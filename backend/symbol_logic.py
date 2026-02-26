"""Symbol aggregation logic and lookup tables for /api/symbols."""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

from weather_codes import ww_to_symbol, ww_severity_rank

# ── Symbol code ↔ type name ──────────────────────────────────────────────────
SYMBOL_CODE_TO_TYPE: dict[int, str] = {
    0: "clear", 1: "st", 2: "ac", 3: "ci",
    4: "blue_thermal", 5: "cu_hum", 6: "cu_con", 7: "cb",
    20: "fog", 21: "rime_fog",
    22: "drizzle_light", 23: "drizzle_moderate", 24: "drizzle_dense",
    25: "freezing_drizzle", 26: "freezing_drizzle_heavy",
    27: "rain_slight", 28: "rain_moderate", 29: "rain_heavy",
    30: "freezing_rain",
    31: "snow_slight", 32: "snow_moderate", 33: "snow_heavy", 34: "snow_grains",
    35: "rain_shower", 36: "rain_shower_moderate",
    37: "snow_shower", 38: "snow_shower_heavy",
    39: "thunderstorm", 40: "thunderstorm_hail",
}

# Severity priority order: highest-severity first (index 0 = most severe).
SYMBOL_PRIORITY: list[int] = [
    40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
    7, 6, 5, 4, 1, 2, 3, 0,
]

# Fast O(1) priority rank lookup by integer symbol code.
SYMBOL_CODE_RANK_LUT: np.ndarray = np.full(256, -1, dtype=np.int16)
for _rk, _code in enumerate(SYMBOL_PRIORITY):
    if 0 <= _code < SYMBOL_CODE_RANK_LUT.shape[0]:
        SYMBOL_CODE_RANK_LUT[_code] = _rk
del _rk, _code
from constants import (
    CAPE_CONV_THRESHOLD,
    CAPE_CB_STRONG_THRESHOLD,
    LPI_CB_THRESHOLD,
    CLOUD_DEPTH_CU_CON_THRESHOLD,
    CLOUD_DEPTH_CB_THRESHOLD,
    AGL_CONV_MIN_METERS,
    CEILING_VALID_MAX_METERS,
)


def aggregate_symbol_cell(
    cli: np.ndarray,
    clo: np.ndarray,
    cell_ww: np.ndarray,
    ceil_arr: np.ndarray,
    c_clcl: np.ndarray,
    c_clcm: np.ndarray,
    c_clch: np.ndarray,
    c_cape: np.ndarray,
    c_htop_dc: np.ndarray,
    c_hbas_sc: np.ndarray,
    c_htop_sc: np.ndarray,
    c_lpi: np.ndarray,
    c_hsurf: np.ndarray,
    classify_point_fn: Callable,
    zoom: int = 12,
    pre_has_cape: bool = False,
    pre_has_ceil: bool = False,
) -> Tuple[str, int | None, int, int]:
    """Aggregate one cell into (symbol, cloud_base_hm, best_ii, best_jj).

    P0/P2 optimised: the caller (weather.py) guarantees that sig-wx and
    pure-clear cells are handled before this function is called.  This
    function therefore only handles the cloud-classification branch
    (ww <= 3, with possible CAPE or ceiling signal).

    P2: the redundant full-cell (f_*) extraction pass has been removed.
    Only strided (s_*) arrays are used for both detection and representative
    point selection.  pre_has_cape / pre_has_ceil are hints from the
    scatter_cell_stats pre-pass; when a strided sample misses convection
    we use the cell centre as a conservative fallback.
    """
    sym = "clear"
    cb_hm = None
    best_ii = int(cli[len(cli) // 2])
    best_jj = int(clo[len(clo) // 2])

    # Safety guard: sig-wx should be handled by the caller, but handle here
    # too so the function remains self-contained if called directly.
    sig_mask = np.isfinite(cell_ww) & (cell_ww > 10)
    if np.any(sig_mask):
        i_loc, j_loc = np.where(sig_mask)
        ww_vals = cell_ww[i_loc, j_loc].astype(int)
        k = max(range(len(ww_vals)), key=lambda idx: (ww_severity_rank(int(ww_vals[idx])), int(ww_vals[idx])))
        best_ii = int(cli[int(i_loc[k])])
        best_jj = int(clo[int(j_loc[k])])
        sym = ww_to_symbol(int(ww_vals[k])) or "clear"
        return sym, None, best_ii, best_jj

    # Adaptive stride for representative-point selection (P2: used for
    # detection too — the pre-pass pre_has_cape / pre_has_ceil guards
    # us against false-negative classification at low zoom).
    if zoom >= 11:
        stride = 1
    elif zoom >= 9:
        stride = 2
    else:
        stride = 3

    iter_cli = cli[::stride] if len(cli) > 0 else cli
    iter_clo = clo[::stride] if len(clo) > 0 else clo

    if len(iter_cli) == 0 or len(iter_clo) == 0:
        return sym, cb_hm, best_ii, best_jj

    # ── P2: single extraction pass (strided arrays only) ─────────────────────
    s_clcl   = c_clcl[np.ix_(iter_cli, iter_clo)]
    s_clcm   = c_clcm[np.ix_(iter_cli, iter_clo)]
    s_clch   = c_clch[np.ix_(iter_cli, iter_clo)]
    s_cape   = c_cape[np.ix_(iter_cli, iter_clo)]
    s_htop_dc = c_htop_dc[np.ix_(iter_cli, iter_clo)]
    s_hbas_sc = c_hbas_sc[np.ix_(iter_cli, iter_clo)]
    s_htop_sc = c_htop_sc[np.ix_(iter_cli, iter_clo)]
    s_lpi    = c_lpi[np.ix_(iter_cli, iter_clo)]
    s_hsurf  = c_hsurf[np.ix_(iter_cli, iter_clo)]
    s_ceil   = ceil_arr[np.ix_(iter_cli, iter_clo)]

    # ── Convection branch ─────────────────────────────────────────────────────
    conv_mask_s   = np.isfinite(s_cape) & (s_cape > CAPE_CONV_THRESHOLD)
    cloud_depth_s = np.maximum(0.0, np.where(
        np.isfinite(s_htop_sc) & np.isfinite(s_hbas_sc),
        s_htop_sc - s_hbas_sc, 0.0,
    ))
    hbas_agl_s = s_hbas_sc - s_hsurf
    p2_mask_s  = conv_mask_s & np.isfinite(hbas_agl_s) & (hbas_agl_s >= AGL_CONV_MIN_METERS) & (s_hbas_sc > 0)

    if np.any(p2_mask_s):
        cb_mask_s   = p2_mask_s & (
            (np.isfinite(s_lpi) & (s_lpi > LPI_CB_THRESHOLD))
            | ((cloud_depth_s > CLOUD_DEPTH_CB_THRESHOLD) & np.isfinite(s_cape) & (s_cape > CAPE_CB_STRONG_THRESHOLD))
        )
        cu_con_mask_s = p2_mask_s & (~cb_mask_s) & (cloud_depth_s > CLOUD_DEPTH_CU_CON_THRESHOLD)

        i_loc, j_loc = np.where(p2_mask_s)
        vals = s_hbas_sc[i_loc, j_loc]
        k = int(np.argmax(vals))
        ii_s, jj_s = int(i_loc[k]), int(j_loc[k])
        best_ii = int(iter_cli[ii_s])
        best_jj = int(iter_clo[jj_s])
        if cb_mask_s[ii_s, jj_s]:
            sym = "cb"
        elif cu_con_mask_s[ii_s, jj_s]:
            sym = "cu_con"
        else:
            sym = "cu_hum"
        cb_hm = int((float(vals[k]) + 50) / 100) if float(vals[k]) > 0 else None

    elif pre_has_cape:
        # Pre-pass confirmed cape present but stride missed it.
        # Check dry-convection (blue thermal) in sampled arrays.
        htop_dc_agl_s = s_htop_dc - s_hsurf
        p3_mask_s = conv_mask_s & np.isfinite(htop_dc_agl_s) & (htop_dc_agl_s >= AGL_CONV_MIN_METERS)
        if np.any(p3_mask_s):
            i_loc, j_loc = np.where(p3_mask_s)
            vals = s_htop_dc[i_loc, j_loc]
            k = int(np.argmax(vals))
            best_ii = int(iter_cli[int(i_loc[k])])
            best_jj = int(iter_clo[int(j_loc[k])])
            cb_val = float(vals[k])
            sym = "blue_thermal"
            cb_hm = int((cb_val + 50) / 100) if cb_val > 0 else None
        else:
            # Stride missed convective cloud base; cannot confirm hbas_sc > 0
            # so do not emit a cloud symbol.
            sym = "clear"
            cb_hm = None

    else:
        # ── Dry convection (blue thermal) check ───────────────────────────────
        htop_dc_agl_s = s_htop_dc - s_hsurf
        p3_mask_s = conv_mask_s & np.isfinite(htop_dc_agl_s) & (htop_dc_agl_s >= AGL_CONV_MIN_METERS)

        if np.any(p3_mask_s):
            i_loc, j_loc = np.where(p3_mask_s)
            vals = s_htop_dc[i_loc, j_loc]
            k = int(np.argmax(vals))
            best_ii = int(iter_cli[int(i_loc[k])])
            best_jj = int(iter_clo[int(j_loc[k])])
            cb_val = float(vals[k])
            sym = "blue_thermal"
            cb_hm = int((cb_val + 50) / 100) if cb_val > 0 else None

        else:
            # ── Non-convective ceiling fallback ───────────────────────────────
            ceil_mask_s = np.isfinite(s_ceil) & (s_ceil > 0) & (s_ceil < CEILING_VALID_MAX_METERS)
            if not np.any(ceil_mask_s):
                sym = "clear"
            else:
                i_loc, j_loc = np.where(ceil_mask_s)
                vals = s_ceil[i_loc, j_loc]
                avg_ceil = float(np.mean(vals))
                k = int(np.argmin(np.abs(vals - avg_ceil)))
                ii_s, jj_s = int(i_loc[k]), int(j_loc[k])
                best_ii = int(iter_cli[ii_s])
                best_jj = int(iter_clo[jj_s])

                chosen_ceil = float(vals[k])
                chosen_clcl = float(s_clcl[ii_s, jj_s]) if np.isfinite(s_clcl[ii_s, jj_s]) else np.nan
                chosen_clcm = float(s_clcm[ii_s, jj_s]) if np.isfinite(s_clcm[ii_s, jj_s]) else np.nan
                chosen_clch = float(s_clch[ii_s, jj_s]) if np.isfinite(s_clch[ii_s, jj_s]) else np.nan

                if chosen_ceil < 2000:
                    sym = "st" if np.isfinite(chosen_clcl) and chosen_clcl >= 30 else "clear"
                elif chosen_ceil < 7000:
                    sym = "ac" if np.isfinite(chosen_clcm) and chosen_clcm >= 30 else "clear"
                else:
                    sym = "ci" if np.isfinite(chosen_clch) and chosen_clch >= 30 else "clear"

                cb_hm = int((chosen_ceil + 50) / 100) if sym != "clear" else None

    if cb_hm is not None and cb_hm > 99:
        cb_hm = 99

    return sym, cb_hm, best_ii, best_jj
