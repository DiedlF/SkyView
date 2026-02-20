"""Symbol aggregation logic for /api/symbols."""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

from weather_codes import ww_to_symbol, ww_severity_rank
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
) -> Tuple[str, int | None, int, int]:
    """Aggregate one cell into (symbol, cloud_base_hm, best_ii, best_jj)."""
    sym = "clear"
    cb_hm = None
    best_ii = int(cli[len(cli) // 2])
    best_jj = int(clo[len(clo) // 2])

    # 1) Significant weather aggregation (ww > 10)
    sig_mask = np.isfinite(cell_ww) & (cell_ww > 10)
    if np.any(sig_mask):
        i_loc, j_loc = np.where(sig_mask)
        ww_vals = cell_ww[i_loc, j_loc].astype(int)
        # Highest severity rank wins; tie-break on larger ww
        k = max(range(len(ww_vals)), key=lambda idx: (ww_severity_rank(int(ww_vals[idx])), int(ww_vals[idx])))
        best_ii = int(cli[int(i_loc[k])])
        best_jj = int(clo[int(j_loc[k])])
        best_ww = int(ww_vals[k])
        sym = ww_to_symbol(best_ww) or "clear"
        cb_hm = None
    else:
        # Priority 2/3 aggregation order (explicit):
        # P2: hbas_sc > 0 && cape_ml > CAPE_CONV_THRESHOLD
        # P3: htop_dc > 0 && cape_ml > CAPE_CONV_THRESHOLD
        # else: non-convective ceiling fallback
        # Adaptive sampling at lower zoom to keep response times stable.
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

        # Full-cell arrays for robust signal detection (continuity-safe across zoom levels)
        f_clcl = c_clcl[np.ix_(cli, clo)]
        f_clcm = c_clcm[np.ix_(cli, clo)]
        f_clch = c_clch[np.ix_(cli, clo)]
        f_cape = c_cape[np.ix_(cli, clo)]
        f_htop_dc = c_htop_dc[np.ix_(cli, clo)]
        f_hbas_sc = c_hbas_sc[np.ix_(cli, clo)]
        f_htop_sc = c_htop_sc[np.ix_(cli, clo)]
        f_lpi = c_lpi[np.ix_(cli, clo)]
        f_hsurf = c_hsurf[np.ix_(cli, clo)]
        f_ceil = ceil_arr[np.ix_(cli, clo)]

        # Sampled sub-cell arrays (still used for representative point selection where possible)
        s_clcl = c_clcl[np.ix_(iter_cli, iter_clo)]
        s_clcm = c_clcm[np.ix_(iter_cli, iter_clo)]
        s_clch = c_clch[np.ix_(iter_cli, iter_clo)]
        s_cape = c_cape[np.ix_(iter_cli, iter_clo)]
        s_htop_dc = c_htop_dc[np.ix_(iter_cli, iter_clo)]
        s_hbas_sc = c_hbas_sc[np.ix_(iter_cli, iter_clo)]
        s_htop_sc = c_htop_sc[np.ix_(iter_cli, iter_clo)]
        s_lpi = c_lpi[np.ix_(iter_cli, iter_clo)]
        s_hsurf = c_hsurf[np.ix_(iter_cli, iter_clo)]
        s_ceil = ceil_arr[np.ix_(iter_cli, iter_clo)]

        # Vectorized emulation of classify_point() decision tree.
        # Detection is performed on full arrays; sampled arrays are used for faster representative-point selection.
        conv_mask_f = np.isfinite(f_cape) & (f_cape > CAPE_CONV_THRESHOLD)
        cloud_depth_f = np.maximum(0.0, np.where(np.isfinite(f_htop_sc) & np.isfinite(f_hbas_sc), f_htop_sc - f_hbas_sc, 0.0))
        hbas_agl_f = f_hbas_sc - f_hsurf
        p2_mask_f = conv_mask_f & np.isfinite(hbas_agl_f) & (hbas_agl_f >= AGL_CONV_MIN_METERS)

        conv_mask_s = np.isfinite(s_cape) & (s_cape > CAPE_CONV_THRESHOLD)
        cloud_depth_s = np.maximum(0.0, np.where(np.isfinite(s_htop_sc) & np.isfinite(s_hbas_sc), s_htop_sc - s_hbas_sc, 0.0))
        hbas_agl_s = s_hbas_sc - s_hsurf
        p2_mask_s = conv_mask_s & np.isfinite(hbas_agl_s) & (hbas_agl_s >= AGL_CONV_MIN_METERS)

        if np.any(p2_mask_f):
            cb_mask_f = p2_mask_f & (
                (np.isfinite(f_lpi) & (f_lpi > LPI_CB_THRESHOLD)) | ((cloud_depth_f > CLOUD_DEPTH_CB_THRESHOLD) & np.isfinite(f_cape) & (f_cape > CAPE_CB_STRONG_THRESHOLD))
            )
            cu_con_mask_f = p2_mask_f & (~cb_mask_f) & (cloud_depth_f > CLOUD_DEPTH_CU_CON_THRESHOLD)

            cb_mask_s = p2_mask_s & (
                (np.isfinite(s_lpi) & (s_lpi > LPI_CB_THRESHOLD)) | ((cloud_depth_s > CLOUD_DEPTH_CB_THRESHOLD) & np.isfinite(s_cape) & (s_cape > CAPE_CB_STRONG_THRESHOLD))
            )
            cu_con_mask_s = p2_mask_s & (~cb_mask_s) & (cloud_depth_s > CLOUD_DEPTH_CU_CON_THRESHOLD)

            if np.any(p2_mask_s):
                i_loc, j_loc = np.where(p2_mask_s)
                vals = s_hbas_sc[i_loc, j_loc]
                k = int(np.argmax(vals))
                ii_s = int(i_loc[k])
                jj_s = int(j_loc[k])
                best_ii = int(iter_cli[ii_s])
                best_jj = int(iter_clo[jj_s])
                if cb_mask_s[ii_s, jj_s]:
                    sym = "cb"
                elif cu_con_mask_s[ii_s, jj_s]:
                    sym = "cu_con"
                else:
                    sym = "cu_hum"
                cb_hm = int((float(vals[k]) + 50) / 100) if float(vals[k]) > 0 else None
            else:
                i_loc, j_loc = np.where(p2_mask_f)
                vals = f_hbas_sc[i_loc, j_loc]
                k = int(np.argmax(vals))
                ii_f = int(i_loc[k])
                jj_f = int(j_loc[k])
                best_ii = int(cli[ii_f])
                best_jj = int(clo[jj_f])
                if cb_mask_f[ii_f, jj_f]:
                    sym = "cb"
                elif cu_con_mask_f[ii_f, jj_f]:
                    sym = "cu_con"
                else:
                    sym = "cu_hum"
                cb_hm = int((float(vals[k]) + 50) / 100) if float(vals[k]) > 0 else None
        else:
            # Priority 3: dry convection candidates (htop_dc-hsurf >= AGL minimum with CAPE above convective threshold)
            htop_dc_agl_f = f_htop_dc - f_hsurf
            p3_mask_f = conv_mask_f & np.isfinite(htop_dc_agl_f) & (htop_dc_agl_f >= AGL_CONV_MIN_METERS)

            htop_dc_agl_s = s_htop_dc - s_hsurf
            p3_mask_s = conv_mask_s & np.isfinite(htop_dc_agl_s) & (htop_dc_agl_s >= AGL_CONV_MIN_METERS)

            if np.any(p3_mask_f):
                if np.any(p3_mask_s):
                    i_loc, j_loc = np.where(p3_mask_s)
                    vals = s_htop_dc[i_loc, j_loc]
                    k = int(np.argmax(vals))
                    ii_s = int(i_loc[k])
                    jj_s = int(j_loc[k])
                    best_ii = int(iter_cli[ii_s])
                    best_jj = int(iter_clo[jj_s])
                    cb_val = float(vals[k])
                else:
                    i_loc, j_loc = np.where(p3_mask_f)
                    vals = f_htop_dc[i_loc, j_loc]
                    k = int(np.argmax(vals))
                    ii_f = int(i_loc[k])
                    jj_f = int(j_loc[k])
                    best_ii = int(cli[ii_f])
                    best_jj = int(clo[jj_f])
                    cb_val = float(vals[k])

                sym = "blue_thermal"
                cb_hm = int((cb_val + 50) / 100) if cb_val > 0 else None
            else:
                # Non-convective fallback: representative-cell method
                # 1) detect any valid ceiling on full cell
                # 2) pick representative from sampled subset if possible, else full cell
                ceil_mask_f = np.isfinite(f_ceil) & (f_ceil > 0) & (f_ceil < CEILING_VALID_MAX_METERS)
                ceil_valid_f = f_ceil[ceil_mask_f]
                if len(ceil_valid_f) == 0:
                    sym = "clear"
                else:
                    avg_ceil = float(np.mean(ceil_valid_f))
                    ceil_mask_s = np.isfinite(s_ceil) & (s_ceil > 0) & (s_ceil < CEILING_VALID_MAX_METERS)

                    if np.any(ceil_mask_s):
                        i_loc, j_loc = np.where(ceil_mask_s)
                        vals = s_ceil[i_loc, j_loc]
                        k = int(np.argmin(np.abs(vals - avg_ceil)))
                        ii_s = int(i_loc[k])
                        jj_s = int(j_loc[k])
                        best_ii = int(iter_cli[ii_s])
                        best_jj = int(iter_clo[jj_s])

                        chosen_ceil = float(vals[k])
                        chosen_clcl = float(s_clcl[ii_s, jj_s]) if np.isfinite(s_clcl[ii_s, jj_s]) else np.nan
                        chosen_clcm = float(s_clcm[ii_s, jj_s]) if np.isfinite(s_clcm[ii_s, jj_s]) else np.nan
                        chosen_clch = float(s_clch[ii_s, jj_s]) if np.isfinite(s_clch[ii_s, jj_s]) else np.nan
                    else:
                        i_loc, j_loc = np.where(ceil_mask_f)
                        vals = f_ceil[i_loc, j_loc]
                        k = int(np.argmin(np.abs(vals - avg_ceil)))
                        ii_f = int(i_loc[k])
                        jj_f = int(j_loc[k])
                        best_ii = int(cli[ii_f])
                        best_jj = int(clo[jj_f])

                        chosen_ceil = float(vals[k])
                        chosen_clcl = float(f_clcl[ii_f, jj_f]) if np.isfinite(f_clcl[ii_f, jj_f]) else np.nan
                        chosen_clcm = float(f_clcm[ii_f, jj_f]) if np.isfinite(f_clcm[ii_f, jj_f]) else np.nan
                        chosen_clch = float(f_clch[ii_f, jj_f]) if np.isfinite(f_clch[ii_f, jj_f]) else np.nan

                    if chosen_ceil < 2000:
                        sym = "st" if np.isfinite(chosen_clcl) and chosen_clcl >= 30 else "clear"
                    elif chosen_ceil < 7000:
                        sym = "ac" if np.isfinite(chosen_clcm) and chosen_clcm >= 30 else "clear"
                    else:
                        sym = "ci" if np.isfinite(chosen_clch) and chosen_clch >= 30 else "clear"

                    cb_hm = int((chosen_ceil + 50) / 100) if sym != "clear" else None

    # Keep label tied to the winning symbol point altitude.
    if cb_hm is not None and cb_hm > 99:
        cb_hm = 99

    return sym, cb_hm, best_ii, best_jj
