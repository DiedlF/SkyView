"""Shared symbol payload computation for live API and direct precompute."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from classify import classify_point as _classify_point_core
from constants import (
    CAPE_CONV_THRESHOLD,
    CEILING_VALID_MAX_METERS,
    CELL_SIZES_BY_ZOOM,
)
from convective_filters import filter_hbas_with_mh
from grid_aggregation import build_grid_context, choose_cell_groups, scatter_cell_stats, scatter_best_symbol
from grid_utils import bbox_indices as _bbox_indices, slice_array as _slice_array
from symbol_logic import SYMBOL_CODE_RANK_LUT, SYMBOL_CODE_TO_TYPE, aggregate_symbol_cell
from weather_codes import ww_to_symbol, ww_severity_rank

SYMBOL_KEYS: list[str] = [
    "ww", "ceiling", "clcl", "clcm", "clch",
    "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf", "mh",
    "sym_code", "cb_hm",
]


def compute_symbols_payload(
    *,
    zoom: int,
    bbox: str,
    time: str,
    model: Optional[str],
    symbol_mode: str,
    resolve_time_with_cache_context: Callable[[str, Optional[str]], tuple[str, int, str]],
    load_data: Callable[[str, int, str, Optional[List[str]]], Dict[str, Any]],
    load_eu_data_strict: Callable[[str, List[str]], Any],
    freshness_minutes_from_run: Callable[[str], Any],
    strict_window_hours: float,
    load_coverage_damping_cfg: Callable[[], dict],
) -> dict:
    cell_size = CELL_SIZES_BY_ZOOM[zoom]
    parts = bbox.split(",")
    if len(parts) != 4:
        raise ValueError("bbox: lat_min,lon_min,lat_max,lon_max")
    req_lat_min, req_lon_min, req_lat_max, req_lon_max = map(float, parts)
    lat_min, lon_min, lat_max, lon_max = req_lat_min, req_lon_min, req_lat_max, req_lon_max
    pad = cell_size * 0.5

    requested_model_for_mode = "icon_d2" if symbol_mode == "native" else model
    run, step, model_used = resolve_time_with_cache_context(time, requested_model_for_mode)

    d = load_data(run, step, model_used, keys=SYMBOL_KEYS)
    lat = d["lat"]
    lon = d["lon"]
    d2_lat_min = float(d.get("_latMin", np.min(lat)))
    d2_lat_max = float(d.get("_latMax", np.max(lat)))
    d2_lon_min = float(d.get("_lonMin", np.min(lon)))
    d2_lon_max = float(d.get("_lonMax", np.max(lon)))

    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is None or lo is None or len(li) == 0 or len(lo) == 0:
        return {
            "symbols": [],
            "run": run,
            "model": model_used,
            "validTime": d["validTime"],
            "cellSize": cell_size,
            "count": 0,
            "diagnostics": {
                "dataFreshnessMinutes": freshness_minutes_from_run(run),
                "fallbackDecision": "primary_model_only",
                "requestedModel": model,
                "requestedTime": time,
                "sourceModel": model_used,
                "strictWindowHours": strict_window_hours,
                "euDataMissing": False,
                "euCells": 0,
                "d2Cells": 0,
                "euShare": 0.0,
                "servedFrom": "computed",
                "symbolMode": symbol_mode,
            },
        }

    c_lat = lat[li] if li is not None else lat
    c_lon = lon[lo] if lo is not None else lon
    ww = _slice_array(d["ww"], li, lo)
    ceil_arr = _slice_array(d["ceiling"], li, lo)
    c_clcl = _slice_array(d["clcl"], li, lo) if "clcl" in d else np.zeros_like(ww)
    c_clcm = _slice_array(d["clcm"], li, lo) if "clcm" in d else np.zeros_like(ww)
    c_clch = _slice_array(d["clch"], li, lo) if "clch" in d else np.zeros_like(ww)
    c_cape = _slice_array(d["cape_ml"], li, lo) if "cape_ml" in d else np.zeros_like(ww)
    c_htop_dc = _slice_array(d["htop_dc"], li, lo) if "htop_dc" in d else np.zeros_like(ww)
    c_hbas_sc = _slice_array(d["hbas_sc"], li, lo) if "hbas_sc" in d else np.zeros_like(ww)
    c_htop_sc = _slice_array(d["htop_sc"], li, lo) if "htop_sc" in d else np.zeros_like(ww)
    c_lpi = (
        _slice_array(d["lpi_max"], li, lo) if "lpi_max" in d
        else (_slice_array(d["lpi"], li, lo) if "lpi" in d else np.zeros_like(ww))
    )
    c_hsurf = _slice_array(d["hsurf"], li, lo) if "hsurf" in d else np.zeros_like(ww)
    c_mh = _slice_array(d["mh"], li, lo) if "mh" in d else np.zeros_like(ww)
    c_sym_code = _slice_array(d["sym_code"], li, lo) if "sym_code" in d else None
    c_cb_hm = _slice_array(d["cb_hm"], li, lo) if "cb_hm" in d else None

    if c_hbas_sc is not None and c_hsurf is not None and c_mh is not None:
        c_hbas_sc, _ = filter_hbas_with_mh(c_hbas_sc, c_hsurf, c_mh, margin_m=500.0, hard_cap_agl_m=6500.0)

    d_eu = c_lat_eu = c_lon_eu = ww_eu = ceil_arr_eu = None
    c_clcl_eu = c_clcm_eu = c_clch_eu = c_cape_eu = None
    c_htop_dc_eu = c_hbas_sc_eu = c_htop_sc_eu = c_lpi_eu = c_hsurf_eu = c_mh_eu = None
    c_sym_code_eu = c_cb_hm_eu = None
    eu_data_missing = False

    if model_used == "icon_d2":
        bbox_inside_d2 = (
            lat_min >= d2_lat_min and lat_max <= d2_lat_max
            and lon_min >= d2_lon_min and lon_max <= d2_lon_max
        )
        finite_ratio = float(np.mean(np.isfinite(ww))) if ww.size else 0.0
        allow_eu_fallback = not (bbox_inside_d2 and finite_ratio >= 0.995)
        needs_eu_for_coverage = (
            (lat_min - pad) < d2_lat_min or (lat_max + pad) > d2_lat_max
            or (lon_min - pad) < d2_lon_min or (lon_max + pad) > d2_lon_max
        )
        needs_eu_for_signal = bool(ww.size) and bool(np.any(~np.isfinite(ww)))
        if allow_eu_fallback and (needs_eu_for_coverage or needs_eu_for_signal):
            eu_fb = load_eu_data_strict(time, SYMBOL_KEYS)
            if eu_fb is not None and eu_fb.get("missing"):
                eu_data_missing = True
                eu_fb = None
            if eu_fb is not None:
                d_eu = eu_fb["data"]
                lat_eu = d_eu["lat"]
                lon_eu = d_eu["lon"]
                li_eu, lo_eu = _bbox_indices(lat_eu, lon_eu, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
                if not (li_eu is not None and len(li_eu) == 0):
                    c_lat_eu = lat_eu[li_eu] if li_eu is not None else lat_eu
                    c_lon_eu = lon_eu[lo_eu] if lo_eu is not None else lon_eu
                    ww_eu = _slice_array(d_eu["ww"], li_eu, lo_eu)
                    ceil_arr_eu = _slice_array(d_eu["ceiling"], li_eu, lo_eu)
                    c_clcl_eu = _slice_array(d_eu["clcl"], li_eu, lo_eu) if "clcl" in d_eu else np.zeros_like(ww_eu)
                    c_clcm_eu = _slice_array(d_eu["clcm"], li_eu, lo_eu) if "clcm" in d_eu else np.zeros_like(ww_eu)
                    c_clch_eu = _slice_array(d_eu["clch"], li_eu, lo_eu) if "clch" in d_eu else np.zeros_like(ww_eu)
                    c_cape_eu = _slice_array(d_eu["cape_ml"], li_eu, lo_eu) if "cape_ml" in d_eu else np.zeros_like(ww_eu)
                    c_htop_dc_eu = _slice_array(d_eu["htop_dc"], li_eu, lo_eu) if "htop_dc" in d_eu else np.zeros_like(ww_eu)
                    c_hbas_sc_eu = _slice_array(d_eu["hbas_sc"], li_eu, lo_eu) if "hbas_sc" in d_eu else np.zeros_like(ww_eu)
                    c_htop_sc_eu = _slice_array(d_eu["htop_sc"], li_eu, lo_eu) if "htop_sc" in d_eu else np.zeros_like(ww_eu)
                    c_lpi_eu = (
                        _slice_array(d_eu["lpi_max"], li_eu, lo_eu) if "lpi_max" in d_eu
                        else (_slice_array(d_eu["lpi"], li_eu, lo_eu) if "lpi" in d_eu else np.zeros_like(ww_eu))
                    )
                    c_hsurf_eu = _slice_array(d_eu["hsurf"], li_eu, lo_eu) if "hsurf" in d_eu else np.zeros_like(ww_eu)
                    c_mh_eu = _slice_array(d_eu["mh"], li_eu, lo_eu) if "mh" in d_eu else np.zeros_like(ww_eu)
                    c_hbas_sc_eu, _ = filter_hbas_with_mh(c_hbas_sc_eu, c_hsurf_eu, c_mh_eu, margin_m=500.0, hard_cap_agl_m=6500.0)
                    c_sym_code_eu = _slice_array(d_eu["sym_code"], li_eu, lo_eu) if "sym_code" in d_eu else None
                    c_cb_hm_eu = _slice_array(d_eu["cb_hm"], li_eu, lo_eu) if "cb_hm" in d_eu else None

    ctx = build_grid_context(
        lat=lat, lon=lon, c_lat=c_lat, c_lon=c_lon,
        lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max,
        cell_size=cell_size, zoom=zoom,
        d2_lat_min=d2_lat_min, d2_lat_max=d2_lat_max,
        d2_lon_min=d2_lon_min, d2_lon_max=d2_lon_max,
        c_lat_eu=c_lat_eu if d_eu is not None else None,
        c_lon_eu=c_lon_eu if d_eu is not None else None,
    )

    pre_d2 = scatter_cell_stats(c_lat, c_lon, ctx, ww, c_cape, ceil_arr, CAPE_CONV_THRESHOLD, CEILING_VALID_MAX_METERS)
    pre_eu = None
    if d_eu is not None and c_lat_eu is not None and ww_eu is not None:
        pre_eu = scatter_cell_stats(c_lat_eu, c_lon_eu, ctx, ww_eu, c_cape_eu, ceil_arr_eu, CAPE_CONV_THRESHOLD, CEILING_VALID_MAX_METERS)

    cd_cfg = load_coverage_damping_cfg()
    best_sym_d2 = best_cb_d2 = best_lat_d2 = best_lon_d2 = None
    if c_sym_code is not None and c_cb_hm is not None:
        best_sym_d2, best_cb_d2, best_lat_d2, best_lon_d2 = scatter_best_symbol(
            c_lat, c_lon, ctx, c_sym_code, c_cb_hm, SYMBOL_CODE_RANK_LUT,
            coverage_damping_enabled=cd_cfg["enabled"],
            coverage_min_fraction=cd_cfg["min_fraction"],
            coverage_rank_tolerance=cd_cfg["rank_tolerance"],
        )
    best_sym_eu = best_cb_eu = best_lat_eu = best_lon_eu = None
    if c_sym_code_eu is not None and c_cb_hm_eu is not None and c_lat_eu is not None and c_lon_eu is not None:
        best_sym_eu, best_cb_eu, best_lat_eu, best_lon_eu = scatter_best_symbol(
            c_lat_eu, c_lon_eu, ctx, c_sym_code_eu, c_cb_hm_eu, SYMBOL_CODE_RANK_LUT,
            coverage_damping_enabled=cd_cfg["enabled"],
            coverage_min_fraction=cd_cfg["min_fraction"],
            coverage_rank_tolerance=cd_cfg["rank_tolerance"],
        )

    symbols: List[dict] = []
    used_eu_cells = used_d2_cells = 0
    lat_edges = ctx.lat_edges
    lon_edges = ctx.lon_edges

    for i in range(ctx.lat_cell_count):
        for j in range(ctx.lon_cell_count):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2
            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            in_d2_domain = bool(ctx.in_d2_grid[i, j]) if ctx.in_d2_grid.size else False
            use_eu, cli_list, clo_list = choose_cell_groups(ctx, i, j, prefer_eu=((not in_d2_domain) and (ctx.eu is not None)))
            if use_eu:
                src_lat, src_lon = c_lat_eu, c_lon_eu
                src_ww, src_ceil = ww_eu, ceil_arr_eu
                src_clcl, src_clcm, src_clch = c_clcl_eu, c_clcm_eu, c_clch_eu
                src_cape, src_htop_dc = c_cape_eu, c_htop_dc_eu
                src_hbas_sc, src_htop_sc = c_hbas_sc_eu, c_htop_sc_eu
                src_lpi, src_hsurf, src_mh = c_lpi_eu, c_hsurf_eu, c_mh_eu
                src_sym_code, src_cb_hm = c_sym_code_eu, c_cb_hm_eu
            else:
                src_lat, src_lon = c_lat, c_lon
                src_ww, src_ceil = ww, ceil_arr
                src_clcl, src_clcm, src_clch = c_clcl, c_clcm, c_clch
                src_cape, src_htop_dc = c_cape, c_htop_dc
                src_hbas_sc, src_htop_sc = c_hbas_sc, c_htop_sc
                src_lpi, src_hsurf, src_mh = c_lpi, c_hsurf, c_mh
                src_sym_code, src_cb_hm = c_sym_code, c_cb_hm

            cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
            clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

            if (not use_eu) and (ctx.eu is not None) and len(cli) > 0 and len(clo) > 0 and np.isnan(pre_d2[0][i, j]):
                use_eu = True
                src_lat, src_lon = c_lat_eu, c_lon_eu
                src_ww, src_ceil = ww_eu, ceil_arr_eu
                src_clcl, src_clcm, src_clch = c_clcl_eu, c_clcm_eu, c_clch_eu
                src_cape, src_htop_dc = c_cape_eu, c_htop_dc_eu
                src_hbas_sc, src_htop_sc = c_hbas_sc_eu, c_htop_sc_eu
                src_lpi, src_hsurf, src_mh = c_lpi_eu, c_hsurf_eu, c_mh_eu
                src_sym_code, src_cb_hm = c_sym_code_eu, c_cb_hm_eu
                cli_list = ctx.eu.lat_groups[i]
                clo_list = ctx.eu.lon_groups[j]
                cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

            if len(cli) == 0 or len(clo) == 0:
                continue

            pre = pre_eu if (use_eu and pre_eu is not None) else pre_d2
            pre_max_ww = float(pre[0][i, j])
            pre_any_cape = bool(pre[1][i, j])
            pre_any_ceil = bool(pre[2][i, j])
            max_ww = int(pre_max_ww) if np.isfinite(pre_max_ww) else 0

            best_sym = best_sym_eu if use_eu else best_sym_d2
            best_cb = best_cb_eu if use_eu else best_cb_d2
            best_li = best_lat_eu if use_eu else best_lat_d2
            best_lj = best_lon_eu if use_eu else best_lon_d2

            if best_sym is not None and src_sym_code is not None and src_cb_hm is not None:
                code = int(best_sym[i, j])
                sym = SYMBOL_CODE_TO_TYPE.get(code, "clear")
                vcb = int(best_cb[i, j])
                cb_hm = vcb if vcb >= 0 else None
                li_idx = int(best_li[i, j])
                lj_idx = int(best_lj[i, j])
                if li_idx >= 0 and lj_idx >= 0:
                    best_ii, best_jj = li_idx, lj_idx
                else:
                    best_ii = int(cli_list[len(cli_list) // 2])
                    best_jj = int(clo_list[len(clo_list) // 2])
            else:
                if np.isnan(pre_max_ww) or (pre_max_ww <= 3 and not pre_any_cape and not pre_any_ceil):
                    sym, cb_hm = "clear", None
                    best_ii = int(cli_list[len(cli_list) // 2])
                    best_jj = int(clo_list[len(clo_list) // 2])
                elif np.isfinite(pre_max_ww) and pre_max_ww > 10:
                    cell_ww_only = src_ww[np.ix_(cli, clo)]
                    sig_mask = np.isfinite(cell_ww_only) & (cell_ww_only > 10)
                    if np.any(sig_mask):
                        i_loc, j_loc = np.where(sig_mask)
                        ww_vals = cell_ww_only[i_loc, j_loc].astype(int)
                        k = max(range(len(ww_vals)), key=lambda idx: (ww_severity_rank(int(ww_vals[idx])), int(ww_vals[idx])))
                        sym = ww_to_symbol(int(ww_vals[k])) or "clear"
                        best_ii = int(cli[int(i_loc[k])])
                        best_jj = int(clo[int(j_loc[k])])
                        max_ww = int(ww_vals[k])
                    else:
                        sym = ww_to_symbol(max_ww) or "clear"
                        best_ii = int(cli_list[len(cli_list) // 2])
                        best_jj = int(clo_list[len(clo_list) // 2])
                        cb_hm = None
                    cb_hm = None
                else:
                    cell_ww = src_ww[np.ix_(cli, clo)]
                    max_ww = int(np.nanmax(cell_ww)) if np.any(np.isfinite(cell_ww)) else 0
                    sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                        cli=cli, clo=clo, cell_ww=cell_ww,
                        ceil_arr=src_ceil, c_clcl=src_clcl, c_clcm=src_clcm,
                        c_clch=src_clch, c_cape=src_cape, c_htop_dc=src_htop_dc,
                        c_hbas_sc=src_hbas_sc, c_htop_sc=src_htop_sc,
                        c_lpi=src_lpi, c_hsurf=src_hsurf, c_mh=src_mh,
                        classify_point_fn=_classify_point_core,
                        zoom=zoom, pre_has_cape=pre_any_cape, pre_has_ceil=pre_any_ceil,
                    )

            label = None
            if cb_hm is not None:
                cb_hm = min(cb_hm, 99)
                label = str(cb_hm)

            source_model = "icon_eu" if (use_eu or model_used == "icon_eu") else "icon_d2"
            if source_model == "icon_eu":
                used_eu_cells += 1
            else:
                used_d2_cells += 1

            symbols.append({
                "lat": round(float(lat_c), 4),
                "lon": round(float(lon_c), 4),
                "clickLat": round(float(src_lat[best_ii]), 4),
                "clickLon": round(float(src_lon[best_jj]), 4),
                "type": sym,
                "ww": max_ww,
                "cloudBase": cb_hm,
                "label": label,
                "clickable": True,
                "sourceModel": source_model,
            })

    total_cells = used_eu_cells + used_d2_cells
    eu_share = (used_eu_cells / total_cells) if total_cells else 0.0
    significant_blend = used_eu_cells >= 3 and eu_share >= 0.03
    if used_eu_cells and not used_d2_cells:
        resolved_model = "icon_eu"
        effective_run = d_eu.get("_run", run) if d_eu is not None else run
        effective_valid_time = d_eu.get("validTime", d["validTime"]) if d_eu is not None else d["validTime"]
        fallback_decision = "eu_only_in_viewport"
    elif used_eu_cells and used_d2_cells and significant_blend:
        resolved_model = "ICON-D2 + EU"
        effective_run = run
        effective_valid_time = d["validTime"]
        fallback_decision = "blended_d2_eu"
    elif used_eu_cells and used_d2_cells:
        resolved_model = model_used
        effective_run = run
        effective_valid_time = d["validTime"]
        fallback_decision = "primary_model_with_eu_assist"
    else:
        resolved_model = model_used
        effective_run = run
        effective_valid_time = d["validTime"]
        fallback_decision = "primary_model_only"

    return {
        "symbols": symbols,
        "run": effective_run,
        "model": resolved_model,
        "validTime": effective_valid_time,
        "cellSize": cell_size,
        "count": len(symbols),
        "diagnostics": {
            "dataFreshnessMinutes": freshness_minutes_from_run(effective_run),
            "fallbackDecision": fallback_decision,
            "requestedModel": model,
            "requestedTime": time,
            "sourceModel": resolved_model,
            "strictWindowHours": strict_window_hours,
            "euDataMissing": eu_data_missing,
            "euCells": used_eu_cells,
            "d2Cells": used_d2_cells,
            "euShare": round(eu_share, 4),
            "servedFrom": "computed",
            "symbolMode": symbol_mode,
        },
    }
