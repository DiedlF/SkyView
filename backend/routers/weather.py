"""Weather map endpoints: symbols, wind barbs, emagram, meteogram.

Handler bodies are defined inside build_weather_router() so they close over
injected app-level dependencies (data loaders, caches, state dicts) without
importing from app.py and creating a circular dependency.

Direct module imports are used for everything that lives outside app.py.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import uuid
from collections import OrderedDict
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from cache_state import symbols_cache_get, symbols_cache_set
from classify import classify_point as _classify_point_core
from constants import (
    CAPE_CONV_THRESHOLD,
    CEILING_VALID_MAX_METERS,
    CELL_SIZES_BY_ZOOM,
    EMAGRAM_D2_LEVELS_HPA,
    G0,
    LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM,
)
from grid_aggregation import build_grid_context, choose_cell_groups, scatter_cell_stats
from grid_utils import bbox_indices as _bbox_indices, slice_array as _slice_array
from services.symbol_ops import filter_symbols_to_bbox, load_symbols_precomputed
from symbol_logic import SYMBOL_CODE_RANK_LUT, SYMBOL_CODE_TO_TYPE, aggregate_symbol_cell


def build_weather_router(
    *,
    resolve_time_with_cache_context,
    load_data,
    _load_eu_data_strict,
    fallback_stats: dict,
    _set_fallback_current,
    _freshness_minutes_from_run,
    EU_STRICT_MAX_DELTA_HOURS: float,
    low_zoom_symbols_cache_metrics: dict,
    data_dir: str,
    meteogram_cache: OrderedDict,
    METEOGRAM_CACHE_MAX_ITEMS: int,
    get_merged_timeline,
    logger,
):
    router = APIRouter()

    # ── /api/symbols ──────────────────────────────────────────────────────────

    @router.get("/api/symbols")
    async def api_symbols(
        request: Request,
        zoom: int = Query(8, ge=5, le=12),
        bbox: str = Query("30,-30,72,45"),
        time: str = Query("latest"),
        model: Optional[str] = Query(None),
    ):
        t0 = perf_counter()
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t_load_ms = 0.0
        t_grid_ms = 0.0
        t_agg_ms = 0.0
        cell_size = CELL_SIZES_BY_ZOOM[zoom]

        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(400, "bbox: lat_min,lon_min,lat_max,lon_max")
        lat_min, lon_min, lat_max, lon_max = map(float, parts)

        symbol_keys = [
            "ww", "ceiling", "clcl", "clcm", "clch",
            "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf",
        ]
        run, step, model_used = resolve_time_with_cache_context(time, model)

        is_low_zoom_global = zoom <= LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM
        cache_bbox = f"{lat_min:.4f},{lon_min:.4f},{lat_max:.4f},{lon_max:.4f}"
        symbols_cache_key = (
            f"{model_used}|{run}|{step}|z{zoom}|global"
            if is_low_zoom_global
            else f"{model_used}|{run}|{step}|z{zoom}|{cache_bbox}"
        )
        cached_symbols = symbols_cache_get(symbols_cache_key)
        served_from = None
        cache_load_ms = 0.0

        if is_low_zoom_global:
            if cached_symbols is not None:
                low_zoom_symbols_cache_metrics["hits"] += 1
                served_from = "cache-memory"
            else:
                low_zoom_symbols_cache_metrics["misses"] += 1
                t_disk0 = perf_counter()
                cached_symbols = load_symbols_precomputed(data_dir, model_used, run, step, zoom)
                cache_load_ms = (perf_counter() - t_disk0) * 1000.0
                if cached_symbols is not None:
                    low_zoom_symbols_cache_metrics["diskHits"] += 1
                    symbols_cache_set(symbols_cache_key, cached_symbols)
                    served_from = "cache-disk"
                else:
                    low_zoom_symbols_cache_metrics["diskMisses"] += 1
        elif cached_symbols is not None:
            served_from = "cache-memory"

        if cached_symbols is not None:
            out_payload = (
                filter_symbols_to_bbox(cached_symbols, lat_min, lon_min, lat_max, lon_max)
                if is_low_zoom_global
                else cached_symbols
            )
            total_ms = (perf_counter() - t0) * 1000.0
            logger.info(
                "/api/symbols rid=%s served=%s zoom=%s count=%s cacheLoadMs=%.2f totalMs=%.2f",
                rid, served_from or "cache-memory", zoom, out_payload.get("count"),
                cache_load_ms, total_ms,
            )
            return out_payload

        if is_low_zoom_global:
            symbols_cache_key = f"{model_used}|{run}|{step}|z{zoom}|{cache_bbox}"

        t_load0 = perf_counter()
        d = load_data(run, step, model_used, keys=symbol_keys)
        t_load_ms += (perf_counter() - t_load0) * 1000.0

        lat = d["lat"]
        lon = d["lon"]
        d2_lat_min = float(d.get("_latMin", np.min(lat)))
        d2_lat_max = float(d.get("_latMax", np.max(lat)))
        d2_lon_min = float(d.get("_lonMin", np.min(lon)))
        d2_lon_max = float(d.get("_lonMax", np.max(lon)))

        d_eu = None
        c_lat_eu = c_lon_eu = None
        ww_eu = ceil_arr_eu = c_clcl_eu = c_clcm_eu = c_clch_eu = None
        c_cape_eu = c_htop_dc_eu = c_hbas_sc_eu = c_htop_sc_eu = c_lpi_eu = c_hsurf_eu = None
        c_sym_code_eu = c_cb_hm_eu = None

        pad = cell_size
        li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
        if li is not None and len(li) == 0:
            c_lat = np.array([], dtype=float)
            c_lon = np.array([], dtype=float)
            ww = np.zeros((0, 0), dtype=float)
            ceil_arr = np.zeros((0, 0), dtype=float)
            c_clcl = c_clcm = c_clch = c_cape = c_htop_dc = np.zeros((0, 0), dtype=float)
            c_hbas_sc = c_htop_sc = c_lpi = c_hsurf = np.zeros((0, 0), dtype=float)
            c_sym_code = c_cb_hm = None
        else:
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
            c_sym_code = _slice_array(d["sym_code"], li, lo) if "sym_code" in d else None
            c_cb_hm = _slice_array(d["cb_hm"], li, lo) if "cb_hm" in d else None

        eu_data_missing = False
        if model_used == "icon_d2":
            needs_eu_for_coverage = (
                (lat_min - pad) < d2_lat_min or (lat_max + pad) > d2_lat_max
                or (lon_min - pad) < d2_lon_min or (lon_max + pad) > d2_lon_max
            )
            needs_eu_for_signal = bool(ww.size) and bool(np.any(~np.isfinite(ww)))
            if needs_eu_for_coverage or needs_eu_for_signal:
                try:
                    eu_fb = _load_eu_data_strict(time, symbol_keys)
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
                            c_sym_code_eu = _slice_array(d_eu["sym_code"], li_eu, lo_eu) if "sym_code" in d_eu else None
                            c_cb_hm_eu = _slice_array(d_eu["cb_hm"], li_eu, lo_eu) if "cb_hm" in d_eu else None
                except Exception:
                    d_eu = None

        t_grid0 = perf_counter()
        ctx = build_grid_context(
            lat=lat, lon=lon, c_lat=c_lat, c_lon=c_lon,
            lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max,
            cell_size=cell_size, zoom=zoom,
            d2_lat_min=d2_lat_min, d2_lat_max=d2_lat_max,
            d2_lon_min=d2_lon_min, d2_lon_max=d2_lon_max,
            c_lat_eu=c_lat_eu if d_eu is not None else None,
            c_lon_eu=c_lon_eu if d_eu is not None else None,
        )
        t_grid_ms += (perf_counter() - t_grid0) * 1000.0

        _pre_d2 = scatter_cell_stats(
            c_lat, c_lon, ctx, ww, c_cape, ceil_arr,
            CAPE_CONV_THRESHOLD, CEILING_VALID_MAX_METERS,
        )
        _pre_eu: Any = None
        if d_eu is not None and c_lat_eu is not None and ww_eu is not None:
            _pre_eu = scatter_cell_stats(
                c_lat_eu, c_lon_eu, ctx, ww_eu, c_cape_eu, ceil_arr_eu,
                CAPE_CONV_THRESHOLD, CEILING_VALID_MAX_METERS,
            )

        t_agg0 = perf_counter()
        symbols: List[dict] = []
        used_eu_any = used_d2_any = False
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
                use_eu, cli_list, clo_list = choose_cell_groups(
                    ctx, i, j, prefer_eu=((not in_d2_domain) and (ctx.eu is not None)),
                )

                if use_eu:
                    used_eu_any = True
                    src_lat, src_lon = c_lat_eu, c_lon_eu
                    src_ww, src_ceil = ww_eu, ceil_arr_eu
                    src_clcl, src_clcm, src_clch = c_clcl_eu, c_clcm_eu, c_clch_eu
                    src_cape, src_htop_dc = c_cape_eu, c_htop_dc_eu
                    src_hbas_sc, src_htop_sc = c_hbas_sc_eu, c_htop_sc_eu
                    src_lpi, src_hsurf = c_lpi_eu, c_hsurf_eu
                    src_sym_code, src_cb_hm = c_sym_code_eu, c_cb_hm_eu
                else:
                    src_lat, src_lon = c_lat, c_lon
                    src_ww, src_ceil = ww, ceil_arr
                    src_clcl, src_clcm, src_clch = c_clcl, c_clcm, c_clch
                    src_cape, src_htop_dc = c_cape, c_htop_dc
                    src_hbas_sc, src_htop_sc = c_hbas_sc, c_htop_sc
                    src_lpi, src_hsurf = c_lpi, c_hsurf
                    src_sym_code, src_cb_hm = c_sym_code, c_cb_hm

                cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

                # Signal-based EU fallback for D2-selected cells with no finite ww
                if (not use_eu) and (ctx.eu is not None) and len(cli) > 0 and len(clo) > 0:
                    if np.isnan(_pre_d2[0][i, j]):
                        use_eu = True
                        used_eu_any = True
                        src_lat, src_lon = c_lat_eu, c_lon_eu
                        src_ww, src_ceil = ww_eu, ceil_arr_eu
                        src_clcl, src_clcm, src_clch = c_clcl_eu, c_clcm_eu, c_clch_eu
                        src_cape, src_htop_dc = c_cape_eu, c_htop_dc_eu
                        src_hbas_sc, src_htop_sc = c_hbas_sc_eu, c_htop_sc_eu
                        src_lpi, src_hsurf = c_lpi_eu, c_hsurf_eu
                        src_sym_code, src_cb_hm = c_sym_code_eu, c_cb_hm_eu
                        cli_list = ctx.eu.lat_groups[i]
                        clo_list = ctx.eu.lon_groups[j]
                        cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                        clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

                if len(cli) == 0 or len(clo) == 0:
                    continue

                # Fast path: ingest-precomputed sym_code / cb_hm
                if src_sym_code is not None and src_cb_hm is not None:
                    cell_codes = src_sym_code[np.ix_(cli, clo)]
                    cell_cb = src_cb_hm[np.ix_(cli, clo)]
                    cell_ww = src_ww[np.ix_(cli, clo)]
                    max_ww = int(np.nanmax(cell_ww)) if not np.all(np.isnan(cell_ww)) else 0
                    best_ii = int(cli_list[len(cli_list) // 2])
                    best_jj = int(clo_list[len(clo_list) // 2])
                    sym = "clear"
                    cb_hm = None

                    flat_codes = cell_codes.ravel().astype(np.int16, copy=False)
                    valid = (flat_codes >= 0) & (flat_codes < SYMBOL_CODE_RANK_LUT.shape[0])
                    if np.any(valid):
                        valid_idx = np.flatnonzero(valid)
                        ranks = SYMBOL_CODE_RANK_LUT[flat_codes[valid_idx]]
                        keep = ranks >= 0
                        if np.any(keep):
                            best_k = int(np.argmin(ranks[keep]))
                            flat_idx = int(valid_idx[np.flatnonzero(keep)[best_k]])
                        else:
                            flat_idx = int(valid_idx[0])
                        ii_f, jj_f = np.unravel_index(flat_idx, cell_codes.shape)
                        code = int(cell_codes[ii_f, jj_f])
                        best_ii = int(cli[ii_f])
                        best_jj = int(clo[jj_f])
                        sym = SYMBOL_CODE_TO_TYPE.get(code, "clear")
                        vcb = int(cell_cb[ii_f, jj_f])
                        cb_hm = vcb if vcb >= 0 else None
                else:
                    # Legacy fallback: classify from raw arrays
                    _pre = _pre_eu if (use_eu and _pre_eu is not None) else _pre_d2
                    _pre_max_ww = float(_pre[0][i, j])
                    _pre_any_cape = bool(_pre[1][i, j])
                    _pre_any_ceil = bool(_pre[2][i, j])

                    if (not np.isnan(_pre_max_ww)) and _pre_max_ww <= 3 and not _pre_any_cape and not _pre_any_ceil:
                        sym, cb_hm = "clear", None
                        best_ii = int(cli_list[len(cli_list) // 2])
                        best_jj = int(clo_list[len(clo_list) // 2])
                        max_ww = int(_pre_max_ww) if np.isfinite(_pre_max_ww) else 0
                    else:
                        cell_ww = src_ww[np.ix_(cli, clo)]
                        max_ww = int(np.nanmax(cell_ww)) if not np.all(np.isnan(cell_ww)) else 0
                        sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                            cli=cli, clo=clo, cell_ww=cell_ww,
                            ceil_arr=src_ceil, c_clcl=src_clcl, c_clcm=src_clcm,
                            c_clch=src_clch, c_cape=src_cape, c_htop_dc=src_htop_dc,
                            c_hbas_sc=src_hbas_sc, c_htop_sc=src_htop_sc,
                            c_lpi=src_lpi, c_hsurf=src_hsurf,
                            classify_point_fn=_classify_point_core,
                            zoom=zoom,
                        )

                label = None
                if cb_hm is not None:
                    cb_hm = min(cb_hm, 99)
                    label = str(cb_hm)

                plot_lat = float(lat_c)
                plot_lon = float(lon_c)
                rep_lat = float(src_lat[best_ii])
                rep_lon = float(src_lon[best_jj])

                source_model = "icon_eu" if (use_eu or model_used == "icon_eu") else "icon_d2"
                if source_model == "icon_eu":
                    used_eu_any = True
                    used_eu_cells += 1
                else:
                    used_d2_any = True
                    used_d2_cells += 1

                symbols.append({
                    "lat": round(plot_lat, 4),
                    "lon": round(plot_lon, 4),
                    "clickLat": round(rep_lat, 4),
                    "clickLon": round(rep_lon, 4),
                    "type": sym,
                    "ww": max_ww,
                    "cloudBase": cb_hm,
                    "label": label,
                    "clickable": True,
                    "sourceModel": source_model,
                })

        t_agg_ms += (perf_counter() - t_agg0) * 1000.0

        if used_eu_any and used_d2_any:
            fallback_stats["symbolsBlended"] += 1

        effective_run = run
        effective_valid_time = d["validTime"]
        total_cells = used_eu_cells + used_d2_cells
        eu_share = (used_eu_cells / total_cells) if total_cells else 0.0
        significant_blend = used_eu_cells >= 3 and eu_share >= 0.03

        if used_eu_any and not used_d2_any:
            resolved_model = "icon_eu"
            if d_eu is not None:
                effective_run = d_eu.get("_run", run)
                effective_valid_time = d_eu.get("validTime", d["validTime"])
            fallback_decision = "eu_only_in_viewport"
        elif used_eu_any and used_d2_any and significant_blend:
            resolved_model = "ICON-D2 + EU"
            fallback_decision = "blended_d2_eu"
        elif used_eu_any and used_d2_any:
            resolved_model = model_used
            fallback_decision = "primary_model_with_eu_assist"
        else:
            resolved_model = model_used
            fallback_decision = "primary_model_only"

        _set_fallback_current(
            "symbols", fallback_decision,
            source_model=resolved_model, detail={"requestedTime": time},
        )

        result: Dict[str, Any] = {
            "symbols": symbols,
            "run": effective_run,
            "model": resolved_model,
            "validTime": effective_valid_time,
            "cellSize": cell_size,
            "count": len(symbols),
            "diagnostics": {
                "dataFreshnessMinutes": _freshness_minutes_from_run(effective_run),
                "fallbackDecision": fallback_decision,
                "requestedModel": model,
                "requestedTime": time,
                "sourceModel": resolved_model,
                "strictWindowHours": EU_STRICT_MAX_DELTA_HOURS,
                "euDataMissing": eu_data_missing,
                "euCells": used_eu_cells,
                "d2Cells": used_d2_cells,
                "euShare": round(eu_share, 4),
                "timingsMs": {
                    "load": round(t_load_ms, 2),
                    "grid": round(t_grid_ms, 2),
                    "aggregate": round(t_agg_ms, 2),
                },
                "servedFrom": "computed",
            },
        }
        symbols_cache_set(symbols_cache_key, result)
        total_ms = (perf_counter() - t0) * 1000.0
        logger.info(
            "/api/symbols rid=%s served=computed zoom=%s count=%s euCells=%s d2Cells=%s "
            "loadMs=%.2f gridMs=%.2f aggMs=%.2f totalMs=%.2f",
            rid, zoom, result["count"],
            result["diagnostics"]["euCells"], result["diagnostics"]["d2Cells"],
            t_load_ms, t_grid_ms, t_agg_ms, total_ms,
        )
        return result

    # ── /api/wind ─────────────────────────────────────────────────────────────

    @router.get("/api/wind")
    async def api_wind(
        zoom: int = Query(8, ge=5, le=12),
        bbox: str = Query("30,-30,72,45"),
        time: str = Query("latest"),
        model: Optional[str] = Query(None),
        level: str = Query("10m"),
    ):
        t0 = perf_counter()
        cell_size = CELL_SIZES_BY_ZOOM[zoom]

        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(400, "bbox: lat_min,lon_min,lat_max,lon_max")
        lat_min, lon_min, lat_max, lon_max = map(float, parts)

        gust_mode = level == "gust10m"
        u_key = "u_10m" if (level == "10m" or gust_mode) else f"u_{level}hpa"
        v_key = "v_10m" if (level == "10m" or gust_mode) else f"v_{level}hpa"

        run, step, model_used = resolve_time_with_cache_context(time, model)
        wind_keys = [u_key, v_key] + (["vmax_10m"] if gust_mode else [])
        d = load_data(run, step, model_used, keys=wind_keys)

        lat = d["lat"]
        lon = d["lon"]
        d2_lat_min = float(d.get("_latMin", np.min(lat)))
        d2_lat_max = float(d.get("_latMax", np.max(lat)))
        d2_lon_min = float(d.get("_lonMin", np.min(lon)))
        d2_lon_max = float(d.get("_lonMax", np.max(lon)))

        d_eu = None
        c_lat_eu = c_lon_eu = u_eu = v_eu = gust_eu = None

        if u_key not in d or v_key not in d or (gust_mode and "vmax_10m" not in d):
            return {
                "barbs": [], "run": run, "model": model_used,
                "validTime": d["validTime"], "level": level, "count": 0,
            }

        pad = cell_size
        li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
        if li is not None and len(li) == 0:
            c_lat = np.array([], dtype=float)
            c_lon = np.array([], dtype=float)
            u = v = np.zeros((0, 0), dtype=float)
            gust = None
        else:
            c_lat = lat[li] if li is not None else lat
            c_lon = lon[lo] if lo is not None else lon
            u = _slice_array(d[u_key], li, lo)
            v = _slice_array(d[v_key], li, lo)
            gust = _slice_array(d["vmax_10m"], li, lo) if gust_mode and "vmax_10m" in d else None

        wind_eu_data_missing = False
        if model_used == "icon_d2":
            needs_eu = (
                (lat_min - pad) < d2_lat_min or (lat_max + pad) > d2_lat_max
                or (lon_min - pad) < d2_lon_min or (lon_max + pad) > d2_lon_max
                or (bool(u.size) and (np.any(~np.isfinite(u)) or np.any(~np.isfinite(v))))
            )
            if needs_eu:
                eu_fb_wind = _load_eu_data_strict(time, wind_keys)
                if eu_fb_wind is not None and eu_fb_wind.get("missing"):
                    wind_eu_data_missing = True
                elif eu_fb_wind is not None:
                    d_eu = eu_fb_wind["data"]
                    if u_key in d_eu and v_key in d_eu and (not gust_mode or "vmax_10m" in d_eu):
                        lat_eu = d_eu["lat"]
                        lon_eu = d_eu["lon"]
                        li_eu, lo_eu = _bbox_indices(lat_eu, lon_eu, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
                        if not (li_eu is not None and len(li_eu) == 0):
                            c_lat_eu = lat_eu[li_eu] if li_eu is not None else lat_eu
                            c_lon_eu = lon_eu[lo_eu] if lo_eu is not None else lon_eu
                            u_eu = _slice_array(d_eu[u_key], li_eu, lo_eu)
                            v_eu = _slice_array(d_eu[v_key], li_eu, lo_eu)
                            gust_eu = _slice_array(d_eu["vmax_10m"], li_eu, lo_eu) if gust_mode and "vmax_10m" in d_eu else None

        ctx = build_grid_context(
            lat=lat, lon=lon, c_lat=c_lat, c_lon=c_lon,
            lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max,
            cell_size=cell_size, zoom=zoom,
            d2_lat_min=d2_lat_min, d2_lat_max=d2_lat_max,
            d2_lon_min=d2_lon_min, d2_lon_max=d2_lon_max,
            c_lat_eu=c_lat_eu if (c_lat_eu is not None and u_eu is not None and v_eu is not None) else None,
            c_lon_eu=c_lon_eu if (c_lon_eu is not None and u_eu is not None and v_eu is not None) else None,
        )

        barbs: List[dict] = []
        used_eu_any = False

        for i in range(ctx.lat_cell_count):
            for j in range(ctx.lon_cell_count):
                lat_lo, lat_hi = ctx.lat_edges[i], ctx.lat_edges[i + 1]
                lon_lo, lon_hi = ctx.lon_edges[j], ctx.lon_edges[j + 1]
                lat_c = (lat_lo + lat_hi) / 2
                lon_c = (lon_lo + lon_hi) / 2

                if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                    continue

                in_d2_domain = bool(ctx.in_d2_grid[i, j]) if ctx.in_d2_grid.size else False
                use_eu, cli_list, clo_list = choose_cell_groups(
                    ctx, i, j, prefer_eu=((not in_d2_domain) and (ctx.eu is not None)),
                )

                src_lat = c_lat_eu if use_eu else c_lat
                src_lon = c_lon_eu if use_eu else c_lon
                src_u = u_eu if use_eu else u
                src_v = v_eu if use_eu else v
                src_gust = gust_eu if use_eu else gust

                if use_eu:
                    used_eu_any = True

                cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

                cell_u = src_u[np.ix_(cli, clo)] if len(cli) and len(clo) else np.array([[]])
                cell_v = src_v[np.ix_(cli, clo)] if len(cli) and len(clo) else np.array([[]])
                mean_u = float(np.nanmean(cell_u))
                mean_v = float(np.nanmean(cell_v))

                # Per-cell EU fallback on NaN wind
                if (not use_eu) and (ctx.eu is not None) and (np.isnan(mean_u) or np.isnan(mean_v)):
                    used_eu_any = True
                    src_lat, src_lon = c_lat_eu, c_lon_eu
                    src_u, src_v = u_eu, v_eu
                    src_gust = gust_eu
                    cli_list = ctx.eu.lat_groups[i]
                    clo_list = ctx.eu.lon_groups[j]
                    cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                    clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)
                    if len(cli) == 0 or len(clo) == 0:
                        continue
                    cell_u = src_u[np.ix_(cli, clo)]
                    cell_v = src_v[np.ix_(cli, clo)]
                    mean_u = float(np.nanmean(cell_u))
                    mean_v = float(np.nanmean(cell_v))

                if np.isnan(mean_u) or np.isnan(mean_v):
                    continue

                if gust_mode and src_gust is not None and len(cli) and len(clo):
                    cell_g = src_gust[np.ix_(cli, clo)]
                    speed_ms = float(np.nanmax(cell_g)) if np.any(np.isfinite(cell_g)) else float("nan")
                else:
                    speed_ms = math.sqrt(mean_u ** 2 + mean_v ** 2)

                speed_kt = speed_ms * 1.94384
                dir_deg = (math.degrees(math.atan2(-mean_u, -mean_v)) + 360) % 360

                if speed_kt < 1:
                    continue

                rep_i = int(cli[len(cli) // 2])
                rep_j = int(clo[len(clo) // 2])
                plot_lat = float(src_lat[rep_i]) if zoom >= 12 else float(lat_c)
                plot_lon = float(src_lon[rep_j]) if zoom >= 12 else float(lon_c)
                barbs.append({
                    "lat": round(plot_lat, 4),
                    "lon": round(plot_lon, 4),
                    "speed_kt": round(speed_kt, 1),
                    "dir_deg": round(dir_deg, 0),
                    "speed_ms": round(speed_ms, 1),
                })

        if used_eu_any:
            fallback_stats["windBlended"] += 1

        resolved_model = "blended" if used_eu_any else model_used
        fallback_dec = "blended_d2_eu" if used_eu_any else "primary_model_only"
        _set_fallback_current(
            "wind", fallback_dec, source_model=resolved_model, detail={"requestedTime": time},
        )
        return {
            "barbs": barbs,
            "run": run,
            "model": resolved_model,
            "validTime": d["validTime"],
            "level": level,
            "count": len(barbs),
            "diagnostics": {
                "dataFreshnessMinutes": _freshness_minutes_from_run(run),
                "fallbackDecision": fallback_dec,
                "requestedModel": model,
                "requestedTime": time,
                "sourceModel": resolved_model,
                "euDataMissing": wind_eu_data_missing,
            },
        }

    # ── /api/emagram_point ────────────────────────────────────────────────────

    @router.get("/api/emagram_point")
    async def api_emagram_point(
        request: Request,
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        time: str = Query("latest"),
        model: Optional[str] = Query("icon_d2"),
        stream: bool = Query(False),
        _internal: bool = False,
    ):
        if stream and not _internal:
            async def _gen():
                cancel_event = threading.Event()
                yield json.dumps({"type": "progress", "message": "starting emagram"}) + "\n"
                task = asyncio.create_task(
                    run_in_threadpool(lambda: asyncio.run(
                        api_emagram_point(request=request, lat=lat, lon=lon, time=time,
                                          model=model, stream=False, _internal=True)
                    ))
                )
                while not task.done():
                    if await request.is_disconnected():
                        cancel_event.set()
                        task.cancel()
                        return
                    yield json.dumps({"type": "heartbeat", "message": "working"}) + "\n"
                    await asyncio.sleep(1.0)
                try:
                    payload = await task
                    yield json.dumps({"type": "done", "data": payload}) + "\n"
                except Exception as exc:
                    if cancel_event.is_set():
                        return
                    yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            return StreamingResponse(_gen(), media_type="application/x-ndjson")

        requested_model = model or "icon_d2"
        if requested_model not in ("icon_d2", "icon-d2"):
            raise HTTPException(400, "api_emagram_point currently supports model=icon_d2 only")

        run, step, model_used = resolve_time_with_cache_context(time, "icon_d2")

        keys = (
            [f"t_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
            + [f"fi_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
            + [f"relhum_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
            + [f"u_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
            + [f"v_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
        )
        d = load_data(run, step, model_used, keys=keys)
        lat_arr = d["lat"]
        lon_arr = d["lon"]
        if len(lat_arr) == 0 or len(lon_arr) == 0:
            raise HTTPException(404, "No grid coordinates available")

        i = int(np.argmin(np.abs(lat_arr - lat)))
        j = int(np.argmin(np.abs(lon_arr - lon)))

        def _dewpoint_c(temp_c: float, rh_pct: float) -> Optional[float]:
            if not np.isfinite(temp_c) or not np.isfinite(rh_pct):
                return None
            rh = max(1e-4, min(100.0, float(rh_pct)))
            a, b = 17.625, 243.04
            gamma = math.log(rh / 100.0) + (a * float(temp_c)) / (b + float(temp_c))
            td = (b * gamma) / (a - gamma)
            return float(td) if np.isfinite(td) else None

        levels: List[dict] = []
        for lev in EMAGRAM_D2_LEVELS_HPA:
            t_key, fi_key = f"t_{lev}hpa", f"fi_{lev}hpa"
            rh_key, u_key, v_key = f"relhum_{lev}hpa", f"u_{lev}hpa", f"v_{lev}hpa"
            t_val = d[t_key][i, j] if t_key in d else np.nan
            fi_val = d[fi_key][i, j] if fi_key in d else np.nan
            rh_val = d[rh_key][i, j] if rh_key in d else np.nan
            u_val = d[u_key][i, j] if u_key in d else np.nan
            v_val = d[v_key][i, j] if v_key in d else np.nan

            if not any(np.isfinite(x) for x in (t_val, fi_val, rh_val, u_val, v_val)):
                continue

            temp_c = (float(t_val) - 273.15) if np.isfinite(t_val) else None
            alt_m = (float(fi_val) / G0) if np.isfinite(fi_val) else None
            rh_pct = float(rh_val) if np.isfinite(rh_val) else None
            dew_c = _dewpoint_c(temp_c, rh_pct) if (temp_c is not None and rh_pct is not None) else None
            u_ms = float(u_val) if np.isfinite(u_val) else None
            v_ms = float(v_val) if np.isfinite(v_val) else None
            wind_ms = math.hypot(u_ms, v_ms) if (u_ms is not None and v_ms is not None) else None
            wind_kt = wind_ms * 1.943844 if wind_ms is not None else None
            wind_dir = (
                (270.0 - math.degrees(math.atan2(v_ms, u_ms))) % 360.0
                if (u_ms is not None and v_ms is not None) else None
            )
            levels.append({
                "pressureHpa": lev,
                "temperatureC": round(temp_c, 2) if temp_c is not None else None,
                "dewpointC": round(dew_c, 2) if dew_c is not None else None,
                "relativeHumidityPct": round(rh_pct, 1) if rh_pct is not None else None,
                "uMs": round(u_ms, 3) if u_ms is not None else None,
                "vMs": round(v_ms, 3) if v_ms is not None else None,
                "windSpeedMs": round(wind_ms, 2) if wind_ms is not None else None,
                "windSpeedKt": round(wind_kt, 1) if wind_kt is not None else None,
                "windDirDeg": round(wind_dir, 1) if wind_dir is not None else None,
                "geopotential": round(float(fi_val), 2) if np.isfinite(fi_val) else None,
                "altitudeM": round(alt_m, 1) if alt_m is not None else None,
            })

        levels.sort(key=lambda x: (x["altitudeM"] is None, x["altitudeM"] if x["altitudeM"] is not None else -x["pressureHpa"]))

        return {
            "model": model_used, "run": run, "step": step, "validTime": d.get("validTime"),
            "point": {
                "requestedLat": round(float(lat), 5), "requestedLon": round(float(lon), 5),
                "gridLat": round(float(lat_arr[i]), 5), "gridLon": round(float(lon_arr[j]), 5),
                "i": i, "j": j,
            },
            "levels": levels, "count": len(levels),
        }

    # ── /api/meteogram_point ──────────────────────────────────────────────────

    @router.get("/api/meteogram_point")
    async def api_meteogram_point(
        request: Request,
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        model: Optional[str] = Query("icon_d2"),
        stream: bool = Query(False),
        _internal: bool = False,
    ):
        if stream and not _internal:
            async def _gen():
                cancel_event = threading.Event()
                yield json.dumps({"type": "progress", "message": "starting meteogram"}) + "\n"
                task = asyncio.create_task(
                    run_in_threadpool(lambda: asyncio.run(
                        api_meteogram_point(request=request, lat=lat, lon=lon,
                                            model=model, stream=False, _internal=True)
                    ))
                )
                while not task.done():
                    if await request.is_disconnected():
                        cancel_event.set()
                        task.cancel()
                        return
                    yield json.dumps({"type": "heartbeat", "message": "working"}) + "\n"
                    await asyncio.sleep(1.0)
                try:
                    payload = await task
                    yield json.dumps({"type": "done", "data": payload}) + "\n"
                except Exception as exc:
                    if cancel_event.is_set():
                        return
                    yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            return StreamingResponse(_gen(), media_type="application/x-ndjson")

        merged = get_merged_timeline()
        if not merged or not merged.get("steps"):
            raise HTTPException(404, "No timeline available")

        m = (model or "icon_d2").replace("-", "_")
        if m != "icon_d2":
            raise HTTPException(400, "api_meteogram_point currently supports model=icon_d2 only")
        steps = [s for s in merged.get("steps", []) if s.get("model") == "icon_d2"]
        if not steps:
            raise HTTPException(404, "No timeline for model=icon_d2")

        run_key = str(steps[0].get("run") or "")
        cache_key = f"{m}|{run_key}|{round(float(lat), 4)}|{round(float(lon), 4)}"
        cached = meteogram_cache.get(cache_key)
        if cached is not None:
            meteogram_cache.move_to_end(cache_key)
            return cached

        level_keys: List[str] = []
        for lev in EMAGRAM_D2_LEVELS_HPA:
            level_keys += [f"u_{lev}hpa", f"v_{lev}hpa"]

        needed_keys = ["lat", "lon", "validTime", "tot_prec", "h_snow", "t_2m", "td_2m"] + level_keys

        out: List[dict] = []
        grid_point: Optional[dict] = None

        for s in steps:
            run_i, step_i, model_i = s.get("run"), int(s.get("step")), s.get("model")
            try:
                d = load_data(run_i, step_i, model_i, keys=needed_keys)
            except Exception:
                continue

            lat_arr = d.get("lat")
            lon_arr = d.get("lon")
            if lat_arr is None or lon_arr is None or len(lat_arr) == 0 or len(lon_arr) == 0:
                continue

            ii = int(np.argmin(np.abs(lat_arr - lat)))
            jj = int(np.argmin(np.abs(lon_arr - lon)))
            if grid_point is None:
                grid_point = {
                    "requestedLat": round(float(lat), 5), "requestedLon": round(float(lon), 5),
                    "gridLat": round(float(lat_arr[ii]), 5), "gridLon": round(float(lon_arr[jj]), 5),
                    "i": ii, "j": jj,
                }

            def _g(k: str) -> Optional[float]:
                arr = d.get(k)
                if arr is None:
                    return None
                try:
                    v = arr[ii, jj]
                except Exception:
                    return None
                return float(v) if np.isfinite(v) else None

            t2k = _g("t_2m")
            tdk = _g("td_2m")
            wind_levels: List[dict] = []
            for lev in EMAGRAM_D2_LEVELS_HPA:
                uu, vv = _g(f"u_{lev}hpa"), _g(f"v_{lev}hpa")
                if uu is None or vv is None:
                    wind_levels.append({"pressureHpa": lev, "speedKt": None, "dirDeg": None})
                    continue
                sp = math.hypot(uu, vv) * 1.943844
                dr = (270.0 - math.degrees(math.atan2(vv, uu))) % 360.0
                wind_levels.append({"pressureHpa": lev, "speedKt": round(sp, 1), "dirDeg": round(dr, 1)})

            out.append({
                "validTime": d.get("validTime") or s.get("validTime"),
                "model": model_i, "run": run_i, "step": step_i,
                "windLevels": wind_levels,
                "precipTotal": _g("tot_prec"),
                "snowDepthM": _g("h_snow"),
                "t2mC": round(t2k - 273.15, 2) if t2k is not None else None,
                "dewpoint2mC": round(tdk - 273.15, 2) if tdk is not None else None,
            })

        if not out:
            raise HTTPException(404, "No meteogram data available")

        out.sort(key=lambda r: r.get("validTime") or "")

        # De-accumulate precipitation
        prev_tot = prev_step = prev_run = None
        for r in out:
            tot = r.get("precipTotal")
            step_i = r.get("step")
            run_i = r.get("run")
            rate = None
            if tot is not None and prev_tot is not None and prev_step is not None and run_i == prev_run:
                dt_h = max(1, int(step_i) - int(prev_step))
                delta = float(tot) - float(prev_tot)
                if np.isfinite(delta):
                    rate = max(0.0, delta / float(dt_h))
            r["precipRateTotal"] = round(rate, 3) if rate is not None else None
            if tot is not None:
                prev_tot, prev_step, prev_run = float(tot), int(step_i), run_i

        payload = {"point": grid_point, "count": len(out), "series": out}
        meteogram_cache[cache_key] = payload
        meteogram_cache.move_to_end(cache_key)
        while len(meteogram_cache) > METEOGRAM_CACHE_MAX_ITEMS:
            meteogram_cache.popitem(last=False)
        return payload

    return router
