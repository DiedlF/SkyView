#!/usr/bin/env python3
"""Skyview FastAPI backend — serves ICON-D2 weather symbols for Leaflet frontend."""

import os
import sys
import math
import time
import atexit
import uuid
from time import perf_counter
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse

# Add backend dir to path for classify import
sys.path.insert(0, os.path.dirname(__file__))
from logging_config import setup_logging
from overlay_render import OVERLAY_CONFIGS as RENDER_OVERLAY_CONFIGS, colorize_layer_vectorized
from overlay_data import (
    build_overlay_keys,
    compute_computed_field_cropped,
    compute_computed_field_full,
    normalize_clouds_total_mod,
)
from weather_codes import ww_to_symbol
from symbol_logic import aggregate_symbol_cell
from point_data import build_overlay_values
from time_contract import get_available_runs as tc_get_available_runs, get_merged_timeline as tc_get_merged_timeline, resolve_time as tc_resolve_time
from grid_utils import bbox_indices as _bbox_indices, slice_array as _slice_array
from status_ops import build_status_payload, build_perf_payload
from feedback_ops import make_feedback_entry, append_feedback, read_feedback_list
from model_caps import get_models_payload
from response_headers import build_overlay_headers, build_tile_headers
from cache_state import (
    TILE_CACHE_MAX_ITEMS_DESKTOP, TILE_CACHE_MAX_ITEMS_MOBILE, TILE_CACHE_TTL_SECONDS,
    tile_cache_desktop, tile_cache_mobile, cache_stats, perf_recent, perf_totals, computed_field_cache,
    perf_record, computed_cache_get, computed_cache_set, tile_cache_prune, tile_cache_get, tile_cache_set,
    symbols_cache_get, symbols_cache_set, symbols_cache_stats_payload,
    rotate_caches_for_context, cache_context_stats_payload,
)

logger = setup_logging(__name__, level="INFO")

app = FastAPI(title="Skyview API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_error_counters = {"4xx": 0, "5xx": 0}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "requestId": rid}, headers={"X-Request-Id": rid})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    logger.exception(f"Unhandled error rid={rid}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "requestId": rid}, headers={"X-Request-Id": rid})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log API requests with method, path, and response time."""
    import time
    start_time = time.time()
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-Id"] = request_id
    
    if 400 <= response.status_code < 500:
        api_error_counters["4xx"] += 1
    elif response.status_code >= 500:
        api_error_counters["5xx"] += 1

    # Skip logging routine polling endpoints to reduce noise
    skip_paths = ['/api/timesteps', '/api/models']
    if request.url.path not in skip_paths or response.status_code >= 400:
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms - rid={request_id}"
        )
    
    return response


@app.on_event("startup")
async def startup_event():
    """Log server startup information."""
    logger.info("Skyview API server starting")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")
    
    # Check for available data
    runs = get_available_runs()
    logger.info(f"Found {len(runs)} available model runs")
    if runs:
        latest = runs[0]
        logger.info(f"Latest run: {latest['model']} {latest['run']} ({len(latest['steps'])} timesteps)")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FRONTEND_DIR = os.path.join(SCRIPT_DIR, "..", "frontend")
PID_FILE = os.path.join(SCRIPT_DIR, "logs", "skyview.pid")

data_cache: Dict[str, Dict[str, Any]] = {}


def _acquire_single_instance_or_exit(pid_file: str):
    """Simple PID-file guard to prevent accidental multi-process launches."""
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)

    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            if old_pid > 0:
                os.kill(old_pid, 0)  # check process exists
                raise SystemExit(f"Skyview backend already running with pid {old_pid} (pid file: {pid_file})")
        except ProcessLookupError:
            # stale pid file -> continue and overwrite
            pass
        except ValueError:
            # malformed pid file -> overwrite
            pass

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    def _cleanup_pid_file():
        try:
            if os.path.exists(pid_file):
                with open(pid_file, "r") as pf:
                    cur = pf.read().strip()
                if cur == str(os.getpid()):
                    os.remove(pid_file)
        except Exception:
            pass

    atexit.register(_cleanup_pid_file)

# ─── Grid / Bbox helpers ───

def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling):
    """Classify a single grid point's cloud type.

    Non-convective cloud classification is based solely on ceiling:
    - no valid ceiling (>0): clear
    - ceiling < 2500m: st
    - 2500..7000m: ac
    - >7000m: ci
    """
    # Convective branch
    is_conv = cape_ml > 50
    if is_conv:
        cloud_depth = max(0.0, htop_sc - hbas_sc) if np.isfinite(htop_sc) and np.isfinite(hbas_sc) else 0.0
        if hbas_sc <= 0 or clcl < 5:
            return "blue_thermal"
        if lpi > 0 or cloud_depth > 4000 or cape_ml > 1000:
            return "cb"
        if cloud_depth > 500:
            return "cu_con"
        return "cu_hum"

    # Non-convective: ceiling-only logic
    if not np.isfinite(ceiling) or ceiling <= 0 or ceiling >= 20000:
        return "clear"
    if ceiling < 2500:
        return "st"
    if ceiling > 7000:
        return "ci"
    return "ac"


def load_data(run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load .npz data for a given run/step/model.
    
    Args:
        keys: If provided, only load these keys (plus lat/lon). Saves memory for large grids.
              If None, loads all keys (backward compat).
    """
    cache_key = f"{model}/{run}/{step:03d}"
    
    # Check cache — if we have it and it has all requested keys, use it
    if cache_key in data_cache:
        cached = data_cache[cache_key]
        if keys is None or all(k in cached for k in keys):
            logger.debug(f"Cache hit: {cache_key}")
            return cached

    model_dir = model.replace("_", "-")
    path = os.path.join(DATA_DIR, model_dir, run, f"{step:03d}.npz")
    if not os.path.exists(path):
        logger.error(f"Data not found: {path}")
        raise FileNotFoundError(f"Data not found: {path}")

    logger.debug(f"Loading data: {cache_key}" + (f" (keys: {len(keys)})" if keys else " (all)"))
    npz = np.load(path)
    
    if keys is not None:
        # Selective loading: always include lat/lon + requested keys
        load_keys = set(keys) | {"lat", "lon"}
        arrays = {k: npz[k] for k in load_keys if k in npz.files}
        # Also load any keys already in cache
        if cache_key in data_cache:
            for k, v in data_cache[cache_key].items():
                if k not in arrays:
                    arrays[k] = v
    else:
        arrays = {k: npz[k] for k in npz.files}

    # Compute valid time
    run_dt = datetime.strptime(run, "%Y%m%d%H")
    valid_dt = run_dt + timedelta(hours=step)
    arrays["validTime"] = valid_dt.isoformat() + "Z"
    arrays["_run"] = run
    arrays["_step"] = step

    data_cache[cache_key] = arrays
    return arrays


def get_available_runs():
    return tc_get_available_runs(DATA_DIR)


def get_merged_timeline():
    return tc_get_merged_timeline(DATA_DIR)


def resolve_time(time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
    return tc_resolve_time(DATA_DIR, time_str, model)


def resolve_time_with_cache_context(time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
    run, step, model_used = resolve_time(time_str, model)
    rotate_caches_for_context(f"{model_used}|{run}|{step}")
    return run, step, model_used


# ─── API Endpoints ───


@app.get("/api/health")
async def health():
    runs = get_available_runs()
    return {"status": "ok", "runs": len(runs), "cache": len(data_cache)}


@app.get("/api/models")
async def api_models():
    """Return model capabilities for frontend timestep filtering."""
    return get_models_payload()


@app.get("/api/timesteps")
async def api_timesteps():
    merged = get_merged_timeline()
    return {"runs": get_available_runs(), "merged": merged}


@app.get("/api/symbols")
async def api_symbols(
    zoom: int = Query(8, ge=5, le=12),
    bbox: str = Query("43.18,-3.94,58.08,20.34"),
    time: str = Query("latest"),
    model: Optional[str] = Query(None),
):
    cell_sizes = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.02}
    cell_size = cell_sizes[zoom]

    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(400, "bbox: lat_min,lon_min,lat_max,lon_max")
    lat_min, lon_min, lat_max, lon_max = map(float, parts)

    # Keys needed for symbols endpoint
    symbol_keys = ["ww", "ceiling", "clcl", "clcm", "clch", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi"]
    run, step, model_used = resolve_time_with_cache_context(time, model)

    # Short-TTL response cache for repeated pan/zoom requests
    cache_bbox = f"{lat_min:.4f},{lon_min:.4f},{lat_max:.4f},{lon_max:.4f}"
    symbols_cache_key = f"{model_used}|{run}|{step}|z{zoom}|{cache_bbox}"
    cached_symbols = symbols_cache_get(symbols_cache_key)
    if cached_symbols is not None:
        return cached_symbols

    d = load_data(run, step, model_used, keys=symbol_keys)

    lat = d["lat"]  # 1D
    lon = d["lon"]  # 1D

    # Bbox-slice BEFORE heavy computation.
    # Use one-cell padding around viewport to stabilize aggregation at screen borders during pan.
    pad = cell_size
    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is not None and len(li) == 0:
        result = {"symbols": [], "run": run, "model": model_used, "validTime": d["validTime"], "cellSize": cell_size, "count": 0}
        symbols_cache_set(symbols_cache_key, result)
        return result

    # Work with cropped arrays
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
    c_lpi = _slice_array(d["lpi"], li, lo) if "lpi" in d else np.zeros_like(ww)

    # **GLOBAL FIXED GRID** for absolute stability
    # Use data grid origin as fixed anchor (stable across requests)
    # Align grid to global anchor (data origin) but start from bbox for performance
    anchor_lat = float(lat.min())
    anchor_lon = float(lon.min())
    # At highest zoom, shift anchor by half cell so symbol centers match native pixel centers
    # while preserving strict equidistant spacing.
    if zoom >= 12:
        anchor_lat -= cell_size / 2.0
        anchor_lon -= cell_size / 2.0
    lat_start = anchor_lat + np.floor((lat_min - anchor_lat) / cell_size) * cell_size
    lon_start = anchor_lon + np.floor((lon_min - anchor_lon) / cell_size) * cell_size
    lat_edges = np.arange(lat_start, lat_max + cell_size, cell_size)
    lon_edges = np.arange(lon_start, lon_max + cell_size, cell_size)

    # Pre-bin cropped lat/lon indices onto aggregation grid once (avoids per-cell mask scans)
    lat_cell_count = max(0, len(lat_edges) - 1)
    lon_cell_count = max(0, len(lon_edges) - 1)
    lat_groups = [[] for _ in range(lat_cell_count)]
    lon_groups = [[] for _ in range(lon_cell_count)]

    for idx, v in enumerate(c_lat):
        bi = int(np.floor((float(v) - lat_start) / cell_size))
        if 0 <= bi < lat_cell_count:
            lat_groups[bi].append(idx)
    for idx, v in enumerate(c_lon):
        bj = int(np.floor((float(v) - lon_start) / cell_size))
        if 0 <= bj < lon_cell_count:
            lon_groups[bj].append(idx)

    symbols = []
    for i in range(lat_cell_count):
        for j in range(lon_cell_count):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2

            # Skip cells outside bbox
            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            cli_list = lat_groups[i]
            clo_list = lon_groups[j]
            cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
            clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

            if len(cli) == 0 or len(clo) == 0:
                # Avoid artificial "clear stripes" from boundary/rounding gaps:
                # fall back to nearest available data index per axis.
                if len(c_lat) == 0 or len(c_lon) == 0:
                    continue
                if len(cli) == 0:
                    cli = np.array([int(np.argmin(np.abs(c_lat - lat_c)))], dtype=int)
                if len(clo) == 0:
                    clo = np.array([int(np.argmin(np.abs(c_lon - lon_c)))], dtype=int)

            # Extract cell data
            cell_ww = ww[np.ix_(cli, clo)]
            max_ww = int(np.nanmax(cell_ww)) if not np.all(np.isnan(cell_ww)) else 0

            # Fast-path for likely clear/non-convective low-zoom cells to reduce first-hit latency.
            if zoom <= 9 and max_ww <= 3:
                cell_cape = c_cape[np.ix_(cli, clo)]
                conv_signal = np.any(np.isfinite(cell_cape) & (cell_cape > 50))
                cell_ceil = ceil_arr[np.ix_(cli, clo)]
                ceil_valid = cell_ceil[np.isfinite(cell_ceil) & (cell_ceil > 0) & (cell_ceil < 20000)]
                if (not conv_signal) and len(ceil_valid) == 0:
                    sym, cb_hm = "clear", None
                    best_ii, best_jj = int(cli[len(cli) // 2]), int(clo[len(clo) // 2])
                else:
                    sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                        cli=cli,
                        clo=clo,
                        cell_ww=cell_ww,
                        ceil_arr=ceil_arr,
                        c_clcl=c_clcl,
                        c_clcm=c_clcm,
                        c_clch=c_clch,
                        c_cape=c_cape,
                        c_htop_dc=c_htop_dc,
                        c_hbas_sc=c_hbas_sc,
                        c_htop_sc=c_htop_sc,
                        c_lpi=c_lpi,
                        classify_point_fn=classify_point,
                        zoom=zoom,
                    )
            else:
                # Determine aggregated symbol in helper module
                sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                    cli=cli,
                    clo=clo,
                    cell_ww=cell_ww,
                    ceil_arr=ceil_arr,
                    c_clcl=c_clcl,
                    c_clcm=c_clcm,
                    c_clch=c_clch,
                    c_cape=c_cape,
                    c_htop_dc=c_htop_dc,
                    c_hbas_sc=c_hbas_sc,
                    c_htop_sc=c_htop_sc,
                    c_lpi=c_lpi,
                    classify_point_fn=classify_point,
                    zoom=zoom,
                )
            label = None

            # Label formatting
            if cb_hm is not None:
                if cb_hm > 99:
                    cb_hm = 99
                label = str(cb_hm)

            # Placement strategy:
            # Render symbols on equidistant aggregation grid centers at all zoom levels.
            # (At z12, grid anchor is half-cell shifted above to align centers with native pixels.)
            rep_lat = float(c_lat[best_ii])
            rep_lon = float(c_lon[best_jj])
            plot_lat = float(lat_c)
            plot_lon = float(lon_c)

            symbols.append({
                "lat": round(plot_lat, 4),
                "lon": round(plot_lon, 4),
                "clickLat": round(rep_lat, 4),
                "clickLon": round(rep_lon, 4),
                "type": sym,
                "ww": max_ww,
                "cloudBase": cb_hm,
                "label": label,
                "clickable": True
            })

    result = {
        "symbols": symbols,
        "run": run,
        "model": model_used,
        "validTime": d["validTime"],
        "cellSize": cell_size,
        "count": len(symbols)
    }
    symbols_cache_set(symbols_cache_key, result)
    return result


@app.get("/api/wind")
async def api_wind(
    zoom: int = Query(8, ge=5, le=12),
    bbox: str = Query("43.18,-3.94,58.08,20.34"),
    time: str = Query("latest"),
    model: Optional[str] = Query(None),
    level: str = Query("10m"),
):
    """Return wind barb data on the same grid as convection symbols."""
    cell_sizes = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.02}
    cell_size = cell_sizes[zoom]

    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(400, "bbox: lat_min,lon_min,lat_max,lon_max")
    lat_min, lon_min, lat_max, lon_max = map(float, parts)

    # Select wind variables based on level
    if level == "10m":
        u_key, v_key = "u_10m", "v_10m"
    else:
        u_key, v_key = f"u_{level}hpa", f"v_{level}hpa"

    run, step, model_used = resolve_time_with_cache_context(time, model)
    d = load_data(run, step, model_used, keys=[u_key, v_key])

    lat = d["lat"]
    lon = d["lon"]

    # Check if wind data is available
    if u_key not in d or v_key not in d:
        return {"barbs": [], "run": run, "model": model_used, "validTime": d["validTime"], "level": level, "count": 0}

    # Bbox-slice before computation.
    # Use one-cell padding around viewport to stabilize border cells during pan.
    pad = cell_size
    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is not None and len(li) == 0:
        return {"barbs": [], "run": run, "model": model_used, "validTime": d["validTime"], "level": level, "count": 0}

    c_lat = lat[li] if li is not None else lat
    c_lon = lon[lo] if lo is not None else lon
    u = _slice_array(d[u_key], li, lo)
    v = _slice_array(d[v_key], li, lo)

    # Same fixed grid as symbols for alignment
    # Align grid to global anchor (data origin) but start from bbox for performance
    anchor_lat = float(lat.min())
    anchor_lon = float(lon.min())
    if zoom >= 12:
        anchor_lat -= cell_size / 2.0
        anchor_lon -= cell_size / 2.0
    lat_start = anchor_lat + np.floor((lat_min - anchor_lat) / cell_size) * cell_size
    lon_start = anchor_lon + np.floor((lon_min - anchor_lon) / cell_size) * cell_size
    lat_edges = np.arange(lat_start, lat_max + cell_size, cell_size)
    lon_edges = np.arange(lon_start, lon_max + cell_size, cell_size)

    barbs = []
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2

            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            lat_mask = (c_lat >= lat_lo) & (c_lat < lat_hi)
            lon_mask = (c_lon >= lon_lo) & (c_lon < lon_hi)
            cli = np.where(lat_mask)[0]
            clo = np.where(lon_mask)[0]

            if len(cli) == 0 or len(clo) == 0:
                continue

            cell_u = u[np.ix_(cli, clo)]
            cell_v = v[np.ix_(cli, clo)]
            mean_u = float(np.nanmean(cell_u))
            mean_v = float(np.nanmean(cell_v))

            if np.isnan(mean_u) or np.isnan(mean_v):
                continue

            # Wind speed in m/s → knots (1 m/s = 1.94384 kt)
            speed_ms = math.sqrt(mean_u ** 2 + mean_v ** 2)
            speed_kt = speed_ms * 1.94384

            # Wind direction: meteorological convention (direction wind comes FROM)
            # atan2(-u, -v) gives direction in radians from North, clockwise
            dir_deg = (math.degrees(math.atan2(-mean_u, -mean_v)) + 360) % 360

            if speed_kt < 1:  # Calm wind
                continue

            # Match symbol placement strategy:
            # - highest zoom: representative data point
            # - lower zoom: strict equidistant aggregation grid
            rep_i = int(cli[len(cli) // 2])
            rep_j = int(clo[len(clo) // 2])
            plot_lat = float(c_lat[rep_i]) if zoom >= 12 else float(lat_c)
            plot_lon = float(c_lon[rep_j]) if zoom >= 12 else float(lon_c)
            barbs.append({
                "lat": round(plot_lat, 4),
                "lon": round(plot_lon, 4),
                "speed_kt": round(speed_kt, 1),
                "dir_deg": round(dir_deg, 0),
                "speed_ms": round(speed_ms, 1),
            })

    return {
        "barbs": barbs,
        "run": run,
        "model": model_used,
        "validTime": d["validTime"],
        "level": level,
        "count": len(barbs)
    }


@app.get("/api/point")
async def api_point(
    lat: float = Query(...),
    lon: float = Query(...),
    time: str = Query("latest"),
    model: Optional[str] = Query(None),
    wind_level: str = Query("10m"),
    zoom: Optional[int] = Query(None, ge=5, le=12),
):
    # Load all keys for point endpoint (needs many variables for overlay_values)
    run, step, model_used = resolve_time_with_cache_context(time, model)
    d = load_data(run, step, model_used)

    # Match nearest model gridpoint to avoid empty-cell errors at later timesteps
    lat_arr = d["lat"]
    lon_arr = d["lon"]
    li0 = int(np.argmin(np.abs(lat_arr - lat)))
    lo0 = int(np.argmin(np.abs(lon_arr - lon)))
    li = np.array([li0], dtype=int)
    lo = np.array([lo0], dtype=int)

    vars_out = ["ww", "clcl", "clcm", "clch", "clct", "cape_ml",
                "htop_dc", "hbas_sc", "htop_sc", "lpi", "ceiling"]
    result = {}

    def scalar(vname, reducer="max"):
        if vname not in d:
            return None
        cell = d[vname][np.ix_(li, lo)]
        val = float(np.nanmax(cell)) if reducer == "max" else float(np.nanmean(cell))
        return None if np.isnan(val) else val

    for v in vars_out:
        val = scalar(v, "max")
        if val is None:
            result[v] = None
        elif v == "ceiling" and val > 20000:
            result[v] = None
        else:
            result[v] = round(val, 1)

    clcl = scalar("clcl") or 0.0
    clcm = scalar("clcm") or 0.0
    clch = scalar("clch") or 0.0
    cape_ml = scalar("cape_ml") or 0.0
    htop_dc = scalar("htop_dc") or 0.0
    hbas_sc = scalar("hbas_sc") or 0.0
    htop_sc = scalar("htop_sc") or 0.0
    lpi = scalar("lpi") or 0.0

    ceiling_val = scalar("ceiling") or 0.0
    best_type = classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling_val)
    result["cloudType"] = best_type

    ww_max = int(scalar("ww") or 0)
    ww_sym = ww_to_symbol(ww_max) if ww_max > 10 else None
    sym = ww_sym or best_type
    result["symbol"] = sym
    result["cloudTypeName"] = sym.title()
    result["ww"] = ww_max

    # Cloud base: mirror symbol label logic
    ceil_cell = d["ceiling"][np.ix_(li, lo)] if "ceiling" in d else np.array([])
    htop_cell = d["htop_dc"][np.ix_(li, lo)] if "htop_dc" in d else np.array([])
    hbas_cell = d["hbas_sc"][np.ix_(li, lo)] if "hbas_sc" in d else np.array([])
    valid_ceil = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20000)] if ceil_cell.size else np.array([])
    valid_hbas = hbas_cell[(hbas_cell > 0) & np.isfinite(hbas_cell)] if hbas_cell.size else np.array([])

    if sym in ("cu_hum", "cu_con", "cb") and len(valid_hbas) > 0:
        result["cloudBaseHm"] = int(np.min(valid_hbas) / 100)
    elif sym in ("st", "ac", "ci", "fog", "rime_fog") and len(valid_ceil) > 0:
        result["cloudBaseHm"] = int(np.min(valid_ceil) / 100)
    elif sym == "blue_thermal" and htop_cell.size and np.any(htop_cell > 0):
        result["cloudBaseHm"] = int(np.max(htop_cell[htop_cell > 0]) / 100)
    else:
        result["cloudBaseHm"] = None

    # Overlay values at this point (for displaying active overlay info)
    overlay_values = build_overlay_values(
        d=d,
        li=li,
        lo=lo,
        ww_max=ww_max,
        ceil_cell=ceil_cell,
        wind_level=wind_level,
        zoom=zoom,
        lat=lat,
        lon=lon,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
    )
    result["overlay_values"] = overlay_values

    # Explorer/Skyview contract convergence: provide raw values dictionary as well
    result["values"] = {k: result.get(k) for k in [
        "ww", "clcl", "clcm", "clch", "clct", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi", "ceiling"
    ]}

    # Closest point lat/lon for precision
    li0 = int(np.argmin(np.abs(lat_arr - lat)))
    lo0 = int(np.argmin(np.abs(lon_arr - lon)))
    result["lat"] = round(float(lat_arr[li0]), 4)
    result["lon"] = round(float(lon_arr[lo0]), 4)
    result["validTime"] = d["validTime"]
    result["run"] = run
    result["model"] = model_used

    return result


# ─── Feedback endpoint ───

FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

@app.post("/api/feedback")
async def api_feedback(request: Request):
    """Store user feedback with context."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    message = str(body.get("message", "")).strip()
    if not message:
        raise HTTPException(400, "Message is required")

    entry = make_feedback_entry(body)
    append_feedback(FEEDBACK_FILE, entry)

    logger.info(f"Feedback received: [{entry['type']}] {message[:80]}")
    return {"status": "ok", "id": entry["id"]}


@app.get("/api/feedback")
async def api_feedback_list():
    """List all feedback entries (for admin use)."""
    feedback = read_feedback_list(FEEDBACK_FILE)
    return {"feedback": feedback, "count": len(feedback)}


# ─── Overlay endpoint ───

# Overlay image helpers
import io
from PIL import Image

# Phase-2 modularization: use shared overlay registry from backend/overlay_render.py
OVERLAY_CONFIGS = RENDER_OVERLAY_CONFIGS


@app.get("/api/overlay")
async def api_overlay(
    layer: str = Query(...),
    bbox: str = Query("43.18,-3.94,58.08,20.34"),
    time: str = Query("latest"),
    model: Optional[str] = Query(None),
    width: int = Query(400, ge=50, le=1200),
):
    if layer not in OVERLAY_CONFIGS:
        raise HTTPException(400, f"Unknown layer: {layer}. Available: {list(OVERLAY_CONFIGS.keys())}")

    cfg = OVERLAY_CONFIGS[layer]
    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(400, "bbox: lat_min,lon_min,lat_max,lon_max")
    lat_min, lon_min, lat_max, lon_max = map(float, parts)

    # Determine which keys to load based on layer
    overlay_keys = build_overlay_keys(cfg)

    run, step, model_used = resolve_time_with_cache_context(time, model)
    d = load_data(run, step, model_used, keys=overlay_keys)

    lat = d["lat"]
    lon = d["lon"]
    cmap_fn = cfg["cmap"]

    # Clamp bbox to actual data extent to prevent stretching
    data_lat_min, data_lat_max = float(lat.min()), float(lat.max())
    data_lon_min, data_lon_max = float(lon.min()), float(lon.max())
    
    lat_min = max(lat_min, data_lat_min)
    lat_max = min(lat_max, data_lat_max)
    lon_min = max(lon_min, data_lon_min)
    lon_max = min(lon_max, data_lon_max)
    
    if lat_min >= lat_max or lon_min >= lon_max:
        raise HTTPException(404, "No data in bbox")

    # Expand bbox to at least one grid cell if needed
    grid_res = float(lat[1] - lat[0]) if len(lat) > 1 else 0.02
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span < abs(grid_res):
        lat_mid = (lat_min + lat_max) / 2
        lat_min = lat_mid - abs(grid_res) / 2
        lat_max = lat_mid + abs(grid_res) / 2
    if lon_span < abs(grid_res):
        lon_mid = (lon_min + lon_max) / 2
        lon_min = lon_mid - abs(grid_res) / 2
        lon_max = lon_mid + abs(grid_res) / 2
    
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]

    if len(li) == 0 or len(lo) == 0:
        raise HTTPException(404, "No data in bbox")

    # Calculate actual grid cell bounds
    # Grid coordinates are cell centers; we need cell edges for image bounds.
    # Use abs() and min/max to remain correct for either coordinate ordering.
    lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.02
    c_lat = lat[li]
    c_lon = lon[lo]

    # Actual bounds = edges of selected cells (not requested bbox)
    actual_lat_min = float(np.min(c_lat)) - lat_res / 2
    actual_lat_max = float(np.max(c_lat)) + lat_res / 2
    actual_lon_min = float(np.min(c_lon)) - lon_res / 2
    actual_lon_max = float(np.max(c_lon)) + lon_res / 2

    # Handle computed variables
    if cfg.get("computed"):
        cropped = compute_computed_field_cropped(cfg["var"], d, li, lo)
        h, w = cropped.shape
    else:
        var_name = cfg["var"]
        if var_name not in d and layer == "clouds_total_mod" and "clct" in d:
            var_name = "clct"  # fallback when modified total cloud is unavailable
        if var_name not in d:
            raise HTTPException(404, f"Variable {cfg['var']} not available for this timestep")
        var_data = d[var_name]
        cropped = var_data[np.ix_(li, lo)]
        if layer == "clouds_total_mod":
            cropped = normalize_clouds_total_mod(cropped)
        h, w = cropped.shape

    # Reproject rows to Web-Mercator spacing before rasterization.
    # Without this, a lat/lon-spaced raster in an EPSG:3857 map appears to drift
    # slightly relative to basemap during zoom/pan.
    lat_src = np.asarray(c_lat, dtype=np.float64)
    data_src = np.asarray(cropped, dtype=np.float64)

    # Ensure south->north ordering for interpolation
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        data_src = data_src[::-1, :]

    def merc_y(lat_deg: np.ndarray) -> np.ndarray:
        lat_clamped = np.clip(lat_deg, -85.05112878, 85.05112878)
        lat_rad = np.deg2rad(lat_clamped)
        return np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))

    y_src = merc_y(lat_src)
    y_min = float(merc_y(np.array([actual_lat_min]))[0])
    y_max = float(merc_y(np.array([actual_lat_max]))[0])

    # Keep row count, remap to equal Web-Mercator Y spacing
    target_rows = max(2, data_src.shape[0])
    y_tgt = np.linspace(y_min, y_max, target_rows)
    nearest_idx = np.searchsorted(y_src, y_tgt)
    nearest_idx = np.clip(nearest_idx, 0, len(y_src) - 1)
    left_idx = np.clip(nearest_idx - 1, 0, len(y_src) - 1)
    choose_left = np.abs(y_tgt - y_src[left_idx]) <= np.abs(y_tgt - y_src[nearest_idx])
    row_idx = np.where(choose_left, left_idx, nearest_idx)
    cropped_merc = data_src[row_idx, :]

    # Compute output dimensions using Web-Mercator aspect ratio
    x_min = math.radians(actual_lon_min)
    x_max = math.radians(actual_lon_max)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    aspect = x_span / y_span
    out_w = width
    out_h = max(1, int(out_w / aspect))

    # Create RGBA image at native mercator-reprojected resolution, then resize
    h_merc, w_merc = cropped_merc.shape
    img = Image.new("RGBA", (w_merc, h_merc), (0, 0, 0, 0))
    pixels = img.load()

    # Single variable rendering
    for y in range(h_merc):
        for x in range(w_merc):
            val = cropped_merc[y, x]
            if np.isnan(val):
                continue
            color = cmap_fn(val)
            if color:
                pixels[x, h_merc - 1 - y] = color

    # Resize to output — always nearest-neighbor for crisp pixelated look
    img = img.resize((out_w, out_h), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers=build_overlay_headers(
            run=run,
            valid_time=d["validTime"],
            model=model_used,
            bbox=f"{actual_lat_min},{actual_lon_min},{actual_lat_max},{actual_lon_max}",
        ),
    )


def _tile_bounds_3857(z: int, x: int, y: int):
    origin = 20037508.342789244
    world = origin * 2.0
    tile_size = world / (2 ** z)
    minx = -origin + x * tile_size
    maxx = minx + tile_size
    maxy = origin - y * tile_size
    miny = maxy - tile_size
    return minx, miny, maxx, maxy


def _merc_to_lonlat(mx, my):
    lon = (mx / 20037508.342789244) * 180.0
    lat = np.degrees(2.0 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2.0)
    return lon, lat


def _regular_grid_indices(vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(vals) < 2:
        return np.zeros_like(q, dtype=np.int32)
    step = float(vals[1] - vals[0])
    if step == 0:
        return np.zeros_like(q, dtype=np.int32)
    idx = np.rint((q - float(vals[0])) / step).astype(np.int32)
    return np.clip(idx, 0, len(vals) - 1)


@app.get("/api/status")
async def api_status():
    """Aggregate ingest freshness, model/run state, cache/perf metrics and error counters."""
    runs = get_available_runs()
    merged = get_merged_timeline()
    return build_status_payload(
        runs=runs,
        merged=merged,
        tile_cache_prune_fn=tile_cache_prune,
        tile_cache_desktop=tile_cache_desktop,
        tile_cache_mobile=tile_cache_mobile,
        tile_cache_max_desktop=TILE_CACHE_MAX_ITEMS_DESKTOP,
        tile_cache_max_mobile=TILE_CACHE_MAX_ITEMS_MOBILE,
        tile_cache_ttl=TILE_CACHE_TTL_SECONDS,
        cache_stats=cache_stats,
        computed_field_cache=computed_field_cache,
        symbols_cache_stats=symbols_cache_stats_payload(),
        cache_context_stats=cache_context_stats_payload(),
        perf_recent=perf_recent,
        perf_totals=perf_totals,
        api_error_counters=api_error_counters,
    )


@app.get("/api/cache_stats")
async def api_cache_stats():
    tile_cache_prune("desktop")
    tile_cache_prune("mobile")
    return {
        "tileCache": {
            "desktopItems": len(tile_cache_desktop),
            "desktopMax": TILE_CACHE_MAX_ITEMS_DESKTOP,
            "mobileItems": len(tile_cache_mobile),
            "mobileMax": TILE_CACHE_MAX_ITEMS_MOBILE,
            "ttlSeconds": TILE_CACHE_TTL_SECONDS,
        },
        "metrics": cache_stats,
        "computedFieldCacheItems": len(computed_field_cache),
    }


@app.get("/api/perf_stats")
async def api_perf_stats(reset: bool = Query(False, description="Reset perf counters after returning stats")):
    payload = build_perf_payload(perf_recent, perf_totals)

    if reset:
        perf_recent.clear()
        perf_totals['requests'] = 0
        perf_totals['hits'] = 0
        perf_totals['misses'] = 0
        perf_totals['totalMs'] = 0.0
        payload['reset'] = True

    return payload


@app.get("/api/overlay_tile/{z}/{x}/{y}.png")
async def api_overlay_tile(
    z: int,
    x: int,
    y: int,
    layer: str = Query(...),
    time: str = Query("latest"),
    model: Optional[str] = Query(None),
    clientClass: str = Query("desktop"),
):
    t0 = perf_counter()

    if layer not in OVERLAY_CONFIGS:
        raise HTTPException(400, f"Unknown layer: {layer}")

    client_class = "mobile" if clientClass == "mobile" else "desktop"
    cfg = OVERLAY_CONFIGS[layer]

    overlay_keys = build_overlay_keys(cfg)

    run, step, model_used = resolve_time_with_cache_context(time, model)
    d = load_data(run, step, model_used, keys=overlay_keys)
    lat = d["lat"]
    lon = d["lon"]

    cache_key = f"{client_class}|{model_used}|{run}|{step}|{layer}|{z}|{x}|{y}"
    cached = tile_cache_get(client_class, cache_key)
    if cached is not None:
        perf_record((perf_counter() - t0) * 1000.0, True)
        return Response(
            content=cached,
            media_type="image/png",
            headers=build_tile_headers(
                run=run,
                valid_time=d["validTime"],
                model=model_used,
                cache="HIT",
            ),
        )

    minx, miny, maxx, maxy = _tile_bounds_3857(z, x, y)
    lon0, lat0 = _merc_to_lonlat(minx, miny)
    lon1, lat1 = _merc_to_lonlat(maxx, maxy)
    lon_min, lon_max = min(lon0, lon1), max(lon0, lon1)
    lat_min, lat_max = min(lat0, lat1), max(lat0, lat1)

    data_lat_min, data_lat_max = float(np.min(lat)), float(np.max(lat))
    data_lon_min, data_lon_max = float(np.min(lon)), float(np.max(lon))
    if lat_max < data_lat_min or lat_min > data_lat_max or lon_max < data_lon_min or lon_min > data_lon_max:
        empty = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        b = io.BytesIO(); empty.save(b, format="PNG", optimize=True); png = b.getvalue()
        tile_cache_set(client_class, cache_key, png)
        perf_record((perf_counter() - t0) * 1000.0, False)
        return Response(
            content=png,
            media_type="image/png",
            headers=build_tile_headers(
                run=run,
                valid_time=d["validTime"],
                model=model_used,
                cache="MISS",
            ),
        )

    # source field (full grid), with memoization for expensive computed layers
    if cfg.get("computed"):
        comp_key = f"{model_used}|{run}|{step}|{layer}"
        src = computed_cache_get(comp_key)
        if src is None:
            src = compute_computed_field_full(cfg["var"], d)
            computed_cache_set(comp_key, src)
    else:
        vname = cfg["var"]
        if vname not in d and layer == "clouds_total_mod" and "clct" in d:
            vname = "clct"
        if vname not in d:
            raise HTTPException(404, f"Variable {cfg['var']} unavailable")
        src = d[vname]
        if layer == "clouds_total_mod":
            src = normalize_clouds_total_mod(src)

    xs = np.linspace(minx, maxx, 256, endpoint=False) + (maxx - minx) / 512.0
    ys = np.linspace(maxy, miny, 256, endpoint=False) - (maxy - miny) / 512.0
    mx, my = np.meshgrid(xs, ys)
    qlon, qlat = _merc_to_lonlat(mx, my)

    inside = ((qlat >= data_lat_min) & (qlat <= data_lat_max) & (qlon >= data_lon_min) & (qlon <= data_lon_max))
    li = _regular_grid_indices(lat, qlat)
    lo = _regular_grid_indices(lon, qlon)
    sampled = src[li, lo]
    valid = inside & np.isfinite(sampled)

    rgba = colorize_layer_vectorized(layer, sampled, valid)

    img = Image.fromarray(rgba, mode="RGBA")
    b = io.BytesIO(); img.save(b, format="PNG", optimize=True); png = b.getvalue()
    tile_cache_set(client_class, cache_key, png)
    perf_record((perf_counter() - t0) * 1000.0, False)

    return Response(
        content=png,
        media_type="image/png",
        headers=build_tile_headers(
            run=run,
            valid_time=d["validTime"],
            model=model_used,
            cache="MISS",
        ),
    )


# ─── Static Frontend (must be LAST) ───
# html=True serves index.html for directory requests
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    _acquire_single_instance_or_exit(PID_FILE)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
