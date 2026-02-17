#!/usr/bin/env python3
"""Skyview FastAPI backend — serves ICON-D2 weather symbols for Leaflet frontend."""

import os
import sys
import math
import time
import atexit
import uuid
import json
import base64
import hmac
import hashlib
import requests
from time import perf_counter
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Deque
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
import threading

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
from classify import classify_cloud_type, classify_point as classify_point_core
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FRONTEND_DIR = os.path.join(SCRIPT_DIR, "..", "frontend")
PID_FILE = os.path.join(SCRIPT_DIR, "logs", "skyview.pid")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook (startup/shutdown)."""
    logger.info("Skyview API server starting")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")

    runs = get_available_runs()
    logger.info(f"Found {len(runs)} available model runs")
    if runs:
        latest = runs[0]
        logger.info(f"Latest run: {latest['model']} {latest['run']} ({len(latest['steps'])} timesteps)")

    workers_env = os.environ.get("WEB_CONCURRENCY") or os.environ.get("UVICORN_WORKERS") or "1"
    try:
        worker_count = int(workers_env)
    except Exception:
        worker_count = 1
    if worker_count > 1:
        logger.warning(
            "Running with multiple workers (%s): process-local counters/caches are per-worker only.",
            worker_count,
        )

    yield


app = FastAPI(title="Skyview API", lifespan=lifespan)

# CORS: default to a safe local allowlist; override via SKYVIEW_CORS_ORIGINS (comma-separated).
# Use SKYVIEW_CORS_ORIGINS=* only in trusted dev setups.
_cors_env = os.environ.get("SKYVIEW_CORS_ORIGINS", "").strip()
if _cors_env:
    allow_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    allow_origins = [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_error_counters = {"4xx": 0, "5xx": 0}

# Fallback/blending observability counters (process-local, reset on restart)
fallback_stats = {
    "euResolveAttempts": 0,
    "euResolveSuccess": 0,
    "strictTimeDenied": 0,
    "overlayFallback": 0,
    "overlayTileFallback": 0,
    "symbolsBlended": 0,
    "windBlended": 0,
    "pointFallback": 0,
}

# Small process-local cache to avoid repeated strict EU time resolution work
# for identical (time_str, max_delta_hours) pairs during bursty requests.
_eu_strict_cache: OrderedDict[tuple[str, float], Optional[tuple[str, int, str]]] = OrderedDict()
_EU_STRICT_CACHE_MAX = 64

last_nominatim_request: float = 0.0

feedback_rates: Dict[str, Deque[float]] = {}


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


data_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()


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

def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=0.0):
    """Backward-compatible wrapper around canonical scalar classifier."""
    return classify_point_core(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf)


def load_data(run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load .npz data for a given run/step/model.
    
    Args:
        keys: If provided, only load these keys (plus lat/lon). Saves memory for large grids.
              If None, loads all keys (backward compat).
    """
    cache_key = f"{model}/{run}/{step:03d}"
    
    # Check cache — if we have it and it has all requested keys, use it
    if cache_key in data_cache:
        data_cache.move_to_end(cache_key)
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

    if len(data_cache) >= 8:
        evicted_key, _ = data_cache.popitem(last=False)
        logger.info(f"LRU Cache eviction: {evicted_key}")
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


def _freshness_minutes_from_run(run: str) -> Optional[float]:
    try:
        run_dt = datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - run_dt).total_seconds() / 60.0, 1)
    except Exception:
        return None


def _merge_axis_aligned_segments(segments, eps: float = 1e-9):
    """Merge contiguous horizontal/vertical boundary segments."""
    if not segments:
        return []

    horiz = {}
    vert = {}

    def _q(x: float) -> float:
        return round(float(x), 8)

    for seg in segments:
        (a_lat, a_lon), (b_lat, b_lon) = seg
        a_lat, a_lon, b_lat, b_lon = float(a_lat), float(a_lon), float(b_lat), float(b_lon)
        if abs(a_lat - b_lat) <= eps:
            y = _q(a_lat)
            lo, hi = sorted((a_lon, b_lon))
            horiz.setdefault(y, []).append((lo, hi))
        elif abs(a_lon - b_lon) <= eps:
            x = _q(a_lon)
            lo, hi = sorted((a_lat, b_lat))
            vert.setdefault(x, []).append((lo, hi))

    merged = []

    for y, ivals in horiz.items():
        ivals.sort(key=lambda t: (t[0], t[1]))
        cur_lo, cur_hi = ivals[0]
        for lo, hi in ivals[1:]:
            if lo <= cur_hi + eps:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append([[float(y), cur_lo], [float(y), cur_hi]])
                cur_lo, cur_hi = lo, hi
        merged.append([[float(y), cur_lo], [float(y), cur_hi]])

    for x, ivals in vert.items():
        ivals.sort(key=lambda t: (t[0], t[1]))
        cur_lo, cur_hi = ivals[0]
        for lo, hi in ivals[1:]:
            if lo <= cur_hi + eps:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append([[cur_lo, float(x)], [cur_hi, float(x)]])
                cur_lo, cur_hi = lo, hi
        merged.append([[cur_lo, float(x)], [cur_hi, float(x)]])

    return merged


# ─── API Endpoints ───


@app.get("/api/health")
async def health():
    runs = get_available_runs()
    return {"status": "ok", "runs": len(runs), "cache": len(data_cache)}


@app.get("/api/d2_domain")
async def api_d2_domain(time: str = Query("latest")):
    """Return ICON-D2 domain bounds and boundary of last valid cells."""
    run, step, _ = resolve_time_with_cache_context(time, "icon_d2")
    d = load_data(run, step, "icon_d2", keys=["ww"])

    # Fast path: use precomputed run-level boundary generated at ingestion.
    run_dir = os.path.join(DATA_DIR, "icon-d2", run)
    boundary_cache_path = os.path.join(run_dir, "_d2_boundary.json")
    if os.path.exists(boundary_cache_path):
        try:
            with open(boundary_cache_path, "r", encoding="utf-8") as f:
                bc = json.load(f)
            return {
                "model": "icon_d2",
                "run": run,
                "validTime": d["validTime"],
                "bbox": bc.get("bbox"),
                "cellEdgeBbox": {
                    **(bc.get("cellEdgeBbox") or {}),
                    "latRes": bc.get("latRes"),
                    "lonRes": bc.get("lonRes"),
                },
                "boundarySegments": bc.get("boundarySegments", []),
                "diagnostics": {
                    "dataFreshnessMinutes": _freshness_minutes_from_run(run),
                    "validCells": bc.get("validCells", 0),
                    "boundarySegmentCount": bc.get("boundarySegmentCount", 0),
                    "source": "precomputed",
                },
            }
        except Exception:
            pass

    # Fallback path: compute boundary if cache file is not present.
    lat = d["lat"]
    lon = d["lon"]
    ww = d.get("ww")
    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    lat_res = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    lon_res = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.02
    valid = np.isfinite(ww) if ww is not None else np.ones((len(lat), len(lon)), dtype=bool)

    segments = []
    n_i, n_j = valid.shape
    for i in range(n_i):
        lat_lo = float(lat[i]) - lat_res / 2.0
        lat_hi = float(lat[i]) + lat_res / 2.0
        for j in range(n_j):
            if not valid[i, j]:
                continue
            lon_lo = float(lon[j]) - lon_res / 2.0
            lon_hi = float(lon[j]) + lon_res / 2.0
            if i == n_i - 1 or not valid[i + 1, j]:
                segments.append([[lat_hi, lon_lo], [lat_hi, lon_hi]])
            if i == 0 or not valid[i - 1, j]:
                segments.append([[lat_lo, lon_lo], [lat_lo, lon_hi]])
            if j == n_j - 1 or not valid[i, j + 1]:
                segments.append([[lat_lo, lon_hi], [lat_hi, lon_hi]])
            if j == 0 or not valid[i, j - 1]:
                segments.append([[lat_lo, lon_lo], [lat_hi, lon_lo]])

    segments = _merge_axis_aligned_segments(segments)

    return {
        "model": "icon_d2",
        "run": run,
        "validTime": d["validTime"],
        "bbox": {
            "latMin": lat_min,
            "lonMin": lon_min,
            "latMax": lat_max,
            "lonMax": lon_max,
        },
        "cellEdgeBbox": {
            "latMin": lat_min - lat_res / 2.0,
            "lonMin": lon_min - lon_res / 2.0,
            "latMax": lat_max + lat_res / 2.0,
            "lonMax": lon_max + lon_res / 2.0,
            "latRes": lat_res,
            "lonRes": lon_res,
        },
        "boundarySegments": segments,
        "diagnostics": {
            "dataFreshnessMinutes": _freshness_minutes_from_run(run),
            "validCells": int(np.count_nonzero(valid)),
            "boundarySegmentCount": len(segments),
            "source": "computed",
        },
    }


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
    bbox: str = Query("30,-30,72,45"),
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
    symbol_keys = ["ww", "ceiling", "clcl", "clcm", "clch", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi", "hsurf"]
    run, step, model_used = resolve_time_with_cache_context(time, model)

    # Short-TTL response cache for repeated pan/zoom requests
    cache_bbox = f"{lat_min:.4f},{lon_min:.4f},{lat_max:.4f},{lon_max:.4f}"
    eu_cache_key = ""
    if model_used == "icon_d2":
        eu_strict = _resolve_eu_time_strict(time)
        if eu_strict is not None:
            run_eu_k, step_eu_k, model_eu_k = eu_strict
            eu_cache_key = f"|eu:{run_eu_k}:{step_eu_k}"
    symbols_cache_key = f"{model_used}|{run}|{step}{eu_cache_key}|z{zoom}|{cache_bbox}"
    cached_symbols = symbols_cache_get(symbols_cache_key)
    if cached_symbols is not None:
        return cached_symbols

    d = load_data(run, step, model_used, keys=symbol_keys)

    lat = d["lat"]  # 1D
    lon = d["lon"]  # 1D
    d2_lat_min, d2_lat_max = float(np.min(lat)), float(np.max(lat))
    d2_lon_min, d2_lon_max = float(np.min(lon)), float(np.max(lon))

    # Phase 2: EU fallback outside D2 domain
    d_eu = None
    c_lat_eu = c_lon_eu = None
    ww_eu = ceil_arr_eu = c_clcl_eu = c_clcm_eu = c_clch_eu = None
    c_cape_eu = c_htop_dc_eu = c_hbas_sc_eu = c_htop_sc_eu = c_lpi_eu = c_hsurf_eu = None

    # Bbox-slice BEFORE heavy computation.
    # Use one-cell padding around viewport to stabilize aggregation at screen borders during pan.
    pad = cell_size
    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is not None and len(li) == 0:
        c_lat = np.array([], dtype=float)
        c_lon = np.array([], dtype=float)
        ww = np.zeros((0, 0), dtype=float)
        ceil_arr = np.zeros((0, 0), dtype=float)
        c_clcl = c_clcm = c_clch = c_cape = c_htop_dc = c_hbas_sc = c_htop_sc = c_lpi = c_hsurf = np.zeros((0, 0), dtype=float)
    else:
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
        c_hsurf = _slice_array(d["hsurf"], li, lo) if "hsurf" in d else np.zeros_like(ww)

    # Prepare EU fallback arrays over same padded bbox when D2 is primary model
    if model_used == "icon_d2":
        try:
            eu_strict = _resolve_eu_time_strict(time)
            if eu_strict is not None:
                run_eu, step_eu, model_eu = eu_strict
                d_eu = load_data(run_eu, step_eu, model_eu, keys=symbol_keys)
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
                    c_lpi_eu = _slice_array(d_eu["lpi"], li_eu, lo_eu) if "lpi" in d_eu else np.zeros_like(ww_eu)
                    c_hsurf_eu = _slice_array(d_eu["hsurf"], li_eu, lo_eu) if "hsurf" in d_eu else np.zeros_like(ww_eu)
        except Exception:
            d_eu = None

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

    lat_groups_eu = lon_groups_eu = None
    if d_eu is not None and c_lat_eu is not None and c_lon_eu is not None:
        lat_groups_eu = [[] for _ in range(lat_cell_count)]
        lon_groups_eu = [[] for _ in range(lon_cell_count)]
        for idx, v in enumerate(c_lat_eu):
            bi = int(np.floor((float(v) - lat_start) / cell_size))
            if 0 <= bi < lat_cell_count:
                lat_groups_eu[bi].append(idx)
        for idx, v in enumerate(c_lon_eu):
            bj = int(np.floor((float(v) - lon_start) / cell_size))
            if 0 <= bj < lon_cell_count:
                lon_groups_eu[bj].append(idx)

    symbols = []
    used_eu_any = False
    used_d2_any = False
    for i in range(lat_cell_count):
        for j in range(lon_cell_count):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2

            # Skip cells outside bbox
            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            # Select source model by location: D2 inside D2 domain, EU outside.
            in_d2_domain = (d2_lat_min <= lat_c <= d2_lat_max) and (d2_lon_min <= lon_c <= d2_lon_max)
            use_eu = (not in_d2_domain) and (lat_groups_eu is not None and lon_groups_eu is not None)

            if use_eu:
                used_eu_any = True
                src_lat = c_lat_eu
                src_lon = c_lon_eu
                src_ww = ww_eu
                src_ceil = ceil_arr_eu
                src_clcl = c_clcl_eu
                src_clcm = c_clcm_eu
                src_clch = c_clch_eu
                src_cape = c_cape_eu
                src_htop_dc = c_htop_dc_eu
                src_hbas_sc = c_hbas_sc_eu
                src_htop_sc = c_htop_sc_eu
                src_lpi = c_lpi_eu
                src_hsurf = c_hsurf_eu
                cli_list = lat_groups_eu[i]
                clo_list = lon_groups_eu[j]
            else:
                src_lat = c_lat
                src_lon = c_lon
                src_ww = ww
                src_ceil = ceil_arr
                src_clcl = c_clcl
                src_clcm = c_clcm
                src_clch = c_clch
                src_cape = c_cape
                src_htop_dc = c_htop_dc
                src_hbas_sc = c_hbas_sc
                src_htop_sc = c_htop_sc
                src_lpi = c_lpi
                src_hsurf = c_hsurf
                cli_list = lat_groups[i]
                clo_list = lon_groups[j]

            cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
            clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

            # If D2-selected cell has no finite weather signal, fall back to EU where available.
            if (not use_eu) and (lat_groups_eu is not None and lon_groups_eu is not None) and len(cli) > 0 and len(clo) > 0:
                d2_has_signal = np.any(np.isfinite(src_ww[np.ix_(cli, clo)]))
                if not d2_has_signal:
                    use_eu = True
                    used_eu_any = True
                    src_lat = c_lat_eu
                    src_lon = c_lon_eu
                    src_ww = ww_eu
                    src_ceil = ceil_arr_eu
                    src_clcl = c_clcl_eu
                    src_clcm = c_clcm_eu
                    src_clch = c_clch_eu
                    src_cape = c_cape_eu
                    src_htop_dc = c_htop_dc_eu
                    src_hbas_sc = c_hbas_sc_eu
                    src_htop_sc = c_htop_sc_eu
                    src_lpi = c_lpi_eu
                    src_hsurf = c_hsurf_eu
                    cli_list = lat_groups_eu[i]
                    clo_list = lon_groups_eu[j]
                    cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
                    clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)

            if len(cli) == 0 or len(clo) == 0:
                # No source data in this aggregation cell (outside model domain or boundary gap).
                # Skip instead of nearest-neighbor extrapolation to avoid artificial value copying.
                continue

            # Extract cell data
            cell_ww = src_ww[np.ix_(cli, clo)]
            max_ww = int(np.nanmax(cell_ww)) if not np.all(np.isnan(cell_ww)) else 0

            # Fast-path for likely clear/non-convective low-zoom cells to reduce first-hit latency.
            if zoom <= 9 and max_ww <= 3:
                cell_cape = src_cape[np.ix_(cli, clo)]
                conv_signal = np.any(np.isfinite(cell_cape) & (cell_cape > 50))
                cell_ceil = src_ceil[np.ix_(cli, clo)]
                ceil_valid = cell_ceil[np.isfinite(cell_ceil) & (cell_ceil > 0) & (cell_ceil < 20000)]
                if (not conv_signal) and len(ceil_valid) == 0:
                    sym, cb_hm = "clear", None
                    best_ii, best_jj = int(cli[len(cli) // 2]), int(clo[len(clo) // 2])
                else:
                    sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                        cli=cli,
                        clo=clo,
                        cell_ww=cell_ww,
                        ceil_arr=src_ceil,
                        c_clcl=src_clcl,
                        c_clcm=src_clcm,
                        c_clch=src_clch,
                        c_cape=src_cape,
                        c_htop_dc=src_htop_dc,
                        c_hbas_sc=src_hbas_sc,
                        c_htop_sc=src_htop_sc,
                        c_lpi=src_lpi,
                        c_hsurf=src_hsurf,
                        classify_point_fn=classify_point,
                        zoom=zoom,
                    )
            else:
                # Determine aggregated symbol in helper module
                sym, cb_hm, best_ii, best_jj = aggregate_symbol_cell(
                    cli=cli,
                    clo=clo,
                    cell_ww=cell_ww,
                    ceil_arr=src_ceil,
                    c_clcl=src_clcl,
                    c_clcm=src_clcm,
                    c_clch=src_clch,
                    c_cape=src_cape,
                    c_htop_dc=src_htop_dc,
                    c_hbas_sc=src_hbas_sc,
                    c_htop_sc=src_htop_sc,
                    c_lpi=src_lpi,
                    c_hsurf=src_hsurf,
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
            rep_lat = float(src_lat[best_ii])
            rep_lon = float(src_lon[best_jj])
            plot_lat = float(lat_c)
            plot_lon = float(lon_c)

            source_model = "icon_eu" if (use_eu or model_used == "icon_eu") else "icon_d2"
            if source_model == "icon_eu":
                used_eu_any = True
            else:
                used_d2_any = True

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

    if used_eu_any and used_d2_any:
        fallback_stats["symbolsBlended"] += 1

    effective_run = run
    effective_valid_time = d["validTime"]
    if used_eu_any and not used_d2_any:
        # Viewport fully covered by EU source.
        resolved_model = "icon_eu"
        if d_eu is not None:
            effective_run = d_eu.get("_run", run)
            effective_valid_time = d_eu.get("validTime", d["validTime"])
        fallback_decision = "eu_only_in_viewport"
    elif used_eu_any and used_d2_any:
        resolved_model = "blended"
        fallback_decision = "blended_d2_eu"
    else:
        resolved_model = model_used
        fallback_decision = "primary_model_only"

    result = {
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
        },
    }
    symbols_cache_set(symbols_cache_key, result)
    return result


@app.get("/api/wind")
async def api_wind(
    zoom: int = Query(8, ge=5, le=12),
    bbox: str = Query("30,-30,72,45"),
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
    d2_lat_min, d2_lat_max = float(np.min(lat)), float(np.max(lat))
    d2_lon_min, d2_lon_max = float(np.min(lon)), float(np.max(lon))

    d_eu = None
    c_lat_eu = c_lon_eu = u_eu = v_eu = None

    # Check if wind data is available
    if u_key not in d or v_key not in d:
        return {"barbs": [], "run": run, "model": model_used, "validTime": d["validTime"], "level": level, "count": 0}

    # Bbox-slice before computation.
    # Use one-cell padding around viewport to stabilize border cells during pan.
    pad = cell_size
    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is not None and len(li) == 0:
        # keep path alive; EU fallback may still provide coverage
        c_lat = np.array([], dtype=float)
        c_lon = np.array([], dtype=float)
        u = np.zeros((0, 0), dtype=float)
        v = np.zeros((0, 0), dtype=float)
    else:
        c_lat = lat[li] if li is not None else lat
        c_lon = lon[lo] if lo is not None else lon
        u = _slice_array(d[u_key], li, lo)
        v = _slice_array(d[v_key], li, lo)

    if model_used == "icon_d2":
        try:
            eu_strict = _resolve_eu_time_strict(time)
            if eu_strict is not None:
                run_eu, step_eu, model_eu = eu_strict
                d_eu = load_data(run_eu, step_eu, model_eu, keys=[u_key, v_key])
                if u_key in d_eu and v_key in d_eu:
                    lat_eu = d_eu["lat"]
                    lon_eu = d_eu["lon"]
                    li_eu, lo_eu = _bbox_indices(lat_eu, lon_eu, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
                    if not (li_eu is not None and len(li_eu) == 0):
                        c_lat_eu = lat_eu[li_eu] if li_eu is not None else lat_eu
                        c_lon_eu = lon_eu[lo_eu] if lo_eu is not None else lon_eu
                        u_eu = _slice_array(d_eu[u_key], li_eu, lo_eu)
                        v_eu = _slice_array(d_eu[v_key], li_eu, lo_eu)
        except Exception:
            d_eu = None

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

    lat_cell_count = max(0, len(lat_edges) - 1)
    lon_cell_count = max(0, len(lon_edges) - 1)

    # Pre-bin indices once (same strategy as /api/symbols)
    lat_groups = [[] for _ in range(lat_cell_count)]
    lon_groups = [[] for _ in range(lon_cell_count)]
    for idx, v_lat in enumerate(c_lat):
        bi = int(np.floor((float(v_lat) - lat_start) / cell_size))
        if 0 <= bi < lat_cell_count:
            lat_groups[bi].append(idx)
    for idx, v_lon in enumerate(c_lon):
        bj = int(np.floor((float(v_lon) - lon_start) / cell_size))
        if 0 <= bj < lon_cell_count:
            lon_groups[bj].append(idx)

    lat_groups_eu = lon_groups_eu = None
    if c_lat_eu is not None and c_lon_eu is not None and u_eu is not None and v_eu is not None:
        lat_groups_eu = [[] for _ in range(lat_cell_count)]
        lon_groups_eu = [[] for _ in range(lon_cell_count)]
        for idx, v_lat in enumerate(c_lat_eu):
            bi = int(np.floor((float(v_lat) - lat_start) / cell_size))
            if 0 <= bi < lat_cell_count:
                lat_groups_eu[bi].append(idx)
        for idx, v_lon in enumerate(c_lon_eu):
            bj = int(np.floor((float(v_lon) - lon_start) / cell_size))
            if 0 <= bj < lon_cell_count:
                lon_groups_eu[bj].append(idx)

    barbs = []
    used_eu_any = False
    for i in range(lat_cell_count):
        for j in range(lon_cell_count):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2

            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            in_d2_domain = (d2_lat_min <= lat_c <= d2_lat_max) and (d2_lon_min <= lon_c <= d2_lon_max)
            use_eu = (not in_d2_domain) and (lat_groups_eu is not None and lon_groups_eu is not None)

            if use_eu:
                used_eu_any = True
                src_lat = c_lat_eu
                src_lon = c_lon_eu
                src_u = u_eu
                src_v = v_eu
                cli_list = lat_groups_eu[i]
                clo_list = lon_groups_eu[j]
            else:
                src_lat = c_lat
                src_lon = c_lon
                src_u = u
                src_v = v
                cli_list = lat_groups[i]
                clo_list = lon_groups[j]

            cli = np.asarray(cli_list, dtype=int) if cli_list else np.empty((0,), dtype=int)
            clo = np.asarray(clo_list, dtype=int) if clo_list else np.empty((0,), dtype=int)
            if len(cli) == 0 or len(clo) == 0:
                continue

            cell_u = src_u[np.ix_(cli, clo)]
            cell_v = src_v[np.ix_(cli, clo)]
            mean_u = float(np.nanmean(cell_u))
            mean_v = float(np.nanmean(cell_v))

            # If D2-selected cell has no finite wind, try EU fallback for this cell.
            if (not use_eu) and (lat_groups_eu is not None and lon_groups_eu is not None) and (np.isnan(mean_u) or np.isnan(mean_v)):
                used_eu_any = True
                src_lat = c_lat_eu
                src_lon = c_lon_eu
                src_u = u_eu
                src_v = v_eu
                cli_list = lat_groups_eu[i]
                clo_list = lon_groups_eu[j]
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
    return {
        "barbs": barbs,
        "run": run,
        "model": resolved_model,
        "validTime": d["validTime"],
        "level": level,
        "count": len(barbs),
        "diagnostics": {
            "dataFreshnessMinutes": _freshness_minutes_from_run(run),
            "fallbackDecision": "blended_d2_eu" if used_eu_any else "primary_model_only",
            "requestedModel": model,
            "requestedTime": time,
            "sourceModel": resolved_model,
        },
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
    fallback_decision = "primary_model_only"

    # Phase 3: model-source transparency + EU fallback outside D2 domain.
    if model_used == "icon_d2":
        lat_d2 = d["lat"]
        lon_d2 = d["lon"]
        in_d2_domain = (float(np.min(lat_d2)) <= lat <= float(np.max(lat_d2))) and (float(np.min(lon_d2)) <= lon <= float(np.max(lon_d2)))
        li_d2 = int(np.argmin(np.abs(lat_d2 - lat)))
        lo_d2 = int(np.argmin(np.abs(lon_d2 - lon)))
        d2_has_signal = False
        for k in ("ww", "ceiling", "cape_ml", "hbas_sc"):
            if k in d:
                v = float(d[k][li_d2, lo_d2])
                if np.isfinite(v):
                    d2_has_signal = True
                    break
        if (not in_d2_domain) or (not d2_has_signal):
            trigger = "outside_d2_domain" if (not in_d2_domain) else "d2_missing_signal"
            try:
                eu_strict = _resolve_eu_time_strict(time)
                if eu_strict is not None:
                    run_eu, step_eu, model_eu = eu_strict
                    rotate_caches_for_context(f"{model_eu}|{run_eu}|{step_eu}")
                    d = load_data(run_eu, step_eu, model_eu)
                    run, step, model_used = run_eu, step_eu, model_eu
                    fallback_stats["pointFallback"] += 1
                    fallback_decision = f"eu_fallback:{trigger}"
                else:
                    fallback_decision = f"strict_time_denied:{trigger}"
            except Exception:
                fallback_decision = f"fallback_error:{trigger}"

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
    hsurf = scalar("hsurf") or 0.0

    ceiling_val = scalar("ceiling") or 0.0
    best_type = classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling_val, hsurf)
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
    result["sourceModel"] = model_used
    result["diagnostics"] = {
        "dataFreshnessMinutes": _freshness_minutes_from_run(run),
        "fallbackDecision": fallback_decision,
        "requestedModel": model,
        "requestedTime": time,
        "sourceModel": model_used,
    }

    return result


# ─── Feedback endpoint ───

FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
MARKERS_FILE = os.path.join(DATA_DIR, "markers.json")
OPENAIP_SEED_FILE = os.path.join(SCRIPT_DIR, "openaip_seed.json")
MARKER_AUTH_SECRET = os.environ.get("SKYVIEW_MARKER_AUTH_SECRET", "")
MARKER_AUTH_CONFIGURED = bool(MARKER_AUTH_SECRET and len(MARKER_AUTH_SECRET) >= 16 and MARKER_AUTH_SECRET != "dev-marker-secret-change-me")
MARKER_TOKEN_TTL_SECONDS = 12 * 3600
markers_lock = threading.Lock()
DEFAULT_MARKER = {
    "name": "Geitau",
    "note": "Default marker",
    "lat": 47.6836,
    "lon": 11.9610,
}


def _read_markers() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(MARKERS_FILE):
            return []
        with open(MARKERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _write_markers(markers: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(MARKERS_FILE), exist_ok=True)
    tmp = MARKERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(markers, f, ensure_ascii=False)
    os.replace(tmp, MARKERS_FILE)


def _load_openaip_seed() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(OPENAIP_SEED_FILE):
            return []
        with open(OPENAIP_SEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _sign_marker_token(payload_json: str) -> str:
    sig = hmac.new(MARKER_AUTH_SECRET.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")


def _make_marker_token(client_id: str) -> Dict[str, Any]:
    exp = int(time.time()) + MARKER_TOKEN_TTL_SECONDS
    payload = {"cid": client_id, "exp": exp}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    sig_b64 = _sign_marker_token(payload_json)
    return {
        "token": f"{payload_b64}.{sig_b64}",
        "expiresAt": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _verify_marker_token(client_id: str, token: str) -> bool:
    try:
        if not token or "." not in token:
            return False
        payload_b64, sig_b64 = token.split(".", 1)
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8")).decode("utf-8")
        expected_sig = _sign_marker_token(payload_json)
        if not hmac.compare_digest(expected_sig, sig_b64):
            return False
        payload = json.loads(payload_json)
        if payload.get("cid") != client_id:
            return False
        if int(payload.get("exp", 0)) < int(time.time()):
            return False
        return True
    except Exception:
        return False

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

    ip = str(request.client.host)
    now = time.time()
    times = feedback_rates.setdefault(ip, deque())
    while times and times[0] < now - 60:
        times.popleft()
    if len(times) >= 5:
        raise HTTPException(429, "Rate limit exceeded. Max 5 feedback per minute per IP.")
    times.append(now)

    if os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 10 * 1024 * 1024:
        raise HTTPException(507, "Feedback storage full (10MB limit).")

    entry = make_feedback_entry(body)
    append_feedback(FEEDBACK_FILE, entry)

    logger.info(f"Feedback received: [{entry['type']}] {message[:80]}")
    return {"status": "ok", "id": entry["id"]}


@app.get("/api/feedback")
async def api_feedback_list():
    """List all feedback entries (for admin use)."""
    feedback = read_feedback_list(FEEDBACK_FILE)
    return {"feedback": feedback, "count": len(feedback)}


def _default_marker_for_client(client_id: str) -> Dict[str, Any]:
    return {
        "id": "default-geitau",
        "clientId": client_id,
        "name": DEFAULT_MARKER["name"],
        "note": DEFAULT_MARKER["note"],
        "lat": DEFAULT_MARKER["lat"],
        "lon": DEFAULT_MARKER["lon"],
        "createdAt": None,
        "isDefault": True,
    }


def _marker_for_client(client_id: str) -> Dict[str, Any]:
    # Fallback policy: if marker auth secret is not configured, show default marker.
    if not MARKER_AUTH_CONFIGURED:
        return _default_marker_for_client(client_id)

    markers = [m for m in _read_markers() if m.get("clientId") == client_id]
    if markers:
        markers.sort(key=lambda m: m.get("createdAt", ""), reverse=True)
        return markers[0]
    return _default_marker_for_client(client_id)


@app.get("/api/marker_profile")
async def api_marker_profile(clientId: str = Query(..., min_length=3, max_length=128)):
    marker = _marker_for_client(clientId)
    return {
        "marker": marker,
        "markerAuthConfigured": MARKER_AUTH_CONFIGURED,
        "markerEditable": MARKER_AUTH_CONFIGURED,
        "fallbackToDefault": (not MARKER_AUTH_CONFIGURED),
    }


@app.get("/api/marker_auth")
async def api_marker_auth(clientId: str = Query(..., min_length=3, max_length=128)):
    if not MARKER_AUTH_CONFIGURED:
        raise HTTPException(503, "Marker auth is not configured; marker editing disabled")
    tok = _make_marker_token(clientId)
    return {"clientId": clientId, **tok}


@app.post("/api/marker_profile")
async def api_marker_profile_set(request: Request):
    if not MARKER_AUTH_CONFIGURED:
        raise HTTPException(503, "Marker auth is not configured; marker editing disabled")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    client_id = str(body.get("clientId", "")).strip()
    if len(client_id) < 3:
        raise HTTPException(400, "clientId is required")

    token = str(body.get("markerAuthToken") or "").strip()
    if not _verify_marker_token(client_id, token):
        raise HTTPException(401, "Invalid or expired marker auth token")

    try:
        lat = float(body.get("lat"))
        lon = float(body.get("lon"))
    except Exception:
        raise HTTPException(400, "lat/lon are required")

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "lat/lon out of range")

    name = str(body.get("name") or "Marker").strip()[:80] or "Marker"
    note = str(body.get("note") or "").strip()[:280]

    marker = {
        "id": uuid.uuid4().hex[:12],
        "clientId": client_id,
        "name": name,
        "note": note,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with markers_lock:
        markers = [m for m in _read_markers() if m.get("clientId") != client_id]
        markers.append(marker)
        _write_markers(markers)
    return {"status": "ok", "marker": marker}


@app.get("/api/location_search")
async def api_location_search(q: str = Query(..., min_length=2, max_length=120), limit: int = Query(8, ge=1, le=20)):
    """Free-text search biased towards glider fields / airfields / airports.

    Uses a small local OpenAIP-style seed list first, then enriches with Nominatim.
    """
    ql = q.strip().lower()

    def _score_item(name: str, display_name: str, cls: str, typ: str, seed_boost: int = 0) -> int:
        s = seed_boost
        text = f"{name} {display_name} {cls} {typ}".lower()
        if ql in text:
            s += 20
        if name.lower().startswith(ql):
            s += 12
        for kw, w in [("glider", 20), ("gliding", 18), ("airfield", 18), ("airstrip", 16), ("aerodrome", 14), ("airport", 12), ("flugplatz", 12), ("flug", 8), ("segelflug", 10)]:
            if kw in text:
                s += w
        if cls == "aeroway":
            s += 10
        return s

    out = []

    # 1) Local seed hits (fast, curated)
    for it in _load_openaip_seed():
        try:
            name = str(it.get("name", "")).strip()
            display_name = str(it.get("displayName", "")).strip()
            if not name:
                continue
            if ql not in (name + " " + display_name).lower() and len(ql) >= 3:
                continue
            out.append({
                "name": name[:120],
                "displayName": display_name,
                "lat": float(it.get("lat")),
                "lon": float(it.get("lon")),
                "class": "aeroway",
                "type": it.get("type") or "airfield",
                "score": _score_item(name, display_name, "aeroway", str(it.get("type") or "airfield"), seed_boost=25),
                "source": "seed",
            })
        except Exception:
            continue

    # 2) Nominatim enrichment (rate-limited: max 1 req/s)
    global last_nominatim_request
    now = time.monotonic()
    if now - last_nominatim_request < 1.0:
        raise HTTPException(429, "Rate limit exceeded. Max 1 Nominatim request per second.")
    last_nominatim_request = now

    rows = []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": q,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": max(limit * 3, 15),
                "extratags": 1,
            },
            headers={"User-Agent": "skyview/1.0"},
            timeout=8,
        )
        if r.status_code == 200:
            j = r.json()
            rows = j if isinstance(j, list) else []
    except Exception:
        rows = []

    for it in rows:
        try:
            name = (it.get("name") or it.get("display_name", "").split(",")[0]).strip()[:120]
            display_name = it.get("display_name", "")
            cls = str(it.get("class", ""))
            typ = str(it.get("type", ""))
            out.append({
                "name": name,
                "displayName": display_name,
                "lat": float(it.get("lat")),
                "lon": float(it.get("lon")),
                "class": cls,
                "type": typ,
                "score": _score_item(name, display_name, cls, typ),
                "source": "nominatim",
            })
        except Exception:
            continue

    # De-duplicate by rounded lat/lon + name
    seen = set()
    dedup = []
    for r in sorted(out, key=lambda x: x["score"], reverse=True):
        key = (round(float(r["lat"]), 4), round(float(r["lon"]), 4), r["name"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    return {"results": dedup[:limit], "count": min(len(dedup), limit)}


# Backward-compat endpoints (legacy marker list API)
@app.get("/api/markers")
async def api_markers_list(clientId: str = Query(..., min_length=3, max_length=128)):
    marker = _marker_for_client(clientId)
    return {"markers": [marker], "count": 1}


@app.delete("/api/markers")
async def api_markers_reset_default(
    clientId: str = Query(..., min_length=3, max_length=128),
    markerAuthToken: str = Query(""),
):
    if not MARKER_AUTH_CONFIGURED:
        return {"status": "ok", "deleted": 0, "marker": _default_marker_for_client(clientId), "fallbackToDefault": True}
    if not _verify_marker_token(clientId, markerAuthToken):
        raise HTTPException(401, "Invalid or expired marker auth token")
    with markers_lock:
        markers = [m for m in _read_markers() if m.get("clientId") != clientId]
        _write_markers(markers)
    return {"status": "ok", "deleted": 1, "marker": _marker_for_client(clientId)}


# ─── Overlay endpoint ───

# Overlay image helpers
import io
from PIL import Image

# Phase-2 modularization: use shared overlay registry from backend/overlay_render.py
OVERLAY_CONFIGS = RENDER_OVERLAY_CONFIGS


@app.get("/api/overlay")
async def api_overlay(
    layer: str = Query(...),
    bbox: str = Query("30,-30,72,45"),
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

    # Optional EU fallback outside D2 domain (Phase 1)
    eu_fb = _try_load_eu_fallback(time, cfg) if model_used == "icon_d2" else None
    overlay_fallback_used = False

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
        # If primary D2 has no coverage in bbox, fallback to EU for this request.
        if eu_fb is not None:
            d = eu_fb["data"]
            run, step, model_used = eu_fb["run"], eu_fb["step"], eu_fb["model"]
            overlay_fallback_used = True
            fallback_stats["overlayFallback"] += 1
            lat = d["lat"]
            lon = d["lon"]
            data_lat_min, data_lat_max = float(lat.min()), float(lat.max())
            data_lon_min, data_lon_max = float(lon.min()), float(lon.max())
            lat_min = max(lat_min, data_lat_min)
            lat_max = min(lat_max, data_lat_max)
            lon_min = max(lon_min, data_lon_min)
            lon_max = min(lon_max, data_lon_max)
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
    valid_merc = np.isfinite(cropped_merc)
    rgba_merc = colorize_layer_vectorized(layer, cropped_merc, valid_merc)
    # Match prior orientation (legacy path wrote rows bottom-up)
    rgba_merc = np.flipud(rgba_merc)
    img = Image.fromarray(rgba_merc, mode="RGBA")

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
            extra={
                "X-Source-Model": model_used,
                "X-Fallback-Decision": "eu_fallback:outside_d2_domain" if overlay_fallback_used else "primary_model_only",
                "X-Data-Freshness-Minutes": str(_freshness_minutes_from_run(run)),
            },
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


def _overlay_source_field(layer: str, cfg: dict, d: dict, model_used: str, run: str, step: int):
    """Return source field for overlay rendering (full-grid) with computed caching."""
    if cfg.get("computed"):
        comp_key = f"{model_used}|{run}|{step}|{layer}"
        src = computed_cache_get(comp_key)
        if src is None:
            src = compute_computed_field_full(cfg["var"], d)
            computed_cache_set(comp_key, src)
        return src

    vname = cfg["var"]
    if vname not in d and layer == "clouds_total_mod" and "clct" in d:
        vname = "clct"
    if vname not in d:
        raise HTTPException(404, f"Variable {cfg['var']} unavailable")
    src = d[vname]
    if layer == "clouds_total_mod":
        src = normalize_clouds_total_mod(src)
    return src


def _resolve_eu_time_strict(time_str: str, max_delta_hours: float = 2.0):
    """Resolve EU run/step only if close enough to requested time.

    Accepts ISO-8601 timestamps or the sentinel "latest".
    """
    t = (time_str or "").strip()
    cache_key = (t or "latest", float(max_delta_hours))
    if cache_key in _eu_strict_cache:
        _eu_strict_cache.move_to_end(cache_key)
        return _eu_strict_cache[cache_key]

    fallback_stats["euResolveAttempts"] += 1
    try:
        run_eu, step_eu, model_eu = resolve_time(t or "latest", "icon_eu")
        if model_eu != "icon_eu":
            result = None
        else:
            # lightweight validTime computation without loading fields
            run_dt = datetime.strptime(run_eu, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            vt = run_dt + timedelta(hours=step_eu)

            # Explicit non-ISO handling: latest/current request should pass strictness check.
            if t == "" or t.lower() == "latest":
                fallback_stats["euResolveSuccess"] += 1
                result = (run_eu, step_eu, model_eu)
            else:
                # ISO target time (timezone-naive is treated as UTC)
                try:
                    target = datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                except Exception:
                    fallback_stats["strictTimeDenied"] += 1
                    result = None
                else:
                    if abs((vt - target).total_seconds()) > max_delta_hours * 3600.0:
                        fallback_stats["strictTimeDenied"] += 1
                        result = None
                    else:
                        fallback_stats["euResolveSuccess"] += 1
                        result = (run_eu, step_eu, model_eu)
    except Exception:
        result = None

    _eu_strict_cache[cache_key] = result
    _eu_strict_cache.move_to_end(cache_key)
    while len(_eu_strict_cache) > _EU_STRICT_CACHE_MAX:
        _eu_strict_cache.popitem(last=False)
    return result


def _try_load_eu_fallback(time_str: str, cfg: dict, max_delta_hours: float = 2.0):
    """Load ICON-EU fallback, but only when temporally consistent with request time."""
    overlay_keys = build_overlay_keys(cfg)
    try:
        eu_strict = _resolve_eu_time_strict(time_str, max_delta_hours=max_delta_hours)
        if eu_strict is None:
            return None
        run_eu, step_eu, model_eu = eu_strict
        d_eu = load_data(run_eu, step_eu, model_eu, keys=overlay_keys)
        return {
            "run": run_eu,
            "step": step_eu,
            "model": model_eu,
            "data": d_eu,
        }
    except Exception:
        return None


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
        fallback_stats=fallback_stats,
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

    eu_fb = _try_load_eu_fallback(time, cfg) if model_used == "icon_d2" else None
    eu_key = f"|eu:{eu_fb['run']}:{eu_fb['step']}" if eu_fb else ""
    cache_key = f"{client_class}|{model_used}|{run}|{step}{eu_key}|{layer}|{z}|{x}|{y}"
    cached = tile_cache_get(client_class, cache_key)
    if cached is not None:
        perf_record((perf_counter() - t0) * 1000.0, True)
        source_model_hdr = "blended" if eu_fb else model_used
        return Response(
            content=cached,
            media_type="image/png",
            headers=build_tile_headers(
                run=run,
                valid_time=d["validTime"],
                model=model_used,
                cache="HIT",
                extra={
                    "X-Source-Model": source_model_hdr,
                    "X-Fallback-Decision": "blended_d2_eu" if source_model_hdr == "blended" else "primary_model_only",
                    "X-Data-Freshness-Minutes": str(_freshness_minutes_from_run(run)),
                },
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
        # If primary D2 misses this tile, allow EU fallback coverage.
        if eu_fb is None:
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
                    extra={
                        "X-Source-Model": model_used,
                        "X-Fallback-Decision": "primary_model_only",
                        "X-Data-Freshness-Minutes": str(_freshness_minutes_from_run(run)),
                    },
                ),
            )

    # source field(s): primary + optional EU fallback
    src = _overlay_source_field(layer, cfg, d, model_used, run, step)

    d_eu = None
    src_eu = None
    lat_eu = lon_eu = None
    data_lat_min_eu = data_lat_max_eu = data_lon_min_eu = data_lon_max_eu = None
    if eu_fb is not None:
        d_eu = eu_fb["data"]
        src_eu = _overlay_source_field(layer, cfg, d_eu, eu_fb["model"], eu_fb["run"], eu_fb["step"])
        lat_eu = d_eu["lat"]
        lon_eu = d_eu["lon"]
        data_lat_min_eu, data_lat_max_eu = float(np.min(lat_eu)), float(np.max(lat_eu))
        data_lon_min_eu, data_lon_max_eu = float(np.min(lon_eu)), float(np.max(lon_eu))

    xs = np.linspace(minx, maxx, 256, endpoint=False) + (maxx - minx) / 512.0
    ys = np.linspace(maxy, miny, 256, endpoint=False) - (maxy - miny) / 512.0
    mx, my = np.meshgrid(xs, ys)
    qlon, qlat = _merc_to_lonlat(mx, my)

    inside = ((qlat >= data_lat_min) & (qlat <= data_lat_max) & (qlon >= data_lon_min) & (qlon <= data_lon_max))
    li = _regular_grid_indices(lat, qlat)
    lo = _regular_grid_indices(lon, qlon)
    sampled = src[li, lo]
    valid = inside & np.isfinite(sampled)

    source_model_hdr = model_used
    # EU fallback outside D2: fill only where primary is invalid/outside
    if src_eu is not None:
        inside_eu = ((qlat >= data_lat_min_eu) & (qlat <= data_lat_max_eu) & (qlon >= data_lon_min_eu) & (qlon <= data_lon_max_eu))
        li_eu = _regular_grid_indices(lat_eu, qlat)
        lo_eu = _regular_grid_indices(lon_eu, qlon)
        sampled_eu = src_eu[li_eu, lo_eu]
        valid_eu = inside_eu & np.isfinite(sampled_eu)
        fill_mask = (~valid) & valid_eu
        if np.any(fill_mask):
            sampled = np.array(sampled, copy=True)
            sampled[fill_mask] = sampled_eu[fill_mask]
            valid = valid | fill_mask
            source_model_hdr = "blended"
            fallback_stats["overlayTileFallback"] += 1

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
            extra={
                "X-Source-Model": source_model_hdr,
                "X-Fallback-Decision": "blended_d2_eu" if source_model_hdr == "blended" else "primary_model_only",
                "X-Data-Freshness-Minutes": str(_freshness_minutes_from_run(run)),
            },
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
