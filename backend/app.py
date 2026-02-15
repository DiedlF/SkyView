#!/usr/bin/env python3
"""Skyview FastAPI backend — serves ICON-D2 weather symbols for Leaflet frontend."""

import os
import sys
import math
import time
from time import perf_counter
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from collections import OrderedDict, deque

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# Add backend dir to path for classify import
sys.path.insert(0, os.path.dirname(__file__))
from classify import classify_cloud_type, get_cloud_base
from soaring import calc_wstar, calc_climb_rate, calc_lcl, calc_cu_potential, calc_thermal_height, calc_reachable_distance
from logging_config import setup_logging

logger = setup_logging(__name__, level="INFO")

app = FastAPI(title="Skyview API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log API requests with method, path, and response time."""
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    
    # Skip logging routine polling endpoints to reduce noise
    skip_paths = ['/api/timesteps', '/api/models']
    if request.url.path not in skip_paths or response.status_code >= 400:
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms"
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

data_cache: Dict[str, Dict[str, Any]] = {}

# Overlay tile cache (desktop/mobile split, LRU + TTL)
TILE_CACHE_MAX_ITEMS_DESKTOP = 1400
TILE_CACHE_MAX_ITEMS_MOBILE = 700
TILE_CACHE_TTL_SECONDS = 900

tile_cache_desktop = OrderedDict()  # key -> (png_bytes, ts)
tile_cache_mobile = OrderedDict()
cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}

# Overlay tile performance telemetry
perf_recent = deque(maxlen=400)  # [{'ms':..., 'hit':0/1, 'ts':...}, ...]
perf_totals = {
    'requests': 0,
    'hits': 0,
    'misses': 0,
    'totalMs': 0.0,
}

def _perf_record(ms: float, cache_hit: bool):
    now = time.time()
    perf_recent.append({'ms': float(ms), 'hit': 1 if cache_hit else 0, 'ts': now})
    perf_totals['requests'] += 1
    perf_totals['totalMs'] += float(ms)
    if cache_hit:
        perf_totals['hits'] += 1
    else:
        perf_totals['misses'] += 1

# Cache for expensive full-grid computed overlay fields (run/step/layer scoped)
COMPUTED_FIELD_CACHE_MAX_ITEMS = 48
computed_field_cache = OrderedDict()  # key -> ndarray

def _computed_cache_get(key: str):
    arr = computed_field_cache.get(key)
    if arr is None:
        return None
    computed_field_cache.move_to_end(key)
    return arr

def _computed_cache_set(key: str, arr):
    computed_field_cache[key] = arr
    computed_field_cache.move_to_end(key)
    while len(computed_field_cache) > COMPUTED_FIELD_CACHE_MAX_ITEMS:
        computed_field_cache.popitem(last=False)

def _cache_for_class(client_class: str):
    if client_class == "mobile":
        return tile_cache_mobile, TILE_CACHE_MAX_ITEMS_MOBILE
    return tile_cache_desktop, TILE_CACHE_MAX_ITEMS_DESKTOP

def _tile_cache_prune(client_class: str):
    cache, max_items = _cache_for_class(client_class)
    now = time.time()
    while cache:
        _, (_, ts) = next(iter(cache.items()))
        if (now - ts) <= TILE_CACHE_TTL_SECONDS:
            break
        cache.popitem(last=False)
        cache_stats["expired"] += 1
    while len(cache) > max_items:
        cache.popitem(last=False)
        cache_stats["evictions"] += 1

def _tile_cache_get(client_class: str, key: str):
    cache, _ = _cache_for_class(client_class)
    _tile_cache_prune(client_class)
    item = cache.get(key)
    if item is None:
        cache_stats["misses"] += 1
        return None
    png, ts = item
    if (time.time() - ts) > TILE_CACHE_TTL_SECONDS:
        del cache[key]
        cache_stats["expired"] += 1
        cache_stats["misses"] += 1
        return None
    cache.move_to_end(key)
    cache_stats["hits"] += 1
    return png

def _tile_cache_set(client_class: str, key: str, png: bytes):
    cache, _ = _cache_for_class(client_class)
    cache[key] = (png, time.time())
    cache.move_to_end(key)
    _tile_cache_prune(client_class)

# ─── Grid / Bbox helpers ───

def _get_grid_bounds(lat, lon):
    """Return (lat_min, lat_max, lon_min, lon_max) from 1D coordinate arrays."""
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max):
    """Return (lat_indices, lon_indices) for points within the bbox.
    Returns None, None if bbox covers the entire grid."""
    grid_lat_min, grid_lat_max = float(lat.min()), float(lat.max())
    grid_lon_min, grid_lon_max = float(lon.min()), float(lon.max())

    # If bbox covers entire grid, no need to slice
    if lat_min <= grid_lat_min and lat_max >= grid_lat_max and lon_min <= grid_lon_min and lon_max >= grid_lon_max:
        return None, None

    eps = 0.001
    lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
    lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]
    if len(li) == 0 or len(lo) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return li, lo


def _slice_array(arr, li, lo):
    """Slice a 2D array by lat/lon index arrays. If li/lo is None, return as-is."""
    if li is None:
        return arr
    return arr[np.ix_(li, lo)]


def ww_to_symbol(ww: int) -> Optional[str]:
    """Map WMO ww code to symbol type string."""
    if ww == 96: return "thunderstorm_hail"
    if 95 <= ww <= 99: return "thunderstorm"
    if ww == 86: return "snow_shower_heavy"
    if ww == 85: return "snow_shower"
    if ww == 81: return "rain_shower_moderate"
    if ww == 80: return "rain_shower"
    if ww == 82: return "rain_shower_moderate"
    if ww == 75: return "snow_heavy"
    if ww == 73: return "snow_moderate"
    if ww == 71: return "snow_slight"
    if ww == 77: return "snow_grains"
    if ww == 65: return "rain_heavy"
    if ww == 63: return "rain_moderate"
    if ww == 61: return "rain_slight"
    if ww in (66, 67): return "freezing_rain"
    if ww == 55: return "drizzle_dense"
    if ww == 53: return "drizzle_moderate"
    if ww == 51: return "drizzle_light"
    if ww == 57: return "freezing_drizzle_heavy"
    if ww == 56: return "freezing_drizzle"
    if ww == 45: return "fog"
    if ww == 48: return "rime_fog"
    return None  # ww below significant range uses cloud classification


def ww_severity_rank(ww: int) -> int:
    """Aviation-oriented ww severity ordering for aggregation."""
    if 95 <= ww <= 99: return 100
    if 71 <= ww <= 77: return 90
    if 85 <= ww <= 86: return 85
    if 66 <= ww <= 67: return 80
    if 56 <= ww <= 57: return 75
    if 61 <= ww <= 65: return 70
    if 80 <= ww <= 84: return 65
    if 51 <= ww <= 55: return 60
    if 45 <= ww <= 48: return 50
    return ww


# Cloud type priority for cloud-only fallback aggregation
CLOUD_PRIORITY = {
    "clear": 0,
    "ci": 1,
    "ac": 2,
    "st": 3,
    "blue_thermal": 4,
    "cu_hum": 5,
    "cu_con": 6,
    "cb": 7,
}


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
    """Scan data/icon-d2 and data/icon-eu dirs for available runs and steps."""
    runs = []
    for model_dir in ["icon-d2", "icon-eu"]:
        model_path = os.path.join(DATA_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        model_type = model_dir.replace("-", "_")
        for d in sorted(os.listdir(model_path), reverse=True):
            run_path = os.path.join(model_path, d)
            if not os.path.isdir(run_path):
                continue
            try:
                run_dt = datetime.strptime(d, "%Y%m%d%H")
            except ValueError:
                continue
            npz_files = sorted([f for f in os.listdir(run_path) if f.endswith(".npz")])
            steps = []
            for f in npz_files:
                try:
                    step = int(f[:-4])
                    vt = run_dt + timedelta(hours=step)
                    steps.append({"step": step, "validTime": vt.isoformat() + "Z"})
                except ValueError:
                    continue
            if steps:
                runs.append({"run": d, "model": model_type, "runTime": run_dt.isoformat() + "Z", "steps": steps})
    runs.sort(key=lambda x: x["runTime"], reverse=True)
    return runs


def get_merged_timeline():
    """Build a unified timeline: ICON-D2 for first 48h, ICON-EU for 49-120h.
    Returns a combined list of timesteps with model info, plus metadata."""
    runs = get_available_runs()
    if not runs:
        return None
    
    # Find latest D2 and EU runs
    # Prefer complete D2 runs (48 steps) — fall back to partial if no complete run exists
    d2_complete = next((r for r in runs if r["model"] == "icon_d2" and len(r["steps"]) >= 48), None)
    d2_any = next((r for r in runs if r["model"] == "icon_d2"), None)
    d2_run = d2_complete or d2_any
    eu_run = next((r for r in runs if r["model"] == "icon_eu"), None)
    
    if not d2_run and not eu_run:
        return None
    
    # Primary run is the latest D2 (or EU if no D2)
    primary = d2_run or eu_run
    
    # Build merged steps: D2 for first 48h, EU only AFTER D2's last valid time
    merged_steps = []
    d2_last_valid = None
    
    if d2_run:
        for s in d2_run["steps"]:
            merged_steps.append({**s, "model": "icon_d2", "run": d2_run["run"]})
            if d2_last_valid is None or s["validTime"] > d2_last_valid:
                d2_last_valid = s["validTime"]
    
    # Add EU steps strictly AFTER D2's last valid time
    if eu_run and d2_last_valid:
        for s in eu_run["steps"]:
            if s["validTime"] > d2_last_valid:
                merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})
    elif eu_run and not d2_run:
        # No D2 at all — use EU only
        for s in eu_run["steps"]:
            merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})
    
    # Sort by valid time
    merged_steps.sort(key=lambda x: x["validTime"])
    
    return {
        "run": primary["run"],
        "runTime": primary["runTime"],
        "model": primary["model"],
        "steps": merged_steps,
        "d2Run": d2_run["run"] if d2_run else None,
        "euRun": eu_run["run"] if eu_run else None,
    }


def resolve_time(time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
    """Resolve time string to (run, step, model)."""
    runs = get_available_runs()
    if not runs:
        raise HTTPException(404, "No data available")

    if time_str == "latest":
        d2_runs = [r for r in runs if r["model"] == "icon_d2"]
        if d2_runs:
            return d2_runs[0]["run"], d2_runs[0]["steps"][-1]["step"], "icon_d2"
        eu_runs = [r for r in runs if r["model"] == "icon_eu"]
        if eu_runs:
            return eu_runs[0]["run"], eu_runs[0]["steps"][-1]["step"], "icon_eu"
        return runs[0]["run"], runs[0]["steps"][-1]["step"], runs[0]["model"]

    try:
        target = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "Invalid time format")
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    lead_hours = max(0, (target - now).total_seconds() / 3600.0)

    candidates = runs
    if model is None:
        pref_model = "icon_d2" if lead_hours <= 48 else "icon_eu"
        candidates = [r for r in runs if r["model"] == pref_model]
    else:
        candidates = [r for r in runs if r["model"] == model]

    if not candidates:
        candidates = runs

    best_dist = float("inf")
    best_run = best_step = best_model = None
    for r in candidates:
        run_dt = datetime.fromisoformat(r["runTime"].replace("Z", "+00:00"))
        for s in r["steps"]:
            vt = datetime.fromisoformat(s["validTime"].replace("Z", "+00:00"))
            dist = abs((vt - target).total_seconds())
            if dist < best_dist:
                best_dist = dist
                best_run = r["run"]
                best_step = s["step"]
                best_model = r["model"]
    if best_run is None:
        raise HTTPException(404, "No matching timestep")
    return best_run, best_step, best_model


# ─── API Endpoints ───


@app.get("/api/health")
async def health():
    runs = get_available_runs()
    return {"status": "ok", "runs": len(runs), "cache": len(data_cache)}


@app.get("/api/models")
async def api_models():
    """Return model capabilities for frontend timestep filtering."""
    return {
        "models": [
            {
                "name": "icon-d2",
                "label": "ICON-D2 (2.2km)",
                "maxHours": 48,
                "timesteps": list(range(1, 49)),  # Every hour 1-48
                "resolution": 2.2,
                "updateInterval": 3
            },
            {
                "name": "icon-eu",
                "label": "ICON-EU (6.5km)",
                "maxHours": 120,
                "timesteps": list(range(49, 79)) + list(range(81, 121, 3)),  # 49-78, then 81,84,87...120
                "resolution": 6.5,
                "updateInterval": 6
            }
        ]
    }


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
    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used, keys=symbol_keys)

    lat = d["lat"]  # 1D
    lon = d["lon"]  # 1D

    # Bbox-slice BEFORE heavy computation.
    # Use one-cell padding around viewport to stabilize aggregation at screen borders during pan.
    pad = cell_size
    li, lo = _bbox_indices(lat, lon, lat_min - pad, lon_min - pad, lat_max + pad, lon_max + pad)
    if li is not None and len(li) == 0:
        return {"symbols": [], "run": run, "model": model_used, "validTime": d["validTime"], "cellSize": cell_size, "count": 0}

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

    symbols = []
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            lat_c = (lat_lo + lat_hi) / 2
            lon_c = (lon_lo + lon_hi) / 2
            
            # Skip cells outside bbox
            if lat_hi < lat_min or lat_lo > lat_max or lon_hi < lon_min or lon_lo > lon_max:
                continue

            # Find grid indices in cell (using cropped coordinate arrays)
            lat_mask = (c_lat >= lat_lo) & (c_lat < lat_hi)
            lon_mask = (c_lon >= lon_lo) & (c_lon < lon_hi)
            cli = np.where(lat_mask)[0]
            clo = np.where(lon_mask)[0]

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

            # Determine aggregated symbol
            # Rule set:
            # 1) if any ww>10 is present -> show worst ww by severity grading
            # 2) else if convection exists -> show convective cloud type tied to highest hbas_sc
            #    (or htop_dc for blue thermal)
            # 3) else -> average ceiling and derive non-convective cloud type from that average
            sym = "clear"
            cb_hm = None
            label = None
            best_ii = int(cli[len(cli) // 2])
            best_jj = int(clo[len(clo) // 2])

            # 1) Significant weather aggregation (ww > 10)
            sig_positions = []
            for i_local, ii in enumerate(cli):
                for j_local, jj in enumerate(clo):
                    ww_raw = cell_ww[i_local, j_local]
                    ww_val = int(ww_raw) if np.isfinite(ww_raw) else 0
                    if ww_val > 10:
                        sig_positions.append((ii, jj, ww_val))

            if sig_positions:
                # Highest severity rank wins; tie-break on larger ww code
                best_sig = max(sig_positions, key=lambda x: (ww_severity_rank(x[2]), x[2]))
                best_ii, best_jj, best_ww = best_sig
                sym = ww_to_symbol(best_ww) or "clear"

            else:
                # Build cloud candidates for all points in this aggregate cell
                cumulus_candidates = []   # (hbas_sc, ct, ii, jj)
                blue_candidates = []      # (htop_dc, ct, ii, jj)
                ceil_vals = []

                for ii in cli:
                    for jj in clo:
                        ceil_v = float(ceil_arr[ii, jj]) if np.isfinite(ceil_arr[ii, jj]) else float("nan")
                        ct = classify_point(
                            float(c_clcl[ii, jj]), float(c_clcm[ii, jj]),
                            float(c_clch[ii, jj]), float(c_cape[ii, jj]),
                            float(c_htop_dc[ii, jj]), float(c_hbas_sc[ii, jj]),
                            float(c_htop_sc[ii, jj]), float(c_lpi[ii, jj]),
                            ceil_v
                        )

                        if ct in ("cu_hum", "cu_con", "cb"):
                            hbas = float(c_hbas_sc[ii, jj])
                            if np.isfinite(hbas) and hbas > 0:
                                cumulus_candidates.append((hbas, ct, int(ii), int(jj)))
                        elif ct == "blue_thermal":
                            htop = float(c_htop_dc[ii, jj])
                            if not np.isfinite(htop) or htop <= 0:
                                htop = float(c_htop_sc[ii, jj]) if np.isfinite(c_htop_sc[ii, jj]) else -1.0
                            if htop > 0:
                                blue_candidates.append((htop, ct, int(ii), int(jj)))

                        if np.isfinite(ceil_v) and 0 < ceil_v < 20000:
                            ceil_vals.append(ceil_v)

                # 2) Convective aggregation
                # Priority rule:
                # - if any cumulus exists, show highest hbas_sc cumulus type
                # - only if no cumulus exists, allow blue thermal (highest htop_dc)
                if cumulus_candidates:
                    best_metric, best_ct, best_ii, best_jj = max(cumulus_candidates, key=lambda x: x[0])
                    sym = best_ct
                    cb_hm = int(best_metric / 100) if best_metric > 0 else None
                elif blue_candidates:
                    best_metric, best_ct, best_ii, best_jj = max(blue_candidates, key=lambda x: x[0])
                    sym = best_ct
                    cb_hm = int(best_metric / 100) if best_metric > 0 else None

                # 3) Non-convective aggregation by average ceiling
                else:
                    if len(ceil_vals) == 0:
                        sym = "clear"
                    else:
                        avg_ceil = float(np.mean(ceil_vals))
                        if avg_ceil < 2500:
                            sym = "st"
                        elif avg_ceil > 7000:
                            sym = "ci"
                        else:
                            sym = "ac"
                        cb_hm = int(avg_ceil / 100)

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

    return {
        "symbols": symbols,
        "run": run,
        "model": model_used,
        "validTime": d["validTime"],
        "cellSize": cell_size,
        "count": len(symbols)
    }


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

    run, step, model_used = resolve_time(time, model)
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
    run, step, model_used = resolve_time(time, model)
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
    overlay_values = {
        "total_precip": None, "rain": None, "snow": None, "hail": None,
        "clouds": None, "thermals": None, "ceiling": None, "cloud_base": None,
        "sigwx": ww_max,
    }

    try:
        prr = float(np.nanmean(d["prr_gsp"][np.ix_(li, lo)])) if "prr_gsp" in d else 0.0
        prs = float(np.nanmean(d["prs_gsp"][np.ix_(li, lo)])) if "prs_gsp" in d else 0.0
        prg = float(np.nanmean(d["prg_gsp"][np.ix_(li, lo)])) if "prg_gsp" in d else 0.0
        overlay_values["total_precip"] = round((prr + prs + prg) * 3600, 2)
        overlay_values["rain"] = round(prr * 3600, 2)
        overlay_values["snow"] = round(prs * 3600, 2)
        overlay_values["hail"] = round(prg * 3600, 2)
    except Exception:
        pass

    if "clcl" in d:
        overlay_values["clouds_low"] = round(float(np.nanmean(d["clcl"][np.ix_(li, lo)])), 1)
    if "clcm" in d:
        overlay_values["clouds_mid"] = round(float(np.nanmean(d["clcm"][np.ix_(li, lo)])), 1)
    if "clch" in d:
        overlay_values["clouds_high"] = round(float(np.nanmean(d["clch"][np.ix_(li, lo)])), 1)
    if "clct" in d:
        overlay_values["clouds_total"] = round(float(np.nanmean(d["clct"][np.ix_(li, lo)])), 1)
    if "clct_mod" in d:
        clct_mod_val = float(np.nanmean(d["clct_mod"][np.ix_(li, lo)]))
        # Normalize to percent if data appears to be fractional (0..1)
        if np.isfinite(clct_mod_val) and clct_mod_val <= 1.5:
            clct_mod_val *= 100.0
        overlay_values["clouds_total_mod"] = round(clct_mod_val, 1) if np.isfinite(clct_mod_val) else None
    if "cape_ml" in d:
        overlay_values["thermals"] = round(float(np.nanmax(d["cape_ml"][np.ix_(li, lo)])), 1)

    ceil_vals = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20000)] if ceil_cell.size else np.array([])
    overlay_values["ceiling"] = round(float(np.min(ceil_vals)), 0) if len(ceil_vals) > 0 else None
    if "hbas_sc" in d:
        hbas_vals = d["hbas_sc"][np.ix_(li, lo)]
        hbas_valid = hbas_vals[hbas_vals > 0]
        overlay_values["cloud_base"] = round(float(np.min(hbas_valid)), 0) if len(hbas_valid) > 0 else None
    if "htop_dc" in d:
        htop_dc_vals = d["htop_dc"][np.ix_(li, lo)]
        htop_dc_valid = htop_dc_vals[htop_dc_vals > 0]
        overlay_values["dry_conv_top"] = round(float(np.max(htop_dc_valid)), 0) if len(htop_dc_valid) > 0 else None
    if "htop_sc" in d and "hbas_sc" in d:
        thick = np.maximum(0, d["htop_sc"][np.ix_(li, lo)] - d["hbas_sc"][np.ix_(li, lo)])
        thick_valid = thick[thick > 0]
        overlay_values["conv_thickness"] = round(float(np.max(thick_valid)), 0) if len(thick_valid) > 0 else None

    # Soaring data
    if "ashfl_s" in d and "mh" in d and "t_2m" in d:
        ashfl_cell = d["ashfl_s"][np.ix_(li, lo)]
        mh_cell = d["mh"][np.ix_(li, lo)]
        t2m_cell = d["t_2m"][np.ix_(li, lo)]
        wstar = calc_wstar(ashfl_cell, None, mh_cell, t2m_cell)
        overlay_values["wstar"] = round(float(np.nanmean(wstar)), 1)
        overlay_values["climb_rate"] = round(max(float(np.nanmean(wstar)) - 0.7, 0.0), 1)
        overlay_values["bl_height"] = round(float(np.nanmean(mh_cell)), 0)
        if "td_2m" in d and "hsurf" in d:
            td2m_cell = d["td_2m"][np.ix_(li, lo)]
            hsurf_cell = d["hsurf"][np.ix_(li, lo)]
            lcl = calc_lcl(t2m_cell, td2m_cell, hsurf_cell)
            thermal_agl = calc_thermal_height(mh_cell, lcl, hsurf_cell)
            reachable = calc_reachable_distance(thermal_agl)
            overlay_values["lcl"] = round(float(np.nanmean(lcl)), 0)
            overlay_values["reachable"] = round(float(np.nanmean(reachable)), 1)

    # Wind data (match selected wind level and aggregation cell used by wind symbols when zoom is provided)
    if wind_level == "10m":
        u_key, v_key = "u_10m", "v_10m"
    else:
        u_key, v_key = f"u_{wind_level}hpa", f"v_{wind_level}hpa"

    if u_key in d and v_key in d:
        if zoom is not None:
            cell_sizes = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.02}
            cell_size = cell_sizes.get(int(zoom), 0.25)
            anchor_lat = float(lat_arr.min())
            anchor_lon = float(lon_arr.min())
            lat_lo = anchor_lat + math.floor((lat - anchor_lat) / cell_size) * cell_size
            lon_lo = anchor_lon + math.floor((lon - anchor_lon) / cell_size) * cell_size
            lat_hi = lat_lo + cell_size
            lon_hi = lon_lo + cell_size

            wli = np.where((lat_arr >= lat_lo) & (lat_arr < lat_hi))[0]
            wlo = np.where((lon_arr >= lon_lo) & (lon_arr < lon_hi))[0]
            if len(wli) == 0 or len(wlo) == 0:
                wli, wlo = li, lo
        else:
            wli, wlo = li, lo

        u_cell = d[u_key][np.ix_(wli, wlo)]
        v_cell = d[v_key][np.ix_(wli, wlo)]
        u_mean = float(np.nanmean(u_cell))
        v_mean = float(np.nanmean(v_cell))
        if np.isfinite(u_mean) and np.isfinite(v_mean):
            overlay_values["wind_speed"] = round(math.sqrt(u_mean**2 + v_mean**2) * 1.94384, 1)  # knots
            overlay_values["wind_dir"] = round((math.degrees(math.atan2(-u_mean, -v_mean)) + 360) % 360, 0)
    result["overlay_values"] = overlay_values

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

import json
from datetime import datetime as dt

FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

@app.post("/api/feedback")
async def api_feedback(request: Request):
    """Store user feedback with context."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(400, "Message is required")

    entry = {
        "id": int(dt.now(timezone.utc).timestamp() * 1000),
        "timestamp": dt.now(timezone.utc).isoformat() + "Z",
        "type": body.get("type", "general"),
        "message": message,
        "context": body.get("context", {}),
        "userAgent": body.get("userAgent", ""),
        "screen": body.get("screen", ""),
        "status": "new",
    }

    # Load existing feedback
    feedback = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedback = json.load(f)
        except Exception:
            feedback = []

    feedback.append(entry)

    # Write back
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)

    logger.info(f"Feedback received: [{entry['type']}] {message[:80]}")
    return {"status": "ok", "id": entry["id"]}


@app.get("/api/feedback")
async def api_feedback_list():
    """List all feedback entries (for admin use)."""
    if not os.path.exists(FEEDBACK_FILE):
        return {"feedback": [], "count": 0}
    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedback = json.load(f)
        return {"feedback": feedback, "count": len(feedback)}
    except Exception:
        return {"feedback": [], "count": 0}


# ─── Overlay endpoint ───

# Colormaps for overlay layers
import io
import colorsys
from PIL import Image

def colormap_total_precip(total_rate):
    """Total precipitation - sum of rain/snow/graupel. Blue colormap."""
    mmh = total_rate * 3600  # kg/m²/s to mm/h
    if mmh < 0.1:
        return None
    intensity = min(mmh / 5.0, 1.0)  # 0-5 mm/h scale
    r = int(150 * (1 - intensity))
    g = int(180 + 75 * (1 - intensity))
    b = 255
    alpha = int(120 + 135 * intensity)
    return (r, g, b, alpha)

def colormap_rain(rain_rate):
    """Rain only - light cyan (light rain) → dark blue (heavy rain)."""
    mmh = rain_rate * 3600
    if mmh < 0.1:
        return None
    intensity = min(mmh / 5.0, 1.0)  # 0-5 mm/h scale
    r = int(180 - 160 * intensity)
    g = int(220 - 160 * intensity)
    b = int(255 - 75 * intensity)
    alpha = int(130 + 125 * intensity)
    return (r, g, b, alpha)

def colormap_snow(snow_rate):
    """Snow only - light pink (light snow) → dark purple (heavy snow)."""
    mmh = snow_rate * 3600
    if mmh < 0.1:
        return None
    intensity = min(mmh / 5.0, 1.0)  # 0-5 mm/h scale
    r = int(255 - 135 * intensity)
    g = int(200 - 160 * intensity)
    b = int(255 - 95 * intensity)
    alpha = int(130 + 125 * intensity)
    return (r, g, b, alpha)

def colormap_hail(graupel_rate):
    """Hail/graupel - red/orange colormap."""
    mmh = graupel_rate * 3600
    if mmh < 0.1:
        return None
    intensity = min(mmh / 5.0, 1.0)  # 0-5 mm/h scale
    r = int(200 + 55 * intensity)
    g = int(80 + 80 * (1 - intensity))
    b = int(20 * (1 - intensity))
    alpha = int(130 + 125 * intensity)
    return (r, g, b, alpha)

def colormap_sigwx(val):
    """Significant weather coloring by ww code.

    Rules:
    - ww <= 3: transparent
    - ww 4..9: gray shades
    - ww >= 10: unique color per ww code (deterministic), grouped by broad weather family
    """
    ww = int(val)
    # Low ww classes in grayscale with explicit contrast for 0..3
    if ww == 0:
        return None
    if ww == 1:
        return (205, 205, 205, 165)  # slight gray
    if ww == 2:
        return (145, 145, 145, 175)  # middle gray
    if ww == 3:
        return (85, 85, 85, 190)     # dark gray
    if ww < 10:
        # ww 4..9 continue darker grayscale ramp
        g = int(120 - (ww - 4) * 8)  # 120 -> 80
        g = max(75, min(130, g))
        return (g, g, g, 185)

    # Broad group base hues (keeps weather-family feel)
    if ww in (45, 48):          # fog
        base_h = 45 / 360.0
    elif 50 <= ww <= 59:        # drizzle/freezing drizzle
        base_h = 95 / 360.0
    elif 60 <= ww <= 69:        # rain/freezing rain
        base_h = 130 / 360.0
    elif 70 <= ww <= 79:        # snow
        base_h = 265 / 360.0
    elif 80 <= ww <= 84:        # rain showers
        base_h = 175 / 360.0
    elif 85 <= ww <= 86:        # snow showers
        base_h = 280 / 360.0
    elif 95 <= ww <= 99:        # thunderstorm
        base_h = 350 / 360.0
    else:
        base_h = 210 / 360.0

    # Unique per-code hue jitter: golden-angle style distribution
    hue = (base_h + ((ww * 0.61803398875) % 1.0) * 0.18) % 1.0
    sat = 0.82 if ww >= 95 else 0.78
    val_v = 0.95 if ww >= 95 else 0.88
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val_v)
    r, g, b = int(r_f * 255), int(g_f * 255), int(b_f * 255)
    alpha = 210 if ww >= 95 else 190
    return (r, g, b, alpha)


def colormap_clouds(val):
    """Cloud cover 0-100% → high-contrast grayscale with constant opacity."""
    pct = float(val)
    if pct < 1:
        return None
    t = min(pct / 100.0, 1.0)
    # Increased contrast: very light gray (low cloud) -> dark gray (overcast)
    grey = int(225 - 180 * t)  # 225 -> 45
    alpha = 210  # fixed opacity (no opacity encoding)
    return (grey, grey, grey, alpha)

def colormap_thermals(val):
    """Thermal strength based on CAPE (J/kg).

    Show only CAPE >= 50 J/kg.
    Clamp color scale to 50..1000 J/kg (values above use max color).
    """
    cape = float(val)
    if cape < 50:
        return None
    t = min(max((cape - 50.0) / 950.0, 0.0), 1.0)  # 50..1000
    r = int(50 + 205 * t)
    g = int(180 * (1 - t))
    b = int(50 * (1 - t))
    alpha = int(90 + 130 * t)
    return (r, g, b, alpha)

def colormap_ceiling(ceiling_m):
    """Ceiling height - red (low) → yellow → green (high). Range 0-9900m."""
    if ceiling_m >= 9900 or ceiling_m <= 0:
        return None
    h = max(0, min(ceiling_m, 9900))
    t = h / 9900.0  # 0 at 0m (low/red), 1 at 9900m (high/green)
    r = int(220 * (1 - t))
    g = int(60 + 180 * t)
    b = int(30 + 50 * t)
    alpha = int(200 - 60 * t)
    return (r, g, b, alpha)

def colormap_hbas_sc(hbas_m):
    """Stratocumulus cloud base - red (low) → yellow → green (high). Range 0-5000m."""
    if hbas_m <= 0:
        return None
    h = max(0, min(hbas_m, 5000))
    t = h / 5000.0  # 0 at 0m (low/red), 1 at 5000m (high/green)
    r = int(220 * (1 - t))
    g = int(60 + 180 * t)
    b = int(30 + 50 * t)
    alpha = int(200 - 60 * t)
    return (r, g, b, alpha)

def colormap_wstar(val):
    """Thermal strength W* — green (weak) → yellow → red (strong). Range 0-5 m/s."""
    if val < 0.2:
        return None
    t = min(val / 5.0, 1.0)
    r = int(50 + 205 * t)
    g = int(200 - 80 * t)
    b = int(50 * (1 - t))
    alpha = int(100 + 130 * t)
    return (r, g, b, alpha)

def colormap_climb_rate(val):
    """Expected climb rate — green (weak) → yellow → red (strong). Range 0-5 m/s, capped at 0."""
    if val < 0.1:
        return None
    t = min(val / 5.0, 1.0)
    r = int(50 + 205 * t)
    g = int(200 - 80 * t)
    b = int(50 * (1 - t))
    alpha = int(100 + 130 * t)
    return (r, g, b, alpha)

def colormap_lcl(val):
    """LCL / cumulus cloud base — red (low) → green (high). Range 0-5000m MSL."""
    if val < 50:
        return None
    t = min(val / 5000.0, 1.0)
    r = int(220 * (1 - t))
    g = int(60 + 180 * t)
    b = int(30 + 50 * t)
    alpha = int(180 - 40 * t)
    return (r, g, b, alpha)

def colormap_reachable(val):
    """Reachable distance — red (short) → yellow → green (far). Range 0-200 km."""
    if val < 1:
        return None
    t = min(val / 200.0, 1.0)
    r = int(220 * (1 - t))
    g = int(80 + 160 * t)
    b = int(50)
    alpha = int(120 + 100 * t)
    return (r, g, b, alpha)

def colormap_conv_thickness(val):
    """Convective cloud thickness (htop_sc - hbas_sc), 0-6000m."""
    if val is None:
        return None
    d = float(val)
    if d <= 0:
        return None
    t = min(d / 6000.0, 1.0)
    r = int(240 * t + 40 * (1 - t))
    g = int(220 * (1 - t) + 80 * t)
    b = int(60 * (1 - t) + 40 * t)
    alpha = 190
    return (r, g, b, alpha)


OVERLAY_CONFIGS = {
    "total_precip":    {"var": "total_precip", "cmap": colormap_total_precip, "computed": True},
    "rain":            {"var": "prr_gsp", "cmap": colormap_rain},
    "snow":            {"var": "prs_gsp", "cmap": colormap_snow},
    "hail":            {"var": "prg_gsp", "cmap": colormap_hail},
    "clouds_low":      {"var": "clcl", "cmap": colormap_clouds},
    "clouds_mid":      {"var": "clcm", "cmap": colormap_clouds},
    "clouds_high":     {"var": "clch", "cmap": colormap_clouds},
    "clouds_total":    {"var": "clct", "cmap": colormap_clouds},
    "clouds_total_mod": {"var": "clct_mod", "cmap": colormap_clouds},
    "dry_conv_top":    {"var": "htop_dc", "cmap": colormap_ceiling},
    "sigwx":           {"var": "ww", "cmap": colormap_sigwx},
    "ceiling":         {"var": "ceiling", "cmap": colormap_ceiling},
    "cloud_base":      {"var": "hbas_sc", "cmap": colormap_hbas_sc},
    "conv_thickness":  {"var": "conv_thickness", "cmap": colormap_conv_thickness, "computed": True},
    "thermals":        {"var": "cape_ml", "cmap": colormap_thermals},
    "wstar":           {"var": "wstar", "cmap": colormap_wstar, "computed": True},
    "climb_rate":      {"var": "climb_rate", "cmap": colormap_climb_rate, "computed": True},
    "lcl":             {"var": "lcl", "cmap": colormap_lcl, "computed": True},
    "reachable":       {"var": "reachable", "cmap": colormap_reachable, "computed": True},
}


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
    overlay_keys = ["lat", "lon"]
    if cfg.get("computed"):
        if cfg["var"] == "total_precip":
            overlay_keys += ["prr_gsp", "prs_gsp", "prg_gsp"]
        elif cfg["var"] == "conv_thickness":
            overlay_keys += ["htop_sc", "hbas_sc"]
        elif cfg["var"] in ("wstar", "climb_rate"):
            overlay_keys += ["ashfl_s", "mh", "t_2m"]
        elif cfg["var"] in ("lcl", "reachable"):
            overlay_keys += ["ashfl_s", "mh", "t_2m", "td_2m", "hsurf"]
    else:
        overlay_keys.append(cfg["var"])

    run, step, model_used = resolve_time(time, model)
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
        if cfg["var"] == "total_precip":
            prr = d["prr_gsp"][np.ix_(li, lo)]
            prs = d["prs_gsp"][np.ix_(li, lo)]
            prg = d["prg_gsp"][np.ix_(li, lo)]
            cropped = prr + prs + prg
        elif cfg["var"] == "conv_thickness":
            htop_sc = d["htop_sc"][np.ix_(li, lo)] if "htop_sc" in d else np.zeros((len(li), len(lo)))
            hbas_sc = d["hbas_sc"][np.ix_(li, lo)] if "hbas_sc" in d else np.zeros((len(li), len(lo)))
            cropped = np.maximum(0, htop_sc - hbas_sc)
        elif cfg["var"] in ("wstar", "climb_rate", "lcl", "reachable"):
            # Soaring calculations require specific variables
            if "ashfl_s" not in d or "mh" not in d or "t_2m" not in d:
                raise HTTPException(404, "Soaring data not available for this timestep (re-ingestion needed)")
            ashfl_s = d["ashfl_s"][np.ix_(li, lo)]
            mh = d["mh"][np.ix_(li, lo)]
            t_2m = d["t_2m"][np.ix_(li, lo)]

            if cfg["var"] == "wstar":
                wstar = calc_wstar(ashfl_s, None, mh, t_2m, dt_seconds=3600)
                cropped = wstar
            elif cfg["var"] == "climb_rate":
                wstar = calc_wstar(ashfl_s, None, mh, t_2m, dt_seconds=3600)
                cropped = calc_climb_rate(wstar)
            elif cfg["var"] == "lcl":
                if "td_2m" not in d or "hsurf" not in d:
                    raise HTTPException(404, "LCL data not available (re-ingestion needed)")
                td_2m = d["td_2m"][np.ix_(li, lo)]
                hsurf = d["hsurf"][np.ix_(li, lo)]
                lcl_amsl = calc_lcl(t_2m, td_2m, hsurf)
                # Show as MSL
                cropped = lcl_amsl
            elif cfg["var"] == "reachable":
                if "td_2m" not in d or "hsurf" not in d:
                    raise HTTPException(404, "Reachable data not available (re-ingestion needed)")
                td_2m = d["td_2m"][np.ix_(li, lo)]
                hsurf = d["hsurf"][np.ix_(li, lo)]
                lcl_amsl = calc_lcl(t_2m, td_2m, hsurf)
                thermal_agl = calc_thermal_height(mh, lcl_amsl, hsurf)
                cropped = calc_reachable_distance(thermal_agl)
        else:
            raise HTTPException(400, f"Unknown computed variable: {cfg['var']}")
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
            # Normalize to percent if clct_mod is fractional (0..1)
            vmax_local = float(np.nanmax(cropped)) if np.size(cropped) else float("nan")
            if np.isfinite(vmax_local) and vmax_local <= 1.5:
                cropped = cropped * 100.0
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
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Run": run,
            "X-ValidTime": d["validTime"],
            "X-Model": model_used,
            "X-Bbox": f"{actual_lat_min},{actual_lon_min},{actual_lat_max},{actual_lon_max}",
            "Access-Control-Expose-Headers": "X-Bbox, X-Run, X-ValidTime, X-Model",
        }
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


def _colorize_layer_vectorized(layer: str, sampled: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fast vectorized RGBA colorization for common Skyview overlay layers."""
    h, w = sampled.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if not np.any(valid):
        return rgba

    v = sampled

    def set_rgba(mask, r, g, b, a):
        if np.any(mask):
            rgba[..., 0][mask] = np.clip(r[mask] if isinstance(r, np.ndarray) else r, 0, 255).astype(np.uint8) if isinstance(r, np.ndarray) else np.uint8(r)
            rgba[..., 1][mask] = np.clip(g[mask] if isinstance(g, np.ndarray) else g, 0, 255).astype(np.uint8) if isinstance(g, np.ndarray) else np.uint8(g)
            rgba[..., 2][mask] = np.clip(b[mask] if isinstance(b, np.ndarray) else b, 0, 255).astype(np.uint8) if isinstance(b, np.ndarray) else np.uint8(b)
            rgba[..., 3][mask] = np.clip(a[mask] if isinstance(a, np.ndarray) else a, 0, 255).astype(np.uint8) if isinstance(a, np.ndarray) else np.uint8(a)

    if layer in ("total_precip", "rain", "snow", "hail"):
        mmh = v * 3600.0
        m = valid & (mmh >= 0.1)
        if np.any(m):
            t = np.clip(mmh / 5.0, 0.0, 1.0)
            if layer == "total_precip":
                r = 150 * (1 - t)
                g = 180 + 75 * (1 - t)
                b = np.full_like(t, 255.0)
                a = 120 + 135 * t
            elif layer == "rain":
                r = 180 - 160 * t
                g = 220 - 160 * t
                b = 255 - 75 * t
                a = 130 + 125 * t
            elif layer == "snow":
                r = 255 - 135 * t
                g = 200 - 160 * t
                b = 255 - 95 * t
                a = 130 + 125 * t
            else:  # hail
                r = 200 + 55 * t
                g = 80 + 80 * (1 - t)
                b = 20 * (1 - t)
                a = 130 + 125 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer.startswith("clouds_"):
        m = valid & (v >= 1)
        if np.any(m):
            t = np.clip(v / 100.0, 0.0, 1.0)
            grey = 225 - 180 * t
            set_rgba(m, grey, grey, grey, 210)
        return rgba

    if layer in ("ceiling", "dry_conv_top"):
        m = valid & (v > 0) & (v < 9900)
        if np.any(m):
            t = np.clip(v / 9900.0, 0.0, 1.0)
            r = 220 * (1 - t)
            g = 60 + 180 * t
            b = 30 + 50 * t
            a = 200 - 60 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer == "cloud_base":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 5000.0, 0.0, 1.0)
            r = 220 * (1 - t)
            g = 60 + 180 * t
            b = 30 + 50 * t
            a = 200 - 60 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer == "conv_thickness":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 6000.0, 0.0, 1.0)
            r = 240 * t + 40 * (1 - t)
            g = 220 * (1 - t) + 80 * t
            b = 60 * (1 - t) + 40 * t
            set_rgba(m, r, g, b, 190)
        return rgba

    if layer == "thermals":
        # CAPE_ml coloring only from 50 J/kg upwards; clamp scale at 1000 J/kg
        m = valid & (v >= 50)
        if np.any(m):
            t = np.clip((v - 50.0) / 950.0, 0.0, 1.0)
            r = 50 + 205 * t
            g = 180 * (1 - t)
            b = 50 * (1 - t)
            a = 90 + 130 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer in ("wstar", "climb_rate"):
        th = 0.2 if layer == "wstar" else 0.1
        m = valid & (v >= th)
        if np.any(m):
            t = np.clip(v / 5.0, 0.0, 1.0)
            r = 50 + 205 * t
            g = 200 - 80 * t
            b = 50 * (1 - t)
            a = 100 + 130 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer == "lcl":
        m = valid & (v >= 50)
        if np.any(m):
            t = np.clip(v / 5000.0, 0.0, 1.0)
            r = 220 * (1 - t)
            g = 60 + 180 * t
            b = 30 + 50 * t
            a = 180 - 40 * t
            set_rgba(m, r, g, b, a)
        return rgba

    if layer == "reachable":
        m = valid & (v >= 1)
        if np.any(m):
            t = np.clip(v / 200.0, 0.0, 1.0)
            r = 220 * (1 - t)
            g = 80 + 160 * t
            b = np.full_like(t, 50.0)
            a = 120 + 100 * t
            set_rgba(m, r, g, b, a)
        return rgba

    # sigwx and any uncommon/fallback layers: keep exact semantics via existing per-pixel cmap.
    cmap_fn = OVERLAY_CONFIGS[layer]["cmap"]
    for yy in range(h):
        for xx in range(w):
            if not valid[yy, xx]:
                continue
            color = cmap_fn(v[yy, xx])
            if color:
                rgba[yy, xx] = color
    return rgba


@app.get("/api/cache_stats")
async def api_cache_stats():
    _tile_cache_prune("desktop")
    _tile_cache_prune("mobile")
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
    recent = list(perf_recent)
    recent_n = len(recent)
    if recent_n:
        recent_avg_ms = sum(r['ms'] for r in recent) / recent_n
        recent_hit_rate = sum(r['hit'] for r in recent) / recent_n
    else:
        recent_avg_ms = None
        recent_hit_rate = None

    total_req = perf_totals['requests']
    total_avg_ms = (perf_totals['totalMs'] / total_req) if total_req else None
    total_hit_rate = (perf_totals['hits'] / total_req) if total_req else None

    payload = {
        'recentWindow': {
            'size': recent_n,
            'maxSize': perf_recent.maxlen,
            'avgMs': recent_avg_ms,
            'hitRate': recent_hit_rate,
        },
        'totals': {
            'requests': total_req,
            'hits': perf_totals['hits'],
            'misses': perf_totals['misses'],
            'avgMs': total_avg_ms,
            'hitRate': total_hit_rate,
        }
    }

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

    overlay_keys = ["lat", "lon"]
    if cfg.get("computed"):
        if cfg["var"] == "total_precip":
            overlay_keys += ["prr_gsp", "prs_gsp", "prg_gsp"]
        elif cfg["var"] == "conv_thickness":
            overlay_keys += ["htop_sc", "hbas_sc"]
        elif cfg["var"] in ("wstar", "climb_rate"):
            overlay_keys += ["ashfl_s", "mh", "t_2m"]
        elif cfg["var"] in ("lcl", "reachable"):
            overlay_keys += ["ashfl_s", "mh", "t_2m", "td_2m", "hsurf"]
    else:
        overlay_keys.append(cfg["var"])

    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used, keys=overlay_keys)
    lat = d["lat"]
    lon = d["lon"]

    cache_key = f"{client_class}|{model_used}|{run}|{step}|{layer}|{z}|{x}|{y}"
    cached = _tile_cache_get(client_class, cache_key)
    if cached is not None:
        _perf_record((perf_counter() - t0) * 1000.0, True)
        return Response(content=cached, media_type="image/png", headers={
            "Cache-Control": "public, max-age=300",
            "X-Run": run,
            "X-ValidTime": d["validTime"],
            "X-Model": model_used,
            "X-Cache": "HIT",
            "Access-Control-Expose-Headers": "X-Run, X-ValidTime, X-Model, X-Cache",
        })

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
        _tile_cache_set(client_class, cache_key, png)
        _perf_record((perf_counter() - t0) * 1000.0, False)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=300", "X-Cache": "MISS"})

    # source field (full grid), with memoization for expensive computed layers
    if cfg.get("computed"):
        comp_key = f"{model_used}|{run}|{step}|{layer}"
        src = _computed_cache_get(comp_key)
        if src is None:
            v = cfg["var"]
            if v == "total_precip":
                src = d["prr_gsp"] + d["prs_gsp"] + d["prg_gsp"]
            elif v == "conv_thickness":
                src = np.maximum(0, d["htop_sc"] - d["hbas_sc"])
            elif v in ("wstar", "climb_rate"):
                wstar = calc_wstar(d["ashfl_s"], None, d["mh"], d["t_2m"], dt_seconds=3600)
                src = wstar if v == "wstar" else calc_climb_rate(wstar)
            elif v in ("lcl", "reachable"):
                lcl_amsl = calc_lcl(d["t_2m"], d["td_2m"], d["hsurf"])
                if v == "lcl":
                    src = lcl_amsl
                else:
                    thermal_agl = calc_thermal_height(d["mh"], lcl_amsl, d["hsurf"])
                    src = calc_reachable_distance(thermal_agl)
            else:
                raise HTTPException(400, f"Unsupported computed layer: {v}")
            _computed_cache_set(comp_key, src)
    else:
        vname = cfg["var"]
        if vname not in d and layer == "clouds_total_mod" and "clct" in d:
            vname = "clct"
        if vname not in d:
            raise HTTPException(404, f"Variable {cfg['var']} unavailable")
        src = d[vname]
        if layer == "clouds_total_mod":
            vmax_local = float(np.nanmax(src)) if np.size(src) else float("nan")
            if np.isfinite(vmax_local) and vmax_local <= 1.5:
                src = src * 100.0

    xs = np.linspace(minx, maxx, 256, endpoint=False) + (maxx - minx) / 512.0
    ys = np.linspace(maxy, miny, 256, endpoint=False) - (maxy - miny) / 512.0
    mx, my = np.meshgrid(xs, ys)
    qlon, qlat = _merc_to_lonlat(mx, my)

    inside = ((qlat >= data_lat_min) & (qlat <= data_lat_max) & (qlon >= data_lon_min) & (qlon <= data_lon_max))
    li = _regular_grid_indices(lat, qlat)
    lo = _regular_grid_indices(lon, qlon)
    sampled = src[li, lo]
    valid = inside & np.isfinite(sampled)

    rgba = _colorize_layer_vectorized(layer, sampled, valid)

    img = Image.fromarray(rgba, mode="RGBA")
    b = io.BytesIO(); img.save(b, format="PNG", optimize=True); png = b.getvalue()
    _tile_cache_set(client_class, cache_key, png)
    _perf_record((perf_counter() - t0) * 1000.0, False)

    return Response(content=png, media_type="image/png", headers={
        "Cache-Control": "public, max-age=300",
        "X-Run": run,
        "X-ValidTime": d["validTime"],
        "X-Model": model_used,
        "X-Cache": "MISS",
        "Access-Control-Expose-Headers": "X-Run, X-ValidTime, X-Model, X-Cache",
    })


# ─── Static Frontend (must be LAST) ───
# html=True serves index.html for directory requests
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
