#!/usr/bin/env python3
"""Skyview FastAPI backend — serves ICON-D2 weather symbols for Leaflet frontend."""

import os
import sys
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

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
    return None  # ww 0-3: use cloud classification


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


def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi):
    """Classify a single grid point's cloud type using SPEC-aligned rules."""
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

    # Stratiform branch (no convection)
    if clcl > 30 and clcm < 20:
        return "st"
    if clcm > 30 and clcl < 20:
        return "ac"
    if clcl < 10 and clcm < 10 and clch > 30:
        return "ci"
    return "clear"


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

    # Bbox-slice BEFORE heavy computation: find indices within requested bbox
    li, lo = _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max)
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
                continue

            # Extract cell data
            cell_ww = ww[np.ix_(cli, clo)]
            max_ww = int(np.nanmax(cell_ww)) if not np.all(np.isnan(cell_ww)) else 0

            # Determine symbol
            # If there is significant weather (>10), always show the worst ww symbol.
            sig_mask = cell_ww > 10
            sym = None
            best_ii = int(cli[len(cli) // 2])
            best_jj = int(clo[len(clo) // 2])

            if np.any(sig_mask):
                ww_sig = cell_ww[sig_mask]
                max_sig = int(np.nanmax(ww_sig))
                sym = ww_to_symbol(max_sig)
                # anchor click point at a location where this worst ww occurs
                sig_positions = np.argwhere(cell_ww == max_sig)
                if len(sig_positions) > 0:
                    i0, j0 = sig_positions[0]
                    best_ii = int(cli[i0])
                    best_jj = int(clo[j0])
            else:
                # Cloud-only cell:
                # - prefer strongest convective symbol by highest convective top
                # - otherwise use lowest cloud-base stratiform type (St > Ac > Ci > clear)
                best_conv_top = -1.0
                best_strat_alt = float("inf")
                best_strat_type = "clear"

                for ii in cli:
                    for jj in clo:
                        ct = classify_point(
                            float(c_clcl[ii, jj]), float(c_clcm[ii, jj]),
                            float(c_clch[ii, jj]), float(c_cape[ii, jj]),
                            float(c_htop_dc[ii, jj]), float(c_hbas_sc[ii, jj]),
                            float(c_htop_sc[ii, jj]), float(c_lpi[ii, jj])
                        )

                        if ct in ("blue_thermal", "cu_hum", "cu_con", "cb"):
                            conv_top = float(c_htop_dc[ii, jj]) if np.isfinite(c_htop_dc[ii, jj]) else 0.0
                            if conv_top <= 0 and np.isfinite(c_htop_sc[ii, jj]):
                                conv_top = float(c_htop_sc[ii, jj])
                            if conv_top > best_conv_top:
                                best_conv_top = conv_top
                                sym = ct
                                best_ii, best_jj = int(ii), int(jj)
                        else:
                            # stratiform fallback: lowest altitude is most restrictive
                            alt = float(ceil_arr[ii, jj]) if np.isfinite(ceil_arr[ii, jj]) else float("inf")
                            if alt <= 0 or alt >= 20000:
                                alt = float(c_hbas_sc[ii, jj]) if np.isfinite(c_hbas_sc[ii, jj]) else float("inf")
                            if alt < best_strat_alt:
                                best_strat_alt = alt
                                best_strat_type = ct
                                best_ii, best_jj = int(ii), int(jj)
                            elif alt == best_strat_alt and CLOUD_PRIORITY.get(ct, 0) > CLOUD_PRIORITY.get(best_strat_type, 0):
                                best_strat_type = ct
                                best_ii, best_jj = int(ii), int(jj)

                if sym is None:
                    sym = best_strat_type

            # Cloud base logic
            cb_hm = None
            label = None
            if sym in ("st", "ac", "ci", "cu_hum", "cu_con", "cb"):
                cell_ceil = ceil_arr[np.ix_(cli, clo)]
                valid_ceil = cell_ceil[(cell_ceil > 0) & (cell_ceil < 20000)]
                if len(valid_ceil) > 0:
                    cb_hm = int(np.min(valid_ceil) / 100)
            elif sym == "blue_thermal":
                cell_htop = c_htop_dc[np.ix_(cli, clo)]
                valid_htop = cell_htop[cell_htop > 0]
                if len(valid_htop) > 0:
                    cb_hm = int(np.max(valid_htop) / 100)
            if cb_hm is not None:
                if cb_hm > 99:
                    cb_hm = 99  # Cap at 9900m
                label = str(cb_hm)

            symbols.append({
                "lat": round(lat_c, 4),
                "lon": round(lon_c, 4),
                "clickLat": round(float(c_lat[best_ii]), 4),
                "clickLon": round(float(c_lon[best_jj]), 4),
                "type": sym,
                "ww": max_ww,
                "cloudBase": cb_hm,
                "label": label,
                "clickable": sym != "clear"
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

    # Bbox-slice before computation
    li, lo = _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max)
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

            barbs.append({
                "lat": round(lat_c, 4),
                "lon": round(lon_c, 4),
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

    best_type = classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi)
    result["cloudType"] = best_type

    ww_max = int(scalar("ww") or 0)
    ww_sym = ww_to_symbol(ww_max) if ww_max > 10 else None
    sym = ww_sym or best_type
    result["symbol"] = sym
    result["cloudTypeName"] = sym.title()
    result["ww"] = ww_max

    # Cloud base: mirror symbols (min ceil or max htop_dc in cell)
    ceil_cell = d["ceiling"][np.ix_(li, lo)] if "ceiling" in d else np.array([])
    htop_cell = d["htop_dc"][np.ix_(li, lo)] if "htop_dc" in d else np.array([])
    valid_ceil = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20000)] if ceil_cell.size else np.array([])
    if sym in ("st", "ac", "ci", "cu_hum", "cu_con", "cb", "fog", "rime_fog") and len(valid_ceil) > 0:
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

    if "clct" in d:
        overlay_values["clouds"] = round(float(np.nanmean(d["clct"][np.ix_(li, lo)])), 1)
    if "cape_ml" in d:
        overlay_values["thermals"] = round(float(np.nanmax(d["cape_ml"][np.ix_(li, lo)])), 1)

    ceil_vals = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20000)] if ceil_cell.size else np.array([])
    overlay_values["ceiling"] = round(float(np.min(ceil_vals)), 0) if len(ceil_vals) > 0 else None
    if "hbas_sc" in d:
        hbas_vals = d["hbas_sc"][np.ix_(li, lo)]
        hbas_valid = hbas_vals[hbas_vals > 0]
        overlay_values["cloud_base"] = round(float(np.min(hbas_valid)), 0) if len(hbas_valid) > 0 else None

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

    # Wind data
    if "u_10m" in d and "v_10m" in d:
        u_cell = d["u_10m"][np.ix_(li, lo)]
        v_cell = d["v_10m"][np.ix_(li, lo)]
        u_mean = float(np.nanmean(u_cell))
        v_mean = float(np.nanmean(v_cell))
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

    - ww < 10: gray shades
    - ww >= 10: distinct per-code colors while preserving broad weather-group hue families
    """
    ww = int(val)
    if ww <= 3:
        return None

    if ww < 10:
        # deterministic gray shades for low/non-sig codes
        g = 180 - (ww * 10)
        return (max(90, g), max(90, g), max(90, g), 150)

    # Grouped hue families with per-code variation
    if ww in (45, 48):  # fog
        base = (190, 150, 40)
    elif 50 <= ww <= 59:  # drizzle/freezing drizzle
        base = (110, 190, 110)
    elif 60 <= ww <= 69:  # rain/freezing rain
        base = (20, 160, 40)
    elif 70 <= ww <= 79:  # solid snow
        base = (140, 110, 230)
    elif 80 <= ww <= 84:  # rain showers
        base = (0, 150, 120)
    elif 85 <= ww <= 86:  # snow showers
        base = (120, 90, 240)
    elif 95 <= ww <= 99:  # thunderstorms
        base = (220, 20, 60)
    else:
        base = (130, 130, 130)

    # Per-code tint for visual separation inside each group
    offset = (ww * 17) % 36 - 18
    r = max(0, min(255, base[0] + offset))
    g = max(0, min(255, base[1] - offset // 2))
    b = max(0, min(255, base[2] + offset // 2))
    alpha = 185 if ww >= 95 else 170
    return (r, g, b, alpha)

def colormap_clouds(val):
    """Cloud cover 0-100% → gray gradient with constant opacity."""
    pct = float(val)
    if pct < 1:
        return None
    t = min(pct / 100.0, 1.0)
    grey = int(170 - 130 * t)  # light gray -> dark gray
    alpha = 180  # fixed opacity (no opacity encoding)
    return (grey, grey, grey, alpha)

def colormap_thermals(val):
    """Thermal strength based on CAPE (J/kg). 0→transparent, >2000→deep red."""
    cape = float(val)
    if cape < 10: return None
    # Scale: 10-2000 J/kg
    t = min(cape / 2000.0, 1.0)
    r = int(50 + 205 * t)
    g = int(180 * (1 - t))
    b = int(50 * (1 - t))
    alpha = int(80 + 140 * t)
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

OVERLAY_CONFIGS = {
    "total_precip":  {"var": "total_precip", "cmap": colormap_total_precip, "computed": True},
    "rain":          {"var": "prr_gsp", "cmap": colormap_rain},
    "snow":          {"var": "prs_gsp", "cmap": colormap_snow},
    "hail":          {"var": "prg_gsp", "cmap": colormap_hail},
    "sigwx":         {"var": "ww", "cmap": colormap_sigwx},
    "clouds":        {"var": "clct", "cmap": colormap_clouds},
    "thermals":      {"var": "cape_ml", "cmap": colormap_thermals},
    "ceiling":       {"var": "ceiling", "cmap": colormap_ceiling},
    "cloud_base":    {"var": "hbas_sc", "cmap": colormap_hbas_sc},
    "wstar":         {"var": "wstar", "cmap": colormap_wstar, "computed": True},
    "climb_rate":    {"var": "climb_rate", "cmap": colormap_climb_rate, "computed": True},
    "lcl":           {"var": "lcl", "cmap": colormap_lcl, "computed": True},
    "reachable":     {"var": "reachable", "cmap": colormap_reachable, "computed": True},
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
    # Grid coordinates are cell centers; we need cell edges for image bounds
    lat_res = float(lat[1] - lat[0]) if len(lat) > 1 else 0.02
    lon_res = float(lon[1] - lon[0]) if len(lon) > 1 else 0.02
    
    # Actual bounds = edges of selected cells (not requested bbox)
    actual_lat_min = float(lat[li[0]]) - lat_res / 2
    actual_lat_max = float(lat[li[-1]]) + lat_res / 2
    actual_lon_min = float(lon[lo[0]]) - lon_res / 2
    actual_lon_max = float(lon[lo[-1]]) + lon_res / 2

    # Handle computed variables
    if cfg.get("computed"):
        if cfg["var"] == "total_precip":
            prr = d["prr_gsp"][np.ix_(li, lo)]
            prs = d["prs_gsp"][np.ix_(li, lo)]
            prg = d["prg_gsp"][np.ix_(li, lo)]
            cropped = prr + prs + prg
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
        var_data = d[cfg["var"]]
        cropped = var_data[np.ix_(li, lo)]
        h, w = cropped.shape

    # Compute output dimensions preserving aspect ratio
    # Use actual cell bounds (not requested bbox) for correct aspect ratio
    aspect = (actual_lon_max - actual_lon_min) / (actual_lat_max - actual_lat_min) if (actual_lat_max - actual_lat_min) > 0 else 1
    out_w = width
    out_h = max(1, int(out_w / aspect))

    # Create RGBA image at native resolution, then resize
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pixels = img.load()
    
    # Single variable rendering
    for y in range(h):
        for x in range(w):
            val = cropped[y, x]
            if np.isnan(val):
                continue
            color = cmap_fn(val)
            if color:
                pixels[x, h - 1 - y] = color

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


# ─── Static Frontend (must be LAST) ───
# html=True serves index.html for directory requests
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
