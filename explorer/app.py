#!/usr/bin/env python3
"""ICON-Explorer Backend — raw data visualization API for ICON weather model data."""

import os
import sys
import io
import math
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

app = FastAPI(title="ICON-Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
EXPLORER_DIR = SCRIPT_DIR

data_cache: Dict[str, Dict[str, Any]] = {}

# Small in-memory LRU caches for rendered overlay tiles (split by client class)
TILE_CACHE_MAX_ITEMS_DESKTOP = 1400
TILE_CACHE_MAX_ITEMS_MOBILE = 700
TILE_CACHE_TTL_SECONDS = 900  # periodic age-based pruning

# key -> (png_bytes, stored_at_epoch)
tile_cache_desktop = OrderedDict()
tile_cache_mobile = OrderedDict()

# basic cache metrics
cache_stats = {
    'hits': 0,
    'misses': 0,
    'evictions': 0,
    'expired': 0,
}

def _cache_for_class(client_class: str):
    if client_class == 'mobile':
        return tile_cache_mobile, TILE_CACHE_MAX_ITEMS_MOBILE
    return tile_cache_desktop, TILE_CACHE_MAX_ITEMS_DESKTOP

def _tile_cache_prune(client_class: str):
    cache, max_items = _cache_for_class(client_class)
    now = time.time()

    # Age pruning (front of OrderedDict is oldest)
    while cache:
        _, (_, ts) = next(iter(cache.items()))
        if (now - ts) <= TILE_CACHE_TTL_SECONDS:
            break
        cache.popitem(last=False)
        cache_stats['expired'] += 1

    # Size pruning
    while len(cache) > max_items:
        cache.popitem(last=False)
        cache_stats['evictions'] += 1

def _tile_cache_get(client_class: str, key: str):
    cache, _ = _cache_for_class(client_class)
    _tile_cache_prune(client_class)

    item = cache.get(key)
    if item is None:
        cache_stats['misses'] += 1
        return None

    png_bytes, ts = item
    if (time.time() - ts) > TILE_CACHE_TTL_SECONDS:
        del cache[key]
        cache_stats['expired'] += 1
        cache_stats['misses'] += 1
        return None

    cache.move_to_end(key)
    cache_stats['hits'] += 1
    return png_bytes

def _tile_cache_set(client_class: str, key: str, png_bytes: bytes):
    cache, _ = _cache_for_class(client_class)
    cache[key] = (png_bytes, time.time())
    cache.move_to_end(key)
    _tile_cache_prune(client_class)

# ─── Variable Metadata ───

VARIABLE_METADATA = {
    # Aviation group
    "ww": {"group": "aviation", "unit": "code", "desc": "Significant weather"},
    "cape_ml": {"group": "aviation", "unit": "J/kg", "desc": "Mixed-layer CAPE"},
    "cin_ml": {"group": "aviation", "unit": "J/kg", "desc": "Mixed-layer CIN"},
    "htop_dc": {"group": "aviation", "unit": "m", "desc": "Dry convection top height"},
    "hbas_sc": {"group": "aviation", "unit": "m", "desc": "Shallow convection cloud base"},
    "htop_sc": {"group": "aviation", "unit": "m", "desc": "Shallow convection cloud top"},
    "lpi": {"group": "aviation", "unit": "J/kg", "desc": "Lightning potential index"},
    "lpi_max": {"group": "aviation", "unit": "J/kg", "desc": "Max lightning potential index"},
    "ceiling": {"group": "aviation", "unit": "m", "desc": "Cloud ceiling height"},
    "clcl": {"group": "aviation", "unit": "%", "desc": "Low cloud cover"},
    "clcm": {"group": "aviation", "unit": "%", "desc": "Medium cloud cover"},
    "clch": {"group": "aviation", "unit": "%", "desc": "High cloud cover"},
    "clct": {"group": "aviation", "unit": "%", "desc": "Total cloud cover"},
    "clct_mod": {"group": "aviation", "unit": "%", "desc": "Modified total cloud cover"},
    "cldepth": {"group": "aviation", "unit": "m", "desc": "Cloud depth"},
    "vis": {"group": "aviation", "unit": "m", "desc": "Visibility"},
    "hzerocl": {"group": "aviation", "unit": "m", "desc": "0°C level height"},
    "snowlmt": {"group": "aviation", "unit": "m", "desc": "Snow line altitude"},
    "mh": {"group": "aviation", "unit": "m", "desc": "Boundary layer height"},
    "hsurf": {"group": "aviation", "unit": "m", "desc": "Surface elevation"},
    
    # Surface group
    "t_2m": {"group": "surface", "unit": "K", "desc": "2m Temperature"},
    "td_2m": {"group": "surface", "unit": "K", "desc": "2m Dew point"},
    "tmax_2m": {"group": "surface", "unit": "K", "desc": "Max 2m temperature"},
    "tmin_2m": {"group": "surface", "unit": "K", "desc": "Min 2m temperature"},
    "relhum_2m": {"group": "surface", "unit": "%", "desc": "2m Relative humidity"},
    "pmsl": {"group": "surface", "unit": "Pa", "desc": "Mean sea level pressure"},
    "ps": {"group": "surface", "unit": "Pa", "desc": "Surface pressure"},
    "u_10m": {"group": "surface", "unit": "m/s", "desc": "10m U wind"},
    "v_10m": {"group": "surface", "unit": "m/s", "desc": "10m V wind"},
    "vmax_10m": {"group": "surface", "unit": "m/s", "desc": "10m Max wind gust"},
    "tot_prec": {"group": "surface", "unit": "kg/m²", "desc": "Total precipitation"},
    "prr_gsp": {"group": "surface", "unit": "kg/m²/s", "desc": "Rain rate (grid-scale)"},
    "prs_gsp": {"group": "surface", "unit": "kg/m²/s", "desc": "Snow rate (grid-scale)"},
    "prg_gsp": {"group": "surface", "unit": "kg/m²/s", "desc": "Graupel rate (grid-scale)"},
    "rain_gsp": {"group": "surface", "unit": "kg/m²", "desc": "Rain amount (grid-scale)"},
    "rain_con": {"group": "surface", "unit": "kg/m²", "desc": "Rain amount (convective)"},
    "snow_gsp": {"group": "surface", "unit": "kg/m²", "desc": "Snow amount (grid-scale)"},
    "snow_con": {"group": "surface", "unit": "kg/m²", "desc": "Snow amount (convective)"},
    "grau_gsp": {"group": "surface", "unit": "kg/m²", "desc": "Graupel amount"},
    "h_snow": {"group": "surface", "unit": "m", "desc": "Snow depth"},
    "freshsnw": {"group": "surface", "unit": "kg/m²", "desc": "Fresh snow amount"},
    "t_g": {"group": "surface", "unit": "K", "desc": "Ground temperature"},
    
    # Severe weather group
    "dbz_cmax": {"group": "severe", "unit": "dBZ", "desc": "Column-max radar reflectivity"},
    "dbz_ctmax": {"group": "severe", "unit": "dBZ", "desc": "Column-max composite reflectivity"},
    "dbz_850": {"group": "severe", "unit": "dBZ", "desc": "850 hPa radar reflectivity"},
    "sdi_2": {"group": "severe", "unit": "index", "desc": "Supercell detection index"},
    "vorw_ctmax": {"group": "severe", "unit": "1/s", "desc": "Max vertical vorticity"},
    "w_ctmax": {"group": "severe", "unit": "m/s", "desc": "Max vertical velocity"},
    "uh_max": {"group": "severe", "unit": "m²/s²", "desc": "Max updraft helicity"},
    "uh_max_low": {"group": "severe", "unit": "m²/s²", "desc": "Max low-level updraft helicity"},
    "uh_max_med": {"group": "severe", "unit": "m²/s²", "desc": "Max mid-level updraft helicity"},
    "echotop": {"group": "severe", "unit": "m", "desc": "Echo top height"},
    "tcond_max": {"group": "severe", "unit": "kg/kg", "desc": "Max total condensate"},
    "tcond10_mx": {"group": "severe", "unit": "kg/kg", "desc": "Max 10min total condensate"},
    
    # Soaring group
    "ashfl_s": {"group": "soaring", "unit": "W/m²", "desc": "Surface sensible heat flux"},
    
    # Wind group (includes surface winds u_10m, v_10m already defined)
    "u_950hpa": {"group": "wind", "unit": "m/s", "desc": "950 hPa U wind"},
    "v_950hpa": {"group": "wind", "unit": "m/s", "desc": "950 hPa V wind"},
    "u_850hpa": {"group": "wind", "unit": "m/s", "desc": "850 hPa U wind"},
    "v_850hpa": {"group": "wind", "unit": "m/s", "desc": "850 hPa V wind"},
    "u_700hpa": {"group": "wind", "unit": "m/s", "desc": "700 hPa U wind"},
    "v_700hpa": {"group": "wind", "unit": "m/s", "desc": "700 hPa V wind"},
    "u_500hpa": {"group": "wind", "unit": "m/s", "desc": "500 hPa U wind"},
    "v_500hpa": {"group": "wind", "unit": "m/s", "desc": "500 hPa V wind"},
    "u_300hpa": {"group": "wind", "unit": "m/s", "desc": "300 hPa U wind"},
    "v_300hpa": {"group": "wind", "unit": "m/s", "desc": "300 hPa V wind"},
}

# ─── Data Loading Helpers ───

def load_data(run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load .npz data for a given run/step/model."""
    cache_key = f"{model}/{run}/{step:03d}"
    
    if cache_key in data_cache:
        cached = data_cache[cache_key]
        if keys is None or all(k in cached for k in keys):
            return cached

    model_dir = model.replace("_", "-")
    path = os.path.join(DATA_DIR, model_dir, run, f"{step:03d}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")

    npz = np.load(path)
    
    if keys is not None:
        load_keys = set(keys) | {"lat", "lon"}
        arrays = {k: npz[k] for k in load_keys if k in npz.files}
        if cache_key in data_cache:
            for k, v in data_cache[cache_key].items():
                if k not in arrays:
                    arrays[k] = v
    else:
        arrays = {k: npz[k] for k in npz.files}

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
    """Build a unified timeline: ICON-D2 for first 48h, ICON-EU for 49-120h."""
    runs = get_available_runs()
    if not runs:
        return None
    
    d2_complete = next((r for r in runs if r["model"] == "icon_d2" and len(r["steps"]) >= 48), None)
    d2_any = next((r for r in runs if r["model"] == "icon_d2"), None)
    d2_run = d2_complete or d2_any
    eu_run = next((r for r in runs if r["model"] == "icon_eu"), None)
    
    if not d2_run and not eu_run:
        return None
    
    primary = d2_run or eu_run
    
    merged_steps = []
    d2_last_valid = None
    
    if d2_run:
        for s in d2_run["steps"]:
            merged_steps.append({**s, "model": "icon_d2", "run": d2_run["run"]})
            if d2_last_valid is None or s["validTime"] > d2_last_valid:
                d2_last_valid = s["validTime"]
    
    if eu_run and d2_last_valid:
        for s in eu_run["steps"]:
            if s["validTime"] > d2_last_valid:
                merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})
    elif eu_run and not d2_run:
        for s in eu_run["steps"]:
            merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})
    
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


def _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max):
    """Return (lat_indices, lon_indices) for points within the bbox."""
    eps = 0.001
    lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
    lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]
    if len(li) == 0 or len(lo) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return li, lo


def _slice_array(arr, li, lo):
    """Slice a 2D array by lat/lon index arrays."""
    return arr[np.ix_(li, lo)]


# ─── API Endpoints ───

@app.get("/api/variables")
async def api_variables():
    """List all available variables with metadata and sample min/max."""
    variables = []
    
    # Try to load a sample timestep to get min/max values
    runs = get_available_runs()
    sample_data = None
    if runs:
        try:
            run = runs[0]["run"]
            step = runs[0]["steps"][0]["step"]
            model = runs[0]["model"]
            sample_data = load_data(run, step, model)
        except Exception:
            pass
    
    for var_name, meta in VARIABLE_METADATA.items():
        var_info = {
            "name": var_name,
            "group": meta["group"],
            "unit": meta["unit"],
            "desc": meta["desc"],
        }
        
        # Add min/max from sample if available
        if sample_data and var_name in sample_data:
            arr = sample_data[var_name]
            if isinstance(arr, np.ndarray) and arr.size > 0:
                valid_data = arr[~np.isnan(arr)]
                if len(valid_data) > 0:
                    var_info["min"] = float(np.min(valid_data))
                    var_info["max"] = float(np.max(valid_data))
        
        variables.append(var_info)
    
    return {"variables": variables}


@app.get("/api/timesteps")
async def api_timesteps():
    """Return available runs and timesteps."""
    merged = get_merged_timeline()
    return {"runs": get_available_runs(), "merged": merged}


@app.get("/api/overlay")
async def api_overlay(
    var: str = Query(..., description="Variable name"),
    bbox: str = Query("43.18,-3.94,58.08,20.34", description="lat_min,lon_min,lat_max,lon_max"),
    time: str = Query("latest", description="ISO time or 'latest'"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu"),
    width: int = Query(800, ge=100, le=2000, description="Output width in pixels"),
    palette: str = Query("viridis", description="Colormap name"),
    vmin: Optional[float] = Query(None, description="Min value for color scale"),
    vmax: Optional[float] = Query(None, description="Max value for color scale"),
):
    """Render a variable as a color-mapped PNG overlay."""
    # Validate variable
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f"Unknown variable: {var}")
    
    # Parse bbox
    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(400, "bbox format: lat_min,lon_min,lat_max,lon_max")
    lat_min, lon_min, lat_max, lon_max = map(float, parts)
    
    # Validate palette
    valid_palettes = ["viridis", "plasma", "inferno", "magma", "coolwarm", 
                      "RdBu_r", "Blues", "Greens", "Reds", "YlOrRd"]
    if palette not in valid_palettes:
        raise HTTPException(400, f"Invalid palette. Valid: {valid_palettes}")
    
    # Load data
    client_class = 'mobile' if clientClass == 'mobile' else 'desktop'

    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used, keys=[var])
    
    if var not in d:
        raise HTTPException(404, f"Variable {var} not found in data")
    
    lat = d["lat"]
    lon = d["lon"]
    
    # Crop to bbox
    li, lo = _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max)
    if len(li) == 0 or len(lo) == 0:
        raise HTTPException(404, "No data in bbox")
    
    c_lat = lat[li]
    c_lon = lon[lo]
    data_arr = _slice_array(d[var], li, lo)
    
    # Compute actual bbox (cell edges)
    lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.02
    actual_lat_min = float(np.min(c_lat)) - lat_res / 2
    actual_lat_max = float(np.max(c_lat)) + lat_res / 2
    actual_lon_min = float(np.min(c_lon)) - lon_res / 2
    actual_lon_max = float(np.max(c_lon)) + lon_res / 2

    # Auto-range if vmin/vmax not provided
    if vmin is None or vmax is None:
        valid_data = data_arr[~np.isnan(data_arr)]
        if len(valid_data) == 0:
            raise HTTPException(404, "No valid data in bbox")
        if vmin is None:
            vmin = float(np.min(valid_data))
        if vmax is None:
            vmax = float(np.max(valid_data))

    # Reproject rows from regular latitude spacing to regular Web-Mercator spacing.
    # Leaflet image overlays are drawn in EPSG:3857 space; if we render the raster
    # in geographic row spacing, it appears to drift/stretch during zoom/pan.
    lat_src = np.asarray(c_lat, dtype=np.float64)
    data_src = np.asarray(data_arr, dtype=np.float64)

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

    # Keep row count; remap rows to equal spacing in Web-Mercator Y.
    target_rows = max(2, data_src.shape[0])
    y_tgt = np.linspace(y_min, y_max, target_rows)
    nearest_idx = np.searchsorted(y_src, y_tgt)
    nearest_idx = np.clip(nearest_idx, 0, len(y_src) - 1)
    left_idx = np.clip(nearest_idx - 1, 0, len(y_src) - 1)
    choose_left = np.abs(y_tgt - y_src[left_idx]) <= np.abs(y_tgt - y_src[nearest_idx])
    row_idx = np.where(choose_left, left_idx, nearest_idx)
    data_merc = data_src[row_idx, :]

    # Compute output dimensions in Web-Mercator aspect ratio
    x_min = math.radians(actual_lon_min)
    x_max = math.radians(actual_lon_max)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    aspect = x_span / y_span
    out_w = width
    out_h = max(1, int(out_w / aspect))

    # Create matplotlib figure with exact pixel dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(out_w/dpi, out_h/dpi), dpi=dpi)
    ax.set_position([0, 0, 1, 1])  # Fill entire figure
    ax.axis('off')

    # Render with colormap (origin='lower' keeps north at top in final PNG)
    cmap = cm.get_cmap(palette)
    cmap.set_bad(alpha=0)  # Transparent for NaN

    ax.imshow(
        data_merc,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        interpolation='nearest',
        aspect='auto'
    )
    
    # Save to bytes
    # IMPORTANT: avoid bbox_inches='tight' here, because it can crop/resize by
    # sub-pixel amounts and cause geographic misalignment (overlay appears to
    # drift relative to basemap during zoom/pan).
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format='png',
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Bbox": f"{actual_lat_min},{actual_lon_min},{actual_lat_max},{actual_lon_max}",
            "X-Run": run,
            "X-ValidTime": d["validTime"],
            "X-Model": model_used,
            "X-VMin": str(vmin),
            "X-VMax": str(vmax),
            "Access-Control-Expose-Headers": "X-Bbox, X-Run, X-ValidTime, X-Model, X-VMin, X-VMax",
        }
    )


@app.get("/api/point")
async def api_point(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    time: str = Query("latest", description="ISO time or 'latest'"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu"),
):
    """Return all variable values at a specific point for a given timestep."""
    # Load all data for this timestep
    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used)
    
    lat_arr = d["lat"]
    lon_arr = d["lon"]
    
    # Find nearest grid point
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
    
    # Extract values for all variables
    values = {}
    for var_name in VARIABLE_METADATA.keys():
        if var_name in d:
            arr = d[var_name]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                val = float(arr[lat_idx, lon_idx])
                values[var_name] = None if np.isnan(val) else round(val, 3)
    
    return {
        "lat": round(float(lat_arr[lat_idx]), 4),
        "lon": round(float(lon_arr[lon_idx]), 4),
        "validTime": d["validTime"],
        "run": run,
        "model": model_used,
        "values": values
    }


# ─── Tiled overlay endpoints (Web-Mercator aligned) ───

def _tile_bounds_3857(z: int, x: int, y: int):
    """Return Web-Mercator bounds (meters): (minx, miny, maxx, maxy)."""
    origin = 20037508.342789244
    world = origin * 2.0
    tile_size = world / (2 ** z)
    minx = -origin + x * tile_size
    maxx = minx + tile_size
    maxy = origin - y * tile_size
    miny = maxy - tile_size
    return minx, miny, maxx, maxy


def _merc_to_lonlat(mx, my):
    """Vectorized EPSG:3857 meters -> lon/lat degrees."""
    lon = (mx / 20037508.342789244) * 180.0
    lat = np.degrees(2.0 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2.0)
    return lon, lat


def _regular_grid_indices(vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Fast nearest-index lookup for regular 1D grids (asc or desc)."""
    if len(vals) < 2:
        return np.zeros_like(q, dtype=np.int32)
    step = float(vals[1] - vals[0])
    if step == 0:
        return np.zeros_like(q, dtype=np.int32)
    idx = np.rint((q - float(vals[0])) / step).astype(np.int32)
    return np.clip(idx, 0, len(vals) - 1)


@app.get('/api/overlay_range')
async def api_overlay_range(
    var: str = Query(..., description='Variable name'),
    bbox: str = Query('43.18,-3.94,58.08,20.34', description='lat_min,lon_min,lat_max,lon_max'),
    time: str = Query('latest', description='ISO time or latest'),
    model: Optional[str] = Query(None, description='icon_d2 or icon_eu'),
):
    """Compute vmin/vmax for current viewport (used to keep tile colors consistent)."""
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f'Unknown variable: {var}')

    parts = bbox.split(',')
    if len(parts) != 4:
        raise HTTPException(400, 'bbox format: lat_min,lon_min,lat_max,lon_max')
    lat_min, lon_min, lat_max, lon_max = map(float, parts)

    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used, keys=[var])
    lat = d['lat']
    lon = d['lon']

    li, lo = _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max)
    if len(li) == 0 or len(lo) == 0:
        raise HTTPException(404, 'No data in bbox')

    data_arr = _slice_array(d[var], li, lo)
    valid = data_arr[~np.isnan(data_arr)]
    if len(valid) == 0:
        raise HTTPException(404, 'No valid data in bbox')

    return {
        'vmin': float(np.min(valid)),
        'vmax': float(np.max(valid)),
        'run': run,
        'model': model_used,
        'validTime': d['validTime'],
    }


@app.get('/api/overlay_tile/{z}/{x}/{y}.png')
async def api_overlay_tile(
    z: int,
    x: int,
    y: int,
    var: str = Query(..., description='Variable name'),
    time: str = Query('latest', description='ISO time or latest'),
    model: Optional[str] = Query(None, description='icon_d2 or icon_eu'),
    palette: str = Query('viridis', description='Colormap name'),
    vmin: Optional[float] = Query(None, description='Fixed min value for color scale'),
    vmax: Optional[float] = Query(None, description='Fixed max value for color scale'),
    clientClass: str = Query('desktop', description='desktop or mobile'),
):
    """Render one 256x256 Web-Mercator tile for overlay variable."""
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f'Unknown variable: {var}')

    valid_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm',
                      'RdBu_r', 'Blues', 'Greens', 'Reds', 'YlOrRd']
    if palette not in valid_palettes:
        raise HTTPException(400, f'Invalid palette. Valid: {valid_palettes}')

    client_class = 'mobile' if clientClass == 'mobile' else 'desktop'

    run, step, model_used = resolve_time(time, model)
    d = load_data(run, step, model_used, keys=[var])
    arr = d[var]
    lat = d['lat']
    lon = d['lon']

    # Tile-level cache key (must include full rendering state)
    cache_key = f"{client_class}|{model_used}|{run}|{step}|{var}|{palette}|{vmin}|{vmax}|{z}|{x}|{y}"
    cached_png = _tile_cache_get(client_class, cache_key)
    if cached_png is not None:
        return Response(
            content=cached_png,
            media_type='image/png',
            headers={
                'Cache-Control': 'public, max-age=300',
                'X-Run': run,
                'X-ValidTime': d['validTime'],
                'X-Model': model_used,
                'X-Cache': 'HIT',
                'Access-Control-Expose-Headers': 'X-Run, X-ValidTime, X-Model, X-Cache',
            }
        )

    # Tile bounds in lon/lat
    minx, miny, maxx, maxy = _tile_bounds_3857(z, x, y)
    lon0, lat0 = _merc_to_lonlat(minx, miny)
    lon1, lat1 = _merc_to_lonlat(maxx, maxy)
    lon_min, lon_max = (min(lon0, lon1), max(lon0, lon1))
    lat_min, lat_max = (min(lat0, lat1), max(lat0, lat1))

    # Skip tiles outside data extent quickly
    data_lat_min, data_lat_max = float(np.min(lat)), float(np.max(lat))
    data_lon_min, data_lon_max = float(np.min(lon)), float(np.max(lon))
    if lat_max < data_lat_min or lat_min > data_lat_max or lon_max < data_lon_min or lon_min > data_lon_max:
        empty = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO(); empty.save(buf, format='PNG', optimize=True); buf.seek(0)
        return Response(content=buf.getvalue(), media_type='image/png', headers={'Cache-Control': 'public, max-age=300'})

    # Build pixel-center coordinates in Web-Mercator then convert to lon/lat
    xs = np.linspace(minx, maxx, 256, endpoint=False) + (maxx - minx) / 512.0
    ys = np.linspace(maxy, miny, 256, endpoint=False) - (maxy - miny) / 512.0  # north->south
    mx, my = np.meshgrid(xs, ys)
    qlon, qlat = _merc_to_lonlat(mx, my)

    # Mask outside data extent
    inside = (
        (qlat >= data_lat_min) & (qlat <= data_lat_max) &
        (qlon >= data_lon_min) & (qlon <= data_lon_max)
    )

    # Resolve scale if not provided
    if vmin is None or vmax is None:
        # derive from tile intersection for reasonable fallback
        li, lo = _bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max)
        if len(li) == 0 or len(lo) == 0:
            empty = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            buf = io.BytesIO(); empty.save(buf, format='PNG', optimize=True); buf.seek(0)
            return Response(content=buf.getvalue(), media_type='image/png', headers={'Cache-Control': 'public, max-age=300'})
        local = _slice_array(arr, li, lo)
        valid = local[~np.isnan(local)]
        if len(valid) == 0:
            empty = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
            buf = io.BytesIO(); empty.save(buf, format='PNG', optimize=True); buf.seek(0)
            return Response(content=buf.getvalue(), media_type='image/png', headers={'Cache-Control': 'public, max-age=300'})
        if vmin is None:
            vmin = float(np.min(valid))
        if vmax is None:
            vmax = float(np.max(valid))

    # Nearest-neighbor sample from source grid
    lat_idx = _regular_grid_indices(lat, qlat)
    lon_idx = _regular_grid_indices(lon, qlon)
    sampled = arr[lat_idx, lon_idx]

    # Apply mask + NaN handling
    valid_mask = inside & np.isfinite(sampled)

    # Colormap to RGBA
    norm = np.clip((sampled - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    cmap = cm.get_cmap(palette)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (H,W,4)

    # Transparent where invalid
    rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)

    img = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)

    png_bytes = buf.getvalue()
    _tile_cache_set(client_class, cache_key, png_bytes)

    return Response(
        content=png_bytes,
        media_type='image/png',
        headers={
            'Cache-Control': 'public, max-age=300',
            'X-Run': run,
            'X-ValidTime': d['validTime'],
            'X-Model': model_used,
            'X-Cache': 'MISS',
            'Access-Control-Expose-Headers': 'X-Run, X-ValidTime, X-Model, X-Cache',
        }
    )


@app.get("/api/timeseries")
async def api_timeseries(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    var: str = Query(..., description="Variable name"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu (if None, uses merged timeline)"),
):
    """Return values of one variable across all available timesteps at a point."""
    # Validate variable
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f"Unknown variable: {var}")
    
    # Get available runs
    runs = get_available_runs()
    if not runs:
        raise HTTPException(404, "No data available")
    
    # If model specified, filter to that model; otherwise use all runs
    if model:
        runs = [r for r in runs if r["model"] == model]
        if not runs:
            raise HTTPException(404, f"No data for model: {model}")
    
    # Collect timeseries data
    data_points = []
    
    for run_info in runs:
        run = run_info["run"]
        run_model = run_info["model"]
        
        for step_info in run_info["steps"]:
            step = step_info["step"]
            try:
                d = load_data(run, step, run_model, keys=[var])
                
                if var not in d:
                    continue
                
                lat_arr = d["lat"]
                lon_arr = d["lon"]
                
                # Find nearest grid point
                lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
                lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
                
                arr = d[var]
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    val = float(arr[lat_idx, lon_idx])
                    if not np.isnan(val):
                        data_points.append({
                            "validTime": d["validTime"],
                            "value": round(val, 3),
                            "run": run,
                            "step": step,
                            "model": run_model
                        })
            except Exception:
                continue
    
    # Sort by valid time
    data_points.sort(key=lambda x: x["validTime"])
    
    # Find the actual lat/lon used (from first successful load)
    actual_lat = lat
    actual_lon = lon
    if data_points and runs:
        try:
            first_run = runs[0]["run"]
            first_step = runs[0]["steps"][0]["step"]
            first_model = runs[0]["model"]
            d = load_data(first_run, first_step, first_model, keys=[var])
            lat_arr = d["lat"]
            lon_arr = d["lon"]
            lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
            lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
            actual_lat = round(float(lat_arr[lat_idx]), 4)
            actual_lon = round(float(lon_arr[lon_idx]), 4)
        except Exception:
            pass
    
    return {
        "lat": actual_lat,
        "lon": actual_lon,
        "var": var,
        "data": data_points,
        "count": len(data_points)
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    runs = get_available_runs()
    return {"status": "ok", "runs": len(runs), "cache": len(data_cache)}


@app.get("/api/cache_stats")
async def api_cache_stats():
    """Overlay tile cache stats for debugging/perf tuning."""
    _tile_cache_prune('desktop')
    _tile_cache_prune('mobile')
    return {
        "tileCache": {
            "desktopItems": len(tile_cache_desktop),
            "desktopMax": TILE_CACHE_MAX_ITEMS_DESKTOP,
            "mobileItems": len(tile_cache_mobile),
            "mobileMax": TILE_CACHE_MAX_ITEMS_MOBILE,
            "ttlSeconds": TILE_CACHE_TTL_SECONDS,
        },
        "metrics": cache_stats,
    }


# ─── Static Files (serve explorer frontend) ───
# Must be last - serves index.html for directory requests
if os.path.isdir(EXPLORER_DIR):
    app.mount("/", StaticFiles(directory=EXPLORER_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
