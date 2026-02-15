#!/usr/bin/env python3
"""ICON-Explorer Backend — raw data visualization API for ICON weather model data."""

import os
import sys
import io
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm

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

# ─── Variable Metadata ───

VARIABLE_METADATA = {
    # Aviation group
    "ww": {"group": "aviation", "unit": "code", "desc": "Significant weather code"},
    "cape_ml": {"group": "aviation", "unit": "J/kg", "desc": "Mixed-layer CAPE"},
    "cin_ml": {"group": "aviation", "unit": "J/kg", "desc": "Mixed-layer CIN"},
    "htop_dc": {"group": "aviation", "unit": "m", "desc": "Convective cloud top height"},
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
    lat_res = float(lat[1] - lat[0]) if len(lat) > 1 else 0.02
    lon_res = float(lon[1] - lon[0]) if len(lon) > 1 else 0.02
    actual_lat_min = float(c_lat[0]) - lat_res / 2
    actual_lat_max = float(c_lat[-1]) + lat_res / 2
    actual_lon_min = float(c_lon[0]) - lon_res / 2
    actual_lon_max = float(c_lon[-1]) + lon_res / 2
    
    # Auto-range if vmin/vmax not provided
    if vmin is None or vmax is None:
        valid_data = data_arr[~np.isnan(data_arr)]
        if len(valid_data) == 0:
            raise HTTPException(404, "No valid data in bbox")
        if vmin is None:
            vmin = float(np.min(valid_data))
        if vmax is None:
            vmax = float(np.max(valid_data))
    
    # Compute output dimensions preserving aspect ratio
    aspect = (actual_lon_max - actual_lon_min) / (actual_lat_max - actual_lat_min)
    out_w = width
    out_h = max(1, int(out_w / aspect))
    
    # Create matplotlib figure with exact pixel dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(out_w/dpi, out_h/dpi), dpi=dpi)
    ax.set_position([0, 0, 1, 1])  # Fill entire figure
    ax.axis('off')
    
    # Render with colormap (origin='lower' for correct orientation)
    cmap = cm.get_cmap(palette)
    cmap.set_bad(alpha=0)  # Transparent for NaN
    
    im = ax.imshow(
        data_arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[actual_lon_min, actual_lon_max, actual_lat_min, actual_lat_max],
        origin='lower',
        interpolation='nearest',
        aspect='auto'
    )
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                pad_inches=0, transparent=True)
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


# ─── Static Files (serve explorer frontend) ───
# Must be last - serves index.html for directory requests
if os.path.isdir(EXPLORER_DIR):
    app.mount("/", StaticFiles(directory=EXPLORER_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
