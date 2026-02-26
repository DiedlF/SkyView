#!/usr/bin/env python3
"""ICON-Explorer Backend — raw data visualization API for ICON weather model data."""

import os
import sys
import io
import math
import time
import atexit
import uuid
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, JSONResponse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

from data_provider import build_provider, LocalNpzProvider

app = FastAPI(title="ICON-Explorer API")

logger = logging.getLogger("explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "requestId": rid}, headers={"X-Request-Id": rid})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    logger.exception(f"Unhandled explorer error rid={rid}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "requestId": rid}, headers={"X-Request-Id": rid})


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
EXPLORER_DIR = SCRIPT_DIR
PID_FILE = os.path.join(SCRIPT_DIR, "logs", "explorer.pid")

# Shared helpers (keeps API alias + point normalization in one place)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "backend"))
from api_contract import resolve_layer_alias as _resolve_layer_alias
from time_contract import get_available_runs as tc_get_available_runs, get_merged_timeline as tc_get_merged_timeline, resolve_time as tc_resolve_time
from grid_utils import bbox_indices as _bbox_indices, slice_array as _slice_array
from response_headers import build_overlay_headers, build_tile_headers
from model_caps import get_models_payload

DATA_PROVIDER_MODE = os.environ.get("EXPLORER_DATA_PROVIDER", "dwd")
SOURCE_API_TOKEN = os.environ.get("EXPLORER_SOURCE_API_TOKEN", "").strip()


def _acquire_single_instance_or_exit(pid_file: str):
    """Simple PID-file guard to prevent accidental multi-process launches."""
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)

    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            if old_pid > 0:
                os.kill(old_pid, 0)
                raise SystemExit(f"Explorer backend already running with pid {old_pid} (pid file: {pid_file})")
        except ProcessLookupError:
            pass
        except ValueError:
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

# layer alias resolver imported from backend/api_contract.py

# ─── Data Loading Helpers ───

source_provider = LocalNpzProvider(
    DATA_DIR,
    tc_get_available_runs,
    tc_get_merged_timeline,
    tc_resolve_time,
)

try:
    provider = build_provider(
        DATA_PROVIDER_MODE,
        data_dir=DATA_DIR,
        get_runs_fn=tc_get_available_runs,
        get_merged_timeline_fn=tc_get_merged_timeline,
        resolve_time_fn=tc_resolve_time,
    )
    ACTIVE_PROVIDER_MODE = DATA_PROVIDER_MODE
except Exception as e:
    logger.warning(f"Failed to initialize provider '{DATA_PROVIDER_MODE}' ({e}), falling back to local_npz")
    provider = build_provider(
        "local_npz",
        data_dir=DATA_DIR,
        get_runs_fn=tc_get_available_runs,
        get_merged_timeline_fn=tc_get_merged_timeline,
        resolve_time_fn=tc_resolve_time,
    )
    ACTIVE_PROVIDER_MODE = "local_npz"

logger.info(f"Explorer data provider: {ACTIVE_PROVIDER_MODE}")


def _provider_call(fn_name: str, *args, **kwargs):
    fn = getattr(provider, fn_name)
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _require_source_auth(request: Request):
    if not SOURCE_API_TOKEN:
        return
    token = request.headers.get("X-Source-Token", "")
    if token != SOURCE_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized source access")


def load_data(run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    return _provider_call("load_data", run, step, model, keys=keys)


def get_available_runs():
    return _provider_call("get_available_runs")


def get_merged_timeline():
    return _provider_call("get_merged_timeline")


def resolve_time(time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
    return _provider_call("resolve_time", time_str, model)


def resolve_time_with_run(time_str: str, model: Optional[str] = None, run: Optional[str] = None) -> tuple[str, int, str]:
    """Resolve time with optional explicit run pinning."""
    if not run:
        return resolve_time(time_str, model)

    runs = get_available_runs()
    candidates = [r for r in runs if r.get("run") == run and (model is None or r.get("model") == model)]
    if not candidates:
        raise HTTPException(404, f"Run not found: {run}")

    info = candidates[0]
    steps = info.get("steps", [])
    if not steps:
        raise HTTPException(404, f"No steps for run: {run}")

    if time_str == "latest":
        step = max(int(s.get("step", 0)) for s in steps)
        return run, step, info.get("model")

    for s in steps:
        if s.get("validTime") == time_str:
            return run, int(s.get("step")), info.get("model")

    raise HTTPException(404, f"Time not found in run {run}: {time_str}")


def _build_model_capabilities(p):
    caps: Dict[str, Any] = {}
    runs = p.get_available_runs()
    for r in runs:
        model = r.get("model")
        if not model or model in caps:
            continue
        steps = r.get("steps") or []
        if not steps:
            continue
        try:
            step = int(steps[0].get("step"))
            d = p.load_data(r["run"], step, model, keys=None)  # load full sample for var discovery
        except Exception:
            continue

        if hasattr(p, "list_variables"):
            try:
                vars_available = sorted(getattr(p, "list_variables")(model, run=r.get("run")))
            except Exception:
                vars_available = sorted([k for k, v in d.items() if isinstance(v, np.ndarray) and k not in ("lat", "lon")])
        else:
            vars_available = sorted([k for k, v in d.items() if isinstance(v, np.ndarray) and k not in ("lat", "lon")])

        levels = sorted(
            {k.split("_")[1].replace("hpa", "") for k in vars_available if k.startswith("u_") and k.endswith("hpa")}
        )
        lat = d.get("lat")
        lon = d.get("lon")
        bbox = None
        if isinstance(lat, np.ndarray) and isinstance(lon, np.ndarray) and lat.size and lon.size:
            bbox = {
                "latMin": float(np.min(lat)),
                "latMax": float(np.max(lat)),
                "lonMin": float(np.min(lon)),
                "lonMax": float(np.max(lon)),
            }
        caps[model] = {
            "run": r.get("run"),
            "step": step,
            "variables": vars_available,
            "pressureLevels": levels,
            "bbox": bbox,
        }
    return {"models": caps}


def _provider_health():
    stats_payload = provider.get_stats() if hasattr(provider, "get_stats") else {"mode": ACTIVE_PROVIDER_MODE}
    mode = stats_payload.get("mode", ACTIVE_PROVIDER_MODE)
    if mode not in ("remote", "dwd"):
        return {"status": "ok", "mode": mode, "reason": "threshold checks apply to remote/dwd modes", "stats": stats_payload}

    stats = stats_payload.get("stats", {})
    fetches = int(stats.get("field_fetches", 0) or 0)
    errors = int(stats.get("http_errors", 0) or 0)
    misses = int(stats.get("field_cache_misses", 0) or 0)
    mem_hits = int(stats.get("field_mem_hits", 0) or 0)
    disk_hits = int(stats.get("field_disk_hits", 0) or 0)
    avg_ms = float(stats.get("fetch_avg_ms", 0.0) or 0.0)

    err_warn = float(os.environ.get("EXPLORER_PROVIDER_WARN_ERROR_RATE", "0.08"))
    err_crit = float(os.environ.get("EXPLORER_PROVIDER_CRIT_ERROR_RATE", "0.2"))
    lat_warn = float(os.environ.get("EXPLORER_PROVIDER_WARN_FETCH_MS", "2500"))
    lat_crit = float(os.environ.get("EXPLORER_PROVIDER_CRIT_FETCH_MS", "5000"))
    hit_warn = float(os.environ.get("EXPLORER_PROVIDER_WARN_CACHE_HIT_RATIO", "0.35"))

    err_rate = (errors / fetches) if fetches > 0 else 0.0
    hit_ratio = ((mem_hits + disk_hits) / (mem_hits + disk_hits + misses)) if (mem_hits + disk_hits + misses) > 0 else 1.0

    status = "ok"
    reasons = []
    if err_rate >= err_crit or avg_ms >= lat_crit:
        status = "critical"
    elif err_rate >= err_warn or avg_ms >= lat_warn or hit_ratio < hit_warn:
        status = "warning"

    if err_rate >= err_warn:
        reasons.append(f"high error rate ({err_rate:.2%})")
    if avg_ms >= lat_warn:
        reasons.append(f"high avg fetch latency ({avg_ms:.0f} ms)")
    if hit_ratio < hit_warn:
        reasons.append(f"low cache hit ratio ({hit_ratio:.2%})")

    return {
        "status": status,
        "mode": mode,
        "reasons": reasons,
        "metrics": {
            "errorRate": round(err_rate, 4),
            "cacheHitRatio": round(hit_ratio, 4),
            "avgFetchMs": round(avg_ms, 2),
        },
        "thresholds": {
            "warnErrorRate": err_warn,
            "critErrorRate": err_crit,
            "warnFetchMs": lat_warn,
            "critFetchMs": lat_crit,
            "warnCacheHitRatio": hit_warn,
        },
        "stats": stats_payload,
    }


# ─── API Endpoints ───

@app.get("/api/variables")
async def api_variables(
    include_unavailable: bool = Query(False, description="Include variables not present in latest timestep"),
    model: Optional[str] = Query(None, description="Optional model filter"),
    run: Optional[str] = Query(None, description="Optional run filter"),
):
    """List variables with metadata, availability, and sample min/max."""
    variables = []

    # If provider supports directory-based variable listing (DWD), use it directly.
    listed_vars = None
    if model and hasattr(provider, "list_variables"):
        try:
            listed_vars = set(getattr(provider, "list_variables")(model, run=run))
        except Exception:
            listed_vars = set()

    # Try to load a sample timestep to get min/max values + live availability.
    runs = get_available_runs()
    if model:
        runs = [r for r in runs if r.get("model") == model]
    if run:
        runs = [r for r in runs if r.get("run") == run]

    sample_data = None
    if runs:
        try:
            run0 = runs[0]["run"]
            step = runs[0]["steps"][0]["step"]
            model0 = runs[0]["model"]
            sample_data = load_data(run0, step, model0)
        except Exception:
            pass

    # Build variable universe:
    # - preferred: listed vars from provider
    # - fallback: known metadata keys
    if listed_vars is not None:
        universe = sorted(listed_vars)
    else:
        universe = list(VARIABLE_METADATA.keys())

    for var_name in universe:
        meta = VARIABLE_METADATA.get(var_name, {"group": "raw", "unit": "", "desc": "DWD variable"})
        var_available = bool(sample_data is not None and var_name in sample_data)
        if not include_unavailable and not var_available and listed_vars is None:
            continue

        var_info = {
            "name": var_name,
            "group": meta["group"],
            "unit": meta["unit"],
            "desc": meta["desc"],
            "available": (var_name in listed_vars) if listed_vars is not None else var_available,
        }

        # Add min/max from sample if available
        if sample_data is not None and var_name in sample_data:
            arr = sample_data[var_name]
            if isinstance(arr, np.ndarray) and arr.size > 0:
                valid_data = arr[~np.isnan(arr)]
                if len(valid_data) > 0:
                    var_info["min"] = float(np.min(valid_data))
                    var_info["max"] = float(np.max(valid_data))

        variables.append(var_info)

    return {"variables": variables}


@app.get("/api/models")
async def api_models():
    """Compatibility endpoint with Skyview model metadata."""
    return get_models_payload()


@app.get("/api/capabilities")
async def api_capabilities():
    """Available variables/levels per model for current provider data source."""
    return _build_model_capabilities(provider)


@app.get("/api/timesteps")
async def api_timesteps(
    model: Optional[str] = Query(None, description="Filter runs by model (icon_d2|icon_eu)"),
    run: Optional[str] = Query(None, description="Select specific run id"),
):
    """Return available runs and timesteps, optionally filtered by model/run."""
    runs = get_available_runs()

    if model:
        runs = [r for r in runs if r.get("model") == model]

    if run:
        selected = [r for r in runs if r.get("run") == run]
        if not selected:
            raise HTTPException(404, f"Run not found: {run}")
        return {"runs": selected, "merged": None}

    # If no model filter: keep merged behavior for convenience.
    if model is None:
        merged = get_merged_timeline()
        return {"runs": runs, "merged": merged}

    return {"runs": runs, "merged": None}


# --- Source endpoints for RemoteProvider (Phase 2) ---

@app.get("/api/source/runs")
async def api_source_runs(request: Request):
    _require_source_auth(request)
    return source_provider.get_available_runs()


@app.get("/api/source/timeline")
async def api_source_timeline(request: Request):
    _require_source_auth(request)
    return source_provider.get_merged_timeline()


@app.get("/api/source/resolve_time")
async def api_source_resolve_time(
    request: Request,
    time: str = Query("latest", description="ISO time or 'latest'"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu"),
):
    _require_source_auth(request)
    run, step, model_used = source_provider.resolve_time(time, model)
    return {"run": run, "step": step, "model": model_used}


@app.get("/api/source/capabilities")
async def api_source_capabilities(request: Request):
    _require_source_auth(request)
    return _build_model_capabilities(source_provider)


@app.get("/api/source/field.npz")
async def api_source_field_npz(
    request: Request,
    run: str = Query(...),
    step: int = Query(..., ge=0),
    model: str = Query(...),
    keys: Optional[str] = Query(None, description="Comma-separated variable keys"),
):
    _require_source_auth(request)
    key_list = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    d = source_provider.load_data(run, step, model, keys=key_list)

    out = io.BytesIO()
    arrays: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
    # Include validTime as scalar array for metadata consistency
    arrays["validTime"] = np.array(str(d.get("validTime", "")))
    np.savez_compressed(out, **arrays)
    out.seek(0)

    return Response(
        content=out.getvalue(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=60"},
    )


@app.get("/api/source/contract")
async def api_source_contract():
    """Machine-readable source endpoint contract for remote provider compatibility."""
    return {
        "version": "v1",
        "auth": {
            "header": "X-Source-Token",
            "requiredWhenConfigured": True,
        },
        "endpoints": {
            "runs": {
                "method": "GET",
                "path": "/api/source/runs",
                "response": "list[runs] as in time_contract.get_available_runs",
            },
            "timeline": {
                "method": "GET",
                "path": "/api/source/timeline",
                "response": "merged timeline object as in time_contract.get_merged_timeline",
            },
            "resolve_time": {
                "method": "GET",
                "path": "/api/source/resolve_time",
                "query": {"time": "str (latest|ISO)", "model": "optional str"},
                "response": {"run": "str", "step": "int", "model": "str"},
            },
            "field_npz": {
                "method": "GET",
                "path": "/api/source/field.npz",
                "query": {
                    "run": "str YYYYMMDDHH",
                    "step": "int",
                    "model": "str (icon_d2|icon_eu)",
                    "keys": "optional comma-separated variable keys",
                },
                "response": "npz bytes with at least lat, lon, validTime and requested variables when present",
            },
        },
    }


@app.get("/api/overlay")
async def api_overlay(
    var: Optional[str] = Query(None, description="Variable name"),
    layer: Optional[str] = Query(None, description="Skyview-compatible layer alias"),
    bbox: str = Query("43.18,-3.94,58.08,20.34", description="lat_min,lon_min,lat_max,lon_max"),
    time: str = Query("latest", description="ISO time or 'latest'"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu"),
    run: Optional[str] = Query(None, description="Optional explicit run id YYYYMMDDHH"),
    width: int = Query(800, ge=100, le=2000, description="Output width in pixels"),
    palette: str = Query("viridis", description="Colormap name"),
    vmin: Optional[float] = Query(None, description="Min value for color scale"),
    vmax: Optional[float] = Query(None, description="Max value for color scale"),
):
    """Render a variable as a color-mapped PNG overlay."""
    # Skyview-compat alias support
    if var is None and layer is not None:
        var = _resolve_layer_alias(layer)

    if var is None:
        raise HTTPException(400, "Missing required query param: var (or layer alias)")

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
    run, step, model_used = resolve_time_with_run(time, model, run)
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
        headers=build_overlay_headers(
            run=run,
            valid_time=d["validTime"],
            model=model_used,
            bbox=f"{actual_lat_min},{actual_lon_min},{actual_lat_max},{actual_lon_max}",
            extra={"X-VMin": str(vmin), "X-VMax": str(vmax)},
        ),
    )





@app.get("/api/point")
async def api_point(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    time: str = Query("latest", description="ISO time or 'latest'"),
    model: Optional[str] = Query(None, description="icon_d2 or icon_eu"),
    run: Optional[str] = Query(None, description="Optional explicit run id YYYYMMDDHH"),
    var: Optional[str] = Query(None, description="Rendered variable key"),
    layer: Optional[str] = Query(None, description="Optional Skyview layer alias"),
):
    """Return only one point metric: the currently rendered overlay value."""
    if var is None and layer is not None:
        var = _resolve_layer_alias(layer)
    if not var:
        raise HTTPException(400, "Missing required query param: var (or layer alias)")

    run, step, model_used = resolve_time_with_run(time, model, run)
    d = load_data(run, step, model_used, keys=[var])

    lat_arr = d["lat"]
    lon_arr = d["lon"]
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))

    value = None
    if var in d:
        arr = d[var]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            val = float(arr[lat_idx, lon_idx])
            value = None if np.isnan(val) else round(val, 3)

    return {
        "lat": round(float(lat_arr[lat_idx]), 4),
        "lon": round(float(lon_arr[lon_idx]), 4),
        "validTime": d["validTime"],
        "run": run,
        "model": model_used,
        "var": var,
        "value": value,
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
    run: Optional[str] = Query(None, description='Optional explicit run id YYYYMMDDHH'),
):
    """Compute vmin/vmax for current viewport (used to keep tile colors consistent)."""
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f'Unknown variable: {var}')

    parts = bbox.split(',')
    if len(parts) != 4:
        raise HTTPException(400, 'bbox format: lat_min,lon_min,lat_max,lon_max')
    lat_min, lon_min, lat_max, lon_max = map(float, parts)

    run, step, model_used = resolve_time_with_run(time, model, run)
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
    var: Optional[str] = Query(None, description='Variable name'),
    layer: Optional[str] = Query(None, description='Skyview-compatible layer alias'),
    time: str = Query('latest', description='ISO time or latest'),
    model: Optional[str] = Query(None, description='icon_d2 or icon_eu'),
    run: Optional[str] = Query(None, description='Optional explicit run id YYYYMMDDHH'),
    palette: str = Query('viridis', description='Colormap name'),
    vmin: Optional[float] = Query(None, description='Fixed min value for color scale'),
    vmax: Optional[float] = Query(None, description='Fixed max value for color scale'),
    clientClass: str = Query('desktop', description='desktop or mobile'),
):
    """Render one 256x256 Web-Mercator tile for overlay variable."""
    if var is None and layer is not None:
        var = _resolve_layer_alias(layer)
    if var is None:
        raise HTTPException(400, 'Missing required query param: var (or layer alias)')
    if var not in VARIABLE_METADATA:
        raise HTTPException(400, f'Unknown variable: {var}')

    valid_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm',
                      'RdBu_r', 'Blues', 'Greens', 'Reds', 'YlOrRd']
    if palette not in valid_palettes:
        raise HTTPException(400, f'Invalid palette. Valid: {valid_palettes}')

    client_class = 'mobile' if clientClass == 'mobile' else 'desktop'

    run, step, model_used = resolve_time_with_run(time, model, run)
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
            headers=build_tile_headers(
                run=run,
                valid_time=d['validTime'],
                model=model_used,
                cache='HIT',
            ),
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
        headers=build_tile_headers(
            run=run,
            valid_time=d['validTime'],
            model=model_used,
            cache='MISS',
        ),
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
    return {"status": "ok", "runs": len(runs), "provider": ACTIVE_PROVIDER_MODE}


@app.get("/api/provider_stats")
async def api_provider_stats():
    """Provider-level metrics (remote cache/fetch stats, local cache size)."""
    if hasattr(provider, "get_stats"):
        return provider.get_stats()
    return {"mode": ACTIVE_PROVIDER_MODE}


@app.get("/api/provider_health")
async def api_provider_health():
    """Threshold-based provider health state for canary/cutover operations."""
    return _provider_health()


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
    _acquire_single_instance_or_exit(PID_FILE)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
