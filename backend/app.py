#!/usr/bin/env python3
"""Skyview FastAPI backend — serves ICON-D2 weather symbols for Leaflet frontend."""

import os
import sys
import math
import asyncio

# ─── Auto-load .marker_auth_secret.env (and any other *.env files in backend dir) ───
# This ensures secrets are available regardless of how the server is launched
# (direct python3 call, nohup, systemd, etc.) without requiring manual `source`.
def _load_env_file(path: str) -> None:
    """Parse shell-style `export KEY="VALUE"` or `KEY=VALUE` env files into os.environ."""
    import re
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip leading 'export ' if present
                line = re.sub(r"^export\s+", "", line)
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                # Strip surrounding quotes (single or double)
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass
    except Exception as _e:
        print(f"Warning: could not load env file {path}: {_e}", file=sys.stderr)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
for _env_file in [
    os.path.join(_BACKEND_DIR, ".marker_auth_secret.env"),
    os.path.join(_BACKEND_DIR, ".env"),
]:
    _load_env_file(_env_file)
import time
import atexit
import uuid
import json
import glob
import subprocess
import requests
from email.utils import parsedate_to_datetime
from time import perf_counter
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
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
from services.symbol_ops import seed_symbols_cache_from_disk as _seed_symbols_cache_from_disk_svc
from point_data import build_overlay_values, POINT_KEYS, POINT_KEYS_MINIMAL
from classify import classify_point as classify_point_core
from time_contract import get_available_runs as tc_get_available_runs, get_merged_timeline as tc_get_merged_timeline, resolve_time as tc_resolve_time
from grid_aggregation import build_grid_context, choose_cell_groups, scatter_cell_stats
from status_ops import build_status_payload, build_perf_payload
from feedback_ops import make_feedback_entry, append_feedback, read_feedback_list, update_feedback_status
from model_caps import get_models_payload
from response_headers import build_overlay_headers, build_tile_headers
from usage_stats import record_visit, get_usage_stats, get_marker_stats
import marker_auth as _marker_auth
from services.model_select import resolve_eu_time_strict as svc_resolve_eu_time_strict, load_eu_data_strict as svc_load_eu_data_strict
from services.data_loader import load_step_data
from services.app_state import AppState
from routers.core import build_core_router
from routers.point import build_point_router
from routers.domain import build_domain_router
from routers.weather import build_weather_router
from routers.overlay import build_overlay_router
from routers.ops import build_ops_router
from routers.admin import build_admin_router
from constants import (
    CELL_SIZES_BY_ZOOM,
    CAPE_CONV_THRESHOLD,
    CEILING_VALID_MAX_METERS,
    ICON_EU_STEP_3H_START,
    EU_STRICT_MAX_DELTA_HOURS_DEFAULT,
    DATA_CACHE_MAX_ITEMS,
    LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM,
    LOW_ZOOM_GLOBAL_BBOX,
    EMAGRAM_D2_LEVELS_HPA,
    G0,
)
from cache_state import (
    TILE_CACHE_MAX_ITEMS_DESKTOP, TILE_CACHE_MAX_ITEMS_MOBILE, TILE_CACHE_TTL_SECONDS,
    tile_cache_desktop, tile_cache_mobile, cache_stats, perf_recent, perf_totals, computed_field_cache,
    overlay_phase_recent, overlay_phase_totals, overlay_phase_record,
    perf_record, computed_cache_get_or_compute, computed_metrics_payload,
    tile_cache_prune, tile_cache_get, tile_cache_set,
    symbols_cache_get, symbols_cache_set, symbols_cache_stats_payload,
    rotate_caches_for_context, cache_context_stats_payload,
)

logger = setup_logging(__name__, level="INFO")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FRONTEND_DIR = os.path.join(SCRIPT_DIR, "..", "frontend")
PID_FILE = os.path.join(SCRIPT_DIR, "logs", "skyview.pid")
# Low-zoom symbols request-rate metrics (runtime counters — kept in app.py
# because api_status and api_perf_stats reference them directly).
low_zoom_symbols_cache_metrics = {
    "hits": 0,
    "misses": 0,
    "diskHits": 0,
    "diskMisses": 0,
}


def _model_dir_name(model_used: str) -> str:
    return "icon-d2" if model_used == "icon_d2" else ("icon-eu" if model_used == "icon_eu" else model_used)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook (startup/shutdown)."""
    logger.info("Skyview API server starting")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")

    _auth_check = _marker_auth.startup_check_with_level()
    if _auth_check:
        _auth_severity, _auth_msg = _auth_check
        _banner = "\n" + "=" * 72 + f"\n  SKYVIEW SECURITY: {_auth_msg}\n" + "=" * 72
        print(_banner, file=sys.stderr, flush=True)
        if _auth_severity == "error":
            logger.error("Marker auth: %s", _auth_msg)
        else:
            logger.warning("Marker auth: %s", _auth_msg)

    runs = get_available_runs()
    logger.info(f"Found {len(runs)} available model runs")
    if runs:
        latest = runs[0]
        logger.info(f"Latest run: {latest['model']} {latest['run']} ({len(latest['steps'])} timesteps)")
    seeded = _seed_symbols_cache_from_disk_svc(DATA_DIR, symbols_cache_set, max_runs_per_model=1)
    if seeded:
        logger.info("Seeded low-zoom symbols cache entries from disk: %s", seeded)

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

    try:
        _flush_fallback_stats(force=True)
    except Exception:
        pass


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

# Admin overview telemetry (in-process rolling windows)
REQUEST_METRICS_WINDOW = int(os.environ.get("SKYVIEW_REQUEST_METRICS_WINDOW", "400"))
request_metrics: Dict[str, Dict[str, Any]] = {}
overlay_layer_metrics: Dict[str, Dict[str, Any]] = {}
feature_usage_counters: Dict[str, int] = {
    "page": 0,
    "overlay_tile": 0,
    "symbols": 0,
    "point": 0,
    "meteogram": 0,
    "emagram": 0,
    "other_api": 0,
}
capacity_samples = deque(maxlen=288)  # ~24h at 5min cadence
_last_capacity_sample_mono = 0.0
_cpu_sample_prev: Optional[tuple[int, int]] = None

# Consolidated runtime state (PR3)
app_state = AppState(
    fallback_stats={
        "euResolveAttempts": 0,
        "euResolveSuccess": 0,
        "strictTimeDenied": 0,
        "overlayFallback": 0,
        "overlayTileFallback": 0,
        "symbolsBlended": 0,
        "windBlended": 0,
        "pointFallback": 0,
    }
)

_FALLBACK_STATS_PATH = os.path.join(DATA_DIR, "fallback_stats.json")
_fallback_stats_lock = threading.Lock()
_last_fallback_flush_mono = 0.0
_FALLBACK_FLUSH_INTERVAL_SECONDS = 30.0


def _load_persisted_fallback_stats() -> None:
    try:
        if not os.path.exists(_FALLBACK_STATS_PATH):
            return
        with open(_FALLBACK_STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for k in app_state.fallback_stats.keys():
            v = data.get(k)
            if isinstance(v, int):
                app_state.fallback_stats[k] = v
    except Exception:
        pass


def _flush_fallback_stats(force: bool = False) -> None:
    global _last_fallback_flush_mono
    now = time.monotonic()
    if (not force) and (now - _last_fallback_flush_mono) < _FALLBACK_FLUSH_INTERVAL_SECONDS:
        return
    with _fallback_stats_lock:
        os.makedirs(os.path.dirname(_FALLBACK_STATS_PATH), exist_ok=True)
        tmp = _FALLBACK_STATS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(app_state.fallback_stats, f, ensure_ascii=False)
        os.replace(tmp, _FALLBACK_STATS_PATH)
        _last_fallback_flush_mono = now


_load_persisted_fallback_stats()


def _set_fallback_current(endpoint: str, decision: str, source_model: Optional[str] = None, detail: Optional[dict] = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    app_state.fallback_current["updatedAt"] = now_iso
    app_state.fallback_current["endpoints"][endpoint] = {
        "updatedAt": now_iso,
        "decision": decision,
        "sourceModel": source_model,
        "detail": detail or {},
    }


def _classify_feature(path: str) -> str:
    if path in ("/", "/index.html"):
        return "page"
    if path == "/api/overlay_tile":
        return "overlay_tile"
    if path == "/api/symbols":
        return "symbols"
    if path == "/api/point":
        return "point"
    if path == "/api/meteogram_point":
        return "meteogram"
    if path == "/api/emagram_point":
        return "emagram"
    if path.startswith("/api/"):
        return "other_api"
    return "other_api"


def _record_request_metric(path: str, status_code: int, duration_ms: float) -> None:
    m = request_metrics.setdefault(path, {
        "count": 0,
        "errors4xx": 0,
        "errors5xx": 0,
        "slowOver1s": 0,
        "slowOver2s": 0,
        "latencyMs": deque(maxlen=REQUEST_METRICS_WINDOW),
        "lastSeen": None,
    })
    m["count"] += 1
    if 400 <= status_code < 500:
        m["errors4xx"] += 1
    elif status_code >= 500:
        m["errors5xx"] += 1
    if duration_ms > 1000:
        m["slowOver1s"] += 1
    if duration_ms > 2000:
        m["slowOver2s"] += 1
    m["latencyMs"].append(float(duration_ms))
    m["lastSeen"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_server_health() -> Dict[str, Any]:
    global _cpu_sample_prev
    cpu_percent = None
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline().strip()
        parts = [int(x) for x in line.split()[1:]]
        total = sum(parts)
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        if _cpu_sample_prev is not None:
            p_total, p_idle = _cpu_sample_prev
            dt, di = total - p_total, idle - p_idle
            if dt > 0:
                cpu_percent = max(0.0, min(100.0, (1.0 - (di / dt)) * 100.0))
        _cpu_sample_prev = (total, idle)
    except Exception:
        pass

    mem_total = mem_available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            info = {}
            for ln in f:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    info[k.strip()] = int(v.strip().split()[0]) * 1024
        mem_total = info.get("MemTotal")
        mem_available = info.get("MemAvailable")
    except Exception:
        pass

    mem_used = (mem_total - mem_available) if (mem_total is not None and mem_available is not None) else None
    mem_pct = ((mem_used / mem_total) * 100.0) if (mem_used is not None and mem_total) else None

    load1 = load5 = load15 = None
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        pass

    uptime_s = None
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            uptime_s = float(f.readline().split()[0])
    except Exception:
        pass

    return {
        "cpuPercent": round(cpu_percent, 1) if cpu_percent is not None else None,
        "memoryUsedBytes": int(mem_used) if mem_used is not None else None,
        "memoryTotalBytes": int(mem_total) if mem_total is not None else None,
        "memoryPercent": round(mem_pct, 1) if mem_pct is not None else None,
        "loadAvg1m": round(load1, 2) if load1 is not None else None,
        "loadAvg5m": round(load5, 2) if load5 is not None else None,
        "loadAvg15m": round(load15, 2) if load15 is not None else None,
        "uptimeSeconds": int(uptime_s) if uptime_s is not None else None,
    }


def _sample_capacity(total_bytes: int, est_tile_bytes: int) -> None:
    global _last_capacity_sample_mono
    now_m = time.monotonic()
    if (now_m - _last_capacity_sample_mono) < 300 and capacity_samples:
        return
    _last_capacity_sample_mono = now_m
    capacity_samples.append({
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "totalBytes": int(total_bytes),
        "estTileBytes": int(est_tile_bytes),
    })


def _record_overlay_layer_metric(layer: str, cache_hit: bool, total_ms: float) -> None:
    row = overlay_layer_metrics.setdefault(layer, {
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "slowOver1s": 0,
        "slowOver2s": 0,
        "latencyMs": deque(maxlen=REQUEST_METRICS_WINDOW),
    })
    row["requests"] += 1
    if cache_hit:
        row["hits"] += 1
    else:
        row["misses"] += 1
    if total_ms > 1000:
        row["slowOver1s"] += 1
    if total_ms > 2000:
        row["slowOver2s"] += 1
    row["latencyMs"].append(float(total_ms))

EU_STRICT_MAX_DELTA_HOURS = EU_STRICT_MAX_DELTA_HOURS_DEFAULT
EU_STRICT_MAX_DELTA_HOURS_3H = float(os.environ.get("SKYVIEW_EU_STRICT_MAX_DELTA_HOURS_3H", "1.6"))

# Optional proactive overlay warmup to smooth first-tile latency on context switches.
OVERLAY_WARMUP_ENABLED = os.environ.get("SKYVIEW_OVERLAY_WARMUP", "1").strip().lower() not in ("0", "false", "no", "off")
OVERLAY_WARMUP_MIN_INTERVAL_SECONDS = float(os.environ.get("SKYVIEW_OVERLAY_WARMUP_MIN_INTERVAL", "2.0"))
_warmup_lock = threading.Lock()
_last_overlay_warmup: Dict[str, float] = {"key": "", "ts": 0.0}


# Small process-local cache to avoid repeated strict EU time resolution work
# for identical (time_str, max_delta_hours) pairs during bursty requests.
# value: (result, monotonic_ts)
_EU_STRICT_CACHE_MAX = 64
_EU_STRICT_CACHE_TTL_SECONDS = float(os.environ.get("SKYVIEW_EU_STRICT_CACHE_TTL_SECONDS", "300"))

# Circuit-breaker/backoff for repeated EU missing-on-disk situations
_EU_MISSING_BACKOFF_SECONDS = float(os.environ.get("SKYVIEW_EU_MISSING_BACKOFF_SECONDS", "60"))

# /api/location_search hardening: process-local IP rate limiting + response cache
LOCATION_SEARCH_WINDOW_SECONDS = float(os.environ.get("SKYVIEW_LOCATION_SEARCH_WINDOW_SECONDS", "60"))
LOCATION_SEARCH_MAX_REQUESTS_PER_WINDOW = int(os.environ.get("SKYVIEW_LOCATION_SEARCH_MAX_PER_WINDOW", "30"))
LOCATION_SEARCH_CACHE_TTL_SECONDS = float(os.environ.get("SKYVIEW_LOCATION_SEARCH_CACHE_TTL_SECONDS", "300"))
_LOCATION_SEARCH_CACHE_MAX_ITEMS = 512


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

    _record_request_metric(request.url.path, response.status_code, duration_ms)
    feat = _classify_feature(request.url.path)
    feature_usage_counters[feat] = feature_usage_counters.get(feat, 0) + 1

    # Skip logging routine polling endpoints to reduce noise
    skip_paths = ['/api/timesteps', '/api/models']
    if request.url.path not in skip_paths or response.status_code >= 400:
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms - rid={request_id}"
        )

    # Usage tracking: count real page loads only (avoid inflating with API/static requests).
    if (
        request.method == "GET"
        and response.status_code < 400
        and request.url.path in ("/", "/index.html")
    ):
        try:
            client_ip = request.client.host if request.client else "unknown"
            ua = request.headers.get("user-agent", "")
            al = request.headers.get("accept-language", "")
            record_visit(
                USAGE_STATS_FILE,
                ip=client_ip,
                user_agent=ua,
                accept_lang=al,
                salt=USAGE_HASH_SALT,
                path=request.url.path,
            )
        except Exception as e:
            logger.debug(f"Usage tracking skipped: {e}")

    try:
        _flush_fallback_stats()
    except Exception:
        pass
    return response


data_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
meteogram_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
METEOGRAM_CACHE_MAX_ITEMS = int(os.environ.get("SKYVIEW_METEOGRAM_CACHE_MAX_ITEMS", "16"))

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
    """Load .npz data for a given run/step/model (service-backed)."""
    return load_step_data(
        data_dir=DATA_DIR,
        model=model,
        run=run,
        step=step,
        cache=data_cache,
        cache_max_items=DATA_CACHE_MAX_ITEMS,
        keys=keys,
        logger=logger,
    )


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


# Phase-2 modularization: core health/model/timeline endpoints moved to router
app.include_router(build_core_router(
    get_available_runs=get_available_runs,
    get_merged_timeline=get_merged_timeline,
    get_models_payload=get_models_payload,
    data_cache=data_cache,
))
app.include_router(build_domain_router(
    get_merged_timeline=get_merged_timeline,
    resolve_time_with_cache_context=resolve_time_with_cache_context,
    load_data=load_data,
    _freshness_minutes_from_run=_freshness_minutes_from_run,
    _merge_axis_aligned_segments=_merge_axis_aligned_segments,
    DATA_DIR=DATA_DIR,
))

# ─── Feedback endpoint ───

FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
MARKERS_FILE = os.path.join(DATA_DIR, "markers.json")
USAGE_STATS_FILE = os.path.join(DATA_DIR, "usage_stats.json")
USAGE_HASH_SALT = os.environ.get("SKYVIEW_USAGE_SALT", "skyview-usage-default-salt")
OPENAIP_SEED_FILE = os.path.join(SCRIPT_DIR, "openaip_seed.json")
MARKER_AUTH_SECRET = os.environ.get("SKYVIEW_MARKER_AUTH_SECRET", "")
MARKER_AUTH_CONFIGURED = _marker_auth.is_configured()
MARKER_TOKEN_TTL_SECONDS = _marker_auth.TOKEN_TTL_SECONDS
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


def _make_marker_token(client_id: str) -> Dict[str, Any]:
    return _marker_auth.make_token(client_id)


def _verify_marker_token(client_id: str, token: str) -> bool:
    return _marker_auth.verify_token(client_id, token)

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
    times = app_state.feedback_rates.setdefault(ip, deque())
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


async def api_feedback_list(
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    """List feedback entries with optional status/text filtering (for admin use)."""
    feedback = read_feedback_list(FEEDBACK_FILE)

    if status:
        status_l = status.strip().lower()
        feedback = [f for f in feedback if str(f.get("status", "new")).lower() == status_l]

    if q:
        ql = q.strip().lower()
        if ql:
            def _hit(it: dict) -> bool:
                msg = str(it.get("message", "")).lower()
                typ = str(it.get("type", "")).lower()
                ctx = json.dumps(it.get("context", {}), ensure_ascii=False).lower()
                return (ql in msg) or (ql in typ) or (ql in ctx)
            feedback = [f for f in feedback if _hit(f)]

    feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    total = len(feedback)
    feedback = feedback[:limit]
    return {"feedback": feedback, "count": len(feedback), "total": total}


async def api_feedback_update(item_id: int, request: Request):
    """Update feedback workflow status (new|triaged|resolved)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    new_status = str(body.get("status", "")).strip().lower()
    if new_status not in {"new", "triaged", "resolved"}:
        raise HTTPException(400, "status must be one of: new, triaged, resolved")

    updated = update_feedback_status(FEEDBACK_FILE, item_id, new_status)
    if updated is None:
        raise HTTPException(404, "feedback item not found")
    return {"status": "ok", "feedback": updated}


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


async def api_marker_profile(clientId: str = Query(..., min_length=3, max_length=128)):
    marker = _marker_for_client(clientId)
    return {
        "marker": marker,
        "markerAuthConfigured": MARKER_AUTH_CONFIGURED,
        "markerEditable": MARKER_AUTH_CONFIGURED,
        "fallbackToDefault": (not MARKER_AUTH_CONFIGURED),
    }


async def api_marker_auth(clientId: str = Query(..., min_length=3, max_length=128)):
    if not MARKER_AUTH_CONFIGURED:
        raise HTTPException(503, "Marker auth is not configured; marker editing disabled")
    tok = _make_marker_token(clientId)
    return {"clientId": clientId, **tok}


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


def _location_search_client_key(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()[:128]
    client = request.client.host if request.client else "unknown"
    return str(client)[:128]


def _location_search_check_rate_limit(client_key: str):
    now = time.monotonic()
    with app_state.location_search_lock:
        dq = app_state.location_search_rate.setdefault(client_key, deque())
        while dq and (now - dq[0]) > LOCATION_SEARCH_WINDOW_SECONDS:
            dq.popleft()
        if len(dq) >= LOCATION_SEARCH_MAX_REQUESTS_PER_WINDOW:
            raise HTTPException(429, "Rate limit exceeded for location search")
        dq.append(now)


def _location_search_cache_get(cache_key: tuple[str, int]):
    now = time.monotonic()
    with app_state.location_search_lock:
        item = app_state.location_search_cache.get(cache_key)
        if item is None:
            return None
        payload, ts = item
        if (now - ts) > LOCATION_SEARCH_CACHE_TTL_SECONDS:
            try:
                del app_state.location_search_cache[cache_key]
            except Exception:
                pass
            return None
        app_state.location_search_cache.move_to_end(cache_key)
        return payload


def _location_search_cache_set(cache_key: tuple[str, int], payload: dict):
    with app_state.location_search_lock:
        app_state.location_search_cache[cache_key] = (payload, time.monotonic())
        app_state.location_search_cache.move_to_end(cache_key)
        while len(app_state.location_search_cache) > _LOCATION_SEARCH_CACHE_MAX_ITEMS:
            app_state.location_search_cache.popitem(last=False)


async def api_location_search(request: Request, q: str = Query(..., min_length=2, max_length=120), limit: int = Query(8, ge=1, le=20)):
    """Free-text search biased towards glider fields / airfields / airports.

    Uses a small local OpenAIP-style seed list first, then enriches with Nominatim.
    """
    qn = q.strip()
    ql = qn.lower()

    # Robust server-side rate limit by client key (in addition to upstream-friendly Nominatim pacing below).
    _location_search_check_rate_limit(_location_search_client_key(request))

    cache_key = (ql, int(limit))
    cached = _location_search_cache_get(cache_key)
    if cached is not None:
        return cached

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

    # 2) Nominatim enrichment (paced: max ~1 req/s process-local)
    now = time.monotonic()
    wait_s = 1.0 - (now - app_state.last_nominatim_request)
    if wait_s > 0:
        await asyncio.sleep(min(wait_s, 1.0))
    app_state.last_nominatim_request = time.monotonic()

    rows = []
    try:
        def _nominatim_fetch():
            return requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": qn,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "limit": max(limit * 3, 15),
                    "extratags": 1,
                },
                headers={"User-Agent": "skyview/1.0"},
                timeout=8,
            )
        r = await asyncio.to_thread(_nominatim_fetch)
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

    payload = {"results": dedup[:limit], "count": min(len(dedup), limit)}
    _location_search_cache_set(cache_key, payload)
    return payload


# Backward-compat endpoints (legacy marker list API)
async def api_markers_list(clientId: str = Query(..., min_length=3, max_length=128)):
    marker = _marker_for_client(clientId)
    return {"markers": [marker], "count": 1}


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

    lat = d["lat"]
    lon = d["lon"]

    # EU spatial fill for areas outside D2 domain (strict-only, no recovery)
    eu_fb = None
    if model_used == "icon_d2":
        d2_lat_min, d2_lat_max = float(np.min(lat)), float(np.max(lat))
        d2_lon_min, d2_lon_max = float(np.min(lon)), float(np.max(lon))
        overlaps_outside_d2 = (
            lat_min < d2_lat_min or lat_max > d2_lat_max or lon_min < d2_lon_min or lon_max > d2_lon_max
        )
        if overlaps_outside_d2:
            _eu_raw = _try_load_eu_fallback(time, cfg)
            if _eu_raw is not None and not _eu_raw.get("missing"):
                eu_fb = _eu_raw
    overlay_fallback_used = False
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
            app_state.fallback_stats["overlayFallback"] += 1
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
        cropped = compute_computed_field_cropped(cfg["var"], d, li, lo, model_used, step)
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

    _set_fallback_current(
        "overlay",
        "eu_fallback:outside_d2_domain" if overlay_fallback_used else "primary_model_only",
        source_model=model_used,
        detail={"requestedTime": time, "layer": layer},
    )

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


def _precip_prev_step_and_dt(model_used: str, step: int) -> tuple[Optional[int], float]:
    if model_used == "icon_eu" and step >= ICON_EU_STEP_3H_START:
        prev = step - 3
        return (prev if prev >= 1 else None), 3.0
    if step >= 2:
        return step - 1, 1.0
    return None, 1.0


def _overlay_source_field(layer: str, cfg: dict, d: dict, model_used: str, run: str, step: int):
    """Return source field for overlay rendering (full-grid) with computed caching."""
    if cfg.get("computed"):
        comp_key = f"{model_used}|{run}|{step}|{layer}"
        return computed_cache_get_or_compute(
            comp_key,
            layer=layer,
            compute_fn=lambda: compute_computed_field_full(cfg["var"], d, model_used, step),
        )

    vname = cfg["var"]
    if vname not in d and layer == "clouds_total_mod" and "clct" in d:
        vname = "clct"
    if vname not in d:
        raise HTTPException(404, f"Variable {cfg['var']} unavailable")
    src = d[vname]
    if layer == "clouds_total_mod":
        src = normalize_clouds_total_mod(src)
    return src


def _maybe_warmup_overlay_context(layer: str, cfg: dict, d: dict, model_used: str, run: str, step: int):
    """Proactively precompute field cache once per context/layer to reduce burst misses."""
    if not OVERLAY_WARMUP_ENABLED:
        return
    ctx_key = f"{model_used}|{run}|{step}|{layer}"
    now = time.time()
    should_run = False
    with _warmup_lock:
        if _last_overlay_warmup.get("key") != ctx_key or (now - float(_last_overlay_warmup.get("ts", 0.0))) > OVERLAY_WARMUP_MIN_INTERVAL_SECONDS:
            _last_overlay_warmup["key"] = ctx_key
            _last_overlay_warmup["ts"] = now
            should_run = True
    if not should_run:
        return
    try:
        _overlay_source_field(layer, cfg, d, model_used, run, step)
    except Exception:
        # Warmup must never fail the request path.
        pass


def _resolve_eu_time_strict(time_str: str, max_delta_hours: float = EU_STRICT_MAX_DELTA_HOURS):
    """Resolve EU run/step only if close enough to requested time (service-backed)."""
    return svc_resolve_eu_time_strict(
        time_str=time_str,
        max_delta_hours=max_delta_hours,
        max_delta_hours_3h=EU_STRICT_MAX_DELTA_HOURS_3H,
        resolve_time_fn=resolve_time,
        cache=app_state.eu_strict_cache,
        cache_ttl_seconds=_EU_STRICT_CACHE_TTL_SECONDS,
        cache_max=_EU_STRICT_CACHE_MAX,
        fallback_stats=app_state.fallback_stats,
        logger=logger,
    )


def _load_eu_data_strict(
    time_str: str,
    keys: list[str],
    max_delta_hours: float = EU_STRICT_MAX_DELTA_HOURS,
) -> Optional[Dict[str, Any]]:
    """Load ICON-EU data strictly matching the requested time (service-backed)."""
    payload, updated_until = svc_load_eu_data_strict(
        time_str=time_str,
        keys=keys,
        max_delta_hours=max_delta_hours,
        resolve_eu_time_strict_fn=_resolve_eu_time_strict,
        load_data_fn=load_data,
        eu_missing_until_mono=app_state.eu_missing_until_mono,
        eu_missing_backoff_seconds=_EU_MISSING_BACKOFF_SECONDS,
        logger=logger,
    )
    app_state.eu_missing_until_mono = updated_until
    return payload


def _try_load_eu_fallback(time_str: str, cfg: dict, max_delta_hours: float = EU_STRICT_MAX_DELTA_HOURS):
    """Load ICON-EU fallback for overlay endpoints (strict, no recovery)."""
    overlay_keys = build_overlay_keys(cfg)
    return _load_eu_data_strict(time_str, overlay_keys, max_delta_hours=max_delta_hours)

app.include_router(build_point_router(
    resolve_time_with_cache_context=resolve_time_with_cache_context,
    load_data=load_data,
    POINT_KEYS=POINT_KEYS,
    POINT_KEYS_MINIMAL=POINT_KEYS_MINIMAL,
    _resolve_eu_time_strict=_resolve_eu_time_strict,
    _load_eu_data_strict=_load_eu_data_strict,
    rotate_caches_for_context=rotate_caches_for_context,
    fallback_stats=app_state.fallback_stats,
    classify_point=classify_point,
    ww_to_symbol=ww_to_symbol,
    build_overlay_values=build_overlay_values,
    _freshness_minutes_from_run=_freshness_minutes_from_run,
    _set_fallback_current=_set_fallback_current,
))


async def api_status():
    """Aggregate ingest freshness, model/run state, cache/perf metrics and error counters."""
    runs = get_available_runs()
    merged = get_merged_timeline()
    payload = build_status_payload(
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
        overlay_phase_recent=overlay_phase_recent,
        overlay_phase_totals=overlay_phase_totals,
        computed_field_metrics=computed_metrics_payload(),
        api_error_counters=api_error_counters,
        fallback_stats=app_state.fallback_stats,
    )

    # Override freshness semantics: minutes since latest fully ingested run became complete.
    timings = await asyncio.to_thread(_ingest_model_timings)
    full_times = [
        t.get("latestRunFullyIngestedAt")
        for t in timings.values()
        if t.get("latestRunFullyIngestedAt")
    ]
    if full_times:
        latest_full_iso = sorted(full_times)[-1]
        try:
            dt = datetime.fromisoformat(latest_full_iso.replace("Z", "+00:00"))
            payload["ingest"]["freshnessMinutes"] = round((datetime.now(timezone.utc) - dt).total_seconds() / 60.0, 1)
            payload["ingest"]["latestFullIngestedAt"] = latest_full_iso
        except Exception:
            pass

    for mk in ("icon_d2", "icon_eu"):
        if mk in payload.get("ingestHealth", {}).get("models", {}) and mk in timings:
            payload["ingestHealth"]["models"][mk]["latestRunAvailableAt"] = timings[mk].get("latestRunAvailableAt")
            payload["ingestHealth"]["models"][mk]["latestRunFullyAvailableOnDwdAt"] = timings[mk].get("latestRunFullyAvailableOnDwdAt")
            payload["ingestHealth"]["models"][mk]["modelCalculationMinutes"] = timings[mk].get("modelCalculationMinutes")
            payload["ingestHealth"]["models"][mk]["latestRunFullyIngestedAt"] = timings[mk].get("latestRunFullyIngestedAt")
            payload["ingestHealth"]["models"][mk]["ingestDurationMinutes"] = timings[mk].get("ingestDurationMinutes")
            payload["ingestHealth"]["models"][mk]["freshnessMinutesSinceFullIngest"] = timings[mk].get("freshnessMinutesSinceFullIngest")

    # Panel-relevant fallback status: current snapshot, not cumulative counters.
    # Merge per-request snapshot fields into the fallback counter dict built by
    # build_status_payload — use update() so the cumulative counters are preserved.
    payload["fallback"].update({
        "updatedAt": app_state.fallback_current.get("updatedAt"),
        "endpoints": app_state.fallback_current.get("endpoints", {}),
        "strictWindowHours": EU_STRICT_MAX_DELTA_HOURS,
    })

    lz_total = low_zoom_symbols_cache_metrics["hits"] + low_zoom_symbols_cache_metrics["misses"]
    payload["symbolsLowZoomCache"] = {
        "maxZoom": LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM,
        "metrics": low_zoom_symbols_cache_metrics,
        "hitRate": (low_zoom_symbols_cache_metrics["hits"] / lz_total) if lz_total else None,
    }

    # Cache visibility for low-memory VPS sizing (rough in-process estimate).
    tile_desktop_bytes = sum(len(v[0]) for v in tile_cache_desktop.values()) if tile_cache_desktop else 0
    tile_mobile_bytes = sum(len(v[0]) for v in tile_cache_mobile.values()) if tile_cache_mobile else 0
    payload.setdefault("cache", {})["server"] = {
        "npzDataItems": len(data_cache),
        "npzDataMax": DATA_CACHE_MAX_ITEMS,
        "meteogramItems": len(meteogram_cache),
        "meteogramMax": METEOGRAM_CACHE_MAX_ITEMS,
        "estTileBytes": tile_desktop_bytes + tile_mobile_bytes,
    }

    return payload


async def api_cache_stats():
    tile_cache_prune("desktop")
    tile_cache_prune("mobile")
    lz_total = low_zoom_symbols_cache_metrics["hits"] + low_zoom_symbols_cache_metrics["misses"]
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
        "computedFieldMetrics": computed_metrics_payload(),
        "symbolsLowZoomCache": {
            "maxZoom": LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM,
            "metrics": low_zoom_symbols_cache_metrics,
            "hitRate": (low_zoom_symbols_cache_metrics["hits"] / lz_total) if lz_total else None,
        },
    }


async def api_usage_stats(days: int = Query(30, ge=1, le=365)):
    usage = get_usage_stats(USAGE_STATS_FILE, days=days)
    usage["markers"] = get_marker_stats(MARKERS_FILE)
    usage["notes"] = {
        "privacy": "unique visitors are estimated via hashed fingerprint (ip+ua+lang+salt)",
        "scope": "visits counted on '/' and '/index.html' page loads",
    }
    return usage


async def api_perf_stats(reset: bool = Query(False, description="Reset perf counters after returning stats")):
    payload = build_perf_payload(perf_recent, perf_totals)

    if reset:
        perf_recent.clear()
        perf_totals['requests'] = 0
        perf_totals['hits'] = 0
        perf_totals['misses'] = 0
        perf_totals['totalMs'] = 0.0
        overlay_phase_recent.clear()
        overlay_phase_totals['requests'] = 0
        overlay_phase_totals['hits'] = 0
        overlay_phase_totals['misses'] = 0
        overlay_phase_totals['loadMs'] = 0.0
        overlay_phase_totals['sourceMs'] = 0.0
        overlay_phase_totals['colorizeMs'] = 0.0
        overlay_phase_totals['encodeMs'] = 0.0
        overlay_phase_totals['totalMs'] = 0.0
        low_zoom_symbols_cache_metrics['hits'] = 0
        low_zoom_symbols_cache_metrics['misses'] = 0
        low_zoom_symbols_cache_metrics['diskHits'] = 0
        low_zoom_symbols_cache_metrics['diskMisses'] = 0
        payload['reset'] = True

    return payload


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
    t_load_ms = 0.0
    t_source_ms = 0.0
    t_colorize_ms = 0.0
    t_encode_ms = 0.0

    if layer not in OVERLAY_CONFIGS:
        raise HTTPException(400, f"Unknown layer: {layer}")

    client_class = "mobile" if clientClass == "mobile" else "desktop"
    cfg = OVERLAY_CONFIGS[layer]

    # Always load hsurf alongside overlay keys: it reflects the true D2 domain validity
    # (NaN outside the irregular rotated-lat-lon domain, not just at the rectangular bbox edge).
    overlay_keys = build_overlay_keys(cfg)
    if "hsurf" not in overlay_keys:
        overlay_keys = list(overlay_keys) + ["hsurf"]

    t_load0 = perf_counter()
    run, step, model_used = resolve_time_with_cache_context(time, model)
    d = load_data(run, step, model_used, keys=overlay_keys)
    lat = d["lat"]
    lon = d["lon"]

    # EU load decision: check whether this tile has any pixels where D2 has no valid data.
    # ICON-D2 uses a rotated lat-lon grid whose domain is highly irregular in regular lat-lon.
    # The rectangular bbox (np.min/max lat/lon) is much larger than the actual valid domain:
    # e.g. at lat=50°N valid D2 lons start at ~-1.9°E, not at the bbox edge of -3.94°E.
    # We use hsurf (terrain elevation) as the domain mask — it is NaN exactly where D2 has
    # no data, regardless of the cause (irregular boundary, relaxation zone, etc.).
    _eu_raw_tile = None
    eu_fb = None
    d2_lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    d2_lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.02
    if model_used == "icon_d2":
        d2_lat_min, d2_lat_max = float(np.min(lat)), float(np.max(lat))
        d2_lon_min, d2_lon_max = float(np.min(lon)), float(np.max(lon))
        minx, miny, maxx, maxy = _tile_bounds_3857(z, x, y)
        lon0, lat0 = _merc_to_lonlat(minx, miny)
        lon1, lat1 = _merc_to_lonlat(maxx, maxy)
        t_lon_min, t_lon_max = min(lon0, lon1), max(lon0, lon1)
        t_lat_min, t_lat_max = min(lat0, lat1), max(lat0, lat1)

        # Fast path: tile entirely outside D2 rectangular bbox → definitely need EU.
        outside_d2_rect = (
            t_lat_min >= d2_lat_max or t_lat_max <= d2_lat_min
            or t_lon_min >= d2_lon_max or t_lon_max <= d2_lon_min
        )
        # Tile overlaps D2 bbox but might have NaN in the irregular domain interior.
        needs_eu_fill = outside_d2_rect
        if not needs_eu_fill and "hsurf" in d:
            # Slice hsurf to tile bounds (+1 cell padding) and check for any NaN.
            # This correctly handles the irregular D2 boundary in all directions.
            lat_lo = int(max(0, np.searchsorted(lat, t_lat_min) - 1))
            lat_hi = int(min(len(lat) - 1, np.searchsorted(lat, t_lat_max) + 1))
            lon_lo = int(max(0, np.searchsorted(lon, t_lon_min) - 1))
            lon_hi = int(min(len(lon) - 1, np.searchsorted(lon, t_lon_max) + 1))
            tile_hsurf = d["hsurf"][lat_lo : lat_hi + 1, lon_lo : lon_hi + 1]
            needs_eu_fill = not bool(np.all(np.isfinite(tile_hsurf)))
        elif not needs_eu_fill:
            # hsurf not available — fall back to 5-cell-inward rectangular margin.
            needs_eu_fill = (
                t_lat_min < d2_lat_min + d2_lat_res * 5
                or t_lat_max > d2_lat_max - d2_lat_res * 5
                or t_lon_min < d2_lon_min + d2_lon_res * 5
                or t_lon_max > d2_lon_max - d2_lon_res * 5
            )

        if needs_eu_fill:
            _eu_raw_tile = _try_load_eu_fallback(time, cfg)
            eu_fb = _eu_raw_tile if (_eu_raw_tile is not None and not _eu_raw_tile.get("missing")) else None
    t_load_ms += (perf_counter() - t_load0) * 1000.0

    # Warm cache once per context/layer to smooth bursty tile starts.
    _maybe_warmup_overlay_context(layer, cfg, d, model_used, run, step)
    if eu_fb is not None:
        try:
            _maybe_warmup_overlay_context(layer, cfg, eu_fb["data"], eu_fb["model"], eu_fb["run"], eu_fb["step"])
        except Exception:
            pass
    eu_key = f"|eu:{eu_fb['run']}:{eu_fb['step']}" if eu_fb else ""
    cache_key = f"{client_class}|{model_used}|{run}|{step}{eu_key}|{layer}|{z}|{x}|{y}"
    cached = tile_cache_get(client_class, cache_key)
    if cached is not None:
        total_ms = (perf_counter() - t0) * 1000.0
        perf_record(total_ms, True)
        _record_overlay_layer_metric(layer, True, total_ms)
        overlay_phase_record(
            layer=layer,
            cache_hit=True,
            load_ms=t_load_ms,
            source_ms=t_source_ms,
            colorize_ms=t_colorize_ms,
            encode_ms=t_encode_ms,
            total_ms=total_ms,
        )
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
            t_enc0 = perf_counter()
            b = io.BytesIO(); empty.save(b, format="PNG", optimize=True); png = b.getvalue()
            t_encode_ms += (perf_counter() - t_enc0) * 1000.0
            tile_cache_set(client_class, cache_key, png)
            total_ms = (perf_counter() - t0) * 1000.0
            perf_record(total_ms, False)
            _record_overlay_layer_metric(layer, False, total_ms)
            overlay_phase_record(
                layer=layer,
                cache_hit=False,
                load_ms=t_load_ms,
                source_ms=t_source_ms,
                colorize_ms=t_colorize_ms,
                encode_ms=t_encode_ms,
                total_ms=total_ms,
            )
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
    t_src0 = perf_counter()
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
    t_source_ms += (perf_counter() - t_src0) * 1000.0

    xs = np.linspace(minx, maxx, 256, endpoint=False) + (maxx - minx) / 512.0
    ys = np.linspace(maxy, miny, 256, endpoint=False) - (maxy - miny) / 512.0
    mx, my = np.meshgrid(xs, ys)
    qlon, qlat = _merc_to_lonlat(mx, my)

    # Expand inside mask by half a grid cell so D2 fills right to the outer edge of its last cells.
    # Without this, there is a half-cell transparent strip at each D2 boundary edge.
    inside = (
        (qlat >= data_lat_min - d2_lat_res / 2) & (qlat <= data_lat_max + d2_lat_res / 2) &
        (qlon >= data_lon_min - d2_lon_res / 2) & (qlon <= data_lon_max + d2_lon_res / 2)
    )
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
            app_state.fallback_stats["overlayTileFallback"] += 1

    t_col0 = perf_counter()
    rgba = colorize_layer_vectorized(layer, sampled, valid)
    t_colorize_ms += (perf_counter() - t_col0) * 1000.0

    img = Image.fromarray(rgba, mode="RGBA")
    t_enc0 = perf_counter()
    b = io.BytesIO(); img.save(b, format="PNG", optimize=True); png = b.getvalue()
    t_encode_ms += (perf_counter() - t_enc0) * 1000.0
    tile_cache_set(client_class, cache_key, png)
    total_ms = (perf_counter() - t0) * 1000.0
    perf_record(total_ms, False)
    _record_overlay_layer_metric(layer, False, total_ms)
    overlay_phase_record(
        layer=layer,
        cache_hit=False,
        load_ms=t_load_ms,
        source_ms=t_source_ms,
        colorize_ms=t_colorize_ms,
        encode_ms=t_encode_ms,
        total_ms=total_ms,
    )

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


# Phase-3 modularization: bind large endpoint groups via routers
app.include_router(build_weather_router(
    resolve_time_with_cache_context=resolve_time_with_cache_context,
    load_data=load_data,
    _load_eu_data_strict=_load_eu_data_strict,
    fallback_stats=app_state.fallback_stats,
    _set_fallback_current=_set_fallback_current,
    _freshness_minutes_from_run=_freshness_minutes_from_run,
    EU_STRICT_MAX_DELTA_HOURS=EU_STRICT_MAX_DELTA_HOURS,
    low_zoom_symbols_cache_metrics=low_zoom_symbols_cache_metrics,
    data_dir=DATA_DIR,
    meteogram_cache=meteogram_cache,
    METEOGRAM_CACHE_MAX_ITEMS=METEOGRAM_CACHE_MAX_ITEMS,
    get_merged_timeline=get_merged_timeline,
    logger=logger,
))
app.include_router(build_overlay_router(
    api_overlay=api_overlay,
    api_overlay_tile=api_overlay_tile,
))


def _tail_lines(path: str, limit: int = 300) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-limit:]]
    except Exception:
        return []


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return int(total)


_dwd_run_available_cache: OrderedDict[str, tuple[Optional[str], float]] = OrderedDict()


def _fetch_dwd_run_fully_available_at(model_key: str, run: Optional[str]) -> Optional[str]:
    """Best-effort DWD full-run availability via Last-Modified on final-step ww file."""
    if not run:
        return None
    cache_key = f"{model_key}:{run}"
    now_mono = time.monotonic()
    cached = _dwd_run_available_cache.get(cache_key)
    if cached and (now_mono - cached[1]) < 900:
        _dwd_run_available_cache.move_to_end(cache_key)
        return cached[0]

    hh = run[-2:]
    if model_key == "icon_d2":
        url = (
            f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{hh}/ww/"
            f"icon-d2_germany_regular-lat-lon_single-level_{run}_048_2d_ww.grib2.bz2"
        )
    else:
        url = (
            f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{hh}/ww/"
            f"icon-eu_europe_regular-lat-lon_single-level_{run}_120_WW.grib2.bz2"
        )

    iso = None
    try:
        r = requests.head(url, timeout=6)
        if r.status_code < 400:
            lm = r.headers.get("Last-Modified")
            if lm:
                dt = parsedate_to_datetime(lm)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                iso = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        iso = None

    _dwd_run_available_cache[cache_key] = (iso, now_mono)
    _dwd_run_available_cache.move_to_end(cache_key)
    while len(_dwd_run_available_cache) > 32:
        _dwd_run_available_cache.popitem(last=False)
    return iso


def _ingest_model_timings() -> dict[str, dict[str, Any]]:
    """Per-model run timing diagnostics from local files.

    latestRunAvailableAt uses runTime from timeline; latestRunFullyIngestedAt uses max mtime of
    npz files in the newest run that has full expected step coverage.
    """
    now = datetime.now(timezone.utc)
    expected_steps = {"icon_d2": 48, "icon_eu": 92}
    runs = get_available_runs()
    by_model = {
        "icon_d2": next((r for r in runs if r.get("model") == "icon_d2"), None),
        "icon_eu": next((r for r in runs if r.get("model") == "icon_eu"), None),
    }
    out: dict[str, dict[str, Any]] = {}

    for model_dir, model_key in (("icon-d2", "icon_d2"), ("icon-eu", "icon_eu")):
        base = os.path.join(DATA_DIR, model_dir)
        run_dirs = sorted([d for d in os.listdir(base) if d.isdigit()], reverse=True) if os.path.isdir(base) else []
        full_dt = None
        full_steps = None
        full_run = None
        for run in run_dirs:
            npz_files = sorted(glob.glob(os.path.join(base, run, "*.npz")))
            if len(npz_files) >= expected_steps[model_key]:
                full_dt = datetime.fromtimestamp(max(os.path.getmtime(p) for p in npz_files), tz=timezone.utc)
                full_steps = len(npz_files)
                full_run = run
                break

        freshness = round((now - full_dt).total_seconds() / 60.0, 1) if full_dt else None
        latest = by_model.get(model_key)
        latest_run = latest.get("run") if latest else None
        latest_run_start_iso = None
        if latest_run:
            try:
                latest_run_start_iso = datetime.strptime(latest_run, "%Y%m%d%H").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                latest_run_start_iso = None

        dwd_available_iso = _fetch_dwd_run_fully_available_at(model_key, latest_run)

        latest_run_first_ingested_iso = None
        ingest_running_minutes = None
        latest_run_step_count = 0
        if latest_run:
            latest_run_npz = sorted(glob.glob(os.path.join(base, latest_run, "*.npz")))
            latest_run_step_count = len(latest_run_npz)
            if latest_run_npz:
                first_dt = datetime.fromtimestamp(min(os.path.getmtime(p) for p in latest_run_npz), tz=timezone.utc)
                latest_run_first_ingested_iso = first_dt.isoformat().replace("+00:00", "Z")
                if latest_run_step_count < expected_steps[model_key]:
                    ingest_running_minutes = round((now - first_dt).total_seconds() / 60.0, 1)

        calc_minutes = None
        if latest_run_start_iso and dwd_available_iso:
            try:
                rs = datetime.fromisoformat(latest_run_start_iso.replace("Z", "+00:00"))
                da = datetime.fromisoformat(dwd_available_iso.replace("Z", "+00:00"))
                calc_minutes = round((da - rs).total_seconds() / 60.0, 1)
            except Exception:
                calc_minutes = None

        ingest_minutes = None
        full_iso = full_dt.isoformat().replace("+00:00", "Z") if full_dt else None
        if dwd_available_iso and full_iso and latest_run and full_run == latest_run:
            try:
                da = datetime.fromisoformat(dwd_available_iso.replace("Z", "+00:00"))
                fi = datetime.fromisoformat(full_iso.replace("Z", "+00:00"))
                ingest_minutes = round((fi - da).total_seconds() / 60.0, 1)
            except Exception:
                ingest_minutes = None

        out[model_key] = {
            "latestRun": latest_run,
            "latestRunStartAt": latest_run_start_iso,
            "latestRunAvailableAt": latest.get("runTime") if latest else None,
            "latestRunFullyAvailableOnDwdAt": dwd_available_iso,
            "modelCalculationMinutes": calc_minutes,
            "latestRunFirstIngestedAt": latest_run_first_ingested_iso,
            "ingestRunningMinutes": ingest_running_minutes,
            "latestRunFullyIngestedRun": full_run,
            "latestRunFullyIngestedAt": full_iso,
            "latestRunFullyIngestedSteps": full_steps,
            "ingestDurationMinutes": ingest_minutes,
            "freshnessMinutesSinceFullIngest": freshness,
        }
    return out


async def api_admin_storage():
    """Storage + ingest/admin overview (runs, timings, tmp state, markers, usage)."""
    expected_steps = {"icon_d2": 48, "icon_eu": 92}
    model_timings = await asyncio.to_thread(_ingest_model_timings)

    out: dict[str, Any] = {
        "models": {},
        "totalBytes": 0,
        "ingest": {},
        "tmp": {},
        "markers": get_marker_stats(MARKERS_FILE),
        "usage": get_usage_stats(USAGE_STATS_FILE, days=30),
    }

    for model_dir, model_key in (("icon-d2", "icon_d2"), ("icon-eu", "icon_eu")):
        base = os.path.join(DATA_DIR, model_dir)
        model_total = _dir_size_bytes(base)
        runs = sorted([d for d in os.listdir(base) if d.isdigit()], reverse=True) if os.path.isdir(base) else []
        run_items: list[dict[str, Any]] = []
        latest_full_ingested_at = None
        latest_full_ingested_step_count = None

        for run in runs:
            rp = os.path.join(base, run)
            npz_files = sorted(glob.glob(os.path.join(rp, "*.npz")))
            run_bytes = sum(os.path.getsize(p) for p in npz_files if os.path.exists(p))
            latest_npz_mtime = None
            if npz_files:
                latest_npz_mtime = datetime.fromtimestamp(max(os.path.getmtime(p) for p in npz_files), tz=timezone.utc)

            run_items.append({
                "run": run,
                "npzFiles": len(npz_files),
                "bytes": int(run_bytes),
                "latestNpzsUpdatedAt": latest_npz_mtime.isoformat().replace("+00:00", "Z") if latest_npz_mtime else None,
            })

            if latest_full_ingested_at is None and len(npz_files) >= expected_steps[model_key]:
                latest_full_ingested_at = latest_npz_mtime
                latest_full_ingested_step_count = len(npz_files)

        mt = model_timings.get(model_key, {})

        out["models"][model_key] = {
            "path": base,
            "totalBytes": int(model_total),
            "runs": run_items,
            "latestRun": mt.get("latestRun") or (run_items[0]["run"] if run_items else None),
            "latestRunStartAt": mt.get("latestRunStartAt"),
            "latestRunAvailableAt": mt.get("latestRunAvailableAt"),
            "latestRunFullyAvailableOnDwdAt": mt.get("latestRunFullyAvailableOnDwdAt"),
            "modelCalculationMinutes": mt.get("modelCalculationMinutes"),
            "latestRunFirstIngestedAt": mt.get("latestRunFirstIngestedAt"),
            "ingestRunningMinutes": mt.get("ingestRunningMinutes"),
            "latestRunFullyIngestedAt": mt.get("latestRunFullyIngestedAt"),
            "latestRunFullyIngestedSteps": mt.get("latestRunFullyIngestedSteps") or latest_full_ingested_step_count,
            "ingestDurationMinutes": mt.get("ingestDurationMinutes"),
            "freshnessMinutesSinceFullIngest": mt.get("freshnessMinutesSinceFullIngest"),
        }
        out["totalBytes"] += int(model_total)

    # tmp folder visibility
    tmp_dir = os.path.join(DATA_DIR, "tmp")
    tmp_entries = []
    if os.path.isdir(tmp_dir):
        for name in sorted(os.listdir(tmp_dir)):
            p = os.path.join(tmp_dir, name)
            try:
                sz = _dir_size_bytes(p) if os.path.isdir(p) else os.path.getsize(p)
            except Exception:
                sz = 0
            tmp_entries.append({"name": name, "isDir": os.path.isdir(p), "bytes": int(sz)})

    lock_path = "/tmp/skyview-ingest.lock"
    lock_info = {"exists": os.path.exists(lock_path), "path": lock_path, "ageSeconds": None}
    if lock_info["exists"]:
        try:
            lock_info["ageSeconds"] = int(time.time() - os.path.getmtime(lock_path))
        except Exception:
            pass

    running = []
    try:
        def _find_ingest_procs():
            pr = subprocess.run(["pgrep", "-af", "ingest.py"], capture_output=True, text=True, timeout=2)
            if pr.returncode == 0 and pr.stdout.strip():
                return [ln.strip() for ln in pr.stdout.strip().splitlines() if ln.strip()]
            return []
        running = await asyncio.to_thread(_find_ingest_procs)
    except Exception:
        pass

    out["tmp"] = {
        "path": tmp_dir,
        "count": len(tmp_entries),
        "entries": tmp_entries,
    }
    out["ingest"] = {
        "lock": lock_info,
        "runningProcesses": running,
        "isRunning": bool(running) or bool(lock_info.get("exists")),
    }

    return out


async def api_admin_logs(
    limit: int = Query(5000, ge=50, le=5000),
    level: str = Query("all", description="all|error|warn|info|debug"),
    source: str = Query("all", description="all|backend|ingest|stdout"),
):
    """Recent backend and ingest logs for admin/status view.

    Each log entry includes a ``source`` field: "backend" (skyview.log),
    "ingest" (ingest.log), or "stdout" (.out files).  The optional ``source``
    query param filters to one group.
    """

    def _source_for_file(basename: str) -> str:
        b = basename.lower()
        if b.startswith("ingest"):
            return "ingest"
        if b.endswith(".out"):
            return "stdout"
        return "backend"

    def _collect_logs_payload():
        logs_dir = os.path.join(SCRIPT_DIR, "logs")
        all_files = sorted(
            glob.glob(os.path.join(logs_dir, "*.log"))
            + glob.glob(os.path.join(logs_dir, "*.out"))
        )

        lvl = (level or "all").strip().lower()
        needles = {
            "error": ["error", "exception", "traceback", "critical"],
            "warn": ["warn", "warning"],
            "info": ["info"],
            "debug": ["debug"],
        }.get(lvl)

        src_filter = (source or "all").strip().lower()

        payload: list[dict[str, Any]] = []
        for fp in all_files:
            basename = os.path.basename(fp)
            file_source = _source_for_file(basename)

            if src_filter != "all" and file_source != src_filter:
                continue

            lines = _tail_lines(fp, limit=limit)
            if needles:
                lines = [ln for ln in lines if any(n in ln.lower() for n in needles)]

            payload.append({
                "file": basename,
                "path": fp,
                "source": file_source,
                "sizeBytes": os.path.getsize(fp) if os.path.exists(fp) else 0,
                "tail": lines,
            })

        # Sort: backend first, then ingest, then stdout
        _order = {"backend": 0, "ingest": 1, "stdout": 2}
        payload.sort(key=lambda e: (_order.get(e["source"], 9), e["file"]))

        ingest_artifacts = [
            {"name": os.path.basename(p), "path": os.path.abspath(p), "sizeBytes": os.path.getsize(p)}
            for p in sorted(glob.glob(os.path.join(logs_dir, "*ingest*")))
            if os.path.isfile(p)
        ]
        regression_artifacts = [
            {"name": os.path.basename(p), "path": os.path.abspath(p), "sizeBytes": os.path.getsize(p)}
            for p in sorted(glob.glob(os.path.join(SCRIPT_DIR, "..", "scripts", "qa_*.py")))
            if os.path.isfile(p)
        ]

        return {
            "logs": payload,
            "count": len(payload),
            "artifacts": {
                "ingest": ingest_artifacts,
                "regression": regression_artifacts,
            },
        }

    return await asyncio.to_thread(_collect_logs_payload)


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * float(p)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


async def api_admin_overview_metrics():
    """Extended admin metrics: reliability, endpoint perf, capacity, product usage and server health."""
    storage = await api_admin_storage()
    status = await api_status()

    # Reliability: endpoint error counters + top failing endpoints
    endpoint_rows = []
    for path, m in request_metrics.items():
        total = int(m.get("count", 0))
        e4 = int(m.get("errors4xx", 0))
        e5 = int(m.get("errors5xx", 0))
        endpoint_rows.append({
            "path": path,
            "requests": total,
            "errors4xx": e4,
            "errors5xx": e5,
            "errorTotal": e4 + e5,
            "lastSeen": m.get("lastSeen"),
        })
    top_failing = sorted(endpoint_rows, key=lambda r: (r["errorTotal"], r["errors5xx"]), reverse=True)[:10]

    # Ingest failure streak: consecutive latest runs with incomplete coverage
    streak = 0
    expected_steps = {"icon_d2": 48, "icon_eu": 92}
    for mk, mdir in (("icon_d2", "icon-d2"), ("icon_eu", "icon-eu")):
        runs = (storage.get("models", {}).get(mk, {}) or {}).get("runs", []) or []
        for r in runs:
            npz = int(r.get("npzFiles") or 0)
            if npz >= expected_steps[mk]:
                break
            streak += 1

    # Performance: per-endpoint latencies and slow buckets
    endpoint_perf = []
    for path, m in request_metrics.items():
        lats = list(m.get("latencyMs", []))
        endpoint_perf.append({
            "path": path,
            "requests": int(m.get("count", 0)),
            "p50Ms": _pct(lats, 0.5),
            "p95Ms": _pct(lats, 0.95),
            "p99Ms": _pct(lats, 0.99),
            "slowOver1s": int(m.get("slowOver1s", 0)),
            "slowOver2s": int(m.get("slowOver2s", 0)),
        })
    endpoint_perf.sort(key=lambda r: r["requests"], reverse=True)

    overlay_phase_series = [{
        "ts": datetime.fromtimestamp(float(r.get("ts", time.time())), tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "loadMs": r.get("loadMs"),
        "sourceMs": r.get("sourceMs"),
        "colorizeMs": r.get("colorizeMs"),
        "encodeMs": r.get("encodeMs"),
        "totalMs": r.get("totalMs"),
        "layer": r.get("layer"),
    } for r in list(overlay_phase_recent)[-120:]]

    cache_efficiency_by_layer = []
    for layer, m in overlay_layer_metrics.items():
        req = int(m.get("requests", 0))
        hit = int(m.get("hits", 0))
        lats = list(m.get("latencyMs", []))
        cache_efficiency_by_layer.append({
            "layer": layer,
            "requests": req,
            "hitRate": (hit / req) if req else None,
            "p95Ms": _pct(lats, 0.95),
        })
    cache_efficiency_by_layer.sort(key=lambda r: r["requests"], reverse=True)

    # Capacity trends
    total_bytes = int(storage.get("totalBytes") or 0)
    est_tile_bytes = int((((status.get("cache") or {}).get("server") or {}).get("estTileBytes") or 0))
    _sample_capacity(total_bytes, est_tile_bytes)

    disk_growth_per_day = None
    cache_growth_per_day = None
    if len(capacity_samples) >= 2:
        first, last = capacity_samples[0], capacity_samples[-1]
        try:
            t0 = datetime.fromisoformat(str(first["ts"]).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(last["ts"]).replace("Z", "+00:00"))
            ddays = max((t1 - t0).total_seconds() / 86400.0, 0.0001)
            disk_growth_per_day = (int(last["totalBytes"]) - int(first["totalBytes"])) / ddays
            cache_growth_per_day = (int(last["estTileBytes"]) - int(first["estTileBytes"])) / ddays
        except Exception:
            pass

    # Product usage
    total_feature = sum(int(v) for v in feature_usage_counters.values())
    feature_split = []
    for k, v in feature_usage_counters.items():
        n = int(v)
        feature_split.append({"feature": k, "count": n, "share": (n / total_feature) if total_feature else None})
    feature_split.sort(key=lambda r: r["count"], reverse=True)

    usage7 = get_usage_stats(USAGE_STATS_FILE, days=7)
    usage30 = get_usage_stats(USAGE_STATS_FILE, days=30)
    markers = get_marker_stats(MARKERS_FILE)

    phase_recent = list(overlay_phase_recent)
    def _phase_pct(field: str, pct: float):
        return _pct([float(r.get(field, 0.0)) for r in phase_recent], pct)

    return {
        "reliability": {
            "globalErrors": dict(api_error_counters),
            "topFailingEndpoints": top_failing,
            "ingestFailureStreak": streak,
            "fallback": dict(app_state.fallback_stats),
        },
        "performance": {
            "endpointLatency": endpoint_perf[:20],
            "overlayPhaseSeries": overlay_phase_series,
            "overlayPhaseP95": {
                "loadMs": _phase_pct("loadMs", 0.95),
                "sourceMs": _phase_pct("sourceMs", 0.95),
                "colorizeMs": _phase_pct("colorizeMs", 0.95),
                "encodeMs": _phase_pct("encodeMs", 0.95),
                "totalMs": _phase_pct("totalMs", 0.95),
            },
            "cacheEfficiencyByLayer": cache_efficiency_by_layer,
        },
        "capacity": {
            "totalBytes": total_bytes,
            "estTileBytes": est_tile_bytes,
            "diskGrowthBytesPerDay": disk_growth_per_day,
            "cacheGrowthBytesPerDay": cache_growth_per_day,
            "sampleCount": len(capacity_samples),
        },
        "product": {
            "featureUsageSplit": feature_split,
            "usage7dVisits": usage7.get("totalVisits", 0),
            "usage30dVisits": usage30.get("totalVisits", 0),
            "usage7dUnique": usage7.get("estimatedUniqueVisitors", 0),
            "usage30dUnique": usage30.get("estimatedUniqueVisitors", 0),
            "markerActive24h": markers.get("activeLast24h", 0),
            "markerActive7d": markers.get("activeLast7d", 0),
            "markerActive30d": markers.get("activeLast30d", 0),
        },
        "serverHealth": _read_server_health(),
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


async def admin_view():
    """Simple Skyview admin/status dashboard page."""
    p = os.path.join(FRONTEND_DIR, "admin.html")
    if not os.path.isfile(p):
        raise HTTPException(404, "admin.html not found")
    return FileResponse(p)



# Phase-4 modularization: ops/admin endpoints bound via routers
app.include_router(build_ops_router(
    api_feedback=api_feedback,
    api_feedback_list=api_feedback_list,
    api_feedback_update=api_feedback_update,
    api_marker_profile=api_marker_profile,
    api_marker_auth=api_marker_auth,
    api_marker_profile_set=api_marker_profile_set,
    api_location_search=api_location_search,
    api_markers_list=api_markers_list,
    api_markers_reset_default=api_markers_reset_default,
))
app.include_router(build_admin_router(
    api_status=api_status,
    api_cache_stats=api_cache_stats,
    api_usage_stats=api_usage_stats,
    api_perf_stats=api_perf_stats,
    api_admin_storage=api_admin_storage,
    api_admin_logs=api_admin_logs,
    api_admin_overview_metrics=api_admin_overview_metrics,
    admin_view=admin_view,
))

# ─── Static Frontend (must be LAST) ───
# html=True serves index.html for directory requests
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    _acquire_single_instance_or_exit(PID_FILE)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
