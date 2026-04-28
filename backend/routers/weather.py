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
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from time import perf_counter
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from cache_state import symbols_cache_get, symbols_cache_set
from constants import (
    CELL_SIZES_BY_ZOOM,
    EMAGRAM_D2_LEVELS_HPA,
    METEOGRAM_D2_LEVELS_HPA,
    G0,
    SYMBOL_MODE_NATIVE_ZOOM_D2,
    SYMBOL_MODE_NATIVE_ZOOM_EU,
    SYMBOL_MODE_PRECOMPUTED_MAX_ZOOM,
    WORLD_GRID_ANCHOR_LAT,
    WORLD_GRID_ANCHOR_LON,
    LOW_ZOOM_PRECOMPUTED_BINS_ENABLED,
)
from grid_aggregation import build_grid_context, choose_cell_groups
from grid_utils import bbox_indices as _bbox_indices, slice_array as _slice_array
from services.symbol_ops import (
    filter_symbols_to_bbox,
    load_symbols_precomputed,
    load_symbols_precomputed_bins_merged,
    save_symbols_precomputed_bin,
    symbols_bin_bbox,
    symbols_bin_indices_for_bbox,
)
from services.symbol_compute import compute_symbols_payload, load_coverage_damping_cfg
from services.storage_io import read_step_point_arrays


def _native_zoom_threshold_for_model(model_name: Optional[str]) -> int:
    normalized = str(model_name or "icon_d2").replace("-", "_")
    return SYMBOL_MODE_NATIVE_ZOOM_EU if normalized == "icon_eu" else SYMBOL_MODE_NATIVE_ZOOM_D2


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
    meteogram_inflight: dict[str, threading.Event] = {}
    meteogram_inflight_lock = threading.Lock()
    meteogram_build_semaphore = threading.Semaphore(
        max(1, int(os.environ.get("SKYVIEW_METEOGRAM_BUILD_CONCURRENCY", "1")))
    )
    METEOGRAM_POINT_CACHE_VERSION = 1
    EMAGRAM_POINT_CACHE_VERSION = 1
    emagram_cache: OrderedDict[str, dict] = OrderedDict()
    emagram_inflight: dict[str, threading.Event] = {}
    emagram_inflight_lock = threading.Lock()
    EMAGRAM_CACHE_MAX_ITEMS = int(os.environ.get("SKYVIEW_EMAGRAM_CACHE_MAX_ITEMS", "128"))

    def _meteogram_cache_set(cache_key: str, payload: dict) -> None:
        meteogram_cache[cache_key] = payload
        meteogram_cache.move_to_end(cache_key)
        while len(meteogram_cache) > METEOGRAM_CACHE_MAX_ITEMS:
            meteogram_cache.popitem(last=False)

    def _meteogram_disk_path(model_key: str, run_key: str, i: int, j: int) -> str:
        return os.path.join(data_dir, "cache", "meteogram-point", model_key, run_key, f"{int(i)}_{int(j)}.json")

    def _read_meteogram_disk_cache(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                wrapper = json.load(f)
            if wrapper.get("version") != METEOGRAM_POINT_CACHE_VERSION:
                return None
            payload = wrapper.get("payload")
            return payload if isinstance(payload, dict) else None
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.debug("Meteogram disk cache read failed for %s: %s", path, exc)
            return None

    def _write_meteogram_disk_cache(path: str, payload: dict) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": METEOGRAM_POINT_CACHE_VERSION, "payload": payload}, f, separators=(",", ":"))
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.debug("Meteogram disk cache write failed for %s: %s", path, exc)
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _versioned_disk_path(kind: str, model_key: str, run_key: str, name: str) -> str:
        return os.path.join(data_dir, "cache", kind, model_key, run_key, name)

    def _read_versioned_disk_cache(path: str, version: int) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                wrapper = json.load(f)
            if wrapper.get("version") != version:
                return None
            payload = wrapper.get("payload")
            return payload if isinstance(payload, dict) else None
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.debug("Disk cache read failed for %s: %s", path, exc)
            return None

    def _write_versioned_disk_cache(path: str, version: int, payload: dict) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": version, "payload": payload}, f, separators=(",", ":"))
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.debug("Disk cache write failed for %s: %s", path, exc)
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _cache_set(cache: OrderedDict, cache_key: str, payload: dict, max_items: int) -> None:
        cache[cache_key] = payload
        cache.move_to_end(cache_key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    def _load_point_arrays(
        *,
        run: str,
        step: int,
        model: str,
        keys: list[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        i: Optional[int] = None,
        j: Optional[int] = None,
    ) -> Optional[dict]:
        return read_step_point_arrays(
            data_dir=data_dir,
            model=model,
            run=run,
            step=step,
            keys=keys,
            lat=lat,
            lon=lon,
            i=i,
            j=j,
            substep_supported_vars=set(),
            logger=logger,
        )

    def _with_requested_point(payload: dict, lat: float, lon: float) -> dict:
        out = dict(payload)
        point = dict(out.get("point") or {})
        point["requestedLat"] = round(float(lat), 5)
        point["requestedLon"] = round(float(lon), 5)
        out["point"] = point
        return out

    def _meteogram_steps(model: Optional[str]) -> tuple[str, list[dict]]:
        merged = get_merged_timeline()
        if not merged or not merged.get("steps"):
            raise HTTPException(404, "No timeline available")

        m = (model or "icon_d2").replace("-", "_")
        if m != "icon_d2":
            raise HTTPException(400, "api_meteogram_point currently supports model=icon_d2 only")
        steps = [s for s in merged.get("steps", []) if s.get("model") == "icon_d2"]
        if not steps:
            raise HTTPException(404, "No timeline for model=icon_d2")
        return m, steps

    def _meteogram_grid_point(steps: list[dict], lat: float, lon: float) -> tuple[int, int, dict]:
        for s in steps:
            run_i, step_i, model_i = s.get("run"), int(s.get("step")), s.get("model")
            try:
                d = load_data(run_i, step_i, model_i, keys=["lat", "lon"])
            except Exception:
                continue
            lat_arr = d.get("lat")
            lon_arr = d.get("lon")
            if lat_arr is None or lon_arr is None or len(lat_arr) == 0 or len(lon_arr) == 0:
                continue
            ii = int(np.argmin(np.abs(lat_arr - lat)))
            jj = int(np.argmin(np.abs(lon_arr - lon)))
            return ii, jj, {
                "requestedLat": round(float(lat), 5), "requestedLon": round(float(lon), 5),
                "gridLat": round(float(lat_arr[ii]), 5), "gridLon": round(float(lon_arr[jj]), 5),
                "i": ii, "j": jj,
            }
        raise HTTPException(404, "No meteogram grid available")

    def _build_meteogram_payload(
        *,
        steps: list[dict],
        grid_point: dict,
        ii: int,
        jj: int,
        needed_keys: list[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        out: List[dict] = []
        for s in steps:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("meteogram request cancelled")
            run_i, step_i, model_i = s.get("run"), int(s.get("step")), s.get("model")
            try:
                d = _load_point_arrays(run=run_i, step=step_i, model=model_i, keys=needed_keys, i=ii, j=jj)
                local_i = local_j = 0
                if d is None:
                    d = load_data(run_i, step_i, model_i, keys=needed_keys)
                    local_i, local_j = ii, jj
            except Exception:
                continue

            def _g(k: str) -> Optional[float]:
                arr = d.get(k)
                if arr is None:
                    return None
                try:
                    v = arr[local_i, local_j]
                except Exception:
                    return None
                return float(v) if np.isfinite(v) else None

            t2k = _g("t_2m")
            tdk = _g("td_2m")
            wind_levels: List[dict] = []
            for lev in METEOGRAM_D2_LEVELS_HPA:
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
                "hsurfM": _g("hsurf"),
                "zeroDegAltM": _g("hzerocl"),
                "t2mC": round(t2k - 273.15, 2) if t2k is not None else None,
                "dewpoint2mC": round(tdk - 273.15, 2) if tdk is not None else None,
            })

        if not out:
            raise HTTPException(404, "No meteogram data available")

        out.sort(key=lambda r: r.get("validTime") or "")
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

        return {"point": grid_point, "count": len(out), "series": out}

    def _get_or_build_meteogram_payload(
        *,
        lat: float,
        lon: float,
        model: Optional[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        m, steps = _meteogram_steps(model)
        run_key = str(steps[0].get("run") or "")
        ii, jj, grid_point = _meteogram_grid_point(steps, lat, lon)
        cache_key = f"{m}|{run_key}|{ii}|{jj}"
        disk_path = _meteogram_disk_path(m, run_key, ii, jj)

        cached = meteogram_cache.get(cache_key)
        if cached is not None:
            meteogram_cache.move_to_end(cache_key)
            return _with_requested_point(cached, lat, lon)

        disk_payload = _read_meteogram_disk_cache(disk_path)
        if disk_payload is not None:
            _meteogram_cache_set(cache_key, disk_payload)
            return _with_requested_point(disk_payload, lat, lon)

        while True:
            with meteogram_inflight_lock:
                evt = meteogram_inflight.get(cache_key)
                if evt is None:
                    evt = threading.Event()
                    meteogram_inflight[cache_key] = evt
                    owner = True
                    break
                owner = False
            if not owner:
                while not evt.wait(timeout=0.5):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("meteogram request cancelled")
                cached = meteogram_cache.get(cache_key)
                if cached is not None:
                    meteogram_cache.move_to_end(cache_key)
                    return _with_requested_point(cached, lat, lon)
                disk_payload = _read_meteogram_disk_cache(disk_path)
                if disk_payload is not None:
                    _meteogram_cache_set(cache_key, disk_payload)
                    return _with_requested_point(disk_payload, lat, lon)

        acquired = False
        try:
            while not acquired:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("meteogram request cancelled")
                acquired = meteogram_build_semaphore.acquire(timeout=0.5)

            level_keys: List[str] = []
            for lev in METEOGRAM_D2_LEVELS_HPA:
                level_keys += [f"u_{lev}hpa", f"v_{lev}hpa"]
            needed_keys = ["lat", "lon", "validTime", "tot_prec", "h_snow", "t_2m", "td_2m", "hsurf", "hzerocl"] + level_keys
            payload = _build_meteogram_payload(
                steps=steps,
                grid_point=grid_point,
                ii=ii,
                jj=jj,
                needed_keys=needed_keys,
                cancel_event=cancel_event,
            )
            _meteogram_cache_set(cache_key, payload)
            _write_meteogram_disk_cache(disk_path, payload)
            return _with_requested_point(payload, lat, lon)
        finally:
            if acquired:
                meteogram_build_semaphore.release()
            with meteogram_inflight_lock:
                meteogram_inflight.pop(cache_key, None)
                evt.set()

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
        req_lat_min, req_lon_min, req_lat_max, req_lon_max = map(float, parts)

        # Working bbox may be expanded to fixed world bins at low zoom.
        lat_min, lon_min, lat_max, lon_max = req_lat_min, req_lon_min, req_lat_max, req_lon_max

        symbol_keys = [
            "ww", "ceiling", "clcl", "clcm", "clch",
            "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf", "mh",
            "sym_code", "cb_hm",
        ]
        requested_model_normalized = str(model or "icon_d2").replace("-", "_")
        native_zoom_threshold = _native_zoom_threshold_for_model(requested_model_normalized)
        symbol_mode = (
            "precomputed" if (zoom <= SYMBOL_MODE_PRECOMPUTED_MAX_ZOOM and LOW_ZOOM_PRECOMPUTED_BINS_ENABLED) else
            ("native" if zoom >= native_zoom_threshold else "fixed_grid")
        )
        requested_model_for_mode = requested_model_normalized if symbol_mode == "native" else model
        run, step, model_used = resolve_time_with_cache_context(time, requested_model_for_mode)
        is_low_zoom_global = symbol_mode == "precomputed"
        is_native_zoom = symbol_mode == "native"

        bin_ids = symbols_bin_indices_for_bbox(req_lat_min, req_lon_min, req_lat_max, req_lon_max, cell_size) if is_low_zoom_global else []

        # Low zoom: stabilize panning by serving world-fixed bin unions.
        # Compute/cache full bin payloads, then filter to requested viewport.
        if is_low_zoom_global and bin_ids:
            bboxes = [symbols_bin_bbox(i, j, cell_size) for i, j in bin_ids]
            lat_min = min(b[0] for b in bboxes)
            lon_min = min(b[1] for b in bboxes)
            lat_max = max(b[2] for b in bboxes)
            lon_max = max(b[3] for b in bboxes)

        if is_low_zoom_global:
            bins_key = ";".join(f"{i}:{j}" for i, j in bin_ids)
            symbols_cache_key = f"{model_used}|{run}|{step}|z{zoom}|bins|{bins_key}"
        elif not is_native_zoom:
            i0 = int(math.floor((req_lat_min - WORLD_GRID_ANCHOR_LAT) / cell_size))
            i1 = int(math.floor((req_lat_max - WORLD_GRID_ANCHOR_LAT) / cell_size))
            j0 = int(math.floor((req_lon_min - WORLD_GRID_ANCHOR_LON) / cell_size))
            j1 = int(math.floor((req_lon_max - WORLD_GRID_ANCHOR_LON) / cell_size))
            symbols_cache_key = f"{model_used}|{run}|{step}|z{zoom}|cells|{i0}:{i1}:{j0}:{j1}"
        else:
            symbols_cache_key = None

        cached_symbols = symbols_cache_get(symbols_cache_key) if symbols_cache_key else None
        served_from = None
        cache_load_ms = 0.0

        if is_low_zoom_global:
            if cached_symbols is not None:
                low_zoom_symbols_cache_metrics["hits"] += 1
                served_from = "cache-memory"
            else:
                low_zoom_symbols_cache_metrics["misses"] += 1
                t_disk0 = perf_counter()
                cached_symbols = load_symbols_precomputed_bins_merged(
                    data_dir=data_dir,
                    model_used=model_used,
                    run=run,
                    step=step,
                    zoom=zoom,
                    lat_min=lat_min,
                    lon_min=lon_min,
                    lat_max=lat_max,
                    lon_max=lon_max,
                    cell_size=cell_size,
                )
                if cached_symbols is None:
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
                filter_symbols_to_bbox(cached_symbols, req_lat_min, req_lon_min, req_lat_max, req_lon_max)
                if is_low_zoom_global
                else cached_symbols
            )
            diag = dict(out_payload.get("diagnostics") or {})
            diag["symbolMode"] = symbol_mode
            if served_from is not None:
                diag["servedFrom"] = served_from
            out_payload["diagnostics"] = diag
            total_ms = (perf_counter() - t0) * 1000.0
            logger.info(
                "/api/symbols rid=%s served=%s zoom=%s count=%s cacheLoadMs=%.2f totalMs=%.2f",
                rid, served_from or "cache-memory", zoom, out_payload.get("count"),
                cache_load_ms, total_ms,
            )
            return out_payload

        t_comp0 = perf_counter()
        result = compute_symbols_payload(
            zoom=zoom,
            bbox=f"{lat_min},{lon_min},{lat_max},{lon_max}",
            time=time,
            model=model,
            symbol_mode=symbol_mode,
            resolve_time_with_cache_context=resolve_time_with_cache_context,
            load_data=load_data,
            load_eu_data_strict=_load_eu_data_strict,
            freshness_minutes_from_run=_freshness_minutes_from_run,
            strict_window_hours=EU_STRICT_MAX_DELTA_HOURS,
            load_coverage_damping_cfg=load_coverage_damping_cfg,
        )
        t_agg_ms += (perf_counter() - t_comp0) * 1000.0

        if result.get("diagnostics", {}).get("euCells") and result.get("diagnostics", {}).get("d2Cells"):
            fallback_stats["symbolsBlended"] += 1

        _set_fallback_current(
            "symbols",
            result.get("diagnostics", {}).get("fallbackDecision", "primary_model_only"),
            source_model=result.get("model"),
            detail={"requestedTime": time},
        )

        result.setdefault("diagnostics", {})["timingsMs"] = {
            "load": round(t_load_ms, 2),
            "grid": round(t_grid_ms, 2),
            "aggregate": round(t_agg_ms, 2),
        }

        if symbols_cache_key:
            symbols_cache_set(symbols_cache_key, result)

        # Persist world-anchored bin payloads for low-zoom requests when the
        # request exactly matches a single bin. This enables stable pan-serving
        # from disk precompute tiles/bins.
        if is_low_zoom_global and len(bin_ids) == 1:
            bi, bj = bin_ids[0]
            b_lat_min, b_lon_min, b_lat_max, b_lon_max = symbols_bin_bbox(bi, bj, cell_size)
            if (
                abs(lat_min - b_lat_min) < 1e-6 and abs(lon_min - b_lon_min) < 1e-6
                and abs(lat_max - b_lat_max) < 1e-6 and abs(lon_max - b_lon_max) < 1e-6
            ):
                save_symbols_precomputed_bin(
                    data_dir=data_dir,
                    model_used=model_used,
                    run=run,
                    step=step,
                    zoom=zoom,
                    i=bi,
                    j=bj,
                    payload=result,
                    logger=logger,
                )

        out_payload = (
            filter_symbols_to_bbox(result, req_lat_min, req_lon_min, req_lat_max, req_lon_max)
            if is_low_zoom_global
            else result
        )

        total_ms = (perf_counter() - t0) * 1000.0
        logger.info(
            "/api/symbols rid=%s served=computed zoom=%s count=%s euCells=%s d2Cells=%s "
            "loadMs=%.2f gridMs=%.2f aggMs=%.2f totalMs=%.2f",
            rid, zoom, out_payload["count"],
            out_payload.get("diagnostics", {}).get("euCells"), out_payload.get("diagnostics", {}).get("d2Cells"),
            t_load_ms, t_grid_ms, t_agg_ms, total_ms,
        )
        return out_payload

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

        requested_model_normalized = str(model or "icon_d2").replace("-", "_")
        native_zoom_threshold = _native_zoom_threshold_for_model(requested_model_normalized)
        wind_mode = "native" if zoom >= native_zoom_threshold else "fixed_grid"

        requested_model_for_mode = requested_model_normalized if wind_mode == "native" else model
        run, step, model_used = resolve_time_with_cache_context(time, requested_model_for_mode)
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
        c_clat_eu = c_clon_eu = None

        if u_key not in d or v_key not in d or (gust_mode and "vmax_10m" not in d):
            return {
                "barbs": [], "run": run, "model": model_used,
                "validTime": d["validTime"], "level": level, "count": 0,
            }

        pad = cell_size if wind_mode != "native" else (cell_size * 0.5)
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

        if wind_mode == "native":
            barbs: List[dict] = []
            seen_coords = set()
            used_eu_any = False
            skipped_nan = 0
            skipped_bbox = 0
            skipped_dedup = 0
            native_sources = [(c_lat, c_lon, u, v, gust, model_used)]
            if d_eu is not None and u_eu is not None and v_eu is not None:
                native_sources.append((c_lat_eu, c_lon_eu, u_eu, v_eu, gust_eu, "icon_eu"))

            for src_lat_1d, src_lon_1d, src_u, src_v, src_gust, src_model in native_sources:
                rows, cols = src_u.shape
                for ii in range(rows):
                    for jj in range(cols):
                        lat_v = float(src_lat_1d[ii])
                        lon_v = float(src_lon_1d[jj])
                        if not (math.isfinite(lat_v) and math.isfinite(lon_v)):
                            skipped_nan += 1
                            continue
                        if lat_v < lat_min or lat_v > lat_max or lon_v < lon_min or lon_v > lon_max:
                            skipped_bbox += 1
                            continue
                        key = (round(lat_v, 6), round(lon_v, 6))
                        if key in seen_coords:
                            skipped_dedup += 1
                            continue
                        u_v = float(src_u[ii, jj]) if np.isfinite(src_u[ii, jj]) else float("nan")
                        v_v = float(src_v[ii, jj]) if np.isfinite(src_v[ii, jj]) else float("nan")
                        if np.isnan(u_v) or np.isnan(v_v):
                            skipped_nan += 1
                            continue
                        if gust_mode and src_gust is not None and np.isfinite(src_gust[ii, jj]):
                            speed_ms = float(src_gust[ii, jj])
                        else:
                            speed_ms = math.sqrt(u_v ** 2 + v_v ** 2)
                        speed_kt = speed_ms * 1.94384
                        dir_deg = (math.degrees(math.atan2(-u_v, -v_v)) + 360) % 360
                        seen_coords.add(key)
                        if src_model == "icon_eu":
                            used_eu_any = True
                        barbs.append({
                            "lat": round(lat_v, 4),
                            "lon": round(lon_v, 4),
                            "speed_kt": round(speed_kt, 1),
                            "dir_deg": round(dir_deg, 0),
                            "speed_ms": round(speed_ms, 1),
                            "sourceModel": src_model,
                        })

            if used_eu_any:
                fallback_stats["windBlended"] += 1

            resolved_model = "blended" if used_eu_any else model_used
            fallback_dec = "native_point_fallback_blended" if used_eu_any else "native_point_primary_model_only"
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
                    "windMode": wind_mode,
                    "nativeSkipNaN": skipped_nan,
                    "nativeSkipBBox": skipped_bbox,
                    "nativeSkipDedup": skipped_dedup,
                },
            }

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
                "windMode": wind_mode,
            },
        }

    def _emagram_with_requested_point(payload: dict, lat: float, lon: float) -> dict:
        out = dict(payload)
        point = dict(out.get("point") or {})
        point["requestedLat"] = round(float(lat), 5)
        point["requestedLon"] = round(float(lon), 5)
        out["point"] = point
        return out

    def _build_emagram_payload(*, lat: float, lon: float, time: str, model: Optional[str]) -> dict:
        requested_model = model or "icon_d2"
        if requested_model not in ("icon_d2", "icon-d2"):
            raise HTTPException(400, "api_emagram_point currently supports model=icon_d2 only")

        run, step, model_used = resolve_time_with_cache_context(time, "icon_d2")

        coord_data = load_data(run, step, model_used, keys=["lat", "lon"])
        lat_arr = coord_data["lat"]
        lon_arr = coord_data["lon"]
        if len(lat_arr) == 0 or len(lon_arr) == 0:
            raise HTTPException(404, "No grid coordinates available")
        i = int(np.argmin(np.abs(lat_arr - lat)))
        j = int(np.argmin(np.abs(lon_arr - lon)))

        cache_key = f"{model_used}|{run}|{step}|{i}|{j}"
        disk_path = _versioned_disk_path("emagram-point", model_used, run, f"{int(step):03d}_{i}_{j}.json")
        cached = emagram_cache.get(cache_key)
        if cached is not None:
            emagram_cache.move_to_end(cache_key)
            return _emagram_with_requested_point(cached, lat, lon)
        disk_payload = _read_versioned_disk_cache(disk_path, EMAGRAM_POINT_CACHE_VERSION)
        if disk_payload is not None:
            _cache_set(emagram_cache, cache_key, disk_payload, EMAGRAM_CACHE_MAX_ITEMS)
            return _emagram_with_requested_point(disk_payload, lat, lon)

        while True:
            with emagram_inflight_lock:
                evt = emagram_inflight.get(cache_key)
                if evt is None:
                    evt = threading.Event()
                    emagram_inflight[cache_key] = evt
                    owner = True
                    break
                owner = False
            if not owner:
                evt.wait(timeout=30.0)
                cached = emagram_cache.get(cache_key)
                if cached is not None:
                    emagram_cache.move_to_end(cache_key)
                    return _emagram_with_requested_point(cached, lat, lon)
                disk_payload = _read_versioned_disk_cache(disk_path, EMAGRAM_POINT_CACHE_VERSION)
                if disk_payload is not None:
                    _cache_set(emagram_cache, cache_key, disk_payload, EMAGRAM_CACHE_MAX_ITEMS)
                    return _emagram_with_requested_point(disk_payload, lat, lon)

        try:
            keys = (
                [f"t_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
                + [f"fi_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
                + [f"relhum_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
                + [f"u_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
                + [f"v_{lev}hpa" for lev in EMAGRAM_D2_LEVELS_HPA]
            )
            d = _load_point_arrays(run=run, step=step, model=model_used, keys=keys, i=i, j=j)
            local_i = local_j = 0
            if d is None:
                d = load_data(run, step, model_used, keys=keys)
                local_i, local_j = i, j

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
                t_val = d[t_key][local_i, local_j] if t_key in d else np.nan
                fi_val = d[fi_key][local_i, local_j] if fi_key in d else np.nan
                rh_val = d[rh_key][local_i, local_j] if rh_key in d else np.nan
                u_val = d[u_key][local_i, local_j] if u_key in d else np.nan
                v_val = d[v_key][local_i, local_j] if v_key in d else np.nan

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

            payload = {
                "model": model_used, "run": run, "step": step, "validTime": d.get("validTime"),
                "point": {
                    "requestedLat": round(float(lat), 5), "requestedLon": round(float(lon), 5),
                    "gridLat": round(float(lat_arr[i]), 5), "gridLon": round(float(lon_arr[j]), 5),
                    "i": i, "j": j,
                },
                "levels": levels, "count": len(levels),
            }
            _cache_set(emagram_cache, cache_key, payload, EMAGRAM_CACHE_MAX_ITEMS)
            _write_versioned_disk_cache(disk_path, EMAGRAM_POINT_CACHE_VERSION, payload)
            return payload
        finally:
            with emagram_inflight_lock:
                emagram_inflight.pop(cache_key, None)
                evt.set()

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
                    run_in_threadpool(lambda: _build_emagram_payload(lat=lat, lon=lon, time=time, model=model))
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

        return await run_in_threadpool(lambda: _build_emagram_payload(lat=lat, lon=lon, time=time, model=model))

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
                    run_in_threadpool(lambda: _get_or_build_meteogram_payload(
                        lat=lat, lon=lon, model=model, cancel_event=cancel_event
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

        return await run_in_threadpool(lambda: _get_or_build_meteogram_payload(
            lat=lat, lon=lon, model=model
        ))

    @router.get("/api/nowcast_point")
    async def api_nowcast_point(
        request: Request,
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        model: Optional[str] = Query("icon_d2"),
        hours: int = Query(24, ge=1, le=24),
        stream: bool = Query(False),
        _internal: bool = False,
    ):
        if stream and not _internal:
            async def _gen():
                cancel_event = threading.Event()
                yield json.dumps({"type": "progress", "message": "starting nowcast"}) + "\n"
                task = asyncio.create_task(
                    run_in_threadpool(lambda: asyncio.run(
                        api_nowcast_point(request=request, lat=lat, lon=lon,
                                          model=model, hours=hours,
                                          stream=False, _internal=True)
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

        m = (model or "icon_d2").replace("-", "_")
        if m != "icon_d2":
            raise HTTPException(400, "api_nowcast_point currently supports model=icon_d2 only")

        merged = get_merged_timeline()
        if not merged or not merged.get("steps"):
            raise HTTPException(404, "No timeline available")
        steps = [s for s in merged.get("steps", []) if s.get("model") == "icon_d2"]
        if not steps:
            raise HTTPException(404, "No timeline for model=icon_d2")

        try:
            start_dt = datetime.fromisoformat(str(steps[0].get("validTime")).replace("Z", "+00:00"))
        except Exception:
            start_dt = None
        horizon_limit = (start_dt + timedelta(hours=int(hours))) if start_dt is not None else None
        if horizon_limit is not None:
            filtered_steps = []
            for s in steps:
                try:
                    vt = datetime.fromisoformat(str(s.get("validTime")).replace("Z", "+00:00"))
                except Exception:
                    continue
                if vt <= horizon_limit:
                    filtered_steps.append(s)
                else:
                    break
            steps = filtered_steps
            if not steps:
                raise HTTPException(404, "No nowcast data available")

        run_key = str(steps[0].get("run") or "")
        cache_key = f"{m}|{run_key}|{round(float(lat), 4)}|{round(float(lon), 4)}|h{int(hours)}"
        cached = meteogram_cache.get(cache_key)
        if cached is not None:
            meteogram_cache.move_to_end(cache_key)
            return cached

        wanted_keys = [
            "lat", "lon", "cape_ml", "cin_ml", "hbas_sc", "htop_sc", "lpi",
            "cape_ml_substeps", "cape_ml_substep_minutes",
            "cin_ml_substeps", "cin_ml_substep_minutes",
            "hbas_sc_substeps", "hbas_sc_substep_minutes",
            "htop_sc_substeps", "htop_sc_substep_minutes",
            "lpi_substeps", "lpi_substep_minutes",
        ]

        out = []
        grid_point = None
        ii = jj = None

        for s in steps:
            run_i, step_i, model_i = s.get("run"), int(s.get("step")), s.get("model")
            try:
                d = load_data(run_i, step_i, model_i, keys=wanted_keys)
            except Exception:
                continue

            lat_arr = d.get("lat")
            lon_arr = d.get("lon")
            if lat_arr is None or lon_arr is None or len(lat_arr) == 0 or len(lon_arr) == 0:
                continue

            if ii is None or jj is None:
                ii = int(np.argmin(np.abs(lat_arr - lat)))
                jj = int(np.argmin(np.abs(lon_arr - lon)))
                grid_point = {
                    "requestedLat": round(float(lat), 5), "requestedLon": round(float(lon), 5),
                    "gridLat": round(float(lat_arr[ii]), 5), "gridLon": round(float(lon_arr[jj]), 5),
                    "i": ii, "j": jj,
                }

            base_vt = d.get("validTime") or s.get("validTime")
            try:
                base_dt = datetime.fromisoformat(str(base_vt).replace("Z", "+00:00"))
            except Exception:
                continue
            if horizon_limit is not None and base_dt > horizon_limit:
                break

            def _scalar(key: str):
                arr = d.get(key)
                if arr is None:
                    return None
                try:
                    val = float(arr[ii, jj])
                    return val if np.isfinite(val) else None
                except Exception:
                    return None

            def _collect_series(key: str):
                arr = d.get(f"{key}_substeps")
                mins = d.get(f"{key}_substep_minutes")
                if arr is None or mins is None:
                    return {0: _scalar(key)}
                minute_list = [int(x) for x in np.asarray(mins).tolist()]
                out_map = {}
                for idx, minute in enumerate(minute_list):
                    try:
                        val = float(arr[idx, ii, jj])
                        out_map[minute] = val if np.isfinite(val) else None
                    except Exception:
                        out_map[minute] = None
                return out_map

            cape_map = _collect_series("cape_ml")
            cin_map = _collect_series("cin_ml")
            hbas_map = _collect_series("hbas_sc")
            htop_map = _collect_series("htop_sc")
            lpi_map = _collect_series("lpi")
            all_minutes = sorted(set(cape_map) | set(cin_map) | set(hbas_map) | set(htop_map) | set(lpi_map) | {0})

            for minute in all_minutes:
                vt = base_dt + timedelta(minutes=int(minute))
                if horizon_limit is not None and vt > horizon_limit:
                    continue
                hbas_raw = hbas_map.get(minute)
                htop_raw = htop_map.get(minute)
                hbas_v = 0.0 if hbas_raw is None else float(hbas_raw)
                htop_v = 0.0 if htop_raw is None else float(htop_raw)
                conv_thickness = max(0.0, htop_v - hbas_v)
                out.append({
                    "validTime": vt.isoformat().replace("+00:00", "Z"),
                    "run": run_i,
                    "step": step_i,
                    "substepMinutes": int(minute),
                    "capeMl": cape_map.get(minute),
                    "cinMl": cin_map.get(minute),
                    "hbasSc": hbas_v,
                    "htopSc": htop_v,
                    "cloudThickness": conv_thickness,
                    "lpi": lpi_map.get(minute),
                })

        if not out:
            raise HTTPException(404, "No nowcast data available")
        out.sort(key=lambda r: r.get("validTime") or "")
        payload = {"point": grid_point, "count": len(out), "hours": int(hours), "series": out}
        meteogram_cache[cache_key] = payload
        meteogram_cache.move_to_end(cache_key)
        while len(meteogram_cache) > METEOGRAM_CACHE_MAX_ITEMS:
            meteogram_cache.popitem(last=False)
        return payload

    return router
