from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import threading

from services.storage_io import read_static_arrays, read_step_arrays


_data_inflight: Dict[str, threading.Event] = {}
_data_inflight_lock = threading.Lock()
_data_cache_meta: Dict[str, Dict[str, Any]] = {}
_static_grid_cache: Dict[str, Dict[str, Any]] = {}
_static_grid_lock = threading.Lock()

SUBSTEP_SUPPORTED_VARS = {"cape_ml", "cin_ml", "hbas_sc", "htop_sc", "lpi"}
STATIC_GRID_KEYS = {"hsurf"}


def _cache_has_requested_keys(
    *,
    cache_key: str,
    cached: Dict[str, Any],
    keys: Optional[List[str]],
    model: str,
    substep_minutes: int,
) -> bool:
    meta = _data_cache_meta.get(cache_key) or {}
    if keys is None:
        has_keys = bool(meta.get("full"))
    else:
        has_keys = all(k in cached for k in keys)

    if has_keys and substep_minutes > 0 and model == "icon_d2" and keys is not None:
        for k in keys:
            if k in SUBSTEP_SUPPORTED_VARS and (f"{k}_substeps" not in cached or f"{k}_substep_minutes" not in cached):
                return False
    return has_keys


def _remember_cache_meta(cache_key: str, arrays: Dict[str, Any], *, full: bool) -> None:
    loaded = set(_data_cache_meta.get(cache_key, {}).get("keys") or set())
    loaded.update(arrays.keys())
    _data_cache_meta[cache_key] = {"full": bool(full), "keys": loaded}


def _evict_cache_item(cache: Dict[str, Any], logger) -> None:
    evicted_key, _ = cache.popitem(last=False)
    _data_cache_meta.pop(evicted_key, None)
    logger.info(f"LRU eviction: {evicted_key}")


def _load_static_grid_arrays(data_dir: str, model: str, logger) -> Dict[str, Any]:
    cache_key = f"{data_dir}|{model}"
    with _static_grid_lock:
        cached = _static_grid_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        out = read_static_arrays(data_dir=data_dir, model=model, logger=logger)
        if out:
            with _static_grid_lock:
                _static_grid_cache[cache_key] = out
        return out
    except Exception as exc:
        logger.warning(f"Static grid load failed for {model}: {exc}")
        return {}


def _maybe_attach_static_grid(arrays: Dict[str, Any], data_dir: str, model: str, logger) -> Dict[str, Any]:
    missing = [k for k in STATIC_GRID_KEYS if k not in arrays]
    if not missing:
        return arrays
    static_arrays = _load_static_grid_arrays(data_dir, model, logger)
    if not static_arrays:
        return arrays
    out = dict(arrays)
    for key in missing:
        if key in static_arrays:
            out[key] = static_arrays[key]
    return out


def _apply_substep_aliases(arrays: Dict[str, Any], model: str, step: int, substep_minutes: int, keys: Optional[List[str]]):
    if model != "icon_d2" or substep_minutes <= 0:
        return arrays

    out = dict(arrays)
    requested = set(keys or [])
    touched = False
    for key in SUBSTEP_SUPPORTED_VARS:
        sub_key = f"{key}_substeps"
        mins_key = f"{key}_substep_minutes"
        if requested and key not in requested and sub_key not in requested:
            continue
        if sub_key not in arrays or mins_key not in arrays:
            continue
        minutes = [int(x) for x in np.asarray(arrays[mins_key]).tolist()]
        if substep_minutes not in minutes:
            continue
        idx = minutes.index(substep_minutes)
        out[key] = np.asarray(arrays[sub_key][idx], dtype=np.float32)
        touched = True

    if touched and isinstance(out.get("validTime"), str):
        try:
            dt = datetime.fromisoformat(out["validTime"].replace("Z", ""))
            out["validTime"] = (dt + timedelta(minutes=substep_minutes)).isoformat() + "Z"
            out["_substepMinutes"] = substep_minutes
        except Exception:
            out["_substepMinutes"] = substep_minutes
    return out


def load_step_data(
    *,
    data_dir: str,
    model: str,
    run: str,
    step: int,
    cache: Dict[str, Any],
    cache_max_items: int,
    keys: Optional[List[str]],
    logger,
    substep_minutes: int = 0,
) -> Dict[str, Any]:
    """Load step data with selective-key support, LRU + singleflight."""
    cache_key = f"{model}/{run}/{step:03d}"

    if cache_key in cache:
        cached = cache[cache_key]
        if _cache_has_requested_keys(
            cache_key=cache_key,
            cached=cached,
            keys=keys,
            model=model,
            substep_minutes=substep_minutes,
        ):
            cache.move_to_end(cache_key)
            logger.debug(f"Cache hit: {cache_key}")
            return _apply_substep_aliases(cached, model, step, substep_minutes, keys)

    # Singleflight: coord concurrent partial/full misses
    owner = False
    evt: threading.Event | None = None
    with _data_inflight_lock:
        evt = _data_inflight.get(cache_key)
        if evt is None:
            evt = threading.Event()
            _data_inflight[cache_key] = evt
            owner = True

    if owner:
        try:
            logger.debug(f"Owner load: {cache_key}" + (f" (keys: {len(keys)})" if keys else " (all)"))

            if keys is not None:
                load_keys = set(keys) | {"lat", "lon"}
                if substep_minutes > 0 and model == "icon_d2":
                    for key in list(load_keys):
                        if key in SUBSTEP_SUPPORTED_VARS:
                            load_keys.add(f"{key}_substeps")
                            load_keys.add(f"{key}_substep_minutes")
                arrays = read_step_arrays(
                    data_dir=data_dir, model=model, run=run, step=step,
                    keys=load_keys, logger=logger,
                )
                if cache_key in cache:
                    for k, v in cache[cache_key].items():
                        if k not in arrays:
                            arrays[k] = v  # atomic merge under lock
            else:
                arrays = read_step_arrays(
                    data_dir=data_dir, model=model, run=run, step=step,
                    keys=None, logger=logger,
                )

            arrays = _maybe_attach_static_grid(arrays, data_dir, model, logger)

            run_dt = datetime.strptime(run, "%Y%m%d%H")
            valid_dt = run_dt + timedelta(hours=step)
            arrays["validTime"] = valid_dt.isoformat() + "Z"
            arrays["_run"] = run
            arrays["_step"] = step
            if "lat" in arrays and "lon" in arrays:
                try:
                    arrays["_latMin"] = float(np.min(arrays["lat"]))
                    arrays["_latMax"] = float(np.max(arrays["lat"]))
                    arrays["_lonMin"] = float(np.min(arrays["lon"]))
                    arrays["_lonMax"] = float(np.max(arrays["lon"]))
                except Exception:
                    pass

            if len(cache) >= cache_max_items:
                _evict_cache_item(cache, logger)
            cache[cache_key] = arrays
            _remember_cache_meta(cache_key, arrays, full=(keys is None))
            cache.move_to_end(cache_key)
            return _apply_substep_aliases(arrays, model, step, substep_minutes, keys)
        finally:
            with _data_inflight_lock:
                _data_inflight.pop(cache_key, None)
                if evt:
                    evt.set()
    else:
        logger.debug(f"Singleflight wait: {cache_key}")
        evt.wait(timeout=30.0)  # NPZ timeout

        # Post-wait recheck (owner may have filled)
        if cache_key in cache:
            cached = cache[cache_key]
            if _cache_has_requested_keys(
                cache_key=cache_key,
                cached=cached,
                keys=keys,
                model=model,
                substep_minutes=substep_minutes,
            ):
                cache.move_to_end(cache_key)
                logger.debug(f"Singleflight hit: {cache_key}")
                return _apply_substep_aliases(cached, model, step, substep_minutes, keys)

        # Fallback: owner crashed/partial fail
        logger.warning(f"Singleflight fallback: {cache_key}")
        logger.debug(f"Fallback load: {cache_key}")

        if keys is not None:
            load_keys = set(keys) | {"lat", "lon"}
            if substep_minutes > 0 and model == "icon_d2":
                for key in list(load_keys):
                    if key in SUBSTEP_SUPPORTED_VARS:
                        load_keys.add(f"{key}_substeps")
                        load_keys.add(f"{key}_substep_minutes")
            arrays = read_step_arrays(
                data_dir=data_dir, model=model, run=run, step=step,
                keys=load_keys, logger=logger,
            )
            if cache_key in cache:
                for k, v in cache[cache_key].items():
                    if k not in arrays:
                        arrays[k] = v
        else:
            arrays = read_step_arrays(
                data_dir=data_dir, model=model, run=run, step=step,
                keys=None, logger=logger,
            )

        arrays = _maybe_attach_static_grid(arrays, data_dir, model, logger)

        run_dt = datetime.strptime(run, "%Y%m%d%H")
        valid_dt = run_dt + timedelta(hours=step)
        arrays["validTime"] = valid_dt.isoformat() + "Z"
        arrays["_run"] = run
        arrays["_step"] = step
        if "lat" in arrays and "lon" in arrays:
            try:
                arrays["_latMin"] = float(np.min(arrays["lat"]))
                arrays["_latMax"] = float(np.max(arrays["lat"]))
                arrays["_lonMin"] = float(np.min(arrays["lon"]))
                arrays["_lonMax"] = float(np.max(arrays["lon"]))
            except Exception:
                pass

        if len(cache) >= cache_max_items:
            _evict_cache_item(cache, logger)
        cache[cache_key] = arrays
        _remember_cache_meta(cache_key, arrays, full=(keys is None))
        cache.move_to_end(cache_key)
        return _apply_substep_aliases(arrays, model, step, substep_minutes, keys)
