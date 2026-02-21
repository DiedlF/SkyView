from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import threading


_data_inflight: Dict[str, threading.Event] = {}
_data_inflight_lock = threading.Lock()

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
) -> Dict[str, Any]:
    """Load .npz for model/run/step with selective-key support, LRU + singleflight."""
    cache_key = f"{model}/{run}/{step:03d}"

    if cache_key in cache:
        cached = cache[cache_key]
        if keys is None or all(k in cached for k in keys):
            cache.move_to_end(cache_key)
            logger.debug(f"Cache hit: {cache_key}")
            return cached

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
            model_dir = model.replace("_", "-")
            path = os.path.join(data_dir, model_dir, run, f"{step:03d}.npz")
            if not os.path.exists(path):
                logger.error(f"Data not found: {path}")
                raise FileNotFoundError(f"Data not found: {path}")

            logger.debug(f"Owner load: {cache_key}" + (f" (keys: {len(keys)})" if keys else " (all)"))
            npz = np.load(path)

            if keys is not None:
                load_keys = set(keys) | {"lat", "lon"}
                arrays: Dict[str, Any] = {k: npz[k] for k in load_keys if k in npz.files}
                if cache_key in cache:
                    for k, v in cache[cache_key].items():
                        if k not in arrays:
                            arrays[k] = v  # atomic merge under lock
            else:
                arrays = {k: npz[k] for k in npz.files}

            run_dt = datetime.strptime(run, "%Y%m%d%H")
            valid_dt = run_dt + timedelta(hours=step)
            arrays["validTime"] = valid_dt.isoformat() + "Z"
            arrays["_run"] = run
            arrays["_step"] = step

            if len(cache) >= cache_max_items:
                evicted_key, _ = cache.popitem(last=False)
                logger.info(f"LRU eviction: {evicted_key}")
            cache[cache_key] = arrays
            cache.move_to_end(cache_key)
            return arrays
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
            if keys is None or all(k in cached for k in keys):
                cache.move_to_end(cache_key)
                logger.debug(f"Singleflight hit: {cache_key}")
                return cached

        # Fallback: owner crashed/partial fail
        logger.warning(f"Singleflight fallback: {cache_key}")
        # Repeat load (duplicated for simplicity; refactor to _do_load if needed)
        model_dir = model.replace("_", "-")
        path = os.path.join(data_dir, model_dir, run, f"{step:03d}.npz")
        if not os.path.exists(path):
            logger.error(f"Data not found: {path}")
            raise FileNotFoundError(f"Data not found: {path}")

        logger.debug(f"Fallback load: {cache_key}")
        npz = np.load(path)

        if keys is not None:
            load_keys = set(keys) | {"lat", "lon"}
            arrays = {k: npz[k] for k in load_keys if k in npz.files}
            if cache_key in cache:
                for k, v in cache[cache_key].items():
                    if k not in arrays:
                        arrays[k] = v
        else:
            arrays = {k: npz[k] for k in npz.files}

        run_dt = datetime.strptime(run, "%Y%m%d%H")
        valid_dt = run_dt + timedelta(hours=step)
        arrays["validTime"] = valid_dt.isoformat() + "Z"
        arrays["_run"] = run
        arrays["_step"] = step

        if len(cache) >= cache_max_items:
            evicted_key, _ = cache.popitem(last=False)
            logger.info(f"LRU eviction: {evicted_key}")
        cache[cache_key] = arrays
        cache.move_to_end(cache_key)
        return arrays
