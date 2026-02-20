from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


def load_step_data(
    *,
    data_dir: str,
    model: str,
    run: str,
    step: int,
    cache,
    cache_max_items: int,
    keys: Optional[List[str]],
    logger,
) -> Dict[str, Any]:
    """Load .npz for model/run/step with selective-key support and LRU cache."""
    cache_key = f"{model}/{run}/{step:03d}"

    if cache_key in cache:
        cache.move_to_end(cache_key)
        cached = cache[cache_key]
        if keys is None or all(k in cached for k in keys):
            logger.debug(f"Cache hit: {cache_key}")
            return cached

    model_dir = model.replace("_", "-")
    path = os.path.join(data_dir, model_dir, run, f"{step:03d}.npz")
    if not os.path.exists(path):
        logger.error(f"Data not found: {path}")
        raise FileNotFoundError(f"Data not found: {path}")

    logger.debug(f"Loading data: {cache_key}" + (f" (keys: {len(keys)})" if keys else " (all)"))
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
        logger.info(f"LRU Cache eviction: {evicted_key}")
    cache[cache_key] = arrays
    return arrays
