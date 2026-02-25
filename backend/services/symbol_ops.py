"""Symbol precompute I/O and cache-seeding helpers.

Extracted from app.py so routers/weather.py can import them directly without
creating a circular dependency on the full application module.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Optional

from constants import LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM


# ── Path helpers ─────────────────────────────────────────────────────────────

def _model_dir_name(model_used: str) -> str:
    return "icon-d2" if model_used == "icon_d2" else (
        "icon-eu" if model_used == "icon_eu" else model_used
    )


def symbols_precomputed_path(data_dir: str, model_used: str, run: str, step: int, zoom: int) -> str:
    return os.path.join(
        data_dir, _model_dir_name(model_used), run,
        f"_symbols_z{zoom}_{int(step):03d}.json",
    )


# ── Disk read / write ─────────────────────────────────────────────────────────

def load_symbols_precomputed(
    data_dir: str, model_used: str, run: str, step: int, zoom: int,
) -> Optional[dict]:
    path = symbols_precomputed_path(data_dir, model_used, run, step, zoom)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_symbols_precomputed(
    data_dir: str,
    model_used: str,
    run: str,
    step: int,
    zoom: int,
    payload: dict,
    logger,
) -> None:
    path = symbols_precomputed_path(data_dir, model_used, run, step, zoom)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Failed writing symbols precompute %s: %s", path, exc)


# ── Viewport filter ───────────────────────────────────────────────────────────

def filter_symbols_to_bbox(
    payload: dict,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
) -> dict:
    """Clip a global symbol payload to a viewport bbox, recomputing blend flags."""
    symbols = payload.get("symbols", [])
    filtered = [
        s for s in symbols
        if lat_min <= float(s.get("lat", 0.0)) <= lat_max
        and lon_min <= float(s.get("lon", 0.0)) <= lon_max
    ]
    out = dict(payload)
    out["symbols"] = filtered
    out["count"] = len(filtered)

    eu_cells = sum(1 for s in filtered if s.get("sourceModel") == "icon_eu")
    d2_cells = sum(1 for s in filtered if s.get("sourceModel") == "icon_d2")
    total = eu_cells + d2_cells
    eu_share = (eu_cells / total) if total else 0.0
    significant_blend = eu_cells >= 3 and eu_share >= 0.03

    primary_model = (
        "icon_eu" if str(payload.get("model", "")).lower().startswith("icon_eu") else "icon_d2"
    )
    if eu_cells > 0 and d2_cells == 0:
        out["model"] = "icon_eu"
        fb = "eu_only_in_viewport"
    elif eu_cells > 0 and d2_cells > 0 and significant_blend:
        out["model"] = "ICON-D2 + EU"
        fb = "blended_d2_eu"
    elif eu_cells > 0 and d2_cells > 0:
        out["model"] = primary_model
        fb = "primary_model_with_eu_assist"
    else:
        out["model"] = primary_model
        fb = "primary_model_only"

    diag = dict(out.get("diagnostics") or {})
    diag.update(
        fallbackDecision=fb,
        sourceModel=out["model"],
        euCells=eu_cells,
        d2Cells=d2_cells,
        euShare=round(eu_share, 4),
    )
    if "servedFrom" not in diag:
        diag["servedFrom"] = "cache"
    out["diagnostics"] = diag
    return out


# ── Startup cache seed ────────────────────────────────────────────────────────

def seed_symbols_cache_from_disk(
    data_dir: str,
    symbols_cache_set_fn: Callable[[str, dict], None],
    max_runs_per_model: int = 1,
) -> int:
    """Warm the in-memory symbols cache from precomputed disk files at startup."""
    loaded = 0
    for model_key, model_dir in (("icon_d2", "icon-d2"), ("icon_eu", "icon-eu")):
        run_root = os.path.join(data_dir, model_dir)
        if not os.path.isdir(run_root):
            continue
        runs = sorted(
            [d for d in os.listdir(run_root) if len(d) == 10 and d.isdigit()],
            reverse=True,
        )[:max_runs_per_model]
        for run in runs:
            run_dir = os.path.join(run_root, run)
            for zoom in range(5, LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM + 1):
                for p in glob.glob(os.path.join(run_dir, f"_symbols_z{zoom}_*.json")):
                    try:
                        step = int(os.path.basename(p).split("_")[-1].split(".")[0])
                        with open(p, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        key = f"{model_key}|{run}|{step}|z{zoom}|global"
                        symbols_cache_set_fn(key, payload)
                        loaded += 1
                    except Exception:
                        continue
    return loaded
