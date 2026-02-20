from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Callable

from constants import ICON_EU_STEP_3H_START


def resolve_eu_time_strict(
    *,
    time_str: str,
    max_delta_hours: float,
    max_delta_hours_3h: float,
    resolve_time_fn: Callable[[str, Optional[str]], tuple[str, int, str]],
    cache,
    cache_ttl_seconds: float,
    cache_max: int,
    fallback_stats: dict,
    logger,
):
    """Resolve EU run/step only if close enough to requested time.

    Stateless service function with explicit dependency injection.
    """
    t = (time_str or "").strip()
    cache_key = (t or "latest", float(max_delta_hours))
    cached = cache.get(cache_key)
    if cached is not None:
        cached_result, cached_ts = cached
        if (time.monotonic() - cached_ts) <= cache_ttl_seconds:
            cache.move_to_end(cache_key)
            return cached_result
        try:
            del cache[cache_key]
        except Exception:
            pass

    fallback_stats["euResolveAttempts"] += 1
    try:
        run_eu, step_eu, model_eu = resolve_time_fn(t or "latest", "icon_eu")
        if model_eu != "icon_eu":
            result = None
        else:
            run_dt = datetime.strptime(run_eu, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            vt = run_dt + timedelta(hours=step_eu)

            step_window = max_delta_hours_3h if int(step_eu) >= ICON_EU_STEP_3H_START else max_delta_hours
            effective_window = max(float(max_delta_hours), float(step_window))

            if t == "" or t.lower() == "latest":
                fallback_stats["euResolveSuccess"] += 1
                result = (run_eu, step_eu, model_eu)
            else:
                try:
                    target = datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                except Exception:
                    fallback_stats["strictTimeDenied"] += 1
                    logger.error(
                        "EU strict fallback denied: invalid requested timestamp time=%s (missing timestep within window %.2fh)",
                        t,
                        effective_window,
                    )
                    result = None
                else:
                    delta_h = abs((vt - target).total_seconds()) / 3600.0
                    if delta_h > effective_window:
                        fallback_stats["strictTimeDenied"] += 1
                        logger.error(
                            "EU strict fallback denied: missing timestep within window requested=%s strict=%s run=%s step=%s delta=%.2fh window=%.2fh",
                            target.isoformat().replace('+00:00', 'Z'),
                            vt.isoformat().replace('+00:00', 'Z'),
                            run_eu,
                            step_eu,
                            delta_h,
                            effective_window,
                        )
                        result = None
                    else:
                        fallback_stats["euResolveSuccess"] += 1
                        result = (run_eu, step_eu, model_eu)
    except Exception:
        result = None

    cache[cache_key] = (result, time.monotonic())
    cache.move_to_end(cache_key)
    while len(cache) > cache_max:
        cache.popitem(last=False)
    return result


def load_eu_data_strict(
    *,
    time_str: str,
    keys: list[str],
    max_delta_hours: float,
    resolve_eu_time_strict_fn: Callable[..., Optional[tuple[str, int, str]]],
    load_data_fn: Callable[[str, int, str, Optional[list[str]]], Dict[str, Any]],
    eu_missing_until_mono: float,
    eu_missing_backoff_seconds: float,
    logger,
):
    """Strict EU load with ingest-gap backoff; returns (payload, updated_missing_until)."""
    eu_strict = resolve_eu_time_strict_fn(time_str=time_str, max_delta_hours=max_delta_hours)
    if eu_strict is None:
        return None, eu_missing_until_mono

    run_eu, step_eu, model_eu = eu_strict
    now_mono = time.monotonic()
    if now_mono < eu_missing_until_mono:
        return {
            "run": run_eu,
            "step": step_eu,
            "model": model_eu,
            "data": None,
            "missing": True,
            "backoff": True,
        }, eu_missing_until_mono

    try:
        d_eu = load_data_fn(run_eu, step_eu, model_eu, keys=keys)
        return {"run": run_eu, "step": step_eu, "model": model_eu, "data": d_eu, "missing": False}, 0.0
    except FileNotFoundError:
        new_until = time.monotonic() + eu_missing_backoff_seconds
        logger.warning(
            "EU data missing from disk (ingest gap): time=%s run=%s step=%s (backoff %.0fs)",
            time_str, run_eu, step_eu, eu_missing_backoff_seconds,
        )
        return {"run": run_eu, "step": step_eu, "model": model_eu, "data": None, "missing": True}, new_until
    except Exception as e:
        logger.error("EU strict load error: time=%s run=%s step=%s: %s", time_str, run_eu, step_eu, e)
        return None, eu_missing_until_mono
