"""Status/cache/perf payload assembly helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def build_status_payload(
    *,
    runs,
    merged,
    tile_cache_prune_fn,
    tile_cache_desktop,
    tile_cache_mobile,
    tile_cache_max_desktop,
    tile_cache_max_mobile,
    tile_cache_ttl,
    cache_stats,
    computed_field_cache,
    symbols_cache_stats,
    cache_context_stats,
    perf_recent,
    perf_totals,
    api_error_counters,
    fallback_stats,
):
    now = datetime.now(timezone.utc)
    latest_run_time = None
    if runs:
        try:
            latest_run_time = datetime.fromisoformat(runs[0]["runTime"].replace("Z", "+00:00"))
        except Exception:
            latest_run_time = None

    ingest = {
        "hasData": bool(runs),
        "runCount": len(runs),
        "latestRun": runs[0]["run"] if runs else None,
        "latestModel": runs[0]["model"] if runs else None,
        "latestRunTime": runs[0]["runTime"] if runs else None,
        "freshnessMinutes": round((now - latest_run_time).total_seconds() / 60.0, 1) if latest_run_time else None,
    }

    # Ingest health panel (latest run + expected/available/missing steps per model)
    expected_steps = {"icon_d2": 48, "icon_eu": 92}
    by_model = {}
    for m in ("icon_d2", "icon_eu"):
        latest_m = next((r for r in runs if r.get("model") == m), None)
        if latest_m is None:
            by_model[m] = {
                "hasRun": False,
                "latestRun": None,
                "latestRunTime": None,
                "availableSteps": 0,
                "expectedSteps": expected_steps[m],
                "missingSteps": expected_steps[m],
                "coverage": 0.0,
            }
            continue
        available = len(latest_m.get("steps", []))
        expected = expected_steps[m]
        missing = max(0, expected - available)
        by_model[m] = {
            "hasRun": True,
            "latestRun": latest_m.get("run"),
            "latestRunTime": latest_m.get("runTime"),
            "availableSteps": available,
            "expectedSteps": expected,
            "missingSteps": missing,
            "coverage": round((available / expected), 3) if expected else None,
        }

    tile_cache_prune_fn("desktop")
    tile_cache_prune_fn("mobile")

    recent = list(perf_recent)
    recent_n = len(recent)
    recent_avg_ms = (sum(r["ms"] for r in recent) / recent_n) if recent_n else None
    recent_hit_rate = (sum(r["hit"] for r in recent) / recent_n) if recent_n else None

    total_req = perf_totals["requests"]
    total_avg_ms = (perf_totals["totalMs"] / total_req) if total_req else None
    total_hit_rate = (perf_totals["hits"] / total_req) if total_req else None

    return {
        "ingest": ingest,
        "models": {
            "mergedPrimary": {
                "run": merged["run"] if merged else None,
                "model": merged["model"] if merged else None,
                "runTime": merged["runTime"] if merged else None,
                "stepCount": len(merged["steps"]) if merged else 0,
                "d2Run": merged.get("d2Run") if merged else None,
                "euRun": merged.get("euRun") if merged else None,
            }
        },
        "cache": {
            "tile": {
                "desktopItems": len(tile_cache_desktop),
                "desktopMax": tile_cache_max_desktop,
                "desktopFillRatio": (len(tile_cache_desktop) / tile_cache_max_desktop) if tile_cache_max_desktop else 0.0,
                "mobileItems": len(tile_cache_mobile),
                "mobileMax": tile_cache_max_mobile,
                "mobileFillRatio": (len(tile_cache_mobile) / tile_cache_max_mobile) if tile_cache_max_mobile else 0.0,
                "ttlSeconds": tile_cache_ttl,
                "metrics": cache_stats,
                "hitRate": (cache_stats.get("hits", 0) / (cache_stats.get("hits", 0) + cache_stats.get("misses", 0))) if (cache_stats.get("hits", 0) + cache_stats.get("misses", 0)) else None,
            },
            "computedField": {
                "items": len(computed_field_cache),
            },
            "symbols": symbols_cache_stats,
            "context": cache_context_stats,
        },
        "perf": {
            "recentWindow": {
                "size": recent_n,
                "maxSize": perf_recent.maxlen,
                "avgMs": recent_avg_ms,
                "hitRate": recent_hit_rate,
            },
            "totals": {
                "requests": total_req,
                "hits": perf_totals["hits"],
                "misses": perf_totals["misses"],
                "avgMs": total_avg_ms,
                "hitRate": total_hit_rate,
            },
        },
        "errors": {
            "http4xx": api_error_counters["4xx"],
            "http5xx": api_error_counters["5xx"],
        },
        "ingestHealth": {
            "models": by_model,
        },
        "fallback": {
            "euResolveAttempts": fallback_stats.get("euResolveAttempts", 0),
            "euResolveSuccess": fallback_stats.get("euResolveSuccess", 0),
            "strictTimeDenied": fallback_stats.get("strictTimeDenied", 0),
            "overlayFallback": fallback_stats.get("overlayFallback", 0),
            "overlayTileFallback": fallback_stats.get("overlayTileFallback", 0),
            "symbolsBlended": fallback_stats.get("symbolsBlended", 0),
            "windBlended": fallback_stats.get("windBlended", 0),
            "pointFallback": fallback_stats.get("pointFallback", 0),
        },
        "serverTime": now.isoformat().replace("+00:00", "Z"),
    }


def build_perf_payload(perf_recent, perf_totals):
    recent = list(perf_recent)
    recent_n = len(recent)
    if recent_n:
        recent_avg_ms = sum(r['ms'] for r in recent) / recent_n
        recent_hit_rate = sum(r['hit'] for r in recent) / recent_n
    else:
        recent_avg_ms = None
        recent_hit_rate = None

    total_req = perf_totals['requests']
    total_avg_ms = (perf_totals['totalMs'] / total_req) if total_req else None
    total_hit_rate = (perf_totals['hits'] / total_req) if total_req else None

    return {
        'recentWindow': {
            'size': recent_n,
            'maxSize': perf_recent.maxlen,
            'avgMs': recent_avg_ms,
            'hitRate': recent_hit_rate,
        },
        'totals': {
            'requests': total_req,
            'hits': perf_totals['hits'],
            'misses': perf_totals['misses'],
            'avgMs': total_avg_ms,
            'hitRate': total_hit_rate,
        }
    }
