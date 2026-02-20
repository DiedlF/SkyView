"""Tile/perf/computed cache state and helpers."""

from __future__ import annotations

import os
import time
import threading
from collections import OrderedDict, deque
from typing import Callable, Any

# Overlay tile cache (desktop/mobile split, LRU + TTL)
TILE_CACHE_MAX_ITEMS_DESKTOP = int(os.environ.get('SKYVIEW_TILE_CACHE_MAX_DESKTOP', '1400'))
TILE_CACHE_MAX_ITEMS_MOBILE = int(os.environ.get('SKYVIEW_TILE_CACHE_MAX_MOBILE', '700'))
TILE_CACHE_TTL_SECONDS = int(os.environ.get('SKYVIEW_TILE_CACHE_TTL_SECONDS', '900'))

tile_cache_desktop = OrderedDict()  # key -> (png_bytes, ts)
tile_cache_mobile = OrderedDict()
cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}

# Overlay tile performance telemetry
perf_recent = deque(maxlen=400)  # [{'ms':..., 'hit':0/1, 'ts':...}, ...
perf_totals = {
    'requests': 0,
    'hits': 0,
    'misses': 0,
    'totalMs': 0.0,
}

# Overlay phase telemetry (for /api/overlay_tile timing breakdown)
overlay_phase_recent = deque(maxlen=400)  # [{'layer':..., 'hit':..., 'loadMs':..., 'sourceMs':..., 'colorizeMs':..., 'encodeMs':..., 'totalMs':...}, ...]
overlay_phase_totals = {
    'requests': 0,
    'hits': 0,
    'misses': 0,
    'loadMs': 0.0,
    'sourceMs': 0.0,
    'colorizeMs': 0.0,
    'encodeMs': 0.0,
    'totalMs': 0.0,
}

# Computed full-field cache (for expensive derived layers used by tile endpoint)
# Slightly larger default improves warmup retention under layer/time churn.
COMPUTED_CACHE_MAX_ITEMS = int(os.environ.get('SKYVIEW_COMPUTED_CACHE_MAX_ITEMS', '192'))
COMPUTED_CACHE_TTL_SECONDS = int(os.environ.get('SKYVIEW_COMPUTED_CACHE_TTL_SECONDS', '2400'))
computed_field_cache = OrderedDict()  # key -> (np.ndarray, ts)
computed_field_metrics = {
    'hits': 0,
    'misses': 0,
    'singleflightWaits': 0,
    'byLayer': {},  # layer -> {'hits':..., 'misses':...}
}

# Singleflight coordination for computed field misses
_computed_inflight = {}  # key -> threading.Event
_computed_inflight_lock = threading.Lock()

# Symbols response cache (JSON payload cache for pan/zoom repeats)
SYMBOLS_CACHE_MAX_ITEMS = 256
SYMBOLS_CACHE_TTL_SECONDS = 60
symbols_cache = OrderedDict()  # key -> (payload_dict, ts)
symbols_cache_metrics = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}

# Run/step-aware cache invalidation state
cache_context = {"key": None, "rotations": 0}


def perf_record(ms: float, cache_hit: bool):
    now = time.time()
    perf_recent.append({'ms': float(ms), 'hit': 1 if cache_hit else 0, 'ts': now})
    perf_totals['requests'] += 1
    perf_totals['totalMs'] += float(ms)
    if cache_hit:
        perf_totals['hits'] += 1
    else:
        perf_totals['misses'] += 1


def overlay_phase_record(*, layer: str, cache_hit: bool, load_ms: float, source_ms: float, colorize_ms: float, encode_ms: float, total_ms: float):
    now = time.time()
    row = {
        'layer': layer,
        'hit': 1 if cache_hit else 0,
        'loadMs': float(load_ms),
        'sourceMs': float(source_ms),
        'colorizeMs': float(colorize_ms),
        'encodeMs': float(encode_ms),
        'totalMs': float(total_ms),
        'ts': now,
    }
    overlay_phase_recent.append(row)
    overlay_phase_totals['requests'] += 1
    overlay_phase_totals['totalMs'] += row['totalMs']
    overlay_phase_totals['loadMs'] += row['loadMs']
    overlay_phase_totals['sourceMs'] += row['sourceMs']
    overlay_phase_totals['colorizeMs'] += row['colorizeMs']
    overlay_phase_totals['encodeMs'] += row['encodeMs']
    if cache_hit:
        overlay_phase_totals['hits'] += 1
    else:
        overlay_phase_totals['misses'] += 1


def _computed_metrics_layer(layer: str):
    return computed_field_metrics['byLayer'].setdefault(layer, {'hits': 0, 'misses': 0})


def computed_cache_get(key: str, layer: str | None = None):
    now = time.time()
    item = computed_field_cache.get(key)
    if item is None:
        computed_field_metrics['misses'] += 1
        if layer is not None:
            _computed_metrics_layer(layer)['misses'] += 1
        return None
    arr, ts = item
    if now - ts > COMPUTED_CACHE_TTL_SECONDS:
        del computed_field_cache[key]
        computed_field_metrics['misses'] += 1
        if layer is not None:
            _computed_metrics_layer(layer)['misses'] += 1
        return None
    computed_field_cache.move_to_end(key)
    computed_field_metrics['hits'] += 1
    if layer is not None:
        _computed_metrics_layer(layer)['hits'] += 1
    return arr


def computed_cache_set(key: str, arr) -> None:
    now = time.time()
    computed_field_cache[key] = (arr, now)
    computed_field_cache.move_to_end(key)
    while len(computed_field_cache) > COMPUTED_CACHE_MAX_ITEMS:
        computed_field_cache.popitem(last=False)


def computed_cache_get_or_compute(key: str, layer: str, compute_fn: Callable[[], Any]):
    """Singleflight wrapper for computed full-grid fields.

    Only one caller computes a given key at a time; concurrent callers wait and reuse cache.
    """
    arr = computed_cache_get(key, layer=layer)
    if arr is not None:
        return arr

    owner = False
    evt = None
    with _computed_inflight_lock:
        evt = _computed_inflight.get(key)
        if evt is None:
            evt = threading.Event()
            _computed_inflight[key] = evt
            owner = True

    if owner:
        try:
            arr = compute_fn()
            computed_cache_set(key, arr)
            return arr
        finally:
            with _computed_inflight_lock:
                done_evt = _computed_inflight.pop(key, None)
                if done_evt is not None:
                    done_evt.set()

    computed_field_metrics['singleflightWaits'] += 1
    evt.wait(timeout=10.0)
    arr = computed_cache_get(key, layer=layer)
    if arr is not None:
        return arr

    # Fallback if owner failed before filling cache
    arr = compute_fn()
    computed_cache_set(key, arr)
    return arr


def _tile_cache_select(client_class: str):
    if client_class == "mobile":
        return tile_cache_mobile, TILE_CACHE_MAX_ITEMS_MOBILE
    return tile_cache_desktop, TILE_CACHE_MAX_ITEMS_DESKTOP


def tile_cache_prune(client_class: str):
    cache, max_items = _tile_cache_select(client_class)
    now = time.time()
    expired_keys = [k for k, (_v, ts) in cache.items() if now - ts > TILE_CACHE_TTL_SECONDS]
    for k in expired_keys:
        try:
            del cache[k]
        except KeyError:
            pass
        cache_stats["expired"] += 1
    while len(cache) > max_items:
        cache.popitem(last=False)
        cache_stats["evictions"] += 1


def tile_cache_get(client_class: str, key: str):
    cache, _ = _tile_cache_select(client_class)
    item = cache.get(key)
    if item is None:
        cache_stats["misses"] += 1
        return None
    png, ts = item
    now = time.time()
    if now - ts > TILE_CACHE_TTL_SECONDS:
        del cache[key]
        cache_stats["expired"] += 1
        cache_stats["misses"] += 1
        return None
    cache.move_to_end(key)
    cache_stats["hits"] += 1
    return png


def tile_cache_set(client_class: str, key: str, png: bytes):
    cache, max_items = _tile_cache_select(client_class)
    cache[key] = (png, time.time())
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)
        cache_stats["evictions"] += 1


def symbols_cache_get(key: str):
    now = time.time()
    item = symbols_cache.get(key)
    if item is None:
        symbols_cache_metrics["misses"] += 1
        return None
    payload, ts = item
    if now - ts > SYMBOLS_CACHE_TTL_SECONDS:
        del symbols_cache[key]
        symbols_cache_metrics["expired"] += 1
        symbols_cache_metrics["misses"] += 1
        return None
    symbols_cache.move_to_end(key)
    symbols_cache_metrics["hits"] += 1
    return payload


def symbols_cache_set(key: str, payload):
    symbols_cache[key] = (payload, time.time())
    symbols_cache.move_to_end(key)
    while len(symbols_cache) > SYMBOLS_CACHE_MAX_ITEMS:
        symbols_cache.popitem(last=False)
        symbols_cache_metrics["evictions"] += 1


def symbols_cache_stats_payload():
    total = symbols_cache_metrics["hits"] + symbols_cache_metrics["misses"]
    hit_rate = (symbols_cache_metrics["hits"] / total) if total else None
    return {
        "items": len(symbols_cache),
        "maxItems": SYMBOLS_CACHE_MAX_ITEMS,
        "ttlSeconds": SYMBOLS_CACHE_TTL_SECONDS,
        "fillRatio": (len(symbols_cache) / SYMBOLS_CACHE_MAX_ITEMS) if SYMBOLS_CACHE_MAX_ITEMS else 0.0,
        "metrics": symbols_cache_metrics,
        "hitRate": hit_rate,
    }


def rotate_caches_for_context(context_key: str):
    """Invalidate hot caches when model/run/step context changes."""
    if cache_context["key"] == context_key:
        return False
    cache_context["key"] = context_key
    cache_context["rotations"] += 1

    tile_cache_desktop.clear()
    tile_cache_mobile.clear()
    computed_field_cache.clear()
    symbols_cache.clear()
    return True


def computed_metrics_payload():
    return {
        'hits': computed_field_metrics['hits'],
        'misses': computed_field_metrics['misses'],
        'singleflightWaits': computed_field_metrics['singleflightWaits'],
        'byLayer': computed_field_metrics['byLayer'],
    }


def cache_context_stats_payload():
    return {
        "current": cache_context["key"],
        "rotations": cache_context["rotations"],
    }
