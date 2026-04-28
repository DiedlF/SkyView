# Performance Recommendations - 2026-04-26

## Scope

Static review of likely performance opportunities in the current Skyview codebase. These recommendations focus on interactive map performance: overlay tile bursts, pan/zoom reloads, frontend marker churn, and cold-load/static delivery.

These are code-informed suggestions, not benchmark results. Treat the first step as measurement.

## Guiding Assumptions

- The main user-facing performance risk is map interaction latency, especially after layer or timestep switches.
- The backend already has useful optimizations in place: selective NPZ loading, data-cache singleflight, computed-field cache singleflight, vectorized overlay colorization, tile cache, and optional overlay prewarm.
- The next wins should be surgical: improve cache placement, delivery, and hot-path measurement before changing meteorological logic.

## Recommended Order

1. Extend perf probes to cover the real hot paths.
2. Enable and validate reverse-proxy caching for overlay tiles.
3. Benchmark overlay tile PNG encoding options.
4. Tune frontend overlay prewarm behavior based on backend p95/CPU.
5. Reduce frontend Leaflet marker churn for symbols and wind.
6. Add targeted backend caches where repeated requests still recompute work.

## 1. Expand Performance Probes

Current `scripts/qa_perf.py` probes legacy `/api/overlay` and `/api/wind`, but not `/api/overlay_tile` or `/api/symbols`.

Add probes for:

- `/api/overlay_tile/{z}/{x}/{y}.png` cold tile burst.
- `/api/overlay_tile/{z}/{x}/{y}.png` warm repeat burst.
- `/api/symbols` repeated pan inside the same world-cell cache key.
- `/api/symbols` zoom transition near native/fixed-grid thresholds.
- Overlay layer/time switch scenario with visible tiles plus prewarm enabled.

Suggested measurements:

- avg and p95 latency
- cold vs warm split
- cache hit ratio
- `encodeMs`, `colorizeMs`, `sourceMs`, and `loadMs` from existing overlay telemetry
- CPU peak during a tile burst

Success criterion: every optimization below should show before/after evidence from these probes or from `/api/perf_stats`.

## 2. Put Tile Cache In Front Of Python

`/api/overlay_tile` currently resolves time, loads data, and performs EU fallback checks before consulting the in-process tile cache. A reverse proxy cache can short-circuit repeat tile URLs before FastAPI runs.

Use `docs/REVERSE_PROXY_TILE_CACHE.md` as the deployment guide.

Recommended validation:

- Request the same tile twice and verify proxy HIT on the second request.
- Switch timestep and verify the query string produces a distinct cache key.
- Confirm backend tile misses decrease under repeated pan/zoom.
- Keep max cache size and inactive TTL bounded.

Expected impact:

- Largest benefit for repeated pan/zoom and multiple users viewing the same layer/time.
- Lower Python CPU and lower pressure on in-process cache locks.

## 3. Benchmark Tile Encoding

Overlay tiles are encoded with PIL PNG output. The current `optimize=True` setting can improve byte size but may cost CPU during tile bursts.

Benchmark variants:

- PNG `optimize=True` current behavior
- PNG `optimize=False`
- PNG `compress_level=1`, `3`, and `6`
- Optional: WebP with alpha if browser and Leaflet delivery constraints are acceptable

Measure using existing `encodeMs` telemetry. Do not change the default until p95 tile latency and tile byte size are both compared.

Expected impact:

- Better p95 under cold bursts if encode time is a material part of tile latency.
- Possible bandwidth tradeoff if compression is reduced.

## 4. Make Overlay Prewarm Adaptive

The frontend prewarms viewport-plus-ring tiles after layer/time switches. This improves warm navigation, but it can add backend load at the same moment visible tiles are already loading.

Consider:

- Delay prewarm until visible tiles have settled.
- Lower mobile prewarm concurrency and tile count.
- Disable or shrink prewarm when recent tile p95 or error rate is high.
- Keep climb-rate/CAPE layers conservative because they are already burst-sensitive.

Success criterion:

- Visible tile p95 improves or stays flat.
- CPU peak during layer/time switch does not increase.
- Warm second-pan latency improves enough to justify the extra requests.

## 5. Reduce Frontend Marker Churn

Symbols and wind currently rebuild Leaflet markers after each successful response. This is straightforward, but it creates DOM/icon churn on pan, zoom, and timestep changes.

Options:

- Reuse symbol markers by stable coordinate/type keys.
- Reuse wind markers by stable coordinate/level keys.
- Move wind barbs to a canvas layer because they are non-interactive.
- Keep symbol markers interactive unless a canvas hit-test path is added.

Success criterion:

- Lower main-thread time during pan/zoom reloads.
- No regression in symbol click behavior.
- No visible marker flicker on mobile.

## 6. Add A Wind Response Cache

Symbols have a response cache keyed by stable world cells for fixed-grid modes. Wind recomputes aggregation for repeated viewport/level/time requests.

Add a small LRU cache keyed by:

```text
model|run|step|zoom|level|mode|world-cell-bbox
```

Include gust mode in the key. Keep the TTL short and invalidate on run/step context rotation.

Success criterion:

- Warm repeated `/api/wind` requests become cache hits.
- Wind p95 improves on repeated pan within the same cell bucket.
- Cache size remains bounded.

## 7. Cache Static Grid Arrays In Data Loader

`services/data_loader.py` attaches static grid keys such as `hsurf` from `grid/static.npz`. A per-model static-grid cache would avoid repeated disk reads during cold paths.

Recommendation:

- Cache static arrays by model name.
- Keep this cache tiny: one entry per model.
- Treat static grid arrays as immutable.

Success criterion:

- Fewer `static.npz` reads under cold tile/symbol bursts.
- No change in loaded payload semantics.

## 8. Keep Shared Cache State Thread-Safe

`cache_state.py` mutates `OrderedDict` caches and telemetry counters from request paths. This is listed in the docs backlog as a correctness issue, but it also affects performance under concurrent tile bursts.

Recommendation:

- Add narrow locks around each mutable cache family, or encapsulate cache state in small classes.
- Keep lock scope small: lookup/update only, not expensive compute.
- Preserve current singleflight behavior for computed fields.

Success criterion:

- No duplicate work caused by concurrent cache races.
- No measurable regression from lock contention under tile burst probes.

## Deferred Or Riskier Ideas

- Do not re-enable current low-zoom JSON symbol precompute globally. The VPS benchmark showed no clear win and high disk cost.
- Avoid broad vectorization rewrites of symbol classification unless perf probes show symbol aggregation is the dominant bottleneck.
- Avoid multi-worker cache redesign until deployment actually needs multiple workers; document process-local limits meanwhile.

## Verification Checklist

Before landing any implementation:

- Run targeted `qa_perf.py` probes with warmup and measured runs.
- Capture `/api/perf_stats` before and after.
- Verify visual output for overlays, symbols, and wind.
- Check mobile and desktop viewport behavior.
- Update `TODO.md` acceptance criteria if a recommendation becomes active work.
