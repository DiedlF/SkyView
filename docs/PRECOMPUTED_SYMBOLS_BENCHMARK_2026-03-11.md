# Precomputed Symbols Benchmark — 2026-03-11

## Goal

Measure whether low-zoom precomputed symbol bins are worth their operational cost on the VPS.

Costs under review:
- extra ingest time (~10 minutes)
- extra disk usage
- extra complexity in the low-zoom symbol path

## Method

Compared `/api/symbols` on the VPS in two modes:

1. **Precomputed ON**
   - low zoom (`z <= 9`) served via persisted precomputed bins
2. **Precomputed OFF**
   - low zoom served by normal fixed-grid computation path

Benchmark tool:
- `scripts/benchmark_symbols_precompute.py`

Comparison setup:
- production-like server on `127.0.0.1:8501`
- second temporary server on `127.0.0.1:8511`
- same data, same VPS, same request scenarios

## Scenarios

- `z5_wide_alps`
- `z7_alps`
- `z9_tirol`

Each scenario replays several pan positions and measures average / p95 request latency.

## Results

### z5_wide_alps
- **Precomputed ON:** avg `81.1 ms`, p95 `128.3 ms`
- **Precomputed OFF:** avg `57.6 ms`, p95 `59.5 ms`
- **Conclusion:** precomputed path is clearly worse here

### z7_alps
- **Precomputed ON:** avg `78.1 ms`, p95 `128.5 ms`
- **Precomputed OFF:** avg `77.4 ms`, p95 `140.0 ms`
- **Conclusion:** effectively no meaningful avg gain; p95 mixed

### z9_tirol
- **Precomputed ON:** avg `101.7 ms`, p95 `154.1 ms`
- **Precomputed OFF:** avg `99.1 ms`, p95 `162.7 ms`
- **Conclusion:** negligible difference

## Disk Usage

Persisted low-zoom symbol bins on VPS:
- **99,648 files**
- **4745.69 MB** total (~4.7 GB)

## Decision

### Current implementation is not worth keeping enabled by default.

Reasoning:
- no clear latency win overall
- z5 is materially worse with precomputed bins
- z7/z9 are basically a wash
- ingest cost is high
- disk cost is very high
- current storage format (many small JSON files) is especially inefficient

## Implemented Change

Low-zoom precomputed bins are now **opt-in** via:

```bash
SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS=1
```

Default is now **off**.

## Recommendation for Future Revisit

If we revisit symbol precomputation later, use a different persistence model:
- fewer larger files
- NPZ / msgpack / sqlite instead of many tiny JSON files
- maybe only selected zooms or selected timesteps

Do not re-enable current JSON-bin precompute path globally without a new benchmark.
