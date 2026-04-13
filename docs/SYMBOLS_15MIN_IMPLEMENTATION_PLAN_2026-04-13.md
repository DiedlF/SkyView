# 15-Minute Symbols: Findings and Implementation Plan

Date: 2026-04-13

## Summary

Skyview already ingests quarter-hour substeps for the key ICON-D2 convective fields and already exposes them in `/api/point`. The missing piece is that `/api/symbols` is still hourly-only and does not accept or propagate a substep parameter.

This means the backend can already answer quarter-hour point queries, but the symbol layer cannot yet render symbols for `+15`, `+30`, or `+45` minutes.

## What already exists

### Ingest / storage

Quarter-hour substeps are already extracted during ingest for ICON-D2 in `backend/ingest.py`.

Current substep-capable variables:
- `cape_ml`
- `cin_ml`
- `hbas_sc`
- `htop_sc`
- `lpi`

For these variables, ingest stores:
- `<var>_substeps`
- `<var>_substep_minutes`
- for selected variables also `<var>_hourly_max`

### Data loading

`backend/services/data_loader.py` already supports:
- `load_data(..., substep_minutes=15|30|45)`
- swapping the base field (`cape_ml`, `cin_ml`, `hbas_sc`, `htop_sc`, `lpi`) to the requested quarter-hour slice
- updating `validTime` to the shifted timestamp

Important limitation:
- substep aliasing is currently applied only for `model == "icon_d2"`

### Existing API support

`/api/point` already supports:
- `substep` query param
- quarter-hour field loading
- quarter-hour `validTime`
- consistent point symbol classification from the requested substep values

There is also an existing nowcast-style point timeline path that reads quarter-hour series directly.

## What is missing for `/api/symbols`

### 1. No substep query parameter

`backend/routers/weather.py` `/api/symbols` currently exposes:
- `zoom`
- `bbox`
- `time`
- `model`

It does **not** expose `substep`.

### 2. Symbol compute path is hourly-only

`backend/services/symbol_compute.py`:
- does not accept `substep_minutes`
- calls `load_data(...)` without substep selection
- therefore always classifies from hourly fields

### 3. Current symbol logic prefers hourly-max convective fields

The symbol pipeline currently prefers hourly-max fields where available:
- `cape_ml_hourly_max`
- `hbas_sc_hourly_max`
- `htop_sc_hourly_max`

That is correct for current hourly symbol behavior, but it is **not** correct for true quarter-hour symbols.

If quarter-hour symbols are implemented, substep mode should classify from the actual quarter-hour fields instead of the hourly maxima.

### 4. Precomputed native symbol fields are hourly only

Ingest currently precomputes only hourly native symbol helper fields:
- `sym_code`
- `cb_hm`

There are no quarter-hour equivalents such as:
- `sym_code_substeps`
- `cb_hm_substeps`

This means substep symbols cannot rely on the current precomputed native symbol fast path unless that precompute layer is extended.

### 5. Cache keys are hourly only

Symbol cache keys currently use combinations like:
- `model|run|step|z...`

They do not include substep. If quarter-hour support is added, cache keys must include `substep` to avoid collisions between hourly and quarter-hour symbol payloads.

## Variables used by symbols and their temporal availability

### Variables used by current symbol logic

Used by `/api/symbols` / `compute_symbols_payload`:
- `ww`
- `ceiling`
- `clcl`
- `clcm`
- `clch`
- `cape_ml`
- `cin_ml`
- `htop_dc`
- `hbas_sc`
- `htop_sc`
- `lpi` / `lpi_max`
- `hsurf`
- `mh`
- optional precomputed helpers: `sym_code`, `cb_hm`

### Variables with quarter-hour support

Quarter-hour-capable now:
- `cape_ml`
- `cin_ml`
- `hbas_sc`
- `htop_sc`
- `lpi`

### Variables without quarter-hour support

Still hourly/base only:
- `ww`
- `ceiling`
- `clcl`
- `clcm`
- `clch`
- `htop_dc`
- `hsurf`
- `mh`
- `sym_code`
- `cb_hm`

## Product implication

A “15-minute symbol” would currently be:
- quarter-hour-aware for convective/instability-driven parts of classification
- still hourly for significant weather, cloud cover layers, ceiling, terrain/static helpers, and dry-convection-only inputs

So this is best described as:
- **quarter-hour convective symbol updates**

not yet:
- full quarter-hour all-weather symbol updates

## Recommended implementation approach

### Phase 1: ship quarter-hour symbols with on-the-fly classification

Recommended first version:
- add `substep` to `/api/symbols`
- thread `substep_minutes` through `compute_symbols_payload`
- in substep mode, classify from live quarter-hour fields
- do **not** rely on hourly precomputed `sym_code` / `cb_hm`
- use quarter-hour values directly, not hourly maxima
- separate symbol cache keys by substep
- wire frontend to pass the same substep to symbols and point popup

Why this is the best first step:
- minimal ingest/storage changes
- avoids expanding native symbol precompute complexity immediately
- easiest way to verify product behavior and semantics

### Phase 2: optional performance optimization

If substep symbols are too slow at native zooms, add quarter-hour native symbol precompute:
- `sym_code_substeps`
- `cb_hm_substeps`

Then extend loader aliasing to map those in substep mode.

## Concrete backend changes

### A. `backend/routers/weather.py`

For `/api/symbols`:
- add `substep: int = Query(0, ge=0, le=45)`
- normalize to `0, 15, 30, 45`
- pass normalized value into `compute_symbols_payload(...)`
- include substep in cache keys
- include substep diagnostics in the response

### B. `backend/services/symbol_compute.py`

- add `substep_minutes` parameter to `compute_symbols_payload`
- pass it to `load_data(...)`
- if `substep_minutes > 0`, prefer direct quarter-hour fields:
  - `cape_ml`
  - `cin_ml`
  - `hbas_sc`
  - `htop_sc`
  - `lpi`
- avoid using:
  - `cape_ml_hourly_max`
  - `hbas_sc_hourly_max`
  - `htop_sc_hourly_max`
  in substep mode
- in substep mode, bypass hourly native precomputed helpers (`sym_code`, `cb_hm`) unless dedicated substep versions exist

### C. `frontend`

- add substep parameter to symbol requests
- pass same substep to `/api/point` when clicking symbols
- ideally keep symbols, overlays, and point popup on one shared substep control

## Recommended UX behavior

Use one shared temporal offset control for:
- overlays
- symbols
- point popup

This avoids mismatches where map symbols show `+15` but the popup still queries the top-of-hour hourly step.

## Risks / caveats

1. **Semantic mismatch risk**
   Quarter-hour symbols will still be partly hourly because several symbol inputs do not have substeps.

2. **Performance risk**
   If native zooms classify everything on the fly in substep mode, CPU cost may rise.

3. **Cache fragmentation**
   Each step now potentially has four symbol variants: `0, 15, 30, 45`.

4. **EU model behavior**
   Quarter-hour aliasing is D2-only. For EU requests, nonzero substeps should either:
   - degrade cleanly to hourly behavior, or
   - be surfaced in diagnostics as not applied

## Recommendation

Implement the feature in two stages:

1. **Correctness-first**
   - add API + compute support
   - classify from live substep fields
   - keep ingest unchanged

2. **Performance second**
   - only if needed, add substep-native symbol precompute

That gives the fastest path to a working feature with the lowest architectural risk.
