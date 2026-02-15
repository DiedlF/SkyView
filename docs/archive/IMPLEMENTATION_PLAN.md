# Skyview Implementation Plan

**Date:** 2026-02-15  
**Goal:** Consolidate architecture, improve reliability/performance, and unlock faster feature delivery.

> Consolidated master version: `docs/IMPROVEMENT_IDEAS_SUMMARY.md`
> This file is kept as the execution-detail companion.

---

## Phase 0 — Stabilize Baseline (2-3 days)

### Objectives
- Freeze current behavior
- Prevent regressions while refactoring

### Tasks
1. Define "golden" manual test routes (Geitau + border pan + D2/EU handover)
2. Add smoke checks for key endpoints:
   - `/api/health`
   - `/api/timesteps`
   - `/api/symbols`
   - `/api/overlay`
   - `/api/point`
3. Add startup guard to avoid multiple backend instances on same port

### Deliverables
- Stable baseline checklist
- Repeatable smoke script

---

## Phase 1 — Reliability & Test Automation (4-6 days)

### Objectives
- Convert fragile manual checks into automated tests

### Tasks
1. Integration tests (pytest or equivalent):
   - z12 symbol continuity (no missing rows in known areas)
   - border panning consistency
   - D2↔EU transition consistency
   - CAPE threshold behavior (>=50 rendering)
   - wind tooltip parity vs wind-layer selection
2. Add CI job for tests on PR/commit

### Deliverables
- Test suite for critical geospatial/rendering logic
- CI pass/fail guard for regressions

---

## Phase 2 — Backend Modularization (5-8 days)

### Objectives
- Reduce risk/complexity from monolithic backend

### Target module split
- `backend/api/symbols.py`
- `backend/api/overlays.py`
- `backend/api/point.py`
- `backend/core/aggregation.py`
- `backend/core/colormaps.py`
- `backend/core/cache.py`
- `backend/core/models.py`

### Tasks
1. Extract modules without behavior changes
2. Keep contract compatibility during migration
3. Add import-level unit tests for each module

### Deliverables
- Smaller maintainable modules
- Reduced blast radius for future changes

---

## Phase 3 — Explorer/Skyview API Convergence (4-7 days)

### Objectives
- One shared backend contract, minimal duplicate logic

### Tasks
1. Define unified schema:
   - timesteps payload
   - overlay request/metadata
   - point values payload/units
2. Implement adapter layer for Explorer and Skyview frontend specifics
3. Remove duplicate mapping code

### Deliverables
- Shared API contract doc
- Reduced code duplication

---

## Phase 4 — Performance Improvements (3-5 days)

### Objectives
- Better responsiveness under frequent pan/zoom/layer switching

### Tasks
1. Add tile prewarm (viewport + one ring)
2. Tune cache invalidation strategy by `(model, run, step, layer, params)`
3. Benchmark tile generation and optimize remaining hotspots
4. Optional reverse proxy cache (Nginx/Caddy) for tile endpoints

### Deliverables
- Faster perceived rendering
- Lower backend recompute pressure

---

## Phase 5 — Observability & Ops (2-4 days)

### Objectives
- Faster diagnosis and safer operations

### Tasks
1. Add `/api/status` aggregated endpoint
2. Add structured error IDs/correlation IDs
3. Add frontend lightweight health indicator for repeated failures

### Deliverables
- Better operational visibility
- Faster debugging in production-like use

---

## Phase 6 — Feature Expansion (ongoing)

### Priority feature queue
1. Airspace overlay (OpenAir)
2. Persistent custom markers + recommendation hints
3. Feedback notifications
4. Feedback/log admin view

---

## Suggested Execution Order

1. Phase 0 (baseline)
2. Phase 1 (tests)
3. Phase 2 (modularization)
4. Phase 3 (API convergence)
5. Phase 4 (performance)
6. Phase 5 (ops)
7. Phase 6 (features)

---

## Success Criteria

- No recurring symbol-row/edge artifacts
- D2↔EU transition passes automated checks
- Backend split into maintainable modules
- Explorer and Skyview share a stable contract
- Tile interactions feel smooth under normal usage
- Feature work can be added with low regression risk
