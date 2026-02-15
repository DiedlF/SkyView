# Skyview — Consolidated Improvement Roadmap

**Date:** 2026-02-15  
**Purpose:** Single consolidated document combining improvement ideas + implementation plan.

---

## 1) Strategic Priorities

### P0 — Reliability First
- Stabilize startup/runtime behavior (no multi-process ambiguity, clean restarts)
- Lock in regression protection for known fragile geospatial/rendering cases
- Keep current UX behavior stable while refactoring

### P1 — Consolidation & Performance
- Split backend monolith into maintainable modules
- Converge Explorer/Skyview API contracts and reduce duplicate logic
- Improve tile responsiveness via prewarm + stronger cache strategy

### P2 — Product Expansion
- Airspace overlays
- Persistent user markers/recommendations
- Feedback notifications/admin tooling

---

## 2) Consolidated Workstreams

### A. Reliability & Operations
1. Service startup hardening
   - single-instance guard / pre-start port checks
   - safer restart behavior (systemd guardrails where deployed)
2. Structured error visibility
   - correlation IDs + repeat-failure counters
   - frontend health indicator for repeated tile/API failures
3. Aggregated status endpoint
   - `/api/status`: ingest freshness, current runs/models, cache stats, perf stats, error counters

### B. Testing & Quality Gates
1. Baseline smoke checks
   - `/api/health`, `/api/timesteps`, `/api/symbols`, `/api/overlay`, `/api/point`
2. Automated integration tests for known risk zones
   - z12 symbol continuity / no missing-row artifacts
   - border-pan stability
   - D2↔EU transition continuity
   - CAPE threshold behavior (>=50 render; tooltip raw value)
   - wind tooltip parity vs selected wind level + zoom aggregation

### C. Architecture & Convergence
1. Backend modularization
   - split `backend/app.py` into symbols, overlays, point, cache, colormaps, aggregation modules
2. Unified Explorer/Skyview contract
   - common timestep schema
   - common point payload + units/format metadata
   - common overlay metadata semantics (`layer`/`var` alignment)
3. Centralized metadata/format registry
   - variable names, units, precision, descriptions, value formatting hints

### D. Performance & Scalability
1. Tile prewarm on interaction
   - pre-generate viewport + one ring on layer/timestep switches
2. Cache lifecycle strategy
   - deterministic invalidation by `(model, run, step, layer, params)`
   - memory boundaries + predictable TTL behavior
3. Rendering optimization
   - vectorize remaining expensive paths
   - optional reverse-proxy cache (Nginx/Caddy) in front of tile endpoints

### E. Product Features (next wave)
1. Airspace structure overlay (OpenAir)
2. Persistent custom markers + recommendation hints
3. Feedback notifications (Telegram/email)
4. Feedback/log admin web view

### F. Delivery & Governance
1. Push repository to GitHub
2. Add CI (lint + smoke + integration test subset)
3. Enforce quality gate before major feature merges

---

## 3) Execution Plan (Phased)

### Phase 0 (2–3 days) — Baseline Stabilization
- Freeze known-good behavior and add smoke scripts
- Add startup guard to prevent process/port conflicts

### Phase 1 (4–6 days) — Automated Reliability Tests
- Implement high-value integration tests for fragile map/overlay scenarios
- Wire into CI

### Phase 2 (5–8 days) — Backend Modularization
- Extract modules from monolith without changing external behavior
- Add module-level tests

### Phase 3 (4–7 days) — Explorer/Skyview Contract Convergence
- Shared schemas and adapters
- Remove duplicated mapping/format logic

### Phase 4 (3–5 days) — Performance Sprint
- Tile prewarm + cache improvements + profiling cleanup

### Phase 5 (2–4 days) — Observability & Ops
- `/api/status`, correlation IDs, frontend health indicator

### Phase 6 (ongoing) — Feature Expansion
- Airspace + markers + feedback ops features

---

## 4) Success Criteria

- No recurring symbol-row/border artifacts in regression suite
- Stable D2↔EU transition behavior across symbols/overlays/tooltips
- `backend/app.py` responsibilities modularized with lower regression risk
- Explorer and Skyview share a stable, documented API contract
- Faster perceived responsiveness after layer/time switches
- Feature additions can land without destabilizing core rendering logic

---

## 5) Practical Next Batch (recommended)

1. Startup hardening + smoke script
2. z12/border/D2-EU integration tests
3. Begin backend module extraction (overlays + colormaps first)
4. Define shared point/overlay contract doc for Explorer/Skyview
