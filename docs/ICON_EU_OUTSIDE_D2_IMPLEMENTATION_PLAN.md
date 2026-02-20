# ICON-EU Outside ICON-D2 Domain — Implementation Status

**Date:** 2026-02-16  
**Status:** ✅ Implemented (Phases 1–3 complete)

## Objective
In Skyview, render:
- **ICON-D2** where D2 data exists (preferred, higher resolution)
- **ICON-EU** where D2 does not exist / is invalid

This is a spatial domain blend with deterministic D2 priority in overlap.

---

## Product Rules (implemented)

1. **Priority order by location/data**
   - D2 valid at location/cell → use D2
   - otherwise EU valid → use EU
   - otherwise empty/no data

2. **Overlap policy**
   - D2 wins in overlap.

3. **Seam policy**
   - Hard seam (deterministic), no feathering yet.

4. **Point transparency**
   - `/api/point` returns `sourceModel`.

5. **Temporal consistency**
   - EU fallback is only allowed when EU timestep is temporally close to requested time (strict guard).
   - Prevents large fallback time jumps.

---

## Delivered by Phase

### Phase 1 — overlays ✅
Endpoints:
- `/api/overlay`
- `/api/overlay_tile/{z}/{x}/{y}.png`

Delivered:
- EU fallback when D2 coverage is missing/invalid.
- Tile blending where EU fills pixels invalid/outside D2.
- `X-Source-Model` support (`icon_d2` / `blended` / `icon_eu` depending on path).
- cache-key blend context includes EU run/step.

### Phase 2 — symbols + wind ✅
Endpoints:
- `/api/symbols`
- `/api/wind`

Delivered:
- Per-cell source selection (D2 inside domain, EU outside).
- Per-cell EU fallback when D2 signal is missing/NaN.
- Response model reports `blended` when EU was used.

### Phase 3 — point transparency ✅
Endpoint:
- `/api/point`

Delivered:
- Domain/signal-aware fallback to EU when needed.
- `sourceModel` field included in response.
- JSON NaN guard added for `overlay_values` serialization safety.

---

## Ingestion policy update ✅

To support strict-time fallback without horizon gaps, ICON-EU ingestion policy was expanded:
- from: `49..78 hourly + 81..120 every 3h`
- to: **`1..78 hourly + 81..120 every 3h`**

This increases EU per-run volume by +48 steps (~+109% vs previous EU step count).

---

## Residual Risks / Follow-ups

1. **Hard seam visibility**
   - expected at D2 boundary in some layers.
   - optional enhancement: overlay-only feather band.

2. **Runtime counters reset on restart**
   - fallback counters in `/api/status` are process-local.
   - can be persisted later if long-term telemetry is needed.

3. **Variable availability asymmetry**
   - some fields may differ in quality/availability between models.
   - keep parity QA for key layers/symbol logic.

---

## QA/Validation

- Automated gates passed after implementation updates:
  - `scripts/qa_smoke.py`
  - `scripts/qa_regression.py`
  - `scripts/qa_contract.py`
- Boundary evidence and request examples are tracked in:
  - `docs/QA_BASELINE_2026-02-15.md`
