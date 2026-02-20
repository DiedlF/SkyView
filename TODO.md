# Skyview TODO (Consolidated)

**Last updated:** 2026-02-18  
**State:** ✅ Core map/symbol/overlay workflow stable. ✅ D2/EU fallback + marker workflow + localization/help + LPI overlay delivered.

---

## On hold (explicit)

### OpenAir overlay
- Backend parser/index for OpenAir geometry
- Overlay API endpoint + bbox filtering
- Frontend layer toggle + styling by class/type
- Performance sanity pass
- QA + docs

---

## Active open tasks (priority order)

### 1) Security hardening
- ✅ Current Prio-1 security items completed (location_search rate limiting + Nominatim caching)

### 2) Backend architecture / maintainability
- [ ] Continue splitting `backend/app.py` into routers/modules (`symbols`, `overlay`, `markers`, `wind`, `point`)
- [x] Centralize meteorological thresholds / magic numbers in `constants.py` (+ rationale comments)
- [x] De-duplicate shared `cell_sizes` definitions into one source of truth
- [ ] Consolidate repeated D2/EU fallback logic via shared helper(s)
- [x] Add missing type hints on selected backend helpers
- [x] Refactor precip pipeline internals with shared constants/mappings (overlay + point + ingest semantics); deeper module split still open
- [x] Add context-level cache tuning for precomputed precip fields (tile burst warmup behavior)
- [ ] (Later) Optional runtime fallback for missing precomputed precip fields in legacy runs

### 3) Data/model harmonization (ICON-EU ↔ ICON-D2)
- [ ] Align ICON-EU variable semantics/mapping to ICON-D2 for:
  - [ ] precipitation fields
  - [ ] `hbas_sc` / `htop_sc` (comparison done; confirmed proxy/non-1:1 semantics)
  - [ ] `lpi` (comparison done; confirmed proxy/non-1:1 semantics)
- [ ] Finalize parity impacts on symbols/overlays and codify mapping/normalization rules

### 4) New overlays / weather layers
- ✅ Closed (already implemented)

### 5) Overlay interaction UX
- [ ] Change point tooltip behavior to hover-based overlay value info
- [ ] Verify desktop/mobile interaction model and fallback behavior
- [ ] Option to visualize only convective or grid-scale precipitation
- [x] Hide D2 boundary overlay when requested timestep is EU-only (no D2 data displayed)
- [x] Verify symbol gridding logic always covers all visible map cells (no gaps/omitted cells at viewport edges/zoom transitions)
  - Checked with fixed-timestep API sampling across zoom 6–10 and multiple bboxes: no lattice holes detected.
  - Follow-up (optional): add automated regression test for symbol lattice continuity at viewport edges.

### 5b) Meteogram
- [ ] Meteogram functionality

### 6) Soaring model quality research
- [x] Research and improve thermal strength / climb rate parameterization
- [x] Research and improve cloudbase (LCL) parameterization
- [ ] Research gliding potential flying distance metrics:
  - [ ] per-hour potential distance
  - [ ] daily cumulative potential distance

### 7) Feedback pipeline + admin
- [x] Unified admin/status dashboard MVP delivered (`/admin`):
  - ingest health + full-ingest freshness timing
  - ingest/storage visibility (+ tmp + running ingest)
  - fallback/cache/perf diagnostics
  - markers + usage stats
  - feedback inbox + status workflow + search
  - logs tail + baseline quick links
- [ ] Admin follow-ups:
  - [x] richer logs UX (level filters, ingest/regression artifact drilldown)
  - [x] status widgets/tables instead of raw JSON blocks
- [ ] Notification dispatch (Telegram/email)

### 8b) Overlay performance implementation plan (post-precip migration)

#### Phase 1 — Measure first (1–2 days, High confidence)
- [x] Add per-request timing breakdown for `/api/overlay_tile`:
  - [x] NPZ load time
  - [x] source/computed field lookup time
  - [x] colorization time
  - [x] PNG encode time
- [x] Expose phase timings in `/api/status` (rolling p50/p95)
- [x] Add counters for computed-cache hit/miss by layer
- **Expected impact:** observability baseline; identifies true bottleneck before deeper refactors.

#### Phase 2 — Concurrency & cache efficiency (1–2 days, High impact)
- [x] Implement singleflight for computed-field cache misses keyed by `(model,run,step,layer)`
- [x] Prevent duplicate concurrent full-grid computations under tile bursts
- [ ] Tune computed cache retention and eviction for active timestep/layer churn
- **Expected impact:** major cold-burst latency reduction, lower CPU spikes.

#### Phase 3 — Warmup strategy (1 day, Medium–High impact)
- [x] Add optional warmup on layer/time switch for active context:
  - [x] precompute full source field once
  - [ ] optionally pre-render small ring of viewport tiles
- [x] Guard with config flag + rate limits
- **Expected impact:** smoother UX on first interaction after timestep/layer changes.

#### Phase 4 — Storage/IO optimization (2–3 days, Medium impact)
- [ ] Add optional quantized storage for heavy overlay fields (starting with precip rates)
- [ ] Persist scale/offset metadata and decode path
- [ ] Compare compressed size + decode speed vs float32
- **Expected impact:** lower disk IO and memory bandwidth; better throughput under load.

#### Phase 5 — HTTP/cache delivery tuning (0.5–1 day, Medium impact)
- [ ] Revisit `Cache-Control`/ETag policy for tile responses
- [ ] Ensure browser/CDN reuse for identical tile URLs
- [ ] Add cache-hit telemetry for tile cache by client class
- **Expected impact:** fewer repeated backend renders for common pan/zoom patterns.

#### Acceptance criteria
- [ ] p95 `/api/overlay_tile` reduced by >=30% for precip layers in cold-burst scenario
- [ ] p95 reduced by >=20% for non-precip overlays
- [ ] CPU peak during initial tile burst reduced measurably (documented before/after)
- [ ] No visual regressions (color/units) in overlay regression checks

### 9) Project ops / release hygiene
- [ ] Restrict CORS origins for production deployment (`SKYVIEW_CORS_ORIGINS` to real hostnames; avoid `*` outside trusted dev)
- [ ] Harden marker secret handling:
  - [ ] startup warning/error policy when secret missing/weak
  - [ ] optionally rotate/validate secret policy in ops docs
- [ ] Push repository to GitHub
- [ ] Add CI pipeline:
  - [ ] formatting/lint/type checks
  - [ ] `qa_smoke.py`
  - [ ] `qa_regression.py`
  - [ ] `qa_contract.py`
  - [ ] `qa_perf.py`

---

## Recently completed (closed)

- D2→EU fallback phases (strict temporal consistency)
- `/api/status` fallback metrics + ingest health visibility
- D2 border from valid-cell edges (ingest-time precompute)
- Single-marker UX with default Geitau + place search + "use my position"
- Marker auth hardening with safe fallback behavior
- Marker write-path race mitigation (`POST` + `DELETE` now under marker lock)
- Help/onboarding modal with cloud symbol rendering + EN/DE localization
- Locale default handling (EN/DE) + localized layer labels
- Diagnostic LPI layer (legend + point value support)
- Non-convective symbol logic refinement (representative-cell method + cover thresholds)
- Convective symbols (`cb`, `cu_con`, `cu_hum`, `blue_thermal`) now suppressed when AGL < 300 m (using `hsurf` correction)
- Cb threshold harmonization (`lpi > 7`) across grid and point paths
- Overlay performance pass: LPI vectorized path + `/api/overlay` vectorized rendering
- Wind performance pass: pre-binning strategy aligned with symbols endpoint
- Ceiling visualization: values >9900 m clamp to max color; 0/null and no-ceiling sentinel stay transparent
- Canonical scalar cloud classifier moved to `backend/classify.py`; app path now reuses it
- Regression coverage expanded: AGL suppression + blue_thermal-over-cb precedence + strict EU time input handling
- Frontend `symbols.js` `_typeToWw` mapping aligned with backend `weather_codes.py`
- Performance follow-ups completed: fallback colorizer path optimized, strict EU resolve memoized, WW symbol map trimmed
- Boundary segment payload reduced via merged axis-aligned edges (ingest cache + API fallback path)
- Runtime best-practices pass: startup moved to FastAPI lifespan; multi-worker caveat logged for process-local metrics
- Frontend hardening: Leaflet CDN SRI enabled + global unhandled error/rejection banner
- Ingest cleanup hardened: `shutil.rmtree()` replaces shell `rm -rf`
- Usage analytics module added: `/api/usage_stats` with privacy-preserving daily visitors + marker engagement stats
- D2 boundary overlay suppression for EU-only merged timesteps implemented (`/api/d2_domain` returns empty segments in EU-only window)
- DWD variable comparison documentation expanded with measured D2↔EU proxy behavior (`hbas/htop`, `lpi`)

---

## Notes

- Keep help text synchronized with backend thresholds/rules whenever symbol logic changes.
- If OpenAir is resumed, treat it as a separate mini-project with its own implementation + QA checklist.
