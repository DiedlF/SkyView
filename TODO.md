# Skyview TODO (Consolidated)

**Last updated:** 2026-02-17  
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

### 1) Security hardening (from code review)
- [ ] Restrict CORS origins for production deployment (`SKYVIEW_CORS_ORIGINS` to real hostnames; avoid `*` outside trusted dev)
- [ ] Add robust server-side rate-limiting for `/api/location_search` (not only process-local timing)
- [ ] Add result caching for Nominatim proxy responses
- [ ] Harden marker secret handling:
  - [ ] startup warning/error policy when secret missing/weak
  - [ ] optionally rotate/validate secret policy in ops docs

### 2) Correctness + drift prevention (from code review)
- [x] Remove logic duplication long-term:
  - [x] move canonical point classification helper into `classify.py`
  - [x] refactor app usage to avoid ad-hoc 1x1 wrappers
- [x] Add regression checks for cloud-type precedence (`blue_thermal` vs `cb`) in vectorized path
- [x] Harden `_resolve_eu_time_strict` handling for non-ISO time inputs (`latest`, malformed strings) with explicit branches/tests
- [x] Verify/align frontend `_typeToWw` mapping with backend `weather_codes.py`

### 3) Performance follow-ups (remaining)
- [x] Replace residual generic per-pixel fallback in `colorize_layer_vectorized` for any remaining layers
- [x] Boundary generation optimization (segment-heavy path → compact merged axis-aligned segments)
- [x] Reduce duplicate time-resolution work around EU strict fallback paths
- [x] Trim WW symbol mapping/preload to actually used codes only

### 4) Backend architecture / maintainability
- [ ] Continue splitting `backend/app.py` into routers/modules (`symbols`, `overlay`, `markers`, `wind`, `point`)
- [ ] Centralize meteorological thresholds / magic numbers in `constants.py` (+ rationale comments)
- [ ] De-duplicate shared `cell_sizes` definitions into one source of truth
- [ ] Consolidate repeated D2/EU fallback logic via shared helper(s)
- [ ] Add missing type hints on selected backend helpers

### 5) Runtime / platform best practices
- [x] Remove deprecated `@app.on_event("startup")` in favor of FastAPI lifespan handlers
- [x] Validate process-local counters strategy for multi-worker deployment (or export proper metrics)
- [x] Add SRI hashes for Leaflet CDN assets in frontend
- [x] Add frontend unhandled promise rejection user-facing error surface
- [x] Replace `rm -rf` usage in `ingest.py` with `shutil.rmtree()`

### 6) Data/model harmonization (ICON-EU ↔ ICON-D2)
- [ ] Align ICON-EU variable semantics/mapping to ICON-D2 for:
  - [ ] precipitation fields
  - [ ] `hbas_sc` / `htop_sc`
  - [ ] `lpi`
- [ ] Validate parity impacts on symbols/overlays and document final mapping rules

### 7) New overlays / weather layers
- [ ] Add temperature overlay
  - [ ] Research available model altitude levels/heights
  - [ ] Decide default level and selector UX (if multiple levels)
- [ ] Add 10m gust wind layer

### 8) Ceiling/visualization behavior
- [x] Show ceiling altitudes above 9900 m using max color (instead of suppressing)

### 9) Overlay interaction UX
- [ ] Change point tooltip behavior to hover-based overlay value info
- [ ] Verify desktop/mobile interaction model and fallback behavior

### 10) Soaring model quality research
- [ ] Research and improve thermal strength / climb rate parameterization
- [ ] Research and improve cloudbase (LCL) parameterization
- [ ] Research gliding potential flying distance metrics:
  - [ ] per-hour potential distance
  - [ ] daily cumulative potential distance

### 11) Feedback pipeline + admin
- [ ] Notification dispatch (Telegram/email)
- [ ] Minimal admin web UI for feedback/logs
- [ ] Filtering/search/export basics

### 12) Ingest/storage optimization
- [ ] Define safe per-variable quantization policy
- [ ] A/B benchmark (disk + ingest + API + parity)
- [ ] Roll out low-risk subset
- [ ] Rollback switch

### 13) Project ops / release hygiene
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
- Cb threshold harmonization (`lpi > 7`) across grid and point paths
- Overlay performance pass: LPI vectorized path + `/api/overlay` vectorized rendering
- Wind performance pass: pre-binning strategy aligned with symbols endpoint
- Ceiling visualization: values >9900 m now clamp to max color (not suppressed)
- Canonical scalar cloud classifier moved to `backend/classify.py`; app path now reuses it
- Regression coverage expanded: AGL suppression + blue_thermal-over-cb precedence + strict EU time input handling
- Frontend `symbols.js` `_typeToWw` mapping aligned with backend `weather_codes.py`
- Performance follow-ups completed: fallback colorizer path optimized, strict EU resolve memoized, WW symbol map trimmed
- Boundary segment payload reduced via merged axis-aligned edges (ingest cache + API fallback path)
- Runtime best-practices pass: startup moved to FastAPI lifespan; multi-worker caveat logged for process-local metrics
- Frontend hardening: Leaflet CDN SRI enabled + global unhandled error/rejection banner
- Ingest cleanup hardened: `shutil.rmtree()` replaces shell `rm -rf`

---

## Notes

- Keep help text synchronized with backend thresholds/rules whenever symbol logic changes.
- If OpenAir is resumed, treat it as a separate mini-project with its own implementation + QA checklist.
