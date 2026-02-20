# Skyview TODO

**Last updated:** 2026-02-20 (session 2)
**State:** ✅ Core stable. P1 (arch/cache) complete. P2 (cache correctness) complete. P3 (quality gates) in progress. Admin endpoint hardened.

---

## On Hold

### OpenAir overlay (deferred mini-project)
- Backend parser/index for OpenAir geometry
- Overlay API endpoint + bbox filtering
- Frontend layer toggle + styling by class/type
- Performance + QA + docs

---

## Open Tasks (priority order)

### A) Backend / Maintainability

- [x] **Marker auth module** — extracted to `backend/marker_auth.py` (make_token/verify_token/startup_check). app.py thin wrappers; startup warning on weak/missing secret. Tests: `tests/test_marker_auth.py` (17 cases). (Arch #8)
- [~] **Per-cell loop vectorization** — `aggregate_symbol_cell` is already fully vectorized *within* each cell (NumPy on full cell arrays). Across-cell vectorization blocked by per-cell EU/D2 source switching. Fast-path exists (zoom ≤ 9 stride sampling). Further deferred. (Arch #6)
- [ ] **EU fallback helpers consolidation** — overlay/tile both have inline EU load+gate logic. Extract shared `_load_eu_for_tile(time, cfg, tile_bounds)` helper.
- [ ] **Computed cache tune** — eviction policy for active timestep/layer churn (Phase 2 overlay perf).
- [ ] **Tile pre-render warmup** — optional ring of viewport tiles on context switch (Phase 3).
- [ ] **Quantized storage** — optional quantize heavy overlay fields (precip rates); persist scale/offset (Phase 4, medium effort 2–3d).
- [ ] **Multi-worker docs** — document process-local metric limits; defer fix to later if Redis not needed.
- [ ] **Legacy precip fallback** — optional runtime fallback for missing precomputed precip in old runs.

### B) Ops / Release Hygiene (P3)

- [ ] **`test_symbol_zoom_continuity` threshold** — 3 cells at z8→z9 give 98.68% vs 99.5% threshold. Investigate whether these are legitimately ambiguous border cells or a real rendering inconsistency.
- [ ] **GitHub push** — repo not yet pushed to remote.
- [ ] **CI pipeline** (PR9) — lint/type/pytest + qa_smoke/qa_regression/qa_contract/qa_perf workflows.
- [x] **Pytest migration** (PR8) — `tests/test_smoke.py`, `test_regression.py`, `test_contract.py`, `test_perf.py`. Unit tests (no server) run always; integration/perf marked + skipped in fast CI. `pytest.ini` configured. 20 unit tests pass. (Arch #11)
- [ ] **Marker secret startup policy** — warn/error at startup when `SKYVIEW_MARKER_AUTH_SECRET` missing/weak; add rotation notes to ops docs. (PR10)
- [ ] **CORS production** — doc that `SKYVIEW_CORS_ORIGINS` must be set to real hostname before public deploy; default already safe (localhost allowlist). (PR11)

### C) Data / Model Harmonization (ICON-EU ↔ D2)

- [ ] Finalize variable mapping/normalization rules for EU↔D2:
  - [ ] Precipitation fields (semantics differ)
  - [ ] `hbas_sc`/`htop_sc` (proxy, non-1:1 confirmed)
  - [ ] `lpi` (proxy, non-1:1 confirmed)
- [ ] Codify parity impacts on symbols/overlays in docs

### D) UX / Frontend

- [ ] **Hover tooltip** — change point tooltip to hover-based overlay value display
- [ ] **Desktop/mobile verify** — test interaction model and fallback behavior across device types
- [ ] **Precipitation toggle** — option to show only convective or grid-scale precip
- [ ] **Meteogram** — full meteogram functionality (larger feature)

### E) Soaring Model

- [ ] Gliding potential flying distance:
  - [ ] Per-hour potential distance metric
  - [ ] Daily cumulative potential distance metric

### F) Notifications

- [ ] Admin notification dispatch (Telegram / email)

### G) Overlay Perf — HTTP/Cache Delivery (Phase 5)

- [ ] Revisit `Cache-Control`/ETag policy for tile responses (PR7)
- [ ] Verify browser/CDN reuse for identical tile URLs
- [ ] Add hit telemetry split by client class

### H) Overlay Perf — Acceptance Criteria (Phase 5 gates)

- [ ] p95 `/api/overlay_tile` reduced ≥30% for precip layers (cold-burst scenario)
- [ ] p95 reduced ≥20% for non-precip overlays
- [ ] CPU peak during tile burst reduced (before/after documented)
- [ ] No visual regressions in overlay regression checks

---

## Completed ✅

### Architecture (PR1–PR6 + Arch Review)
- ✅ **app.py split** → routers/ (core, domain, weather, overlay, point, ops, admin) — Arch #1
- ✅ **GridContext** shared blend engine (`grid_aggregation.py`: build_grid_context/choose_cell_groups) — Arch #2
- ✅ **EU fallback gated** on tile/overlay bbox-vs-D2 domain check — Arch #3
- ✅ **Blocking calls** wrapped: Nominatim `to_thread`, DWD HEAD inside `_ingest_model_timings` (runs in thread pool), asyncio.sleep — Arch #4
- ✅ **DATA_CACHE_MAX_ITEMS=24** env-configurable in constants.py (was hardcoded 8) — Arch #5
- ✅ **AppState** consolidation — globals → structured AppState + DI — Arch #7
- ✅ **api_point selective keys** — uses POINT_KEYS filter, no full-variable load — Arch #9
- ✅ **data_cache singleflight** + key-merge hardening (PR5, services/data_loader.py) — Arch #10
- ✅ **constants.py** — all thresholds/cell_sizes centralized with rationale comments
- ✅ **Spurious Δ=0h fallback banner** — fixed in backend + frontend (Arch bug fix)
- ✅ services/model_select.py, services/data_loader.py, services/app_state.py

### Performance
- ✅ Overlay perf Phase 1: per-tile timing breakdown + status endpoint telemetry
- ✅ Overlay perf Phase 2: computed-field singleflight (`computed_cache_get_or_compute`)
- ✅ Overlay perf Phase 3: warmup on layer/time switch (guarded, rate-limited)
- ✅ Precip pipeline: vectorized LPI path, shared constants/mappings
- ✅ Wind pre-binning aligned to symbols strategy

### Frontend / UX
- ✅ Leaflet CDN SRI enabled
- ✅ Global unhandled error/rejection banner
- ✅ Help/onboarding modal (EN/DE localized)
- ✅ D2 boundary suppression for EU-only timesteps
- ✅ Symbol gridding verified (no lattice holes at viewport edges)
- ✅ `symbols.js _typeToWw` aligned to backend weather_codes.py

### Data / Ingest
- ✅ D2→EU fallback (strict temporal consistency, no nearby-timestep recovery)
- ✅ Ingest cleanup hardened (`shutil.rmtree`)
- ✅ D2 border from valid-cell edges (ingest-time precompute)
- ✅ Marker write-path race mitigated (markers_lock on POST/DELETE)
- ✅ Usage analytics module (`/api/usage_stats`, privacy-preserving)
- ✅ DWD variable comparison docs (hbas/htop/lpi proxy behavior)
- ✅ Classify.py: canonical scalar cloud classifier; cb/blue_thermal precedence fixed

### Admin / Ops
- ✅ Admin dashboard MVP (`/admin`): ingest health, fallback/cache/perf, feedback inbox, logs
- ✅ Status endpoint richer: widgets/tables, level filters, artifact drilldown
- ✅ Fallback stats persisted to `data/fallback_stats.json`
- ✅ `/api/status` fallback counters fixed — were overwritten by snapshot fields (`.update()` fix)
- ✅ `/api/status` ingestHealth now includes `missingStepNumbers[]` — exact missing steps, not just count
- ✅ `ingest.py --fill-missing` — ingests only absent steps for a run; defaults to full step range
- ✅ EU overlay gap fixed — hsurf NaN slice check replaces rectangular bbox margin (commit `04509e8`)

---

## Notes

- Keep help text in sync with backend thresholds when symbol logic changes.
- Multi-worker deployment unsafe (process-local state) — document, defer Redis until needed.
- `_eu_strict_cache` resets on restart — expected, not a bug.
- OpenAir: resume as separate mini-project with own QA checklist.
