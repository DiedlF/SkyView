# Skyview TODO

**Last updated:** 2026-04-26
**State:** ✅ Core stable. P1 (arch/cache) complete. P2 (cache correctness) complete. P3 (quality gates) in progress. Admin auth shipped (HTTP Basic). Meteogram shipped. Several new overlays shipped (geopotential, wave, hourly-max cloud base, Gold climb-rate, 600 hPa wind, mh-in-MSL). ICON-D2 substep ingest extended through 48h. Next planned feature: 15-minute symbols (plan in `docs/SYMBOLS_15MIN_IMPLEMENTATION_PLAN_2026-04-13.md`).

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
- [ ] **`backend/app.py` is still too large** — move ops/admin/overlay/location handler bodies into router/service modules (continuation of PR1 split). (docs/README backlog #4)
- [ ] **Explorer split/deploy boundary** — do not split the Explorer frontend alone. Keep monorepo for now, but prepare Explorer as its own deployable app by extracting shared Skyview/Explorer API-contract helpers into a small shared module/package, adding configurable Explorer `API_BASE`, documenting Explorer service deployment, and revisiting permissive Explorer CORS before public hosting. Consider a separate repo only after the shared contract boundary is clean.
- [ ] **In-process caches need locking or encapsulation** — `cache_state.py` mutates shared `OrderedDict` caches and counters from request paths without consistent locks. (docs/README backlog #1)
- [ ] **Feedback storage hardening** — `feedback_ops.py` rewrites the whole JSON file in place; switch to temp-file + `os.replace()`, JSONL, or SQLite. (docs/README backlog #2)
- [ ] **Precip precalculation robustness** — check edge cases/robustness. Keep separate precip layers (convective/gridscale) in mind.
- [ ] **Phased ingest pipeline** — split ingest into priority phases so the app becomes usable as soon as convective fields land, instead of waiting for the full run. Suggested order: (1) convection basics (CAPE/CIN/LPI/hbas_sc/htop_sc/htop_dc → symbols' convective inputs), (2) remaining symbol inputs (cloud cover, ww, ceiling, mh, hsurf), (3) overlay-only fields (precip, wave, geopotential, wind levels), (4) meteogram/skew-T/nowcast extras. Each phase should write a "ready" marker so frontend/feature gates can light up incrementally; ingest health needs to report per-phase coverage.
- [ ] **Computed cache tune** — eviction policy for active timestep/layer churn (Phase 2 overlay perf).
- [ ] **Tile pre-render warmup** — optional ring of viewport tiles on context switch (Phase 3).
- [ ] **Quantized storage** — optional quantize heavy overlay fields (precip rates); persist scale/offset (Phase 4, medium effort 2–3d).
- [ ] **Multi-worker docs** — document process-local metric limits; defer fix to later if Redis not needed.
- [ ] **Legacy precip fallback** — optional runtime fallback for missing precomputed precip in old runs.

### B) Ops / Release Hygiene (P3)

- [x] **`test_symbol_zoom_continuity` threshold** — resolved.
- [x] **Precomputed symbols benchmark** — benchmarked on VPS with `scripts/benchmark_symbols_precompute.py`; current low-zoom JSON-bin precompute is now opt-in only (`SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS=1`) because measured gains were negligible and storage cost was ~4.7 GB. (`docs/PRECOMPUTED_SYMBOLS_BENCHMARK_2026-03-11.md`)
- [x] **GitHub push** — repo pushed to remote.
- [x] **CI pipeline** (PR9) — lint/type/pytest + qa_smoke/qa_regression/qa_contract/qa_perf workflows. Live: https://github.com/DiedlF/SkyView/actions
- [x] **GitHub Actions watch scripts** — `scripts/` helpers + `docs/github-actions-watch.md`.
- [x] **Pytest migration** (PR8) — `tests/test_smoke.py`, `test_regression.py`, `test_contract.py`, `test_perf.py`. Unit tests (no server) run always; integration/perf marked + skipped in fast CI. `pytest.ini` configured. 20 unit tests pass. (Arch #11)
- [x] **Marker secret startup policy** — ERROR log + stderr banner on missing secret; WARNING on weak secret; rotation notes in `docs/OPS_SECRETS.md`. (PR10)
- [ ] **CORS production** — doc that `SKYVIEW_CORS_ORIGINS` must be set to real hostname before public deploy; default already safe (localhost allowlist). (PR11)
- [ ] **CI integration coverage can silently skip** — add synthetic `.npz` fixtures + `TestClient` API tests so core endpoints are exercised without live DWD/Explorer. (docs/README backlog #3)
- [ ] **Local developer setup docs** — document `python3`, Ruff installation, test/lint/run commands, expected env vars. (docs/README backlog #8)
- [ ] **Root onboarding README** — short root `README.md` pointing to `SPEC.md`, `TODO.md`, `docs/README.md`. (docs/README backlog #9)

### C) Data / Model Harmonization (ICON-EU ↔ D2)

- [ ] Finalize variable mapping/normalization rules for EU↔D2:
  - [ ] Precipitation fields (semantics differ)
  - [ ] `hbas_sc`/`htop_sc` (proxy, non-1:1 confirmed)
  - [x] `lpi` — D2 switched to `lpi_max` (1h max) via `d2_variable_map`; EU uses `lpi_con_max`; both are time-window maxima. Re-validation recommended.
- [ ] Codify parity impacts on symbols/overlays in docs

### D) UX / Frontend

- [x] **Meteogram** — backend `/api/meteogram_point` (streaming with heartbeats, UTC times, grid-index reuse) + frontend overlay with temp/precip/wind charts and per-point cache. (commits `fe10e93`, `4e09c00`, `56b3fe5`, `22d3d13`, `0a660b6`)
- [ ] **Hover tooltip** — change point tooltip to hover-based overlay value display
- [ ] **Desktop/mobile verify** — test interaction model and fallback behavior across device types
- [ ] **Precipitation layer split** — expose precipitation as three distinct layers in the UI: convective only, grid-scale only, and total/cumulative (current behavior). Requires backend overlay handlers and computed-field paths to surface the component fields separately, plus frontend toggles, legend, and point-popup attribution. Supersedes the earlier "Precipitation toggle" item.
- [ ] **15-minute symbols** — implement Phase 1 of `docs/SYMBOLS_15MIN_IMPLEMENTATION_PLAN_2026-04-13.md`: add `substep` query to `/api/symbols`, thread `substep_minutes` through `compute_symbols_payload`, classify from live quarter-hour fields (not hourly maxima), separate symbol cache keys by substep, wire frontend so symbols/overlays/point popup share one substep control. EU stays hourly with diagnostics. (Plan only; `/api/symbols` currently has no `substep` param.)
- [ ] **Frontend/admin XSS pass** — broader review of `innerHTML` usage in admin/debug views (marker suggestions are safe via DOM rendering). (docs/README backlog #5)

### E) Soaring Model

- [ ] **Climbrate estimation** — add more temperature layers (bigger ingest); use `htop_dc` for z_upper.
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

### I) Admin / Auth (post-MVP polish)

- [x] **Admin HTTP Basic auth** — `backend/admin_auth.py`; `SKYVIEW_ADMIN_USER`/`SKYVIEW_ADMIN_PASSWORD`; admin auth prompt fix. (commits `94eb9e8`, `7b89dea`; `docs/OPS_SECRETS.md`)
- [ ] **Admin auth polish** — startup warning when admin creds unset, rate-limit failed admin auth attempts, improve admin UI auth messaging. (docs/README backlog #6)

### J) Marker / Location

- [x] **ICAO marker lookup** — exact 4-letter ICAO queries (`EDDM`, `LOWI`, `LSZH`, …) routed via `backend/airport_lookup.py`; results carry structured `icao` field. (commit `94eb9e8`; `d70ec65` map/marker search stabilization)
- [ ] **Airport/ICAO seed normalization** — give `openaip_seed.json` explicit `icao` fields, or generate an airport index from a maintained source. (docs/README backlog #7)

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
- ✅ **Native grid placement refactor** for symbols and wind (commit `78305b6`)

### Performance
- ✅ Overlay perf Phase 1: per-tile timing breakdown + status endpoint telemetry
- ✅ Overlay perf Phase 2: computed-field singleflight (`computed_cache_get_or_compute`)
- ✅ Overlay perf Phase 3: warmup on layer/time switch (guarded, rate-limited)
- ✅ Precip pipeline: vectorized LPI path, shared constants/mappings
- ✅ Wind pre-binning aligned to symbols strategy
- ✅ Meteogram endpoint: grid-index reuse, streaming with heartbeats, optimized loading

### Frontend / UX
- ✅ Leaflet CDN SRI enabled
- ✅ Global unhandled error/rejection banner
- ✅ Help/onboarding modal (EN/DE localized)
- ✅ D2 boundary suppression for EU-only timesteps
- ✅ Symbol gridding verified (no lattice holes at viewport edges)
- ✅ `symbols.js _typeToWw` aligned to backend weather_codes.py
- ✅ Meteogram overlay (temp / precip / wind charts, per-point cache)
- ✅ Map layer / marker search stabilization
- ✅ Marker UI tuning + admin auth prompt fix

### Overlays (new since 2026-02)
- ✅ **Geopotential overlay** added; scaling/colormap refined (commits `ce9eae7`, `28be999`, `ede3dd4`)
- ✅ **Wave overlays + 600 hPa wind** with contrast tuning and control cleanup (commits `c3be5d1`, `db4087a`, `3305984`, `c6d7e0f`)
- ✅ **Hourly-max cloud base overlay** (commit `4671388`)
- ✅ **Gold climb-rate overlay**: aligned with `htop_dc` and hourly-max CAPE; CAPE thresholds tuned (commits `faf8299`, `bb13bde`, `472cdc3`, `c3be5d1`'s siblings, `e768cbf`, `54a6095`, `9840417`, `d230b4d`, `251675f`)
- ✅ **`mh` overlay shown in MSL** (commit `8c7bdd8`); hbas-above-mh margin increased (`520d386`)
- ✅ Cumulus symbol cloud-cover gate relaxed; ww 0/1 treated as clear for stratiform (`e614b6f`, `1e67883`)
- ✅ EU symbol CIN slicing fixes + shape guards; CIN threshold constant (`15b4a35`, `485cdd9`, `2b18f42`, `b35b3f7`, `895b397`)
- ✅ Native wind/icon-eu symbol resolution fixes (`92edb57`, `0634a69`)
- ✅ Convective cloud precedence regression updated (`fc2812c`, `7d415e8`)

### Data / Ingest
- ✅ D2→EU fallback (strict temporal consistency, no nearby-timestep recovery)
- ✅ Ingest cleanup hardened (`shutil.rmtree`)
- ✅ D2 border from valid-cell edges (ingest-time precompute)
- ✅ Marker write-path race mitigated (markers_lock on POST/DELETE)
- ✅ Usage analytics module (`/api/usage_stats`, privacy-preserving)
- ✅ DWD variable comparison docs (hbas/htop/lpi proxy behavior)
- ✅ Classify.py: canonical scalar cloud classifier; cb/blue_thermal precedence fixed
- ✅ **ICON-D2 substep ingest extended through 48h** (commit `23d2ab9`)
- ✅ **EU symbol ingest input fix + regression coverage** (commit `1238e96`)
- ✅ Static grid bundle simplified to single file (commit `cf273d0`)

### Admin / Ops
- ✅ Admin dashboard MVP (`/admin`): ingest health, fallback/cache/perf, feedback inbox, logs
- ✅ **Admin HTTP Basic auth** (`backend/admin_auth.py`)
- ✅ Status endpoint richer: widgets/tables, level filters, artifact drilldown
- ✅ Fallback stats persisted to `data/fallback_stats.json`
- ✅ `/api/status` fallback counters fixed — were overwritten by snapshot fields (`.update()` fix)
- ✅ `/api/status` ingestHealth now includes `missingStepNumbers[]` — exact missing steps, not just count
- ✅ `ingest.py --fill-missing` — ingests only absent steps for a run; defaults to full step range
- ✅ EU overlay gap fixed — hsurf NaN slice check replaces rectangular bbox margin (commit `04509e8`)
- ✅ GitHub Actions watch scripts (`docs/github-actions-watch.md`)

### Markers / Location
- ✅ **ICAO marker lookup** (`backend/airport_lookup.py`)
- ✅ Map layer / marker search stabilization

---

## Notes

- Keep help text in sync with backend thresholds when symbol logic changes.
- Multi-worker deployment unsafe (process-local state) — document, defer Redis until needed.
- `_eu_strict_cache` resets on restart — expected, not a bug.
- OpenAir: resume as separate mini-project with own QA checklist.
- The `docs/README.md` "Remaining Findings / Improvement Backlog" section (2026-04 review) is the source of items A/B/D/I/J above marked "(docs/README backlog #N)".
- `docs/SYMBOLS_15MIN_IMPLEMENTATION_PLAN_2026-04-13.md` is the design doc for the next D-section feature.
