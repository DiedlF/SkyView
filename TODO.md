# Skyview TODO

**Last updated:** 2026-05-07
**State:** ✅ Core stable. P1 (arch/cache) complete. P2 (cache correctness) complete. P3 (quality gates) in progress. Admin auth shipped (HTTP Basic). Meteogram shipped and enhanced with high-cloud background layers, terrain masking, and 10 m wind. 15-minute symbol support shipped. Recent overlays include geopotential, wave, hourly-max cloud base, Gold climb-rate, 600 hPa wind, mh-in-MSL, and convective/grid-scale precipitation split. ICON-D2 substep ingest extended through 48h.

---

## On Hold

### OpenAir overlay (deferred mini-project)
- Backend parser/index for OpenAir geometry
- Overlay API endpoint + bbox filtering
- Frontend layer toggle + styling by class/type
- Performance + QA + docs

### Observation layer — radar + satellite (in progress)
- Live EUMETNET OPERA radar (5-min dBZ composite) + EUMETSAT MSG RSS satellite overlays.
- Backend: `backend/observations/` ingest (fetch → temporary native decode → derived PNG render cache), 5 h ring-buffer retention, `/api/observations/frames`, `/api/observations/render/...`, and `/api/status` freshness.
- Frontend: "Observations" layer group with radar/HRV toggles, recent-frames strip + loop animation, and EUMETNET/EUMETSAT attribution.
- Full design + live-verification log: `docs/OBSERVATION_LAYER_IMPLEMENTATION_PLAN.md` (§10).
- **Status:** Derived radar dBZ + satellite HRV ingest, serving endpoints, frontend controls/loop, status freshness, systemd timer files, GitHub deploy wiring, and render-grid georeferencing are implemented and live-verified 2026-06-13. Both products render over a single window a bit wider than the ICON-EU/D2 crop (40–60°N, −8–26°E) and the serving layer advertises that same window as the frame bbox for both sources (regression guard: `tests/test_observations_api.py::test_served_bbox_matches_render_grid`).
  - **Frames are now reprojected to Web Mercator (EPSG:3857), not equirectangular.** The Leaflet basemap is Web Mercator and `L.imageOverlay` stretches the frame linearly in projected pixels; an equirectangular frame was displaced up to ~100 km N–S at mid latitudes (the visible mismatch against the basemap). `reproject.web_mercator_area_definition` is the shared render target for radar (`reproject_odim`) and satellite (`render_satellite_hrv_png`); the served bbox is unchanged because corners map 1:1. Verified against a Mercator graticule (Italian coast / Alps / 12°E meridian align). Remaining: post-deploy visual QA, legal page copy, backfill/refetch tooling, optional numeric-grid path.
  - **Per-image satellite navigation (researched 2026-06-13):** each SEVIRI L1.5 file carries its own `satellite_actual_longitude/latitude/altitude` (MSG-4 RSS is an *inclined* orbit, sub-satellite ≈9.5°E/2.2°N, varying per scan) and the steerable HRV `area_extent` shifts ~3 km E–W between scans. L1.5 is rectified to the nominal grid and we resample with each frame's own area, so the **ground is registered consistently**; residual frame-to-frame motion of features ("fixpoints moving") is **cloud parallax** from the off-nadir/inclined-orbit view — intrinsic to MSG, only removable with cloud-top-height parallax correction (not pursued).
  - **Deploy note:** the persistent `data/observations/` ring buffer can briefly mix pre-fix (equirectangular/old-window) and post-fix (Web Mercator) frames until old ones age out (≤5 h). Clear `data/observations/{radar,satellite}/` once on the deploy that ships this change to avoid showing stale-projection frames in the loop.
- **Known bugs to fix (confirmed live, see plan §10.2):**
  - [x] Stale satellite collection id — `EO:EUM:DAT:MSG:HRSEVIRI-RSS` 404s; use `EO:EUM:DAT:MSG:MSG15-RSS` (`backend/observations/config.py`, `backend/.env.example`).
  - [x] `SatelliteSource.fetch_latest` saves product as `.nc` but the Data Store delivers a `.zip` wrapping a `.nat` — unzip + read inner `.nat` (`backend/observations/satellite_eumetsat.py`).
  - [ ] `satpy` `Scene.resample()` dropped half the swath in testing; validate/replace `reproject.py:reproject_swath` with the cKDTree nearest path if/when numeric satellite grids return.
  - [x] Wire the satellite ingest branch for the selected derived HRV render cache (`backend/observations/ingest_obs.py`).
  - [x] Derived render georeferencing — radar now resamples from ODIM pixel-corner extents; HRV now uses Satpy native geometry + `Scene.resample()` onto the SkyView lat/lon render grid.
- **MTG-I1 (Meteosat-12) FCI comparison layer (added 2026-06-14):** a third observation source (`mtg`) renders FCI L1c HRFI `vis_06` (0.5 km visible, closest analogue to SEVIRI HRV) alongside MSG RSS for side-by-side comparison. Same EUMDAC/Data Store path; collection `EO:EUM:DAT:0665` (HRFI) by default, `EUCOMP_MTG_COLLECTION`/`EUCOMP_MTG_CHANNEL` overridable. New `backend/observations/satellite_mtg.py` (search/download split + `read_fci_valid_time`), `render_fci_vis_png` (satpy `fci_l1c_nc` reader, crop→Web-Mercator resample, shared grayscale stretch with HRV), `ingest_mtg`, router `mtg` source, and frontend toggle/pane. FD cadence is 10 min (context layer, not a 5-min loop match). **Cost guards (the full FCI stream is ~440 GB/day):** (1) `ingest_mtg` de-duplicates against the manifest using the search result's sensing time **before** any download, so the 2-min cron only fetches genuinely new cycles; (2) it downloads **only the northern (Europe) chunk entries** via `product.open(entry=...)`, not the full disk — `select_europe_chunks` keeps the top `chunk_fraction` (default 0.35 ≈ EUMETSAT "Q4" northern quarter + margin) of body chunks by *rank* plus the trailer. Rank-based (not absolute chunk numbers) on purpose: FDHSI FD has 40 body chunks but HRFI ~70, and "Q4=29–40" only holds for the 40-chunk layout; `EUCOMP_MTG_CHUNKS="29-40"` can pin an explicit range. Cron now runs `--source all` (radar+satellite+mtg). **Not yet live-verified against the real Data Store** — needs: FCI NRT licence accepted (`EO:EUM:DAT:0665`), `netCDF4` installed in `.venv-obs`, one real cycle ingested + visual QA (confirm the rendered frame covers 40–60°N — widen `EUCOMP_MTG_CHUNK_FRACTION` if the southern edge clips), and confirmation that `product.sensing_start`/`product.entries` are populated (else the `_OPE_` filename / full-`.nc` fallbacks apply).
- **Ops:** satellite downloads need the per-collection **NRT licence** accepted in the EUMETSAT Data Store (documented in `docs/OPS_SECRETS.md`). This now includes the FCI HRFI collection (`EO:EUM:DAT:0665`) for the MTG-I1 layer.

### Deferred performance/watchlist
- **Per-cell loop vectorization** — `aggregate_symbol_cell` is already vectorized within each cell (NumPy on full cell arrays). Across-cell vectorization is blocked by per-cell EU/D2 source switching. Fast-path exists (zoom ≤ 9 stride sampling). Further work is deferred unless profiling shows it matters. (Arch #6)

---

## Open Tasks (priority order)

### 0) Brand / Public Launch

- [ ] **Rename Skyview to D2View** — complete trademark/domain screening first (DPMA, EUIPO/TMview, WIPO, USPTO, domains/social handles), then update product name across frontend, legal pages, README/docs, GitHub metadata, deployment config, page titles, attribution wording, screenshots/assets, and any public URLs. Avoid implying DWD endorsement; keep DWD attribution as source/license language only.

### A) Backend / Maintainability

- [ ] **EU fallback helpers consolidation** — overlay/tile both have inline EU load+gate logic. Extract shared `_load_eu_for_tile(time, cfg, tile_bounds)` helper.
- [ ] **`backend/app.py` is still too large** — move ops/admin/overlay/location handler bodies into router/service modules (continuation of PR1 split).
- [ ] **Explorer split/deploy boundary** — do not split the Explorer frontend alone. Keep monorepo for now, but prepare Explorer as its own deployable app by extracting shared Skyview/Explorer API-contract helpers into a small shared module/package, adding configurable Explorer `API_BASE`, documenting Explorer service deployment, and revisiting permissive Explorer CORS before public hosting. Consider a separate repo only after the shared contract boundary is clean.
- [ ] **In-process caches need locking or encapsulation** — `cache_state.py` mutates shared `OrderedDict` caches and counters from request paths without consistent locks.
- [ ] **Feedback storage hardening** — `feedback_ops.py` rewrites the whole JSON file in place; switch to temp-file + `os.replace()`, JSONL, or SQLite.
- [ ] **Precip backfill/legacy robustness** — decide how old runs missing `convective_rate`/`gridscale_rate` should behave: explicit re-ingest/backfill requirement, optional runtime fallback, or status/admin warning.
- [ ] **Phased ingest pipeline** — split ingest into priority phases so the app becomes usable as soon as convective fields land, instead of waiting for the full run. Suggested order: (1) convection basics (CAPE/CIN/LPI/hbas_sc/htop_sc/htop_dc → symbols' convective inputs), (2) remaining symbol inputs (cloud cover, ww, ceiling, mh, hsurf), (3) overlay-only fields (precip, wave, geopotential, wind levels), (4) meteogram/skew-T/nowcast extras. Each phase should write a "ready" marker so frontend/feature gates can light up incrementally; ingest health needs to report per-phase coverage.
- [ ] **Computed cache tune** — eviction policy for active timestep/layer churn (Phase 2 overlay perf).
- [ ] **Quantized storage** — optional quantize heavy overlay fields (precip rates); persist scale/offset (Phase 4, medium effort 2–3d).
- [ ] **Multi-worker deployment guidance** — expand the existing process-local caveat into deployment docs; defer Redis/shared-store work until multi-worker hosting is actually needed.

### B) Ops / Release Hygiene (P3)

- [ ] **CI integration coverage can silently skip** — add synthetic `.npz` fixtures + `TestClient` API tests so core endpoints are exercised without live DWD/Explorer.
- [ ] **Local developer setup docs** — document `python3`, Ruff installation, test/lint/run commands, expected env vars.
- [ ] **SPEC/API docs drift** — align `SPEC.md` and the 15-minute symbols design note with current `/api/symbols`, overlay substep, and frontend behavior.

### C) Data / Model Harmonization (ICON-EU ↔ D2)

- [ ] Finalize remaining variable mapping/normalization rules for EU↔D2:
  - [ ] `hbas_sc`/`htop_sc` (proxy, non-1:1 confirmed)
- [ ] Re-validate LPI parity after the D2 `lpi_max` switch; EU uses `lpi_con_max`, and both are time-window maxima.
- [ ] Codify EU↔D2 parity impacts in docs, including precipitation semantics (`tot_prec`, convective/grid-scale split, D2 graupel inclusion), `hbas_sc`/`htop_sc`, and LPI.

### D) UX / Frontend

- [ ] **Hover tooltip** — change point tooltip to hover-based overlay value display.
- [ ] **Desktop/mobile verify** — test interaction model and fallback behavior across device types.
- [ ] **Frontend/admin XSS pass** — broader review of `innerHTML` usage in admin/debug views (marker suggestions are safe via DOM rendering).

### E) Soaring Model

- [ ] **Climbrate estimation** — add more temperature layers (bigger ingest); use `htop_dc` for z_upper.
- [ ] Gliding potential flying distance:
  - [ ] Per-hour potential distance metric
  - [ ] Daily cumulative potential distance metric

### F) Notifications

- [ ] Admin notification dispatch (Telegram / email)

### G) Overlay Perf — HTTP/Cache Delivery (Phase 5)

- [ ] **Performance recommendations follow-up** — use `docs/PERFORMANCE_RECOMMENDATIONS_2026-04-26.md` as the current execution note: expand perf probes for `/api/overlay_tile` and `/api/symbols`, validate reverse-proxy tile caching, benchmark PNG encoding options, tune overlay prewarm, reduce frontend marker churn, and consider wind/static-grid caches only with before/after evidence.
- [ ] Revisit `Cache-Control`/ETag policy for tile responses (PR7)
- [ ] Verify browser/CDN reuse for identical tile URLs
- [ ] Add hit telemetry split by client class

### H) Overlay Perf — Acceptance Criteria (Phase 5 gates)

- [ ] p95 `/api/overlay_tile` reduced ≥30% for precip layers (cold-burst scenario)
- [ ] p95 reduced ≥20% for non-precip overlays
- [ ] CPU peak during tile burst reduced (before/after documented)
- [ ] No visual regressions in overlay regression checks

### I) Admin / Auth (post-MVP polish)

- [ ] **Admin auth polish** — startup warning when admin creds unset, rate-limit failed admin auth attempts, improve admin UI auth messaging.

### J) Marker / Location

- [ ] **Airport/ICAO seed normalization** — give `openaip_seed.json` explicit `icao` fields, or generate an airport index from a maintained source.

---

## Completed ✅

### Architecture (PR1–PR6 + Arch Review)
- ✅ **app.py split** → routers/ (core, domain, weather, overlay, point, ops, admin) — Arch #1
- ✅ **GridContext** shared blend engine (`grid_aggregation.py`: build_grid_context/choose_cell_groups) — Arch #2
- ✅ **EU fallback gated** on tile/overlay bbox-vs-D2 domain check — Arch #3
- ✅ **Blocking calls** wrapped: Nominatim `to_thread`, DWD HEAD inside `_ingest_model_timings` (runs in thread pool), asyncio.sleep — Arch #4
- ✅ **DATA_CACHE_MAX_ITEMS=24** env-configurable in constants.py (was hardcoded 8) — Arch #5
- ✅ **AppState** consolidation — globals → structured AppState + DI — Arch #7
- ✅ **Marker auth module** — extracted to `backend/marker_auth.py` (make_token/verify_token/startup_check); tests in `tests/test_marker_auth.py`. (Arch #8)
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
- ✅ Meteogram wind-panel enhancement: high-cloud background layers, dark gray cloud shading, terrain/ground-height masking, 10 m wind barbs, and below-ground pressure-wind suppression.
- ✅ Meteogram mobile touch tooltip: tap/drag scrubber with pinned cursor and bottom readout; desktop hover unchanged.
- ✅ 15-minute symbols: `/api/symbols` substep mode, live quarter-hour classification, substep-aware cache keys, and frontend substep controls.
- ✅ Map layer / marker search stabilization
- ✅ Marker UI tuning + admin auth prompt fix

### Overlays (new since 2026-02)
- ✅ **Geopotential overlay** added; scaling/colormap refined (commits `ce9eae7`, `28be999`, `ede3dd4`)
- ✅ **Wave overlays + 600 hPa wind** with contrast tuning and control cleanup (commits `c3be5d1`, `db4087a`, `3305984`, `c6d7e0f`)
- ✅ **Hourly-max cloud base overlay** (commit `4671388`)
- ✅ **Convective/grid-scale precipitation split**: ingest writes `convective_rate`/`gridscale_rate`; overlays, frontend selector, legends, and point popups expose the new layers (commit `9e567d1`)
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
- ✅ D2 LPI mapping switched to `lpi_max` (1h max); EU uses `lpi_con_max`.
- ✅ Classify.py: canonical scalar cloud classifier; cb/blue_thermal precedence fixed
- ✅ **ICON-D2 substep ingest extended through 48h** (commit `23d2ab9`)
- ✅ **EU symbol ingest input fix + regression coverage** (commit `1238e96`)
- ✅ Static grid bundle simplified to single file (commit `cf273d0`)

### Admin / Ops
- ✅ Admin dashboard MVP (`/admin`): ingest health, fallback/cache/perf, feedback inbox, logs
- ✅ **Admin HTTP Basic auth** (`backend/admin_auth.py`)
- ✅ CI pipeline (PR9): lint/type/pytest + qa_smoke/qa_regression/qa_contract/qa_perf workflows.
- ✅ Pytest migration (PR8): unit tests run always; integration/perf marked and skipped in fast CI.
- ✅ `test_symbol_zoom_continuity` threshold resolved.
- ✅ Precomputed symbols benchmark completed; low-zoom JSON-bin precompute is opt-in (`SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS=1`).
- ✅ Marker secret startup policy: ERROR/stderr banner on missing secret; WARNING on weak secret; rotation notes in `docs/OPS_SECRETS.md`.
- ✅ Production CORS documentation: `SKYVIEW_CORS_ORIGINS` covered in `README.md` and `docs/OPS_SECRETS.md`.
- ✅ Root onboarding README points to `SPEC.md`, `TODO.md`, `docs/README.md`.
- ✅ Status endpoint richer: widgets/tables, level filters, artifact drilldown
- ✅ Fallback stats persisted to `data/fallback_stats.json`
- ✅ `/api/status` fallback counters fixed — were overwritten by snapshot fields (`.update()` fix)
- ✅ `/api/status` ingestHealth now includes `missingStepNumbers[]` — exact missing steps, not just count
- ✅ `ingest.py --fill-missing` — ingests only absent steps for a run; defaults to full step range
- ✅ EU overlay gap fixed — hsurf NaN slice check replaces rectangular bbox margin (commit `04509e8`)
- ✅ GitHub Actions watch scripts (`docs/github-actions-watch.md`)
- ✅ GitHub push / remote setup.

### Markers / Location
- ✅ **ICAO marker lookup** (`backend/airport_lookup.py`)
- ✅ Map layer / marker search stabilization

---

## Notes

- Keep help text in sync with backend thresholds when symbol logic changes.
- Multi-worker deployment unsafe (process-local state) — document, defer Redis until needed.
- `_eu_strict_cache` resets on restart — expected, not a bug.
- OpenAir: resume as separate mini-project with own QA checklist.
- `TODO.md` is the source of truth for active backlog entries; archived docs are context only.
- `docs/SYMBOLS_15MIN_IMPLEMENTATION_PLAN_2026-04-13.md` is now a historical implementation note and should be reconciled with current substep behavior.
