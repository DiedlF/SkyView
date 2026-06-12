# Observation Layer — Implementation Plan (Radar + Satellite Composites)

**Status:** Phase 1A implemented: live OPERA radar dBZ + MSG RSS HRV ingest writes derived PNG render frames with manifest/ring-buffer retention; serving/frontend/scheduling pending
**Date:** 2026-06-03 (live-verification addendum 2026-06-12 — see §10)
**Scope:** Add a live **observation layer** to SkyView — EUMETNET OPERA radar
(5-min, 1 km max-reflectivity composite) and EUMETSAT MSG RSS satellite (5-min)
— covering both the backend (ingest + serving) and the frontend (UI + animation).

This plan adapts the standalone `eucomposites/` scaffold (see the uploaded
`config.py`, `radar_ord.py`, `satellite_eumetsat.py`, `poller.py`, and
`PLANNING.md`) into SkyView's existing architecture rather than running it as a
separate service. The guiding principle is **maximum reuse of the overlay-tile
pipeline that already powers ICON-D2/EU overlays**.

---

## 1. Goal & user-facing outcome

A glider pilot opens SkyView and, alongside the forecast layers, can switch on:

- **Radar** — the latest OPERA CIRRUS max-reflectivity composite (dBZ).
- **Satellite** — MSG RSS high-resolution visible (HRV) imagery.

Observations have their **own recent-frames time strip** (e.g. the last 2–3 h at
5-minute cadence) with a **loop/play animation**, independent of the forecast
timeline. The native composite files remain the source of truth; everything the
browser sees is a derived, rendered PNG frame/tile.

---

## 2. Why integrate (not run the scaffold standalone)

The uploaded scaffold fetches Europe-wide composites and stores native
ODIM-HDF5 / NetCDF. That is the right **ingest** half, but SkyView already has a
mature **serving** half we should not duplicate:

- `/api/overlay_tile/{z}/{x}/{y}.png` (`backend/app.py:1702`) already does
  bbox → Web-Mercator tile → reproject a **regular lat/lon grid** → colorize →
  PNG → LRU/TTL cache (`backend/cache_state.py:11`), with reverse-proxy cache
  headers (`backend/response_headers.py:25`).
- The frontend already renders overlays as a Leaflet `tileLayer` against that
  endpoint (`frontend/app.js:1638`), with per-layer opacity, legends
  (`LEGEND_CONFIGS`, `frontend/app.js:597`), and a layer control
  (`frontend/index.html:80`).

**Current implementation choice:** for the first live slice we keep only
**derived render frames** (`radar_dbz` PNG and satellite `hrv` PNG) plus a
manifest. Native ODIM/ZIP/NAT files are temporary ingest inputs. Numeric
regular-lat/lon Zarr remains a future option if we later need server-side
physical-value sampling, recoloring, or per-tile composition from arrays.

SkyView only displays the **Eastern Alps** window (`BOUNDS = 45.5–48.5°N,
9–17°E`, zoom 5–12; `SPEC.md:466`), and ingest already crops EU data to
`d2_bounds = 43.18–58.08°N, -3.94–20.34°E` (`backend/ingest_config.yaml:9`). So
the Europe-wide composite is **cropped + downsampled to that same small window**
at ingest — the reprojection target is small and cheap.

---

## 3. Architecture overview

```
        ┌─────────────────────── INGEST (new) ───────────────────────┐
 OPERA EDR / S3  ──▶  fetch native ODIM temp ──▶ reproject + render dBZ PNG
 (radar_ord.py)
 EUMDAC MSG RSS  ──▶  fetch ZIP temp ──▶ unzip NAT temp ──▶ render HRV PNG
 (satellite_…py)                                                                    │
        └────────────────────────────────────────────────────────────────────────┘
                                                                                    ▼
   data/observations/{radar|satellite}/{YYYYMMDDHHMM}_{product}.png
   + manifest.json (5 h ring buffer)
                                                                                    │
        ┌─────────────────────── SERVE (reuse + thin new layer) ──────────────────┘
        ▼
  /api/observations/frames   (NEW: list available timestamps per source)
  /api/observations/render/...png or /api/overlay_tile/...png (serve render frame)
        │
        ▼
        └─────────────────────── FRONTEND (new UI on existing plumbing) ──────────┐
  "Observations" layer section + recent-frames strip + loop animation             │
  Leaflet tileLayer → /api/overlay_tile?layer=obs_radar_dbz&time=<frame ts>        │
        └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Backend design

### 4.1 New package: `backend/observations/`

Port the scaffold into a SkyView sub-package (keeps it isolated, importable, and
testable). Suggested layout:

```
backend/observations/
  __init__.py
  config.py              # adapted from scaffold config.py (env-overridable)
  radar_ord.py           # adapted: OPERA EDR + unsigned-S3 fallback + ODIM reader
  satellite_eumetsat.py  # adapted: EUMDAC fetch + native reader
  reproject.py           # NEW: native grid → regular lat/lon crop (the heavy lift)
  store.py               # NEW: write reprojected Zarr + maintain manifest/ring buffer
  ingest_obs.py          # NEW: orchestration (fetch → reproject → store → retain)
  poller.py              # adapted: interval loop; optional MQTT trigger
```

Adaptation notes vs. the uploaded scaffold:

- **Reuse as-is (logic):** `radar_ord.RadarSource` (EDR + S3 fallback), the ODIM
  reader `read_odim_maxreflectivity()`, `satellite_eumetsat.SatelliteSource`.
- **Change output:** the scaffold stops at "save native file." The live slice
  now adds `render.py` + render-aware `store.py` so the retained cache product is
  a derived PNG frame, not a native provider file.
- **Native files are temporary** for the selected cache policy. Keep native
  files only in explicit debug/live-verification runs.
- **Config** moves under SkyView's env conventions (`SKYVIEW_*` / existing
  `EUCOMP_*` accepted), data root anchored at SkyView `data/`.

### 4.2 Reprojection (`reproject.py`) — the one genuinely new compute

The overlay/tile renderer assumes a **regular lat/lon grid with 1-D `lat`/`lon`**
(like ICON Zarr; see `backend/grid_utils.py:bbox_indices` and the tile reprojection
in `app.py:1819`). So ingest must convert:

- **Radar (OPERA):** ODIM HDF5 in **Lambert Azimuthal Equal Area**. Read the
  `/where` projdef + corner attributes (already returned by the scaffold reader),
  build the source grid with `pyproj`, then resample to a regular lat/lon target
  grid cropped to `d2_bounds` at a chosen resolution (start at **0.02°**, ICON-D2
  native; radar is 1 km so this slightly downsamples — acceptable for the Alps
  window). Nearest-neighbour or `pyresample`/`scipy.griddata`.
- **Satellite (MSG RSS):** geostationary projection. Use `satpy`/`pyresample`
  `Scene.resample()` to an `AreaDefinition` matching the same regular target grid,
  or precompute a **fixed pixel→grid lookup table** (geostationary geometry is
  static) stored under `data/coords/` — mirrors the existing
  `kdtree_indices.npy` reuse pattern.

Output array convention (so it Just Works downstream): 2-D float32, regular
`lat`/`lon` 1-D coords, NaN for nodata/undetect (radar) or off-disk (satellite).

**Target grid is small:** ~ (58.08-43.18)/0.02 × (20.34+3.94)/0.02 ≈ **745 × 1214**,
same order as ICON-D2. Reprojection is sub-second per frame.

### 4.3 Storage layout & retention

```
data/observations/
  radar/
    202606121820_radar_dbz.png
    manifest.json
  satellite/
    202606121824_hrv.png
    manifest.json
  tmp/                        # temporary native ODIM/ZIP/NAT during ingest
```

- `manifest.json` is the **time index** (avoids `listdir` on the hot path) and
  records each frame's valid time, products, attrs, source, and age.
- **Retention = ring buffer**, unlike forecast (`keep_runs: 1`). Configurable
  window, default **5 h** (= 60 frames at 5 min). `store.prune()` deletes render
  PNGs older than the window and rewrites the manifest.
- Add `data/observations/` to `.gitignore` (runtime data; `.gitignore` already
  ignores `data/icon-d2/` etc.).

### 4.4 Serving — reuse the overlay/tile path

**(a) Register observation layers** in `OVERLAY_CONFIGS`
(`backend/overlay_render.py:371`):

```python
"obs_radar_dbz":   {"var": "dbz",   "cmap": colormap_radar_dbz,  "source": "observation"},
"obs_satellite_ir":{"var": "ir_bt", "cmap": colormap_ir_bt,      "source": "observation"},
"obs_satellite_wv":{"var": "wv_bt", "cmap": colormap_wv_bt,      "source": "observation"},
```

Add the matching colormap functions next to the existing ones (e.g.
`colormap_radar_dbz`: the standard NWS/DWD dBZ scale, transparent below ~5 dBZ;
`colormap_ir_bt`: cold-cloud-top greyscale/colour ramp).

**(b) Branch on `source == "observation"`** at the two load sites
(`/api/overlay` `app.py:1226` and `/api/overlay_tile` `app.py:1702`). Instead of
`load_step_data(model, run, step, …)`, call a new
`observations/store.load_frame(source, timestamp, keys)` that reads
`data/observations/<source>/<ts>.zarr`. Everything after the load — bbox crop,
reproject-to-tile, `colorize_layer_vectorized`, PNG encode, cache — is unchanged.

**(c) Observation time resolver.** Forecast uses
`resolve_time_with_cache_context()` (run/step). Observations need a parallel
`resolve_observation_frame(source, time)`:
- `time="latest"` → newest manifest entry.
- `time=<ISO ts>` → exact frame, else nearest within ± half-cadence, else 404
  (strict, mirroring SkyView's no-substitution philosophy in `ARCHITECTURE.md`).
- Returns the frame timestamp used → feeds `X-ValidTime` header and the tile
  cache key (`cache_state.py` key already includes `time`, so no cache changes).

**(d) New metadata endpoint** — the only new route. Register in
`backend/routers/overlay.py` (or a small `routers/observations.py`):

```
GET /api/observations/frames?source=radar|satellite
  → { "source": "radar",
      "cadence_seconds": 300,
      "frames": [ {"time":"2026-06-03T12:30:00Z","age_s":120}, ... ],
      "latest": "2026-06-03T12:30:00Z" }
```

The frontend polls this to build its recent-frames strip and to know "is data
fresh?". Cache-Control short (e.g. `max-age=30`).

**(e) Capabilities.** Optionally surface observations in `/api/models`
(`backend/model_caps.py:1`) as a non-forecast category so the explorer/diagnostics
can see them, but the frontend toggle can rely solely on `/api/observations/frames`.

### 4.5 Ingest orchestration & scheduling

- `ingest_obs.py` exposes `run_once(source)` (fetch newest → if new, reproject →
  store → prune) and is safe to call repeatedly (de-dupes via manifest, like the
  scaffold's last-seen markers).
- **Scheduling options (recommend starting with cron):**
  1. **Cron** — extend `backend/cron-ingest.sh` (currently every 10 min) or add a
     dedicated 2–3-min cron calling `python -m backend.observations.ingest_obs`.
     Use a separate `flock` lock (the script already uses `/tmp/skyview-ingest.lock`).
  2. **Poller** — the scaffold `poller.py` as a long-running service
     (`deploy/skyview-observations.service`, modeled on `deploy/skyview.service`).
  3. **MQTT push** (planning task 4) — subscribe to the OPERA notification broker
     (port 8884) and trigger `run_once("radar")` per new-frame message. Best
     latency; add as an enhancement after cron works.
- On new frames, call `cache_state.rotate_caches_for_context()` with an
  observation context key so stale tiles are dropped.

### 4.6 Dependencies

Add to `backend/requirements.txt` (currently has `numpy, scipy, xarray, zarr,
Pillow, requests`): `h5py` (ODIM), `pyproj` + `pyresample` (reprojection),
`boto3` (S3 fallback), `eumdac` (satellite). `satpy[seviri]` optional (satellite
read/resample). All already listed in the scaffold's `requirements.txt`.

> **Environment note (updated 2026-06-12):** an earlier draft recorded the
> session sandbox as blocking `api.meteogate.eu` / cloudferro S3 /
> `api.eumetsat.int` with `Host not in allowlist`. **This is no longer true on
> the current host** — all three are reachable, and a full live satellite fetch
> succeeded on 2026-06-12 (see §10). The geo deps are not in the default
> interpreter but install cleanly into a dedicated venv (`.venv-obs`:
> `eumdac`, `satpy`, `pyresample`, `matplotlib`). The earlier ORD "open items
> #1/#2" remain resolved per the official docs; satellite open items are now
> resolved against a real file (§10). Unknowns stay isolated inside
> `radar_ord.py` / `reproject.py` / the satellite ingest branch.

---

## 5. Frontend design

The frontend already has all the rendering plumbing; observations need a **new
control group** and a **separate time axis**. Add to `frontend/` (the pilot app);
`explorer/` is the general variable-browser and can adopt observations later via
its `/api/variables` discovery.

### 5.1 Layer controls (`frontend/index.html`)

Add an **"Observations"** section after the Overlay block (`index.html:80`),
following the existing radio/checkbox idiom:

```html
<div class="layer-divider"></div>
<div class="layer-subtitle" data-i18n="layer.observations.title">Observations (live)</div>
<label><input type="checkbox" id="layer-obs-radar"> <span data-i18n="layer.obs.radar">Radar</span></label>
<label><input type="checkbox" id="layer-obs-satellite"> <span>Satellite</span>
  <select id="obs-sat-type"> <!-- ir_bt / wv_bt --> </select></label>
<div id="obs-timebar" class="obs-timebar" style="display:none;">
  <button id="obs-play">▶</button>
  <input type="range" id="obs-frame-slider" min="0" max="0">
  <span id="obs-frame-label">--:--</span>
</div>
```

(Radar and satellite are independent **checkboxes** like Convection/Wind, not the
single-active overlay radios, because a pilot may want radar over the satellite or
ICON overlay. Use stacked panes for ordering.)

### 5.2 Map layers & panes (`frontend/app.js`)

- Add panes (near `app.js:655`): `skyviewObsSatellitePane` (low, e.g. z-index 340,
  under ICON raster) and `skyviewObsRadarPane` (e.g. 360, above satellite). This
  lets radar sit over satellite over base map, all under symbols/wind (610–620).
- Add globals: `obsRadarLayer`, `obsSatLayer`, `obsFrames=[]`,
  `obsFrameIndex=0`, `obsPlaying=false`, `obsTimer=null`.
- New `loadObservationLayer(source)` mirroring `loadOverlay()` (`app.js:1576`):
  builds a `L.tileLayer('/api/overlay_tile/{z}/{x}/{y}.png?' + params)` with
  `layer=obs_radar_dbz` (or `obs_satellite_ir/_wv`), `time=<selected frame ts>`,
  `pane=<obs pane>`, opacity from a new `obs_*` rule in `overlayOpacityForLayer()`
  (`app.js:1508`, e.g. radar 0.7, satellite 0.85).

### 5.3 Recent-frames time strip + animation

This is the main new UX and the only place observations diverge from the forecast
timeline (`buildTimeline()` `app.js:2027`):

- On enabling an observation layer, fetch `/api/observations/frames?source=…`,
  store in `obsFrames`, set the slider `max = frames.length-1`, default to the
  latest frame.
- **Slider** scrubs frames → re-points the tileLayer `time` param (swap the
  layer URL; keep 1 buffered frame for smoothness, like `keepBuffer:1` at
  `app.js`).
- **Play** advances `obsFrameIndex` every ~500 ms, looping, with a brief pause on
  the latest frame — standard radar-loop UX. Preload adjacent frames by letting
  Leaflet fetch the next `time` before display.
- Poll `/api/observations/frames` every ~60 s while active to append newly
  ingested frames and advance "latest".
- This time axis is **independent** of the forecast `timesteps`/step buttons; the
  forecast timeline keeps working unchanged when observations are on.

### 5.4 Legends & info

- Add `LEGEND_CONFIGS.obs_radar_dbz` and `.obs_satellite_ir/_wv`
  (`frontend/app.js:597`) with the dBZ / brightness-temperature gradients and
  labels, so the existing `updateLegend()` (`app.js:1782`) renders them.
- Show frame valid-time + age ("Radar 12:30 UTC · 2 min old") near the timebar;
  warn (amber) if `age_s` exceeds ~2 cadences (data stalled).

### 5.5 Attribution / legal (required by license)

OPERA composites are **CC BY 4.0** (attribute EUMETNET); MSG via EUMETSAT.
Add attribution to the Leaflet attribution control and the legal pages
(`frontend/legal.html`, `frontend/impressum.html`) when a layer is active.

---

## 6. Phased delivery

| Phase | Deliverable | Verifiable in this sandbox? |
|-------|-------------|------------------------------|
| **0. Scaffolding** ✅ **DONE** | `backend/observations/` package (config, radar_ord, satellite_eumetsat, poller, requirements), `.gitignore`, `tests/test_observations_s3keys.py` + `tests/test_observations_odim.py`. S3-key parser extracted as pure/tested helpers; ODIM test uses a synthetic fixture + `importorskip` | ✅ (logic verified; full `pytest`/`h5py` run needs the ingest venv) |
| **1A. Derived render ingest** ✅ **DONE; live-verified 2026-06-12** | `render.py`, render-aware `store.py`, `ingest_obs.py` fetches live OPERA and MSG15 RSS into temp dirs, renders `radar_dbz` + `hrv` PNGs, writes manifests, prunes with 5 h retention. | ✅ live-verified with current provider data |
| **1B. Numeric observation grids** | Optional future path: `reproject.py` + Zarr for dBZ/IR/WV if physical-value overlays or recoloring become necessary. | partial |
| **2. Backend serving** | `/api/observations/frames`, render-frame response endpoint or tile integration, cache headers, age/freshness metadata | ✅ unit-testable with fixture PNGs |
| **3. Frontend** | Observations control group, panes/layers, recent-frames strip + loop, legends, attribution | ✅ (manual/visual; needs backend frames) |
| **4. Scheduling** | cron entry or `skyview-observations.service`; optional MQTT push subscriber | ❌ needs host/network |
| **5. Hardening** | retry/backoff (transient HTTP), disk-space guard, structured logging (`backend/logging_config.py`), perf check that obs tiles stay within budget (`scripts/qa_perf.py`) | partial |

A natural first PR = **Phases 0 + 2 + a fixture-driven test** (all sandbox-
verifiable), landing the serving path behind a feature that returns 404 until real
frames exist. Phases 1/3/4 follow on a networked host.

---

## 7. Testing

Mirror `tests/` conventions (`pytest.ini`, markers `integration`/`perf`):

- `test_observations_odim.py` — ODIM reader gain/offset/nodata masking against a
  small synthetic HDF5 fixture (built in-test with `h5py`); the composite group
  layout (`/dataset1/data1/data`) is asserted and is one of the 3 open items to
  confirm against a real CIRRUS file.
- `test_observations_s3keys.py` — S3 key parsing/sorting for
  `YYYY/MM/DD/OPERA/COMP/OPERA@<ts>@0@DBZH.h5` (newest-wins).
- `test_observations_store.py` — render write→manifest→prune ring-buffer semantics.
- `test_observations_satellite.py` — MSG product valid-time parsing.
- Future `test_observations_overlay.py` — render endpoint/tile path over fixture
  PNGs returns a valid PNG with correct headers; `latest`/exact/nearest/404 time
  resolution.
- Reproject test — Lambert-EA corner → known lat/lon within tolerance.

## 8. Risks & open questions

### MeteoGate access — confirmed June 2026 (ORD official docs)

Re-researched against the [ORD API docs](https://eumetnet.github.io/openradardata-documentation/):

- **Access is OPEN and anonymous** — onboarding to MeteoGate finalised 20 May
  2026, **no whitelisting**. The earlier `403 Forbidden` we saw was **this Claude
  Code on web sandbox's own egress policy** (body literally `Host not in
  allowlist`, empty `server` header), NOT MeteoGate. To exercise the live API
  *from a web session*, the environment's **network policy must allowlist**
  `api.meteogate.eu`, `s3.waw3-1.cloudferro.com`, `radar.meteogate.eu` (and
  `api.eumetsat.int` for satellite). A normal host has no such restriction.
- **Rate limits:** anonymous calls are rate-limited — watch the
  `x-ratelimit-remaining` header. For sustained 5-min polling, register a free API
  key at <https://devportal.meteogate.eu/> (`RadarConfig.api_key`, sent as the
  `apikey` header).
- **Correct OPERA composite id:** `location_id = 0-20010-0-OPERA` (the previous
  `0-*-*-OPERA` was wrong — fixed in `config.py`). Composite query:
  `standard_name=DBZH`, `method=comp`, `format=ODIM`.
- **Open item #1 (href field) RESOLVED:** the `/collections/observations/items`
  FeatureCollection puts the download URL in `properties.data` (our code already
  prefers it). The `/collections/observations/locations/{id}` route returns
  CoverageJSON with the same links.
- **Open item #2 (delivery shape) RESOLVED:** the composite is a **single
  Europe-wide grid file** (Lambert EA, corners ≈70N/30W–32N/30E, 1×1 km CIRRUS),
  downloadable straight from S3:
  `s3://openradar-24h/YYYY/MM/DD/OPERA/COMP/OPERA@<YYYYMMDDTHHMM>@0@DBZH.h5`
  (unsigned `--no-sign-request`, confirmed). No CoverageJSON round-trip needed.
- **Open item #3 (ODIM group layout) still open** — needs a real CIRRUS file.
  Mitigation: composites are **also published as cloud-optimized GeoTIFF**, which
  carries its own georeferencing (rasterio) and sidesteps ODIM `/where` guessing —
  a lower-risk alternative input for `reproject.py` if the HDF5 layout surprises us.
- **MQTT push (Phase 4) confirmed:** `wss://radar.meteogate.eu:8884/ordmqtt`,
  user/pass `everyone`/`everyone`, topic `ORD/eu.eumetnet/0-20010-0-OPERA/DBZH`
  (or `mqtt://radar.meteogate.eu:1883`). Captured in `RadarConfig`.

### Other risks

- **Reprojection fidelity** — Lambert-EA/geostationary → 0.02° lat/lon; validate
  visually against a known event before shipping; consider 0.01° for radar if the
  Alps window benefits.
- **Time-axis UX coupling** — keep observation time strictly separate from the
  forecast timeline to avoid regressions in `buildTimeline()`/keyboard nav.
- **Cadence reality** — only **max reflectivity** is true 5-min; RATE/ACRR are
  coarser (NIMBUS). MSG RSS is 5-min now; MTG FCI (10-min) is the migration target
  (`config.collection_id` switch already supports this).
- **Load** — obs tiles add a second hot tile family; reuse the split desktop/mobile
  LRU and short TTL; precompute dBZ→colour at ingest only if profiling shows need.

## 9. Concrete first-PR checklist (Phases 0 + 2)

1. `backend/observations/` package: `config.py`, `radar_ord.py`,
   `satellite_eumetsat.py`, `store.py`, `reproject.py` (interfaces + radar path),
   `ingest_obs.py`, `poller.py`.
2. `backend/requirements.txt`: add `h5py, pyproj, pyresample, boto3, eumdac`.
3. `.gitignore`: add `data/observations/`.
4. `backend/overlay_render.py`: `obs_*` configs + `colormap_radar_dbz` /
   `colormap_ir_bt` / `colormap_wv_bt`.
5. `backend/app.py`: observation load branch in `/api/overlay` + `/api/overlay_tile`;
   `resolve_observation_frame`.
6. `backend/routers/observations.py` (+ wire in `app.py`): `/api/observations/frames`.
7. Tests under `tests/` per §7 with synthetic fixtures.
8. Docs: link this plan from `docs/README.md` and add an entry to `TODO.md`.

---

## 10. Live verification log — satellite (2026-06-12)

First end-to-end run of the **satellite** path against live EUMETSAT data, on a
real host (not the restricted sandbox of earlier drafts) using a real API key in
`backend/.env` and a dedicated `.venv-obs` (`eumdac 3.1.1`, `satpy 0.60.0`,
`pyresample 1.35.0`, `matplotlib 3.11.0`). Auth → search → download → unzip →
decode → reproject → preview **all succeeded**. Findings below are confirmed
against the real product, not docs.

### 10.1 Confirmed working

- **Credential path is correct.** `EUMETSAT_CONSUMER_KEY`/`_SECRET` in
  `backend/.env` → `eumdac.AccessToken` → token + Data Store reachable.
  `scripts/eumetsat_auth.py` validates it cleanly (exit 0).
- **Live product cadence confirmed:** the MSG Rapid Scan collection serves ~9
  products per 45 min (5-min cadence), newest only a few minutes old.
- **Download + decode confirmed:** a 59 MB product downloaded in ~4 s; it is a
  **ZIP** containing a 102 MB `.nat` (SEVIRI native L1.5) + `EOPMetadata.xml` +
  `manifest.xml`. `satpy` `reader="seviri_l1b_native"` reads it directly.
- **Channels present and physical:** `IR_108` (10.8 µm window) and `WV_062`
  (6.2 µm water vapour) both load in Kelvin with sane ranges over Europe
  (IR ≈ 227–308 K, WV ≈ 222–242 K). All 12 SEVIRI channels are available
  (HRV, VIS006/008, IR_016/039/087/097/108/120/134, WV_062/073).
- **Reprojection to the SkyView grid works** and the Alps window is fully
  covered (native pixels span 14.9–71.9 °N; the 46.5 °N/12 °E point reads a
  valid BT). Previews saved to `.state/sat_preview_{IR_108,WV_062}.png`.

### 10.2 Code fixes landed from live findings

- **Collection id fixed:** default MSG RSS collection is now
  `EO:EUM:DAT:MSG:MSG15-RSS`; the older `HRSEVIRI-RSS` id 404s on the live Data
  Store.
- **ZIP/NAT handling fixed:** `SatelliteSource.fetch_latest()` saves `.zip`, and
  `extract_native_file()` opens the inner `.nat`.
- **Satellite ingest wired for selected cache policy:** `ingest_satellite()`
  renders the HRV channel to a derived PNG frame and does not retain native
  products.
- **Remaining numeric-grid caution:** `satpy`'s `Scene.resample()` silently
  dropped half the swath during numeric-grid testing. Using
   `scn.resample(area, resampler="nearest", radius_of_influence=8000)` produced a
   grid with a clean cut at ~50 °N and only ~52 % coverage — the southern half
   (including the Alps) was discarded even though the source pixels exist. A
   manual **`scipy.spatial.cKDTree` nearest-neighbour** reprojection over the
  native `(lon, lat)` arrays gave **100 % coverage**. **Action if numeric grids
  return:** validate `reproject.py:reproject_swath` against this exact file
  before trusting it; prefer the cKDTree path, or pass `reduce_data=False` and
  verify.

### 10.3 Access / licensing gotchas (operational)

- **NRT licence is per-collection and separate.** Even with a valid key and
  general account licences accepted, SEVIRI **L1.5 image** collections
  (`HRSEVIRI`, `MSG15-RSS`) returned `403` with body
  `NRTLicense required to access this collection` until the licence for **that
  specific image-data collection** was accepted in the Data Store web UI. Other
  MSG products (e.g. Cloud Mask `MSG:CLM`) downloaded fine before that, so a
  blanket "accept all licences" page does **not** necessarily cover the image
  NRT licence. Acceptance can take up to ~1 h to propagate to the API gateway.
- **Ops doc:** record the exact collection(s) whose NRT licence must be accepted
  in `docs/OPS_SECRETS.md` alongside the EUMETSAT key setup, so a new deployer
  doesn't hit the same 403.

### 10.4 Live ingest verification

- `python -m backend.observations.ingest_obs --source both` live-verified on
  2026-06-12 with `.venv-obs`.
- OPERA EDR returned `422` for the query shape, but the unsigned CloudFerro S3
  fallback successfully downloaded `OPERA@20260612T1820@0@DBZH.h5`; ODIM layout
  read and Lambert-EA reprojection succeeded.
- EUMETSAT MSG15 RSS downloaded a live ZIP, extracted the `.nat`, decoded HRV via
  Satpy, and wrote `202606121824_hrv.png`.
- Retained cache after cleanup contained only derived PNGs and manifests under
  `data/observations/{radar,satellite}/`; native inputs lived only in temp dirs.

---

### Reference map (existing code this plan builds on)

| Concern | File:line |
|---|---|
| Tile endpoint (reused) | `backend/app.py:1702` |
| Full-image overlay (reused) | `backend/app.py:1226` |
| Overlay layer configs + colormaps | `backend/overlay_render.py:371` |
| Overlay field assembly | `backend/overlay_data.py:24` |
| Tile/computed caches | `backend/cache_state.py:11` |
| Cache rotation on new data | `backend/cache_state.py:265` |
| Zarr read/write | `backend/services/storage_io.py` |
| Step data loader | `backend/services/data_loader.py:89` |
| BBox indexing / grids | `backend/grid_utils.py`, `backend/grid_aggregation.py` |
| Response headers | `backend/response_headers.py:25` |
| Overlay router | `backend/routers/overlay.py` |
| Models capability payload | `backend/model_caps.py:1` |
| Ingest pipeline + scheduling | `backend/ingest.py:1018`, `backend/cron-ingest.sh` |
| Ingest config (crop bounds/retention) | `backend/ingest_config.yaml:9` |
| Frontend overlay tileLayer | `frontend/app.js:1638` |
| Frontend layer resolve | `frontend/app.js:1494` |
| Frontend legends | `frontend/app.js:597`, `frontend/app.js:1782` |
| Frontend timeline (forecast) | `frontend/app.js:2027` |
| Frontend layer controls (HTML) | `frontend/index.html:80` |
| Display window / zoom map | `SPEC.md:466`, `backend/constants.py:10` |
