# Observation Layer — Implementation Plan (Radar + Satellite Composites)

**Status:** Proposed
**Date:** 2026-06-03
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
- **Satellite** — MSG RSS infrared / water-vapour brightness temperature.

Observations have their **own recent-frames time strip** (e.g. the last 2–3 h at
5-minute cadence) with a **loop/play animation**, independent of the forecast
timeline. The native composite files remain the source of truth; everything the
browser sees is a derived, reprojected, colorized PNG tile.

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

**Key consequence:** if we **reproject each composite onto a regular lat/lon
grid** (the same coordinate convention as ICON output) and store it as Zarr in
the SkyView layout, observations become "just another overlay layer" for ~90% of
the serving path. The only genuinely new backend concept is the **time axis**
(rolling observation frames vs. forecast run/step).

SkyView only displays the **Eastern Alps** window (`BOUNDS = 45.5–48.5°N,
9–17°E`, zoom 5–12; `SPEC.md:466`), and ingest already crops EU data to
`d2_bounds = 43.18–58.08°N, -3.94–20.34°E` (`backend/ingest_config.yaml:9`). So
the Europe-wide composite is **cropped + downsampled to that same small window**
at ingest — the reprojection target is small and cheap.

---

## 3. Architecture overview

```
        ┌─────────────────────── INGEST (new) ───────────────────────┐
 OPERA EDR / S3  ──▶  fetch native ODIM    ──▶  reproject (Lambert EA → reg. lat/lon,
 (radar_ord.py)       (source of truth)         crop to d2_bounds, ~0.02°)  ──▶  Zarr
 EUMDAC MSG RSS  ──▶  fetch native NetCDF  ──▶  reproject (geos → reg. lat/lon)  ──▶  Zarr
 (satellite_…py)                                                                    │
        └────────────────────────────────────────────────────────────────────────┘
                                                                                    ▼
   data/observations/{radar|satellite}/{YYYYMMDDHHMM}.zarr  +  manifest.json (ring buffer)
                                                                                    │
        ┌─────────────────────── SERVE (reuse + thin new layer) ──────────────────┘
        ▼
  /api/observations/frames   (NEW: list available timestamps per source)
  /api/overlay_tile/...png   (REUSE: add observation layers + obs time resolver)
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
- **Change output:** the scaffold stops at "save native file." We add
  `reproject.py` + `store.py` so the pipeline ends at a SkyView-shaped Zarr.
- **Drop satpy RGB** (`to_europe_rgb`) for serving — SkyView colorizes server-side
  from physical values (dBZ, brightness temp), consistent with every other
  overlay. Keep native files as source-of-truth artifacts only.
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
    202606031230.zarr/        # group: dbz (2D), lat (1D), lon (1D) + attrs
    202606031235.zarr/
    ...
    manifest.json             # {"frames": ["202606031230", ...], "updated": "..."}
  satellite/
    202606031230.zarr/        # group: ir_bt, wv_bt, lat, lon + attrs
    ...
    manifest.json
  native/                     # optional: keep source ODIM/NetCDF (source of truth)
```

- Zarr written via the existing `services/storage_io.py:write_zarr_group`
  (Blosc compression), so it's byte-compatible with the loader.
- `manifest.json` is the **time index** (avoids `listdir` on the hot path) and
  records each frame's valid time, source, and age.
- **Retention = ring buffer**, unlike forecast (`keep_runs: 1`). Configurable
  window, default **3 h** (= 36 frames at 5 min). A `prune_old_frames()` in
  `store.py` deletes frames older than the window and rewrites the manifest. This
  is the OPERA S3 cache's own 24 h horizon scaled to what the UI animates.
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

> **Environment note:** this planning session's sandbox has **no network and none
> of these geo deps installed**, so live smoke-tests (PLANNING tasks 1, 3) and the
> 3 "open items" (composite href field, delivery shape, ODIM group layout) must be
> verified on a networked host during Phase 1. The plan is structured so those
> unknowns are isolated inside `radar_ord.py` / `reproject.py`.

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
| **0. Scaffolding** | `backend/observations/` package (port scaffold), config, requirements, `.gitignore`, unit tests for ODIM reader + S3-key parsing using a tiny synthetic fixture | ✅ (code + tests; deps must be installed) |
| **1. Ingest + reproject + store** | `reproject.py`, `store.py`, `ingest_obs.py`, manifest + retention; resolve the 3 PLANNING open items against the live API; one real radar + one real satellite frame on disk as Zarr | ❌ needs network + creds |
| **2. Backend serving** | `OVERLAY_CONFIGS` entries + colormaps, observation load branch, `resolve_observation_frame`, `/api/observations/frames` route, cache rotation | ✅ unit-testable with fixture Zarr |
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
- `test_observations_store.py` — write→manifest→prune ring-buffer semantics.
- `test_observations_overlay.py` — `/api/overlay_tile?layer=obs_radar_dbz` over a
  fixture Zarr returns a valid PNG with correct headers; `latest`/exact/nearest/404
  time resolution.
- Reproject test — Lambert-EA corner → known lat/lon within tolerance.

## 8. Risks & open questions

- **3 PLANNING open items** (composite `href` field, single-grid vs CoverageJSON
  delivery, ODIM composite group layout) — unverifiable offline; isolate in
  `radar_ord.py`/`reproject.py`, confirm via Swagger + 2 real requests in Phase 1.
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
</content>
</invoke>
