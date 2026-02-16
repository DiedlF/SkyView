# QA Baseline Report — 2026-02-15 (updated 2026-02-16)

## Scope
Initial baseline lock for current Skyview state before deeper refactor work, plus P0 completion update.

## Environment
- Backend: `http://127.0.0.1:8501`
- Explorer: `http://127.0.0.1:8502`
- Area checks: around Geitau and broader viewport slices

## Automated Checks

### Core suites
- ✅ `python3 scripts/qa_smoke.py`
- ✅ `python3 scripts/qa_regression.py`
- ✅ `python3 scripts/qa_contract.py`

### Endpoint sanity
- ✅ `/api/health` -> 200
- ✅ `/api/timesteps` -> 200
- ✅ `/api/models` -> 200
- ✅ `/api/status` -> 200

## P0 Manual/Interactive Validation (2026-02-16)

### Overlay + interaction stability
- ✅ Overlay controls render and can be switched (including `CAPE_ml` layer)
- ✅ Zoom interactions (in/out) perform without overlay exceptions after frontend fix
- ✅ Timeline and layer panel remain functional during interactions

### Regression discovered + fixed during P0
- ❗ Found JS runtime error during overlay draw ordering:
  - `TypeError: symbolLayer.bringToFront is not a function` in `frontend/app.js`
- ✅ Fixed by iterating symbol sublayers and calling `bringToFront()` per marker layer.
- ✅ Re-tested with no new console errors observed for this path.

### CAPE threshold + tooltip parity checks
- ✅ CAPE threshold logic verified with direct renderer check:
  - `colorize_layer_vectorized('thermals', ...)` returns alpha=0 for values `<50`
  - values `>=50` produce non-transparent pixels
- ✅ Frontend CAPE tooltip formatting/path verified in code:
  - `thermals: (v) => v != null ? \`CAPE_ml: ${v} J/kg\` : null`
  - tooltip values sourced from `data.overlay_values[overlayKey]`

## Prior baseline notes retained

1. `clouds_total_mod` returns valid rendered output and no endpoint failures.
2. CAPE (`thermals`) endpoint behavior is active and returning thresholded rendering.
3. Some `/api/point` samples for selected wind level (`850`) can return `wind_speed/wind_dir = null` depending on availability.

## Baseline Status
- **Automated smoke/regression/contract:** PASS
- **P0 QA & hardening:** COMPLETE

## Sprint-2 Benchmark Snapshot (2026-02-16)

Backend: `http://127.0.0.1:8501`  
Timestep tested: `2026-02-16T10:00:00Z`

### Symbols
- `z9` same bbox (`46,8,49,14`), 8 calls:
  - first call ~135.7 ms, subsequent ~29.8–31.2 ms
  - avg: **43.4 ms**
- `z9` varied bboxes (cold-ish), 8 calls:
  - avg: **101.8 ms** (min 30.3, max 125.7)
- `z12` same bbox, 8 calls:
  - first call ~32.8 ms, subsequent ~6.3–6.7 ms
  - avg: **9.8 ms**

### Overlay tiles (`/api/overlay_tile`, z9)
- `sigwx`:
  - same tile avg: **4.2 ms** (HIT path ~2 ms after first)
  - varied tiles avg: **11.1 ms**
- `thermals`:
  - same tile avg: **2.8 ms**
  - varied tiles avg: **6.6 ms**
- `clouds_total`:
  - same tile avg: **4.7 ms**
  - varied tiles avg: **9.2 ms**

### Cache/Status metrics snapshot (`/api/status`)
- `cache.context`: `icon_d2|2026021609|1`, rotations: `1`
- `symbols_cache`:
  - items: `11 / 256`
  - hitRate: `0.46875`
  - metrics: `hits=15, misses=17, expired=6, evictions=0`
- `tile cache metrics`:
  - `hits=24, misses=24, evictions=0, expired=0`

## Next Action
Proceed with remaining P1 tasks:
- final backend/explorer convergence audit and residual cleanup
- optional additional first-hit low-zoom optimization if needed
- reverse-proxy tile caching option (Nginx/Caddy)
