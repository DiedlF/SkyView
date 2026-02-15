# Full ICON Data Ingest + Explorer App

**Created:** 2026-02-12
**Status:** Planned

## Overview

Expand Skyview's data pipeline to ingest a curated set of ~66 variables (up from ~20) at full ICON-D2 grid coverage, and build a separate "Explorer" app for raw data visualization. Skyview and Explorer share the same data store.

## Current State

- **Variables:** ~20 (aviation-focused)
- **Region:** Alps crop (45.5–48.5°N, 9–17°E) — 150×400 grid points
- **Storage:** ~200 MB/run (compressed .npz)
- **Models:** ICON-D2 (hours 1–48) + ICON-EU (hours 49–120, cropped to Alps)
- **Retention:** Last 2 runs

## Target State

- **Variables:** 66 (56 single-level + 10 pressure-level)
- **Region:** Full ICON-D2 domain (43.18–58.08°N, -3.94–20.34°E) — 746×1215 grid points
- **Storage:** ~4 GB/run (compressed)
- **Models:** ICON-D2 (full grid) + ICON-EU (cropped to D2 bounds, 238×388 points)
- **Retention:** Latest run only (no history)

---

## Variable Groups

### Aviation (19 vars) — Skyview core
| Variable | Description |
|----------|-------------|
| `ww` | Significant weather (WMO code) |
| `cape_ml` | CAPE mixed layer (J/kg) |
| `cin_ml` | CIN mixed layer (J/kg) |
| `htop_dc` | Dynamic convective cloud top (m) |
| `hbas_sc` | Convective cloud base (m) |
| `htop_sc` | Shallow convection top (m) |
| `lpi` | Lightning potential index |
| `ceiling` | Cloud ceiling (m) |
| `clcl` | Low cloud cover (%) |
| `clcm` | Mid cloud cover (%) |
| `clch` | High cloud cover (%) |
| `clct` | Total cloud cover (%) |
| `clct_mod` | Modified total cloud cover (%) |
| `cldepth` | Cloud depth (optical) |
| `vis` | Visibility (m) |
| `hzerocl` | Zero-degree isotherm height (m) |
| `snowlmt` | Snow line altitude (m) |
| `mh` | Mixing layer height (m) |
| `hsurf` | Terrain height (m, static) |

### Surface Weather (22 vars)
| Variable | Description |
|----------|-------------|
| `t_2m` | 2m temperature (K) |
| `td_2m` | 2m dewpoint (K) |
| `tmax_2m` | 2m temperature max (K) |
| `tmin_2m` | 2m temperature min (K) |
| `relhum_2m` | 2m relative humidity (%) |
| `pmsl` | Mean sea level pressure (Pa) |
| `ps` | Surface pressure (Pa) |
| `u_10m` | 10m wind u-component (m/s) |
| `v_10m` | 10m wind v-component (m/s) |
| `vmax_10m` | 10m max wind gust (m/s) |
| `tot_prec` | Total precipitation (kg/m²) |
| `prr_gsp` | Rain rate grid-scale (kg/m²/h) |
| `prs_gsp` | Snow rate grid-scale (kg/m²/h) |
| `prg_gsp` | Graupel rate grid-scale (kg/m²/h) |
| `rain_gsp` | Rain amount grid-scale (kg/m²) |
| `rain_con` | Rain amount convective (kg/m²) |
| `snow_gsp` | Snow amount grid-scale (kg/m²) |
| `snow_con` | Snow amount convective (kg/m²) |
| `grau_gsp` | Graupel amount grid-scale (kg/m²) |
| `h_snow` | Snow depth (m) |
| `freshsnw` | Fresh snow factor (dimensionless) |
| `t_g` | Ground temperature (K) |

### Severe Weather (14 vars)
| Variable | Description |
|----------|-------------|
| `dbz_cmax` | Composite max reflectivity (dBZ) |
| `dbz_ctmax` | Column-total max reflectivity (dBZ) |
| `dbz_850` | Reflectivity at 850 hPa (dBZ) |
| `lpi_max` | Max lightning potential index |
| `sdi_2` | Supercell detection index |
| `vorw_ctmax` | Max rotation (1/s) |
| `w_ctmax` | Max updraft velocity (m/s) |
| `uh_max` | Updraft helicity max (m²/s²) |
| `uh_max_low` | Updraft helicity low-level (m²/s²) |
| `uh_max_med` | Updraft helicity mid-level (m²/s²) |
| `echotop` | Echo top height (m) |
| `tcond_max` | Max condensate (kg/m²) |
| `tcond10_mx` | Max condensate 10-min (kg/m²) |
| `q_sedim` | Sedimentation flux (kg/m²/s) |

### Soaring (1 var)
| Variable | Description |
|----------|-------------|
| `ashfl_s` | Sensible heat flux (W/m²) |

### Pressure-Level Wind (2 vars × 5 levels = 10)
| Variable | Levels (hPa) |
|----------|-------------|
| `u` | 950, 850, 700, 500, 300 |
| `v` | 950, 850, 700, 500, 300 |

---

## Storage Estimates

### Per Run (latest only, no history)

| Component | Uncompressed | Compressed (~3x) |
|-----------|-------------|-------------------|
| ICON-D2 (66 vars × 49 steps, full grid) | 10.9 GB | ~3.6 GB |
| ICON-EU (66 vars × 44 steps, cropped to D2 bounds) | 1.0 GB | ~0.3 GB |
| **Total** | **~12 GB** | **~4 GB** |

### Bandwidth

| Metric | Value |
|--------|-------|
| Download per D2 run (bz2 from DWD) | ~2.5 GB |
| D2 runs per day | 8 |
| Download per EU run | ~0.5 GB |
| EU runs per day | 4 |
| **Daily bandwidth** | **~22 GB** |

### Disk Requirement

~5 GB recommended (4 GB data + buffer during ingest when old and new run coexist briefly).

---

## Architecture

### Data Flow
```
DWD OpenData (GRIB2/bz2)
  → ingest.py (download, decompress, crop EU, extract)
  → .npz files in data/{model}/{run}/{step:03d}.npz
  → Shared by Skyview (port 8501) + Explorer (port 8502)
```

### File Structure
```
skyview/
├── backend/
│   ├── ingest.py            # Extended: variable groups, full D2 grid
│   ├── ingest_config.yaml   # Variable group definitions + toggle
│   ├── app.py               # Skyview API (unchanged, reads same data)
│   ├── classify.py          # Cloud classification (unchanged)
│   ├── soaring.py           # Soaring metrics (unchanged)
│   └── cron-ingest.sh       # Updated for new variable set
├── explorer/
│   ├── app.py               # Raw data viz API (port 8502)
│   ├── index.html           # Leaflet-based variable explorer
│   ├── app.js               # Explorer frontend logic
│   └── style.css
├── data/
│   ├── icon-d2/             # Full Germany grid
│   │   └── {run}/
│   │       └── {step:03d}.npz  # All 66 vars per file
│   └── icon-eu/             # Cropped to D2 bounds
│       └── {run}/
│           └── {step:03d}.npz
└── docs/
    └── FULL-INGEST-PLAN.md  # This file
```

### Config-Driven Ingestion (`ingest_config.yaml`)

```yaml
region:
  icon-d2: full          # Use native grid (no crop)
  icon-eu: crop-to-d2    # Crop to ICON-D2 bounds (43.18-58.08°N, -3.94-20.34°E)

retention:
  keep_runs: 1           # Latest run only

groups:
  aviation: true
  surface_weather: true
  severe_weather: true
  soaring: true

pressure_levels:
  variables: [u, v]
  levels: [950, 850, 700, 500, 300]
```

---

## Implementation Plan

### Phase 1: Extended Ingest + Skyview Adaptation (~2-3 days)

1. **Create `ingest_config.yaml`** with variable group definitions
2. **Refactor `ingest.py`**:
   - Read config to determine which variables to download
   - Remove hardcoded BOUNDS for ICON-D2 (use full grid)
   - Keep crop logic for ICON-EU (crop to D2 extent)
   - Update retention: delete previous run after new run completes
   - Handle variables that don't exist in ICON-EU (graceful skip)
3. **Update ICON-EU variable mapping** for new variables
4. **Update `cron-ingest.sh`** — same 10-min polling, larger download set
5. **Adapt Skyview `app.py` for full grid**:
   - Replace hardcoded `BOUNDS` with grid extent read from data (lat/lon arrays in .npz)
   - Add bbox slicing before processing: all endpoints must crop to requested bbox *before* computing (symbols aggregation, overlay rendering, classification) — avoids 15x overhead from full grid
   - Use `np.load(..., mmap_mode='r')` or selective key loading to avoid reading all 66 vars when only ~15 are needed for Skyview
   - Verify `/api/symbols` bbox filtering works with larger grid
   - Verify `/api/overlay` PNG rendering crops before rasterizing
   - Verify `/api/point` nearest-neighbor lookup uses correct lat/lon arrays
6. **Adapt `classify.py`** — accept pre-cropped arrays or add internal bbox crop to avoid classifying the full 746×1215 grid unnecessarily
7. **Update D2→EU handover** — EU now cropped to D2 bounds (not Alps). Verify `app.py` transition logic handles the different grid resolutions at the boundary
8. **Test** — full ingest cycle, verify Skyview still works correctly, check memory usage (~70 MB vs ~5 MB per request with full grid), verify disk usage

### Phase 2: Explorer App (~3-5 days)

1. **Backend (`explorer/app.py`)**:
   - `GET /api/variables` — list available variables with metadata (name, unit, group, min/max)
   - `GET /api/overlay` — render any variable as color-mapped PNG (auto-scale or manual range)
   - `GET /api/point` — raw value at lat/lon for all variables at a timestep
   - `GET /api/timeseries` — values over all timesteps at a point
   - `GET /api/timesteps` — available runs and timesteps (reuse from Skyview)
   - Share data directory with Skyview — read-only access to same .npz files

2. **Frontend (`explorer/index.html` + `app.js`)**:
   - Leaflet map, same base layer as Skyview
   - Variable picker (dropdown, grouped by category)
   - Color scale: auto-range with manual override, palette selector (sequential/diverging)
   - Time slider (same concept as Skyview timeline)
   - Click-to-inspect: show raw value + coordinates
   - Model info bar (same as Skyview)
   - Optional: side-by-side comparison mode (two variables or two timesteps)

3. **Deployment**:
   - Run on port 8502 alongside Skyview (8501)
   - Same cron, same data, independent UI

### Phase 3: Refinements (ongoing)

- Variable metadata enrichment (descriptions, typical ranges, color scale presets)
- Export functionality (GeoTIFF, CSV at point)
- Cross-link from Skyview to Explorer ("view raw data" button)
- Additional variable groups (radiation, soil) as needed
- Performance optimization for large grid rendering

---

## Key Design Decisions

1. **Full D2 grid, no crop** — covers all of Germany + neighbors. ICON-EU cropped to match.
2. **Latest run only** — no historical data stored. Keeps disk at ~4 GB.
3. **Curated 66 variables** — not all 130+. Focused on aviation, weather, severe weather. Expandable via config.
4. **Shared data store** — Skyview and Explorer read the same .npz files. No duplication.
5. **Config-driven** — variable groups toggled in YAML. Easy to add/remove without code changes.

---

## ICON-EU Variable Mapping (extended)

New variables to map (in addition to existing):

| D2 Name | EU Name | Notes |
|---------|---------|-------|
| `cin_ml` | `cin_ml` | Same |
| `clct_mod` | `clct_mod` | Same |
| `cldepth` | — | May not exist in EU |
| `vis` | `vis` | Same |
| `hzerocl` | `hzerocl` | Same |
| `snowlmt` | `snowlmt` | Same |
| `tmax_2m` | `tmax_2m` | Same |
| `tmin_2m` | `tmin_2m` | Same |
| `relhum_2m` | `relhum_2m` | Same |
| `pmsl` | `pmsl` | Same |
| `ps` | `ps` | Same |
| `vmax_10m` | `vmax_10m` | Same |
| `tot_prec` | `tot_prec` | Same |
| `grau_gsp` | — | May not exist in EU |
| `h_snow` | `h_snow` | Same |
| `freshsnw` | `freshsnw` | Same |
| `t_g` | `t_g` | Same |
| `dbz_*` | — | Radar reflectivity likely D2 only |
| `sdi_2` | — | Supercell index likely D2 only |
| `vorw_ctmax` | — | Likely D2 only |
| `w_ctmax` | — | Likely D2 only |
| `uh_max*` | — | Likely D2 only |
| `echotop` | — | May not exist in EU |
| `tcond*` | — | May not exist in EU |

**Note:** Several severe weather variables are ICON-D2 specific (high-res convection-resolving). The Explorer should handle missing EU variables gracefully (show D2-only badge).
