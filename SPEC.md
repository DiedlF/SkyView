# Skyview — Convection Visualization for Aviators

## Overview
Modern web-based replacement for the classic Skyview (Flash-based). Shows weather symbols on an adaptive grid over a hillshade map, powered by DWD ICON-D2 and ICON-EU data. Provides real-time forecast visualization with multi-layer overlays including convection symbols, precipitation, significant weather, cloud cover, and thermal indicators.

## Architecture
```
DWD Open Data (GRIB2) → Python ingester → NumPy cache → FastAPI → Leaflet frontend
```

**Status:** ✅ Production-ready with automated data updates

## Directory Structure
```
skyview/
├── SPEC.md                 # This file
├── TODO.md                 # Task tracking
├── backend/
│   ├── app.py              # FastAPI application (REST API + static file serving)
│   ├── ingest.py           # GRIB2 download + processing (ICON-D2 & ICON-EU)
│   ├── classify.py         # Cloud type classification
│   ├── cron-ingest.sh      # Automated ingestion script (runs every 10min)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Main page
│   ├── app.js              # Leaflet map + UI logic
│   ├── symbols.js          # SVG symbol definitions
│   └── style.css           # Styling
├── data/                   # Runtime cache (.npz files, ~200MB/run)
│   ├── icon-d2/            # ICON-D2 forecasts (48h, 2.2km resolution)
│   └── icon-eu/            # ICON-EU forecasts (120h, 6.5km resolution)
└── docs/                   # Documentation & research notes
    ├── ICON-D2_data_research.md
    ├── ICON-D2_quickstart.md
    ├── DATA_PIPELINE_RESEARCH.md
    ├── ICON-EU-IMPLEMENTATION.md
    ├── PRECIPITATION_VARIABLES.md
    └── SkyView_12.01.15_Manual_DE.pdf
```

## Data Pipeline (ingest.py)

### Dual-Model Architecture
**ICON-D2** (High-resolution, short-range) + **ICON-EU** (Medium-resolution, extended-range)

#### ICON-D2
- **Source**: DWD open data, regular-lat-lon GRIB2 format
- **URL pattern**: `https://opendata.dwd.de/weather/nwp/icon-d2/grib/{RUN}/{VAR}/`
- **Resolution**: 2.2 km
- **Runs**: 00, 03, 06, 09, 12, 15, 18, 21 UTC (8 per day)
- **Forecast range**: 1–48 hours (hourly steps)
- **Publication delay**: ~2 hours after run time
- **Update frequency**: Automated check every 10 minutes (fast polling with HEAD requests)

#### ICON-EU
- **Source**: DWD open data, regular-lat-lon GRIB2 format
- **URL pattern**: `https://opendata.dwd.de/weather/nwp/icon-eu/grib/{RUN}/{VAR}/`
- **Resolution**: 6.5 km
- **Runs**: 00, 06, 12, 18 UTC (4 per day)
- **Forecast range**: 49–120 hours (hourly 49-78, then 3-hourly 81-120)
- **Publication delay**: ~4 hours after run time
- **Note**: Only hours 49+ are ingested (ICON-D2 covers 1-48 at higher resolution)

### Variables Downloaded
All variables are from **regular-lat-lon** single-level files (no regridding needed):

| Variable | ICON-D2 Name | ICON-EU Name | Description |
|----------|--------------|--------------|-------------|
| `ww` | `ww` | `ww` | Significant weather (WMO code) |
| `cape_ml` | `cape_ml` | `cape_ml` | CAPE - Mixed Layer (J/kg) |
| `htop_dc` | `htop_dc` | `htop_dc` | Dynamic convective cloud top (m) |
| `hbas_sc` | `hbas_sc` | `hbas_con` | Convection base height (m) |
| `htop_sc` | `htop_sc` | `htop_con` | Convection top height (m) |
| `lpi` | `lpi` | `lpi_con_max` | Lightning potential index |
| `ceiling` | `ceiling` | `ceiling` | Cloud ceiling (m) |
| `clcl` | `clcl` | `clcl` | Low cloud cover (%) |
| `clcm` | `clcm` | `clcm` | Mid cloud cover (%) |
| `clch` | `clch` | `clch` | High cloud cover (%) |
| `clct` | `clct` | `clct` | Total cloud cover (%) |
| `prr_gsp` | `prr_gsp` | `rain_gsp` | Rain rate - grid-scale (kg/m²/h) |
| `prs_gsp` | `prs_gsp` | `snow_gsp` | Snow rate - grid-scale (kg/m²/h) |
| `prg_gsp` | `prg_gsp` | `rain_con` | Graupel/hail rate (kg/m²/h) |

### Processing Steps
1. **Fast polling**: Check every 10 minutes via HTTP HEAD request (0.6s, minimal overhead)
2. **Download trigger**: When new data detected, download all variables for all timesteps
3. **Parallel download**: Uses `curl` with retry logic and timeouts
4. **Decompression**: bz2 → GRIB2 (automatic)
5. **Grid extraction**: 
   - Read regular-lat-lon data from GRIB2 using `cfgrib`
   - Crop to region bounds (45.5-48.5°N, 9-17°E)
   - No regridding needed (already regular grid)
6. **Classification**: Run cloud type classifier on cropped data
7. **Storage**: Save as compressed NumPy `.npz` files in `data/{model}/{run}/{step:03d}.npz`
8. **Cleanup**: Delete forecasts older than 2 runs (auto-cleanup)

### Automation
- **Cron schedule**: `*/10 * * * *` (every 10 minutes)
- **Script**: `backend/cron-ingest.sh`
- **Strategy**: Fast HEAD check → full ingest only when new data available
- **Result**: New forecasts appear within ~10 minutes of DWD publication
- **Previous latency**: 2.5 hours → **Optimized to**: <10 minutes

## Cloud Type Classification (classify.py)

For grid cells where ww ∈ {0, 1, 2, 3} (no significant weather), derive cloud type:

### Decision Tree
```
1. Is there convective activity? (cape_ml > 50)
   ├─ NO → stratiform
   │   ├─ clcl > 30% AND clcm < 20% → St (Stratus)
   │   ├─ clcm > 30% AND clcl < 20% → Ac (Altocumulus)
   │   └─ clcl < 10% AND clcm < 10% AND clch > 30% → Ci (Cirrus)
   │
   └─ YES → convective
       ├─ hbas_sc = 0 (no cloud formed yet) → Blue Thermal
       ├─ lpi > 0 OR (htop_sc - hbas_sc) > 4000 OR cape_ml > 1000 → Cb (Cumulonimbus)
       ├─ (htop_sc - hbas_sc) > 500 → Cu con (Cumulus congestus)
       └─ else → Cu hum (Cumulus humilis)
```

### Cloud Base Height
- Use `hbas_sc` where available
- `htop_dc` for blue thermals
- Display as hectometers MSL (e.g., ceiling 1500m → "15")
- Round to nearest hectometer

## Symbol Grid (symbols.py)

### Zoom-Adaptive Grid
- At each zoom level, divide the visible area into cells
- Cell size decreases with zoom (symbols must not overlap)
- Suggested cell sizes:
  - Zoom 5:
  - Zoom 6:
  - Zoom 7: ~0.5° (~50km)
  - Zoom 8: ~0.25° (~25km)
  - Zoom 9: ~0.12° (~12km)
  - Zoom 10: ~0.06° (~6km)
  - Zoom 11: ~0.03° (~3km)
  - Zoom 12: ~0.02° (~2km, native resolution)

### "Worst Weather Wins" Aggregation
For each cell, aggregate all grid points within it:
1. Find the maximum ww code in the cell (worst weather)
2. If max ww > 3: use that ww code's symbol
3. If max ww ≤ 3: use the cloud type classification
   - Priority: Cb > Cu con > Cu hum > St > Ac > Blue Thermal > Clear
   - Take the "worst" (most significant for aviation) cloud type
4. Cloud base: use MINIMUM cloud base in cell (conservative for aviation)

### WMO ww Code Priority (higher = worse)
95–99: Thunderstorm (worst)
71–77: Snowfall
85–86: Snow showers
66-67: Freezing rain
56-57: Freezing drizzle
61–65: Rain
80–82: Rain showers
51–55: Drizzle
45–48: Fog
3: Overcast
2: Partly cloudy
1: Mostly clear
0: Clear (least significant)

## API Endpoints (app.py)

FastAPI backend serves both REST API and static frontend files.

### GET /api/models
Returns available forecast models and their capabilities.

**Response:**
```json
{
  "models": [
    {
      "name": "icon-d2",
      "label": "ICON-D2 (2.2km)",
      "timesteps": [1, 2, 3, ..., 48],
      "maxHours": 48
    },
    {
      "name": "icon-eu",
      "label": "ICON-EU (6.5km)",
      "timesteps": [49, 50, ..., 78, 81, 84, 87, ..., 120],
      "maxHours": 120
    }
  ]
}
```

### GET /api/symbols
Returns symbol data for the visible area at the current zoom level.

**Query params:**
- `zoom` (int): Map zoom level (7-12)
- `bbox` (string): "lat_min,lon_min,lat_max,lon_max"
- `time` (string): ISO timestamp "2026-02-09T12:00Z" or forecast hour (int)
- `run` (string, optional): Specific model run "2026020906"
- `model` (string, optional): "icon-d2" or "icon-eu" (auto-selected if omitted)

**Response:**
```json
{
  "symbols": [
    {
      "lat": 47.5,
      "lon": 12.0,
      "type": "cu_hum",
      "ww": 2,
      "cloudBase": 15,
      "label": "15"
    }
  ],
  "run": "2026020906",
  "validTime": "2026-02-09T12:00:00Z",
  "cellSize": 0.25,
  "model": "icon-d2"
}
```

### GET /api/timesteps
Returns available forecast timesteps across all models.

**Query params:**
- `run` (string, optional): Specific run, or latest

**Response:**
```json
{
  "runs": {
    "icon-d2": "2026020906",
    "icon-eu": "2026020900"
  },
  "timesteps": [
    {"step": 1, "validTime": "2026-02-09T07:00:00Z", "model": "icon-d2"},
    {"step": 2, "validTime": "2026-02-09T08:00:00Z", "model": "icon-d2"},
    ...
    {"step": 48, "validTime": "2026-02-10T06:00:00Z", "model": "icon-d2"},
    {"step": 49, "validTime": "2026-02-10T07:00:00Z", "model": "icon-eu"},
    ...
    {"step": 120, "validTime": "2026-02-14T06:00:00Z", "model": "icon-eu"}
  ]
}
```

### GET /api/point
Returns point information for a clicked location. A response can include a
symbol and/or wind values and/or one selected overlay value.

**Query params:**
- `lat`, `lon` (float): Coordinates
- `time` (string): ISO timestamp or forecast hour
- `model` (string, optional): Force specific model
- `wind_level` (string, optional): Wind source (`10m`, `gust10m`, or pressure level like `850`)
- `include_symbol` (bool, optional): Include symbol/cloud classification data
- `include_wind` (bool, optional): Include wind speed/direction in `overlay_values`
- `include_overlay` (bool, optional): Include one overlay value in `overlay_values`
- `overlay_key` (string, optional): Overlay identifier (for example `ceiling`, `rain`, `climb_rate`)
- `zoom` (int, optional): Current map zoom, used for a few derived point values

**Response:**
```json
{
  "lat": 47.6836,
  "lon": 11.961,
  "ww": 2,
  "symbol": "cu_hum",
  "cloudType": "cu_hum",
  "cloudTypeName": "Cu_Hum",
  "clcl": 35.2,
  "clcm": 5.1,
  "clch": 0.0,
  "clct": 40.3,
  "cape_ml": 320.5,
  "htop_dc": 1800.0,
  "hbas_sc": 1500.0,
  "htop_sc": 2600.0,
  "lpi_max": 0.0,
  "ceiling": 1500.0,
  "values": {
    "ww": 2,
    "clcl": 35.2,
    "clcm": 5.1,
    "clch": 0.0,
    "clct": 40.3,
    "cape_ml": 320.5,
    "htop_dc": 1800.0,
    "hbas_sc": 1500.0,
    "htop_sc": 2600.0,
    "lpi_max": 0.0,
    "ceiling": 1500.0
  },
  "overlay_values": {
    "wind_speed": 12.4,
    "wind_dir": 230,
    "ceiling": 1500.0
  },
  "validTime": "2026-02-09T12:00:00Z",
  "run": "2026020906",
  "model": "icon_d2",
  "sourceModel": "icon_d2",
  "diagnostics": {
    "dataFreshnessMinutes": 42,
    "fallbackDecision": "primary_model_only",
    "requestedModel": null,
    "requestedTime": "latest",
    "sourceModel": "icon_d2",
    "euDataMissing": false
  }
}
```

**Notes:**
- `overlay_values` is sparse: it only includes requested wind values and/or the selected overlay.
- `cloudBaseHm` has been removed from the point payload.
- Point overlay `ceiling` now comes directly from the clicked grid point; no separate `ceil_cell` path is used.

### GET /api/overlay
Generates PNG overlay for specified layer type.

**Query params:**
- `layer` (string): "precipitation", "sigwx", "clouds", or "thermals"
- `time` (string): ISO timestamp or forecast hour
- `bbox` (string): "lat_min,lon_min,lat_max,lon_max"
- `width`, `height` (int): Image dimensions in pixels
- `model` (string, optional): Force specific model

**Response:**
- Content-Type: `image/png`
- PNG image with transparency (Base64-encoded in frontend)
- Geographic bounds encoded in response headers

### Static Files
- `/` → `frontend/index.html`
- `/app.js`, `/style.css`, `/symbols.js` → Frontend assets
- CORS enabled for all origins (development/production)
```

## Frontend (index.html + app.js)

### Map Setup
- **Library**: Leaflet 1.9.4
- **Base layer**: ESRI World Hillshade (no labels, clean topographic view)
- **Bounds**: Eastern Alps (45.5-48.5°N, 9-17°E), maxBoundsViscosity 1.0
- **Zoom levels**: 5-12
  - Z5
  - Z6
  - Z7: 55km grid spacing
  - Z8: 28km
  - Z9: 13km
  - Z10: 7km
  - Z11: 3km
  - Z12: 2km (native resolution)
- **Initial view**: Geitau (47.6836°N, 11.9610°E) at zoom 9

### Layer System
**Multi-layer architecture with toggle controls:**

1. **Convection Symbols Layer** (checkbox, default ON)
   - Zoom-adaptive grid of weather symbols
   - Fetches from `/api/symbols` on pan/zoom
   - SVG markers (~32px) with cloud base labels (hectometers)
   - "Worst weather wins" aggregation per grid cell
   
2. **Overlay Layers** (radio buttons, one active at a time)
   - **None** (default)
   - **Precipitation**: Rain/snow/hail rates (blue/pink/red color scale)
   - **Significant Weather**: WMO ww codes (pixelated, color-coded)
   - **Cloud Cover**: Total cloud fraction (white-to-gray scale)
   - **Thermals**: CAPE visualization (green-yellow-red heat map)

### Overlay Rendering
- Server-side PNG generation from NumPy arrays
- Base64-encoded, transparency-masked
- Positioned as Leaflet `ImageOverlay` with correct geographic bounds
- Auto-refresh on time change or layer toggle
- Smooth transitions (no flicker)

### Time Controls
**Horizontal scrollable timeline** (redesigned 2026-02-10):
- Each timestep is a clickable button
- Inline date labels at day boundaries
- Current time highlighted
- ±1 hour navigation buttons (◀ 1h / 1h ▶)
- Auto-scrolls to keep current time visible
- Shows model transition (D2 → EU at hour 48/49)
- Handles missing timesteps gracefully (hours 79-80 for ICON-EU)

### Info Panel
Top-right corner shows:
- **Model**: ICON-D2 or ICON-EU (with resolution)
- **Run time**: Model initialization timestamp
- **Valid time**: Forecast valid time
- **Zoom/Grid**: Current zoom level + grid spacing

### Interactive Features
- **Click on symbol**: Popup with detailed forecast (ww, cloud type, CAPE, cloud cover, etc.)
- **Geitau marker**: Red circle marker at reference location
- **Touch support**: Mobile-friendly (pinch-zoom, tap, swipe)
- **Debounced updates**: Prevents API spam during rapid pan/zoom

### SVG Symbols (symbols.js)
Comprehensive symbol library with WMO/ICAO-style icons:
- **Cloud types**: clear, blue_thermal, cu_hum, cu_con, cb, st, ac
- **Precipitation**: drizzle (light/moderate/dense), rain (slight/moderate/heavy), snow (slight/moderate/heavy)
- **Showers**: rain_shower, snow_shower
- **Severe**: thunderstorm, cb (cumulonimbus)
- **Obscuration**: fog, rime_fog
- **Freezing**: freezing_rain, freezing_drizzle
- **Special**: snow_grains

All symbols use crisp SVG paths, optimized for rendering at multiple zoom levels.

### Responsive Design
- **Desktop**: Full-width map, side panel for layers
- **Mobile**: Touch-optimized controls, collapsible layer panel
- **Performance**: Efficient symbol batching, minimal redraws

## Configuration

### Region Bounds
```python
BOUNDS = (45.5, 48.5, 9.0, 17.0)  # lat_min, lat_max, lon_min, lon_max
# Eastern Alps: Germany (south), Austria, Switzerland (east), northern Italy
```

### Server Settings
```python
HOST = "0.0.0.0"
PORT = 8501
DATA_DIR = "/root/.openclaw/workspace/skyview/data"
```

### Ingestion Settings
- **Fast poll interval**: 10 minutes (HTTP HEAD check only)
- **Full ingest trigger**: When new run detected
- **Retention**: Keep last 2 model runs, auto-delete older
- **Parallel downloads**: 8 concurrent `curl` processes
- **Timeout per file**: 120 seconds
- **Compression**: bz2 (automatic decompression)

## Installation & Deployment

### Dependencies
```bash
# System packages
apt-get install -y python3-cfgrib eccodes curl

# Python packages
pip install fastapi uvicorn cfgrib scipy numpy
```

### Initial Data Ingestion
```bash
# ICON-D2 (required, covers hours 1-48)
python3 backend/ingest.py --model icon-d2 --steps all

# ICON-EU (optional, extends to hour 120)
python3 backend/ingest.py --model icon-eu --steps all

# Check data
ls -lh data/icon-d2/  # Should show run directories (e.g., 2026021106/)
```

### Running the Server
```bash
# Development mode
cd /root/.openclaw/workspace/skyview
python3 backend/app.py
# → http://localhost:8501

# Production mode (with uvicorn)
uvicorn backend.app:app --host 0.0.0.0 --port 8501 --workers 2

# Background process
nohup uvicorn backend.app:app --host 0.0.0.0 --port 8501 > server.log 2>&1 &
```

### Automated Updates (Cron)
Add to crontab (`crontab -e`):
```bash
# Check for new ICON-D2 data every 10 minutes
*/10 * * * * /root/.openclaw/workspace/skyview/backend/cron-ingest.sh icon-d2 >> /var/log/skyview-d2.log 2>&1

# Check for new ICON-EU data every 10 minutes
*/10 * * * * /root/.openclaw/workspace/skyview/backend/cron-ingest.sh icon-eu >> /var/log/skyview-eu.log 2>&1
```

The cron script (`cron-ingest.sh`) performs:
1. Fast HEAD check to detect new runs
2. Full download only when new data is available
3. Automatic cleanup of old forecasts
4. Logging of all operations

### Verification
```bash
# Check if data is current
ls -lt data/icon-d2/ | head -5

# Test API endpoints
curl http://localhost:8501/api/models
curl http://localhost:8501/api/timesteps

# Check logs
tail -f /var/log/skyview-d2.log
```

## Performance Metrics
- **Data size**: ~200 MB per model run (48 timesteps × 14 variables)
- **Ingest time**: ~5-10 minutes (parallel download)
- **API response time**: <100ms for symbol queries
- **Overlay generation**: <500ms for PNG rendering
- **Frontend load time**: <2 seconds (first load)
- **Memory usage**: ~500 MB (with 2 cached runs)
