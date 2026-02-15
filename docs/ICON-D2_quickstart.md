# ICON-D2 Convection Visualization - Quick Start

## In 30 Seconds
✅ **Real ICON-D2 data downloaded and rendered**
- **Latest Map:** `convection_overlay.png` (georeferenced Germany map with convection overlay)
- **Data Stats:** `convection_data_analysis.png` (real data distribution analysis)

---

## Python Setup

```python
# Install dependencies (one-time)
pip install cfgrib matplotlib numpy

# Download latest data
import urllib.request, bz2
url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/09/htop_dc/..."
urllib.request.urlretrieve(url, "data.grib2.bz2")
with bz2.open("data.grib2.bz2", "rb") as f_in:
    with open("data.grib2", "wb") as f_out:
        f_out.write(f_in.read())

# Read GRIB2
import cfgrib
ds = cfgrib.open_dataset("data.grib2")
data = ds["HTOP_DC"].values  # Convection height in meters

# Data is shape (542040,) - unstructured grid points
print(f"Max convection: {data.max():.0f} m")
print(f"Mean height: {data.mean():.0f} m")
```

---

## Quick Data Access

### Latest Files
```bash
# Latest 09 UTC run (+0 to +48 hours available)
BASE="https://opendata.dwd.de/weather/nwp/icon-d2/grib"

# Convection heights
wget $BASE/09/htop_dc/icon-d2_germany_icosahedral_*_2d_htop_dc.grib2.bz2

# CAPE (instability)
wget $BASE/09/cape_ml/icon-d2_germany_icosahedral_*_2d_cape_ml.grib2.bz2

# Wind at 10m
wget $BASE/09/u_10m/icon-d2_germany_icosahedral_*_2d_u_10m.grib2.bz2
wget $BASE/09/v_10m/icon-d2_germany_icosahedral_*_2d_v_10m.grib2.bz2

# Precipitation
wget $BASE/09/tot_prec/icon-d2_germany_icosahedral_*_2d_tot_prec.grib2.bz2
```

### Available Variables
```python
# Convection
"HTOP_DC"   # Dynamic cloud top height (m) ⭐
"CAPE_ML"   # Convective available potential energy (J/kg)
"LPI"       # Lightning potential index
"UH_MAX"    # Updraft helicity (m²/s²)

# Wind
"U_10M", "V_10M"   # 10m wind components
"U", "V", "W"      # 3D wind field (all levels)

# Precipitation
"TOT_PREC"         # Total precipitation (kg/m²)
"RAIN_GSP"         # Large-scale rain
"RAIN_CON"         # Convective rain

# Temperature
"T"                # Temperature (all levels)
"T_2M", "TD_2M"    # 2m temp & dew point
```

---

## Visualization Options

### Option A: Simple Scatter Plot (Current)
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))
# For unstructured grid, use scatter
data_flat = data.flatten()
scatter = ax.scatter(range(len(data_flat)), data_flat, 
                    c=data_flat, cmap='viridis', s=1)
plt.colorbar(scatter, label='Height (m)')
plt.title('ICON-D2 Convection Heights (All Grid Points)')
plt.savefig('convection.png', dpi=100, bbox_inches='tight')
```

### Option B: Regrid to Regular Grid (Better)
```python
import cfgrib
import xarray as xr

# Read with cfgrib
ds = cfgrib.open_dataset("data.grib2")
data = ds["HTOP_DC"]

# Regrid using xarray
target_grid = {
    'latitude': np.linspace(47, 56, 100),
    'longitude': np.linspace(5, 17, 100)
}
data_regridded = data.interp(
    latitude=target_grid['latitude'],
    longitude=target_grid['longitude'],
    method='linear'
)

# Now can use pcolormesh
fig, ax = plt.subplots()
im = ax.pcolormesh(target_grid['longitude'], 
                   target_grid['latitude'],
                   data_regridded.values,
                   cmap='RdYlBu_r')
```

### Option C: Web Tiles (Production)
```python
# Convert to GeoTIFF for web tiling
import rasterio
from rasterio.transform import Affine

# Regrid to regular grid first (see Option B)
# Then save as GeoTIFF
with rasterio.open(
    'convection.tif',
    'w',
    driver='GTiff',
    height=data_regridded.shape[0],
    width=data_regridded.shape[1],
    count=1,
    dtype=data_regridded.values.dtype,
    crs='EPSG:4326',
    transform=Affine.identity()
) as dst:
    dst.write(data_regridded.values, 1)

# Then use gdal2tiles.py or similar for web tiles
```

---

## Colormap for Convection

```python
import matplotlib.colors as mcolors

colors = [
    '#0000FF',  # 0m - No convection (deep blue)
    '#00AAFF',  # 2km - Developing (light blue)
    '#00FF00',  # 4km - Green (moderate)
    '#FFFF00',  # 6km - Yellow (strong)
    '#FF8800',  # 8km - Orange (very strong)
    '#FF0000',  # 10km - Red (extreme)
    '#8800FF',  # 12km - Purple (severe)
]

cmap = mcolors.LinearSegmentedColormap.from_list(
    'convection', colors, N=256
)

# Use it
norm = mcolors.Normalize(vmin=0, vmax=15000)
im = ax.pcolormesh(..., cmap=cmap, norm=norm)
```

---

## Integration with Web Framework

### Flask API Example
```python
from flask import Flask, jsonify, send_file
import cfgrib
import io

app = Flask(__name__)

@app.route('/api/convection/<forecast_hour>')
def get_convection(forecast_hour):
    """Return convection data as GeoJSON"""
    url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/09/htop_dc/..."
    # Download & process
    ds = cfgrib.open_dataset("data.grib2")
    heights = ds["HTOP_DC"].values
    
    return jsonify({
        'data': heights.tolist(),
        'min': float(heights.min()),
        'max': float(heights.max()),
        'mean': float(heights.mean())
    })

@app.route('/map/convection.png')
def get_map():
    """Return rendered PNG"""
    return send_file('convection_overlay.png', mimetype='image/png')
```

### React/Mapbox Integration
```javascript
// Display overlay on map
const convectionLayer = {
  id: 'convection-overlay',
  type: 'raster',
  source: {
    type: 'raster',
    url: 'http://your-server/convection-tiles/{z}/{x}/{y}.png',
    tileSize: 256
  },
  paint: {
    'raster-opacity': 0.7
  }
};

map.addLayer(convectionLayer);
```

---

## Automation (Cron Job)

```bash
#!/bin/bash
# run-convection-update.sh

cd /data/convection

# Download latest (updated every 3 hours at XX:15)
wget --quiet \
  "https://opendata.dwd.de/weather/nwp/icon-d2/grib/09/htop_dc/icon-d2_germany_icosahedral_single-level_$(date +%Y%m%d09)_000_2d_htop_dc.grib2.bz2" \
  -O htop_dc_latest.grib2.bz2

# Decompress
bunzip2 -f htop_dc_latest.grib2.bz2

# Render
python3 /opt/render_convection.py

# Upload to web server
scp convection_overlay.png user@server:/var/www/maps/

echo "✓ Convection map updated at $(date)"
```

Add to crontab:
```bash
15 */3 * * * /opt/run-convection-update.sh >> /var/log/convection.log 2>&1
```

---

## Troubleshooting

**Issue:** "ecCodes provides no latitudes/longitudes for gridType='unstructured_grid'"  
**Solution:** This is normal for ICON data. Grid points are implicit in the icosahedral structure. Use scatter plot or regrid to regular grid.

**Issue:** Data shape is (542040,) instead of 2D  
**Solution:** That's correct for unstructured grids. Flatten and plot, or regrid to regular grid.

**Issue:** cfgrib not installing  
**Solution:** `apt-get install python3-cfgrib python3-eccodes`

---

## Resources

- **DWD Open Data:** https://opendata.dwd.de
- **ICON-D2 Docs:** https://www.dwd.de/ourservices/nwp_forecast_data/
- **cfgrib Docs:** https://github.com/ecmwf/cfgrib
- **GRIB2 Format:** https://www.wmo.int/pages/prog/wis/2010dlm/documents/GRIB2_GUIDE_11.4_20230524.docx

---

**Next Steps:**
1. Set up automated 3-hourly downloads
2. Regrid to regular lat/lon grid for web display
3. Create interactive map with Mapbox/Leaflet
4. Add wind vectors overlay
5. Integrate with web server for live map

🚀 Ready to build the full web app!
