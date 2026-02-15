# ICON-D2 Data Access Research

## Executive Summary
✅ **ICON-D2 data is freely available**, no restrictions, and includes **convection height data**.

---

## Data Source & Access

### Primary Source: DWD Open Data Server
**URL:** `https://opendata.dwd.de/weather/nwp/icon-d2/`

- **Access:** Completely free, no authentication required
- **Format:** GRIB2 (standard meteorological format)
- **Compression:** Files compressed with bzip2 (.bz2)
- **License:** Public domain (German open data)

---

## ICON-D2 Model Specifications

| Property | Value |
|----------|-------|
| **Resolution** | 2.2 km (ultra-high for regional forecasts) |
| **Coverage** | Germany + Benelux, Switzerland, Austria, neighboring areas |
| **Vertical Levels** | 65 atmosphere levels |
| **Forecast Range** | 48 hours (+0 to +48 hours) |
| **Update Frequency** | 8 runs daily: 00, 03, 06, 09, 12, 15, 18, 21 UTC |
| **Time Steps** | Hourly forecasts |
| **Data Updated** | New forecasts every 3 hours |

---

## Available Convection Variables

### ✅ Convection Height Data (PRIMARY)
1. **`htop_dc`** – **Dynamic Cloud Top Height** ⭐ **(EXACT MATCH for SkyView)**
   - Height of the top of dynamic convection
   - Perfect replacement for SkyView's "convection height"
   - Files available: Compressed GRIB2, ~600-700 KB per forecast hour

2. **`htop_sc`** – Shallow Convection Top Height
   - Separate metric for shallow convection
   
3. **`hbas_sc`** – Shallow Convection Base Height
   - Height of cloud base for shallow convection

### ✅ Convection Instability Indices
4. **`cape_ml`** – **Convective Available Potential Energy (CAPE)**
   - Most-lifted parcel (ML) version
   - Key indicator of convective potential
   - Files: ~2-2.2 MB per hour (larger due to high variability)

5. **`cin_ml`** – Convective Inhibition (CIN)
   - Stability barrier to convection
   - Complement to CAPE analysis

### ✅ Radar/Reflectivity Proxies
6. **`dbz_ctmax`** – Composite Reflectivity Maximum
   - Simulated reflectivity at cloud top
   - Shows convective intensity

7. **`dbz_850`** – Reflectivity at 850 hPa level
   - Mid-level convective cell identification

8. **`echotop`** – Echo Top Height
   - Height where reflectivity falls below threshold
   - Another convection height metric

### ✅ Storm Severity Indices
9. **`lpi`** – Lightning Potential Index (hourly)
   - Thunderstorm severity indicator
   - Good for identifying dangerous convection

10. **`lpi_max`** – Maximum LPI
    - Highest potential lightning within forecast period

11. **`uh_max`**, **`uh_max_low`**, **`uh_max_med`** – Updraft Helicity
    - Rotation strength indicator
    - Used for severe weather/tornado potential

### ✅ Cloud Cover Data (by Level)
12. **`clcl`** – Low Cloud Cover (0-2000m)
13. **`clcm`** – Mid Cloud Cover (2000-7000m)
14. **`clch`** – High Cloud Cover (5000-13000m)
15. **`clct`** – Total Cloud Cover

### ✅ Wind Data
16. **`u_10m`, `v_10m`** – 10-meter Wind Components
17. **`vmax_10m`** – Wind Gust Maxima
18. **`u`, `v`, `w`** – 3D Wind Field (all pressure levels)

### ✅ Temperature Data
19. **`t`** – Temperature (all levels)
20. **`t_2m`** – 2-meter Temperature
21. **`td_2m`** – Dew Point (2m)

### ✅ Precipitation
22. **`tot_prec`** – Total Precipitation
23. **`rain_gsp`** – Large-scale Precipitation
24. **`rain_con`** – Convective Precipitation
25. **`grau_gsp`** – Graupel (hail-like ice)

---

## File Structure Example

### Recent Download (08 Feb 2026, 09 UTC Run)
```
Name: icon-d2_germany_icosahedral_single-level_2026020809_000_2d_htop_dc.grib2.bz2
Compressed: 564 KB
Uncompressed: 1.1 MB
Forecast Hour: 000 (Analysis time)
Variable: htop_dc (Dynamic cloud top height)
Grid: Icosahedral (native), can be regridded to regular lat/lon
```

**48-hour forecast provides:**
- Forecast hours 000-048 (49 timesteps total)
- 1 file per variable per forecast hour
- ~600KB-2.2MB per file (compressed)

---

## Data Format & Tools

### GRIB2 Format
- **Standard format** used by all major weather agencies
- **Binary** = efficient storage & transmission
- **Compressed** with bzip2 for further reduction

### Python Libraries for Reading
```python
# Most popular option
import cfgrib
ds = cfgrib.open_dataset('file.grib2')

# Alternative
import xarray as xr
ds = xr.open_dataset('file.grib2', engine='cfgrib')

# Low-level access
import eccodes
# Direct GRIB2 reading
```

### Command-Line Tools
```bash
# View GRIB2 contents
grib_dump file.grib2
wgrib2 file.grib2

# Convert to NetCDF
cdo -f nc copy file.grib2 file.nc

# Regrid from icosahedral to regular grid
cdo -f nc remapbil,target_grid.txt file.grib2 regridded.nc
```

---

## Data Access Strategy

### Option A: Direct Download (Simplest)
```bash
wget https://opendata.dwd.de/weather/nwp/icon-d2/grib/09/htop_dc/\
icon-d2_germany_icosahedral_single-level_2026020809_000_2d_htop_dc.grib2.bz2
bunzip2 file.grib2.bz2
```

### Option B: Python via HTTP
```python
import requests
import bz2

url = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/09/..."
response = requests.get(url)
data = bz2.decompress(response.content)
```

### Option C: Automated Sync
- Use `rsync` or `rclone` to mirror latest forecasts
- Set up cron job to download new runs automatically
- Example: Download all convection variables every 3 hours

---

## Comparison: ICON-D2 vs SkyView

| Feature | SkyView | ICON-D2 Available? |
|---------|---------|------------------|
| **Convection Height** | ✅ Yes | ✅ htop_dc (EXACT MATCH) |
| **Wind Visualization** | ✅ Yes | ✅ u_10m, v_10m, u, v, w |
| **Precipitation** | ✅ Yes | ✅ tot_prec, rain_gsp, rain_con |
| **Temperature Forecast** | ✅ Yes | ✅ t, t_2m, td_2m |
| **Cloud Cover** | ✅ Yes | ✅ clcl, clcm, clch, clct |
| **Grid Resolution** | 6.5 km | 2.2 km (**HIGHER**) |
| **Update Frequency** | 4 runs/day | 8 runs/day (**MORE FREQUENT**) |
| **Forecast Horizon** | 120 hours | 48 hours (acceptable for regional) |
| **CAPE/Stability** | ✅ Likely | ✅ cape_ml, cin_ml (EXPLICIT) |

---

## Next Steps for Implementation

### Phase 1: Data Pipeline
1. Set up automated download from DWD server
2. Decompress GRIB2 files
3. Parse with `cfgrib` or similar
4. Extract convection height + wind data
5. Store in efficient format (NetCDF or PostGIS)

### Phase 2: API Layer
1. Create REST API to query data by region/time
2. Return JSON for frontend consumption
3. Handle regridding from icosahedral to regular lat/lon

### Phase 3: Visualization
1. WebGL map rendering (Mapbox/Three.js)
2. Color-code convection heights
3. Overlay wind barbs/vectors
4. Interactive time slider for 48-hour forecast

### Phase 4: Enhancements
- Real-time alerts for high CAPE (> 2000 J/kg)
- Storm tracking using convection height changes
- Export formats (GeoTIFF, KML, etc.)

---

## Summary
✅ **Convection height data is 100% available** via DWD ICON-D2  
✅ **No access restrictions** – completely public  
✅ **High resolution** (2.2 km) – better than original SkyView  
✅ **Frequent updates** (every 3 hours vs. SkyView's 4x/day)  
✅ **Rich metadata** – CAPE, wind, precip all included  

**Recommendation:** Use ICON-D2 `htop_dc` as primary convection height layer, supplement with `cape_ml` for instability visualization.
