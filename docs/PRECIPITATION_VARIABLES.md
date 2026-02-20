# ICON-D2 Precipitation Variables
**Complete reference for Skyview**

---

## Available Variables (DWD ICON-D2)

### 1. Total Precipitation (Accumulated)
**`tot_prec`** — Total precipitation (all types combined)
- **Units:** kg/m² (equivalent to mm)
- **Type:** Accumulated since run start
- **Use case:** Overall precipitation amount forecast
- **Example values:** 0-15+ mm per timestep
- **Coverage:** ~47% of grid points in typical forecast

---

### 2. Precipitation by Type (Accumulated)

**`rain_gsp`** — Grid-scale (stratiform) rain
- Steady, widespread rain from large-scale weather systems
- **Units:** kg/m²

**`rain_con`** — Convective rain
- Showers, thunderstorms
- More localized, intense
- **Units:** kg/m²

**`snow_gsp`** — Grid-scale snow
- **Units:** kg/m²

**`snow_con`** — Convective snow
- Snow showers
- **Units:** kg/m²

---

### 3. Precipitation Rates (Instantaneous)

**`prr_gsp`** — Large-scale rain rate
- **Units:** kg/m²/s (multiply by 3600 for mm/h)
- **Typical range:** 0-0.01 kg/m²/s (0-36 mm/h)
- **Example:** Max observed 22 mm/h in sample forecast
- **Use case:** Rainfall intensity at specific time

**`prs_gsp`** — Large-scale snow rate
- **Units:** kg/m²/s

**`prg_gsp`** — Large-scale graupel (soft hail) rate
- **Units:** kg/m²/s

---

### 4. Snow Coverage

**`snowc`** — Snow cover fraction
- **Units:** 0-1 (0% to 100%)
- Current snow on ground

**`snowlmt`** — Snow fall limit
- **Units:** m above sea level
- Elevation where rain transitions to snow

---

### 5. Weather Code (Currently Used)

**`ww`** — Significant weather (WMO 4680)
- **Units:** Integer code (0-99)
- **Type:** Categorical
- **Includes:** All precipitation types + intensity + special conditions
- **Codes:**
  - 0-3: Clear to overcast (no precip)
  - 51-57: Drizzle (light/moderate/dense, freezing)
  - 61-67: Rain (slight/moderate/heavy, freezing)
  - 71-77: Snow (slight/moderate/heavy, grains)
  - 80-86: Showers (rain/snow)
  - 95-99: Thunderstorms (with/without hail)

---

## Current Skyview implementation (2026-02)

- Precip overlays (`total_precip`, `rain`, `snow`, `hail`) use **precomputed ingest fields** derived from amount fields via de-accumulation between consecutive timesteps.
- Reported values are normalized to **mm/h-equivalent** by dividing with timestep length `Δt`.
  - `Δt=1h` for hourly steps
  - `Δt=3h` for ICON-EU long-range steps (`>=81`)
- Sources:
  - `total_precip`: `tot_prec`
  - `rain`: `rain_gsp + rain_con`
  - `snow`: `snow_gsp + snow_con`
  - `hail`: `grau_gsp` (if unavailable: treated as 0)
- Optional missing ICON-EU variables are now **skipped** in ingest (no zero-fill placeholders).

## Recommendations for Skyview

### Current Implementation
✅ Using `ww` for precipitation overlay (categorical, includes intensity)

### Potential Enhancements

**Option A: Quantitative Precipitation**
- Add `tot_prec` layer → show actual mm amounts
- Color scale: 0mm (transparent) → 1mm (light blue) → 10mm (dark blue) → 50mm+ (purple)
- More informative for planning (exact amounts vs. categories)

**Option B: Rainfall Intensity**
- Add `prr_gsp` layer → show instantaneous rain rate in mm/h
- Useful for "how heavy is the rain right now"
- Complements ww (which is categorical)

**Option C: Convective vs. Stratiform**
- Compare `rain_con` vs. `rain_gsp` to identify shower/thunderstorm risk
- Helpful for soaring: convective rain = unstable air = potential thermals

**Option D: Snow Forecast**
- Add `snow_gsp` + `snowlmt` for winter soaring
- Show where/when snow is expected

---

## Data Characteristics (2026-02-10 15 UTC +1h sample)

| Variable | Coverage | Max Value | Notes |
|----------|----------|-----------|-------|
| tot_prec | 47% of grid | 15.5 mm | Widespread light precip |
| prr_gsp | 38% of grid | 22 mm/h | Peak intensity moderate |
| ww | 100% coverage | Codes 0-82 | Most comprehensive |

---

## Next Steps

1. **Decide priority:** Which layer(s) would be most useful for soaring forecasts?
2. **Implement:** Add to `VARIABLES` list in `ingest.py`
3. **Colormap:** Design appropriate scale (e.g., 0-50mm for tot_prec)
4. **UI:** Add to overlay selector (radio buttons or multi-select)

---

**References:**
- DWD Open Data: https://opendata.dwd.de/weather/nwp/icon-d2/grib/
- WMO Code 4680: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
