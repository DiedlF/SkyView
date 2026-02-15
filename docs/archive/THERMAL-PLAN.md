# Plan: Thermal Strength, Cloud Base & Flying Distance

## Overview

Three interconnected features using ICON model forecast data to provide soaring-specific predictions, following established meteorological methods (RASP/BLIPMAP by Dr. Jack Glendening, soaringmeteo.org, Windy soaring layer).

---

## 1. Thermal Strength (W*)

### Method: Convective Velocity Scale (W-star)

The standard approach used by RASP/BLIPMAP and validated by the soaring community:

```
W* = [ (g / T₀) × Qₛ × D ]^(1/3)
```

Where:
- **g** = 9.81 m/s² (gravitational acceleration)
- **T₀** = mean boundary layer temperature (K)
- **Qₛ** = surface sensible heat flux (W/m² → K·m/s)
- **D** = boundary layer depth (m)

### ICON Variables Needed

| Variable | DWD Name | Description | Currently Ingested? |
|----------|----------|-------------|-------------------|
| Surface sensible heat flux | `ashfl_s` | Accumulated sensible heat flux at surface | ❌ **NEW** |
| Boundary layer height | `mh` | Mixed layer height (BL depth) | ❌ **NEW** |
| 2m temperature | `t_2m` | Temperature at 2m height | ❌ **NEW** |
| Surface temperature | `t_g` | Ground temperature | ❌ **NEW** |

### Calculation Steps

1. **Boundary layer depth (D):** Use `mh` directly from ICON (it provides mixed-layer height)
2. **Surface heating (Qₛ):** Use `ashfl_s` (accumulated sensible heat flux). Convert from accumulated (J/m²) to instantaneous rate (W/m²) by differencing successive timesteps, then to kinematic flux: `Qₛ_kinematic = Qₛ / (ρ × cₚ)` where ρ ≈ 1.225 kg/m³, cₚ = 1004 J/(kg·K)
3. **Mean temperature (T₀):** Approximate using `t_2m` (good enough for the buoyancy parameter)
4. **W-star:** Apply formula above
5. **Expected vario reading:** W* minus glider sink rate (configurable, ~1.0 m/s for paraglider, ~0.7 m/s for sailplane)

### Output
- **Thermal strength overlay**: Color map of W* (m/s), e.g. 0–5 m/s scale
- **Climb rate overlay**: W* minus configurable sink rate
- **Symbol enhancement**: Show expected climb rate on convection symbols

### Validation
- W* ≈ 1-2 m/s = weak thermals (early morning, winter)
- W* ≈ 2-3 m/s = moderate thermals (typical summer day)
- W* ≈ 3-5 m/s = strong thermals (hot summer day, good CAPE)
- Compare against existing CAPE overlay for consistency

---

## 2. Cloud Base Height (improved)

### Current State
We already have `ceiling` and `hbas_sc` (stratocumulus cloud base). These are model-direct outputs.

### Enhancement: LCL-based Cloud Base (Cumulus Cloudbase)

The **Lifting Condensation Level (LCL)** gives the theoretical cumulus cloud base — where rising air first condenses:

```
LCL_height ≈ 125 × (T₂ₘ - Td₂ₘ)   [meters above ground]
```

Where:
- **T₂ₘ** = 2m temperature (°C)
- **Td₂ₘ** = 2m dew point temperature (°C)

This is the **Espy formula** — simple but surprisingly accurate for convective situations.

### ICON Variables Needed

| Variable | DWD Name | Description | Currently Ingested? |
|----------|----------|-------------|-------------------|
| 2m temperature | `t_2m` | Temperature at 2m | ❌ **NEW** |
| 2m dew point | `td_2m` | Dew point at 2m | ❌ **NEW** |
| Surface elevation | `hsurf` | Terrain height (static, download once) | ❌ **NEW** |

### Calculation
1. `spread = t_2m - td_2m` (in °C)
2. `lcl_agl = 125 × spread` (meters above ground)
3. `lcl_amsl = lcl_agl + hsurf` (meters above sea level)
4. Compare with `mh` (BL height): if `lcl_amsl < mh` → cumulus expected, if `lcl_amsl > mh` → blue thermals (dry)

### Output
- **Cumulus cloud base overlay**: LCL height (colored, similar to ceiling)
- **Cu potential indicator**: LCL vs BL top comparison (green = Cu, blue = dry)
- Enhance existing convection symbols with LCL-derived cloud base

---

## 3. Flying Distance Estimate

### Method: Glide Cone from Thermal Top

Simple geometric calculation:

```
reachable_radius = (thermal_height_agl - safety_margin) × glide_ratio
```

### Parameters (user-configurable)

| Parameter | Paraglider | Sailplane | Hang Glider |
|-----------|-----------|-----------|-------------|
| Glide ratio | 9–11 | 30–50 | 12–16 |
| Sink rate | 1.0–1.2 m/s | 0.6–0.8 m/s | 0.9–1.1 m/s |
| Safety margin | 300m | 200m | 300m |
| Min thermalling height | 500m AGL | 300m AGL | 400m AGL |

### Calculation
1. **Usable height** = min(BL_top, LCL) - terrain_height - safety_margin
2. **Reachable distance** = usable_height × glide_ratio
3. **Wind correction**: Adjust for BL-average wind (extends range downwind, reduces upwind)
4. **XC potential score**: Combine W*, usable height, and inter-thermal distance

### Advanced: Inter-thermal Glide
More realistic XC estimate considering multiple thermals:
```
XC_distance = N_thermals × avg_thermal_spacing
N_thermals ≈ soarable_hours × 60 / avg_thermal_time
avg_thermal_time = climb_time + glide_time
climb_time = usable_height / (W* - sink_rate)
glide_time = avg_thermal_spacing / glide_speed
```

### Output
- **Reachable radius overlay**: Circle/ring around each grid point showing glide range
- **XC potential score**: Composite rating (1-5 stars or color scale)
- **User's location**: Show reachable area from their marker position

---

## Implementation Plan

### Phase 1: Additional Data Ingestion
**New variables to add to `ingest.py`:**
- `ashfl_s` — surface sensible heat flux
- `mh` — mixed layer height (BL depth)
- `t_2m` — 2m temperature
- `td_2m` — 2m dew point
- `hsurf` — surface elevation (static field, only step 000)

**Effort:** Small — just add to VARIABLES list, verify DWD availability, re-ingest.

### Phase 2: Backend Calculations
**New module: `soaring.py`**
- `calc_wstar(ashfl_s, mh, t_2m)` → W* array
- `calc_lcl(t_2m, td_2m, hsurf)` → cloud base array (m AMSL)
- `calc_cu_potential(lcl, mh)` → boolean Cu/dry array
- `calc_reachable(mh, lcl, hsurf, glide_ratio, safety_margin)` → radius array

**New API endpoints:**
- Extend `/api/overlay` with new layers: `wstar`, `climb_rate`, `lcl`, `cu_potential`, `xc_potential`
- Extend `/api/point` with soaring values

**Effort:** Medium — core calculation module + overlay integration.

### Phase 3: Frontend
- Add new overlay options to layer selector
- Aircraft type selector (paraglider/sailplane/hang glider) for glide ratio presets
- Show soaring values on symbol click popup
- Reachable radius visualization (optional, more complex)

**Effort:** Small-medium — mostly adding overlay options.

### Phase 4: Validation & Tuning
- Compare W* predictions against pilot reports
- Tune proportionality constant if needed
- Validate LCL cloud base against actual observations
- Adjust safety margins and glide ratios based on feedback

---

## DWD Variable Availability Check

All required variables confirmed available for both models:

| Variable | ICON-D2 | ICON-EU | Format |
|----------|---------|---------|--------|
| `ashfl_s` | ✅ single-level | ✅ single-level | Accumulated J/m² |
| `mh` | ✅ single-level | ✅ single-level | Meters |
| `t_2m` | ✅ single-level | ✅ single-level | Kelvin |
| `td_2m` | ✅ single-level | ✅ single-level | Kelvin |
| `hsurf` | ✅ single-level (step 0 only) | ✅ single-level | Meters |

---

## Summary

| Feature | Key Formula | New Variables | Complexity |
|---------|------------|---------------|------------|
| Thermal strength (W*) | `[(g/T₀)×Qₛ×D]^(1/3)` | ashfl_s, mh, t_2m | Medium |
| Cloud base (LCL) | `125 × (T - Td)` | t_2m, td_2m, hsurf | Low |
| Flying distance | `height × glide_ratio` | (uses above) | Low-Medium |

**Total new DWD variables: 5** (ashfl_s, mh, t_2m, td_2m, hsurf)
**Estimated implementation time: 2-3 sessions**
