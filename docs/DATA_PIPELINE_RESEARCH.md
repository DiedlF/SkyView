# ICON-D2 Data Pipeline Research
**Date:** 2026-02-10

## Summary
DWD ICON-D2 data is published faster than our current cron expects. We're fetching ~2.5h old data when we could get near-realtime forecasts.

---

## Model Overview

### ICON-D2 (Deterministic)
- **Domain:** Germany + neighboring countries
- **Resolution:** 2.2 km (native icosahedral), 0.02° regular-lat-lon (~2 km)
- **Forecast horizon:** 48 hours
- **Temporal resolution:** 15 minutes (first 6h), 1h (6-48h for most vars)
- **Run frequency:** Every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)

### ICON-D2-EPS (Ensemble)
- Same domain/resolution as ICON-D2
- Probabilistic forecasts (ensemble members)
- Also available on opendata.dwd.de

---

## Data Availability Timing

Based on analysis of actual DWD opendata.dwd.de files:

| Event | Time offset | Example (12 UTC run) |
|-------|-------------|----------------------|
| Run starts | +0 min | 12:00 UTC |
| First files (t=0h) | **+44 min** | 12:44 UTC |
| Full run complete (t=48h) | **+80 min** | 13:20 UTC |

**Key findings:**
- Icosahedral and regular-lat-lon grids published **simultaneously** (within seconds)
- No advantage to using icosahedral for availability
- Files are published incrementally as model computes

---

## Current vs Optimal Scheduling

### Current Setup
```cron
0 */3 * * *  # Runs at 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
```

**Problem:** Fetches the **previous** run's data, which is ~2h20min old by the time cron executes.

Example:
- 12:00 UTC run starts
- 15:00 UTC cron fires → fetches 12 UTC data (already 3h old, but published at ~13:20)
- Actually getting data that's been available for ~1h40min

### Optimal Schedule Options

#### Option A: Early Bird (Partial Forecast)
```cron
50 */3 * * *  # Runs at 00:50, 03:50, 06:50, etc.
```
- Gets first ~10 timesteps (0-2.5h forecast) **immediately** when available
- Full 48h forecast available 30 minutes later
- **Latency:** ~6 minutes (50 min after run start, files at +44 min)

#### Option B: Complete Forecast
```cron
25 1-23/3 * * *  # Runs at 01:25, 04:25, 07:25, 10:25, 13:25, 16:25, 19:25, 22:25
```
- Waits for full 48h forecast to be complete
- **Latency:** ~5 minutes (85 min after run start, complete at +80 min)
- More predictable (always complete data)

#### Option C: Two-Stage (Responsive + Complete)
```cron
50 */3 * * *   # Quick update with partial data
30 1-23/3 * * *  # Full update 40 min later
```
- Best user experience: near-instant initial update, full data follows
- Higher API/bandwidth usage

---

## Grid Type Comparison

| Feature | Icosahedral | Regular-lat-lon |
|---------|-------------|-----------------|
| Native resolution | 2.2 km | 0.02° (~2 km at 47°N = 1.5-2.2 km) |
| File size | Smaller (compressed) | Slightly larger |
| Processing complexity | High (triangular grid) | Low (rectangular) |
| Availability timing | Same (within seconds) | Same |
| Variables available | All | All |
| Leaflet integration | Requires regridding | Direct use |

**Decision:** Stick with **regular-lat-lon**. No availability advantage, much simpler processing.

---

## Recommendations

### Immediate (Priority 1)
**Change cron schedule to Option B (Complete Forecast):**
```cron
25 1-23/3 * * *
```
- Balances freshness (~5min latency) with completeness (full 48h)
- Users get forecasts 2+ hours sooner than current setup

### Future Enhancements (Priority 3)
1. **ICON-D2-EPS integration:** Add probabilistic forecasts (ensemble spread for uncertainty)
2. **ICON-EU fallback:** Use ICON-EU (6.5km, 120h forecast) for t+48h to t+120h
3. **Two-stage updates:** Implement Option C for super-responsive UX

### Data Retention
Current: Keep last 2 runs (~6h of data)
Consider: Keep last 8 runs (24h) for time-series analysis / trends

---

## Implementation Checklist

- [x] Update cron job schedule to `*/10 * * * *` (every 10 minutes)
- [x] Add `--check-only` fast mode to ingest.py (~0.6s vs 6-30s)
- [x] Optimize cron payload to check first, ingest only if needed
- [x] Update TODO.md with findings
- [x] Consider frontend "Data age" indicator (show run time + valid time)
- [ ] Monitor for missed runs or timing issues

## Final Implementation

**Fast polling strategy:**
- Check every 10 minutes with `--check-only` (0.6 second HEAD request)
- Full ingest (~60-120s) only runs when new data detected
- **Result:** New forecasts appear within 10 minutes of DWD publication
- **Cost:** 6 API calls/hour (all under 1 second except when downloading)

---

## References
- DWD OpenData Server: https://opendata.dwd.de/weather/nwp/icon-d2/
- ICON Database Reference: https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf
- DWD NWP Overview: https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html
