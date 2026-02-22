# ICON variable notes for Skyview (DWD context)

This file extracts **Skyview-relevant** variable context from the downloaded DWD references and OpenData indexes.

Sources used in this note:
- `icon-d2-grib-00-index.html`
- `icon-eu-grib-00-index.html`
- `icon_description.html`
- `opendata.html`
- `icon_database_main.pdf` (downloaded as authoritative reference; full text extraction pending tool support)

---

## 1) Skyview critical variable availability (D2 vs EU)

### Present in both ICON-D2 and ICON-EU
- `tot_prec`
- `rain_gsp`
- `rain_con`
- `snow_gsp`
- `snow_con`
- `mh`
- `ceiling`
- `ww`
- `cape_ml`
- `htop_dc`
- `t_2m`
- `td_2m`
- `hsurf`
- `clcl`, `clcm`, `clch`, `clct`, `clct_mod`

### Present in D2, missing in EU (same shortName)
- `grau_gsp` (used for hail/graupel amount in Skyview)
- `hbas_sc`
- `htop_sc`
- `lpi`

### EU-only nearby/alternative-looking names seen in index
- `hbas_con`
- `htop_con`
- `lpi_con_max`
- `cape_con`

Implication: EU currently lacks 1:1 availability for some D2 convective diagnostics by shortName; harmonization must use explicit mapping policy.

---

## 2) Precipitation semantics used in Skyview

Current Skyview implementation (already migrated):
- Uses amount fields (`tot_prec`, `rain_gsp+rain_con`, `snow_gsp+snow_con`, `grau_gsp`) and computes de-accumulated rates.
- Stored as ingest-time precomputed fields:
  - `tp_rate`, `rain_rate`, `snow_rate`, `hail_rate`
- Normalization:
  - `Δt=1h` normally
  - ICON-EU long-range (`step >= 81`): `Δt=3h`
- Negative diffs clamped to 0.

Rationale: robust/consistent handling across D2/EU despite known field availability differences.

---

## 3) Boundary layer depth (`mh`) note

- `mh` is available in both D2 and EU indexes.
- The visual “3000 m sentinel” observed in map rendering is likely from Skyview color scaling (`mh` colormap cap at 3000m), not a DWD data sentinel.
- If desired, visualization range can be raised (e.g. 5000m) or made percentile-adaptive.

---

## 4) Recommended next extraction pass from DWD PDF

When PDF text extraction tooling is available, pull exact official definitions/units for:
- `tot_prec`, `rain_gsp`, `rain_con`, `snow_gsp`, `snow_con`, `grau_gsp`
- `mh`, `hbas_sc`, `htop_sc`, `hbas_con`, `htop_con`
- `lpi`, `lpi_con_max`
- `ceiling`, `ww`

and append them here with quoted snippets/page references.

---

## 5) Comparison study: `hbas_sc/htop_sc` (D2) vs `hbas_con/htop_con` (EU)

### Metadata-level interpretation
- D2 `hbas_sc`: `Cloud base above MSL, shallow convection`
- D2 `htop_sc`: `Cloud top above MSL, shallow convection`
- EU `hbas_con`: `Height of Convective Cloud Base above msl`
- EU `htop_con`: `Height of Convective Cloud Top above msl`

All are in meters above MSL, but semantics differ:
- D2: explicitly shallow convection
- EU: broader convective cloud representation

### Validation snapshot (current local runs)
- D2 run: `2026021812`
- EU run: `2026021806`
- Common timesteps: `48` (`2026-02-18T13:00:00Z` → `2026-02-20T12:00:00Z`)
- Points tested (EU points inside D2 extent): `4,462,608`

Occurrence (valid where finite and `top > base`):
- D2: `1.39%`
- EU: `27.49%`
- Both valid: `0.93%`

On overlapping valid points (`EU - D2`):
- Base bias: `+45.2 m`, base MAE: `504.6 m`
- Top bias: `+758.8 m`, top MAE: `1309.7 m`
- Thickness bias: `+713.6 m`, thickness MAE: `1111.3 m`
- Correlation: base `0.41`, top `0.166`, thickness `0.14`

Conclusion: usable as fallback proxy only; not 1:1 equivalent.

---

## 6) Comparison study: `lpi_max` (D2) vs `lpi_con_max` (EU)

> **Note:** Skyview switched from D2 `lpi` (instantaneous) to `lpi_max` (1-hour rolling maximum)
> in Feb 2026. The variable stored in NPZ is still keyed as `lpi`; the change is handled via
> `d2_variable_map` in `ingest_config.yaml`. The original comparison below used D2 `lpi`
> (instantaneous) and is preserved for reference. A fresh comparison with `lpi_max` vs
> `lpi_con_max` would be more meaningful — both are hourly-window maxima.

### Metadata-level interpretation
- D2 `lpi_max`: `Maximum Lightning Potential Index` (1-hour rolling max; paramId `503142`)
- EU `lpi_con_max`: `Maximum Lightning Potential Index from convection scheme` (paramId `503673`)

Both are `J kg-1` and both represent a maximum over a time window, making them semantically closer
than the old instantaneous D2 `lpi` vs EU `lpi_con_max` pairing.

### Legacy validation snapshot (D2 `lpi` instantaneous vs EU `lpi_con_max`)
- Points tested: `4,090,724`
- Finite overlap: `83.14%`
- Bias (`EU - D2`): `+0.207`
- MAE: `0.207`
- RMSE: `1.485`
- Correlation: `0.038`

Threshold exceedance comparison (instantaneous D2 `lpi`):
- `>1`: D2 `0.00%`, EU `5.70%` (Jaccard `0.001`)
- `>2`: D2 `0.00%`, EU `3.81%` (Jaccard `0.000`)
- `>5`: D2 `0.00%`, EU `1.61%` (Jaccard `0.000`)
- `>7`: D2 `0.00%`, EU `1.00%` (Jaccard `0.000`)
- `>10`: D2 `0.00%`, EU `0.56%` (Jaccard `0.000`)

Additional distribution signal (instantaneous):
- D2 `lpi` p99 median: `0.0`
- EU `lpi_con_max` p99 median: `5.11`

> TODO: Re-run comparison with D2 `lpi_max` — expected to show higher D2 exceedance rates
> and improved Jaccard overlap with EU `lpi_con_max`.

Conclusion: `lpi_con_max` is a fallback proxy; `lpi_max` (D2) vs `lpi_con_max` (EU) is a
better-matched pair than the previous `lpi` (instantaneous) vs `lpi_con_max` pairing.

---

## 7) Practical guidance for Skyview

1. Keep D2-native shortNames as primary semantics baseline.
2. For EU, keep explicit compatibility mappings where names differ (`*_sc` vs `*_con`, `lpi_max` vs `lpi_con_max`) but label them internally as proxy semantics.
3. For D2, prefer time-window maxima over instantaneous fields where available (`lpi_max` over `lpi`, etc.) — handled via `d2_variable_map` in `ingest_config.yaml`.
3. Keep precip on ingest-precomputed de-accum rates for performance + consistency.
4. Treat missing optional EU vars as missing (no silent zero-fill).
5. If symbol parity is required, add explicit EU-side normalization/calibration layer (not raw threshold reuse).
