# Explorer ↔ Skyview API Convergence Contract (v1.0)

**Date:** 2026-02-16  
**Status:** Active baseline contract for both backends.

## Goal
Provide one stable API contract so Skyview and Explorer frontends can share semantics with thin adapters.

---

## Canonical endpoints

## 1) `GET /api/models`
Required response shape:
```json
{
  "models": [
    {"name":"icon-d2","label":"ICON-D2 (2.2km)","maxHours":48,"timesteps":[1,2],"resolution":2.2,"updateInterval":3},
    {"name":"icon-eu","label":"ICON-EU (6.5km)","maxHours":120,"timesteps":[49,50],"resolution":6.5,"updateInterval":6}
  ]
}
```

## 2) `GET /api/timesteps`
Required merged timeline fields:
- `merged.run`
- `merged.runTime`
- `merged.steps[]` with: `validTime`, `model`, `run`

## 3) `GET /api/overlay`
Canonical request params:
- `time`
- `bbox`
- `width`
- one of:
  - `layer` (semantic alias)
  - `var` (raw variable)

Required response headers:
- `X-Run`
- `X-ValidTime`
- `X-Model`
- `X-Bbox`

(plus optional backend-specific headers like `X-VMin`/`X-VMax`)

## 4) `GET /api/overlay_tile/{z}/{x}/{y}.png`
Canonical request params:
- one of `layer` or `var`
- `time`
- optional rendering params (`palette`, `vmin`, `vmax`, `clientClass`)

Required response headers:
- `X-Run`
- `X-ValidTime`
- `X-Model`
- `X-Cache`

## 5) `GET /api/point`
Canonical response fields:
- `lat`, `lon`, `validTime`, `run`, `model`
- `values` (raw variable key/value map)
- `overlay_values` (semantic values map)

Recommended point params:
- `lat`, `lon`, `time`, `model`
- optional: `wind_level`

---

## Layer alias baseline (semantic `layer` → raw `var`)

- `sigwx` -> `ww`
- `clouds_low` -> `clcl`
- `clouds_mid` -> `clcm`
- `clouds_high` -> `clch`
- `clouds_total` -> `clct`
- `clouds_total_mod` -> `clct_mod`
- `dry_conv_top` -> `htop_dc`
- `ceiling` -> `ceiling`
- `cloud_base` -> `hbas_sc`
- `thermals` -> `cape_ml`
- `rain` -> `prr_gsp`
- `snow` -> `prs_gsp`
- `hail` -> `prg_gsp`
- `total_precip` -> backend-specific computed/derived precipitation field

---

## Point payload field conventions

### `overlay_values` keys (current baseline)
- `total_precip`, `rain`, `snow`, `hail` -> mm/h
- `clouds_low`, `clouds_mid`, `clouds_high`, `clouds_total`, `clouds_total_mod` -> %
- `sigwx` -> WMO ww code
- `thermals` -> CAPE_ml (J/kg, raw value)
- `ceiling`, `cloud_base`, `dry_conv_top`, `conv_thickness` -> meters
- wind (if available):
  - `wind_speed` -> knots
  - `wind_dir` -> degrees (meteorological FROM)

### Numeric precision
- Recommended: 1 decimal where continuous values are displayed; integer for discrete codes.

---

## Compatibility matrix (implemented)

| Feature | Skyview | Explorer |
|---|---:|---:|
| `/api/models` | ✅ | ✅ |
| merged `/api/timesteps` | ✅ | ✅ |
| `/api/overlay?layer=...` | ✅ | ✅ |
| `/api/overlay_tile?...layer=...` | ✅ | ✅ |
| `/api/point` includes `values` | ✅ | ✅ |
| `/api/point` includes `overlay_values` | ✅ | ✅ |
| Overlay/tile required headers | ✅ | ✅ |

---

## Contract verification

Use:
```bash
python3 scripts/qa_contract.py
```

This validates both backends against this baseline contract.
