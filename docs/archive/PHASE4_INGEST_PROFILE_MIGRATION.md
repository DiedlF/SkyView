# Phase 4 Migration — Profile-based Ingest

## Goal
Reduce ingest scope for Skyview core while keeping:
- precipitation precompute (`tp_rate`, `rain_rate`, `snow_rate`, `hail_rate`)
- D2 boundary cache (`_d2_boundary.json`)

Explorer is allowed to degrade temporarily (reduced variable availability) until on-demand loading is implemented.

## Implemented Changes

### 1) `backend/ingest_config.yaml`
- Added ingest profiles:
  - `full` (legacy broad ingest)
  - `skyview_core` (minimal variable set for Skyview operation)

### 2) `backend/ingest.py`
- Added `--profile` CLI option.
- Profile-aware variable/static/pressure selection.
- `check-only` and variable URL checks now respect profile.
- Ingest + run-availability checks use profile-scoped variables.
- Precompute and D2 boundary cache remain unchanged.

### 3) `backend/cron-ingest.sh`
- Uses profile via env var:
  - `SKYVIEW_INGEST_PROFILE` (default: `skyview_core`)
- Passes `--profile` to all ingest/check invocations.

### 4) `explorer/app.py`
- `/api/variables` now reports only currently available vars by default.
- Optional `include_unavailable=true` to return full catalog with `available: false` entries.

## Operations

### Default (core mode)
```bash
cd skyview/backend
./cron-ingest.sh
```
(uses `skyview_core` unless overridden)

### Override profile
```bash
SKYVIEW_INGEST_PROFILE=full ./cron-ingest.sh
```

### Manual ingest examples
```bash
python3 ingest.py --model icon-d2 --profile skyview_core --steps all
python3 ingest.py --model icon-eu --profile skyview_core --steps all
```

### Rollback
Switch to full profile:
```bash
SKYVIEW_INGEST_PROFILE=full ./cron-ingest.sh
```
or for one-off runs add `--profile full`.

## Expected Impact
- Lower ingest bandwidth/CPU/storage.
- Explorer shows only variables present in current ingested data.
- No change intended for core Skyview overlays using retained variables.
