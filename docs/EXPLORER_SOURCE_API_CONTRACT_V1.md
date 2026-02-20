# Explorer Source API Contract v1

This contract defines compatibility requirements for Explorer remote provider mode.

## Auth
- Header: `X-Source-Token`
- Required when source server sets `EXPLORER_SOURCE_API_TOKEN`.

## Endpoints

### 1) `GET /api/source/runs`
Returns available runs list (same semantics as `time_contract.get_available_runs`).

### 2) `GET /api/source/timeline`
Returns merged timeline object (same semantics as `time_contract.get_merged_timeline`).

### 3) `GET /api/source/resolve_time`
Query:
- `time`: string (`latest` or ISO time)
- `model`: optional model id

Response JSON:
```json
{ "run": "YYYYMMDDHH", "step": 12, "model": "icon_d2" }
```

### 4) `GET /api/source/field.npz`
Query:
- `run`: string run id (`YYYYMMDDHH`)
- `step`: integer step
- `model`: model id (`icon_d2` / `icon_eu`)
- `keys`: optional comma-separated variables

Response:
- `application/octet-stream`
- NPZ containing at least:
  - `lat` (1D float array)
  - `lon` (1D float array)
  - `validTime` (scalar string array)
  - requested variables if present in source data

## Introspection
- `GET /api/source/contract` returns machine-readable contract metadata.

## Compatibility guarantees
- Existing field names and response shape are backward compatible within `v1`.
- New optional keys may be added without breaking clients.
- Breaking changes require contract version bump.
