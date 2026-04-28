# Phase 2 — Explorer Remote Provider + Cache

## Goal
Enable Explorer to fetch model fields from a remote source per request (instead of only local NPZ), with basic caching.

## Implemented

## 1) Functional remote provider
File: `explorer/data_provider.py`

- Added `RemoteProvider` (replacing stub) with:
  - metadata cache (runs/timeline)
  - field cache (NPZ payloads per run/step/model/keys)
  - TTL controls via env vars
  - simple cache-size pruning
- Kept `LocalNpzProvider` as default/fallback.

### Remote provider env vars
- `EXPLORER_DATA_PROVIDER=remote`
- `EXPLORER_REMOTE_BASE_URL=https://<source-host>` (required in remote mode)
- `EXPLORER_REMOTE_TIMEOUT_SECONDS` (default `12`)
- `EXPLORER_REMOTE_META_TTL_SECONDS` (default `30`)
- `EXPLORER_REMOTE_FIELD_TTL_SECONDS` (default `120`)
- `EXPLORER_REMOTE_FIELD_CACHE_ITEMS` (default `48`)

## 2) Source endpoints for remote fetch
File: `explorer/app.py`

Added local-source endpoints used by `RemoteProvider`:
- `GET /api/source/runs`
- `GET /api/source/timeline`
- `GET /api/source/resolve_time?time=...&model=...`
- `GET /api/source/field.npz?run=...&step=...&model=...&keys=a,b,c`

These endpoints always read from local NPZ through a dedicated `source_provider`.

## 3) Provider seam remains global
All existing Explorer APIs continue to use provider wrappers:
- `load_data`
- `get_available_runs`
- `get_merged_timeline`
- `resolve_time`

So switching provider mode changes backend data source without touching endpoint logic.

## Notes
- This is a practical Phase 2 baseline (remote fetch + cache), not final optimized architecture.
- Next improvements:
  - request coalescing for concurrent identical field fetches
  - disk cache for remote fields
  - optional auth header/token for source endpoints
  - transport format optimization (e.g. zarr/chunked arrays)
