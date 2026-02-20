# Phase 1 — Explorer Provider Abstraction

## Objective
Prepare Explorer for on-demand remote data access without breaking current local NPZ mode.

## Implemented

### New module
- `explorer/data_provider.py`
  - `ExplorerDataProvider` protocol
  - `LocalNpzProvider` (current behavior)
  - `RemoteProviderStub` (placeholder for upcoming remote mode)
  - `build_provider(...)` factory

### Explorer backend integration
- `explorer/app.py`
  - uses provider wrappers for:
    - `load_data(...)`
    - `get_available_runs()`
    - `get_merged_timeline()`
    - `resolve_time(...)`
  - provider selected via env:
    - `EXPLORER_DATA_PROVIDER=local_npz` (default)
    - `EXPLORER_DATA_PROVIDER=remote` (stub, returns 503)
  - `/api/health` reports `provider` mode.

## Why this matters
All overlay/point/tile endpoints now depend on a provider interface instead of hard-wired local file logic. This is the required seam for Phase 2 (real remote fetch + caching).

## Runtime notes
Default behavior is unchanged.

To test remote-stub wiring:
```bash
EXPLORER_DATA_PROVIDER=remote python3 app.py
```
Endpoints should return `503` with a clear message that remote mode is not implemented yet.
