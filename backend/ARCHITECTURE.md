# Skyview Backend Architecture (Current)

## Entry point
- `app.py`
  - app/bootstrap, middleware, exception handlers
  - shared utilities/services (time/fallback/cache helpers)
  - mounts routers and static frontend

## Routers
- `routers/core.py`
  - `/api/health`
  - `/api/models`
  - `/api/timesteps`

- `routers/domain.py`
  - `/api/d2_domain`

- `routers/point.py`
  - `/api/point`

- `routers/weather.py`
  - `/api/symbols`
  - `/api/wind`

- `routers/overlay.py`
  - `/api/overlay`
  - `/api/overlay_tile/{z}/{x}/{y}.png`

- `routers/ops.py`
  - feedback endpoints
  - marker endpoints
  - location search

- `routers/admin.py`
  - status/cache/perf/usage/admin endpoints
  - `/admin`

## Shared modules
- `grid_aggregation.py` - fixed-grid/domain/grouping helpers used by symbols/wind
- `point_data.py` - point overlay/soaring data assembly
- `cache_state.py` - cache objects + metrics
- `time_contract.py` - run/timestep resolution and merged timeline

## Design notes
- EU fallback is strict-only (no nearby-timestep substitution)
- Missing EU timestep is surfaced via diagnostics (`euDataMissing`) and UI banner
- Fallback counters persist to `data/fallback_stats.json`
