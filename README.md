# Skyview

Skyview is a web-based convection and weather visualization tool for aviators. It ingests DWD ICON-D2 and ICON-EU forecast data, serves a FastAPI backend, and renders an interactive Leaflet frontend with weather symbols, overlays, markers, meteograms, and admin/ops views.

## Start Here

- `SPEC.md` - product/API overview and broader architecture reference.
- `TODO.md` - current roadmap, completed work, and prioritized open tasks.
- `docs/README.md` - documentation index and reading guide.
- `backend/ARCHITECTURE.md` - concise current backend module map.

## Main Directories

- `backend/` - FastAPI app, ingest pipeline, weather logic, caches, routers, and services.
- `frontend/` - static Leaflet frontend and assets.
- `data/` - runtime forecast/cache data; not source documentation.
- `docs/` - active docs, runbooks, plans, reference notes, and archived history.
- `tests/` - pytest coverage and integration/performance wrappers.
- `scripts/` - QA and operations helper scripts.

## Operational Notes

- Admin/private ops endpoints use HTTP Basic auth via `SKYVIEW_ADMIN_USER` and `SKYVIEW_ADMIN_PASSWORD`; see `docs/OPS_SECRETS.md`.
- Marker editing uses `SKYVIEW_MARKER_AUTH_SECRET`; missing or weak secrets disable marker editing.
- Production CORS must be configured with `SKYVIEW_CORS_ORIGINS`.
- Performance follow-up work is tracked in `docs/PERFORMANCE_RECOMMENDATIONS_2026-04-26.md`.

## Local Commands

Common project commands are documented in the relevant docs and scripts. The current high-level entry points are:

```bash
python3 backend/app.py
pytest
python3 scripts/qa_smoke.py
python3 scripts/qa_perf.py --base http://127.0.0.1:8501
```

Some integration/performance tests expect a running server and current local data.
