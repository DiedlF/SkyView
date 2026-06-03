# Skyview Documentation

This directory contains active project notes, operational runbooks, implementation plans, and reference material for Skyview. Historical snapshots live in `archive/`.

Last updated: 2026-04-26

## Current Source Of Truth

- `../TODO.md` - Current roadmap, completed work, and prioritized open tasks.
- `../SPEC.md` - Product and API specification. Broad architectural reference; some examples are illustrative.
- `../backend/ARCHITECTURE.md` - Short current backend module map.
- `PERFORMANCE_RECOMMENDATIONS_2026-04-26.md` - Current performance follow-up plan.
- `SYMBOLS_15MIN_IMPLEMENTATION_PLAN_2026-04-13.md` - Current design note for the next symbol-timing feature.

## Operations

- `OPS_SECRETS.md` - Marker auth, admin auth, and CORS environment variables.
- `LOGGING.md` - Logging format, locations, and maintenance.
- `github-actions-watch.md` - GitHub Actions failure watcher usage.
- `REVERSE_PROXY_TILE_CACHE.md` - Nginx/Caddy guidance for overlay tile caching.
- `TESTING_CHECKLIST.md` - Focused manual overlay-positioning regression checklist.
- `EXPLORER_REMOTE_CANARY_RUNBOOK.md` - Explorer remote-provider canary procedure.

## Architecture And API

- `API_CONVERGENCE_CONTRACT.md` - Explorer/Skyview API compatibility contract.
- `ICON_EU_OUTSIDE_D2_IMPLEMENTATION_PLAN.md` - D2-to-EU fallback implementation status.
- `OBSERVATION_LAYER_IMPLEMENTATION_PLAN.md` - Plan for adding live radar (OPERA) + satellite (MSG RSS) observation overlays (backend + frontend).
- `ICON-EU-IMPLEMENTATION.md` - ICON-EU integration summary and operational notes.
- `EXPLORER_SOURCE_API_CONTRACT_V1.md` - Remote Explorer source API contract.
- `EXPLORER_REMOTE_PROVIDER.md` - Consolidated Explorer local/remote provider and ingest-profile summary.
- `EXPLORER_MIGRATION_EXECUTION_CHECKLIST.md` - Explorer migration checklist.

## Data And Model Reference

- `ICON-D2_data_research.md` - Initial ICON-D2 data access research.
- `ICON-D2_quickstart.md` - Hands-on ICON-D2 GRIB2 examples and tooling notes.
- `DATA_PIPELINE_RESEARCH.md` - DWD data availability and fast-polling strategy.
- `PRECIPITATION_VARIABLES.md` - Precipitation variable semantics and implementation notes.
- `reference/dwd-icon/README.md` - Downloaded DWD ICON reference bundle index.
- `SkyView_12.01.15_Manual_DE.pdf` - Historical SkyView manual.
- `icon_database_main.pdf` - DWD ICON database reference.
- `Lightning Potential for ICON.pdf` - LPI reference material.

## Benchmarks And Performance

- `PERFORMANCE_RECOMMENDATIONS_2026-04-26.md` - Current performance recommendations and verification criteria.
- `PRECOMPUTED_SYMBOLS_BENCHMARK_2026-03-11.md` - VPS benchmark for low-zoom symbol precompute; current JSON-bin precompute remains opt-in.

## Archive

Historical reviews, phase notes, baselines, prototype reports, and superseded implementation plans live in `archive/`. Start with `archive/ARCHIVE_INDEX.md` if archaeology is needed.

## Remaining Documentation Backlog

These are the main documentation-related items still worth improving:

1. Add synthetic `.npz` fixture documentation once TestClient integration tests exist.
2. Expand local developer setup notes with exact install/test/lint/run commands.
3. Keep `SPEC.md` aligned with current endpoint behavior, especially newer overlays and substep behavior.
4. Keep this index updated whenever a new top-level doc is added.
