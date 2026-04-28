# Explorer Remote Provider

Current summary of Explorer's local/remote data-provider architecture and ingest-profile support. Detailed phase implementation notes are archived under `archive/`.

## Current Shape

Explorer reads weather fields through `explorer/data_provider.py` instead of hard-wired local NPZ calls.

Provider modes:

- `local_npz` - default local filesystem mode.
- `remote` - fetches model metadata and field NPZ payloads from a configured source API.

The provider abstraction is used by Explorer endpoint logic for:

- `load_data(...)`
- `get_available_runs()`
- `get_merged_timeline()`
- `resolve_time(...)`

## Remote Source API

Remote mode uses source endpoints exposed by an Explorer/Skyview-compatible source service:

- `GET /api/source/runs`
- `GET /api/source/timeline`
- `GET /api/source/resolve_time?time=...&model=...`
- `GET /api/source/field.npz?run=...&step=...&model=...&keys=a,b,c`

See `EXPLORER_SOURCE_API_CONTRACT_V1.md` for the wire contract.

## Cache Layers

Remote provider cache flow:

1. L1 in-memory metadata/field cache.
2. Per-field in-flight request coalescing for concurrent identical field fetches.
3. Optional L2 disk cache for remote field NPZ payloads.
4. Remote source fetch.

Useful environment variables:

```bash
EXPLORER_DATA_PROVIDER=remote
EXPLORER_REMOTE_BASE_URL=https://<source-host>
EXPLORER_REMOTE_TIMEOUT_SECONDS=12
EXPLORER_REMOTE_META_TTL_SECONDS=30
EXPLORER_REMOTE_FIELD_TTL_SECONDS=180
EXPLORER_REMOTE_FIELD_CACHE_ITEMS=128
EXPLORER_REMOTE_FIELD_WAIT_TIMEOUT_SECONDS=20
EXPLORER_REMOTE_DISK_CACHE_DIR=/tmp/explorer-remote-cache
EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS=600
```

## Ingest Profiles

Skyview ingest supports profiles via `backend/ingest_config.yaml` and `backend/ingest.py --profile`.

Profiles:

- `skyview_core` - default for operational Skyview ingest. Keeps core overlays, symbols, precipitation precompute, and D2 boundary cache.
- `full` - broader legacy ingest profile for full variable availability.

Operational override:

```bash
SKYVIEW_INGEST_PROFILE=full backend/cron-ingest.sh
```

Explorer variable listing reflects currently available variables by default. Use `include_unavailable=true` where supported to show the full catalog with availability flags.

## Canary Procedure

Use `EXPLORER_REMOTE_CANARY_RUNBOOK.md` for rollout checks and rollback steps. The historical canary/tuning baseline is archived at `archive/CANARY_TUNING_BASELINE_2026-02-18.md`.

## Archived Phase Notes

- `archive/PHASE1_EXPLORER_PROVIDER_ABSTRACTION.md`
- `archive/PHASE2_EXPLORER_REMOTE_PROVIDER.md`
- `archive/PHASE3_REMOTE_CACHE_COALESCING.md`
- `archive/PHASE4_INGEST_PROFILE_MIGRATION.md`
