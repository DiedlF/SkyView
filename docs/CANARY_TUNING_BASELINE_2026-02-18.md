# Canary & Tuning Baseline — 2026-02-18

## What was executed
- Checked runtime provider endpoints on running Explorer instance:
  - `/api/health`
  - `/api/provider_health`
  - `/api/provider_stats`

## Observed
- Provider mode at runtime: `local_npz`
- Health: `ok` (remote thresholds not applicable in local mode)

## Interpretation
- Code default now prefers `remote`, but currently running process is still in local mode (or remote fallback due missing remote env/base URL).
- For meaningful canary metrics, restart Explorer with explicit remote env.

## Recommended canary env
```bash
EXPLORER_DATA_PROVIDER=remote
EXPLORER_REMOTE_BASE_URL=https://<source-host>
EXPLORER_REMOTE_SOURCE_TOKEN=<shared-secret>

EXPLORER_REMOTE_TIMEOUT_SECONDS=12
EXPLORER_REMOTE_META_TTL_SECONDS=30
EXPLORER_REMOTE_FIELD_TTL_SECONDS=180
EXPLORER_REMOTE_FIELD_CACHE_ITEMS=128
EXPLORER_REMOTE_DISK_CACHE_DIR=/tmp/explorer-remote-cache
EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS=600

EXPLORER_PROVIDER_WARN_ERROR_RATE=0.08
EXPLORER_PROVIDER_CRIT_ERROR_RATE=0.20
EXPLORER_PROVIDER_WARN_FETCH_MS=2500
EXPLORER_PROVIDER_CRIT_FETCH_MS=5000
EXPLORER_PROVIDER_WARN_CACHE_HIT_RATIO=0.35
```

## Acceptance targets
- warm fetch avg typically < 1000 ms
- cold fetch avg typically 1000–4000 ms
- sustained `provider_health=status=ok` (brief warning spikes acceptable)
