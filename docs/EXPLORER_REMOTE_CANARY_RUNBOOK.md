# Explorer Remote Canary Runbook

## Goal
Validate remote provider mode against production-like traffic before default cutover.

## 1) Configure source (upstream)
Set on source Explorer instance:

```bash
EXPLORER_SOURCE_API_TOKEN=<shared-secret>
```

## 2) Configure canary Explorer instance

```bash
EXPLORER_DATA_PROVIDER=remote
EXPLORER_REMOTE_BASE_URL=https://<source-host>
EXPLORER_REMOTE_SOURCE_TOKEN=<shared-secret>

# Suggested initial cache tuning
EXPLORER_REMOTE_FIELD_TTL_SECONDS=180
EXPLORER_REMOTE_FIELD_CACHE_ITEMS=128
EXPLORER_REMOTE_DISK_CACHE_DIR=/tmp/explorer-remote-cache
EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS=600
```

## 3) Verify endpoints
- `GET /api/health`
- `GET /api/provider_stats`
- `GET /api/provider_health`
- `GET /api/capabilities`

## 4) SLO targets for canary
- Warm interactions: typically < 1s
- Cold interactions: typically 1–4s
- Provider health status should stay `ok` (occasional brief `warning` tolerated)

## 5) Watch signals
From `/api/provider_stats`:
- `stats.http_errors`
- `stats.fetch_avg_ms`
- `stats.fetch_max_ms`
- `stats.field_mem_hits`, `stats.field_disk_hits`, `stats.field_cache_misses`

From `/api/provider_health`:
- `status` (`ok`/`warning`/`critical`)
- `reasons`

## 6) Rollback
Switch back instantly:

```bash
EXPLORER_DATA_PROVIDER=local_npz
```

No data migration needed for rollback.
