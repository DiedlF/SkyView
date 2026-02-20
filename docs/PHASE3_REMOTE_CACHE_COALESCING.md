# Phase 3 — Remote fetch coalescing + optional disk cache

## Implemented
File: `explorer/data_provider.py` (`RemoteProvider`)

### 1) In-flight request coalescing
- Added per-field in-flight map using `threading.Event`.
- If multiple requests ask for the same field key (`model/run/step/keys`) concurrently:
  - one request performs network fetch
  - others wait for completion and reuse cache result
- Wait timeout configurable via:
  - `EXPLORER_REMOTE_FIELD_WAIT_TIMEOUT_SECONDS` (default `20`)

### 2) Optional disk cache (L2)
- Added optional disk cache for remote field NPZ payloads.
- Enabled when env var is set:
  - `EXPLORER_REMOTE_DISK_CACHE_DIR=/path/to/cache`
- TTL configurable:
  - `EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS` (default follows field TTL)

Cache flow now:
1. L1 memory cache
2. L2 disk cache (optional)
3. remote fetch

### 3) Existing memory cache remains
- In-memory field TTL + max-item pruning still active:
  - `EXPLORER_REMOTE_FIELD_TTL_SECONDS`
  - `EXPLORER_REMOTE_FIELD_CACHE_ITEMS`

## Suggested runtime settings

```bash
EXPLORER_DATA_PROVIDER=remote
EXPLORER_REMOTE_BASE_URL=https://<source-host>
EXPLORER_REMOTE_FIELD_TTL_SECONDS=180
EXPLORER_REMOTE_FIELD_CACHE_ITEMS=128
EXPLORER_REMOTE_DISK_CACHE_DIR=/tmp/explorer-remote-cache
EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS=600
```

## Expected benefit
- Lower duplicate upstream fetches during tile bursts.
- Better warm performance across worker reloads/restarts when disk cache is enabled.
