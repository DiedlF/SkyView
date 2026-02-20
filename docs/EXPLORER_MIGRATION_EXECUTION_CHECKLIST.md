# Explorer Migration — Execution Checklist (to target architecture)

## Current status
- ✅ Phase 4 ingest profile migration in place (`skyview_core` default for cron)
- ✅ Phase 1 provider abstraction in Explorer
- ✅ Phase 2 remote provider baseline (metadata + field fetch + cache)
- ✅ Phase 3 perf baseline (request coalescing + optional disk cache)
- ✅ Security/observability pass started (token auth + provider stats endpoint)

## Remaining plan (ordered)

### A) Security + observability hardening (0.5–1.5 days)
- [x] Add source endpoint auth token check (`X-Source-Token`)
- [x] Add remote-provider token forwarding (`EXPLORER_REMOTE_SOURCE_TOKEN`)
- [x] Add provider stats endpoint (`/api/provider_stats`)
- [x] Add structured remote-fetch logging + provider latency/error counters
- [x] Add threshold-based provider health state (`/api/provider_health`)
- [ ] Wire external alerting notifications from provider health/status

### B) Source contract stabilization (0.5–1 day)
- [x] Freeze endpoint contract for:
  - `/api/source/runs`
  - `/api/source/timeline`
  - `/api/source/resolve_time`
  - `/api/source/field.npz`
- [ ] Add explicit schema notes + compatibility guarantees

### C) On-demand model capability expansion (1–2 days)
- [x] Enrich metadata with per-model variable/level availability (`/api/capabilities`, `/api/source/capabilities`)
- [ ] Support icon-eu/global capability flags cleanly in UI and API
- [x] Keep graceful handling of unavailable variables (model-aware variable selector fallback)

### D) Explorer UX hardening for on-demand (1–2 days)
- [x] Better loading/error states in `app.js`
- [ ] Prefetch near-neighbor timeline slots/tiles (optional)
- [x] Improve user feedback when variables are temporarily unavailable

### E) Performance tuning + canary (1–2 days)
- [ ] Tune TTL/caches with real traffic
- [x] Canary runbook documented (`docs/EXPLORER_REMOTE_CANARY_RUNBOOK.md`)
- [ ] Measure warm/cold path:
  - warm target: <1s for common layer updates
  - cold target: ~1–4s typical
- [ ] Canary rollout with fallback toggle

### F) Cutover + cleanup (0.5–1 day)
- [ ] Default Explorer to remote mode in runtime config
- [ ] Keep rollback path documented (`EXPLORER_DATA_PROVIDER=local_npz`)
- [ ] Remove dead code paths after stabilization

## Ops variables (important)

### Source-side protection
- `EXPLORER_SOURCE_API_TOKEN=<secret>`

### Remote-side access
- `EXPLORER_DATA_PROVIDER=remote`
- `EXPLORER_REMOTE_BASE_URL=https://<source-host>`
- `EXPLORER_REMOTE_SOURCE_TOKEN=<same-secret>`

### Cache tuning
- `EXPLORER_REMOTE_FIELD_TTL_SECONDS`
- `EXPLORER_REMOTE_FIELD_CACHE_ITEMS`
- `EXPLORER_REMOTE_DISK_CACHE_DIR`
- `EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS`

## Success criteria
- Explorer can access broad model/variable space on demand.
- Core Skyview ingest remains minimal and stable.
- Cache hit ratio high enough to keep perceived latency low.
- Clear rollback and monitoring in place.
