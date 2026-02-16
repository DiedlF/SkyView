# Skyview TODO List (Consolidated)

**Last updated:** 2026-02-16 13:55 UTC  
**Status:** ✅ P0/P1 complete. Active backlog is P2 feature delivery + project ops.

---

## ✅ Completed Milestones

- P0 QA/hardening complete (manual + automated baseline + QA report)
- P1 backend modularization + convergence complete
  - shared helpers/modules for overlays, symbols, point payloads, time/timelines, grid slicing, headers, status/perf, feedback, model caps
  - Explorer/Skyview converged contract paths validated (`qa_contract.py`)
- Performance and operability improvements delivered
  - symbols/tile caching improvements
  - run/step-aware cache rotation
  - SigWX LUT optimization
  - tile prewarm (viewport + ring)
- Symbol aggregation priority update delivered
  - Prio 1: `WW > 10`
  - Prio 2: `hbas_sc > 0 && cape_ml > 50`
  - Prio 3: `htop_dc` with `cape_ml > 50`
  - hectometer label now follows the winning symbol point altitude
- Observability baseline live
  - `/api/status`
  - request correlation IDs (`X-Request-Id`)
  - frontend degraded-data indicator
- Reverse-proxy tile caching guidance documented
  - `docs/REVERSE_PROXY_TILE_CACHE.md`

---

## 🚀 Open Tasks

## E. Product Features (P2)
- [ ] Airspace structure overlay (OpenAir)
- [ ] Persistent per-user custom location markers + recommendations
- [ ] Feedback notifications (Telegram/email)
- [ ] Feedback/log admin view (simple web UI)
- [ ] Evaluate ingest/storage optimization via reduced precision persistence
- [ ] Improve ICON-EU coverage around D2→EU transition for problematic variables

## F. Delivery & Project Ops (P2 / later)
- [ ] Push repository to GitHub
- [ ] Add CI pipeline (lint, style/type checks, smoke/regression/contract checks)

---

## 🧭 References

- Main roadmap: `docs/IMPROVEMENT_IDEAS_SUMMARY.md`
- P1 convergence audit: `docs/P1_CONVERGENCE_AUDIT_2026-02-16.md`
- Reverse proxy cache guide: `docs/REVERSE_PROXY_TILE_CACHE.md`

---
**Created:** 2026-02-09
