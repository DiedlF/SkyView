# Skyview TODO List (Consolidated)

**Last updated:** 2026-02-15 22:45 UTC  
**Status:** ✅ Core product works; now focus is robustness, consolidation, and scalable feature delivery.

---

## ✅ Completed (high-level)

- ICON-D2 + ICON-EU ingestion with automatic handover
- Fast polling ingest (<10 min latency)
- Multi-model timeline + API
- Skyview core map + overlays + wind + timeline + mobile
- Explorer raw-data app foundation
- Major symbol/overlay logic iteration (classification, clickability, gridding)
- Logging + docs consolidation

---

## 🚀 Open Tasks (Consolidated)

### A. QA & Hardening (P0)
- [ ] Full regression pass via `docs/TESTING_CHECKLIST.md`
- [ ] Add automated integration checks for known fragile cases:
  - [ ] z12 symbol continuity around Geitau / border panning
  - [ ] D2↔EU transition continuity (symbols/overlays/tooltips)
  - [ ] CAPE threshold rendering (50+ only) and tooltip value parity
  - [ ] Wind tooltip parity with selected wind level + zoom aggregation
- [ ] Eliminate multi-process/restart ambiguity (single-instance startup guard)

### B. Backend Refactor & API Convergence (P0/P1)
- [ ] Split `backend/app.py` into modules (symbols, overlays, point, cache, colormaps)
- [ ] Define unified Explorer/Skyview backend contract (timesteps, point schema, overlay metadata)
- [ ] Remove duplicated variable/format logic between Explorer and Skyview

### C. Performance & Scalability (P1)
- [ ] Add tile prewarm for current viewport (+ ring) on layer/time switch
- [ ] Strengthen cache strategy (run/step-aware invalidation + memory bounds)
- [ ] Add reverse-proxy tile caching option (Nginx/Caddy)
- [ ] Profile and vectorize any remaining expensive non-vectorized rendering paths

### D. Observability & Ops (P1)
- [ ] Add `/api/status` endpoint aggregating:
  - ingest freshness
  - current runs/models
  - cache stats
  - perf stats
  - recent error counters
- [ ] Add correlation/error IDs for frontend-visible failures
- [ ] Add lightweight frontend health indicator for repeated tile/API failures

### E. Product Features (P2)
- [ ] Airspace structure overlay (OpenAir)
- [ ] Persistent per-user custom location markers + recommendations
- [ ] Feedback notifications (Telegram/email)
- [ ] Feedback/log admin view (simple web UI)

### F. Delivery & Project Ops (P2)
- [ ] Push repository to GitHub
- [ ] Add CI pipeline (lint, type/style checks, smoke tests)

---

## 🧭 Implementation Plan Reference

See: `docs/IMPLEMENTATION_PLAN.md`

---
**Created:** 2026-02-09
