# Skyview TODO List (Consolidated)

**Last updated:** 2026-02-15 11:00 UTC  
**Status:** ✅ Core complete, actively iterating on quality and productization

---

## ✅ Completed (high-level)

- ICON-D2 + ICON-EU ingestion with automatic handover
- Fast polling ingest (<10 min latency)
- Multi-model API + Explorer app foundation
- Full basemap/overlay stack (precip, ww, cloud cover, thermals, ceiling/cloud base)
- Wind barbs (multi-altitude), timeline UX, mobile support
- Feedback modal + backend storage
- Major map alignment and rendering bug fixes
- Logging + docs consolidation

---

## 🚀 Open Tasks (Alphabetic Task Groups)

### **A. Core Reliability & Interaction (P0)**
- [ ] Fix click-info failures at later timesteps ("Error loading details")
- [ ] Restrict clicks to valid symbol locations only
- [ ] Ensure popup shows exactly: clicked symbol values + active overlay values
- [ ] Eliminate occasional stale overlay persistence when switching layers

### **B. Symbol & Weather Logic (P0/P1)**
- [ ] Rework cloud classification per SPEC strategy:
  - no convection: dominant cloud layer type (low→St, mid→Ac, high→Ci)
  - with convection: detect blue thermals (dry convection top logic), classify Cu/Cb via vertical extent + CAPE
  - if convection exists but no `htop_dc`: default to Cu
- [ ] Ensure every map grid point has a symbol state (blank for clear sky), clickable where symbol exists
- [ ] Revisit and improve significant-weather (ww) prioritization logic
- [ ] Improve zoom-dependent symbol gridding:
  - high zoom: center symbols in data cells
  - lower zoom: progressive aggregation
  - preserve worst ww where ww > 10
  - cloud-only aggregation: use defined altitude/type fallback rules

### **C. Overlay Semantics & Visuals (P1)**
- [ ] Sig weather overlay: distinct colors for all ww classes, retain grouped palette feel, ww < 10 in gray tones
- [ ] Cloud cover overlay: use color gradient only (no opacity-based encoding)

### **D. Explorer App Stabilization & Parity (P1)**
- [ ] Fix explorer overlay reload behavior on zoom/pan
- [ ] Fix explorer click tooltip stale/multiple values when variable changes
- [ ] Phase 2b parity: copy key Skyview UX patterns (time selector, control placement, cleaner map chrome)
- [ ] Phase 3: generalize and combine Explorer/Skyview backend APIs

### **E. Product Features (P2)**
- [ ] Airspace structure overlay (OpenAir data, e.g., soaringweb)
- [ ] Persistent per-user custom location markers + recommendations
- [ ] Feedback notifications (Telegram/email)
- [ ] Feedback/log admin view (simple web UI)

### **F. Delivery & Project Ops (P2)**
- [ ] Push repository to GitHub

---

## 🧪 QA Gate (applies before/after A–D)

Use `docs/TESTING_CHECKLIST.md` as the execution checklist for:
- Overlay lock under zoom/pan
- D2↔EU transition stability
- Symbol/overlay consistency at click points
- Regression checks (timeline, symbols, controls, auto-refresh)

---

## Deferred / Historical (not active backlog)

Older unchecked TODOs in prototype/research docs are considered historical unless explicitly promoted back into groups A–F.

---
**Created:** 2026-02-09
