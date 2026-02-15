# Skyview Documentation

This directory contains research notes, implementation guides, and reference materials for the Skyview project.

---

## 📚 Contents

### Reference Materials
- **SkyView_12.01.15_Manual_DE.pdf** — Original SkyView (Flash version) manual in German. Historical reference for feature comparison.

### Data Source Research
- **ICON-D2_data_research.md** — Comprehensive investigation of DWD ICON-D2 data availability, variables, format, and access methods. Confirms convection height data is freely available and details all meteorological variables.

- **ICON-D2_quickstart.md** — Quick reference for working with ICON-D2 GRIB2 files. Includes Python code snippets, download commands, and visualization examples.

- **DATA_PIPELINE_RESEARCH.md** — Analysis of data update frequency optimization. Documents the transition from 2.5-hour latency to <10-minute latency via fast polling strategy.

- **ICON-EU-IMPLEMENTATION.md** — Implementation notes for ICON-EU integration (6.5km, 120h forecasts). Covers variable name mapping, timestep handling, and dual-model architecture.

- **PRECIPITATION_VARIABLES.md** — Details on precipitation data variables (rain, snow, graupel rates) and how they differ between ICON-D2 and ICON-EU.

### Planning & Improvements
- **IMPROVEMENT_IDEAS_SUMMARY.md** — Consolidated roadmap and phased implementation plan.

### Archive
- Historical/superseded docs are in `archive/` (prototype reports, completed fix summaries, and superseded plans).

---

## 📖 Reading Order (for New Contributors)

1. **Start here:** `ICON-D2_data_research.md` — Understand the data source
2. **Quick reference:** `ICON-D2_quickstart.md` — Get hands-on with GRIB2 files
3. **Architecture:** `ICON-EU-IMPLEMENTATION.md` — How the dual-model system works
4. **Optimization:** `DATA_PIPELINE_RESEARCH.md` — Why we poll every 10 minutes
5. **Details:** `PRECIPITATION_VARIABLES.md` — Specific variable handling

Prototype/fix-history docs are archived under `archive/` — useful for archaeology, not required for current development.

---

## 🔗 Related Files

- **Project root:** `/root/.openclaw/workspace/skyview/`
- **Main spec:** `../SPEC.md` — Complete technical specification
- **Task list:** `../TODO.md` — Current priorities and roadmap
- **Source code:** `../backend/`, `../frontend/`

---

**Last updated:** 2026-02-11
