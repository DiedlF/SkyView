"""SkyView observation layer — Europe-wide radar + satellite composites.

Phase 0 scaffolding: source fetchers and config. Fetch + reprojection to a
regular lat/lon Zarr (Phase 1) and serving via the overlay-tile pipeline
(Phase 2) build on this package. See
``docs/OBSERVATION_LAYER_IMPLEMENTATION_PLAN.md``.
"""

from __future__ import annotations

from .config import Config, RadarConfig, SatelliteConfig

__all__ = ["Config", "RadarConfig", "SatelliteConfig"]
