"""Shared constants for Skyview backend.

Keep meteorological thresholds and frequently reused config in one place.
"""

from __future__ import annotations
import os

# Grid aggregation cell sizes by map zoom
CELL_SIZES_BY_ZOOM: dict[int, float] = {
    5: 2.0,
    6: 1.0,
    7: 0.5,
    8: 0.25,
    9: 0.12,
    10: 0.06,
    11: 0.03,
    12: 0.02,
}

# Fallback strictness
EU_STRICT_MAX_DELTA_HOURS_DEFAULT: float = 0.5

# ICON-EU forecast cadence change for precip accumulation/rates
ICON_EU_STEP_3H_START: int = 81

# Convection/cloud classification thresholds
CAPE_CONV_THRESHOLD: float = 50.0
CAPE_CB_STRONG_THRESHOLD: float = 1000.0
LPI_CB_THRESHOLD: float = 7.0
CLOUD_DEPTH_CU_CON_THRESHOLD: float = 2000.0
CLOUD_DEPTH_CB_THRESHOLD: float = 4000.0
AGL_CONV_MIN_METERS: float = 300.0

# Non-convective ceiling bands
CEILING_LOW_MAX_METERS: float = 2000.0
CEILING_MID_MAX_METERS: float = 7000.0
CEILING_VALID_MAX_METERS: float = 20000.0

# Precomputed precip-rate fields used by overlays and point payloads
PRECIP_RATE_FIELD_BY_LAYER_VAR: dict[str, str] = {
    "total_precip": "tp_rate",
    "rain_amount": "rain_rate",
    "snow_amount": "snow_rate",
    "hail_amount": "hail_rate",
}

DATA_CACHE_MAX_ITEMS: int = int(os.environ.get('SKYVIEW_DATA_CACHE_MAX_ITEMS', '24'))

# Low-zoom global symbols settings (also used by routers)
LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM: int = 9
LOW_ZOOM_GLOBAL_BBOX: tuple[float, float, float, float] = (30.0, -30.0, 72.0, 45.0)

# Emagram pressure levels (D2)
EMAGRAM_D2_LEVELS_HPA: list[int] = [1000, 975, 950, 850, 700, 600, 500, 400, 300, 200]

# Standard gravity (m/s²)
G0: float = 9.80665
