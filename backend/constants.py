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
CAPE_CONV_THRESHOLD: float = 2.0
CIN_CONV_THRESHOLD: float = -100.0
CAPE_CB_STRONG_THRESHOLD: float = 1000.0
LPI_CB_THRESHOLD: float = 7.0
CLOUD_DEPTH_CU_CON_THRESHOLD: float = 2000.0
CLOUD_DEPTH_CB_THRESHOLD: float = 4000.0
AGL_CONV_MIN_METERS: float = 0.0

# Non-convective ceiling bands
CEILING_LOW_MAX_METERS: float = 2000.0
CEILING_MID_MAX_METERS: float = 7000.0
CEILING_VALID_MAX_METERS: float = 20000.0

# Precomputed precip-rate fields used by overlays and point payloads
PRECIP_RATE_FIELD_BY_LAYER_VAR: dict[str, str] = {
    "total_precip": "tp_rate",
    "convective_precip": "convective_rate",
    "gridscale_precip": "gridscale_rate",
    "rain_amount": "rain_rate",
    "snow_amount": "snow_rate",
    "hail_amount": "hail_rate",
}

DATA_CACHE_MAX_ITEMS: int = int(os.environ.get('SKYVIEW_DATA_CACHE_MAX_ITEMS', '24'))

# Symbol rendering modes
SYMBOL_MODE_PRECOMPUTED_MAX_ZOOM: int = 9
SYMBOL_MODE_FIXED_GRID_MAX_ZOOM: int = 11
SYMBOL_MODE_NATIVE_ZOOM: int = 12
SYMBOL_MODE_NATIVE_ZOOM_D2: int = 12
SYMBOL_MODE_NATIVE_ZOOM_EU: int = 11

# World anchoring for fixed symbol grids / bins
WORLD_GRID_ANCHOR_LAT: float = -90.0
WORLD_GRID_ANCHOR_LON: float = -180.0

# Low-zoom precompute domain (operational Europe view)
LOW_ZOOM_GLOBAL_CACHE_MAX_ZOOM: int = SYMBOL_MODE_PRECOMPUTED_MAX_ZOOM
LOW_ZOOM_GLOBAL_BBOX: tuple[float, float, float, float] = (30.0, -30.0, 72.0, 45.0)
# Low-zoom precomputed JSON bins are opt-in. VPS benchmark (2026-03-11) showed
# little to no latency win, plus ~10 min ingest overhead and ~4.7 GB disk use.
LOW_ZOOM_PRECOMPUTED_BINS_ENABLED: bool = os.environ.get("SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS", "0").strip().lower() not in {"0", "false", "no", "off"}

# Emagram pressure levels (D2)
EMAGRAM_D2_LEVELS_HPA: list[int] = [1000, 975, 950, 850, 700, 600, 500, 400, 300, 200]

# Meteogram wind profile levels (D2) — trimmed for faster loads and better readability.
# Approx. tops out around 7 km (~400 hPa).
METEOGRAM_D2_LEVELS_HPA: list[int] = [1000, 975, 950, 850, 700, 600, 500, 400]

# ICON-D2 model-level cloud-cover levels for the meteogram wind-panel background.
# These span roughly surface to 9.25 km, matching the extended high-cloud panel without ingesting
# all 44 available levels in that altitude range.
METEOGRAM_D2_CLC_MODEL_LEVELS: list[int] = [16, 18, 22, 24, 27, 30, 32, 35, 38, 42, 45, 48, 52, 55, 58, 61, 63, 65]

# Standard ICON-D2 full-level heights for zero topography height, in meters AMSL.
# Source: DWD ICON Database Reference, Table A.4.
ICON_D2_FULL_LEVEL_HEIGHT_M: dict[int, float] = {
    1: 20700.926, 2: 18707.630, 3: 17459.836, 4: 16432.216, 5: 15538.089,
    6: 14738.074, 7: 14009.789, 8: 13338.901, 9: 12715.508, 10: 12132.398,
    11: 11584.105, 12: 11066.360, 13: 10575.747, 14: 10109.477, 15: 9665.235,
    16: 9241.077, 17: 8835.344, 18: 8446.611, 19: 8073.640, 20: 7715.350,
    21: 7370.787, 22: 7039.106, 23: 6719.557, 24: 6411.462, 25: 6114.219,
    26: 5827.280, 27: 5550.148, 28: 5282.374, 29: 5023.548, 30: 4773.294,
    31: 4531.269, 32: 4297.157, 33: 4070.672, 34: 3851.546, 35: 3639.535,
    36: 3434.414, 37: 3235.976, 38: 3044.029, 39: 2858.399, 40: 2678.926,
    41: 2505.461, 42: 2337.870, 43: 2176.032, 44: 2019.836, 45: 1869.185,
    46: 1723.991, 47: 1584.179, 48: 1449.686, 49: 1320.458, 50: 1196.457,
    51: 1077.658, 52: 964.048, 53: 855.630, 54: 752.427, 55: 654.479,
    56: 561.856, 57: 474.652, 58: 393.002, 59: 317.092, 60: 247.172,
    61: 183.592, 62: 126.857, 63: 77.745, 64: 37.606, 65: 10.000,
}

# Standard gravity (m/s²)
G0: float = 9.80665
