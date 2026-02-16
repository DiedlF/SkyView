"""WMO ww mapping and severity ranking helpers."""

from __future__ import annotations

from typing import Optional


def ww_to_symbol(ww: int) -> Optional[str]:
    """Map WMO ww code to symbol type string."""
    if ww == 96: return "thunderstorm_hail"
    if 95 <= ww <= 99: return "thunderstorm"
    if ww == 86: return "snow_shower_heavy"
    if ww == 85: return "snow_shower"
    if ww == 81: return "rain_shower_moderate"
    if ww == 80: return "rain_shower"
    if ww == 82: return "rain_shower_moderate"
    if ww == 75: return "snow_heavy"
    if ww == 73: return "snow_moderate"
    if ww == 71: return "snow_slight"
    if ww == 77: return "snow_grains"
    if ww == 65: return "rain_heavy"
    if ww == 63: return "rain_moderate"
    if ww == 61: return "rain_slight"
    if ww in (66, 67): return "freezing_rain"
    if ww == 55: return "drizzle_dense"
    if ww == 53: return "drizzle_moderate"
    if ww == 51: return "drizzle_light"
    if ww == 57: return "freezing_drizzle_heavy"
    if ww == 56: return "freezing_drizzle"
    if ww == 45: return "fog"
    if ww == 48: return "rime_fog"
    return None


def ww_severity_rank(ww: int) -> int:
    """Aviation-oriented ww severity ordering for aggregation."""
    if 95 <= ww <= 99: return 100
    if 71 <= ww <= 77: return 90
    if 85 <= ww <= 86: return 85
    if 66 <= ww <= 67: return 80
    if 56 <= ww <= 57: return 75
    if 61 <= ww <= 65: return 70
    if 80 <= ww <= 84: return 65
    if 51 <= ww <= 55: return 60
    if 45 <= ww <= 48: return 50
    return ww
