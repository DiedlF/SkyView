from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from constants import (  # noqa: E402
    ICON_D2_FULL_LEVEL_HEIGHT_M,
    METEOGRAM_D2_CLC_MODEL_LEVELS,
    METEOGRAM_D2_LEVELS_HPA,
)


def test_meteogram_wind_levels_stop_near_7000m():
    assert METEOGRAM_D2_LEVELS_HPA == [1000, 975, 950, 850, 700, 600, 500, 400]


def test_meteogram_cloud_levels_extend_above_7000m_with_consistent_spacing():
    assert METEOGRAM_D2_CLC_MODEL_LEVELS[:2] == [16, 18]
    assert ICON_D2_FULL_LEVEL_HEIGHT_M[22] > 7000
    assert 8300 <= ICON_D2_FULL_LEVEL_HEIGHT_M[18] <= 8600
    assert ICON_D2_FULL_LEVEL_HEIGHT_M[16] > 9000
