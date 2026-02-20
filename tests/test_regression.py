"""Pytest wrapper for scripts/qa_regression.py.

Unit checks (no server needed) run always.
Integration checks (need Skyview) are skipped when backend not reachable.

Run:  pytest tests/test_regression.py -v
      pytest tests/test_regression.py -v -m "not integration"  # unit-only
"""

from __future__ import annotations

import os
import sys

import pytest

# Make scripts importable
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, BACKEND_DIR)

import qa_regression  # noqa: E402


# ---------------------------------------------------------------------------
# Pure unit tests (no server required)
# ---------------------------------------------------------------------------

def test_convective_agl_suppression_logic():
    """Convective symbols suppressed below AGL threshold — pure logic."""
    qa_regression.check_convective_agl_suppression_logic()


def test_blue_thermal_precedence_over_cb():
    """blue_thermal takes priority over cb when hbas_sc <= 0 — pure logic."""
    qa_regression.check_blue_thermal_precedence_over_cb()


def test_resolve_eu_time_strict_input_handling():
    """_resolve_eu_time_strict handles 'latest' and non-ISO strings gracefully."""
    qa_regression.check_resolve_eu_time_strict_input_handling()


# ---------------------------------------------------------------------------
# Integration tests (require live Skyview backend)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_z12_row_continuity(skyview_base):
    """Symbol grid has uniform row sizes at zoom 12 (no lattice gaps)."""
    steps = qa_regression.get_merged_steps(skyview_base)
    t = next((s["validTime"] for s in steps if s["validTime"][11:13] == "23"), steps[0]["validTime"])
    qa_regression.check_z12_row_continuity(skyview_base, t)


@pytest.mark.integration
def test_border_pan_stability(skyview_base):
    """Overlapping viewport requests return consistent symbols in shared area."""
    steps = qa_regression.get_merged_steps(skyview_base)
    t = steps[0]["validTime"]
    qa_regression.check_border_pan_stability(skyview_base, t)


@pytest.mark.integration
def test_d2_eu_handover(skyview_base):
    """D2/EU handover region returns blended or single-model consistently."""
    steps = qa_regression.get_merged_steps(skyview_base)
    qa_regression.check_d2_eu_handover(skyview_base, steps)


@pytest.mark.integration
def test_wind_point_parity(skyview_base):
    """Wind barb endpoint and point endpoint agree on wind direction/speed."""
    steps = qa_regression.get_merged_steps(skyview_base)
    t = steps[0]["validTime"]
    qa_regression.check_wind_point_parity(skyview_base, t)


@pytest.mark.integration
def test_symbol_zoom_continuity(skyview_base):
    """Same location returns same dominant symbol class across zoom levels."""
    steps = qa_regression.get_merged_steps(skyview_base)
    t = next((s["validTime"] for s in steps if s["validTime"][11:13] == "14"), steps[0]["validTime"])
    qa_regression.check_symbol_zoom_continuity(skyview_base, t)
