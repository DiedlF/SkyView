"""Pytest wrapper for scripts/qa_contract.py.

Checks API contract parity between Skyview and Explorer.
Both backends must be running; skipped otherwise.

Run:  pytest tests/test_contract.py -v
      SKYVIEW_BASE=http://x:8501 EXPLORER_BASE=http://x:8502 pytest tests/test_contract.py
"""

from __future__ import annotations

import os
import sys

import pytest
import requests

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import qa_contract  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sample_time(skyview_base):
    r = requests.get(skyview_base + "/api/timesteps", timeout=20)
    steps = r.json().get("merged", {}).get("steps", [])
    assert steps, "No merged steps from /api/timesteps"
    return steps[0]["validTime"]


def test_contract_models_skyview(skyview_base):
    qa_contract.check_models(skyview_base, "skyview")


def test_contract_models_explorer(explorer_base):
    qa_contract.check_models(explorer_base, "explorer")


def test_contract_timesteps_skyview(skyview_base):
    qa_contract.check_timesteps(skyview_base, "skyview")


def test_contract_timesteps_explorer(explorer_base):
    qa_contract.check_timesteps(explorer_base, "explorer")


def test_contract_overlay_skyview(skyview_base, sample_time):
    qa_contract.check_overlay(skyview_base, "skyview", sample_time)


def test_contract_overlay_explorer(explorer_base, sample_time):
    qa_contract.check_overlay(explorer_base, "explorer", sample_time)


def test_contract_overlay_tile_skyview(skyview_base, sample_time):
    qa_contract.check_overlay_tile(skyview_base, "skyview", sample_time)


def test_contract_overlay_tile_explorer(explorer_base, sample_time):
    qa_contract.check_overlay_tile(explorer_base, "explorer", sample_time)


def test_contract_point_skyview(skyview_base, sample_time):
    qa_contract.check_point(skyview_base, "skyview", sample_time)


def test_contract_point_explorer(explorer_base, sample_time):
    qa_contract.check_point(explorer_base, "explorer", sample_time)
