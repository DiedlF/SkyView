"""Shared pytest fixtures for Skyview integration tests."""

from __future__ import annotations

import os

import pytest
import requests

SKYVIEW_BASE = os.environ.get("SKYVIEW_BASE", "http://127.0.0.1:8501")
EXPLORER_BASE = os.environ.get("EXPLORER_BASE", "http://127.0.0.1:8502")
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(url + "/api/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def skyview_base():
    """URL of a running Skyview backend. Skip session if not reachable."""
    if not _reachable(SKYVIEW_BASE):
        pytest.skip(f"Skyview server not reachable at {SKYVIEW_BASE} — set SKYVIEW_BASE or start backend.")
    return SKYVIEW_BASE


@pytest.fixture(scope="session")
def explorer_base():
    """URL of a running Explorer backend. Skip session if not reachable."""
    if not _reachable(EXPLORER_BASE):
        pytest.skip(f"Explorer server not reachable at {EXPLORER_BASE} — set EXPLORER_BASE or start explorer.")
    return EXPLORER_BASE
