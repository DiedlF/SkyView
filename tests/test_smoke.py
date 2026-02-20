"""Pytest wrapper for scripts/qa_smoke.py.

Marks: integration (requires live Skyview backend).
Run:   pytest tests/test_smoke.py -v
       SKYVIEW_BASE=http://myserver:8501 pytest tests/test_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
import os

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
pytestmark = pytest.mark.integration


def test_smoke_core_endpoints(skyview_base):
    """Core endpoints (/health, /timesteps, /models, /status) return 200."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "qa_smoke.py"), "--base", skyview_base],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"qa_smoke.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, f"Expected PASS in output:\n{result.stdout}"
