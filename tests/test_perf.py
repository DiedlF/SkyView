"""Pytest wrapper for scripts/qa_perf.py.

Lightweight performance probes with threshold checks.
Marked 'perf' — skip by default in fast CI; run explicitly when needed.

Run:  pytest tests/test_perf.py -v -m perf
      SKYVIEW_BASE=http://myserver:8501 pytest tests/test_perf.py -m perf
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
pytestmark = [pytest.mark.integration, pytest.mark.perf]


def test_perf_probes(skyview_base):
    """All performance probes stay within configured ms thresholds."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "qa_perf.py"),
         "--base", skyview_base, "--runs", "6"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"qa_perf.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
