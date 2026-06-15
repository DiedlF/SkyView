"""Unit tests for the cadence fetch-gate in observation ingest.

Pure: exercises ``fresh_within_cadence`` without network or filesystem.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations import store  # noqa: E402
from observations.ingest_obs import fresh_within_cadence  # noqa: E402

_NOW = dt.datetime(2026, 6, 15, 9, 11, 0, tzinfo=dt.timezone.utc)


def _frame(minutes_ago: float) -> dict:
    when = _NOW - dt.timedelta(minutes=minutes_ago)
    return {"frame_id": store.frame_id(when)}


def test_fresh_frame_skips_fetch():
    # Radar cadence 300s: a 2-minute-old frame means no new frame is due yet.
    assert fresh_within_cadence(_frame(2), 300, _NOW) is True


def test_stale_frame_allows_fetch():
    # Older than one cadence -> a new frame should exist, so fetch.
    assert fresh_within_cadence(_frame(6), 300, _NOW) is False


def test_no_frame_allows_fetch():
    assert fresh_within_cadence(None, 300, _NOW) is False
    assert fresh_within_cadence({}, 300, _NOW) is False


def test_missing_cadence_does_not_gate():
    assert fresh_within_cadence(_frame(1), None, _NOW) is False


def test_unparseable_frame_id_allows_fetch():
    assert fresh_within_cadence({"frame_id": "not-a-time"}, 300, _NOW) is False
