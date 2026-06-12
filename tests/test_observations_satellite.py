"""Pure tests for EUMETSAT MSG product helpers."""

from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.satellite_eumetsat import read_msg_valid_time  # noqa: E402


def test_read_msg_valid_time_from_live_msg15_shape():
    got = read_msg_valid_time(
        "MSG4-SEVI-MSG15-0100-NA-20260612164916.217000000Z-NA.nat"
    )
    assert got == dt.datetime(2026, 6, 12, 16, 49, 16, tzinfo=dt.timezone.utc)


def test_read_msg_valid_time_returns_none_for_unknown_name():
    assert read_msg_valid_time("not-a-msg-product.nat") is None
