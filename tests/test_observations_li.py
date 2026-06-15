"""Pure tests for the MTG-I1 Lightning Imager (LI) source wiring."""

from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.config import LiConfig  # noqa: E402
from observations.satellite_li import product_sensing_time  # noqa: E402


def test_li_default_collection_and_dataset():
    cfg = LiConfig()
    assert cfg.collection_id == "EO:EUM:DAT:0686"  # LI L2 Accumulated Flashes
    assert cfg.dataset == "flash_accumulation"
    assert cfg.cadence_seconds == 600


def test_li_reuses_wmo_sensing_time_helper():
    # LI products share the WMO filename layout, so the shared helper resolves
    # the sensing start from the product identifier when no attr is present.
    class FakeProduct:
        sensing_start = None

        def __str__(self):
            return (
                "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AF--FD--CHK-BODY--ARC-NC4E"
                "_C_EUMT_20260615080025_L2PF_OPE_20260615075000_20260615080000_N__O_0048_0001.nc"
            )

    got = product_sensing_time(FakeProduct())
    assert got == dt.datetime(2026, 6, 15, 7, 50, 0, tzinfo=dt.timezone.utc)
