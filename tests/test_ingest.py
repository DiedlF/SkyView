from __future__ import annotations

import os
import sys

import numpy as np

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from ingest import _select_spatial_dataset  # noqa: E402


class _Coord:
    def __init__(self, value):
        self.values = value


class _Dataset:
    def __init__(self, valid_time: str, has_spatial: bool = True):
        self.coords = {}
        if has_spatial:
            self.coords["latitude"] = _Coord(np.array([47.0]))
            self.coords["longitude"] = _Coord(np.array([11.0]))
        self.coords["valid_time"] = _Coord(np.datetime64(valid_time))
        self.data_vars = {"dummy": object()}


def test_select_spatial_dataset_prefers_nominal_hour_for_d2_multi_message_var():
    datasets = [
        _Dataset("2026-03-11T05:15:00"),
        _Dataset("2026-03-11T05:00:00"),
        _Dataset("2026-03-11T05:30:00"),
        _Dataset("2026-03-11T05:45:00"),
    ]
    selected = _select_spatial_dataset(
        datasets,
        "/tmp/icon-d2_germany_regular-lat-lon_single-level_2026031100_005_2d_cape_ml.grib2",
    )
    assert str(selected.coords["valid_time"].values) == "2026-03-11T05:00:00"


def test_select_spatial_dataset_leaves_other_vars_on_first_spatial_dataset():
    datasets = [
        _Dataset("2026-03-11T05:15:00"),
        _Dataset("2026-03-11T05:00:00"),
    ]
    selected = _select_spatial_dataset(
        datasets,
        "/tmp/icon-d2_germany_regular-lat-lon_single-level_2026031100_005_2d_ceiling.grib2",
    )
    assert str(selected.coords["valid_time"].values) == "2026-03-11T05:15:00"
