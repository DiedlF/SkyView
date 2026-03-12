from __future__ import annotations

import os
import sys

import numpy as np

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from ingest import _select_spatial_dataset, load_grib_with_substeps  # noqa: E402


class _Coord:
    def __init__(self, value):
        self.values = value


class _Var:
    def __init__(self, value):
        self.values = value


class _Dataset:
    def __init__(self, valid_time: str, has_spatial: bool = True, data=None, lat=None, lon=None):
        self.coords = {}
        if has_spatial:
            self.coords["latitude"] = _Coord(np.array([47.0, 48.0]) if lat is None else np.asarray(lat))
            self.coords["longitude"] = _Coord(np.array([11.0, 12.0]) if lon is None else np.asarray(lon))
        self.coords["valid_time"] = _Coord(np.datetime64(valid_time))
        default = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self.data_vars = {"dummy": _Var(default if data is None else data)}


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


def test_select_spatial_dataset_works_with_temp_filename_when_hints_are_provided():
    datasets = [
        _Dataset("2026-03-11T05:15:00"),
        _Dataset("2026-03-11T05:00:00"),
        _Dataset("2026-03-11T05:30:00"),
    ]
    selected = _select_spatial_dataset(
        datasets,
        "/tmp/tmpabcd1234.grib2",
        var_name_hint="cape_ml",
        nominal_hour_hint=5,
    )
    assert str(selected.coords["valid_time"].values) == "2026-03-11T05:00:00"


def test_select_spatial_dataset_prefers_expected_valid_time_for_nonzero_run_hour():
    datasets = [
        _Dataset("2026-03-11T05:00:00"),
        _Dataset("2026-03-11T08:00:00"),
        _Dataset("2026-03-11T08:15:00"),
    ]
    selected = _select_spatial_dataset(
        datasets,
        "/tmp/icon-d2_germany_regular-lat-lon_single-level_2026031103_005_2d_cape_ml.grib2",
    )
    assert str(selected.coords["valid_time"].values) == "2026-03-11T08:00:00"


def test_load_grib_with_substeps_filters_to_expected_valid_hour_for_nonzero_run(monkeypatch):
    datasets = [
        _Dataset("2026-03-11T05:00:00", data=np.full((2, 2), 5.0, dtype=np.float32)),
        _Dataset("2026-03-11T05:15:00", data=np.full((2, 2), 15.0, dtype=np.float32)),
        _Dataset("2026-03-11T08:00:00", data=np.full((2, 2), 80.0, dtype=np.float32)),
        _Dataset("2026-03-11T08:15:00", data=np.full((2, 2), 81.0, dtype=np.float32)),
        _Dataset("2026-03-11T08:30:00", data=np.full((2, 2), 82.0, dtype=np.float32)),
        _Dataset("2026-03-11T08:45:00", data=np.full((2, 2), 83.0, dtype=np.float32)),
    ]

    monkeypatch.setattr("ingest.cfgrib.open_datasets", lambda _path: datasets)

    data, lat, lon, substeps, minutes = load_grib_with_substeps(
        "/tmp/icon-d2_germany_regular-lat-lon_single-level_2026031103_005_2d_cape_ml.grib2"
    )

    assert data.shape == (2, 2)
    assert float(data[0, 0]) == 80.0
    assert lat.shape == (2,)
    assert lon.shape == (2,)
    assert minutes == [0, 15, 30, 45]
    assert substeps.shape == (4, 2, 2)
    assert [float(substeps[i, 0, 0]) for i in range(4)] == [80.0, 81.0, 82.0, 83.0]


def test_load_grib_with_substeps_uses_expected_valid_hint_for_temp_files(monkeypatch):
    datasets = [
        _Dataset("2026-03-11T16:00:00", data=np.full((2, 2), 16.0, dtype=np.float32)),
        _Dataset("2026-03-11T16:15:00", data=np.full((2, 2), 16.25, dtype=np.float32)),
        _Dataset("2026-03-11T19:00:00", data=np.full((2, 2), 19.0, dtype=np.float32)),
        _Dataset("2026-03-11T19:15:00", data=np.full((2, 2), 19.25, dtype=np.float32)),
        _Dataset("2026-03-11T19:30:00", data=np.full((2, 2), 19.5, dtype=np.float32)),
        _Dataset("2026-03-11T19:45:00", data=np.full((2, 2), 19.75, dtype=np.float32)),
    ]

    monkeypatch.setattr("ingest.cfgrib.open_datasets", lambda _path: datasets)

    data, lat, lon, substeps, minutes = load_grib_with_substeps(
        "/tmp/tmpabcd1234.grib2",
        var_name_hint="cape_ml",
        nominal_hour_hint=16,
        expected_valid_dt_hint=np.datetime64("2026-03-11T19:00:00").astype("datetime64[m]").astype(object),
    )

    assert data.shape == (2, 2)
    assert float(data[0, 0]) == 19.0
    assert lat.shape == (2,)
    assert lon.shape == (2,)
    assert minutes == [0, 15, 30, 45]
    assert substeps.shape == (4, 2, 2)
    assert [float(substeps[i, 0, 0]) for i in range(4)] == [19.0, 19.25, 19.5, 19.75]
