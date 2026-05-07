from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from ingest import (  # noqa: E402
    _compute_precip_rate_fields,
    _load_precip_acc_from_step,
    _select_spatial_dataset,
    build_d2_boundary_cache,
    check_new_data_available,
    cleanup_old_runs,
    load_grib_with_substeps,
)
from services import storage_io  # noqa: E402


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


def test_check_new_data_available_counts_zarr_steps(tmp_path, monkeypatch):
    run = "2026042600"
    run_dir = tmp_path / "icon-d2" / run
    run_dir.mkdir(parents=True)
    for step in range(1, 49):
        (run_dir / f"{step:03d}.zarr").mkdir()

    monkeypatch.setattr("ingest.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("ingest.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    available, has_local = check_new_data_available(run, model="icon-d2")

    assert available is True
    assert has_local is True


def test_cleanup_old_runs_removes_matching_disk_cache_dirs(tmp_path, monkeypatch):
    for run in ("2026042600", "2026042606", "2026042612"):
        (tmp_path / "icon-d2" / run).mkdir(parents=True)

    for kind in ("meteogram-point", "emagram-point", "nowcast-point"):
        for run in ("2026042600", "2026042606", "2026042612"):
            cache_dir = tmp_path / "cache" / kind / "icon_d2" / run
            cache_dir.mkdir(parents=True)
            (cache_dir / "entry.json").write_text("{}", encoding="utf-8")

    other_model_cache = tmp_path / "cache" / "meteogram-point" / "icon_eu" / "2026042600"
    other_model_cache.mkdir(parents=True)
    (other_model_cache / "entry.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("ingest.DATA_DIR", str(tmp_path))

    cleanup_old_runs("icon-d2", keep_runs=1)

    assert (tmp_path / "icon-d2" / "2026042612").exists()
    assert not (tmp_path / "icon-d2" / "2026042606").exists()
    assert not (tmp_path / "icon-d2" / "2026042600").exists()

    for kind in ("meteogram-point", "emagram-point", "nowcast-point"):
        assert (tmp_path / "cache" / kind / "icon_d2" / "2026042612").exists()
        assert not (tmp_path / "cache" / kind / "icon_d2" / "2026042606").exists()
        assert not (tmp_path / "cache" / kind / "icon_d2" / "2026042600").exists()

    assert other_model_cache.exists()


def test_load_precip_acc_from_zarr_step(tmp_path, monkeypatch):
    if not storage_io.zarr_available():
        return
    run = "2026042600"
    run_dir = tmp_path / "icon-d2" / run
    run_dir.mkdir(parents=True)
    assert storage_io.write_zarr_group(
        str(run_dir / "001.zarr"),
        {
            "lat": np.array([47.0, 48.0], dtype=np.float32),
            "lon": np.array([11.0, 12.0], dtype=np.float32),
            "tot_prec": np.ones((2, 2), dtype=np.float32),
        },
    )
    monkeypatch.setattr("ingest.DATA_DIR", str(tmp_path))

    acc = _load_precip_acc_from_step("icon-d2", run, 1)

    assert acc is not None
    assert float(acc["tot_prec"][0, 0]) == 1.0
    assert acc["rain_gsp"].shape == (2, 2)


def test_compute_precip_rate_fields_splits_convective_and_gridscale():
    shape = (2, 2)
    zeros = np.zeros(shape, dtype=np.float32)
    fields = _compute_precip_rate_fields(
        tp=np.full(shape, 10.0, dtype=np.float32),
        rg=np.full(shape, 4.0, dtype=np.float32),
        rc=np.full(shape, 2.0, dtype=np.float32),
        sg=np.full(shape, 3.0, dtype=np.float32),
        sc=np.full(shape, 1.0, dtype=np.float32),
        gg=np.full(shape, 0.5, dtype=np.float32),
        dt_h=1.0,
        prev_acc=None,
    )

    assert float(fields["convective_rate"][0, 0]) == 3.0
    assert float(fields["gridscale_rate"][0, 0]) == 7.5
    assert float(fields["rain_rate"][0, 0]) == 6.0
    assert float(fields["snow_rate"][0, 0]) == 4.0
    assert float(fields["hail_rate"][0, 0]) == 0.5
    assert float(fields["tp_rate"][0, 0]) == 10.0

    fields = _compute_precip_rate_fields(
        tp=np.full(shape, 16.0, dtype=np.float32),
        rg=np.full(shape, 7.0, dtype=np.float32),
        rc=np.full(shape, 5.0, dtype=np.float32),
        sg=np.full(shape, 5.0, dtype=np.float32),
        sc=np.full(shape, 2.0, dtype=np.float32),
        gg=np.full(shape, 1.0, dtype=np.float32),
        dt_h=3.0,
        prev_acc={
            "tot_prec": np.full(shape, 10.0, dtype=np.float32),
            "rain_gsp": np.full(shape, 4.0, dtype=np.float32),
            "rain_con": np.full(shape, 2.0, dtype=np.float32),
            "snow_gsp": np.full(shape, 3.0, dtype=np.float32),
            "snow_con": np.full(shape, 1.0, dtype=np.float32),
            "grau_gsp": zeros,
        },
    )

    assert np.isclose(float(fields["convective_rate"][0, 0]), 4.0 / 3.0)
    assert np.isclose(float(fields["gridscale_rate"][0, 0]), 6.0 / 3.0)


def test_build_d2_boundary_cache_reads_zarr_only(tmp_path, monkeypatch):
    if not storage_io.zarr_available():
        return
    run = "2026042600"
    run_dir = tmp_path / "icon-d2" / run
    run_dir.mkdir(parents=True)
    assert storage_io.write_zarr_group(
        str(tmp_path / "icon-d2" / "grid" / "static.zarr"),
        {
            "lat": np.array([47.0, 48.0], dtype=np.float32),
            "lon": np.array([11.0, 12.0], dtype=np.float32),
            "hsurf": np.ones((2, 2), dtype=np.float32),
        },
    )
    assert storage_io.write_zarr_group(
        str(run_dir / "001.zarr"),
        {
            "lat": np.array([47.0, 48.0], dtype=np.float32),
            "lon": np.array([11.0, 12.0], dtype=np.float32),
            "ww": np.ones((2, 2), dtype=np.float32),
        },
    )
    monkeypatch.setattr("ingest.DATA_DIR", str(tmp_path))

    build_d2_boundary_cache(run)

    assert (run_dir / "_d2_boundary.json").exists()
