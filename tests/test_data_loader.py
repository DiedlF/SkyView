from __future__ import annotations

from collections import OrderedDict
import os
import sys

import numpy as np

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from services.data_loader import load_step_data  # noqa: E402
from services import storage_io  # noqa: E402


class _Logger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _write_step(tmp_path):
    run_dir = tmp_path / "icon-d2" / "2026042600"
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "001.npz",
        lat=np.array([47.0, 48.0], dtype=np.float32),
        lon=np.array([11.0, 12.0], dtype=np.float32),
        ww=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        cape_ml=np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        ceiling=np.array([[1000.0, 1100.0], [1200.0, 1300.0]], dtype=np.float32),
    )


def _write_step_zarr(tmp_path):
    run_dir = tmp_path / "icon-d2" / "2026042600"
    run_dir.mkdir(parents=True, exist_ok=True)
    return storage_io.write_zarr_group(
        str(run_dir / "001.zarr"),
        {
            "lat": np.array([47.0, 48.0], dtype=np.float32),
            "lon": np.array([11.0, 12.0], dtype=np.float32),
            "ww": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            "cape_ml": np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32),
        },
    )


def _load(tmp_path, cache, keys):
    return load_step_data(
        data_dir=str(tmp_path),
        model="icon_d2",
        run="2026042600",
        step=1,
        cache=cache,
        cache_max_items=4,
        keys=keys,
        logger=_Logger(),
    )


def test_partial_cache_does_not_satisfy_full_load(tmp_path):
    _write_step(tmp_path)
    cache = OrderedDict()

    partial = _load(tmp_path, cache, keys=["ww"])
    assert "ww" in partial
    assert "cape_ml" not in partial

    full = _load(tmp_path, cache, keys=None)
    assert "ww" in full
    assert "cape_ml" in full
    assert "ceiling" in full


def test_selective_loads_merge_into_cache(tmp_path):
    _write_step(tmp_path)
    cache = OrderedDict()

    first = _load(tmp_path, cache, keys=["ww"])
    assert "cape_ml" not in first

    second = _load(tmp_path, cache, keys=["cape_ml"])
    assert "ww" in second
    assert "cape_ml" in second

    third = _load(tmp_path, cache, keys=["ww", "cape_ml"])
    assert "ww" in third
    assert "cape_ml" in third


def test_auto_layout_prefers_zarr_when_present(tmp_path, monkeypatch):
    if not storage_io.zarr_available():
        return
    _write_step(tmp_path)
    assert _write_step_zarr(tmp_path)
    monkeypatch.setenv("SKYVIEW_DATA_LAYOUT", "auto")
    cache = OrderedDict()

    data = _load(tmp_path, cache, keys=["ww"])

    assert float(data["ww"][0, 0]) == 10.0
    assert "cape_ml" not in data


def test_zarr_layout_discovers_steps(tmp_path):
    if not storage_io.zarr_available():
        return
    run_dir = tmp_path / "icon-d2" / "2026042600"
    run_dir.mkdir(parents=True)
    assert storage_io.write_zarr_group(str(run_dir / "001.zarr"), {"lat": np.array([1], dtype=np.float32)})

    assert storage_io.step_numbers_from_dir(str(run_dir)) == [1]


def test_zarr_point_reader_returns_compact_arrays(tmp_path, monkeypatch):
    if not storage_io.zarr_available():
        return
    _write_step_zarr(tmp_path)
    monkeypatch.setenv("SKYVIEW_DATA_LAYOUT", "auto")

    data = storage_io.read_step_point_arrays(
        data_dir=str(tmp_path),
        model="icon_d2",
        run="2026042600",
        step=1,
        keys=["ww", "cape_ml"],
        lat=47.8,
        lon=11.8,
    )

    assert data is not None
    assert data["_gridI"] == 1
    assert data["_gridJ"] == 1
    assert data["ww"].shape == (1, 1)
    assert float(data["ww"][0, 0]) == 40.0
    assert float(data["cape_ml"][0, 0]) == 4000.0
    assert data["validTime"] == "2026-04-26T01:00:00Z"
