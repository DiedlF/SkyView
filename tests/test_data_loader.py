from __future__ import annotations

from collections import OrderedDict
import os
import sys

import numpy as np

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from services.data_loader import load_step_data  # noqa: E402


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
