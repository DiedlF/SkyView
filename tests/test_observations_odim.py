"""Unit test for the ODIM HDF5 radar composite reader.

Builds a tiny synthetic ODIM-shaped HDF5 file and verifies the gain/offset
scaling and nodata/undetect masking. Skips automatically where h5py/numpy are
not installed (e.g. the core-server environment without ingest deps).
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.radar_ord import read_odim_maxreflectivity  # noqa: E402


def _write_synthetic_odim(path, raw, gain, offset, nodata, undetect, where=None):
    """Write a minimal ODIM-like file: /dataset1/data1/{data,what} + /where."""
    with h5py.File(path, "w") as f:
        d1 = f.create_group("dataset1").create_group("data1")
        d1.create_dataset("data", data=raw)
        what = d1.create_group("what")
        what.attrs["gain"] = gain
        what.attrs["offset"] = offset
        what.attrs["nodata"] = nodata
        what.attrs["undetect"] = undetect
        w = f.create_group("where")
        for k, v in (where or {"LL_lat": 31.7, "LL_lon": -10.4}).items():
            w.attrs[k] = v


def test_read_odim_scaling_and_masking(tmp_path):
    # raw codes: 0 = undetect, 255 = nodata, others = real echoes
    raw = np.array([[0, 10, 20], [128, 255, 200]], dtype="uint8")
    gain, offset, nodata, undetect = 0.5, -32.0, 255, 0
    path = tmp_path / "opera_cirrus.h5"
    _write_synthetic_odim(path, raw, gain, offset, nodata, undetect)

    phys, where = read_odim_maxreflectivity(path)

    assert phys.shape == raw.shape
    assert phys.dtype == np.float32
    # real echoes scaled: raw*gain + offset
    assert phys[0, 1] == pytest.approx(10 * 0.5 - 32.0)   # -27.0 dBZ
    assert phys[1, 0] == pytest.approx(128 * 0.5 - 32.0)  # 32.0 dBZ
    assert phys[1, 2] == pytest.approx(200 * 0.5 - 32.0)  # 68.0 dBZ
    # undetect (0) and nodata (255) masked to NaN
    assert np.isnan(phys[0, 0])
    assert np.isnan(phys[1, 1])
    # where attributes round-tripped
    assert where["LL_lat"] == pytest.approx(31.7)
