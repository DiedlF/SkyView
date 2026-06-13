"""Unit tests for the observation target grid + ODIM attribute coercion.

Pure (no numpy/pyproj): exercises GridSpec index math and the attribute helpers
used by the reprojection step.
"""

from __future__ import annotations

import os
import sys

import pytest

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.config import GridSpec  # noqa: E402
from observations.reproject import (  # noqa: E402
    attr_float,
    attr_str,
    decode_attr,
    reproject_odim,
    target_area_definition,
)


def test_grid_shape_matches_d2_window():
    g = GridSpec()
    assert g.n_lat == 746
    assert g.n_lon == 1215
    assert g.shape == (746, 1215)


def test_grid_endpoints_inclusive():
    g = GridSpec()
    assert g.lat_at(0) == pytest.approx(43.18)
    assert g.lat_at(g.n_lat - 1) == pytest.approx(58.08)
    assert g.lon_at(0) == pytest.approx(-3.94)
    assert g.lon_at(g.n_lon - 1) == pytest.approx(20.34)


def test_grid_custom_resolution():
    g = GridSpec(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=2.0, resolution=0.5)
    assert g.shape == (3, 5)  # 0,0.5,1.0  and 0,0.5,1.0,1.5,2.0


def test_decode_attr_handles_bytes_and_scalars():
    assert decode_attr(b"+proj=laea") == "+proj=laea"
    assert decode_attr("plain") == "plain"
    assert decode_attr(1000) == 1000


def test_attr_helpers():
    where = {"xscale": b"1000.0", "projdef": b"+proj=laea +lat_0=55", "xsize": 1900}
    assert attr_float(where, "xscale") == pytest.approx(1000.0)
    assert attr_float(where, "xsize") == pytest.approx(1900.0)
    assert attr_str(where, "projdef") == "+proj=laea +lat_0=55"
    assert attr_float(where, "missing") is None
    assert attr_str(where, "missing") is None


def test_target_area_uses_grid_centers():
    pytest.importorskip("pyresample")
    g = GridSpec(lat_min=0.5, lat_max=1.5, lon_min=10.5, lon_max=11.5, resolution=1.0)
    area = target_area_definition(g)

    assert area.width == 2
    assert area.height == 2
    assert list(area.projection_x_coords) == pytest.approx([10.5, 11.5])
    # Pyresample AreaDefinition rows are north-to-south for image output.
    assert list(area.projection_y_coords) == pytest.approx([1.5, 0.5])


def test_reproject_odim_respects_pixel_corner_extent():
    np = pytest.importorskip("numpy")
    pytest.importorskip("pyproj")
    pytest.importorskip("pyresample")

    # ODIM image data are stored north-to-south. The /where corners below are
    # pixel edges, so target centers at 0.5/1.5 degrees should hit each cell.
    phys = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
    where = {
        "projdef": "+proj=longlat +datum=WGS84 +no_defs",
        "xsize": 2,
        "ysize": 2,
        "xscale": 1.0,
        "yscale": 1.0,
        "LL_lon": 0.0,
        "LL_lat": 0.0,
        "UL_lon": 0.0,
        "UL_lat": 2.0,
        "UR_lon": 2.0,
        "UR_lat": 2.0,
        "LR_lon": 2.0,
        "LR_lat": 0.0,
    }
    grid = GridSpec(lat_min=0.5, lat_max=1.5, lon_min=0.5, lon_max=1.5, resolution=1.0)

    got = reproject_odim(phys, where, grid)

    assert got.shape == (2, 2)
    # reproject_odim keeps SkyView's internal south-to-north latitude order.
    np.testing.assert_allclose(got, [[30.0, 40.0], [10.0, 20.0]])
