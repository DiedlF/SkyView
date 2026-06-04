"""Reprojection: native composite grids → regular lat/lon target grid.

The serving pipeline expects a regular lat/lon grid (1-D ``lat``/``lon`` + 2-D
field), so ingest reprojects:

  * Radar (OPERA ODIM): Lambert Azimuthal Equal Area → regular lat/lon, via a
    pyproj transform + nearest-neighbour sampling of the source raster.
  * Satellite (MSG/MTG geostationary): a generic swath resample (pyresample
    nearest) given the source lon/lat arrays + data.

Both crop/downsample to the SkyView ``TARGET_GRID`` (≈ d2_bounds at 0.02°). All
heavy dependencies (numpy, pyproj, pyresample) are imported lazily so this module
imports cleanly in the core-server environment.

NOTE: the exact ODIM ``/where`` attribute set for the CIRRUS *composite* is one
of the open items to confirm against a real file in Phase 1 (single-site layouts
differ). The reader below uses the standard ODIM composite georeferencing
(projdef + UL corner + xscale/yscale + xsize/ysize).
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import TARGET_GRID, GridSpec

log = logging.getLogger("skyview.observations.reproject")


# -- pure attribute coercion (no heavy deps; unit-tested) ------------------
def decode_attr(value):
    """Coerce an HDF5 attribute (bytes / 0-d array / scalar) to a Python scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    # numpy 0-d arrays / numpy scalars expose .item(); plain scalars do not.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def attr_float(where: dict, key: str) -> Optional[float]:
    if key not in where:
        return None
    try:
        return float(decode_attr(where[key]))
    except (TypeError, ValueError):
        return None


def attr_str(where: dict, key: str) -> Optional[str]:
    if key not in where:
        return None
    return str(decode_attr(where[key]))


# -- target grid coords (lazy numpy) --------------------------------------
def build_target_coords(grid: GridSpec = TARGET_GRID):
    """Return ascending 1-D ``(lat, lon)`` numpy arrays for the target grid."""
    import numpy as np

    lat = grid.lat_min + np.arange(grid.n_lat, dtype="float64") * grid.resolution
    lon = grid.lon_min + np.arange(grid.n_lon, dtype="float64") * grid.resolution
    return lat.astype("float32"), lon.astype("float32")


# -- radar: ODIM Lambert EA → regular lat/lon -----------------------------
def reproject_odim(phys, where: dict, grid: GridSpec = TARGET_GRID):
    """Nearest-neighbour reproject an ODIM composite array onto the target grid.

    ``phys`` is the 2-D physical field (shape ``(ysize, xsize)``) and ``where``
    the ODIM ``/where`` attributes. Returns a ``(n_lat, n_lon)`` float32 array
    with NaN where the target falls outside the source raster.
    """
    import numpy as np
    import pyproj

    projdef = attr_str(where, "projdef")
    if not projdef:
        raise ValueError("ODIM /where has no 'projdef'; cannot reproject")

    xscale = attr_float(where, "xscale")
    yscale = attr_float(where, "yscale")
    xsize = int(attr_float(where, "xsize") or phys.shape[1])
    ysize = int(attr_float(where, "ysize") or phys.shape[0])
    if not xscale or not yscale:
        raise ValueError("ODIM /where missing xscale/yscale")

    src_crs = pyproj.CRS.from_proj4(projdef) if projdef.strip().startswith("+") \
        else pyproj.CRS.from_user_input(projdef)
    to_src = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

    # Source raster origin: upper-left corner projected into source CRS.
    ul_lon = attr_float(where, "UL_lon")
    ul_lat = attr_float(where, "UL_lat")
    if ul_lon is None or ul_lat is None:
        raise ValueError("ODIM /where missing UL_lon/UL_lat corner")
    x_ul, y_ul = to_src.transform(ul_lon, ul_lat)

    lat1d, lon1d = build_target_coords(grid)
    lon_mesh, lat_mesh = np.meshgrid(lon1d.astype("float64"), lat1d.astype("float64"))
    xs, ys = to_src.transform(lon_mesh, lat_mesh)

    # Source pixel indices (col increases east, row increases south).
    col = np.round((xs - x_ul) / xscale).astype("int64")
    row = np.round((y_ul - ys) / yscale).astype("int64")
    inside = (col >= 0) & (col < xsize) & (row >= 0) & (row < ysize)

    out = np.full(lat_mesh.shape, np.nan, dtype="float32")
    src = np.asarray(phys, dtype="float32")
    out[inside] = src[row[inside], col[inside]]
    return out


# -- satellite: generic swath (geostationary) → regular lat/lon -----------
def reproject_swath(src_lons, src_lats, src_data, grid: GridSpec = TARGET_GRID,
                    radius_of_influence: float = 6000.0):
    """Nearest-neighbour resample source swath/geos data onto the target grid.

    ``src_lons``/``src_lats``/``src_data`` are 2-D source arrays (e.g. a SEVIRI
    area). Off-disk / fill pixels should already be NaN in ``src_data``.
    """
    import numpy as np
    from pyresample import geometry, kd_tree

    lat1d, lon1d = build_target_coords(grid)
    lon_mesh, lat_mesh = np.meshgrid(lon1d.astype("float64"), lat1d.astype("float64"))

    src_def = geometry.SwathDefinition(lons=np.asarray(src_lons), lats=np.asarray(src_lats))
    tgt_def = geometry.GridDefinition(lons=lon_mesh, lats=lat_mesh)
    out = kd_tree.resample_nearest(
        src_def,
        np.asarray(src_data, dtype="float32"),
        tgt_def,
        radius_of_influence=radius_of_influence,
        fill_value=np.nan,
    )
    return out.astype("float32")
