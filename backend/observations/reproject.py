"""Reprojection: native composite grids -> regular lat/lon target grid.

The serving pipeline expects a regular lat/lon grid (1-D ``lat``/``lon`` + 2-D
field), so ingest reprojects:

  * Radar (OPERA ODIM): ODIM ``/where`` projection metadata -> regular
    lat/lon, via pyresample nearest-neighbour area resampling.
  * Satellite (MSG/MTG geostationary): Satpy reads the official native
    geostationary geometry and resamples to the same target area.

Both crop/downsample to the SkyView ``TARGET_GRID`` (a regular lat/lon window a
bit wider than the ICON-EU/D2 crop, at 0.02°), which is also the exact extent the
serving layer advertises as the frame bbox — render and bbox must stay equal. All
heavy dependencies (numpy, pyproj, pyresample) are imported lazily so this module
imports cleanly in the core-server environment.

ODIM geographical image products define corner coordinates as pixel corners. The
helpers below preserve that convention by building pyresample area extents from
pixel edges, then returning arrays in SkyView's south-to-north latitude order.
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


def target_area_definition(grid: GridSpec = TARGET_GRID):
    """Return a north-up pyresample area for SkyView's regular lon/lat grid."""
    from pyresample.geometry import AreaDefinition

    half = grid.resolution / 2.0
    return AreaDefinition(
        "skyview_latlon",
        "SkyView regular lon/lat observation grid",
        "latlon",
        {"proj": "longlat", "datum": "WGS84", "no_defs": None, "type": "crs"},
        grid.n_lon,
        grid.n_lat,
        (
            grid.lon_min - half,
            grid.lat_min - half,
            grid.lon_max + half,
            grid.lat_max + half,
        ),
    )


def web_mercator_area_definition(grid: GridSpec = TARGET_GRID):
    """Return a north-up EPSG:3857 (Web Mercator) area for the grid's geo corners.

    The Leaflet basemap is Web Mercator, and ``L.imageOverlay`` stretches the
    frame *linearly in projected (Mercator) pixels* between the corner lat/lons.
    So the frame must itself be uniform in Mercator metres to line up with the
    map. An equirectangular (uniform-in-latitude) frame is displaced by up to
    ~100 km N-S at mid latitudes — the visible mismatch against the basemap.

    The served bbox is unchanged: corners map 1:1 between lat/lon and Mercator,
    so the frontend still places the image at [lat_min/lon_min .. lat_max/lon_max].
    """
    import pyproj
    from pyresample.geometry import AreaDefinition

    to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = to_merc.transform(grid.lon_min, grid.lat_min)
    x_max, y_max = to_merc.transform(grid.lon_max, grid.lat_max)

    width = grid.n_lon
    # Square-ish Mercator pixels: scale height to the projected aspect ratio.
    height = max(1, int(round(width * (y_max - y_min) / (x_max - x_min))))
    return AreaDefinition(
        "skyview_webmerc",
        "SkyView Web Mercator observation grid",
        "webmerc",
        "EPSG:3857",
        width,
        height,
        (x_min, y_min, x_max, y_max),
    )


def odim_area_definition(where: dict, fallback_shape: tuple[int, int]):
    """Build a pyresample source area from ODIM Cartesian ``/where`` metadata."""
    import pyproj
    from pyresample.geometry import AreaDefinition

    projdef = attr_str(where, "projdef")
    if not projdef:
        raise ValueError("ODIM /where has no 'projdef'; cannot reproject")

    xscale = attr_float(where, "xscale")
    yscale = attr_float(where, "yscale")
    xsize = int(attr_float(where, "xsize") or fallback_shape[1])
    ysize = int(attr_float(where, "ysize") or fallback_shape[0])
    if not xscale or not yscale:
        raise ValueError("ODIM /where missing xscale/yscale")

    src_crs = (
        pyproj.CRS.from_proj4(projdef)
        if projdef.strip().startswith("+")
        else pyproj.CRS.from_user_input(projdef)
    )
    to_src = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

    corners = {
        key: (attr_float(where, f"{key}_lon"), attr_float(where, f"{key}_lat"))
        for key in ("LL", "UL", "UR", "LR")
    }
    if all(lon is not None and lat is not None for lon, lat in corners.values()):
        xs, ys = zip(*(to_src.transform(lon, lat) for lon, lat in corners.values()))
        extent = (min(xs), min(ys), max(xs), max(ys))
    else:
        ul_lon = attr_float(where, "UL_lon")
        ul_lat = attr_float(where, "UL_lat")
        if ul_lon is None or ul_lat is None:
            raise ValueError("ODIM /where missing corner lon/lat metadata")
        x_ul, y_ul = to_src.transform(ul_lon, ul_lat)
        extent = (x_ul, y_ul - yscale * ysize, x_ul + xscale * xsize, y_ul)

    return AreaDefinition(
        "odim_cartesian",
        "ODIM Cartesian radar image",
        "odim",
        projdef,
        xsize,
        ysize,
        extent,
    )


# -- radar: ODIM projected image -> regular lat/lon -----------------------
def reproject_odim(phys, where: dict, grid: GridSpec = TARGET_GRID):
    """Nearest-neighbour reproject an ODIM composite array onto the target grid.

    ``phys`` is the 2-D physical field (shape ``(ysize, xsize)``) and ``where``
    the ODIM ``/where`` attributes. Returns a ``(n_lat, n_lon)`` float32 array
    with NaN where the target falls outside the source raster.
    """
    import numpy as np
    from pyresample import kd_tree

    src = np.asarray(phys, dtype="float32")
    src_def = odim_area_definition(where, src.shape)
    tgt_def = web_mercator_area_definition(grid)
    north_up = kd_tree.resample_nearest(
        src_def,
        src,
        tgt_def,
        radius_of_influence=max(grid.resolution * 111_000.0 * 2.0, 5000.0),
        fill_value=np.nan,
    ).astype("float32")
    return np.flipud(north_up)


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
