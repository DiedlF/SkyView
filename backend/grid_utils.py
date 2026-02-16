"""Shared grid/bbox slicing helpers for Skyview + Explorer."""

from __future__ import annotations

import numpy as np


def get_grid_bounds(lat, lon):
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def bbox_indices(lat, lon, lat_min, lon_min, lat_max, lon_max):
    """Return (lat_indices, lon_indices) for points within bbox.

    Returns (None, None) when bbox covers full grid.
    """
    grid_lat_min, grid_lat_max = float(lat.min()), float(lat.max())
    grid_lon_min, grid_lon_max = float(lon.min()), float(lon.max())

    if lat_min <= grid_lat_min and lat_max >= grid_lat_max and lon_min <= grid_lon_min and lon_max >= grid_lon_max:
        return None, None

    eps = 0.001
    lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
    lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]
    if len(li) == 0 or len(lo) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return li, lo


def slice_array(arr, li, lo):
    if li is None:
        return arr
    return arr[np.ix_(li, lo)]
