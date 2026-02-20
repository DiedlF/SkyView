"""Shared fixed-grid aggregation helpers for symbols/wind endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GridModelContext:
    lat: np.ndarray
    lon: np.ndarray
    lat_groups: list[list[int]]
    lon_groups: list[list[int]]


@dataclass
class GridContext:
    lat_start: float
    lon_start: float
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    lat_cell_count: int
    lon_cell_count: int
    in_d2_grid: np.ndarray
    primary: GridModelContext
    eu: Optional[GridModelContext] = None


def build_fixed_grid(lat: np.ndarray, lon: np.ndarray, lat_min: float, lon_min: float, lat_max: float, lon_max: float, cell_size: float, zoom: int):
    """Build globally anchored fixed grid edges for the current viewport."""
    anchor_lat = float(lat.min())
    anchor_lon = float(lon.min())
    if zoom >= 12:
        anchor_lat -= cell_size / 2.0
        anchor_lon -= cell_size / 2.0

    lat_start = anchor_lat + np.floor((lat_min - anchor_lat) / cell_size) * cell_size
    lon_start = anchor_lon + np.floor((lon_min - anchor_lon) / cell_size) * cell_size
    lat_edges = np.arange(lat_start, lat_max + cell_size, cell_size)
    lon_edges = np.arange(lon_start, lon_max + cell_size, cell_size)

    lat_cell_count = max(0, len(lat_edges) - 1)
    lon_cell_count = max(0, len(lon_edges) - 1)
    return lat_start, lon_start, lat_edges, lon_edges, lat_cell_count, lon_cell_count


def build_index_groups(values: np.ndarray, start: float, cell_size: float, cell_count: int):
    """Pre-bin 1D coordinate values into fixed-grid cell index lists."""
    groups = [[] for _ in range(cell_count)]
    for idx, v in enumerate(values):
        bi = int(np.floor((float(v) - start) / cell_size))
        if 0 <= bi < cell_count:
            groups[bi].append(idx)
    return groups


def build_domain_mask(lat_edges: np.ndarray, lon_edges: np.ndarray, d2_lat_min: float, d2_lat_max: float, d2_lon_min: float, d2_lon_max: float) -> np.ndarray:
    """Boolean [i,j] mask: True where aggregation cell center is inside D2 domain."""
    lat_cell_count = max(0, len(lat_edges) - 1)
    lon_cell_count = max(0, len(lon_edges) - 1)
    if lat_cell_count == 0 or lon_cell_count == 0:
        return np.zeros((lat_cell_count, lon_cell_count), dtype=bool)

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_in = (lat_centers >= d2_lat_min) & (lat_centers <= d2_lat_max)
    lon_in = (lon_centers >= d2_lon_min) & (lon_centers <= d2_lon_max)
    return np.outer(lat_in, lon_in)


def build_grid_context(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    c_lat: np.ndarray,
    c_lon: np.ndarray,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    cell_size: float,
    zoom: int,
    d2_lat_min: float,
    d2_lat_max: float,
    d2_lon_min: float,
    d2_lon_max: float,
    c_lat_eu: Optional[np.ndarray] = None,
    c_lon_eu: Optional[np.ndarray] = None,
) -> GridContext:
    """Build shared GridContext used by both symbols and wind endpoints."""
    lat_start, lon_start, lat_edges, lon_edges, lat_cell_count, lon_cell_count = build_fixed_grid(
        lat, lon, lat_min, lon_min, lat_max, lon_max, cell_size, zoom
    )
    in_d2_grid = build_domain_mask(lat_edges, lon_edges, d2_lat_min, d2_lat_max, d2_lon_min, d2_lon_max)

    primary = GridModelContext(
        lat=c_lat,
        lon=c_lon,
        lat_groups=build_index_groups(c_lat, lat_start, cell_size, lat_cell_count),
        lon_groups=build_index_groups(c_lon, lon_start, cell_size, lon_cell_count),
    )

    eu = None
    if c_lat_eu is not None and c_lon_eu is not None:
        eu = GridModelContext(
            lat=c_lat_eu,
            lon=c_lon_eu,
            lat_groups=build_index_groups(c_lat_eu, lat_start, cell_size, lat_cell_count),
            lon_groups=build_index_groups(c_lon_eu, lon_start, cell_size, lon_cell_count),
        )

    return GridContext(
        lat_start=lat_start,
        lon_start=lon_start,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
        lat_cell_count=lat_cell_count,
        lon_cell_count=lon_cell_count,
        in_d2_grid=in_d2_grid,
        primary=primary,
        eu=eu,
    )


def choose_cell_groups(ctx: GridContext, i: int, j: int, prefer_eu: bool):
    """Return (use_eu, cli_list, clo_list) for a given cell."""
    if prefer_eu and ctx.eu is not None:
        return True, ctx.eu.lat_groups[i], ctx.eu.lon_groups[j]
    return False, ctx.primary.lat_groups[i], ctx.primary.lon_groups[j]
