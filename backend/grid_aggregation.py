"""Shared fixed-grid aggregation helpers for symbols/wind endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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


def scatter_cell_stats(
    c_lat: np.ndarray,
    c_lon: np.ndarray,
    ctx: "GridContext",
    ww: np.ndarray,
    cape: np.ndarray,
    ceil_arr: np.ndarray,
    cape_conv_threshold: float,
    ceil_valid_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized O(N_grid) per-cell aggregate stats — no Python cell loop.

    Computes three stats arrays shaped (lat_cell_count, lon_cell_count):
      cell_max_ww   — float32, NaN where cell has no finite ww data
      cell_any_cape — bool, True where any grid point has cape > cape_conv_threshold
      cell_any_ceil — bool, True where any grid point has a valid ceiling

    Used as a pre-pass before the symbol cell loop so that "clear" cells can be
    resolved with O(1) array lookups instead of per-cell np.ix_ extractions.
    Eliminates the two most expensive per-cell np.ix_ calls for the common case.

    Memory overhead (D2 full domain ~730×1200 = 876K pts):
      flat_cell broadcast: ~7 MB (int64), freed immediately after scatter
      result arrays: 3 × (n_cells × dtype) — tiny (< 1 KB for zoom 5)
    """
    shape = (ctx.lat_cell_count, ctx.lon_cell_count)
    nan_ww = np.full(shape, np.nan, dtype=np.float32)
    no_cape = np.zeros(shape, dtype=bool)
    no_ceil = np.zeros(shape, dtype=bool)

    if ctx.lat_cell_count == 0 or ctx.lon_cell_count == 0:
        return nan_ww, no_cape, no_ceil
    if c_lat.size == 0 or c_lon.size == 0:
        return nan_ww, no_cape, no_ceil
    if ww.shape != (c_lat.size, c_lon.size):
        return nan_ww, no_cape, no_ceil

    # Derive cell size from stored edges (same cell_size used to build ctx)
    if len(ctx.lat_edges) < 2 or len(ctx.lon_edges) < 2:
        return nan_ww, no_cape, no_ceil
    lat_cell_size = float(ctx.lat_edges[1] - ctx.lat_edges[0])
    lon_cell_size = float(ctx.lon_edges[1] - ctx.lon_edges[0])

    # Bin each coordinate into its cell index  (n_lat,) and (n_lon,)
    lat_bins = np.floor((c_lat.astype(np.float64) - ctx.lat_start) / lat_cell_size).astype(np.intp)
    lon_bins = np.floor((c_lon.astype(np.float64) - ctx.lon_start) / lon_cell_size).astype(np.intp)

    # Domain validity masks for each axis
    lat_ok = (lat_bins >= 0) & (lat_bins < ctx.lat_cell_count)  # (n_lat,)
    lon_ok = (lon_bins >= 0) & (lon_bins < ctx.lon_cell_count)  # (n_lon,)

    # Broadcast to full 2D grid: domain_ok[r, c] = True if both axes in range
    domain_ok = lat_ok[:, None] & lon_ok[None, :]               # (n_lat, n_lon)

    # Flat cell index: cell_flat[r, c] = lat_bin[r] * ncols + lon_bin[c]
    flat_cell = (lat_bins[:, None] * ctx.lon_cell_count + lon_bins[None, :]).ravel()  # (n_lat*n_lon,)
    domain_v  = domain_ok.ravel()
    n_cells   = ctx.lat_cell_count * ctx.lon_cell_count

    # ── max ww per cell (scatter-max) ─────────────────────────────────────────
    # Initialize with -inf so np.maximum.at works correctly (NaN would propagate
    # and leave everything as NaN). Convert remaining -inf (empty cells) to NaN afterward.
    cell_max_ww_flat = np.full(n_cells, -np.inf, dtype=np.float32)
    ww_v = ww.ravel()
    fin_ww = domain_v & np.isfinite(ww_v)
    if fin_ww.any():
        np.maximum.at(cell_max_ww_flat, flat_cell[fin_ww], ww_v[fin_ww])
    cell_max_ww_flat[cell_max_ww_flat == -np.inf] = np.nan  # empty cells → NaN

    # ── any cape > threshold per cell (scatter-set) ───────────────────────────
    cell_any_cape_flat = np.zeros(n_cells, dtype=bool)
    cape_v = cape.ravel()
    cape_hit = domain_v & np.isfinite(cape_v) & (cape_v > cape_conv_threshold)
    if cape_hit.any():
        cell_any_cape_flat[flat_cell[cape_hit]] = True

    # ── any valid ceiling per cell (scatter-set) ──────────────────────────────
    cell_any_ceil_flat = np.zeros(n_cells, dtype=bool)
    ceil_v = ceil_arr.ravel()
    ceil_hit = domain_v & np.isfinite(ceil_v) & (ceil_v > 0) & (ceil_v < ceil_valid_max)
    if ceil_hit.any():
        cell_any_ceil_flat[flat_cell[ceil_hit]] = True

    return (
        cell_max_ww_flat.reshape(shape),
        cell_any_cape_flat.reshape(shape),
        cell_any_ceil_flat.reshape(shape),
    )
