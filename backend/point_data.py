"""Helpers for /api/point payload assembly."""

from __future__ import annotations

import math
import logging
import numpy as np

from soaring import (
    calc_lcl,
    classify_thermal_strength,
    calc_climb_rate_from_thermal_class,
)
from constants import CELL_SIZES_BY_ZOOM, ICON_EU_STEP_3H_START

logger = logging.getLogger(__name__)

# All NPZ keys consumed by a full point query.
# Pass this to load_data(keys=POINT_KEYS) to avoid loading unused variables.
POINT_KEYS = [
    # Core weather / cloud
    "ww", "ceiling", "clcl", "clcm", "clch", "clct", "clct_mod",
    "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf",
    # Precipitation (pre-computed rate fields, already mm/h equivalent)
    "tp_rate", "rain_rate", "snow_rate", "hail_rate",
    # Boundary layer / atmosphere
    "mh", "ashfl_s", "relhum_2m",
    "t_2m", "td_2m",
    "t_950hpa", "t_850hpa", "t_700hpa", "t_500hpa", "t_300hpa",
    # Wind
    "u_10m", "v_10m", "vmax_10m",
    "u_850hpa", "v_850hpa",
    "u_700hpa", "v_700hpa",
    "u_500hpa", "v_500hpa",
    "u_300hpa", "v_300hpa",
]

# Temperature: values above this threshold are treated as Kelvin → convert to Celsius
_KELVIN_THRESHOLD = 200.0
_KELVIN_OFFSET = 273.15

_MOISTURE_LABELS = ("sehr_feucht", "feucht", "moderat", "trocken")


# ─── Scalar helpers ────────────────────────────────────────────────────────────

def _safe_get(d: dict, key: str, i: int, j: int) -> float | None:
    """Extract a single finite scalar from a 2-D field.

    Returns None if the key is absent, the value is NaN/Inf, or indexing fails.
    This replaces the old pattern:
        float(np.nanmean(d[key][np.ix_([i], [j])]))
    which was O(np overhead) on a trivial 1×1 slice.
    """
    if key not in d:
        return None
    try:
        v = float(d[key][i, j])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _cell_agg(d: dict, key: str, ix, stat: str = "mean") -> float | None:
    """Aggregate over a cell defined by a pre-computed np.ix_ index pair.

    Used only where cell-averaging is semantically meaningful (wind).
    """
    if key not in d:
        return None
    try:
        cell = d[key][ix]
        fn = np.nanmax if stat == "max" else (np.nanmin if stat == "min" else np.nanmean)
        v = float(fn(cell))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _to_celsius(v: float) -> float:
    """Convert Kelvin → Celsius if value looks like Kelvin, else pass through."""
    return v - _KELVIN_OFFSET if v > _KELVIN_THRESHOLD else v


def _precip_step_hours(model_used: str | None, step: int | None) -> float:
    if model_used == "icon_eu" and step is not None and int(step) >= ICON_EU_STEP_3H_START:
        return 3.0
    return 1.0


# ─── Explorer path (scalar dict → overlay values) ──────────────────────────────

def build_overlay_values_from_raw(
    values: dict,
    wind_level: str = "10m",
    model_used: str | None = None,
    step: int | None = None,
) -> dict:
    """Build semantic overlay_values from a raw single-point variable dict (scalars).

    Used by Explorer /api/point; kept here to avoid semantic drift with Skyview.
    """
    ov: dict = {}
    step_h = _precip_step_hours(model_used, step)

    # Precipitation
    tp = values.get("tot_prec")
    rg = values.get("rain_gsp")
    rc = values.get("rain_con")
    sg = values.get("snow_gsp")
    sc = values.get("snow_con")
    gg = values.get("grau_gsp")
    if tp is not None:
        ov["total_precip"] = round(float(tp) / step_h, 2)
    if rg is not None or rc is not None:
        ov["rain"] = round((float(rg or 0.0) + float(rc or 0.0)) / step_h, 2)
    if sg is not None or sc is not None:
        ov["snow"] = round((float(sg or 0.0) + float(sc or 0.0)) / step_h, 2)
    if gg is not None:
        ov["hail"] = round(float(gg) / step_h, 2)

    # Cloud cover
    for out_key, src_key in [
        ("clouds_low", "clcl"), ("clouds_mid", "clcm"),
        ("clouds_high", "clch"), ("clouds_total", "clct"),
    ]:
        if values.get(src_key) is not None:
            ov[out_key] = round(values[src_key], 1)
    if values.get("clct_mod") is not None:
        v = values["clct_mod"]
        if v <= 1.5:
            v *= 100.0
        ov["clouds_total_mod"] = round(v, 1)

    # Scalars
    for k in ("mh", "ashfl_s", "relhum_2m"):
        if values.get(k) is not None:
            ov[k] = round(float(values[k]), 1)
    if values.get("t_2m") is not None and values.get("td_2m") is not None:
        ov["dew_spread_2m"] = round(float(values["t_2m"]) - float(values["td_2m"]), 1)

    # Temperature levels
    for tk in ("t_2m", "t_950hpa", "t_850hpa", "t_700hpa", "t_500hpa", "t_300hpa"):
        tv = values.get(tk)
        if tv is not None:
            ov[tk] = round(_to_celsius(float(tv)), 1)

    # Weather
    if values.get("ww") is not None:
        ov["sigwx"] = int(values["ww"])
    if values.get("cape_ml") is not None:
        ov["thermals"] = round(values["cape_ml"], 1)
    if values.get("ceiling") is not None and values["ceiling"] > 0:
        ov["ceiling"] = round(values["ceiling"], 0)
    if values.get("hbas_sc") is not None and values["hbas_sc"] > 0:
        ov["cloud_base"] = round(values["hbas_sc"], 0)
    if values.get("htop_dc") is not None and values["htop_dc"] > 0:
        ov["dry_conv_top"] = round(values["htop_dc"], 0)
    if values.get("lpi_max") is not None:
        ov["lpi"] = round(values["lpi_max"], 1)
    elif values.get("lpi") is not None:
        ov["lpi"] = round(values["lpi"], 1)
    if values.get("htop_sc") is not None and values.get("hbas_sc") is not None:
        thick = max(0.0, values["htop_sc"] - values["hbas_sc"])
        ov["conv_thickness"] = round(thick, 0)

    # Wind
    gust_mode = (wind_level == "gust10m")
    uk = "u_10m" if (wind_level == "10m" or gust_mode) else f"u_{wind_level}hpa"
    vk = "v_10m" if (wind_level == "10m" or gust_mode) else f"v_{wind_level}hpa"
    if values.get(uk) is not None and values.get(vk) is not None:
        u = float(values[uk])
        v = float(values[vk])
        if gust_mode and values.get("vmax_10m") is not None:
            sp = float(values["vmax_10m"])
        else:
            sp = math.sqrt(u * u + v * v)
        ov["wind_speed"] = round(sp * 1.94384, 1)
        ov["wind_dir"] = round((math.degrees(math.atan2(-u, -v)) + 360) % 360, 0)

    return ov


# ─── Skyview path (NPZ arrays → overlay values) ───────────────────────────────

def build_overlay_values(
    d: dict,
    li: np.ndarray,
    lo: np.ndarray,
    ww_max: int,
    ceil_cell: np.ndarray,
    wind_level: str,
    zoom: int | None,
    lat: float,
    lon: float,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    model_used: str,
    step: int,
) -> dict:
    """Build overlay_values for /api/point from NPZ data arrays.

    li / lo are always single-element arrays (nearest-neighbour point query).
    All non-wind lookups use direct scalar indexing d[var][i0, j0] instead
    of np.ix_ + nanmean on a 1×1 sub-array.
    """
    i0, j0 = int(li[0]), int(lo[0])

    ov: dict = {"sigwx": ww_max}

    # ── Precipitation (pre-computed rate fields) ──────────────────────────────
    # These are already in mm/h; no step-hour division needed here.
    for out_key, src_key in [
        ("total_precip", "tp_rate"),
        ("rain",         "rain_rate"),
        ("snow",         "snow_rate"),
        ("hail",         "hail_rate"),
    ]:
        v = _safe_get(d, src_key, i0, j0)
        if v is not None:
            ov[out_key] = round(max(v, 0.0), 2)
        else:
            ov[out_key] = None

    # ── Cloud cover ───────────────────────────────────────────────────────────
    for out_key, src_key in [
        ("clouds_low",  "clcl"),
        ("clouds_mid",  "clcm"),
        ("clouds_high", "clch"),
        ("clouds_total","clct"),
    ]:
        v = _safe_get(d, src_key, i0, j0)
        if v is not None:
            ov[out_key] = round(v, 1)

    clct_mod = _safe_get(d, "clct_mod", i0, j0)
    if clct_mod is not None:
        if clct_mod <= 1.5:
            clct_mod *= 100.0
        ov["clouds_total_mod"] = round(clct_mod, 1)

    # ── Thermodynamics ────────────────────────────────────────────────────────
    cape = _safe_get(d, "cape_ml", i0, j0)
    if cape is not None:
        ov["thermals"] = round(cape, 1)

    for key in ("mh", "ashfl_s", "relhum_2m"):
        v = _safe_get(d, key, i0, j0)
        if v is not None:
            ov[key] = round(v, 1)

    # Extract shared temperature/humidity scalars once for reuse below
    t2m  = _safe_get(d, "t_2m",  i0, j0)
    td2m = _safe_get(d, "td_2m", i0, j0)
    hsurf = _safe_get(d, "hsurf", i0, j0)
    mh    = _safe_get(d, "mh",    i0, j0)
    ashfl = _safe_get(d, "ashfl_s", i0, j0)

    if t2m is not None and td2m is not None:
        ov["dew_spread_2m"] = round(t2m - td2m, 1)

    # Temperature levels → Celsius
    for tk in ("t_2m", "t_950hpa", "t_850hpa", "t_700hpa", "t_500hpa", "t_300hpa"):
        v = _safe_get(d, tk, i0, j0)
        if v is not None:
            ov[tk] = round(_to_celsius(v), 1)

    # ── Cloud heights ─────────────────────────────────────────────────────────
    ceil_vals = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20_000)] if ceil_cell.size else np.array([])
    ov["ceiling"] = round(float(np.min(ceil_vals)), 0) if len(ceil_vals) > 0 else None

    hbas = _safe_get(d, "hbas_sc", i0, j0)
    ov["cloud_base"] = round(hbas, 0) if (hbas is not None and hbas > 0) else None

    htop_dc = _safe_get(d, "htop_dc", i0, j0)
    ov["dry_conv_top"] = round(htop_dc, 0) if (htop_dc is not None and htop_dc > 0) else None

    htop_sc = _safe_get(d, "htop_sc", i0, j0)
    if htop_sc is not None and hbas is not None:
        thick = max(0.0, htop_sc - hbas)
        ov["conv_thickness"] = round(thick, 0) if thick > 0 else None

    lpi = _safe_get(d, "lpi_max", i0, j0)
    if lpi is None:
        lpi = _safe_get(d, "lpi", i0, j0)
    if lpi is not None:
        ov["lpi"] = round(lpi, 1)

    # ── Soaring / thermal parameters ──────────────────────────────────────────
    # Requires: ashfl, mh, t2m, td2m, hsurf + at least one upper-level temperature
    if ashfl is not None and mh is not None and t2m is not None:
        ov["bl_height"] = round(mh, 0)

        if td2m is not None and hsurf is not None:
            # LCL (Lifting Condensation Level) — cloud base estimate
            # calc_lcl accepts numpy arrays; scalars work via numpy ufuncs
            lcl = calc_lcl(
                np.float64(t2m),
                np.float64(td2m),
                np.float64(hsurf),
            )
            ov["lcl"] = round(float(lcl), 0)

            # Thermal class — pick lapse-rate reference level by terrain elevation
            t_850 = _safe_get(d, "t_850hpa", i0, j0)
            t_700 = _safe_get(d, "t_700hpa", i0, j0)
            t_500 = _safe_get(d, "t_500hpa", i0, j0)
            t_300 = _safe_get(d, "t_300hpa", i0, j0)

            tupper: float | None = t_850
            z_upper_m = 1500.0
            if hsurf > 1000.0 and t_700 is not None:  #intentionally lower threshold than 1500m to avoid using 700 hPa level in cases where 850 hPa is just slightly above ground (e.g. high terrain or warm low pressure)
                tupper, z_upper_m = t_700, 3000.0
            if hsurf > 2500.0 and t_500 is not None: #intentionally lower threshold than 3000m to avoid using 500 hPa level in cases where 700 hPa is just slightly above ground
                tupper, z_upper_m = t_500, 5500.0
            if hsurf > 4000.0 and t_300 is not None: #intentionally lower threshold than 5500m to avoid using 300 hPa level in cases where 500 hPa is just slightly above ground
                tupper, z_upper_m = t_300, 9000.0

            if tupper is not None:
                delta_alt_km = max((z_upper_m - hsurf) / 1000.0, 0.1)
                t2m_c  = _to_celsius(t2m)
                td2m_c = _to_celsius(td2m)
                tu_c   = _to_celsius(tupper)

                thermal_class, lapse_factor, moisture_code = classify_thermal_strength(
                    np.float64(t2m_c),
                    np.float64(td2m_c),
                    np.float64(tu_c),
                    np.float64(delta_alt_km),
                )
                climb = calc_climb_rate_from_thermal_class(thermal_class)

                ov["climb_rate"]    = round(float(climb), 1)
                ov["thermal_class"] = int(round(float(thermal_class)))
                ov["lapse_factor"]  = round(float(lapse_factor), 2)
                mc = int(round(float(moisture_code)))
                ov["moisture_class"] = _MOISTURE_LABELS[max(0, min(3, mc))]

    # ── Wind (cell-averaged to match map barb display) ────────────────────────
    gust_mode = (wind_level == "gust10m")
    u_key = "u_10m" if (wind_level == "10m" or gust_mode) else f"u_{wind_level}hpa"
    v_key = "v_10m" if (wind_level == "10m" or gust_mode) else f"v_{wind_level}hpa"

    if u_key in d and v_key in d:
        # Determine aggregation cell: match the symbol/barb grid shown on the map
        if zoom is not None:
            cell_size = CELL_SIZES_BY_ZOOM.get(int(zoom), 0.25)
            anchor_lat = float(lat_arr.min())
            anchor_lon = float(lon_arr.min())
            lat_lo = anchor_lat + math.floor((lat - anchor_lat) / cell_size) * cell_size
            lon_lo = anchor_lon + math.floor((lon - anchor_lon) / cell_size) * cell_size
            wli = np.where((lat_arr >= lat_lo) & (lat_arr < lat_lo + cell_size))[0]
            wlo = np.where((lon_arr >= lon_lo) & (lon_arr < lon_lo + cell_size))[0]
            if len(wli) == 0 or len(wlo) == 0:
                wli, wlo = li, lo
        else:
            wli, wlo = li, lo

        wix = np.ix_(wli, wlo)
        u_mean = _cell_agg(d, u_key, wix, "mean")
        v_mean = _cell_agg(d, v_key, wix, "mean")

        if u_mean is not None and v_mean is not None:
            if gust_mode:
                gust = _cell_agg(d, "vmax_10m", wix, "max")
                speed_ms = gust if gust is not None else math.sqrt(u_mean**2 + v_mean**2)
            else:
                speed_ms = math.sqrt(u_mean**2 + v_mean**2)
            ov["wind_speed"] = round(speed_ms * 1.94384, 1)
            ov["wind_dir"]   = round((math.degrees(math.atan2(-u_mean, -v_mean)) + 360) % 360, 0)

    # ── NaN/Inf guard (final pass) ────────────────────────────────────────────
    for k, v in list(ov.items()):
        if isinstance(v, float) and not math.isfinite(v):
            ov[k] = None

    return ov
