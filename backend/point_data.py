"""Helpers for /api/point payload assembly."""

from __future__ import annotations

import math
import numpy as np

from soaring import calc_wstar, calc_lcl, calc_thermal_height, calc_reachable_distance


CELL_SIZES = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.02}


def build_overlay_values_from_raw(values: dict, wind_level: str = "10m") -> dict:
    """Build semantic overlay_values from raw single-point variable values.

    Used by Explorer /api/point; kept here to avoid drift with Skyview semantics.
    """
    ov = {}

    prr = values.get("prr_gsp")
    prs = values.get("prs_gsp")
    prg = values.get("prg_gsp")
    if prr is not None and prs is not None and prg is not None:
        ov["total_precip"] = round((prr + prs + prg) * 3600.0, 2)
        ov["rain"] = round(prr * 3600.0, 2)
        ov["snow"] = round(prs * 3600.0, 2)
        ov["hail"] = round(prg * 3600.0, 2)

    if values.get("clcl") is not None:
        ov["clouds_low"] = round(values["clcl"], 1)
    if values.get("clcm") is not None:
        ov["clouds_mid"] = round(values["clcm"], 1)
    if values.get("clch") is not None:
        ov["clouds_high"] = round(values["clch"], 1)
    if values.get("clct") is not None:
        ov["clouds_total"] = round(values["clct"], 1)
    if values.get("clct_mod") is not None:
        v = values["clct_mod"]
        if v <= 1.5:
            v *= 100.0
        ov["clouds_total_mod"] = round(v, 1)

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

    if values.get("htop_sc") is not None and values.get("hbas_sc") is not None:
        thick = max(0.0, values["htop_sc"] - values["hbas_sc"])
        ov["conv_thickness"] = round(thick, 0)

    if wind_level == "10m":
        uk, vk = "u_10m", "v_10m"
    else:
        uk, vk = f"u_{wind_level}hpa", f"v_{wind_level}hpa"
    if values.get(uk) is not None and values.get(vk) is not None:
        u = float(values[uk])
        v = float(values[vk])
        sp = (u * u + v * v) ** 0.5
        ov["wind_speed"] = round(sp * 1.94384, 1)
        ov["wind_dir"] = round((math.degrees(math.atan2(-u, -v)) + 360) % 360, 0)

    return ov


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
) -> dict:
    overlay_values = {
        "total_precip": None, "rain": None, "snow": None, "hail": None,
        "clouds": None, "thermals": None, "ceiling": None, "cloud_base": None,
        "sigwx": ww_max,
    }

    try:
        prr = float(np.nanmean(d["prr_gsp"][np.ix_(li, lo)])) if "prr_gsp" in d else 0.0
        prs = float(np.nanmean(d["prs_gsp"][np.ix_(li, lo)])) if "prs_gsp" in d else 0.0
        prg = float(np.nanmean(d["prg_gsp"][np.ix_(li, lo)])) if "prg_gsp" in d else 0.0
        overlay_values["total_precip"] = round((prr + prs + prg) * 3600, 2)
        overlay_values["rain"] = round(prr * 3600, 2)
        overlay_values["snow"] = round(prs * 3600, 2)
        overlay_values["hail"] = round(prg * 3600, 2)
    except Exception:
        pass

    if "clcl" in d:
        overlay_values["clouds_low"] = round(float(np.nanmean(d["clcl"][np.ix_(li, lo)])), 1)
    if "clcm" in d:
        overlay_values["clouds_mid"] = round(float(np.nanmean(d["clcm"][np.ix_(li, lo)])), 1)
    if "clch" in d:
        overlay_values["clouds_high"] = round(float(np.nanmean(d["clch"][np.ix_(li, lo)])), 1)
    if "clct" in d:
        overlay_values["clouds_total"] = round(float(np.nanmean(d["clct"][np.ix_(li, lo)])), 1)
    if "clct_mod" in d:
        clct_mod_val = float(np.nanmean(d["clct_mod"][np.ix_(li, lo)]))
        if np.isfinite(clct_mod_val) and clct_mod_val <= 1.5:
            clct_mod_val *= 100.0
        overlay_values["clouds_total_mod"] = round(clct_mod_val, 1) if np.isfinite(clct_mod_val) else None
    if "cape_ml" in d:
        overlay_values["thermals"] = round(float(np.nanmax(d["cape_ml"][np.ix_(li, lo)])), 1)

    ceil_vals = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20000)] if ceil_cell.size else np.array([])
    overlay_values["ceiling"] = round(float(np.min(ceil_vals)), 0) if len(ceil_vals) > 0 else None
    if "hbas_sc" in d:
        hbas_vals = d["hbas_sc"][np.ix_(li, lo)]
        hbas_valid = hbas_vals[hbas_vals > 0]
        overlay_values["cloud_base"] = round(float(np.min(hbas_valid)), 0) if len(hbas_valid) > 0 else None
    if "htop_dc" in d:
        htop_dc_vals = d["htop_dc"][np.ix_(li, lo)]
        htop_dc_valid = htop_dc_vals[htop_dc_vals > 0]
        overlay_values["dry_conv_top"] = round(float(np.max(htop_dc_valid)), 0) if len(htop_dc_valid) > 0 else None
    if "htop_sc" in d and "hbas_sc" in d:
        thick = np.maximum(0, d["htop_sc"][np.ix_(li, lo)] - d["hbas_sc"][np.ix_(li, lo)])
        thick_valid = thick[thick > 0]
        overlay_values["conv_thickness"] = round(float(np.max(thick_valid)), 0) if len(thick_valid) > 0 else None

    if "ashfl_s" in d and "mh" in d and "t_2m" in d:
        ashfl_cell = d["ashfl_s"][np.ix_(li, lo)]
        mh_cell = d["mh"][np.ix_(li, lo)]
        t2m_cell = d["t_2m"][np.ix_(li, lo)]
        wstar = calc_wstar(ashfl_cell, None, mh_cell, t2m_cell)
        overlay_values["wstar"] = round(float(np.nanmean(wstar)), 1)
        overlay_values["climb_rate"] = round(max(float(np.nanmean(wstar)) - 0.7, 0.0), 1)
        overlay_values["bl_height"] = round(float(np.nanmean(mh_cell)), 0)
        if "td_2m" in d and "hsurf" in d:
            td2m_cell = d["td_2m"][np.ix_(li, lo)]
            hsurf_cell = d["hsurf"][np.ix_(li, lo)]
            lcl = calc_lcl(t2m_cell, td2m_cell, hsurf_cell)
            thermal_agl = calc_thermal_height(mh_cell, lcl, hsurf_cell)
            reachable = calc_reachable_distance(thermal_agl)
            overlay_values["lcl"] = round(float(np.nanmean(lcl)), 0)
            overlay_values["reachable"] = round(float(np.nanmean(reachable)), 1)

    # Wind
    if wind_level == "10m":
        u_key, v_key = "u_10m", "v_10m"
    else:
        u_key, v_key = f"u_{wind_level}hpa", f"v_{wind_level}hpa"

    if u_key in d and v_key in d:
        if zoom is not None:
            cell_size = CELL_SIZES.get(int(zoom), 0.25)
            anchor_lat = float(lat_arr.min())
            anchor_lon = float(lon_arr.min())
            lat_lo = anchor_lat + math.floor((lat - anchor_lat) / cell_size) * cell_size
            lon_lo = anchor_lon + math.floor((lon - anchor_lon) / cell_size) * cell_size
            lat_hi = lat_lo + cell_size
            lon_hi = lon_lo + cell_size

            wli = np.where((lat_arr >= lat_lo) & (lat_arr < lat_hi))[0]
            wlo = np.where((lon_arr >= lon_lo) & (lon_arr < lon_hi))[0]
            if len(wli) == 0 or len(wlo) == 0:
                wli, wlo = li, lo
        else:
            wli, wlo = li, lo

        u_cell = d[u_key][np.ix_(wli, wlo)]
        v_cell = d[v_key][np.ix_(wli, wlo)]
        u_mean = float(np.nanmean(u_cell))
        v_mean = float(np.nanmean(v_cell))
        if np.isfinite(u_mean) and np.isfinite(v_mean):
            overlay_values["wind_speed"] = round(math.sqrt(u_mean**2 + v_mean**2) * 1.94384, 1)
            overlay_values["wind_dir"] = round((math.degrees(math.atan2(-u_mean, -v_mean)) + 360) % 360, 0)

    return overlay_values
