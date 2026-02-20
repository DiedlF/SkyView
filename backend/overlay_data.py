"""Shared overlay data-key and computed-field helpers."""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from soaring import (
    calc_wstar,
    calc_lcl,
    calc_thermal_height,
    calc_reachable_distance,
    classify_thermal_strength,
    calc_climb_rate_from_thermal_class,
)
from constants import PRECIP_RATE_FIELD_BY_LAYER_VAR


def build_overlay_keys(cfg: dict) -> list[str]:
    """Determine data keys required for a given overlay config."""
    overlay_keys = ["lat", "lon"]
    if cfg.get("computed"):
        v = cfg["var"]
        if v in ("total_precip", "rain_amount", "snow_amount", "hail_amount"):
            overlay_keys += ["tp_rate", "rain_rate", "snow_rate", "hail_rate"]
        elif v == "conv_thickness":
            overlay_keys += ["htop_sc", "hbas_sc"]
        elif v == "wstar":
            overlay_keys += ["ashfl_s", "mh", "t_2m"]
        elif v == "climb_rate":
            overlay_keys += ["t_2m", "td_2m", "hsurf", "mh", "ww", "t_850hpa", "t_700hpa", "t_500hpa", "t_300hpa"]
        elif v in ("lcl", "reachable"):
            overlay_keys += ["ashfl_s", "mh", "t_2m", "td_2m", "hsurf"]
        elif v == "dew_spread_2m":
            overlay_keys += ["t_2m", "td_2m"]
    else:
        overlay_keys.append(cfg["var"])
    return overlay_keys


def normalize_clouds_total_mod(arr: np.ndarray) -> np.ndarray:
    """Normalize clct_mod to percent if it appears fractional (0..1)."""
    vmax_local = float(np.nanmax(arr)) if np.size(arr) else float("nan")
    if np.isfinite(vmax_local) and vmax_local <= 1.5:
        return arr * 100.0
    return arr


def _precip_precomputed_field(var: str, d: dict) -> np.ndarray:
    key = PRECIP_RATE_FIELD_BY_LAYER_VAR.get(var)
    if not key:
        raise HTTPException(400, f"Unknown precip variable: {var}")
    if key not in d:
        raise HTTPException(404, f"Precomputed precipitation field missing: {key}")
    return d[key]


def compute_computed_field_cropped(var: str, d: dict, li: np.ndarray, lo: np.ndarray, model_used: str, step: int) -> np.ndarray:
    """Compute cropped computed overlay field for /api/overlay."""
    if var in ("total_precip", "rain_amount", "snow_amount", "hail_amount"):
        return _precip_precomputed_field(var, d)[np.ix_(li, lo)]

    if var == "conv_thickness":
        htop_sc = d["htop_sc"][np.ix_(li, lo)] if "htop_sc" in d else np.zeros((len(li), len(lo)))
        hbas_sc = d["hbas_sc"][np.ix_(li, lo)] if "hbas_sc" in d else np.zeros((len(li), len(lo)))
        return np.maximum(0, htop_sc - hbas_sc)

    if var == "dew_spread_2m":
        if "t_2m" not in d or "td_2m" not in d:
            raise HTTPException(404, "Dew spread data not available for this timestep")
        return d["t_2m"][np.ix_(li, lo)] - d["td_2m"][np.ix_(li, lo)]

    if var in ("wstar", "climb_rate", "lcl", "reachable"):
        if var == "wstar":
            if "ashfl_s" not in d or "mh" not in d or "t_2m" not in d:
                raise HTTPException(404, "Soaring data not available for this timestep (re-ingestion needed)")
            ashfl_s = d["ashfl_s"][np.ix_(li, lo)]
            mh = d["mh"][np.ix_(li, lo)]
            t_2m = d["t_2m"][np.ix_(li, lo)]
            return calc_wstar(ashfl_s, None, mh, t_2m, dt_seconds=3600)

        if var == "climb_rate":
            if "t_2m" not in d or "td_2m" not in d or "t_850hpa" not in d or "hsurf" not in d:
                raise HTTPException(404, "Climb-rate data not available for this timestep (missing t_2m/td_2m/hsurf/t_850hpa)")
            t2m_c = d["t_2m"][np.ix_(li, lo)] - 273.15
            td2m_c = d["td_2m"][np.ix_(li, lo)] - 273.15
            hsurf = d["hsurf"][np.ix_(li, lo)]
            t_upper_c = d["t_850hpa"][np.ix_(li, lo)] - 273.15
            z_upper_m = np.full_like(hsurf, np.nan, dtype=np.float32)
            valid = np.isfinite(hsurf) & np.isfinite(t2m_c) & np.isfinite(td2m_c) & np.isfinite(t_upper_c)
            z_upper_m = np.where(valid, 1500.0, z_upper_m)
            # High terrain adjustment: above ~850 hPa level, use next available upper level.
            if "t_700hpa" in d:
                t700_c = d["t_700hpa"][np.ix_(li, lo)] - 273.15
                m700 = valid & (hsurf > 1500.0) & np.isfinite(t700_c)
                t_upper_c = np.where(m700, t700_c, t_upper_c)
                z_upper_m = np.where(m700, 3000.0, z_upper_m)
            if "t_500hpa" in d:
                t500_c = d["t_500hpa"][np.ix_(li, lo)] - 273.15
                m500 = valid & (hsurf > 3000.0) & np.isfinite(t500_c)
                t_upper_c = np.where(m500, t500_c, t_upper_c)
                z_upper_m = np.where(m500, 5500.0, z_upper_m)
            if "t_300hpa" in d:
                t300_c = d["t_300hpa"][np.ix_(li, lo)] - 273.15
                m300 = valid & (hsurf > 5500.0) & np.isfinite(t300_c)
                t_upper_c = np.where(m300, t300_c, t_upper_c)
                z_upper_m = np.where(m300, 9000.0, z_upper_m)

            delta_alt_km = np.where(valid, np.maximum((z_upper_m - hsurf) / 1000.0, 0.1), np.nan)
            thermal_class, _, _ = classify_thermal_strength(t2m_c, td2m_c, t_upper_c, delta_alt_km)
            out = calc_climb_rate_from_thermal_class(thermal_class)
            domain_mask = np.ones_like(valid, dtype=bool)
            if "mh" in d:
                domain_mask = np.isfinite(d["mh"][np.ix_(li, lo)])
            elif "ww" in d:
                domain_mask = np.isfinite(d["ww"][np.ix_(li, lo)])
            out = np.where(valid & domain_mask, out, np.nan)
            return out

        if "ashfl_s" not in d or "mh" not in d or "t_2m" not in d:
            raise HTTPException(404, "Soaring data not available for this timestep (re-ingestion needed)")
        ashfl_s = d["ashfl_s"][np.ix_(li, lo)]
        mh = d["mh"][np.ix_(li, lo)]
        t_2m = d["t_2m"][np.ix_(li, lo)]

        if "td_2m" not in d or "hsurf" not in d:
            if var == "lcl":
                raise HTTPException(404, "LCL data not available (re-ingestion needed)")
            raise HTTPException(404, "Reachable data not available (re-ingestion needed)")

        td_2m = d["td_2m"][np.ix_(li, lo)]
        hsurf = d["hsurf"][np.ix_(li, lo)]
        lcl_amsl = calc_lcl(t_2m, td_2m, hsurf)
        if var == "lcl":
            return lcl_amsl
        thermal_agl = calc_thermal_height(mh, lcl_amsl, hsurf)
        return calc_reachable_distance(thermal_agl)

    raise HTTPException(400, f"Unknown computed variable: {var}")


def compute_computed_field_full(var: str, d: dict, model_used: str, step: int) -> np.ndarray:
    """Compute full-grid computed overlay field for /api/overlay_tile."""
    if var in ("total_precip", "rain_amount", "snow_amount", "hail_amount"):
        return _precip_precomputed_field(var, d)
    if var == "conv_thickness":
        return np.maximum(0, d["htop_sc"] - d["hbas_sc"])
    if var == "wstar":
        if "ashfl_s" not in d or "mh" not in d or "t_2m" not in d:
            raise HTTPException(404, "Soaring data not available for this timestep (re-ingestion needed)")
        return calc_wstar(d["ashfl_s"], None, d["mh"], d["t_2m"], dt_seconds=3600)
    if var == "climb_rate":
        if "t_2m" not in d or "td_2m" not in d or "t_850hpa" not in d or "hsurf" not in d:
            raise HTTPException(404, "Climb-rate data not available for this timestep (missing t_2m/td_2m/hsurf/t_850hpa)")
        t2m_c = d["t_2m"] - 273.15
        td2m_c = d["td_2m"] - 273.15
        hsurf = d["hsurf"]
        t_upper_c = d["t_850hpa"] - 273.15
        z_upper_m = np.full_like(hsurf, np.nan, dtype=np.float32)
        valid = np.isfinite(hsurf) & np.isfinite(t2m_c) & np.isfinite(td2m_c) & np.isfinite(t_upper_c)
        z_upper_m = np.where(valid, 1500.0, z_upper_m)
        if "t_700hpa" in d:
            t700_c = d["t_700hpa"] - 273.15
            m700 = valid & (hsurf > 1000.0) & np.isfinite(t700_c) #intentionally lower threshold than 1500m to avoid using 700 hPa level in cases where 850 hPa is just slightly above ground (e.g. high terrain or warm low pressure)
            t_upper_c = np.where(m700, t700_c, t_upper_c)
            z_upper_m = np.where(m700, 3000.0, z_upper_m)
        if "t_500hpa" in d:
            t500_c = d["t_500hpa"] - 273.15
            m500 = valid & (hsurf > 2500.0) & np.isfinite(t500_c) #intentionally lower threshold than 3000m to avoid using 500 hPa level in cases where 700 hPa is just slightly above ground
            t_upper_c = np.where(m500, t500_c, t_upper_c)
            z_upper_m = np.where(m500, 5500.0, z_upper_m)
        if "t_300hpa" in d:
            t300_c = d["t_300hpa"] - 273.15
            m300 = valid & (hsurf > 4000.0) & np.isfinite(t300_c) #intentionally lower threshold than 5500m to avoid using 300 hPa level in cases where 500 hPa is just slightly above ground
            t_upper_c = np.where(m300, t300_c, t_upper_c)
            z_upper_m = np.where(m300, 9000.0, z_upper_m)
        delta_alt_km = np.where(valid, np.maximum((z_upper_m - hsurf) / 1000.0, 0.1), np.nan)
        thermal_class, _, _ = classify_thermal_strength(t2m_c, td2m_c, t_upper_c, delta_alt_km)
        out = calc_climb_rate_from_thermal_class(thermal_class)
        return np.where(valid, out, np.nan)
    if var == "dew_spread_2m":
        if "t_2m" not in d or "td_2m" not in d:
            raise HTTPException(404, "Dew spread data not available for this timestep")
        return d["t_2m"] - d["td_2m"]
    if var in ("lcl", "reachable"):
        if "t_2m" not in d or "td_2m" not in d or "hsurf" not in d:
            if var == "lcl":
                raise HTTPException(404, "LCL data not available for this timestep")
            raise HTTPException(404, "Reachable data not available for this timestep")
        lcl_amsl = calc_lcl(d["t_2m"], d["td_2m"], d["hsurf"])
        if var == "lcl":
            return lcl_amsl
        thermal_agl = calc_thermal_height(d["mh"], lcl_amsl, d["hsurf"])
        return calc_reachable_distance(thermal_agl)
    raise HTTPException(400, f"Unsupported computed layer: {var}")
