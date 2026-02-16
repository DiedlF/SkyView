"""Shared overlay data-key and computed-field helpers."""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from soaring import calc_wstar, calc_climb_rate, calc_lcl, calc_thermal_height, calc_reachable_distance


def build_overlay_keys(cfg: dict) -> list[str]:
    """Determine data keys required for a given overlay config."""
    overlay_keys = ["lat", "lon"]
    if cfg.get("computed"):
        v = cfg["var"]
        if v == "total_precip":
            overlay_keys += ["prr_gsp", "prs_gsp", "prg_gsp"]
        elif v == "conv_thickness":
            overlay_keys += ["htop_sc", "hbas_sc"]
        elif v in ("wstar", "climb_rate"):
            overlay_keys += ["ashfl_s", "mh", "t_2m"]
        elif v in ("lcl", "reachable"):
            overlay_keys += ["ashfl_s", "mh", "t_2m", "td_2m", "hsurf"]
    else:
        overlay_keys.append(cfg["var"])
    return overlay_keys


def normalize_clouds_total_mod(arr: np.ndarray) -> np.ndarray:
    """Normalize clct_mod to percent if it appears fractional (0..1)."""
    vmax_local = float(np.nanmax(arr)) if np.size(arr) else float("nan")
    if np.isfinite(vmax_local) and vmax_local <= 1.5:
        return arr * 100.0
    return arr


def compute_computed_field_cropped(var: str, d: dict, li: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Compute cropped computed overlay field for /api/overlay."""
    if var == "total_precip":
        return d["prr_gsp"][np.ix_(li, lo)] + d["prs_gsp"][np.ix_(li, lo)] + d["prg_gsp"][np.ix_(li, lo)]

    if var == "conv_thickness":
        htop_sc = d["htop_sc"][np.ix_(li, lo)] if "htop_sc" in d else np.zeros((len(li), len(lo)))
        hbas_sc = d["hbas_sc"][np.ix_(li, lo)] if "hbas_sc" in d else np.zeros((len(li), len(lo)))
        return np.maximum(0, htop_sc - hbas_sc)

    if var in ("wstar", "climb_rate", "lcl", "reachable"):
        if "ashfl_s" not in d or "mh" not in d or "t_2m" not in d:
            raise HTTPException(404, "Soaring data not available for this timestep (re-ingestion needed)")
        ashfl_s = d["ashfl_s"][np.ix_(li, lo)]
        mh = d["mh"][np.ix_(li, lo)]
        t_2m = d["t_2m"][np.ix_(li, lo)]

        if var == "wstar":
            return calc_wstar(ashfl_s, None, mh, t_2m, dt_seconds=3600)
        if var == "climb_rate":
            wstar = calc_wstar(ashfl_s, None, mh, t_2m, dt_seconds=3600)
            return calc_climb_rate(wstar)

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


def compute_computed_field_full(var: str, d: dict) -> np.ndarray:
    """Compute full-grid computed overlay field for /api/overlay_tile."""
    if var == "total_precip":
        return d["prr_gsp"] + d["prs_gsp"] + d["prg_gsp"]
    if var == "conv_thickness":
        return np.maximum(0, d["htop_sc"] - d["hbas_sc"])
    if var in ("wstar", "climb_rate"):
        wstar = calc_wstar(d["ashfl_s"], None, d["mh"], d["t_2m"], dt_seconds=3600)
        return wstar if var == "wstar" else calc_climb_rate(wstar)
    if var in ("lcl", "reachable"):
        lcl_amsl = calc_lcl(d["t_2m"], d["td_2m"], d["hsurf"])
        if var == "lcl":
            return lcl_amsl
        thermal_agl = calc_thermal_height(d["mh"], lcl_amsl, d["hsurf"])
        return calc_reachable_distance(thermal_agl)
    raise HTTPException(400, f"Unsupported computed layer: {var}")
