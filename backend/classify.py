"""Cloud type classification for Skyview."""
import numpy as np
from logging_config import setup_logging
from convective_filters import convective_cloud_mask
from constants import (
    CAPE_CONV_THRESHOLD,
    CAPE_CB_STRONG_THRESHOLD,
    LPI_CB_THRESHOLD,
    CLOUD_DEPTH_CU_CON_THRESHOLD,
    CLOUD_DEPTH_CB_THRESHOLD,
    AGL_CONV_MIN_METERS,
    CEILING_LOW_MAX_METERS,
    CEILING_MID_MAX_METERS,
    CEILING_VALID_MAX_METERS,
)

logger = setup_logging(__name__, level="WARNING")


def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=0.0, mh=None):
    """Canonical scalar cloud-type classification.

    Inputs use AMSL for htop_dc/hbas_sc; AGL thresholds are applied via hsurf.
    """
    ww = 0.0
    if not (np.isfinite(ww) and ww <= 3):
        return "clear"

    is_conv = np.isfinite(cape_ml) and (cape_ml > CAPE_CONV_THRESHOLD)
    if is_conv:
        cloud_depth = max(0.0, htop_sc - hbas_sc) if np.isfinite(htop_sc) and np.isfinite(hbas_sc) else 0.0
        htop_dc_agl = (htop_dc - hsurf) if (np.isfinite(htop_dc) and np.isfinite(hsurf)) else np.nan
        conv_cloud_ok = bool(convective_cloud_mask(
            np.asarray([hbas_sc]), np.asarray([hsurf]), None if mh is None else np.asarray([mh]),
            min_agl_m=AGL_CONV_MIN_METERS,
        )[0])

        if (not conv_cloud_ok or clcl < 5) and np.isfinite(htop_dc_agl) and htop_dc_agl >= AGL_CONV_MIN_METERS:
            return "blue_thermal"
        if conv_cloud_ok:
            if (lpi > LPI_CB_THRESHOLD) or ((cloud_depth > CLOUD_DEPTH_CB_THRESHOLD) and (cape_ml > CAPE_CB_STRONG_THRESHOLD)):
                return "cb"
            if cloud_depth > CLOUD_DEPTH_CU_CON_THRESHOLD:
                return "cu_con"
            return "cu_hum"
        return "clear"

    if not np.isfinite(ceiling) or ceiling <= 0 or ceiling >= CEILING_VALID_MAX_METERS:
        return "clear"
    if ceiling < CEILING_LOW_MAX_METERS:
        if np.isfinite(clcl) and clcl >= 30:
            return "st"
    elif ceiling < CEILING_MID_MAX_METERS:
        if np.isfinite(clcm) and clcm >= 30:
            return "ac"
    else:
        if np.isfinite(clch) and clch >= 30:
            return "ci"

    # Fallback: if a valid ceiling exists but the band-matched layer is below
    # threshold, still emit the strongest remaining cloud layer instead of
    # collapsing to clear. Prefer lower layers because they are operationally
    # more relevant on the map.
    if np.isfinite(clcl) and clcl >= 30:
        return "st"
    if np.isfinite(clcm) and clcm >= 30:
        return "ac"
    if np.isfinite(clch) and clch >= 30:
        return "ci"
    return "clear"


def crop_to_bbox(arrays_dict, lat, lon, bbox):
    """Crop multiple 2D arrays to bbox (lat_min, lat_max, lon_min, lon_max).
    
    Args:
        arrays_dict: dict of {name: 2D array}
        lat: 1D latitude array
        lon: 1D longitude array
        bbox: (lat_min, lat_max, lon_min, lon_max) or None
    
    Returns:
        (cropped_dict, cropped_lat, cropped_lon)
    """
    if bbox is None:
        return arrays_dict, lat, lon
    lat_min, lat_max, lon_min, lon_max = bbox
    eps = 0.001
    lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
    lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]
    if len(li) == 0 or len(lo) == 0:
        return {k: v[:0, :0] for k, v in arrays_dict.items()}, lat[:0], lon[:0]
    cropped = {k: v[np.ix_(li, lo)] for k, v in arrays_dict.items()}
    return cropped, lat[li], lon[lo]


def classify_cloud_type(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=None, mh=None):
    """Classify cloud type for ww<=3 grid points.

    Non-convective class uses ceiling band + min layer cloud cover:
    - st: ceiling < CEILING_LOW_MAX_METERS and clcl >= 30
    - ac: 2000 <= ceiling < CEILING_MID_MAX_METERS and clcm >= 30
    - ci: ceiling >= 7000 and clch >= 30
    otherwise clear.
    """
    height, width = ww.shape
    logger.debug(f"Classifying cloud types for grid: {height}x{width}")

    cloud_type = np.full((height, width), "clear", dtype=object)
    mask = (ww <= 3) & np.isfinite(ww)
    is_convective = (cape_ml > CAPE_CONV_THRESHOLD)

    # Convective
    conv_mask = mask & is_convective
    cloud_depth = np.maximum(0, htop_sc - hbas_sc)

    # hbas_sc / htop_dc are AMSL; convert to AGL using hsurf when available.
    if hsurf is None:
        hsurf = np.zeros_like(hbas_sc)
    htop_dc_agl = htop_dc - hsurf

    # Unified convective plausibility: cloud-forming convection requires a
    # physically plausible convective cloud base; blue thermals use htop_dc.
    conv_cloud_ok = convective_cloud_mask(
        hbas_sc,
        hsurf,
        mh,
        min_agl_m=AGL_CONV_MIN_METERS,
    )
    blue_ok = np.isfinite(htop_dc_agl) & (htop_dc_agl >= AGL_CONV_MIN_METERS)

    blue_mask = conv_mask & ((~conv_cloud_ok) | (clcl < 5)) & blue_ok
    cloud_type[blue_mask] = "blue_thermal"
    cloud_type[conv_mask & conv_cloud_ok & ((lpi > LPI_CB_THRESHOLD) | ((cloud_depth > CLOUD_DEPTH_CB_THRESHOLD) & (cape_ml > CAPE_CB_STRONG_THRESHOLD))) & (cloud_type != "blue_thermal")] = "cb"
    cu_con_mask = conv_mask & conv_cloud_ok & (cloud_depth > CLOUD_DEPTH_CU_CON_THRESHOLD) & (cloud_type != "cb") & (cloud_type != "blue_thermal")
    cloud_type[cu_con_mask] = "cu_con"
    cu_hum_mask = conv_mask & conv_cloud_ok & (cloud_type == "clear")
    cloud_type[cu_hum_mask] = "cu_hum"

    # Non-convective (ceiling band + min layer cloud cover)
    strat_mask = mask & ~is_convective
    valid_ceiling = strat_mask & np.isfinite(ceiling) & (ceiling > 0) & (ceiling < CEILING_VALID_MAX_METERS)

    low_band = valid_ceiling & (ceiling < CEILING_LOW_MAX_METERS)
    mid_band = valid_ceiling & (ceiling >= CEILING_LOW_MAX_METERS) & (ceiling < CEILING_MID_MAX_METERS)
    high_band = valid_ceiling & (ceiling >= 7000)

    cloud_type[low_band & np.isfinite(clcl) & (clcl >= 30)] = "st"
    cloud_type[mid_band & np.isfinite(clcm) & (clcm >= 30)] = "ac"
    cloud_type[high_band & np.isfinite(clch) & (clch >= 30)] = "ci"

    # Fallback for cases where ceiling is valid but the band-matched layer does
    # not clear the threshold while another cloud layer does.
    unresolved = valid_ceiling & (cloud_type == "clear")
    cloud_type[unresolved & np.isfinite(clcl) & (clcl >= 30)] = "st"
    unresolved = valid_ceiling & (cloud_type == "clear")
    cloud_type[unresolved & np.isfinite(clcm) & (clcm >= 30)] = "ac"
    unresolved = valid_ceiling & (cloud_type == "clear")
    cloud_type[unresolved & np.isfinite(clch) & (clch >= 30)] = "ci"

    # Log classification summary
    unique, counts = np.unique(cloud_type, return_counts=True)
    summary = dict(zip(unique, counts))
    logger.debug(f"Cloud type distribution: {summary}")

    return cloud_type


def get_cloud_base(ceiling, hbas_sc):
    """Cloud base in hectometers. -1 if invalid.

    Priority:
    - If hbas_sc > 0 (convective cloud formed): use hbas_sc directly.
    - Else if ceiling is valid (finite, >0, <CEILING_VALID_MAX_METERS): use ceiling
      (stratiform base).
    - Else: return -1 (no valid base).
    """
    hbas_valid = np.isfinite(hbas_sc) & (hbas_sc > 0)
    ceil_valid  = np.isfinite(ceiling) & (ceiling > 0) & (ceiling < CEILING_VALID_MAX_METERS)
    base = np.where(hbas_valid, hbas_sc, np.where(ceil_valid, ceiling, np.nan))

    base_hm = np.full(base.shape, -1, dtype=np.int16)
    finite = np.isfinite(base)
    if np.any(finite):
        vals = np.floor(np.maximum(base[finite], 0) / 100)
        vals = np.clip(vals, 0, 32767).astype(np.int16)
        base_hm[finite] = vals

    base_hm[base_hm > 150] = -1
    return base_hm
