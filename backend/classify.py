"""Cloud type classification for Skyview."""
import numpy as np
from logging_config import setup_logging

logger = setup_logging(__name__, level="WARNING")


def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=0.0):
    """Canonical scalar cloud-type classification.

    Inputs use AMSL for htop_dc/hbas_sc; AGL thresholds are applied via hsurf.
    """
    ww = 0.0
    if not (np.isfinite(ww) and ww <= 3):
        return "clear"

    is_conv = np.isfinite(cape_ml) and (cape_ml > 50)
    if is_conv:
        cloud_depth = max(0.0, htop_sc - hbas_sc) if np.isfinite(htop_sc) and np.isfinite(hbas_sc) else 0.0
        hbas_agl = (hbas_sc - hsurf) if (np.isfinite(hbas_sc) and np.isfinite(hsurf)) else np.nan
        htop_dc_agl = (htop_dc - hsurf) if (np.isfinite(htop_dc) and np.isfinite(hsurf)) else np.nan

        if (hbas_sc <= 0 or clcl < 5) and np.isfinite(htop_dc_agl) and htop_dc_agl >= 300:
            return "blue_thermal"
        if np.isfinite(hbas_agl) and hbas_agl >= 300:
            if (lpi > 7) or ((cloud_depth > 4000) and (cape_ml > 1000)):
                return "cb"
            if cloud_depth > 2000:
                return "cu_con"
            return "cu_hum"
        return "clear"

    if not np.isfinite(ceiling) or ceiling <= 0 or ceiling >= 20000:
        return "clear"
    if ceiling < 2000:
        return "st" if np.isfinite(clcl) and clcl >= 30 else "clear"
    if ceiling < 7000:
        return "ac" if np.isfinite(clcm) and clcm >= 30 else "clear"
    return "ci" if np.isfinite(clch) and clch >= 30 else "clear"


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


def classify_cloud_type(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=None):
    """Classify cloud type for ww<=3 grid points.

    Non-convective class uses ceiling band + min layer cloud cover:
    - st: ceiling < 2000 and clcl >= 30
    - ac: 2000 <= ceiling < 7000 and clcm >= 30
    - ci: ceiling >= 7000 and clch >= 30
    otherwise clear.
    """
    height, width = ww.shape
    logger.debug(f"Classifying cloud types for grid: {height}x{width}")

    cloud_type = np.full((height, width), "clear", dtype=object)
    mask = (ww <= 3) & np.isfinite(ww)
    is_convective = (cape_ml > 50)

    # Convective
    conv_mask = mask & is_convective
    cloud_depth = np.maximum(0, htop_sc - hbas_sc)

    # hbas_sc / htop_dc are AMSL; convert to AGL using hsurf when available.
    if hsurf is None:
        hsurf = np.zeros_like(hbas_sc)
    hbas_agl = hbas_sc - hsurf
    htop_dc_agl = htop_dc - hsurf

    # Suppress convective symbols when AGL signal is too low:
    # - convective clouds (cu_hum/cu_con/cb): cloud base must be >= 300 m AGL
    # - blue_thermal: dry convection top must be >= 300 m AGL
    conv_cloud_ok = np.isfinite(hbas_agl) & (hbas_agl >= 300)
    blue_ok = np.isfinite(htop_dc_agl) & (htop_dc_agl >= 300)

    blue_mask = conv_mask & ((hbas_sc <= 0) | (clcl < 5)) & blue_ok
    cloud_type[blue_mask] = "blue_thermal"
    cloud_type[conv_mask & conv_cloud_ok & ((lpi > 7) | ((cloud_depth > 4000) & (cape_ml > 1000))) & (cloud_type != "blue_thermal")] = "cb"
    cu_con_mask = conv_mask & conv_cloud_ok & (cloud_depth > 2000) & (cloud_type != "cb") & (cloud_type != "blue_thermal")
    cloud_type[cu_con_mask] = "cu_con"
    cu_hum_mask = conv_mask & conv_cloud_ok & (cloud_type == "clear")
    cloud_type[cu_hum_mask] = "cu_hum"

    # Non-convective (ceiling band + min layer cloud cover)
    strat_mask = mask & ~is_convective
    valid_ceiling = strat_mask & np.isfinite(ceiling) & (ceiling > 0) & (ceiling < 20000)

    low_band = valid_ceiling & (ceiling < 2000)
    mid_band = valid_ceiling & (ceiling >= 2000) & (ceiling < 7000)
    high_band = valid_ceiling & (ceiling >= 7000)

    cloud_type[low_band & np.isfinite(clcl) & (clcl >= 30)] = "st"
    cloud_type[mid_band & np.isfinite(clcm) & (clcm >= 30)] = "ac"
    cloud_type[high_band & np.isfinite(clch) & (clch >= 30)] = "ci"
    
    # Log classification summary
    unique, counts = np.unique(cloud_type, return_counts=True)
    summary = dict(zip(unique, counts))
    logger.debug(f"Cloud type distribution: {summary}")

    return cloud_type


def get_cloud_base(ceiling, hbas_sc):
    """Cloud base in hectometers. -1 if invalid."""
    mask_valid = np.isfinite(ceiling) & (ceiling > 0) & (ceiling < 20000)
    base = np.where(mask_valid, ceiling, hbas_sc)
    base_hm = np.floor(np.maximum(base, 0) / 100).astype(np.int16)
    base_hm[~np.isfinite(base)] = -1
    base_hm[base_hm > 150] = -1
    return base_hm
