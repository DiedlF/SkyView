"""Cloud type classification for Skyview."""
import numpy as np
from logging_config import setup_logging

logger = setup_logging(__name__, level="WARNING")


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


def classify_cloud_type(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling):
    """Classify cloud type per decision tree, only where ww <= 3.
    
    All input arrays should be pre-cropped to the region of interest
    (callers should bbox-slice before calling this function).
    
    Returns 2D array of strings."""
    height, width = ww.shape
    logger.debug(f"Classifying cloud types for grid: {height}x{width}")
    
    cloud_type = np.full((height, width), "clear", dtype=object)
    mask = (ww <= 3) & np.isfinite(ww)
    is_convective = (cape_ml > 50)

    # Convective
    conv_mask = mask & is_convective
    cloud_depth = np.maximum(0, htop_sc - hbas_sc)
    cloud_type[conv_mask & ((hbas_sc <= 0) | (clcl < 5))] = "blue_thermal"
    cloud_type[conv_mask & ((lpi > 0) | (cloud_depth > 4000) | (cape_ml > 1000))] = "cb"
    cu_con_mask = conv_mask & (cloud_depth > 500) & (cloud_type != "cb") & (cloud_type != "blue_thermal")
    cloud_type[cu_con_mask] = "cu_con"
    cu_hum_mask = conv_mask & (cloud_type == "clear")
    cloud_type[cu_hum_mask] = "cu_hum"

    # Stratiform
    strat_mask = mask & ~is_convective
    cloud_type[strat_mask & (clcl > 30) & (clcm < 20)] = "st"
    cloud_type[strat_mask & (clcm > 30) & (clcl < 20)] = "ac"
    cloud_type[strat_mask & (clcl < 10) & (clcm < 10) & (clch > 30)] = "ci"
    
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
