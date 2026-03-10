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


def _meters_to_hm_scalar(value: float) -> int | None:
    if not np.isfinite(value) or value <= 0:
        return None
    hm = int(np.floor(max(float(value), 0.0) / 100.0))
    if hm > 150:
        return None
    return hm


def _meters_to_hm_array(values: np.ndarray) -> np.ndarray:
    base_hm = np.full(values.shape, -1, dtype=np.int16)
    finite = np.isfinite(values) & (values > 0)
    if np.any(finite):
        vals = np.floor(np.maximum(values[finite], 0) / 100)
        vals = np.clip(vals, 0, 32767).astype(np.int16)
        base_hm[finite] = vals
    base_hm[base_hm > 150] = -1
    return base_hm


def classify_point_with_base(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=0.0, mh=None):
    """Canonical scalar symbol/base decision.

    Returns (cloud_type, cb_hm). Inputs use AMSL for htop_dc/hbas_sc;
    AGL thresholds are applied via hsurf.
    """
    ww = 0.0
    if not (np.isfinite(ww) and ww <= 3):
        return "clear", None

    is_conv = np.isfinite(cape_ml) and (cape_ml > CAPE_CONV_THRESHOLD)
    if is_conv:
        cloud_depth = max(0.0, htop_sc - hbas_sc) if np.isfinite(htop_sc) and np.isfinite(hbas_sc) else 0.0
        htop_dc_agl = (htop_dc - hsurf) if (np.isfinite(htop_dc) and np.isfinite(hsurf)) else np.nan
        conv_cloud_ok = bool(convective_cloud_mask(
            np.asarray([hbas_sc]), np.asarray([hsurf]), None if mh is None else np.asarray([mh]),
            min_agl_m=AGL_CONV_MIN_METERS,
        )[0])

        if (not conv_cloud_ok or clcl < 5) and np.isfinite(htop_dc_agl) and htop_dc_agl >= AGL_CONV_MIN_METERS:
            return "blue_thermal", _meters_to_hm_scalar(htop_dc)
        if conv_cloud_ok:
            cb_hm = _meters_to_hm_scalar(hbas_sc)
            if (lpi > LPI_CB_THRESHOLD) or ((cloud_depth > CLOUD_DEPTH_CB_THRESHOLD) and (cape_ml > CAPE_CB_STRONG_THRESHOLD)):
                return "cb", cb_hm
            if cloud_depth > CLOUD_DEPTH_CU_CON_THRESHOLD:
                return "cu_con", cb_hm
            return "cu_hum", cb_hm
        return "clear", None

    valid_ceiling = np.isfinite(ceiling) and (ceiling > 0) and (ceiling < CEILING_VALID_MAX_METERS)
    if not valid_ceiling:
        return "clear", None

    cb_hm = _meters_to_hm_scalar(ceiling)
    if ceiling < CEILING_LOW_MAX_METERS:
        if np.isfinite(clcl) and clcl >= 30:
            return "st", cb_hm
    elif ceiling < CEILING_MID_MAX_METERS:
        if np.isfinite(clcm) and clcm >= 30:
            return "ac", cb_hm
    else:
        if np.isfinite(clch) and clch >= 30:
            return "ci", cb_hm

    # Fallback: if a valid ceiling exists but the band-matched layer is below
    # threshold, still emit the strongest remaining cloud layer instead of
    # collapsing to clear. Prefer lower layers because they are operationally
    # more relevant on the map.
    if np.isfinite(clcl) and clcl >= 30:
        return "st", cb_hm
    if np.isfinite(clcm) and clcm >= 30:
        return "ac", cb_hm
    if np.isfinite(clch) and clch >= 30:
        return "ci", cb_hm
    return "clear", None


def classify_point(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=0.0, mh=None):
    return classify_point_with_base(clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf, mh)[0]


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


def classify_clouds_and_bases(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=None, mh=None):
    """Classify grid-point cloud type and cb_hm together.

    Returns (cloud_type, cb_hm) where cb_hm is the label height in hectometers
    matched to the chosen symbol semantics.
    """
    height, width = ww.shape
    logger.debug(f"Classifying cloud types for grid: {height}x{width}")

    cloud_type = np.full((height, width), "clear", dtype=object)
    cb_hm = np.full((height, width), -1, dtype=np.int16)
    mask = (ww <= 3) & np.isfinite(ww)
    is_convective = np.isfinite(cape_ml) & (cape_ml > CAPE_CONV_THRESHOLD)

    cloud_depth = np.maximum(0, np.where(
        np.isfinite(htop_sc) & np.isfinite(hbas_sc),
        htop_sc - hbas_sc,
        0.0,
    ))

    if hsurf is None:
        hsurf = np.zeros_like(hbas_sc)
    htop_dc_agl = htop_dc - hsurf

    conv_cloud_ok = convective_cloud_mask(
        hbas_sc,
        hsurf,
        mh,
        min_agl_m=AGL_CONV_MIN_METERS,
    )
    blue_ok = np.isfinite(htop_dc_agl) & (htop_dc_agl >= AGL_CONV_MIN_METERS)

    conv_mask = mask & is_convective
    hbas_hm = _meters_to_hm_array(hbas_sc)
    ceil_hm = _meters_to_hm_array(ceiling)
    htop_dc_hm = _meters_to_hm_array(htop_dc)

    blue_mask = conv_mask & ((~conv_cloud_ok) | (clcl < 5)) & blue_ok
    cloud_type[blue_mask] = "blue_thermal"
    cb_hm[blue_mask] = htop_dc_hm[blue_mask]

    cb_mask = conv_mask & conv_cloud_ok & ((lpi > LPI_CB_THRESHOLD) | ((cloud_depth > CLOUD_DEPTH_CB_THRESHOLD) & (cape_ml > CAPE_CB_STRONG_THRESHOLD))) & (cloud_type != "blue_thermal")
    cloud_type[cb_mask] = "cb"
    cb_hm[cb_mask] = hbas_hm[cb_mask]

    cu_con_mask = conv_mask & conv_cloud_ok & (cloud_depth > CLOUD_DEPTH_CU_CON_THRESHOLD) & (cloud_type != "cb") & (cloud_type != "blue_thermal")
    cloud_type[cu_con_mask] = "cu_con"
    cb_hm[cu_con_mask] = hbas_hm[cu_con_mask]

    cu_hum_mask = conv_mask & conv_cloud_ok & (cloud_type == "clear")
    cloud_type[cu_hum_mask] = "cu_hum"
    cb_hm[cu_hum_mask] = hbas_hm[cu_hum_mask]

    strat_mask = mask & ~is_convective
    valid_ceiling = strat_mask & np.isfinite(ceiling) & (ceiling > 0) & (ceiling < CEILING_VALID_MAX_METERS)

    low_band = valid_ceiling & (ceiling < CEILING_LOW_MAX_METERS)
    mid_band = valid_ceiling & (ceiling >= CEILING_LOW_MAX_METERS) & (ceiling < CEILING_MID_MAX_METERS)
    high_band = valid_ceiling & (ceiling >= CEILING_MID_MAX_METERS)

    st_mask = low_band & np.isfinite(clcl) & (clcl >= 30)
    ac_mask = mid_band & np.isfinite(clcm) & (clcm >= 30)
    ci_mask = high_band & np.isfinite(clch) & (clch >= 30)

    cloud_type[st_mask] = "st"
    cb_hm[st_mask] = ceil_hm[st_mask]
    cloud_type[ac_mask] = "ac"
    cb_hm[ac_mask] = ceil_hm[ac_mask]
    cloud_type[ci_mask] = "ci"
    cb_hm[ci_mask] = ceil_hm[ci_mask]

    unresolved = valid_ceiling & (cloud_type == "clear")
    st_fallback = unresolved & np.isfinite(clcl) & (clcl >= 30)
    cloud_type[st_fallback] = "st"
    cb_hm[st_fallback] = ceil_hm[st_fallback]

    unresolved = valid_ceiling & (cloud_type == "clear")
    ac_fallback = unresolved & np.isfinite(clcm) & (clcm >= 30)
    cloud_type[ac_fallback] = "ac"
    cb_hm[ac_fallback] = ceil_hm[ac_fallback]

    unresolved = valid_ceiling & (cloud_type == "clear")
    ci_fallback = unresolved & np.isfinite(clch) & (clch >= 30)
    cloud_type[ci_fallback] = "ci"
    cb_hm[ci_fallback] = ceil_hm[ci_fallback]

    unique, counts = np.unique(cloud_type, return_counts=True)
    summary = dict(zip(unique, counts))
    logger.debug(f"Cloud type distribution: {summary}")

    return cloud_type, cb_hm


def classify_cloud_type(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=None, mh=None):
    return classify_clouds_and_bases(ww, clcl, clcm, clch, cape_ml, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf, mh)[0]


def get_cloud_base(ceiling, hbas_sc):
    """Legacy helper; prefer classify_clouds_and_bases for new code."""
    hbas_valid = np.isfinite(hbas_sc) & (hbas_sc > 0)
    ceil_valid = np.isfinite(ceiling) & (ceiling > 0) & (ceiling < CEILING_VALID_MAX_METERS)
    base = np.where(hbas_valid, hbas_sc, np.where(ceil_valid, ceiling, np.nan))
    return _meters_to_hm_array(base)
