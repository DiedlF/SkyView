from __future__ import annotations

import numpy as np


def convective_cloud_mask(
    hbas_amsl: np.ndarray,
    hsurf_amsl: np.ndarray,
    mh_agl: np.ndarray | None,
    *,
    min_agl_m: float,
    margin_m: float = 500.0,
    hard_cap_agl_m: float = 6500.0,
) -> np.ndarray:
    """Return mask where convective cloud base is physically plausible."""
    hbas = np.asarray(hbas_amsl, dtype=float)
    hsurf = np.asarray(hsurf_amsl, dtype=float)

    hbas_agl = hbas - hsurf
    ok = np.isfinite(hbas_agl) & (hbas > 0.0) & (hbas_agl >= float(min_agl_m)) & (hbas_agl <= float(hard_cap_agl_m))

    if mh_agl is not None:
        mh = np.asarray(mh_agl, dtype=float)
        mh_finite = np.isfinite(mh)
        ok &= (~mh_finite) | (hbas_agl <= (mh + float(margin_m)))

    return ok


def filter_hbas_with_mh(
    hbas_amsl: np.ndarray,
    hsurf_amsl: np.ndarray,
    mh_agl: np.ndarray | None,
    *,
    margin_m: float = 500.0,
    hard_cap_agl_m: float = 6500.0,
    return_quality: bool = False,
):
    """Sanity-filter convective cloud base against boundary-layer depth.

    Args:
        hbas_amsl: Convective cloud base [m MSL].
        hsurf_amsl: Surface elevation [m MSL].
        mh_agl: Mixed-layer height [m AGL]. If missing, only physical checks apply.
        margin_m: Allowed overshoot above mh before hard rejection.
        hard_cap_agl_m: Upper physical cap for base above ground.
        return_quality: If True, also returns quality score in [0, 1].

    Returns:
        filtered_hbas_amsl, reject_mask[, quality]
    """
    hbas = np.asarray(hbas_amsl, dtype=float)
    hsurf = np.asarray(hsurf_amsl, dtype=float)

    hbas_agl = hbas - hsurf
    finite = np.isfinite(hbas_agl)
    phys_ok = finite & (hbas_agl >= 0.0) & (hbas_agl <= hard_cap_agl_m)

    reject = ~phys_ok
    quality = None

    if mh_agl is not None:
        mh = np.asarray(mh_agl, dtype=float)
        mh_finite = np.isfinite(mh)

        overshoot = hbas_agl - (mh + margin_m)
        # only apply mh gate where mh is finite
        reject |= mh_finite & (overshoot > 0.0)

        if return_quality:
            quality = np.full(hbas.shape, np.nan, dtype=np.float32)
            quality[phys_ok] = 1.0
            weak_k = 1200.0
            gated = phys_ok & mh_finite
            quality[gated] *= np.exp(-np.maximum(0.0, overshoot[gated]) / weak_k)
            quality[reject] = 0.0
    elif return_quality:
        quality = np.full(hbas.shape, np.nan, dtype=np.float32)
        quality[phys_ok] = 1.0
        quality[reject] = 0.0

    out = hbas.copy()
    out[reject] = np.nan

    if return_quality:
        return out, reject, quality
    return out, reject
