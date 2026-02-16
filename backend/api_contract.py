"""Shared API contract helpers for Skyview/Explorer convergence."""

from __future__ import annotations

from typing import Optional

LAYER_ALIAS_MAP = {
    "rain": "prr_gsp",
    "snow": "prs_gsp",
    "hail": "prg_gsp",
    "sigwx": "ww",
    "clouds_low": "clcl",
    "clouds_mid": "clcm",
    "clouds_high": "clch",
    "clouds_total": "clct",
    "clouds_total_mod": "clct_mod",
    "dry_conv_top": "htop_dc",
    "ceiling": "ceiling",
    "cloud_base": "hbas_sc",
    "thermals": "cape_ml",
    "total_precip": "tot_prec",
}


def resolve_layer_alias(layer: Optional[str]) -> Optional[str]:
    if layer is None:
        return None
    return LAYER_ALIAS_MAP.get(layer)
