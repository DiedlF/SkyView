"""Shared HTTP response header builders for API convergence."""

from __future__ import annotations


# Overlay tiles whose URL carries the current timeline token are safe to cache
# well beyond the default: the token changes as soon as any model step lands.
# One hour (not "immutable") bounds the damage if a step file is ever rewritten
# in place; model runs rotate every ~3 h anyway, so longer buys little.
TILE_CACHE_VERSIONED = "public, max-age=3600"
TILE_CACHE_DEFAULT = "public, max-age=300"


def tile_cache_control_for(
    *,
    version: str | None,
    current_version: str,
    time_str: str,
    model: str | None,
    substep_minutes: int,
    eu_fill_missing: bool,
) -> str:
    """Pick the Cache-Control for an overlay tile response.

    The long TTL applies only when the response is fully determined by the URL:
    the client sent the current version token (``v=``), pinned the model
    (without it the D2/EU choice depends on wall-clock lead time), asked for a
    concrete hourly time (15-min substep data can arrive after its hourly
    step), and the tile did not need an ICON-EU fill that was unavailable when
    it was rendered. Everything else keeps the short default.
    """
    if (
        version
        and model
        and time_str != "latest"
        and substep_minutes == 0
        and not eu_fill_missing
        and version == current_version
    ):
        return TILE_CACHE_VERSIONED
    return TILE_CACHE_DEFAULT


def build_overlay_headers(*, run: str, valid_time: str, model: str, bbox: str, extra: dict | None = None) -> dict:
    headers = {
        "Cache-Control": "public, max-age=300",
        "X-Run": run,
        "X-ValidTime": valid_time,
        "X-Model": model,
        "X-Bbox": bbox,
        "Access-Control-Expose-Headers": "X-Bbox, X-Run, X-ValidTime, X-Model",
    }
    if extra:
        headers.update(extra)
        expose = [x.strip() for x in headers["Access-Control-Expose-Headers"].split(",") if x.strip()]
        for k in extra.keys():
            if k not in expose and k not in ("Cache-Control",):
                expose.append(k)
        headers["Access-Control-Expose-Headers"] = ", ".join(expose)
    return headers


def build_tile_headers(*, run: str, valid_time: str, model: str, cache: str, extra: dict | None = None) -> dict:
    headers = {
        "Cache-Control": "public, max-age=300",
        "X-Run": run,
        "X-ValidTime": valid_time,
        "X-Model": model,
        "X-Cache": cache,
        "Access-Control-Expose-Headers": "X-Run, X-ValidTime, X-Model, X-Cache",
    }
    if extra:
        headers.update(extra)
        expose = [x.strip() for x in headers["Access-Control-Expose-Headers"].split(",") if x.strip()]
        for k in extra.keys():
            if k not in expose and k not in ("Cache-Control",):
                expose.append(k)
        headers["Access-Control-Expose-Headers"] = ", ".join(expose)
    return headers
