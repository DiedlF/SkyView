"""Shared HTTP response header builders for API convergence."""

from __future__ import annotations


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
