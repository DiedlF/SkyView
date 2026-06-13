from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

try:
    from observations import store
    from observations.config import TARGET_GRID
except ImportError:  # pragma: no cover - package launch style
    from backend.observations import store
    from backend.observations.config import TARGET_GRID


# Both products are reprojected/resampled onto ``TARGET_GRID`` at ingest, so the
# served bbox is derived from that same grid for both. Keeping a single source of
# truth prevents the render-extent vs served-bbox drift that misregistered the
# satellite overlay (the image is placed by the frontend using exactly this bbox).
_GRID_BBOX = [TARGET_GRID.lat_min, TARGET_GRID.lon_min, TARGET_GRID.lat_max, TARGET_GRID.lon_max]

SOURCE_PRODUCTS = {
    "radar": {
        "default": "radar_dbz",
        "products": {"radar_dbz"},
        "label": "OPERA radar dBZ",
        "bbox": list(_GRID_BBOX),
    },
    "satellite": {
        "default": "hrv",
        "products": {"hrv"},
        "label": "MSG RSS HRV",
        "bbox": list(_GRID_BBOX),
    },
}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_iso(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _frame_age_s(frame: dict, now: Optional[dt.datetime] = None) -> Optional[int]:
    when = _parse_iso(str(frame.get("valid_time") or ""))
    if when is None:
        return None
    return max(0, int(((now or _now()) - when).total_seconds()))


def _validate_source(source: str) -> str:
    source = str(source or "").strip().lower()
    if source not in SOURCE_PRODUCTS:
        raise HTTPException(400, f"unknown observation source: {source}")
    return source


def _validate_product(source: str, product: Optional[str]) -> str:
    product = str(product or SOURCE_PRODUCTS[source]["default"]).strip()
    if product not in SOURCE_PRODUCTS[source]["products"]:
        raise HTTPException(400, f"unknown {source} observation product: {product}")
    return product


def resolve_frame(source: str, time: str = "latest", *, max_delta_seconds: Optional[int] = None) -> dict:
    source = _validate_source(source)
    frames = store.list_frames(source)
    if not frames:
        raise HTTPException(404, f"no {source} observation frames available")
    frames = sorted(frames, key=lambda f: str(f.get("frame_id") or ""))
    requested = str(time or "latest").strip()
    if requested == "latest":
        return frames[-1]
    if requested.isdigit() and len(requested) == 12:
        for frame in frames:
            if frame.get("frame_id") == requested:
                return frame
        raise HTTPException(404, f"{source} frame not found: {requested}")

    target = _parse_iso(requested)
    if target is None:
        raise HTTPException(422, "time must be latest, frame_id YYYYMMDDHHMM, or ISO timestamp")

    candidates = []
    for frame in frames:
        when = _parse_iso(str(frame.get("valid_time") or ""))
        if when is not None:
            candidates.append((abs((when - target).total_seconds()), frame))
    if not candidates:
        raise HTTPException(404, f"no {source} observation frames with valid timestamps")
    delta, frame = min(candidates, key=lambda x: x[0])
    if max_delta_seconds is not None and delta > max_delta_seconds:
        raise HTTPException(404, f"no {source} frame close to requested time")
    return frame


def frame_payload(source: str, frame: dict, *, now: Optional[dt.datetime] = None) -> dict:
    products = frame.get("products") or {}
    attrs = frame.get("attrs") or {}
    out = {
        "frame_id": frame.get("frame_id"),
        "valid_time": frame.get("valid_time"),
        "age_s": _frame_age_s(frame, now),
        "products": products,
        "attrs": attrs,
    }
    for product in products.keys():
        out.setdefault("render_urls", {})[product] = (
            f"/api/observations/render/{source}/{product}/{frame.get('frame_id')}.png"
        )
    return out


def observations_summary() -> dict:
    now = _now()
    sources = {}
    for source, meta in SOURCE_PRODUCTS.items():
        manifest = store.read_manifest(source)
        frames = sorted(manifest.get("frames") or [], key=lambda f: str(f.get("frame_id") or ""))
        latest = frames[-1] if frames else None
        age_s = _frame_age_s(latest, now) if latest else None
        cadence = manifest.get("cadence_seconds") or 300
        stale = age_s is None or age_s > int(cadence) * 2
        sources[source] = {
            "label": meta["label"],
            "cadence_seconds": cadence,
            "frame_count": len(frames),
            "latest": frame_payload(source, latest, now=now) if latest else None,
            "fresh": not stale,
            "stale": stale,
        }
    return {"sources": sources, "updated": now.replace(microsecond=0).isoformat().replace("+00:00", "Z")}


def build_observations_router():
    router = APIRouter()

    @router.get("/api/observations/frames")
    async def api_observation_frames(source: str = Query(...), product: Optional[str] = Query(None)):
        source_v = _validate_source(source)
        product_v = _validate_product(source_v, product)
        manifest = store.read_manifest(source_v)
        now = _now()
        frames = [
            frame_payload(source_v, frame, now=now)
            for frame in sorted(manifest.get("frames") or [], key=lambda f: str(f.get("frame_id") or ""))
            if product_v in (frame.get("products") or {})
        ]
        latest = frames[-1]["valid_time"] if frames else None
        return {
            "source": source_v,
            "product": product_v,
            "label": SOURCE_PRODUCTS[source_v]["label"],
            "cadence_seconds": manifest.get("cadence_seconds") or 300,
            "bbox": SOURCE_PRODUCTS[source_v]["bbox"],
            "latest": latest,
            "frames": frames,
        }

    @router.get("/api/observations/status")
    async def api_observation_status():
        return observations_summary()

    @router.get("/api/observations/render/{source}/{product}/{time_str}.png")
    async def api_observation_render(source: str, product: str, time_str: str):
        source_v = _validate_source(source)
        product_v = _validate_product(source_v, product)
        frame = resolve_frame(source_v, time_str, max_delta_seconds=150)
        fid = str(frame.get("frame_id") or "")
        try:
            path = store.render_file_for_frame(source_v, fid, product_v)
        except FileNotFoundError:
            raise HTTPException(404, "observation render not found")
        if not Path(path).is_file():
            raise HTTPException(404, "observation render file missing")

        valid_time = str(frame.get("valid_time") or "")
        headers = {
            "Cache-Control": "public, max-age=240",
            "X-Observation-Source": source_v,
            "X-Observation-Product": product_v,
            "X-Frame-Id": fid,
            "X-ValidTime": valid_time,
            "X-Bbox": ",".join(str(x) for x in SOURCE_PRODUCTS[source_v]["bbox"]),
            "Access-Control-Expose-Headers": (
                "X-Observation-Source, X-Observation-Product, X-Frame-Id, X-ValidTime, X-Bbox"
            ),
        }
        return FileResponse(path, media_type="image/png", headers=headers)

    return router
