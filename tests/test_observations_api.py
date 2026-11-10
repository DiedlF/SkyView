from __future__ import annotations

import datetime as dt
import os
import sys
import asyncio

from fastapi import Response

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations import store  # noqa: E402
from observations.config import TARGET_GRID  # noqa: E402
from routers.observations import (  # noqa: E402
    SOURCE_PRODUCTS,
    build_observations_router,
    observations_summary,
)


def _dt(fid: str) -> dt.datetime:
    return dt.datetime.strptime(fid, "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)


def _endpoint(router, name: str):
    for route in router.routes:
        if getattr(route.endpoint, "__name__", "") == name:
            return route.endpoint
    raise AssertionError(f"endpoint not found: {name}")


def test_observations_frames_and_render_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_ROOT", tmp_path)
    src_png = tmp_path / "source.png"
    src_png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00"
        b"\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    store.write_frame_render(
        "radar",
        _dt("202606121820"),
        "radar_dbz",
        src_png,
        cadence_seconds=300,
        prune_after=False,
    )

    router = build_observations_router()
    frames_endpoint = _endpoint(router, "api_observation_frames")
    render_endpoint = _endpoint(router, "api_observation_render")

    frames_response = Response()
    payload = asyncio.run(frames_endpoint(response=frames_response, source="radar", product=None))
    assert frames_response.headers["cache-control"] == "public, max-age=20"
    assert payload["source"] == "radar"
    assert payload["product"] == "radar_dbz"
    assert payload["frames"][0]["frame_id"] == "202606121820"
    assert payload["frames"][0]["render_urls"]["radar_dbz"].endswith(
        "/api/observations/render/radar/radar_dbz/202606121820.png"
    )

    render = asyncio.run(render_endpoint(source="radar", product="radar_dbz", time_str="latest"))
    assert render.media_type == "image/png"
    assert render.headers["x-frame-id"] == "202606121820"
    # "latest" moves to a newer frame as data arrives: must stay short-lived.
    assert render.headers["cache-control"] == "public, max-age=60"

    # A URL naming the frame id can never change content: cache it for hours.
    pinned = asyncio.run(
        render_endpoint(source="radar", product="radar_dbz", time_str="202606121820")
    )
    assert pinned.headers["cache-control"] == "public, max-age=21600, immutable"


def test_webp_render_keeps_its_format_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_ROOT", tmp_path)
    src = tmp_path / "source.webp"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    dest = store.write_frame_render(
        "satellite", _dt("202606121820"), "hrv", src, cadence_seconds=300, prune_after=False
    )
    assert dest.name == "202606121820_hrv.webp"

    router = build_observations_router()
    payload = asyncio.run(
        _endpoint(router, "api_observation_frames")(
            response=Response(), source="satellite", product=None
        )
    )
    url = payload["frames"][0]["render_urls"]["hrv"]
    assert url == "/api/observations/render/satellite/hrv/202606121820.webp"

    render = asyncio.run(
        _endpoint(router, "api_observation_render")(
            source="satellite", product="hrv", time_str="202606121820"
        )
    )
    assert render.media_type == "image/webp"

    # Both URL suffixes are routed, so links to older PNG frames keep working.
    paths = {route.path for route in router.routes}
    assert "/api/observations/render/{source}/{product}/{time_str}.png" in paths
    assert "/api/observations/render/{source}/{product}/{time_str}.webp" in paths

    # Retention must also delete WebP renders, or the ring buffer leaks disk.
    store.prune("satellite", keep_seconds=1, keep_frames=None)
    assert not dest.exists()


def test_served_bbox_matches_render_grid():
    # Frames are reprojected/resampled onto TARGET_GRID at ingest, and the
    # frontend places the image overlay using exactly the served bbox. If the
    # two ever diverge the overlay is misregistered (the satellite bug). Pin
    # both sources to the render grid extent: [lat_min, lon_min, lat_max, lon_max].
    expected = [
        TARGET_GRID.lat_min,
        TARGET_GRID.lon_min,
        TARGET_GRID.lat_max,
        TARGET_GRID.lon_max,
    ]
    for source in ("radar", "satellite", "mtg", "li"):
        assert SOURCE_PRODUCTS[source]["bbox"] == expected


def test_observations_summary_marks_missing_sources_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_ROOT", tmp_path)
    summary = observations_summary()
    assert summary["sources"]["radar"]["frame_count"] == 0
    assert summary["sources"]["radar"]["stale"] is True
