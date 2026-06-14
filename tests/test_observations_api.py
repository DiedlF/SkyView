from __future__ import annotations

import datetime as dt
import os
import sys
import asyncio

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

    payload = asyncio.run(frames_endpoint(source="radar", product=None))
    assert payload["source"] == "radar"
    assert payload["product"] == "radar_dbz"
    assert payload["frames"][0]["frame_id"] == "202606121820"
    assert payload["frames"][0]["render_urls"]["radar_dbz"].endswith(
        "/api/observations/render/radar/radar_dbz/202606121820.png"
    )

    render = asyncio.run(render_endpoint(source="radar", product="radar_dbz", time_str="latest"))
    assert render.media_type == "image/png"
    assert render.headers["x-frame-id"] == "202606121820"


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
    for source in ("radar", "satellite", "mtg"):
        assert SOURCE_PRODUCTS[source]["bbox"] == expected


def test_observations_summary_marks_missing_sources_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_ROOT", tmp_path)
    summary = observations_summary()
    assert summary["sources"]["radar"]["frame_count"] == 0
    assert summary["sources"]["radar"]["stale"] is True
