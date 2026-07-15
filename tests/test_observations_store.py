"""Unit tests for the observation frame store: manifest + ring-buffer prune.

Pure stdlib (json/os/datetime/shutil) — no numpy/zarr needed. The Zarr write/read
paths (write_frame_zarr/load_frame) are exercised on a host with the ingest deps.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations import store  # noqa: E402


def _dt(fid: str) -> dt.datetime:
    return dt.datetime.strptime(fid, "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    """Point the store at a temp data root for every test."""
    monkeypatch.setattr(store, "DATA_ROOT", tmp_path)


def _make_frame_dir(source: str, fid: str) -> None:
    d = store.frame_zarr_path(source, fid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "marker").write_text("x")


def test_frame_id_and_valid_time_are_utc():
    naive = dt.datetime(2026, 6, 3, 12, 30)  # treated as UTC
    assert store.frame_id(naive) == "202606031230"
    assert store.valid_time_iso(naive) == "2026-06-03T12:30:00Z"
    assert store.parse_frame_id("202606031230") == _dt("202606031230")


def test_record_frame_sorts_and_dedupes():
    store.record_frame("radar", _dt("202606031230"), cadence_seconds=300)
    store.record_frame("radar", _dt("202606031225"))
    store.record_frame("radar", _dt("202606031230"))  # duplicate
    frames = store.list_frames("radar")
    assert [f["frame_id"] for f in frames] == ["202606031225", "202606031230"]
    assert store.latest_frame("radar")["frame_id"] == "202606031230"
    assert store.read_manifest("radar")["cadence_seconds"] == 300
    assert store.has_frame("radar", "202606031225")
    assert not store.has_frame("radar", "202606031200")


def test_write_render_frame_records_product_and_prunes_png(tmp_path):
    old_png = tmp_path / "old.png"
    old_png.write_bytes(b"old")
    new_png = tmp_path / "new.png"
    new_png.write_bytes(b"new")

    store.write_frame_render(
        "radar",
        _dt("202606031210"),
        "radar_dbz",
        old_png,
        cadence_seconds=300,
        prune_after=False,
    )
    store.write_frame_render(
        "radar",
        _dt("202606031230"),
        "radar_dbz",
        new_png,
        attrs={"cache": "derived_render"},
        prune_after=False,
    )

    frames = store.list_frames("radar")
    assert frames[-1]["products"]["radar_dbz"] == "202606031230_radar_dbz.png"
    assert frames[-1]["attrs"]["cache"] == "derived_render"
    assert store.frame_render_path("radar", "202606031210", "radar_dbz").exists()

    removed = store.prune("radar", keep_seconds=600, keep_frames=None, now=_dt("202606031230"))

    assert removed == ["202606031210"]
    assert not store.frame_render_path("radar", "202606031210", "radar_dbz").exists()
    assert store.frame_render_path("radar", "202606031230", "radar_dbz").exists()


def test_write_frame_points_records_json_and_resolves():
    flashes = [{"lon": 8.5, "lat": 50.1, "r": 1.2, "n": 3}, {"lon": 8.6, "lat": 50.2}]
    store.write_frame_points(
        "li",
        _dt("202606031230"),
        "flashes",
        flashes,
        attrs={"product": "lightning_flashes", "count": len(flashes)},
        cadence_seconds=600,
        prune_after=False,
    )

    frame = store.latest_frame("li")
    assert frame["products"]["flashes"] == "202606031230_flashes.json"
    assert frame["attrs"]["product"] == "lightning_flashes"

    path = store.render_file_for_frame("li", "202606031230", "flashes")
    assert path.exists() and path.suffix == ".json"
    import json

    payload = json.loads(path.read_text())
    assert payload["count"] == 2
    assert payload["flashes"][0]["lat"] == 50.1


def test_prune_removes_points_json():
    store.write_frame_points("li", _dt("202606031210"), "flashes", [{"lon": 1.0, "lat": 2.0}], prune_after=False)
    store.write_frame_points("li", _dt("202606031230"), "flashes", [{"lon": 1.0, "lat": 2.0}], prune_after=False)
    old = store.source_dir("li") / "202606031210_flashes.json"
    assert old.exists()

    removed = store.prune("li", keep_seconds=600, keep_frames=None, now=_dt("202606031230"))

    assert removed == ["202606031210"]
    assert not old.exists()
    assert (store.source_dir("li") / "202606031230_flashes.json").exists()


def test_read_manifest_missing_is_empty():
    m = store.read_manifest("satellite")
    assert m["source"] == "satellite"
    assert m["frames"] == []
    assert store.latest_frame("satellite") is None


def test_prune_by_age_deletes_dirs_and_manifest_entries():
    for fid in ("202606031210", "202606031225", "202606031230"):
        store.record_frame("radar", _dt(fid))
        _make_frame_dir("radar", fid)

    removed = store.prune("radar", keep_seconds=600, keep_frames=None, now=_dt("202606031230"))

    assert removed == ["202606031210"]  # 20 min old, cutoff is 10 min
    assert not store.frame_zarr_path("radar", "202606031210").exists()
    assert store.frame_zarr_path("radar", "202606031225").exists()
    assert [f["frame_id"] for f in store.list_frames("radar")] == [
        "202606031225",
        "202606031230",
    ]


def test_prune_by_count_keeps_newest_n():
    for fid in ("202606031210", "202606031215", "202606031220", "202606031225"):
        store.record_frame("radar", _dt(fid))
        _make_frame_dir("radar", fid)

    removed = store.prune("radar", keep_seconds=None, keep_frames=2, now=_dt("202606031225"))

    assert removed == ["202606031210", "202606031215"]
    assert [f["frame_id"] for f in store.list_frames("radar")] == [
        "202606031220",
        "202606031225",
    ]


def test_prune_noop_when_unlimited():
    store.record_frame("radar", _dt("202606031230"))
    assert store.prune("radar", keep_seconds=None, keep_frames=None) == []
    assert len(store.list_frames("radar")) == 1


# -- consecutive-failure memo ---------------------------------------------
# NB: unlike frames (pruned only on an explicit prune() call), the failure memo
# is pruned against wall-clock now on every write, so these tests must use
# now-relative frame ids — the fixed 2026-06-03 ids used above are older than the
# retention window and would be dropped as soon as they were written.
def _now_fid(minutes_ago: float = 0) -> tuple[dt.datetime, str]:
    when = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_ago)).replace(
        second=0, microsecond=0
    )
    return when, store.frame_id(when)


def test_failure_count_increments_and_persists():
    _, fid = _now_fid()
    _, other_fid = _now_fid(minutes_ago=5)
    assert store.failure_count("satellite", fid) == 0
    assert store.record_failure("satellite", fid) == 1
    assert store.record_failure("satellite", fid) == 2
    # Re-read from the manifest on disk, not in-memory state.
    assert store.failure_count("satellite", fid) == 2
    # Counts are per frame id and per source.
    assert store.failure_count("satellite", other_fid) == 0
    assert store.failure_count("mtg", fid) == 0


def test_successful_record_frame_clears_failure_memo():
    when, fid = _now_fid()
    store.record_failure("satellite", fid)
    store.record_failure("satellite", fid)
    assert store.failure_count("satellite", fid) == 2  # guard: memo really is set
    store.record_frame("satellite", when, products={"hrv": "x.png"})
    assert store.failure_count("satellite", fid) == 0


def test_record_failure_prunes_entries_older_than_retention():
    now = dt.datetime.now(dt.timezone.utc)
    old = store.frame_id(now - dt.timedelta(hours=48))
    recent = store.frame_id(now)
    store.record_failure("mtg", old)
    store.record_failure("mtg", recent)  # write prunes the aged-out entry
    failed = store.read_manifest("mtg").get("failed", {})
    assert old not in failed
    assert recent in failed


def test_prune_failures_drops_unparseable_frame_ids():
    assert store._prune_failures({"not-a-frame-id": 3}) == {}
