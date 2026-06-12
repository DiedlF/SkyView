"""Frame storage + manifest/ring-buffer for the observation layer.

Each reprojected frame is written as a Zarr group under
``data/observations/<source>/<frame_id>.zarr`` and indexed in a per-source
``manifest.json``. The manifest is the time index the serving layer (Phase 2)
and frontend read instead of listing the directory on the hot path.

Retention is a rolling window (``prune``), unlike forecast data which keeps only
the latest run.

The manifest/retention logic is pure stdlib (json/os/datetime/shutil) so it is
unit-testable without numpy/zarr; only ``write_frame_zarr`` / ``load_frame``
touch the heavy Zarr dependencies, and those imports are lazy.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from .config import DATA_ROOT, RETENTION_MAX_FRAMES, RETENTION_SECONDS

log = logging.getLogger("skyview.observations.store")

MANIFEST_NAME = "manifest.json"
FRAME_ID_FMT = "%Y%m%d%H%M"  # minute resolution


# -- time/path helpers (pure) ---------------------------------------------
def frame_id(when: dt.datetime) -> str:
    """Canonical frame id, e.g. ``202606031230`` (UTC, minute resolution)."""
    return _as_utc(when).strftime(FRAME_ID_FMT)


def parse_frame_id(fid: str) -> dt.datetime:
    return dt.datetime.strptime(fid, FRAME_ID_FMT).replace(tzinfo=dt.timezone.utc)


def valid_time_iso(when: dt.datetime) -> str:
    return _as_utc(when).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_dir(source: str) -> Path:
    return DATA_ROOT / source


def frame_zarr_path(source: str, fid: str) -> Path:
    return source_dir(source) / f"{fid}.zarr"


def manifest_path(source: str) -> Path:
    return source_dir(source) / MANIFEST_NAME


def _as_utc(when: dt.datetime) -> dt.datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc)


# -- manifest (pure) -------------------------------------------------------
def read_manifest(source: str) -> dict:
    """Return the manifest dict, or a fresh empty one if absent/corrupt."""
    path = manifest_path(source)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("frames"), list):
            return data
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Manifest %s unreadable (%s); starting fresh", path, exc)
    return {"source": source, "cadence_seconds": None, "frames": []}


def write_manifest(source: str, manifest: dict) -> None:
    path = manifest_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated"] = valid_time_iso(dt.datetime.now(dt.timezone.utc))
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def record_frame(source: str, when: dt.datetime, *, cadence_seconds: Optional[int] = None) -> dict:
    """Add (or refresh) a frame entry, keeping ``frames`` sorted & de-duplicated."""
    fid = frame_id(when)
    manifest = read_manifest(source)
    if cadence_seconds is not None:
        manifest["cadence_seconds"] = cadence_seconds
    frames = {f["frame_id"]: f for f in manifest.get("frames", [])}
    frames[fid] = {"frame_id": fid, "valid_time": valid_time_iso(when)}
    manifest["frames"] = [frames[k] for k in sorted(frames)]
    write_manifest(source, manifest)
    return manifest


def list_frames(source: str) -> list[dict]:
    return read_manifest(source).get("frames", [])


def latest_frame(source: str) -> Optional[dict]:
    frames = list_frames(source)
    return frames[-1] if frames else None


def has_frame(source: str, fid: str) -> bool:
    return any(f["frame_id"] == fid for f in list_frames(source))


# -- retention (pure filesystem) ------------------------------------------
def prune(
    source: str,
    *,
    keep_seconds: Optional[int] = RETENTION_SECONDS,
    keep_frames: Optional[int] = RETENTION_MAX_FRAMES,
    now: Optional[dt.datetime] = None,
) -> list[str]:
    """Drop frames older than ``keep_seconds`` and/or beyond ``keep_frames``.

    Deletes each removed frame's Zarr directory and rewrites the manifest.
    Returns the list of removed frame ids. With both limits ``None`` this is a
    no-op (other than rewriting the manifest's ``updated`` stamp is avoided).
    """
    if keep_seconds is None and keep_frames is None:
        return []

    now = _as_utc(now or dt.datetime.now(dt.timezone.utc))
    manifest = read_manifest(source)
    frames = sorted(manifest.get("frames", []), key=lambda f: f["frame_id"])

    keep_flags = [True] * len(frames)
    if keep_seconds is not None:
        cutoff = now - dt.timedelta(seconds=keep_seconds)
        for i, f in enumerate(frames):
            if parse_frame_id(f["frame_id"]) < cutoff:
                keep_flags[i] = False
    if keep_frames is not None and keep_frames > 0 and len(frames) > keep_frames:
        # keep only the newest `keep_frames` of those still flagged
        kept_idx = [i for i, k in enumerate(keep_flags) if k]
        for i in kept_idx[:-keep_frames]:
            keep_flags[i] = False

    removed: list[str] = []
    kept: list[dict] = []
    for f, keep in zip(frames, keep_flags):
        if keep:
            kept.append(f)
            continue
        removed.append(f["frame_id"])
        _remove_frame_dir(source, f["frame_id"])

    if removed:
        manifest["frames"] = kept
        write_manifest(source, manifest)
        log.info("Pruned %d %s frame(s): %s", len(removed), source, ", ".join(removed))
    return removed


def _remove_frame_dir(source: str, fid: str) -> None:
    path = frame_zarr_path(source, fid)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():  # stray file
        try:
            path.unlink()
        except OSError:
            pass


# -- Zarr I/O (lazy heavy deps) -------------------------------------------
def _storage_io():
    """Import the backend Zarr helpers regardless of how we were launched."""
    try:
        from services import storage_io  # backend dir on sys.path (server style)
    except ImportError:  # pragma: no cover - alt launch style
        from backend.services import storage_io  # repo root on sys.path
    return storage_io


def write_frame_zarr(
    source: str,
    when: dt.datetime,
    arrays: dict,
    attrs: Optional[dict] = None,
    *,
    cadence_seconds: Optional[int] = None,
    prune_after: bool = True,
) -> Path:
    """Write a reprojected frame as Zarr and index it in the manifest.

    ``arrays`` should contain the 2-D field(s) plus 1-D ``lat``/``lon`` axes.
    ``attrs`` is merged with standard provenance (source, valid_time, frame_id).
    """
    storage_io = _storage_io()
    if not storage_io.zarr_available():
        raise RuntimeError("zarr is not installed — cannot write observation frames")

    fid = frame_id(when)
    path = frame_zarr_path(source, fid)
    meta = {
        "source": source,
        "frame_id": fid,
        "valid_time": valid_time_iso(when),
        "grid": "regular_lat_lon",
    }
    if attrs:
        meta.update(attrs)

    ok = storage_io.write_zarr_group(str(path), arrays, meta)
    if not ok:
        raise RuntimeError(f"failed to write frame zarr at {path}")

    record_frame(source, when, cadence_seconds=cadence_seconds)
    log.info("Wrote %s frame %s -> %s", source, fid, path)
    if prune_after:
        prune(source)
    return path


def load_frame(source: str, frame: str, keys: Optional[Iterable[str]] = None) -> dict:
    """Load arrays for one frame. ``frame`` is a frame_id (or 'latest')."""
    if frame == "latest":
        latest = latest_frame(source)
        if latest is None:
            raise FileNotFoundError(f"no {source} frames available")
        frame = latest["frame_id"]
    path = frame_zarr_path(source, frame)
    if not path.is_dir():
        raise FileNotFoundError(f"frame zarr not found: {path}")
    storage_io = _storage_io()
    return storage_io._read_zarr(str(path), keys)
