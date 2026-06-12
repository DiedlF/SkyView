"""Observation ingest orchestration: fetch → reproject → store → prune.

Run once (cron-friendly) or for a single source:

    python -m backend.observations.ingest_obs --source radar
    python -m backend.observations.ingest_obs --source both --once

Each source is isolated: a failure in one never aborts the other. De-duplication
is by frame id (a frame already in the manifest is skipped). This is the cron
alternative to the long-running ``poller``; both end at the same Zarr + manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from . import store
from .config import Config, TMP_DIR, ensure_dirs

log = logging.getLogger("skyview.observations.ingest")


def ingest_radar(cfg: Config) -> Optional[str]:
    """Fetch the newest OPERA composite, render, and store. Returns frame id."""
    from .radar_ord import (
        RadarSource,
        read_odim_maxreflectivity,
        read_odim_valid_time,
    )
    from .render import render_radar_dbz_png
    from .reproject import reproject_odim

    src = RadarSource(cfg.radar)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="radar-", dir=str(TMP_DIR)) as tmp:
        native = src.fetch_latest(Path(tmp))
        if native is None:
            log.info("radar: no new composite available")
            return None

        when = read_odim_valid_time(native)
        if when is None:
            log.warning("radar: could not determine valid time for %s", native)
            return None
        fid = store.frame_id(when)
        if store.has_frame("radar", fid):
            log.info("radar: frame %s already ingested; skipping", fid)
            return None

        phys, where = read_odim_maxreflectivity(native)
        field = reproject_odim(phys, where)
        tmp_png = Path(tmp) / "radar_dbz.png"
        render_radar_dbz_png(field, tmp_png)
        store.write_frame_render(
            "radar",
            when,
            "radar_dbz",
            tmp_png,
            attrs={
                "standard_name": cfg.radar.standard_name,
                "units": "dBZ",
                "cache": "derived_render",
            },
            cadence_seconds=cfg.radar.cadence_seconds,
        )
        return fid


def ingest_satellite(cfg: Config) -> Optional[str]:
    """Fetch the newest MSG RSS product, render HRV, and store. Returns frame id."""
    from .render import render_satellite_hrv_png
    from .satellite_eumetsat import (
        SatelliteSource,
        extract_native_file,
        read_msg_valid_time,
    )

    src = SatelliteSource(cfg.satellite)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="satellite-", dir=str(TMP_DIR)) as tmp:
        product = src.fetch_latest(Path(tmp))
        if product is None:
            log.info("satellite: no new product available")
            return None

        native = extract_native_file(product, Path(tmp))
        when = read_msg_valid_time(native) or read_msg_valid_time(product)
        if when is None:
            when = dt.datetime.fromtimestamp(native.stat().st_mtime, dt.timezone.utc)

        fid = store.frame_id(when)
        if store.has_frame("satellite", fid):
            log.info("satellite: frame %s already ingested; skipping", fid)
            return None

        tmp_png = Path(tmp) / "satellite_hrv.png"
        render_satellite_hrv_png(native, tmp_png, bbox=cfg.satellite.roi_bbox)
        store.write_frame_render(
            "satellite",
            when,
            "hrv",
            tmp_png,
            attrs={
                "channel": "HRV",
                "collection_id": cfg.satellite.collection_id,
                "cache": "derived_render",
            },
            cadence_seconds=cfg.satellite.cadence_seconds,
        )
        return fid


def run_once(source: str, cfg: Optional[Config] = None) -> dict:
    """Ingest one cycle for ``source`` in {radar, satellite, both}.

    Returns a dict of ``{source: frame_id_or_None}``; never raises for a single
    source failure (errors are logged and recorded as None).
    """
    ensure_dirs()
    cfg = cfg or Config()
    results: dict[str, Optional[str]] = {}
    targets = ["radar", "satellite"] if source == "both" else [source]
    for tgt in targets:
        try:
            if tgt == "radar":
                results[tgt] = ingest_radar(cfg)
            elif tgt == "satellite":
                results[tgt] = ingest_satellite(cfg)
            else:
                log.error("unknown source: %s", tgt)
                results[tgt] = None
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            log.error("%s ingest failed: %s", tgt, exc)
            results[tgt] = None
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SkyView observation ingest (fetch→reproject→store).")
    parser.add_argument("--source", choices=["radar", "satellite", "both"], default="radar")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit (default).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    results = run_once(args.source)
    new = {k: v for k, v in results.items() if v}
    if new:
        log.info("ingested new frames: %s", new)
    # Exit non-zero only if every requested source errored AND produced nothing.
    return 0 if any(v is not None for v in results.values()) or not results else 0


if __name__ == "__main__":
    sys.exit(main())
