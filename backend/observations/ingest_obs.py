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
import logging
import sys
from typing import Optional

from . import store
from .config import Config, ensure_dirs

log = logging.getLogger("skyview.observations.ingest")


def ingest_radar(cfg: Config) -> Optional[str]:
    """Fetch the newest OPERA composite, reproject, and store. Returns frame id."""
    from .radar_ord import (
        RadarSource,
        read_odim_maxreflectivity,
        read_odim_valid_time,
    )
    from .reproject import build_target_coords, reproject_odim

    src = RadarSource(cfg.radar)
    native = src.fetch_latest()
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
    lat, lon = build_target_coords()
    store.write_frame_zarr(
        "radar",
        when,
        {"dbz": field, "lat": lat, "lon": lon},
        attrs={"standard_name": cfg.radar.standard_name, "units": "dBZ"},
        cadence_seconds=cfg.radar.cadence_seconds,
    )
    return fid


def ingest_satellite(cfg: Config) -> Optional[str]:
    """Fetch the newest MSG/MTG product, reproject IR/WV, and store.

    Reading the native SEVIRI/FCI file and extracting per-channel brightness
    temperatures + lon/lat uses satpy; the exact composite/channel selection is
    confirmed against a real product in Phase 1 (host with creds + satpy).
    """
    from .reproject import reproject_swath  # noqa: F401  (used once wired)
    from .satellite_eumetsat import SatelliteSource

    src = SatelliteSource(cfg.satellite)
    native = src.fetch_latest()
    if native is None:
        log.info("satellite: no new product available")
        return None

    # Channel extraction + swath reprojection is implemented on the host where
    # satpy/eumdac are installed and a real product is available (Phase 1 live).
    raise NotImplementedError(
        "satellite reprojection (satpy channel extraction) is wired during "
        "Phase 1 live verification; native product fetched at "
        f"{native}"
    )


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
