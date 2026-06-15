"""Observation ingest orchestration: fetch → reproject → store → prune.

Run once (cron-friendly) or for a single source:

    python -m backend.observations.ingest_obs --source radar
    python -m backend.observations.ingest_obs --source mtg --once
    python -m backend.observations.ingest_obs --source all --once

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


def ingest_mtg(cfg: Config) -> Optional[str]:
    """Fetch the newest MTG-I1 FCI cycle, render vis_06, and store. Returns frame id.

    Two cost guards: (1) de-duplicate against the manifest using the search
    result's sensing time BEFORE any download, and (2) download only the northern
    (Europe) chunk entries, not the full disk.
    """
    from .render import render_fci_vis_png
    from .satellite_mtg import (
        MtgSource,
        parse_chunk_spec,
        product_sensing_time,
        select_europe_chunks,
    )

    src = MtgSource(cfg.mtg)
    product = src.search_latest()
    if product is None:
        log.info("mtg: no new product available")
        return None

    when = product_sensing_time(product)
    if when is None:
        log.warning("mtg: could not determine sensing time for %s", product)
        return None
    fid = store.frame_id(when)
    if store.has_frame("mtg", fid):
        log.info("mtg: frame %s already ingested; skipping download", fid)
        return None

    entries = src.product_entries(product)
    selected = select_europe_chunks(
        entries,
        fraction=cfg.mtg.chunk_fraction,
        explicit=parse_chunk_spec(cfg.mtg.chunks),
    )
    if not selected:
        log.warning("mtg: product %s exposed no chunk entries to download", product)
        return None
    log.info("mtg: downloading %d/%d chunks for %s", len(selected), len(entries), fid)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mtg-", dir=str(TMP_DIR)) as tmp:
        chunks = src.download_entries(product, selected, Path(tmp))
        if not chunks:
            log.warning("mtg: no chunk files downloaded for %s", product)
            return None

        tmp_png = Path(tmp) / "mtg_vis06.png"
        render_fci_vis_png(chunks, tmp_png, bbox=cfg.mtg.roi_bbox, channel=cfg.mtg.channel)
        store.write_frame_render(
            "mtg",
            when,
            cfg.mtg.channel,
            tmp_png,
            attrs={
                "channel": cfg.mtg.channel,
                "collection_id": cfg.mtg.collection_id,
                "satellite": "MTG-I1",
                "chunk_count": len(chunks),
                "cache": "derived_render",
            },
            cadence_seconds=cfg.mtg.cadence_seconds,
        )
        return fid


def ingest_li(cfg: Config) -> Optional[str]:
    """Fetch the newest MTG-I1 LI accumulated-flashes product, render, and store.

    De-duplicates on the search result's sensing time before downloading. The AF
    product is a single small ``.nc`` body, so there is no chunk subsetting.
    """
    from .render import render_li_accum_png
    from .satellite_li import LiSource, product_sensing_time

    src = LiSource(cfg.li)
    product = src.search_latest()
    if product is None:
        log.info("li: no new product available")
        return None

    when = product_sensing_time(product)
    if when is None:
        log.warning("li: could not determine sensing time for %s", product)
        return None
    fid = store.frame_id(when)
    if store.has_frame("li", fid):
        log.info("li: frame %s already ingested; skipping download", fid)
        return None

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="li-", dir=str(TMP_DIR)) as tmp:
        bodies = src.download_body(product, Path(tmp))
        if not bodies:
            log.warning("li: product %s had no NetCDF body to render", product)
            return None

        tmp_png = Path(tmp) / "li_af.png"
        render_li_accum_png(bodies, tmp_png, bbox=cfg.li.roi_bbox, dataset=cfg.li.dataset)
        store.write_frame_render(
            "li",
            when,
            "af",
            tmp_png,
            attrs={
                "product": "accumulated_flashes",
                "dataset": cfg.li.dataset,
                "collection_id": cfg.li.collection_id,
                "satellite": "MTG-I1",
                "instrument": "LI",
                "cache": "derived_render",
            },
            cadence_seconds=cfg.li.cadence_seconds,
        )
        return fid


_INGESTORS = {
    "radar": ingest_radar,
    "satellite": ingest_satellite,
    "mtg": ingest_mtg,
    "li": ingest_li,
}


def run_once(source: str, cfg: Optional[Config] = None) -> dict:
    """Ingest one cycle for ``source`` in {radar, satellite, mtg, both, all}.

    ``both`` = radar + satellite (legacy); ``all`` = every source. Returns a
    dict of ``{source: frame_id_or_None}``; never raises for a single source
    failure (errors are logged and recorded as None).
    """
    ensure_dirs()
    cfg = cfg or Config()
    results: dict[str, Optional[str]] = {}
    if source == "both":
        targets = ["radar", "satellite"]
    elif source == "all":
        targets = list(_INGESTORS)
    else:
        targets = [source]
    for tgt in targets:
        ingest = _INGESTORS.get(tgt)
        if ingest is None:
            log.error("unknown source: %s", tgt)
            results[tgt] = None
            continue
        try:
            results[tgt] = ingest(cfg)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            log.error("%s ingest failed: %s", tgt, exc)
            results[tgt] = None
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SkyView observation ingest (fetch→reproject→store).")
    parser.add_argument("--source", choices=["radar", "satellite", "mtg", "li", "both", "all"], default="radar")
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
