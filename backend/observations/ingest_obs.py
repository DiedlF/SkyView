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


def _skip_failed(source: str, fid: str) -> bool:
    """True if ``fid`` has already failed too many times to retry this tick.

    Guards the download+render of a frame that keeps failing (e.g. a broken
    upstream product) so it is not re-fetched every tick until it ages out.
    """
    count = store.failure_count(source, fid)
    if count >= store.FRAME_FAILURE_LIMIT:
        log.warning(
            "%s: frame %s failed %d× already; skipping until it ages out",
            source, fid, count,
        )
        return True
    return False


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
    """Fetch the newest MSG RSS product, render HRV, and store. Returns frame id.

    De-duplicates against the manifest using the product id's nominal time BEFORE
    downloading the (large) body, and skips frames that have already failed
    ``store.FRAME_FAILURE_LIMIT`` times — so a single broken product is not
    re-downloaded and re-rendered on every 2-min tick until it ages out.
    """
    from .render import render_satellite_hrv_png
    from .satellite_eumetsat import (
        SatelliteSource,
        extract_native_file,
        read_msg_valid_time,
    )

    src = SatelliteSource(cfg.satellite)
    product = src.search_latest()
    if product is None:
        log.info("satellite: no new product available")
        return None

    when = read_msg_valid_time(str(product))
    if when is None:
        log.warning("satellite: could not determine valid time for %s", product)
        return None
    fid = store.frame_id(when)
    if store.has_frame("satellite", fid):
        log.info("satellite: frame %s already ingested; skipping download", fid)
        return None
    if _skip_failed("satellite", fid):
        return None

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="satellite-", dir=str(TMP_DIR)) as tmp:
        try:
            product_path = src.download_product(product, Path(tmp))
            native = extract_native_file(product_path, Path(tmp))
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
        except Exception:
            store.record_failure("satellite", fid)
            raise
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
    if _skip_failed("mtg", fid):
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
        try:
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
        except Exception:
            store.record_failure("mtg", fid)
            raise
        return fid


def ingest_li(cfg: Config) -> Optional[str]:
    """Fetch the latest MTG-I1 LI Lightning-Flashes cycle, extract points, and store.

    LFL is delivered as many small ``.nc`` granules per cycle, so we collect a
    trailing window of granules, merge their flashes into one frame keyed by the
    newest sensing time, and store them as a JSON point list. De-duplicates on
    that frame id before downloading.
    """
    from .render import extract_li_flashes
    from .satellite_li import LiSource, product_sensing_time

    src = LiSource(cfg.li)
    products = src.search_window(cfg.li.window_seconds)
    if not products:
        log.info("li: no new product available")
        return None

    when = product_sensing_time(products[0])  # newest granule -> frame/dedup key
    if when is None:
        log.warning("li: could not determine sensing time for %s", products[0])
        return None
    fid = store.frame_id(when)
    if store.has_frame("li", fid):
        log.info("li: frame %s already ingested; skipping download", fid)
        return None
    if _skip_failed("li", fid):
        return None

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="li-", dir=str(TMP_DIR)) as tmp:
        try:
            bodies: list[Path] = []
            for product in products:
                bodies.extend(src.download_body(product, Path(tmp)))
            if not bodies:
                log.warning("li: cycle %s had no NetCDF bodies to read", fid)
                return None

            flashes = extract_li_flashes(bodies, bbox=cfg.li.roi_bbox, dataset=cfg.li.dataset)
            log.info("li: extracted %d flashes for frame %s", len(flashes), fid)
            store.write_frame_points(
                "li",
                when,
                "flashes",
                flashes,
                attrs={
                    "product": "lightning_flashes",
                    "dataset": cfg.li.dataset,
                    "collection_id": cfg.li.collection_id,
                    "satellite": "MTG-I1",
                    "instrument": "LI",
                    "granules": len(products),
                    "count": len(flashes),
                    "cache": "derived_points",
                },
                cadence_seconds=cfg.li.cadence_seconds,
            )
        except Exception:
            store.record_failure("li", fid)
            raise
        return fid


_INGESTORS = {
    "radar": ingest_radar,
    "satellite": ingest_satellite,
    "mtg": ingest_mtg,
    "li": ingest_li,
}


def _source_cadences(cfg: Config) -> dict[str, int]:
    return {
        "radar": cfg.radar.cadence_seconds,
        "satellite": cfg.satellite.cadence_seconds,
        "mtg": cfg.mtg.cadence_seconds,
        "li": cfg.li.cadence_seconds,
    }


def fresh_within_cadence(
    latest: Optional[dict], cadence_seconds: Optional[int], now: dt.datetime
) -> bool:
    """True if the newest stored frame is younger than one cadence.

    When the latest frame is still within its cadence window, no new frame is
    published yet, so there's nothing to fetch. Gating on this skips the upstream
    request entirely — it stops the redundant re-downloads (and, for radar, the
    EDR API hits that trip MeteoGate's anonymous rate limit) that happen because
    the cron fires more often (2 min) than the products update (5–10 min).
    """
    if not latest or not cadence_seconds:
        return False
    fid = latest.get("frame_id")
    if not fid:
        return False
    try:
        when = store.parse_frame_id(str(fid))
    except ValueError:
        return False
    return (now - when).total_seconds() < cadence_seconds


def run_once(
    source: str, cfg: Optional[Config] = None, *, report: Optional[dict] = None
) -> dict:
    """Ingest one cycle for ``source`` in {radar, satellite, mtg, li, both, all}.

    ``both`` = radar + satellite (legacy); ``all`` = every source. Returns a
    dict of ``{source: frame_id_or_None}``; never raises for a single source
    failure (errors are logged and recorded as None). Sources whose latest frame
    is still within its cadence window are skipped without an upstream request.

    A ``None`` result is ambiguous on its own — it covers both "nothing new
    upstream" (benign) and "this source raised". When a ``report`` dict is
    passed it is populated with ``attempted`` and ``errored`` source-name sets so
    the caller (``main``) can tell a genuinely failed run from a quiet one.
    """
    ensure_dirs()
    cfg = cfg or Config()
    cadences = _source_cadences(cfg)
    now = dt.datetime.now(dt.timezone.utc)
    results: dict[str, Optional[str]] = {}
    attempted: set[str] = set()
    errored: set[str] = set()
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
            attempted.add(tgt)
            errored.add(tgt)
            continue
        if fresh_within_cadence(store.latest_frame(tgt), cadences.get(tgt), now):
            log.debug("%s: latest frame within cadence; skipping fetch", tgt)
            results[tgt] = None
            continue
        attempted.add(tgt)
        try:
            results[tgt] = ingest(cfg)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            log.error("%s ingest failed: %s", tgt, exc)
            results[tgt] = None
            errored.add(tgt)
    if report is not None:
        report["attempted"] = attempted
        report["errored"] = errored
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

    report: dict = {}
    results = run_once(args.source, report=report)
    new = {k: v for k, v in results.items() if v}
    if new:
        log.info("ingested new frames: %s", new)

    # Exit non-zero only when every source we actually tried raised. A quiet run
    # (nothing new upstream, or every source skipped inside its cadence window)
    # is a success, and a partial failure still leaves the served frames current
    # — but a run where nothing worked should be visible in systemd rather than
    # reported as OK.
    attempted: set = report.get("attempted", set())
    errored: set = report.get("errored", set())
    if attempted and errored == attempted:
        log.error("all attempted sources failed: %s", ", ".join(sorted(errored)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
