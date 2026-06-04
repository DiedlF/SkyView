"""Poller: drives both observation sources on their cadence without re-downloading.

Run as:  python -m backend.observations.poller
A simple last-seen marker per source prevents duplicate fetches. Designed to be
resilient -- a failure in one source never stops the other. This is the interval
alternative to the MQTT push trigger (Phase 4).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import STATE_DIR, Config, ensure_dirs
from .radar_ord import RadarSource
from .satellite_eumetsat import SatelliteSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("skyview.observations.poller")


def _marker(name: str) -> Path:
    return STATE_DIR / f"{name}.last"


def _seen(name: str, key: str) -> bool:
    m = _marker(name)
    return m.exists() and m.read_text().strip() == key


def _mark(name: str, key: str) -> None:
    _marker(name).write_text(key)


def run() -> None:
    ensure_dirs()
    cfg = Config()
    radar = RadarSource(cfg.radar)
    sat = SatelliteSource(cfg.satellite)

    # One-time startup check. ORD is openly available via MeteoGate since
    # 20 May 2026 -- no whitelisting. EDR falls back to the unsigned S3 cache.
    try:
        cols = radar.list_collections()
        log.info("Radar collections available: %d", len(cols))
    except Exception as exc:  # noqa: BLE001 - startup diagnostics only
        log.warning("Could not list radar collections: %s", exc)

    next_radar = 0.0
    next_sat = 0.0
    while True:
        now = time.monotonic()

        if now >= next_radar:
            try:
                path = radar.fetch_latest()
                if path and not _seen("radar", path.name):
                    _mark("radar", path.name)
                    log.info("New radar frame: %s", path.name)
            except Exception as exc:  # noqa: BLE001
                log.error("Radar fetch error: %s", exc)
            next_radar = now + cfg.radar.poll_interval

        if now >= next_sat:
            try:
                path = sat.fetch_latest()
                if path and not _seen("sat", path.name):
                    _mark("sat", path.name)
                    log.info("New satellite product: %s", path.name)
            except Exception as exc:  # noqa: BLE001
                log.error("Satellite fetch error: %s", exc)
            next_sat = now + cfg.satellite.poll_interval

        time.sleep(5)


if __name__ == "__main__":
    run()
