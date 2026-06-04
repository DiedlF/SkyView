"""Satellite source: EUMETSAT Data Store via EUMDAC.

Default collection is MSG Rapid Scanning Service (5-minute, Europe). Switch the
collection id in config to MTG FCI (10-minute) when you want the new generation.

Requires:  pip install eumdac (and satpy[seviri] netCDF4 for optional rendering).
Credentials: set EUMETSAT_CONSUMER_KEY / _SECRET, or run `eumdac set-credentials`.
Verify with: python3 scripts/eumetsat_auth.py

The native NetCDF file is the source of truth; server-side colorization (Phase 2)
works from physical brightness-temperature values rather than a baked RGB, so the
satpy RGB helper below is a diagnostic convenience only.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Optional

from .config import SAT_DIR, SatelliteConfig

log = logging.getLogger("skyview.observations.satellite")


class SatelliteSource:
    def __init__(self, cfg: SatelliteConfig):
        self.cfg = cfg
        self._token = None
        self._datastore = None

    def _connect(self):
        if self._datastore is not None:
            return
        import eumdac

        if not (self.cfg.consumer_key and self.cfg.consumer_secret):
            raise RuntimeError(
                "Missing EUMETSAT credentials. Set EUMETSAT_CONSUMER_KEY and "
                "EUMETSAT_CONSUMER_SECRET, or run `eumdac set-credentials`. "
                "Verify with: python3 scripts/eumetsat_auth.py"
            )
        self._token = eumdac.AccessToken(
            (self.cfg.consumer_key, self.cfg.consumer_secret)
        )
        self._datastore = eumdac.DataStore(self._token)
        log.info("Connected to EUMETSAT Data Store")

    def fetch_latest(self, dest_dir: Path = SAT_DIR) -> Optional[Path]:
        """Download the most recent product in the configured collection."""
        self._connect()
        dest_dir.mkdir(parents=True, exist_ok=True)

        collection = self._datastore.get_collection(self.cfg.collection_id)
        # Search the last 30 minutes, newest first, take one.
        now = dt.datetime.now(dt.timezone.utc)
        products = collection.search(
            dtstart=now - dt.timedelta(minutes=30),
            dtend=now,
        )
        products = sorted(products, key=lambda p: str(p), reverse=True)
        if not products:
            log.info("No satellite products in the last 30 min")
            return None

        product = products[0]
        out = dest_dir / f"{product}.nc"
        with product.open() as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 16):
                dst.write(chunk)
        log.info("Saved satellite product -> %s", out)
        return out


def to_europe_rgb(native_file: Path, out_png: Path) -> Path:
    """Render a quick natural-color RGB cropped to Europe using satpy.

    Diagnostic convenience only; for serving/analysis keep the native NetCDF file.
    """
    from satpy import Scene
    from satpy.writers import to_image

    scn = Scene(reader="seviri_l1b_native", filenames=[str(native_file)])
    composite = "natural_color"
    scn.load([composite])
    europe = scn.crop(ll_bbox=(-15.0, 32.0, 45.0, 72.0))
    img = to_image(europe[composite])
    img.save(str(out_png))
    log.info("Rendered Europe RGB -> %s", out_png)
    return out_png
