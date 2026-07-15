"""Satellite source: EUMETSAT Data Store via EUMDAC.

Default collection is MSG Rapid Scanning Service (5-minute, Europe). Switch the
collection id in config to MTG FCI (10-minute) when you want the new generation.

Requires:  pip install eumdac (and satpy[seviri] netCDF4 for optional rendering).
Credentials: set EUMETSAT_CONSUMER_KEY / _SECRET, or run `eumdac set-credentials`.
Verify with: python3 scripts/eumetsat_auth.py

The MSG RSS Data Store payload is a ZIP containing a SEVIRI native ``.nat`` file.
In normal ingest this native file is only a temporary render input; the retained
cache product is a derived PNG frame indexed by ``store.manifest.json``.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import zipfile
from pathlib import Path
from typing import Optional

from .config import SAT_DIR, SatelliteConfig

log = logging.getLogger("skyview.observations.satellite")


class SatelliteSource:
    def __init__(self, cfg: SatelliteConfig):
        self.cfg = cfg
        self._datastore = None

    def _connect(self):
        if self._datastore is not None:
            return
        from .eumdac_client import get_datastore

        self._datastore = get_datastore(
            self.cfg.consumer_key, self.cfg.consumer_secret
        )

    def search_latest(self):
        """Return the most recent product (metadata only; no body download).

        Split from the download so ``ingest_satellite`` can derive the frame time
        from the product id and de-duplicate against the manifest *before* pulling
        the (large) body — matching the MTG/LI cost guards.
        """
        self._connect()
        collection = self._datastore.get_collection(self.cfg.collection_id)
        # Search the last 30 minutes, newest first, take one.
        now = dt.datetime.now(dt.timezone.utc)
        products = sorted(
            collection.search(dtstart=now - dt.timedelta(minutes=30), dtend=now),
            key=lambda p: str(p),
            reverse=True,
        )
        if not products:
            log.info("No satellite products in the last 30 min")
            return None
        return products[0]

    def download_product(self, product, dest_dir: Path = SAT_DIR) -> Path:
        """Download one product's body as a ZIP. Call after dedup/selection."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{product}.zip"
        with product.open() as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 16):
                dst.write(chunk)
        log.info("Saved satellite product -> %s", out)
        return out

    def fetch_latest(self, dest_dir: Path = SAT_DIR) -> Optional[Path]:
        """Search + download the most recent product (one-shot; poller path)."""
        product = self.search_latest()
        if product is None:
            return None
        return self.download_product(product, dest_dir)


def extract_native_file(product_path: Path, dest_dir: Optional[Path] = None) -> Path:
    """Return a native ``.nat`` file for a downloaded EUMETSAT product.

    The live MSG RSS product is a ZIP even when older code saved it with a
    misleading suffix. Sniff the file contents, not just the extension.
    """
    product_path = Path(product_path)
    if product_path.suffix.lower() == ".nat":
        return product_path
    if not zipfile.is_zipfile(product_path):
        raise ValueError(f"satellite product is not a ZIP or .nat file: {product_path}")

    out_dir = dest_dir or product_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(product_path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".nat")]
        if not names:
            raise ValueError(f"satellite product ZIP has no .nat member: {product_path}")
        member = names[0]
        out = out_dir / Path(member).name
        with zf.open(member) as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        return out


def read_msg_valid_time(path: Path) -> Optional[dt.datetime]:
    """Parse the nominal MSG product time from a native/ZIP filename."""
    match = re.search(r"-(\d{14})(?:\.\d+)?Z-", Path(path).name)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError:
        return None


def to_europe_rgb(native_file: Path, out_png: Path) -> Path:
    """Render a quick natural-color RGB cropped to Europe using satpy.

    Diagnostic convenience only; for serving/analysis keep the native NetCDF file.
    """
    from satpy import Scene
    scn = Scene(reader="seviri_l1b_native", filenames=[str(native_file)])
    composite = "natural_color"
    scn.load([composite])
    europe = scn.crop(ll_bbox=(-15.0, 32.0, 45.0, 72.0))
    europe.save_dataset(composite, filename=str(out_png))
    log.info("Rendered Europe RGB -> %s", out_png)
    return out_png
