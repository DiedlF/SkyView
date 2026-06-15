"""Lightning source: MTG-I1 (Meteosat-12) Lightning Imager (LI) L2 via the Data Store.

LI is a separate optical lightning instrument on MTG-I1 (not an FCI channel), so
this is its own observation source rendering a coloured, transparent overlay that
sits on top of either satellite image. We use the gridded "Accumulated Flashes"
(AF) product (``EO:EUM:DAT:0686``): a sparse 2-D field on the FCI 2 km
geostationary grid, satpy reader ``li_l2_nc``, dataset ``flash_accumulation``.

Unlike FCI, the AF product is small (one ~2–3 MB single body chunk per 10-min
cycle), so there is no chunk subsetting — ``ingest_li`` just de-duplicates on the
search result's sensing time (no wasted downloads) and fetches the ``.nc`` body.

Credentials are shared with the MSG/MTG sources (EUMETSAT_CONSUMER_KEY / _SECRET).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from .config import LI_DIR, LiConfig
# product_sensing_time / read_fci_valid_time are generic WMO-filename helpers
# (not FCI-specific): LI products share the same naming, so reuse them.
from .satellite_mtg import product_sensing_time

log = logging.getLogger("skyview.observations.li")


class LiSource:
    def __init__(self, cfg: LiConfig):
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
        log.info("Connected to EUMETSAT Data Store (LI)")

    def search_latest(self):
        """Return the most recent LI AF product (metadata only; no body download)."""
        self._connect()
        collection = self._datastore.get_collection(self.cfg.collection_id)
        now = dt.datetime.now(dt.timezone.utc)
        products = list(
            collection.search(dtstart=now - dt.timedelta(minutes=30), dtend=now)
        )
        if not products:
            log.info("No MTG LI products in the last 30 min")
            return None
        epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        products.sort(key=lambda p: (product_sensing_time(p) or epoch, str(p)), reverse=True)
        return products[0]

    def download_body(self, product, dest_dir: Path = LI_DIR) -> list[Path]:
        """Download the product's NetCDF body file(s). Call after dedup.

        The accumulated product has a single ``-BODY-`` ``.nc`` data file plus a
        tiny trailer; satpy reads only the body, so we fetch the ``.nc`` entries
        (small) and skip the XML manifests.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_paths: list[Path] = []
        for name in (str(e) for e in (getattr(product, "entries", None) or [])):
            low = name.lower()
            if not low.endswith(".nc") or "-trail-" in low:
                continue
            out = dest_dir / Path(name).name
            with product.open(entry=name) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 16):
                    dst.write(chunk)
            out_paths.append(out)
        log.info("Downloaded %d MTG LI body file(s) -> %s", len(out_paths), dest_dir)
        return out_paths
