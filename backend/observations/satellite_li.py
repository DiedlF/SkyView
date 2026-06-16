"""Lightning source: MTG-I1 (Meteosat-12) Lightning Imager (LI) L2 via the Data Store.

LI is a separate optical lightning instrument on MTG-I1 (not an FCI channel), so
this is its own observation source. We use the point-based "Lightning Flashes"
(LFL) product (``EO:EUM:DAT:0691``): a list of individual detected flashes, each
with its own lat/lon, satpy reader ``li_l2_nc``. ``ingest_li`` extracts the flash
points and the frontend draws them as red dots (see ``render.extract_li_flashes``).

LFL is delivered as many small ``.nc`` granules per 10-min cycle, so
``search_window`` collects a trailing window of granules and ``ingest_li`` merges
their flashes into one frame. Bodies are small, so there is no chunk subsetting.

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
        """Return the most recent LI LFL product (metadata only; no body download)."""
        products = self.search_window(30 * 60)
        return products[0] if products else None

    def search_window(self, window_seconds: int):
        """Return all LFL products sensed within the trailing ``window_seconds``.

        Sorted newest-first. LFL is delivered as many small granules per cycle, so
        ``ingest_li`` merges a window's flashes into one frame; ``search_latest``
        (the newest of these) provides the frame's dedup/sensing time.
        """
        self._connect()
        collection = self._datastore.get_collection(self.cfg.collection_id)
        now = dt.datetime.now(dt.timezone.utc)
        window = max(int(window_seconds), 60)
        products = list(
            collection.search(dtstart=now - dt.timedelta(seconds=window), dtend=now)
        )
        if not products:
            log.info("No MTG LI products in the last %d s", window)
            return []
        epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        products.sort(key=lambda p: (product_sensing_time(p) or epoch, str(p)), reverse=True)
        return products

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
