"""Satellite source: MTG-I1 (Meteosat-12) FCI Level 1c via the EUMETSAT Data Store.

This is a separate source from the MSG RSS layer so the new-generation imagery
can be shown side-by-side with SEVIRI HRV for comparison. Default collection is
FCI L1c HRFI (``EO:EUM:DAT:0665``), and the rendered channel is ``vis_06`` — the
closest high-res visible analogue to SEVIRI HRV (0.5 km in HRFI).

An FCI L1c product is one (large) product per repeat cycle, internally split into
many "chunk" NetCDF files that are stacked vertically by image row — i.e. chunk
number maps to latitude band, with the highest-numbered body chunks at the north
(EUMETSAT's "Q4" northern-quarter coverage = the top chunks, which is Europe).
The full HRFI disk is hundreds of MB per cycle; SkyView only needs the northern
band over 40–60°N, so this module downloads **only the northern fraction of
chunks** (plus the trailer) as individual Data Store *entries* rather than the
whole product. That keeps both bandwidth and transient disk small.

Chunk selection is by *rank*, not absolute number, on purpose: FDHSI full disk is
40 body chunks but HRFI is ~70, and the absolute "Q4 = 29–40" numbers only hold
for the 40-chunk layout. Selecting the top ``chunk_fraction`` of whatever chunks
the product actually has is robust to either layout. ``EUCOMP_MTG_CHUNKS`` can
pin an explicit range (e.g. ``"29-40"``) if you know the layout.

To avoid even listing/fetching a cycle we already have, ``ingest_mtg`` derives the
frame time from the *search result* and de-duplicates before any download.

Requires:  pip install eumdac satpy netCDF4 pyproj pyresample
Credentials: shared with the MSG source (EUMETSAT_CONSUMER_KEY / _SECRET).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from .config import MTG_DIR, MtgConfig

log = logging.getLogger("skyview.observations.mtg")

# Sensing-start timestamp inside a WMO-format FCI filename / product id:
#   ..._OPE_<sensing_start>_<sensing_end>_...  (14 digits each, YYYYMMDDhhmmss)
_OPE_TIME_RE = re.compile(r"_OPE_(\d{14})_")
_ANY_TIME_RE = re.compile(r"(\d{14})")
# Trailing chunk number of an FCI L1c entry, e.g. ``..._0073_0067.nc`` -> 67.
_CHUNK_RE = re.compile(r"_(\d{3,4})\.nc$", re.IGNORECASE)


class MtgSource:
    def __init__(self, cfg: MtgConfig):
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
        log.info("Connected to EUMETSAT Data Store (MTG)")

    def search_latest(self):
        """Return the most recent FCI product (metadata only; no body download)."""
        self._connect()
        collection = self._datastore.get_collection(self.cfg.collection_id)
        now = dt.datetime.now(dt.timezone.utc)
        products = list(
            collection.search(dtstart=now - dt.timedelta(minutes=30), dtend=now)
        )
        if not products:
            log.info("No MTG FCI products in the last 30 min")
            return None
        # Newest first, by sensing time when available (str fallback).
        epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        products.sort(key=lambda p: (product_sensing_time(p) or epoch, str(p)), reverse=True)
        return products[0]

    @staticmethod
    def product_entries(product) -> list[str]:
        """List the chunk entry filenames inside a product (no body download)."""
        return [str(e) for e in (getattr(product, "entries", None) or [])]

    def download_entries(self, product, entries: Iterable[str], dest_dir: Path = MTG_DIR) -> list[Path]:
        """Download only the named chunk entries. Call after dedup + selection."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_paths: list[Path] = []
        for name in entries:
            out = dest_dir / Path(name).name
            with product.open(entry=name) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 16):
                    dst.write(chunk)
            out_paths.append(out)
        log.info("Downloaded %d MTG FCI chunk(s) -> %s", len(out_paths), dest_dir)
        return out_paths


def product_sensing_time(product) -> Optional[dt.datetime]:
    """Best-effort UTC sensing-start of an eumdac product, without downloading."""
    val = getattr(product, "sensing_start", None)
    if isinstance(val, dt.datetime):
        return val if val.tzinfo else val.replace(tzinfo=dt.timezone.utc)
    return read_fci_valid_time(str(product))


def read_fci_valid_time(name) -> Optional[dt.datetime]:
    """Parse the sensing-start time from an FCI chunk filename or product id."""
    text = Path(str(name)).name
    match = _OPE_TIME_RE.search(text) or _ANY_TIME_RE.search(text)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError:
        return None


def chunk_number(name) -> Optional[int]:
    """Return the trailing chunk number of an FCI L1c entry, or None."""
    match = _CHUNK_RE.search(Path(str(name)).name)
    return int(match.group(1)) if match else None


def is_trailer(name) -> bool:
    """The per-cycle TRAIL chunk carries metadata satpy needs; always keep it."""
    return "-TRAIL-" in Path(str(name)).name.upper()


def parse_chunk_spec(spec: Optional[str]) -> Optional[set[int]]:
    """Parse ``"29-40,55"`` into ``{29..40, 55}``. Empty/None -> None (use fraction)."""
    if not spec:
        return None
    wanted: set[int] = set()
    for token in str(spec).replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            lo, _, hi = token.partition("-")
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(token))
    return wanted or None


def select_europe_chunks(
    entries: Iterable[str],
    *,
    fraction: float = 0.35,
    explicit: Optional[set[int]] = None,
) -> list[str]:
    """Pick the FCI BODY chunk entries covering the northern (Europe) band.

    Chunks are stacked south->north, so the Europe band is the highest-numbered
    body chunks. We keep the top ``fraction`` of body chunks by number (or an
    ``explicit`` chunk-number set). The per-cycle TRAIL chunk is excluded: the
    satpy ``fci_l1c_nc`` reader cannot open it ("Don't know how to open" warning)
    and only the BODY chunks carry image data, so downloading/passing it is pure
    waste. Unknown layouts (no parseable chunk numbers) fall back to every ``.nc``
    entry so we never silently render an empty/partial frame.
    """
    ncs = [e for e in entries if str(e).lower().endswith(".nc")]
    body = [e for e in ncs if not is_trailer(e) and chunk_number(e) is not None]
    if not body:
        return sorted(ncs)

    if explicit:
        wanted = explicit
    else:
        frac = min(max(fraction, 0.0), 1.0)
        numbers = sorted({chunk_number(e) for e in body})
        keep_count = max(1, round(len(numbers) * frac))
        wanted = set(numbers[-keep_count:])  # top (northern) chunks

    return sorted(e for e in body if chunk_number(e) in wanted)
