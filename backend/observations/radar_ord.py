"""Radar source: EUMETNET OPERA CIRRUS max-reflectivity composite (5-min, 1 km).

Access (verified June 2026, post-MeteoGate onboarding -- no whitelisting):
  1. EDR API via the MeteoGate gateway -- openly available.
  2. Public S3 24h cache via unsigned requests -- openly available.

Both deliver ODIM HDF5. Reading uses h5py directly (wradlib/xradar are nicer if
already installed). Heavy/optional dependencies (h5py, numpy, boto3) are imported
lazily inside the functions that need them so this module imports cleanly even in
environments where only the serving deps are present.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import requests

from .config import RADAR_DIR, RadarConfig

log = logging.getLogger("skyview.observations.radar")


class RadarSource:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.session = requests.Session()
        # Anonymous access is open but rate-limited; an optional API key raises
        # the limit (see RadarConfig.api_key / devportal.meteogate.eu).
        if cfg.api_key:
            self.session.headers[cfg.api_key_header] = cfg.api_key

    # -- discovery --------------------------------------------------------
    def list_collections(self) -> list[dict]:
        """Confirm live collections from the EDR /collections endpoint."""
        url = f"{self.cfg.edr_base}/collections"
        r = self.session.get(url, params={"f": "json"}, timeout=30)
        r.raise_for_status()
        return r.json().get("collections", [])

    # -- fetch via EDR API ------------------------------------------------
    def fetch_latest_edr(self, dest_dir: Path = RADAR_DIR) -> Optional[Path]:
        """Fetch the most recent OPERA max-reflectivity composite via EDR.

        Uses the OGC-EDR ``locations/{id}`` query for the OPERA composite over a
        short trailing window (the ``items?id=`` form returns HTTP 422). The
        response is a CoverageJSON document whose ``links`` carry the ODIM
        ``.h5`` download href (which itself points at the open S3 cache); we pick
        the newest and download it. ``f=CoverageJSON`` is the only accepted
        output format for this endpoint.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(minutes=15)
        datetime_range = (
            f"{start.strftime('%Y-%m-%dT%H:%MZ')}/{now.strftime('%Y-%m-%dT%H:%MZ')}"
        )

        url = f"{self.cfg.edr_base}/collections/observations/locations/{self.cfg.location_id}"
        params = {
            "standard_name": self.cfg.standard_name,  # DBZH
            "method": self.cfg.method,                # comp
            "format": self.cfg.odim_format,           # ODIM (data file format)
            "datetime": datetime_range,
            "f": "CoverageJSON",                      # only accepted value here
        }
        r = self.session.get(url, params=params, timeout=60)
        if r.status_code == 204:
            log.info("EDR returned 204 (no data for window)")
            return None
        r.raise_for_status()

        href = newest_odim_href(odim_hrefs_from_coveragejson(r.json()))
        if not href:
            log.warning("EDR CoverageJSON had no ODIM (.h5) download link")
            return None

        out = dest_dir / Path(href).name
        return self._download(href, out)

    # -- fetch via unsigned S3 cache -------------------------------------
    def fetch_latest_s3(self, dest_dir: Path = RADAR_DIR) -> Optional[Path]:
        """Fetch newest OPERA composite directly from the open S3 24h cache.

        Path layout (verified):
            s3://openradar-24h/YYYY/MM/DD/OPERA/COMP/OPERA@<ts>@0@DBZH.h5
        Uses boto3 with unsigned config (equivalent to --no-sign-request).
        """
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config as BotoConfig

        dest_dir.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client(
            "s3",
            endpoint_url=self.cfg.s3_endpoint,
            config=BotoConfig(signature_version=UNSIGNED),
        )
        today = dt.datetime.now(dt.timezone.utc)
        prefix = f"{today:%Y/%m/%d}/OPERA/COMP/"
        resp = s3.list_objects_v2(Bucket=self.cfg.s3_bucket, Prefix=prefix)
        keys = [o["Key"] for o in resp.get("Contents", [])]

        newest = select_newest_composite_key(keys, self.cfg.standard_name)
        if not newest:
            log.info("No S3 composite keys under %s", prefix)
            return None

        out = dest_dir / Path(newest).name
        s3.download_file(self.cfg.s3_bucket, newest, str(out))
        log.info("Saved radar frame (S3) -> %s", out)
        return out

    def fetch_latest(self, dest_dir: Path = RADAR_DIR) -> Optional[Path]:
        """Try the EDR API first, fall back to the S3 cache on failure."""
        try:
            return self.fetch_latest_edr(dest_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("EDR fetch failed (%s); falling back to S3", exc)
            return self.fetch_latest_s3(dest_dir)

    def _download(self, href: str, out: Path) -> Path:
        with self.session.get(href, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        log.info("Saved radar frame (EDR) -> %s", out)
        return out


# -- ODIM HDF5 reading -----------------------------------------------------
def read_odim_maxreflectivity(path: Path):
    """Return (phys_dbz_array, where_attrs) for an ODIM HDF5 composite.

    ODIM stores the grid under ``/dataset1/data1/data`` with gain/offset/nodata
    in the corresponding ``/what`` attributes. nodata/undetect are masked to NaN.

    NOTE: the composite group layout is one of the open items to confirm against
    a real CIRRUS file in Phase 1 (single-site files may differ).
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        ds = f["dataset1"]["data1"]
        raw = ds["data"][:].astype("float32")
        what = ds["what"].attrs
        gain = float(what.get("gain", 1.0))
        offset = float(what.get("offset", 0.0))
        nodata = float(what.get("nodata", np.nan))
        undetect = float(what.get("undetect", np.nan))

        phys = raw * gain + offset
        phys[(raw == nodata) | (raw == undetect)] = np.nan
        where = dict(f["where"].attrs)
    return phys, where


def read_odim_valid_time(path: Path) -> Optional[dt.datetime]:
    """Return the composite's nominal valid time from ODIM ``/what`` attrs.

    ODIM stores ``date`` (YYYYMMDD) and ``time`` (HHMMSS) on the root ``/what``
    group (and/or per-dataset). Falls back to the file mtime if absent. Returns a
    timezone-aware UTC datetime, or ``None`` if nothing usable is found.
    """
    import h5py

    def _parse(date_v, time_v) -> Optional[dt.datetime]:
        if date_v is None or time_v is None:
            return None
        d = date_v.decode() if isinstance(date_v, bytes) else str(date_v)
        t = time_v.decode() if isinstance(time_v, bytes) else str(time_v)
        t = t.zfill(6)
        try:
            return dt.datetime.strptime(d + t[:6], "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            return None

    try:
        with h5py.File(path, "r") as f:
            for grp in ("what", "dataset1/what"):
                if grp in f:
                    a = f[grp].attrs
                    got = _parse(a.get("date"), a.get("time"))
                    if got:
                        return got
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read ODIM valid time from %s: %s", path, exc)

    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    except OSError:
        return None


# -- pure helpers (no network / heavy deps; unit-tested) -------------------
def parse_composite_key(key: str) -> Optional[dict]:
    """Parse an OPERA composite S3 key/filename into its components.

    Expected filename: ``OPERA@<timestamp>@<elevation>@<standard_name>.h5``
    (full key may be prefixed with ``YYYY/MM/DD/OPERA/COMP/``). Returns a dict
    with ``timestamp``, ``elevation``, ``standard_name``, ``name`` and ``key``,
    or ``None`` if the name does not match the expected shape.
    """
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(".h5"):
        return None
    stem = name[:-3]
    parts = stem.split("@")
    if len(parts) != 4 or parts[0] != "OPERA":
        return None
    return {
        "timestamp": parts[1],
        "elevation": parts[2],
        "standard_name": parts[3],
        "name": name,
        "key": key,
    }


def select_newest_composite_key(keys, standard_name: str) -> Optional[str]:
    """Return the key of the newest composite for ``standard_name``.

    Filters to well-formed keys matching the requested standard name and returns
    the one with the lexicographically greatest timestamp (ISO/compact timestamps
    sort chronologically). Malformed or non-matching keys are ignored.
    """
    parsed = []
    for k in keys:
        p = parse_composite_key(k)
        if p and p["standard_name"] == standard_name:
            parsed.append(p)
    if not parsed:
        return None
    parsed.sort(key=lambda p: p["timestamp"])
    return parsed[-1]["key"]


def odim_hrefs_from_coveragejson(doc: dict) -> list[str]:
    """Collect the ODIM ``.h5`` download hrefs from a CoverageJSON ``links`` list.

    The OPERA composite ``locations`` response carries the data file(s) as
    ``links`` entries whose href ends in ``.h5`` (alongside docs/license links we
    ignore). The href points at the open S3 cache, so the download itself is an
    unauthenticated GET.
    """
    hrefs = []
    for link in doc.get("links") or []:
        href = link.get("href") if isinstance(link, dict) else None
        if href and href.lower().endswith(".h5"):
            hrefs.append(href)
    return hrefs


def newest_odim_href(hrefs: list[str]) -> Optional[str]:
    """Return the newest OPERA composite href by its embedded timestamp."""
    if not hrefs:
        return None
    parsed = [(parse_composite_key(h), h) for h in hrefs]
    dated = [(p["timestamp"], h) for p, h in parsed if p]
    if not dated:
        return hrefs[-1]  # unparseable names: fall back to last listed
    return max(dated, key=lambda th: th[0])[1]
