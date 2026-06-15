#!/usr/bin/env python3
"""Probe the MeteoGate OPERA EDR endpoint and dump its response shape.

Diagnostic for the radar observation source: it queries the OGC-EDR
``/collections/observations/locations/{id}`` endpoint (the correct pattern for
the OPERA composite — the old ``/items?id=`` form returns HTTP 422) using the
``EUCOMP_RADAR_API_KEY`` from ``backend/.env``, then prints the status and enough
of the response structure to write the download parser:

  * HTTP status + Content-Type
  * top-level JSON keys (and CoverageJSON markers if present)
  * any candidate ODIM download hrefs (strings ending .h5/.hdf/.nc or http links)

It tries a few ``f`` output formats so one run reveals the working combination.

Usage:
  python3 scripts/radar_edr_probe.py
  PYTHONPATH=backend .venv-obs/bin/python scripts/radar_edr_probe.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import requests  # noqa: E402

from observations.config import RadarConfig  # noqa: E402


def _find_links(obj, out, depth=0):
    """Recursively collect strings that look like download hrefs / file links."""
    if depth > 8:
        return
    if isinstance(obj, str):
        low = obj.lower()
        if low.endswith((".h5", ".hdf", ".hdf5", ".nc")) or (
            obj.startswith("http") and ("download" in low or low.endswith(".h5"))
        ):
            out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("href", "data", "value", "url") and isinstance(v, str) and v.startswith("http"):
                out.add(f"{k}={v}")
            _find_links(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:20]:
            _find_links(v, out, depth + 1)


def main() -> int:
    cfg = RadarConfig()
    sess = requests.Session()
    if cfg.api_key:
        sess.headers[cfg.api_key_header] = cfg.api_key
        print(f"Using API key header '{cfg.api_key_header}' (key present, {len(cfg.api_key)} chars)")
    else:
        print("WARNING: no EUCOMP_RADAR_API_KEY set — expect 429 rate-limiting")

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(minutes=20)
    datetime_range = f"{start:%Y-%m-%dT%H:%MZ}/{now:%Y-%m-%dT%H:%MZ}"
    base = f"{cfg.edr_base}/collections/observations/locations/{cfg.location_id}"

    for fmt in ("CoverageJSON", "GeoJSON", "json"):
        params = {
            "standard_name": cfg.standard_name,
            "method": cfg.method,
            "format": cfg.odim_format,   # ODIM (the data file format we want)
            "datetime": datetime_range,
            "f": fmt,
        }
        print("\n" + "=" * 72)
        print(f"GET {base}")
        print(f"    params: {params}")
        try:
            r = sess.get(base, params=params, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"    request error: {type(exc).__name__}: {exc}")
            continue
        ct = r.headers.get("Content-Type", "")
        print(f"--> HTTP {r.status_code} | Content-Type: {ct} | {len(r.content)} bytes")
        if r.status_code != 200:
            print(f"    body (first 400 chars): {r.text[:400]}")
            continue
        if "json" not in ct.lower():
            print("    (non-JSON body — likely the ODIM file delivered directly)")
            continue
        try:
            data = r.json()
        except ValueError:
            print("    (could not parse JSON)")
            continue
        if isinstance(data, dict):
            print(f"    top-level keys: {list(data.keys())}")
            if data.get("type"):
                print(f"    type: {data.get('type')}")
            if "features" in data:
                print(f"    features: {len(data.get('features') or [])}")
            for marker in ("coverages", "ranges", "parameters", "domain"):
                if marker in data:
                    print(f"    CoverageJSON marker present: {marker}")
        links = set()
        _find_links(data, links)
        if links:
            print("    candidate download hrefs:")
            for ln in sorted(links)[:10]:
                print(f"      - {ln}")
        else:
            print("    no obvious .h5/href links found — paste the structure below")
    print("\nDone. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
