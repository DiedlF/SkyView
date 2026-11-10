"""Reverse proxies whose X-Forwarded-* headers Skyview trusts.

The public site is served through the IPv64 CDN64 edge servers, so the TCP peer
of every request is an edge server, not the visitor. Uvicorn rewrites
``request.client`` / ``request.url.scheme`` from ``X-Forwarded-For`` /
``X-Forwarded-Proto`` only for peers listed here; anyone else's forwarded
headers are ignored, so they cannot be used to spoof a client address.

Override with ``SKYVIEW_TRUSTED_PROXIES`` (comma-separated addresses or CIDR
networks; empty string = trust nobody but localhost).
"""

from __future__ import annotations

import os
from pathlib import Path

EDGE_IPS_FILE = Path(__file__).resolve().parent.parent / "deploy" / "cdn64-edge-ips.txt"
_ALWAYS_TRUSTED = ("127.0.0.1", "::1")


def read_address_file(path: Path) -> list[str]:
    """Read one address per line, ignoring blank lines and ``#`` comments."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        value = line.split("#", 1)[0].strip()
        if value:
            out.append(value)
    return out


def load_trusted_proxies(env: dict | None = None, path: Path = EDGE_IPS_FILE) -> list[str]:
    """Return the trusted proxy addresses (localhost first, no duplicates)."""
    env = os.environ if env is None else env
    raw = env.get("SKYVIEW_TRUSTED_PROXIES")
    if raw is not None:
        configured = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        configured = read_address_file(path)
    out: list[str] = []
    for value in (*_ALWAYS_TRUSTED, *configured):
        if value not in out:
            out.append(value)
    return out
