"""Shared EUMETSAT Data Store connection.

The satellite (MSG RSS), MTG (FCI), and LI sources all authenticate with the same
``EUMETSAT_CONSUMER_KEY`` / ``_SECRET``. Building a separate ``eumdac.AccessToken``
+ ``DataStore`` in each source means three token requests per ingest tick (every
2 min) even when nothing new is available. This module caches one ``DataStore``
per credential pair for the lifetime of the process, so a single tick shares one
token across all three EUMETSAT sources.

Ingest runs as a fresh short-lived process each tick, so the cache is naturally
per-run; ``reset_cache`` exists only as a test seam.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("skyview.observations.eumdac")

_CACHE: dict[tuple[str, str], object] = {}


def get_datastore(consumer_key: Optional[str], consumer_secret: Optional[str]):
    """Return a process-cached ``eumdac.DataStore`` for these credentials."""
    if not (consumer_key and consumer_secret):
        raise RuntimeError(
            "Missing EUMETSAT credentials. Set EUMETSAT_CONSUMER_KEY and "
            "EUMETSAT_CONSUMER_SECRET, or run `eumdac set-credentials`. "
            "Verify with: python3 scripts/eumetsat_auth.py"
        )
    key = (consumer_key, consumer_secret)
    datastore = _CACHE.get(key)
    if datastore is None:
        import eumdac

        token = eumdac.AccessToken((consumer_key, consumer_secret))
        datastore = eumdac.DataStore(token)
        _CACHE[key] = datastore
        log.info("Connected to EUMETSAT Data Store (shared token)")
    return datastore


def reset_cache() -> None:
    """Drop cached datastores. Test seam; not used by the ingest path."""
    _CACHE.clear()
