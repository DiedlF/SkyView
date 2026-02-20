"""Marker authentication — HMAC-based token issuance and verification.

Custom token scheme: ``{payload_b64}.{sig_b64}``
where payload = JSON ``{"cid": client_id, "exp": unix_ts}``.

This is a simplified JWT-alike; migrate to python-jose if requirements grow.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration (read once at module import; tests may monkey-patch SECRET)
# ---------------------------------------------------------------------------

SECRET: str = os.environ.get("SKYVIEW_MARKER_AUTH_SECRET", "")
TOKEN_TTL_SECONDS: int = 12 * 3600

_WEAK_SECRETS = {"", "dev-marker-secret-change-me"}


def is_configured(secret: Optional[str] = None) -> bool:
    """Return True when a strong secret is present and auth is usable."""
    s = secret if secret is not None else SECRET
    return bool(s and len(s) >= 16 and s not in _WEAK_SECRETS)


def startup_check(secret: Optional[str] = None) -> Optional[str]:
    """Return a warning string if secret is missing/weak, else None."""
    s = secret if secret is not None else SECRET
    if not s:
        return "SKYVIEW_MARKER_AUTH_SECRET is not set — marker editing disabled."
    if s in _WEAK_SECRETS or len(s) < 16:
        return (
            f"SKYVIEW_MARKER_AUTH_SECRET is too weak (len={len(s)}) — "
            "marker editing disabled. Set a strong secret (>= 16 chars)."
        )
    return None


# ---------------------------------------------------------------------------
# Core signing helpers
# ---------------------------------------------------------------------------

def _sign(payload_json: str, secret: str) -> str:
    """Return URL-safe base64 HMAC-SHA256 signature (no padding)."""
    sig = hmac.new(secret.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_token(client_id: str, secret: Optional[str] = None, ttl: Optional[int] = None) -> Dict[str, Any]:
    """Issue a new marker token for *client_id*.

    Returns ``{"token": "...", "expiresAt": "<iso8601>"}`` on success.
    Raises ``RuntimeError`` if secret is not configured.
    """
    s = secret if secret is not None else SECRET
    if not is_configured(s):
        raise RuntimeError("Marker auth secret not configured or too weak.")
    ttl = ttl if ttl is not None else TOKEN_TTL_SECONDS
    exp = int(time.time()) + ttl
    payload = {"cid": client_id, "exp": exp}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8").rstrip("=")
    sig_b64 = _sign(payload_json, s)
    return {
        "token": f"{payload_b64}.{sig_b64}",
        "expiresAt": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def verify_token(client_id: str, token: str, secret: Optional[str] = None) -> bool:
    """Return True iff *token* is a valid, unexpired token for *client_id*.

    Never raises; returns False on any parse/crypto error.
    """
    s = secret if secret is not None else SECRET
    if not is_configured(s):
        return False
    try:
        if not token or "." not in token:
            return False
        payload_b64, sig_b64 = token.split(".", 1)
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode((payload_b64 + padding).encode("utf-8")).decode("utf-8")
        expected_sig = _sign(payload_json, s)
        if not hmac.compare_digest(expected_sig, sig_b64):
            return False
        payload = json.loads(payload_json)
        if payload.get("cid") != client_id:
            return False
        if int(payload.get("exp", 0)) < int(time.time()):
            return False
        return True
    except Exception:
        return False
