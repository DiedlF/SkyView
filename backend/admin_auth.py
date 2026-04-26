"""HTTP Basic authentication for private admin/ops endpoints."""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic(auto_error=True)


def _configured_credentials() -> tuple[str, str] | None:
    user = os.environ.get("SKYVIEW_ADMIN_USER", "").strip()
    password = os.environ.get("SKYVIEW_ADMIN_PASSWORD", "")
    if not user or not password:
        return None
    return user, password


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Require configured HTTP Basic credentials and return the admin user."""
    configured = _configured_credentials()
    if configured is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin auth is not configured",
            headers={"WWW-Authenticate": "Basic"},
        )

    expected_user, expected_password = configured
    user_ok = hmac.compare_digest(credentials.username.encode("utf-8"), expected_user.encode("utf-8"))
    password_ok = hmac.compare_digest(credentials.password.encode("utf-8"), expected_password.encode("utf-8"))
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return expected_user
