from __future__ import annotations

import os
import sys

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from admin_auth import require_admin  # noqa: E402


def test_require_admin_accepts_configured_credentials(monkeypatch):
    monkeypatch.setenv("SKYVIEW_ADMIN_USER", "admin")
    monkeypatch.setenv("SKYVIEW_ADMIN_PASSWORD", "secret")

    user = require_admin(HTTPBasicCredentials(username="admin", password="secret"))

    assert user == "admin"


def test_require_admin_rejects_bad_credentials(monkeypatch):
    monkeypatch.setenv("SKYVIEW_ADMIN_USER", "admin")
    monkeypatch.setenv("SKYVIEW_ADMIN_PASSWORD", "secret")

    with pytest.raises(HTTPException) as exc:
        require_admin(HTTPBasicCredentials(username="admin", password="wrong"))

    assert exc.value.status_code == 401


def test_require_admin_fails_closed_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SKYVIEW_ADMIN_USER", raising=False)
    monkeypatch.delenv("SKYVIEW_ADMIN_PASSWORD", raising=False)

    with pytest.raises(HTTPException) as exc:
        require_admin(HTTPBasicCredentials(username="admin", password="secret"))

    assert exc.value.status_code == 503


def test_http_exception_handler_preserves_basic_auth_challenge():
    import asyncio
    import app as skyview_app
    from fastapi import HTTPException

    class _State:
        request_id = "req-test"

    class _Request:
        state = _State()

    exc = HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Basic"})
    response = asyncio.run(skyview_app.http_exception_handler(_Request(), exc))

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"
    assert response.headers["x-request-id"] == "req-test"
