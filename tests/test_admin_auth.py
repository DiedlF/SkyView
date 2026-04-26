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
