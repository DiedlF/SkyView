"""Tests for backend/marker_auth.py — token issuance, verification, edge cases."""

from __future__ import annotations


import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import marker_auth

SECRET = "test-secret-strong-enough-32chars"


def test_is_configured_strong_secret():
    assert marker_auth.is_configured(SECRET) is True


def test_is_configured_empty():
    assert marker_auth.is_configured("") is False


def test_is_configured_too_short():
    assert marker_auth.is_configured("short") is False


def test_is_configured_known_weak():
    assert marker_auth.is_configured("dev-marker-secret-change-me") is False


def test_startup_check_no_secret():
    warn = marker_auth.startup_check("")
    assert warn is not None
    assert "not set" in warn.lower() or "missing" in warn.lower() or "not" in warn.lower()


def test_startup_check_weak_secret():
    warn = marker_auth.startup_check("tooshort")
    assert warn is not None


def test_startup_check_good_secret():
    assert marker_auth.startup_check(SECRET) is None


def test_make_token_returns_expected_keys():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    assert "token" in tok
    assert "expiresAt" in tok
    assert "." in tok["token"]


def test_verify_token_valid():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    assert marker_auth.verify_token("client-abc", tok["token"], secret=SECRET) is True


def test_verify_token_wrong_client():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    assert marker_auth.verify_token("other-client", tok["token"], secret=SECRET) is False


def test_verify_token_tampered():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    tampered = tok["token"][:-3] + "xxx"
    assert marker_auth.verify_token("client-abc", tampered, secret=SECRET) is False


def test_verify_token_expired():
    tok = marker_auth.make_token("client-abc", secret=SECRET, ttl=-1)
    assert marker_auth.verify_token("client-abc", tok["token"], secret=SECRET) is False


def test_verify_token_empty():
    assert marker_auth.verify_token("client-abc", "", secret=SECRET) is False


def test_verify_token_garbage():
    assert marker_auth.verify_token("client-abc", "notavalidtoken", secret=SECRET) is False


def test_verify_token_wrong_secret():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    assert marker_auth.verify_token("client-abc", tok["token"], secret="different-secret-entirely") is False


def test_make_token_raises_without_secret():
    with pytest.raises(RuntimeError, match="not configured"):
        marker_auth.make_token("client-abc", secret="")


def test_verify_token_unconfigured_secret_returns_false():
    tok = marker_auth.make_token("client-abc", secret=SECRET)
    assert marker_auth.verify_token("client-abc", tok["token"], secret="") is False
