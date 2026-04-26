from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from airport_lookup import curated_matches, item_icao, normalize_icao_query  # noqa: E402


def test_normalize_icao_query_accepts_four_letters_case_insensitive():
    assert normalize_icao_query("eddm") == "EDDM"
    assert normalize_icao_query(" LOWI ") == "LOWI"


def test_normalize_icao_query_rejects_non_icao_text():
    assert normalize_icao_query("Munich") is None
    assert normalize_icao_query("EDDM airport") is None
    assert normalize_icao_query("ED1M") is None


def test_item_icao_extracts_from_seed_style_fields():
    assert item_icao({"name": "Unterwoessen (EDMU)", "displayName": "Unterwoessen Airfield EDMU, DE"}) == "EDMU"


def test_curated_matches_returns_known_airport():
    matches = curated_matches("EDDM")
    assert matches
    assert matches[0]["icao"] == "EDDM"
    assert "Munich" in matches[0]["name"]


def test_curated_matches_includes_burg_feuerstein():
    matches = curated_matches("EDQE")
    assert matches
    assert matches[0]["icao"] == "EDQE"
    assert "Burg Feuerstein" in matches[0]["name"]


def test_location_search_icao_payload_uses_local_fast_path(monkeypatch):
    import app as skyview_app

    class _Request:
        client = type("Client", (), {"host": "127.0.0.1"})()
        headers = {}

    monkeypatch.setattr(skyview_app.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Nominatim should not be called")))

    import asyncio
    payload = asyncio.run(skyview_app.api_location_search(_Request(), q="EDQE", limit=8))

    assert payload["source"] == "local_icao"
    assert payload["results"]
    assert payload["results"][0]["icao"] == "EDQE"
