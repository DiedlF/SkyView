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
