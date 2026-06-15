"""Unit tests for OPERA composite S3 key parsing/selection.

Pure-Python helpers (no network, no numpy/h5py/boto3), so these run anywhere.
"""

from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.radar_ord import (  # noqa: E402
    newest_odim_href,
    odim_hrefs_from_coveragejson,
    parse_composite_key,
    select_newest_composite_key,
)

PREFIX = "2026/06/03/OPERA/COMP/"

# A trimmed CoverageJSON like the live OPERA locations response: the ODIM file
# is a `.h5` link alongside docs/license links we must ignore.
_COVERAGE_JSON = {
    "type": "Coverage",
    "links": [
        {"href": "https://api.meteogate.eu/eu-eumetnet-weather-radar/docs", "rel": "service-doc"},
        {
            "href": "https://s3.waw3-1.cloudferro.com/openradar-24h/2026/06/15/OPERA/COMP/OPERA@20260615T1045@0@DBZH.h5",
            "rel": "data",
        },
        {
            "href": "https://s3.waw3-1.cloudferro.com/openradar-24h/2026/06/15/OPERA/COMP/OPERA@20260615T1050@0@DBZH.h5",
            "rel": "data",
        },
        {"href": "https://www.eumetnet.eu/observations/weather-radar-network/", "rel": "license"},
    ],
}


def test_odim_hrefs_ignores_non_data_links():
    hrefs = odim_hrefs_from_coveragejson(_COVERAGE_JSON)
    assert all(h.endswith(".h5") for h in hrefs)
    assert len(hrefs) == 2


def test_newest_odim_href_picks_latest_timestamp():
    href = newest_odim_href(odim_hrefs_from_coveragejson(_COVERAGE_JSON))
    assert href.endswith("OPERA@20260615T1050@0@DBZH.h5")


def test_newest_odim_href_empty():
    assert newest_odim_href([]) is None
    assert odim_hrefs_from_coveragejson({"links": []}) == []


def test_parse_valid_key():
    p = parse_composite_key(PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5")
    assert p is not None
    assert p["timestamp"] == "20260603T123000Z"
    assert p["elevation"] == "0"
    assert p["standard_name"] == "DBZH"
    assert p["name"] == "OPERA@20260603T123000Z@0@DBZH.h5"


def test_parse_bare_filename_without_prefix():
    p = parse_composite_key("OPERA@20260603T120500Z@0@RATE.h5")
    assert p is not None and p["standard_name"] == "RATE"


def test_parse_rejects_malformed():
    assert parse_composite_key(PREFIX + "not-a-composite.h5") is None
    assert parse_composite_key(PREFIX + "OPERA@ts@0@DBZH.txt") is None  # wrong ext
    assert parse_composite_key(PREFIX + "OPERA@ts@DBZH.h5") is None      # too few parts
    assert parse_composite_key("") is None


def test_select_newest_picks_greatest_timestamp():
    keys = [
        PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5",
        PREFIX + "OPERA@20260603T124500Z@0@DBZH.h5",  # newest
        PREFIX + "OPERA@20260603T120000Z@0@DBZH.h5",
    ]
    newest = select_newest_composite_key(keys, "DBZH")
    assert newest == PREFIX + "OPERA@20260603T124500Z@0@DBZH.h5"


def test_select_filters_by_standard_name():
    keys = [
        PREFIX + "OPERA@20260603T124500Z@0@RATE.h5",   # newer, but wrong product
        PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5",   # the only DBZH
    ]
    assert select_newest_composite_key(keys, "DBZH") == (
        PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5"
    )


def test_select_ignores_malformed_keys():
    keys = [
        PREFIX + "junk.h5",
        PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5",
        "index.html",
    ]
    assert select_newest_composite_key(keys, "DBZH") == (
        PREFIX + "OPERA@20260603T123000Z@0@DBZH.h5"
    )


def test_select_returns_none_when_empty_or_no_match():
    assert select_newest_composite_key([], "DBZH") is None
    assert select_newest_composite_key(
        [PREFIX + "OPERA@20260603T123000Z@0@RATE.h5"], "DBZH"
    ) is None
