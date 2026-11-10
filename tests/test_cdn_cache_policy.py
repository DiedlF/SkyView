"""Cache/proxy policy behind the IPv64 CDN64 edge (which honours origin headers)."""

from __future__ import annotations

import os
import sys

import pytest

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from response_headers import (  # noqa: E402
    TILE_CACHE_DEFAULT,
    TILE_CACHE_VERSIONED,
    build_tile_headers,
    tile_cache_control_for,
)
from static_cache import (  # noqa: E402
    STATIC_CACHE_DEFAULT,
    STATIC_CACHE_HTML,
    STATIC_CACHE_VERSIONED,
    CachedStaticFiles,
    static_cache_control,
)
from time_contract import timeline_version  # noqa: E402
from trusted_proxies import EDGE_IPS_FILE, load_trusted_proxies, read_address_file  # noqa: E402
from usage_stats import is_automated_agent  # noqa: E402


@pytest.mark.parametrize(
    "path,query,expected",
    [
        # Versioned assets never change under the same URL.
        ("app.js", b"v=73", STATIC_CACHE_VERSIONED),
        ("style.css", "v=38", STATIC_CACHE_VERSIONED),
        # Without a version the URL can change content on the next deploy.
        ("app.js", b"", STATIC_CACHE_DEFAULT),
        ("app.js", b"v=", STATIC_CACHE_DEFAULT),
        ("geodata/ww-symbols/wmo4677_ww00.svg", b"", STATIC_CACHE_DEFAULT),
        # HTML carries the version numbers, so it must always be revalidated -
        # also when it is reached as a directory or with a stray ?v=.
        ("index.html", b"", STATIC_CACHE_HTML),
        ("index.html", b"v=1", STATIC_CACHE_HTML),
        (".", b"", STATIC_CACHE_HTML),
        ("", b"", STATIC_CACHE_HTML),
        ("de", b"", STATIC_CACHE_HTML),
        ("/srv/frontend/de/privacy.html", b"", STATIC_CACHE_HTML),
    ],
)
def test_static_cache_control(path, query, expected):
    assert static_cache_control(path, query) == expected


def test_static_files_send_cache_headers(tmp_path):
    import asyncio

    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "app.js").write_text("console.log(1)")
    static = CachedStaticFiles(directory=tmp_path, html=True)

    def get(path, query=b"", headers=()):
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/" if path == "." else f"/{path}",
            "headers": list(headers),
            "query_string": query,
        }
        return asyncio.run(static.get_response(path, scope))

    assert get(".").headers["cache-control"] == STATIC_CACHE_HTML
    versioned = get("app.js", b"v=73")
    assert versioned.status_code == 200
    assert versioned.headers["cache-control"] == STATIC_CACHE_VERSIONED
    assert get("app.js").headers["cache-control"] == STATIC_CACHE_DEFAULT

    # Revalidation answers must carry the policy too, or a 304 resets it.
    etag = versioned.headers["etag"].encode()
    not_modified = get("app.js", b"v=73", headers=[(b"if-none-match", etag)])
    assert not_modified.status_code == 304
    assert not_modified.headers["cache-control"] == STATIC_CACHE_VERSIONED

    # A missing file must not come back as a cacheable asset.
    from starlette.exceptions import HTTPException

    with pytest.raises(HTTPException) as exc:
        get("nope.js", b"v=1")
    assert exc.value.status_code == 404


def _run(model, run, steps):
    return {"model": model, "run": run, "steps": [{"step": s} for s in steps]}


def test_timeline_version_changes_with_every_ingested_step():
    base = [_run("icon_d2", "2026111012", [1, 2, 3]), _run("icon_eu", "2026111009", [1, 2])]
    token = timeline_version(base)
    assert token == timeline_version([dict(r) for r in base])
    assert len(token) == 10

    one_more_d2_step = [_run("icon_d2", "2026111012", [1, 2, 3, 4]), base[1]]
    one_more_eu_step = [base[0], _run("icon_eu", "2026111009", [1, 2, 3])]
    new_run = [_run("icon_d2", "2026111015", [1, 2, 3]), base[1]]
    tokens = {token, *(timeline_version(r) for r in (one_more_d2_step, one_more_eu_step, new_run))}
    assert len(tokens) == 4
    assert timeline_version([]) == timeline_version(None)


def test_trusted_proxies_default_to_the_edge_list_file():
    edge = read_address_file(EDGE_IPS_FILE)
    assert "91.99.98.96" in edge and "2a01:4f8:1c18:5887::1" in edge
    assert all("#" not in value and " " not in value for value in edge)

    trusted = load_trusted_proxies(env={})
    assert trusted[:2] == ["127.0.0.1", "::1"]
    assert set(edge) <= set(trusted)
    assert len(trusted) == len(set(trusted))


def test_trusted_proxies_env_override_and_missing_file(tmp_path):
    assert load_trusted_proxies(env={"SKYVIEW_TRUSTED_PROXIES": "10.0.0.1, 10.0.0.0/8,"}) == [
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "10.0.0.0/8",
    ]
    # Explicitly empty: trust nobody beyond localhost, do not fall back to the file.
    assert load_trusted_proxies(env={"SKYVIEW_TRUSTED_PROXIES": ""}) == ["127.0.0.1", "::1"]
    assert load_trusted_proxies(env={}, path=tmp_path / "absent.txt") == ["127.0.0.1", "::1"]


def test_forwarded_client_ip_only_trusted_from_edge():
    """request.client must be the visitor for edge peers and the peer otherwise."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    import asyncio

    seen = {}

    async def app(scope, receive, send):
        seen["client"] = scope["client"][0]
        seen["scheme"] = scope["scheme"]

    wrapped = ProxyHeadersMiddleware(app, trusted_hosts=load_trusted_proxies(env={}))
    headers = [(b"x-forwarded-for", b"203.0.113.9"), (b"x-forwarded-proto", b"https")]

    def call(peer):
        scope = {"type": "http", "scheme": "http", "client": (peer, 1234), "headers": headers}
        asyncio.run(wrapped(scope, None, None))
        return seen["client"], seen["scheme"]

    assert call("91.99.98.96") == ("203.0.113.9", "https")
    # A scanner hitting the origin directly cannot pose as someone else.
    assert call("198.51.100.7") == ("198.51.100.7", "http")


@pytest.mark.parametrize(
    "ua,automated",
    [
        ("node64", True),
        ("", True),
        ("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", True),
        ("curl/8.5.0", True),
        ("python-requests/2.32", True),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36", False),
        ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) Version/18.1 Mobile Safari/604.1", False),
    ],
)
def test_is_automated_agent(ua, automated):
    assert is_automated_agent(ua) is automated


_VERSIONED_TILE = dict(
    version="abc1234567",
    current_version="abc1234567",
    time_str="2026-11-11T10:00:00Z",
    model="icon_d2",
    substep_minutes=0,
    eu_fill_missing=False,
)


def test_tile_gets_long_ttl_only_when_url_pins_the_content():
    assert tile_cache_control_for(**_VERSIONED_TILE) == TILE_CACHE_VERSIONED


@pytest.mark.parametrize(
    "override",
    [
        {"version": None},  # old client without a token
        {"version": ""},
        {"version": "0ld0ld0ld0"},  # token from before the last ingested step
        {"model": None},  # D2/EU choice then depends on wall-clock time
        {"time_str": "latest"},
        {"substep_minutes": 15},  # 15-min data may land after the hourly step
        {"eu_fill_missing": True},  # border tile rendered with a hole
    ],
)
def test_tile_keeps_short_ttl_when_content_can_still_change(override):
    assert tile_cache_control_for(**{**_VERSIONED_TILE, **override}) == TILE_CACHE_DEFAULT


def test_tile_headers_take_cache_control_override_without_exposing_it():
    headers = build_tile_headers(
        run="r", valid_time="t", model="m", cache="MISS", extra={"Cache-Control": "no-store", "X-Overlay-Status": "empty:404"}
    )
    assert headers["Cache-Control"] == "no-store"
    exposed = headers["Access-Control-Expose-Headers"]
    assert "X-Overlay-Status" in exposed and "Cache-Control" not in exposed
