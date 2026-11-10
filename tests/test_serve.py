from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from serve import Listener, build_servers, listeners_from_env  # noqa: E402


def _cert_pair(tmp_path):
    cert, key = tmp_path / "origin.crt", tmp_path / "origin.key"
    cert.write_text("cert")
    key.write_text("key")
    return str(cert), str(key)


def test_default_is_plain_http_on_8501():
    assert listeners_from_env({}) == [Listener("0.0.0.0", 8501)]


def test_https_listener_added_when_cert_and_key_exist(tmp_path):
    cert, key = _cert_pair(tmp_path)
    got = listeners_from_env(
        {"SKYVIEW_HTTP_HOST": "127.0.0.1", "SKYVIEW_TLS_CERT": cert, "SKYVIEW_TLS_KEY": key}
    )
    assert got == [Listener("127.0.0.1", 8501), Listener("0.0.0.0", 8443, cert, key)]
    assert got[1].tls and got[1].describe() == "https://0.0.0.0:8443"


def test_missing_certificate_never_takes_the_site_down(tmp_path):
    cert, _key = _cert_pair(tmp_path)
    for env in (
        {"SKYVIEW_TLS_CERT": cert},  # key not configured
        {"SKYVIEW_TLS_CERT": cert, "SKYVIEW_TLS_KEY": str(tmp_path / "absent.key")},
        {"SKYVIEW_TLS_KEY": cert},
    ):
        assert listeners_from_env(env) == [Listener("0.0.0.0", 8501)]


def test_bad_ports_fall_back_and_port_clash_drops_https(tmp_path):
    cert, key = _cert_pair(tmp_path)
    got = listeners_from_env(
        {"SKYVIEW_HTTP_PORT": "http", "SKYVIEW_TLS_PORT": "70000", "SKYVIEW_TLS_CERT": cert, "SKYVIEW_TLS_KEY": key}
    )
    assert [lst.port for lst in got] == [8501, 8443]
    clash = listeners_from_env(
        {"SKYVIEW_TLS_PORT": "8501", "SKYVIEW_TLS_CERT": cert, "SKYVIEW_TLS_KEY": key}
    )
    assert clash == [Listener("0.0.0.0", 8501)]


def test_only_first_listener_runs_the_app_lifespan(tmp_path):
    cert, key = _cert_pair(tmp_path)
    servers = build_servers(
        object(), [Listener("127.0.0.1", 8501), Listener("127.0.0.1", 8443, cert, key)], proxy_headers=True
    )
    assert [s.config.lifespan for s in servers] == ["auto", "off"]
    assert servers[1].config.ssl_certfile == cert and servers[0].config.ssl_certfile is None
    assert all(s.config.proxy_headers for s in servers)
