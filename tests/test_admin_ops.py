from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

import admin_ops  # noqa: E402
from health_checks import data_freshness  # noqa: E402

NOW = dt.datetime(2026, 11, 11, 9, 0, tzinfo=dt.timezone.utc)


def _iso(minutes_ago: float) -> str:
    return (NOW - dt.timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


def _healthy_freshness() -> dict:
    return data_freshness(
        runs=[
            {"model": "icon_d2", "run": "2026111106", "steps": [1]},
            {"model": "icon_eu", "run": "2026111100", "steps": [1]},
        ],
        latest_observation_times={"radar": _iso(10), "satellite": _iso(6), "mtg": _iso(25), "li": _iso(15)},
        now=NOW,
    )


def _rows(health: dict) -> dict:
    return {r["key"]: r for r in health["rows"]}


def test_metric_key_groups_by_route_template():
    assert admin_ops.metric_key("/api/overlay_tile/{z}/{x}/{y}.png", "/api/overlay_tile/8/1/2.png", 200) == "/api/overlay_tile/{z}/{x}/{y}.png"
    assert admin_ops.metric_key(None, "/", 200) == "/ (page)"
    assert admin_ops.metric_key(None, "/app.js", 200) == "(static files)"
    # Scanner probes must not create one key each.
    assert admin_ops.metric_key(None, "/wp-login.php", 404) == "(not found)"
    assert admin_ops.metric_key(None, "/xmlrpc.php", 404) == "(not found)"


def test_error_line_matches_levels_not_words():
    assert admin_ops.is_error_line("2026-11-11 08:00:00 | ERROR    | app | boom")
    assert admin_ops.is_error_line("Traceback (most recent call last):")
    assert admin_ops.is_error_line("OSError: [Errno 28] No space left on device")
    assert not admin_ops.is_error_line("2026-11-11 08:00:00 | INFO     | app | GET /api/x - 200 - errors=0")
    assert not admin_ops.is_error_line("2026-11-11 13:15:20,351 eumdac WARNING Trying again for GET https://x/failed")


def test_healthy_system_is_ok_even_without_traffic():
    """No tile samples and a 9-hour-old ICON-EU run are normal, not warnings."""
    health = admin_ops.build_health(
        freshness=_healthy_freshness(),
        disk={"usedPercent": 59.0, "freeBytes": 15 * 1024 ** 3},
        server_health={"memoryPercent": 19.0},
        pipelines=admin_ops.parse_process_table(""),
    )
    assert health["status"] == "ok"
    assert health["issues"] == []
    rows = _rows(health)
    assert rows["icon_eu"]["status"] == "ok"
    assert rows["icon_eu"]["limit"] == "max 15 h"
    assert rows["icon_eu"]["value"] == "run 2026111100, 9.0 h old"
    assert {"icon_d2", "icon_eu", "radar", "satellite", "mtg", "li", "disk", "memory", "server_errors"} <= set(rows)


def test_stale_stream_is_critical_like_the_external_monitor():
    freshness = data_freshness(
        runs=[
            {"model": "icon_d2", "run": "2026111106", "steps": [1]},
            {"model": "icon_eu", "run": "2026111100", "steps": [1]},
        ],
        latest_observation_times={"radar": _iso(61), "satellite": _iso(6), "mtg": _iso(25), "li": _iso(15)},
        now=NOW,
    )
    assert freshness["status"] == "stale"
    health = admin_ops.build_health(freshness=freshness)
    assert health["status"] == "crit"
    assert _rows(health)["radar"]["status"] == "crit"
    assert any(issue.startswith("Radar") for issue in health["issues"])


def test_absent_observation_source_does_not_fail():
    freshness = data_freshness(
        runs=[
            {"model": "icon_d2", "run": "2026111106", "steps": [1]},
            {"model": "icon_eu", "run": "2026111100", "steps": [1]},
        ],
        latest_observation_times={"radar": _iso(10)},
        now=NOW,
    )
    health = admin_ops.build_health(
        freshness=freshness, disk={"usedPercent": 10.0, "freeBytes": 1}, server_health={"memoryPercent": 10.0}
    )
    assert _rows(health)["mtg"]["status"] == "info"
    assert health["status"] == "ok"


def test_disk_and_memory_thresholds():
    def status(disk_pct, mem_pct):
        health = admin_ops.build_health(
            freshness=_healthy_freshness(),
            disk={"usedPercent": disk_pct, "freeBytes": 1024 ** 3},
            server_health={"memoryPercent": mem_pct},
        )
        rows = _rows(health)
        return rows["disk"]["status"], rows["memory"]["status"], health["status"]

    assert status(79.9, 50) == ("ok", "ok", "ok")
    assert status(80.0, 50) == ("warn", "ok", "warn")
    assert status(90.0, 50) == ("crit", "ok", "crit")
    assert status(50, 85.0) == ("ok", "warn", "warn")
    assert status(50, 95.0) == ("ok", "crit", "crit")


def test_server_errors_use_the_last_hour_not_the_total():
    old_only = admin_ops.build_health(
        freshness=_healthy_freshness(), disk={"usedPercent": 1.0, "freeBytes": 1}, server_health={"memoryPercent": 1.0},
        server_errors_last_hour=0, server_errors_since_start=40,
    )
    assert _rows(old_only)["server_errors"]["status"] == "ok"
    assert old_only["status"] == "ok"
    recent = admin_ops.build_health(freshness=_healthy_freshness(), server_errors_last_hour=3)
    assert _rows(recent)["server_errors"]["status"] == "warn"
    burst = admin_ops.build_health(freshness=_healthy_freshness(), server_errors_last_hour=20)
    assert _rows(burst)["server_errors"]["status"] == "crit"


def test_process_table_detects_a_hung_observation_ingest():
    ps = "\n".join([
        "  901       2 /bin/bash ./cron-ingest.sh",
        "  902       1 ../venv/bin/python ingest.py --model icon-eu --profile auto --check-only",
        "  950  518400 .venv-obs/bin/python -m backend.observations.ingest_obs --source all",
        "  999      10 /usr/sbin/sshd -D",
    ])
    pipelines = admin_ops.parse_process_table(ps)
    # The every-minute probe is not an ingest run.
    assert pipelines["forecast"]["running"] is False
    assert pipelines["observations"]["running"] is True
    assert pipelines["observations"]["longestSeconds"] == 518400

    health = admin_ops.build_health(freshness=_healthy_freshness(), pipelines=pipelines)
    rows = _rows(health)
    assert rows["observation_ingest"]["status"] == "crit"
    assert rows["forecast_ingest"]["value"] == "idle"
    assert health["status"] == "crit"


def test_process_table_reports_a_normal_forecast_ingest_as_ok():
    pipelines = admin_ops.parse_process_table("  77  600 ../venv/bin/python ingest.py --model icon-d2 --profile auto")
    assert pipelines["forecast"]["running"] is True
    row = _rows(admin_ops.build_health(freshness=_healthy_freshness(), pipelines=pipelines))["forecast_ingest"]
    assert row["status"] == "ok"
    assert row["value"] == "running for 10 min"


def test_slow_forecast_ingest_of_a_newer_run_warns():
    health = admin_ops.build_health(
        freshness=_healthy_freshness(),
        model_timings={"icon_d2": {"ingestRunningMinutes": 150.0}, "icon_eu": {"ingestRunningMinutes": 12.0}},
    )
    rows = _rows(health)
    assert rows["icon_d2"]["status"] == "warn"
    assert rows["icon_eu"]["status"] == "ok"
    assert "ingesting" in rows["icon_eu"]["detail"]


def test_observation_buffer_counts_gaps_and_failures():
    frames = [{"valid_time": _iso(m)} for m in (40, 35, 25, 20, 10)]  # 15 and 30 missing
    buffer = admin_ops.observation_buffer(
        {"cadence_seconds": 300, "frames": frames, "failed": {"202611110845": 2, "202611110830": 1}}, now=NOW
    )
    assert buffer["frameCount"] == 5
    assert buffer["expectedFrames"] == 7
    assert buffer["ageSeconds"] == 600
    assert buffer["failedFrames"] == 2
    assert buffer["failedAttempts"] == 3


def test_observation_buffer_empty():
    buffer = admin_ops.observation_buffer({"frames": []}, now=NOW)
    assert buffer["frameCount"] == 0
    assert buffer["latestValidTime"] is None
    assert buffer["expectedFrames"] is None


def test_disk_usage_and_reboot_flag(tmp_path):
    usage = admin_ops.disk_usage(str(tmp_path))
    assert usage["totalBytes"] > 0
    assert 0 <= usage["usedPercent"] <= 100
    assert admin_ops.disk_usage(str(tmp_path / "missing"))["usedPercent"] is None

    flag = tmp_path / "reboot-required"
    assert admin_ops.reboot_required(str(flag)) == {"required": False, "packages": []}
    flag.write_text("*** System restart required ***\n")
    (tmp_path / "reboot-required.pkgs").write_text("linux-image-6.8.0-150-generic\nlinux-base\n")
    assert admin_ops.reboot_required(str(flag)) == {
        "required": True,
        "packages": ["linux-base", "linux-image-6.8.0-150-generic"],
    }


def test_swap_usage_parses_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 4000000 kB\nSwapTotal: 2097148 kB\nSwapFree: 2096124 kB\n")
    swap = admin_ops.swap_usage(str(meminfo))
    assert swap == {"totalBytes": 2097148 * 1024, "usedBytes": 1024 * 1024}
    assert admin_ops.swap_usage(str(tmp_path / "none")) == {"totalBytes": None, "usedBytes": None}


def test_overview_endpoint_carries_everything_the_page_needs():
    """The admin page loads this one payload; it must work on a box without data."""
    import asyncio

    import app as skyview_app

    skyview_app._record_request_metric(
        admin_ops.metric_key("/api/overlay_tile/{z}/{x}/{y}.png", "/api/overlay_tile/8/1/2.png", 200), 200, 12.0
    )
    skyview_app._record_request_metric(admin_ops.metric_key(None, "/wp-login.php", 404), 404, 1.0)
    skyview_app._record_request_metric(admin_ops.metric_key(None, "/xmlrpc.php", 404), 404, 1.0)

    payload = asyncio.run(skyview_app.api_admin_overview_metrics())

    for key in ("health", "freshness", "process", "host", "observations", "storage", "status", "endpoints", "product"):
        assert key in payload, key
    assert payload["health"]["status"] in ("ok", "warn", "crit")
    for row in payload["health"]["rows"]:
        assert {"key", "group", "label", "status", "value", "limit", "detail"} <= set(row)
    # Same verdict source as the external monitor.
    assert set(payload["freshness"]["checks"]) >= {"icon_d2", "icon_eu", "radar", "satellite", "mtg", "li"}
    assert payload["process"]["startedAt"]
    assert {"disk", "swap", "reboot"} <= set(payload["host"])
    assert len(payload["product"]["usageDaily"]) == 30

    by_path = {e["path"]: e for e in payload["endpoints"]}
    assert by_path["/api/overlay_tile/{z}/{x}/{y}.png"]["requests"] >= 1
    assert by_path["(not found)"]["errors4xx"] >= 2
    assert not any("wp-login" in path or "/8/1/2" in path for path in by_path)


def test_log_sources_and_tail(tmp_path):
    import app as skyview_app

    assert skyview_app._admin_log_source("/var/log/skyview-observations.log") == "observations"
    assert skyview_app._admin_log_source("/opt/skyview/backend/logs/ingest.log") == "ingest"
    assert skyview_app._admin_log_source("/opt/skyview/backend/logs/skyview.log") == "backend"
    assert skyview_app._admin_log_source("/opt/skyview/backend/logs/server.out") == "stdout"

    log = tmp_path / "big.log"
    log.write_text("".join(f"line {i} " + "x" * 700 + "\n" for i in range(500)))
    tail = skyview_app._tail_lines(str(log), limit=100)
    assert len(tail) == 100
    assert tail[-1].startswith("line 499 ")
    assert all(line.startswith("line ") for line in tail)
    small = tmp_path / "small.log"
    small.write_text("a\nb\n")
    assert skyview_app._tail_lines(str(small), limit=100) == ["a", "b"]
    assert skyview_app._tail_lines(str(tmp_path / "missing.log")) == []
