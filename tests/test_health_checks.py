from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from health_checks import (  # noqa: E402
    MODEL_MAX_AGE_HOURS,
    OBSERVATION_MAX_AGE_SECONDS,
    data_freshness,
)

NOW = dt.datetime(2026, 11, 10, 15, 0, tzinfo=dt.timezone.utc)


def _run(model: str, hours_old: float, steps: int = 3) -> dict:
    when = NOW - dt.timedelta(hours=hours_old)
    return {"model": model, "run": when.strftime("%Y%m%d%H"), "steps": [{"step": i} for i in range(steps)]}


def _obs(**minutes_old) -> dict:
    return {
        source: (NOW - dt.timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
        for source, m in minutes_old.items()
    }


HEALTHY_RUNS = [_run("icon_d2", 3), _run("icon_eu", 6)]
HEALTHY_OBS = _obs(radar=15, satellite=6, mtg=25, li=20)


def test_everything_current_is_ok():
    result = data_freshness(runs=HEALTHY_RUNS, latest_observation_times=HEALTHY_OBS, now=NOW)
    assert result["status"] == "ok" and result["stale"] == []
    assert result["checks"]["icon_d2"]["ageHours"] == 3.0
    assert result["checks"]["mtg"]["ageSeconds"] == 25 * 60


def test_stalled_observation_ingest_is_reported():
    # The 2026-07 failure: process alive, frames frozen for days.
    frozen = _obs(radar=6 * 24 * 60, satellite=6 * 24 * 60, mtg=6 * 24 * 60, li=6 * 24 * 60)
    result = data_freshness(runs=HEALTHY_RUNS, latest_observation_times=frozen, now=NOW)
    assert result["status"] == "stale"
    assert set(result["stale"]) == {"radar", "satellite", "mtg", "li"}


def test_single_stale_stream_names_only_that_stream():
    obs = {**HEALTHY_OBS, **_obs(radar=61)}
    result = data_freshness(runs=HEALTHY_RUNS, latest_observation_times=obs, now=NOW)
    assert result["stale"] == ["radar"]
    assert result["checks"]["satellite"]["status"] == "ok"


def test_limits_are_inclusive_at_the_boundary():
    at_limit = _obs(radar=OBSERVATION_MAX_AGE_SECONDS["radar"] / 60)
    result = data_freshness(
        runs=[_run("icon_d2", MODEL_MAX_AGE_HOURS["icon_d2"]), _run("icon_eu", 1)],
        latest_observation_times={**HEALTHY_OBS, **at_limit},
        now=NOW,
    )
    assert result["status"] == "ok"


def test_old_model_run_is_stale_and_newest_run_wins():
    runs = [_run("icon_d2", 20), _run("icon_d2", 12), _run("icon_eu", 6)]
    result = data_freshness(runs=runs, latest_observation_times=HEALTHY_OBS, now=NOW)
    assert result["stale"] == ["icon_d2"]
    assert result["checks"]["icon_d2"]["ageHours"] == 12.0

    # A newer run counts even if it is listed after an older one...
    runs = [_run("icon_d2", 20), _run("icon_d2", 2), _run("icon_eu", 6)]
    assert data_freshness(runs=runs, latest_observation_times=HEALTHY_OBS, now=NOW)["status"] == "ok"
    # ...but not while it has no ingested step yet.
    runs = [_run("icon_d2", 20), _run("icon_d2", 2, steps=0), _run("icon_eu", 6)]
    assert data_freshness(runs=runs, latest_observation_times=HEALTHY_OBS, now=NOW)["stale"] == ["icon_d2"]


def test_missing_model_is_stale_but_never_configured_source_is_not():
    result = data_freshness(
        runs=[_run("icon_d2", 3)],
        latest_observation_times={"radar": HEALTHY_OBS["radar"], "satellite": None},
        now=NOW,
    )
    assert result["stale"] == ["icon_eu"]
    assert result["checks"]["icon_eu"]["run"] is None
    for source in ("satellite", "mtg", "li"):
        assert result["checks"][source]["status"] == "absent"


def test_garbage_input_does_not_raise():
    result = data_freshness(
        runs=[{"model": "icon_d2", "run": "not-a-run", "steps": [1]}, {"model": "icon_eu"}],
        latest_observation_times={"radar": "yesterday-ish", "li": ""},
        now=NOW,
    )
    assert set(result["stale"]) == {"icon_d2", "icon_eu"}
    assert result["checks"]["radar"]["status"] == "absent"
