"""Data-freshness health check for external monitoring.

``/api/health`` only says the process is up. The 2026-07 incident showed that is
not enough: ingest hung for six days while the app kept answering 200 and
serving old frames. This module answers "is the data still being updated?" so
an HTTP monitor (IPv64 Healthchecks, any uptime checker) can alert on it:
the endpoint returns 503 as soon as one data stream is older than its limit.

Limits are deliberately generous - several times the normal delivery delay -
because a false alarm at 3 am is worse than learning about a stall an hour in.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

# Maximum age of the newest model run (hours since the run's reference time).
# ICON-D2 runs every 3 h and is published ~1.5 h later, so its newest run is
# normally at most ~5 h old; ICON-EU is ingested from the 6-hourly cycle with
# ~2-3 h delay, so up to ~9 h is normal.
MODEL_MAX_AGE_HOURS = {"icon_d2": 9.0, "icon_eu": 15.0}

# Maximum age of the newest observation frame (seconds since its valid time).
# Normal ages: MSG satellite 5-10 min, radar 10-25 min, MTG/LI 15-30 min.
OBSERVATION_MAX_AGE_SECONDS = {"radar": 3600, "satellite": 3600, "mtg": 5400, "li": 5400}


def _parse_run(run: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.strptime(str(run), "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _parse_iso(value: str) -> Optional[dt.datetime]:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def data_freshness(
    *,
    runs: list,
    latest_observation_times: dict,
    now: Optional[dt.datetime] = None,
) -> dict:
    """Evaluate model and observation freshness.

    ``runs`` is ``get_available_runs()`` (newest first per model);
    ``latest_observation_times`` maps source -> ISO valid time of its newest
    frame, or ``None`` when the source has no frames at all.

    A model without any run is stale (the app cannot show a forecast). An
    observation source without any frame is reported as ``absent`` and does not
    fail the check: that is a source which was never set up (e.g. no EUMETSAT
    credentials), not one that stopped - a stopped source keeps its old frames,
    because the ring buffer is only pruned when a new frame is written.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    checks: dict = {}
    stale: list[str] = []

    for model, max_hours in MODEL_MAX_AGE_HOURS.items():
        newest = None
        for run in runs or []:
            if run.get("model") != model or not run.get("steps"):
                continue
            when = _parse_run(run.get("run"))
            if when is not None and (newest is None or when > newest[0]):
                newest = (when, run.get("run"))
        if newest is None:
            checks[model] = {"status": "stale", "run": None, "ageHours": None, "maxAgeHours": max_hours}
            stale.append(model)
            continue
        age_h = round((now - newest[0]).total_seconds() / 3600.0, 2)
        ok = age_h <= max_hours
        checks[model] = {
            "status": "ok" if ok else "stale",
            "run": newest[1],
            "ageHours": age_h,
            "maxAgeHours": max_hours,
        }
        if not ok:
            stale.append(model)

    for source, max_seconds in OBSERVATION_MAX_AGE_SECONDS.items():
        when = _parse_iso(latest_observation_times.get(source) or "")
        if when is None:
            checks[source] = {"status": "absent", "ageSeconds": None, "maxAgeSeconds": max_seconds}
            continue
        age_s = int((now - when).total_seconds())
        ok = age_s <= max_seconds
        checks[source] = {
            "status": "ok" if ok else "stale",
            "ageSeconds": age_s,
            "maxAgeSeconds": max_seconds,
        }
        if not ok:
            stale.append(source)

    return {
        "status": "stale" if stale else "ok",
        "stale": stale,
        "checks": checks,
        "checkedAt": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
