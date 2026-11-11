"""Facts and the health verdict for the admin page.

The admin page used to judge health in the browser with its own thresholds,
which disagreed with the external monitor (``/api/health/data``): it warned
about "stale" ICON-EU data that was perfectly normal, and about missing tile
telemetry after every restart. Everything that decides a state lives here now,
is computed server-side and reuses the monitor's limits, so the page and the
alert can never tell different stories.

Pure functions take plain values so they can be tested without the app.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
from typing import Any, Optional

# --- limits (the only place they are defined) ------------------------------
DISK_WARN_PERCENT = 80.0
DISK_CRIT_PERCENT = 90.0
MEMORY_WARN_PERCENT = 85.0
MEMORY_CRIT_PERCENT = 95.0
# A forecast ingest takes 15-25 minutes; the observation ingest is killed by
# its own 5-minute timeout. Anything far beyond that is a hung run - the cause
# of the 2026-07 outage.
FORECAST_INGEST_WARN_SECONDS = 2 * 3600
OBSERVATION_INGEST_CRIT_SECONDS = 15 * 60
SERVER_ERRORS_WARN_PER_HOUR = 1
SERVER_ERRORS_CRIT_PER_HOUR = 20

_SEVERITY = {"ok": 0, "info": 0, "warn": 1, "crit": 2}

# A log line counts as an error when its level says so or it carries one of a
# few unmistakable markers. Matching the bare word "error" also caught INFO
# lines such as "errors=0" and URLs.
_ERROR_LINE = re.compile(
    r"\b(ERROR|CRITICAL)\b|Traceback \(most recent call last\)|Segmentation fault|No space left on device|ABORT"
)


def is_error_line(line: str) -> bool:
    return bool(_ERROR_LINE.search(line or ""))


# --- request metrics --------------------------------------------------------
def metric_key(route_path: Optional[str], request_path: str, status_code: int) -> str:
    """Group a request under its route template instead of its literal URL.

    Keying by URL made every tile, frame and scanner probe its own row (the
    table was unreadable and the dict grew until the next restart). The route
    template gives one row per endpoint and a bounded number of keys.
    """
    if route_path:
        return route_path
    if request_path in ("/", "/index.html"):
        return "/ (page)"
    if status_code == 404:
        return "(not found)"
    return "(static files)"


# --- host facts --------------------------------------------------------------
def disk_usage(path: str) -> dict[str, Any]:
    """Usage of the filesystem holding ``path`` (the data directory)."""
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"path": path, "totalBytes": None, "usedBytes": None, "freeBytes": None, "usedPercent": None}
    # Percent as df shows it: used / (used + available), which accounts for
    # the blocks reserved for root.
    denominator = usage.used + usage.free
    return {
        "path": path,
        "totalBytes": int(usage.total),
        "usedBytes": int(usage.used),
        "freeBytes": int(usage.free),
        "usedPercent": round(usage.used / denominator * 100.0, 1) if denominator else None,
    }


def swap_usage(meminfo_path: str = "/proc/meminfo") -> dict[str, Any]:
    total = free = None
    try:
        with open(meminfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    used = (total - free) if (total is not None and free is not None) else None
    return {"totalBytes": total, "usedBytes": used}


def reboot_required(flag_path: str = "/var/run/reboot-required") -> dict[str, Any]:
    """Whether an installed update is waiting for a reboot (Debian/Ubuntu flag file)."""
    if not os.path.exists(flag_path):
        return {"required": False, "packages": []}
    packages: list[str] = []
    try:
        with open(flag_path + ".pkgs", "r", encoding="utf-8") as f:
            packages = sorted({ln.strip() for ln in f if ln.strip()})
    except OSError:
        pass
    return {"required": True, "packages": packages[:20]}


def deployed_commit(repo_dir: str) -> Optional[dict[str, Any]]:
    """Short hash, commit time and subject of the checked-out commit, if this is a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "log", "-1", "--format=%h%x09%cI%x09%s"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    parts = result.stdout.strip().split("\t", 2)
    if result.returncode != 0 or len(parts) != 3:
        return None
    return {"sha": parts[0], "committedAt": parts[1], "subject": parts[2][:160]}


# --- ingest processes --------------------------------------------------------
def classify_pipeline(command: str) -> Optional[str]:
    if "ingest_obs" in command or "cron-observations.sh" in command:
        return "observations"
    if "ingest.py" in command:
        return "forecast"
    return None


def parse_process_table(ps_output: str) -> dict[str, dict[str, Any]]:
    """Turn ``ps -eo pid=,etimes=,args=`` output into per-pipeline run state.

    Run time comes from the process itself rather than from a lock file's
    mtime: the lock files are touched by every cron tick, including the ticks
    that find the lock held, so their age says nothing about a hung run.
    """
    pipelines: dict[str, dict[str, Any]] = {
        name: {"running": False, "longestSeconds": None, "processes": []} for name in ("forecast", "observations")
    }
    for raw in (ps_output or "").splitlines():
        parts = raw.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        command = parts[2].strip()
        name = classify_pipeline(command)
        if name is None:
            continue
        elapsed = int(parts[1])
        entry = pipelines[name]
        entry["processes"].append({"pid": int(parts[0]), "elapsedSeconds": elapsed, "command": command[:300]})
        # The every-minute "is there a new run?" probe is not an ingest run.
        if "--check-only" not in command:
            entry["running"] = True
        if entry["longestSeconds"] is None or elapsed > entry["longestSeconds"]:
            entry["longestSeconds"] = elapsed
    return pipelines


def read_pipelines() -> dict[str, dict[str, Any]]:
    try:
        result = subprocess.run(["ps", "-eo", "pid=,etimes=,args="], capture_output=True, text=True, timeout=3)
        return parse_process_table(result.stdout if result.returncode == 0 else "")
    except (OSError, subprocess.SubprocessError):
        return parse_process_table("")


# --- observation buffer ------------------------------------------------------
def _parse_iso(value: Any) -> Optional[dt.datetime]:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def observation_buffer(manifest: dict, *, now: Optional[dt.datetime] = None) -> dict[str, Any]:
    """Summarise one source's ring buffer: newest/oldest frame, gaps, failed attempts."""
    now = now or dt.datetime.now(dt.timezone.utc)
    times = sorted(t for t in (_parse_iso(f.get("valid_time")) for f in manifest.get("frames") or []) if t)
    cadence = int(manifest.get("cadence_seconds") or 0) or None
    failed = manifest.get("failed") or {}
    out: dict[str, Any] = {
        "cadenceSeconds": cadence,
        "frameCount": len(times),
        "expectedFrames": None,
        "oldestValidTime": None,
        "latestValidTime": None,
        "ageSeconds": None,
        "failedFrames": len(failed),
        "failedAttempts": sum(int(v) for v in failed.values() if isinstance(v, (int, float))),
    }
    if not times:
        return out
    out["oldestValidTime"] = times[0].isoformat().replace("+00:00", "Z")
    out["latestValidTime"] = times[-1].isoformat().replace("+00:00", "Z")
    out["ageSeconds"] = int((now - times[-1]).total_seconds())
    if cadence:
        # Frames a gap-free buffer would hold between its oldest and newest frame.
        out["expectedFrames"] = int(round((times[-1] - times[0]).total_seconds() / cadence)) + 1
    return out


# --- the verdict -------------------------------------------------------------
def _row(key: str, group: str, label: str, status: str, value: str, limit: str = "", detail: str = "") -> dict[str, str]:
    return {"key": key, "group": group, "label": label, "status": status, "value": value, "limit": limit, "detail": detail}


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def _fmt_gb(value: Optional[float]) -> str:
    return "-" if value is None else f"{value / 1024 ** 3:.1f} GB"


MODEL_LABELS = {"icon_d2": "ICON-D2 forecast", "icon_eu": "ICON-EU forecast"}
OBSERVATION_LABELS = {"radar": "Radar", "satellite": "Satellite (MSG)", "mtg": "Satellite (MTG)", "li": "Lightning (LI)"}


def build_health(
    *,
    freshness: dict,
    model_timings: Optional[dict] = None,
    disk: Optional[dict] = None,
    server_health: Optional[dict] = None,
    pipelines: Optional[dict] = None,
    server_errors_last_hour: int = 0,
    server_errors_since_start: int = 0,
    active_log_error_groups: int = 0,
) -> dict[str, Any]:
    """One row per thing that can break, and the worst state as the verdict.

    ``freshness`` is the ``health_checks.data_freshness`` payload, i.e. exactly
    what the external monitor sees; a stale stream is critical here because it
    is what pages there.
    """
    rows: list[dict[str, str]] = []
    checks = (freshness or {}).get("checks") or {}
    timings = model_timings or {}

    for model, label in MODEL_LABELS.items():
        check = checks.get(model) or {}
        max_hours = check.get("maxAgeHours")
        limit = f"max {max_hours:g} h" if isinstance(max_hours, (int, float)) else "-"
        if check.get("run") is None:
            rows.append(_row(model, "Forecast", label, "crit", "no run on disk", limit))
            continue
        status = "ok" if check.get("status") == "ok" else "crit"
        age_hours = check.get("ageHours")
        value = f"run {check['run']}, {age_hours:.1f} h old" if isinstance(age_hours, (int, float)) else f"run {check['run']}"
        detail = ""
        running_min = (timings.get(model) or {}).get("ingestRunningMinutes")
        if running_min is not None:
            detail = f"newer run ingesting for {_fmt_duration(float(running_min) * 60)}"
            if status == "ok" and float(running_min) * 60 > FORECAST_INGEST_WARN_SECONDS:
                status = "warn"
        rows.append(_row(model, "Forecast", label, status, value, limit, detail))

    for source, label in OBSERVATION_LABELS.items():
        check = checks.get(source)
        if check is None:
            continue
        limit = f"max {_fmt_duration(check.get('maxAgeSeconds'))}"
        if check.get("status") == "absent":
            rows.append(_row(source, "Observations", label, "info", "no frames (source not set up)", limit))
            continue
        status = "ok" if check.get("status") == "ok" else "crit"
        rows.append(_row(source, "Observations", label, status, f"newest frame {_fmt_duration(check.get('ageSeconds'))} old", limit))

    pipelines = pipelines or {}
    forecast = pipelines.get("forecast") or {}
    observations = pipelines.get("observations") or {}
    longest = forecast.get("longestSeconds")
    if forecast.get("running") or (longest or 0) > FORECAST_INGEST_WARN_SECONDS:
        stuck = (longest or 0) > FORECAST_INGEST_WARN_SECONDS
        rows.append(_row(
            "forecast_ingest", "Ingest", "Forecast ingest", "warn" if stuck else "ok",
            f"running for {_fmt_duration(longest)}", f"warn after {_fmt_duration(FORECAST_INGEST_WARN_SECONDS)}",
        ))
    else:
        rows.append(_row("forecast_ingest", "Ingest", "Forecast ingest", "ok", "idle", f"warn after {_fmt_duration(FORECAST_INGEST_WARN_SECONDS)}"))
    longest = observations.get("longestSeconds")
    if longest is not None:
        stuck = longest > OBSERVATION_INGEST_CRIT_SECONDS
        rows.append(_row(
            "observation_ingest", "Ingest", "Observation ingest", "crit" if stuck else "ok",
            f"running for {_fmt_duration(longest)}", f"max {_fmt_duration(OBSERVATION_INGEST_CRIT_SECONDS)}",
            "a run this long is hung and blocks every later tick" if stuck else "",
        ))
    else:
        rows.append(_row("observation_ingest", "Ingest", "Observation ingest", "ok", "idle", f"max {_fmt_duration(OBSERVATION_INGEST_CRIT_SECONDS)}"))

    disk = disk or {}
    used_pct = disk.get("usedPercent")
    disk_limit = f"warn {DISK_WARN_PERCENT:.0f}%, critical {DISK_CRIT_PERCENT:.0f}%"
    if used_pct is None:
        rows.append(_row("disk", "Server", "Disk", "info", "unknown", disk_limit))
    else:
        status = "crit" if used_pct >= DISK_CRIT_PERCENT else "warn" if used_pct >= DISK_WARN_PERCENT else "ok"
        rows.append(_row("disk", "Server", "Disk", status, f"{used_pct:.0f}% used, {_fmt_gb(disk.get('freeBytes'))} free", disk_limit))

    mem_pct = (server_health or {}).get("memoryPercent")
    mem_limit = f"warn {MEMORY_WARN_PERCENT:.0f}%, critical {MEMORY_CRIT_PERCENT:.0f}%"
    if mem_pct is None:
        rows.append(_row("memory", "Server", "Memory", "info", "unknown", mem_limit))
    else:
        status = "crit" if mem_pct >= MEMORY_CRIT_PERCENT else "warn" if mem_pct >= MEMORY_WARN_PERCENT else "ok"
        rows.append(_row("memory", "Server", "Memory", status, f"{mem_pct:.0f}% used", mem_limit))

    status = (
        "crit" if server_errors_last_hour >= SERVER_ERRORS_CRIT_PER_HOUR
        else "warn" if server_errors_last_hour >= SERVER_ERRORS_WARN_PER_HOUR
        else "ok"
    )
    rows.append(_row(
        "server_errors", "App", "Server errors (5xx)", status,
        f"{server_errors_last_hour} in the last hour", f"critical from {SERVER_ERRORS_CRIT_PER_HOUR} per hour",
        f"{server_errors_since_start} since the app started",
    ))
    rows.append(_row(
        "log_errors", "App", "Errors in logs", "warn" if active_log_error_groups else "ok",
        f"{active_log_error_groups} kind(s) in the last 10 min", "any is a warning",
    ))

    worst = max((_SEVERITY.get(r["status"], 0) for r in rows), default=0)
    issues = [r for r in rows if _SEVERITY.get(r["status"], 0) > 0]
    issues.sort(key=lambda r: _SEVERITY.get(r["status"], 0), reverse=True)
    return {
        "status": "crit" if worst == 2 else "warn" if worst == 1 else "ok",
        "rows": rows,
        "issues": [f"{r['label']}: {r['value']}" for r in issues],
        "checkedAt": (freshness or {}).get("checkedAt"),
    }
