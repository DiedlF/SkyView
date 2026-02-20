"""Lightweight usage stats helpers (privacy-preserving)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

usage_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _load_stats(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema": "skyview.usage.v1", "daily": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("daily"), dict):
                return data
    except Exception:
        pass
    return {"schema": "skyview.usage.v1", "daily": {}}


def _save_stats(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _fingerprint_hash(ip: str, user_agent: str, accept_lang: str, salt: str) -> str:
    raw = f"{ip}|{user_agent}|{accept_lang}|{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:20]


def record_visit(stats_file: str, *, ip: str, user_agent: str, accept_lang: str, salt: str, path: str = "/"):
    """Record a page visit for current UTC day."""
    now = _utc_now()
    day = _day_key(now)
    with usage_lock:
        data = _load_stats(stats_file)
        daily = data.setdefault("daily", {})
        row = daily.setdefault(day, {
            "visits": 0,
            "uniqueVisitors": 0,
            "visitorHashes": [],
            "pathCounts": {},
        })
        row["visits"] = int(row.get("visits", 0)) + 1

        vhash = _fingerprint_hash(ip, user_agent, accept_lang, salt)
        hashes = set(row.get("visitorHashes", []))
        hashes.add(vhash)
        # Keep bounded to avoid unbounded growth.
        if len(hashes) > 200000:
            hashes = set(list(hashes)[:200000])
        row["visitorHashes"] = sorted(hashes)
        row["uniqueVisitors"] = len(row["visitorHashes"])

        pc = row.setdefault("pathCounts", {})
        pc[path] = int(pc.get(path, 0)) + 1

        data["updatedAt"] = now.isoformat().replace("+00:00", "Z")
        _save_stats(stats_file, data)


def get_usage_stats(stats_file: str, *, days: int = 30) -> Dict[str, Any]:
    """Return aggregate usage stats for last `days` days (UTC)."""
    with usage_lock:
        data = _load_stats(stats_file)

    daily = data.get("daily", {})
    now = _utc_now()
    keys = [
        _day_key(now - timedelta(days=i))
        for i in range(max(1, days))
    ]

    selected = []
    total_visits = 0
    unique_global = set()
    for k in reversed(keys):
        row = daily.get(k)
        if not row:
            selected.append({"date": k, "visits": 0, "uniqueVisitors": 0})
            continue
        visits = int(row.get("visits", 0))
        uv = int(row.get("uniqueVisitors", 0))
        total_visits += visits
        for h in row.get("visitorHashes", []):
            unique_global.add(h)
        selected.append({"date": k, "visits": visits, "uniqueVisitors": uv})

    return {
        "windowDays": max(1, days),
        "totalVisits": total_visits,
        "estimatedUniqueVisitors": len(unique_global),
        "daily": selected,
        "updatedAt": data.get("updatedAt"),
        "schema": data.get("schema", "skyview.usage.v1"),
    }


def get_marker_stats(markers_file: str) -> Dict[str, Any]:
    """Derive marker engagement stats from markers.json."""
    if not os.path.exists(markers_file):
        return {
            "totalMarkers": 0,
            "distinctClientIds": 0,
            "activeLast24h": 0,
            "activeLast7d": 0,
            "activeLast30d": 0,
        }

    try:
        with open(markers_file, "r", encoding="utf-8") as f:
            markers = json.load(f)
        if not isinstance(markers, list):
            markers = []
    except Exception:
        markers = []

    now = _utc_now()
    cids = set()
    last_seen: Dict[str, datetime] = {}
    for m in markers:
        if not isinstance(m, dict):
            continue
        cid = str(m.get("clientId") or "").strip()
        if not cid:
            continue
        cids.add(cid)
        ts = m.get("createdAt")
        if isinstance(ts, str) and ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if cid not in last_seen or dt > last_seen[cid]:
                    last_seen[cid] = dt
            except Exception:
                pass

    def _active(delta_days: int) -> int:
        threshold = now - timedelta(days=delta_days)
        return sum(1 for dt in last_seen.values() if dt >= threshold)

    return {
        "totalMarkers": len(markers),
        "distinctClientIds": len(cids),
        "activeLast24h": _active(1),
        "activeLast7d": _active(7),
        "activeLast30d": _active(30),
    }
