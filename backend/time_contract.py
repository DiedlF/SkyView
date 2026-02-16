"""Shared run/timeline/time-resolution helpers for Skyview + Explorer."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException


def get_available_runs(data_dir: str):
    runs = []
    for model_dir in ["icon-d2", "icon-eu"]:
        model_path = os.path.join(data_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        model_type = model_dir.replace("-", "_")
        for d in sorted(os.listdir(model_path), reverse=True):
            run_path = os.path.join(model_path, d)
            if not os.path.isdir(run_path):
                continue
            try:
                run_dt = datetime.strptime(d, "%Y%m%d%H")
            except ValueError:
                continue
            npz_files = sorted([f for f in os.listdir(run_path) if f.endswith(".npz")])
            steps = []
            for f in npz_files:
                try:
                    step = int(f[:-4])
                    vt = run_dt.replace(tzinfo=timezone.utc)
                    vt = vt.timestamp() + step * 3600
                    steps.append({"step": step, "validTime": datetime.fromtimestamp(vt, tz=timezone.utc).isoformat().replace('+00:00', 'Z')})
                except ValueError:
                    continue
            if steps:
                runs.append({
                    "run": d,
                    "model": model_type,
                    "runTime": run_dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z'),
                    "steps": steps,
                })
    runs.sort(key=lambda x: x["runTime"], reverse=True)
    return runs


def get_merged_timeline(data_dir: str):
    runs = get_available_runs(data_dir)
    if not runs:
        return None

    d2_complete = next((r for r in runs if r["model"] == "icon_d2" and len(r["steps"]) >= 48), None)
    d2_any = next((r for r in runs if r["model"] == "icon_d2"), None)
    d2_run = d2_complete or d2_any
    eu_run = next((r for r in runs if r["model"] == "icon_eu"), None)
    if not d2_run and not eu_run:
        return None

    primary = d2_run or eu_run
    merged_steps = []
    d2_last_valid = None

    if d2_run:
        for s in d2_run["steps"]:
            merged_steps.append({**s, "model": "icon_d2", "run": d2_run["run"]})
            if d2_last_valid is None or s["validTime"] > d2_last_valid:
                d2_last_valid = s["validTime"]

    if eu_run and d2_last_valid:
        for s in eu_run["steps"]:
            if s["validTime"] > d2_last_valid:
                merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})
    elif eu_run and not d2_run:
        for s in eu_run["steps"]:
            merged_steps.append({**s, "model": "icon_eu", "run": eu_run["run"]})

    merged_steps.sort(key=lambda x: x["validTime"])

    return {
        "run": primary["run"],
        "runTime": primary["runTime"],
        "model": primary["model"],
        "steps": merged_steps,
        "d2Run": d2_run["run"] if d2_run else None,
        "euRun": eu_run["run"] if eu_run else None,
    }


def resolve_time(data_dir: str, time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
    runs = get_available_runs(data_dir)
    if not runs:
        raise HTTPException(404, "No data available")

    if time_str == "latest":
        d2_runs = [r for r in runs if r["model"] == "icon_d2"]
        if d2_runs:
            return d2_runs[0]["run"], d2_runs[0]["steps"][-1]["step"], "icon_d2"
        eu_runs = [r for r in runs if r["model"] == "icon_eu"]
        if eu_runs:
            return eu_runs[0]["run"], eu_runs[0]["steps"][-1]["step"], "icon_eu"
        return runs[0]["run"], runs[0]["steps"][-1]["step"], runs[0]["model"]

    try:
        target = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "Invalid time format")
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    lead_hours = max(0, (target - now).total_seconds() / 3600.0)

    candidates = runs
    if model is None:
        pref_model = "icon_d2" if lead_hours <= 48 else "icon_eu"
        candidates = [r for r in runs if r["model"] == pref_model]
    else:
        candidates = [r for r in runs if r["model"] == model]
    if not candidates:
        candidates = runs

    best_dist = float("inf")
    best_run = best_step = best_model = None
    for r in candidates:
        for s in r["steps"]:
            vt = datetime.fromisoformat(s["validTime"].replace("Z", "+00:00"))
            dist = abs((vt - target).total_seconds())
            if dist < best_dist:
                best_dist = dist
                best_run = r["run"]
                best_step = s["step"]
                best_model = r["model"]
    if best_run is None:
        raise HTTPException(404, "No matching timestep")
    return best_run, best_step, best_model
