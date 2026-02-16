#!/usr/bin/env python3
"""Skyview regression checks for fragile map behaviors.

Usage:
  python3 scripts/qa_regression.py [--base http://127.0.0.1:8501]
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
import requests


def fail(msg: str):
    raise AssertionError(msg)


def get_merged_steps(base: str):
    r = requests.get(base + "/api/timesteps", timeout=20)
    if r.status_code != 200:
        fail(f"/api/timesteps failed: {r.status_code}")
    merged = r.json().get("merged") or {}
    steps = merged.get("steps") or []
    if not steps:
        fail("No merged timeline steps")
    return steps


def check_z12_row_continuity(base: str, t: str):
    bbox = "47.60,11.85,47.92,12.08"  # around Geitau + north band
    r = requests.get(base + "/api/symbols", params={"bbox": bbox, "zoom": 12, "time": t}, timeout=30)
    if r.status_code != 200:
        fail(f"z12 symbols failed: {r.status_code}")
    sy = r.json().get("symbols", [])
    if len(sy) < 80:
        fail(f"z12 returned too few symbols: {len(sy)}")

    by_row = Counter(round(s["lat"], 4) for s in sy)
    row_sizes = sorted(set(by_row.values()))
    if len(row_sizes) != 1:
        fail(f"z12 uneven row sizes detected: {row_sizes}")


def check_border_pan_stability(base: str, t: str):
    # Two overlapping viewports: symbols in overlap should mostly match by grid cell
    b1 = "47.60,11.85,47.84,12.09"
    b2 = "47.62,11.87,47.86,12.11"
    common = "47.62,11.87,47.84,12.09"

    r1 = requests.get(base + "/api/symbols", params={"bbox": b1, "zoom": 11, "time": t}, timeout=30)
    r2 = requests.get(base + "/api/symbols", params={"bbox": b2, "zoom": 11, "time": t}, timeout=30)
    if r1.status_code != 200 or r2.status_code != 200:
        fail(f"border symbols failed: {r1.status_code}/{r2.status_code}")

    def in_common(s):
        return (47.62 <= s["lat"] <= 47.84) and (11.87 <= s["lon"] <= 12.09)

    m1 = {(round(s["lat"], 4), round(s["lon"], 4)): s["type"] for s in r1.json().get("symbols", []) if in_common(s)}
    m2 = {(round(s["lat"], 4), round(s["lon"], 4)): s["type"] for s in r2.json().get("symbols", []) if in_common(s)}

    keys = sorted(set(m1.keys()) & set(m2.keys()))
    if len(keys) < 30:
        fail(f"Too few overlap cells for stability check: {len(keys)}")

    mismatches = sum(1 for k in keys if m1[k] != m2[k])
    ratio = mismatches / len(keys)
    if ratio > 0.08:
        fail(f"Border pan instability too high: {mismatches}/{len(keys)} ({ratio:.1%})")


def check_d2_eu_handover(base: str, steps):
    # Find model change in merged steps
    switch_idx = None
    for i in range(1, len(steps)):
        if steps[i - 1].get("model") != steps[i].get("model"):
            switch_idx = i
            break
    if switch_idx is None:
        fail("No D2/EU handover found in merged timeline")

    idxs = [max(0, switch_idx - 1), switch_idx, min(len(steps) - 1, switch_idx + 1)]
    for i in idxs:
        t = steps[i]["validTime"]
        r = requests.get(base + "/api/symbols", params={"bbox": "47.2,10.8,48.2,12.3", "zoom": 10, "time": t}, timeout=30)
        if r.status_code != 200:
            fail(f"handover symbol fetch failed at idx {i}: {r.status_code}")
        c = r.json().get("count", 0)
        if c <= 0:
            fail(f"handover symbol fetch returned zero symbols at idx {i}")


def check_wind_point_parity(base: str, t: str):
    # Compare a rendered wind-barb cell against /api/point wind values for same level/zoom/time
    level = "850"
    zoom = 10
    bbox = "47.40,11.20,47.95,12.20"
    rw = requests.get(
        base + "/api/wind",
        params={"bbox": bbox, "zoom": zoom, "time": t, "level": level},
        timeout=30,
    )
    if rw.status_code != 200:
        fail(f"/api/wind failed: {rw.status_code}")
    barbs = rw.json().get("barbs", [])
    if not barbs:
        # skip if no wind at this sample time/area
        return

    b = barbs[len(barbs) // 2]
    rp = requests.get(
        base + "/api/point",
        params={
            "lat": b["lat"],
            "lon": b["lon"],
            "time": t,
            "wind_level": level,
            "zoom": zoom,
        },
        timeout=30,
    )
    if rp.status_code != 200:
        fail(f"/api/point failed for wind parity: {rp.status_code}")
    ov = rp.json().get("overlay_values", {})
    sp = ov.get("wind_speed")
    dr = ov.get("wind_dir")
    if sp is None or dr is None:
        fail("/api/point missing wind_speed/wind_dir for wind parity check")

    if abs(float(sp) - float(b["speed_kt"])) > 2.5:
        fail(f"wind speed mismatch too high: point={sp} kt vs barb={b['speed_kt']} kt")

    d = abs((float(dr) - float(b["dir_deg"]) + 180) % 360 - 180)
    if d > 20:
        fail(f"wind direction mismatch too high: point={dr}° vs barb={b['dir_deg']}° (Δ={d:.1f}°)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8501")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    steps = get_merged_steps(base)
    t23 = next((s["validTime"] for s in steps if s["validTime"][11:13] == "23"), steps[0]["validTime"])

    check_z12_row_continuity(base, t23)
    check_border_pan_stability(base, t23)
    check_d2_eu_handover(base, steps)
    check_wind_point_parity(base, t23)

    print("PASS: Skyview regression checks passed")


if __name__ == "__main__":
    main()
