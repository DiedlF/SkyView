#!/usr/bin/env python3
"""Skyview regression checks for fragile map behaviors.

Usage:
  python3 scripts/qa_regression.py [--base http://127.0.0.1:8501]
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import sys
import numpy as np
import requests


def fail(msg: str):
    raise AssertionError(msg)


def get_merged_steps(base: str):
    r = requests.get(base + "/api/timesteps", timeout=20)
    if r.status_code != 200:
        fail(f"/api/timesteps failed: {r.status_code}")
    merged = r.json().get("merged") or {}
    return merged.get("steps") or []


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
        # Valid operational state when merged timeline currently comes from a single model window.
        return

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


def check_convective_agl_suppression_logic():
    """Regression guard: convective/cloud-free decision must use AGL (MSL-hsurf), threshold 300m."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from classify import classify_cloud_type  # local import to keep API-only path lightweight

    # Eight synthetic points:
    # [cb_suppressed, cb_allowed, cu_con_suppressed, cu_con_allowed,
    #  cu_hum_suppressed, cu_hum_allowed, blue_suppressed, blue_allowed]
    ww = np.zeros((1, 8), dtype=float)
    clcl = np.array([[40, 40, 40, 40, 40, 40, 0, 0]], dtype=float)
    clcm = np.zeros((1, 8), dtype=float)
    clch = np.zeros((1, 8), dtype=float)
    cape = np.array([[1200, 1200, 600, 600, 200, 200, 200, 200]], dtype=float)
    htop_dc = np.array([[1800, 1800, 1800, 1800, 1800, 1800, 1290, 1310]], dtype=float)
    hbas_sc = np.array([[1200, 1400, 1200, 1400, 1200, 1400, -1, -1]], dtype=float)
    # depth: [5000,5000,2500,2500,1000,1000,1001,1001]
    htop_sc = np.array([[6200, 6400, 3700, 3900, 2200, 2400, 1000, 1000]], dtype=float)
    lpi = np.array([[9, 9, 0, 0, 0, 0, 0, 0]], dtype=float)
    ceiling = np.zeros((1, 8), dtype=float)
    hsurf = np.array([[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]], dtype=float)

    out = classify_cloud_type(ww, clcl, clcm, clch, cape, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=hsurf)
    got = [str(x) for x in out[0, :]]
    want = ["clear", "cb", "clear", "cu_con", "clear", "cu_hum", "clear", "blue_thermal"]
    if got != want:
        fail(f"AGL suppression regression: got={got}, want={want}")


def check_blue_thermal_precedence_over_cb():
    """Regression guard: blue_thermal must win over cb when both conditions are true."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from classify import classify_cloud_type

    ww = np.zeros((1, 1), dtype=float)
    clcl = np.array([[0.0]], dtype=float)        # triggers blue_thermal branch
    clcm = np.zeros((1, 1), dtype=float)
    clch = np.zeros((1, 1), dtype=float)
    cape = np.array([[1500.0]], dtype=float)
    htop_dc = np.array([[1500.0]], dtype=float)  # blue_ok true with hsurf=1000
    hbas_sc = np.array([[1200.0]], dtype=float)
    htop_sc = np.array([[6200.0]], dtype=float)
    lpi = np.array([[12.0]], dtype=float)        # would trigger cb too
    ceiling = np.zeros((1, 1), dtype=float)
    hsurf = np.array([[1000.0]], dtype=float)

    out = classify_cloud_type(ww, clcl, clcm, clch, cape, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf=hsurf)
    got = str(out[0, 0])
    if got != "blue_thermal":
        fail(f"Precedence regression: expected blue_thermal over cb, got={got}")


def check_symbol_zoom_continuity(base: str, t: str):
    """Regression guard: non-clear parent symbols should persist into at least one non-clear child when zooming in."""
    bbox = "46,8,49,13"
    cell_sizes = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, 9: 0.12, 10: 0.06, 11: 0.03, 12: 0.02}

    def fetch(z: int):
        r = requests.get(base + "/api/symbols", params={"bbox": bbox, "zoom": z, "time": t, "model": "icon_d2"}, timeout=60)
        if r.status_code != 200:
            fail(f"zoom continuity symbols failed at z{z}: {r.status_code}")
        return r.json().get("symbols", [])

    def continuity(parent, child, z_parent: int):
        cs = cell_sizes[z_parent]
        total = 0
        ok = 0
        for p in parent:
            p_type = p.get("type")
            if p_type in (None, "clear"):
                continue
            total += 1
            plat = float(p["lat"])
            plon = float(p["lon"])
            lat_lo, lat_hi = plat - cs / 2.0, plat + cs / 2.0
            lon_lo, lon_hi = plon - cs / 2.0, plon + cs / 2.0
            kids = [
                c for c in child
                if lat_lo - 1e-6 <= float(c["lat"]) <= lat_hi + 1e-6
                and lon_lo - 1e-6 <= float(c["lon"]) <= lon_hi + 1e-6
            ]
            if any(k.get("type") not in (None, "clear") for k in kids):
                ok += 1
        return total, ok

    s8 = fetch(8)
    s9 = fetch(9)
    s10 = fetch(10)

    for z, parent, child in [(8, s8, s9), (9, s9, s10)]:
        total, ok = continuity(parent, child, z)
        if total == 0:
            continue
        ratio = ok / total
        # Allow tiny edge effects but guard against stride-induced widespread disappearances.
        if ratio < 0.995:
            fail(f"symbol zoom continuity too low at z{z}->z{z+1}: {ok}/{total} ({ratio:.2%})")


def check_resolve_eu_time_strict_input_handling():
    """Regression guard for explicit latest/malformed handling in strict EU resolver."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    import app as backend_app

    latest = backend_app._resolve_eu_time_strict("latest")
    # If EU data is unavailable, None is acceptable; otherwise expect tuple(run, step, model).
    if latest is not None:
        if not (isinstance(latest, tuple) and len(latest) == 3 and latest[2] == "icon_eu"):
            fail(f"_resolve_eu_time_strict('latest') returned unexpected payload: {latest!r}")

    malformed = backend_app._resolve_eu_time_strict("not-a-time")
    if malformed is not None:
        fail("_resolve_eu_time_strict('not-a-time') should return None")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8501")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # Logic-only checks run regardless of data availability
    check_convective_agl_suppression_logic()
    check_blue_thermal_precedence_over_cb()
    check_resolve_eu_time_strict_input_handling()

    steps = get_merged_steps(base)
    if not steps:
        print("SKIP: No ingested data — skipping HTTP regression checks (CI/empty server)")
        print("PASS: Skyview regression checks passed (logic only)")
        return

    t23 = next((s["validTime"] for s in steps if s["validTime"][11:13] == "23"), steps[0]["validTime"])
    check_z12_row_continuity(base, t23)
    check_border_pan_stability(base, t23)
    check_d2_eu_handover(base, steps)
    check_wind_point_parity(base, t23)
    check_symbol_zoom_continuity(base, t23)

    print("PASS: Skyview regression checks passed")


if __name__ == "__main__":
    main()
