#!/usr/bin/env python3
"""Skyview smoke/regression checks (non-visual).

Usage:
  python3 scripts/qa_smoke.py [--base http://127.0.0.1:8501]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
import requests


def assert_ok(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8501")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # 1) Core endpoints
    for ep in ("/api/health", "/api/timesteps", "/api/models", "/api/status"):
        r = requests.get(base + ep, timeout=20)
        assert_ok(r.status_code == 200, f"{ep} returned {r.status_code}")

    st = requests.get(base + "/api/status", timeout=20).json()
    fb = st.get("fallback", {})
    for k in ("euResolveAttempts", "euResolveSuccess", "strictTimeDenied", "symbolsBlended", "windBlended", "pointFallback"):
        assert_ok(k in fb, f"/api/status missing fallback.{k}")
    ih = st.get("ingestHealth", {}).get("models", {})
    assert_ok("icon_d2" in ih and "icon_eu" in ih, "/api/status missing ingestHealth models")
    for m in ("icon_d2", "icon_eu"):
        for k in ("availableSteps", "expectedSteps", "missingSteps", "coverage"):
            assert_ok(k in ih[m], f"/api/status missing ingestHealth.models.{m}.{k}")

    ts = requests.get(base + "/api/timesteps", timeout=20).json().get("merged", {}).get("steps", [])
    assert_ok(len(ts) > 10, "Merged timeline too short / unavailable")

    # Use a 23 UTC sample if available (known problematic region checks)
    sample = next((s for s in ts if s["validTime"][11:13] == "23"), ts[0])
    t = sample["validTime"]

    # 2) Symbol continuity around Geitau area at z12
    bbox = "47.60,11.85,47.92,12.08"
    r = requests.get(base + "/api/symbols", params={"bbox": bbox, "zoom": 12, "time": t}, timeout=30)
    assert_ok(r.status_code == 200, f"/api/symbols failed: {r.status_code}")
    sy = r.json().get("symbols", [])
    assert_ok(len(sy) > 50, "Too few symbols returned for z12 continuity check")

    by_row = Counter(round(s["lat"], 4) for s in sy)
    row_sizes = set(by_row.values())
    assert_ok(len(row_sizes) == 1, f"Inconsistent row sizes detected: {sorted(row_sizes)}")

    # 3) Overlay endpoints (key layers)
    layers = [
        "sigwx",
        "clouds_total",
        "clouds_total_mod",
        "thermals",
        "ceiling",
        "cloud_base",
        "conv_thickness",
    ]
    for layer in layers:
        ro = requests.get(
            base + "/api/overlay",
            params={"layer": layer, "bbox": "47,10,48,12", "time": t, "width": 400},
            timeout=30,
        )
        assert_ok(ro.status_code == 200, f"overlay {layer} failed: {ro.status_code}")
        assert_ok(len(ro.content) > 200, f"overlay {layer} returned suspiciously tiny payload")

    # 4) CAPE threshold sanity (thermals endpoint available + point returns raw value)
    rp = requests.get(
        base + "/api/point",
        params={"lat": 47.6836, "lon": 11.9610, "time": t, "wind_level": "10m", "zoom": 12},
        timeout=30,
    )
    assert_ok(rp.status_code == 200, f"/api/point failed: {rp.status_code}")
    j = rp.json()
    assert_ok("overlay_values" in j, "/api/point missing overlay_values")
    assert_ok("thermals" in j["overlay_values"], "/api/point missing thermals value")
    diag = j.get("diagnostics", {})
    for k in ("dataFreshnessMinutes", "fallbackDecision", "requestedTime", "sourceModel"):
        assert_ok(k in diag, f"/api/point missing diagnostics.{k}")

    print("PASS: Skyview smoke checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}")
        raise
