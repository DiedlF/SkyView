#!/usr/bin/env python3
"""Skyview lightweight performance probe.

Usage:
  python3 scripts/qa_perf.py [--base http://127.0.0.1:8501] [--runs 6]

Notes:
- Executes 1 warmup + N-1 measured requests per probe.
- Prints per-probe timing stats and optional threshold checks.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests


@dataclass
class Probe:
    name: str
    path: str
    params: Dict[str, str]
    max_avg_ms: float


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[idx]


def run_probe(base: str, probe: Probe, runs: int, timeout_s: float) -> Tuple[float, List[float]]:
    url = base + probe.path
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = requests.get(url, params=probe.params, timeout=timeout_s)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            raise RuntimeError(f"{probe.name}: HTTP {r.status_code}")
        # Access body to ensure full response read/decode path.
        _ = r.content
        times.append(dt)

    warm = times[0]
    measured = times[1:] if len(times) > 1 else times
    return warm, measured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8501")
    ap.add_argument("--runs", type=int, default=6, help="Total runs per probe (first run is warmup)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--strict", action="store_true", help="Fail with non-zero exit code when avg exceeds threshold")
    args = ap.parse_args()

    if args.runs < 2:
        raise SystemExit("--runs must be >= 2")

    base = args.base.rstrip("/")

    probes = [
        Probe(
            name="overlay_lpi_small_w400",
            path="/api/overlay",
            params={"layer": "lpi", "bbox": "47,10,48,12", "time": "latest", "width": "400"},
            max_avg_ms=30.0,
        ),
        Probe(
            name="overlay_lpi_medium_w700",
            path="/api/overlay",
            params={"layer": "lpi", "bbox": "46.8,9.8,48.6,14.2", "time": "latest", "width": "700"},
            max_avg_ms=45.0,
        ),
        Probe(
            name="wind_850_small_z10",
            path="/api/wind",
            params={"bbox": "47.40,11.20,47.95,12.20", "zoom": "10", "time": "latest", "level": "850"},
            max_avg_ms=80.0,
        ),
        Probe(
            name="wind_850_medium_z9",
            path="/api/wind",
            params={"bbox": "46.8752,9.8959,48.4802,14.0267", "zoom": "9", "time": "latest", "level": "850"},
            max_avg_ms=140.0,
        ),
    ]

    failed = []

    print(f"PERF base={base} runs={args.runs} (1 warmup + {args.runs-1} measured)")
    for p in probes:
        warm, vals = run_probe(base, p, args.runs, args.timeout)
        avg = statistics.mean(vals)
        p95 = percentile(vals, 0.95)
        vmin = min(vals)
        vmax = max(vals)
        status = "PASS" if avg <= p.max_avg_ms else "SLOW"
        print(
            f"{status} {p.name}: warm={warm:.1f}ms avg={avg:.1f}ms p95={p95:.1f}ms "
            f"min={vmin:.1f}ms max={vmax:.1f}ms limit_avg={p.max_avg_ms:.1f}ms"
        )
        if avg > p.max_avg_ms:
            failed.append((p.name, avg, p.max_avg_ms))

    if failed and args.strict:
        details = ", ".join(f"{name} avg={avg:.1f}>{limit:.1f}ms" for name, avg, limit in failed)
        raise SystemExit(f"FAIL: {details}")

    if failed:
        print(f"WARN: {len(failed)} probe(s) above threshold")
    else:
        print("PASS: all perf probes within thresholds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
