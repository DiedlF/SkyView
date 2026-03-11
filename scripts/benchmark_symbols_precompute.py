#!/usr/bin/env python3
"""Benchmark the effectiveness/cost of low-zoom precomputed symbol bins.

Measures:
- /api/symbols latency with current server (usually precomputed bins ON)
- /api/symbols latency with a temporary comparison server (precomputed bins OFF)
- optional disk usage of persisted low-zoom symbol bin JSON files

Typical VPS usage:
  cd /opt/skyview
  venv/bin/python scripts/benchmark_symbols_precompute.py \
    --base http://127.0.0.1:8501 \
    --app-dir /opt/skyview/backend \
    --venv-python /opt/skyview/venv/bin/python \
    --data-dir /opt/skyview/data
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class Scenario:
    name: str
    zoom: int
    bbox: str
    pans: list[str]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="z5_wide_alps",
        zoom=5,
        bbox="45.5,5.0,49.5,16.0",
        pans=[
            "45.5,5.0,49.5,16.0",
            "45.5,5.4,49.5,16.4",
            "45.7,5.4,49.7,16.4",
            "45.7,5.8,49.7,16.8",
            "45.3,5.8,49.3,16.8",
        ],
    ),
    Scenario(
        name="z7_alps",
        zoom=7,
        bbox="46.2,8.0,48.9,14.6",
        pans=[
            "46.2,8.0,48.9,14.6",
            "46.2,8.3,48.9,14.9",
            "46.4,8.3,49.1,14.9",
            "46.4,8.6,49.1,15.2",
            "46.1,8.6,48.8,15.2",
        ],
    ),
    Scenario(
        name="z9_tirol",
        zoom=9,
        bbox="46.75,9.70,48.35,13.95",
        pans=[
            "46.75,9.70,48.35,13.95",
            "46.75,9.82,48.35,14.07",
            "46.82,9.82,48.42,14.07",
            "46.82,9.94,48.42,14.19",
            "46.68,9.94,48.28,14.19",
        ],
    ),
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[idx]


def wait_for_http(base: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(base.rstrip("/") + "/api/health", timeout=2)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Server at {base} did not become ready: {last_err}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def get_latest_valid_time(base: str) -> str:
    r = requests.get(base.rstrip("/") + "/api/timesteps", timeout=20)
    r.raise_for_status()
    merged = (r.json() or {}).get("merged") or {}
    steps = merged.get("steps") or []
    if not steps:
        raise RuntimeError("No merged timesteps available")
    return str(steps[0]["validTime"])


def run_request(base: str, scenario: Scenario, bbox: str, valid_time: str, timeout_s: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.get(
        base.rstrip("/") + "/api/symbols",
        params={"zoom": scenario.zoom, "bbox": bbox, "time": valid_time},
        timeout=timeout_s,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    payload = r.json()
    diag = payload.get("diagnostics") or {}
    return {
        "elapsed_ms": elapsed_ms,
        "count": int(payload.get("count", 0)),
        "served_from": diag.get("servedFrom"),
        "symbol_mode": diag.get("symbolMode"),
        "aggregate_ms": ((diag.get("timingsMs") or {}).get("aggregate")),
        "load_ms": ((diag.get("timingsMs") or {}).get("load")),
        "grid_ms": ((diag.get("timingsMs") or {}).get("grid")),
    }


def benchmark_server(base: str, valid_time: str, runs: int, timeout_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {"base": base, "scenarios": {}}
    for scenario in SCENARIOS:
        warmup = run_request(base, scenario, scenario.bbox, valid_time, timeout_s)
        results = []
        for _ in range(runs):
            for bbox in scenario.pans:
                results.append(run_request(base, scenario, bbox, valid_time, timeout_s))
        elapsed = [x["elapsed_ms"] for x in results]
        agg_vals = [float(x["aggregate_ms"]) for x in results if x["aggregate_ms"] is not None]
        load_vals = [float(x["load_ms"]) for x in results if x["load_ms"] is not None]
        counts = [x["count"] for x in results]
        out["scenarios"][scenario.name] = {
            "zoom": scenario.zoom,
            "samples": len(results),
            "warmup_ms": round(warmup["elapsed_ms"], 1),
            "avg_ms": round(statistics.mean(elapsed), 1),
            "p95_ms": round(percentile(elapsed, 0.95), 1),
            "min_ms": round(min(elapsed), 1),
            "max_ms": round(max(elapsed), 1),
            "avg_count": round(statistics.mean(counts), 1) if counts else 0,
            "served_from": sorted({str(x["served_from"]) for x in results}),
            "symbol_mode": sorted({str(x["symbol_mode"]) for x in results}),
            "avg_aggregate_ms": round(statistics.mean(agg_vals), 1) if agg_vals else None,
            "avg_load_ms": round(statistics.mean(load_vals), 1) if load_vals else None,
        }
    return out


def disk_usage_for_precomputed_bins(data_dir: Path) -> dict[str, Any]:
    files = list(data_dir.rglob("_symbols_z*_b*_*.json"))
    total_bytes = sum(p.stat().st_size for p in files if p.exists())
    by_zoom: dict[str, dict[str, int]] = {}
    for p in files:
        name = p.name
        z_part = name.split("_")[1] if len(name.split("_")) > 1 else "z?"
        zoom = z_part.replace("symbols", "").strip() or "unknown"
        bucket = by_zoom.setdefault(zoom, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += p.stat().st_size
    return {
        "files": len(files),
        "bytes": total_bytes,
        "megabytes": round(total_bytes / (1024 * 1024), 2),
        "by_zoom": by_zoom,
    }


def launch_compare_server(venv_python: str, app_dir: str, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = app_dir
    env["SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS"] = "0"
    cmd = [
        venv_python,
        "-m",
        "uvicorn",
        "app:app",
        "--app-dir",
        app_dir,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def print_report(current: dict[str, Any], compare: dict[str, Any], disk: dict[str, Any] | None) -> None:
    print("\n=== Symbol precompute benchmark ===")
    for name, cur in current["scenarios"].items():
        cmp = compare["scenarios"][name]
        delta_ms = cmp["avg_ms"] - cur["avg_ms"]
        delta_pct = ((cmp["avg_ms"] / cur["avg_ms"]) - 1.0) * 100.0 if cur["avg_ms"] else 0.0
        p95_delta_ms = cmp["p95_ms"] - cur["p95_ms"]
        p95_delta_pct = ((cmp["p95_ms"] / cur["p95_ms"]) - 1.0) * 100.0 if cur["p95_ms"] else 0.0
        print(f"\n[{name}] z{cur['zoom']}")
        print(
            f"  precomputed ON : avg={cur['avg_ms']}ms p95={cur['p95_ms']}ms "
            f"served={','.join(cur['served_from'])} mode={','.join(cur['symbol_mode'])}"
        )
        print(
            f"  precomputed OFF: avg={cmp['avg_ms']}ms p95={cmp['p95_ms']}ms "
            f"served={','.join(cmp['served_from'])} mode={','.join(cmp['symbol_mode'])}"
        )
        avg_word = "faster" if delta_ms > 0 else "slower"
        p95_word = "faster" if p95_delta_ms > 0 else "slower"
        print(
            f"  delta          : avg {abs(delta_ms):.1f}ms ({abs(delta_pct):.1f}%) {avg_word} with precompute, "
            f"p95 {abs(p95_delta_ms):.1f}ms ({abs(p95_delta_pct):.1f}%) {p95_word}"
        )
    if disk is not None:
        print("\nDisk usage of persisted low-zoom symbol bins:")
        print(f"  files={disk['files']} total={disk['megabytes']} MB")
        for zoom, info in sorted(disk.get("by_zoom", {}).items()):
            print(f"  {zoom}: files={info['files']} size={round(info['bytes'] / (1024 * 1024), 2)} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8501", help="Current Skyview server (normally precomputed bins ON)")
    ap.add_argument("--runs", type=int, default=4, help="Number of pan cycles per scenario")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--compare-port", type=int, default=8511)
    ap.add_argument("--compare-base", default="", help="Optional already-running comparison server URL; skips auto-launch when set")
    ap.add_argument("--app-dir", default=str(Path(__file__).resolve().parents[1] / "backend"))
    ap.add_argument("--venv-python", default=str(Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"))
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    wait_for_http(args.base, timeout_s=20)
    valid_time = get_latest_valid_time(args.base)
    current = benchmark_server(args.base, valid_time, args.runs, args.timeout)

    compare_proc = None
    compare_base = args.compare_base.strip() or f"http://127.0.0.1:{args.compare_port}"
    try:
        if args.compare_base.strip():
            wait_for_http(compare_base, timeout_s=20)
        else:
            compare_proc = launch_compare_server(args.venv_python, args.app_dir, args.compare_port)
            wait_for_http(compare_base, timeout_s=90)
        compare = benchmark_server(compare_base, valid_time, args.runs, args.timeout)
    finally:
        if compare_proc is not None:
            terminate_process(compare_proc)

    disk = None
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        disk = disk_usage_for_precomputed_bins(data_dir)

    report = {
        "valid_time": valid_time,
        "current": current,
        "compare": compare,
        "disk": disk,
    }
    print_report(current, compare, disk)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
