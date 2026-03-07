#!/usr/bin/env python3
"""Precompute low-zoom symbols payloads and persist them for fast startup/warm cache."""

import os
import argparse
import asyncio
import numpy as np
from fastapi.testclient import TestClient

import app
from constants import LOW_ZOOM_GLOBAL_BBOX, CELL_SIZES_BY_ZOOM
from services.symbol_ops import symbols_bin_bbox, symbols_bin_indices_for_bbox


def _model_api_name(model: str) -> str:
    return model.replace("-", "_")


def _model_dir_name(model: str) -> str:
    return model


def _iter_steps(run_dir: str):
    for f in sorted(os.listdir(run_dir)):
        if f.endswith(".npz") and f[:-4].isdigit():
            yield int(f[:-4]), os.path.join(run_dir, f)


async def _run(model: str, run: str, zooms: list[int], steps_filter: set[int] | None = None, mode: str = "direct"):
    api_model = _model_api_name(model)
    run_dir = os.path.join(app.DATA_DIR, _model_dir_name(model), run)
    if not os.path.isdir(run_dir):
        print(f"Run dir not found: {run_dir}")
        return 1

    # Most .npz files do not carry validTime; build run+step -> validTime map
    # from timeline metadata instead of relying on npz payload internals.
    step_to_valid_time: dict[int, str] = {}
    try:
        runs = app.get_available_runs()
        match = next((r for r in runs if r.get("model") == api_model and r.get("run") == run), None)
        if match:
            for s in match.get("steps", []):
                st = s.get("step")
                vt = s.get("validTime")
                if isinstance(st, int) and isinstance(vt, str) and vt:
                    step_to_valid_time[st] = vt
    except Exception as e:
        print(f"warning: failed to build step->validTime map for {model} {run}: {e}")

    done = 0

    bins_by_zoom: dict[int, list[tuple[int, int]]] = {}
    for zoom in zooms:
        cs = CELL_SIZES_BY_ZOOM.get(int(zoom))
        if cs is None:
            continue
        bins_by_zoom[int(zoom)] = symbols_bin_indices_for_bbox(
            LOW_ZOOM_GLOBAL_BBOX[0], LOW_ZOOM_GLOBAL_BBOX[1],
            LOW_ZOOM_GLOBAL_BBOX[2], LOW_ZOOM_GLOBAL_BBOX[3],
            cs,
        )

    client = TestClient(app.app)

    for step, path in _iter_steps(run_dir):
        if steps_filter and step not in steps_filter:
            continue
        try:
            valid_time = step_to_valid_time.get(step)
            if not valid_time:
                # Backward fallback for datasets that embed validTime directly.
                z = np.load(path)
                valid_time = z["validTime"].item() if "validTime" in z.files else None
            if not valid_time:
                continue
            for zoom in zooms:
                cs = CELL_SIZES_BY_ZOOM.get(int(zoom))
                if cs is None:
                    continue
                for bi, bj in bins_by_zoom.get(int(zoom), []):
                    b = symbols_bin_bbox(bi, bj, cs)
                    bbox = f"{b[0]},{b[1]},{b[2]},{b[3]}"
                    if mode == "direct":
                        resp = client.get(
                            "/api/symbols",
                            params={"zoom": zoom, "bbox": bbox, "time": str(valid_time), "model": api_model},
                        )
                    else:
                        base_url = os.environ.get("SKYVIEW_PRECOMPUTE_BASE_URL", "http://127.0.0.1:8501")
                        import requests
                        resp = requests.get(
                            f"{base_url}/api/symbols",
                            params={"zoom": zoom, "bbox": bbox, "time": str(valid_time), "model": api_model},
                            timeout=60,
                        )
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    done += 1
        except Exception as e:
            print(f"precompute failed {model} {run} step={step}: {e}")
    print(f"Precomputed symbols entries: {done}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="icon-d2 or icon-eu")
    p.add_argument("--run", required=True)
    p.add_argument("--zooms", default="5,6,7,8,9")
    p.add_argument("--steps", default="", help="Optional comma-separated steps, e.g. 15 or 1,2,3")
    p.add_argument("--mode", default="direct", choices=["direct", "http"], help="direct=in-process API calls, http=network requests")
    args = p.parse_args()

    zooms = [int(x) for x in args.zooms.split(",") if x.strip()]
    steps_filter = {int(x) for x in args.steps.split(",") if x.strip()} if args.steps else None
    raise SystemExit(asyncio.run(_run(args.model, args.run, zooms, steps_filter=steps_filter, mode=args.mode)))


if __name__ == "__main__":
    main()
