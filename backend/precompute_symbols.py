#!/usr/bin/env python3
"""Precompute low-zoom symbols payloads and persist them for fast startup/warm cache."""

import os
import argparse
import asyncio
import numpy as np

import app


def _model_api_name(model: str) -> str:
    return model.replace("-", "_")


def _model_dir_name(model: str) -> str:
    return model


def _iter_steps(run_dir: str):
    for f in sorted(os.listdir(run_dir)):
        if f.endswith(".npz") and f[:-4].isdigit():
            yield int(f[:-4]), os.path.join(run_dir, f)


async def _run(model: str, run: str, zooms: list[int]):
    api_model = _model_api_name(model)
    run_dir = os.path.join(app.DATA_DIR, _model_dir_name(model), run)
    if not os.path.isdir(run_dir):
        print(f"Run dir not found: {run_dir}")
        return 1

    done = 0
    for step, path in _iter_steps(run_dir):
        try:
            z = np.load(path)
            valid_time = z["validTime"].item() if "validTime" in z.files else None
            if not valid_time:
                continue
            for zoom in zooms:
                await app.api_symbols(
                    zoom=zoom,
                    bbox=f"{app.LOW_ZOOM_GLOBAL_BBOX[0]},{app.LOW_ZOOM_GLOBAL_BBOX[1]},{app.LOW_ZOOM_GLOBAL_BBOX[2]},{app.LOW_ZOOM_GLOBAL_BBOX[3]}",
                    time=str(valid_time),
                    model=api_model,
                )
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
    args = p.parse_args()

    zooms = [int(x) for x in args.zooms.split(",") if x.strip()]
    raise SystemExit(asyncio.run(_run(args.model, args.run, zooms)))


if __name__ == "__main__":
    main()
