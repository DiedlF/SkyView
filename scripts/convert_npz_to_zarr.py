#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from services.storage_io import step_zarr_path, static_zarr_path, write_zarr_group, zarr_available  # noqa: E402


def _read_npz(path: str) -> dict:
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert existing Skyview NPZ forecast data to side-by-side Zarr groups.")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    parser.add_argument("--model", required=True, choices=["icon-d2", "icon-eu"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing .zarr directories")
    args = parser.parse_args()

    if not zarr_available():
        print("zarr/numcodecs is not installed", file=sys.stderr)
        return 2

    run_dir = os.path.join(args.data_dir, args.model, args.run)
    if not os.path.isdir(run_dir):
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 1

    done = 0
    for name in sorted(os.listdir(run_dir)):
        if not (name.endswith(".npz") and name[:-4].isdigit()):
            continue
        step = int(name[:-4])
        out = step_zarr_path(args.data_dir, args.model, args.run, step)
        if os.path.isdir(out) and not args.force:
            continue
        arrays = _read_npz(os.path.join(run_dir, name))
        write_zarr_group(out, arrays, attrs={"model": args.model.replace("-", "_"), "run": args.run, "step": step})
        done += 1

    static_npz = os.path.join(args.data_dir, args.model, "grid", "static.npz")
    if os.path.exists(static_npz):
        out = static_zarr_path(args.data_dir, args.model)
        if args.force or not os.path.isdir(out):
            write_zarr_group(out, _read_npz(static_npz))

    print(f"converted {done} step(s) for {args.model} {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
