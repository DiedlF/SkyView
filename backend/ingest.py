#!/usr/bin/env python3
"""ICON-D2 and ICON-EU data ingester for Skyview.
Downloads regular-lat-lon GRIB2 from DWD, optionally crops, saves .npz.

Config-driven: reads ingest_config.yaml for variable groups and region settings.
ICON-D2: full native grid (746×1215).
ICON-EU: cropped to D2 bounds (43.18–58.08°N, -3.94–20.34°E).
Retention: latest run only.
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import cfgrib
import yaml
from datetime import datetime, timedelta, timezone
from logging_config import setup_logging

logger = setup_logging(__name__, level="INFO")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "ingest_config.yaml")


def load_config():
    """Load ingest configuration from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_active_variables(config):
    """Collect all enabled single-level variables from config groups."""
    variables = []
    for group_name, group in config.get("groups", {}).items():
        if group.get("enabled", False):
            variables.extend(group.get("variables", []))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def get_static_variables(config):
    return config.get("static_variables", ["hsurf"])


def get_pressure_config(config):
    pc = config.get("pressure_levels", {})
    return pc.get("variables", []), pc.get("levels", [])


def get_d2_only_variables(config):
    return set(config.get("d2_only_variables", []))


def get_eu_var_name(var, config):
    """Get ICON-EU variable name (may differ from D2)."""
    eu_map = config.get("eu_variable_map", {})
    return eu_map.get(var, var)


# ── Legacy bounds for backward compat (Alps crop) ──
LEGACY_BOUNDS = (45.5, 48.5, 9.0, 17.0)

# Model configurations
MODEL_CONFIG = {
    "icon-d2": {
        "runs_per_day": 8,
        "run_interval": 3,
        "base_url": "https://opendata.dwd.de/weather/nwp/icon-d2/grib",
        "filename_pattern": "icon-d2_germany_regular-lat-lon_single-level_{run}_{step:03d}_2d_{var}.grib2.bz2",
        "max_forecast_hours": 48,
        "steps": list(range(1, 49)),
        "short_steps": list(range(1, 13)),
    },
    "icon-eu": {
        "runs_per_day": 4,
        "run_interval": 6,
        "base_url": "https://opendata.dwd.de/weather/nwp/icon-eu/grib",
        "filename_pattern": "icon-eu_europe_regular-lat-lon_single-level_{run}_{step:03d}_{var_upper}.grib2.bz2",
        "max_forecast_hours": 120,
        # Full relevant EU horizon: hourly 1..78, then 3-hourly 81..120.
        "steps": list(range(1, 79)) + list(range(81, 121, 3)),
        "short_steps": list(range(1, 25)),
    },
}


def get_latest_run(model="icon-d2"):
    """Determine latest available run, accounting for publication delay."""
    now = datetime.now(timezone.utc)
    delay_hours = 2 if model == "icon-d2" else 4
    ref = now - timedelta(hours=delay_hours)
    cfg = MODEL_CONFIG[model]
    hour = (ref.hour // cfg["run_interval"]) * cfg["run_interval"]
    return ref.strftime("%Y%m%d") + f"{hour:02d}"


def build_url(run, step, var, model="icon-d2", pressure_level=None, config=None):
    """Build DWD download URL for regular-lat-lon data."""
    cfg = MODEL_CONFIG[model]
    run_hour = run[-2:]

    # Map variable name for EU
    if model == "icon-eu" and config:
        var_name = get_eu_var_name(var, config)
    elif model == "icon-eu":
        var_name = var
    else:
        var_name = var

    static_vars = get_static_variables(config) if config else ["hsurf"]

    if var in static_vars:
        if model == "icon-d2":
            return (f"{cfg['base_url']}/{run_hour}/{var}/"
                    f"icon-d2_germany_regular-lat-lon_time-invariant_{run}_000_0_{var}.grib2.bz2")
        else:
            return (f"{cfg['base_url']}/{run_hour}/{var_name}/"
                    f"icon-eu_europe_regular-lat-lon_time-invariant_{run}_000_0_{var_name.upper()}.grib2.bz2")

    if pressure_level:
        if model == "icon-d2":
            return (f"{cfg['base_url']}/{run_hour}/{var}/"
                    f"icon-d2_germany_regular-lat-lon_pressure-level_{run}_{step:03d}_{pressure_level}_{var}.grib2.bz2")
        else:
            return (f"{cfg['base_url']}/{run_hour}/{var_name}/"
                    f"icon-eu_europe_regular-lat-lon_pressure-level_{run}_{step:03d}_{pressure_level}_{var_name.upper()}.grib2.bz2")

    if model == "icon-d2":
        return (f"{cfg['base_url']}/{run_hour}/{var}/"
                f"{cfg['filename_pattern'].format(run=run, step=step, var=var)}")
    else:
        return (f"{cfg['base_url']}/{run_hour}/{var_name}/"
                f"{cfg['filename_pattern'].format(run=run, step=step, var_upper=var_name.upper())}")


def download(url, dest):
    """Download file, skip if already exists and non-empty."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return True
    try:
        r = subprocess.run(["curl", "-sfL", url, "-o", dest], timeout=120, capture_output=True)
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 100:
            return True
        if os.path.exists(dest):
            os.unlink(dest)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(dest):
            os.unlink(dest)
    return False


def load_grib(filepath, bounds=None):
    """Load GRIB2, optionally crop to bounds, return (data_2d, lat_1d, lon_1d)."""
    ds = cfgrib.open_datasets(filepath)[0]
    var = list(ds.data_vars.values())[0]
    data = var.values.squeeze()
    lat = ds.coords["latitude"].values
    lon = ds.coords["longitude"].values

    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        eps = 0.001
        lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
        lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
        return data[np.ix_(lat_mask, lon_mask)], lat[lat_mask], lon[lon_mask]

    return data, lat, lon


def get_bounds_for_model(model, config):
    """Return crop bounds for a model, or None for full grid."""
    region_cfg = config.get("region", {})
    setting = region_cfg.get(model, "full")

    if setting == "full":
        return None  # No cropping
    elif setting == "crop-to-d2":
        d2b = config.get("d2_bounds", {})
        return (d2b["lat_min"], d2b["lat_max"], d2b["lon_min"], d2b["lon_max"])
    else:
        return None


def ingest_step(run, step, tmp_dir, out_dir, model="icon-d2", config=None):
    """Download all variables for one step, optionally crop, save as .npz."""
    out_path = os.path.join(out_dir, f"{step:03d}.npz")
    if os.path.exists(out_path):
        logger.debug(f"Step {step:03d}: already exists, skipping")
        return True

    if config is None:
        config = load_config()

    variables = get_active_variables(config)
    static_vars = get_static_variables(config)
    wind_vars, wind_levels = get_pressure_config(config)
    d2_only = get_d2_only_variables(config)
    bounds = get_bounds_for_model(model, config)

    arrays = {}
    lat_1d = lon_1d = None

    # Determine optional/skippable variables for this model
    optional_vars = set()
    if model == "icon-eu":
        eu_optional = set(config.get("eu_optional_variables", []))
        optional_vars = d2_only | eu_optional

    for var in variables:
        if model == "icon-eu" and var in d2_only:
            logger.debug(f"Step {step:03d}: {var} is D2-only, skipping for EU")
            continue

        url = build_url(run, step, var, model, config=config)
        bz2_path = os.path.join(tmp_dir, f"{var}_{step:03d}.grib2.bz2")
        grib_path = bz2_path[:-4]

        if not download(url, bz2_path):
            if var in optional_vars:
                if lat_1d is not None:
                    arrays[var] = np.zeros((len(lat_1d), len(lon_1d)), dtype=np.float32)
                    logger.debug(f"Step {step:03d}: {var} not available, using zeros")
                continue
            else:
                logger.warning(f"Step {step:03d}: {var} download failed, skipping step")
                return False

        subprocess.run(["bunzip2", "-f", bz2_path], capture_output=True)
        if not os.path.exists(grib_path):
            if var in optional_vars:
                if lat_1d is not None:
                    arrays[var] = np.zeros((len(lat_1d), len(lon_1d)), dtype=np.float32)
                continue
            logger.error(f"Step {step:03d}: {var} decompress failed")
            return False

        try:
            data, lat, lon = load_grib(grib_path, bounds)
            arrays[var] = data.astype(np.float32)
            if lat_1d is None:
                lat_1d = lat.astype(np.float32)
                lon_1d = lon.astype(np.float32)
        except Exception as e:
            if var in optional_vars:
                if lat_1d is not None:
                    arrays[var] = np.zeros((len(lat_1d), len(lon_1d)), dtype=np.float32)
                continue
            logger.error(f"Step {step:03d}: {var} parse failed: {e}")
            return False
        finally:
            if os.path.exists(grib_path):
                os.unlink(grib_path)

    if lat_1d is None:
        return False

    # Static variables (hsurf etc.) — cached per run
    for svar in static_vars:
        if model == "icon-eu" and svar in d2_only:
            continue
        svar_cache = os.path.join(out_dir, f"_static_{svar}.npy")
        if os.path.exists(svar_cache):
            arrays[svar] = np.load(svar_cache)
        else:
            url = build_url(run, 0, svar, model, config=config)
            bz2_path = os.path.join(tmp_dir, f"{svar}_000.grib2.bz2")
            grib_path = bz2_path[:-4]
            if download(url, bz2_path):
                subprocess.run(["bunzip2", "-f", bz2_path], capture_output=True)
                if os.path.exists(grib_path):
                    try:
                        data, _, _ = load_grib(grib_path, bounds)
                        arr = data.astype(np.float32)
                        arrays[svar] = arr
                        os.makedirs(out_dir, exist_ok=True)
                        np.save(svar_cache, arr)
                    except Exception as e:
                        logger.debug(f"Step {step:03d}: {svar} parse failed: {e}")
                    finally:
                        if os.path.exists(grib_path):
                            os.unlink(grib_path)

    # Pressure-level wind data
    for plev in wind_levels:
        for wind_var in wind_vars:
            url = build_url(run, step, wind_var, model, pressure_level=plev, config=config)
            bz2_path = os.path.join(tmp_dir, f"{wind_var}_{plev}hpa_{step:03d}.grib2.bz2")
            grib_path = bz2_path[:-4]

            if not download(url, bz2_path):
                logger.debug(f"Step {step:03d}: {wind_var}_{plev}hPa not available, skipping")
                continue

            subprocess.run(["bunzip2", "-f", bz2_path], capture_output=True)
            if not os.path.exists(grib_path):
                continue

            try:
                data, _, _ = load_grib(grib_path, bounds)
                arrays[f"{wind_var}_{plev}hpa"] = data.astype(np.float32)
            except Exception as e:
                logger.debug(f"Step {step:03d}: {wind_var}_{plev}hPa parse failed: {e}")
            finally:
                if os.path.exists(grib_path):
                    os.unlink(grib_path)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, lat=lat_1d, lon=lon_1d, **arrays)
    logger.info(f"Step {step:03d}: saved ({len(arrays)} vars, shape {arrays.get('ww', next(iter(arrays.values()))).shape})")
    return True


def check_new_data_available(run, model="icon-d2", config=None):
    """Quick check: is new run available AND not yet downloaded?"""
    cfg = MODEL_CONFIG[model]

    out_dir = os.path.join(DATA_DIR, model, run)
    if os.path.isdir(out_dir):
        npz_files = [f for f in os.listdir(out_dir) if f.endswith('.npz')]
        expected_count = len(cfg["steps"])
        has_local = len(npz_files) >= expected_count
    else:
        has_local = False

    # Quick HEAD check on DWD for first timestep
    first_step = cfg["steps"][0]
    test_var = "t_2m"  # reliable variable available in both models
    test_url = build_url(run, first_step, test_var, model, config=config)
    r = subprocess.run(["curl", "-sfI", test_url], capture_output=True, timeout=10)
    available_dwd = (r.returncode == 0)

    return (available_dwd, has_local)


def check_variable_urls(run, model, config):
    """Check which variables resolve (HTTP HEAD) on DWD servers. For --check-only diagnostics."""
    variables = get_active_variables(config)
    d2_only = get_d2_only_variables(config)
    cfg = MODEL_CONFIG[model]
    first_step = cfg["steps"][0]

    ok = []
    fail = []
    skipped = []

    for var in variables:
        if model == "icon-eu" and var in d2_only:
            skipped.append(var)
            continue
        url = build_url(run, first_step, var, model, config=config)
        try:
            r = subprocess.run(["curl", "-sfI", url], capture_output=True, timeout=10)
            if r.returncode == 0:
                ok.append(var)
            else:
                fail.append(var)
        except Exception:
            fail.append(var)

    # Check pressure levels
    wind_vars, wind_levels = get_pressure_config(config)
    for plev in wind_levels:
        for wv in wind_vars:
            url = build_url(run, first_step, wv, model, pressure_level=plev, config=config)
            key = f"{wv}_{plev}hpa"
            try:
                r = subprocess.run(["curl", "-sfI", url], capture_output=True, timeout=10)
                if r.returncode == 0:
                    ok.append(key)
                else:
                    fail.append(key)
            except Exception:
                fail.append(key)

    return ok, fail, skipped


def cleanup_old_runs(model, keep_runs, current_run=None):
    """Remove old runs, keeping only `keep_runs` most recent."""
    model_dir = os.path.join(DATA_DIR, model)
    if not os.path.isdir(model_dir):
        return
    runs = sorted([d for d in os.listdir(model_dir) if len(d) == 10 and d.isdigit()], reverse=True)
    for old_run in runs[keep_runs:]:
        old_path = os.path.join(model_dir, old_run)
        logger.info(f"Cleaning up old run: {model}/{old_run}")
        shutil.rmtree(old_path, ignore_errors=True)


def _merge_axis_aligned_segments(segments, eps=1e-9):
    """Merge contiguous horizontal/vertical boundary segments to reduce payload size."""
    if not segments:
        return []

    horiz = {}
    vert = {}

    def _q(x):
        return round(float(x), 8)

    for seg in segments:
        (a_lat, a_lon), (b_lat, b_lon) = seg
        a_lat, a_lon, b_lat, b_lon = float(a_lat), float(a_lon), float(b_lat), float(b_lon)
        if abs(a_lat - b_lat) <= eps:
            y = _q(a_lat)
            lo, hi = sorted((a_lon, b_lon))
            horiz.setdefault(y, []).append((lo, hi))
        elif abs(a_lon - b_lon) <= eps:
            x = _q(a_lon)
            lo, hi = sorted((a_lat, b_lat))
            vert.setdefault(x, []).append((lo, hi))

    merged = []

    for y, ivals in horiz.items():
        ivals.sort(key=lambda t: (t[0], t[1]))
        cur_lo, cur_hi = ivals[0]
        for lo, hi in ivals[1:]:
            if lo <= cur_hi + eps:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append([[float(y), cur_lo], [float(y), cur_hi]])
                cur_lo, cur_hi = lo, hi
        merged.append([[float(y), cur_lo], [float(y), cur_hi]])

    for x, ivals in vert.items():
        ivals.sort(key=lambda t: (t[0], t[1]))
        cur_lo, cur_hi = ivals[0]
        for lo, hi in ivals[1:]:
            if lo <= cur_hi + eps:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append([[cur_lo, float(x)], [cur_hi, float(x)]])
                cur_lo, cur_hi = lo, hi
        merged.append([[cur_lo, float(x)], [cur_hi, float(x)]])

    return merged


def build_d2_boundary_cache(run: str):
    """Precompute ICON-D2 valid-cell boundary once per run for fast API serving."""
    run_dir = os.path.join(DATA_DIR, "icon-d2", run)
    if not os.path.isdir(run_dir):
        return

    npz_files = sorted([f for f in os.listdir(run_dir) if f.endswith('.npz')])
    if not npz_files:
        return

    src = None
    for f in npz_files:
        p = os.path.join(run_dir, f)
        try:
            z = np.load(p)
            if 'lat' in z and 'lon' in z and 'ww' in z:
                src = z
                break
        except Exception:
            continue
    if src is None:
        return

    lat = src['lat']
    lon = src['lon']
    ww = src['ww']

    lat_res = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    lon_res = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.02
    valid = np.isfinite(ww)

    segments = []
    n_i, n_j = valid.shape
    for i in range(n_i):
        lat_lo = float(lat[i]) - lat_res / 2.0
        lat_hi = float(lat[i]) + lat_res / 2.0
        for j in range(n_j):
            if not valid[i, j]:
                continue
            lon_lo = float(lon[j]) - lon_res / 2.0
            lon_hi = float(lon[j]) + lon_res / 2.0
            if i == n_i - 1 or not valid[i + 1, j]:
                segments.append([[lat_hi, lon_lo], [lat_hi, lon_hi]])
            if i == 0 or not valid[i - 1, j]:
                segments.append([[lat_lo, lon_lo], [lat_lo, lon_hi]])
            if j == n_j - 1 or not valid[i, j + 1]:
                segments.append([[lat_lo, lon_hi], [lat_hi, lon_hi]])
            if j == 0 or not valid[i, j - 1]:
                segments.append([[lat_lo, lon_lo], [lat_hi, lon_lo]])

    segments = _merge_axis_aligned_segments(segments)

    payload = {
        'run': run,
        'latRes': lat_res,
        'lonRes': lon_res,
        'bbox': {
            'latMin': float(np.min(lat)), 'lonMin': float(np.min(lon)),
            'latMax': float(np.max(lat)), 'lonMax': float(np.max(lon)),
        },
        'cellEdgeBbox': {
            'latMin': float(np.min(lat)) - lat_res / 2.0,
            'lonMin': float(np.min(lon)) - lon_res / 2.0,
            'latMax': float(np.max(lat)) + lat_res / 2.0,
            'lonMax': float(np.max(lon)) + lon_res / 2.0,
        },
        'boundarySegments': segments,
        'validCells': int(np.count_nonzero(valid)),
        'boundarySegmentCount': len(segments),
    }
    out = os.path.join(run_dir, '_d2_boundary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    logger.info(f"D2 boundary cache written: {out} ({len(segments)} segments)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest ICON-D2/ICON-EU data for Skyview")
    parser.add_argument("run", nargs="?", default="latest",
                        help="Run ID (e.g. 2026021012) or 'latest'")
    parser.add_argument("--model", type=str, default="icon-d2", choices=["icon-d2", "icon-eu"])
    parser.add_argument("--steps", type=str, default="short",
                        help="'short', 'all', or comma-separated list")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if .npz exists")
    parser.add_argument("--check-only", action="store_true",
                        help="Fast check: exit 0 if new data available, 1 otherwise. With --verbose, also checks variable URLs.")
    parser.add_argument("--verbose", action="store_true",
                        help="With --check-only: print variable URL resolution results")
    args = parser.parse_args()

    config = load_config()
    model = args.model
    cfg = MODEL_CONFIG[model]

    run = get_latest_run(model) if args.run == "latest" else args.run

    # Fast check mode
    if args.check_only:
        available, has_local = check_new_data_available(run, model, config)

        if args.verbose:
            logger.info(f"Checking variable URLs for {model} run {run}...")
            ok, fail, skipped = check_variable_urls(run, model, config)
            logger.info(f"  OK ({len(ok)}): {', '.join(ok)}")
            if fail:
                logger.warning(f"  FAIL ({len(fail)}): {', '.join(fail)}")
            if skipped:
                logger.info(f"  SKIPPED/D2-only ({len(skipped)}): {', '.join(skipped)}")
            logger.info(f"  Total: {len(ok)} ok, {len(fail)} fail, {len(skipped)} skipped")

        if available and not has_local:
            logger.info(f"NEW: {model} {run} available, not yet downloaded")
            sys.exit(0)
        elif available and has_local:
            logger.debug(f"OK: {model} {run} already downloaded")
            sys.exit(1)
        else:
            logger.debug(f"WAIT: {model} {run} not available yet")
            sys.exit(1)

    if args.steps == "short":
        steps = cfg["short_steps"]
    elif args.steps == "all":
        steps = cfg["steps"]
    else:
        steps = [int(s) for s in args.steps.split(",")]

    variables = get_active_variables(config)
    logger.info(f"Ingesting {model} run {run}, steps {steps[0]}-{steps[-1]} ({len(steps)} total), {len(variables)} variables")

    bounds = get_bounds_for_model(model, config)
    if bounds:
        logger.info(f"  Region: crop to {bounds}")
    else:
        logger.info(f"  Region: full grid (no crop)")

    # Check if run data is available
    first_step = steps[0]
    test_url = build_url(run, first_step, "t_2m", model, config=config)
    r = subprocess.run(["curl", "-sfI", test_url], capture_output=True, timeout=15)
    if r.returncode != 0:
        logger.warning(f"Run {run} not available yet on DWD")
        sys.exit(0)

    out_dir = os.path.join(DATA_DIR, model, run)
    tmp_dir = os.path.join(DATA_DIR, "tmp", f"{model}_{run}")
    os.makedirs(tmp_dir, exist_ok=True)

    ok = 0
    for step in steps:
        if args.force:
            npz = os.path.join(out_dir, f"{step:03d}.npz")
            if os.path.exists(npz):
                os.unlink(npz)
        if ingest_step(run, step, tmp_dir, out_dir, model, config):
            ok += 1

    # Build cached D2 boundary geometry once per run (for fast frontend border rendering)
    if model == "icon-d2" and ok > 0:
        try:
            build_d2_boundary_cache(run)
        except Exception as e:
            logger.warning(f"D2 boundary cache build failed: {e}")

    # Cleanup tmp
    subprocess.run(["rm", "-rf", tmp_dir], capture_output=True)

    # Cleanup old runs — keep only latest (configurable)
    keep = config.get("retention", {}).get("keep_runs", 1)
    cleanup_old_runs(model, keep)

    logger.info(f"Done: {ok}/{len(steps)} steps ingested for {model} run {run}")


if __name__ == "__main__":
    main()
