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
import bz2
import tempfile
import warnings
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
import cfgrib
import requests as _requests
import yaml
from datetime import datetime, timedelta, timezone
import time
from logging_config import setup_logging
from constants import ICON_EU_STEP_3H_START, LOW_ZOOM_PRECOMPUTED_BINS_ENABLED
from classify import classify_clouds_and_bases
from convective_filters import filter_hbas_with_mh

logger = setup_logging(__name__, level="INFO", log_name="ingest")
# Reduce cfgrib index-cache noise like "Ignoring index file ... older than GRIB file".
try:
    import logging as _logging
    _logging.getLogger("cfgrib").setLevel(_logging.ERROR)
except Exception:
    pass

# Silence xarray/cfgrib FutureWarning about decode_timedelta('step') to keep ingest logs clean.
warnings.filterwarnings(
    "ignore",
    message=r".*will not decode the variable 'step' into a timedelta64 dtype.*",
    category=FutureWarning,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "ingest_config.yaml")

# Number of parallel download workers per step.
# Each worker downloads + bz2-decompresses one variable simultaneously.
# DWD OpenData is the real bottleneck; 6 workers saturates it without overloading
# either side. Reduce (e.g. SKYVIEW_INGEST_WORKERS=3) if you see HTTP 429s.
INGEST_WORKERS = int(os.environ.get("SKYVIEW_INGEST_WORKERS", "6"))


def _symbol_code_map() -> dict[str, int]:
    return {
        "clear": 0,
        "st": 1, "ac": 2, "ci": 3,
        "blue_thermal": 4, "cu_hum": 5, "cu_con": 6, "cb": 7,
        "fog": 20, "rime_fog": 21,
        "drizzle_light": 22, "drizzle_moderate": 23, "drizzle_dense": 24,
        "freezing_drizzle": 25, "freezing_drizzle_heavy": 26,
        "rain_slight": 27, "rain_moderate": 28, "rain_heavy": 29,
        "freezing_rain": 30,
        "snow_slight": 31, "snow_moderate": 32, "snow_heavy": 33, "snow_grains": 34,
        "rain_shower": 35, "rain_shower_moderate": 36,
        "snow_shower": 37, "snow_shower_heavy": 38,
        "thunderstorm": 39, "thunderstorm_hail": 40,
    }


def _precompute_symbol_native_fields(arrays: dict, step: int | None = None, model: str | None = None, run: str | None = None) -> tuple[bool, str]:
    required = {"ww", "ceiling", "clcl", "clcm", "clch", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "hsurf"}
    missing = sorted(required - set(arrays.keys()))
    ctx = f"{model or '?'} {run or '?'} step {step:03d}" if step is not None else f"{model or '?'} {run or '?'}"
    if missing:
        msg = f"symbol precompute skipped for {ctx}: missing required vars {missing}"
        logger.warning(msg)
        return False, msg

    ww = arrays["ww"]
    ceiling = arrays["ceiling"]
    clcl = arrays["clcl"]
    clcm = arrays["clcm"]
    clch = arrays["clch"]
    cape = arrays["cape_ml"]
    htop_dc = arrays["htop_dc"]
    hbas_sc = arrays["hbas_sc"]
    htop_sc = arrays["htop_sc"]
    lpi = arrays.get("lpi_max", np.zeros_like(ww))
    hsurf = arrays["hsurf"]

    cloud_type, cb_hm = classify_clouds_and_bases(ww, clcl, clcm, clch, cape, htop_dc, hbas_sc, htop_sc, lpi, ceiling, hsurf, arrays.get("mh"))

    sym_code = np.zeros(ww.shape, dtype=np.int16)
    m = _symbol_code_map()
    for k, v in m.items():
        if k == "clear":
            continue
        sym_code[cloud_type == k] = v

    # overwrite with significant weather symbol classes where ww indicates wx event
    ww_i = np.where(np.isfinite(ww), ww, -999).astype(np.int16)
    wx = (ww_i > 10)
    sym_code[wx & (ww_i == 96)] = m["thunderstorm_hail"]
    sym_code[wx & (ww_i >= 95) & (ww_i <= 99) & (ww_i != 96)] = m["thunderstorm"]
    sym_code[wx & (ww_i == 86)] = m["snow_shower_heavy"]
    sym_code[wx & (ww_i == 85)] = m["snow_shower"]
    sym_code[wx & ((ww_i == 81) | (ww_i == 82))] = m["rain_shower_moderate"]
    sym_code[wx & (ww_i == 80)] = m["rain_shower"]
    sym_code[wx & (ww_i == 75)] = m["snow_heavy"]
    sym_code[wx & (ww_i == 73)] = m["snow_moderate"]
    sym_code[wx & (ww_i == 71)] = m["snow_slight"]
    sym_code[wx & (ww_i == 77)] = m["snow_grains"]
    sym_code[wx & (ww_i == 65)] = m["rain_heavy"]
    sym_code[wx & (ww_i == 63)] = m["rain_moderate"]
    sym_code[wx & (ww_i == 61)] = m["rain_slight"]
    sym_code[wx & ((ww_i == 66) | (ww_i == 67))] = m["freezing_rain"]
    sym_code[wx & (ww_i == 55)] = m["drizzle_dense"]
    sym_code[wx & (ww_i == 53)] = m["drizzle_moderate"]
    sym_code[wx & (ww_i == 51)] = m["drizzle_light"]
    sym_code[wx & (ww_i == 57)] = m["freezing_drizzle_heavy"]
    sym_code[wx & (ww_i == 56)] = m["freezing_drizzle"]
    sym_code[wx & (ww_i == 45)] = m["fog"]
    sym_code[wx & (ww_i == 48)] = m["rime_fog"]

    # ranking shortcut for fast per-cell weather pick

    arrays["sym_code"] = sym_code
    arrays["cb_hm"] = cb_hm
    # Climb rate estimate from CAPE: sqrt(CAPE_ML)/10, clipped to >= 0
    cr_cape = np.where(
        np.isfinite(cape),
        np.maximum(0.0, np.sqrt(np.maximum(cape, 0.0)) / 10.0),
        np.nan,
    ).astype(np.float32)
    arrays["climb_rate_cape"] = cr_cape
    msg = f"symbol precompute ok for {ctx}: wrote sym_code/cb_hm/climb_rate_cape"
    return True, msg


def load_config():
    """Load ingest configuration from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for v in items:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _get_profile(config, profile_name):
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown ingest profile: {profile_name}. Available: {sorted(profiles.keys())}")
    return profiles[profile_name]


def resolve_profile_name(config, model: str, profile_name: str) -> str:
    """Resolve effective profile, supporting model-specific defaults.

    - explicit profile (e.g. full, skyview_d2_core, skyview_eu_core) wins
    - auto/default uses config.default_profiles[model]
    """
    p = (profile_name or "auto").strip()
    if p not in ("auto", "default"):
        _get_profile(config, p)  # validate
        return p

    defaults = config.get("default_profiles", {})
    model_key = model.replace("_", "-")
    eff = defaults.get(model_key) or defaults.get(model)
    if eff:
        _get_profile(config, eff)  # validate
        return eff

    # backward-compatible fallback
    if "full" in (config.get("profiles") or {}):
        return "full"
    raise ValueError("No default ingest profile configured and 'full' profile missing")


def get_active_variables(config, profile_name="full"):
    """Collect single-level variables, optionally profile-scoped."""
    profile = _get_profile(config, profile_name) if config.get("profiles") else None

    if profile and profile.get("variables"):
        return _dedupe_keep_order(profile.get("variables", []))

    include_groups = profile.get("include_groups") if profile else None

    variables = []
    for group_name, group in config.get("groups", {}).items():
        if include_groups is not None and group_name not in include_groups:
            continue
        if group.get("enabled", False):
            variables.extend(group.get("variables", []))

    return _dedupe_keep_order(variables)


def get_static_variables(config, profile_name="full"):
    profile = _get_profile(config, profile_name) if config.get("profiles") else None
    if profile and "static_variables" in profile:
        return profile.get("static_variables", ["hsurf"])
    return config.get("static_variables", ["hsurf"])


def get_pressure_config(config, profile_name="full"):
    profile = _get_profile(config, profile_name) if config.get("profiles") else None
    if profile and "pressure_levels" in profile:
        pc = profile.get("pressure_levels", {})
    else:
        pc = config.get("pressure_levels", {})
    return pc.get("variables", []), pc.get("levels", [])


def get_d2_only_variables(config):
    return set(config.get("d2_only_variables", []))


def get_d2_var_name(var, config):
    """Get ICON-D2 DWD filename variable name (may differ from canonical NPZ key).

    d2_variable_map maps NPZ key → DWD filename variable, e.g.:
      lpi: lpi_max   →  download lpi_max from DWD, store as 'lpi' in NPZ
    """
    d2_map = config.get("d2_variable_map", {})
    return d2_map.get(var, var)


def get_eu_var_name(var, config):
    """Get ICON-EU DWD filename variable name (may differ from canonical NPZ key)."""
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


def get_latest_run(model="icon-d2", config=None):
    """Determine latest expected run, accounting for publication delay/holdoff."""
    now = datetime.now(timezone.utc)

    default_delay_hours = 0 if model == "icon-d2" else 2
    delay_cfg = (config or {}).get("publication_delay_hours", {}) if isinstance(config, dict) else {}
    model_key = model.replace("_", "-")
    delay_hours = float(delay_cfg.get(model, delay_cfg.get(model_key, default_delay_hours)))

    ref = now - timedelta(hours=delay_hours)
    cfg = MODEL_CONFIG[model]
    hour = (ref.hour // cfg["run_interval"]) * cfg["run_interval"]
    return ref.strftime("%Y%m%d") + f"{hour:02d}"


def expected_next_run_time(model: str, from_utc: datetime | None = None) -> datetime:
    """Next expected model run timestamp in UTC (clock schedule, not DWD publish completion)."""
    now = from_utc or datetime.now(timezone.utc)
    cfg = MODEL_CONFIG[model]
    interval = int(cfg["run_interval"])
    next_hour = ((now.hour // interval) + 1) * interval
    day = now.date()
    if next_hour >= 24:
        next_hour -= 24
        day = (now + timedelta(days=1)).date()
    return datetime(day.year, day.month, day.day, next_hour, 0, 0, tzinfo=timezone.utc)


def build_url(run, step, var, model="icon-d2", pressure_level=None, config=None, profile_name="full"):
    """Build DWD download URL for regular-lat-lon data."""
    cfg = MODEL_CONFIG[model]
    run_hour = run[-2:]

    # Map canonical NPZ key → model-specific DWD filename variable
    if model == "icon-eu" and config:
        var_name = get_eu_var_name(var, config)
    elif model == "icon-d2" and config:
        var_name = get_d2_var_name(var, config)
    else:
        var_name = var

    static_vars = get_static_variables(config, profile_name=profile_name) if config else ["hsurf"]

    if var in static_vars:
        if model == "icon-d2":
            return (f"{cfg['base_url']}/{run_hour}/{var}/"
                    f"icon-d2_germany_regular-lat-lon_time-invariant_{run}_000_0_{var}.grib2.bz2")
        else:
            # ICON-EU time-invariant files use a shorter naming scheme (no _000_0 segment).
            return (f"{cfg['base_url']}/{run_hour}/{var_name}/"
                    f"icon-eu_europe_regular-lat-lon_time-invariant_{run}_{var_name.upper()}.grib2.bz2")

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


def _download_and_decompress(url: str, retries: int = 3) -> Optional[bytes]:
    """Stream-download a .bz2 URL and return decompressed GRIB bytes in memory.

    Replaces the old download() + bunzip2 subprocess pattern:
      - No .bz2 or .grib2 temp files on disk
      - No subprocess overhead for decompression
      - Returns None on persistent 404 / network / decompression failure

    Includes lightweight retries for transient DWD/network issues.
    """
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = _requests.get(url, timeout=120, stream=True)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.debug(f"HTTP {resp.status_code} fetching {url} (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(0.6 * attempt)
                    continue
                return None
            decomp = bz2.BZ2Decompressor()
            parts: List[bytes] = []
            for chunk in resp.iter_content(chunk_size=131072):  # 128 KB chunks
                if chunk:
                    parts.append(decomp.decompress(chunk))
            return b"".join(parts)
        except Exception as e:
            logger.debug(f"_download_and_decompress failed ({url}) attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(0.6 * attempt)
                continue
            return None
    return None


def _parse_grib_from_bytes(grib_bytes: bytes, bounds=None) -> Tuple:
    """Write GRIB bytes to a NamedTemporaryFile and parse with cfgrib.

    The temp file is deleted immediately after parsing regardless of success/failure.
    Raises the same exceptions as load_grib() on parse errors.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(grib_bytes)
        return load_grib(tmp_path, bounds)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def download(url, dest):
    """Download file to disk via curl. Used for HEAD checks in availability probes."""
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


MULTI_MESSAGE_HOURLY_SELECT_VARS = {"cape_ml", "cin_ml", "hbas_sc", "htop_sc", "lpi"}
# Quarter-hour substeps are most useful for short-range nowcast/overlay work.
# Keep later D2 steps on the cheaper hourly path to limit ingest overhead.
D2_SUBSTEP_MAX_STEP = int(os.environ.get("SKYVIEW_D2_SUBSTEP_MAX_STEP", "24"))


def _guess_var_name_from_path(filepath: str) -> str | None:
    name = os.path.basename(filepath)
    m = re.search(r'_\d{3}_(?:2d_)?([A-Za-z0-9_]+)\.grib2$', name)
    if not m:
        return None
    return m.group(1).lower()


def _extract_nominal_hour_from_filename(filepath: str) -> int | None:
    name = os.path.basename(filepath)
    m = re.search(r'_(\d{3})_(?:2d_)?[A-Za-z0-9_]+\.grib2$', name)
    if not m:
        return None
    return int(m.group(1))


def _extract_expected_valid_datetime_from_filename(filepath: str):
    """Return expected valid datetime from DWD filename when available.

    Example:
      icon-d2_..._2026031103_005_2d_cape_ml.grib2 -> 2026-03-11 08:00 UTC
    """
    name = os.path.basename(filepath)
    m = re.search(r'_(\d{10})_(\d{3})_(?:2d_)?[A-Za-z0-9_]+\.grib2$', name)
    if not m:
        return None
    try:
        run_dt = datetime.strptime(m.group(1), "%Y%m%d%H")
        step_h = int(m.group(2))
        return run_dt + timedelta(hours=step_h)
    except Exception:
        return None


def _first_scalar_like(v):
    arr = np.asarray(v)
    if arr.size == 0:
        return None
    if arr.size == 1:
        try:
            return arr.item()
        except Exception:
            return arr.reshape(-1)[0]
    return arr.reshape(-1)[0]


def _dataset_valid_datetime(ds):
    if "valid_time" in ds.coords:
        raw = ds.coords["valid_time"].values
        try:
            if isinstance(raw, np.datetime64):
                return raw.astype("datetime64[m]").astype(object)
            if isinstance(raw, np.ndarray) and raw.shape == () and np.issubdtype(raw.dtype, np.datetime64):
                return raw.astype("datetime64[m]").astype(object)
        except Exception:
            pass

        v = _first_scalar_like(raw)
        if v is not None:
            try:
                return np.datetime64(v).astype("datetime64[m]").astype(object)
            except Exception:
                pass
            text = str(v)
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(text, fmt)
                except Exception:
                    pass
    return None


def _dataset_valid_hour_and_minute(ds) -> tuple[int | None, int | None]:
    dt = _dataset_valid_datetime(ds)
    if dt is not None:
        return int(dt.hour), int(dt.minute)
    if "step" in ds.coords:
        v = _first_scalar_like(ds.coords["step"].values)
        text = str(v)
        m = re.search(r'(\d+)', text)
        if m:
            total_minutes = int(m.group(1)) // 60000000000 if 'nanoseconds' in text else None
            if total_minutes is not None:
                return total_minutes // 60, total_minutes % 60
    return None, None


def _spatial_datasets(datasets):
    return [d for d in datasets if "latitude" in d.coords and "longitude" in d.coords and d.data_vars]


def _select_spatial_dataset_with_reason(
    datasets,
    filepath: str,
    var_name_hint: str | None = None,
    nominal_hour_hint: int | None = None,
    expected_valid_dt_hint=None,
):
    spatial = _spatial_datasets(datasets)
    if not spatial:
        return None, "no_spatial"

    var_name = var_name_hint or _guess_var_name_from_path(filepath)
    nominal_hour = nominal_hour_hint if nominal_hour_hint is not None else _extract_nominal_hour_from_filename(filepath)
    expected_valid_dt = expected_valid_dt_hint or _extract_expected_valid_datetime_from_filename(filepath)
    if var_name not in MULTI_MESSAGE_HOURLY_SELECT_VARS or nominal_hour is None:
        return spatial[0], "default_first"

    exact_valid = []
    exact_hour = []
    zero_minute = []
    for d in spatial:
        dt = _dataset_valid_datetime(d)
        hour, minute = _dataset_valid_hour_and_minute(d)
        if minute == 0:
            zero_minute.append(d)
            if expected_valid_dt is not None and dt is not None and dt == expected_valid_dt:
                exact_valid.append(d)
            if hour == nominal_hour:
                exact_hour.append(d)

    if exact_valid:
        return exact_valid[0], "exact_valid"
    if exact_hour:
        return exact_hour[0], "fallback_exact_hour"
    if zero_minute:
        return zero_minute[0], "fallback_zero_minute"
    return spatial[0], "fallback_first_spatial"


def _select_spatial_dataset(datasets, filepath: str, var_name_hint: str | None = None, nominal_hour_hint: int | None = None):
    ds, _reason = _select_spatial_dataset_with_reason(
        datasets,
        filepath,
        var_name_hint=var_name_hint,
        nominal_hour_hint=nominal_hour_hint,
    )
    return ds


def _reduce_dataset_to_2d(ds, filepath: str):
    var = list(ds.data_vars.values())[0]
    lat = ds.coords["latitude"].values
    lon = ds.coords["longitude"].values

    if lat.ndim > 1:
        lat = lat[:, 0]
    if lon.ndim > 1:
        lon = lon[0, :]

    data = np.squeeze(var.values)
    while data.ndim > 2:
        data = data[0]

    if data.shape != (len(lat), len(lon)):
        raise ValueError(
            f"Shape mismatch after reduction: data={data.shape}, "
            f"lat={len(lat)}, lon={len(lon)} in {filepath}"
        )
    return data, lat, lon


def _crop_field(data, lat, lon, bounds=None):
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        eps = 0.001
        lat_mask = (lat >= lat_min - eps) & (lat <= lat_max + eps)
        lon_mask = (lon >= lon_min - eps) & (lon <= lon_max + eps)
        return data[np.ix_(lat_mask, lon_mask)], lat[lat_mask], lon[lon_mask]
    return data, lat, lon


def load_grib(filepath, bounds=None):
    """Load GRIB2, optionally crop to bounds, return (data_2d, lat_1d, lon_1d).

    cfgrib ≥ 0.9.14 compatibility notes vs 0.9.10.x:
    - open_datasets() may return multiple datasets; we search for the first with
      latitude/longitude coords rather than blindly taking [0].
    - For D2 multi-message quarter-hour files (`cape_ml`, `cin_ml`, `hbas_sc`,
      `htop_sc`, `lpi`), we explicitly select the on-the-hour message matching
      the nominal forecast step from the filename (e.g. `_005_` -> 05:00),
      instead of accidentally ingesting whichever slice cfgrib returns first.
    - step may be an explicit dimension (not squeezed) for accumulated fields
      (tot_prec, rain_*, snow_*, grau_gsp, h_snow, freshsnw, prr/prs/prg_gsp)
      and time-range statistics (vmax_10m, tmax/tmin_2m, lpi_max, dbz_*, etc.).
      We reduce any residual leading dims via data[0] rather than data[-1].
    - alternativeRowScanning is now handled correctly; lat/lon stay 1-D for
      regular-lat-lon grids but ordering may differ — we sort ascending if needed.
    - Non-hourly steps (ICON-EU 81-120h, 3h intervals) parse correctly since 0.9.12.
    """
    datasets = cfgrib.open_datasets(filepath)
    if not datasets:
        raise ValueError(f"No datasets in {filepath}")

    ds, _selection_reason = _select_spatial_dataset_with_reason(datasets, filepath)
    if ds is None:
        raise ValueError(f"No spatial dataset found in {filepath}")

    data, lat, lon = _reduce_dataset_to_2d(ds, filepath)
    return _crop_field(data, lat, lon, bounds)


def load_grib_with_substeps(
    filepath,
    bounds=None,
    var_name_hint: str | None = None,
    nominal_hour_hint: int | None = None,
    expected_valid_dt_hint=None,
):
    """Load GRIB2 and also return quarter-hour stacked substeps when present.

    Returns (data_2d, lat_1d, lon_1d, substeps_3d|None, minutes_list|None)
    where substeps_3d has shape (N, lat, lon).
    """
    datasets = cfgrib.open_datasets(filepath)
    if not datasets:
        raise ValueError(f"No datasets in {filepath}")

    spatial = _spatial_datasets(datasets)
    if not spatial:
        raise ValueError(f"No spatial dataset found in {filepath}")

    selected, selection_reason = _select_spatial_dataset_with_reason(
        datasets,
        filepath,
        var_name_hint=var_name_hint,
        nominal_hour_hint=nominal_hour_hint,
        expected_valid_dt_hint=expected_valid_dt_hint,
    )
    data, lat, lon = _reduce_dataset_to_2d(selected, filepath)
    data, lat, lon = _crop_field(data, lat, lon, bounds)

    var_name = var_name_hint or _guess_var_name_from_path(filepath)
    nominal_hour = nominal_hour_hint if nominal_hour_hint is not None else _extract_nominal_hour_from_filename(filepath)
    expected_valid_dt = expected_valid_dt_hint or _extract_expected_valid_datetime_from_filename(filepath)
    if var_name not in MULTI_MESSAGE_HOURLY_SELECT_VARS or nominal_hour is None:
        return data, lat, lon, None, None

    substep_entries = []
    for ds in spatial:
        dt = _dataset_valid_datetime(ds)
        hour, minute = _dataset_valid_hour_and_minute(ds)
        if minute is None:
            continue
        if expected_valid_dt is not None and dt is not None:
            if (dt.year, dt.month, dt.day, dt.hour) != (
                expected_valid_dt.year,
                expected_valid_dt.month,
                expected_valid_dt.day,
                expected_valid_dt.hour,
            ):
                continue
        elif hour != nominal_hour:
            continue
        s_data, s_lat, s_lon = _reduce_dataset_to_2d(ds, filepath)
        s_data, s_lat, s_lon = _crop_field(s_data, s_lat, s_lon, bounds)
        if s_data.shape != data.shape:
            continue
        substep_entries.append((int(minute), s_data))

    if not substep_entries:
        if selection_reason != "exact_valid":
            logger.warning(
                "Substep selection fallback for %s (%s) produced no quarter-hour slices",
                os.path.basename(filepath),
                selection_reason,
            )
        return data, lat, lon, None, None

    substep_entries.sort(key=lambda item: item[0])
    minutes = [m for m, _ in substep_entries]
    stacked = np.stack([arr for _, arr in substep_entries], axis=0)
    if selection_reason != "exact_valid":
        logger.warning(
            "Substep selection fallback for %s used %s with minutes=%s",
            os.path.basename(filepath),
            selection_reason,
            minutes,
        )
    return data, lat, lon, stacked, minutes


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


def ingest_step(run, step, tmp_dir, out_dir, model="icon-d2", config=None, profile_name="full",
                prev_acc=None, prev_step=None):
    """Download all variables for one step, optionally crop, save as .npz.

    tmp_dir is accepted for backward-compat but no longer used: all downloads and
    decompression are now handled in memory.

    Uses an interleaved parallel download + sequential parse approach:
      Downloads — INGEST_WORKERS threads fetch and bz2-decompress variable URLs concurrently.
                  No .bz2 or .grib2 temp files touch disk.
      Parsing   — The main thread parses each result with cfgrib as soon as its future
                  completes. cfgrib stays single-threaded (eccodes C library).

    Precip rates (tp_rate, rain_rate, snow_rate, hail_rate) are computed inline and written into
    the same .npz, avoiding a separate post-processing read+rewrite pass.

    Args:
        prev_acc: dict of raw precip accumulation arrays from the previous step
                  (keys: tot_prec, rain_gsp, rain_con, snow_gsp, snow_con, grau_gsp).
        prev_step: step number that prev_acc corresponds to.

    Returns:
        (success: bool, curr_acc: dict | None)
        curr_acc holds the current step's raw precip accumulations for the next iteration.
        curr_acc is None when the step was skipped (already existed) or had no precip data.
    """
    out_path = os.path.join(out_dir, f"{step:03d}.npz")
    if os.path.exists(out_path):
        logger.debug(f"Step {step:03d}: already exists, skipping")
        return True, None

    if config is None:
        config = load_config()

    variables = get_active_variables(config, profile_name=profile_name)
    static_vars = get_static_variables(config, profile_name=profile_name)
    wind_vars, wind_levels = get_pressure_config(config, profile_name=profile_name)
    d2_only = get_d2_only_variables(config)
    bounds = get_bounds_for_model(model, config)

    # Determine optional/skippable variables for this model
    optional_vars: set = set()
    if model == "icon-eu":
        eu_optional = set(config.get("eu_optional_variables", []))
        # DWD sometimes publishes lpi_con_max incompletely for specific runs/steps.
        # Keep timestep ingesting and backfill zeros when missing.
        eu_optional.add("lpi_max")
        optional_vars = d2_only | eu_optional

    # ── Build download task lists ─────────────────────────────────────────────
    # Each task: (array_key, url, is_optional)
    # array_key is the canonical NPZ key (may differ from the DWD variable name
    # for mapped variables like lpi→lpi_max handled inside build_url).

    var_tasks: List[Tuple[str, str, bool]] = []
    for var in variables:
        if model == "icon-eu" and var in d2_only:
            logger.debug(f"Step {step:03d}: {var} is D2-only, skipping for EU")
            continue
        url = build_url(run, step, var, model, config=config, profile_name=profile_name)
        var_tasks.append((var, url, var in optional_vars))

    plev_tasks: List[Tuple[str, str, bool]] = []
    for plev in wind_levels:
        for wind_var in wind_vars:
            url = build_url(run, step, wind_var, model, pressure_level=plev,
                            config=config, profile_name=profile_name)
            plev_tasks.append((f"{wind_var}_{plev}hpa", url, True))  # pressure wind always optional

    all_tasks = var_tasks + plev_tasks

    # ── Interleaved parallel download + sequential parse ──────────────────────
    # Downloads run concurrently in a thread pool; the main thread parses each
    # result as soon as its future completes — so at most INGEST_WORKERS
    # decompressed GRIBs (~5 MB each) are held in memory at any one time.
    # cfgrib parsing stays on the main thread (eccodes C library is not thread-safe).
    #
    # Memory profile (D2, 68 vars, 6 workers):
    #   Old sequential:  ~10 MB peak in-flight  (1 GRIB at a time)
    #   Previous impl:  ~360 MB peak (held all decompressed bytes before parsing)
    #   This impl:       ~30 MB peak in-flight  (6 GRIBs × 5 MB)
    #   + NumPy arrays grow to ~245 MB by end of step (unavoidable — that's the data)

    workers = INGEST_WORKERS
    arrays: Dict[str, np.ndarray] = {}
    lat_1d = lon_1d = None
    out_tmp_path = os.path.join(out_dir, f".{step:03d}.tmp.npz")

    # Track which required variables failed so we can abort after the pool drains.
    failed_required: Optional[str] = None

    # var_tasks is ordered; we need to honour required-var failure semantics after
    # the parallel phase.  Build a lookup for quick metadata access by key.
    task_meta: Dict[str, Tuple[str, bool]] = {
        key: (url, is_opt) for key, url, is_opt in all_tasks
    }
    # Separate required vars from optional/pressure — only required failures abort.
    required_var_keys = {key for key, _url, is_opt in var_tasks if not is_opt}
    var_task_keys = {key for key, _url, _is_opt in var_tasks}

    def _process_future(key: str, grib_bytes: Optional[bytes]) -> bool:
        """Parse one result on the calling (main) thread. Returns False to abort step."""
        nonlocal lat_1d, lon_1d, failed_required

        is_optional = key not in required_var_keys

        if grib_bytes is None:
            if is_optional:
                logger.debug(f"Step {step:03d}: optional {key} not available, skipping")
                return True
            url = task_meta.get(key, ("", False))[0]
            logger.warning(f"Step {step:03d}: {key} download failed ({url}), skipping step")
            failed_required = key
            return False

        try:
            if model == "icon-d2" and key in MULTI_MESSAGE_HOURLY_SELECT_VARS and int(step) <= D2_SUBSTEP_MAX_STEP:
                fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(grib_bytes)
                    data, lat, lon, substeps, minutes = load_grib_with_substeps(
                        tmp_path,
                        bounds,
                        var_name_hint=key,
                        nominal_hour_hint=int(step),
                        expected_valid_dt_hint=datetime.strptime(str(run), "%Y%m%d%H") + timedelta(hours=int(step)),
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                arrays[key] = data.astype(np.float32)
                if substeps is not None and minutes:
                    arrays[f"{key}_substeps"] = substeps.astype(np.float32)
                    arrays[f"{key}_substep_minutes"] = np.asarray(minutes, dtype=np.int16)
                    logger.debug(f"Step {step:03d}: {key} substeps extracted minutes={minutes}")
                else:
                    logger.warning(f"Step {step:03d}: {key} expected quarter-hour substeps but none were extracted")
            else:
                data, lat, lon = _parse_grib_from_bytes(grib_bytes, bounds)
                arrays[key] = data.astype(np.float32)
            if lat_1d is None and key in var_task_keys:
                lat_1d = lat.astype(np.float32)
                lon_1d = lon.astype(np.float32)
        except Exception as e:
            if is_optional:
                logger.debug(f"Step {step:03d}: optional {key} parse failed ({e}), skipping")
                return True
            logger.error(f"Step {step:03d}: {key} parse failed: {e}")
            failed_required = key
            return False

        return True

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(_download_and_decompress, url): key
                for key, url, _ in all_tasks
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    grib_bytes = future.result()
                except Exception as exc:
                    logger.debug(f"Step {step:03d}: download future for {key} raised: {exc}")
                    grib_bytes = None
                # Parse immediately — bytes go out of scope here after this call
                if not _process_future(key, grib_bytes):
                    # Cancel pending futures for required-var failure; executor drains gracefully
                    for f in future_to_key:
                        f.cancel()
                    break
    else:
        # Sequential path (INGEST_WORKERS=1)
        for key, url, _ in all_tasks:
            if not _process_future(key, _download_and_decompress(url)):
                break

    if failed_required is not None:
        return False, None

    if lat_1d is None:
        return False, None

    # EU lpi_max (mapped to lpi_con_max) may be missing for some runs/steps on DWD.
    # Ensure downstream APIs still get a deterministic field.
    if model == "icon-eu" and ("lpi_max" in variables) and ("lpi_max" not in arrays):
        arrays["lpi_max"] = np.zeros((len(lat_1d), len(lon_1d)), dtype=np.float32)
        if _eu_lpi_max_expected_for_step(step):
            logger.warning(f"Step {step:03d}: lpi_max missing on EU source for expected step — filled with zeros")
        else:
            logger.debug(f"Step {step:03d}: lpi_max not scheduled on EU source cadence — filled with zeros")

    # ── Static variables (hsurf etc.) — cached per run ───────────────────────
    for svar in static_vars:
        if model == "icon-eu" and svar in d2_only:
            continue
        svar_cache = os.path.join(out_dir, f"_static_{svar}.npy")
        if os.path.exists(svar_cache):
            arrays[svar] = np.load(svar_cache)
        else:
            url = build_url(run, 0, svar, model, config=config, profile_name=profile_name)
            grib_bytes = _download_and_decompress(url)
            if grib_bytes is not None:
                try:
                    data, _, _ = _parse_grib_from_bytes(grib_bytes, bounds)
                    arr = data.astype(np.float32)
                    arrays[svar] = arr
                    os.makedirs(out_dir, exist_ok=True)
                    static_tmp = os.path.join(out_dir, f"._static_{svar}.npy.tmp")
                    with open(static_tmp, "wb") as f:
                        np.save(f, arr)
                    os.replace(static_tmp, svar_cache)
                except Exception as e:
                    logger.debug(f"Step {step:03d}: static {svar} parse failed: {e}")
                    try:
                        if 'static_tmp' in locals() and os.path.exists(static_tmp):
                            os.unlink(static_tmp)
                    except OSError:
                        pass

    # ── Inline precip rate computation ────────────────────────────────────────
    # De-accumulate precipitation and compute mm/h rates in the same pass,
    # so we never need a separate post-processing read+rewrite of the .npz.
    curr_acc = None
    if 'tot_prec' in arrays or any(k in arrays for k in ('rain_gsp', 'rain_con', 'snow_gsp', 'snow_con', 'grau_gsp')):
        shape = arrays.get('tot_prec', next(iter(arrays.values()))).shape
        zeros = np.zeros(shape, dtype=np.float32)

        tp = arrays.get('tot_prec', zeros)
        rg = arrays.get('rain_gsp', zeros)
        rc = arrays.get('rain_con', zeros)
        sg = arrays.get('snow_gsp', zeros)
        sc = arrays.get('snow_con', zeros)
        gg = arrays.get('grau_gsp', zeros)

        prev_step_expected, dt_h = _precip_prev_step_and_dt(model, step)
        use_prev = (prev_acc is not None and prev_step == prev_step_expected)

        if use_prev:
            tp_rate   = (tp - prev_acc['tot_prec']) / dt_h
            rain_rate = ((rg + rc) - (prev_acc['rain_gsp'] + prev_acc['rain_con'])) / dt_h
            snow_rate = ((sg + sc) - (prev_acc['snow_gsp'] + prev_acc['snow_con'])) / dt_h
            hail_rate = (gg - prev_acc['grau_gsp']) / dt_h
        else:
            tp_rate   = tp / dt_h
            rain_rate = (rg + rc) / dt_h
            snow_rate = (sg + sc) / dt_h
            hail_rate = gg / dt_h

        arrays['tp_rate']   = np.maximum(tp_rate,   0.0).astype(np.float32)
        arrays['rain_rate'] = np.maximum(rain_rate, 0.0).astype(np.float32)
        arrays['snow_rate'] = np.maximum(snow_rate, 0.0).astype(np.float32)
        arrays['hail_rate'] = np.maximum(hail_rate, 0.0).astype(np.float32)

        curr_acc = {
            'tot_prec': tp, 'rain_gsp': rg, 'rain_con': rc,
            'snow_gsp': sg, 'snow_con': sc, 'grau_gsp': gg,
        }

    # EU sanity filter: suppress implausible convective cloud-base spikes
    # using boundary-layer depth (mh) consistency.
    hbas_filter_cfg = (config or {}).get("hbas_filter", {}) if config else {}
    hbas_filter_enabled = bool(hbas_filter_cfg.get("enabled", True))
    hbas_filter_models = set(hbas_filter_cfg.get("models", ["icon-eu"]))
    if (
        hbas_filter_enabled
        and model in hbas_filter_models
        and ("hbas_sc" in arrays)
        and ("hsurf" in arrays)
        and ("mh" in arrays)
    ):
        margin_m = float(hbas_filter_cfg.get("margin_m", 500.0))
        hard_cap_agl_m = float(hbas_filter_cfg.get("hard_cap_agl_m", 6500.0))
        hbas_filtered, reject_mask, _quality = filter_hbas_with_mh(
            arrays["hbas_sc"],
            arrays["hsurf"],
            arrays["mh"],
            margin_m=margin_m,
            hard_cap_agl_m=hard_cap_agl_m,
            return_quality=True,
        )
        rejected = int(np.count_nonzero(reject_mask))
        if rejected > 0:
            frac = (rejected / reject_mask.size) * 100.0
            logger.debug(
                f"Step {step:03d}: hbas_sc sanity filter rejected {rejected}/{reject_mask.size} cells ({frac:.2f}%)"
            )
        arrays["hbas_sc"] = hbas_filtered.astype(np.float32)

    _precompute_symbol_native_fields(arrays, step=step, model=model, run=run)

    os.makedirs(out_dir, exist_ok=True)
    try:
        np.savez_compressed(out_tmp_path, lat=lat_1d, lon=lon_1d, **arrays)
        os.replace(out_tmp_path, out_path)
    except Exception:
        try:
            if os.path.exists(out_tmp_path):
                os.unlink(out_tmp_path)
        except OSError:
            pass
        raise
    logger.debug(f"Step {step:03d}: saved ({len(arrays)} vars, shape {arrays.get('ww', next(iter(arrays.values()))).shape})")
    return True, curr_acc


def _availability_probe_target(cfg: dict, variables: list[str]) -> tuple[int, str]:
    """Use latest expected step and a late-published variable (prefer mh) for run-availability checks."""
    probe_step = int(cfg["steps"][-1])
    probe_var = "mh" if "mh" in variables else (variables[0] if variables else "t_2m")
    return probe_step, probe_var


def check_new_data_available(run, model="icon-d2", config=None, profile_name="full"):
    """Quick check: is new run available AND not yet downloaded?"""
    cfg = MODEL_CONFIG[model]

    out_dir = os.path.join(DATA_DIR, model, run)
    if os.path.isdir(out_dir):
        npz_files = [f for f in os.listdir(out_dir) if f.endswith('.npz')]
        expected_count = len(cfg["steps"])
        has_local = len(npz_files) >= expected_count
    else:
        has_local = False

    # Quick HEAD check on DWD for latest expected timestep.
    active_vars = get_active_variables(config, profile_name=profile_name) if config else []
    probe_step, probe_var = _availability_probe_target(cfg, active_vars)
    test_url = build_url(run, probe_step, probe_var, model, config=config, profile_name=profile_name)
    r = subprocess.run(["curl", "-sfI", test_url], capture_output=True, timeout=10)
    available_dwd = (r.returncode == 0)

    return (available_dwd, has_local)


def check_variable_urls(run, model, config, profile_name="full"):
    """Check which variables resolve (HTTP HEAD) on DWD servers. For --check-only diagnostics."""
    variables = get_active_variables(config, profile_name=profile_name)
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
        url = build_url(run, first_step, var, model, config=config, profile_name=profile_name)
        try:
            r = subprocess.run(["curl", "-sfI", url], capture_output=True, timeout=10)
            if r.returncode == 0:
                ok.append(var)
            else:
                fail.append(var)
        except Exception:
            fail.append(var)

    # Check pressure levels
    wind_vars, wind_levels = get_pressure_config(config, profile_name=profile_name)
    for plev in wind_levels:
        for wv in wind_vars:
            url = build_url(run, first_step, wv, model, pressure_level=plev, config=config, profile_name=profile_name)
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


def _load_precip_acc_from_step(run_dir: str, step: int) -> dict | None:
    """Load raw precip accumulation arrays from an existing step .npz file.

    Used to seed prev_acc when --fill-missing processes non-contiguous steps or
    when ingest resumes after a partial run.
    Returns None if the file doesn't exist or can't be read.
    """
    path = os.path.join(run_dir, f"{step:03d}.npz")
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        if 'lat' not in z.files or 'lon' not in z.files:
            return None
        shape = (len(z['lat']), len(z['lon']))
        zeros = np.zeros(shape, dtype=np.float32)
        return {k: z[k] if k in z.files else zeros.copy()
                for k in ('tot_prec', 'rain_gsp', 'rain_con', 'snow_gsp', 'snow_con', 'grau_gsp')}
    except Exception:
        return None


def _precip_prev_step_and_dt(model: str, step: int) -> tuple[int | None, float]:
    if model == "icon-eu" and step >= ICON_EU_STEP_3H_START:
        prev = step - 3
        return (prev if prev >= 1 else None), 3.0
    if step >= 2:
        return step - 1, 1.0
    return None, 1.0


def _eu_lpi_max_expected_for_step(step: int) -> bool:
    """ICON-EU lpi_con_max availability pattern (mapped to lpi_max).

    Observed DWD cadence:
      - 0..48: hourly
      - 51..72: 3-hourly
      - 78..120: 6-hourly
    Ingest uses forecast steps starting at 1.
    """
    if 1 <= step <= 48:
        return True
    if 51 <= step <= 72 and (step % 3 == 0):
        return True
    if 78 <= step <= 120 and (step % 6 == 0):
        return True
    return False


def precompute_precip_rates_for_run(model: str, run: str):
    """Backfill de-accumulated precip rate fields for an already-ingested run.

    This is a repair/migration utility for runs that were ingested before the
    inline rate computation was added to ingest_step(). New ingests compute
    rates on the fly and do not need to call this function.

    Rewrites each .npz with added fields:
      - tp_rate, rain_rate, snow_rate, hail_rate  (mm/h-equivalent)
    """
    run_dir = os.path.join(DATA_DIR, model, run)
    if not os.path.isdir(run_dir):
        return

    npz_files = sorted([f for f in os.listdir(run_dir) if f.endswith('.npz')])
    if not npz_files:
        return

    prev_acc = None
    prev_step_seen = None

    for fname in npz_files:
        step = int(fname[:-4])
        path = os.path.join(run_dir, fname)
        z = np.load(path)
        arrays = {k: z[k] for k in z.files}

        lat = arrays.get('lat')
        lon = arrays.get('lon')
        if lat is None or lon is None:
            continue

        shape = (len(lat), len(lon))
        zeros = np.zeros(shape, dtype=np.float32)

        tp = arrays.get('tot_prec', zeros)
        rg = arrays.get('rain_gsp', zeros)
        rc = arrays.get('rain_con', zeros)
        sg = arrays.get('snow_gsp', zeros)
        sc = arrays.get('snow_con', zeros)
        gg = arrays.get('grau_gsp', zeros)

        prev_step_expected, dt_h = _precip_prev_step_and_dt(model, step)
        use_prev = prev_acc is not None and prev_step_seen == prev_step_expected

        if use_prev:
            tp_rate = (tp - prev_acc['tot_prec']) / dt_h
            rain_rate = ((rg + rc) - (prev_acc['rain_gsp'] + prev_acc['rain_con'])) / dt_h
            snow_rate = ((sg + sc) - (prev_acc['snow_gsp'] + prev_acc['snow_con'])) / dt_h
            hail_rate = (gg - prev_acc['grau_gsp']) / dt_h
        else:
            tp_rate = tp / dt_h
            rain_rate = (rg + rc) / dt_h
            snow_rate = (sg + sc) / dt_h
            hail_rate = gg / dt_h

        tp_rate = np.maximum(tp_rate, 0.0).astype(np.float32)
        rain_rate = np.maximum(rain_rate, 0.0).astype(np.float32)
        snow_rate = np.maximum(snow_rate, 0.0).astype(np.float32)
        hail_rate = np.maximum(hail_rate, 0.0).astype(np.float32)

        arrays['tp_rate'] = tp_rate
        arrays['rain_rate'] = rain_rate
        arrays['snow_rate'] = snow_rate
        arrays['hail_rate'] = hail_rate

        tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

        prev_acc = {
            'tot_prec': tp,
            'rain_gsp': rg,
            'rain_con': rc,
            'snow_gsp': sg,
            'snow_con': sc,
            'grau_gsp': gg,
        }
        prev_step_seen = step

    logger.info(f"Precomputed precip rates for {model} run {run} ({len(npz_files)} steps)")


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

    lat_res = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.02
    lon_res = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.02

    # Boundary mask: use hsurf (terrain elevation) to identify valid D2 domain cells.
    # hsurf is always defined wherever the D2 grid has a valid point (only NaN outside domain).
    # Avoid mh (mixing height = NaN at night/stable conditions) and ww (NaN with no weather
    # event) — both have meteorological NaN that pulls the border inward beyond domain edges.
    # Fall back to ww, then all-valid if hsurf is unavailable.
    if 'hsurf' in src.files:
        valid = np.isfinite(src['hsurf'])
    elif 'ww' in src.files:
        valid = np.isfinite(src['ww'])
    else:
        valid = np.ones((len(lat), len(lon)), dtype=bool)

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
    parser.add_argument("--fill-missing", action="store_true",
                        help="Ingest only steps not yet on disk for this run. Defaults to --steps all if --steps not set explicitly.")
    parser.add_argument("--check-only", action="store_true",
                        help="Fast check: exit 0 if new data available, 1 otherwise. With --verbose, also checks variable URLs.")
    parser.add_argument("--verbose", action="store_true",
                        help="With --check-only: print variable URL resolution results")
    parser.add_argument("--profile", type=str, default="auto",
                        help="Ingest profile from ingest_config.yaml (e.g. auto, full, skyview_d2_core, skyview_eu_core)")
    args = parser.parse_args()

    config = load_config()
    model = args.model
    cfg = MODEL_CONFIG[model]
    profile_name = resolve_profile_name(config, model, args.profile)

    run = get_latest_run(model, config=config) if args.run == "latest" else args.run

    # Fast check mode
    if args.check_only:
        available, has_local = check_new_data_available(run, model, config, profile_name=profile_name)

        if args.verbose:
            logger.info(f"Checking variable URLs for {model} run {run} (profile={profile_name})...")
            ok, fail, skipped = check_variable_urls(run, model, config, profile_name=profile_name)
            logger.info(f"  OK ({len(ok)}): {', '.join(ok)}")
            if fail:
                logger.warning(f"  FAIL ({len(fail)}): {', '.join(fail)}")
            if skipped:
                logger.info(f"  SKIPPED/D2-only ({len(skipped)}): {', '.join(skipped)}")
            logger.info(f"  Total: {len(ok)} ok, {len(fail)} fail, {len(skipped)} skipped")

        nxt = expected_next_run_time(model)
        nxt_iso = nxt.isoformat().replace("+00:00", "Z")


        if available and not has_local:
            logger.info(f"NEW: {model} {run} available, not yet downloaded")
            logger.debug(f"Next expected {model} run time (UTC): {nxt_iso}")
            sys.exit(0)
        elif available and has_local:
            logger.debug(f"NOOP: ingest check ran, no new data for {model}. Current run {run} already downloaded.")
            logger.debug(f"Next expected {model} run time (UTC): {nxt_iso}")
            sys.exit(1)
        else:
            logger.debug(f"NOOP: ingest check ran, no new data for {model}. DWD run {run} not yet available.")
            logger.debug(f"Next expected {model} run time (UTC): {nxt_iso}")
            sys.exit(1)

    # --fill-missing implies "all" steps as the target set unless --steps was given explicitly
    if args.steps == "short" and args.fill_missing:
        steps = cfg["steps"]
    elif args.steps == "short":
        steps = cfg["short_steps"]
    elif args.steps == "all":
        steps = cfg["steps"]
    else:
        steps = [int(s) for s in args.steps.split(",")]

    variables = get_active_variables(config, profile_name=profile_name)
    logger.info(f"Ingesting {model} run {run}, profile={profile_name}, steps {steps[0]}-{steps[-1]} ({len(steps)} total), {len(variables)} variables")

    bounds = get_bounds_for_model(model, config)
    if bounds:
        logger.debug(f"Region: crop to {bounds}")
    else:
        logger.debug("Region: full grid (no crop)")

    # Check if run data is available (latest expected step; prefer late var 'mh').
    probe_step, probe_var = _availability_probe_target(cfg, variables)
    test_url = build_url(run, probe_step, probe_var, model, config=config, profile_name=profile_name)
    r = subprocess.run(["curl", "-sfI", test_url], capture_output=True, timeout=15)
    if r.returncode != 0:
        logger.warning(f"Run {run} not available yet on DWD")
        sys.exit(0)

    out_dir = os.path.join(DATA_DIR, model, run)
    tmp_dir = os.path.join(DATA_DIR, "tmp", f"{model}_{run}")
    os.makedirs(tmp_dir, exist_ok=True)

    if args.fill_missing:
        existing = {
            int(os.path.splitext(f)[0])
            for f in (os.listdir(out_dir) if os.path.isdir(out_dir) else [])
            if f.endswith(".npz") and os.path.splitext(f)[0].isdigit()
        }
        steps = sorted(s for s in steps if s not in existing)
        if not steps:
            logger.info(f"No missing steps for {model} run {run} — already complete")
            sys.exit(0)
        logger.info(f"--fill-missing: {len(steps)} step(s) to ingest")
        logger.debug(f"--fill-missing steps: {steps}")

    ok = 0
    prev_acc: dict | None = None
    prev_step_done: int | None = None

    sp_cfg = config.get("symbol_precompute", {}) if isinstance(config, dict) else {}
    sp_enabled = bool(sp_cfg.get("enabled", False)) and LOW_ZOOM_PRECOMPUTED_BINS_ENABLED
    sp_fresh_only = bool(sp_cfg.get("on_fresh_ingest_only", True))
    sp_zooms = ",".join(str(int(z)) for z in sp_cfg.get("zooms", [5, 6, 7, 8, 9]))

    if model == "icon-d2":
        logger.info(
            "D2 substep ingest: vars=%s max_step=%s",
            ",".join(sorted(MULTI_MESSAGE_HOURLY_SELECT_VARS)),
            D2_SUBSTEP_MAX_STEP,
        )

    if bool(sp_cfg.get("enabled", False)) and not LOW_ZOOM_PRECOMPUTED_BINS_ENABLED:
        logger.info(
            "Low-zoom symbol bin precompute disabled by SKYVIEW_LOW_ZOOM_PRECOMPUTED_BINS=0; native symbol fields remain enabled"
        )

    for step in steps:
        # Ensure prev_acc is consistent with what this step expects.
        # If we have the right one in memory, use it; otherwise fall back to disk.
        # This handles both normal sequential runs and --fill-missing with gaps.
        expected_prev, _ = _precip_prev_step_and_dt(model, step)
        if prev_step_done != expected_prev:
            if expected_prev is not None:
                disk_acc = _load_precip_acc_from_step(out_dir, expected_prev)
                prev_acc = disk_acc
                prev_step_done = expected_prev if disk_acc is not None else None
            else:
                prev_acc = None
                prev_step_done = None

        if args.force:
            npz = os.path.join(out_dir, f"{step:03d}.npz")
            if os.path.exists(npz):
                os.unlink(npz)

        step_npz = os.path.join(out_dir, f"{step:03d}.npz")
        existed_before = os.path.exists(step_npz)

        success, curr_acc = ingest_step(run, step, tmp_dir, out_dir, model, config,
                                        profile_name=profile_name,
                                        prev_acc=prev_acc, prev_step=prev_step_done)
        if success:
            ok += 1
            if curr_acc is not None:
                prev_acc = curr_acc
                prev_step_done = step

            fresh_step = (not existed_before) and os.path.exists(step_npz)
            should_precompute = sp_enabled and ((not sp_fresh_only) or fresh_step)
            if should_precompute:
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            os.path.join(SCRIPT_DIR, "precompute_symbols.py"),
                            "--model", model,
                            "--run", run,
                            "--zooms", sp_zooms,
                            "--steps", str(step),
                            "--mode", "direct",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0:
                        logger.warning(
                            "Step %03d symbol precompute returned %s (stdout=%s stderr=%s)",
                            step,
                            proc.returncode,
                            (proc.stdout or "").strip()[:400],
                            (proc.stderr or "").strip()[:400],
                        )
                except Exception as e:
                    logger.warning(f"Step {step:03d}: symbol precompute failed: {e}")

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
