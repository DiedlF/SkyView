from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import zarr
    from numcodecs import Blosc
except Exception:  # pragma: no cover - exercised only where zarr is absent
    zarr = None
    Blosc = None


DATA_LAYOUT_ENV = "SKYVIEW_DATA_LAYOUT"
WRITE_ZARR_ENV = "SKYVIEW_WRITE_ZARR"


def data_layout() -> str:
    value = os.environ.get(DATA_LAYOUT_ENV, "auto").strip().lower()
    return value if value in {"auto", "npz", "zarr"} else "auto"


def should_write_zarr() -> bool:
    return os.environ.get(WRITE_ZARR_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def zarr_available() -> bool:
    return zarr is not None and Blosc is not None


def model_dir_name(model: str) -> str:
    return str(model).replace("_", "-")


def step_npz_path(data_dir: str, model: str, run: str, step: int) -> str:
    return os.path.join(data_dir, model_dir_name(model), str(run), f"{int(step):03d}.npz")


def step_zarr_path(data_dir: str, model: str, run: str, step: int) -> str:
    return os.path.join(data_dir, model_dir_name(model), str(run), f"{int(step):03d}.zarr")


def static_npz_path(data_dir: str, model: str) -> str:
    return os.path.join(data_dir, model_dir_name(model), "grid", "static.npz")


def static_zarr_path(data_dir: str, model: str) -> str:
    return os.path.join(data_dir, model_dir_name(model), "grid", "static.zarr")


def step_numbers_from_dir(run_path: str) -> list[int]:
    steps: set[int] = set()
    if not os.path.isdir(run_path):
        return []
    for name in os.listdir(run_path):
        stem = None
        if name.endswith(".npz"):
            stem = name[:-4]
        elif name.endswith(".zarr") and not name.startswith("."):
            stem = name[:-5]
        if stem and stem.isdigit():
            steps.add(int(stem))
    return sorted(steps)


def _selected_keys(available: Iterable[str], keys: Optional[Iterable[str]]) -> list[str]:
    available_set = set(available)
    if keys is None:
        return sorted(available_set)
    return [k for k in keys if k in available_set]


def _read_npz(path: str, keys: Optional[Iterable[str]]) -> Dict[str, Any]:
    with np.load(path) as npz:
        selected = _selected_keys(npz.files, keys)
        return {k: npz[k] for k in selected}


def _read_zarr(path: str, keys: Optional[Iterable[str]]) -> Dict[str, Any]:
    if not zarr_available():
        raise RuntimeError("zarr is not installed")
    group = zarr.open_group(path, mode="r")
    selected = _selected_keys(group.array_keys(), keys)
    return {k: group[k][...] for k in selected}


def _open_step_zarr_for_direct(
    *,
    data_dir: str,
    model: str,
    run: str,
    step: int,
    logger,
):
    layout = data_layout()
    path = step_zarr_path(data_dir, model, run, step)
    if layout == "npz":
        return None
    if not os.path.isdir(path):
        if layout == "zarr":
            raise FileNotFoundError(f"Zarr data not found: {path}")
        return None
    if not zarr_available():
        if layout == "zarr":
            raise RuntimeError("zarr is not installed")
        return None
    try:
        return zarr.open_group(path, mode="r")
    except Exception as exc:
        if layout == "zarr":
            raise
        if logger is not None:
            logger.warning("Zarr direct read failed for %s, falling back to NPZ: %s", path, exc)
        return None


def _valid_time(run: str, step: int) -> str:
    run_dt = datetime.strptime(str(run), "%Y%m%d%H")
    return (run_dt + timedelta(hours=int(step))).isoformat() + "Z"


def _as_scalar_array(value: Any) -> np.ndarray:
    return np.asarray([[value]], dtype=np.asarray(value).dtype if np.asarray(value).dtype.kind != "O" else np.float32)


def _read_static_point_value(
    *,
    data_dir: str,
    model: str,
    key: str,
    i: int,
    j: int,
    logger,
):
    zarr_path = static_zarr_path(data_dir, model)
    layout = data_layout()
    if layout in {"auto", "zarr"} and os.path.isdir(zarr_path) and zarr_available():
        try:
            group = zarr.open_group(zarr_path, mode="r")
            if key in group:
                return group[key][i, j]
        except Exception as exc:
            if layout == "zarr":
                raise
            if logger is not None:
                logger.warning("Static Zarr direct read failed for %s: %s", zarr_path, exc)

    if layout == "zarr":
        return None
    arrays = read_static_arrays(data_dir=data_dir, model=model, logger=logger)
    if key in arrays:
        return arrays[key][i, j]
    return None


def read_step_point_arrays(
    *,
    data_dir: str,
    model: str,
    run: str,
    step: int,
    keys: Iterable[str],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    i: Optional[int] = None,
    j: Optional[int] = None,
    substep_minutes: int = 0,
    substep_supported_vars: Optional[set[str]] = None,
    logger=None,
) -> Optional[Dict[str, Any]]:
    group = _open_step_zarr_for_direct(data_dir=data_dir, model=model, run=run, step=step, logger=logger)
    if group is None:
        return None
    if "lat" not in group or "lon" not in group:
        return None

    lat_arr = group["lat"][...]
    lon_arr = group["lon"][...]
    if len(lat_arr) == 0 or len(lon_arr) == 0:
        return None

    if i is None:
        if lat is None:
            raise ValueError("lat is required when i is not provided")
        i = int(np.argmin(np.abs(lat_arr - float(lat))))
    if j is None:
        if lon is None:
            raise ValueError("lon is required when j is not provided")
        j = int(np.argmin(np.abs(lon_arr - float(lon))))

    i = max(0, min(int(i), len(lat_arr) - 1))
    j = max(0, min(int(j), len(lon_arr) - 1))

    out: Dict[str, Any] = {
        "lat": np.asarray([lat_arr[i]], dtype=lat_arr.dtype),
        "lon": np.asarray([lon_arr[j]], dtype=lon_arr.dtype),
        "validTime": _valid_time(run, step),
        "_run": str(run),
        "_step": int(step),
        "_gridI": i,
        "_gridJ": j,
        "_latMin": float(np.min(lat_arr)),
        "_latMax": float(np.max(lat_arr)),
        "_lonMin": float(np.min(lon_arr)),
        "_lonMax": float(np.max(lon_arr)),
    }

    supported = substep_supported_vars or set()
    requested = set(keys or [])
    for key in requested:
        if key in {"lat", "lon", "validTime"}:
            continue
        value = None
        if (
            substep_minutes > 0
            and key in supported
            and f"{key}_substeps" in group
            and f"{key}_substep_minutes" in group
        ):
            minutes = [int(x) for x in np.asarray(group[f"{key}_substep_minutes"][...]).tolist()]
            if int(substep_minutes) in minutes:
                sub_idx = minutes.index(int(substep_minutes))
                value = group[f"{key}_substeps"][sub_idx, i, j]
                out["_substepMinutes"] = int(substep_minutes)
        if value is None and key in group:
            arr = group[key]
            if arr.ndim == 0:
                value = arr[()]
            elif arr.ndim == 2:
                value = arr[i, j]
            elif arr.ndim == 3:
                value = arr[0, i, j]
        if value is None:
            value = _read_static_point_value(data_dir=data_dir, model=model, key=key, i=i, j=j, logger=logger)
        if value is not None:
            out[key] = _as_scalar_array(value)

    return out


def _read_layout(
    *,
    npz_path: str,
    zarr_path: str,
    keys: Optional[Iterable[str]],
    logger,
) -> Dict[str, Any]:
    layout = data_layout()
    if layout in {"auto", "zarr"} and os.path.isdir(zarr_path):
        try:
            return _read_zarr(zarr_path, keys)
        except Exception as exc:
            if layout == "zarr":
                raise
            if logger is not None:
                logger.warning("Zarr read failed for %s, falling back to NPZ: %s", zarr_path, exc)

    if layout == "zarr":
        raise FileNotFoundError(f"Zarr data not found: {zarr_path}")

    if os.path.exists(npz_path):
        return _read_npz(npz_path, keys)

    raise FileNotFoundError(f"Data not found: {zarr_path if layout == 'zarr' else npz_path}")


def read_step_arrays(
    *,
    data_dir: str,
    model: str,
    run: str,
    step: int,
    keys: Optional[Iterable[str]],
    logger=None,
) -> Dict[str, Any]:
    return _read_layout(
        npz_path=step_npz_path(data_dir, model, run, step),
        zarr_path=step_zarr_path(data_dir, model, run, step),
        keys=keys,
        logger=logger,
    )


def read_static_arrays(*, data_dir: str, model: str, logger=None) -> Dict[str, Any]:
    npz_path = static_npz_path(data_dir, model)
    zarr_path = static_zarr_path(data_dir, model)
    if not os.path.exists(npz_path) and not os.path.isdir(zarr_path):
        return {}
    return _read_layout(npz_path=npz_path, zarr_path=zarr_path, keys=None, logger=logger)


def _zarr_compressor():
    if not zarr_available():
        return None
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _chunks_for(arr: np.ndarray):
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return (min(int(arr.shape[0]), 4096),)
    if arr.ndim == 2:
        return (min(int(arr.shape[0]), 256), min(int(arr.shape[1]), 256))
    if arr.ndim == 3:
        return (min(int(arr.shape[0]), 1), min(int(arr.shape[1]), 256), min(int(arr.shape[2]), 256))
    return tuple(min(int(dim), 64) for dim in arr.shape)


def write_zarr_group(path: str, arrays: Dict[str, Any], attrs: Optional[Dict[str, Any]] = None) -> bool:
    if not zarr_available() or not should_write_zarr():
        return False

    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".{os.path.basename(path)}.tmp")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    group = zarr.open_group(tmp_path, mode="w")
    if attrs:
        group.attrs.update(attrs)

    compressor = _zarr_compressor()
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"O", "U", "S"}:
            group.attrs[key] = arr.item() if arr.shape == () else arr.tolist()
            continue
        group.create_dataset(
            key,
            data=arr,
            chunks=_chunks_for(arr),
            compressor=compressor,
            overwrite=True,
        )

    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)
    return True
