"""Explorer data-provider abstraction.

Phase 1: local provider + remote stub.
Phase 2: functional remote provider with metadata/field caching.
"""

from __future__ import annotations

import io
import bz2
import json
import os
import time
import hashlib
import logging
import threading
import urllib.parse
import urllib.request
import tempfile
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import cfgrib

logger = logging.getLogger("explorer.provider")


class ExplorerDataProvider(Protocol):
    def load_data(self, run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]: ...
    def get_available_runs(self) -> list[dict]: ...
    def get_merged_timeline(self) -> dict: ...
    def resolve_time(self, time_str: str, model: Optional[str] = None) -> tuple[str, int, str]: ...


class LocalNpzProvider:
    def __init__(self, data_dir: str, get_runs_fn, get_merged_timeline_fn, resolve_time_fn):
        self.data_dir = data_dir
        self._get_runs = get_runs_fn
        self._get_merged_timeline = get_merged_timeline_fn
        self._resolve_time = resolve_time_fn
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_data(self, run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        cache_key = f"{model}/{run}/{step:03d}"

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if keys is None or all(k in cached for k in keys):
                return cached

        model_dir = model.replace("_", "-")
        path = os.path.join(self.data_dir, model_dir, run, f"{step:03d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}")

        npz = np.load(path)

        if keys is not None:
            load_keys = set(keys) | {"lat", "lon"}
            arrays = {k: npz[k] for k in load_keys if k in npz.files}
            if cache_key in self._cache:
                for k, v in self._cache[cache_key].items():
                    if k not in arrays:
                        arrays[k] = v
        else:
            arrays = {k: npz[k] for k in npz.files}

        run_dt = datetime.strptime(run, "%Y%m%d%H")
        valid_dt = run_dt + timedelta(hours=step)
        arrays["validTime"] = valid_dt.isoformat() + "Z"
        arrays["_run"] = run
        arrays["_step"] = step

        self._cache[cache_key] = arrays
        return arrays

    def get_available_runs(self) -> list[dict]:
        return self._get_runs(self.data_dir)

    def get_merged_timeline(self) -> dict:
        return self._get_merged_timeline(self.data_dir)

    def resolve_time(self, time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
        return self._resolve_time(self.data_dir, time_str, model)

    def get_stats(self) -> dict:
        return {
            "mode": "local_npz",
            "cache_items": len(self._cache),
        }


class RemoteProvider:
    """Remote explorer provider using source endpoints + local TTL caches."""

    def __init__(self, base_url: Optional[str] = None, timeout_seconds: Optional[float] = None):
        self.base_url = (base_url or os.environ.get("EXPLORER_REMOTE_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("EXPLORER_REMOTE_BASE_URL is required for remote provider")

        self.timeout_seconds = float(timeout_seconds or os.environ.get("EXPLORER_REMOTE_TIMEOUT_SECONDS", "12"))
        self.auth_token = os.environ.get("EXPLORER_REMOTE_SOURCE_TOKEN", "").strip()
        self.meta_ttl = float(os.environ.get("EXPLORER_REMOTE_META_TTL_SECONDS", "30"))
        self.field_ttl = float(os.environ.get("EXPLORER_REMOTE_FIELD_TTL_SECONDS", "120"))
        self.field_cache_max_items = int(os.environ.get("EXPLORER_REMOTE_FIELD_CACHE_ITEMS", "48"))
        self.field_wait_timeout = float(os.environ.get("EXPLORER_REMOTE_FIELD_WAIT_TIMEOUT_SECONDS", "20"))

        # Optional disk cache (shared across workers/restarts)
        self.disk_cache_dir = os.environ.get("EXPLORER_REMOTE_DISK_CACHE_DIR", "").strip()
        self.disk_cache_ttl = float(os.environ.get("EXPLORER_REMOTE_DISK_CACHE_TTL_SECONDS", str(int(self.field_ttl))))
        if self.disk_cache_dir:
            os.makedirs(self.disk_cache_dir, exist_ok=True)

        self._meta_cache: Dict[str, tuple[float, Any]] = {}
        self._field_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._field_inflight: Dict[str, threading.Event] = {}
        self._field_inflight_lock = threading.Lock()
        self._stats: Dict[str, int | float] = {
            "meta_cache_hits": 0,
            "meta_cache_misses": 0,
            "field_mem_hits": 0,
            "field_disk_hits": 0,
            "field_cache_misses": 0,
            "field_fetches": 0,
            "field_waits": 0,
            "field_wait_timeouts": 0,
            "http_errors": 0,
            "last_fetch_ms": 0.0,
            "fetch_total_ms": 0.0,
            "fetch_avg_ms": 0.0,
            "fetch_max_ms": 0.0,
        }

    def _make_request(self, path: str, query: Optional[dict] = None) -> urllib.request.Request:
        q = f"?{urllib.parse.urlencode(query)}" if query else ""
        url = f"{self.base_url}{path}{q}"
        headers = {}
        if self.auth_token:
            headers["X-Source-Token"] = self.auth_token
        return urllib.request.Request(url=url, headers=headers)

    def _http_json(self, path: str, query: Optional[dict] = None) -> Any:
        req = self._make_request(path, query)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            self._stats["http_errors"] += 1
            logger.warning("remote_http_json_error path=%s err=%s", path, e)
            raise

    def _http_bytes(self, path: str, query: Optional[dict] = None) -> bytes:
        req = self._make_request(path, query)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as r:
                return r.read()
        except Exception as e:
            self._stats["http_errors"] += 1
            logger.warning("remote_http_bytes_error path=%s err=%s", path, e)
            raise

    def _cache_get(self, cache: Dict[str, tuple[float, Any]], key: str):
        item = cache.get(key)
        if not item:
            return None
        exp, val = item
        if time.time() > exp:
            cache.pop(key, None)
            return None
        return val

    def _cache_set(self, cache: Dict[str, tuple[float, Any]], key: str, val: Any, ttl: float):
        cache[key] = (time.time() + ttl, val)

    def _field_prune(self):
        if len(self._field_cache) <= self.field_cache_max_items:
            return
        # Remove oldest-expiring entries first
        for k, _ in sorted(self._field_cache.items(), key=lambda kv: kv[1][0])[: max(1, len(self._field_cache) - self.field_cache_max_items)]:
            self._field_cache.pop(k, None)

    def _disk_key_path(self, cache_key: str) -> Optional[str]:
        if not self.disk_cache_dir:
            return None
        h = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return os.path.join(self.disk_cache_dir, f"{h}.npz")

    def _disk_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = self._disk_key_path(cache_key)
        if not path or not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > self.disk_cache_ttl:
            try:
                os.unlink(path)
            except Exception:
                pass
            return None
        try:
            npz = np.load(path)
            arrays = {k: npz[k] for k in npz.files}
            if "validTime" in arrays and isinstance(arrays["validTime"], np.ndarray):
                arrays["validTime"] = str(arrays["validTime"].item())
            return arrays
        except Exception:
            return None

    def _disk_set(self, cache_key: str, arrays: Dict[str, Any]):
        path = self._disk_key_path(cache_key)
        if not path:
            return
        tmp = f"{path}.tmp"
        to_save = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray):
                to_save[k] = v
        if "validTime" in arrays and not isinstance(arrays["validTime"], np.ndarray):
            to_save["validTime"] = np.array(str(arrays["validTime"]))
        if "_run" in arrays and not isinstance(arrays["_run"], np.ndarray):
            to_save["_run"] = np.array(str(arrays["_run"]))
        if "_step" in arrays and not isinstance(arrays["_step"], np.ndarray):
            to_save["_step"] = np.array(int(arrays["_step"]))

        try:
            np.savez_compressed(tmp, **to_save)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass

    def load_data(self, run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        requested = sorted(set(keys or []))
        cache_key = f"{model}/{run}/{step:03d}|{','.join(requested) if requested else '*'}"

        # L1 memory cache
        cached = self._cache_get(self._field_cache, cache_key)
        if cached is not None:
            self._stats["field_mem_hits"] += 1
            return cached

        # L2 disk cache
        disk_cached = self._disk_get(cache_key)
        if disk_cached is not None:
            self._stats["field_disk_hits"] += 1
            self._cache_set(self._field_cache, cache_key, disk_cached, ttl=self.field_ttl)
            return disk_cached

        self._stats["field_cache_misses"] += 1

        # In-flight coalescing: only one network fetch per field key.
        owner = False
        with self._field_inflight_lock:
            evt = self._field_inflight.get(cache_key)
            if evt is None:
                evt = threading.Event()
                self._field_inflight[cache_key] = evt
                owner = True

        if not owner:
            self._stats["field_waits"] += 1
            waited = evt.wait(timeout=self.field_wait_timeout)
            if waited:
                cached_after = self._cache_get(self._field_cache, cache_key)
                if cached_after is not None:
                    return cached_after
                disk_after = self._disk_get(cache_key)
                if disk_after is not None:
                    self._cache_set(self._field_cache, cache_key, disk_after, ttl=self.field_ttl)
                    return disk_after
            self._stats["field_wait_timeouts"] += 1
            # timeout or no result -> fall through and fetch ourselves

        try:
            query = {
                "run": run,
                "step": int(step),
                "model": model,
            }
            if requested:
                query["keys"] = ",".join(requested)

            self._stats["field_fetches"] += 1
            t0 = time.perf_counter()
            blob = self._http_bytes("/api/source/field.npz", query=query)
            npz = np.load(io.BytesIO(blob))
            arrays = {k: npz[k] for k in npz.files}
            arrays["_run"] = run
            arrays["_step"] = int(step)
            if "validTime" in arrays and isinstance(arrays["validTime"], np.ndarray):
                arrays["validTime"] = str(arrays["validTime"].item())

            self._cache_set(self._field_cache, cache_key, arrays, ttl=self.field_ttl)
            self._field_prune()
            self._disk_set(cache_key, arrays)
            fetch_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self._stats["last_fetch_ms"] = fetch_ms
            self._stats["fetch_total_ms"] = round(float(self._stats["fetch_total_ms"]) + fetch_ms, 2)
            self._stats["fetch_avg_ms"] = round(float(self._stats["fetch_total_ms"]) / max(1, int(self._stats["field_fetches"])), 2)
            self._stats["fetch_max_ms"] = round(max(float(self._stats["fetch_max_ms"]), fetch_ms), 2)
            logger.info(
                "remote_field_fetch model=%s run=%s step=%s keys=%s bytes=%s ms=%s",
                model,
                run,
                step,
                ",".join(requested) if requested else "*",
                len(blob),
                fetch_ms,
            )
            return arrays
        finally:
            if owner:
                with self._field_inflight_lock:
                    evt = self._field_inflight.pop(cache_key, None)
                    if evt is not None:
                        evt.set()

    def get_available_runs(self) -> list[dict]:
        key = "runs"
        cached = self._cache_get(self._meta_cache, key)
        if cached is not None:
            self._stats["meta_cache_hits"] += 1
            return cached
        self._stats["meta_cache_misses"] += 1
        val = self._http_json("/api/source/runs")
        self._cache_set(self._meta_cache, key, val, ttl=self.meta_ttl)
        return val

    def get_merged_timeline(self) -> dict:
        key = "timeline"
        cached = self._cache_get(self._meta_cache, key)
        if cached is not None:
            self._stats["meta_cache_hits"] += 1
            return cached
        self._stats["meta_cache_misses"] += 1
        val = self._http_json("/api/source/timeline")
        self._cache_set(self._meta_cache, key, val, ttl=self.meta_ttl)
        return val

    def resolve_time(self, time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
        query = {"time": time_str}
        if model:
            query["model"] = model
        data = self._http_json("/api/source/resolve_time", query=query)
        return data["run"], int(data["step"]), data["model"]

    def get_stats(self) -> dict:
        return {
            "mode": "remote",
            "base_url": self.base_url,
            "meta_cache_items": len(self._meta_cache),
            "field_mem_cache_items": len(self._field_cache),
            "field_inflight": len(self._field_inflight),
            "disk_cache_enabled": bool(self.disk_cache_dir),
            "disk_cache_dir": self.disk_cache_dir or None,
            "stats": dict(self._stats),
        }


class DwdOpenDataProvider:
    """Direct on-demand provider for DWD OpenData ICON products."""

    MODEL_CFG = {
        "icon_d2": {
            "run_interval": 3,
            "delay_h": 2,
            "steps": list(range(1, 49)),
            "base": "https://opendata.dwd.de/weather/nwp/icon-d2/grib",
            "prefix": "icon-d2_germany_regular-lat-lon",
        },
        "icon_eu": {
            "run_interval": 6,
            "delay_h": 4,
            "steps": list(range(1, 79)) + list(range(81, 121, 3)),
            "base": "https://opendata.dwd.de/weather/nwp/icon-eu/grib",
            "prefix": "icon-eu_europe_regular-lat-lon",
        },
        "icon_global": {
            "run_interval": 6,
            "delay_h": 5,
            "steps": list(range(0, 181, 3)),
            "base": "https://opendata.dwd.de/weather/nwp/icon/grib",
            "prefix": "icon_global_icosahedral_single-level",
        },
    }

    def __init__(self):
        self.timeout_seconds = float(os.environ.get("EXPLORER_DWD_TIMEOUT_SECONDS", "20"))
        self.field_ttl = float(os.environ.get("EXPLORER_DWD_FIELD_TTL_SECONDS", "300"))
        self.meta_ttl = float(os.environ.get("EXPLORER_DWD_META_TTL_SECONDS", "180"))
        self._field_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._meta_cache: Dict[str, tuple[float, Any]] = {}

    def _head_ok(self, url: str) -> bool:
        req = urllib.request.Request(url=url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds):
                return True
        except Exception:
            return False

    def _get_bytes(self, url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as r:
            return r.read()

    def _list_subdirs(self, url: str) -> List[str]:
        try:
            html = self._get_bytes(url).decode("utf-8", errors="ignore")
        except Exception:
            return []
        names = set()
        for m in re.finditer(r'href="([^"]+/)"', html):
            n = m.group(1).strip("/")
            if not n or n == "..":
                continue
            # Keep plain variable dir names, ignore query-like entries
            if all(ch.isalnum() or ch in ("_", "-") for ch in n):
                names.add(n.lower())
        return sorted(names)

    def _latest_run(self, model: str) -> str:
        cfg = self.MODEL_CFG[model]
        ref = datetime.now(timezone.utc) - timedelta(hours=cfg["delay_h"])
        h = (ref.hour // cfg["run_interval"]) * cfg["run_interval"]
        return ref.strftime("%Y%m%d") + f"{h:02d}"

    def _recent_runs(self, model: str, max_count: int = 12) -> List[str]:
        cfg = self.MODEL_CFG[model]
        out = []
        cur = datetime.strptime(self._latest_run(model), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        for _ in range(max_count):
            out.append(cur.strftime("%Y%m%d%H"))
            cur -= timedelta(hours=cfg["run_interval"])
        return out

    def _canon_model(self, model: Optional[str]) -> Optional[str]:
        if model is None:
            return None
        m = model.lower().replace("-", "_")
        if m == "icon":
            return "icon_global"
        return m

    def _build_url(self, model: str, run: str, step: int, var: str, pressure_level: Optional[int] = None) -> str:
        cfg = self.MODEL_CFG[model]
        run_h = run[-2:]
        var_dir = var
        if pressure_level is not None:
            if model == "icon_d2":
                fn = f"icon-d2_germany_regular-lat-lon_pressure-level_{run}_{step:03d}_{pressure_level}_{var}.grib2.bz2"
            elif model == "icon_eu":
                fn = f"icon-eu_europe_regular-lat-lon_pressure-level_{run}_{step:03d}_{pressure_level}_{var.upper()}.grib2.bz2"
            else:
                fn = f"icon_global_icosahedral_pressure-level_{run}_{step:03d}_{pressure_level}_{var.upper()}.grib2.bz2"
            return f"{cfg['base']}/{run_h}/{var_dir}/{fn}"

        if var == "hsurf":
            if model == "icon_d2":
                fn = f"icon-d2_germany_regular-lat-lon_time-invariant_{run}_000_0_hsurf.grib2.bz2"
            elif model == "icon_eu":
                fn = f"icon-eu_europe_regular-lat-lon_time-invariant_{run}_000_0_HSURF.grib2.bz2"
            else:
                fn = f"icon_global_icosahedral_time-invariant_{run}_000_0_HSURF.grib2.bz2"
            return f"{cfg['base']}/{run_h}/{var_dir}/{fn}"

        if model == "icon_d2":
            fn = f"icon-d2_germany_regular-lat-lon_single-level_{run}_{step:03d}_2d_{var}.grib2.bz2"
        elif model == "icon_eu":
            fn = f"icon-eu_europe_regular-lat-lon_single-level_{run}_{step:03d}_{var.upper()}.grib2.bz2"
        else:
            fn = f"icon_global_icosahedral_single-level_{run}_{step:03d}_{var.upper()}.grib2.bz2"
        return f"{cfg['base']}/{run_h}/{var_dir}/{fn}"

    def _read_grib_array(self, raw_bz2: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = bz2.decompress(raw_bz2)
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tf:
            tf.write(raw)
            tf.flush()
            ds = cfgrib.open_datasets(tf.name)[0]
            var = list(ds.data_vars.values())[0]
            data = var.values.squeeze().astype(np.float32)
            lat = ds.coords["latitude"].values.astype(np.float32)
            lon = ds.coords["longitude"].values.astype(np.float32)
            return data, lat, lon

    def get_available_runs(self) -> list[dict]:
        ck = "runs"
        now = time.time()
        if ck in self._meta_cache and self._meta_cache[ck][0] > now:
            return self._meta_cache[ck][1]

        runs: List[dict] = []
        runs_back = int(os.environ.get("EXPLORER_DWD_RUNS_BACK", "10"))

        for model in ("icon_d2", "icon_eu", "icon_global"):
            cfg = self.MODEL_CFG[model]
            discovered: List[str] = []

            # Probe candidate run folders backward in solver interval steps.
            # This mirrors Skyview ingest behavior: new solver run => new folder content appears.
            for run in self._recent_runs(model, max_count=max(1, runs_back)):
                test = self._build_url(model, run, cfg["steps"][0], "t_2m")
                if self._head_ok(test):
                    discovered.append(run)

            for run in discovered:
                steps = []
                for st in cfg["steps"]:
                    vd = datetime.strptime(run, "%Y%m%d%H") + timedelta(hours=int(st))
                    steps.append({"step": int(st), "validTime": vd.isoformat() + "Z"})
                runs.append({
                    "model": model,
                    "run": run,
                    "runTime": datetime.strptime(run, "%Y%m%d%H").isoformat() + "Z",
                    "steps": steps,
                })

        # Newest first per model/run
        runs.sort(key=lambda r: (r.get("model", ""), r.get("run", "")), reverse=True)

        self._meta_cache[ck] = (now + self.meta_ttl, runs)
        return runs

    def get_merged_timeline(self) -> dict:
        runs = self.get_available_runs()
        steps = []
        for r in runs:
            for s in r.get("steps", []):
                steps.append({"validTime": s["validTime"], "step": s["step"], "model": r["model"], "run": r["run"]})
        # prefer higher-res models for identical validTime
        pref = {"icon_d2": 0, "icon_eu": 1, "icon_global": 2}
        dedup = {}
        for s in sorted(steps, key=lambda x: (x["validTime"], pref.get(x["model"], 9))):
            dedup.setdefault(s["validTime"], s)
        merged = sorted(dedup.values(), key=lambda x: x["validTime"])
        return {"run": None, "runTime": None, "steps": merged}

    def resolve_time(self, time_str: str, model: Optional[str] = None) -> tuple[str, int, str]:
        m = self._canon_model(model)
        runs = self.get_available_runs()
        if m:
            runs = [r for r in runs if r["model"] == m]
        if not runs:
            raise RuntimeError("No DWD runs available")

        if time_str == "latest":
            best = None
            for r in runs:
                for s in r["steps"]:
                    if best is None or s["validTime"] > best[0]:
                        best = (s["validTime"], r, s)
            if best is None:
                raise RuntimeError("No DWD timesteps available")
            _, r, s = best
            return r["run"], int(s["step"]), r["model"]

        for r in runs:
            for s in r["steps"]:
                if s["validTime"] == time_str:
                    return r["run"], int(s["step"]), r["model"]
        raise RuntimeError(f"Time not found in DWD source: {time_str}")

    def load_data(self, run: str, step: int, model: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        model = self._canon_model(model) or "icon_d2"
        requested = sorted(set(keys or []))
        ck = f"{model}/{run}/{int(step):03d}|{','.join(requested) if requested else '*'}"
        now = time.time()
        if ck in self._field_cache and self._field_cache[ck][0] > now:
            return self._field_cache[ck][1]

        # If no keys given, keep cost controlled by loading common variables only.
        if not requested:
            requested = ["t_2m", "td_2m", "u_10m", "v_10m", "ww", "clct"]

        out: Dict[str, Any] = {}
        lat_1d = None
        lon_1d = None

        for k in requested:
            pressure_level = None
            var = k
            if k.startswith(("u_", "v_", "t_")) and k.endswith("hpa"):
                parts = k.split("_")
                var = parts[0]
                pressure_level = int(parts[1].replace("hpa", ""))

            try:
                url = self._build_url(model, run, int(step), var, pressure_level=pressure_level)
                blob = self._get_bytes(url)
                arr, lat, lon = self._read_grib_array(blob)
                out[k] = arr
                if lat_1d is None:
                    lat_1d = lat
                    lon_1d = lon
            except Exception as e:
                logger.debug("dwd_load_skip model=%s run=%s step=%s var=%s err=%s", model, run, step, k, e)
                continue

        if lat_1d is None or lon_1d is None:
            raise RuntimeError(f"No fields could be loaded from DWD for {model} run={run} step={step}")

        out["lat"] = lat_1d
        out["lon"] = lon_1d
        out["_run"] = run
        out["_step"] = int(step)
        vd = datetime.strptime(run, "%Y%m%d%H") + timedelta(hours=int(step))
        out["validTime"] = vd.isoformat() + "Z"

        self._field_cache[ck] = (now + self.field_ttl, out)
        return out

    def list_variables(self, model: str, run: Optional[str] = None) -> List[str]:
        m = self._canon_model(model) or "icon_d2"
        r = run
        if not r:
            runs = [x for x in self.get_available_runs() if x.get("model") == m]
            if not runs:
                return []
            r = runs[0]["run"]
        run_h = str(r)[-2:]
        cfg = self.MODEL_CFG[m]
        # Variables are represented as subfolders under /<run_hour>/
        base = f"{cfg['base']}/{run_h}/"
        return self._list_subdirs(base)

    def get_stats(self) -> dict:
        return {
            "mode": "dwd",
            "meta_cache_items": len(self._meta_cache),
            "field_cache_items": len(self._field_cache),
        }


def build_provider(kind: str, *, data_dir: str, get_runs_fn, get_merged_timeline_fn, resolve_time_fn):
    mode = (kind or "local_npz").strip().lower()
    if mode in ("local", "local_npz", "npz"):
        return LocalNpzProvider(data_dir, get_runs_fn, get_merged_timeline_fn, resolve_time_fn)
    if mode in ("remote", "http"):
        return RemoteProvider()
    if mode in ("dwd", "dwd_open", "opendata"):
        return DwdOpenDataProvider()
    raise ValueError(f"Unknown EXPLORER_DATA_PROVIDER mode: {kind}")
