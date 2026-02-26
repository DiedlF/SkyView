from __future__ import annotations

from typing import Optional
import time as _time
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, Query


def build_point_router(
    *,
    resolve_time_with_cache_context,
    load_data,
    POINT_KEYS,
    POINT_KEYS_MINIMAL,
    _resolve_eu_time_strict,
    _load_eu_data_strict,
    rotate_caches_for_context,
    fallback_stats,
    classify_point,
    ww_to_symbol,
    build_overlay_values,
    _freshness_minutes_from_run,
    _set_fallback_current,
):
    router = APIRouter()
    point_cache = OrderedDict()
    POINT_CACHE_TTL_SECONDS = 90.0
    POINT_CACHE_MAX_ITEMS = 2000

    @router.get("/api/point")
    async def api_point(
        lat: float = Query(...),
        lon: float = Query(...),
        time: str = Query("latest"),
        model: Optional[str] = Query(None),
        wind_level: str = Query("10m"),
        zoom: Optional[int] = Query(None, ge=5, le=12),
        include_overlay: bool = Query(True),
        include_symbol: bool = Query(True),
        include_wind: bool = Query(False),
        overlay_key: Optional[str] = Query(None),
    ):
        # Selective key loading based on currently rendered layers.
        run, step, model_used = resolve_time_with_cache_context(time, model)

        ov = (overlay_key or "").strip().lower()
        ov_alias = {
            "rain_amount": "rain",
            "snow_amount": "snow",
            "hail_amount": "hail",
        }
        ov = ov_alias.get(ov, ov)
        need_overlay = bool(include_overlay and ov and ov != "none")

        req_set = {"lat", "lon", "validTime"}
        symbol_keys = {"ww", "ceiling", "clcl", "clcm", "clch", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf"}
        req_set.update(symbol_keys)  # also used for D2 signal checks and fallback decisions

        if include_wind:
            if wind_level == "10m" or wind_level == "gust10m":
                req_set.update({"u_10m", "v_10m"})
                if wind_level == "gust10m":
                    req_set.add("vmax_10m")
            else:
                req_set.update({f"u_{wind_level}hpa", f"v_{wind_level}hpa"})

        if need_overlay:
            overlay_needs = {
                "clouds_low": {"clcl"},
                "clouds_mid": {"clcm"},
                "clouds_high": {"clch"},
                "clouds_total": {"clct"},
                "clouds_total_mod": {"clct_mod"},
                "thermals": {"cape_ml"},
                "dry_conv_top": {"htop_dc"},
                "ceiling": {"ceiling"},
                "cloud_base": {"hbas_sc"},
                "rain": {"rain_rate"},
                "snow": {"snow_rate"},
                "hail": {"hail_rate"},
                "total_precip": {"tp_rate"},
                "h_snow": {"h_snow"},
                "dew_spread_2m": {"t_2m", "td_2m"},
                "conv_thickness": {"htop_sc", "hbas_sc"},
                "lpi": {"lpi_max"},
                "t_2m": {"t_2m"},
                "t_950hpa": {"t_950hpa"},
                "t_850hpa": {"t_850hpa"},
                "t_700hpa": {"t_700hpa"},
                "t_500hpa": {"t_500hpa"},
                "t_300hpa": {"t_300hpa"},
                "relhum_2m": {"relhum_2m"},
                "mh": {"mh"},
                "ashfl_s": {"ashfl_s"},
                "climb_rate": {"ashfl_s", "mh", "t_2m", "td_2m", "hsurf", "t_850hpa", "t_700hpa", "t_500hpa", "t_300hpa"},
    "climb_rate_cape": {"climb_rate_cape"},
                "lcl": {"t_2m", "td_2m", "hsurf"},
            }
            req_set.update(overlay_needs.get(ov, set()))

        req_keys = sorted(req_set)
        d = load_data(run, step, model_used, keys=req_keys)
        fallback_decision = "primary_model_only"

        cache_key = (
            model_used,
            run,
            int(step),
            round(float(lat), 4),
            round(float(lon), 4),
            str(wind_level),
            int(zoom) if zoom is not None else None,
            bool(include_overlay),
            bool(include_symbol),
            bool(include_wind),
            ov,
        )
        now = _time.time()
        cached = point_cache.get(cache_key)
        if cached is not None:
            payload, ts = cached
            if (now - ts) <= POINT_CACHE_TTL_SECONDS:
                point_cache.move_to_end(cache_key)
                return payload
            point_cache.pop(cache_key, None)

        # EU fallback outside D2 domain or when D2 has no signal at the queried point.
        def _nearest_idx(arr: np.ndarray, value: float) -> int:
            pos = int(np.searchsorted(arr, value, side="left"))
            if pos <= 0:
                return 0
            if pos >= len(arr):
                return len(arr) - 1
            return pos - 1 if abs(value - float(arr[pos - 1])) <= abs(float(arr[pos]) - value) else pos

        if model_used == "icon_d2":
            lat_d2 = d["lat"]
            lon_d2 = d["lon"]
            lat_min = float(d.get("_latMin", np.min(lat_d2)))
            lat_max = float(d.get("_latMax", np.max(lat_d2)))
            lon_min = float(d.get("_lonMin", np.min(lon_d2)))
            lon_max = float(d.get("_lonMax", np.max(lon_d2)))
            in_d2_domain = (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)
            li_d2 = _nearest_idx(lat_d2, lat)
            lo_d2 = _nearest_idx(lon_d2, lon)
            d2_has_signal = any(
                np.isfinite(float(d[k][li_d2, lo_d2]))
                for k in ("ww", "ceiling", "cape_ml", "hbas_sc")
                if k in d
            )
            if (not in_d2_domain) or (not d2_has_signal):
                trigger = "outside_d2_domain" if (not in_d2_domain) else "d2_missing_signal"
                eu_fb_point = _load_eu_data_strict(time, req_keys)
                if eu_fb_point is not None and not eu_fb_point.get("missing"):
                    run_eu, step_eu, model_eu = eu_fb_point["run"], eu_fb_point["step"], eu_fb_point["model"]
                    rotate_caches_for_context(f"{model_eu}|{run_eu}|{step_eu}")
                    d = eu_fb_point["data"]
                    run, step, model_used = run_eu, step_eu, model_eu
                    fallback_stats["pointFallback"] += 1
                    fallback_decision = f"eu_fallback:{trigger}"
                elif eu_fb_point is not None and eu_fb_point.get("missing"):
                    fallback_decision = f"eu_data_missing:{trigger}"
                else:
                    fallback_decision = f"strict_time_denied:{trigger}"

        # Nearest grid point — computed once, reused throughout.
        lat_arr = d["lat"]
        lon_arr = d["lon"]
        li0 = _nearest_idx(lat_arr, lat)
        lo0 = _nearest_idx(lon_arr, lon)
        li = np.array([li0], dtype=int)
        lo = np.array([lo0], dtype=int)

        # ── Variable extraction using direct scalar indexing ──────────────────────
        def _get(key: str) -> float | None:
            if key not in d:
                return None
            v = float(d[key][li0, lo0])
            return None if not np.isfinite(v) else v

        vars_out = ["ww", "clcl", "clcm", "clch", "clct", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "ceiling"]
        result = {}
        for v in vars_out:
            val = _get(v)
            if val is None:
                result[v] = None
            elif v == "ceiling" and val > 20_000:
                result[v] = None
            else:
                result[v] = round(val, 1)

        ww_max = int(_get("ww") or 0)
        best_type = None
        sym = None

        if include_symbol:
            best_type = classify_point(
                clcl=_get("clcl") or 0.0,
                clcm=_get("clcm") or 0.0,
                clch=_get("clch") or 0.0,
                cape_ml=_get("cape_ml") or 0.0,
                htop_dc=_get("htop_dc") or 0.0,
                hbas_sc=_get("hbas_sc") or 0.0,
                htop_sc=_get("htop_sc") or 0.0,
                lpi=(_get("lpi_max") if _get("lpi_max") is not None else (_get("lpi") or 0.0)),
                ceiling=_get("ceiling") or 0.0,
                hsurf=_get("hsurf") or 0.0,
            )
            result["cloudType"] = best_type
            ww_sym = ww_to_symbol(ww_max) if ww_max > 10 else None
            sym = ww_sym or best_type
            result["symbol"] = sym
            result["cloudTypeName"] = sym.title()
            result["ww"] = ww_max

            # Cloud base height (hectometers), matching symbol label logic
            ceil_cell = d["ceiling"][np.ix_(li, lo)] if "ceiling" in d else np.array([])
            htop_cell = d["htop_dc"][np.ix_(li, lo)] if "htop_dc" in d else np.array([])
            hbas_cell = d["hbas_sc"][np.ix_(li, lo)] if "hbas_sc" in d else np.array([])
            valid_ceil = ceil_cell[(ceil_cell > 0) & (ceil_cell < 20_000)] if ceil_cell.size else np.array([])
            valid_hbas = hbas_cell[(hbas_cell > 0) & np.isfinite(hbas_cell)] if hbas_cell.size else np.array([])

            if sym in ("cu_hum", "cu_con", "cb") and len(valid_hbas) > 0:
                result["cloudBaseHm"] = int(np.min(valid_hbas) / 100)
            elif sym in ("st", "ac", "ci", "fog", "rime_fog") and len(valid_ceil) > 0:
                result["cloudBaseHm"] = int(np.min(valid_ceil) / 100)
            elif sym == "blue_thermal" and htop_cell.size and np.any(htop_cell > 0):
                result["cloudBaseHm"] = int(np.max(htop_cell[htop_cell > 0]) / 100)
            else:
                result["cloudBaseHm"] = None
        else:
            result["ww"] = ww_max
            result["symbol"] = None
            result["cloudType"] = None
            result["cloudTypeName"] = None
            result["cloudBaseHm"] = None
            ceil_cell = d["ceiling"][np.ix_(li, lo)] if "ceiling" in d else np.array([])

        # Overlay values (only active basemap data: max 1 overlay + optional wind)
        if include_overlay or include_wind:
            full_overlay = build_overlay_values(
                d=d,
                li=li,
                lo=lo,
                ww_max=ww_max,
                ceil_cell=ceil_cell,
                wind_level=wind_level,
                zoom=zoom,
                lat=lat,
                lon=lon,
                lat_arr=lat_arr,
                lon_arr=lon_arr,
                model_used=model_used,
                step=step,
            )
            ov_out = {}
            if include_wind:
                if "wind_speed" in full_overlay:
                    ov_out["wind_speed"] = full_overlay.get("wind_speed")
                if "wind_dir" in full_overlay:
                    ov_out["wind_dir"] = full_overlay.get("wind_dir")
            if need_overlay:
                ov_val = full_overlay.get(ov)
                if ov_val is None:
                    ov_val = full_overlay.get(ov_alias.get(ov, ov))
                ov_out[ov] = ov_val
                # Keep thermal_class as fallback helper for climb-rate UI derivation.
                if ov == "climb_rate" and "thermal_class" in full_overlay:
                    ov_out["thermal_class"] = full_overlay.get("thermal_class")
            result["overlay_values"] = ov_out if ov_out else None
        else:
            result["overlay_values"] = None

        # Explorer/Skyview contract convergence: raw values dict
        result["values"] = {k: result.get(k) for k in [
            "ww", "clcl", "clcm", "clch", "clct", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "ceiling"
        ]}

        # Closest grid point coordinates (li0/lo0 already computed above — no duplicate argmin)
        result["lat"] = round(float(lat_arr[li0]), 4)
        result["lon"] = round(float(lon_arr[lo0]), 4)
        result["validTime"] = d["validTime"]
        result["run"] = run
        result["model"] = model_used
        result["sourceModel"] = model_used
        result["diagnostics"] = {
            "dataFreshnessMinutes": _freshness_minutes_from_run(run),
            "fallbackDecision": fallback_decision,
            "requestedModel": model,
            "requestedTime": time,
            "sourceModel": model_used,
            "euDataMissing": fallback_decision.startswith("eu_data_missing:"),
        }

        _set_fallback_current(
            "point",
            fallback_decision,
            source_model=model_used,
            detail={"requestedTime": time},
        )

        point_cache[cache_key] = (result, now)
        point_cache.move_to_end(cache_key)
        while len(point_cache) > POINT_CACHE_MAX_ITEMS:
            point_cache.popitem(last=False)

        return result

    return router
