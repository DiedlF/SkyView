from __future__ import annotations

from typing import Optional
import time as _time
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, Query


OV_ALIAS = {
    "rain_amount": "rain",
    "snow_amount": "snow",
    "hail_amount": "hail",
}

SYMBOL_KEYS = {
    "ww", "ceiling", "clcl", "clcm", "clch", "cape_ml",
    "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "hsurf",
}

OVERLAY_NEEDS = {
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

RAW_VALUE_KEYS = ["ww", "clcl", "clcm", "clch", "clct", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi_max", "ceiling"]


def _nearest_idx(arr: np.ndarray, value: float) -> int:
    pos = int(np.searchsorted(arr, value, side="left"))
    if pos <= 0:
        return 0
    if pos >= len(arr):
        return len(arr) - 1
    return pos - 1 if abs(value - float(arr[pos - 1])) <= abs(float(arr[pos]) - value) else pos


def _normalize_overlay_key(overlay_key: Optional[str]) -> tuple[str, bool]:
    ov = OV_ALIAS.get((overlay_key or "").strip().lower(), (overlay_key or "").strip().lower())
    need_overlay = bool(ov and ov != "none")
    return ov, need_overlay


def _build_req_keys(*, include_wind: bool, wind_level: str, include_symbol: bool, need_overlay: bool, ov: str) -> list[str]:
    req_set = {"lat", "lon", "validTime"}
    # Keep signal keys available for D2/EU fallback decisioning.
    req_set.update(SYMBOL_KEYS)

    if include_wind:
        if wind_level in ("10m", "gust10m"):
            req_set.update({"u_10m", "v_10m"})
            if wind_level == "gust10m":
                req_set.add("vmax_10m")
        else:
            req_set.update({f"u_{wind_level}hpa", f"v_{wind_level}hpa"})

    if need_overlay:
        req_set.update(OVERLAY_NEEDS.get(ov, set()))

    return sorted(req_set)


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
        substep: int = Query(0, ge=0, le=45),
    ):
        run, step, model_used = resolve_time_with_cache_context(time, model)

        ov, has_overlay_request = _normalize_overlay_key(overlay_key)
        need_overlay = bool(include_overlay and has_overlay_request)
        req_keys = _build_req_keys(
            include_wind=include_wind,
            wind_level=wind_level,
            include_symbol=include_symbol,
            need_overlay=need_overlay,
            ov=ov,
        )
        substep_minutes = substep if substep in (0, 15, 30, 45) else 0
        d = load_data(run, step, model_used, keys=req_keys, substep_minutes=substep_minutes)
        fallback_decision = "primary_model_only"

        cache_key = (
            model_used,
            run,
            int(step),
            int(substep_minutes),
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

        # ── Scalar extraction at the nearest clicked grid point ───────────────
        def _get(key: str) -> float | None:
            if key not in d:
                return None
            v = float(d[key][li0, lo0])
            return None if not np.isfinite(v) else v

        scalar_values = {k: _get(k) for k in set(RAW_VALUE_KEYS) | {"hsurf", "lpi"}}

        result = {}
        for key in RAW_VALUE_KEYS:
            val = scalar_values.get(key)
            if val is None:
                result[key] = None
            elif key == "ceiling" and val > 20_000:
                result[key] = None
            else:
                result[key] = round(val, 1)

        ww_max = int(scalar_values.get("ww") or 0)
        result["ww"] = ww_max

        if include_symbol:
            lpi_val = scalar_values.get("lpi_max")
            if lpi_val is None:
                lpi_val = scalar_values.get("lpi") or 0.0
            best_type = classify_point(
                clcl=scalar_values.get("clcl") or 0.0,
                clcm=scalar_values.get("clcm") or 0.0,
                clch=scalar_values.get("clch") or 0.0,
                cape_ml=scalar_values.get("cape_ml") or 0.0,
                htop_dc=scalar_values.get("htop_dc") or 0.0,
                hbas_sc=scalar_values.get("hbas_sc") or 0.0,
                htop_sc=scalar_values.get("htop_sc") or 0.0,
                lpi=lpi_val,
                ceiling=scalar_values.get("ceiling") or 0.0,
                hsurf=scalar_values.get("hsurf") or 0.0,
            )
            ww_sym = ww_to_symbol(ww_max) if ww_max > 10 else None
            sym = ww_sym or best_type
            result["cloudType"] = best_type
            result["symbol"] = sym
            result["cloudTypeName"] = sym.title()
        else:
            result["symbol"] = None
            result["cloudType"] = None
            result["cloudTypeName"] = None

        if include_overlay or include_wind:
            full_overlay = build_overlay_values(
                d=d,
                li=li,
                lo=lo,
                ww_max=ww_max,
                wind_level=wind_level,
                zoom=zoom,
                lat=lat,
                lon=lon,
                lat_arr=lat_arr,
                lon_arr=lon_arr,
                model_used=model_used,
                step=step,
            )
            overlay_values = {}
            if include_wind:
                if "wind_speed" in full_overlay:
                    overlay_values["wind_speed"] = full_overlay.get("wind_speed")
                if "wind_dir" in full_overlay:
                    overlay_values["wind_dir"] = full_overlay.get("wind_dir")
            if need_overlay:
                overlay_values[ov] = full_overlay.get(ov)
                if ov == "climb_rate" and "thermal_class" in full_overlay:
                    overlay_values["thermal_class"] = full_overlay.get("thermal_class")
            result["overlay_values"] = overlay_values or None
        else:
            result["overlay_values"] = None

        result["values"] = {k: result.get(k) for k in RAW_VALUE_KEYS}

        # Closest grid point coordinates (li0/lo0 already computed above — no duplicate argmin)
        result["lat"] = round(float(lat_arr[li0]), 4)
        result["lon"] = round(float(lon_arr[lo0]), 4)
        result["validTime"] = d["validTime"]
        result["substepMinutes"] = int(d.get("_substepMinutes", 0) or 0)
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
