from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, Query


def build_point_router(
    *,
    resolve_time_with_cache_context,
    load_data,
    POINT_KEYS,
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

    @router.get("/api/point")
    async def api_point(
        lat: float = Query(...),
        lon: float = Query(...),
        time: str = Query("latest"),
        model: Optional[str] = Query(None),
        wind_level: str = Query("10m"),
        zoom: Optional[int] = Query(None, ge=5, le=12),
    ):
        # Selective key loading: only fetch variables actually needed for a point query.
        run, step, model_used = resolve_time_with_cache_context(time, model)
        d = load_data(run, step, model_used, keys=POINT_KEYS)
        fallback_decision = "primary_model_only"

        # EU fallback outside D2 domain or when D2 has no signal at the queried point.
        if model_used == "icon_d2":
            lat_d2 = d["lat"]
            lon_d2 = d["lon"]
            in_d2_domain = (float(np.min(lat_d2)) <= lat <= float(np.max(lat_d2))) and (float(np.min(lon_d2)) <= lon <= float(np.max(lon_d2)))
            li_d2 = int(np.argmin(np.abs(lat_d2 - lat)))
            lo_d2 = int(np.argmin(np.abs(lon_d2 - lon)))
            d2_has_signal = any(
                np.isfinite(float(d[k][li_d2, lo_d2]))
                for k in ("ww", "ceiling", "cape_ml", "hbas_sc")
                if k in d
            )
            if (not in_d2_domain) or (not d2_has_signal):
                trigger = "outside_d2_domain" if (not in_d2_domain) else "d2_missing_signal"
                eu_fb_point = _load_eu_data_strict(time, POINT_KEYS)
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
        li0 = int(np.argmin(np.abs(lat_arr - lat)))
        lo0 = int(np.argmin(np.abs(lon_arr - lon)))
        li = np.array([li0], dtype=int)
        lo = np.array([lo0], dtype=int)

        # ── Variable extraction using direct scalar indexing ──────────────────────
        def _get(key: str) -> float | None:
            if key not in d:
                return None
            v = float(d[key][li0, lo0])
            return None if not np.isfinite(v) else v

        vars_out = ["ww", "clcl", "clcm", "clch", "clct", "cape_ml",
                    "htop_dc", "hbas_sc", "htop_sc", "lpi", "ceiling"]
        result = {}
        for v in vars_out:
            val = _get(v)
            if val is None:
                result[v] = None
            elif v == "ceiling" and val > 20_000:
                result[v] = None
            else:
                result[v] = round(val, 1)

        # Cloud type classification
        best_type = classify_point(
            clcl=_get("clcl") or 0.0,
            clcm=_get("clcm") or 0.0,
            clch=_get("clch") or 0.0,
            cape_ml=_get("cape_ml") or 0.0,
            htop_dc=_get("htop_dc") or 0.0,
            hbas_sc=_get("hbas_sc") or 0.0,
            htop_sc=_get("htop_sc") or 0.0,
            lpi=_get("lpi") or 0.0,
            ceiling=_get("ceiling") or 0.0,
            hsurf=_get("hsurf") or 0.0,
        )
        result["cloudType"] = best_type

        ww_max = int(_get("ww") or 0)
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

        # Overlay values (active overlay info shown in click popup)
        overlay_values = build_overlay_values(
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
        result["overlay_values"] = overlay_values

        # Explorer/Skyview contract convergence: raw values dict
        result["values"] = {k: result.get(k) for k in [
            "ww", "clcl", "clcm", "clch", "clct", "cape_ml", "htop_dc", "hbas_sc", "htop_sc", "lpi", "ceiling"
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

        return result

    return router
