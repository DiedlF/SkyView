from __future__ import annotations

from datetime import datetime, timezone

import json
import numpy as np
from fastapi import APIRouter, Query


def build_domain_router(
    *,
    get_merged_timeline,
    resolve_time_with_cache_context,
    load_data,
    _freshness_minutes_from_run,
    _merge_axis_aligned_segments,
    DATA_DIR,
):
    router = APIRouter()

    @router.get("/api/d2_domain")
    async def api_d2_domain(time: str = Query("latest")):
        """Return ICON-D2 domain bounds and boundary of last valid cells.

        Boundary is hidden when requested time resolves to non-D2 primary model (EU-only window).
        """
        merged = get_merged_timeline()
        merged_model = None
        if merged and merged.get("steps"):
            steps = merged.get("steps", [])
            t_req = (time or "latest").strip()
            if t_req == "" or t_req.lower() == "latest":
                merged_model = steps[-1].get("model")
            else:
                exact = next((s for s in steps if s.get("validTime") == t_req), None)
                if exact is not None:
                    merged_model = exact.get("model")
                else:
                    try:
                        target = datetime.fromisoformat(t_req.replace("Z", "+00:00"))
                        if target.tzinfo is None:
                            target = target.replace(tzinfo=timezone.utc)
                        best = min(
                            steps,
                            key=lambda s: abs((datetime.fromisoformat(s["validTime"].replace("Z", "+00:00")) - target).total_seconds())
                        )
                        merged_model = best.get("model")
                    except Exception:
                        merged_model = None

        if merged_model is not None and merged_model != "icon_d2":
            return {
                "model": "icon_d2",
                "run": None,
                "validTime": None,
                "bbox": None,
                "cellEdgeBbox": None,
                "boundarySegments": [],
                "diagnostics": {
                    "dataFreshnessMinutes": None,
                    "validCells": 0,
                    "boundarySegmentCount": 0,
                    "source": "suppressed_non_d2_timestep",
                    "requestedModel": merged_model,
                    "requestedTime": time,
                },
            }

        run, step, _ = resolve_time_with_cache_context(time, "icon_d2")
        d = load_data(run, step, "icon_d2", keys=["ww", "mh"])

        # Fast path: use precomputed run-level boundary generated at ingestion.
        run_dir = f"{DATA_DIR}/icon-d2/{run}"
        boundary_cache_path = f"{run_dir}/_d2_boundary.json"
        try:
            with open(boundary_cache_path, "r", encoding="utf-8") as f:
                bc = json.load(f)
            return {
                "model": "icon_d2",
                "run": run,
                "validTime": d["validTime"],
                "bbox": bc.get("bbox"),
                "cellEdgeBbox": {
                    **(bc.get("cellEdgeBbox") or {}),
                    "latRes": bc.get("latRes"),
                    "lonRes": bc.get("lonRes"),
                },
                "boundarySegments": bc.get("boundarySegments", []),
                "diagnostics": {
                    "dataFreshnessMinutes": _freshness_minutes_from_run(run),
                    "validCells": bc.get("validCells", 0),
                    "boundarySegmentCount": bc.get("boundarySegmentCount", 0),
                    "source": "precomputed",
                },
            }
        except Exception:
            pass

        # Fallback path: compute boundary if cache file is not present.
        lat = d["lat"]
        lon = d["lon"]
        ww = d.get("ww")
        mh = d.get("mh")
        lat_min = float(np.min(lat))
        lat_max = float(np.max(lat))
        lon_min = float(np.min(lon))
        lon_max = float(np.max(lon))
        lat_res = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.02
        lon_res = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.02
        if mh is not None:
            valid = np.isfinite(mh)
        else:
            valid = np.isfinite(ww) if ww is not None else np.ones((len(lat), len(lon)), dtype=bool)

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

        return {
            "model": "icon_d2",
            "run": run,
            "validTime": d["validTime"],
            "bbox": {
                "latMin": lat_min,
                "lonMin": lon_min,
                "latMax": lat_max,
                "lonMax": lon_max,
            },
            "cellEdgeBbox": {
                "latMin": lat_min - lat_res / 2.0,
                "lonMin": lon_min - lon_res / 2.0,
                "latMax": lat_max + lat_res / 2.0,
                "lonMax": lon_max + lon_res / 2.0,
                "latRes": lat_res,
                "lonRes": lon_res,
            },
            "boundarySegments": segments,
            "diagnostics": {
                "dataFreshnessMinutes": _freshness_minutes_from_run(run),
                "validCells": int(np.count_nonzero(valid)),
                "boundarySegmentCount": len(segments),
                "source": "computed",
            },
        }

    return router
