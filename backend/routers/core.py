from __future__ import annotations

from fastapi import APIRouter


def build_core_router(
    *,
    get_available_runs,
    get_merged_timeline,
    get_models_payload,
    data_cache,
    get_tile_version=None,
):
    router = APIRouter()

    @router.get("/api/health")
    async def health():
        runs = get_available_runs()
        return {"status": "ok", "runs": len(runs), "cache": len(data_cache)}

    @router.get("/api/models")
    async def api_models():
        return get_models_payload()

    @router.get("/api/timesteps")
    async def api_timesteps():
        # Deliberately not cacheable: the frontend refetches this exactly when
        # the ingest pulse reports a change, and must then see the new state.
        merged = get_merged_timeline()
        payload = {"runs": get_available_runs(), "merged": merged}
        if get_tile_version is not None:
            # Token the frontend appends to overlay tile URLs (``v=``).
            payload["tileVersion"] = get_tile_version()
        return payload

    return router
