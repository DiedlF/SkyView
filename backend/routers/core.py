from __future__ import annotations

from fastapi import APIRouter


def build_core_router(
    *,
    get_available_runs,
    get_merged_timeline,
    get_models_payload,
    data_cache,
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
        merged = get_merged_timeline()
        return {"runs": get_available_runs(), "merged": merged}

    return router
