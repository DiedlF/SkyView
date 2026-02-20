from __future__ import annotations

from fastapi import APIRouter


def build_overlay_router(*, api_overlay, api_overlay_tile):
    """Register overlay endpoints using injected handlers."""
    router = APIRouter()
    router.add_api_route("/api/overlay", api_overlay, methods=["GET"])
    router.add_api_route("/api/overlay_tile/{z}/{x}/{y}.png", api_overlay_tile, methods=["GET"])
    return router
