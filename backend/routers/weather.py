from __future__ import annotations

from fastapi import APIRouter


def build_weather_router(*, api_symbols, api_wind, api_emagram_point):
    """Register weather-map endpoints using injected handlers."""
    router = APIRouter()
    router.add_api_route("/api/symbols", api_symbols, methods=["GET"])
    router.add_api_route("/api/wind", api_wind, methods=["GET"])
    router.add_api_route("/api/emagram_point", api_emagram_point, methods=["GET"])
    return router
