"""Static frontend serving with explicit cache policy.

Starlette's ``StaticFiles`` only sends ETag/Last-Modified. Browsers then guess a
lifetime and a CDN that honours origin headers (IPv64 CDN64) caches nothing.
"""

from __future__ import annotations

from urllib.parse import parse_qs

from fastapi.staticfiles import StaticFiles

# index.html references app.js?v=NN etc.; a URL with a version never changes
# content, so it can be cached "forever". Bump the number whenever the file
# changes - that was already required, browsers heuristically cached these
# files for weeks.
STATIC_CACHE_VERSIONED = "public, max-age=31536000, immutable"
# Icons, symbol SVGs, geodata: change rarely, no version in the URL.
STATIC_CACHE_DEFAULT = "public, max-age=86400"
# HTML must always be revalidated: it carries the asset version numbers, and
# every real page load should reach the app (visit statistics, fast rollouts).
STATIC_CACHE_HTML = "no-cache"


def static_cache_control(path: str, query_string: bytes | str = b"") -> str:
    """Cache-Control value for a static file request path + query string."""
    if isinstance(query_string, bytes):
        query_string = query_string.decode("latin-1")
    name = str(path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    if name in ("", ".") or "." not in name or name.endswith((".html", ".htm")):
        return STATIC_CACHE_HTML
    if parse_qs(query_string).get("v"):
        return STATIC_CACHE_VERSIONED
    return STATIC_CACHE_DEFAULT


class CachedStaticFiles(StaticFiles):
    """``StaticFiles`` that adds Cache-Control to successful responses."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code in (200, 304):
            served = getattr(response, "path", None) or path
            response.headers["Cache-Control"] = static_cache_control(
                str(served), scope.get("query_string", b"")
            )
        return response
