from __future__ import annotations

from fastapi import APIRouter


def build_admin_router(*, api_status, api_cache_stats, api_usage_stats, api_perf_stats, api_admin_storage, api_admin_logs, api_admin_overview_metrics, admin_view):
    router = APIRouter()
    router.add_api_route('/api/status', api_status, methods=['GET'])
    router.add_api_route('/api/cache_stats', api_cache_stats, methods=['GET'])
    router.add_api_route('/api/usage_stats', api_usage_stats, methods=['GET'])
    router.add_api_route('/api/perf_stats', api_perf_stats, methods=['GET'])
    router.add_api_route('/api/admin/storage', api_admin_storage, methods=['GET'])
    router.add_api_route('/api/admin/logs', api_admin_logs, methods=['GET'])
    router.add_api_route('/api/admin/overview_metrics', api_admin_overview_metrics, methods=['GET'])
    router.add_api_route('/admin', admin_view, methods=['GET'])
    return router
