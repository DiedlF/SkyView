from __future__ import annotations

from fastapi import APIRouter


def build_ops_router(*, api_feedback, api_feedback_list, api_feedback_update, api_marker_profile, api_marker_auth, api_marker_profile_set, api_location_search, api_markers_list, api_markers_reset_default):
    router = APIRouter()
    router.add_api_route('/api/feedback', api_feedback, methods=['POST'])
    router.add_api_route('/api/feedback', api_feedback_list, methods=['GET'])
    router.add_api_route('/api/feedback/{item_id}', api_feedback_update, methods=['PATCH'])
    router.add_api_route('/api/marker_profile', api_marker_profile, methods=['GET'])
    router.add_api_route('/api/marker_auth', api_marker_auth, methods=['GET'])
    router.add_api_route('/api/marker_profile', api_marker_profile_set, methods=['POST'])
    router.add_api_route('/api/location_search', api_location_search, methods=['GET'])
    router.add_api_route('/api/markers', api_markers_list, methods=['GET'])
    router.add_api_route('/api/markers', api_markers_reset_default, methods=['DELETE'])
    return router
