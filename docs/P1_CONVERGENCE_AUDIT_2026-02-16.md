# P1 Convergence Audit — 2026-02-16

## Scope
Audit remaining Skyview vs Explorer differences after shared helper extraction.

## Shared modules now in use
- `backend/api_contract.py` (layer aliases)
- `backend/time_contract.py` (runs/timeline/time resolution)
- `backend/grid_utils.py` (bbox + slicing)
- `backend/point_data.py` (`build_overlay_values_from_raw`)
- `backend/model_caps.py` (`/api/models` payload)
- `backend/response_headers.py` (overlay/tile headers)

## Intentional differences (kept)
1. **Rendering backend**
   - Skyview: PIL + vectorized path optimized for product overlays and tile API.
   - Explorer: matplotlib-centric raw variable explorer.
   - Rationale: Explorer remains a diagnostic/raw-data app.

2. **Endpoint richness**
   - Skyview has domain-specific endpoints (`/api/symbols`, `/api/wind`, richer point semantics).
   - Explorer focuses on generic variable/tile exploration.

3. **UI behavior and controls**
   - Skyview frontend includes pilot-centric layer grouping and workflow.
   - Explorer frontend keeps lower-level controls for debugging and raw inspection.

## Extractable differences still possible (optional)
- Shared error/health middleware package for both backends (currently equivalent behavior but duplicated implementation details).
- Shared tile prewarm/caching policy constants if Explorer adopts identical behavior.

## Conclusion
No critical accidental schema drift remains for converged endpoints (`/api/models`, `/api/timesteps`, `/api/overlay`, `/api/overlay_tile`, `/api/point`).
Remaining differences are intentional product-role differences.
