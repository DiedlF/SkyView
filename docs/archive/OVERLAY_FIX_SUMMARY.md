# Overlay Positioning Bugs - Fixed

## Issues Identified

### Bug 1: Overlay Movement on Zoom/Pan
**Problem**: The overlay image would shift relative to the map when zooming or panning, instead of staying fixed to its geographic position.

**Root Cause**: The `/api/overlay` endpoint was returning the clamped **requested viewport bounds** in the `X-Bbox` header, rather than the actual geographic bounds of the data grid cells contained in the image.

### Bug 2: Misalignment Between Layers
**Problem**: The overlay, convection symbols, and point tooltips appeared misaligned, as if using different coordinate systems.

**Root Cause**: Same as Bug 1. The overlay bounds didn't accurately represent where the image pixels should be placed on the map. Since the data grid has discrete cells with coordinates at cell **centers**, but image bounds specify image **corners**, there was a mismatch.

## The Fix

### Changed: `/root/.openclaw/workspace/skyview/backend/app.py`

In the `/api/overlay` endpoint, added calculation of **actual grid cell edge bounds**:

```python
# Calculate actual grid cell bounds
# Grid coordinates are cell centers; we need cell edges for image bounds
lat_res = float(lat[1] - lat[0]) if len(lat) > 1 else 0.02
lon_res = float(lon[1] - lon[0]) if len(lon) > 1 else 0.02

# Actual bounds = edges of selected cells (not requested bbox)
actual_lat_min = float(lat[li[0]]) - lat_res / 2
actual_lat_max = float(lat[li[-1]]) + lat_res / 2
actual_lon_min = float(lon[lo[0]]) - lon_res / 2
actual_lon_max = float(lon[lo[-1]]) + lon_res / 2
```

**Key changes:**
1. Calculate grid resolution from the lat/lon arrays
2. Use **cell-edge coordinates** instead of cell-center coordinates
3. Update aspect ratio calculation to use actual bounds
4. Return actual bounds in `X-Bbox` header

## How It Works

### Before (Buggy)
1. Client requests overlay for viewport bbox: `47.0,11.0,47.5,12.0`
2. Server selects grid cells whose centers fall within this bbox
3. Server creates image from these cells (e.g., cells from 47.00 to 47.48)
4. Server returns bbox: `47.0,11.0,47.5,12.0` (the requested bounds)
5. **Problem**: Image contains cells 47.00-47.48, but Leaflet stretches it to 47.0-47.5
6. Result: Misalignment and shifting on zoom/pan

### After (Fixed)
1. Client requests overlay for viewport bbox: `47.0,11.0,47.5,12.0`
2. Server selects grid cells whose centers fall within this bbox
3. Server calculates **actual cell edges**: `46.99,11.01,47.49,12.01`
4. Server creates image from these cells
5. Server returns actual bounds: `46.99,11.01,47.49,12.01`
6. **Result**: Image bounds match actual data extent
7. Overlay stays fixed to its geographic position ✓

## Verification

### Test Case
```python
# Data grid: 0.02° resolution
lat = [47.00, 47.02, 47.04, ..., 47.48]
lon = [11.02, 11.04, 11.06, ..., 12.00]

# Selected cells: 25 lat × 50 lon
# Old bbox: 47.0 to 47.5 (doesn't match cell extent)
# New bbox: 46.99 to 47.49 (matches cell edges) ✓
```

### Expected Behavior After Fix
1. **Overlay stability**: Overlay stays fixed when zooming/panning
2. **Alignment**: Overlay pixels align with data grid cells
3. **Symbols alignment**: Symbols (which use aggregated grid cells) align with underlying data
4. **Point tooltip accuracy**: Clicking on features shows correct data

## Related Code Components

### Frontend (`app.js`)
- `loadOverlay()`: Fetches overlay, reads `X-Bbox` header, creates Leaflet imageOverlay
- No changes needed - already correctly using the header

### Backend (`app.py`)
- `/api/overlay`: **FIXED** - Now returns actual cell-edge bounds
- `/api/symbols`: Uses fixed grid with absolute origin (45.5, 9.0) - no changes needed
- `/api/point`: Uses nearest-neighbor lookup - no changes needed

## Testing Recommendations

1. **Zoom in/out**: Overlay should stay locked to geographic features
2. **Pan around**: Overlay should not shift relative to the hillshade
3. **Compare layers**: Click on symbols and verify point data makes sense
4. **Try different overlays**: Precipitation, clouds, thermals should all align
5. **Check at different zoom levels**: Z7-Z12 should all work correctly

## Notes

- The symbols layer uses a **fixed grid** (zoom-dependent cell size) for aggregation and stability
- The overlay uses the **native data grid** (~0.02° for ICON-D2) for full resolution
- Slight visual differences are expected (symbols show aggregated cells, overlay shows full resolution)
- The fix ensures both layers correctly reference the underlying data grid coordinates
