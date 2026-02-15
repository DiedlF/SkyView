# Overlay Positioning Fix - Testing Checklist

## Quick Test (5 minutes)

### 1. Basic Overlay Display
- [ ] Enable any overlay (Precipitation/Clouds/Thermals)
- [ ] Overlay appears without errors
- [ ] Overlay has reasonable opacity and colors

### 2. Zoom Stability Test
- [ ] Enable overlay (e.g., Precipitation)
- [ ] Zoom in from Z7 to Z12
- [ ] **Expected**: Overlay stays locked to geographic features (doesn't shift)
- [ ] **Before fix**: Overlay would shift/jump relative to hillshade

### 3. Pan Stability Test
- [ ] Enable overlay
- [ ] Pan map in all directions (N/S/E/W)
- [ ] **Expected**: Overlay moves smoothly with map (no independent sliding)
- [ ] **Before fix**: Overlay would sometimes appear to slide relative to map

### 4. Symbol Alignment Test
- [ ] Enable both Convection layer and any overlay
- [ ] Click on several convection symbols
- [ ] **Expected**: Point data popup shows values consistent with overlay colors
- [ ] Example: Blue symbols should show low cloud base, red areas should show high CAPE

## Detailed Test (15 minutes)

### 5. Multi-Layer Alignment
- [ ] Enable Convection symbols
- [ ] Enable Precipitation overlay
- [ ] Zoom to Z10 or Z11 (medium zoom)
- [ ] Find an area with precipitation
- [ ] Click on a symbol in the precipitation area
- [ ] **Expected**: Popup shows rain_gsp or similar precipitation values
- [ ] Verify the colored overlay aligns with where symbols indicate weather

### 6. Edge Cases

#### Empty/Low Data Areas
- [ ] Pan to edge of model domain (far north/south/east/west)
- [ ] Enable overlay
- [ ] **Expected**: Overlay appears where data exists, no errors at boundaries

#### Different Overlay Types
- [ ] Test each overlay type:
  - [ ] Precipitation (multi-variable: rain, snow, graupel)
  - [ ] Sig. Weather (discrete ww codes)
  - [ ] Cloud Cover (continuous percentage)
  - [ ] Thermals (CAPE values)
- [ ] **Expected**: All overlays stay stable when zooming/panning

#### Time Navigation
- [ ] Select different timesteps using timeline
- [ ] **Expected**: Overlay updates correctly for each timestep
- [ ] Overlay stays aligned during timestep changes

### 7. Geographic Accuracy Test
- [ ] Enable overlay at Z12 (highest zoom)
- [ ] Zoom in on a known geographic feature (e.g., Geitau marker)
- [ ] Enable Thermals overlay (shows CAPE in red)
- [ ] Click on a red area
- [ ] **Expected**: Popup shows high CAPE_ML value (>500 J/kg)
- [ ] Click on a white/clear area
- [ ] **Expected**: Popup shows low/zero CAPE

### 8. Performance Test
- [ ] Enable overlay
- [ ] Rapidly zoom in/out several times (Z7 → Z12 → Z7)
- [ ] Rapidly pan around map
- [ ] **Expected**: 
  - Overlay debouncing works (doesn't flicker)
  - No console errors
  - Overlay refreshes after ~400ms debounce
  - Symbols stay on top of overlay

### 9. Model Transition Test (D2 → EU)
- [ ] Navigate to timestep around hour 48-50 (D2/EU transition)
- [ ] Enable overlay
- [ ] Step forward through timeline (hour 47 → 48 → 49 → 50)
- [ ] **Expected**: Overlay continues working across model transition
- [ ] Note: Resolution may change (D2: 2.2km, EU: 6.5km) but alignment should stay correct

## Known Behavior (Not Bugs)

### Visual Differences Are Expected
- **Symbols**: Show aggregated data in zoom-dependent grid cells (Z9: 13km cells)
- **Overlay**: Shows full-resolution data grid (~2.2km for ICON-D2)
- **Result**: Symbol grid and overlay pixels won't align 1:1 at most zoom levels
- **This is intentional**: Symbols aggregate for clarity, overlay shows detail

### At Z12 (2km cells)
- Symbol grid size (2km) ≈ Data grid size (2.2km)
- Symbols and overlay should align most closely at this zoom level

### Projection Effects
- Leaflet uses Web Mercator projection
- At higher latitudes (Alps), east-west distances appear compressed
- This affects both symbols and overlay equally, so alignment is maintained

## Regression Tests

### Verify Nothing Broke
- [ ] Symbols still load and display correctly
- [ ] Point data popups still work
- [ ] Timeline navigation still works
- [ ] Auto-refresh (60s) still works
- [ ] Layer controls (enable/disable) still work

## Success Criteria

✅ **Fix is successful if:**
1. Overlays stay locked to geographic position when zooming
2. Overlays stay locked to geographic position when panning
3. Point data (clicked on symbols) matches overlay colors/patterns
4. No console errors related to overlay
5. No visible "jumping" or "shifting" of overlay during user interaction

❌ **Fix needs revision if:**
1. Overlay still shifts when zooming/panning
2. Overlay appears stretched or distorted
3. Overlay misaligns with hillshade or geographic features
4. Point data contradicts overlay visualization

## Technical Validation

### Check X-Bbox Header
```javascript
// In browser console while overlay is loaded:
// 1. Open DevTools Network tab
// 2. Enable an overlay
// 3. Find the /api/overlay request
// 4. Check Response Headers
// Look for: X-Bbox: 47.39,11.51,47.89,12.51 (example)
// Values should be cell-edge coordinates (half grid resolution outside cell centers)
```

### Verify Grid Math
- Data grid resolution: ~0.02° (ICON-D2), ~0.06° (ICON-EU)
- X-Bbox should extend ±0.01° (ICON-D2) or ±0.03° (ICON-EU) beyond cell centers
- First cell center at 47.40 → X-Bbox min should be ~47.39 ✓

## Reporting Issues

If you find the overlay still shifting or misaligning:
1. Note the zoom level
2. Note which overlay type (precipitation/clouds/etc)
3. Note the timestep (D2 vs EU data)
4. Check browser console for errors
5. Check Network tab for X-Bbox header values
6. Report with screenshots if possible
