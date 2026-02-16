// app.js - Main application logic

let map, symbolLayer, windLayer, overlayLayer = null, debounceTimer, overlayDebounce, windDebounce, timesteps = [], currentTimeIndex = 0, currentRun = '';
let overlayRequestSeq = 0;
let overlayAbortCtrl = null;
let overlayPrewarmCtrl = null;
let overlayObjectUrl = null;
let currentOverlay = 'none';
let windEnabled = false;
let windLevel = '10m';
let modelCapabilities = {};  // Store model capabilities from API

let apiFailureStreak = 0;
let healthBadgeEl = null;

function ensureHealthBadge() {
  if (healthBadgeEl) return healthBadgeEl;
  const el = document.createElement('div');
  el.id = 'api-health-indicator';
  el.style.position = 'fixed';
  el.style.top = '10px';
  el.style.right = '10px';
  el.style.zIndex = '2000';
  el.style.padding = '6px 10px';
  el.style.borderRadius = '8px';
  el.style.fontSize = '12px';
  el.style.background = 'rgba(200,50,50,0.9)';
  el.style.color = '#fff';
  el.style.display = 'none';
  document.body.appendChild(el);
  healthBadgeEl = el;
  return el;
}

function updateHealthIndicator() {
  const el = ensureHealthBadge();
  if (apiFailureStreak >= 3) {
    el.textContent = `Data degraded (${apiFailureStreak} failures)`;
    el.style.display = 'block';
  } else {
    el.style.display = 'none';
  }
}

function markApiSuccess() {
  if (apiFailureStreak !== 0) {
    apiFailureStreak = 0;
    updateHealthIndicator();
  }
}

function markApiFailure(context, err) {
  apiFailureStreak += 1;
  updateHealthIndicator();
  if (err && err.requestId) {
    console.error(`${context} failed (requestId=${err.requestId})`, err);
  }
}

async function throwHttpError(res, context = 'request') {
  let detail = `HTTP ${res.status}`;
  let requestId = res.headers.get('X-Request-Id') || null;
  try {
    const j = await res.json();
    if (j && j.detail) detail = j.detail;
    if (j && j.requestId) requestId = j.requestId;
  } catch (_e) {
    // ignore non-JSON body
  }
  const err = new Error(`${context}: ${detail}`);
  err.status = res.status;
  err.requestId = requestId;
  throw err;
}

// Legend definitions for each effective backend overlay layer
const LEGEND_CONFIGS = {
  total_precip: { title: 'Precipitation: Total', gradient: 'linear-gradient(to right, rgb(150,255,255), rgb(100,200,255), rgb(50,150,255), rgb(0,100,255))', labels: ['0.1 mm/h', '5+ mm/h'] },
  rain: { title: 'Precipitation: Rain', gradient: 'linear-gradient(to right, rgb(180,220,255), rgb(100,160,230), rgb(20,60,180))', labels: ['0.1 mm/h', '5+ mm/h'] },
  snow: { title: 'Precipitation: Snow', gradient: 'linear-gradient(to right, rgb(255,200,255), rgb(210,120,230), rgb(120,40,160))', labels: ['0.1 mm/h', '5+ mm/h'] },
  hail: { title: 'Precipitation: Hail/Graupel', gradient: 'linear-gradient(to right, rgb(200,160,20), rgb(240,100,30), rgb(255,80,20))', labels: ['0.1 mm/h', '5+ mm/h'] },
  clouds_low: { title: 'Cloud Cover: Low', gradient: 'linear-gradient(to right, rgb(225,225,225), rgb(45,45,45))', labels: ['1%', '100%'] },
  clouds_mid: { title: 'Cloud Cover: Mid', gradient: 'linear-gradient(to right, rgb(225,225,225), rgb(45,45,45))', labels: ['1%', '100%'] },
  clouds_high: { title: 'Cloud Cover: High', gradient: 'linear-gradient(to right, rgb(225,225,225), rgb(45,45,45))', labels: ['1%', '100%'] },
  clouds_total: { title: 'Cloud Cover: Total', gradient: 'linear-gradient(to right, rgb(225,225,225), rgb(45,45,45))', labels: ['1%', '100%'] },
  clouds_total_mod: { title: 'Cloud Cover: Total_mod', gradient: 'linear-gradient(to right, rgb(225,225,225), rgb(45,45,45))', labels: ['1%', '100%'] },
  dry_conv_top: { title: 'Dry Convection Top', gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))', labels: ['0m', '9900m'] },
  sigwx: { title: 'Significant weather', gradient: 'linear-gradient(to right, rgba(0,0,0,0), rgb(205,205,205), rgb(145,145,145), rgb(85,85,85), rgb(160,170,40), rgb(70,180,80), rgb(70,180,210), rgb(145,110,230), rgb(220,40,80))', labels: ['ww0 clear', 'ww severe'] },
  ceiling: { title: 'Ceiling', gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))', labels: ['0m', '9900m'] },
  cloud_base: { title: 'Cloud base (convective)', gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))', labels: ['0m', '5000m'] },
  conv_thickness: { title: 'Cloud thickness (convective)', gradient: 'linear-gradient(to right, rgb(40,220,60), rgb(200,200,40), rgb(240,80,40))', labels: ['0m', '6000m'] },
  thermals: { title: 'CAPE_ml', gradient: 'linear-gradient(to right, rgb(50,180,50), rgb(150,150,50), rgb(220,100,30), rgb(255,50,50))', labels: ['50 J/kg', '1000+ J/kg'] },
  climb_rate: { title: 'Climb Rate', gradient: 'linear-gradient(to right, rgb(50,200,50), rgb(180,200,50), rgb(220,150,30), rgb(255,50,50))', labels: ['0 m/s', '5 m/s'] },
  lcl: { title: 'Cloud Base (LCL) MSL', gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))', labels: ['0m', '5000m MSL'] }
};

// Initialize map
map = L.map('map', {
  center: [47.6836, 11.9610],
  zoom: 9,
  minZoom: 5,
  maxZoom: 12,
  maxBounds: L.latLngBounds([[42, -5], [59, 21]]),
  maxBoundsViscosity: 1.0,
  zoomControl: false
});
L.control.zoom({ position: 'bottomleft' }).addTo(map);

// Ocean base provides water coloring (ocean, lakes, rivers) with matching coastlines
// maxNativeZoom=10: inland tiles unavailable at z11+, so upscale z10 tiles
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
  attribution: '',
  maxNativeZoom: 10,
  maxZoom: 12
}).addTo(map);

// Hillshade on top with multiply blend mode — shading applies over the ocean base colors
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}', {
  attribution: '',
  className: 'hillshade-layer'
}).addTo(map);

// Geitau marker
L.circleMarker([47.6836, 11.9610], {
  radius: 6,
  fillColor: '#ff0000',
  color: '#8B0000',
  weight: 2,
  opacity: 0.8,
  fillOpacity: 0.7
}).bindTooltip('Geitau').addTo(map);

symbolLayer = L.layerGroup().addTo(map);
windLayer = L.layerGroup();

// Format date as "DayOfWeek, dd.mm., HH UTC"
function formatDateShort(d) {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${days[d.getUTCDay()]}, ${dd}.${mm}., ${hh} UTC`;
}

// Update info panel zoom/grid
const CELL_KM = {5:200, 6:110, 7:55, 8:28, 9:13, 10:7, 11:3, 12:2};
function updateZoom() {
  const z = map.getZoom();
  document.getElementById('zoom-grid').textContent = `Z${z}, ${CELL_KM[z] || 2}km`;
}
map.on('zoomend', updateZoom);
updateZoom(); // initial

// Load model capabilities
async function loadModelCapabilities() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error('Failed to fetch model info');
    const data = await res.json();
    markApiSuccess();
    
    // Index by model name
    data.models.forEach(m => {
      modelCapabilities[m.name] = m;
    });
  } catch (e) {
    console.error('Error loading model capabilities:', e);
    markApiFailure('model capabilities', e);
    // Fallback: assume ICON-D2 capabilities
    modelCapabilities = {
      'icon_d2': {
        name: 'icon-d2',
        label: 'ICON-D2 (2.2km)',
        timesteps: [...Array(48).keys()].map(i => i + 1),
        maxHours: 48
      }
    };
  }
}

// Check if a forecast hour is available for a given model
function isTimestepAvailable(forecastHour, modelName) {
  const model = modelCapabilities[modelName] || modelCapabilities['icon_d2'];
  return model.timesteps.includes(forecastHour);
}

// Load timesteps
async function loadTimesteps() {
  try {
    // Load model capabilities first if not loaded
    if (Object.keys(modelCapabilities).length === 0) {
      await loadModelCapabilities();
    }
    
    const res = await fetch('/api/timesteps');
    if (!res.ok) throw new Error('Failed to fetch timesteps');
    const data = await res.json();
    
    // Use merged timeline (D2 for first 48h + EU for extended range)
    const merged = data.merged;
    if (merged && merged.steps.length) {
      timesteps = merged.steps;
      currentTimeIndex = 0;
      currentRun = merged.run;
      document.getElementById('run-time').textContent = merged.runTime ? formatDateShort(new Date(merged.runTime)) : 'N/A';
      document.getElementById('model').textContent = 'ICON-D2 + EU';
    } else {
      // Fallback: use first available run
      const latestRun = data.runs && data.runs[0];
      if (!latestRun || !latestRun.steps.length) throw new Error('No data available');
      timesteps = latestRun.steps;
      currentTimeIndex = 0;
      currentRun = latestRun.run;
      document.getElementById('run-time').textContent = latestRun.runTime ? formatDateShort(new Date(latestRun.runTime)) : 'N/A';
      document.getElementById('model').textContent = latestRun.model ? latestRun.model.toUpperCase().replace('_', '-') : 'ICON-D2';
    }
    
    buildTimeline();
    loadSymbols();
    markApiSuccess();
  } catch (e) {
    console.error('Error loading timesteps:', e);
    markApiFailure('timesteps', e);
    document.getElementById('run-time').textContent = 'N/A';
  }
}

// Update valid time in info panel
function updateValidTime() {
  // Valid time display removed — kept as no-op for call sites
}

// Load symbols
async function loadSymbols() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    const b = map.getBounds();
    const bbox = `${b.getSouth()},${b.getWest()},${b.getNorth()},${b.getEast()}`;
    const zoom = map.getZoom();
    const time = timesteps[currentTimeIndex]?.validTime || '';
    try {
      // Pass model hint if timestep has model info (from merged timeline)
      const step = timesteps[currentTimeIndex];
      const modelParam = step && step.model ? `&model=${step.model}` : '';
      const res = await fetch(`/api/symbols?bbox=${bbox}&zoom=${zoom}&time=${encodeURIComponent(time)}${modelParam}`);
      if (!res.ok) await throwHttpError(res, 'API');
      const data = await res.json();
      markApiSuccess();
      symbolLayer.clearLayers();
      data.symbols.forEach(sym => {
        if (!sym.clickable) return;

        const icon = createSymbolIcon(sym.type, sym.cloudBase);
        const marker = L.marker([sym.lat, sym.lon], {
          icon,
          bubblingMouseEvents: true  // Allow touch events to bubble for pinch-zoom
        }).addTo(symbolLayer);
        marker.on('click', () => {
          const curStep = timesteps[currentTimeIndex];
          const curTime = curStep?.validTime || time;
          const curModel = curStep?.model || null;
          loadPoint(sym.clickLat ?? sym.lat, sym.clickLon ?? sym.lon, curTime, curModel, windLevel, map.getZoom());
        });
      });
      
      // Update info panel
      // Override with precise cellSize if available
      if (data.cellSize) {
        const preciseKm = Math.round(data.cellSize * 111);
        document.getElementById('zoom-grid').textContent = `Z${zoom}, ${preciseKm}km`;
      }
      
      // (valid time display removed from UI)
      
      // Update model name from response (changes when switching between D2/EU timesteps)
      if (data.model) {
        document.getElementById('model').textContent = data.model.toUpperCase().replace('_', '-');
      }
    } catch (e) {
      console.error('Error loading symbols:', e);
      markApiFailure('symbols', e);
      // Optionally show error in UI
      symbolLayer.clearLayers();
    }
  }, 300);
}

// Overlay value formatting (aligned with API_CONVERGENCE_CONTRACT units)
const OVERLAY_META = {
  total_precip: { label: 'Total precip', unit: 'mm/h', decimals: 2 },
  rain: { label: 'Rain', unit: 'mm/h', decimals: 2 },
  snow: { label: 'Snow', unit: 'mm/h', decimals: 2 },
  hail: { label: 'Hail/Graupel', unit: 'mm/h', decimals: 2 },
  sigwx: { label: 'Sig. Weather ww', unit: '', integer: true },
  clouds_low: { label: 'Cloud cover low', unit: '%', decimals: 1 },
  clouds_mid: { label: 'Cloud cover mid', unit: '%', decimals: 1 },
  clouds_high: { label: 'Cloud cover high', unit: '%', decimals: 1 },
  clouds_total: { label: 'Cloud cover total', unit: '%', decimals: 1 },
  clouds_total_mod: { label: 'Cloud cover total mod', unit: '%', decimals: 1 },
  ceiling: { label: 'Ceiling', unit: 'm', integer: true },
  cloud_base: { label: 'Cloud base', unit: 'm', integer: true },
  dry_conv_top: { label: 'Dry convection top', unit: 'm', integer: true },
  conv_thickness: { label: 'Convective thickness', unit: 'm', integer: true },
  thermals: { label: 'CAPE_ml', unit: 'J/kg', decimals: 1 },
  climb_rate: { label: 'Climb', unit: 'm/s', decimals: 1 },
  lcl: { label: 'Cloud base (LCL)', unit: 'm MSL', integer: true },
};

function formatOverlayValue(key, value) {
  if (value == null) return null;
  const meta = OVERLAY_META[key];
  if (!meta) return null;

  let out;
  if (meta.integer) out = Math.round(Number(value));
  else if (typeof meta.decimals === 'number') out = Number(value).toFixed(meta.decimals);
  else out = String(value);

  return meta.unit ? `${meta.label}: ${out} ${meta.unit}` : `${meta.label}: ${out}`;
}

// Symbol type display names
const SYMBOL_NAMES = {
  clear: 'Clear', ci: 'Cirrus', cs: 'Cirrostratus', cc: 'Cirrocumulus',
  ac: 'Altocumulus', as: 'Altostratus', st: 'Stratus', sc: 'Stratocumulus',
  cu_hum: 'Cu humilis', cu_con: 'Cu congestus', cb: 'Cumulonimbus',
  blue_thermal: 'Blue thermal',
  fog: 'Fog', rime_fog: 'Rime fog',
  drizzle_light: 'Drizzle (light)', drizzle_moderate: 'Drizzle (mod)', drizzle_dense: 'Drizzle (dense)',
  freezing_drizzle: 'Freezing drizzle', freezing_drizzle_heavy: 'Freezing drizzle (heavy)',
  rain_slight: 'Rain (slight)', rain_moderate: 'Rain (moderate)', rain_heavy: 'Rain (heavy)',
  freezing_rain: 'Freezing rain', freezing_rain_heavy: 'Freezing rain (heavy)',
  rain_shower: 'Rain shower', rain_shower_moderate: 'Rain shower (mod)', rain_shower_heavy: 'Rain shower (heavy)',
  snow_slight: 'Snow (slight)', snow_moderate: 'Snow (moderate)', snow_heavy: 'Snow (heavy)',
  snow_grains: 'Snow grains',
  snow_shower: 'Snow shower', snow_shower_heavy: 'Snow shower (heavy)',
  thunderstorm: 'Thunderstorm', thunderstorm_hail: 'Thunderstorm (hail)',
};

// Load point details
async function loadPoint(lat, lon, time, model, windLvl = '10m', zoom = null) {
  try {
    if (!time) time = 'latest';
    const modelParam = model ? `&model=${model}` : '';
    const windParam = windLvl ? `&wind_level=${encodeURIComponent(windLvl)}` : '';
    const zoomParam = (zoom != null) ? `&zoom=${encodeURIComponent(zoom)}` : '';
    const res = await fetch(`/api/point?lat=${lat}&lon=${lon}&time=${encodeURIComponent(time)}${modelParam}${windParam}${zoomParam}`);
    if (!res.ok) await throwHttpError(res, 'API');
    const data = await res.json();

    const symbolName = SYMBOL_NAMES[data.symbol] || data.symbol || 'N/A';
    let lines = [`<b>${symbolName}</b>`];

    // Show active overlay value if an overlay is selected
    const overlayKey = getEffectiveOverlayLayer();
    if (overlayKey !== 'none' && data.overlay_values) {
      const formatted = formatOverlayValue(overlayKey, data.overlay_values[overlayKey]);
      if (formatted) lines.push(formatted);
    }

    // Wind info in tooltip (only when wind layer is enabled)
    if (windEnabled && data.overlay_values && data.overlay_values.wind_speed != null) {
      const kt = Number(data.overlay_values.wind_speed);
      const kmh = kt * 1.852;
      const dir = Number(data.overlay_values.wind_dir);
      const windText = `Wind: ${Math.round(kmh)} km/h (${Math.round(kt)} kt)`;
      if (Number.isFinite(dir)) {
        lines.push(`${windText}, ${Math.round(dir)}°`);
      } else {
        lines.push(windText);
      }
    }

    L.popup({ maxWidth: 200 })
      .setLatLng([lat, lon])
      .setContent(lines.join('<br/>'))
      .openOn(map);
  } catch (e) {
    console.error('Error loading point:', e);
    markApiFailure('point', e);
    L.popup()
      .setLatLng([lat, lon])
      .setContent('Error loading details')
      .openOn(map);
  }
}

// Overlay loading — uses FIXED full-extent bbox so overlay never shifts on pan/zoom
// Use viewport bbox for overlays (not full domain) for performance
function getOverlayParams() {
  const b = map.getBounds();
  // Clamp to data extent
  const latMin = Math.max(b.getSouth(), 43.18);
  const lonMin = Math.max(b.getWest(), -3.94);
  const latMax = Math.min(b.getNorth(), 58.08);
  const lonMax = Math.min(b.getEast(), 20.34);
  const bbox = `${latMin},${lonMin},${latMax},${lonMax}`;
  // Width proportional to viewport, capped
  const width = Math.min(800, Math.round(window.innerWidth * 1.2));
  return { bbox, width };
}

function getEffectiveOverlayLayer() {
  if (currentOverlay === 'precip') {
    return document.getElementById('precip-type')?.value || 'total_precip';
  }
  if (currentOverlay === 'clouds') {
    return document.getElementById('clouds-type')?.value || 'clouds_total';
  }
  return currentOverlay;
}

function overlayOpacityForLayer(layer) {
  if (layer && layer.startsWith('clouds_')) return 0.9;
  if (layer === 'sigwx') return 0.85;
  return 0.6;
}

function lonLatToTile(lat, lon, z) {
  const n = 2 ** z;
  const x = Math.floor((lon + 180) / 360 * n);
  const latRad = lat * Math.PI / 180;
  const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
  return { x, y };
}

async function prewarmOverlayTiles(params) {
  if (!overlayLayer || !map) return;
  if (overlayPrewarmCtrl) overlayPrewarmCtrl.abort();
  overlayPrewarmCtrl = new AbortController();
  const signal = overlayPrewarmCtrl.signal;

  const z = map.getZoom();
  const b = map.getBounds();
  const nw = lonLatToTile(b.getNorth(), b.getWest(), z);
  const se = lonLatToTile(b.getSouth(), b.getEast(), z);
  const minX = Math.min(nw.x, se.x) - 1;
  const maxX = Math.max(nw.x, se.x) + 1;
  const minY = Math.min(nw.y, se.y) - 1;
  const maxY = Math.max(nw.y, se.y) + 1;

  const maxTile = (2 ** z) - 1;
  const urls = [];
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      if (x < 0 || y < 0 || x > maxTile || y > maxTile) continue;
      urls.push(`/api/overlay_tile/${z}/${x}/${y}.png?${params.toString()}`);
    }
  }

  // Limit prewarm burst size to avoid spiking backend/network.
  const limited = urls.slice(0, 36);
  const concurrency = 6;
  let idx = 0;
  const workers = Array.from({ length: concurrency }, async () => {
    while (idx < limited.length) {
      const i = idx++;
      const u = limited[i];
      try {
        await fetch(u, { signal, cache: 'force-cache' });
      } catch (_e) {
        // best-effort prewarm; ignore failures
      }
    }
  });
  await Promise.all(workers);
}

async function loadOverlay() {
  clearTimeout(overlayDebounce);
  overlayDebounce = setTimeout(async () => {
    const reqId = ++overlayRequestSeq;

    // Cancel previous in-flight overlay request
    if (overlayAbortCtrl) {
      overlayAbortCtrl.abort();
      overlayAbortCtrl = null;
    }
    overlayAbortCtrl = new AbortController();

    // Remove existing overlay
    if (overlayLayer) {
      map.removeLayer(overlayLayer);
      overlayLayer = null;
    }
    if (overlayObjectUrl) {
      URL.revokeObjectURL(overlayObjectUrl);
      overlayObjectUrl = null;
    }

    if (currentOverlay === 'none') return;

    const effectiveOverlay = getEffectiveOverlayLayer();
    if (!effectiveOverlay || effectiveOverlay === 'none') return;

    const time = timesteps[currentTimeIndex]?.validTime || '';

    try {
      const overlayStep = timesteps[currentTimeIndex];
      const clientClass = window.innerWidth <= 768 ? 'mobile' : 'desktop';

      const params = new URLSearchParams({
        layer: effectiveOverlay,
        time,
        clientClass,
      });
      if (overlayStep && overlayStep.model) params.append('model', overlayStep.model);

      // Ignore stale responses that arrive out of order
      if (reqId !== overlayRequestSeq) return;

      const tileUrl = `/api/overlay_tile/{z}/{x}/{y}.png?${params.toString()}`;
      overlayLayer = L.tileLayer(tileUrl, {
        tileSize: 256,
        opacity: overlayOpacityForLayer(effectiveOverlay),
        updateWhenZooming: false,
        updateWhenIdle: true,
        keepBuffer: 1,
        crossOrigin: true,
        zIndex: 250,
      }).addTo(map);

      // Best-effort prewarm: viewport + 1-ring tiles after layer/time switch.
      prewarmOverlayTiles(params).catch(() => {});

      if (symbolLayer && typeof symbolLayer.eachLayer === 'function') {
        symbolLayer.eachLayer(layer => {
          if (layer && typeof layer.bringToFront === 'function') layer.bringToFront();
        });
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Overlay error:', e);
        markApiFailure('overlay', e);
      }
    }
  }, 220);
}

// Load wind barbs
async function loadWind() {
  clearTimeout(windDebounce);
  windDebounce = setTimeout(async () => {
    windLayer.clearLayers();
    if (!windEnabled) return;

    const b = map.getBounds();
    const bbox = `${b.getSouth()},${b.getWest()},${b.getNorth()},${b.getEast()}`;
    const zoom = map.getZoom();
    const time = timesteps[currentTimeIndex]?.validTime || '';
    const step = timesteps[currentTimeIndex];
    const modelParam = step && step.model ? `&model=${step.model}` : '';

    try {
      const res = await fetch(`/api/wind?bbox=${bbox}&zoom=${zoom}&time=${encodeURIComponent(time)}${modelParam}&level=${windLevel}`);
      if (!res.ok) await throwHttpError(res, 'API');
      const data = await res.json();
      markApiSuccess();

      data.barbs.forEach(b => {
        const icon = createWindBarbIcon(b.speed_kt, b.dir_deg);
        const marker = L.marker([b.lat, b.lon], {
          icon,
          bubblingMouseEvents: true,
          interactive: false  // Don't interfere with symbol clicks
        }).addTo(windLayer);
      });

      if (symbolLayer && typeof symbolLayer.eachLayer === 'function') {
        symbolLayer.eachLayer(layer => {
          if (layer && typeof layer.bringToFront === 'function') layer.bringToFront();
        });
      }
    } catch (e) {
      console.error('Wind barb error:', e);
      markApiFailure('wind', e);
    }
  }, 350);
}

// Update legend display
function updateLegend() {
  const legendEl = document.getElementById('legend');
  const effectiveOverlay = getEffectiveOverlayLayer();

  if (effectiveOverlay === 'none' || !LEGEND_CONFIGS[effectiveOverlay]) {
    legendEl.style.display = 'none';
    return;
  }

  const config = LEGEND_CONFIGS[effectiveOverlay];
  legendEl.innerHTML = `
    <div class="legend-title">${config.title}</div>
    <div class="legend-gradient" style="background: ${config.gradient};"></div>
    <div class="legend-labels">
      <span>${config.labels[0]}</span>
      <span>${config.labels[config.labels.length - 1]}</span>
    </div>
  `;
  legendEl.style.display = 'block';
}

// Layer toggle for mobile
document.getElementById('layers-toggle').addEventListener('click', (e) => {
  const content = document.getElementById('layers-content');
  const toggle = e.target;
  content.classList.toggle('expanded');
  toggle.classList.toggle('rotated');
});

// Layer controls
document.getElementById('layer-convection').addEventListener('change', (e) => {
  if (e.target.checked) {
    symbolLayer.addTo(map);
    loadSymbols();
  } else {
    map.removeLayer(symbolLayer);
  }
});

document.getElementById('layer-wind').addEventListener('change', (e) => {
  windEnabled = e.target.checked;
  if (windEnabled) {
    windLayer.addTo(map);
    loadWind();
  } else {
    map.removeLayer(windLayer);
    // Remove stale tooltip content that may include wind info
    map.closePopup();
  }
});

document.getElementById('wind-level').addEventListener('change', (e) => {
  windLevel = e.target.value;
  if (windEnabled) loadWind();
});

document.querySelectorAll('input[name="overlay"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentOverlay = e.target.value;
    updateLegend();
    loadOverlay();
  });
});

const precipType = document.getElementById('precip-type');
if (precipType) {
  precipType.addEventListener('change', () => {
    if (currentOverlay === 'precip') {
      updateLegend();
      loadOverlay();
    }
  });
}

const cloudsType = document.getElementById('clouds-type');
if (cloudsType) {
  cloudsType.addEventListener('change', () => {
    if (currentOverlay === 'clouds') {
      updateLegend();
      loadOverlay();
    }
  });
}

// Event listeners
// Overlay uses fixed extent — only refresh symbols on pan/zoom
map.on('moveend zoomend', () => { loadSymbols(); loadWind(); });

// Don't clear symbols on zoomstart - causes issues on mobile pinch-zoom
// The debounced loadSymbols() will clear and reload on zoomend

// Enable click tooltip when convection is hidden but wind is active
map.on('click', (e) => {
  const convectionOn = document.getElementById('layer-convection')?.checked;
  if (convectionOn) return;
  if (!windEnabled) return;

  const curStep = timesteps[currentTimeIndex];
  const curTime = curStep?.validTime || 'latest';
  const curModel = curStep?.model || null;
  loadPoint(e.latlng.lat, e.latlng.lng, curTime, curModel, windLevel, map.getZoom());
});

// Timeline controls
const timeline = document.getElementById('timeline');
const dateStrip = document.getElementById('date-strip');
const prev1hBtn = document.getElementById('prev-1h');
const next1hBtn = document.getElementById('next-1h');
let currentStep = 1;  // Step size in hours
let dayMeta = [];  // [{dateStr, firstIdx, noonIdx}]

// Step size buttons
document.querySelectorAll('.step-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.step-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentStep = parseInt(btn.dataset.step);
  });
});

function buildTimeline() {
  timeline.innerHTML = '';
  dayMeta = [];
  let lastDate = null;
  let lastModel = null;
  
  timesteps.forEach((step, idx) => {
    const d = new Date(step.validTime);
    const day = String(d.getUTCDate()).padStart(2, '0');
    const month = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dateStr = `${day}.${month}.`;
    const hour = d.getUTCHours();
    const hourStr = String(hour).padStart(2, '0');
    
    // Track day boundaries
    if (dateStr !== lastDate) {
      const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const dayLabel = `${dayNames[d.getUTCDay()]} ${dateStr}`;
      dayMeta.push({ dateStr: dayLabel, firstIdx: idx, noonIdx: null });
      lastDate = dateStr;
    }
    // Track noon (or closest to 12)
    const curDay = dayMeta[dayMeta.length - 1];
    if (hour === 12) curDay.noonIdx = idx;
    
    const hourEl = document.createElement('div');
    hourEl.className = 'timeline-hour';
    hourEl.textContent = hourStr;
    hourEl.dataset.index = idx;
    
    // Mark day boundaries
    if (idx === curDay.firstIdx && idx > 0) {
      hourEl.classList.add('day-start');
    }
    
    // Mark model switch (D2 → EU handover)
    if (step.model && lastModel && step.model !== lastModel) {
      hourEl.classList.add('model-switch');
      hourEl.title = `Model switch: ${lastModel.toUpperCase().replace('_', '-')} → ${step.model.toUpperCase().replace('_', '-')}`;
    }
    lastModel = step.model;
    
    if (idx === currentTimeIndex) {
      hourEl.classList.add('active');
    }
    
    hourEl.addEventListener('click', () => {
      currentTimeIndex = idx;
      updateTimeline();
      loadSymbols();
      loadOverlay();
      loadWind();
    });
    
    timeline.appendChild(hourEl);
  });
  
  // For days without noon, pick midpoint
  dayMeta.forEach((dm, i) => {
    if (dm.noonIdx === null) {
      const nextStart = i < dayMeta.length - 1 ? dayMeta[i + 1].firstIdx : timesteps.length;
      dm.noonIdx = Math.floor((dm.firstIdx + nextStart) / 2);
    }
  });
  
  updateValidTime();
  updateDateStrip();
}

function updateDateStrip() {
  dateStrip.innerHTML = '';
  const hours = timeline.querySelectorAll('.timeline-hour');
  if (hours.length === 0) return;
  
  const wrapperRect = timeline.parentElement.getBoundingClientRect();
  const wrapperLeft = wrapperRect.left;
  const wrapperRight = wrapperRect.right;
  const wrapperWidth = wrapperRect.width;
  const scrollLeft = timeline.scrollLeft;
  
  dayMeta.forEach((dm, i) => {
    const label = document.createElement('div');
    label.className = 'date-label';
    label.textContent = dm.dateStr;
    dateStrip.appendChild(label);
    
    // Position at noon hour element
    const noonEl = hours[dm.noonIdx];
    const firstEl = hours[dm.firstIdx];
    const lastIdx = i < dayMeta.length - 1 ? dayMeta[i + 1].firstIdx - 1 : hours.length - 1;
    const lastEl = hours[lastIdx];
    
    if (!noonEl || !firstEl || !lastEl) return;
    
    const noonRect = noonEl.getBoundingClientRect();
    const firstRect = firstEl.getBoundingClientRect();
    const lastRect = lastEl.getBoundingClientRect();
    const labelWidth = label.offsetWidth;
    
    // Ideal position: centered on noon
    let idealLeft = noonRect.left + noonRect.width / 2 - wrapperLeft - labelWidth / 2;
    
    // Clamp: don't go past first hour of this day
    const minLeft = firstRect.left - wrapperLeft;
    // Clamp: don't go past last hour of this day
    const maxLeft = lastRect.right - wrapperLeft - labelWidth;
    
    // Also clamp to viewport edges of the wrapper
    let finalLeft = Math.max(0, Math.min(wrapperWidth - labelWidth, Math.max(minLeft, Math.min(maxLeft, idealLeft))));
    
    // Hide if day is completely scrolled out
    if (lastRect.right < wrapperLeft || firstRect.left > wrapperRight) {
      label.style.display = 'none';
    } else {
      label.style.left = finalLeft + 'px';
    }
  });
}

// Update date strip on scroll
timeline.addEventListener('scroll', updateDateStrip);

function updateTimeline() {
  const hours = timeline.querySelectorAll('.timeline-hour');
  hours.forEach((el) => {
    el.classList.toggle('active', parseInt(el.dataset.index) === currentTimeIndex);
  });
  
  const activeEl = hours[currentTimeIndex];
  if (activeEl) {
    activeEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
  }
  
  updateValidTime();
  // Delay date strip update to let scroll settle
  setTimeout(updateDateStrip, 100);
}

function stepForward() {
  // Find next timestep that's >= currentStep hours ahead
  if (currentStep === 1) {
    if (currentTimeIndex < timesteps.length - 1) currentTimeIndex++;
  } else {
    const curTime = new Date(timesteps[currentTimeIndex].validTime).getTime();
    const targetTime = curTime + currentStep * 3600000;
    let bestIdx = currentTimeIndex;
    for (let i = currentTimeIndex + 1; i < timesteps.length; i++) {
      const t = new Date(timesteps[i].validTime).getTime();
      if (t >= targetTime) { bestIdx = i; break; }
      if (i === timesteps.length - 1) bestIdx = i;
    }
    if (bestIdx > currentTimeIndex) currentTimeIndex = bestIdx;
  }
  updateTimeline();
  loadSymbols();
  loadOverlay();
  loadWind();
}

function stepBackward() {
  if (currentStep === 1) {
    if (currentTimeIndex > 0) currentTimeIndex--;
  } else {
    const curTime = new Date(timesteps[currentTimeIndex].validTime).getTime();
    const targetTime = curTime - currentStep * 3600000;
    let bestIdx = currentTimeIndex;
    for (let i = currentTimeIndex - 1; i >= 0; i--) {
      const t = new Date(timesteps[i].validTime).getTime();
      if (t <= targetTime) { bestIdx = i; break; }
      if (i === 0) bestIdx = i;
    }
    if (bestIdx < currentTimeIndex) currentTimeIndex = bestIdx;
  }
  updateTimeline();
  loadSymbols();
  loadOverlay();
  loadWind();
}

prev1hBtn.addEventListener('click', stepBackward);
next1hBtn.addEventListener('click', stepForward);

// ─── Feedback system ───
const feedbackBtn = document.getElementById('feedback-btn');
const feedbackOverlay = document.getElementById('feedback-overlay');
const feedbackClose = document.getElementById('feedback-close');
const feedbackSubmit = document.getElementById('feedback-submit');
const feedbackText = document.getElementById('feedback-text');
const feedbackType = document.getElementById('feedback-type');
const feedbackStatus = document.getElementById('feedback-status');
const feedbackContext = document.getElementById('feedback-context');

function openFeedback() {
  feedbackOverlay.style.display = 'flex';
  feedbackText.value = '';
  feedbackStatus.textContent = '';
  feedbackSubmit.disabled = false;
  // Show current context
  const step = timesteps[currentTimeIndex];
  const ctx = [];
  if (step) ctx.push(`Time: ${step.validTime}`);
  if (step?.model) ctx.push(`Model: ${step.model.toUpperCase().replace('_','-')}`);
  ctx.push(`Zoom: ${map.getZoom()}`);
  if (currentOverlay !== 'none') ctx.push(`Overlay: ${getEffectiveOverlayLayer()}`);
  if (windEnabled) ctx.push(`Wind: ${windLevel}`);
  feedbackContext.textContent = ctx.length ? 'Context: ' + ctx.join(' · ') : '';
  setTimeout(() => feedbackText.focus(), 100);
}

function closeFeedback() {
  feedbackOverlay.style.display = 'none';
}

async function submitFeedback() {
  const text = feedbackText.value.trim();
  if (!text) {
    feedbackStatus.textContent = 'Please enter a message.';
    feedbackStatus.style.color = '#ff6b6b';
    return;
  }

  feedbackSubmit.disabled = true;
  feedbackStatus.textContent = 'Sending...';
  feedbackStatus.style.color = 'rgba(255,255,255,0.6)';

  const step = timesteps[currentTimeIndex];
  const payload = {
    type: feedbackType.value,
    message: text,
    context: {
      validTime: step?.validTime || null,
      model: step?.model || null,
      run: currentRun,
      zoom: map.getZoom(),
      center: [map.getCenter().lat, map.getCenter().lng],
      overlay: getEffectiveOverlayLayer(),
      windEnabled: windEnabled,
      windLevel: windEnabled ? windLevel : null,
    },
    userAgent: navigator.userAgent,
    screen: `${window.innerWidth}x${window.innerHeight}`,
  };

  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) await throwHttpError(res, 'API');
    markApiSuccess();
    feedbackStatus.textContent = '✓ Thank you for your feedback!';
    feedbackStatus.style.color = '#66cc66';
    feedbackText.value = '';
    setTimeout(closeFeedback, 1500);
  } catch (e) {
    console.error('Feedback error:', e);
    markApiFailure('feedback', e);
    feedbackStatus.textContent = 'Failed to send. Please try again.';
    feedbackStatus.style.color = '#ff6b6b';
    feedbackSubmit.disabled = false;
  }
}

feedbackBtn.addEventListener('click', openFeedback);
feedbackClose.addEventListener('click', closeFeedback);
feedbackOverlay.addEventListener('click', (e) => {
  if (e.target === feedbackOverlay) closeFeedback();
});
feedbackSubmit.addEventListener('click', submitFeedback);
// Allow Ctrl+Enter to submit
feedbackText.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') submitFeedback();
});

// Initial load
loadTimesteps();

// Auto-refresh: check for new runs every 60s
setInterval(async () => {
  try {
    const res = await fetch('/api/timesteps');
    if (!res.ok) return;
    const data = await res.json();
    
    // Use merged timeline
    const merged = data.merged;
    const source = merged && merged.steps.length ? merged : (data.runs && data.runs[0]);
    if (!source) return;
    
    const newRun = source.run || source.run;
    const newSteps = source.steps;
    const newRunTime = source.runTime;

    if (currentRun && newRun !== currentRun) {
      // New run available — remember relative position
      const wasAtStart = currentTimeIndex === 0;
      const wasAtEnd = currentTimeIndex === timesteps.length - 1;
      const oldValidTime = timesteps[currentTimeIndex]?.validTime;

      // Reload timesteps
      timesteps = newSteps;
      currentRun = newRun;
      document.getElementById('run-time').textContent = newRunTime ? formatDateShort(new Date(newRunTime)) : 'N/A';
      document.getElementById('model').textContent = merged ? 'ICON-D2 + EU' : (source.model ? source.model.toUpperCase().replace('_', '-') : 'ICON-D2');

      // Try to keep same valid time, otherwise go to start
      if (wasAtStart) {
        currentTimeIndex = 0;
      } else if (wasAtEnd) {
        currentTimeIndex = timesteps.length - 1;
      } else {
        // Find closest valid time
        let bestIdx = 0, bestDist = Infinity;
        timesteps.forEach((s, i) => {
          const dist = Math.abs(new Date(s.validTime) - new Date(oldValidTime));
          if (dist < bestDist) { bestDist = dist; bestIdx = i; }
        });
        currentTimeIndex = bestIdx;
      }

      buildTimeline();
      loadSymbols();
      loadWind();

      // Flash the run label to signal update
      const runEl = document.getElementById('run-time');
      runEl.style.color = '#00AA00';
      setTimeout(() => { runEl.style.color = ''; }, 3000);
    } else if (newRun === currentRun && newSteps.length > timesteps.length) {
      // Same run but more steps available
      const oldLen = timesteps.length;
      timesteps = newSteps;
      // If user was at the end, stay at end
      if (currentTimeIndex === oldLen - 1) {
        currentTimeIndex = timesteps.length - 1;
      }
      buildTimeline();
    }
  } catch (e) {
    // Silently ignore polling errors
  }
}, 60000);
