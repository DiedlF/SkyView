// app.js - Main application logic

let map, symbolLayer, windLayer, overlayLayer = null, debounceTimer, overlayDebounce, windDebounce, timesteps = [], currentTimeIndex = 0, currentRun = '';
let overlayRequestSeq = 0;
let overlayAbortCtrl = null;
let overlayObjectUrl = null;
let currentOverlay = 'none';
let windEnabled = false;
let windLevel = '10m';
let modelCapabilities = {};  // Store model capabilities from API

// Legend definitions for each overlay
const LEGEND_CONFIGS = {
  total_precip: {
    title: 'Total Precipitation',
    gradient: 'linear-gradient(to right, rgb(150,255,255), rgb(100,200,255), rgb(50,150,255), rgb(0,100,255))',
    labels: ['0.1 mm/h', '5+ mm/h']
  },
  rain: {
    title: 'Rain',
    gradient: 'linear-gradient(to right, rgb(180,220,255), rgb(100,160,230), rgb(20,60,180))',
    labels: ['0.1 mm/h', '5+ mm/h']
  },
  snow: {
    title: 'Snow',
    gradient: 'linear-gradient(to right, rgb(255,200,255), rgb(210,120,230), rgb(120,40,160))',
    labels: ['0.1 mm/h', '5+ mm/h']
  },
  hail: {
    title: 'Hail/Graupel',
    gradient: 'linear-gradient(to right, rgb(200,160,20), rgb(240,100,30), rgb(255,80,20))',
    labels: ['0.1 mm/h', '5+ mm/h']
  },
  sigwx: {
    title: 'Significant Weather',
    gradient: 'linear-gradient(to right, rgb(180,150,0), rgb(100,200,100), rgb(0,170,0), rgb(160,120,255), rgb(220,0,50))',
    labels: ['Fog', 'Drizzle', 'Rain', 'Snow', 'T-Storm']
  },
  clouds: {
    title: 'Cloud Cover',
    gradient: 'linear-gradient(to right, rgb(140,140,140), rgb(40,40,40))',
    labels: ['5%', '100%']
  },
  thermals: {
    title: 'Thermal Activity (CAPE)',
    gradient: 'linear-gradient(to right, rgb(50,180,50), rgb(150,150,50), rgb(220,100,30), rgb(255,50,50))',
    labels: ['10 J/kg', '2000+ J/kg']
  },
  ceiling: {
    title: 'Ceiling Height',
    gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))',
    labels: ['0m', '9900m']
  },
  cloud_base: {
    title: 'Cloud Base (SC)',
    gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))',
    labels: ['0m', '5000m']
  },
  wstar: {
    title: 'Thermal Strength (W*)',
    gradient: 'linear-gradient(to right, rgb(50,200,50), rgb(180,200,50), rgb(220,150,30), rgb(255,50,50))',
    labels: ['0.2 m/s', '5+ m/s']
  },
  climb_rate: {
    title: 'Climb Rate',
    gradient: 'linear-gradient(to right, rgb(50,200,50), rgb(180,200,50), rgb(220,150,30), rgb(255,50,50))',
    labels: ['0 m/s', '5 m/s']
  },
  lcl: {
    title: 'Cloud Base (LCL) MSL',
    gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))',
    labels: ['0m', '5000m MSL']
  },
  reachable: {
    title: 'Reachable Distance (L/D 40)',
    gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,180,50), rgb(80,240,80))',
    labels: ['0 km', '200 km']
  }
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
    
    // Index by model name
    data.models.forEach(m => {
      modelCapabilities[m.name] = m;
    });
  } catch (e) {
    console.error('Error loading model capabilities:', e);
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
  } catch (e) {
    console.error('Error loading timesteps:', e);
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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      symbolLayer.clearLayers();
      data.symbols.forEach(sym => {
        // Clear sky remains an internal symbol state but is rendered as blank/non-clickable.
        if (!sym.clickable || sym.type === 'clear') return;

        const icon = createSymbolIcon(sym.type, sym.cloudBase);
        const marker = L.marker([sym.lat, sym.lon], {
          icon,
          bubblingMouseEvents: true  // Allow touch events to bubble for pinch-zoom
        }).addTo(symbolLayer);
        marker.on('click', () => {
          const curStep = timesteps[currentTimeIndex];
          const curTime = curStep?.validTime || time;
          const curModel = curStep?.model || null;
          loadPoint(sym.clickLat ?? sym.lat, sym.clickLon ?? sym.lon, curTime, curModel);
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
      // Optionally show error in UI
      symbolLayer.clearLayers();
    }
  }, 300);
}

// Overlay value formatting
const OVERLAY_FORMAT = {
  total_precip: (v) => v != null ? `Total precip: ${v} mm/h` : null,
  rain:         (v) => v != null ? `Rain: ${v} mm/h` : null,
  snow:         (v) => v != null ? `Snow: ${v} mm/h` : null,
  hail:         (v) => v != null ? `Hail/Graupel: ${v} mm/h` : null,
  sigwx:        (v) => v != null ? `Sig. Weather: ww ${v}` : null,
  clouds:       (v) => v != null ? `Cloud cover: ${v}%` : null,
  thermals:     (v) => v != null ? `CAPE: ${v} J/kg` : null,
  ceiling:      (v) => v != null ? `Ceiling: ${Math.round(v)} m` : null,
  cloud_base:   (v) => v != null ? `Cloud base (SC): ${Math.round(v)} m` : null,
  wind:         (v) => v != null ? `Wind: ${v} kt` : null,
  wstar:        (v) => v != null ? `W*: ${v} m/s` : null,
  climb_rate:   (v) => v != null ? `Climb: ${v} m/s` : null,
  lcl:          (v) => v != null ? `Cloud base (LCL): ${Math.round(v)} m MSL` : null,
  reachable:    (v) => v != null ? `Reachable: ${v} km` : null,
};

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
async function loadPoint(lat, lon, time, model) {
  try {
    if (!time) time = 'latest';
    const modelParam = model ? `&model=${model}` : '';
    const res = await fetch(`/api/point?lat=${lat}&lon=${lon}&time=${encodeURIComponent(time)}${modelParam}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const symbolName = SYMBOL_NAMES[data.symbol] || data.symbol || 'N/A';
    let lines = [`<b>${symbolName}</b>`];

    // Show active overlay value if an overlay is selected
    if (currentOverlay !== 'none' && data.overlay_values) {
      const fmt = OVERLAY_FORMAT[currentOverlay];
      if (fmt) {
        const formatted = fmt(data.overlay_values[currentOverlay]);
        if (formatted) lines.push(formatted);
      }
    }

    L.popup({ maxWidth: 200 })
      .setLatLng([lat, lon])
      .setContent(lines.join('<br/>'))
      .openOn(map);
  } catch (e) {
    console.error('Error loading point:', e);
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

    const time = timesteps[currentTimeIndex]?.validTime || '';

    try {
      const overlayStep = timesteps[currentTimeIndex];
      const overlayModelParam = overlayStep && overlayStep.model ? `&model=${overlayStep.model}` : '';
      const { bbox: oBbox, width: oWidth } = getOverlayParams();
      const url = `/api/overlay?layer=${currentOverlay}&bbox=${oBbox}&time=${encodeURIComponent(time)}&width=${oWidth}${overlayModelParam}`;

      const overlayRes = await fetch(url, { signal: overlayAbortCtrl.signal });
      if (!overlayRes.ok) throw new Error(`Overlay HTTP ${overlayRes.status}`);
      const clampedBbox = overlayRes.headers.get('X-Bbox');
      const blob = await overlayRes.blob();

      // Ignore stale responses that arrive out of order
      if (reqId !== overlayRequestSeq) return;

      overlayObjectUrl = URL.createObjectURL(blob);

      let bounds;
      if (clampedBbox) {
        const [cLatMin, cLonMin, cLatMax, cLonMax] = clampedBbox.split(',').map(Number);
        bounds = [[cLatMin, cLonMin], [cLatMax, cLonMax]];
      } else {
        const fb = map.getBounds();
        bounds = [[fb.getSouth(), fb.getWest()], [fb.getNorth(), fb.getEast()]];
      }

      overlayLayer = L.imageOverlay(overlayObjectUrl, bounds, { opacity: 0.6, interactive: false }).addTo(map);
      if (symbolLayer) symbolLayer.bringToFront();
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Overlay error:', e);
      }
    }
  }, 400);
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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      data.barbs.forEach(b => {
        const icon = createWindBarbIcon(b.speed_kt, b.dir_deg);
        const marker = L.marker([b.lat, b.lon], {
          icon,
          bubblingMouseEvents: true,
          interactive: false  // Don't interfere with symbol clicks
        }).addTo(windLayer);
      });

      if (symbolLayer) symbolLayer.bringToFront();
    } catch (e) {
      console.error('Wind barb error:', e);
    }
  }, 350);
}

// Update legend display
function updateLegend() {
  const legendEl = document.getElementById('legend');
  
  if (currentOverlay === 'none' || !LEGEND_CONFIGS[currentOverlay]) {
    legendEl.style.display = 'none';
    return;
  }
  
  const config = LEGEND_CONFIGS[currentOverlay];
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

// Event listeners
// Overlay uses fixed extent — only refresh symbols on pan/zoom
map.on('moveend zoomend', () => { loadSymbols(); loadWind(); loadOverlay(); });

// Don't clear symbols on zoomstart - causes issues on mobile pinch-zoom
// The debounced loadSymbols() will clear and reload on zoomend

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
  if (currentOverlay !== 'none') ctx.push(`Overlay: ${currentOverlay}`);
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
      overlay: currentOverlay,
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    feedbackStatus.textContent = '✓ Thank you for your feedback!';
    feedbackStatus.style.color = '#66cc66';
    feedbackText.value = '';
    setTimeout(closeFeedback, 1500);
  } catch (e) {
    console.error('Feedback error:', e);
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
