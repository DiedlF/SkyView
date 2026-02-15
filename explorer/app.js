// ICON-Explorer App
// ICON Weather Data Explorer

const STATE = {
    map: null,
    overlay: null,
    overlayObjectUrl: null,
    overlayAbortCtrl: null,
    overlayRequestSeq: 0,
    overlayUpdateDebounce: null,
    variables: [],
    timesteps: [],
    selectedVariable: null,
    selectedTime: null,
    selectedPalette: 'viridis',
    autoRange: true,
    rangeMin: null,
    rangeMax: null,
    opacity: 0.6,
    stepSize: 1
};

// Initialize app on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initControls();
    loadVariables();
    loadTimesteps();
});

// ===== MAP INITIALIZATION =====

function initMap() {
    // Create map
    STATE.map = L.map('map', {
        center: [50.5, 10.0],
        zoom: 6,
        minZoom: 5,
        maxZoom: 12,
        maxBounds: [[42, -5], [59, 21]],
        zoomControl: false,
        attributionControl: false
    });

    // Match Skyview: zoom controls bottom-left
    L.control.zoom({ position: 'bottomleft' }).addTo(STATE.map);

    // ESRI World Ocean Base
    const oceanBase = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        {
            attribution: 'ESRI World Ocean Base',
            maxNativeZoom: 10,
            maxZoom: 12
        }
    );

    // ESRI World Hillshade with blend mode
    const hillshade = L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
        {
            attribution: 'ESRI',
            maxNativeZoom: 10,
            maxZoom: 12,
            className: 'hillshade-layer'
        }
    );

    oceanBase.addTo(STATE.map);
    hillshade.addTo(STATE.map);

    // Map click handler
    STATE.map.on('click', handleMapClick);

    // Refresh overlay when viewport changes (fixes zoom/pan stale overlays)
    const onViewportChange = () => {
        if (STATE.selectedVariable && STATE.selectedTime) {
            clearTimeout(STATE.overlayUpdateDebounce);
            STATE.overlayUpdateDebounce = setTimeout(() => updateOverlay(), 250);
        }
    };
    STATE.map.on('moveend zoomend', onViewportChange);
}

// ===== CONTROLS INITIALIZATION =====

function initControls() {
    // Variable selector
    const variableSelect = document.getElementById('variable-select');
    variableSelect.addEventListener('change', (e) => {
        const varData = STATE.variables.find(v => v.name === e.target.value);
        STATE.selectedVariable = varData;
        if (STATE.map) STATE.map.closePopup();
        updateVariableInfo();
        updateOverlay();
    });

    // Palette selector
    const paletteSelect = document.getElementById('palette-select');
    paletteSelect.addEventListener('change', (e) => {
        STATE.selectedPalette = e.target.value;
        updateLegendGradient();
        updateOverlay();
    });

    // Auto range checkbox
    const autoRangeCheckbox = document.getElementById('auto-range');
    autoRangeCheckbox.addEventListener('change', (e) => {
        STATE.autoRange = e.target.checked;
        document.getElementById('range-min').disabled = e.target.checked;
        document.getElementById('range-max').disabled = e.target.checked;
        if (!e.target.checked) {
            STATE.rangeMin = parseFloat(document.getElementById('range-min').value);
            STATE.rangeMax = parseFloat(document.getElementById('range-max').value);
        }
        updateOverlay();
    });

    // Range inputs
    document.getElementById('range-min').addEventListener('change', (e) => {
        STATE.rangeMin = parseFloat(e.target.value);
        updateOverlay();
    });

    document.getElementById('range-max').addEventListener('change', (e) => {
        STATE.rangeMax = parseFloat(e.target.value);
        updateOverlay();
    });

    // Opacity slider
    const opacitySlider = document.getElementById('opacity-slider');
    opacitySlider.addEventListener('input', (e) => {
        STATE.opacity = e.target.value / 100;
        document.getElementById('opacity-value').textContent = e.target.value + '%';
        if (STATE.overlay) {
            STATE.overlay.setOpacity(STATE.opacity);
        }
    });

    // Time navigation
    document.getElementById('step-back').addEventListener('click', () => stepTime(-1));
    document.getElementById('step-forward').addEventListener('click', () => stepTime(1));

    // Step size buttons
    document.querySelectorAll('.step-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.step-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            STATE.stepSize = parseInt(e.target.dataset.step);
        });
    });

    // Sidebar toggle (mobile)
    document.getElementById('sidebar-toggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });
}

// ===== DATA LOADING =====

async function loadVariables() {
    try {
        const response = await fetch('/api/variables');
        const data = await response.json();
        STATE.variables = data.variables || data;
        populateVariableSelect();
    } catch (error) {
        console.error('Failed to load variables:', error);
        showError('Failed to load variables');
    }
}

async function loadTimesteps() {
    try {
        const response = await fetch('/api/timesteps');
        const data = await response.json();
        // Build timesteps from merged timeline
        const merged = data.merged;
        if (merged && merged.steps) {
            STATE.timesteps = merged.steps.map(s => s.validTime);
            STATE._mergedSteps = merged.steps;  // keep model/run info per step
        } else {
            STATE.timesteps = data.timesteps || [];
        }
        populateTimeButtons();
        updateModelInfo(data);
    } catch (error) {
        console.error('Failed to load timesteps:', error);
        showError('Failed to load timesteps');
    }
}

// ===== UI POPULATION =====

function populateVariableSelect() {
    const select = document.getElementById('variable-select');
    select.innerHTML = '';

    // Group variables by category
    const grouped = {};
    STATE.variables.forEach(v => {
        if (!grouped[v.group]) grouped[v.group] = [];
        grouped[v.group].push(v);
    });

    // Create optgroups
    Object.keys(grouped).sort().forEach(group => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = group;
        grouped[group].forEach(v => {
            const option = document.createElement('option');
            option.value = v.name;
            option.textContent = `${v.name} (${v.unit})`;
            optgroup.appendChild(option);
        });
        select.appendChild(optgroup);
    });

    // Select first variable
    if (STATE.variables.length > 0) {
        STATE.selectedVariable = STATE.variables[0];
        select.value = STATE.selectedVariable.name;
        updateVariableInfo();
    }
}

function populateTimeButtons() {
    const container = document.getElementById('time-buttons');
    container.innerHTML = '';

    let lastDate = null;

    STATE.timesteps.forEach((ts, index) => {
        const date = new Date(ts);
        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

        const slot = document.createElement('div');
        slot.className = 'time-slot';

        // Add date label if date changed
        if (dateStr !== lastDate) {
            const dateLabel = document.createElement('div');
            dateLabel.className = 'date-label';
            dateLabel.textContent = dateStr;
            slot.appendChild(dateLabel);
            lastDate = dateStr;
        }

        const btn = document.createElement('button');
        btn.className = 'hour-btn';
        btn.textContent = timeStr;
        btn.dataset.time = ts;
        btn.addEventListener('click', () => selectTime(ts));

        slot.appendChild(btn);
        container.appendChild(slot);
    });

    // Select first timestep
    if (STATE.timesteps.length > 0) {
        selectTime(STATE.timesteps[0]);
    }
}

// ===== TIME SELECTION =====

function selectTime(time) {
    STATE.selectedTime = time;
    if (STATE.map) STATE.map.closePopup();

    // Update active button
    document.querySelectorAll('.hour-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.time === time);
    });

    // Scroll to active button
    const activeBtn = document.querySelector('.hour-btn.active');
    if (activeBtn) {
        activeBtn.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }

    updateModelInfo();
    updateOverlay();
}

function stepTime(direction) {
    const currentIndex = STATE.timesteps.indexOf(STATE.selectedTime);
    if (currentIndex === -1) return;

    const newIndex = currentIndex + (direction * STATE.stepSize);
    if (newIndex >= 0 && newIndex < STATE.timesteps.length) {
        selectTime(STATE.timesteps[newIndex]);
    }
}

// ===== INFO UPDATES =====

function updateVariableInfo() {
    const infoDiv = document.getElementById('variable-info');
    if (STATE.selectedVariable) {
        infoDiv.textContent = STATE.selectedVariable.desc || '';
        
        // Update range inputs with variable defaults
        if (STATE.autoRange) {
            document.getElementById('range-min').value = STATE.selectedVariable.min || '';
            document.getElementById('range-max').value = STATE.selectedVariable.max || '';
        }
    }
}

function updateModelInfo(data) {
    const infoDiv = document.getElementById('model-info');
    if (!STATE.selectedTime) {
        infoDiv.textContent = 'Loading...';
        return;
    }

    const validTime = new Date(STATE.selectedTime);
    const validStr = validTime.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
    });

    let modelText = '';
    if (data && data.merged && data.merged.runTime) {
        const runTime = new Date(data.merged.runTime);
        const runStr = runTime.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        modelText = `Model: ICON-D2/EU<br>Run: ${runStr}<br>Valid: ${validStr}`;
    } else {
        modelText = `Valid: ${validStr}`;
    }

    infoDiv.innerHTML = modelText;
}

// ===== OVERLAY UPDATE =====

async function updateOverlay() {
    if (!STATE.selectedVariable || !STATE.selectedTime || !STATE.map) return;

    const reqId = ++STATE.overlayRequestSeq;

    // Cancel previous in-flight request
    if (STATE.overlayAbortCtrl) {
        STATE.overlayAbortCtrl.abort();
    }
    STATE.overlayAbortCtrl = new AbortController();

    try {
        // Get map bounds
        const bounds = STATE.map.getBounds();
        const bbox = [
            bounds.getSouth(),
            bounds.getWest(),
            bounds.getNorth(),
            bounds.getEast()
        ].join(',');

        // Get map width for resolution
        const width = STATE.map.getSize().x;

        // Build URL
        const params = new URLSearchParams({
            var: STATE.selectedVariable.name,
            bbox: bbox,
            time: STATE.selectedTime,
            width: width,
            palette: STATE.selectedPalette
        });

        // Add range if not auto
        if (!STATE.autoRange && STATE.rangeMin !== null && STATE.rangeMax !== null) {
            params.append('vmin', STATE.rangeMin);
            params.append('vmax', STATE.rangeMax);
        }

        const url = `/api/overlay?${params}`;

        const response = await fetch(url, { signal: STATE.overlayAbortCtrl.signal });
        if (!response.ok) return;

        const bboxHeader = response.headers.get('x-bbox');
        const vminHeader = response.headers.get('x-vmin');
        const vmaxHeader = response.headers.get('x-vmax');

        const blob = await response.blob();

        // Ignore out-of-order stale responses
        if (reqId !== STATE.overlayRequestSeq) return;

        const imageUrl = URL.createObjectURL(blob);

        // Remove old overlay only when new one is ready
        if (STATE.overlay) {
            STATE.map.removeLayer(STATE.overlay);
            STATE.overlay = null;
        }
        if (STATE.overlayObjectUrl) {
            URL.revokeObjectURL(STATE.overlayObjectUrl);
            STATE.overlayObjectUrl = null;
        }

        if (bboxHeader) {
            const [latMin, lonMin, latMax, lonMax] = bboxHeader.split(',').map(Number);
            const overlayBounds = [[latMin, lonMin], [latMax, lonMax]];

            STATE.overlayObjectUrl = imageUrl;
            STATE.overlay = L.imageOverlay(imageUrl, overlayBounds, {
                opacity: STATE.opacity,
                interactive: false
            }).addTo(STATE.map);

            if (vminHeader && vmaxHeader) {
                updateLegend(parseFloat(vminHeader), parseFloat(vmaxHeader));
            }
        } else {
            URL.revokeObjectURL(imageUrl);
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Failed to update overlay:', error);
        }
    }
}

// ===== LEGEND =====

function updateLegend(min, max) {
    document.getElementById('legend-min').textContent = min.toFixed(2);
    document.getElementById('legend-max').textContent = max.toFixed(2);
}

function updateLegendGradient() {
    const gradients = {
        viridis: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde724)',
        plasma: 'linear-gradient(to right, #0d0887, #7e03a8, #cc4778, #f89540, #f0f921)',
        inferno: 'linear-gradient(to right, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)',
        coolwarm: 'linear-gradient(to right, #3b4cc0, #7396f5, #b0b0b0, #f07e6e, #b40426)',
        RdBu_r: 'linear-gradient(to right, #053061, #2166ac, #d1e5f0, #fddbc7, #b2182b, #67001f)',
        Blues: 'linear-gradient(to right, #f7fbff, #6baed6, #08519c)',
        Greens: 'linear-gradient(to right, #f7fcf5, #74c476, #00441b)',
        Reds: 'linear-gradient(to right, #fff5f0, #fb6a4a, #67000d)',
        YlOrRd: 'linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #e31a1c, #800026)'
    };

    const gradient = gradients[STATE.selectedPalette] || gradients.viridis;
    document.getElementById('legend-gradient').style.background = gradient;
}

// ===== MAP CLICK HANDLER =====

async function handleMapClick(e) {
    if (!STATE.selectedTime) return;

    try {
        const params = new URLSearchParams({
            lat: e.latlng.lat,
            lon: e.latlng.lng,
            time: STATE.selectedTime
        });

        const response = await fetch(`/api/point?${params}`);
        const data = await response.json();

        if (data.values) {
            showPointPopup(e.latlng, data.values);
        }
    } catch (error) {
        console.error('Failed to fetch point data:', error);
    }
}

function showPointPopup(latlng, values) {
    const selected = STATE.selectedVariable;
    if (!selected) return;

    // Keep popup concise and synced to current selected variable.
    const value = values[selected.name];
    const valueText = (value === null || value === undefined)
        ? 'n/a'
        : `${Number(value).toFixed(2)} ${selected.unit}`;

    const html = `
        <div class="popup-group">
            <div class="popup-group-title">${selected.group}</div>
            <div class="popup-var">
                <span class="popup-var-name">${selected.name}</span>
                <span class="popup-var-value">${valueText}</span>
            </div>
            <div class="popup-meta">${selected.desc || ''}</div>
        </div>
    `;

    L.popup({
        maxWidth: 300,
        maxHeight: 220
    })
        .setLatLng(latlng)
        .setContent(html)
        .openOn(STATE.map);
}

// ===== ERROR HANDLING =====

function showError(message) {
    console.error(message);
    // Could add a toast notification here
}

// ===== MAP EVENT HANDLERS =====
// Bound inside initMap() after map creation.
