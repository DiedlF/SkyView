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
    stepSize: 1,
    dayMeta: [],
    mergedRunTime: null,
    mergedLabel: null,
    currentRun: null,
    selectedModel: '',
    selectedRun: '',
    overlayReady: false,
    pendingOverlayUpdate: false,
    modelCapabilities: {},
    currentModel: null,
    runs: []
};

// Initialize app on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initControls();
    loadCapabilities();
    loadVariables();
    loadTimesteps();
    startTimelineAutoRefresh();
    startProviderHealthPolling();

    // Overlay-ready gate: prevent first render until layout/map metrics are stable.
    window.addEventListener('load', () => {
        setTimeout(() => {
            if (!STATE.map) return;
            STATE.map.invalidateSize();
            STATE.overlayReady = true;
            if (STATE.pendingOverlayUpdate || (STATE.selectedVariable && STATE.selectedTime)) {
                STATE.pendingOverlayUpdate = false;
                updateOverlay();
            }
        }, 120);
    }, { once: true });
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
    // Data source selectors
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', async (e) => {
            STATE.selectedModel = e.target.value || '';
            STATE.selectedRun = '';
            populateRunSelect();
            await loadVariables();
            await loadTimesteps();
            applyModelMapBounds();
        });
    }

    const runSelect = document.getElementById('run-select');
    if (runSelect) {
        runSelect.addEventListener('change', async (e) => {
            STATE.selectedRun = e.target.value || '';
            await loadVariables();
            await loadTimesteps();
        });
    }

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

    // Date strip follows horizontal timeline scroll
    const timeline = document.getElementById('time-buttons');
    timeline.addEventListener('scroll', updateDateStrip);
    window.addEventListener('resize', updateDateStrip);

    // Sidebar toggle (mobile)
    document.getElementById('sidebar-toggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });
}

// ===== DATA LOADING =====

function setStatus(message = '', level = '') {
    const el = document.getElementById('status-text');
    if (!el) return;
    el.textContent = message;
    el.classList.remove('loading', 'error');
    if (level) el.classList.add(level);
}

function getSelectedModelId() {
    const currentIndex = STATE.timesteps.indexOf(STATE.selectedTime);
    const stepInfo = currentIndex >= 0 ? STATE._mergedSteps?.[currentIndex] : null;
    return stepInfo?.model || null;
}

function applyModelMapBounds() {
    if (!STATE.map) return;
    const model = STATE.selectedModel || STATE.currentModel || getSelectedModelId();
    const caps = model ? STATE.modelCapabilities[model] : null;
    const b = caps?.bbox;
    if (b && Number.isFinite(b.latMin) && Number.isFinite(b.lonMin) && Number.isFinite(b.latMax) && Number.isFinite(b.lonMax)) {
        const pad = 1.0;
        const south = Math.max(-85, b.latMin - pad);
        const west = Math.max(-180, b.lonMin - pad);
        const north = Math.min(85, b.latMax + pad);
        const east = Math.min(180, b.lonMax + pad);
        STATE.map.setMaxBounds([[south, west], [north, east]]);
    } else {
        STATE.map.setMaxBounds([[-85, -180], [85, 180]]);
    }
}

function applyVariableAvailabilityForModel() {
    // Rebuild visible variable list whenever model/capabilities change.
    populateVariableSelect();
}

async function loadCapabilities() {
    try {
        const response = await fetch('/api/capabilities');
        if (!response.ok) return;
        const data = await response.json();
        STATE.modelCapabilities = data.models || {};
        applyVariableAvailabilityForModel();
        applyModelMapBounds();
    } catch (_err) {
        // Keep graceful fallback when capability endpoint is temporarily unavailable.
    }
}

async function loadVariables() {
    try {
        const qs = new URLSearchParams();
        if (STATE.selectedModel) qs.set('model', STATE.selectedModel);
        if (STATE.selectedRun) qs.set('run', STATE.selectedRun);
        const response = await fetch(`/api/variables${qs.toString() ? `?${qs.toString()}` : ''}`);
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
        // 1) Always refresh full run list (keep model selector stable)
        const runsResp = await fetch('/api/timesteps');
        const runsData = await runsResp.json();
        STATE.runs = runsData.runs || [];
        populateModelSelect();
        populateRunSelect();

        // 2) Fetch timeline with current model/run selection
        const qs = new URLSearchParams();
        if (STATE.selectedModel) qs.set('model', STATE.selectedModel);
        if (STATE.selectedRun) qs.set('run', STATE.selectedRun);

        const response = await fetch(`/api/timesteps${qs.toString() ? `?${qs.toString()}` : ''}`);
        const data = await response.json();

        const merged = data.merged;
        if (merged && merged.steps) {
            STATE.timesteps = merged.steps.map(s => s.validTime);
            STATE._mergedSteps = merged.steps;
            STATE.mergedRunTime = merged.runTime || null;
            STATE.mergedLabel = "ICON-D2 + EU";
            STATE.currentRun = merged.run || null;
        } else {
            const src = getTimelineSource(data);
            if (src && src.steps) {
                STATE.timesteps = src.steps.map(s => s.validTime);
                STATE._mergedSteps = src.steps;
                STATE.mergedRunTime = src.runTime || null;
                STATE.mergedLabel = src.label || null;
                STATE.currentRun = src.run || null;
            } else {
                STATE.timesteps = [];
                STATE._mergedSteps = null;
                STATE.mergedRunTime = null;
                STATE.mergedLabel = null;
                STATE.currentRun = null;
            }
        }
        populateTimeButtons();
        updateModelInfo();

    } catch (error) {
        console.error('Failed to load timesteps:', error);
        showError('Failed to load timesteps');
    }
}

function getTimelineSource(data) {
    const merged = data?.merged;
    if (merged && Array.isArray(merged.steps) && merged.steps.length) {
        return {
            run: merged.run || null,
            runTime: merged.runTime || null,
            label: 'ICON-D2 + EU',
            steps: merged.steps,
        };
    }

    const firstRun = data?.runs?.[0];
    if (firstRun && Array.isArray(firstRun.steps) && firstRun.steps.length) {
        return {
            run: firstRun.run || null,
            runTime: firstRun.runTime || null,
            label: firstRun.model ? firstRun.model.toUpperCase().replace('_', '-') : null,
            steps: firstRun.steps,
        };
    }

    if (Array.isArray(data?.timesteps) && data.timesteps.length) {
        return {
            run: null,
            runTime: null,
            label: null,
            steps: data.timesteps.map(vt => ({ validTime: vt })),
        };
    }

    return null;
}

function populateModelSelect() {
    const sel = document.getElementById('model-select');
    if (!sel) return;

    // Keep full model list stable (do not collapse to currently filtered runs).
    const capModels = Object.keys(STATE.modelCapabilities || {});
    const runModels = [...new Set((STATE.runs || []).map(r => r.model).filter(Boolean))];
    const models = [...new Set([...capModels, ...runModels])].sort();
    const prev = STATE.selectedModel || '';

    sel.innerHTML = '';
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m.toUpperCase().replace('_', '-');
        sel.appendChild(opt);
    });

    // Blended/merged disabled for now: always keep an explicit model selected.
    if (models.includes(prev)) {
        sel.value = prev;
    } else if (models.includes('icon_d2')) {
        sel.value = 'icon_d2';
    } else {
        sel.value = models[0] || '';
    }
    STATE.selectedModel = sel.value;
}

function populateRunSelect() {
    const sel = document.getElementById('run-select');
    if (!sel) return;

    const runs = (STATE.runs || []).filter(r => !STATE.selectedModel || r.model === STATE.selectedModel);
    const prev = STATE.selectedRun || '';

    sel.innerHTML = '';
    const latest = document.createElement('option');
    latest.value = '';
    latest.textContent = 'Latest';
    sel.appendChild(latest);

    runs.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.run;
        opt.textContent = `${r.run} (${(r.model || '').toUpperCase().replace('_','-')})`;
        sel.appendChild(opt);
    });

    sel.value = runs.some(r => r.run === prev) ? prev : '';
    STATE.selectedRun = sel.value;
}

function startTimelineAutoRefresh() {
    // Keep explorer timeline fresh like Skyview frontend.
    setInterval(async () => {
        try {
            const qs = new URLSearchParams();
            if (STATE.selectedModel) qs.set('model', STATE.selectedModel);
            if (STATE.selectedRun) qs.set('run', STATE.selectedRun);
            const response = await fetch(`/api/timesteps${qs.toString() ? `?${qs.toString()}` : ''}`);
            if (!response.ok) return;
            const data = await response.json();
            const source = getTimelineSource(data);
            if (!source || !source.steps.length) return;

            const newTimesteps = source.steps.map(s => s.validTime);
            const newRun = source.run;

            if (STATE.currentRun && newRun && newRun !== STATE.currentRun) {
                const oldTimes = STATE.timesteps.slice();
                const oldIdx = oldTimes.indexOf(STATE.selectedTime);
                const wasAtStart = oldIdx === 0;
                const wasAtEnd = oldIdx === oldTimes.length - 1;
                const oldSelected = STATE.selectedTime;

                STATE.timesteps = newTimesteps;
                STATE._mergedSteps = source.steps;
                STATE.currentRun = newRun;
                STATE.mergedRunTime = source.runTime;
                STATE.mergedLabel = source.label;

                if (wasAtStart) {
                    STATE.selectedTime = STATE.timesteps[0] || null;
                } else if (wasAtEnd) {
                    STATE.selectedTime = STATE.timesteps[STATE.timesteps.length - 1] || null;
                } else if (oldSelected) {
                    let bestIdx = 0;
                    let bestDist = Infinity;
                    const oldMs = new Date(oldSelected).getTime();
                    STATE.timesteps.forEach((vt, i) => {
                        const dist = Math.abs(new Date(vt).getTime() - oldMs);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestIdx = i;
                        }
                    });
                    STATE.selectedTime = STATE.timesteps[bestIdx] || null;
                } else {
                    STATE.selectedTime = STATE.timesteps[0] || null;
                }

                populateTimeButtons();
                updateModelInfo();
                updateOverlay();

                const runEl = document.getElementById('run-time');
                if (runEl) {
                    runEl.style.color = '#00AA00';
                    setTimeout(() => { runEl.style.color = ''; }, 3000);
                }
            } else if (
                (newRun && newRun === STATE.currentRun && newTimesteps.length > STATE.timesteps.length) ||
                (!newRun && newTimesteps.length > STATE.timesteps.length)
            ) {
                const oldLen = STATE.timesteps.length;
                const oldIdx = STATE.timesteps.indexOf(STATE.selectedTime);
                const wasAtEnd = oldIdx === oldLen - 1;

                STATE.timesteps = newTimesteps;
                STATE._mergedSteps = source.steps;
                STATE.currentRun = newRun || STATE.currentRun;
                STATE.mergedRunTime = source.runTime;
                STATE.mergedLabel = source.label;

                if (wasAtEnd) {
                    STATE.selectedTime = STATE.timesteps[STATE.timesteps.length - 1] || STATE.selectedTime;
                }

                populateTimeButtons();
                updateModelInfo();
                updateOverlay();
            }
        } catch (_err) {
            // Silent polling; ignore temporary errors.
        }
    }, 60000);
}

function startProviderHealthPolling() {
    const run = async () => {
        try {
            const res = await fetch('/api/provider_health');
            if (!res.ok) return;
            const data = await res.json();
            if (data.status === 'critical') {
                setStatus(`Provider critical: ${(data.reasons || []).join(', ') || 'check backend'}`, 'error');
            } else if (data.status === 'warning') {
                setStatus(`Provider warning: ${(data.reasons || []).join(', ')}`, 'error');
            }
        } catch (_e) {
            // Ignore transient provider health polling errors.
        }
    };

    run();
    setInterval(run, 45000);
}

// ===== UI POPULATION =====

function populateVariableSelect() {
    const select = document.getElementById('variable-select');
    if (!select) return;
    select.innerHTML = '';

    const model = STATE.selectedModel || STATE.currentModel || getSelectedModelId();
    const caps = model ? STATE.modelCapabilities[model] : null;
    const allowed = caps?.variables ? new Set(caps.variables) : null;

    const visibleVariables = allowed
        ? STATE.variables.filter(v => allowed.has(v.name))
        : STATE.variables.slice();

    // Group variables by category
    const grouped = {};
    visibleVariables.forEach(v => {
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
            option.textContent = (v.unit && v.unit !== 'code') ? `${v.name} (${v.unit})` : `${v.name}`;
            optgroup.appendChild(option);
        });
        select.appendChild(optgroup);
    });

    // Keep current variable if still available, otherwise fallback to first visible variable.
    const keep = STATE.selectedVariable && visibleVariables.find(v => v.name === STATE.selectedVariable.name);
    if (keep) {
        STATE.selectedVariable = keep;
        select.value = keep.name;
    } else if (visibleVariables.length > 0) {
        STATE.selectedVariable = visibleVariables[0];
        select.value = STATE.selectedVariable.name;
        setStatus(`Variables updated for ${model?.toUpperCase().replace('_','-') || 'model'}`, 'loading');
    } else {
        STATE.selectedVariable = null;
        setStatus(`No variables available for ${model?.toUpperCase().replace('_','-') || 'model'}`, 'error');
    }

    updateVariableInfo();
}

function populateTimeButtons() {
    const timeline = document.getElementById('time-buttons');
    timeline.innerHTML = '';
    STATE.dayMeta = [];

    let lastDate = null;
    let lastModel = null;

    STATE.timesteps.forEach((ts, index) => {
        const date = new Date(ts);
        const day = String(date.getUTCDate()).padStart(2, '0');
        const month = String(date.getUTCMonth() + 1).padStart(2, '0');
        const dateKey = `${day}.${month}.`;
        const hourStr = String(date.getUTCHours()).padStart(2, '0');

        if (dateKey !== lastDate) {
            const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
            STATE.dayMeta.push({
                dateStr: `${dayNames[date.getUTCDay()]} ${dateKey}`,
                firstIdx: index,
                noonIdx: null
            });
            lastDate = dateKey;
        }

        const curDay = STATE.dayMeta[STATE.dayMeta.length - 1];
        if (date.getUTCHours() === 12) curDay.noonIdx = index;

        const btn = document.createElement('button');
        btn.className = 'hour-btn';
        btn.textContent = hourStr;
        btn.dataset.time = ts;
        btn.dataset.index = index;

        if (index === curDay.firstIdx && index > 0) {
            btn.classList.add('day-start');
        }

        const stepInfo = STATE._mergedSteps?.[index];
        if (stepInfo?.model && lastModel && stepInfo.model !== lastModel) {
            btn.classList.add('model-switch');
            btn.title = `Model switch: ${lastModel.toUpperCase().replace('_', '-')} → ${stepInfo.model.toUpperCase().replace('_', '-')}`;
        }
        lastModel = stepInfo?.model || lastModel;

        btn.addEventListener('click', () => selectTime(ts));
        timeline.appendChild(btn);
    });

    STATE.dayMeta.forEach((dm, i) => {
        if (dm.noonIdx === null) {
            const nextStart = i < STATE.dayMeta.length - 1 ? STATE.dayMeta[i + 1].firstIdx : STATE.timesteps.length;
            dm.noonIdx = Math.floor((dm.firstIdx + nextStart) / 2);
        }
    });

    updateDateStrip();

    // Keep current selection if possible; otherwise select first timestep
    if (STATE.timesteps.length > 0) {
        const preferred = STATE.selectedTime && STATE.timesteps.includes(STATE.selectedTime)
            ? STATE.selectedTime
            : STATE.timesteps[0];
        selectTime(preferred);
    }
}

// ===== TIME SELECTION =====

function updateDateStrip() {
    const timeline = document.getElementById('time-buttons');
    const dateStrip = document.getElementById('date-strip');
    dateStrip.innerHTML = '';

    const hours = timeline.querySelectorAll('.hour-btn');
    if (!hours.length || !STATE.dayMeta.length) return;

    const wrapperRect = timeline.parentElement.getBoundingClientRect();
    const wrapperLeft = wrapperRect.left;
    const wrapperRight = wrapperRect.right;
    const wrapperWidth = wrapperRect.width;

    STATE.dayMeta.forEach((dm, i) => {
        const label = document.createElement('div');
        label.className = 'date-label';
        label.textContent = dm.dateStr;
        dateStrip.appendChild(label);

        const noonEl = hours[dm.noonIdx];
        const firstEl = hours[dm.firstIdx];
        const lastIdx = i < STATE.dayMeta.length - 1 ? STATE.dayMeta[i + 1].firstIdx - 1 : hours.length - 1;
        const lastEl = hours[lastIdx];
        if (!noonEl || !firstEl || !lastEl) return;

        const noonRect = noonEl.getBoundingClientRect();
        const firstRect = firstEl.getBoundingClientRect();
        const lastRect = lastEl.getBoundingClientRect();
        const labelWidth = label.offsetWidth;

        let idealLeft = noonRect.left + noonRect.width / 2 - wrapperLeft - labelWidth / 2;
        const minLeft = firstRect.left - wrapperLeft;
        const maxLeft = lastRect.right - wrapperLeft - labelWidth;
        let finalLeft = Math.max(0, Math.min(wrapperWidth - labelWidth, Math.max(minLeft, Math.min(maxLeft, idealLeft))));

        if (lastRect.right < wrapperLeft || firstRect.left > wrapperRight) {
            label.style.display = 'none';
        } else {
            label.style.left = `${finalLeft}px`;
        }
    });
}

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
        setTimeout(updateDateStrip, 100);
    }

    updateModelInfo();
    updateOverlay();
}

function stepTime(direction) {
    const currentIndex = STATE.timesteps.indexOf(STATE.selectedTime);
    if (currentIndex === -1) return;

    if (STATE.stepSize === 1) {
        const newIndex = currentIndex + direction;
        if (newIndex >= 0 && newIndex < STATE.timesteps.length) {
            selectTime(STATE.timesteps[newIndex]);
        }
        return;
    }

    const currentMs = new Date(STATE.timesteps[currentIndex]).getTime();
    const targetMs = currentMs + (direction * STATE.stepSize * 3600000);
    let bestIdx = currentIndex;

    if (direction > 0) {
        for (let i = currentIndex + 1; i < STATE.timesteps.length; i++) {
            const t = new Date(STATE.timesteps[i]).getTime();
            if (t >= targetMs) {
                bestIdx = i;
                break;
            }
            if (i === STATE.timesteps.length - 1) bestIdx = i;
        }
    } else {
        for (let i = currentIndex - 1; i >= 0; i--) {
            const t = new Date(STATE.timesteps[i]).getTime();
            if (t <= targetMs) {
                bestIdx = i;
                break;
            }
            if (i === 0) bestIdx = i;
        }
    }

    if (bestIdx !== currentIndex) {
        selectTime(STATE.timesteps[bestIdx]);
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

function formatDateShortUTC(date) {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const dd = String(date.getUTCDate()).padStart(2, '0');
    const mm = String(date.getUTCMonth() + 1).padStart(2, '0');
    const hh = String(date.getUTCHours()).padStart(2, '0');
    return `${days[date.getUTCDay()]}, ${dd}.${mm}., ${hh} UTC`;
}

function updateModelInfo() {
    const modelEl = document.getElementById('model-name');
    const runEl = document.getElementById('run-time');
    const validEl = document.getElementById('valid-time');
    if (!modelEl || !runEl || !validEl) return;

    const currentIndex = STATE.timesteps.indexOf(STATE.selectedTime);
    const currentStep = currentIndex >= 0 ? STATE._mergedSteps?.[currentIndex] : null;

    const modelId = currentStep?.model || null;
    STATE.currentModel = modelId;

    const model = modelId
        ? modelId.toUpperCase().replace('_', '-')
        : (STATE.mergedLabel || '--');
    modelEl.textContent = model;
    applyVariableAvailabilityForModel();
    applyModelMapBounds();

    if (STATE.mergedRunTime) {
        runEl.textContent = formatDateShortUTC(new Date(STATE.mergedRunTime));
    } else {
        runEl.textContent = 'N/A';
    }

    if (STATE.selectedTime) {
        validEl.textContent = formatDateShortUTC(new Date(STATE.selectedTime));
    } else {
        validEl.textContent = '--';
    }
}

// ===== OVERLAY UPDATE =====

async function updateOverlay() {
    if (!STATE.selectedVariable || !STATE.selectedTime || !STATE.map) return;

    setStatus('Loading layer…', 'loading');

    if (!STATE.overlayReady) {
        STATE.pendingOverlayUpdate = true;
        return;
    }

    const reqId = ++STATE.overlayRequestSeq;

    // Cancel previous in-flight request (range fetch)
    if (STATE.overlayAbortCtrl) {
        STATE.overlayAbortCtrl.abort();
    }
    STATE.overlayAbortCtrl = new AbortController();

    try {
        const bounds = STATE.map.getBounds();
        const bbox = [
            bounds.getSouth(),
            bounds.getWest(),
            bounds.getNorth(),
            bounds.getEast()
        ].join(',');

        // Resolve color range once per viewport so all fetched tiles share one scale
        // (prevents tile seams and keeps legend/range UI in sync).
        const rangeParams = new URLSearchParams({
            var: STATE.selectedVariable.name,
            bbox: bbox,
            time: STATE.selectedTime,
            palette: STATE.selectedPalette,
        });
        if (STATE.selectedModel) rangeParams.append('model', STATE.selectedModel);
        if (STATE.selectedRun) rangeParams.append('run', STATE.selectedRun);

        if (!STATE.autoRange && STATE.rangeMin !== null && STATE.rangeMax !== null) {
            rangeParams.append('vmin', STATE.rangeMin);
            rangeParams.append('vmax', STATE.rangeMax);
        }

        let effectiveMin = STATE.rangeMin;
        let effectiveMax = STATE.rangeMax;

        if (STATE.autoRange) {
            const rangeRes = await fetch(`/api/overlay_range?${rangeParams.toString()}`, {
                signal: STATE.overlayAbortCtrl.signal
            });
            if (!rangeRes.ok) return;
            const rangeData = await rangeRes.json();
            effectiveMin = rangeData.vmin;
            effectiveMax = rangeData.vmax;

            updateLegend(effectiveMin, effectiveMax);
            const minInput = document.getElementById('range-min');
            const maxInput = document.getElementById('range-max');
            if (minInput) minInput.value = Number.isFinite(effectiveMin) ? effectiveMin.toFixed(3) : '';
            if (maxInput) maxInput.value = Number.isFinite(effectiveMax) ? effectiveMax.toFixed(3) : '';
        } else {
            if (Number.isFinite(effectiveMin) && Number.isFinite(effectiveMax)) {
                updateLegend(effectiveMin, effectiveMax);
            }
        }

        // Ignore stale async result
        if (reqId !== STATE.overlayRequestSeq) return;

        const clientClass = window.innerWidth <= 768 ? 'mobile' : 'desktop';
        const tileParams = new URLSearchParams({
            var: STATE.selectedVariable.name,
            time: STATE.selectedTime,
            palette: STATE.selectedPalette,
            clientClass,
        });
        if (STATE.selectedModel) tileParams.append('model', STATE.selectedModel);
        if (STATE.selectedRun) tileParams.append('run', STATE.selectedRun);
        if (Number.isFinite(effectiveMin)) tileParams.append('vmin', String(effectiveMin));
        if (Number.isFinite(effectiveMax)) tileParams.append('vmax', String(effectiveMax));

        const tileUrl = `/api/overlay_tile/{z}/{x}/{y}.png?${tileParams.toString()}`;

        // Replace overlay layer atomically
        if (STATE.overlay) {
            STATE.map.removeLayer(STATE.overlay);
            STATE.overlay = null;
        }

        STATE.overlay = L.tileLayer(tileUrl, {
            tileSize: 256,
            opacity: STATE.opacity,
            updateWhenZooming: false,
            updateWhenIdle: true,
            keepBuffer: 1,
            crossOrigin: true,
            zIndex: 250,
        }).addTo(STATE.map);
        setStatus('');
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Failed to update overlay:', error);
            setStatus('Overlay update failed', 'error');
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


function wwInterpretation(code) {
    if (!Number.isFinite(code)) return null;
    const ww = Math.round(code);
    if (ww === 0) return 'None';
    if (ww >= 1 && ww <= 3) return 'Cloud dev.';
    if (ww >= 4 && ww <= 9) return 'Haze/smoke/dust';
    if (ww === 45) return 'Fog';
    if (ww === 48) return 'Rime fog';
    if (ww >= 50 && ww <= 59) return 'Drizzle';
    if (ww >= 60 && ww <= 69) return 'Rain / freezing';
    if (ww >= 70 && ww <= 79) return 'Snow';
    if (ww >= 80 && ww <= 84) return 'Rain showers';
    if (ww >= 85 && ww <= 86) return 'Snow showers';
    if (ww >= 95 && ww <= 99) return 'Thunderstorm';
    return 'Other';
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
        if (STATE.selectedModel) params.append('model', STATE.selectedModel);
        if (STATE.selectedRun) params.append('run', STATE.selectedRun);

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

    const value = values[selected.name];

    function formatValue(v, variable) {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
        const n = Number(v);
        const unit = variable.unit || '';

        let digits = 1;
        if (variable.name === 'ww' || unit === 'code' || unit === 'index') digits = 0;
        else if (unit === 'm' || unit === 'Pa' || unit === '%') digits = 0;
        else if (unit === 'K') digits = 1;
        else if (unit === 'm/s' || unit === 'J/kg') digits = 1;

        const valueStr = digits === 0 ? String(Math.round(n)) : n.toFixed(digits);
        if (variable.name === 'ww' || unit === 'code') return valueStr;
        return unit ? `${valueStr} ${unit}` : valueStr;
    }

    const valueText = formatValue(value, selected);

    let extra = '';
    if (selected.name === 'ww' && value !== null && value !== undefined) {
        const wwText = wwInterpretation(Number(value));
        if (wwText) {
            extra = `<div class="popup-meta"><strong>WW:</strong> ${wwText}</div>`;
        }
    }

    const html = `
        <div class="popup-group">
            <div class="popup-group-title">${selected.group}</div>
            <div class="popup-var">
                <span class="popup-var-name">${selected.name}</span>
                <span class="popup-var-value">${valueText}</span>
            </div>
            <div class="popup-meta">${selected.desc || ''}</div>
            ${extra}
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
    setStatus(message, 'error');
}

// ===== MAP EVENT HANDLERS =====
// Bound inside initMap() after map creation.
