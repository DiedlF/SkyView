// app.js - Main application logic

let map, symbolLayer, windLayer, markerLayer, d2BorderLayer = null, overlayLayer = null, debounceTimer, overlayDebounce, windDebounce, timesteps = [], currentTimeIndex = 0, currentRun = '';
let overlayRequestSeq = 0;
let overlayAbortCtrl = null;
let overlayPrewarmCtrl = null;
let overlayObjectUrl = null;
let currentOverlay = 'none';
let windEnabled = false;
let windLevel = '10m';
let modelCapabilities = {};  // Store model capabilities from API

function showGlobalError(msg) {
  let el = document.getElementById('global-error-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'global-error-banner';
    el.style.position = 'fixed';
    el.style.left = '50%';
    el.style.bottom = '12px';
    el.style.transform = 'translateX(-50%)';
    el.style.background = 'rgba(140,30,30,0.95)';
    el.style.color = '#fff';
    el.style.padding = '8px 12px';
    el.style.borderRadius = '8px';
    el.style.fontSize = '12px';
    el.style.zIndex = '9999';
    el.style.maxWidth = '85vw';
    el.style.textAlign = 'center';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(showGlobalError._t);
  showGlobalError._t = setTimeout(() => {
    if (el) el.style.display = 'none';
  }, 6000);
}

function ensureFallbackBanner() {
  let el = document.getElementById('fallback-timestep-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'fallback-timestep-banner';
    el.style.position = 'fixed';
    el.style.left = '50%';
    el.style.top = '12px';
    el.style.transform = 'translateX(-50%)';
    el.style.background = 'rgba(160,110,20,0.95)';
    el.style.color = '#fff';
    el.style.padding = '8px 12px';
    el.style.borderRadius = '8px';
    el.style.fontSize = '12px';
    el.style.zIndex = '9998';
    el.style.maxWidth = '90vw';
    el.style.textAlign = 'center';
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

function ensureEuMissingBanner() {
  let el = document.getElementById('eu-missing-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'eu-missing-banner';
    el.style.position = 'fixed';
    el.style.left = '50%';
    el.style.top = '12px';
    el.style.transform = 'translateX(-50%)';
    el.style.background = 'rgba(160,110,20,0.95)';
    el.style.color = '#fff';
    el.style.padding = '8px 14px';
    el.style.borderRadius = '8px';
    el.style.fontSize = '12px';
    el.style.zIndex = '9998';
    el.style.maxWidth = '90vw';
    el.style.textAlign = 'center';
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

function updateFallbackBanner(data) {
  // EU data missing: show infobox when the expected EU timestep is absent from disk (ingest gap).
  // Areas outside ICON-D2 domain will have no data for this timestep.
  const el = ensureEuMissingBanner();
  const infoEl = document.getElementById('fallback-info');
  const infoSep = document.getElementById('fallback-info-sep');
  const missing = data?.diagnostics?.euDataMissing;

  if (!missing) {
    el.style.display = 'none';
    if (infoEl) infoEl.style.display = 'none';
    if (infoSep) infoSep.style.display = 'none';
    return;
  }

  el.textContent = '⚠ EU weather data missing for this timestep — areas outside ICON-D2 domain may be incomplete';
  el.style.display = 'block';

  if (infoEl) {
    infoEl.textContent = 'EU data missing';
    infoEl.style.display = 'inline';
  }
  if (infoSep) infoSep.style.display = 'inline';
}

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  showGlobalError('Network/API error. Please retry or reload.');
});

window.addEventListener('error', (event) => {
  console.error('Unhandled frontend error:', event.error || event.message);
  showGlobalError('Unexpected frontend error. Please reload if this persists.');
});

const I18N = {
  en: {
    'layers.title': 'Layers',
    'info.model': 'Model:',
    'info.run': 'Run:',
    'info.zoom': 'Zoom:',
    'layer.convection': 'Convection',
    'layer.wind': 'Wind',
    'layer.wind.gust10m': '10m Gusts',
    'layer.overlay.title': 'Overlay (ICON)',
    'layer.none': 'None',
    'layer.precip': 'Precipitation',
    'layer.precip.total': 'Total',
    'layer.precip.rain': 'Rain',
    'layer.precip.snow': 'Snow',
    'layer.precip.hail': 'Hail/Graupel',
    'layer.cloudcover': 'Cloud Cover',
    'layer.clouds.low': 'Low',
    'layer.clouds.mid': 'Mid',
    'layer.clouds.high': 'High',
    'layer.clouds.total': 'Total',
    'layer.temperature': 'Temperature',
    'layer.temp.t2m': '2m',
    'layer.diagnostic': 'Diagnostic',
    'layer.cloudbase': 'Cloud base (convective)',
    'layer.dryconv': 'Dry Convection Top',
    'layer.mh': 'Boundary layer depth',
    'layer.ashfl': 'Surface heat flux',
    'layer.relhum': 'Relative humidity 2m',
    'layer.dewspread': 'Dew point spread 2m',
    'layer.sigwx': 'Significant weather',
    'layer.ceiling': 'Ceiling',
    'layer.convthickness': 'Cloud thickness (convective)',
    'layer.lpi': 'LPI',
    'layer.experimental': 'Experimental',
    'layer.climbrate': 'Climb Rate (lapse_rate)',
    'layer.lcl': 'Cloud Base (spread * 125)',
    'layer.marker': 'Marker',
    helpTitle: 'How to read Skyview',
    helpHtml: `
      <h4>Data sources</h4>
      <ul>
        <li>Primary model: ICON-D2 (high resolution).</li>
        <li>Fallback/outside D2: ICON-EU where available.</li>
      </ul>
      <h4>Symbols</h4>
      <p style="margin:4px 0 8px"><b>Priority:</b> significant weather (ww) &rarr; convective cloud &rarr; non-convective cloud.</p>
      <div class="cloud-sym-grid">
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"><line x1="3" y1="18" x2="33" y2="18" stroke="#333" stroke-width="3" stroke-linecap="round"/></svg></div>
          <div><b>St</b> &mdash; Stratus<br><span class="cloud-sym-desc">ceiling &lt; 2000 m, no convection, low cloud cover (clcl) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"><path d="M4 14 Q4 26 11 26 Q18 26 18 14 Q18 26 25 26 Q32 26 32 14" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/></svg></div>
          <div><b>Ac</b> &mdash; Altocumulus<br><span class="cloud-sym-desc">ceiling 2000&ndash;7000 m, no convection, mid cloud cover (clcm) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M 6,25 h 20 A 5,5 0 0 0 26,15" fill="none" stroke="#333" stroke-width="2.6" stroke-linecap="round"/>
          </svg></div>
          <div><b>Ci</b> &mdash; Cirrus<br><span class="cloud-sym-desc">ceiling &ge; 7000 m, no convection, high cloud cover (clch) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <text x="18" y="26" text-anchor="middle" font-size="22" font-weight="bold" fill="#333">b</text></svg></div>
          <div><b>b</b> &mdash; Blue thermal<br><span class="cloud-sym-desc">convection &ge; 300 mAGL, no visible cloud base (dry thermals)
          </span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/></svg></div>
          <div><b>Cu</b> &mdash; Cu humilis<br><span class="cloud-sym-desc">convection &ge; 300 mAGL, shallow convection, depth &le; 2000 m</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <path d="M11 19 Q18 0 25 19" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="18" y1="10" x2="18" y2="23" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          </svg></div>
          <div><b>Cu con</b> &mdash; Cu congestus<br><span class="cloud-sym-desc">convection &ge; 300 mAGL, depth &gt; 2000 m, no LPI/anvil</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="12" y1="17" x2="9" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="24" y1="17" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="9" y1="10" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          </svg></div>
          <div><b>Cb</b> &mdash; Cumulonimbus<br><span class="cloud-sym-desc">convection &ge; 300 mAGL and (LPI &gt; 7 or (depth &gt; 4 km and CAPE &gt; 1000 J/kg))</span></div>
        </div>
      </div>
      
      <h4>Overlays</h4>
      <ul>
        <li>Overlays visualize one field at a time (precip, clouds, diagnostics).</li>
        <li>Colors are relative field scales; use point click for exact values.</li>
      </ul>
      <h4>Time &amp; model info</h4>
      <ul>
        <li>Run = forecast initialization time (UTC).</li>
        <li>Model label reflects what is visible in your current viewport.</li>
      </ul>
    `,
  },
  de: {
    'layers.title': 'Ebenen',
    'info.model': 'Modell:',
    'info.run': 'Lauf:',
    'info.zoom': 'Zoom:',
    'layer.convection': 'Konvektion',
    'layer.wind': 'Wind',
    'layer.wind.gust10m': '10m Böen',
    'layer.overlay.title': 'Overlay (ICON)',
    'layer.none': 'Keines',
    'layer.precip': 'Niederschlag',
    'layer.precip.total': 'Gesamt',
    'layer.precip.rain': 'Regen',
    'layer.precip.snow': 'Schnee',
    'layer.precip.hail': 'Hagel/Graupel',
    'layer.cloudcover': 'Bewölkung',
    'layer.clouds.low': 'Tief',
    'layer.clouds.mid': 'Mittel',
    'layer.clouds.high': 'Hoch',
    'layer.clouds.total': 'Gesamt',
    'layer.temperature': 'Temperatur',
    'layer.temp.t2m': '2m',
    'layer.diagnostic': 'Diagnostik',
    'layer.cloudbase': 'Wolkenbasis (konvektiv)',
    'layer.dryconv': 'Konvektionshöhe trocken',
    'layer.mh': 'Durchmischungshöhe',
    'layer.ashfl': 'Bodenwärmestrom',
    'layer.relhum': 'Relative Feuchte 2m',
    'layer.dewspread': 'Taupunktdifferenz 2m',
    'layer.sigwx': 'Signifikantes Wetter',
    'layer.ceiling': 'Haupt-Wolkenuntergrenze',
    'layer.convthickness': 'Wolkenmächtigkeit (konvektiv)',
    'layer.lpi': 'LPI',
    'layer.experimental': 'Experimentell',
    'layer.climbrate': 'Steigwerte (lapse_rate)',
    'layer.lcl': 'Wolkenbasis (Spread * 125)',
    'layer.marker': 'Markierung',
    helpTitle: 'Skyview Erklärung',
    helpHtml: `
      <h4>Datenquellen</h4>
      <ul>
        <li>Primärmodell: ICON-D2 (hohe Auflösung).</li>
        <li>Fallback/außerhalb D2: ICON-EU, sofern verfügbar.</li>
      </ul>
      <h4>Symbole</h4>
      <p style="margin:4px 0 8px"><b>Priorität:</b> Signifikantes Wetter (ww) &rarr; Konvektionswolke &rarr; nicht-konvektive Wolke.</p>
      <div class="cloud-sym-grid">
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"><line x1="3" y1="18" x2="33" y2="18" stroke="#333" stroke-width="3" stroke-linecap="round"/></svg></div>
          <div><b>St</b> &mdash; Stratus<br><span class="cloud-sym-desc">Haupt-Wolkenuntergrenze &lt; 2000 m, keine Konvektion, tiefe Bewölkung (clcl) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"><path d="M4 14 Q4 26 11 26 Q18 26 18 14 Q18 26 25 26 Q32 26 32 14" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/></svg></div>
          <div><b>Ac</b> &mdash; Altocumulus<br><span class="cloud-sym-desc">Haupt-Wolkenuntergrenze 2000&ndash;7000 m, keine Konvektion, mittlere Bewölkung (clcm) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M 6,25 h 20 A 5,5 0 0 0 26,15" fill="none" stroke="#333" stroke-width="2.6" stroke-linecap="round"/>
          </svg></div>
          <div><b>Ci</b> &mdash; Cirrus<br><span class="cloud-sym-desc">Haupt-Wolkenuntergrenze &ge; 7000 m, keine Konvektion, hohe Bewölkung (clch) &ge; 30%</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <text x="18" y="26" text-anchor="middle" font-size="22" font-weight="bold" fill="#333">b</text>
          </svg></div>
          <div><b>b</b> &mdash; Blauthermik<br><span class="cloud-sym-desc">Konvektion &ge; 300 mAGL, keine sichtbare Wolkenbasis (Blauthermik)</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          </svg></div>
          <div><b>Cu</b> &mdash; Cu humilis<br><span class="cloud-sym-desc">Konvektion &ge; 300 mAGL, flache Konvektion, Tiefe &le; 2000 m</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <path d="M11 19 Q18 0 25 19" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="18" y1="10" x2="18" y2="23" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          </svg></div>
          <div><b>Cu con</b> &mdash; Cu congestus<br><span class="cloud-sym-desc">Konvektion &ge; 300 mAGL, Tiefe &gt; 2000 m, kein LPI/Amboss</span></div>
        </div>
        <div class="cloud-sym-row">
          <div class="cloud-sym-cell"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
          <path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="12" y1="17" x2="9" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="24" y1="17" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="9" y1="10" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          </svg></div>
          <div><b>Cb</b> &mdash; Cumulonimbus<br><span class="cloud-sym-desc">Konvektion &ge; 300 mAGL und (LPI &gt; 7 oder (Tiefe &gt; 4 km und CAPE &gt; 1000 J/kg))</span></div>
        </div>
      </div>
      
      <h4>Overlays</h4>
      <ul>
        <li>Overlays zeigen jeweils ein Feld (Niederschlag, Bewölkung, Diagnostik).</li>
        <li>Farben sind Feldskalen; exakte Werte per Klick auf einen Punkt.</li>
      </ul>
      <h4>Zeit &amp; Modellinfo</h4>
      <ul>
        <li>Lauf = Initialisierungszeit des Forecasts (UTC).</li>
        <li>Die Modellanzeige bezieht sich auf den aktuell sichtbaren Kartenausschnitt.</li>
      </ul>
    `,
  },
};

let currentLang = localStorage.getItem('skyview_lang') || ((navigator.language || navigator.userLanguage || 'en').toLowerCase().startsWith('de') ? 'de' : 'en');

function t(key) {
  const dict = I18N[currentLang] || I18N.en;
  return dict[key] || I18N.en[key] || key;
}

function applyLocale(lang) {
  currentLang = (lang === 'de') ? 'de' : 'en';
  localStorage.setItem('skyview_lang', currentLang);
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key);
  });
  const helpTitle = document.getElementById('help-title');
  const helpBody = document.getElementById('help-body');
  if (helpTitle) helpTitle.textContent = t('helpTitle');
  if (helpBody) helpBody.innerHTML = t('helpHtml');
  const langSel = document.getElementById('lang-select');
  if (langSel && langSel.value !== currentLang) langSel.value = currentLang;
}

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

function getClientId() {
  const key = 'skyview_client_id';
  let v = localStorage.getItem(key);
  if (!v) {
    v = (crypto?.randomUUID ? crypto.randomUUID() : `cid_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`);
    localStorage.setItem(key, v);
  }
  return v;
}

let markerAuthToken = null;
let markerAuthExpMs = 0;
let markerEditingEnabled = true;

async function ensureMarkerAuthToken(force = false) {
  const now = Date.now();
  if (!force && markerAuthToken && now < markerAuthExpMs - 60_000) return markerAuthToken;
  const cid = getClientId();
  const res = await fetch(`/api/marker_auth?clientId=${encodeURIComponent(cid)}`);
  if (!res.ok) await throwHttpError(res, 'API');
  const data = await res.json();
  markerAuthToken = data.token;
  markerAuthExpMs = Date.parse(data.expiresAt || '') || (now + 6 * 3600 * 1000);
  return markerAuthToken;
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
  mh: { title: 'Boundary layer depth', gradient: 'linear-gradient(to right, rgb(220,90,40), rgb(160,170,60), rgb(80,240,100))', labels: ['0m', '3000m+'] },
  relhum_2m: { title: 'Relative humidity 2m', gradient: 'linear-gradient(to right, rgb(230,230,230), rgb(120,150,190), rgb(60,110,190))', labels: ['0%', '100%'] },
  dew_spread_2m: { title: 'Dew point spread 2m', gradient: 'linear-gradient(to right, rgb(70,200,220), rgb(170,140,110), rgb(255,70,40))', labels: ['0 K', '25+ K'] },
  conv_thickness: { title: 'Cloud thickness (convective)', gradient: 'linear-gradient(to right, rgb(40,220,60), rgb(200,200,40), rgb(240,80,40))', labels: ['0m', '6000m'] },
  lpi: { title: 'LPI', gradient: 'linear-gradient(to right, rgb(70,190,80), rgb(160,150,60), rgb(255,70,40))', labels: ['0', '20+'] },
  thermals: { title: 'CAPE_ml', gradient: 'linear-gradient(to right, rgb(50,180,50), rgb(150,150,50), rgb(220,100,30), rgb(255,50,50))', labels: ['50 J/kg', '1000+ J/kg'] },
  climb_rate: { title: 'Climb Rate', gradient: 'linear-gradient(to right, rgb(50,200,50), rgb(180,200,50), rgb(220,150,30), rgb(255,50,50))', labels: ['0 m/s', '5 m/s'] },
  lcl: { title: 'Cloud Base (LCL) MSL', gradient: 'linear-gradient(to right, rgb(220,60,60), rgb(240,150,60), rgb(180,220,60), rgb(80,240,80))', labels: ['0m', '5000m MSL'] },
  h_snow: { title: 'Snow depth', gradient: 'linear-gradient(to right, rgba(255,255,255,0), rgb(220,235,255), rgb(160,200,255), rgb(100,150,240))', labels: ['0', '100+ cm'] }
};

// Initialize map
map = L.map('map', {
  center: [47.6836, 11.9610],
  zoom: 9,
  minZoom: 5,
  maxZoom: 12,
  // Expanded pan limits to full ICON-EU operational area (approx Europe view)
  maxBounds: L.latLngBounds([[30, -30], [72, 45]]),
  maxBoundsViscosity: 1.0,
  zoomControl: false
});
L.control.zoom({ position: 'bottomleft' }).addTo(map);

// Ocean base provides water coloring (ocean, lakes, rivers) with matching coastlines
// maxNativeZoom=10: inland tiles unavailable at z11+, so upscale z10 tiles
const baseTileOpts = {
  attribution: '',
  updateWhenZooming: false,
  updateWhenIdle: true,
  keepBuffer: 2,
  detectRetina: false,
};

L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
  ...baseTileOpts,
  maxNativeZoom: 10,
  maxZoom: 12,
}).addTo(map);

// Hillshade on top with multiply blend mode — shading applies over the ocean base colors
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}', {
  ...baseTileOpts,
  className: 'hillshade-layer',
}).addTo(map);

symbolLayer = L.layerGroup().addTo(map);
windLayer = L.layerGroup();
markerLayer = L.layerGroup().addTo(map);
d2BorderLayer = L.layerGroup().addTo(map);

// Format date as "DayOfWeek, dd.mm., HH UTC"
function formatDateShort(d) {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${days[d.getUTCDay()]}, ${dd}.${mm}., ${hh} UTC`;
}

function parseRunToDate(run) {
  if (!run || !/^\d{10}$/.test(run)) return null;
  const y = Number(run.slice(0, 4));
  const m = Number(run.slice(4, 6)) - 1;
  const d = Number(run.slice(6, 8));
  const h = Number(run.slice(8, 10));
  return new Date(Date.UTC(y, m, d, h, 0, 0));
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
    loadD2Border();
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
      updateFallbackBanner(data);
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
      
      // Update model/run from response (handles D2/EU-only viewport cases).
      if (data.model) {
        document.getElementById('model').textContent = data.model.toUpperCase().replace('_', '-');
      }
      if (data.run) {
        const runDt = parseRunToDate(data.run);
        if (runDt) {
          document.getElementById('run-time').textContent = formatDateShort(runDt);
        }
      }
    } catch (e) {
      console.error('Error loading symbols:', e);
      markApiFailure('symbols', e);
      updateFallbackBanner(null);
      // Optionally show error in UI
      symbolLayer.clearLayers();
    }
  }, 300);
}

// Overlay value formatting (aligned with API_CONVERGENCE_CONTRACT units)
const OVERLAY_META = {
  total_precip: { label: 'Total precip', unit: 'mm/h', decimals: 2 },
  rain: { label: 'Rain', unit: 'mm/h', decimals: 2 },
  rain_amount: { label: 'Rain', unit: 'mm/h', decimals: 2 },
  snow: { label: 'Snow', unit: 'mm/h', decimals: 2 },
  snow_amount: { label: 'Snow', unit: 'mm/h', decimals: 2 },
  hail: { label: 'Hail/Graupel', unit: 'mm/h', decimals: 2 },
  hail_amount: { label: 'Hail/Graupel', unit: 'mm/h', decimals: 2 },
  sigwx: { label: 'Sig. Weather ww', unit: '', integer: true },
  clouds_low: { label: 'Cloud cover low', unit: '%', decimals: 1 },
  clouds_mid: { label: 'Cloud cover mid', unit: '%', decimals: 1 },
  clouds_high: { label: 'Cloud cover high', unit: '%', decimals: 1 },
  clouds_total: { label: 'Cloud cover total', unit: '%', decimals: 1 },
  clouds_total_mod: { label: 'Cloud cover total mod', unit: '%', decimals: 1 },
  ceiling: { label: 'Ceiling', unit: 'm', integer: true },
  cloud_base: { label: 'Cloud base', unit: 'm', integer: true },
  dry_conv_top: { label: 'Dry convection top', unit: 'm', integer: true },
  mh: { label: 'Boundary layer depth', unit: 'm', decimals: 1 },
  relhum_2m: { label: 'Relative humidity 2m', unit: '%', decimals: 1 },
  dew_spread_2m: { label: 'Dew point spread 2m', unit: 'K', decimals: 1 },
  conv_thickness: { label: 'Convective thickness', unit: 'm', integer: true },
  lpi: { label: 'LPI', unit: '', decimals: 1 },
  thermals: { label: 'CAPE_ml', unit: 'J/kg', decimals: 1 },
  climb_rate: { label: 'Climb', unit: 'm/s', decimals: 1 },
  lcl: { label: 'Cloud base (LCL)', unit: 'm MSL', integer: true },
  h_snow: { label: 'Snow depth', unit: 'm', decimals: 2 },
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
      const aliasKey = ({ rain_amount: 'rain', snow_amount: 'snow', hail_amount: 'hail' })[overlayKey] || overlayKey;
      let ovVal = data.overlay_values[overlayKey];
      if (ovVal == null) ovVal = data.overlay_values[aliasKey];

      // Climb-rate robustness: if direct value missing, derive from thermal class.
      if (overlayKey === 'climb_rate' && (ovVal == null) && data.overlay_values.thermal_class != null) {
        const tc = Number(data.overlay_values.thermal_class);
        if (Number.isFinite(tc)) {
          ovVal = ({0: 0.0, 1: 1.0, 2: 2.0, 3: 3.2})[Math.max(0, Math.min(3, Math.round(tc)))];
        }
      }
      const formatted = formatOverlayValue(overlayKey, ovVal);
      if (formatted) {
        lines.push(formatted);
      } else {
        const meta = OVERLAY_META[overlayKey] || OVERLAY_META[aliasKey];
        if (meta) lines.push(`${meta.label}: n/a`);
      }
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

    const pointModel = String(data.model || model || '').replace('-', '_');
    const showSkewT = pointModel === 'icon_d2';
    let btns = `<div style="margin-top:8px;display:flex;gap:6px;align-items:center">`;
    if (showSkewT) {
      btns += `<button onclick="openEmagramAt(${Number(lat).toFixed(5)},${Number(lon).toFixed(5)},'${String(time || 'latest').replace(/'/g, "&#39;")}','icon_d2')" style="font-size:11px;padding:2px 6px;line-height:1.1;">Skew-T</button>`;
    }
    btns += `<button onclick="openMeteogramAt(${Number(lat).toFixed(5)},${Number(lon).toFixed(5)},'icon_d2')" style="font-size:11px;padding:2px 6px;line-height:1.1;">Meteogram</button>`;
    btns += `</div>`;
    L.popup({ maxWidth: 280 })
      .setLatLng([lat, lon])
      .setContent(lines.join('<br/>') + btns)
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

async function loadD2Border() {
  try {
    const step = timesteps[currentTimeIndex];
    const time = step?.validTime || 'latest';
    const res = await fetch(`/api/d2_domain?time=${encodeURIComponent(time)}`);
    if (!res.ok) await throwHttpError(res, 'API');
    const data = await res.json();
    d2BorderLayer.clearLayers();

    const segments = data?.boundarySegments || [];
    const freshness = data?.diagnostics?.dataFreshnessMinutes;
    const freshText = (freshness != null) ? `${freshness} min` : 'n/a';
    const tip = `ICON-D2 valid-cell border<br/>Run: ${data.run}<br/>Valid: ${data.validTime}<br/>Freshness: ${freshText}`;

    if (segments.length > 0) {
      const line = L.polyline(segments, {
        color: '#ffd84d',
        weight: 2,
        opacity: 0.95,
        interactive: true,
      }).addTo(d2BorderLayer);
      line.bindTooltip(tip, { sticky: true, direction: 'top', opacity: 0.92 });
    } else {
      const b = data?.cellEdgeBbox || data?.bbox;
      if (!b) return;
      const rect = L.rectangle([[b.latMin, b.lonMin], [b.latMax, b.lonMax]], {
        color: '#ffd84d',
        weight: 2,
        opacity: 0.95,
        fill: false,
        interactive: true,
        dashArray: '4,3',
      }).addTo(d2BorderLayer);
      rect.bindTooltip(tip, { sticky: true, direction: 'top', opacity: 0.92 });
    }
    markApiSuccess();
  } catch (e) {
    console.error('Error loading D2 border:', e);
    markApiFailure('d2 border', e);
  }
}

let currentMarker = null;
let markerSuggestions = [];
let selectedSuggestionIdx = -1;
let markerSearchDebounce = null;

async function loadMarkerProfile() {
  try {
    const cid = getClientId();
    const res = await fetch(`/api/marker_profile?clientId=${encodeURIComponent(cid)}`);
    if (!res.ok) await throwHttpError(res, 'API');
    const data = await res.json();
    const m = data.marker;
    if (!m) return;
    currentMarker = m;
    markerEditingEnabled = !!data.markerEditable;

    markerLayer.clearLayers();
    const marker = L.circleMarker([m.lat, m.lon], {
      radius: 7,
      fillColor: '#ff3b30',
      color: '#8b0000',
      weight: 2,
      opacity: 0.95,
      fillOpacity: 0.9,
    }).addTo(markerLayer);
    marker.bindTooltip(m.name || 'Marker', { direction: 'top', opacity: 0.92 });

    const btn = document.getElementById('marker-picker-open');
    if (btn) {
      btn.textContent = `Marker: ${m.name || 'Marker'}`;
      btn.disabled = !markerEditingEnabled;
      btn.title = markerEditingEnabled ? '' : 'Marker editing disabled: server auth not configured';
    }

    const input = document.getElementById('marker-search');
    const confirmBtn = document.getElementById('marker-confirm');
    const locBtn = document.getElementById('marker-use-location');
    if (input) input.disabled = !markerEditingEnabled;
    if (confirmBtn) confirmBtn.disabled = !markerEditingEnabled;
    if (locBtn) locBtn.disabled = !markerEditingEnabled;

    markApiSuccess();
  } catch (e) {
    console.error('Error loading marker profile:', e);
    markApiFailure('marker profile', e);
  }
}

function renderMarkerSuggestions() {
  const box = document.getElementById('marker-suggestions');
  if (!box) return;
  if (!markerSuggestions.length) {
    box.innerHTML = '<div class="marker-suggestion empty">No matches</div>';
    return;
  }
  box.innerHTML = markerSuggestions.map((s, idx) => {
    const active = idx === selectedSuggestionIdx ? ' active' : '';
    const subtitle = s.displayName ? `<div class="sub">${s.displayName}</div>` : '';
    return `<div class="marker-suggestion${active}" data-idx="${idx}"><div>${s.name}</div>${subtitle}</div>`;
  }).join('');

  box.querySelectorAll('.marker-suggestion').forEach(el => {
    el.addEventListener('click', () => {
      selectedSuggestionIdx = Number(el.dataset.idx);
      renderMarkerSuggestions();
    });
  });
}

async function searchMarkerLocations(query) {
  try {
    const res = await fetch(`/api/location_search?q=${encodeURIComponent(query)}&limit=8`);
    if (!res.ok) await throwHttpError(res, 'API');
    const data = await res.json();
    markerSuggestions = data.results || [];
    selectedSuggestionIdx = markerSuggestions.length ? 0 : -1;
    renderMarkerSuggestions();
    markApiSuccess();
  } catch (e) {
    console.error('Location search failed:', e);
    markApiFailure('location search', e);
  }
}

async function setMarkerProfile({ name, note = '', lat, lon }) {
  const token = await ensureMarkerAuthToken();
  const payload = {
    clientId: getClientId(),
    markerAuthToken: token,
    name,
    note,
    lat,
    lon,
  };
  let res = await fetch('/api/marker_profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (res.status === 401) {
    const newTok = await ensureMarkerAuthToken(true);
    payload.markerAuthToken = newTok;
    res = await fetch('/api/marker_profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }
  if (!res.ok) await throwHttpError(res, 'API');
}

async function saveSelectedMarker() {
  if (selectedSuggestionIdx < 0 || !markerSuggestions[selectedSuggestionIdx]) {
    alert('Please select a location first.');
    return;
  }
  const s = markerSuggestions[selectedSuggestionIdx];
  try {
    await setMarkerProfile({ name: s.name, note: s.displayName || '', lat: s.lat, lon: s.lon });
    document.getElementById('marker-picker').style.display = 'none';
    await loadMarkerProfile();
    markApiSuccess();
  } catch (e) {
    console.error('Set marker failed:', e);
    markApiFailure('set marker', e);
  }
}

// Overlay loading — uses FIXED full-extent bbox so overlay never shifts on pan/zoom
// Use viewport bbox for overlays (not full domain) for performance
function getOverlayParams() {
  const b = map.getBounds();
  // No frontend clamp: backend will clip to available model extent.
  const bbox = `${b.getSouth()},${b.getWest()},${b.getNorth()},${b.getEast()}`;
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

document.getElementById('marker-picker-open').addEventListener('click', () => {
  if (!markerEditingEnabled) {
    alert('Marker editing is currently disabled (server auth secret missing). Using default marker Geitau.');
    return;
  }
  const panel = document.getElementById('marker-picker');
  const input = document.getElementById('marker-search');
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  if (panel.style.display === 'block') {
    input.value = '';
    markerSuggestions = [];
    selectedSuggestionIdx = -1;
    renderMarkerSuggestions();
    setTimeout(() => input.focus(), 50);
  }
});

document.getElementById('marker-cancel').addEventListener('click', () => {
  document.getElementById('marker-picker').style.display = 'none';
});

document.getElementById('marker-confirm').addEventListener('click', () => {
  saveSelectedMarker();
});

document.getElementById('marker-use-location').addEventListener('click', () => {
  if (!navigator.geolocation) {
    alert('Geolocation is not supported in this browser.');
    return;
  }
  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      try {
        await setMarkerProfile({
          name: 'My position',
          note: 'Device location',
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
        });
        document.getElementById('marker-picker').style.display = 'none';
        await loadMarkerProfile();
        markApiSuccess();
      } catch (e) {
        console.error('Set marker from location failed:', e);
        markApiFailure('marker local position', e);
      }
    },
    () => alert('Could not get your location. Please allow location permission.'),
    { enableHighAccuracy: true, timeout: 10000 }
  );
});

document.getElementById('marker-search').addEventListener('input', (e) => {
  const q = e.target.value.trim();
  clearTimeout(markerSearchDebounce);
  if (q.length < 2) {
    markerSuggestions = [];
    selectedSuggestionIdx = -1;
    renderMarkerSuggestions();
    return;
  }
  markerSearchDebounce = setTimeout(() => searchMarkerLocations(q), 220);
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
      loadD2Border();
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
  if (emagramState.open && Number.isFinite(emagramState.lat) && Number.isFinite(emagramState.lon)) {
    const step = timesteps[currentTimeIndex] || null;
    openEmagramAt(emagramState.lat, emagramState.lon, step?.validTime || 'latest', step?.model || emagramState.model || '');
  }
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
  loadD2Border();
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
  loadD2Border();
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

const helpBtn = document.getElementById('help-btn');
const helpOverlay = document.getElementById('help-overlay');
const helpClose = document.getElementById('help-close');
const emagramOverlay = document.getElementById('emagram-overlay');
const emagramClose = document.getElementById('emagram-close');
const emagramTitle = document.getElementById('emagram-title');
const emagramBody = document.getElementById('emagram-body');
const meteogramOverlay = document.getElementById('meteogram-overlay');
const meteogramClose = document.getElementById('meteogram-close');
const meteogramTitle = document.getElementById('meteogram-title');
const meteogramBody = document.getElementById('meteogram-body');
let emagramState = { open: false, lat: null, lon: null, model: '', zoom: null, loading: false, reqId: 0, allowedIndices: [] };
let meteogramState = { open: false, lat: null, lon: null, model: '' };
const emagramCache = new Map();
const meteogramCache = new Map();
const EMAGRAM_CACHE_MAX = 96;
const METEOGRAM_CACHE_MAX = 32;
const langSelect = document.getElementById('lang-select');

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

function openHelp() {
  helpOverlay.style.display = 'flex';
}

function closeHelp() {
  helpOverlay.style.display = 'none';
}

function closeEmagram() {
  emagramState.open = false;
  if (emagramOverlay) emagramOverlay.style.display = 'none';
}

function closeMeteogram() {
  meteogramState.open = false;
  if (meteogramOverlay) meteogramOverlay.style.display = 'none';
}

function renderEmagramSvg(levels) {
  const rows = (levels || [])
    .filter(l => Number.isFinite(Number(l.pressureHpa)) && (Number.isFinite(Number(l.temperatureC)) || Number.isFinite(Number(l.dewpointC))))
    .sort((a, b) => Number(b.pressureHpa) - Number(a.pressureHpa));

  if (!rows.length) return '<div style="color:#ffb3b3">No profile data available for this point/time.</div>';

  const W = 620, H = 388;
  const m = { l: 112, r: 120, t: 40, b: 36 };
  const plotRight = W - m.r;
  const axisAlt = m.l - 54;
  const iw = plotRight - m.l;
  const ih = H - m.t - m.b;
  const pTop = 200;
  const pBot = 1000;
  const skew = 0.26;

  const temps = rows.flatMap(r => [Number(r.temperatureC), Number(r.dewpointC)]).filter(Number.isFinite);
  let tMin = Math.floor((Math.min(...temps) - 8) / 5) * 5;
  let tMax = Math.ceil((Math.max(...temps) + 8) / 5) * 5;
  if (!(Number.isFinite(tMin) && Number.isFinite(tMax)) || tMax <= tMin) { tMin = -40; tMax = 30; }

  const y = (p) => m.t + ((Math.log(p) - Math.log(pTop)) / (Math.log(pBot) - Math.log(pTop))) * ih;
  const baseX = (t) => m.l + ((t - tMin) / (tMax - tMin)) * iw;
  const x = (t, p) => baseX(t) + (H - m.b - y(p)) * skew;

  const pressureLines = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200].filter(p => p <= pBot && p >= pTop);
  const rowsByP = rows
    .filter(r => Number.isFinite(Number(r.pressureHpa)) && Number.isFinite(Number(r.altitudeM)))
    .map(r => ({ p: Number(r.pressureHpa), z: Number(r.altitudeM) }))
    .sort((a, b) => b.p - a.p);
  const altitudeAtP = (p) => {
    if (!rowsByP.length) return null;
    for (let i = 0; i < rowsByP.length - 1; i++) {
      const a = rowsByP[i], b = rowsByP[i + 1];
      if ((a.p >= p && p >= b.p) || (a.p <= p && p <= b.p)) {
        const t = (p - a.p) / (b.p - a.p || 1);
        return a.z + t * (b.z - a.z);
      }
    }
    return rowsByP.reduce((best, r) => (Math.abs(r.p - p) < Math.abs(best.p - p) ? r : best), rowsByP[0]).z;
  };

  let grid = '';
  let tempGrid = '';
  for (const p of pressureLines) {
    const yy = y(p);
    grid += `<line x1="${m.l}" y1="${yy}" x2="${plotRight}" y2="${yy}" stroke="rgba(255,255,255,0.14)"/>`;
    // Pressure labels on the plot border axis
    grid += `<text x="${m.l - 6}" y="${yy + 4}" fill="rgba(255,255,255,0.80)" font-size="11" text-anchor="end">${p} hPa</text>`;
    // Altitude axis tick + label on dedicated left line
    grid += `<line x1="${axisAlt - 4}" y1="${yy}" x2="${axisAlt}" y2="${yy}" stroke="rgba(255,255,255,0.65)"/>`;
    const alt = altitudeAtP(p);
    if (Number.isFinite(alt)) {
      grid += `<text x="${axisAlt - 8}" y="${yy + 4}" fill="rgba(255,255,255,0.80)" font-size="10" text-anchor="end">${Math.round(alt)}m</text>`;
    }
  }
  for (let t = tMin; t <= tMax; t += 10) {
    const x1 = x(t, pBot), x2 = x(t, pTop);
    tempGrid += `<line x1="${x1}" y1="${y(pBot)}" x2="${x2}" y2="${y(pTop)}" stroke="rgba(255,255,255,0.12)"/>`;
    grid += `<text x="${x1}" y="${H - 21}" fill="rgba(255,255,255,0.75)" font-size="11" text-anchor="middle">${t}</text>`;
  }

  const kappa = 0.286;
  const dryThetas = [260, 270, 280, 290, 300, 310, 320, 330, 340];
  let dryGrid = '';
  for (const th of dryThetas) {
    const pts = [];
    for (let p = pBot; p >= pTop; p -= 25) {
      const tK = th * Math.pow(p / 1000.0, kappa);
      const tC = tK - 273.15;
      if (tC < tMin - 20 || tC > tMax + 20) continue;
      pts.push(`${x(tC, p).toFixed(1)},${y(p).toFixed(1)}`);
    }
    if (pts.length > 1) dryGrid += `<path d="M${pts.join(' L')}" fill="none" stroke="rgba(255,170,70,0.20)" stroke-width="1"/>`;
  }

  const moistStarts = [-10, 0, 10, 20, 30];
  let moistGrid = '';
  for (const t0 of moistStarts) {
    const pts = [];
    for (let p = pBot; p >= pTop; p -= 25) {
      const frac = (1000 - p) / 800;
      const tC = t0 - (24 * frac) - (8 * frac * frac);
      pts.push(`${x(tC, p).toFixed(1)},${y(p).toFixed(1)}`);
    }
    if (pts.length > 1) moistGrid += `<path d="M${pts.join(' L')}" fill="none" stroke="rgba(90,200,255,0.20)" stroke-width="1"/>`;
  }

  const toPath = (key) => rows
    .filter(r => Number.isFinite(Number(r[key])) && Number.isFinite(Number(r.pressureHpa)))
    .map((r, i) => `${i ? 'L' : 'M'}${x(Number(r[key]), Number(r.pressureHpa)).toFixed(1)},${y(Number(r.pressureHpa)).toFixed(1)}`)
    .join(' ');

  const tPath = toPath('temperatureC');
  const tdPath = toPath('dewpointC');

  const mkBarb = (xx, yy, speedKt = 0, dirDeg = 0) => {
    if (!(Number.isFinite(speedKt) && Number.isFinite(dirDeg))) return '';
    let s = Math.max(0, Math.round(speedKt / 5) * 5);
    const flags50 = Math.floor(s / 50); s -= flags50 * 50;
    const flags10 = Math.floor(s / 10); s -= flags10 * 10;
    const flags5 = s >= 5 ? 1 : 0;
    let g = `<g transform="translate(${xx},${yy}) rotate(${dirDeg})">`;
    g += `<line x1="0" y1="0" x2="0" y2="-22" stroke="#d1d5db" stroke-width="1.6"/>`;
    let yyf = -22;
    for (let i = 0; i < flags50; i++) { g += `<polygon points="0,${yyf} 11,${yyf+3} 0,${yyf+6}" fill="#d1d5db"/>`; yyf += 6; }
    for (let i = 0; i < flags10; i++) { g += `<line x1="0" y1="${yyf}" x2="11" y2="${yyf+3}" stroke="#d1d5db" stroke-width="1.6"/>`; yyf += 4; }
    if (flags5) g += `<line x1="0" y1="${yyf}" x2="7" y2="${yyf+2}" stroke="#d1d5db" stroke-width="1.6"/>`;
    g += `</g>`;
    return g;
  };

  const barbX = plotRight + 46;
  const barbs = rows
    .filter(r => Number.isFinite(Number(r.windSpeedKt)) && Number.isFinite(Number(r.windDirDeg)))
    .map(r => mkBarb(barbX, y(Number(r.pressureHpa)), Number(r.windSpeedKt), Number(r.windDirDeg)))
    .join('');

  return `
    <svg width="100%" viewBox="0 0 ${W} ${H}" role="img" aria-label="Skew-T profile">
      <rect x="0" y="0" width="${W}" height="${H}" fill="#151b2d" rx="8"/>
      ${grid}
      <defs>
        <clipPath id="skewPlotClip"><rect x="${m.l}" y="${m.t}" width="${plotRight - m.l}" height="${H - m.t - m.b}" /></clipPath>
      </defs>
      <line x1="${axisAlt}" y1="${m.t}" x2="${axisAlt}" y2="${H - m.b}" stroke="rgba(255,255,255,0.55)"/>
      <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="rgba(255,255,255,0.50)"/>
      <line x1="${m.l}" y1="${H - m.b}" x2="${plotRight}" y2="${H - m.b}" stroke="rgba(255,255,255,0.5)"/>
      <g clip-path="url(#skewPlotClip)">
        ${tempGrid}
        ${dryGrid}
        ${moistGrid}
        ${tPath ? `<path d="${tPath}" fill="none" stroke="#ff6b6b" stroke-width="2.2"/>` : ''}
        ${tdPath ? `<path d="${tdPath}" fill="none" stroke="#69b1ff" stroke-width="2.2"/>` : ''}
      </g>
      ${barbs}
      <text x="${(m.l + plotRight)/2}" y="${H-2}" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-size="11">Skewed temperature lines (°C)</text>
      <text x="${m.l-40}" y="${m.t-26}" text-anchor="start" fill="rgba(255,255,255,0.9)" font-size="10">Pressure</text>
      <text x="${axisAlt-33}" y="${m.t-26}" text-anchor="start" fill="rgba(255,255,255,0.9)" font-size="10">Altitude</text>
      <text x="${barbX}" y="${m.t-26}" text-anchor="middle" fill="rgba(255,255,255,0.85)" font-size="10">Wind</text>
    </svg>
    <div style="display:flex;gap:14px;font-size:12px;margin-top:6px;align-items:center;flex-wrap:wrap">
      <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;height:2px;background:#ff6b6b;display:inline-block"></span>T</span>
      <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;height:2px;background:#69b1ff;display:inline-block"></span>Td</span>
      <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;height:2px;background:rgba(255,170,70,0.65);display:inline-block"></span>dry adiabat</span>
      <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;height:2px;background:rgba(90,200,255,0.65);display:inline-block"></span>moist adiabat (approx)</span>
      <span style="opacity:.75">barbs in kt</span>
    </div>`;
}

function emagramCurrentStep() {
  return timesteps[currentTimeIndex] || null;
}

function emagramAllowedIndices() {
  const idx = [];
  for (let i = 0; i < timesteps.length; i++) {
    const m = String(timesteps[i]?.model || '').replace('-', '_');
    if (m === 'icon_d2') idx.push(i);
  }
  return idx;
}

function emagramNeighborIndex(delta) {
  const allowed = emagramState.allowedIndices?.length ? emagramState.allowedIndices : emagramAllowedIndices();
  if (!allowed.length) return currentTimeIndex;
  let pos = allowed.indexOf(currentTimeIndex);
  if (pos < 0) {
    // Snap to nearest available index
    pos = 0;
    let best = Infinity;
    for (let i = 0; i < allowed.length; i++) {
      const d = Math.abs(allowed[i] - currentTimeIndex);
      if (d < best) { best = d; pos = i; }
    }
  }
  const nextPos = Math.max(0, Math.min(allowed.length - 1, pos + delta));
  return allowed[nextPos];
}

function meteogramCacheSet(key, value) {
  if (meteogramCache.has(key)) meteogramCache.delete(key);
  meteogramCache.set(key, value);
  if (meteogramCache.size > METEOGRAM_CACHE_MAX) {
    const oldest = meteogramCache.keys().next().value;
    meteogramCache.delete(oldest);
  }
}

function renderMeteogramSvg(series) {
  const rows = (series || []).filter(r => r && r.validTime);
  if (!rows.length) return '<div style="color:#ffb3b3">No meteogram data.</div>';

  const W = 760, H = 520;
  const m = { l: 48, r: 40, t: 18, b: 44 };
  const panels = [
    { key: 'wind', h: 175 },
    { key: 'precip', h: 110 },
    { key: 'temp', h: 85 },
  ];
  const panelGap = 15;
  const totalH = panels.reduce((a, p) => a + p.h, 0);
  const scaleY = (H - m.t - m.b - panelGap * (panels.length - 1)) / totalH;
  const pxW = W - m.l - m.r;

  let y0 = m.t;
  for (let i = 0; i < panels.length; i++) {
    const p = panels[i];
    p.y = y0;
    p.ph = p.h * scaleY;
    y0 += p.ph + (i < panels.length - 1 ? panelGap : 0);
  }

  const x = (idx) => m.l + (rows.length <= 1 ? 0 : (idx / (rows.length - 1)) * pxW);
  const v = (row, key) => Number(row[key]);

  const linePath = (key, panel, min, max) => rows.map((r, i) => {
    const vv = v(r, key);
    if (!Number.isFinite(vv)) return null;
    const yy = panel.y + panel.ph - ((vv - min) / (max - min || 1)) * panel.ph;
    return `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${yy.toFixed(1)}`;
  }).filter(Boolean).join(' ');

  const precipVals = rows.map(r => v(r, 'precipRateTotal')).filter(Number.isFinite);
  const precipMax = Math.max(1, ...(precipVals.length ? precipVals : [0]));

  const temps = rows.flatMap(r => [v(r, 't2mC'), v(r, 'dewpoint2mC')]).filter(Number.isFinite);
  let tMin = Math.floor((Math.min(...(temps.length ? temps : [-10])) - 2) / 5) * 5;
  let tMax = Math.ceil((Math.max(...(temps.length ? temps : [20])) + 2) / 5) * 5;
  if (!(Number.isFinite(tMin) && Number.isFinite(tMax)) || tMax <= tMin) { tMin = -20; tMax = 30; }

  const snowVals = rows.map(r => v(r, 'snowDepthM')).filter(Number.isFinite);
  const snowMax = Math.max(0, ...(snowVals.length ? snowVals : [0]));
  const hasSnow = snowMax > 0;

  const pWind = panels[0], pPre = panels[1], pTemp = panels[2];

  let svg = `<svg width="100%" viewBox="0 0 ${W} ${H}" role="img" aria-label="Meteogram">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#151b2d" rx="8"/>`;

  for (const p of panels) {
    svg += `<line x1="${m.l}" y1="${p.y.toFixed(1)}" x2="${W - m.r}" y2="${p.y.toFixed(1)}" stroke="rgba(255,255,255,0.2)"/>`;
    svg += `<line x1="${m.l}" y1="${(p.y + p.ph).toFixed(1)}" x2="${W - m.r}" y2="${(p.y + p.ph).toFixed(1)}" stroke="rgba(255,255,255,0.2)"/>`;
  }

  const wd = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  for (let i = 0; i < rows.length; i++) {
    const xx = x(i);
    const dt = new Date(rows[i].validTime);
    if (Number.isNaN(dt.getTime())) continue;
    const hh = dt.getUTCHours();
    const dd = String(dt.getUTCDate()).padStart(2,'0');
    const mm = String(dt.getUTCMonth()+1).padStart(2,'0');
    const day = wd[dt.getUTCDay()];
    const isMajorHour = (hh % 6 === 0);
    const isMidnight = hh === 0;
    if (isMajorHour) {
      const strokeW = isMidnight ? 1.8 : 0.8;
      svg += `<line x1="${xx}" y1="${m.t}" x2="${xx}" y2="${H - m.b}" stroke="rgba(255,255,255,0.20)" stroke-width="${strokeW}"/>`;
    }
    const tickLen = isMidnight ? 11 : (isMajorHour ? 8 : 4);
    svg += `<line x1="${xx}" y1="${H - m.b}" x2="${xx}" y2="${H - m.b + tickLen}" stroke="rgba(255,255,255,0.55)"/>`;
    if (isMajorHour || i === 0 || i === rows.length - 1) {
      svg += `<text x="${xx}" y="${H - m.b + 30}" fill="rgba(255,255,255,0.72)" font-size="10" text-anchor="middle">${String(hh).padStart(2,'0')}Z</text>`;
    }
    if (hh === 12) {
      svg += `<text x="${xx}" y="${H - m.b + 18}" fill="rgba(255,255,255,0.78)" font-size="10" text-anchor="middle">${day} ${dd}.${mm}.</text>`;
    }
  }

  // Wind panel: small barbs for each timestep x pressure level
  const levels = [1000, 975, 950, 850, 700, 600, 500, 400, 300, 200];
  const yWind = (lev) => pWind.y + ((lev - 200) / (1000 - 200)) * pWind.ph;
  const mkBarb = (xx, yy, speedKt = 0, dirDeg = 0) => {
    if (!(Number.isFinite(speedKt) && Number.isFinite(dirDeg))) return '';
    let s = Math.max(0, Math.round(speedKt / 5) * 5);
    const f50 = Math.floor(s / 50); s -= f50 * 50;
    const f10 = Math.floor(s / 10); s -= f10 * 10;
    const f5 = s >= 5 ? 1 : 0;
    let g = `<g transform="translate(${xx},${yy}) rotate(${dirDeg})">`;
    g += `<line x1="0" y1="0" x2="0" y2="-9" stroke="#dbeafe" stroke-width="1.1"/>`;
    let yyf = -9;
    for (let i = 0; i < f50; i++) { g += `<polygon points="0,${yyf} 5,${yyf+1.4} 0,${yyf+2.8}" fill="#dbeafe"/>`; yyf += 3; }
    for (let i = 0; i < f10; i++) { g += `<line x1="0" y1="${yyf}" x2="5" y2="${yyf+1.4}" stroke="#dbeafe" stroke-width="1.1"/>`; yyf += 2.2; }
    if (f5) g += `<line x1="0" y1="${yyf}" x2="3.2" y2="${yyf+0.9}" stroke="#dbeafe" stroke-width="1.1"/>`;
    g += `</g>`;
    return g;
  };
  for (let i = 0; i < rows.length; i++) {
    const xx = x(i);
    const wl = rows[i].windLevels || [];
    const byLev = {};
    for (const w of wl) byLev[Number(w.pressureHpa)] = w;
    for (const lev of levels) {
      const w = byLev[lev];
      if (!w) continue;
      svg += mkBarb(xx, yWind(lev), Number(w.speedKt), Number(w.dirDeg));
    }
  }

  // Approx altitude reference lines in wind panel (reversed orientation)
  const altRefs = [
    { z: 0, p: 1000 },
    { z: 3000, p: 700 },
    { z: 5000, p: 540 },
    { z: 10000, p: 260 },
  ];
  for (const a of altRefs) {
    const yy = yWind(a.p);
    svg += `<line x1="${m.l}" y1="${yy.toFixed(1)}" x2="${W - m.r}" y2="${yy.toFixed(1)}" stroke="rgba(255,255,255,0.20)" stroke-dasharray="3 3"/>`;
    svg += `<text x="6" y="${(yy - 3).toFixed(1)}" fill="rgba(255,255,255,0.62)" font-size="9" text-anchor="start">~${a.z}m</text>`;
  }

  for (let i = 0; i < rows.length; i++) {
    const xx = x(i);
    const bw = Math.max(2, pxW / Math.max(rows.length, 24));
    const pr = Math.max(0, v(rows[i], 'precipRateTotal') || 0);
    const hP = (pr / precipMax) * pPre.ph;
    const yBase = pPre.y + pPre.ph;
    if (hP > 0) svg += `<rect x="${(xx - bw/2).toFixed(1)}" y="${(yBase - hP).toFixed(1)}" width="${bw.toFixed(1)}" height="${hP.toFixed(1)}" fill="rgba(80,170,255,0.82)"/>`;
  }

  let snowUnit = 'm';
  let snowTopLabel = '0.00';
  if (hasSnow) {
    const snowUseCm = snowMax < 1.0;
    snowUnit = snowUseCm ? 'cm' : 'm';
    snowTopLabel = snowUseCm ? (snowMax * 100).toFixed(0) : snowMax.toFixed(2);
    const snowPath = linePath('snowDepthM', pPre, 0, snowMax);
    if (snowPath) svg += `<path d="${snowPath}" fill="none" stroke="#ffffff" stroke-width="1.4"/>`;
  }

  const tPath = linePath('t2mC', pTemp, tMin, tMax);
  if (tPath) svg += `<path d="${tPath}" fill="none" stroke="#ff9a66" stroke-width="1.8"/>`;
  const tdPath = linePath('dewpoint2mC', pTemp, tMin, tMax);
  if (tdPath) svg += `<path d="${tdPath}" fill="none" stroke="#7cc3ff" stroke-width="1.6"/>`;
  const yZero = pTemp.y + pTemp.ph - ((0 - tMin) / (tMax - tMin || 1)) * pTemp.ph;
  if (Number.isFinite(yZero) && yZero >= pTemp.y && yZero <= pTemp.y + pTemp.ph) {
    svg += `<line x1="${m.l}" y1="${yZero.toFixed(1)}" x2="${W - m.r}" y2="${yZero.toFixed(1)}" stroke="rgba(255,255,255,0.35)" stroke-dasharray="4 3"/>`;
  }

  svg += `<text x="6" y="${(pWind.y + 0).toFixed(1)}" fill="rgba(255,255,255,0.82)" font-size="10">Wind</text>`;
  svg += `<text x="6" y="${(pPre.y + 24).toFixed(1)}" fill="rgba(255,255,255,0.82)" font-size="10">Precip</text>`;
  if (hasSnow) svg += `<text x="${W - m.r + 8}" y="${(pPre.y + 24).toFixed(1)}" fill="rgba(255,255,255,0.82)" font-size="10" text-anchor="end">Snow</text>`;
  svg += `<text x="6" y="${(pTemp.y + 24).toFixed(1)}" fill="rgba(255,255,255,0.82)" font-size="10">Temp</text>`;

  // Precip axis (left): 0 .. precipMax
  svg += `<line x1="${m.l}" y1="${(pPre.y + pPre.ph).toFixed(1)}" x2="${(m.l-4)}" y2="${(pPre.y + pPre.ph).toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
  svg += `<line x1="${m.l}" y1="${pPre.y.toFixed(1)}" x2="${(m.l-4)}" y2="${pPre.y.toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
  svg += `<text x="${m.l-6}" y="${(pPre.y + pPre.ph + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="end">0 mm/h</text>`;
  svg += `<text x="${m.l-6}" y="${(pPre.y + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="end">${precipMax.toFixed(1)} mm/h</text>`;

  // Snow axis (right): 0 .. snowMax
  if (hasSnow) {
    svg += `<line x1="${W - m.r}" y1="${(pPre.y + pPre.ph).toFixed(1)}" x2="${(W - m.r + 4)}" y2="${(pPre.y + pPre.ph).toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
    svg += `<line x1="${W - m.r}" y1="${pPre.y.toFixed(1)}" x2="${(W - m.r + 4)}" y2="${pPre.y.toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
    svg += `<text x="${W - m.r + 6}" y="${(pPre.y + pPre.ph + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="start">0 ${snowUnit}</text>`;
    svg += `<text x="${W - m.r + 6}" y="${(pPre.y + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="start">${snowTopLabel} ${snowUnit}</text>`;
  }

  // Temp axis (left): tMin .. tMax
  svg += `<line x1="${m.l}" y1="${(pTemp.y + pTemp.ph).toFixed(1)}" x2="${(m.l-4)}" y2="${(pTemp.y + pTemp.ph).toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
  svg += `<line x1="${m.l}" y1="${pTemp.y.toFixed(1)}" x2="${(m.l-4)}" y2="${pTemp.y.toFixed(1)}" stroke="rgba(255,255,255,0.65)"/>`;
  svg += `<text x="${m.l-6}" y="${(pTemp.y + pTemp.ph + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="end">${tMin.toFixed(0)} °C</text>`;
  svg += `<text x="${m.l-6}" y="${(pTemp.y + 4).toFixed(1)}" fill="rgba(255,255,255,0.78)" font-size="9" text-anchor="end">${tMax.toFixed(0)} °C</text>`;

  svg += `</svg>`;
  return svg;
}

async function openMeteogramAt(lat, lon, model = 'icon_d2') {
  if (!meteogramOverlay || !meteogramBody) return;
  const key = `${Number(lat).toFixed(4)}|${Number(lon).toFixed(4)}|${model || ''}`;
  meteogramState = { open: true, lat: Number(lat), lon: Number(lon), model: model || '' };
  meteogramOverlay.style.display = 'flex';
  meteogramBody.innerHTML = '<div style="opacity:.8">Loading meteogram…</div>';
  try {
    let data = meteogramCache.get(key);
    if (!data) {
      const modelPart = model ? `&model=${encodeURIComponent(model)}` : '';
      const res = await fetch(`/api/meteogram_point?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}${modelPart}`);
      if (!res.ok) await throwHttpError(res, 'API');
      data = await res.json();
      meteogramCacheSet(key, data);
    }
    const p = data.point || {};
    const modelUsed = data.series?.[0]?.model || (model || 'icon_d2');
    const runUsed = data.series?.[0]?.run || '-';
    const glat = (p.gridLat ?? lat);
    const glon = (p.gridLon ?? lon);
    if (meteogramTitle) meteogramTitle.textContent = `Meteogram · ${glat}, ${glon} · ${String(modelUsed).toUpperCase().replace('_','-')} · Run ${runUsed}`;
    meteogramBody.innerHTML = renderMeteogramSvg(data.series || []);
  } catch (e) {
    meteogramBody.innerHTML = `<div style="color:#ff9f9f">Failed to load meteogram: ${String(e.message || e)}</div>`;
  }
}

window.openMeteogramAt = openMeteogramAt;


function emagramSetLoadingUI(isLoading) {
  const prevBtn = document.getElementById('emNavPrev');
  const nextBtn = document.getElementById('emNavNext');
  const st = document.getElementById('emNavStatus');
  const allowed = emagramState.allowedIndices?.length ? emagramState.allowedIndices : emagramAllowedIndices();
  const atStart = !allowed.length || currentTimeIndex === allowed[0];
  const atEnd = !allowed.length || currentTimeIndex === allowed[allowed.length - 1];
  if (prevBtn) {
    prevBtn.disabled = isLoading || atStart;
    prevBtn.style.opacity = (isLoading || atStart) ? '0.45' : '1';
    prevBtn.style.cursor = (isLoading || atStart) ? 'default' : 'pointer';
  }
  if (nextBtn) {
    nextBtn.disabled = isLoading || atEnd;
    nextBtn.style.opacity = (isLoading || atEnd) ? '0.45' : '1';
    nextBtn.style.cursor = (isLoading || atEnd) ? 'default' : 'pointer';
  }
  if (st) st.textContent = isLoading ? 'loading…' : '';
}

function emagramCacheKey({ lat, lon, model, zoom, time }) {
  return `${Number(lat).toFixed(4)}|${Number(lon).toFixed(4)}|${model || ''}|z${zoom ?? ''}|${time || 'latest'}`;
}

function emagramCacheSet(key, value) {
  if (emagramCache.has(key)) emagramCache.delete(key);
  emagramCache.set(key, value);
  if (emagramCache.size > EMAGRAM_CACHE_MAX) {
    const oldest = emagramCache.keys().next().value;
    emagramCache.delete(oldest);
  }
}

function emagramNav(delta) {
  const nextIdx = emagramNeighborIndex(delta);
  if (nextIdx === currentTimeIndex || emagramState.loading) return;
  currentTimeIndex = nextIdx;
  emagramState.loading = true;
  emagramSetLoadingUI(true);
  // In emagram mode keep map requests stable; only refresh sounding for step changes.
  updateTimeline();
}

async function openEmagramAt(lat, lon, time = 'latest', model = '') {
  if (!emagramOverlay || !emagramBody) return;
  const fixedModel = 'icon_d2';
  const fixedZoom = (emagramState.zoom != null) ? emagramState.zoom : map.getZoom();
  const allowedIndices = emagramAllowedIndices();
  if (allowedIndices.length && !allowedIndices.includes(currentTimeIndex)) {
    currentTimeIndex = emagramNeighborIndex(0);
  }
  const hadContent = emagramState.open && emagramBody.innerHTML.trim().length > 0;
  const reqId = (emagramState.reqId || 0) + 1;
  emagramState = { open: true, lat: Number(lat), lon: Number(lon), model: fixedModel, zoom: fixedZoom, loading: true, reqId, allowedIndices };
  emagramOverlay.style.display = 'flex';
  if (!hadContent) emagramBody.innerHTML = '<div style="opacity:.8">Loading profile…</div>';
  else emagramSetLoadingUI(true);
  try {
    const cacheKey = emagramCacheKey({ lat, lon, model: fixedModel, zoom: fixedZoom, time });
    let d = emagramCache.get(cacheKey);
    if (!d) {
      const modelPart = fixedModel ? `&model=${encodeURIComponent(fixedModel)}` : '';
      const res = await fetch(`/api/emagram_point?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&time=${encodeURIComponent(time || 'latest')}${modelPart}`);
      if (!res.ok) await throwHttpError(res, 'API');
      d = await res.json();
      emagramCacheSet(cacheKey, d);
    }
    if (reqId !== emagramState.reqId) return;
    const p = d.point || {};
    const step = emagramCurrentStep();
    const allowed = emagramState.allowedIndices?.length ? emagramState.allowedIndices : emagramAllowedIndices();
    const atStart = !allowed.length || currentTimeIndex === allowed[0];
    const atEnd = !allowed.length || currentTimeIndex === allowed[allowed.length - 1];
    if (emagramTitle) emagramTitle.textContent = `Emagram · ${d.validTime || d.run || ''}`;
    const nav = `<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
      <button id="emNavPrev" onclick="emagramNav(-1)" ${atStart ? 'disabled' : ''} style="font-size:12px;padding:2px 8px;${atStart ? 'opacity:.45;cursor:default;' : ''}">◀</button>
      <div style="font-size:12px;opacity:.85;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${step?.validTime || d.validTime || time || 'latest'}</div>
      <span id="emNavStatus" style="font-size:11px;opacity:.75;min-width:62px;text-align:center"></span>
      <button id="emNavNext" onclick="emagramNav(1)" ${atEnd ? 'disabled' : ''} style="font-size:12px;padding:2px 8px;${atEnd ? 'opacity:.45;cursor:default;' : ''}">▶</button>
    </div>`;
    const meta = `<div style="font-size:12px;opacity:.88;margin-bottom:8px">Point ${p.gridLat ?? lat}, ${p.gridLon ?? lon} · ${d.model || ''} · run ${d.run || ''} step ${d.step ?? ''}</div>`;
    emagramBody.innerHTML = nav + meta + renderEmagramSvg(d.levels || []);
    emagramState.loading = false;
    emagramSetLoadingUI(false);
  } catch (e) {
    if (!hadContent) emagramBody.innerHTML = `<div style="color:#ff9f9f">Failed to load emagram: ${String(e.message || e)}</div>`;
    else {
      const st = document.getElementById('emNavStatus');
      if (st) st.textContent = 'error';
    }
    emagramState.loading = false;
    emagramSetLoadingUI(false);
  }
}

window.openEmagramAt = openEmagramAt;
window.emagramNav = emagramNav;

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

if (helpBtn) helpBtn.addEventListener('click', openHelp);
if (helpClose) helpClose.addEventListener('click', closeHelp);
if (helpOverlay) {
  helpOverlay.addEventListener('click', (e) => {
    if (e.target === helpOverlay) closeHelp();
  });
}
if (emagramClose) emagramClose.addEventListener('click', closeEmagram);
if (emagramOverlay) {
  emagramOverlay.addEventListener('click', (e) => {
    if (e.target === emagramOverlay) closeEmagram();
  });
}
if (meteogramClose) meteogramClose.addEventListener('click', closeMeteogram);
if (meteogramOverlay) {
  meteogramOverlay.addEventListener('click', (e) => {
    if (e.target === meteogramOverlay) closeMeteogram();
  });
}
if (langSelect) {
  langSelect.addEventListener('change', (e) => applyLocale(e.target.value));
}

// Initial localization + load
applyLocale(currentLang);
loadTimesteps();
loadMarkerProfile();

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
      loadOverlay();
      loadWind();
      loadD2Border();

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
      loadOverlay();
      loadD2Border();
    }
  } catch (e) {
    // Silently ignore polling errors
  }
}, 60000);
