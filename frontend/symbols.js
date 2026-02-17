// symbols.js — Weather symbols for Skyview
// WMO ww symbols: from weewx-DWD / weathericons (Public Domain, colored)
// Cloud types: hand-drawn SVG (black #333)

const SYM_SIZE = 36;
const LABEL_OFFSET = 14;

// ─── WMO ww code → symbol file mapping ───
// Keep this list trimmed to codes actually used by backend/weather_codes.py.
const ICON_D2_WW = [0,1,2,3,45,48,51,53,55,56,57,61,63,65,66,67,71,73,75,77,80,81,82,85,86,95,96];
const WW_SYMBOLS = Object.fromEntries(
  ICON_D2_WW.map((ww) => [ww, `/geodata/ww-symbols/wmo4677_ww${String(ww).padStart(2, '0')}.svg`])
);

// ww code display names
const WW_NAMES = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Rime fog',
  51: 'Drizzle (light)', 53: 'Drizzle (mod)', 55: 'Drizzle (dense)',
  56: 'Freezing drizzle (light)', 57: 'Freezing drizzle (heavy)',
  61: 'Rain (slight)', 63: 'Rain (moderate)', 65: 'Rain (heavy)',
  66: 'Freezing rain (light)', 67: 'Freezing rain (heavy)',
  71: 'Snow (slight)', 73: 'Snow (moderate)', 75: 'Snow (heavy)',
  77: 'Snow grains',
  80: 'Rain shower (slight)', 81: 'Rain shower (mod)', 82: 'Rain shower (violent)',
  85: 'Snow shower (slight)', 86: 'Snow shower (heavy)',
  95: 'Thunderstorm', 96: 'Thunderstorm with hail',
};

// ─── Cloud type symbols (hand-drawn, used when ww 0-3) ───
const CLOUD_SYMBOLS = {
  st: {
    svg: `<line x1="3" y1="18" x2="33" y2="18" stroke="#333" stroke-width="3" stroke-linecap="round"/>`,
    label: 'Stratus'
  },
  ac: {
    svg: `<path d="M4 14 Q4 26 11 26 Q18 26 18 14 Q18 26 25 26 Q32 26 32 14" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>`,
    label: 'Altocumulus'
  },
  ci: {
    svg: `<path d="M 6,25 h 20 A 5,5 0 0 0 26,15" fill="none" stroke="#333" stroke-width="2.6" stroke-linecap="round"/>`,
    label: 'Cirrus'
  },
  cu_hum: {
    // Reference style: simple semicircle with baseline
    svg: `<path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>`,
    label: 'Cu humilis'
  },
  cu_con: {
    // Reference style: developed cumulus with upper dome and vertical core
    svg: `<path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <path d="M11 19 Q18 0 25 19" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="18" y1="10" x2="18" y2="23" stroke="#333" stroke-width="3" stroke-linecap="round"/>`,
    label: 'Cu congestus'
  },
  cb: {
    // Reference style: dome with trapezoid anvil top
    svg: `<path d="M6 26 Q18 7 30 26" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="12" y1="17" x2="9" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="24" y1="17" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <line x1="9" y1="10" x2="27" y2="10" stroke="#333" stroke-width="3" stroke-linecap="round"/>`,
    label: 'Cb'
  },
  blue_thermal: {
    svg: `<text x="18" y="26" text-anchor="middle" font-size="22" font-weight="bold" fill="#333">b</text>`,
    label: 'Blue thermal'
  },
};

// ─── Preload cache for WMO SVG icons ───
const _svgCache = {};

async function _preloadWwSvg(ww) {
  if (_svgCache[ww] !== undefined) return _svgCache[ww];
  const url = WW_SYMBOLS[ww];
  if (!url) return null;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const text = await res.text();
    _svgCache[ww] = text;
    return text;
  } catch (e) {
    return null;
  }
}

// Preload all configured ww codes on startup
ICON_D2_WW.forEach(ww => _preloadWwSvg(ww));

/**
 * Create a Leaflet DivIcon for a weather symbol.
 * For cloud types (st, ac, cu_hum, etc.): uses inline SVG.
 * For ww codes: uses the WMO SVG file as an <img>.
 */
function createSymbolIcon(type, cloudBase) {
  const hasLabel = cloudBase != null && cloudBase > 0;
  const totalHeight = hasLabel ? SYM_SIZE + LABEL_OFFSET : SYM_SIZE;

  // Clear sky: blank but clickable hit area
  if (type === 'clear') {
    return L.divIcon({
      // Very large transparent hit area to avoid unclickable gaps in the grid.
      html: `<div style="width:36px;height:36px;background:transparent;"></div>`,
      className: 'weather-symbol clear-symbol',
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });
  }

  // Check if it's a cloud type symbol
  const cloud = CLOUD_SYMBOLS[type];
  if (cloud) {
    let html = `<svg xmlns="http://www.w3.org/2000/svg" width="${SYM_SIZE}" height="${totalHeight}" viewBox="0 0 ${SYM_SIZE} ${totalHeight}">`;
    html += cloud.svg;
    if (hasLabel) {
      html += `<text x="18" y="${SYM_SIZE + 11}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#333" font-weight="bold">${cloudBase}</text>`;
    }
    html += '</svg>';
    return L.divIcon({
      html: html,
      className: 'weather-symbol',
      iconSize: [SYM_SIZE, totalHeight],
      // Georeference to symbol center (ignore lower label extension)
      iconAnchor: [SYM_SIZE / 2, SYM_SIZE / 2],
    });
  }

  // It's a ww-based type — look up the ww code from the type name
  const wwCode = _typeToWw(type);
  if (wwCode !== null && WW_SYMBOLS[wwCode]) {
    let html = `<img src="${WW_SYMBOLS[wwCode]}" width="${SYM_SIZE}" height="${SYM_SIZE}" style="display:block;">`;
    if (hasLabel) {
      html += `<div style="text-align:center;font-size:11px;font-weight:bold;color:#333;font-family:sans-serif;margin-top:-2px;">${cloudBase}</div>`;
    }
    return L.divIcon({
      html: html,
      className: 'weather-symbol',
      iconSize: [SYM_SIZE, totalHeight],
      // Georeference to symbol center (ignore lower label extension)
      iconAnchor: [SYM_SIZE / 2, SYM_SIZE / 2],
    });
  }

  // Fallback: empty
  return L.divIcon({ html: '', className: 'weather-symbol', iconSize: [0, 0] });
}

// Map symbol type names back to ww codes
function _typeToWw(type) {
  // Keep aligned with backend/weather_codes.py ww_to_symbol mapping.
  const map = {
    'fog': 45, 'rime_fog': 48,
    'drizzle_light': 51, 'drizzle_moderate': 53, 'drizzle_dense': 55,
    'freezing_drizzle': 56, 'freezing_drizzle_heavy': 57,
    'rain_slight': 61, 'rain_moderate': 63, 'rain_heavy': 65,
    'freezing_rain': 66, 'freezing_rain_heavy': 67,
    'rain_shower': 80, 'rain_shower_moderate': 81,
    // Backend currently maps ww82 to rain_shower_moderate; keep alias for compatibility.
    'rain_shower_heavy': 82,
    'snow_slight': 71, 'snow_moderate': 73, 'snow_heavy': 75,
    'snow_grains': 77,
    'snow_shower': 85, 'snow_shower_heavy': 86,
    'thunderstorm': 95, 'thunderstorm_hail': 96,
  };
  return map[type] ?? null;
}
