// symbols.js — Weather symbols for Skyview
// WMO ww symbols: from weewx-DWD / weathericons (Public Domain, colored)
// Cloud types: hand-drawn SVG (black #333)

const SYM_SIZE = 36;
const LABEL_OFFSET = 14;

// ─── WMO ww code → symbol file mapping ───
// Maps ww codes produced by ICON-D2 to SVG filenames
const WW_SYMBOLS = {};
// Generate mappings for all 100 ww codes (00-99)
for (let i = 0; i < 100; i++) {
  const code = String(i).padStart(2, '0');
  WW_SYMBOLS[i] = `/geodata/ww-symbols/wmo4677_ww${code}.svg`;
}

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
  cu_hum: {
    svg: `<line x1="6" y1="26" x2="30" y2="26" stroke="#333" stroke-width="3"/>
          <path d="M6 26 Q6 12 18 12 Q30 12 30 26" fill="none" stroke="#333" stroke-width="3"/>`,
    label: 'Cu humilis'
  },
  cu_con: {
    svg: `<line x1="6" y1="30" x2="30" y2="30" stroke="#333" stroke-width="3"/>
          <path d="M6 30 L6 20 Q6 14 11 14 Q14 8 18 8 Q22 8 25 14 Q30 14 30 20 L30 30" fill="none" stroke="#333" stroke-width="3"/>`,
    label: 'Cu congestus'
  },
  cb: {
    svg: `<line x1="6" y1="30" x2="30" y2="30" stroke="#333" stroke-width="3"/>
          <path d="M6 30 L6 20 Q6 14 11 14 Q14 8 18 8 Q22 8 25 14 Q30 14 30 20 L30 30" fill="none" stroke="#333" stroke-width="3"/>
          <line x1="4" y1="8" x2="32" y2="8" stroke="#333" stroke-width="3"/>`,
    label: 'Cb'
  },
  blue_thermal: {
    svg: `<text x="18" y="24" text-anchor="middle" font-size="22" font-weight="bold" fill="#333">b</text>`,
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

// Preload all ICON-D2 ww codes on startup
const ICON_D2_WW = [0,1,2,3,45,48,51,53,55,56,57,61,63,65,66,67,71,73,75,77,80,81,82,85,86,95,96];
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
  const map = {
    'fog': 45, 'rime_fog': 48,
    'drizzle_light': 50, 'drizzle_moderate': 51, 'drizzle_dense': 53,
    'freezing_drizzle': 56, 'freezing_drizzle_heavy': 57,
    'rain_slight': 60, 'rain_moderate': 61, 'rain_heavy': 63,
    'freezing_rain': 66, 'freezing_rain_heavy': 67,
    'rain_shower': 80, 'rain_shower_moderate': 81, 'rain_shower_heavy': 82,
    'snow_slight': 70, 'snow_moderate': 71, 'snow_heavy': 73,
    'snow_grains': 77,
    'snow_shower': 85, 'snow_shower_heavy': 86,
    'thunderstorm': 95, 'thunderstorm_hail': 96,
  };
  return map[type] ?? null;
}
