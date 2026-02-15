// windbarbs.js — Wind barb SVG generation for Skyview
// Standard meteorological wind barbs:
//   Staff points in direction wind comes FROM
//   Short barb = 5 kt, long barb = 10 kt, pennant (flag) = 50 kt

const BARB_SIZE = 40;
const BARB_COLOR = '#333';

/**
 * Decompose wind speed (knots) into pennants, long barbs, and short barbs.
 * Uses standard meteorological rounding: round to nearest 5 kt.
 */
function decomposeSpeed(speedKt) {
  // Round to nearest 5
  const rounded = Math.round(speedKt / 5) * 5;
  const pennants = Math.floor(rounded / 50);
  const remaining = rounded - pennants * 50;
  const longBarbs = Math.floor(remaining / 10);
  const shortBarbs = Math.floor((remaining - longBarbs * 10) / 5);
  return { pennants, longBarbs, shortBarbs };
}

/**
 * Build SVG path elements for a wind barb at origin pointing UP (North).
 * The barb is drawn from bottom (tail) to top (tip), with barbs on the right side.
 * Caller rotates the whole thing based on wind direction.
 *
 * @param {number} speedKt - Wind speed in knots
 * @returns {string} SVG elements as string
 */
function buildBarbElements(speedKt) {
  if (speedKt < 1) {
    // Calm: circle
    return `<circle cx="20" cy="20" r="4" fill="none" stroke="${BARB_COLOR}" stroke-width="1.5"/>`;
  }

  const { pennants, longBarbs, shortBarbs } = decomposeSpeed(speedKt);

  // Staff: vertical line from bottom center up
  // Staff starts at (20, 36) bottom, goes to (20, 4) top
  const staffTop = 4;
  const staffBottom = 36;
  const staffLen = staffBottom - staffTop;

  let svg = `<line x1="20" y1="${staffBottom}" x2="20" y2="${staffTop}" stroke="${BARB_COLOR}" stroke-width="1.8" stroke-linecap="round"/>`;

  // Place barbs from top down
  let y = staffTop;
  const barbSpacing = 4;
  const barbLength = 10;  // length of long barb line
  const shortBarbLength = 6;
  const pennantWidth = 5;

  // Pennants (filled triangles)
  for (let i = 0; i < pennants; i++) {
    svg += `<polygon points="20,${y} 30,${y + pennantWidth / 2} 20,${y + pennantWidth}" fill="${BARB_COLOR}" stroke="none"/>`;
    y += pennantWidth + 1;
  }

  // Long barbs (full-length lines to the right)
  for (let i = 0; i < longBarbs; i++) {
    svg += `<line x1="20" y1="${y}" x2="${20 + barbLength}" y2="${y - 3}" stroke="${BARB_COLOR}" stroke-width="1.5" stroke-linecap="round"/>`;
    y += barbSpacing;
  }

  // Short barbs (half-length lines)
  for (let i = 0; i < shortBarbs; i++) {
    // If this is the only barb, offset it slightly from the tip
    if (pennants === 0 && longBarbs === 0 && i === 0) {
      y += barbSpacing;
    }
    svg += `<line x1="20" y1="${y}" x2="${20 + shortBarbLength}" y2="${y - 2}" stroke="${BARB_COLOR}" stroke-width="1.5" stroke-linecap="round"/>`;
    y += barbSpacing;
  }

  // Dot at base (wind observation point)
  svg += `<circle cx="20" cy="${staffBottom}" r="2" fill="${BARB_COLOR}"/>`;

  return svg;
}

/**
 * Create a Leaflet DivIcon for a wind barb.
 * @param {number} speedKt - Wind speed in knots
 * @param {number} dirDeg - Direction wind comes FROM (meteorological, degrees from N clockwise)
 * @returns {L.DivIcon}
 */
function createWindBarbIcon(speedKt, dirDeg) {
  const elements = buildBarbElements(speedKt);

  // Rotate: dirDeg=0 means from North (pointing down toward South)
  // Our SVG draws staff pointing UP, so rotate by dirDeg (FROM direction)
  // to point the staff into the wind direction it comes from
  const html = `<svg xmlns="http://www.w3.org/2000/svg" width="${BARB_SIZE}" height="${BARB_SIZE}" viewBox="0 0 ${BARB_SIZE} ${BARB_SIZE}" style="transform: rotate(${dirDeg}deg);">${elements}</svg>`;

  return L.divIcon({
    html: html,
    className: 'wind-barb',
    iconSize: [BARB_SIZE, BARB_SIZE],
    iconAnchor: [BARB_SIZE / 2, BARB_SIZE / 2],
  });
}
