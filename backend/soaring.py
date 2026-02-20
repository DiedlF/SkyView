"""Soaring forecast calculations for Skyview.

Implements:
- W* (convective velocity scale / thermal strength)
- LCL (lifting condensation level / cumulus cloud base)
- Reachable distance (glide cone)

Based on RASP/BLIPMAP methods (Dr. Jack Glendening).
"""

import numpy as np

# Constants
G = 9.81          # gravitational acceleration (m/s²)
RHO = 1.225       # air density at sea level (kg/m³)
CP = 1004.0       # specific heat of dry air (J/(kg·K))

# Glider parameters (sailplane)
GLIDE_RATIO = 40.0
SINK_RATE = 0.7        # m/s
SAFETY_MARGIN = 200.0  # meters


def calc_wstar(ashfl_s, ashfl_s_prev, mh, t_2m, dt_seconds=3600):
    """Calculate convective velocity scale W*.

    W* = [(g/T₀) × Qₛ × D]^(1/3)

    Notes:
    - DWD `ashfl_s` in current ICON products behaves like instantaneous sensible heat flux
      (W/m²), not accumulated J/m². Older/other datasets may be accumulated.
    - We therefore use a unit heuristic to avoid collapsing W* to ~0 by over-dividing.

    Args:
        ashfl_s: Sensible heat flux field (usually W/m²; possibly accumulated J/m²)
        ashfl_s_prev: Previous-step field for de-accum if needed
        mh: Mixed layer height / boundary layer depth (m)
        t_2m: 2m temperature (K)
        dt_seconds: Time between steps (s)

    Returns:
        W* in m/s (2D array), clipped to >= 0
    """
    # Heuristic: typical instantaneous flux magnitudes are O(10..500) W/m².
    # Accumulated J/m² fields over 1h are O(1e4..1e6).
    try:
        p99_abs = float(np.nanpercentile(np.abs(ashfl_s), 99))
    except Exception:
        p99_abs = 0.0
    looks_instantaneous = p99_abs < 2000.0

    if looks_instantaneous:
        qs_wm2 = ashfl_s
    elif ashfl_s_prev is not None:
        qs_wm2 = (ashfl_s - ashfl_s_prev) / dt_seconds
    else:
        qs_wm2 = ashfl_s / dt_seconds

    # Only positive (upward) heat flux creates thermals
    qs_wm2 = np.maximum(qs_wm2, 0.0)

    # Convert to kinematic flux (K·m/s)
    qs_kin = qs_wm2 / (RHO * CP)

    # Boundary layer depth must be positive
    bl_depth = np.maximum(mh, 10.0)

    # Mean temperature (K) — use t_2m as approximation
    t0 = np.maximum(t_2m, 200.0)

    # W* = [(g/T₀) × Qₛ × D]^(1/3)
    product = (G / t0) * qs_kin * bl_depth
    product = np.maximum(product, 0.0)
    wstar = np.power(product, 1.0 / 3.0)

    return wstar


def calc_climb_rate(wstar, sink_rate=SINK_RATE):
    """Expected vario reading = W* - glider sink rate, capped at 0."""
    return np.maximum(wstar - sink_rate, 0.0)


def compute_lapse_factor(delta_t_forecast_c, delta_altitude_km):
    """Lapse factor = ΔT_forecast / ΔT_theoretical_dry_adiabatic.

    ΔT_theoretical_dry_adiabatic = 9.8 * Δz_km
    """
    delta_altitude_km = np.maximum(delta_altitude_km, 0.1)
    delta_t_theoretical = 9.8 * delta_altitude_km
    return delta_t_forecast_c / delta_t_theoretical


def classify_thermal_strength(t2m_c, td2m_c, t_upper_c, delta_altitude_km):
    """Heuristic thermal class using lapse + near-surface moisture.

    Returns:
        thermal_class: 0..3
        lapse_factor: normalized instability proxy
        moisture_class_code: 0 very moist, 1 moist, 2 moderate, 3 dry
    """
    delta_t_forecast = t2m_c - t_upper_c
    lapse_factor = compute_lapse_factor(delta_t_forecast, delta_altitude_km)
    dT_surface = t2m_c - td2m_c

    base_class = np.where(lapse_factor < 0.6, 0,
                  np.where(lapse_factor < 0.9, 1,
                  np.where(lapse_factor < 1.2, 2, 3))).astype(np.int8)

    moisture_class_code = np.where(dT_surface < 2.0, 0,
                           np.where(dT_surface < 6.0, 1,
                           np.where(dT_surface < 12.0, 2, 3))).astype(np.int8)

    # very moist -> reduce usable thermal class by one
    thermal_class = np.where(moisture_class_code == 0, np.maximum(base_class - 1, 0), base_class).astype(np.int8)
    return thermal_class, lapse_factor, moisture_class_code


def calc_climb_rate_from_thermal_class(thermal_class):
    """Map thermal class (0..3) to representative climb-rate m/s.

    Requested mapping:
    - 0 -> 0 m/s
    - 1 -> 1 m/s
    - 2 -> 2 m/s
    - 3 -> >3 m/s (represented as 3.2 m/s)
    """
    return np.choose(np.clip(thermal_class, 0, 3), [0.0, 1.0, 2.0, 3.2]).astype(np.float32)


def calc_lcl(t_2m, td_2m, hsurf):
    """Calculate Lifting Condensation Level (cumulus cloud base).

    Uses Espy formula: LCL_agl ≈ 125 × (T - Td)

    Args:
        t_2m: 2m temperature (K)
        td_2m: 2m dew point temperature (K)
        hsurf: Surface elevation (m AMSL)

    Returns:
        LCL height in meters AMSL (2D array)
    """
    # Convert to Celsius for the Espy formula
    t_c = t_2m - 273.15
    td_c = td_2m - 273.15

    # Dew point spread
    spread = np.maximum(t_c - td_c, 0.0)

    # LCL above ground level
    lcl_agl = 125.0 * spread

    # Convert to AMSL
    lcl_amsl = lcl_agl + hsurf

    return lcl_amsl


def calc_cu_potential(lcl_amsl, mh, hsurf):
    """Determine cumulus potential.

    Cumulus forms when BL top reaches above LCL.
    BL top AMSL = mh + hsurf (mh is depth above surface).

    Args:
        lcl_amsl: LCL height (m AMSL)
        mh: Mixed layer height / BL depth (m above surface)
        hsurf: Surface elevation (m AMSL)

    Returns:
        cu_potential: Positive = cumulus expected, negative = blue/dry thermals
    """
    bl_top_amsl = mh + hsurf
    return bl_top_amsl - lcl_amsl


def calc_thermal_height(mh, lcl_amsl, hsurf):
    """Usable thermal height for gliders.

    The effective thermalling height is the lower of:
    - BL top (dry thermal top)
    - LCL (cloud base, if cumulus present)

    Args:
        mh: Mixed layer height / BL depth (m above surface)
        lcl_amsl: LCL height (m AMSL)
        hsurf: Surface elevation (m AMSL)

    Returns:
        Usable height AGL in meters
    """
    bl_top_amsl = mh + hsurf
    # Thermal top is min of BL top and LCL (if Cu, you stop at cloud base)
    cu_mask = bl_top_amsl > lcl_amsl
    thermal_top_amsl = np.where(cu_mask, lcl_amsl, bl_top_amsl)

    # Height above ground
    thermal_agl = thermal_top_amsl - hsurf
    return np.maximum(thermal_agl, 0.0)


def calc_reachable_distance(thermal_agl, glide_ratio=GLIDE_RATIO, safety_margin=SAFETY_MARGIN):
    """Calculate reachable glide distance from thermal top.

    Args:
        thermal_agl: Usable thermal height AGL (m)
        glide_ratio: L/D ratio of the glider
        safety_margin: Minimum height reserve (m)

    Returns:
        Reachable distance in km (2D array)
    """
    usable = np.maximum(thermal_agl - safety_margin, 0.0)
    distance_m = usable * glide_ratio
    return distance_m / 1000.0  # Convert to km
