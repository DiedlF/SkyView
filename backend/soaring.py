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

    Args:
        ashfl_s: Accumulated sensible heat flux (J/m²) at current step
        ashfl_s_prev: Accumulated sensible heat flux (J/m²) at previous step (or None)
        mh: Mixed layer height / boundary layer depth (m)
        t_2m: 2m temperature (K)
        dt_seconds: Time between steps (s)

    Returns:
        W* in m/s (2D array), clipped to >= 0
    """
    # Convert accumulated heat flux to instantaneous rate (W/m²)
    if ashfl_s_prev is not None:
        qs_wm2 = (ashfl_s - ashfl_s_prev) / dt_seconds
    else:
        # First timestep: estimate from accumulated value
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
    """Expected vario reading = W* - glider sink rate, capped at 0.

    Args:
        wstar: Thermal updraft velocity (m/s)
        sink_rate: Glider sink rate while thermalling (m/s)

    Returns:
        Expected climb rate (m/s), minimum 0
    """
    return np.maximum(wstar - sink_rate, 0.0)


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
