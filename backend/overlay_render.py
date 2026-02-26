"""Overlay color maps, layer config, and fast colorization helpers."""

from __future__ import annotations

import colorsys
import numpy as np


def colormap_total_precip(total_rate):
    mmh = total_rate * 3600
    if mmh < 0.1:
        return None
    t = min(mmh / 5.0, 1.0)
    return (int(150 * (1 - t)), int(180 + 75 * (1 - t)), 255, int(120 + 135 * t))


def colormap_rain(rain_rate):
    mmh = rain_rate * 3600
    if mmh < 0.1:
        return None
    t = min(mmh / 5.0, 1.0)
    return (int(180 - 160 * t), int(220 - 160 * t), int(255 - 75 * t), int(130 + 125 * t))


def colormap_snow(snow_rate):
    mmh = snow_rate * 3600
    if mmh < 0.1:
        return None
    t = min(mmh / 5.0, 1.0)
    return (int(255 - 135 * t), int(200 - 160 * t), int(255 - 95 * t), int(130 + 125 * t))


def colormap_hail(graupel_rate):
    mmh = graupel_rate * 3600
    if mmh < 0.1:
        return None
    t = min(mmh / 5.0, 1.0)
    return (int(200 + 55 * t), int(80 + 80 * (1 - t)), int(20 * (1 - t)), int(130 + 125 * t))


def colormap_sigwx(val):
    ww = int(val)
    if ww == 0:
        return None
    if ww == 1:
        return (205, 205, 205, 165)
    if ww == 2:
        return (145, 145, 145, 175)
    if ww == 3:
        return (85, 85, 85, 190)
    if ww < 10:
        g = int(120 - (ww - 4) * 8)
        g = max(75, min(130, g))
        return (g, g, g, 185)

    if ww in (45, 48):
        base_h = 45 / 360.0
    elif 50 <= ww <= 59:
        base_h = 95 / 360.0
    elif 60 <= ww <= 69:
        base_h = 130 / 360.0
    elif 70 <= ww <= 79:
        base_h = 265 / 360.0
    elif 80 <= ww <= 84:
        base_h = 175 / 360.0
    elif 85 <= ww <= 86:
        base_h = 280 / 360.0
    elif 95 <= ww <= 99:
        base_h = 350 / 360.0
    else:
        base_h = 210 / 360.0

    hue = (base_h + ((ww * 0.61803398875) % 1.0) * 0.18) % 1.0
    sat = 0.82 if ww >= 95 else 0.78
    val_v = 0.95 if ww >= 95 else 0.88
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val_v)
    r, g, b = int(r_f * 255), int(g_f * 255), int(b_f * 255)
    a = 210 if ww >= 95 else 190
    return (r, g, b, a)


def colormap_clouds(val):
    pct = float(val)
    if pct < 1:
        return None
    t = min(pct / 100.0, 1.0)
    grey = int(225 - 180 * t)
    return (grey, grey, grey, 210)


def _temp_to_c(v):
    tv = float(v)
    # ICON temperatures are in Kelvin; keep Celsius input unchanged if already plausible.
    return tv - 273.15 if tv > 170 else tv


def colormap_temperature(v):
    if v is None:
        return None
    c = _temp_to_c(v)
    # Display range: -30 .. +40 °C
    t = min(max((c + 30.0) / 70.0, 0.0), 1.0)
    return (int(40 + 215 * t), int(90 + 110 * (1 - abs(t - 0.5) * 2)), int(230 - 210 * t), int(95 + 125 * t))


def colormap_mh(v):
    if v is None or float(v) <= 0:
        return None
    t = min(float(v) / 3000.0, 1.0)
    return (int(220 * (1 - t)), int(90 + 150 * t), int(40 + 60 * t), int(100 + 120 * t))


def colormap_ashfl(v):
    if v is None or float(v) < 20:
        return None
    t = min((float(v) - 20.0) / 380.0, 1.0)
    return (int(70 + 185 * t), int(170 - 110 * t), int(240 - 220 * t), int(95 + 120 * t))


def colormap_relhum(v):
    if v is None or float(v) < 1:
        return None
    t = min(max(float(v), 0.0) / 100.0, 1.0)
    return (int(230 - 170 * t), int(230 - 120 * t), int(230 - 40 * t), int(80 + 140 * t))


def colormap_dew_spread(v):
    if v is None or float(v) < 0:
        return None
    d = float(v)
    # Kelvin difference equals Celsius difference.
    t = min(max(d, 0.0) / 25.0, 1.0)
    return (int(70 + 185 * t), int(200 - 130 * t), int(220 - 180 * t), int(90 + 130 * t))


def colormap_thermals(val):
    cape = float(val)
    if cape < 5:
        return None
    t = min(max((cape - 5.0) / 995.0, 0.0), 1.0)
    return (int(50 + 205 * t), int(180 * (1 - t)), int(50 * (1 - t)), int(90 + 130 * t))


def colormap_ceiling(v):
    # Keep missing/no-ceiling markers transparent.
    # ICON uses very high sentinel values (~20700m) for "no relevant ceiling".
    if v <= 0 or v >= 20000:
        return None
    # Real high ceilings above 9900m are clamped to max color.
    t = max(0.0, min(float(v), 9900.0)) / 9900.0
    return (int(220 * (1 - t)), int(60 + 180 * t), int(30 + 50 * t), int(200 - 60 * t))


def colormap_hbas_sc(v):
    if v <= 0:
        return None
    t = max(0.0, min(float(v), 5000.0)) / 5000.0
    return (int(220 * (1 - t)), int(60 + 180 * t), int(30 + 50 * t), int(200 - 60 * t))


def colormap_wstar(v):
    if v < 0.2:
        return None
    t = min(float(v) / 5.0, 1.0)
    return (int(50 + 205 * t), int(200 - 80 * t), int(50 * (1 - t)), int(100 + 130 * t))


def colormap_climb_rate(v):
    if v < 0.1:
        return None
    t = min(float(v) / 5.0, 1.0)
    return (int(50 + 205 * t), int(200 - 80 * t), int(50 * (1 - t)), int(100 + 130 * t))


def colormap_lcl(v):
    if v < 50:
        return None
    t = min(float(v) / 5000.0, 1.0)
    return (int(220 * (1 - t)), int(60 + 180 * t), int(30 + 50 * t), int(180 - 40 * t))


def colormap_reachable(v):
    if v < 1:
        return None
    t = min(float(v) / 200.0, 1.0)
    return (int(220 * (1 - t)), int(80 + 160 * t), 50, int(120 + 100 * t))


def colormap_conv_thickness(v):
    if v is None or float(v) <= 0:
        return None
    t = min(float(v) / 6000.0, 1.0)
    return (int(240 * t + 40 * (1 - t)), int(220 * (1 - t) + 80 * t), int(60 * (1 - t) + 40 * t), 190)


def colormap_lpi(v):
    if v is None or float(v) <= 0:
        return None
    t = min(float(v) / 20.0, 1.0)
    return (int(70 + 185 * t), int(190 * (1 - t) + 70 * t), int(80 * (1 - t) + 40 * t), int(120 + 110 * t))


def _build_sigwx_lut() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.uint8)
    for ww in range(256):
        c = colormap_sigwx(ww)
        if c:
            lut[ww] = np.array(c, dtype=np.uint8)
    return lut


SIGWX_LUT = _build_sigwx_lut()


OVERLAY_CONFIGS = {
    "total_precip": {"var": "total_precip", "cmap": colormap_total_precip, "computed": True},
    "rain": {"var": "rain_amount", "cmap": colormap_rain, "computed": True},
    "snow": {"var": "snow_amount", "cmap": colormap_snow, "computed": True},
    "hail": {"var": "hail_amount", "cmap": colormap_hail, "computed": True},
    "clouds_low": {"var": "clcl", "cmap": colormap_clouds},
    "clouds_mid": {"var": "clcm", "cmap": colormap_clouds},
    "clouds_high": {"var": "clch", "cmap": colormap_clouds},
    "clouds_total": {"var": "clct", "cmap": colormap_clouds},
    "clouds_total_mod": {"var": "clct_mod", "cmap": colormap_clouds},
    "t_2m": {"var": "t_2m", "cmap": colormap_temperature},
    "t_950hpa": {"var": "t_950hpa", "cmap": colormap_temperature},
    "t_850hpa": {"var": "t_850hpa", "cmap": colormap_temperature},
    "t_700hpa": {"var": "t_700hpa", "cmap": colormap_temperature},
    "t_500hpa": {"var": "t_500hpa", "cmap": colormap_temperature},
    "t_300hpa": {"var": "t_300hpa", "cmap": colormap_temperature},
    "mh": {"var": "mh", "cmap": colormap_mh},
    "ashfl_s": {"var": "ashfl_s", "cmap": colormap_ashfl},
    "relhum_2m": {"var": "relhum_2m", "cmap": colormap_relhum},
    "dew_spread_2m": {"var": "dew_spread_2m", "cmap": colormap_dew_spread, "computed": True},
    "dry_conv_top": {"var": "htop_dc", "cmap": colormap_ceiling},
    "sigwx": {"var": "ww", "cmap": colormap_sigwx},
    "ceiling": {"var": "ceiling", "cmap": colormap_ceiling},
    "cloud_base": {"var": "hbas_sc", "cmap": colormap_hbas_sc},
    "conv_thickness": {"var": "conv_thickness", "cmap": colormap_conv_thickness, "computed": True},
    "lpi": {"var": "lpi_max", "cmap": colormap_lpi},
    "thermals": {"var": "cape_ml", "cmap": colormap_thermals},
    "wstar": {"var": "wstar", "cmap": colormap_wstar, "computed": True},
    "climb_rate": {"var": "climb_rate", "cmap": colormap_climb_rate, "computed": True},
    "lcl": {"var": "lcl", "cmap": colormap_lcl, "computed": True},
    "h_snow": {"var": "h_snow", "cmap": colormap_clouds},
    "reachable": {"var": "reachable", "cmap": colormap_reachable, "computed": True},
}


def colorize_layer_vectorized(layer: str, sampled: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = sampled.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if not np.any(valid):
        return rgba
    v = sampled

    def set_rgba(mask, r, g, b, a):
        if np.any(mask):
            rgba[..., 0][mask] = np.clip(r[mask] if isinstance(r, np.ndarray) else r, 0, 255).astype(np.uint8) if isinstance(r, np.ndarray) else np.uint8(r)
            rgba[..., 1][mask] = np.clip(g[mask] if isinstance(g, np.ndarray) else g, 0, 255).astype(np.uint8) if isinstance(g, np.ndarray) else np.uint8(g)
            rgba[..., 2][mask] = np.clip(b[mask] if isinstance(b, np.ndarray) else b, 0, 255).astype(np.uint8) if isinstance(b, np.ndarray) else np.uint8(b)
            rgba[..., 3][mask] = np.clip(a[mask] if isinstance(a, np.ndarray) else a, 0, 255).astype(np.uint8) if isinstance(a, np.ndarray) else np.uint8(a)

    if layer in ("total_precip", "rain", "snow", "hail"):
        mmh = v
        m = valid & (mmh >= 0.1)
        if np.any(m):
            t = np.clip(mmh / 5.0, 0.0, 1.0)
            if layer == "total_precip":
                set_rgba(m, 150 * (1 - t), 180 + 75 * (1 - t), np.full_like(t, 255.0), 120 + 135 * t)
            elif layer == "rain":
                set_rgba(m, 180 - 160 * t, 220 - 160 * t, 255 - 75 * t, 130 + 125 * t)
            elif layer == "snow":
                set_rgba(m, 255 - 135 * t, 200 - 160 * t, 255 - 95 * t, 130 + 125 * t)
            else:
                set_rgba(m, 200 + 55 * t, 80 + 80 * (1 - t), 20 * (1 - t), 130 + 125 * t)
        return rgba

    if layer.startswith("clouds_"):
        m = valid & (v >= 1)
        if np.any(m):
            t = np.clip(v / 100.0, 0.0, 1.0)
            grey = 225 - 180 * t
            set_rgba(m, grey, grey, grey, 210)
        return rgba

    if layer.startswith("t_"):
        m = valid & np.isfinite(v)
        if np.any(m):
            vc = np.where(v > 170.0, v - 273.15, v)
            t = np.clip((vc + 30.0) / 70.0, 0.0, 1.0)
            set_rgba(m, 40 + 215 * t, 90 + 110 * (1 - np.abs(t - 0.5) * 2), 230 - 210 * t, 95 + 125 * t)
        return rgba

    if layer == "mh":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 3000.0, 0.0, 1.0)
            set_rgba(m, 220 * (1 - t), 90 + 150 * t, 40 + 60 * t, 100 + 120 * t)
        return rgba

    if layer == "ashfl_s":
        m = valid & (v >= 20)
        if np.any(m):
            t = np.clip((v - 20.0) / 380.0, 0.0, 1.0)
            set_rgba(m, 70 + 185 * t, 170 - 110 * t, 240 - 220 * t, 95 + 120 * t)
        return rgba

    if layer == "relhum_2m":
        m = valid & (v >= 1)
        if np.any(m):
            t = np.clip(v / 100.0, 0.0, 1.0)
            set_rgba(m, 230 - 170 * t, 230 - 120 * t, 230 - 40 * t, 80 + 140 * t)
        return rgba

    if layer == "dew_spread_2m":
        m = valid & (v >= 0)
        if np.any(m):
            t = np.clip(v / 25.0, 0.0, 1.0)
            set_rgba(m, 70 + 185 * t, 200 - 130 * t, 220 - 180 * t, 90 + 130 * t)
        return rgba

    if layer in ("ceiling", "dry_conv_top"):
        m = valid & (v > 0) & (v < 20000)
        if np.any(m):
            t = np.clip(v / 9900.0, 0.0, 1.0)
            set_rgba(m, 220 * (1 - t), 60 + 180 * t, 30 + 50 * t, 200 - 60 * t)
        return rgba

    if layer == "cloud_base":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 5000.0, 0.0, 1.0)
            set_rgba(m, 220 * (1 - t), 60 + 180 * t, 30 + 50 * t, 200 - 60 * t)
        return rgba

    if layer == "conv_thickness":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 6000.0, 0.0, 1.0)
            set_rgba(m, 240 * t + 40 * (1 - t), 220 * (1 - t) + 80 * t, 60 * (1 - t) + 40 * t, 190)
        return rgba

    if layer == "lpi":
        m = valid & (v > 0)
        if np.any(m):
            t = np.clip(v / 20.0, 0.0, 1.0)
            set_rgba(m, 70 + 185 * t, 190 - 120 * t, 80 - 40 * t, 110 + 120 * t)
        return rgba

    if layer == "thermals":
        m = valid & (v >= 50)
        if np.any(m):
            t = np.clip((v - 50.0) / 950.0, 0.0, 1.0)
            set_rgba(m, 50 + 205 * t, 180 * (1 - t), 50 * (1 - t), 90 + 130 * t)
        return rgba

    if layer in ("wstar", "climb_rate"):
        th = 0.2 if layer == "wstar" else 0.1
        m = valid & (v >= th)
        if np.any(m):
            t = np.clip(v / 5.0, 0.0, 1.0)
            set_rgba(m, 50 + 205 * t, 200 - 80 * t, 50 * (1 - t), 100 + 130 * t)
        return rgba


    if layer == "h_snow":
        m = valid & (v > 0.0)
        if np.any(m):
            # meters; 0 stays transparent
            t = np.clip(v / 1.0, 0.0, 1.0)
            set_rgba(m, 220 - 120 * t, 235 - 85 * t, 255 - 15 * t, 70 + 150 * t)
        return rgba

    if layer == "lcl":
        m = valid & (v >= 50)
        if np.any(m):
            t = np.clip(v / 5000.0, 0.0, 1.0)
            set_rgba(m, 220 * (1 - t), 60 + 180 * t, 30 + 50 * t, 180 - 40 * t)
        return rgba

    if layer == "reachable":
        m = valid & (v >= 1)
        if np.any(m):
            t = np.clip(v / 200.0, 0.0, 1.0)
            set_rgba(m, 220 * (1 - t), 80 + 160 * t, np.full_like(t, 50.0), 120 + 100 * t)
        return rgba

    if layer == "sigwx":
        # Fast lookup-table path for discrete WMO ww codes.
        wwi = np.clip(v.astype(np.int16), 0, 255).astype(np.uint8)
        rgba_lookup = SIGWX_LUT[wwi]
        rgba[valid] = rgba_lookup[valid]
        return rgba

    # Generic fallback for uncommon/custom layers: process only valid samples.
    # (Avoid full h*w nested Python loops.)
    cmap_fn = OVERLAY_CONFIGS[layer]["cmap"]
    iy, ix = np.where(valid)
    for k in range(len(iy)):
        yy = int(iy[k]); xx = int(ix[k])
        color = cmap_fn(v[yy, xx])
        if color:
            rgba[yy, xx] = color
    return rgba
