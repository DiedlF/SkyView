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


def colormap_thermals(val):
    cape = float(val)
    if cape < 50:
        return None
    t = min(max((cape - 50.0) / 950.0, 0.0), 1.0)
    return (int(50 + 205 * t), int(180 * (1 - t)), int(50 * (1 - t)), int(90 + 130 * t))


def colormap_ceiling(v):
    if v >= 9900 or v <= 0:
        return None
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
    "rain": {"var": "prr_gsp", "cmap": colormap_rain},
    "snow": {"var": "prs_gsp", "cmap": colormap_snow},
    "hail": {"var": "prg_gsp", "cmap": colormap_hail},
    "clouds_low": {"var": "clcl", "cmap": colormap_clouds},
    "clouds_mid": {"var": "clcm", "cmap": colormap_clouds},
    "clouds_high": {"var": "clch", "cmap": colormap_clouds},
    "clouds_total": {"var": "clct", "cmap": colormap_clouds},
    "clouds_total_mod": {"var": "clct_mod", "cmap": colormap_clouds},
    "dry_conv_top": {"var": "htop_dc", "cmap": colormap_ceiling},
    "sigwx": {"var": "ww", "cmap": colormap_sigwx},
    "ceiling": {"var": "ceiling", "cmap": colormap_ceiling},
    "cloud_base": {"var": "hbas_sc", "cmap": colormap_hbas_sc},
    "conv_thickness": {"var": "conv_thickness", "cmap": colormap_conv_thickness, "computed": True},
    "thermals": {"var": "cape_ml", "cmap": colormap_thermals},
    "wstar": {"var": "wstar", "cmap": colormap_wstar, "computed": True},
    "climb_rate": {"var": "climb_rate", "cmap": colormap_climb_rate, "computed": True},
    "lcl": {"var": "lcl", "cmap": colormap_lcl, "computed": True},
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
        mmh = v * 3600.0
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

    if layer in ("ceiling", "dry_conv_top"):
        m = valid & (v > 0) & (v < 9900)
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

    cmap_fn = OVERLAY_CONFIGS[layer]["cmap"]
    for yy in range(h):
        for xx in range(w):
            if not valid[yy, xx]:
                continue
            color = cmap_fn(v[yy, xx])
            if color:
                rgba[yy, xx] = color
    return rgba
