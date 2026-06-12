"""Derived PNG renderers for observation frames.

The observation cache keeps rendered products, not native provider files. Native
ODIM/NAT payloads are decoded during ingest, converted to display-ready PNGs,
and then discarded by the caller's temporary directory.
"""

from __future__ import annotations

from pathlib import Path


def render_radar_dbz_png(field, out_png: Path) -> Path:
    """Render a reprojected OPERA dBZ field as a north-up RGBA PNG."""
    import numpy as np
    from PIL import Image

    data = np.asarray(field, dtype="float32")
    if data.ndim != 2:
        raise ValueError("radar field must be a 2-D array")

    # Target-grid latitudes are ascending south->north; image rows are north->south.
    data = np.flipud(data)
    valid = np.isfinite(data)

    stops = np.asarray([-5, 0, 5, 10, 20, 30, 40, 50, 60], dtype="float32")
    colors = np.asarray(
        [
            (30, 20, 70, 0),
            (60, 70, 190, 80),
            (55, 145, 240, 120),
            (40, 200, 220, 150),
            (35, 230, 120, 180),
            (190, 245, 45, 205),
            (255, 185, 45, 225),
            (235, 55, 10, 240),
            (125, 0, 0, 250),
        ],
        dtype="float32",
    )

    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    clipped = np.clip(np.where(valid, data, stops[0]), stops[0], stops[-1])
    for channel in range(4):
        rgba[..., channel] = np.interp(clipped, stops, colors[:, channel]).astype("uint8")
    rgba[~valid | (data < 0.0)] = (0, 0, 0, 0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)
    return out_png


def render_satellite_hrv_png(
    native_file: Path,
    out_png: Path,
    *,
    bbox: tuple[float, float, float, float],
) -> Path:
    """Render the MSG SEVIRI HRV channel as a high-resolution grayscale PNG."""
    import numpy as np
    from PIL import Image
    from satpy import Scene

    scn = Scene(reader="seviri_l1b_native", filenames=[str(native_file)])
    scn.load(["HRV"])
    cropped = scn.crop(ll_bbox=bbox)
    data = np.asarray(cropped["HRV"].values, dtype="float32")
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("HRV scene has no finite pixels in requested bbox")

    lo = float(np.nanpercentile(data[finite], 0.5))
    hi = float(np.nanpercentile(data[finite], 99.8))
    scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    scaled[~finite] = 0.0

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255.0).astype("uint8"), mode="L").save(out_png, optimize=True)
    return out_png
