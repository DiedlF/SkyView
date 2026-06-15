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


def _save_grayscale_png(data, out_png: Path, *, label: str) -> Path:
    """Percentile-stretch a reflectance field to an 8-bit grayscale PNG.

    ``data`` is the resampled, north-up SkyView-grid array (NaN off-disk). The
    0.5/99.8 percentile stretch matches across both visible products so MSG HRV
    and MTG vis_06 frames look comparable side-by-side.
    """
    import numpy as np
    from PIL import Image

    data = np.asarray(data, dtype="float32")
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError(f"{label} scene has no finite pixels in requested bbox")

    lo = float(np.nanpercentile(data[finite], 0.5))
    hi = float(np.nanpercentile(data[finite], 99.8))
    scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    scaled[~finite] = 0.0

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255.0).astype("uint8"), mode="L").save(out_png, optimize=True)
    return out_png


def render_satellite_hrv_png(
    native_file: Path,
    out_png: Path,
    *,
    bbox: tuple[float, float, float, float],
) -> Path:
    """Render MSG SEVIRI HRV as a north-up SkyView-grid grayscale PNG."""
    from satpy import Scene

    from .reproject import web_mercator_area_definition

    scn = Scene(reader="seviri_l1b_native", filenames=[str(native_file)])
    scn.load(["HRV"], upper_right_corner="NE")
    cropped = scn.crop(ll_bbox=bbox)
    resampled = cropped.resample(
        web_mercator_area_definition(),
        datasets=["HRV"],
        resampler="nearest",
        radius_of_influence=5000,
    )
    return _save_grayscale_png(resampled["HRV"].values, out_png, label="HRV")


def render_li_accum_png(
    nc_files,
    out_png: Path,
    *,
    bbox: tuple[float, float, float, float],
    dataset: str = "flash_accumulation",
) -> Path:
    """Render MTG-I1 LI accumulated flashes as a coloured, transparent RGBA overlay.

    The accumulated product is a sparse 2-D field on the FCI 2 km geostationary
    grid (only flash cells are finite). We crop to ``bbox``, resample to the
    SkyView Web-Mercator grid, and colour-map flash count to a yellow→white
    lightning palette, fully transparent where there are no flashes — so it
    overlays cleanly on top of either the HRV or vis_06 image.
    """
    import numpy as np
    from PIL import Image
    from satpy import Scene

    from .reproject import web_mercator_area_definition

    scn = Scene(reader="li_l2_nc", filenames=[str(f) for f in nc_files])
    scn.load([dataset])
    cropped = scn.crop(ll_bbox=bbox)
    resampled = cropped.resample(
        web_mercator_area_definition(),
        datasets=[dataset],
        resampler="nearest",
        radius_of_influence=4000,
    )
    data = np.asarray(resampled[dataset].values, dtype="float32")
    # Target-grid latitudes are ascending south->north; image rows are north->south.
    data = np.flipud(data)
    flashes = np.isfinite(data) & (data > 0.0)

    stops = np.asarray([1, 5, 20, 50, 100], dtype="float32")
    colors = np.asarray(
        [
            (255, 255, 130, 200),  # 1   pale yellow
            (255, 226, 60, 225),   # 5   yellow
            (255, 165, 30, 240),   # 20  orange
            (255, 85, 35, 250),    # 50  red-orange
            (255, 255, 255, 255),  # 100 white-hot
        ],
        dtype="float32",
    )

    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    clipped = np.clip(np.where(flashes, data, stops[0]), stops[0], stops[-1])
    for channel in range(4):
        rgba[..., channel] = np.interp(clipped, stops, colors[:, channel]).astype("uint8")
    rgba[~flashes] = (0, 0, 0, 0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)
    return out_png


def render_fci_vis_png(
    chunk_files,
    out_png: Path,
    *,
    bbox: tuple[float, float, float, float],
    channel: str = "vis_06",
) -> Path:
    """Render MTG-I1 FCI ``vis_06`` as a north-up SkyView-grid grayscale PNG.

    ``chunk_files`` is the list of FCI L1c chunk NetCDF files for one repeat
    cycle. The scene is cropped to ``bbox`` before resampling so only the Europe
    region of the (very large) full disk is decoded into the target grid. TRAIL
    chunks are dropped — the ``fci_l1c_nc`` reader can't open them (it only reads
    BODY chunks) and passing them just logs a "Don't know how to open" warning.
    """
    from satpy import Scene

    from .reproject import web_mercator_area_definition

    body_files = [str(f) for f in chunk_files if "-TRAIL-" not in str(f).upper()]
    scn = Scene(reader="fci_l1c_nc", filenames=body_files)
    scn.load([channel], upper_right_corner="NE")
    cropped = scn.crop(ll_bbox=bbox)
    resampled = cropped.resample(
        web_mercator_area_definition(),
        datasets=[channel],
        resampler="nearest",
        radius_of_influence=2000,
    )
    return _save_grayscale_png(resampled[channel].values, out_png, label=channel)
