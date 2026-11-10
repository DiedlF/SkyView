from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

PIL_Image = pytest.importorskip("PIL.Image")

from observations.render import _save_grayscale_png, render_radar_dbz_png  # noqa: E402


def _reference_rgba(field):
    """The truecolour image the radar renderer produced before palettising."""
    data = np.flipud(np.asarray(field, dtype="float32"))
    valid = np.isfinite(data)
    stops = np.asarray([-5, 0, 5, 10, 20, 30, 40, 50, 60], dtype="float32")
    colors = np.asarray(
        [
            (30, 20, 70, 0), (60, 70, 190, 80), (55, 145, 240, 120), (40, 200, 220, 150),
            (35, 230, 120, 180), (190, 245, 45, 205), (255, 185, 45, 225), (235, 55, 10, 240),
            (125, 0, 0, 250),
        ],
        dtype="float32",
    )
    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    clipped = np.clip(np.where(valid, data, stops[0]), stops[0], stops[-1])
    for channel in range(4):
        rgba[..., channel] = np.interp(clipped, stops, colors[:, channel]).astype("uint8")
    rgba[~valid | (data < 0.0)] = (0, 0, 0, 0)
    return rgba


def test_radar_png_is_palettised_and_pixel_identical(tmp_path):
    rng = np.random.default_rng(7)
    field = np.round(rng.uniform(-10, 65, size=(120, 150)) * 2) / 2  # 0.5 dBZ steps
    field[rng.random(field.shape) < 0.3] = np.nan
    out = render_radar_dbz_png(field, tmp_path / "radar.png")

    img = PIL_Image.open(out)
    assert img.mode == "P"
    assert np.array_equal(np.asarray(img.convert("RGBA")), _reference_rgba(field))


def test_radar_png_all_transparent_and_single_pixel(tmp_path):
    empty = render_radar_dbz_png(np.full((8, 8), np.nan), tmp_path / "empty.png")
    assert np.asarray(PIL_Image.open(empty).convert("RGBA"))[..., 3].max() == 0

    one = np.full((8, 8), np.nan)
    one[0, 0] = 45.0  # row 0 is the southern edge -> bottom-left of the image
    rgba = np.asarray(PIL_Image.open(render_radar_dbz_png(one, tmp_path / "one.png")).convert("RGBA"))
    assert rgba[-1, 0, 3] > 0 and (rgba[..., 3] > 0).sum() == 1


def test_grayscale_format_follows_suffix(tmp_path):
    y, x = np.mgrid[0:200, 0:240]
    data = (np.sin(x / 17.0) + np.cos(y / 23.0)).astype("float32")
    data[:10, :10] = np.nan

    png = _save_grayscale_png(data, tmp_path / "frame.png", label="t")
    webp = _save_grayscale_png(data, tmp_path / "frame.webp", label="t")
    png_img, webp_img = PIL_Image.open(png), PIL_Image.open(webp)
    assert png_img.format == "PNG" and webp_img.format == "WEBP"
    assert webp_img.size == png_img.size

    # Lossy, but must stay close to the lossless render.
    diff = np.abs(
        np.asarray(webp_img.convert("L"), dtype="int16") - np.asarray(png_img, dtype="int16")
    )
    assert diff.mean() < 3


def test_grayscale_rejects_empty_scene(tmp_path):
    with pytest.raises(ValueError):
        _save_grayscale_png(np.full((4, 4), np.nan), tmp_path / "x.webp", label="t")
