# SPDX-License-Identifier: MIT
"""Cached metal surfaces and packaged shaded LaserTag actors."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_propagation


@lru_cache(maxsize=4)
def metal_texture(width: int, height: int) -> Image.Image:
    """Seeded blue-gray steel grain, independent of simulation randomness."""
    rng = np.random.default_rng(26020260905)
    shade = np.zeros((height, width))
    for cells, strength in ((5, 12), (18, 9), (75, 5)):
        noise = Image.fromarray(rng.integers(0, 256, (cells, cells), dtype=np.uint8))
        shade += (
            np.asarray(noise.resize((width, height), Image.Resampling.BICUBIC)) / 255 - 0.5
        ) * strength
    shade += rng.normal(0, 3, (height, width))
    yy, xx = np.mgrid[:height, :width]
    shade += 12 * (1 - ((xx / width - 0.4) ** 2 + (yy / height - 0.4) ** 2))
    return Image.fromarray(
        np.clip(np.array((32, 39, 43)) + shade[:, :, None], 0, 255).astype(np.uint8)
    )


@lru_cache(maxsize=1)
def _sheet() -> Image.Image:
    with Image.open(Path(__file__).with_name("laser_tag_sprite_sheet.png")) as image:
        sheet = image.convert("RGBA")
    # The generated source uses a pale checkerboard. Key only background pixels
    # connected to its border, preserving bright metal highlights inside actors.
    rgb = np.asarray(sheet)[:, :, :3].astype(int)
    candidate = (rgb.min(axis=2) > 210) & (np.ptp(rgb, axis=2) < 20)
    candidate |= np.asarray(sheet.getchannel("A")) == 0
    border = np.zeros(candidate.shape, dtype=bool)
    border[0, :] = candidate[0, :]
    border[-1, :] = candidate[-1, :]
    border[:, 0] = candidate[:, 0]
    border[:, -1] = candidate[:, -1]
    background = binary_propagation(border, mask=candidate)
    alpha = np.asarray(sheet.getchannel("A")).copy()
    alpha[background] = 0
    sheet.putalpha(Image.fromarray(alpha))
    return sheet


@lru_cache(maxsize=12)
def _actor(index: int, size: int) -> np.ndarray:
    sheet = _sheet()
    width = sheet.width // 2
    cell = sheet.crop((index * width, 0, (index + 1) * width, sheet.height))
    bounds = cell.getchannel("A").getbbox()
    if bounds:
        cell = cell.crop(bounds)
    cell.thumbnail((size - 6, size - 6), Image.Resampling.LANCZOS)
    output = Image.new("RGBA", (size, size))
    output.alpha_composite(cell, ((size - cell.width) // 2, (size - cell.height) // 2))
    return np.asarray(output)


def _render_robot_pillow(size_px: int = 64) -> np.ndarray:
    return _actor(0, size_px)


def _render_opponent_pillow(size_px: int = 64) -> np.ndarray:
    return _actor(1, size_px)


def draw_wall(image: Image.Image, bounds: tuple[float, float, float, float]) -> None:
    """Shade a raised steel block entirely within the existing wall footprint."""
    left, top, right, bottom = (round(v) for v in bounds)
    width, height = max(1, right - left + 1), max(1, bottom - top + 1)
    tile = metal_texture(128, 128).resize((width, height), Image.Resampling.BILINEAR)
    image.paste(tile, (left, top))
    d = ImageDraw.Draw(image)
    d.rectangle((left, top, right, bottom), outline=(15, 19, 22), width=2)
    if width > 18 and height > 18:
        d.line(
            (left + 3, bottom - 3, left + 3, top + 3, right - 3, top + 3),
            fill=(135, 146, 149),
            width=3,
        )
        d.line(
            (left + 4, bottom - 4, right - 4, bottom - 4, right - 4, top + 4),
            fill=(12, 16, 18),
            width=5,
        )
        d.rectangle((left + 8, top + 8, right - 8, bottom - 8), outline=(81, 91, 95), width=1)
        for x, y in (
            (left + 6, top + 6),
            (right - 6, top + 6),
            (left + 6, bottom - 6),
            (right - 6, bottom - 6),
        ):
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(163, 169, 166))
