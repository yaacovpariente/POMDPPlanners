# SPDX-License-Identifier: MIT
"""Cached sandstone terrain and packaged rover, crate and goal sprites."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@lru_cache(maxsize=4)
def stone_texture(width: int, height: int, dark: bool = False) -> Image.Image:
    """Layer seeded stone grain over broad warm light, once per canvas size."""
    rng = np.random.default_rng(20260905)
    shade = np.zeros((height, width), dtype=float)
    for cells, strength in ((5, 19), (14, 12), (45, 7), (140, 4)):
        noise = Image.fromarray(rng.integers(0, 256, (cells, cells), dtype=np.uint8))
        shade += (
            np.asarray(noise.resize((width, height), Image.Resampling.BICUBIC)) / 255 - 0.5
        ) * strength
    shade += rng.normal(0, 2.6, (height, width))
    yy, xx = np.mgrid[:height, :width]
    light = 1 - ((xx / width - 0.43) ** 2 + (yy / height - 0.35) ** 2)
    shade += light * 23
    base = np.array((49, 42, 32) if dark else (155, 137, 109))
    return Image.fromarray(np.clip(base + shade[:, :, None], 0, 255).astype(np.uint8))


@lru_cache(maxsize=1)
def _sprite_sheet() -> Image.Image:
    with Image.open(Path(__file__).with_name("push_sprite_sheet.png")) as source:
        return source.convert("RGBA")


@lru_cache(maxsize=24)
def sprite(kind: str, size: int) -> Image.Image:
    """Crop a packaged cell, retain its alpha, and cache its scaled shadowed art."""
    sheet = _sprite_sheet()
    index = ("robot", "object", "target").index(kind)
    width = sheet.width // 3
    cell = sheet.crop((index * width, 0, (index + 1) * width, sheet.height))
    bounds = cell.getchannel("A").getbbox()
    if bounds is not None:
        cell = cell.crop(bounds)
    cell.thumbnail((size - 8, size - 8), Image.Resampling.LANCZOS)
    output = Image.new("RGBA", (size, size))
    position = ((size - cell.width) // 2, (size - cell.height) // 2 - 2)
    shadow = Image.new("RGBA", (size, size), (16, 10, 4, 0))
    shadow.putalpha(Image.new("L", (size, size)))
    shadow.paste(
        (20, 12, 4, 110),
        (
            position[0] + 3,
            position[1] + 5,
            position[0] + 3 + cell.width,
            position[1] + 5 + cell.height,
        ),
        cell.getchannel("A"),
    )
    output.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(2)))
    output.alpha_composite(cell, position)
    return output


def paste_sprite(canvas: Image.Image, kind: str, center: tuple[float, float], size: int) -> None:
    asset = sprite(kind, size)
    canvas.paste(asset, (round(center[0] - size / 2), round(center[1] - size / 2)), asset)


def paste_obstacle(
    canvas: Image.Image, bounds: tuple[float, float, float, float], circle: bool
) -> None:
    """Texture an obstacle inside its collision outline without changing its footprint."""
    left, top, right, bottom = (round(value) for value in bounds)
    width, height = max(1, right - left + 1), max(1, bottom - top + 1)
    stone = np.asarray(
        stone_texture(128, 128).resize((width, height), Image.Resampling.BILINEAR)
    ).astype(float)
    stone *= np.array((0.84, 0.51, 0.38))
    yy, xx = np.mgrid[:height, :width]
    stone += (8 - 14 * (xx / width + yy / height))[:, :, None]
    texture = Image.fromarray(np.clip(stone, 0, 255).astype(np.uint8))
    mask = Image.new("L", (width, height))
    md = ImageDraw.Draw(mask)
    if circle:
        md.ellipse((0, 0, width - 1, height - 1), fill=255)
    else:
        md.rectangle((0, 0, width - 1, height - 1), fill=255)
    canvas.paste(texture, (left, top), mask)
    d = ImageDraw.Draw(canvas)
    if circle:
        d.ellipse((left, top, right, bottom), outline=(74, 41, 29), width=2)
        if width > 8 and height > 8:
            d.arc(
                (left + 3, top + 3, right - 3, bottom - 3), 190, 290, fill=(186, 126, 83), width=2
            )
            d.arc((left + 3, top + 3, right - 3, bottom - 3), 10, 110, fill=(68, 37, 26), width=3)
    else:
        d.rectangle((left, top, right, bottom), outline=(74, 41, 29), width=2)
        if width > 8 and height > 8:
            d.line(
                (left + 3, bottom - 3, right - 3, bottom - 3, right - 3, top + 3),
                fill=(68, 37, 26),
                width=3,
            )
            d.line(
                (left + 3, bottom - 5, left + 3, top + 3, right - 5, top + 3),
                fill=(186, 126, 83),
                width=2,
            )
