# SPDX-License-Identifier: MIT
"""Cached concrete and shaded ant art, generated from a fixed seed."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@lru_cache(maxsize=4)
def concrete_texture(width: int, height: int) -> Image.Image:
    """Layer coarse stone variation, mineral grain, pits, and fine cracks."""
    rng = np.random.default_rng(20260905)
    shade = np.full((height, width), 173.0)
    for cells, strength in ((5, 32), (17, 18), (61, 11), (190, 7)):
        noise = Image.fromarray(rng.integers(0, 256, (cells, cells), dtype=np.uint8))
        field = np.asarray(noise.resize((width, height), Image.Resampling.BICUBIC))
        shade += (field.astype(float) / 255 - 0.5) * strength
    shade += rng.normal(0, 6, (height, width))
    yy, xx = np.mgrid[:height, :width]
    light = 14 * np.exp(
        -((xx - width * 0.35) ** 2 + (yy - height * 0.25) ** 2) / (width * 0.5) ** 2
    )
    shade += light - 15 * ((xx / width - 0.5) ** 2 + (yy / height - 0.5) ** 2)
    rgb = np.stack((shade + 7, shade + 5, shade), axis=-1)
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    draw = ImageDraw.Draw(image)
    for _ in range(width * height // 700):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        r = int(rng.integers(1, 3))
        draw.ellipse((x, y, x + r * 2, y + r), fill=(137, 138, 134))
        draw.line((x, y + r, x + r, y + r), fill=(203, 201, 192))
    for _ in range(28):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        points = [(x, y)]
        for _ in range(int(rng.integers(3, 8))):
            x += int(rng.integers(-10, 11))
            y += int(rng.integers(3, 13))
            points.append((x, y))
        draw.line(points, fill=(146, 146, 138), width=1)
    return image


@lru_cache(maxsize=4)
def ant_sprite(size: int = 100) -> Image.Image:
    """Six jointed legs and three glossy blue body segments with cast shadow."""
    scale = 3
    extent = size * scale
    image = Image.new("RGBA", (extent, extent))
    shadow = Image.new("RGBA", image.size)
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((extent * 0.30, extent * 0.30, extent * 0.75, extent * 0.92), fill=(5, 9, 15, 95))
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))
    draw = ImageDraw.Draw(image)
    for direction in (-1, 1):
        for root_y, knee_y, foot_y in ((0.43, 0.32, 0.24), (0.50, 0.49, 0.52), (0.57, 0.66, 0.78)):
            points = [
                (extent * 0.5, extent * root_y),
                (extent * (0.5 + direction * 0.18), extent * knee_y),
                (extent * (0.5 + direction * 0.30), extent * foot_y),
            ]
            draw.line(points, fill=(9, 24, 39, 255), width=9, joint="curve")
            draw.line([(x - 1, y - 2) for x, y in points], fill=(65, 108, 153, 255), width=3)
        draw.line(
            [
                (extent * (0.5 + direction * 0.07), extent * 0.32),
                (extent * (0.5 + direction * 0.11), extent * 0.18),
                (extent * (0.5 + direction * 0.17), extent * 0.14),
            ],
            fill=(17, 29, 41, 255),
            width=5,
            joint="curve",
        )
    for cx, cy, rx, ry in (
        (0.5, 0.68, 0.13, 0.19),
        (0.5, 0.48, 0.09, 0.10),
        (0.5, 0.33, 0.12, 0.12),
    ):
        yy, xx = np.mgrid[:extent, :extent]
        nx, ny = (xx / extent - cx) / rx, (yy / extent - cy) / ry
        r2 = nx * nx + ny * ny
        mask = r2 <= 1
        nz = np.sqrt(np.clip(1 - r2, 0, 1))
        diffuse = np.clip(-0.45 * nx - 0.50 * ny + 0.74 * nz, 0, 1)
        highlight = np.exp(-((nx + 0.35) ** 2 + (ny + 0.42) ** 2) / 0.055)
        rim = np.clip(nz * 2.3, 0, 1)
        rgba = np.zeros((extent, extent, 4), dtype=np.uint8)
        for channel, base in enumerate((18, 97, 195)):
            rgba[:, :, channel] = np.uint8(
                np.clip((base * (0.20 + 0.80 * diffuse) + highlight * 150) * rim, 0, 255)
            )
        rgba[:, :, 3] = mask * 255
        image.alpha_composite(Image.fromarray(rgba))
    return image.resize((size, size), Image.Resampling.LANCZOS)
