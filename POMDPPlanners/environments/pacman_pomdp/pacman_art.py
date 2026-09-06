# SPDX-License-Identifier: MIT
"""Bounded, deterministic PacMan artwork caches; callers must not mutate images."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@lru_cache(maxsize=32)
def tile(size: int, wall: bool) -> Image.Image:
    """Dark stone floor or raised blue wall, contained within one native cell."""
    scale = max(size, 96)
    yy, xx = np.mgrid[:scale, :scale] / scale
    rng = np.random.default_rng(9042)
    grain = rng.normal(0, 1.2, (scale, scale))
    bands = np.sin(xx * 12 + yy * 8) * np.sin(yy * 17 - xx * 5)
    shade = (1 - yy) * 0.45 + (1 - xx) * 0.2 + bands * 0.06
    base = np.array([8, 15, 31]) if not wall else np.array([4, 22, 128])
    spread = np.array([7, 12, 19]) if not wall else np.array([12, 52, 108])
    rgb = np.clip(base + shade[..., None] * spread + grain[..., None], 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    for _ in range(8):
        x, y = int(rng.integers(scale)), int(rng.integers(scale))
        color = (26, 42, 61, 255) if not wall else (24, 70, 176, 255)
        draw.line(
            [(x, y), (x + scale * 0.12, y - scale * 0.08), (x + scale * 0.2, y - scale * 0.1)],
            fill=color,
        )
    edge = max(1, scale // 48)
    draw.rounded_rectangle(
        (1, 1, scale - 2, scale - 2), radius=scale * 0.08, outline=(2, 4, 8), width=edge * 2
    )
    bright = (134, 193, 255) if wall else (80, 103, 134)
    dark = (12, 25, 92) if wall else (8, 14, 24)
    draw.rounded_rectangle(
        (edge * 2, edge * 2, scale - edge * 2 - 1, scale - edge * 2 - 1),
        radius=scale * 0.06,
        outline=bright,
        width=edge,
    )
    draw.line(
        (edge * 3, scale - edge * 3, scale - edge * 3, scale - edge * 3), fill=dark, width=edge * 2
    )
    draw.line(
        (scale - edge * 3, edge * 3, scale - edge * 3, scale - edge * 3), fill=dark, width=edge * 2
    )
    if wall:
        draw.line(
            (edge * 4, edge * 4, scale - edge * 4, edge * 4), fill=(208, 237, 255), width=edge
        )
    return image.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def character(name: str, size: int, direction: str = "east") -> Image.Image:
    """One shaded entity on an alpha tile, decoded only once per size/heading."""
    path = Path(__file__).with_name("img") / f"{name}_polished.png"
    with Image.open(path) as source:
        art = source.convert("RGBA")
    inset = max(1, round(size * 0.12))
    art.thumbnail((max(1, size - inset * 2), max(1, size - inset * 2)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size))
    canvas.alpha_composite(art, ((size - art.width) // 2, (size - art.height) // 2))
    if name == "player":
        angle = {"east": 0, "north": 90, "west": 180, "south": 270}.get(direction, 0)
        if angle:
            canvas = canvas.rotate(angle)
    return canvas


@lru_cache(maxsize=64)
def ghost(size: int, index: int) -> Image.Image:
    """Keep per-ghost colors while preserving the sprite's eyes and shading."""
    source = character("ghost", size)
    if index % 8 == 0:
        return source
    colors = (
        (1, 0, 0),
        (0, 1, 0),
        (0.3, 0.55, 1),
        (1, 0.3, 1),
        (1, 0.5, 0),
        (0, 1, 1),
        (1, 1, 0),
        (0.6, 0.3, 1),
    )
    rgba = np.array(source)
    rgb = rgba[..., :3].astype(float)
    red = np.maximum(0, rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]))
    rgb += red[..., None] * (np.array(colors[index % 8]) - (1, 0, 0))
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


@lru_cache(maxsize=16)
def pellet(size: int) -> Image.Image:
    """A white pellet with a soft blue pool of light, kept inside its cell."""
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
    alpha = np.clip(125 * np.exp(-((radius / max(size * 0.12, 1)) ** 2)), 0, 255)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = (93, 130, 255)
    rgba[..., 3] = alpha.astype(np.uint8)
    image = Image.fromarray(rgba)
    draw = ImageDraw.Draw(image)
    r = max(1, size * 0.085)
    center = (size - 1) / 2
    draw.ellipse((center - r, center - r, center + r, center + r), fill=(255, 255, 245, 255))
    return image
