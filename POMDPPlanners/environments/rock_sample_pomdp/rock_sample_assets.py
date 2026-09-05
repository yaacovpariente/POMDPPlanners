# SPDX-License-Identifier: MIT
"""Cached terrain and packaged sprite art for RockSample episodes."""

from functools import lru_cache
from pathlib import Path

from PIL import Image


@lru_cache(maxsize=32)
def sprite(name: str, size: int) -> Image.Image:
    """Decode and resize artwork once for each cell size."""
    with Image.open(Path(__file__).with_name("visualization_assets") / f"{name}.png") as source:
        result = source.convert("RGBA")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result


@lru_cache(maxsize=4)
def terrain(width: int, height: int) -> Image.Image:
    """Resize packaged orange soil once for the fixed scene."""
    with Image.open(Path(__file__).with_name("visualization_assets") / "terrain.png") as source:
        return source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
