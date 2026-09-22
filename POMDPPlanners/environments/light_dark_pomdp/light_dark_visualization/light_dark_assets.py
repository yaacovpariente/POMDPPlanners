# SPDX-License-Identifier: MIT

"""Cached terrain and packaged artwork for the Light-Dark visualizer.

The ground uses a fixed seed; sprites use packaged PNGs. Identical arguments
therefore return identical pixels. The golden GIF test compares bytes, and
:func:`functools.lru_cache` avoids rebuilding sprites within a process.

Nothing in this module imports Matplotlib.  The Light-Dark environment imports
its visualizer at module import time, so pulling Matplotlib in here would cost
every planner run that never renders a frame.
"""

from functools import lru_cache
from importlib.resources import files
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as PIL_VERSION

# Seed for every procedural texture. Fixed so renders are reproducible.
ASSET_SEED = 20260905

# Supersampling factor for hand-drawn sprites. Sprites are drawn large and
# downsampled, which is how they get smooth edges without an AA-capable
# drawing backend.
_SS = 4

# --- palette -----------------------------------------------------------------

COLOR_PAGE = (18, 16, 15)
COLOR_GROUND_DARK = (30, 28, 29)
COLOR_GROUND_LIGHT = (112, 97, 79)
COLOR_LIGHT_WARM = (255, 233, 190)
COLOR_HAZARD = (214, 44, 38)
COLOR_PATH = (232, 78, 62)
COLOR_ROVER_BODY = (176, 44, 36)
COLOR_BEACON_MARK = (60, 105, 220)
COLOR_GOAL = (46, 190, 74)
COLOR_START = (222, 52, 46)
COLOR_OBSTACLE_DOT = (24, 24, 26)
COLOR_BELIEF = (255, 214, 46)
COLOR_TEXT = (226, 219, 208)
COLOR_TEXT_DIM = (168, 160, 150)
COLOR_PANEL = (30, 28, 27)
COLOR_PANEL_EDGE = (92, 86, 78)


# --- fonts -------------------------------------------------------------------


@lru_cache(maxsize=8)
def get_font(size: int):
    """Return a font of roughly ``size`` pixels.

    Pillow >= 10.1 can scale its bundled Aileron face, which keeps text legible
    and identical on every machine with the same Pillow.

    Older Pillow ignores the size and hands back one fixed ~11px bitmap face,
    which would silently collapse the 30px title, the 18px legend and the 16px
    overlay text to the same size.  A render that quietly looks wrong is worse
    than one that stops, so this raises instead of falling back.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - only on Pillow < 10.1
        raise RuntimeError(
            "The Light-Dark renderer needs Pillow >= 10.1 for scalable default "
            f"fonts; this environment has Pillow {PIL_VERSION}."
        ) from exc


# --- noise -------------------------------------------------------------------


def _value_noise(rng: np.random.Generator, height: int, width: int, cells: int) -> np.ndarray:
    """Smooth value noise on a ``cells x cells`` lattice, bilinearly upsampled.

    Cheaper than any gradient-noise implementation and good enough for stone:
    the visible structure comes from stacking octaves, not from one layer.
    """
    lattice = rng.random((cells + 1, cells + 1))
    ys = np.linspace(0.0, cells, height)
    xs = np.linspace(0.0, cells, width)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, cells)
    x1 = np.minimum(x0 + 1, cells)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    # Smoothstep removes the lattice-aligned creases plain bilinear leaves.
    ty = ty * ty * (3.0 - 2.0 * ty)
    tx = tx * tx * (3.0 - 2.0 * tx)
    top = lattice[np.ix_(y0, x0)] * (1.0 - tx) + lattice[np.ix_(y0, x1)] * tx
    bottom = lattice[np.ix_(y1, x0)] * (1.0 - tx) + lattice[np.ix_(y1, x1)] * tx
    return top * (1.0 - ty) + bottom * ty


@lru_cache(maxsize=4)
def ground_texture(width: int, height: int, seed: int = ASSET_SEED) -> np.ndarray:
    """Unlit dark stone ground as a float RGB array in ``[0, 1]``.

    Returned unlit so the renderer can multiply in the beacon light field and
    get a single coherent lighting pass instead of alpha-stacking circles.
    """
    rng = np.random.default_rng(seed)
    fbm = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for cells in (3, 6, 12, 24, 48, 96):
        fbm += amplitude * _value_noise(rng, height, width, cells)
        total += amplitude
        amplitude *= 0.55
    fbm /= total
    fbm = (fbm - fbm.min()) / max(float(np.ptp(fbm)), 1e-9)

    # Fine grain: per-pixel speckle keeps the stone from looking like a blur.
    grain = rng.normal(0.0, 0.060, size=(height, width))
    shade = np.clip(fbm * 1.05 - 0.02 + grain, 0.0, 1.0)

    dark = np.array(COLOR_GROUND_DARK, dtype=np.float64) / 255.0
    light = np.array(COLOR_GROUND_LIGHT, dtype=np.float64) / 255.0
    rgb = dark[None, None, :] + (light - dark)[None, None, :] * shade[:, :, None]

    # Hairline cracks: ridged noise, so only the narrow valleys darken.
    ridge = 1.0 - np.abs(2.0 * _value_noise(rng, height, width, 72) - 1.0)
    cracks = np.clip((ridge - 0.972) / 0.028, 0.0, 1.0) ** 1.6
    rgb *= (1.0 - 0.42 * cracks)[:, :, None]

    # Scattered pebbles and grit as small round discs. Rectangles here read as
    # visible square blocks at this canvas size, so the disc mask matters.
    n_specks = max(60, (width * height) // 2600)
    sx = rng.integers(0, width, size=n_specks)
    sy = rng.integers(0, height, size=n_specks)
    sr = rng.integers(1, 4, size=n_specks)
    sv = rng.uniform(-0.07, 0.10, size=n_specks)
    for cx, cy, r, v in zip(sx, sy, sr, sv):
        x_lo, x_hi = max(0, cx - r), min(width, cx + r + 1)
        y_lo, y_hi = max(0, cy - r), min(height, cy + r + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        patch = rgb[y_lo:y_hi, x_lo:x_hi]
        patch[mask] = np.clip(patch[mask] + v, 0.0, 1.0)

    return rgb


# --- sprites -----------------------------------------------------------------


def _new_sprite(size: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size * _SS, size * _SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=48)
def _art_sprite(name: str, size: int) -> Image.Image:
    """Load packaged RGBA art once per size; callers must not modify the cache."""
    asset = files(__package__).joinpath("art", name + ".png")
    with asset.open("rb") as stream:
        with Image.open(stream) as source:
            rgba = source.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] != 0:
        raise ValueError(f"Light-Dark sprite {name} has no transparent background")
    return rgba.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=16)
def rover_sprite(size: int) -> Image.Image:
    """Detailed red mechanical rover, facing world +X (right)."""
    return _art_sprite("rover", max(12, int(size)))


@lru_cache(maxsize=64)
def rover_sprite_facing(size: int, heading_deg: int) -> Image.Image:
    """Rover rotated to face ``heading_deg`` (world degrees, 0 = +x, CCW).

    Headings are bucketed by the caller so the cache stays small; a rover that
    points where it is about to move is the single cheapest readability win in
    the frame.
    """
    base = rover_sprite(size)
    if heading_deg % 360 == 0:
        return base
    return base.rotate(heading_deg, resample=Image.Resampling.BICUBIC, expand=False)


@lru_cache(maxsize=16)
def beacon_sprite(size: int) -> Image.Image:
    """Bronze lamp with a luminous lens and blue navigation badge."""
    return _art_sprite("beacon", max(10, int(size)))


@lru_cache(maxsize=16)
def goal_sprite(size: int) -> Image.Image:
    """Raised green enamel star on a machined metal base."""
    return _art_sprite("goal", max(10, int(size)))


@lru_cache(maxsize=16)
def start_sprite(size: int) -> Image.Image:
    """Red start disc with a dark ring so it reads on a bright light pool."""
    size = max(6, int(size))
    img, d = _new_sprite(size)
    s = size * _SS
    d.ellipse([0, 0, s - 1, s - 1], fill=(70, 14, 12, 255))
    d.ellipse([s * 0.12, s * 0.12, s * 0.88, s * 0.88], fill=tuple(COLOR_START) + (255,))
    d.ellipse([s * 0.28, s * 0.24, s * 0.56, s * 0.46], fill=(255, 150, 142, 200))
    return _finish(img, size)


@lru_cache(maxsize=32)
def glow_sprite(radius_px: int, color: Tuple[int, int, int], power: float = 2.4) -> Image.Image:
    """Soft radial glow used for the belief particles and the goal halo."""
    radius_px = max(2, int(radius_px))
    d = 2 * radius_px + 1
    yy, xx = np.mgrid[0:d, 0:d]
    r = np.sqrt((xx - radius_px) ** 2 + (yy - radius_px) ** 2) / radius_px
    alpha = np.clip(1.0 - r, 0.0, 1.0) ** power
    rgba = np.zeros((d, d, 4), dtype=np.uint8)
    rgba[:, :, 0] = color[0]
    rgba[:, :, 1] = color[1]
    rgba[:, :, 2] = color[2]
    rgba[:, :, 3] = (alpha * 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


__all__ = [
    "ASSET_SEED",
    "COLOR_BEACON_MARK",
    "COLOR_BELIEF",
    "COLOR_GOAL",
    "COLOR_GROUND_DARK",
    "COLOR_GROUND_LIGHT",
    "COLOR_HAZARD",
    "COLOR_LIGHT_WARM",
    "COLOR_OBSTACLE_DOT",
    "COLOR_PAGE",
    "COLOR_PANEL",
    "COLOR_PANEL_EDGE",
    "COLOR_PATH",
    "COLOR_ROVER_BODY",
    "COLOR_START",
    "COLOR_TEXT",
    "COLOR_TEXT_DIM",
    "beacon_sprite",
    "get_font",
    "glow_sprite",
    "goal_sprite",
    "ground_texture",
    "rover_sprite",
    "rover_sprite_facing",
    "start_sprite",
]
