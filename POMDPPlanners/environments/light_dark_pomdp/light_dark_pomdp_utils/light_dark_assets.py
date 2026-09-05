# SPDX-License-Identifier: MIT

"""Deterministic procedural art assets for the Light-Dark visualizer.

Every asset here is a pure function of its arguments plus a fixed seed, so the
same call always returns the same pixels.  That matters twice over: the golden
GIF test compares bytes, and the renderer leans on :func:`functools.lru_cache`
to build each sprite once and paste it many times.

Nothing in this module imports Matplotlib.  The Light-Dark environment imports
its visualizer at module import time, so pulling Matplotlib in here would cost
every planner run that never renders a frame.
"""

from functools import lru_cache
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


@lru_cache(maxsize=16)
def rover_sprite(size: int) -> Image.Image:
    """Top-down rover: four wheels, a red hull, a cabin and two headlights."""
    size = max(12, int(size))
    img, d = _new_sprite(size)
    s = size * _SS
    u = s / 32.0  # design grid: the sprite was laid out on 32 units

    wheel = (28, 26, 28, 255)
    wheel_hi = (58, 54, 56, 255)
    for wx in (2.0, 22.0):
        for wy in (3.0, 21.0):
            d.rounded_rectangle(
                [wx * u, wy * u, (wx + 8) * u, (wy + 8) * u],
                radius=2.6 * u,
                fill=wheel,
            )
            d.rounded_rectangle(
                [(wx + 1.2) * u, (wy + 1.2) * u, (wx + 4.0) * u, (wy + 6.8) * u],
                radius=1.2 * u,
                fill=wheel_hi,
            )

    body = tuple(COLOR_ROVER_BODY) + (255,)
    body_dark = (108, 24, 20, 255)
    body_hi = (226, 92, 76, 255)
    d.rounded_rectangle([4.0 * u, 6.0 * u, 28.0 * u, 26.0 * u], radius=4.0 * u, fill=body_dark)
    d.rounded_rectangle([5.4 * u, 7.4 * u, 26.6 * u, 24.6 * u], radius=3.4 * u, fill=body)
    d.rounded_rectangle([7.0 * u, 9.0 * u, 25.0 * u, 13.5 * u], radius=2.0 * u, fill=body_hi)

    cabin = (44, 48, 58, 255)
    cabin_glass = (120, 168, 196, 255)
    d.rounded_rectangle([10.0 * u, 12.0 * u, 22.0 * u, 22.0 * u], radius=2.6 * u, fill=cabin)
    d.rounded_rectangle([11.4 * u, 13.4 * u, 20.6 * u, 18.4 * u], radius=1.8 * u, fill=cabin_glass)

    lamp = (255, 244, 206, 255)
    d.ellipse([27.0 * u, 10.6 * u, 30.2 * u, 13.8 * u], fill=lamp)
    d.ellipse([27.0 * u, 18.2 * u, 30.2 * u, 21.4 * u], fill=lamp)

    return _finish(img, size)


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
    """Beacon lamp: metal base, glowing glass, and the blue legend marker."""
    size = max(10, int(size))
    img, d = _new_sprite(size)
    s = size * _SS
    u = s / 32.0

    d.ellipse([4.0 * u, 20.0 * u, 28.0 * u, 30.0 * u], fill=(30, 28, 27, 200))
    # Stacked trapezoid rings, light at the top, give a cheap metallic shade.
    rings = [
        (6.0, 22.0, 26.0, 28.5, (74, 66, 58)),
        (7.5, 19.5, 24.5, 24.5, (98, 88, 76)),
        (9.0, 17.0, 23.0, 21.5, (126, 113, 96)),
    ]
    for x0, y0, x1, y1, col in rings:
        d.rounded_rectangle(
            [x0 * u, y0 * u, x1 * u, y1 * u], radius=1.6 * u, fill=col + (255,)
        )
    d.ellipse([9.5 * u, 12.5 * u, 22.5 * u, 21.5 * u], fill=(252, 240, 206, 255))
    d.ellipse([12.0 * u, 14.5 * u, 20.0 * u, 19.5 * u], fill=(255, 253, 240, 255))

    tri = [(16.0 * u, 3.5 * u), (23.0 * u, 15.0 * u), (9.0 * u, 15.0 * u)]
    d.polygon(tri, fill=tuple(COLOR_BEACON_MARK) + (255,), outline=(16, 30, 78, 255))

    return _finish(img, size)


def _star_points(cx: float, cy: float, outer: float, inner: float, n: int = 5):
    pts = []
    for i in range(2 * n):
        r = outer if i % 2 == 0 else inner
        theta = -np.pi / 2.0 + i * np.pi / n
        pts.append((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    return pts


@lru_cache(maxsize=16)
def goal_sprite(size: int) -> Image.Image:
    """Green five-point star with a dark rim, matching the mockup legend."""
    size = max(10, int(size))
    img, d = _new_sprite(size)
    s = size * _SS
    c = s / 2.0
    d.polygon(
        _star_points(c, c, c * 0.96, c * 0.42),
        fill=(16, 74, 26, 255),
    )
    d.polygon(
        _star_points(c, c, c * 0.84, c * 0.36),
        fill=tuple(COLOR_GOAL) + (255,),
    )
    d.polygon(
        _star_points(c, c, c * 0.46, c * 0.20),
        fill=(150, 240, 160, 255),
    )
    return _finish(img, size)


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
