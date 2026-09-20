# SPDX-License-Identifier: MIT

"""Palette, textures, sprites and chrome for the occupancy-grid mapping render.

Everything here is drawn with Pillow and numpy, and everything is a pure
function of its arguments: procedural textures take a fixed seed, sprites are
drawn at :data:`_SUPERSAMPLE` times their final size and downsampled, and each
result is memoised with :func:`functools.lru_cache`.  That matters twice over.
The renderer's golden-file test compares GIF bytes, so a texture that reseeded
itself per call would fail it; and the same panel chrome is drawn for every
episode, so building it once per process is most of what keeps the render
cheap.

Nothing in this module imports Matplotlib.  The environment reaches its
visualizer only through a lazy import, and the point of that laziness is lost
if the renderer then drags a plotting stack in behind it.

Callers must treat every cached return value as read-only; copy before writing.

Functions:
    get_font: A scalable font of a given pixel size.
    mix_colors: Linear blend of two RGB triples.
    value_noise: Smooth value noise, bilinearly upsampled from a lattice.
    static_field: Unit-mean multiplicative speckle for uncertain cells.
    stone_shade: Shared stone relief for the ground-truth panel.
    stone_floor: Dark textured floor for the ground-truth panel.
    robot_sprite: The robot, facing north.
    robot_sprite_facing: The robot, rotated to a heading index.
    draw_bezel: A machined metal surround with corner bolts.
    draw_gradient_bar: A colour ramp drawn as a stepped scale.
    draw_screen_edge: The inner rim of a panel's screen.
"""

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as PIL_VERSION

#: Seed for every procedural texture in this module. Fixed so that two renders
#: of the same episode are byte-for-byte identical, which the golden-file test
#: and the determinism test both depend on.
ASSET_SEED = 20260910

#: Sprites are drawn this many times larger and then downsampled, which is how
#: they get smooth edges from a drawing backend with no antialiasing.
_SUPERSAMPLE = 4

# --- palette -----------------------------------------------------------------
#
# One dark control-room page, three panel materials on top of it. The three
# panels are deliberately different materials rather than three shades of one
# ramp: they mean different things, and a reader must never have to check a
# title to tell the observed map from the belief.

#: Page background.
COLOR_PAGE = (13, 19, 24)
#: Body of a metal panel.
COLOR_METAL = (32, 42, 50)
#: Lit edge of a metal panel, top and left.
COLOR_METAL_LIGHT = (86, 104, 116)
#: Shaded edge of a metal panel, bottom and right.
COLOR_METAL_DARK = (16, 22, 27)
#: Bolt head in a panel corner.
COLOR_BOLT = (108, 124, 134)

#: Primary text.
COLOR_TEXT = (222, 233, 238)
#: Secondary text: descriptions, tick labels, units.
COLOR_TEXT_DIM = (134, 156, 168)

# Observed map panel. A diverging ramp about p = 0.5, because that is what
# the quantity does: log-odds start at zero and evidence pushes a cell either
# way. Unknown is fog, free is dark swept floor, occupied is hot amber.
#: Observed map, believed free (p = 0).
COLOR_CHART_FREE = (16, 36, 43)
#: Observed map, unknown (p = 0.5).
COLOR_CHART_FOG = (84, 95, 105)
#: Observed map, believed occupied (p = 1).
COLOR_CHART_OCCUPIED = (240, 166, 60)

# Map estimate panel. A cyan holographic readout with scanlines, so it cannot
# be mistaken for the amber chart beside it even at a glance.
#: Belief panel, probability 0.
COLOR_HOLO_LOW = (8, 32, 42)
#: Belief panel, probability 0.5.
COLOR_HOLO_MID = (24, 104, 124)
#: Belief panel, probability 1.
COLOR_HOLO_HIGH = (96, 226, 244)
#: Belief panel, no belief recorded for the step. Deliberately a neutral warm
#: grey rather than a dark cyan: "no belief" must not look like "probability
#: near zero", and on this ramp anything cyan would.
COLOR_HOLO_BLANK = (74, 74, 80)

# Ground truth. The level itself: dark stone floor and lit blocks.
#: Ground truth, free space.
COLOR_TRUTH_FLOOR = (27, 38, 45)
#: Ground truth, wall or obstacle.
COLOR_TRUTH_WALL = (140, 158, 172)

#: Robot body. A signal colour used by nothing else in the frame.
COLOR_ROBOT = (255, 82, 66)
#: Robot outline and tread.
COLOR_ROBOT_DARK = (58, 16, 14)
#: Robot cockpit and highlights.
COLOR_ROBOT_LIGHT = (255, 226, 196)
#: The path the robot has already driven.
COLOR_TRAIL = (255, 138, 84)
#: A measured sensor beam.
COLOR_BEAM = (126, 240, 208)

#: Filled part of the entropy meter: the entropy still to be driven out.
COLOR_METER_FILL = (240, 166, 60)
#: The meter's threshold mark, below which the map counts as resolved.
COLOR_METER_GOAL = (126, 240, 208)


def mix_colors(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear blend of two colours; ``t = 0`` gives ``a``.

    Args:
        a: Colour at ``t = 0``.
        b: Colour at ``t = 1``.
        t: Blend position, clamped to ``[0, 1]``.

    Returns:
        The blended RGB triple.
    """
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


# --- fonts -------------------------------------------------------------------


@lru_cache(maxsize=16)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a font of roughly ``size`` pixels.

    Pillow 10.1 and later can scale its bundled Aileron face, which keeps text
    identical on every machine carrying the same Pillow.  Older Pillow ignores
    the argument and returns one fixed bitmap face about 11 pixels tall, which
    would silently collapse the 33px title, the 19px panel headings and the
    11px tick labels into one size.  A render that quietly looks wrong is worse
    than one that stops, so this raises rather than falling back.

    Args:
        size: Requested height in pixels.

    Returns:
        A font object suitable for :meth:`PIL.ImageDraw.ImageDraw.text`.

    Raises:
        RuntimeError: If Pillow is too old to scale its default font, or was
            built without FreeType and can only return the bitmap face.
    """
    try:
        font = ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - only on Pillow < 10.1
        raise RuntimeError(
            "The occupancy-grid mapping renderer needs Pillow >= 10.1 for "
            f"scalable default fonts; this environment has Pillow {PIL_VERSION}."
        ) from exc
    if not isinstance(font, ImageFont.FreeTypeFont):  # pragma: no cover
        # Pillow without FreeType still accepts the size argument and still
        # returns the one fixed bitmap face, so the call succeeds and every
        # text size silently collapses. Checking the type is the only way to
        # tell that apart from a working scalable font.
        raise RuntimeError(
            "The occupancy-grid mapping renderer needs a FreeType-enabled "
            f"Pillow for scalable fonts; Pillow {PIL_VERSION} returned "
            f"{type(font).__name__}."
        )
    return font


# --- noise -------------------------------------------------------------------


def value_noise(rng: np.random.Generator, height: int, width: int, cells: int) -> np.ndarray:
    """Smooth value noise on a ``cells x cells`` lattice, bilinearly upsampled.

    Cheaper than any gradient-noise implementation and good enough for stone:
    the visible structure comes from stacking octaves, not from one layer.

    Args:
        rng: Source of the lattice values.
        height: Output height in pixels.
        width: Output width in pixels.
        cells: Lattice resolution.

    Returns:
        ``(height, width)`` array in ``[0, 1]``.
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
def static_field(width: int, height: int, seed: int = ASSET_SEED) -> np.ndarray:
    """Multiplicative speckle, mean one, for cells the robot is unsure about.

    The observed-map panel scales this by each cell's uncertainty, so an
    unknown cell fizzes like untuned signal and a resolved cell is flat.  That
    is the cheapest honest way to draw confidence: the picture gets calmer
    exactly as the entropy the reward pays for falls.

    Args:
        width: Panel width in pixels.
        height: Panel height in pixels.
        seed: Noise seed. The default keeps renders reproducible.

    Returns:
        ``(height, width)`` array of positive multipliers around ``1.0``.
    """
    rng = np.random.default_rng(seed + 1)
    fine = rng.normal(0.0, 0.11, size=(height, width))
    coarse = value_noise(rng, height, width, 40) - 0.5
    return np.clip(1.0 + fine + 0.22 * coarse, 0.45, 1.6)


@lru_cache(maxsize=4)
def stone_shade(width: int, height: int, seed: int = ASSET_SEED) -> np.ndarray:
    """Multiplicative stone relief, roughly unit-mean, shared by floor and wall.

    Floor and wall cells in the ground-truth panel are the same rock under
    different light, so they are shaded from one field. Two independent fields
    would make the two materials look like two unrelated images stitched at the
    cell boundary.

    Args:
        width: Panel width in pixels.
        height: Panel height in pixels.
        seed: Noise seed. The default keeps renders reproducible.

    Returns:
        ``(height, width)`` array of positive multipliers.
    """
    rng = np.random.default_rng(seed)
    fbm = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for cells in (4, 8, 16, 32, 64):
        fbm += amplitude * value_noise(rng, height, width, cells)
        total += amplitude
        amplitude *= 0.55
    fbm /= total
    fbm = (fbm - fbm.min()) / max(float(np.ptp(fbm)), 1e-9)
    grain = rng.normal(0.0, 0.045, size=(height, width))
    shade = np.clip(0.72 + 0.55 * fbm + grain, 0.05, 1.6)

    # Hairline cracks: ridged noise, so only the narrow valleys darken.
    ridge = 1.0 - np.abs(2.0 * value_noise(rng, height, width, 56) - 1.0)
    cracks = np.clip((ridge - 0.965) / 0.035, 0.0, 1.0) ** 1.6
    return shade * (1.0 - 0.45 * cracks)


def stone_floor(width: int, height: int, seed: int = ASSET_SEED) -> np.ndarray:
    """Dark stone floor for the ground-truth panel, as float RGB in ``[0, 1]``.

    Args:
        width: Panel width in pixels.
        height: Panel height in pixels.
        seed: Noise seed. The default keeps renders reproducible.

    Returns:
        ``(height, width, 3)`` array in ``[0, 1]``.
    """
    base = np.array(COLOR_TRUTH_FLOOR, dtype=np.float64) / 255.0
    shade = stone_shade(width, height, seed)
    return np.clip(base[None, None, :] * shade[:, :, None], 0.0, 1.0)


# --- sprites -----------------------------------------------------------------


def _new_sprite(size: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (size * _SUPERSAMPLE, size * _SUPERSAMPLE), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


@lru_cache(maxsize=16)
def robot_sprite(size: int) -> Image.Image:
    """The mapping robot, drawn facing north (up).

    Drawn here rather than shipped as a PNG because it is simple enough that
    the code is smaller than the asset, and a procedural sprite scales to any
    grid size without a resampling artefact.

    Legibility drives the shape. At the default grid the robot is about forty
    pixels across, so it is built from three marks a reader can still separate
    at that size: a heavy dark outline that holds against both the pale walls
    of the truth panel and the amber cells of the observed one, a bright body,
    and a cream nose wedge that is unambiguous about which way it points.

    Args:
        size: Final sprite width and height in pixels.

    Returns:
        A square RGBA image. Cached; callers must not modify it.
    """
    size = max(10, int(size))
    image, draw = _new_sprite(size)
    span = size * _SUPERSAMPLE

    def box(x0: float, y0: float, x1: float, y1: float):
        return [span * x0, span * y0, span * x1, span * y1]

    # Soft contact shadow, so the robot sits on the map instead of floating.
    draw.ellipse(box(0.16, 0.72, 0.84, 0.96), fill=(0, 0, 0, 70))

    # Treads.
    for x0, x1 in ((0.05, 0.24), (0.76, 0.95)):
        draw.rounded_rectangle(
            box(x0, 0.26, x1, 0.86), radius=span * 0.06, fill=COLOR_ROBOT_DARK + (255,)
        )
        for i in range(3):
            top = 0.34 + i * 0.18
            draw.rectangle(box(x0 + 0.02, top, x1 - 0.02, top + 0.07), fill=(122, 52, 44, 255))

    # Nose wedge: the heading cue, drawn first so the body overlaps its base.
    draw.polygon(
        [(span * 0.50, span * 0.02), (span * 0.87, span * 0.44), (span * 0.13, span * 0.44)],
        fill=COLOR_ROBOT_DARK + (255,),
    )
    draw.polygon(
        [(span * 0.50, span * 0.10), (span * 0.78, span * 0.42), (span * 0.22, span * 0.42)],
        fill=COLOR_ROBOT_LIGHT + (255,),
    )

    # Chassis with a heavy rim.
    draw.rounded_rectangle(
        box(0.14, 0.30, 0.86, 0.90), radius=span * 0.15, fill=COLOR_ROBOT_DARK + (255,)
    )
    draw.rounded_rectangle(
        box(0.20, 0.36, 0.80, 0.84), radius=span * 0.12, fill=COLOR_ROBOT + (255,)
    )
    draw.rounded_rectangle(
        box(0.24, 0.39, 0.76, 0.58), radius=span * 0.09, fill=(255, 140, 118, 200)
    )

    # Sensor dome, the part that does the scanning.
    draw.ellipse(box(0.32, 0.44, 0.68, 0.80), fill=COLOR_ROBOT_DARK + (255,))
    draw.ellipse(box(0.36, 0.48, 0.64, 0.76), fill=COLOR_ROBOT_LIGHT + (255,))
    draw.ellipse(box(0.42, 0.54, 0.58, 0.70), fill=COLOR_BEAM + (255,))
    return image.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def robot_sprite_facing(size: int, heading: int) -> Image.Image:
    """The robot rotated to a heading index.

    Args:
        size: Final sprite width and height in pixels.
        heading: ``0`` north, ``1`` east, ``2`` south, ``3`` west, matching
            :data:`~POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor.HEADING_STEPS`.

    Returns:
        A square RGBA image. Cached; callers must not modify it.
    """
    base = robot_sprite(size)
    turns = int(heading) % 4
    if turns == 0:
        return base
    # The sprite faces north, headings run clockwise, and Pillow rotates
    # counter-clockwise, so east needs -90 degrees.
    return base.rotate(-90.0 * turns, resample=Image.Resampling.BICUBIC, expand=False)


# --- chrome ------------------------------------------------------------------


def draw_bezel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    radius: int = 12,
    fill: Tuple[int, int, int] = COLOR_METAL,
    bolts: bool = True,
) -> None:
    """Draw a machined metal panel: rounded body, bevel, and corner bolts.

    A bevel is two lines, one lit and one shaded, on opposite sides. It costs
    almost nothing and it is what separates a chart from an instrument.

    Args:
        draw: Target draw context, which must accept RGBA fills.
        box: ``(left, top, right, bottom)`` in pixels.
        radius: Corner radius.
        fill: Panel body colour.
        bolts: Whether to add the four corner bolt heads.
    """
    left, top, right, bottom = box
    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=radius,
        fill=fill,
        outline=COLOR_METAL_DARK,
        width=2,
    )
    draw.line(
        [(left + radius, top + 2), (right - radius, top + 2)], fill=COLOR_METAL_LIGHT, width=1
    )
    draw.line(
        [(left + 2, top + radius), (left + 2, bottom - radius)], fill=COLOR_METAL_LIGHT, width=1
    )
    draw.line(
        [(left + radius, bottom - 2), (right - radius, bottom - 2)], fill=COLOR_METAL_DARK, width=1
    )
    if not bolts:
        return
    for bolt_x in (left + 11, right - 11):
        for bolt_y in (top + 11, bottom - 11):
            draw.ellipse(
                (bolt_x - 3, bolt_y - 3, bolt_x + 3, bolt_y + 3),
                fill=COLOR_BOLT,
                outline=COLOR_METAL_DARK,
            )
            draw.line((bolt_x - 2, bolt_y, bolt_x + 2, bolt_y), fill=COLOR_METAL_DARK)


def draw_screen_edge(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    tint: Tuple[int, int, int],
) -> None:
    """Draw the inner rim around a panel's screen.

    The rim is tinted with the panel's own accent, which is a second, redundant
    cue for which panel a reader is looking at.

    Args:
        draw: Target draw context, which must accept RGBA fills.
        box: ``(left, top, right, bottom)`` of the screen itself.
        tint: Accent colour for the inner glow line.
    """
    left, top, right, bottom = box
    draw.rectangle([left - 3, top - 3, right + 2, bottom + 2], outline=COLOR_METAL_DARK, width=3)
    draw.rectangle([left - 1, top - 1, right, bottom], outline=tint + (120,), width=1)


def draw_gradient_bar(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    top: int,
    stops: Sequence[Tuple[float, Tuple[int, int, int]]],
    blocks: int,
) -> None:
    """Draw a colour ramp as a row of stepped blocks.

    Stepped rather than continuous because a GIF carries 256 colours: a smooth
    ramp bands under quantization and the banding reads as a defect, while a
    scale of discrete blocks survives it and reads as an instrument.

    Args:
        draw: Target draw context, which must accept RGBA fills.
        left: Left edge of the bar in page pixels.
        right: Right edge of the bar in page pixels.
        top: Top edge of the bar in page pixels.
        stops: ``(position, colour)`` anchors in ascending position order.
        blocks: Number of discrete blocks to draw.
    """
    edges = np.round(np.linspace(left, right, blocks + 1)).astype(int)
    for block in range(blocks):
        position = (block + 0.5) / blocks
        lower, upper = stops[0], stops[-1]
        for index in range(len(stops) - 1):
            if stops[index][0] <= position <= stops[index + 1][0]:
                lower, upper = stops[index], stops[index + 1]
                break
        span = max(upper[0] - lower[0], 1e-9)
        colour = mix_colors(lower[1], upper[1], (position - lower[0]) / span)
        draw.rectangle(
            [int(edges[block]) + 1, top, int(edges[block + 1]) - 1, top + 15],
            fill=colour,
        )
    draw.rectangle([left - 1, top - 1, right, top + 16], outline=COLOR_METAL_DARK)


__all__ = [
    "ASSET_SEED",
    "COLOR_BEAM",
    "COLOR_BOLT",
    "COLOR_CHART_FOG",
    "COLOR_CHART_FREE",
    "COLOR_CHART_OCCUPIED",
    "COLOR_HOLO_BLANK",
    "COLOR_HOLO_HIGH",
    "COLOR_HOLO_LOW",
    "COLOR_HOLO_MID",
    "COLOR_METAL",
    "COLOR_METAL_DARK",
    "COLOR_METAL_LIGHT",
    "COLOR_METER_FILL",
    "COLOR_METER_GOAL",
    "COLOR_PAGE",
    "COLOR_ROBOT",
    "COLOR_ROBOT_DARK",
    "COLOR_ROBOT_LIGHT",
    "COLOR_TEXT",
    "COLOR_TEXT_DIM",
    "COLOR_TRAIL",
    "COLOR_TRUTH_FLOOR",
    "COLOR_TRUTH_WALL",
    "draw_bezel",
    "draw_gradient_bar",
    "draw_screen_edge",
    "get_font",
    "mix_colors",
    "robot_sprite",
    "robot_sprite_facing",
    "static_field",
    "stone_floor",
    "stone_shade",
    "value_noise",
]
