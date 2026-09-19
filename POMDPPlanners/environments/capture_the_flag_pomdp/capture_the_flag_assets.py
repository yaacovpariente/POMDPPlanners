# SPDX-License-Identifier: MIT

"""Cached terrain and actor art for the CaptureTheFlag visualizer.

Everything here is generated from fixed seeds and memoized, never from the
simulation's random stream: the golden-file test hashes the rendered GIF, so a
field that resampled its grass between frames would read as a regression. The
same rule ``laser_tag_assets.metal_texture`` follows.

The camera is isometric -- a cell maps to a diamond -- because a flat top-down
grid reads as a board rather than a battlefield, and the elevated angle is what
lets keeps, trees and soldiers occlude one another.

Functions:
    iso: Project a grid cell to screen pixels.
    ground_plane: Seamless terrain bitmap for a whole viewport.
    tree_sprite, rock_sprite, banner_sprite, keep_sprite, soldier_sprite:
        Outlined actor art.
"""

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TILE_WIDTH, TILE_HEIGHT = 96, 48
OUTLINE = (20, 17, 22, 255)

BLUE = ((24, 44, 100), (40, 84, 172), (84, 138, 232), (162, 198, 255))
RED = ((104, 26, 24), (168, 50, 42), (212, 92, 76), (250, 172, 152))
SKIN = ((168, 116, 78), (214, 168, 124), (240, 206, 168))
WOOD = ((58, 40, 26), (92, 64, 40), (128, 94, 58), (166, 130, 84))
STONE = ((48, 48, 54), (78, 78, 88), (112, 112, 124), (152, 152, 166))

_NOISE_RESOLUTION = 640
_NOISE_SPAN = (-8.0, 18.0)


def iso(cell_x: float, cell_y: float, origin: Tuple[int, int]) -> Tuple[int, int]:
    """Project a grid cell to its screen centre.

    Args:
        cell_x: Grid x, fractional for a unit caught between cells.
        cell_y: Grid y.
        origin: Screen pixel the grid origin projects to.

    Returns:
        The ``(x, y)`` screen pixel at the cell's centre.
    """
    return (
        int((cell_x - cell_y) * (TILE_WIDTH // 2) + origin[0]),
        int((cell_x + cell_y) * (TILE_HEIGHT // 2) + origin[1]),
    )


@lru_cache(maxsize=4)
def _noise_field(seed: int) -> np.ndarray:
    """Return a memoized multi-octave noise field in grid space."""
    rng = np.random.default_rng(seed)
    field = np.zeros((_NOISE_RESOLUTION, _NOISE_RESOLUTION))
    for cells, amplitude in ((3, 1.0), (7, 0.55), (16, 0.3), (38, 0.16), (90, 0.08)):
        coarse = rng.random((cells, cells))
        field += (
            amplitude
            * np.asarray(
                Image.fromarray((coarse * 255).astype(np.uint8)).resize(
                    (_NOISE_RESOLUTION, _NOISE_RESOLUTION), Image.Resampling.BICUBIC
                )
            )
            / 255.0
        )
    field -= field.min()
    return field / field.max()


def _sample(field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Sample a grid-space noise field at projected pixel coordinates."""
    low, high = _NOISE_SPAN
    columns = np.clip((grid_x - low) / (high - low) * (_NOISE_RESOLUTION - 1), 0, _NOISE_RESOLUTION - 1)
    rows = np.clip((grid_y - low) / (high - low) * (_NOISE_RESOLUTION - 1), 0, _NOISE_RESOLUTION - 1)
    return field[rows.astype(int), columns.astype(int)]


def _segment_distance(
    px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float
) -> np.ndarray:
    """Distance from each point to a line segment, in grid units."""
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def river_axis(grid_y: np.ndarray, midline: int) -> np.ndarray:
    """Return the channel's centreline in grid x for a given grid y.

    The bridge and the water are both drawn from this one function, so the
    crossing cannot drift off the river when the wobble is retuned.

    Args:
        grid_y: Grid y coordinates.
        midline: The neutral column the channel meanders around.

    Returns:
        The channel centre in grid x.
    """
    return midline + 0.22 * np.sin(grid_y * 0.85 + 0.6) + 0.09 * np.sin(grid_y * 2.1)


# pylint: disable-next=too-many-locals
def ground_plane(
    size: Tuple[int, int],
    top: int,
    origin: Tuple[int, int],
    midline: int,
    paths: Sequence[Tuple[float, float, float, float]],
) -> Image.Image:
    """Render the terrain for a whole viewport as one seamless bitmap.

    Every screen pixel is inverse-projected to grid space and coloured from a
    continuous noise field, so there are no tile seams and the terrain reaches
    every edge -- a map that stops mid-viewport reads as a board on a table.

    Args:
        size: Viewport size in pixels.
        top: First screen row the field occupies.
        origin: Screen pixel the grid origin projects to.
        midline: Neutral column the channel meanders around.
        paths: Worn trails as ``(ax, ay, bx, by)`` segments in grid space.

    Returns:
        The terrain image, sized ``(width, height - top)``.
    """
    width, height = size
    field_height = height - top
    screen_x = np.arange(width)[None, :].repeat(field_height, 0).astype(float)
    screen_y = (np.arange(field_height) + top)[:, None].repeat(width, 1).astype(float)
    u = (screen_x - origin[0]) / (TILE_WIDTH / 2)
    v = (screen_y - origin[1]) / (TILE_HEIGHT / 2)
    grid_x, grid_y = (u + v) / 2, (v - u) / 2

    base, detail, coarse = (_noise_field(11), _noise_field(29), _noise_field(47))
    n_base = _sample(base, grid_x, grid_y)
    n_detail = _sample(detail, grid_x, grid_y)
    n_coarse = _sample(coarse, grid_x, grid_y)

    grass = np.array([56.0, 86.0, 42.0]) + (
        np.array([104.0, 144.0, 74.0]) - np.array([56.0, 86.0, 42.0])
    ) * n_base[..., None]
    earth = np.array([104.0, 82.0, 52.0]) + (
        np.array([158.0, 132.0, 88.0]) - np.array([104.0, 82.0, 52.0])
    ) * n_detail[..., None]

    if paths:
        trail = np.minimum.reduce(
            [_segment_distance(grid_x, grid_y, *segment) for segment in paths]
        )
    else:
        trail = np.full_like(grid_x, 10.0)
    worn = np.clip(1.0 - (trail - 0.16) / 0.30, 0, 1) * (0.55 + 0.45 * n_coarse)
    patches = np.clip((n_detail - 0.60) / 0.16, 0, 1) * 0.8
    blend = np.clip(worn + patches, 0, 1)[..., None]
    colour = grass * (1 - blend) + earth * blend

    # Territory is a whisper of tint, not a second biome: saturated blue and red
    # are reserved for the units, roofs and banners, so the eye finds them.
    warm = (grid_x > midline)[..., None]
    colour = colour * np.where(
        warm, np.array([1.05, 0.985, 0.94]), np.array([0.97, 1.0, 1.03])
    )

    light = 0.86 + 0.30 * _sample(coarse, grid_x * 0.5, grid_y * 0.5)
    light = light - 0.055 * np.clip(grid_x / 9, 0, 1) - 0.055 * np.clip(grid_y / 7, 0, 1)
    colour = colour * light[..., None]

    distance = np.abs(grid_x - river_axis(grid_y, midline))
    deep, shallow = np.array([40.0, 92.0, 124.0]), np.array([76.0, 148.0, 182.0])
    water = shallow + (deep - shallow) * np.clip((0.46 - distance) / 0.40, 0, 1)[..., None]
    water = water + (_sample(base, grid_x * 3.0, grid_y * 3.0)[..., None] - 0.5) * 22
    wet = np.clip((0.50 - distance) / 0.05, 0, 1)[..., None]
    shore = np.clip((0.78 - distance) / 0.26, 0, 1)[..., None] * (1 - wet)
    colour = colour * (1 - shore) + earth * 0.92 * shore
    colour = colour * (1 - wet) + water * wet
    foam = (distance > 0.44) & (distance < 0.515) & (_sample(detail, grid_x * 6, grid_y * 6) > 0.52)
    colour[foam] = np.array([196.0, 232.0, 244.0])

    return Image.fromarray(np.clip(colour, 0, 255).astype(np.uint8)).convert("RGBA")


def _outlined(size: Tuple[int, int], paint) -> Image.Image:
    """Draw a sprite and ring it with a one-pixel dark outline.

    Dilating the sprite's own alpha is what separates an actor from the terrain
    behind it; without it the units read as coloured shapes lying on grass.

    Args:
        size: Sprite canvas size.
        paint: Callable given an ``ImageDraw`` to draw the sprite.

    Returns:
        The outlined sprite.
    """
    sprite = Image.new("RGBA", size, (0, 0, 0, 0))
    paint(ImageDraw.Draw(sprite))
    alpha = sprite.getchannel("A").point(lambda value: 255 if value > 128 else 0)
    ring = Image.new("RGBA", size, OUTLINE)
    ring.putalpha(alpha.filter(ImageFilter.MaxFilter(3)))
    return Image.alpha_composite(ring, sprite)


@lru_cache(maxsize=64)
def tree_sprite(seed: int, dry: bool) -> Image.Image:
    """Return a memoized tree with an asymmetric crown.

    Args:
        seed: Per-instance seed; equal seeds give identical trees.
        dry: Whether to use the red half's autumn palette.

    Returns:
        The tree sprite.
    """
    rng = np.random.default_rng(seed)
    leaves = (
        ((78, 60, 26), (112, 88, 36), (146, 120, 52), (180, 152, 76))
        if dry
        else ((28, 62, 32), (40, 88, 42), (56, 116, 54), (84, 150, 70))
    )
    lean = int(rng.integers(-3, 4))
    blobs: List[Tuple[int, int, int]] = []
    for _ in range(11):
        angle = float(rng.random()) * 6.283
        radius = int(rng.integers(4, 17))
        blobs.append(
            (
                27 + lean + int(np.cos(angle) * radius),
                30 + int(np.sin(angle) * radius * 0.8) - int(rng.integers(4, 22)),
                int(rng.integers(9, 16)),
            )
        )

    def paint(draw: ImageDraw.ImageDraw) -> None:
        draw.polygon([(24 + lean, 30), (30 + lean, 30), (33, 78), (25, 78)], fill=WOOD[1])
        draw.polygon([(24 + lean, 30), (27 + lean, 30), (29, 78), (25, 78)], fill=WOOD[2])
        draw.line([(27 + lean, 44), (16, 34)], fill=WOOD[1], width=3)
        for bx, by, br in sorted(blobs, key=lambda blob: -blob[1]):
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=leaves[0])
        for bx, by, br in sorted(blobs, key=lambda blob: -blob[1])[3:]:
            draw.ellipse([bx - br + 2, by - br + 1, bx + br - 3, by + br - 4], fill=leaves[1])
        for bx, by, br in sorted(blobs, key=lambda blob: blob[1])[:5]:
            draw.ellipse([bx - br + 4, by - br + 2, bx + br - 7, by + br - 9], fill=leaves[2])
            draw.ellipse([bx - br + 5, by - br + 3, bx - br + 12, by - br + 9], fill=leaves[3])

    return _outlined((56, 84), paint)


@lru_cache(maxsize=32)
def rock_sprite(seed: int) -> Image.Image:
    """Return a memoized rock cluster.

    Args:
        seed: Per-instance seed.

    Returns:
        The rock sprite.
    """
    del seed  # shape is fixed; the seed only keys the cache per placement

    def paint(draw: ImageDraw.ImageDraw) -> None:
        for ox, oy, scale in ((0, 0, 1.0), (14, 6, 0.6), (-12, 5, 0.5)):
            draw.polygon(
                [
                    (20 + ox - int(16 * scale), 30 + oy),
                    (22 + ox - int(12 * scale), 30 + oy - int(16 * scale)),
                    (24 + ox, 30 + oy - int(22 * scale)),
                    (26 + ox + int(14 * scale), 30 + oy - int(12 * scale)),
                    (26 + ox + int(16 * scale), 30 + oy),
                ],
                fill=STONE[1],
            )
            draw.polygon(
                [
                    (22 + ox - int(12 * scale), 30 + oy - int(16 * scale)),
                    (24 + ox, 30 + oy - int(22 * scale)),
                    (24 + ox, 30 + oy - int(8 * scale)),
                ],
                fill=STONE[3],
            )

    return _outlined((48, 36), paint)


@lru_cache(maxsize=16)
def banner_sprite(team_blue: bool, height: int = 34) -> Image.Image:
    """Return a memoized planted banner.

    Args:
        team_blue: Whether the banner belongs to blue.
        height: Pole height in pixels.

    Returns:
        The banner sprite.
    """
    colour = BLUE if team_blue else RED

    def paint(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([5, 3, 7, height + 8], fill=(206, 198, 180))
        draw.rectangle([5, 3, 6, height + 8], fill=(238, 232, 214))
        draw.polygon([(8, 4), (27, 10), (8, 19)], fill=colour[1])
        draw.polygon([(8, 4), (27, 10), (8, 11)], fill=colour[2])
        draw.ellipse([3, 0, 9, 6], fill=(228, 196, 108))

    return _outlined((30, height + 12), paint)


@lru_cache(maxsize=4)
def keep_sprite(team_blue: bool) -> Image.Image:
    """Return a memoized stronghold with a lit left face and a shaded right one.

    Args:
        team_blue: Whether the keep belongs to blue.

    Returns:
        The keep sprite.
    """
    colour = BLUE if team_blue else RED

    def paint(draw: ImageDraw.ImageDraw) -> None:
        draw.polygon([(6, 74), (52, 48), (98, 74), (52, 100)], fill=STONE[0])
        draw.polygon([(10, 68), (52, 44), (94, 68), (52, 92)], fill=STONE[1])
        draw.polygon([(14, 30), (52, 10), (90, 30), (52, 50)], fill=STONE[3])
        draw.polygon([(14, 30), (52, 50), (52, 78), (14, 58)], fill=STONE[2])
        draw.polygon([(90, 30), (52, 50), (52, 78), (90, 58)], fill=STONE[0])
        for merlon in range(15, 90, 12):
            draw.rectangle(
                [merlon, 22, merlon + 6, 32], fill=STONE[3] if merlon < 52 else STONE[1]
            )
        draw.polygon([(20, 26), (52, 6), (84, 26), (52, 40)], fill=colour[1])
        draw.polygon([(52, 6), (84, 26), (52, 40)], fill=colour[0])
        draw.polygon([(30, 58), (40, 52), (40, 76), (30, 80)], fill=(34, 26, 20))
        draw.polygon([(30, 58), (40, 52), (40, 58), (30, 63)], fill=(58, 44, 32))
        for window_x, window_y in ((22, 44), (76, 46)):
            draw.rectangle([window_x, window_y, window_x + 5, window_y + 9], fill=(30, 26, 30))
            draw.rectangle([window_x, window_y, window_x + 5, window_y + 2], fill=STONE[3])

    return _outlined((105, 105), paint)


# pylint: disable-next=too-many-arguments
@lru_cache(maxsize=512)
def soldier_sprite(
    team_blue: bool,
    facing: int,
    stride_index: int,
    carrying_blue: bool,
    carrying_red: bool,
    frozen: bool,
    selected: bool,
) -> Image.Image:
    """Return a memoized soldier in one frame of its run cycle.

    Args:
        team_blue: Whether the soldier is on blue.
        facing: ``1`` facing right, ``-1`` facing left.
        stride_index: Frame index within the run cycle.
        carrying_blue: Whether the blue banner rides on its back.
        carrying_red: Whether the red banner rides on its back.
        frozen: Whether it is serving a respawn freeze, drawn translucent.
        selected: Whether to draw a selection ellipse under it.

    Returns:
        The composed unit sprite, including shadow and selection ring.
    """
    colour = BLUE if team_blue else RED
    swing = int(round(5 * np.sin((stride_index / 8.0) * np.pi)))

    def paint(draw: ImageDraw.ImageDraw) -> None:
        draw.polygon([(12, 36), (17, 36), (15 - swing, 50), (10 - swing, 50)], fill=(62, 48, 36))
        draw.polygon(
            [(9 - swing, 48), (17 - swing, 47), (17 - swing, 51), (8 - swing, 51)],
            fill=(40, 32, 24),
        )
        draw.polygon([(17, 36), (23, 36), (24 + swing, 50), (19 + swing, 50)], fill=(86, 68, 50))
        draw.polygon(
            [(18 + swing, 48), (26 + swing, 47), (26 + swing, 51), (17 + swing, 51)],
            fill=(40, 32, 24),
        )
        haft = 27 if facing > 0 else 7
        draw.line([(haft, 2), (haft - 4 * facing, 40)], fill=WOOD[2], width=3)
        draw.polygon(
            [(haft + 1, -1), (haft + 6 * facing, 8), (haft - 4 * facing, 7)], fill=STONE[3]
        )
        draw.polygon([(11, 18), (23, 18), (25, 38), (9, 38)], fill=colour[1])
        draw.polygon([(11, 18), (17, 18), (17, 38), (9, 38)], fill=colour[2])
        draw.rectangle([9, 29, 25, 32], fill=(206, 176, 92))
        draw.polygon([(7, 20), (11, 19), (13, 32), (9, 33)], fill=colour[0])
        draw.polygon(
            [(23, 19 + swing // 2), (27, 20 + swing // 2), (25, 33), (21, 32)], fill=colour[0]
        )
        shield_x = 2 if facing > 0 else 22
        draw.ellipse([shield_x, 19, shield_x + 12, 36], fill=colour[0])
        draw.ellipse([shield_x + 1, 20, shield_x + 11, 35], fill=colour[1])
        draw.arc([shield_x + 1, 20, shield_x + 11, 35], 140, 320, fill=colour[2])
        draw.ellipse([shield_x + 4, 25, shield_x + 8, 30], fill=(226, 198, 112))
        draw.polygon([(10, 12), (24, 12), (26, 20), (8, 20)], fill=colour[0])
        draw.rectangle([12, 7, 22, 18], fill=SKIN[1])
        draw.polygon([(11, 9), (17, 2), (23, 9), (23, 12), (11, 12)], fill=STONE[2])
        draw.polygon([(11, 9), (17, 2), (17, 12), (11, 12)], fill=STONE[3])
        draw.rectangle([16, -2, 19, 4], fill=colour[2])
        draw.rectangle([12, 12, 22, 13], fill=STONE[1])
        draw.point((14, 15), fill=(28, 22, 22))
        draw.point((20, 15), fill=(28, 22, 22))

    body = _outlined((35, 54), paint)
    if frozen:
        faded = body.getchannel("A").point(lambda value: int(value * 0.45))
        body.putalpha(faded)

    pad = Image.new("RGBA", (56, 66), (0, 0, 0, 0))
    pad_draw = ImageDraw.Draw(pad)
    pad_draw.ellipse([14, 52, 42, 62], fill=(18, 24, 18, 110))
    if selected:
        pad_draw.ellipse([9, 52, 47, 64], outline=(126, 246, 136, 220))
    pad.alpha_composite(body, (10, 6))
    for carrying, team in ((carrying_blue, True), (carrying_red, False)):
        if carrying:
            pad.alpha_composite(banner_sprite(team, 16), (34 if facing > 0 else 0, 0))
    return pad


def contact_shadow(
    canvas: Image.Image,
    centre: Tuple[int, int],
    radius: Tuple[int, int],
    alpha: int = 105,
    skew: int = 0,
) -> None:
    """Composite a soft shadow under a prop, offset down-right with the key light.

    Args:
        canvas: The image to draw onto.
        centre: Shadow centre in screen pixels.
        radius: Shadow half-extents as ``(rx, ry)``.
        alpha: Shadow opacity.
        skew: Horizontal stretch away from the light.
    """
    rx, ry = radius
    patch = Image.new("RGBA", (rx * 2 + abs(skew) + 4, ry * 2 + 4), (0, 0, 0, 0))
    ImageDraw.Draw(patch).ellipse(
        [2 + max(0, skew), 2, 2 + max(0, skew) + rx * 2, 2 + ry * 2], fill=(18, 24, 18, alpha)
    )
    canvas.alpha_composite(
        patch.filter(ImageFilter.GaussianBlur(1.4)), (centre[0] - rx - 2, centre[1] - ry - 2)
    )


def panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int] = (30, 30, 38),
    edge: Tuple[int, int, int] = (96, 96, 112),
) -> None:
    """Draw a bevelled UI slab.

    Args:
        draw: The drawing context.
        box: ``(x0, y0, x1, y1)`` in screen pixels.
        fill: Slab interior colour.
        edge: Highlight colour for the lit edges.
    """
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    draw.rectangle([x0, y0, x1, y0], fill=edge)
    draw.rectangle([x0, y0, x0, y1], fill=edge)
    draw.rectangle([x0, y1, x1, y1], fill=(14, 14, 18))
    draw.rectangle([x1, y0, x1, y1], fill=(14, 14, 18))


def team_colours(team_blue: bool) -> Tuple[Tuple[int, int, int], ...]:
    """Return a team's four-step palette.

    Args:
        team_blue: Whether to return blue's palette.

    Returns:
        The palette, darkest first.
    """
    return BLUE if team_blue else RED


def belief_marker(colour: Tuple[int, int, int], weight: float) -> Dict[str, object]:
    """Return the draw parameters for one belief marker.

    Args:
        colour: The marker's base colour.
        weight: Normalised probability mass in ``[0, 1]``.

    Returns:
        Radius and alpha for the marker.
    """
    return {"radius": 3 + 9 * weight, "alpha": int(70 + 160 * weight), "colour": colour}
