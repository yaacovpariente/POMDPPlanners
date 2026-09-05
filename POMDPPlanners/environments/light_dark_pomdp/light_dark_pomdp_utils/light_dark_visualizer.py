# SPDX-License-Identifier: MIT

"""Light-Dark POMDP episode renderer.

The picture is a dark, textured floor lit by warm beacons, with the rover, the
goal, the hazard fields and the belief cloud drawn on top.  It is produced with
Pillow and numpy only.  There are two reasons for that:

* **Speed.**  The scene is static except for the rover, its path, the action
  arrow and the belief particles.  So the lit ground, the beacons, the hazard
  overlays, the axes and the legend are rasterised once per episode and every
  frame is that cached background plus a small dynamic overlay.  The previous
  Matplotlib implementation re-rendered the entire figure for each frame.
* **Import cost.**  ``BaseLightDarkPOMDP`` imports this module at import time,
  so importing the environment used to import ``matplotlib.pyplot`` even for
  runs that never render anything.

What the renderer must not change, and does not: the world geometry (the view
still spans ``-1 .. grid_size + 1`` on both axes), the fact that hazards are
probability fields rather than solid walls, the recorded states, actions and
beliefs, and the public API of this class.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_assets import (
    COLOR_BEACON_MARK,
    COLOR_BELIEF,
    COLOR_GOAL,
    COLOR_HAZARD,
    COLOR_LIGHT_WARM,
    COLOR_OBSTACLE_DOT,
    COLOR_PAGE,
    COLOR_PANEL,
    COLOR_PANEL_EDGE,
    COLOR_PATH,
    COLOR_ROVER_BODY,
    COLOR_START,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    beacon_sprite,
    get_font,
    glow_sprite,
    goal_sprite,
    ground_texture,
    rover_sprite_facing,
    start_sprite,
)

# Canvas matches the old Matplotlib figure (10x8 inches at 100 dpi) so that
# before/after comparisons and any downstream consumer see the same size.
CANVAS_SIZE = (1000, 800)

# Plot rectangle inside the canvas; the strip on the right holds the legend.
PLOT_LEFT = 78
PLOT_TOP = 74
PLOT_RIGHT = 792
PLOT_BOTTOM = 748

LEGEND_LEFT = 812
LEGEND_RIGHT = 984

# Frames per second of the saved GIF. Unchanged from the previous renderer.
GIF_FPS = 2

# How much of the floor is visible with no beacon nearby. Low enough that the
# dark region reads as dark, high enough that the texture still shows.
AMBIENT = 0.34

# Longest belief history trail, in frames. Same value the old renderer used.
MAX_BELIEF_TRAIL = 10

# Upper bound on particles drawn per trail layer. A vectorized belief can carry
# thousands of particles; past a few hundred the extra dots are invisible but
# still cost time, so we keep the heaviest ones.
MAX_PARTICLES_PER_LAYER = 400


# Colours that must survive GIF quantization even though they cover few pixels.
# Includes the yellow-to-red belief ramp, because a belief cloud that quantizes
# to one flat colour loses the age information the ramp encodes.
ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    COLOR_GOAL,
    (16, 74, 26),
    (150, 240, 160),
    COLOR_START,
    (255, 150, 142),
    COLOR_BEACON_MARK,
    (16, 30, 78),
    COLOR_PATH,
    (240, 96, 84),
    COLOR_ROVER_BODY,
    (226, 92, 76),
    (120, 168, 196),
    (255, 244, 206),
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_PANEL,
    COLOR_PANEL_EDGE,
    COLOR_OBSTACLE_DOT,
    (255, 255, 255),
    (0, 0, 0),
) + tuple(
    (
        int(COLOR_BELIEF[0] * (1.0 - t) + 220.0 * t),
        int(COLOR_BELIEF[1] * (1.0 - t) + 40.0 * t),
        int(COLOR_BELIEF[2] * (1.0 - t) + 34.0 * t),
    )
    for t in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
)


def _xy(point: Any) -> Tuple[float, float]:
    """First two components of a state, ignoring any terminal-flag slot."""
    arr = np.asarray(point, dtype=float).ravel()
    return float(arr[0]), float(arr[1])


def _pretty_title(environment: Any) -> str:
    """Human title from the environment class name, e.g. 'Discrete Light-Dark'."""
    raw = type(environment).__name__.replace("POMDP", " ")
    words: List[str] = []
    current = ""
    for ch in raw:
        if ch == " ":
            if current:
                words.append(current)
                current = ""
            continue
        if ch.isupper() and current:
            words.append(current)
            current = ch
        else:
            current += ch
    if current:
        words.append(current)
    text = " ".join(words)
    return text.replace("Light Dark", "Light-Dark").strip() or "Light-Dark"


class LightDarkPOMDPVisualizer:
    """Visualizer for Light-Dark POMDP environments.

    Handles all visualization and animation logic for Light-Dark POMDP
    environments, including path visualization, belief particle rendering and
    animation generation.  Shared by the continuous and discrete variants and by
    anything that inherits from them.

    Attributes:
        environment: The Light-Dark POMDP environment instance to visualize.
    """

    def __init__(self, environment: Any):
        """Initialize the visualizer.

        Args:
            environment: The Light-Dark POMDP environment instance to visualize.
                Must have attributes: beacons, goal_state, start_state, obstacles,
                obstacle_radius, beacon_radius, grid_size, action_to_vector.
        """
        self.environment = environment
        self._background: Optional[Image.Image] = None
        self._background_key: Optional[Tuple[Any, ...]] = None

    # --- coordinate transform ------------------------------------------------

    @property
    def _world_limits(self) -> Tuple[float, float]:
        """View bounds, identical to the old ``set_xlim``/``set_ylim`` call."""
        return -1.0, float(self.environment.grid_size) + 1.0

    def _to_px(self, x: float, y: float) -> Tuple[float, float]:
        lo, hi = self._world_limits
        span = hi - lo
        px = PLOT_LEFT + (x - lo) / span * (PLOT_RIGHT - PLOT_LEFT)
        py = PLOT_BOTTOM - (y - lo) / span * (PLOT_BOTTOM - PLOT_TOP)
        return px, py

    @property
    def _px_per_unit(self) -> float:
        """Pixels per world unit, using the smaller axis so sprites never clash."""
        lo, hi = self._world_limits
        span = hi - lo
        return min((PLOT_RIGHT - PLOT_LEFT) / span, (PLOT_BOTTOM - PLOT_TOP) / span)

    # --- static background ---------------------------------------------------

    def _scene_key(self) -> Tuple[Any, ...]:
        """Everything the static background depends on.

        The background is cached on the instance, so a visualizer reused after
        the environment's geometry changed would otherwise draw a stale scene.
        """
        env = self.environment
        return (
            type(env).__name__,
            int(env.grid_size),
            float(env.beacon_radius),
            float(env.obstacle_radius),
            np.asarray(env.beacons, dtype=float).tobytes(),
            np.asarray(env.obstacles, dtype=float).tobytes(),
            np.asarray(env.goal_state, dtype=float).tobytes(),
            np.asarray(env.start_state, dtype=float).tobytes(),
        )

    def _lit_ground(self) -> np.ndarray:
        """Ground texture with the beacon light field and hazard fields baked in.

        Distances are measured in world units, so a hazard of radius ``r`` covers
        exactly the pixels whose world position is within ``r`` of its centre --
        the same region the environment treats as hazardous.
        """
        width = PLOT_RIGHT - PLOT_LEFT
        height = PLOT_BOTTOM - PLOT_TOP
        lo, hi = self._world_limits

        # ground_texture is cached, so never write into what it returns.
        rgb = ground_texture(width, height).copy()

        # Pixel-centre world coordinates.
        wx = lo + (np.arange(width) + 0.5) / width * (hi - lo)
        wy = hi - (np.arange(height) + 0.5) / height * (hi - lo)

        beacons = np.asarray(self.environment.beacons, dtype=float)
        light = np.zeros((height, width), dtype=np.float64)
        radius = float(self.environment.beacon_radius)
        core_sigma = max(0.75 * radius, 1e-6)
        halo_sigma = max(1.6 * radius, 1e-6)
        for i in range(beacons.shape[1]):
            bx, by = float(beacons[0, i]), float(beacons[1, i])
            d2 = (wx[None, :] - bx) ** 2 + (wy[:, None] - by) ** 2
            light += np.exp(-d2 / (2.0 * core_sigma**2))
            light += 0.28 * np.exp(-d2 / (2.0 * halo_sigma**2))
        light = np.clip(light, 0.0, 1.6)

        warm = np.array(COLOR_LIGHT_WARM, dtype=np.float64) / 255.0
        # Diffuse term lights the texture; the cubic term adds a small bloom so
        # the centre of a pool blows out the way a real lamp does.
        rgb *= AMBIENT + 1.45 * light[:, :, None] * warm[None, None, :]
        rgb += 0.12 * (light[:, :, None] ** 3) * warm[None, None, :]

        # Vignette: pulls the eye to the middle and sells the "dark world" read.
        ny = np.linspace(-1.0, 1.0, height)[:, None]
        nx = np.linspace(-1.0, 1.0, width)[None, :]
        rgb *= (1.0 - 0.28 * np.clip((nx**2 + ny**2) / 2.0, 0.0, 1.0))[:, :, None]

        rgb = np.clip(rgb, 0.0, 1.0)

        # Hazards are probability fields, not walls: a translucent red wash that
        # is strongest at the centre and fades out, combined so overlapping
        # fields read as more dangerous rather than replacing each other.
        obstacles = np.asarray(self.environment.obstacles, dtype=float)
        if obstacles.size > 0:
            keep = np.ones((height, width), dtype=np.float64)
            o_radius = max(float(self.environment.obstacle_radius), 1e-6)
            for i in range(obstacles.shape[1]):
                ox, oy = float(obstacles[0, i]), float(obstacles[1, i])
                d2 = (wx[None, :] - ox) ** 2 + (wy[:, None] - oy) ** 2
                t = np.clip(d2 / (o_radius**2), 0.0, 1.0)
                alpha = 0.30 * np.clip(1.0 - t**3, 0.0, 1.0) ** 0.45
                keep *= 1.0 - alpha
            total = 1.0 - keep
            hazard = np.array(COLOR_HAZARD, dtype=np.float64) / 255.0
            rgb = rgb * (1.0 - total[:, :, None]) + hazard[None, None, :] * total[:, :, None]

        return np.clip(rgb, 0.0, 1.0)

    def _draw_axes(self, draw: ImageDraw.ImageDraw) -> None:
        lo, hi = self._world_limits
        font = get_font(17)
        ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
        # Thin out labels when the grid is dense enough that they would collide.
        step_px = (PLOT_RIGHT - PLOT_LEFT) / max(len(ticks) - 1, 1)
        stride = 1 if step_px >= 30 else 2

        for idx, t in enumerate(ticks):
            px, _ = self._to_px(float(t), 0.0)
            _, py = self._to_px(0.0, float(t))
            draw.line([(px, PLOT_BOTTOM), (px, PLOT_BOTTOM + 6)], fill=COLOR_TEXT_DIM, width=1)
            draw.line([(PLOT_LEFT - 6, py), (PLOT_LEFT, py)], fill=COLOR_TEXT_DIM, width=1)
            if idx % stride:
                continue
            label = str(t)
            draw.text((px, PLOT_BOTTOM + 10), label, font=font, fill=COLOR_TEXT_DIM, anchor="ma")
            draw.text((PLOT_LEFT - 11, py), label, font=font, fill=COLOR_TEXT_DIM, anchor="rm")

        draw.rectangle(
            [PLOT_LEFT - 1, PLOT_TOP - 1, PLOT_RIGHT, PLOT_BOTTOM],
            outline=(120, 110, 98),
            width=1,
        )

    def _draw_legend(self, canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        font = get_font(18)
        rows = [
            ("beacon", "Beacons"),
            ("goal", "Goal State"),
            ("start", "Start State"),
            ("obstacle", "Obstacles"),
            ("belief", "Belief Particles"),
        ]
        top = PLOT_TOP
        row_h = 38
        height = row_h * len(rows) + 22
        draw.rounded_rectangle(
            [LEGEND_LEFT, top, LEGEND_RIGHT, top + height],
            radius=10,
            fill=COLOR_PANEL,
            outline=COLOR_PANEL_EDGE,
            width=1,
        )
        icon_x = LEGEND_LEFT + 26
        text_x = LEGEND_LEFT + 48
        for i, (kind, label) in enumerate(rows):
            cy = top + 11 + row_h // 2 + i * row_h
            if kind == "beacon":
                draw.polygon(
                    [(icon_x, cy - 9), (icon_x + 10, cy + 7), (icon_x - 10, cy + 7)],
                    fill=COLOR_BEACON_MARK,
                )
            elif kind == "goal":
                sprite = goal_sprite(22)
                canvas.paste(sprite, (icon_x - 11, cy - 11), sprite)
            elif kind == "start":
                draw.ellipse([icon_x - 9, cy - 9, icon_x + 9, cy + 9], fill=COLOR_START)
            elif kind == "obstacle":
                draw.ellipse([icon_x - 9, cy - 9, icon_x + 9, cy + 9], fill=(58, 56, 58))
            else:
                draw.ellipse([icon_x - 9, cy - 9, icon_x + 9, cy + 9], fill=COLOR_BELIEF)
            draw.text((text_x, cy), label, font=font, fill=COLOR_TEXT, anchor="lm")

    def _build_background(self) -> Image.Image:
        """Rasterise everything that does not move. Called once per episode."""
        canvas = Image.new("RGB", CANVAS_SIZE, COLOR_PAGE)
        ground = (self._lit_ground() * 255.0 + 0.5).astype(np.uint8)
        canvas.paste(Image.fromarray(ground, mode="RGB"), (PLOT_LEFT, PLOT_TOP))

        draw = ImageDraw.Draw(canvas, "RGBA")

        # Hazard rims: an outline makes the extent of each field readable even
        # where two fields overlap and the wash saturates.
        obstacles = np.asarray(self.environment.obstacles, dtype=float)
        for i in range(obstacles.shape[1]):
            ox, oy = float(obstacles[0, i]), float(obstacles[1, i])
            r = float(self.environment.obstacle_radius)
            x_lo, y_hi = self._to_px(ox - r, oy - r)
            x_hi, y_lo = self._to_px(ox + r, oy + r)
            draw.ellipse([x_lo, y_lo, x_hi, y_hi], outline=(240, 96, 84, 130), width=2)
            cx, cy = self._to_px(ox, oy)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=COLOR_OBSTACLE_DOT)

        unit = self._px_per_unit
        beacon_px = int(np.clip(0.85 * unit, 18, 64))
        beacons = np.asarray(self.environment.beacons, dtype=float)
        sprite = beacon_sprite(beacon_px)
        for i in range(beacons.shape[1]):
            cx, cy = self._to_px(float(beacons[0, i]), float(beacons[1, i]))
            canvas.paste(sprite, (int(cx - beacon_px / 2), int(cy - beacon_px / 2)), sprite)

        goal_px = int(np.clip(0.95 * unit, 20, 60))
        gx, gy = self._to_px(*_xy(self.environment.goal_state))
        halo = glow_sprite(int(goal_px * 1.1), COLOR_GOAL, power=2.0)
        canvas.paste(halo, (int(gx - halo.width / 2), int(gy - halo.height / 2)), halo)
        gsprite = goal_sprite(goal_px)
        canvas.paste(gsprite, (int(gx - goal_px / 2), int(gy - goal_px / 2)), gsprite)

        start_px = int(np.clip(0.34 * unit, 10, 26))
        sx, sy = self._to_px(*_xy(self.environment.start_state))
        ssprite = start_sprite(start_px)
        canvas.paste(ssprite, (int(sx - start_px / 2), int(sy - start_px / 2)), ssprite)

        self._draw_axes(draw)
        self._draw_legend(canvas, draw)

        draw.text(
            ((PLOT_LEFT + PLOT_RIGHT) / 2, 12),
            _pretty_title(self.environment),
            font=get_font(30),
            fill=COLOR_TEXT,
            anchor="ma",
        )
        draw.text(
            ((PLOT_LEFT + PLOT_RIGHT) / 2, 47),
            "Agent Path",
            font=get_font(17),
            fill=COLOR_TEXT_DIM,
            anchor="ma",
        )
        return canvas

    # --- dynamic layer -------------------------------------------------------

    @staticmethod
    def _trail_color(index: int, count: int) -> Tuple[int, int, int]:
        """Belief trail colour: old frames red, the newest frame yellow.

        Same yellow-to-red ramp the old renderer used, so a reader who knows the
        old GIFs reads the new ones the same way.
        """
        t = (count - 1 - index) / max(count, 1)
        yellow = np.array(COLOR_BELIEF, dtype=float)
        red = np.array([220.0, 40.0, 34.0])
        return tuple(int(v) for v in (yellow * (1.0 - t) + red * t))  # type: ignore[return-value]

    @property
    def _px_per_unit_axes(self) -> Tuple[float, float]:
        """Pixels per world unit on each axis separately.

        The axes are not equally scaled (they were not in the old figure
        either), so the action arrow has to use the per-axis scale or it points
        somewhere the rover will not go.
        """
        lo, hi = self._world_limits
        span = hi - lo
        return (PLOT_RIGHT - PLOT_LEFT) / span, (PLOT_BOTTOM - PLOT_TOP) / span

    @staticmethod
    def _action_label(action: Any) -> str:
        if action is None:
            return "-"
        if isinstance(action, str):
            return action
        arr = np.asarray(action, dtype=float).ravel()
        return "(" + ", ".join(f"{v:.2f}" for v in arr[:2]) + ")"

    def _action_vector(self, action: Any) -> Tuple[float, float]:
        if action is None:
            return 0.0, 0.0
        if isinstance(action, str):
            dx, dy = self.environment.action_to_vector[action]
            return float(dx), float(dy)
        arr = np.asarray(action, dtype=float).ravel()
        if arr.size < 2:
            return 0.0, 0.0
        return float(arr[0]), float(arr[1])

    def _prepare_belief(self, belief: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert one belief to pixel centres and radii, once per episode.

        A belief is drawn again in every frame of its trail, so converting world
        coordinates to pixels inside the frame loop repeats the same work up to
        ten times per belief.
        """
        values = getattr(belief, "values", None)
        if values is None or len(values) == 0:
            return None
        probs = np.asarray(belief.probs, dtype=float).ravel()
        positions = np.asarray([_xy(v) for v in values], dtype=float)
        if probs.size > MAX_PARTICLES_PER_LAYER:
            # Thousands of particles cost time but add no visible dots, so keep
            # the heaviest ones.
            keep = np.argpartition(-probs, MAX_PARTICLES_PER_LAYER - 1)[:MAX_PARTICLES_PER_LAYER]
            probs = probs[keep]
            positions = positions[keep]

        lo, hi = self._world_limits
        span = hi - lo
        px = PLOT_LEFT + (positions[:, 0] - lo) / span * (PLOT_RIGHT - PLOT_LEFT)
        py = PLOT_BOTTOM - (positions[:, 1] - lo) / span * (PLOT_BOTTOM - PLOT_TOP)
        # Radius mirrors the old scatter sizing (area = prob * 600 points^2)
        # converted from points to pixels at the old 100 dpi figure.
        radii = np.clip(np.sqrt(np.maximum(probs, 0.0) * 600.0 / np.pi) * 1.389, 2.0, 14.0)
        return np.stack([px, py], axis=1), radii

    def _draw_belief(
        self,
        draw: ImageDraw.ImageDraw,
        belief_px: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
        n_path: int,
        frame: int,
    ) -> None:
        # Trail length keys off the path, not the belief list, matching the old
        # renderer. cache_visualization makes the two equal, but visualize_path
        # is public and accepts mismatched lists.
        trail = min(n_path, MAX_BELIEF_TRAIL)
        for layer in range(trail):
            history_frame = frame - (trail - 1 - layer)
            if not (0 <= history_frame < len(belief_px)):
                continue
            prepared = belief_px[history_frame]
            if prepared is None:
                continue
            centres, radii = prepared
            color = self._trail_color(layer, trail)
            alpha = int(255 * (0.22 + 0.62 * (layer + 1) / trail))
            fill = color + (alpha,)
            for (cx, cy), r in zip(centres, radii):
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    @staticmethod
    def _headings(headings_source: Sequence[Tuple[float, float]]) -> List[int]:
        """Rover heading per frame, in 15-degree buckets.

        Computed for the whole episode up front rather than carried on the
        instance: heading held as instance state would leak from one render into
        the next and break the byte-for-byte determinism the golden test needs.
        A frame with no action keeps the previous frame's heading.
        """
        headings: List[int] = []
        current = 0
        for dx, dy in headings_source:
            if dx or dy:
                current = int(round(np.degrees(np.arctan2(dy, dx)) / 15.0) * 15) % 360
            headings.append(current)
        return headings

    def _draw_dynamic(
        self,
        overlay: Image.Image,
        path_px: Sequence[Tuple[float, float]],
        vectors: Sequence[Tuple[float, float]],
        headings: Sequence[int],
        action_labels: Sequence[str],
        belief_px: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
        frame: int,
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """Draw the moving parts into ``overlay``; return the rover to paste.

        The rover is returned rather than pasted here because pasting an RGBA
        sprite into a fully transparent overlay and then compositing that
        overlay multiplies by alpha twice, which leaves a dark fringe on the
        sprite's antialiased rim.  It goes onto the opaque canvas instead.
        """
        draw = ImageDraw.Draw(overlay, "RGBA")
        self._draw_belief(draw, belief_px, len(path_px), frame)

        pts = list(path_px[: frame + 1])
        if len(pts) > 1:
            draw.line(pts, fill=COLOR_PATH + (170,), width=3, joint="curve")
        for px, py in pts[:-1]:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=COLOR_PATH + (200,))

        cx, cy = path_px[frame]
        dx, dy = vectors[frame]
        if dx or dy:
            unit_px = self._px_per_unit_axes
            tx, ty = cx + dx * unit_px[0], cy - dy * unit_px[1]
            vx, vy = tx - cx, ty - cy
            length = float(np.hypot(vx, vy))
            if length > 1.0:
                ux, uy = vx / length, vy / length
                head = min(16.0, max(8.0, length * 0.45))
                bx, by = tx - ux * head, ty - uy * head
                draw.line([(cx, cy), (bx, by)], fill=COLOR_PATH + (255,), width=6)
                draw.polygon(
                    [
                        (tx, ty),
                        (bx - uy * head * 0.55, by + ux * head * 0.55),
                        (bx + uy * head * 0.55, by - ux * head * 0.55),
                    ],
                    fill=COLOR_PATH + (255,),
                )

        draw.text(
            (PLOT_LEFT + 12, PLOT_TOP + 10),
            f"step {frame + 1}/{len(path_px)}   action {action_labels[frame]}",
            font=get_font(16),
            fill=COLOR_TEXT + (215,),
        )

        rover_px = int(np.clip(0.72 * self._px_per_unit, 20, 60))
        sprite = rover_sprite_facing(rover_px, headings[frame])
        return sprite, (int(cx - rover_px / 2), int(cy - rover_px / 2))

    # --- public API ----------------------------------------------------------

    def render_frames(
        self,
        path: List[np.ndarray],
        agent_belief_path: List[DiscreteDistribution],
        actions: List[Any],
    ) -> List[Image.Image]:
        """Render every frame of an episode as RGB images.

        Exposed so tests and contact sheets can look at single frames without
        decoding a GIF.  Everything that does not depend on the frame index --
        the background, the pixel coordinates of the path and of every belief,
        the action vectors and headings -- is computed here, once.
        """
        key = self._scene_key()
        if self._background is None or self._background_key != key:
            self._background = self._build_background()
            self._background_key = key

        path_px = [self._to_px(*_xy(p)) for p in path]
        belief_px = [self._prepare_belief(b) for b in agent_belief_path]
        vectors = [
            self._action_vector(actions[i] if i < len(actions) else None)
            for i in range(len(path))
        ]
        headings = self._headings(vectors)
        labels = [
            self._action_label(actions[i] if i < len(actions) else None)
            for i in range(len(path))
        ]

        frames: List[Image.Image] = []
        for frame in range(len(path)):
            canvas = self._background.copy()
            overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
            rover, rover_at = self._draw_dynamic(
                overlay, path_px, vectors, headings, labels, belief_px, frame
            )
            canvas.paste(overlay, (0, 0), overlay)
            canvas.paste(rover, rover_at, rover)
            frames.append(canvas)
        return frames

    @staticmethod
    def _build_palette(reference: Image.Image) -> Image.Image:
        """One GIF palette for the whole animation.

        Median cut on its own weights colours by how many pixels have them, so
        the huge dark-brown floor swallows almost every slot and the small vivid
        markers -- the goal star, the belief particles, the legend -- come out
        muddy.  So the adaptive palette gets the bulk of the slots and the
        accent colours are appended by hand.

        Every frame then shares this palette, which also lets the encoder skip a
        colour analysis per frame.
        """
        adaptive = reference.quantize(
            colors=256 - len(ACCENT_COLORS), method=Image.Quantize.MEDIANCUT
        )
        entries = adaptive.getpalette() or []
        entries = entries[: 3 * (256 - len(ACCENT_COLORS))]
        for color in ACCENT_COLORS:
            entries.extend(color)
        entries.extend([0] * (768 - len(entries)))
        master = Image.new("P", (1, 1))
        master.putpalette(entries)
        return master

    def visualize_path(
        self,
        path: List[np.ndarray],
        agent_belief_path: List[DiscreteDistribution],
        actions: List[str],
        cache_path: Path,
    ):
        """Create and save an animated visualization of the agent's path.

        Args:
            path: List of state positions (2D numpy arrays) along the agent's trajectory.
            agent_belief_path: List of belief distributions at each step.
            actions: List of actions taken at each step.
            cache_path: Path where to save the visualization (must end with .gif).

        Raises:
            TypeError: If cache_path is not a Path object.
            ValueError: If cache_path doesn't end with .gif.
            IndexError: If ``path`` is empty.
            FileNotFoundError: If the parent directory of ``cache_path`` does
                not exist.
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        # An empty path still raises IndexError here, and the missing parent
        # directory still raises FileNotFoundError on save. Both are the old
        # renderer's behaviour and callers' tests pin them; cache_visualization
        # is the entry point that creates directories.
        frames = self.render_frames(path, agent_belief_path, actions)

        master = self._build_palette(frames[0])
        indexed = [f.quantize(palette=master, dither=Image.Dither.NONE) for f in frames]

        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=indexed[1:],
            duration=int(1000 / GIF_FPS),
            loop=0,
            optimize=True,
        )

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of agent's path and belief.

        Args:
            history: List of step data from an episode.
            cache_path: Path where to save the visualization.

        Raises:
            TypeError: If history is not a List or contains non-StepData objects,
                or if cache_path is not a Path object.
            ValueError: If history is empty or contains invalid data.
        """
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        # Extract data with validation
        agent_path = []
        agent_belief_path: List[DiscreteDistribution] = []
        actions = []

        for step in history:
            if (
                not hasattr(step, "state")
                or not hasattr(step, "belief")
                or not hasattr(step, "action")
            ):
                raise ValueError(f"History step missing required attributes: {step}")

            agent_path.append(step.state)
            if isinstance(step.belief, WeightedParticleBelief):
                agent_belief_path.append(step.belief.to_unique_support_distribution())
            else:
                particles = [step.belief.sample() for _ in range(20)]
                weights = np.ones(len(particles)) / len(particles)
                discrete_distribution = DiscreteDistribution(values=particles, probs=weights)
                agent_belief_path.append(discrete_distribution)

            actions.append(step.action)

        # Validate all lists have same length
        if not len(agent_path) == len(agent_belief_path) == len(actions):
            raise ValueError(
                f"Mismatched lengths: path={len(agent_path)}, belief={len(agent_belief_path)}, actions={len(actions)}"
            )

        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.visualize_path(
            path=agent_path,
            agent_belief_path=agent_belief_path,
            actions=actions,
            cache_path=cache_path,
        )
