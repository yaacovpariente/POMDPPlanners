# SPDX-License-Identifier: MIT

"""Shared direct-frame drawing for the discrete and continuous Push worlds."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.push_pomdp.push_visualization_assets import (
    paste_sprite,
    paste_obstacle,
    stone_texture,
)

CANVAS_SIZE = (1200, 1000)
PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM = 80, 38, 981, 939
GIF_DURATION_MS = 1250

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID = (222, 222, 222)
ROBOT = (0, 0, 255)
ROBOT_EDGE = (0, 0, 139)
ROBOT_RADIUS = (191, 191, 255)
OBJECT = (255, 165, 0)
OBJECT_EDGE = (255, 140, 0)
TARGET = (218, 165, 0)
TARGET_EDGE = (184, 134, 11)
OBSTACLE = (174, 88, 61)
OBSTACLE_EDGE = (94, 43, 29)
DANGER = (255, 178, 178)
ACTION = (255, 0, 0)
PUSH_LINE = (255, 0, 0)
STEP_BOX = (190, 224, 235)
DISTANCE_BOX = (255, 255, 224)
REWARD_BOX = (144, 238, 144)
SUCCESS_BOX = (144, 238, 144)
SUCCESS_TEXT = (0, 100, 0)
COLLISION_BOX = (240, 128, 128)
COLLISION_TEXT = (139, 0, 0)

ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    WHITE,
    BLACK,
    GRID,
    ROBOT,
    ROBOT_EDGE,
    ROBOT_RADIUS,
    OBJECT,
    OBJECT_EDGE,
    TARGET,
    TARGET_EDGE,
    OBSTACLE,
    OBSTACLE_EDGE,
    DANGER,
    ACTION,
    PUSH_LINE,
    STEP_BOX,
    DISTANCE_BOX,
    REWARD_BOX,
    SUCCESS_BOX,
    SUCCESS_TEXT,
    COLLISION_BOX,
    COLLISION_TEXT,
)


def _font(size: int) -> Any:
    """Use Pillow's bundled font so frames do not depend on system fonts."""
    return ImageFont.load_default(size=size)


FONT_SMALL = _font(14)
FONT_NORMAL = _font(17)
FONT_TITLE = _font(24)
FONT_STATUS = _font(24)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    """Draw a line and a filled triangular head."""
    draw.line((start, end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return
    ux, uy = dx / length, dy / length
    head, wing = 11 + width, 6 + width / 2
    base_x, base_y = end[0] - ux * head, end[1] - uy * head
    draw.polygon(
        (end, (base_x - uy * wing, base_y + ux * wing), (base_x + uy * wing, base_y - ux * wing)),
        fill=color,
    )


def draw_star(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    outer: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> None:
    """Draw the target marker without a font glyph."""
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else outer * 0.42
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fill: Tuple[int, int, int],
    font: Any = FONT_NORMAL,
    text_fill: Tuple[int, int, int] = BLACK,
    outline: Tuple[int, int, int] = BLACK,
    padding: int = 8,
) -> Tuple[float, float, float, float]:
    """Draw a measured multiline label and return its box."""
    bounds = draw.multiline_textbbox(xy, text, font=font, spacing=3)
    box = (bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding)
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=2)
    draw.line((box[0] + 8, box[1] + 2, box[2] - 8, box[1] + 2), fill=(245, 235, 205), width=1)
    draw.multiline_text(xy, text, fill=text_fill, font=font, spacing=3)
    return box


class PushRendererBase(ABC):
    """Common cached-background renderer for both Push variants."""

    title = "Push POMDP Episode Visualization"

    def __init__(self, env: Any):
        self.env = env
        self.grid_size = env.grid_size
        self.push_threshold = env.push_threshold
        self.obstacles = env.obstacles
        self.dangerous_areas = env.dangerous_areas
        self.dangerous_area_radius = env.dangerous_area_radius
        self._background: Image.Image | None = None
        self._background_key: Tuple[Any, ...] | None = None
        self._background_build_count = 0

    @property
    def _world_limits(self) -> Tuple[float, float]:
        return -0.5, float(self.grid_size) + 0.5

    @property
    def _px_per_unit(self) -> float:
        lo, hi = self._world_limits
        return min(PLOT_RIGHT - PLOT_LEFT, PLOT_BOTTOM - PLOT_TOP) / (hi - lo)

    def _to_px(self, point: Sequence[float] | np.ndarray) -> Tuple[float, float]:
        lo, hi = self._world_limits
        x, y = float(point[0]), float(point[1])
        px = PLOT_LEFT + (x - lo) / (hi - lo) * (PLOT_RIGHT - PLOT_LEFT)
        py = PLOT_BOTTOM - (y - lo) / (hi - lo) * (PLOT_BOTTOM - PLOT_TOP)
        return px, py

    def _scene_key(self) -> Tuple[Any, ...]:
        return (
            type(self).__name__,
            float(self.grid_size),
            float(self.dangerous_area_radius),
            np.asarray(self.obstacles, dtype=float).tobytes(),
            np.asarray(self.dangerous_areas, dtype=float).tobytes(),
            self._variant_scene_key(),
        )

    def _variant_scene_key(self) -> Tuple[Any, ...]:
        return ()

    def _get_static_background(self) -> Image.Image:
        key = self._scene_key()
        if self._background is None or self._background_key != key:
            self._background = self._build_static_background()
            self._background_key = key
        return self._background

    def _build_static_background(self) -> Image.Image:
        self._background_build_count += 1
        canvas = stone_texture(*CANVAS_SIZE, dark=True).copy()
        canvas.paste(
            stone_texture(PLOT_RIGHT - PLOT_LEFT, PLOT_BOTTOM - PLOT_TOP), (PLOT_LEFT, PLOT_TOP)
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            ((PLOT_LEFT + PLOT_RIGHT) // 2, 5),
            self.title,
            fill=(247, 222, 172),
            font=FONT_TITLE,
            anchor="ma",
        )
        for inset, color in ((-5, (27, 24, 20)), (-3, (167, 145, 108)), (-1, (68, 57, 43))):
            draw.rectangle(
                (PLOT_LEFT + inset, PLOT_TOP + inset, PLOT_RIGHT - inset, PLOT_BOTTOM - inset),
                outline=color,
                width=2,
            )
        for tile in range(0, int(self.grid_size) + 1):
            px, _ = self._to_px((tile, 0))
            _, py = self._to_px((0, tile))
            draw.line((px, PLOT_TOP, px, PLOT_BOTTOM), fill=(132, 116, 91))
            draw.line((px + 1, PLOT_TOP, px + 1, PLOT_BOTTOM), fill=(184, 166, 135))
            draw.line((PLOT_LEFT, py, PLOT_RIGHT, py), fill=(132, 116, 91))
            draw.line((PLOT_LEFT, py + 1, PLOT_RIGHT, py + 1), fill=(184, 166, 135))
        tick_step = max(1, int(round(self.grid_size / 5)))
        for tick in range(0, int(self.grid_size) + 1, tick_step):
            px, _ = self._to_px((tick, 0))
            _, py = self._to_px((0, tick))
            draw.text(
                (px, PLOT_BOTTOM + 8), str(tick), fill=(233, 222, 195), font=FONT_SMALL, anchor="ma"
            )
            draw.text(
                (PLOT_LEFT - 10, py), str(tick), fill=(233, 222, 195), font=FONT_SMALL, anchor="rm"
            )
        draw.text(
            ((PLOT_LEFT + PLOT_RIGHT) // 2, 976),
            "X Position",
            fill=(233, 222, 195),
            font=FONT_NORMAL,
            anchor="mm",
        )
        label = Image.new("RGBA", (130, 30), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label)
        label_draw.text((65, 15), "Y Position", fill=(233, 222, 195), font=FONT_NORMAL, anchor="mm")
        rotated = label.rotate(90, expand=True)
        canvas.paste(rotated, (13, (CANVAS_SIZE[1] - rotated.height) // 2), rotated)
        radius = float(self.dangerous_area_radius) * self._px_per_unit
        for center in self.dangerous_areas:
            x, y = self._to_px(center)
            hazard_draw = ImageDraw.Draw(canvas, "RGBA")
            hazard_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(185, 58, 37, 95),
                outline=(170, 65, 44, 220),
                width=2,
            )
        self._draw_static_obstacles(canvas)
        self._draw_legend(canvas)
        return canvas

    @abstractmethod
    def _draw_static_obstacles(self, canvas: Image.Image) -> None:
        """Draw variant-specific obstacle geometry."""

    def _draw_legend(self, canvas: Image.Image) -> None:
        draw = ImageDraw.Draw(canvas)
        left, top, right = 1032, 46, 1194
        entries = self._legend_entries()
        bottom = top + 30 + len(entries) * 31
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=8,
            fill=(37, 32, 25),
            outline=(181, 152, 102),
            width=2,
        )
        for index, (name, kind) in enumerate(entries):
            y = top + 22 + index * 31
            if kind in ("robot", "object", "target"):
                paste_sprite(canvas, kind, (left + 21, y), 32)
            elif kind == "obstacle":
                paste_obstacle(canvas, (left + 11, y - 10, left + 31, y + 10), circle=True)
            elif kind == "danger":
                canvas.paste(stone_texture(128, 128).crop((50, 50, 74, 64)), (left + 9, y - 7))
                ImageDraw.Draw(canvas, "RGBA").rectangle(
                    (left + 9, y - 7, left + 32, y + 6),
                    fill=(185, 58, 37, 95),
                    outline=(170, 65, 44, 220),
                    width=1,
                )
            else:
                self._draw_legend_symbol(draw, (left + 21, y), kind)
            draw.text((left + 45, y), name, fill=(239, 227, 200), font=FONT_SMALL, anchor="lm")

    def _legend_entries(self) -> List[Tuple[str, str]]:
        return [
            ("Robot", "robot"),
            ("Object", "object"),
            ("Target", "target"),
            ("Obstacles", "obstacle"),
            ("Dangerous Areas", "danger"),
            ("Action", "action"),
        ]

    def _draw_legend_symbol(
        self, draw: ImageDraw.ImageDraw, center: Tuple[int, int], kind: str
    ) -> None:
        x, y = center
        if kind == "robot":
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=ROBOT, outline=ROBOT_EDGE, width=2)
        elif kind == "radius":
            draw.ellipse(
                (x - 10, y - 10, x + 10, y + 10), fill=ROBOT_RADIUS, outline=(120, 120, 220)
            )
        elif kind == "object":
            draw.rectangle((x - 9, y - 9, x + 9, y + 9), fill=OBJECT, outline=OBJECT_EDGE, width=2)
        elif kind == "target":
            draw_star(draw, center, 12, TARGET, TARGET_EDGE)
        elif kind == "obstacle":
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=OBSTACLE)
        elif kind == "danger":
            draw.rectangle((x - 12, y - 7, x + 12, y + 7), fill=DANGER)
        else:
            draw_arrow(draw, (x - 13, y), (x + 14, y), ACTION, 4)

    def _validate_visualization_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        if any(not isinstance(step, StepData) for step in history):
            raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Render one GIF frame per recorded state and create parent folders."""
        self._validate_visualization_inputs(history, cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames = self.render_frames(history)
        master = self._build_palette(frames[0])
        indexed = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in frames]
        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=indexed[1:],
            duration=GIF_DURATION_MS,
            loop=0,
            optimize=False,
            disposal=2,
        )

    def render_frames(self, history: List[StepData]) -> List[Image.Image]:
        """Return RGB frames; useful for tests and GIF encoding."""
        background = self._get_static_background()
        rewards = [step.reward for step in history]
        return [
            self._render_frame(background.copy(), history, rewards, index)
            for index in range(len(history))
        ]

    @staticmethod
    def _build_palette(reference: Image.Image) -> Image.Image:
        adaptive = reference.quantize(
            colors=256 - len(ACCENT_COLORS), method=Image.Quantize.MEDIANCUT
        )
        entries = (adaptive.getpalette() or [])[: 3 * (256 - len(ACCENT_COLORS))]
        for color in ACCENT_COLORS:
            entries.extend(color)
        entries.extend([0] * (768 - len(entries)))
        master = Image.new("P", (1, 1))
        master.putpalette(entries)
        return master

    def _render_frame(
        self, canvas: Image.Image, history: List[StepData], rewards: List[float | None], frame: int
    ) -> Image.Image:
        draw = ImageDraw.Draw(canvas)
        state = np.asarray(history[frame].state, dtype=float)
        robot_pos, object_pos, target_pos = state[:2], state[2:4], state[4:6]
        distance_to_target = float(np.linalg.norm(object_pos - target_pos))
        robot_to_object = float(np.linalg.norm(robot_pos - object_pos))
        action = history[frame].action if frame < len(history) - 1 else None
        self._draw_variant_robot(draw, robot_pos)
        if action is not None:
            direction = self._action_vector(action)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                unit = direction / norm
                start = self._to_px(robot_pos)
                end = self._to_px(robot_pos + unit * 0.6)
                if robot_to_object < self.push_threshold:
                    draw.line((start, self._to_px(object_pos)), fill=PUSH_LINE, width=3)
                    draw_arrow(draw, start, end, ACTION, 6)
                draw_arrow(draw, start, end, ACTION, 4)
        self._draw_entities(canvas, robot_pos, object_pos, target_pos)
        self._draw_frame_text(
            draw, frame, len(history), action, rewards, distance_to_target, robot_to_object
        )
        robot_collision, object_collision = self._collisions(robot_pos, object_pos)
        if distance_to_target < 0.5:
            self._draw_centered_status(
                draw, "TARGET REACHED!\nEpisode Complete", 0.50, SUCCESS_BOX, SUCCESS_TEXT
            )
        if robot_collision or object_collision:
            names = [
                name
                for name, hit in (("Robot", robot_collision), ("Object", object_collision))
                if hit
            ]
            self._draw_centered_status(
                draw, f"{' & '.join(names)} Collision!", 0.70, COLLISION_BOX, COLLISION_TEXT
            )
        return canvas

    def _draw_variant_robot(self, draw: ImageDraw.ImageDraw, robot_pos: np.ndarray) -> None:
        del draw, robot_pos

    def _draw_entities(
        self,
        canvas: Image.Image,
        robot_pos: np.ndarray,
        object_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> None:
        rx, ry = self._to_px(robot_pos)
        ox, oy = self._to_px(object_pos)
        tx, ty = self._to_px(target_pos)
        paste_sprite(canvas, "target", (tx, ty), 52)
        paste_sprite(canvas, "object", (ox, oy), 54)
        paste_sprite(canvas, "robot", (rx, ry), 52)

    def _draw_frame_text(
        self,
        draw: ImageDraw.ImageDraw,
        frame: int,
        frame_count: int,
        action: Any,
        rewards: List[float | None],
        distance_to_target: float,
        robot_to_object: float,
    ) -> None:
        label = "Terminal" if action is None else self._format_action(action)
        draw_text_box(draw, (99, 58), f"Step: {frame + 1}/{frame_count}\nAction: {label}", STEP_BOX)
        arrow = self._distance_arrow()
        draw_text_box(
            draw,
            (99, 151),
            f"Object {arrow} Target: {distance_to_target:.2f}\nRobot {arrow} Object: {robot_to_object:.2f}",
            DISTANCE_BOX,
        )
        current = rewards[frame] if rewards[frame] is not None else 0.0
        total = sum(value for value in rewards[: frame + 1] if value is not None)
        draw_text_box(
            draw, (99, 244), f"Step Reward: {current:.1f}\nTotal Reward: {total:.1f}", REWARD_BOX
        )

    def _distance_arrow(self) -> str:
        return "<->"

    def _draw_centered_status(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        vertical_fraction: float,
        fill: Tuple[int, int, int],
        text_fill: Tuple[int, int, int],
    ) -> None:
        center = (
            (PLOT_LEFT + PLOT_RIGHT) // 2,
            int(PLOT_TOP + vertical_fraction * (PLOT_BOTTOM - PLOT_TOP)),
        )
        bounds = draw.multiline_textbbox(
            center, text, font=FONT_STATUS, anchor="mm", align="center", spacing=4
        )
        box = (bounds[0] - 18, bounds[1] - 16, bounds[2] + 18, bounds[3] + 16)
        draw.rounded_rectangle(box, radius=12, fill=fill, outline=text_fill, width=3)
        draw.multiline_text(
            center, text, fill=text_fill, font=FONT_STATUS, anchor="mm", align="center", spacing=4
        )

    @abstractmethod
    def _action_vector(self, action: Any) -> np.ndarray:
        """Convert the variant's action to a vector."""

    def _format_action(self, action: Any) -> str:
        return str(action)

    @abstractmethod
    def _collisions(self, robot_pos: np.ndarray, object_pos: np.ndarray) -> Tuple[bool, bool]:
        """Return robot and object obstacle collision flags."""
