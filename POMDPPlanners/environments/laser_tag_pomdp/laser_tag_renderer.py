# SPDX-License-Identifier: MIT
"""Shared cached Pillow frame drawing for LaserTag visualizers."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_assets import (
    _render_opponent_pillow,
    _render_robot_pillow,
)

CANVAS_SIZE = (1400, 800)
LASER_COLOR = (153, 204, 153)
ROBOT_BELIEF_COLOR = (246, 179, 179)
OPPONENT_BELIEF_COLOR = (206, 232, 240)


class LaserTagFrameRenderer:
    """Draw the frame parts shared by continuous and discrete LaserTag."""

    def __init__(
        self,
        grid_size: np.ndarray,
        walls: np.ndarray,
        robot_radius: float,
        opponent_radius: float,
        dangerous_areas: List[Tuple[float, float]],
        dangerous_area_radius: float,
    ):
        self.grid_size = np.asarray(grid_size, dtype=float)
        self._wall_rectangles = np.asarray(walls, dtype=float).reshape(-1, 4)
        self.walls: np.ndarray | set[Tuple[int, int]] = self._wall_rectangles
        self.robot_radius = robot_radius
        self.opponent_radius = opponent_radius
        self.dangerous_areas = dangerous_areas
        self.dangerous_area_radius = dangerous_area_radius
        self._title = "Continuous LaserTag POMDP Episode"
        self._x_label = "X"
        self._y_label = "Y"
        self._robot_img = _render_robot_pillow()
        self._opponent_img = _render_opponent_pillow()

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Save one one-second frame per recorded state, including terminal records."""
        self._validate_inputs(history, cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        robot_path, opponent_path, actions, beliefs = self._extract_history(history)
        background = self._build_background()
        palette = self._make_palette(background)
        frames = [
            self._render_frame(
                background,
                rp,
                op,
                action,
                belief,
                index,
                len(history),
                robot_path[: index + 1],
                opponent_path[: index + 1],
            ).quantize(palette=palette, dither=Image.Dither.NONE)
            for index, (rp, op, action, belief) in enumerate(
                zip(robot_path, opponent_path, actions, beliefs)
            )
        ]
        # The visible step counter keeps repeated states from merging in the GIF.
        frames[0].save(
            cache_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,
            loop=0,
            disposal=2,
            optimize=False,
        )

    def _validate_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, list):
            raise TypeError("history must be a list")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must contain StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_history(self, history: List[StepData]) -> Tuple:
        robot_path, opponent_path, actions, beliefs = [], [], [], []
        for step in history:
            if not isinstance(step.state, np.ndarray) or len(step.state) != 5:
                raise ValueError("Expected numpy state with shape (5,)")
            robot_path.append(step.state[:2].copy())
            opponent_path.append(step.state[2:4].copy())
            actions.append(step.action)
            beliefs.append(getattr(step, "belief", None))
        return robot_path, opponent_path, actions, beliefs

    def _world_to_pixel(self, point) -> Tuple[float, float]:
        return (
            self._left + (float(point[0]) + 0.5) * self._scale,
            self._bottom - (float(point[1]) + 0.5) * self._scale,
        )

    def _build_background(self) -> Image.Image:
        """Rasterize the world and axes once for this episode."""
        width, height = self.grid_size
        self._scale = min(1240 / (width + 1), 680 / (height + 1))
        plot_width, plot_height = (width + 1) * self._scale, (height + 1) * self._scale
        self._left = (CANVAS_SIZE[0] - plot_width) / 2
        self._top = 65 + (680 - plot_height) / 2
        self._bottom = self._top + plot_height
        self._right = self._left + plot_width
        self._font = ImageFont.load_default(size=16)
        self._title_font = ImageFont.load_default(size=23)
        self._tag_font = ImageFont.load_default(size=32)
        self._robot_sprite = Image.fromarray(self._robot_img).resize(
            (89, 89), Image.Resampling.LANCZOS
        )
        self._opponent_sprite = Image.fromarray(self._opponent_img).resize(
            (89, 89), Image.Resampling.LANCZOS
        )
        image = Image.new("RGB", CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (700, 23),
            self._title,
            fill="black",
            font=self._title_font,
            anchor="mt",
        )
        draw.rectangle((self._left, self._top, self._right, self._bottom), outline="#444444")
        # Bound the tick count for large arenas.
        tick_step = max(1, int(np.ceil(max(width, height) / 12)))
        for x in np.arange(0, width + 0.001, tick_step):
            px, _ = self._world_to_pixel((x, 0))
            draw.line((px, self._top, px, self._bottom), fill="#eeeeee")
            draw.text((px, self._bottom + 8), f"{x:g}", fill="black", font=self._font, anchor="mt")
        for y in np.arange(0, height + 0.001, tick_step):
            _, py = self._world_to_pixel((0, y))
            draw.line((self._left, py, self._right, py), fill="#eeeeee")
            draw.text((self._left - 10, py), f"{y:g}", fill="black", font=self._font, anchor="rm")
        draw.text((700, 783), self._x_label, fill="black", font=self._font, anchor="mb")
        label_bounds = self._font.getbbox(self._y_label)
        label = Image.new(
            "RGB",
            (
                round(label_bounds[2] - label_bounds[0]) + 2,
                round(label_bounds[3] - label_bounds[1]) + 2,
            ),
            "white",
        )
        ImageDraw.Draw(label).text(
            (0, 0), self._y_label, fill="black", font=self._font, anchor="lt"
        )
        label = label.rotate(90, expand=True)
        image.paste(label, (round(max(8, self._left - 55)), round(400 - label.height / 2)))
        axes = image.copy()
        for cx, cy, hx, hy in self._wall_rectangles:
            first = self._world_to_pixel((cx - hx, cy + hy))
            second = self._world_to_pixel((cx + hx, cy - hy))
            draw.rectangle(
                (
                    min(first[0], second[0]),
                    min(first[1], second[1]),
                    max(first[0], second[0]),
                    max(first[1], second[1]),
                ),
                fill="#4c4c4c",
            )
        overlay = ImageDraw.Draw(image, "RGBA")
        for center in self.dangerous_areas:
            x, y = self._world_to_pixel(center)
            radius = self.dangerous_area_radius * self._scale
            overlay.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0, 76))
        bounds = (
            round(self._left) + 1,
            round(self._top) + 1,
            round(self._right),
            round(self._bottom),
        )
        axes.paste(image.crop(bounds), bounds[:2])
        image = axes
        self._draw_legend(ImageDraw.Draw(image))
        self._legend_bounds = (
            round(self._right - 155),
            round(self._top + 14),
            round(self._right - 14),
            round(self._top + 161),
        )
        self._legend = image.crop(self._legend_bounds)
        return image

    def _make_palette(self, background: Image.Image) -> Image.Image:
        """Reserve state colors before adding background colors."""
        colors = [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 128, 0),
            LASER_COLOR,
            ROBOT_BELIEF_COLOR,
            OPPONENT_BELIEF_COLOR,
            (211, 47, 47),
            (183, 28, 28),
            (25, 118, 210),
            (13, 71, 161),
            (255, 205, 210),
            (144, 202, 249),
            (69, 90, 100),
            (38, 50, 56),
            (144, 164, 174),
            (255, 143, 0),
            (255, 82, 82),
            (245, 222, 179),
            (173, 216, 230),
            (255, 160, 160),
        ]
        palette = Image.new("P", (1, 1))
        adaptive = background.quantize(colors=256 - len(colors))
        values = [channel for color in colors for channel in color]
        values += (adaptive.getpalette() or [])[: 768 - len(values)]
        palette.putpalette(values + [0] * (768 - len(values)))
        return palette

    def _draw_belief(self, draw, belief) -> None:
        if belief is None or not hasattr(belief, "to_unique_support_distribution"):
            return
        markers = []
        try:
            distribution = belief.to_unique_support_distribution()
            for index, state in enumerate(distribution.values):
                if isinstance(state, np.ndarray) and len(state) == 5:
                    radius = self._belief_radius(distribution, index)
                    if not np.isfinite(radius) or radius <= 0:
                        continue
                    for point, color in (
                        (state[2:4], OPPONENT_BELIEF_COLOR),
                        (state[:2], ROBOT_BELIEF_COLOR),
                    ):
                        x, y = self._world_to_pixel(point)
                        if np.isfinite(x) and np.isfinite(y):
                            markers.append(
                                ((x - radius, y - radius, x + radius, y + radius), color)
                            )
        except Exception:  # Malformed optional beliefs do not prevent an episode GIF.
            return
        for bounds, color in markers:
            draw.ellipse(bounds, fill=color)

    def _belief_radius(self, distribution, index) -> float:
        """Continuous beliefs use equally sized support markers."""
        return 4.0

    def _laser_segments(
        self, rp: np.ndarray, op: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Each variant supplies its own ray geometry."""
        raise NotImplementedError

    def _action_info(self, action) -> Tuple:
        """Each variant supplies its action label, tag flag, and direction."""
        raise NotImplementedError

    def _is_successful_tag(self, rp: np.ndarray, op: np.ndarray) -> bool:
        """Each variant supplies its tag rule."""
        raise NotImplementedError

    def _render_frame(
        self,
        background,
        rp,
        op,
        action,
        belief,
        index,
        count,
        robot_path=None,
        opponent_path=None,
    ) -> Image.Image:
        image = background.copy()
        draw = ImageDraw.Draw(image)
        for start_point, end_point in self._laser_segments(rp, op):
            draw.line(
                (self._world_to_pixel(start_point), self._world_to_pixel(end_point)),
                fill=LASER_COLOR,
                width=1,
            )
        self._draw_belief(draw, belief)
        robot_path = [rp] if robot_path is None else robot_path
        opponent_path = [op] if opponent_path is None else opponent_path
        if len(robot_path) > 1:
            draw.line(
                [self._world_to_pixel(point) for point in robot_path], fill=(255, 160, 160), width=3
            )
        if len(opponent_path) > 1:
            draw.line(
                [self._world_to_pixel(point) for point in opponent_path],
                fill=(144, 202, 249),
                width=3,
            )
        text, is_tag, direction = self._action_info(action)
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1e-12:
            start = np.asarray(self._world_to_pixel(rp))
            end = np.asarray(self._world_to_pixel(rp + direction / magnitude * 0.6))
            unit = (end - start) / np.linalg.norm(end - start)
            normal = np.array([-unit[1], unit[0]])
            draw.line((tuple(start), tuple(end)), fill="red", width=3)
            draw.polygon(
                [
                    tuple(end),
                    tuple(end - 12 * unit + 5 * normal),
                    tuple(end - 12 * unit - 5 * normal),
                ],
                fill="red",
            )
        for point, sprite in ((rp, self._robot_sprite), (op, self._opponent_sprite)):
            x, y = self._world_to_pixel(point)
            image.paste(sprite, (round(x - sprite.width / 2), round(y - sprite.height / 2)), sprite)
        # Clip world marks to the half-unit margin; leave axis labels outside it.
        bounds = (
            round(self._left) + 1,
            round(self._top) + 1,
            round(self._right),
            round(self._bottom),
        )
        clipped = background.copy()
        clipped.paste(image.crop(bounds), bounds[:2])
        draw = ImageDraw.Draw(clipped)
        x, y = self._left + 18, self._top + 14
        draw.rounded_rectangle((x, y, x + 225, y + 28), radius=4, fill="wheat")
        draw.text((x + 7, y + 5), f"Step: {index + 1}/{count}", fill="black", font=self._font)
        if text:
            draw.rounded_rectangle((x, y + 48, x + 280, y + 76), radius=4, fill="lightblue")
            draw.text((x + 7, y + 53), text, fill="black", font=self._font)
        if is_tag:
            tagged = self._is_successful_tag(rp, op)
            draw.text(
                (x, self._bottom - 48),
                "TAGGED!" if tagged else "MISSED!",
                fill="green" if tagged else "red",
                font=self._tag_font,
            )
        clipped.paste(self._legend, self._legend_bounds[:2])
        return clipped

    def _draw_legend(self, draw) -> None:
        x, y = self._right - 155, self._top + 14
        draw.rounded_rectangle((x, y, x + 140, y + 146), radius=5, fill="white", outline="#aaaaaa")
        for offset, (label, color) in enumerate(
            (
                ("Robot", "#D32F2F"),
                ("Opponent", "#1976D2"),
                ("Robot belief", ROBOT_BELIEF_COLOR),
                ("Opp. belief", OPPONENT_BELIEF_COLOR),
                ("Laser", LASER_COLOR),
                ("Action", "red"),
            )
        ):
            row = y + 10 + offset * 22
            draw.rectangle((x + 10, row + 3, x + 20, row + 13), fill=color)
            draw.text((x + 28, row), label, fill="black", font=self._font)
