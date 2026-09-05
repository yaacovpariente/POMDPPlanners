# SPDX-License-Identifier: MIT

"""Pillow visualization for Safety Ant Velocity episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_assets import (
    ant_sprite,
    concrete_texture,
)


class SafeAntVelocityVisualizer:
    """Create a two-panel GIF of an ant trajectory and its speed."""

    WIDTH = 1600
    HEIGHT = 800
    FRAME_DURATION_MS = 1250

    WHITE = (255, 255, 255)
    BLACK = (218, 226, 232)
    GRID = (40, 52, 60)
    BLUE = (48, 156, 255)
    GREEN = (98, 225, 157)
    ORANGE = (245, 158, 11)
    RED = (220, 38, 38)
    PURPLE = (188, 72, 220)

    MAIN_BOX = (70, 110, 770, 710)
    SPEED_BOX = (950, 385, 1530, 690)

    def __init__(self, env: Any):
        self.env = env
        self.safe_velocity_threshold = env.safe_velocity_threshold
        self.max_force = env.max_force
        self._fonts = {
            "small": self._load_font(16),
            "body": self._load_font(19),
            "title": self._load_font(34),
            "status": self._load_font(22),
        }

    def create_animation(self, history: List[StepData], cache_path: Path) -> None:
        """Save one GIF frame for every recorded state, including the terminal state."""

        self._validate_visualization_inputs(history, cache_path)
        states, actions, rewards = self._extract_episode_data(history)
        geometry = self._geometry(states)
        background = self._build_static_background(states, geometry)
        palette = background.quantize(colors=192)
        colors = list(palette.getpalette() or [])[:576]
        reserved = [
            self.BLUE,
            self.GREEN,
            self.PURPLE,
            self.ORANGE,
            self.RED,
            self.BLACK,
            self.WHITE,
        ]
        reserved += [
            (round(18 * f), round(97 * f), round(195 * f)) for f in np.linspace(0.15, 1, 32)
        ]
        reserved += [
            (round(18 + 150 * f), round(97 + 130 * f), round(195 + 60 * f))
            for f in np.linspace(0, 1, 16)
        ]
        colors += [channel for color in reserved for channel in color]
        palette.putpalette((colors + [0] * 768)[:768])
        frames = [
            self._render_frame(background, states, actions, rewards, geometry, frame).quantize(
                palette=palette, dither=Image.Dither.NONE
            )
            for frame in range(len(states))
        ]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            cache_path,
            save_all=True,
            append_images=frames[1:],
            duration=self.FRAME_DURATION_MS,
            loop=0,
            disposal=2,
            optimize=False,
        )

    def _validate_visualization_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _extract_episode_data(
        self, history: List[StepData]
    ) -> tuple[list[np.ndarray], list[Any], list[float]]:
        states = [np.asarray(step.state, dtype=float) for step in history]
        actions = [step.action for step in history[:-1]]
        rewards = [0.0 if step.reward is None else float(step.reward) for step in history]
        return states, actions, rewards

    def _geometry(self, states: Sequence[np.ndarray]) -> dict[str, float]:
        positions = np.asarray([state[:2] for state in states])
        max_speed = max(float(np.linalg.norm(state[2:4])) for state in states)
        return {
            "x_min": float(positions[:, 0].min() - 1.0),
            "x_max": float(positions[:, 0].max() + 1.0),
            "y_min": float(positions[:, 1].min() - 1.0),
            "y_max": float(positions[:, 1].max() + 1.0),
            "speed_max": max(max_speed * 1.1, self.safe_velocity_threshold * 2.0, 1.0),
        }

    def _build_static_background(
        self, states: Sequence[np.ndarray], geometry: dict[str, float]
    ) -> Image.Image:
        image = Image.new("RGB", (self.WIDTH, self.HEIGHT), (12, 19, 25))
        image.paste(concrete_texture(840, 800), (0, 0))
        draw = ImageDraw.Draw(image)
        for offset in range(15):
            draw.line(
                (840 + offset, 0, 840 + offset, 800),
                fill=(8 + offset // 2, 13 + offset // 2, 17 + offset // 2),
            )
        draw.rounded_rectangle((24, 22, 814, 777), radius=12, outline=(97, 103, 101), width=2)
        draw.line((30, 24, 807, 24), fill=(229, 224, 207), width=2)
        main = self._main_box(geometry)
        speed = self.SPEED_BOX
        draw.text((49, 46), "FIELD / TRAJECTORY", fill=(55, 65, 64), font=self._fonts["body"])
        draw.text((885, 35), "Safe Ant Velocity", fill=self.WHITE, font=self._fonts["title"])
        draw.text(
            (888, 82),
            "MOTION CONTROL  /  EPISODE TELEMETRY",
            fill=(127, 148, 160),
            font=self._fonts["small"],
        )
        draw.rounded_rectangle(
            (883, 118, 1560, 327), radius=14, fill=(19, 29, 37), outline=(53, 68, 77), width=1
        )
        draw.line((902, 120, 1541, 120), fill=(82, 102, 113))
        draw.rounded_rectangle(
            (883, 345, 1560, 754), radius=14, fill=(15, 24, 31), outline=(49, 64, 74)
        )
        draw.text((914, 355), "SPEED OVER TIME", fill=self.WHITE, font=self._fonts["body"])
        self._draw_grid(draw, speed)
        draw.line(
            (speed[0], speed[1], speed[0], speed[3], speed[2], speed[3]),
            fill=(117, 137, 148),
            width=1,
        )
        self._draw_speed_axis_labels(draw, speed, len(states), geometry["speed_max"])
        draw.text((1210, 727), "Time Step", fill=(142, 164, 177), font=self._fonts["small"])
        self._vertical_text(image, (887, 535), "Speed (Velocity Magnitude)", self._fonts["small"])
        for value, color in (
            (self.safe_velocity_threshold, self.ORANGE),
            (self.safe_velocity_threshold * 1.5, self.RED),
        ):
            y = self._speed_y(value, geometry)
            self._dashed_line(draw, (speed[0], y), (speed[2], y), color, 2)
        draw.text(
            (1185, 354),
            f"Safe {self.safe_velocity_threshold:.1f}",
            fill=self.ORANGE,
            font=self._fonts["small"],
        )
        draw.text(
            (1350, 354),
            f"Critical {self.safe_velocity_threshold * 1.5:.1f}",
            fill=self.RED,
            font=self._fonts["small"],
        )
        for index in range(5):
            f = index / 4
            x = round(main[0] + f * (main[2] - main[0]))
            y = round(main[3] - f * (main[3] - main[1]))
            draw.line((x, main[3], x, main[3] + 6), fill=(63, 71, 72))
            draw.text(
                (x - 12, main[3] + 11),
                f"{geometry['x_min'] + f * (geometry['x_max'] - geometry['x_min']):.1f}",
                fill=(51, 61, 63),
                font=self._fonts["small"],
            )
            draw.text(
                (main[0] - 42, y - 8),
                f"{geometry['y_min'] + f * (geometry['y_max'] - geometry['y_min']):.1f}",
                fill=(51, 61, 63),
                font=self._fonts["small"],
            )
        draw.text(
            ((main[0] + main[2]) // 2, main[3] + 38),
            "X Position",
            fill=(48, 58, 60),
            font=self._fonts["small"],
            anchor="mt",
        )
        self._vertical_text(
            image,
            (max(6, main[0] - 64), (main[1] + main[3]) // 2),
            "Y Position",
            self._fonts["small"],
            color=(48, 58, 60),
        )
        draw.text((885, 769), "VELOCITY", fill=self.GREEN, font=self._fonts["small"])
        draw.text((1020, 769), "APPLIED FORCE", fill=self.PURPLE, font=self._fonts["small"])
        draw.text((1210, 769), "TRAJECTORY", fill=self.BLUE, font=self._fonts["small"])
        return image

    def _render_frame(
        self,
        background: Image.Image,
        states: Sequence[np.ndarray],
        actions: Sequence[Any],
        rewards: Sequence[float],
        geometry: dict[str, float],
        frame: int,
    ) -> Image.Image:
        image = background.copy()
        scene = background.copy()
        draw = ImageDraw.Draw(scene)
        state = states[frame]
        position = state[:2]
        velocity = state[2:4]
        speed = float(np.linalg.norm(velocity))
        point = self._world_point(position, geometry)

        path = [self._world_point(candidate[:2], geometry) for candidate in states[: frame + 1]]
        if len(path) > 1:
            draw.line(
                [(x + 1, y + 2) for x, y in path], fill=(100, 113, 125), width=5, joint="curve"
            )
            draw.line(path, fill=self.BLUE, width=3, joint="curve")

        radius = (
            max(0.1, speed / self.safe_velocity_threshold)
            if self.safe_velocity_threshold > 0
            else 0.1
        )
        self._world_circle(draw, point, radius, geometry, self.ORANGE, dashed=True)
        if speed > self.safe_velocity_threshold:
            critical_radius = (
                max(0.1, speed / (self.safe_velocity_threshold * 1.5) * 1.5)
                if self.safe_velocity_threshold > 0
                else 0.1
            )
            self._world_circle(draw, point, critical_radius, geometry, self.RED)

        if speed > 0.01:
            self._arrow(
                draw, point, self._world_point(position + velocity * 0.5, geometry), self.GREEN
            )
        if frame < len(actions):
            force_scale = (0.0, 0.33, 0.67, 1.0)[int(actions[frame])]
            if force_scale > 0:
                direction = np.array([1.0, 0.5]) / np.sqrt(1.25)
                target = self._world_point(
                    position + direction * force_scale * self.max_force * 0.8, geometry
                )
                self._arrow(draw, point, target, self.PURPLE)

        main = self._main_box(geometry)
        sprite_size = max(8, min(100, round(min(main[2] - main[0], main[3] - main[1]) * 0.35)))
        sprite = ant_sprite(sprite_size)
        scene.paste(sprite, (point[0] - sprite.width // 2, point[1] - sprite.height // 2), sprite)
        main = self._main_box(geometry)
        image.paste(scene.crop(main), main)
        draw = ImageDraw.Draw(image)

        speeds = [float(np.linalg.norm(candidate[2:4])) for candidate in states[: frame + 1]]
        speed_points = [
            self._speed_point(index, value, len(states), geometry)
            for index, value in enumerate(speeds)
        ]
        if len(speed_points) > 1:
            draw.line(speed_points, fill=self.BLUE, width=4)
        for speed_point, value in zip(speed_points, speeds):
            color = self._safety_color(value)
            draw.ellipse(
                (speed_point[0] - 6, speed_point[1] - 6, speed_point[0] + 6, speed_point[1] + 6),
                fill=color,
                outline=self.BLACK,
                width=2,
            )

        action_name = f"Force Level {actions[frame]}" if frame < len(actions) else "Terminal"
        self._text_box(
            draw,
            (907, 145),
            f"Step: {frame + 1}/{len(states)}\nAction: {action_name}",
            (19, 29, 37),
        )
        self._text_box(
            draw,
            (1138, 145),
            f"Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}]\nSpeed: {speed:.2f}",
            (19, 29, 37),
        )
        self._text_box(
            draw,
            (907, 223),
            f"Step Reward: {rewards[frame]:.1f}\nTotal Reward: {sum(rewards[: frame + 1]):.1f}",
            (19, 29, 37),
        )
        self._draw_status(draw, speed)
        return image

    def _draw_status(self, draw: ImageDraw.ImageDraw, speed: float) -> None:
        if speed > self.safe_velocity_threshold * 1.5:
            text, color = "CRITICAL / TERMINAL", self.RED
        elif speed > self.safe_velocity_threshold:
            text, color = "SAFETY VIOLATION", self.ORANGE
        else:
            text, color = "SAFE OPERATION", self.GREEN
        draw.rounded_rectangle(
            (1133, 236, 1539, 296),
            radius=10,
            fill=tuple(int(c * 0.13) for c in color),
            outline=color,
            width=2,
        )
        draw.ellipse((1153, 258, 1167, 272), fill=color)
        draw.text((1183, 255), text, fill=color, font=self._fonts["status"])

    def _draw_grid(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        for index in range(1, 8):
            x = left + (right - left) * index // 8
            y = top + (bottom - top) * index // 8
            draw.line((x, top, x, bottom), fill=self.GRID, width=1)
            draw.line((left, y, right, y), fill=self.GRID, width=1)

    def _draw_speed_axis_labels(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        state_count: int,
        speed_max: float,
    ) -> None:
        left, top, right, bottom = box
        for index in range(5):
            fraction = index / 4
            x = left + int(fraction * (right - left))
            y = bottom - int(fraction * (bottom - top))
            draw.text(
                (x - 8, bottom + 8),
                f"{state_count * fraction:.1f}",
                fill=self.BLACK,
                font=self._fonts["small"],
            )
            draw.text(
                (left - 28, y - 9),
                f"{speed_max * fraction:.1f}",
                fill=self.BLACK,
                font=self._fonts["small"],
            )

    def _world_point(
        self, position: Sequence[float] | np.ndarray, geometry: dict[str, float]
    ) -> tuple[int, int]:
        left, top, right, bottom = self._main_box(geometry)
        x = left + (float(position[0]) - geometry["x_min"]) / (
            geometry["x_max"] - geometry["x_min"]
        ) * (right - left)
        y = bottom - (float(position[1]) - geometry["y_min"]) / (
            geometry["y_max"] - geometry["y_min"]
        ) * (bottom - top)
        return round(x), round(y)

    def _speed_point(
        self, step: int, speed: float, state_count: int, geometry: dict[str, float]
    ) -> tuple[int, int]:
        left, _, right, _ = self.SPEED_BOX
        x = left + step / state_count * (right - left)
        return round(x), self._speed_y(speed, geometry)

    def _speed_y(self, speed: float, geometry: dict[str, float]) -> int:
        _, top, _, bottom = self.SPEED_BOX
        return round(bottom - speed / geometry["speed_max"] * (bottom - top))

    def _world_circle(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: float,
        geometry: dict[str, float],
        color: tuple[int, int, int],
        *,
        dashed: bool = False,
    ) -> None:
        left, top, right, bottom = self._main_box(geometry)
        x_radius = radius / (geometry["x_max"] - geometry["x_min"]) * (right - left)
        y_radius = radius / (geometry["y_max"] - geometry["y_min"]) * (bottom - top)
        box = (
            round(center[0] - x_radius),
            round(center[1] - y_radius),
            round(center[0] + x_radius),
            round(center[1] + y_radius),
        )
        if dashed:
            for start in range(0, 360, 24):
                draw.arc(box, start=start, end=start + 13, fill=color, width=3)
        else:
            draw.ellipse(box, outline=color, width=3)

    @staticmethod
    def _arrow(
        draw: ImageDraw.ImageDraw,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        draw.line((start, end), fill=color, width=5)
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(vector))
        if length == 0:
            return
        unit = vector / length
        normal = np.array([-unit[1], unit[0]])
        tip = np.asarray(end, dtype=float)
        draw.polygon(
            [tuple(tip), tuple(tip - unit * 15 + normal * 8), tuple(tip - unit * 15 - normal * 8)],
            fill=color,
        )

    @staticmethod
    def _dashed_line(
        draw: ImageDraw.ImageDraw,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        for x in range(start[0], end[0], 18):
            draw.line((x, start[1], min(x + 10, end[0]), end[1]), fill=color, width=width)

    def _text_box(
        self,
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        bounds = draw.multiline_textbbox(position, text, font=self._fonts["body"], spacing=3)
        draw.rounded_rectangle(
            (bounds[0] - 5, bounds[1] - 4, bounds[2] + 5, bounds[3] + 4),
            radius=4,
            fill=color,
            outline=(41, 57, 66),
            width=1,
        )
        draw.multiline_text(position, text, fill=self.BLACK, font=self._fonts["body"], spacing=3)

    @staticmethod
    def _vertical_text(
        image: Image.Image,
        center: tuple[int, int],
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        *,
        color: tuple[int, int, int] = BLACK,
    ) -> None:
        bounds = tuple(round(value) for value in font.getbbox(text))
        temporary = Image.new(
            "RGBA", (bounds[2] - bounds[0], bounds[3] - bounds[1]), (255, 255, 255, 0)
        )
        ImageDraw.Draw(temporary).text(
            (-bounds[0], -bounds[1]),
            text,
            fill=color,
            font=font,
        )
        rotated = temporary.rotate(90, expand=True)
        image.paste(rotated, (center[0], center[1] - rotated.height // 2), rotated)

    def _safety_color(self, speed: float) -> tuple[int, int, int]:
        if speed <= self.safe_velocity_threshold:
            return self.GREEN
        if speed <= self.safe_velocity_threshold * 1.5:
            return self.ORANGE
        return self.RED

    def _main_box(self, geometry: dict[str, float]) -> tuple[int, int, int, int]:
        left, top, right, bottom = self.MAIN_BOX
        width = right - left
        height = bottom - top
        x_span = geometry["x_max"] - geometry["x_min"]
        y_span = geometry["y_max"] - geometry["y_min"]
        scale = min(width / x_span, height / y_span)
        plot_width, plot_height = round(x_span * scale), round(y_span * scale)
        left += (width - plot_width) // 2
        top += (height - plot_height) // 2
        right, bottom = left + plot_width, top + plot_height
        return left, top, right, bottom

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow before 10.1 has no size argument.
            return ImageFont.load_default()
