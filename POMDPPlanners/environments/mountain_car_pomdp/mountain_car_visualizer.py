# SPDX-License-Identifier: MIT
"""Cached MountainCar artwork; scenery never defines the state geometry."""

from functools import lru_cache
from math import atan2, cos, degrees, sin
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from .mountain_car_pomdp import MountainCarPOMDP

ASSETS = Path(__file__).with_name("assets")


@lru_cache(maxsize=16)
def _font(size: int):
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=4)
def _asset(name: str) -> Image.Image:
    with Image.open(ASSETS / name) as source:
        return source.convert("RGBA")


@lru_cache(maxsize=256)
def _car(angle: float) -> Image.Image:
    art = _asset("car.png").copy()
    art.thumbnail((64, 40), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (112, 112))
    canvas.alpha_composite(art, ((112 - art.width) // 2, 56 - art.height))
    return canvas.rotate(angle, resample=Image.Resampling.BICUBIC)


class MountainCarVisualizer:
    """One 500 ms frame per recorded row, including terminal bookkeeping."""

    def __init__(self, environment: "MountainCarPOMDP", width: int = 800, height: int = 500):
        if not isinstance(width, int) or not isinstance(height, int) or width < 640 or height < 400:
            raise ValueError("Canvas must be at least 640 by 400 pixels")
        self.env = environment
        self.width, self.height = width, height
        self._background: Image.Image | None = None
        self._key: tuple | None = None
        self._palette: Image.Image | None = None

    @staticmethod
    def hill_height(position: float) -> float:
        """Conventional MountainCar hill matching the native cos(3*x) force."""
        return 0.45 * sin(3 * position) + 0.55

    def point(self, position: float) -> tuple[float, float]:
        x = 45 + (position - self.env.min_position) / (
            self.env.max_position - self.env.min_position
        ) * (self.width - 90)
        return x, self.height - 84 - self.hill_height(position) * (self.height - 275)

    def slope_angle(self, position: float) -> float:
        # Screen axes have different units; use the displayed hill tangent.
        return degrees(
            atan2(
                1.35 * cos(3 * position) * (self.height - 275),
                (self.width - 90) / (self.env.max_position - self.env.min_position),
            )
        )

    def _build_background(self) -> Image.Image:
        width, height = self.width, self.height
        canvas = _asset("vista.jpg").resize((width, height), Image.Resampling.LANCZOS)
        points = [
            self.point(float(x))
            for x in np.linspace(self.env.min_position, self.env.max_position, width * 2)
        ]
        polygon = [(0, points[0][1]), *points, (width, points[-1][1]), (width, height), (0, height)]
        mask = Image.new("L", (width, height))
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        rock = _asset("rock.jpg").resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(rock, (0, 0), mask)
        draw = ImageDraw.Draw(canvas)
        # Soil and grass are below the exact top edge; the track never shifts.
        for offset, color, line_width in (
            (8, (55, 44, 22), 13),
            (4, (91, 95, 38), 9),
            (2, (137, 140, 61), 5),
            (0, (207, 186, 119), 2),
        ):
            draw.line([(x, y + offset) for x, y in points], fill=color, width=line_width)
        rng = np.random.default_rng(5031)
        for position in rng.uniform(self.env.min_position, self.env.max_position, 160):
            x, y = self.point(float(position))
            blade = float(rng.uniform(2, 6))
            draw.line(
                (x, y + 2, x + float(rng.uniform(-2, 2)), y - blade), fill=(107, 119, 50), width=1
            )
        gx, gy = self.point(self.env.goal_position)
        draw.line((gx, gy, gx, gy - 42), fill=(48, 65, 28), width=3)
        draw.line((gx - 1, gy, gx - 1, gy - 42), fill=(184, 188, 106), width=1)
        draw.polygon(
            [(gx + 1, gy - 42), (gx + 24, gy - 37), (gx + 17, gy - 28), (gx + 1, gy - 30)],
            fill=(70, 160, 41),
            outline=(35, 91, 28),
        )
        draw.text(
            (gx - 35, gy - 60),
            f"Goal {self.env.goal_position:g}",
            font=_font(13),
            fill=(21, 49, 27),
            stroke_width=1,
            stroke_fill=(225, 223, 168),
        )
        for x in np.linspace(self.env.min_position, self.env.max_position, 10):
            px, _ = self.point(float(x))
            draw.line((px, height - 35, px, height - 29), fill=(226, 206, 163))
            draw.text(
                (px, height - 23), f"{x:.1f}", font=_font(12), anchor="mm", fill=(250, 230, 188)
            )
        draw.line((45, height - 35, width - 45, height - 35), fill=(176, 153, 110))
        # Dark cards keep recorded data separate from the illustrative scene.
        draw.rounded_rectangle(
            (12, 10, width - 12, 90),
            radius=10,
            fill=(18, 33, 43, 245),
            outline=(101, 132, 145),
            width=1,
        )
        draw.rectangle((12, height - 69, 390, height - 43), fill=(29, 35, 31, 230))
        return canvas

    def _prepare(self):
        key = (
            self.env.min_position,
            self.env.max_position,
            self.env.goal_position,
            self.width,
            self.height,
        )
        if key != self._key:
            if self.env.min_position >= self.env.max_position:
                raise ValueError("Position bounds must increase")
            self._background = self._build_background()
            self._key = key
            self._palette = None

    def _text(self, draw: ImageDraw.ImageDraw, y: int, text: str, size: int, color: tuple) -> None:
        while size > 8 and draw.textlength(text, font=_font(size)) > self.width - 48:
            size -= 1
        draw.text((24, y), text, font=_font(size), fill=color)

    def render_frame(self, step: StepData, index: int, total: int) -> Image.Image:
        self._prepare()
        assert self._background is not None
        state = np.asarray(step.state, dtype=float)
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError("State must contain finite position and velocity")
        position, velocity = map(float, state)
        canvas = self._background.copy()
        draw = ImageDraw.Draw(canvas)
        self._draw_belief(canvas, step.belief)
        px, py = self.point(position)
        canvas.alpha_composite(_car(self.slope_angle(position)), (round(px) - 56, round(py) - 56))
        draw = ImageDraw.Draw(canvas)
        terminal = position >= self.env.goal_position
        action = step.action
        if action is not None and action not in (-1, 0, 1):
            raise ValueError("MountainCar action must be -1, 0, 1, or None")
        action_label = {-1: "-1 LEFT", 0: "0 NEUTRAL", 1: "+1 RIGHT", None: "none"}[action]
        suffix = (
            "  GOAL REACHED" if terminal else ("  END OF RECORDING" if index == total - 1 else "")
        )
        self._text(draw, 18, f"MOUNTAIN CAR   {index+1}/{total}{suffix}", 18, (250, 236, 189))
        self._text(
            draw,
            44,
            f"True pre-action: x {position:+.3f}   velocity {velocity:+.4f}   |   Selected action {action_label}",
            13,
            (213, 231, 239),
        )
        result = (
            "Recorded result: none (bookkeeping row)"
            if action is None
            else f"Recorded result: reward {step.reward}   observation {self._observation(step.observation)}"
        )
        self._text(draw, 67, result, 12, (182, 207, 216))
        return canvas.convert("RGB")

    def _draw_belief(self, canvas: Image.Image, belief: Any) -> None:
        overlay = Image.new("RGBA", canvas.size)
        draw = ImageDraw.Draw(overlay)
        label = "Belief position: unavailable"
        particles = getattr(belief, "particles", None)
        if particles is not None and len(particles):
            positions = np.asarray(particles, dtype=float)[:, 0]
            weights = np.ones(len(positions), dtype=float)
            log_weights = getattr(belief, "log_weights", None)
            if log_weights is not None:
                logs = np.asarray(log_weights, dtype=float)
                if logs.shape != weights.shape or np.isnan(logs).any() or np.isposinf(logs).any():
                    raise ValueError("Invalid particle log weights")
                if np.isfinite(logs).any():
                    weights = np.exp(logs - np.max(logs))
            weights /= weights.sum()
            # Sum mass in fixed position bins so duplicate particles retain their weight.
            mass, edges = np.histogram(
                positions,
                bins=100,
                range=(self.env.min_position, self.env.max_position),
                weights=weights,
            )
            for value, probability in zip((edges[:-1] + edges[1:]) / 2, mass):
                if probability > 0:
                    px, py = self.point(float(value))
                    alpha = round(255 * float(probability / mass.max()))
                    draw.ellipse((px - 3, py + 15, px + 3, py + 21), fill=(235, 235, 220, alpha))
            label = "Gray dots: position mass (brighter = more)"
        elif hasattr(belief, "mean") and hasattr(belief, "covariance"):
            mean = float(np.asarray(belief.mean)[0])
            variance = float(np.asarray(belief.covariance)[0, 0])
            if not np.isfinite([mean, variance]).all() or variance < 0:
                raise ValueError("Invalid Gaussian position belief")
            low = max(self.env.min_position, mean - 2 * variance**0.5)
            high = min(self.env.max_position, mean + 2 * variance**0.5)
            if low <= high:
                points = [
                    (x, y + 18)
                    for x, y in (self.point(float(v)) for v in np.linspace(low, high, 100))
                ]
                draw.line(points, fill=(235, 235, 220, 180), width=4)
            if self.env.min_position <= mean <= self.env.max_position:
                px, py = self.point(mean)
                draw.ellipse((px - 4, py + 14, px + 4, py + 22), fill=(255, 249, 224, 255))
            label = "Gray: belief mean and +/-2 SD (clipped)"
        draw.text((22, self.height - 62), label, font=_font(12), fill=(230, 224, 203))
        canvas.alpha_composite(overlay)

    @staticmethod
    def _observation(observation: Any) -> str:
        if observation is None:
            return "none"
        values = np.asarray(observation).ravel()
        return "[" + ", ".join(f"{float(v):+.3f}" for v in values[:2]) + "] (noisy next state)"

    def save(self, history: List[StepData], path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        if not history:
            raise ValueError("Cannot render empty history")
        if any(not isinstance(step, StepData) for step in history):
            raise TypeError("history must contain StepData")
        if path.suffix.lower() != ".gif":
            raise ValueError("path must end with .gif")
        self._prepare()
        if self._palette is None:
            assert self._background is not None
            self._palette = self._background.convert("RGB").quantize(
                colors=224, method=Image.Quantize.MEDIANCUT
            )
            colors = (self._palette.getpalette() or [])[:672]
            art = _asset("car.png").convert("RGB").quantize(colors=24)
            colors += (art.getpalette() or [])[:72]
            colors += [
                250,
                236,
                189,
                213,
                231,
                239,
                182,
                207,
                216,
                235,
                235,
                220,
                255,
                249,
                224,
                18,
                33,
                43,
                70,
                160,
                41,
                255,
                255,
                255,
            ]
            self._palette.putpalette(colors)
        indexed = [
            self.render_frame(step, i, len(history)).quantize(
                palette=self._palette, dither=Image.Dither.NONE
            )
            for i, step in enumerate(history)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        indexed[0].save(
            path,
            save_all=True,
            append_images=indexed[1:],
            duration=500,
            loop=0,
            optimize=False,
            disposal=1,
        )
