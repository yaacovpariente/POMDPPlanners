# SPDX-License-Identifier: MIT

"""Deterministic CartPole episode replay; no transitions or belief sampling."""

from functools import lru_cache
from importlib.resources import files
import math
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData

CANVAS_SIZE = (800, 500)
FRAME_DURATION_MS = 500
PIVOT_Y = 323.0
RAIL_Y = 365
INK = (36, 43, 50)
BLUE = (27, 83, 164)
RED = (191, 45, 32)


@lru_cache(maxsize=8)
def _font(size: int):
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=16)
def _asset(name: str, size: Tuple[int, int]) -> Image.Image:
    resource = files("POMDPPlanners.environments.cartpole_pomdp")
    with resource.joinpath("visualization_assets").joinpath(name + ".png").open("rb") as stream:
        with Image.open(stream) as source:
            image = source.convert("RGBA" if name == "cart" else "RGB")
    return image.resize(size, Image.Resampling.LANCZOS)


def _state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("CartPole state must contain four finite values")
    return state


def pole_geometry(state: Any, scale: float, half_length: float):
    """Pixel pivot and tip; positive radians lean right, zero is upright."""
    x, _, angle, _ = _state(state)
    pivot = (CANVAS_SIZE[0] / 2 + float(x) * scale, PIVOT_Y)
    length_px = 2 * half_length * scale
    tip = (pivot[0] + length_px * math.sin(angle), pivot[1] - length_px * math.cos(angle))
    return pivot, tip


def _arrow(draw: ImageDraw.ImageDraw, x: float, y: float, direction: int) -> None:
    end = x + direction * 80
    start = x + direction * 42
    draw.line((start, y, end, y), fill=BLUE, width=4)
    draw.polygon(
        [(end, y), (end - direction * 10, y - 6), (end - direction * 10, y + 6)], fill=BLUE
    )


@lru_cache(maxsize=8)
def _background(scale: float, x_threshold: float, theta_threshold: float) -> Image.Image:
    """Shared immutable background, copied before adding recorded data."""
    canvas = _asset("lab", CANVAS_SIZE).copy()
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (14, 12, 786, 100), radius=12, fill=(235, 237, 234), outline=(177, 184, 182)
    )
    draw.text((28, 21), "CartPole", font=_font(25), fill=INK)
    draw.text(
        (28, 75),
        f"Limits: |x| <= {x_threshold:g} m   |angle| <= {math.degrees(theta_threshold):g} deg",
        font=_font(13),
        fill=INK,
    )
    # The rail is drawn here, never baked into generated art.
    for offset, color in enumerate(
        (
            (84, 78, 69),
            (219, 215, 203),
            (73, 79, 82),
            (31, 36, 41),
            (133, 139, 141),
            (232, 233, 229),
        )
    ):
        draw.line((18, RAIL_Y + offset, 782, RAIL_Y + offset), fill=color, width=1)
    draw.rectangle((18, RAIL_Y + 6, 782, 403), fill=(222, 224, 218))
    extent = 365 / scale
    for tick in range(math.ceil(-extent), math.floor(extent) + 1):
        px = 400 + tick * scale
        if px < 60 or px > 730:
            continue
        draw.line((px, RAIL_Y + 8, px, RAIL_Y + 14), fill=INK)
        draw.text((px, RAIL_Y + 16), str(tick), font=_font(12), fill=INK, anchor="mt")
    for limit in (-x_threshold, x_threshold):
        px = 400 + limit * scale
        draw.line((px, RAIL_Y - 8, px, RAIL_Y + 10), fill=RED, width=2)
    draw.text((756, RAIL_Y + 16), "x (m)", font=_font(12), fill=INK, anchor="mt")
    draw.rounded_rectangle(
        (14, 405, 786, 492), radius=10, fill=(235, 237, 234), outline=(177, 184, 182)
    )
    return canvas


def _belief_points(belief: Any):
    """Read stored particles only; unsupported beliefs are not sampled."""
    raw = getattr(belief, "particles", None)
    if raw is None:
        return np.empty((0, 4))
    particles = np.asarray(raw, dtype=float)
    if particles.ndim != 2 or particles.shape[1] != 4:
        return np.empty((0, 4))
    return particles[np.isfinite(particles).all(axis=1)]


class CartPoleVisualizer:
    """Render recorded states using the environment's physical parameters."""

    def __init__(self, environment: Any):
        self.environment = environment

    def render_frames(self, history: List[StepData]) -> List[Image.Image]:
        if not isinstance(history, list) or not all(isinstance(row, StepData) for row in history):
            raise TypeError("history must be a list of StepData")
        if not history:
            raise ValueError("Cannot visualize an empty history")
        states = [_state(row.state) for row in history]
        half_length = float(self.environment.length)
        x_threshold = float(self.environment.x_threshold)
        angle_threshold = float(self.environment.theta_threshold_radians)
        if not all(math.isfinite(v) and v > 0 for v in (half_length, x_threshold, angle_threshold)):
            raise ValueError("CartPole length and thresholds must be finite and positive")
        span = max(x_threshold + 0.6, max(abs(float(s[0])) for s in states) + 0.6)
        scale = min(365 / span, 180 / (2 * half_length))
        background = _background(scale, x_threshold, angle_threshold)
        cart_size = (max(16, round(0.55 * scale)), RAIL_Y - round(PIVOT_Y))
        cart = _asset("cart", cart_size)
        frames = []
        for index, (row, state) in enumerate(zip(history, states)):
            action = row.action
            if action is not None and (not np.isscalar(action) or action not in (0, 1)):
                raise ValueError("CartPole action must be 0, 1 or None")
            canvas = background.copy()
            draw = ImageDraw.Draw(canvas)
            phase = "no-action bookkeeping" if action is None else "before action"
            draw.text((28, 51), f"Recorded true state | {phase}", font=_font(13), fill=INK)
            x, velocity, angle, angular_velocity = state
            draw.text(
                (772, 24), f"Step {index + 1}/{len(history)}", font=_font(18), fill=INK, anchor="rt"
            )
            draw.text(
                (772, 49),
                f"x {x:+.3f} m   v {velocity:+.3f} m/s",
                font=_font(15),
                fill=INK,
                anchor="rt",
            )
            draw.text(
                (772, 72),
                f"angle {angle:+.3f} rad   ang.vel. {angular_velocity:+.3f} rad/s",
                font=_font(15),
                fill=INK,
                anchor="rt",
            )
            pivot, tip = pole_geometry(state, scale, half_length)
            # Dashed upright and threshold guides use the same physical scale.
            pole_px = 2 * half_length * scale
            for yy in np.arange(pivot[1] - pole_px, pivot[1], 8):
                yy = float(yy)
                draw.line((pivot[0], yy, pivot[0], min(yy + 3, pivot[1])), fill=(125, 134, 137))
            for limit in (-angle_threshold, angle_threshold):
                gx = pivot[0] + pole_px * math.sin(limit)
                gy = pivot[1] - pole_px * math.cos(limit)
                draw.line((pivot[0], pivot[1], gx, gy), fill=(183, 163, 126), width=1)
            # Layered highlights shade a cylinder without altering its endpoints.
            draw.line((*pivot, *tip), fill=(96, 29, 25), width=9)
            draw.line((*pivot, *tip), fill=RED, width=7)
            draw.line((pivot[0] - 1, pivot[1], tip[0] - 1, tip[1]), fill=(239, 103, 71), width=2)
            draw.ellipse((tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3), fill=(224, 78, 53))
            canvas.paste(cart, (round(pivot[0] - cart.width / 2), round(PIVOT_Y)), cart)
            draw.ellipse(
                (pivot[0] - 6, pivot[1] - 6, pivot[0] + 6, pivot[1] + 6),
                fill=(71, 77, 80),
                outline=(203, 207, 203),
                width=2,
            )
            draw.ellipse(
                (pivot[0] - 2, pivot[1] - 2, pivot[0] + 2, pivot[1] + 2), fill=(221, 224, 219)
            )
            if action is not None:
                _arrow(draw, pivot[0], PIVOT_Y - 20, -1 if action == 0 else 1)
            particles = _belief_points(row.belief)
            # Dots show stored support, not an unweighted probability estimate.
            for particle in particles[:: max(1, math.ceil(len(particles) / 500))]:
                bx = 400 + float(particle[0]) * scale
                if 20 <= bx <= 780:
                    draw.ellipse((bx - 2, 394, bx + 2, 398), fill=(88, 113, 133))
            draw.rounded_rectangle((631, 106, 779, 129), radius=5, fill=(235, 237, 234))
            draw.text(
                (28, 391),
                f"Belief x support ({len(particles)} particles)"
                if len(particles)
                else "Belief x support unavailable",
                font=_font(11),
                fill=INK,
            )
            outside = abs(x) > x_threshold or abs(angle) > angle_threshold
            status = "OUTSIDE LIMITS" if outside else "WITHIN LIMITS"
            draw.text(
                (772, 111),
                status,
                font=_font(13),
                fill=RED if outside else (45, 106, 69),
                anchor="rt",
            )
            direction = (
                "none (bookkeeping)"
                if action is None
                else f"{int(action)} | {'LEFT' if action == 0 else 'RIGHT'} force {float(self.environment.force_mag):g} N"
            )
            draw.text((28, 414), f"Action: {direction}", font=_font(14), fill=BLUE)
            reward = "--" if row.reward is None else f"{float(row.reward):+.4g}"
            draw.text(
                (772, 414), f"Recorded reward: {reward}", font=_font(14), fill=INK, anchor="rt"
            )
            observation = (
                "not recorded"
                if row.observation is None
                else np.array2string(
                    _state(row.observation),
                    precision=3,
                    separator=", ",
                    suppress_small=True,
                    max_line_width=1000,
                )
            )
            draw.text(
                (28, 440),
                "Recorded outcome after action (noisy observation, not true state):",
                font=_font(12),
                fill=INK,
            )
            draw.text(
                (28, 460), f"[x, v, angle, ang.vel.] = {observation}", font=_font(13), fill=INK
            )
            frames.append(canvas)
        return frames

    def save(self, history: List[StepData], path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        if path.suffix.lower() != ".gif":
            raise ValueError("path must end in .gif")
        frames = self.render_frames(history)
        # Analyse a small reference; all saved frames retain their full size.
        reference = frames[0].resize((400, 250), Image.Resampling.NEAREST)
        palette = reference.quantize(colors=248, method=Image.Quantize.MEDIANCUT)
        entries = (palette.getpalette() or [])[:744]
        entries.extend(
            c
            for color in (
                INK,
                BLUE,
                RED,
                (239, 103, 71),
                (88, 113, 133),
                (235, 237, 234),
                (45, 106, 69),
                (221, 224, 219),
            )
            for c in color
        )
        entries.extend([0] * (768 - len(entries)))
        palette.putpalette(entries)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        indexed[0].save(
            path,
            save_all=True,
            append_images=indexed[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            optimize=True,
        )
