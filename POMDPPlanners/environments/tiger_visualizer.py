# SPDX-License-Identifier: MIT
"""Cached Tiger replay art with explicit pre-action and recorded-outcome phases."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.belief import UnweightedParticleBelief, UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.simulation import StepData

CANVAS_SIZE = (800, 500)
FRAME_DURATION_MS = 500
TEXT = (246, 221, 171)
GOLD = (224, 176, 76)
RED = (241, 123, 107)
GREEN = (153, 214, 149)
PANEL = (24, 20, 17)
BAR_TRACK = (57, 47, 32)
DOOR_CENTERS = (226, 570)


@lru_cache(maxsize=1)
def _background() -> Image.Image:
    path = Path(__file__).with_name("tiger_visualization_assets") / "chamber.png"
    with Image.open(path) as source:
        image = source.convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 799, 65), fill=PANEL)
    draw.line((0, 65, 799, 65), fill=GOLD)
    draw.rectangle((0, 425, 799, 499), fill=PANEL)
    draw.line((0, 425, 799, 425), fill=GOLD)
    for center, label in zip(DOOR_CENTERS, ("LEFT", "RIGHT")):
        draw.text((center, 271), label, font=_font(21), fill=TEXT, anchor="mm")
    return image


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=1)
def _palette() -> Image.Image:
    colors = [TEXT, GOLD, RED, GREEN, PANEL, BAR_TRACK]
    image = _background().quantize(colors=256 - len(colors))
    values = [channel for color in colors for channel in color]
    values += (image.getpalette() or [])[: 768 - len(values)]
    palette = Image.new("P", (1, 1))
    palette.putpalette(values)
    return palette


def _belief_probabilities(belief: Any) -> tuple[float, float] | None:
    """Read weighted string support without sampling or inventing a prior."""
    if belief is None:
        return None
    if isinstance(belief, (UnweightedParticleBelief, UnweightedParticleBeliefStateUpdate)):
        values = list(belief.particles)
        weights = np.ones(len(values))
    elif callable(getattr(belief, "to_unique_support_distribution", None)):
        distribution = belief.to_unique_support_distribution()
        values = list(distribution.values)
        weights = np.asarray(distribution.probs, dtype=float)
    else:
        raise TypeError("Tiger visualization needs a discrete particle belief")
    if (
        weights.shape != (len(values),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
        or any(value not in ("tiger_left", "tiger_right") for value in values)
    ):
        raise ValueError("Tiger belief must have finite nonnegative mass on Tiger states")
    total = float(weights.sum())
    left = sum(float(weight) for value, weight in zip(values, weights) if value == "tiger_left")
    return left / total, (total - left) / total


@dataclass(frozen=True)
class TigerFrameData:
    """Text and belief values derived only from this recorded row."""

    hidden: str
    action: str
    observation: str
    reward: str
    probabilities: tuple[float, float] | None
    opened_side: int | None
    outcome: str


def _frame_data(step: StepData) -> TigerFrameData:
    if step.state not in ("tiger_left", "tiger_right"):
        raise ValueError("Unknown Tiger state")
    if step.action not in (None, "listen", "open_left", "open_right"):
        raise ValueError("Unknown Tiger action")
    if step.observation not in (None, "hear_left", "hear_right", "hear_nothing"):
        raise ValueError("Unknown Tiger observation")
    if step.reward is not None and not np.isfinite(step.reward):
        raise ValueError("Tiger reward must be finite or None")
    opened = {"open_left": 0, "open_right": 1}.get(step.action or "")
    outcome = ""
    if opened is not None:
        hit_tiger = (opened == 0) == (step.state == "tiger_left")
        outcome = "TIGER" if hit_tiger else "TREASURE"
    return TigerFrameData(
        hidden="LEFT" if step.state == "tiger_left" else "RIGHT",
        action=step.action if step.action is not None else "none (bookkeeping)",
        observation=step.observation if step.observation is not None else "none recorded",
        reward=f"{step.reward:+g}" if step.reward is not None else "none recorded",
        probabilities=_belief_probabilities(step.belief),
        opened_side=opened,
        outcome=outcome,
    )


class TigerVisualizer:
    """Save one 500 ms frame per StepData, without running the environment."""

    def _render_frame(self, data: TigerFrameData, index: int, count: int) -> Image.Image:
        image = _background().copy()
        draw = ImageDraw.Draw(image)
        draw.text((16, 10), "TIGER", fill=TEXT, font=_font(24))
        draw.text((785, 14), f"Step {index + 1}/{count}", fill=TEXT, font=_font(15), anchor="ra")
        draw.text((150, 16), "Saved episode replay", fill=GOLD, font=_font(15))
        draw.text((16, 43), f"Recorded action: {data.action}", fill=TEXT, font=_font(15))
        draw.text((785, 43), f"Reward: {data.reward}", fill=TEXT, font=_font(15), anchor="ra")
        draw.rounded_rectangle((183, 75, 617, 101), radius=3, fill=PANEL, outline=GOLD)
        draw.text(
            (400, 87),
            f"Hidden ground truth BEFORE action: {data.hidden}",
            font=_font(14),
            fill=RED,
            anchor="mm",
        )
        if data.opened_side is not None:
            center = DOOR_CENTERS[data.opened_side]
            color = RED if data.outcome == "TIGER" else GREEN
            draw.rounded_rectangle(
                (center - 84, 310, center + 84, 365), radius=5, fill=PANEL, outline=color, width=2
            )
            draw.text((center, 326), "ACTION RESULT", font=_font(13), fill=TEXT, anchor="mm")
            draw.text((center, 348), data.outcome, font=_font(20), fill=color, anchor="mm")
            draw.rounded_rectangle((200, 391, 600, 415), radius=3, fill=PANEL)
            draw.text(
                (400, 403),
                "Outcome uses prior state; next row uses reset state",
                font=_font(13),
                fill=TEXT,
                anchor="mm",
            )
        for side, center in enumerate(DOOR_CENTERS):
            probability = None if data.probabilities is None else data.probabilities[side]
            value = "unavailable" if probability is None else f"{probability:.2f}"
            label = "left" if side == 0 else "right"
            draw.text(
                (center, 439),
                f"Prior belief: tiger {label} {value}",
                font=_font(14),
                fill=TEXT,
                anchor="mm",
            )
            draw.rounded_rectangle((center - 116, 451, center + 116, 459), radius=3, fill=BAR_TRACK)
            if probability is not None and probability > 0:
                draw.rectangle(
                    (center - 115, 452, center - 115 + round(230 * probability), 458), fill=GOLD
                )
        draw.text(
            (400, 482),
            f"Recorded observation AFTER action: {data.observation}",
            font=_font(15),
            fill=TEXT,
            anchor="mm",
        )
        return image

    def create_visualization(self, history: list[StepData], cache_path: Path) -> None:
        if not isinstance(history, list) or any(not isinstance(step, StepData) for step in history):
            raise TypeError("history must be a list of StepData")
        if not history:
            raise ValueError("Cannot visualize empty Tiger history")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path")
        if cache_path.suffix.lower() != ".gif":
            raise ValueError("cache_path must end with .gif")
        data = [_frame_data(step) for step in history]
        frames = [
            self._render_frame(row, index, len(data)).quantize(
                palette=_palette(), dither=Image.Dither.NONE
            )
            for index, row in enumerate(data)
        ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            cache_path,
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            disposal=1,
            optimize=False,
        )
