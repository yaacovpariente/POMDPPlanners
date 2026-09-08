# SPDX-License-Identifier: MIT

"""Deterministic Pillow renderer for recorded Racetrack episode states."""

from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from POMDPPlanners.core.simulation.history import StepData
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_X,
    AGENT_REL_Y,
    EGO_HEADING,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    state_agent_rows,
)

_SIZE = 640
_MARGIN = 48
_FRAME_DURATION_MS = 125
_BACKGROUND = (244, 246, 248)
_TRACK = (87, 96, 106)
_TRAIL = (42, 116, 181)
_EGO = (22, 101, 52)
_OPPONENT = (194, 55, 45)


def _recorded_states(history: Sequence[StepData]) -> List[np.ndarray]:
    """Return each recorded state once, including a non-retired final successor."""
    if not history:
        raise ValueError("history must contain at least one recorded step")

    states = [np.asarray(step.state, dtype=float) for step in history]
    final_successor = history[-1].next_state
    if final_successor is not None:
        states.append(np.asarray(final_successor, dtype=float))
    return states


class RacetrackVisualizer:
    """Render saved episodes from history without touching the live simulator."""

    def __init__(self, max_tracked_agents: int) -> None:
        self.max_tracked_agents = max_tracked_agents

    def _bounds(self, states: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
        # pylint: disable=too-many-locals
        """Return fixed world bounds that contain every recorded vehicle."""
        points = []
        for state in states:
            ego_x, ego_y = float(state[EGO_X]), float(state[EGO_Y])
            points.append((ego_x, ego_y))
            if state.size < EGO_STATE_WIDTH + 5 * self.max_tracked_agents:
                continue
            heading = float(state[EGO_HEADING])
            cosine, sine = float(np.cos(heading)), float(np.sin(heading))
            rows = state_agent_rows(state, self.max_tracked_agents)
            for row in rows[rows[:, AGENT_PRESENT] > 0.5]:
                rel_x, rel_y = float(row[AGENT_REL_X]), float(row[AGENT_REL_Y])
                points.append(
                    (
                        ego_x + cosine * rel_x - sine * rel_y,
                        ego_y + sine * rel_x + cosine * rel_y,
                    )
                )
        points = np.asarray(points, dtype=float)
        low = points.min(axis=0)
        high = points.max(axis=0)
        centre = (low + high) / 2.0
        span = max(float(np.max(high - low)), 20.0) * 1.2
        return (
            float(centre[0] - span / 2.0),
            float(centre[1] - span / 2.0),
            float(centre[0] + span / 2.0),
            float(centre[1] + span / 2.0),
        )

    @staticmethod
    def _project(
        x: float, y: float, bounds: tuple[float, float, float, float]
    ) -> tuple[int, int]:
        left, bottom, right, top = bounds
        scale = (_SIZE - 2 * _MARGIN) / max(right - left, top - bottom)
        return (
            round(_MARGIN + (x - left) * scale),
            round(_SIZE - _MARGIN - (y - bottom) * scale),
        )

    def render_frames(self, history: Sequence[StepData]) -> List[Image.Image]:
        # pylint: disable=too-many-locals
        """Draw one frame for each state in a normalized recorded history."""
        states = _recorded_states(history)
        bounds = self._bounds(states)
        trail = [self._project(float(s[EGO_X]), float(s[EGO_Y]), bounds) for s in states]
        frames: List[Image.Image] = []

        for index, state in enumerate(states):
            image = Image.new("RGB", (_SIZE, _SIZE), _BACKGROUND)
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (_MARGIN, _MARGIN, _SIZE - _MARGIN, _SIZE - _MARGIN),
                outline=_TRACK,
                width=2,
            )
            if index:
                draw.line(trail[: index + 1], fill=_TRAIL, width=5)

            ego_x, ego_y = float(state[EGO_X]), float(state[EGO_Y])
            ego_point = self._project(ego_x, ego_y, bounds)
            heading = float(state[EGO_HEADING])
            direction = (np.cos(heading), np.sin(heading))
            nose = self._project(ego_x + 2.0 * direction[0], ego_y + 2.0 * direction[1], bounds)
            draw.line((ego_point, nose), fill=_EGO, width=6)
            draw.ellipse(
                (ego_point[0] - 7, ego_point[1] - 7, ego_point[0] + 7, ego_point[1] + 7),
                fill=_EGO,
            )

            if state.size >= EGO_STATE_WIDTH + 5 * self.max_tracked_agents:
                rows = state_agent_rows(state, self.max_tracked_agents)
                cosine, sine = float(np.cos(heading)), float(np.sin(heading))
                for row in rows[rows[:, AGENT_PRESENT] > 0.5]:
                    rel_x, rel_y = float(row[AGENT_REL_X]), float(row[AGENT_REL_Y])
                    world_x = ego_x + cosine * rel_x - sine * rel_y
                    world_y = ego_y + sine * rel_x + cosine * rel_y
                    px, py = self._project(world_x, world_y, bounds)
                    draw.rectangle((px - 6, py - 4, px + 6, py + 4), fill=_OPPONENT)

            draw.text((16, 16), f"Recorded frame {index + 1}/{len(states)}", fill=_TRACK)
            frames.append(image)
        return frames

    def save(self, history: Sequence[StepData], path: Path) -> None:
        """Save one recorded frame as PNG or all recorded frames as GIF."""
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        suffix = path.suffix.lower()
        if suffix not in {".gif", ".png"}:
            raise ValueError("path must end in .gif or .png")

        frames = self.render_frames(history)
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".png":
            frames[-1].save(path)
            return
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=_FRAME_DURATION_MS,
            loop=0,
            optimize=False,
        )
