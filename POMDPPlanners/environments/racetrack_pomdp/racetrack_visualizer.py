# SPDX-License-Identifier: MIT
"""Offline Racetrack artwork. Only recorded poses define vehicle movement.

The standard road is a reference map of highway-env 1.12.1's exact lane
parameters. Custom scenarios can supply lane polygons or omit the map. It is
never reconstructed from the planner's approximate curvature profile.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation.history import StepData
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    EGO_HEADING,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    state_agent_rows,
)

WIDTH, HEIGHT = 1100, 760
INK = "#203744"
MUTED = "#617581"
PAPER = "#f6f5ef"
GRASS = "#e3e8dc"
ROAD = "#384953"
EDGE = "#f8f5df"
BLUE = "#287fba"
TRAIL = "#81cde0"
AMBER = "#e2a233"
CORAL = "#c3634d"


@lru_cache(maxsize=16)
def _font(size: int):
    return ImageFont.load_default(size=size)


def _recorded_states(history: Sequence[StepData]) -> List[np.ndarray]:
    """Keep every row and append only an explicitly recorded final successor."""
    if not history:
        raise ValueError("history must contain at least one recorded step")
    states = [np.asarray(step.state, dtype=float) for step in history]
    if history[-1].next_state is not None:
        states.append(np.asarray(history[-1].next_state, dtype=float))
    return states


def _world_points(state: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rotate body-relative positions with this record's own ego heading."""
    angle = float(state[EGO_HEADING])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rows[:, [AGENT_REL_X, AGENT_REL_Y]] @ rotation.T + state[[EGO_X, EGO_Y]]


def _number(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):+.3g}"


class RacetrackVisualizer:
    """One 500 ms frame per record, with a 1500 ms hold on the final frame.

    Cars show the true pre-action pose. Amber dots project the stored belief
    onto ego and opponent position; larger rings indicate greater particle weight.
    Heading/speed uncertainty is omitted from this spatial projection. Observations
    and rewards belong to the displayed action's successor, not its initial pose.
    """

    def __init__(
        self,
        max_tracked_agents: int,
        action_presets: Sequence[tuple[float, float]] = DEFAULT_ACTION_PRESETS,
        track_lanes: Sequence[dict] | None = None,
    ) -> None:
        self.max_tracked_agents = max_tracked_agents
        self.action_presets = action_presets
        # No map is the safe default for arbitrary supplied histories.
        self.track_lanes = list(track_lanes) if track_lanes is not None else []

    def _opponents(self, state: np.ndarray) -> np.ndarray:
        if state.size < EGO_STATE_WIDTH + AGENT_SLOT_WIDTH * self.max_tracked_agents:
            return np.empty((0, 2))
        rows = state_agent_rows(state, self.max_tracked_agents)
        return _world_points(state, rows[rows[:, AGENT_PRESENT] > 0.5])

    def _belief_points(self, belief: Any) -> tuple[np.ndarray, np.ndarray]:
        particles = getattr(belief, "particles", None)
        if particles is None:
            return np.empty((0, 2)), np.empty(0)
        particles = np.asarray(particles, dtype=float)
        if particles.ndim != 2 or particles.shape[1] < EGO_STATE_WIDTH or not len(particles):
            return np.empty((0, 2)), np.empty(0)
        weights = np.asarray(
            getattr(belief, "normalized_weights", np.ones(len(particles))), dtype=float
        )
        if weights.shape != (len(particles),) or not np.all(np.isfinite(weights)):
            return np.empty((0, 2)), np.empty(0)
        points, masses = [], []
        for particle, weight in zip(particles, weights):
            if weight <= 0 or not np.all(np.isfinite(particle)):
                continue
            positions = np.vstack([particle[[EGO_X, EGO_Y]], self._opponents(particle)])
            points.extend(positions)
            masses.extend([weight] * len(positions))
        return np.asarray(points).reshape(-1, 2), np.asarray(masses)

    def _bounds(self, states: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
        points = []
        for state in states:
            points.append(state[[EGO_X, EGO_Y]])
            points.extend(self._opponents(state))
        for lane in self.track_lanes:
            points.extend(lane["polygon"])
        array = np.asarray(points)
        low, high = array.min(axis=0), array.max(axis=0)
        span = np.maximum(high - low, 20.0)
        return (
            float(low[0] - 0.08 * span[0]),
            float(low[1] - 0.12 * span[1]),
            float(high[0] + 0.08 * span[0]),
            float(high[1] + 0.12 * span[1]),
        )

    @staticmethod
    def _project(x: float, y: float, bounds) -> tuple[float, float]:
        left, bottom, right, top = bounds
        scale = min(980 / (right - left), 414 / (top - bottom))
        return (550 + (x - (left + right) / 2) * scale, 338 - (y - (bottom + top) / 2) * scale)

    @staticmethod
    def _text(draw, xy, text, size=17, fill=INK, anchor=None):
        draw.text(xy, text, font=_font(size), fill=fill, anchor=anchor)

    def _background(self, bounds) -> Image.Image:
        image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
        draw = ImageDraw.Draw(image)
        self._text(draw, (40, 24), "RACETRACK", 30)
        self._text(draw, (40, 66), "Recorded driving  /  observer view", 17, MUTED)
        draw.rounded_rectangle((30, 108, 1070, 568), radius=18, fill=GRASS)
        project = lambda points: [self._project(float(x), float(y), bounds) for x, y in points]
        # Paint all surfaces before boundaries, so adjacent lanes have no false seams.
        for lane in self.track_lanes:
            draw.polygon(project(lane["polygon"]), fill=ROAD)
        for lane in self.track_lanes:
            for line in lane["lines"]:
                points = project(line["points"])
                if line["type"] == 1:  # highway-env STRIPED
                    for start in range(0, len(points) - 1, 10):
                        draw.line(points[start : start + 5], fill="#aebbb9", width=1)
                elif line["type"] in (2, 3):
                    draw.line(points, fill=EDGE, width=2)
        note = (
            "Reference road: racetrack-v0 / highway-env 1.12.1"
            if self.track_lanes
            else "Road geometry unavailable"
        )
        self._text(draw, (50, 124), note, 13, MUTED)
        # Metric scale shares the pose transform; decorative borders are never road edges.
        x, y = self._project(bounds[0], bounds[1], bounds)
        p1 = self._project(bounds[0] + 10, bounds[1], bounds)
        bar = p1[0] - x
        draw.line((58, 540, 58 + bar, 540), fill=MUTED, width=2)
        draw.line((58, 536, 58, 544), fill=MUTED, width=2)
        draw.line((58 + bar, 536, 58 + bar, 544), fill=MUTED, width=2)
        self._text(draw, (58, 515), "10 m", 13, MUTED)
        return image

    def _car(self, draw, position, heading: float, bounds) -> None:
        # A 5 x 2 m footprint centred on the exact recorded pose. Artwork is
        # illustrative, not a new collision shape. Front windshield marks heading.
        cx, cy = position
        cosine, sine = np.cos(heading), np.sin(heading)

        def polygon(vertices, color):
            points = [
                self._project(cx + cosine * x - sine * y, cy + sine * x + cosine * y, bounds)
                for x, y in vertices
            ]
            draw.polygon(points, fill=color)

        polygon([(-2.5, -1), (1.9, -1), (2.5, -0.65), (2.5, 0.65), (1.9, 1), (-2.5, 1)], "#173f59")
        polygon(
            [(-2.2, -0.82), (1.9, -0.82), (2.2, -0.5), (2.2, 0.5), (1.9, 0.82), (-2.2, 0.82)], BLUE
        )
        polygon([(0.1, -0.75), (1.05, -0.62), (1.05, 0.62), (0.1, 0.75)], "#c3e5e9")
        polygon([(-1.8, -0.65), (-1.1, -0.65), (-1.1, 0.65), (-1.8, 0.65)], "#173f59")
        polygon([(1.8, -0.75), (2.15, -0.7), (2.15, -0.4), (1.8, -0.4)], "#fff2ca")
        polygon([(1.8, 0.4), (2.15, 0.4), (2.15, 0.7), (1.8, 0.75)], "#fff2ca")

    def _action(self, action: Any) -> str:
        if action is None:
            return "none"
        if isinstance(action, (int, np.integer)) and 0 <= int(action) < len(self.action_presets):
            accel, steer = self.action_presets[int(action)]
            return f"{int(action)}   accel {accel:+g} / steer {steer:+g}"
        return str(action)

    @staticmethod
    def _observation(observation: Any) -> str:
        if observation is None:
            return "unavailable"
        if hasattr(observation, "detections"):
            rows = np.asarray(observation.detections)
            count = int(np.sum(rows[:, 0] > 0.5))
            speed = float(np.asarray(observation.ego_speed).flat[0])
            return f"{count} detections / speed {speed:.2f} m/s"
        table = np.asarray(observation)
        if table.ndim == 2 and table.shape[1] == 5:
            return f"kinematics / {int(np.sum(table[:, 0] > .5))} present rows"
        return "recorded (not projected)"

    def render_frames(self, history: Sequence[StepData]) -> List[Image.Image]:
        """Render history without sampling, stepping, or modifying any record."""
        states = _recorded_states(history)
        bounds = self._bounds(states)
        background = self._background(bounds)
        trail = [self._project(float(s[EGO_X]), float(s[EGO_Y]), bounds) for s in states]
        frames = []
        total = 0.0
        for index, state in enumerate(states):
            step = history[index] if index < len(history) else None
            image = background.copy()
            draw = ImageDraw.Draw(image)
            if index:
                # Connect only actual adjacent transitions, not gaps in supplied history.
                for j in range(index):
                    successor = history[j].next_state
                    if successor is not None and np.array_equal(successor, states[j + 1]):
                        draw.line((trail[j], trail[j + 1]), fill=TRAIL, width=3)
                for point in trail[:index]:
                    draw.ellipse(
                        (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=TRAIL
                    )
            particles, masses = self._belief_points(step.belief if step else None)
            hidden = 0
            for point, mass in zip(particles, masses):
                px, py = self._project(float(point[0]), float(point[1]), bounds)
                if not (35 < px < 1065 and 150 < py < 558):
                    hidden += 1
                    continue
                radius = 1.5 + 3 * float(np.sqrt(mass / masses.max()))
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius), outline=AMBER, width=1
                )
            for point in self._opponents(state):
                px, py = self._project(float(point[0]), float(point[1]), bounds)
                # Opponent heading is absent from state. A round marker does not invent it.
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=CORAL, outline=PAPER, width=2)
            self._car(draw, state[[EGO_X, EGO_Y]], float(state[EGO_HEADING]), bounds)
            self._text(draw, (1060, 32), f"{index + 1:02d} / {len(states):02d}", 23, anchor="ra")
            phase = (
                "Pre-action state"
                if step is not None and step.action is not None
                else "Final recorded state"
            )
            self._text(draw, (1060, 69), phase, 15, MUTED, "ra")
            self._text(draw, (42, 590), "STATE", 13, MUTED)
            self._text(draw, (42, 613), f"{state[EGO_SPEED]:.2f} m/s", 26)
            self._text(draw, (42, 649), f"x {state[EGO_X]:.2f}   y {state[EGO_Y]:.2f} m", 15, MUTED)
            self._text(draw, (300, 590), "SELECTED CONTROL (NORMALISED)", 13, MUTED)
            self._text(draw, (300, 617), self._action(step.action if step else None), 17)
            self._text(draw, (300, 649), f"heading {state[EGO_HEADING]:+.3f} rad", 15, MUTED)
            self._text(draw, (730, 590), "RECORDED RESULT OF THIS ACTION", 13, MUTED)
            reward = step.reward if step else None
            if reward is not None:
                total += float(reward)
            self._text(draw, (730, 617), f"Reward {_number(reward)}   /   total {total:+.3g}", 17)
            self._text(
                draw, (730, 649), self._observation(step.observation if step else None), 15, MUTED
            )
            draw.line((40, 682, 1060, 682), fill="#d7dfd8", width=1)
            labels = [
                (42, BLUE, "ego"),
                (130, CORAL, "other vehicle"),
                (305, TRAIL, "recorded path"),
            ]
            for x, color, label in labels:
                draw.ellipse((x, 704, x + 10, 714), fill=color)
                self._text(draw, (x + 19, 700), label, 15, MUTED)
            belief_note = "belief unavailable"
            if len(particles):
                belief_note = (
                    f"belief: position projection ({hidden} outside view)"
                    if hidden
                    else "belief: position projection / larger = more weight"
                )
            draw.ellipse((495, 703, 507, 715), outline=AMBER, width=2)
            self._text(draw, (517, 700), belief_note, 15, MUTED)
            info = step.info if step else history[-1].info
            if step is not None and step.action is None and index > 0:
                previous = history[index - 1]
                if previous.next_state is not None and np.array_equal(previous.next_state, state):
                    info = previous.info
            if info:
                events = [
                    label
                    for key, label in [
                        ("crashed", "collision"),
                        ("off_road", "off road"),
                        ("time_limit", "time limit"),
                    ]
                    if info.get(key, 0)
                ]
                if events:
                    self._text(
                        draw, (1050, 538), "Recorded result: " + ", ".join(events), 16, CORAL, "ra"
                    )
            frames.append(image)
        return frames

    def save(self, history: Sequence[StepData], path: Path) -> None:
        """Save the final state as PNG, or every recorded frame as GIF."""
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        if path.suffix.lower() not in {".gif", ".png"}:
            raise ValueError("path must end in .gif or .png")
        frames = self.render_frames(history)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A shared palette makes PNG and decoded GIF pixels agree exactly.
        sample = frames[0].copy()
        swatches = ImageDraw.Draw(sample)
        for index, color in enumerate((TRAIL, CORAL, AMBER, BLUE)):
            swatches.rectangle((index * 30, 0, index * 30 + 29, 29), fill=color)
        palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        if path.suffix.lower() == ".png":
            indexed[-1].convert("RGB").save(path)
        else:
            indexed[0].save(
                path,
                save_all=True,
                append_images=indexed[1:],
                duration=[500] * (len(frames) - 1) + [1500],
                loop=0,
                optimize=False,
                disposal=2,
            )
