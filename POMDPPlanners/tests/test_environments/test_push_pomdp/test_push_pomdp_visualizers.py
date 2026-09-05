# SPDX-License-Identifier: MIT

"""Behavior tests for the cached Pillow Push renderers."""

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp_visualizer import (
    ContinuousPushPOMDPVisualizer,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import PushPOMDPVisualizer
from POMDPPlanners.environments.push_pomdp.push_visualization_utils import (
    ACTION,
    CANVAS_SIZE,
    COLLISION_BOX,
    DANGER,
    GIF_DURATION_MS,
    OBJECT,
    OBSTACLE,
    ROBOT,
    ROBOT_RADIUS,
    SUCCESS_BOX,
    TARGET,
)


class _DiscreteFixture:
    grid_size = 10
    push_threshold = 1.0
    obstacles = [(4.0, 4.0), (7.0, 2.0)]
    obstacle_radius = 0.5
    dangerous_areas = [(2.0, 7.0)]
    dangerous_area_radius = 0.8

    def _is_colliding_with_obstacle(self, pos: np.ndarray) -> bool:
        return any(
            np.linalg.norm(pos - np.asarray(center)) <= self.obstacle_radius
            for center in self.obstacles
        )


class _ContinuousFixture:
    grid_size = 10
    push_threshold = 1.0
    robot_radius = 0.3
    obstacles = np.array([[4.0, 4.0, 0.7, 0.5], [7.0, 2.0, 0.5, 0.9]])
    dangerous_areas = [(2.0, 7.0)]
    dangerous_area_radius = 0.8

    def _is_circle_colliding_with_obstacle(self, pos: np.ndarray, radius: float) -> bool:
        return bool(
            abs(pos[0] - 4.0) <= 0.7 + radius
            and abs(pos[1] - 4.0) <= 0.5 + radius
        )

    def _is_point_colliding_with_obstacle(self, pos: np.ndarray) -> bool:
        return bool(abs(pos[0] - 4.0) <= 0.7 and abs(pos[1] - 4.0) <= 0.5)


def _step(state: list[float], action: Any, reward: float | None) -> StepData:
    return StepData(np.asarray(state), action, None, None, reward, None)  # type: ignore[arg-type]


DISCRETE_HISTORY = [
    _step([1, 1, 1.7, 1, 4, 1], "right", -1.0),
    _step([1, 1, 1.7, 1, 4, 1], "right", 2.0),
    _step([4, 4, 3, 1, 4, 1], "up", -10.0),
    _step([4, 4, 4, 1, 4, 1], None, None),
]

CONTINUOUS_HISTORY = [
    _step([1.0, 1.0, 1.7, 1.0, 4.0, 1.0], np.array([1.0, 0.0]), -1.0),
    _step([1.0, 1.0, 1.7, 1.0, 4.0, 1.0], np.array([0.8, 0.2]), 2.0),
    _step([4.0, 4.0, 3.3, 1.0, 4.0, 1.0], np.array([0.0, 1.0]), -10.0),
    _step([4.0, 4.0, 4.0, 1.0, 4.0, 1.0], None, None),
]


@pytest.mark.parametrize(
    ("visualizer", "history"),
    [
        (PushPOMDPVisualizer(_DiscreteFixture()), DISCRETE_HISTORY),  # type: ignore[arg-type]
        (
            ContinuousPushPOMDPVisualizer(_ContinuousFixture()),  # type: ignore[arg-type]
            CONTINUOUS_HISTORY,
        ),
    ],
)
def test_push_gif_keeps_frames_size_duration_and_repeated_states(
    tmp_path: Path, visualizer: Any, history: list[StepData]
) -> None:
    output = tmp_path / "nested" / "episode.gif"
    visualizer.create_visualization(history, output)

    with Image.open(output) as gif:
        assert gif.size == CANVAS_SIZE
        assert gif.n_frames == len(history)
        durations = []
        frames = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
            frames.append(gif.convert("RGB").tobytes())
    assert durations == [GIF_DURATION_MS] * len(history)
    assert frames[0] != frames[1], "step text must preserve repeated recorded states"


@pytest.mark.parametrize(
    ("visualizer", "history"),
    [
        (PushPOMDPVisualizer(_DiscreteFixture()), DISCRETE_HISTORY),  # type: ignore[arg-type]
        (
            ContinuousPushPOMDPVisualizer(_ContinuousFixture()),  # type: ignore[arg-type]
            CONTINUOUS_HISTORY,
        ),
    ],
)
def test_push_gif_bytes_are_deterministic(
    tmp_path: Path, visualizer: Any, history: list[StepData]
) -> None:
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    visualizer.create_visualization(history, first)
    visualizer.create_visualization(history, second)
    assert sha256(first.read_bytes()).digest() == sha256(second.read_bytes()).digest()


@pytest.mark.parametrize(
    ("visualizer", "history"),
    [
        (PushPOMDPVisualizer(_DiscreteFixture()), DISCRETE_HISTORY),  # type: ignore[arg-type]
        (
            ContinuousPushPOMDPVisualizer(_ContinuousFixture()),  # type: ignore[arg-type]
            CONTINUOUS_HISTORY,
        ),
    ],
)
def test_push_static_background_is_built_once(visualizer: Any, history: list[StepData]) -> None:
    visualizer.render_frames(history)
    visualizer.render_frames(history)
    assert visualizer._background_build_count == 1


def _pixel(frame: Image.Image, visualizer: Any, point: tuple[float, float]) -> tuple[int, int, int]:
    x, y = visualizer._to_px(point)
    return frame.getpixel((round(x), round(y)))


def _has_color(frame: Image.Image, color: tuple[int, int, int]) -> bool:
    pixels = np.asarray(frame)
    return bool(np.any(np.all(pixels == color, axis=2)))


def test_discrete_push_frames_preserve_scene_and_events() -> None:
    visualizer = PushPOMDPVisualizer(_DiscreteFixture())  # type: ignore[arg-type]
    first, _, collision, terminal = visualizer.render_frames(DISCRETE_HISTORY)

    assert _pixel(first, visualizer, (2.0, 7.0)) == DANGER
    assert _pixel(first, visualizer, (7.0, 2.0)) == OBSTACLE
    assert _pixel(first, visualizer, (1.0, 1.0)) == ROBOT
    assert _pixel(first, visualizer, (1.7, 1.0)) == OBJECT
    assert _pixel(first, visualizer, (4.0, 1.0)) == TARGET
    assert _has_color(first, ACTION), "push and action arrows must stay visible"
    assert _has_color(collision, COLLISION_BOX)
    assert _has_color(terminal, SUCCESS_BOX)


def test_continuous_push_frames_preserve_scene_and_events() -> None:
    visualizer = ContinuousPushPOMDPVisualizer(_ContinuousFixture())  # type: ignore[arg-type]
    first, _, collision, terminal = visualizer.render_frames(CONTINUOUS_HISTORY)

    assert _pixel(first, visualizer, (2.0, 7.0)) == DANGER
    assert _pixel(first, visualizer, (7.0, 2.0)) == OBSTACLE
    assert _pixel(first, visualizer, (1.0, 1.0)) == ROBOT
    assert _pixel(first, visualizer, (1.0, 1.25)) == ROBOT_RADIUS
    assert _pixel(first, visualizer, (1.7, 1.0)) == OBJECT
    assert _pixel(first, visualizer, (4.0, 1.0)) == TARGET
    assert _has_color(first, ACTION), "push and continuous action arrows must stay visible"
    assert _has_color(collision, COLLISION_BOX)
    assert _has_color(terminal, SUCCESS_BOX)


@pytest.mark.parametrize(
    "visualizer",
    [
        PushPOMDPVisualizer(_DiscreteFixture()),  # type: ignore[arg-type]
        ContinuousPushPOMDPVisualizer(_ContinuousFixture()),  # type: ignore[arg-type]
    ],
)
def test_push_visualizer_keeps_input_validation(tmp_path: Path, visualizer: Any) -> None:
    with pytest.raises(ValueError, match="empty history"):
        visualizer.create_visualization([], tmp_path / "empty.gif")
    with pytest.raises(TypeError, match="Path object"):
        visualizer.create_visualization(DISCRETE_HISTORY, "episode.gif")
    with pytest.raises(ValueError, match="must end with .gif"):
        visualizer.create_visualization(DISCRETE_HISTORY, tmp_path / "episode.png")
