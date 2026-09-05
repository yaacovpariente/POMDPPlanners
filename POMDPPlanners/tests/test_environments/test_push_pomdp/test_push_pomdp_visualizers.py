# SPDX-License-Identifier: MIT

"""Behavior tests for the cached Pillow Push renderers."""

from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import GifImagePlugin, Image
import pytest
from POMDPPlanners.environments.push_pomdp.push_visualization_assets import paste_obstacle

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp_visualizer import (
    ContinuousPushPOMDPVisualizer,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import PushPOMDPVisualizer
from POMDPPlanners.environments.push_pomdp.push_visualization_utils import (
    ACTION,
    CANVAS_SIZE,
    COLLISION_BOX,
    GIF_DURATION_MS,
    OBSTACLE,
    PUSH_LINE,
    ROBOT_RADIUS,
    SUCCESS_BOX,
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
        return bool(abs(pos[0] - 4.0) <= 0.7 + radius and abs(pos[1] - 4.0) <= 0.5 + radius)

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
        assert isinstance(gif, GifImagePlugin.GifImageFile)
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
    return cast(tuple[int, int, int], frame.getpixel((round(x), round(y))))


def _has_color(frame: Image.Image, color: tuple[int, int, int]) -> bool:
    pixels = np.asarray(frame)
    return bool(np.any(np.all(pixels == color, axis=2)))


@pytest.mark.parametrize("circle", [True, False])
def test_tiny_obstacle_keeps_valid_outline(circle: bool) -> None:
    canvas = Image.new("RGB", (20, 20), "white")
    paste_obstacle(canvas, (10.0, 10.0, 11.0, 11.0), circle)
    assert canvas.getpixel((10, 10)) != (255, 255, 255)


def _assert_scene_art(frame: Image.Image, visualizer: Any) -> None:
    for point, kind in (((1.0, 1.0), "robot"), ((1.7, 1.0), "object"), ((4.0, 1.0), "target")):
        x, y = visualizer._to_px(point)
        patch = np.asarray(
            frame.crop((round(x) - 15, round(y) - 15, round(x) + 15, round(y) + 15))
        ).astype(int)
        if kind == "robot":
            assert np.any(
                (patch[:, :, 2] > patch[:, :, 0] + 35) & (patch[:, :, 2] > patch[:, :, 1] + 15)
            )
        else:
            assert np.any(
                (patch[:, :, 0] > patch[:, :, 2] + 60) & (patch[:, :, 0] > patch[:, :, 1] + 20)
            )
        assert len(np.unique(patch.reshape(-1, 3), axis=0)) > 30, "sprites retain shaded detail"
    danger = _pixel(frame, visualizer, (2.0, 7.0))
    assert danger[0] > danger[1] * 1.3, "hazard remains red over terrain"
    obstacle = _pixel(frame, visualizer, (7.0, 2.0))
    assert obstacle[0] > obstacle[1] * 1.3 and obstacle[1] < 110, "obstacle remains dark rust stone"


def test_discrete_push_frames_preserve_scene_and_events() -> None:
    visualizer = PushPOMDPVisualizer(_DiscreteFixture())  # type: ignore[arg-type]
    first, _, collision, terminal = visualizer.render_frames(DISCRETE_HISTORY)

    _assert_scene_art(first, visualizer)
    assert _has_color(first, ACTION), "push and action arrows must stay visible"
    assert _has_color(collision, COLLISION_BOX)
    assert _has_color(terminal, SUCCESS_BOX)


def test_continuous_push_frames_preserve_scene_and_events() -> None:
    visualizer = ContinuousPushPOMDPVisualizer(_ContinuousFixture())  # type: ignore[arg-type]
    first, _, collision, terminal = visualizer.render_frames(CONTINUOUS_HISTORY)

    _assert_scene_art(first, visualizer)
    rx, ry = visualizer._to_px((1.0, 1.0))
    assert _has_color(
        first.crop((round(rx) - 30, round(ry) - 30, round(rx) + 30, round(ry) + 30)), ROBOT_RADIUS
    )
    assert _has_color(first, ACTION), "push and continuous action arrows must stay visible"
    tail = _pixel(first, visualizer, (1.29, 1.0))
    assert (
        tail[0] > 200 and tail[1] < 40 and tail[2] < 40
    ), "arrow tail must overlay the radius disc"
    assert _has_color(collision, COLLISION_BOX)
    assert _has_color(terminal, SUCCESS_BOX)


def test_push_connection_is_distinct_from_obstacles() -> None:
    visualizer = PushPOMDPVisualizer(_DiscreteFixture())  # type: ignore[arg-type]
    history = [
        _step([1, 1, 1, 1.7, 4, 1], "right", -1.0),
        _step([1, 1, 1, 1.7, 4, 1], None, None),
    ]
    frame = visualizer.render_frames(history)[0]
    assert _pixel(frame, visualizer, (1.0, 1.35)) == PUSH_LINE
    rust = _pixel(frame, visualizer, (7.0, 2.0))
    assert (
        rust[0] < 190 and rust[1] > 45 and rust[2] > 25
    ), "obstacle color must remain visibly distinct from pure-red connections"
    assert PUSH_LINE != OBSTACLE


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
