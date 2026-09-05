# SPDX-License-Identifier: MIT
"""Behavior checks for cached discrete LaserTag frames."""

import builtins
import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import GifImagePlugin, Image, ImageChops, ImageDraw
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_renderer import (
    CANVAS_SIZE,
    LASER_COLOR,
    LaserTagFrameRenderer,
    OPPONENT_BELIEF_COLOR,
    ROBOT_BELIEF_COLOR,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualizer import LaserTagVisualizer


@pytest.fixture
def visualizer():
    return LaserTagVisualizer((7, 5), {(3, 2)}, [(1, 3)], 0.8)


def history():
    belief: Any = SimpleNamespace(
        to_unique_support_distribution=lambda: SimpleNamespace(
            values=[np.array([1, 1, 5, 3, 0])], probs=np.array([1.0])
        )
    )
    states = ([1, 1, 5, 3, 0], [2, 1, 4, 3, 0], [2, 2, 2, 2, 1], [2, 2, 2, 2, 1])
    actions = (1, 2, 4, None)
    return [
        StepData(np.asarray(state), action, None, None, None, belief)
        for state, action in zip(states, actions)
    ]


def color_count(image, color):
    return np.count_nonzero(np.all(np.asarray(image.convert("RGB")) == color, axis=-1))


def test_discrete_visualizer_uses_neutral_renderer():
    assert LaserTagVisualizer.__bases__ == (LaserTagFrameRenderer,)


def test_frames_are_deterministic_and_keep_terminal_records(visualizer, tmp_path):
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"
    visualizer.create_visualization(history(), first)
    visualizer.create_visualization(history(), second)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    with Image.open(first) as image:
        assert isinstance(image, GifImagePlugin.GifImageFile)
        assert image.size == CANVAS_SIZE
        assert image.n_frames == len(history())
        for frame in range(image.n_frames):
            image.seek(frame)
            assert image.info["duration"] == 1000


def test_scene_rays_beliefs_paths_and_tag(visualizer):
    background = visualizer._build_background()
    steps = history()
    frame = visualizer._render_frame(
        background,
        steps[2].state[:2],
        steps[2].state[2:4],
        steps[2].action,
        steps[2].belief,
        2,
        4,
        [step.state[:2] for step in steps[:3]],
        [step.state[2:4] for step in steps[:3]],
    )
    assert color_count(frame, LASER_COLOR) > color_count(background, LASER_COLOR)
    assert color_count(frame, ROBOT_BELIEF_COLOR) > color_count(background, ROBOT_BELIEF_COLOR)
    assert color_count(frame, OPPONENT_BELIEF_COLOR) > color_count(
        background, OPPONENT_BELIEF_COLOR
    )
    assert color_count(frame, (0, 128, 0)) > 0
    wall_pixel = tuple(map(round, visualizer._world_to_pixel((3, 2))))
    wall = background.getpixel(wall_pixel)
    assert max(wall) < 90 and wall[2] >= wall[0]
    segments = visualizer._laser_segments(np.array([1, 2]), np.array([6, 4]))
    assert any(np.array_equal(end, np.array([2.0, 2.0])) for _, end in segments)


def test_episode_does_not_import_matplotlib(tmp_path):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("matplotlib")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        LaserTagVisualizer((7, 5), {(3, 2)}, [], 0.8).create_visualization(
            history(), tmp_path / "episode.gif"
        )


def test_discrete_belief_markers_preserve_probability_weight(visualizer):
    visualizer._build_background()
    image = Image.new("RGB", CANVAS_SIZE, "white")
    states = [np.array([row, 1, row, 3, 0]) for row in (1, 3, 5)]
    belief = SimpleNamespace(
        to_unique_support_distribution=lambda: SimpleNamespace(
            values=states, probs=np.array([0.1, 0.9, 0.0])
        )
    )
    visualizer._draw_belief(ImageDraw.Draw(image), belief)
    for coordinate, color in (
        (slice(0, 2), ROBOT_BELIEF_COLOR),
        (slice(2, 4), OPPONENT_BELIEF_COLOR),
    ):
        areas = []
        for state in states:
            x, y = map(round, visualizer._world_to_pixel(state[coordinate]))
            areas.append(color_count(image.crop((x - 12, y - 12, x + 13, y + 13)), color))
        assert 0 < areas[0] < areas[1]
        assert areas[1] > 4 * areas[0]
        assert areas[2] == 0


def test_discrete_walls_keep_public_cell_set(visualizer):
    assert visualizer.walls == {(3, 2)}
    assert (3, 2) in visualizer.walls


def test_column_label_is_vertical_and_clear_of_tick_labels(visualizer):
    labeled = visualizer._build_background()
    visualizer._y_label = ""
    unlabeled = visualizer._build_background()
    bounds = ImageChops.difference(labeled, unlabeled).getbbox()
    assert bounds is not None
    left, top, right, bottom = bounds
    assert bottom - top > right - left
    assert right < visualizer._left - 30
