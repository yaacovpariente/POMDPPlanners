# SPDX-License-Identifier: MIT
"""Behavior checks for cached continuous LaserTag frames."""

import builtins
import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, GifImagePlugin
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
    CANVAS_SIZE,
    LASER_COLOR,
    ROBOT_BELIEF_COLOR,
    OPPONENT_BELIEF_COLOR,
)


@pytest.fixture
def visualizer():
    return ContinuousLaserTagVisualizer(
        np.array([11.0, 7.0]),
        np.array([[5.0, 3.0, 0.5, 1.0]]),
        0.3,
        0.3,
        [(3.0, 5.0)],
        1.0,
    )


def history():
    belief: Any = SimpleNamespace(
        to_unique_support_distribution=lambda: SimpleNamespace(
            values=[np.array([2.3, 2.2, 7.2, 4.1, 0.0])]
        )
    )
    states = [
        [1.25, 1.5, 8.2, 4.5, 0.0],
        [1.8, 1.5, 7.7, 4.2, 0.0],
        [2.1, 2.0, 7.0, 4.0, 0.0],
        [3.0, 2.0, 3.8, 2.0, 0.0],
        [3.0, 2.0, 3.8, 2.0, 1.0],
        [3.0, 2.0, 3.8, 2.0, 1.0],
    ]
    actions = [np.array([1.0, 0.0, 0.0]), "up", "tag", np.array([0.0, 0.0, 1.0]), None, None]
    return [StepData(np.array(s), a, None, None, None, belief) for s, a in zip(states, actions)]


def color_count(image, color):
    return np.count_nonzero(np.all(np.asarray(image) == color, axis=-1))


def test_frames_timing_determinism_and_background_reuse(visualizer, tmp_path):
    first, second = tmp_path / "nested/first.gif", tmp_path / "second.gif"
    with patch.object(visualizer, "_build_background", wraps=visualizer._build_background) as build:
        visualizer.create_visualization(history(), first)
        assert build.call_count == 1
    visualizer.create_visualization(history(), second)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    with Image.open(first) as image:
        assert isinstance(image, GifImagePlugin.GifImageFile)
        assert image.size == CANVAS_SIZE
        assert image.n_frames == 6
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 1000
    one = tmp_path / "single.gif"
    visualizer.create_visualization(history()[-1:], one)
    with Image.open(one) as image:
        assert isinstance(image, GifImagePlugin.GifImageFile)
        assert image.n_frames == 1


def test_scene_geometry_and_background_unchanged(visualizer):
    background = visualizer._build_background()
    original = background.tobytes()
    wall = background.getpixel(tuple(map(round, visualizer._world_to_pixel((5.0, 3.0)))))
    assert max(wall) < 90 and wall[2] >= wall[0], "walls retain dark steel texture"
    danger = background.getpixel(tuple(map(round, visualizer._world_to_pixel((3.2, 5.2)))))
    assert danger[0] > max(danger[1], danger[2])
    p = visualizer._world_to_pixel((1.25, 2.5))
    q = visualizer._world_to_pixel((1.5, 2.75))
    assert q[0] - p[0] == pytest.approx(0.25 * visualizer._scale)
    assert p[1] - q[1] == pytest.approx(0.25 * visualizer._scale)
    step = history()[0]
    frame = visualizer._render_frame(
        background, step.state[:2], step.state[2:4], step.action, step.belief, 0, 6
    )
    assert background.tobytes() == original
    for color in (
        LASER_COLOR,
        ROBOT_BELIEF_COLOR,
        OPPONENT_BELIEF_COLOR,
    ):
        assert color_count(frame, color) > color_count(background, color)
    for point, channel in ((step.state[:2], 0), (step.state[2:4], 2)):
        x, y = map(round, visualizer._world_to_pixel(point))
        pixels = np.asarray(frame.crop((x - 30, y - 30, x + 30, y + 30))).astype(int)
        assert np.any(pixels[:, :, channel] > pixels[:, :, (channel + 1) % 3] + 45)


def test_laser_stops_at_wall_and_beliefs_clear(visualizer):
    background = visualizer._build_background()
    rp, op = np.array([2.0, 3.0]), np.array([8.0, 6.0])
    frame = visualizer._render_frame(background, rp, op, None, None, 0, 1)
    x, y = map(int, visualizer._world_to_pixel((4.0, 3.0)))
    assert frame.getpixel((x, y)) == LASER_COLOR
    x, y = map(int, visualizer._world_to_pixel((5.0, 3.0)))
    assert frame.getpixel((x, y)) == background.getpixel((x, y))
    step = history()[0]
    with_belief = visualizer._render_frame(background, rp, op, None, step.belief, 0, 1)
    assert color_count(with_belief, ROBOT_BELIEF_COLOR) > color_count(frame, ROBOT_BELIEF_COLOR)
    assert color_count(with_belief, OPPONENT_BELIEF_COLOR) > color_count(
        frame, OPPONENT_BELIEF_COLOR
    )


def test_tag_and_action_semantics(visualizer):
    background = visualizer._build_background()

    def render(op, action):
        return visualizer._render_frame(
            background, np.array([3.0, 2.0]), np.array(op), action, None, 0, 1
        )

    missed = render([8.0, 2.0], "tag")
    tagged = render([3.8, 2.0], np.array([0.0, 0.0, 1.0]))
    terminal = render([3.8, 2.0], None)
    bounds = (
        1150,
        665,
        1390,
        735,
    )
    assert color_count(missed.crop(bounds), (255, 0, 0)) > 0
    assert color_count(tagged.crop(bounds), (0, 128, 0)) > 0
    assert color_count(terminal.crop(bounds), (0, 128, 0)) == 0
    arrow = render([8.0, 2.0], "right")
    assert color_count(arrow, (255, 0, 0)) > color_count(terminal, (255, 0, 0))


def test_episode_does_not_import_matplotlib(tmp_path):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("matplotlib")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        renderer = ContinuousLaserTagVisualizer(
            np.array([11.0, 7.0]), np.empty((0, 4)), 0.3, 0.3, [], 1.0
        )
        renderer.create_visualization(history(), tmp_path / "episode.gif")


@pytest.mark.parametrize("bad", [None, (), "history"])
def test_history_type(visualizer, tmp_path, bad):
    with pytest.raises(TypeError, match="list"):
        visualizer.create_visualization(bad, tmp_path / "episode.gif")


def test_bad_state_and_step(visualizer, tmp_path):
    with pytest.raises(TypeError, match="StepData"):
        visualizer.create_visualization([None], tmp_path / "episode.gif")
    with pytest.raises(ValueError, match="state"):
        visualizer.create_visualization(
            [history()[0]._replace(state=np.zeros(4))], tmp_path / "episode.gif"
        )


def test_danger_fields_are_clipped_to_world_margin(visualizer):
    plain = visualizer._build_background()
    visualizer.dangerous_areas = [(0.0, 0.0)]
    background = visualizer._build_background()
    x, y = map(round, visualizer._world_to_pixel((-0.65, 0.25)))
    assert background.getpixel((x, y)) == plain.getpixel((x, y))
    x, y = map(round, visualizer._world_to_pixel((-0.25, 0.25)))
    red, green, blue = background.getpixel((x, y))
    assert red > max(green, blue)


def test_sidebar_never_covers_upper_right_actor(visualizer):
    background = visualizer._build_background()
    assert visualizer._right < visualizer._legend_bounds[0]
    op = visualizer.grid_size - np.array([0.2, 0.2])
    frame = visualizer._render_frame(background, np.array([2.0, 2.0]), op, None, None, 0, 1)
    x, y = map(round, visualizer._world_to_pixel(op))
    before = np.asarray(background.crop((x - 25, y - 25, x + 25, y + 25)))
    after = np.asarray(frame.crop((x - 25, y - 25, x + 25, y + 25)))
    assert np.count_nonzero(np.any(before != after, axis=2)) > 300


def test_actor_assets_have_real_transparency():
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_assets import _sheet

    sheet = _sheet()
    assert sheet.getchannel("A").getextrema() == (0, 255)
    rgba = np.asarray(sheet)
    assert not rgba[0, :, 3].any()
    assert not rgba[-1, :, 3].any()
    assert not rgba[:, 0, 3].any()
    assert not rgba[:, -1, 3].any()
    # Bright, opaque metal highlights survive the connected-background key.
    bright = (rgba[:, :, :3].min(axis=2) > 210) & (rgba[:, :, 3] == 255)
    assert bright.sum() > 100


def test_delta_gif_decodes_to_complete_frames(visualizer, tmp_path):
    steps = history()
    path = tmp_path / "delta.gif"
    visualizer.create_visualization(steps, path)
    background = visualizer._build_background()
    palette = visualizer._make_palette(background)
    robots, opponents, actions, beliefs = visualizer._extract_history(steps)
    with Image.open(path) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.n_frames == len(steps)
        for index in range(len(steps)):
            expected = (
                visualizer._render_frame(
                    background,
                    robots[index],
                    opponents[index],
                    actions[index],
                    beliefs[index],
                    index,
                    len(steps),
                    robots[: index + 1],
                    opponents[: index + 1],
                )
                .quantize(palette=palette, dither=Image.Dither.NONE)
                .convert("RGB")
            )
            gif.seek(index)
            assert gif.info["duration"] == 1000
            assert gif.convert("RGB").tobytes() == expected.tobytes()


def test_malformed_belief_leaves_no_partial_markers(visualizer):
    background = visualizer._build_background()
    before = background.tobytes()
    belief = SimpleNamespace(
        to_unique_support_distribution=lambda: SimpleNamespace(
            values=[np.array([2, 2, 7, 4, 0]), np.zeros((5, 2))]
        )
    )
    visualizer._draw_belief(ImageDraw.Draw(background), belief)
    assert background.tobytes() == before


def test_gif_palette_preserves_robot_trail_color(visualizer, tmp_path):
    path = tmp_path / "trail.gif"
    visualizer.create_visualization(history(), path)
    with Image.open(path) as gif:
        gif.seek(2)
        assert color_count(gif.convert("RGB"), (255, 160, 160)) > 0
