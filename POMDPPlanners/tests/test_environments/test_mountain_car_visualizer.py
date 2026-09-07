# SPDX-License-Identifier: MIT
"""Geometry and saved-episode contracts for MountainCar artwork."""

import math
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import GifImagePlugin, Image

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_visualizer import (
    MountainCarVisualizer,
    _asset,
    _car,
)


def row(position=-0.5, velocity=0.0, action: int | None = 1, belief: Any = None):
    return StepData(
        np.array([position, velocity]),
        action,
        np.array([position, velocity]) if action is not None else None,
        np.array([-0.2, 0.01]) if action is not None else None,
        -1.0 if action is not None else None,
        belief,
    )


@pytest.mark.parametrize("position", [-1.2, -0.9, -math.pi / 6, -0.469, 0.0, 0.5, 0.6])
def test_exact_curve_coordinates_and_display_tangent(position):
    env = MountainCarPOMDP(0.95)
    renderer = MountainCarVisualizer(env)
    expected_height = 0.45 * math.sin(3 * position) + 0.55
    assert renderer.hill_height(position) == pytest.approx(expected_height)
    x, y = renderer.point(position)
    assert x == pytest.approx(45 + (position + 1.2) * 710 / 1.8)
    assert y == pytest.approx(416 - 225 * expected_height)
    slope = math.tan(math.radians(renderer.slope_angle(position)))
    assert slope == pytest.approx(225 * 1.35 * math.cos(3 * position) / (710 / 1.8))
    with patch(
        "POMDPPlanners.environments.mountain_car_pomdp.mountain_car_visualizer._car", wraps=_car
    ) as car:
        renderer.render_frame(row(position), 0, 1)
    assert car.call_args.args[0] == pytest.approx(renderer.slope_angle(position))
    marker = Image.new("RGBA", (112, 112))
    marker.putpixel((56, 56), (255, 0, 255, 255))
    with patch(
        "POMDPPlanners.environments.mountain_car_pomdp.mountain_car_visualizer._car",
        return_value=marker,
    ):
        anchored = renderer.render_frame(row(position), 0, 1)
    assert anchored.getpixel((round(x), round(y))) == (255, 0, 255)


def test_goal_and_background_cache_follow_live_bounds():
    env = MountainCarPOMDP(0.95)
    renderer = MountainCarVisualizer(env)
    with patch.object(renderer, "_build_background", wraps=renderer._build_background) as build:
        renderer.render_frame(row(), 0, 1)
        renderer.render_frame(row(-0.4), 0, 1)
        assert build.call_count == 1
        env.goal_position = 0.4
        renderer.render_frame(row(), 0, 1)
        assert build.call_count == 2
    gx, gy = renderer.point(env.goal_position)
    background = np.asarray(renderer._background)
    assert tuple(background[round(gy) - 35, round(gx) + 5, :3]) == (70, 160, 41)


@pytest.mark.parametrize(
    "action,label", [(-1, "-1 LEFT"), (0, "0 NEUTRAL"), (1, "+1 RIGHT"), (None, "none")]
)
def test_selected_action_not_velocity_and_recorded_observation(action, label):
    renderer = MountainCarVisualizer(MountainCarPOMDP(0.95))
    with patch.object(renderer, "_text", wraps=renderer._text) as text:
        renderer.render_frame(row(-0.4, -0.03, action), 0, 1)
    lines = [call.args[2] for call in text.call_args_list]
    assert f"Selected action {label}" in lines[1]
    assert "-0.0300" in lines[1]
    assert "END OF RECORDING" in lines[0]
    assert ("[-0.200, +0.010]" in lines[2]) if action is not None else ("none" in lines[2])


@pytest.mark.parametrize("position,at_goal", [(0.49, False), (0.5, True), (0.6, True)])
def test_goal_status_uses_position_even_with_negative_velocity(position, at_goal):
    renderer = MountainCarVisualizer(MountainCarPOMDP(0.95))
    with patch.object(renderer, "_text", wraps=renderer._text) as text:
        renderer.render_frame(row(position, -0.02, None), 0, 1)
    assert ("GOAL REACHED" in text.call_args_list[0].args[2]) == at_goal


def test_hook_gif_contract_determinism_and_cache(tmp_path):
    env = MountainCarPOMDP(0.95)
    history = [row(-1.2, 0, -1), row(-0.5, 0, 0), row(0.5, 0, None)]
    env.cache_visualization(history, tmp_path / "nested", 42)
    first = tmp_path / "nested" / "agent_path_42.gif"
    renderer = env._episode_visualizer
    env.cache_visualization(history, tmp_path, 43)
    assert env._episode_visualizer is renderer
    assert first.read_bytes() == (tmp_path / "agent_path_43.gif").read_bytes()
    with Image.open(first) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.size == (800, 500) and gif.n_frames == 3 and gif.info["loop"] == 0
        for i in range(3):
            gif.seek(i)
            assert gif.info["duration"] == 500
    assert _car(20.0) is _car(20.0)
    alpha = np.asarray(_asset("car.png"))[..., 3]
    assert (alpha == 0).any() and (alpha == 255).any()
    bbox = _car(0).getbbox()
    assert bbox is not None and bbox[3] <= 56


def test_state_rng_and_cached_art_unchanged(tmp_path):
    env = MountainCarPOMDP(0.95)
    renderer = MountainCarVisualizer(env)
    particle = np.array([-0.45, 0.02])
    history = [row(belief=SimpleNamespace(particles=[particle]))]
    state = history[0].state.copy()
    rng = cast(tuple, np.random.get_state())
    art = _asset("car.png").tobytes()
    with patch.object(env, "sample_next_step", side_effect=AssertionError("must not step")):
        renderer.save(history, tmp_path / "a.gif")
    after = cast(tuple, np.random.get_state())
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_array_equal(rng[1], after[1])
    np.testing.assert_array_equal(state, history[0].state)
    np.testing.assert_array_equal(particle, [-0.45, 0.02])
    assert art == _asset("car.png").tobytes()


def test_validation_and_small_canvas(tmp_path):
    renderer = MountainCarVisualizer(MountainCarPOMDP(0.95), 640, 400)
    renderer.save([row(), row(0.5, 0, None)], tmp_path / "small.gif")
    with pytest.raises(ValueError, match="empty"):
        renderer.save([], tmp_path / "empty.gif")
    with pytest.raises(TypeError, match="Path"):
        renderer.save([row()], "file.gif")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        renderer.save([row(float("nan"))], tmp_path / "nan.gif")
    with pytest.raises(ValueError, match="action"):
        renderer.save([row(action=2)], tmp_path / "bad.gif")
    with pytest.raises(ValueError, match="640"):
        MountainCarVisualizer(renderer.env, 100, 100)


def test_belief_weights_gaussian_and_missing_are_distinct():
    renderer = MountainCarVisualizer(MountainCarPOMDP(0.95))
    particles = np.array([[-0.9, 0.0], [0.1, 0.0]])
    heavy_left = SimpleNamespace(particles=particles, log_weights=np.log([0.999, 0.001]))
    heavy_right = SimpleNamespace(particles=particles, log_weights=np.log([0.001, 0.999]))
    images = [renderer.render_frame(row(belief=b), 0, 1) for b in (heavy_left, heavy_right)]
    assert images[0].tobytes() != images[1].tobytes()
    gaussian = SimpleNamespace(mean=np.array([-0.5, 0]), covariance=np.diag([0.04, 0.01]))
    with patch("numpy.random.multivariate_normal", side_effect=AssertionError("no sampling")):
        image = renderer.render_frame(row(belief=gaussian), 0, 1)
    assert image.tobytes() != renderer.render_frame(row(), 0, 1).tobytes()
    np.testing.assert_array_equal(gaussian.mean, [-0.5, 0])


def test_render_cache_is_excluded_from_pickle(tmp_path):
    env = MountainCarPOMDP(0.95)
    before = len(pickle.dumps(env))
    env.cache_visualization([row()], tmp_path, 0)
    encoded = pickle.dumps(env)
    assert len(encoded) == before
    restored = pickle.loads(encoded)
    assert not hasattr(restored, "_episode_visualizer")
    restored.cache_visualization([row()], tmp_path, 1)
    assert (tmp_path / "agent_path_0.gif").read_bytes() == (
        tmp_path / "agent_path_1.gif"
    ).read_bytes()


@pytest.mark.skipif(not Path("/.dockerenv").exists(), reason="CI Docker image only")
def test_fixed_docker_golden(tmp_path):
    belief = SimpleNamespace(
        particles=np.array([[-0.9, 0.0], [-0.4, 0.0]]), log_weights=np.log([0.1, 0.9])
    )
    history = [row(-1.2, action=-1, belief=belief), row(-0.5, action=0), row(0.5, action=None)]
    path = tmp_path / "mountain.gif"
    MountainCarVisualizer(MountainCarPOMDP(0.95)).save(history, path)
    golden = Path(__file__).with_name("mountain_car_golden.sha256").read_text().strip()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == golden
