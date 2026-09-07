# SPDX-License-Identifier: MIT

"""Geometry, recorded phases and deterministic saving for CartPole."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import GifImagePlugin, Image, ImageDraw

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp import cartpole_visualizer as viz


@pytest.fixture
def env():
    return CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)


def _step(state=None, action=1, observation=None, belief=None):
    state = np.array([0.0, 0.2, 0.0, -0.1]) if state is None else np.asarray(state, dtype=float)
    return StepData(
        state=state,
        action=action,
        next_state=None if action is None else state.copy(),
        observation=observation,
        reward=None if action is None else 1.0,
        belief=belief,
    )


@pytest.mark.parametrize("angle", [0.0, 0.2, -0.2, np.pi / 2, -np.pi / 2])
def test_exact_pole_geometry(angle):
    pivot, tip = viz.pole_geometry([1.2, -0.4, angle, 0.1], scale=100.0, half_length=0.5)
    assert pivot == (520.0, viz.PIVOT_Y)
    assert tip == pytest.approx((520 + 100 * np.sin(angle), viz.PIVOT_Y - 100 * np.cos(angle)))
    assert np.linalg.norm(np.subtract(tip, pivot)) == pytest.approx(100.0)


def test_uses_full_environment_pole_length():
    pivot, tip = viz.pole_geometry([0, 0, 0, 0], scale=50, half_length=1.7)
    assert pivot[1] - tip[1] == pytest.approx(170.0)


@pytest.mark.parametrize("x,angle", [(-2.4, -0.2), (0.0, 0.0), (2.4, 0.2)])
def test_cart_contact_does_not_rotate_with_pole(env, monkeypatch, x, angle):
    placements = []
    original = Image.Image.paste

    def capture(canvas, sprite, box=None, mask=None):
        if isinstance(sprite, Image.Image) and sprite.mode == "RGBA":
            placements.append((sprite.size, box))
        return original(canvas, sprite, box, mask)

    monkeypatch.setattr(Image.Image, "paste", capture)
    viz.CartPoleVisualizer(env).render_frames([_step([x, 0, angle, 0])])
    size, box = placements[-1]
    assert box[1] == round(viz.PIVOT_Y)
    assert box[1] + size[1] == viz.RAIL_Y
    assert 0 <= box[0] and box[0] + size[0] <= viz.CANVAS_SIZE[0]


def _text_spy(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text

    def capture(draw, xy, text, *args, **kwargs):
        texts.append(str(text))
        return original(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    return texts


@pytest.mark.parametrize("action,velocity,direction", [(0, 2.0, -1), (1, -2.0, 1)])
def test_force_direction_is_not_velocity(env, monkeypatch, action, velocity, direction):
    calls = []
    monkeypatch.setattr(viz, "_arrow", lambda draw, x, y, sign: calls.append(sign))
    viz.CartPoleVisualizer(env).render_frames([_step([0, velocity, 0, 0], action)])
    assert calls == [direction]


@pytest.mark.parametrize(
    "x,angle,outside",
    [
        (2.4, 0, False),
        (2.40001, 0, True),
        (-2.40001, 0, True),
        (0, 1.0, False),
        (0, 1.000001, True),
        (0, -1.000001, True),
    ],
)
def test_thresholds_match_environment(env, monkeypatch, x, angle, outside):
    texts = _text_spy(monkeypatch)
    state = [x, 0, angle * env.theta_threshold_radians, 0]
    viz.CartPoleVisualizer(env).render_frames([_step(state)])
    assert env.is_terminal(np.asarray(state)) == outside
    assert ("OUTSIDE LIMITS" if outside else "WITHIN LIMITS") in texts


def test_terminal_and_truncated_phases(env, monkeypatch):
    texts = _text_spy(monkeypatch)
    calls = []
    monkeypatch.setattr(viz, "_arrow", lambda draw, x, y, sign: calls.append(sign))
    rows = [_step(action=0, observation=[9, 8, 7, 6]), _step(action=None)]
    viz.CartPoleVisualizer(env).render_frames(rows)
    assert calls == [-1]
    assert any("9." in text and "8." in text and "7." in text for text in texts)
    assert "Recorded reward: --" in texts
    assert any("bookkeeping" in text for text in texts)
    texts.clear()
    calls.clear()
    viz.CartPoleVisualizer(env).render_frames([_step(action=0), _step(action=1)])
    assert calls == [-1, 1]
    assert any("1 | RIGHT" in text for text in texts)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_gif_metadata_repeated_states_and_determinism(env, tmp_path, count):
    history = [_step() for _ in range(count)]
    env.cache_visualization(history, tmp_path / "nested", 0)
    env.cache_visualization(history, tmp_path / "nested", 1)
    first = tmp_path / "nested/agent_path_0.gif"
    assert first.read_bytes() == (tmp_path / "nested/agent_path_1.gif").read_bytes()
    with Image.open(first) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.size == viz.CANVAS_SIZE
        assert gif.n_frames == count
        assert gif.info["loop"] == 0
        for index in range(count):
            gif.seek(index)
            assert gif.info["duration"] == 500


@pytest.mark.parametrize("bad", [[], None, (), [None]])
def test_invalid_histories_leave_no_output(env, tmp_path, bad):
    with pytest.raises((TypeError, ValueError)):
        env.cache_visualization(bad, tmp_path / "not-created", 0)
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("state", [[1, 2], [0, np.nan, 0, 0], [0, 0, np.inf, 0]])
def test_invalid_states(env, state):
    with pytest.raises(ValueError, match="four finite"):
        viz.CartPoleVisualizer(env).render_frames([_step(state)])


@pytest.mark.parametrize("action", [-1, 2, "left", [1]])
def test_invalid_actions(env, action):
    with pytest.raises(ValueError, match="action"):
        viz.CartPoleVisualizer(env).render_frames([_step(action=action)])


def test_does_not_step_sample_or_change_rng(env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer must not sample or step")

    monkeypatch.setattr(env, "sample_next_step", forbidden)
    monkeypatch.setattr(env, "is_terminal", forbidden)
    belief = SimpleNamespace(particles=np.zeros((20, 4)), sample=forbidden)
    old = np.random.get_state()
    viz.CartPoleVisualizer(env).render_frames([_step(belief=belief)])
    new = np.random.get_state()
    assert old[0] == new[0] and np.array_equal(old[1], new[1]) and old[2:] == new[2:]


def test_assets_and_background_cached_across_instances(env, tmp_path):
    viz._asset.cache_clear()
    viz._background.cache_clear()
    env.cache_visualization([_step()], tmp_path, 0)
    misses = viz._asset.cache_info().misses
    background_misses = viz._background.cache_info().misses
    env.cache_visualization([_step()], tmp_path, 1)
    assert viz._asset.cache_info().misses == misses == 2
    assert viz._background.cache_info().misses == background_misses == 1
    cart = viz._asset("cart", (64, 40))
    assert cart.mode == "RGBA"
    low, high = cart.getchannel("A").getextrema()
    assert low == 0 and high >= 250


def test_path_validation(env, tmp_path):
    renderer = viz.CartPoleVisualizer(env)
    with pytest.raises(TypeError):
        renderer.save([_step()], "wrong.gif")
    with pytest.raises(ValueError):
        renderer.save([_step()], tmp_path / "wrong.png")


def test_actual_docker_repeatability(env, tmp_path):
    if not Path("/.dockerenv").exists():
        pytest.skip("CI-image repeatability check")
    rows = [
        _step([0, 0, 0, 0], 0),
        _step([0.1, -0.1, 0.15, 0.2], 1),
        _step([0.2, 0.1, -0.22, 0.3], None),
    ]
    env.cache_visualization(rows, tmp_path, 0)
    viz._asset.cache_clear()
    viz._background.cache_clear()
    env.cache_visualization(rows, tmp_path, 1)
    assert (tmp_path / "agent_path_0.gif").read_bytes() == (
        tmp_path / "agent_path_1.gif"
    ).read_bytes()
    golden = Path(__file__).parent / "golden_visualizations/cartpole_visualization.gif"
    assert (tmp_path / "agent_path_0.gif").read_bytes() == golden.read_bytes()
