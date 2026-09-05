# SPDX-License-Identifier: MIT
"""Contracts for the state-driven PacMan artwork exporter."""

import hashlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import GifImagePlugin, Image, ImageDraw

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_visualizer import PacManVisualizer
from POMDPPlanners.environments.pacman_pomdp.pacman_art import character, pellet, tile
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import pacman_pinned_kwargs


def case(shape=(5, 5), ghosts=((4, 4),)):
    env = PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=shape,
            walls={(1, 1)},
            initial_pacman_pos=(0, 0),
            num_ghosts=len(ghosts),
            initial_ghost_positions=list(ghosts),
            initial_pellets=[(2, 1)],
        ),
    )
    state = env.make_state(
        pacman_pos=(0, 0), ghost_positions=ghosts, pellets=((2, 1),), score=0.0, terminal=False
    )
    return env, state


def test_one_sprite_for_each_actual_ghost():
    env, state = case(ghosts=((3, 3), (4, 4)))
    renderer = PacManVisualizer(env, 32)
    canvas = Image.new("RGBA", (160, 240))
    with patch.object(canvas, "alpha_composite", wraps=canvas.alpha_composite) as paste:
        renderer._draw_ghosts(state, canvas, renderer.sprites, 32)
    assert paste.call_count == 2
    assert [call.args[1] for call in paste.call_args_list] == [(96, 96), (128, 128)]
    ghost = character("ghost", 96)
    bbox = ghost.getbbox()
    assert bbox is not None
    assert bbox[2] - bbox[0] > 48
    assert np.asarray(ghost)[0, 0, 3] == 0


def test_tiles_preserve_row_column_geometry_and_cache():
    env, state = case()
    renderer = PacManVisualizer(env, 32)
    with patch.object(renderer, "_build_background", wraps=renderer._build_background) as build:
        first = renderer._generate_frames([state, state], [1, 2], renderer.sprites, 32)
        second = renderer._generate_frames([state], [1], renderer.sprites, 32)
    assert build.call_count == 1
    assert first[0] is not first[1] and first[0] is not second[0]
    assert renderer._background is not None
    assert renderer._background.crop((32, 32, 64, 64)).tobytes() == tile(32, True).tobytes()
    assert renderer._background.crop((64, 32, 96, 64)).tobytes() == tile(32, False).tobytes()
    env.walls = {(1, 2)}
    renderer._render_frame(state, 1, "east", renderer.sprites, 32)
    assert renderer._background.crop((64, 32, 96, 64)).tobytes() == tile(32, True).tobytes()
    assert character("ghost", 32) is character("ghost", 32)
    assert pellet(32) is pellet(32)


def test_ghost_identity_and_exact_score():
    env, state = case(ghosts=((3, 3), (4, 4)))
    renderer = PacManVisualizer(env, 96)
    assert renderer.sprites["ghost_0"].tobytes() != renderer.sprites["ghost_1"].tobytes()
    assert (
        renderer.sprites["ghost_0"].getchannel("A").tobytes()
        == renderer.sprites["ghost_1"].getchannel("A").tobytes()
    )
    state = env.make_state(
        pacman_pos=(0, 0),
        ghost_positions=((3, 3), (4, 4)),
        pellets=((2, 1),),
        score=1234567.0,
        terminal=False,
    )
    draw = ImageDraw.Draw(Image.new("RGBA", (480, 560)))
    with patch.object(draw, "text", wraps=draw.text) as text:
        renderer._draw_text_overlay(state, draw, 1, "east", 96)
    assert "Score: 1234567" in text.call_args_list[1].args[1]


def test_heatmap_relative_mass_and_entities_draw_above_overlays():
    env, state = case()
    second = env.make_state(
        pacman_pos=(0, 0), ghost_positions=((2, 1),), pellets=((2, 1),), score=0, terminal=False
    )
    belief = SimpleNamespace(particles=[state, second], normalized_weights=np.array([0.75, 0.25]))
    renderer = PacManVisualizer(env, 32)
    canvas = Image.new("RGBA", (160, 240))
    renderer._draw_ghost_belief(belief, canvas, 32)  # type: ignore[arg-type]
    assert np.asarray(canvas)[144, 144, 3] == 180
    assert np.asarray(canvas)[80, 48, 3] == 60
    clear = renderer._render_frame(state, 1, "east", renderer.sprites, 32)
    overlay = renderer._render_frame(state, 1, "east", renderer.sprites, 32, belief)  # type: ignore[arg-type]
    for x, y in ((144, 144), (48, 80), (16, 16)):
        assert clear.getpixel((x, y)) == overlay.getpixel((x, y))


def test_hazard_does_not_paint_status_panel():
    env, _ = case()
    env.dangerous_areas = [(4, 4)]
    env.dangerous_area_radius = 2.0
    renderer = PacManVisualizer(env, 16)
    canvas = Image.new("RGBA", (80, 160))
    renderer._draw_dangerous_areas(canvas, 16)
    assert np.asarray(canvas)[72, 72, 3] > 0
    assert canvas.crop((0, 80, 80, 160)).getbbox() is None


def test_one_pixel_collision_tile_does_not_invert_outline(tmp_path):
    env, _ = case()
    state = env.make_state(
        pacman_pos=(0, 0), ghost_positions=((0, 0),), pellets=((2, 1),), score=0.0, terminal=True
    )
    renderer = PacManVisualizer(env, 1)
    renderer.visualize_path([state], [1], tmp_path / "tiny.gif")
    with Image.open(tmp_path / "tiny.gif") as gif:
        assert gif.size == (5, 85)


@pytest.mark.parametrize("size", [8, 16, 32, 96])
def test_gif_contract_determinism_and_truncated_last_action(tmp_path, size):
    env, state = case()
    renderer = PacManVisualizer(env, size)
    history = [StepData(state, 1, state, None, 0.0, None), StepData(state, 2, state, None, 0.0, None)]  # type: ignore[arg-type]
    first = tmp_path / "nested" / "first.gif"
    with patch.object(renderer, "_render_frame", wraps=renderer._render_frame) as render:
        renderer.cache_visualization(history, first)
    assert render.call_args_list[-1].args[2] == "south"
    second = tmp_path / "second.gif"
    renderer.cache_visualization(history, second)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    with Image.open(first) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.size == (5 * size, 5 * size + 80)
        assert gif.n_frames == 2
        for index in range(gif.n_frames):
            gif.seek(index)
            assert gif.info["duration"] == 1000


@pytest.mark.parametrize("win", [True, False])
def test_narrow_terminal_banner_is_in_panel(win):
    env, _ = case((4, 2), ((3, 1),))
    state = env.make_state(
        pacman_pos=(0, 0),
        ghost_positions=((3, 1),),
        pellets=() if win else ((2, 1),),
        score=0,
        terminal=True,
    )
    renderer = PacManVisualizer(env, 32)
    canvas = Image.new("RGBA", (64, 208))
    draw = ImageDraw.Draw(canvas)
    with patch.object(draw, "text", wraps=draw.text) as text:
        renderer._draw_text_overlay(state, draw, 3, "Terminal", 32)
    assert text.call_args_list[-1].args[1] == ("WIN" if win else "OVER")
    for call in text.call_args_list:
        bbox = draw.textbbox(call.args[0], call.args[1], font=call.kwargs["font"])
        assert 0 <= bbox[0] < bbox[2] <= 64
        assert 128 <= bbox[1] < bbox[3] <= 208


def test_validation_and_no_state_rng_mutation(tmp_path):
    env, state = case()
    renderer = PacManVisualizer(env)
    with pytest.raises(ValueError, match="empty"):
        renderer.visualize_path([], [], tmp_path / "empty.gif")
    with pytest.raises(ValueError, match="empty"):
        renderer.cache_visualization([], tmp_path / "empty.gif")
    with pytest.raises(TypeError, match="Path"):
        renderer.visualize_path([state], [1], "x.gif")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        PacManVisualizer(env, 0)
    before = state.copy()
    rng = cast(tuple, np.random.get_state())
    renderer.visualize_path([state], [1], tmp_path / "episode.gif")
    np.testing.assert_array_equal(state, before)
    after = cast(tuple, np.random.get_state())
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_array_equal(rng[1], after[1])
