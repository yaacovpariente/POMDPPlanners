# SPDX-License-Identifier: MIT

"""Behavior tests for the cached Pillow RockSample renderer."""

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import GifImagePlugin, Image, ImageChops
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    create_rock_sample_state,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (
    CANVAS_SIZE,
    COLOR_ARROW,
    COLOR_BAD_ROCK,
    COLOR_DANGER,
    COLOR_EXIT,
    COLOR_FAILURE,
    COLOR_GOOD_ROCK,
    COLOR_ROBOT,
    COLOR_SENSOR,
    COLOR_SUCCESS,
    GIF_DURATION_MS,
    RockSampleVisualizer,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import rock_sample_pinned_kwargs


def _fixed_case():
    env = RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(
            map_size=(4, 5),
            rock_positions=[(0, 1), (2, 3)],
            dangerous_areas=[(1, 1)],
            dangerous_area_radius=0.75,
            init_pos=(0, 0),
        ),
    )
    states = [
        create_rock_sample_state((0, 0), (True, False)),
        create_rock_sample_state((0, 1), (True, False)),
        create_rock_sample_state((0, 1), (True, False)),
        create_rock_sample_state((0, 1), (False, False)),
        create_rock_sample_state((0, 2), (False, False)),
        create_rock_sample_state((-1, -1), (False, False)),
    ]
    actions = [2, 5, 0, 0, 2]
    return env, states, actions


def _color_count(image: Image.Image, color) -> int:
    colors = image.getcolors(maxcolors=CANVAS_SIZE[0] * CANVAS_SIZE[1])
    assert colors is not None
    return sum(count for count, found in colors if found == color)


class TestVisualization:
    """Pin the public contract and every RockSample-specific visual cue."""

    def test_visualize_path_parameter_validation(self):
        env, states, actions = _fixed_case()

        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            env.visualize_path(states, actions, "invalid_path")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            env.visualize_path(states, actions, Path("test.png"))

    def test_visualize_path_rejects_empty_path(self, tmp_path):
        env, _, _ = _fixed_case()
        output = tmp_path / "nested" / "empty.gif"

        with pytest.raises(ValueError, match="Cannot visualize empty path"):
            env.visualize_path([], [], output)

        assert not output.parent.exists()

    def test_danger_areas_are_clipped_to_grid(self):
        env, _, _ = _fixed_case()
        env.dangerous_areas = [(0, 0), (3, 4)]
        env.dangerous_area_radius = 1.0
        visualizer = RockSampleVisualizer(env)
        background = visualizer._build_static_background()
        left, top, right, bottom = visualizer._plot_bounds

        # Exclude the legend, whose danger swatch deliberately uses this color.
        assert _color_count(background.crop((0, 0, right + 20, top)), COLOR_DANGER) == 0
        assert _color_count(background.crop((0, top, left, bottom)), COLOR_DANGER) == 0
        assert (
            _color_count(background.crop((right + 1, top, right + 20, bottom)), COLOR_DANGER) == 0
        )
        assert _color_count(background.crop((0, bottom + 1, right + 20, 800)), COLOR_DANGER) == 0
        env.dangerous_areas = []
        clear = RockSampleVisualizer(env)._build_static_background()
        assert background.getpixel((left + 5, top + 5)) != clear.getpixel((left + 5, top + 5))
        assert background.getpixel((right - 5, bottom - 5)) != clear.getpixel(
            (right - 5, bottom - 5)
        )
        for box in (
            (0, 0, right + 20, top),
            (0, top, left, bottom),
            (right + 1, top, right + 20, bottom),
            (0, bottom + 1, right + 20, 800),
        ):
            assert ImageChops.difference(background.crop(box), clear.crop(box)).getbbox() is None

    def test_cache_visualization_empty_history(self, tmp_path):
        env, _, _ = _fixed_case()
        empty_history = History(
            history=[],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=0,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        with pytest.raises(ValueError, match="Cannot visualize empty history"):
            env.cache_visualization(empty_history.history, tmp_path, 0)

    def test_cache_visualization_writes_expected_path(self, tmp_path):
        env, states, actions = _fixed_case()
        history = [
            StepData(
                state=state,
                action=actions[index] if index < len(actions) else None,
                next_state=states[min(index + 1, len(states) - 1)],
                observation="none",
                reward=0.0,
                belief=Mock(spec=Belief),
            )
            for index, state in enumerate(states)
        ]

        output_dir = tmp_path / "nested"
        env.cache_visualization(history, output_dir, 7)

        assert (output_dir / "agent_path_7.gif").is_file()

    def test_gif_keeps_frame_count_duration_and_dimensions(self, tmp_path):
        env, states, actions = _fixed_case()
        output = tmp_path / "episode.gif"

        RockSampleVisualizer(env).visualize_path(states, actions, output)

        with Image.open(output) as gif:
            assert isinstance(gif, GifImagePlugin.GifImageFile)
            assert gif.size == CANVAS_SIZE
            assert gif.n_frames == len(states)
            durations = []
            for frame in range(gif.n_frames):
                gif.seek(frame)
                durations.append(gif.info["duration"])
        assert durations == [GIF_DURATION_MS] * len(states)

    def test_fixed_history_has_deterministic_bytes(self, tmp_path):
        env, states, actions = _fixed_case()
        first = tmp_path / "first.gif"
        second = tmp_path / "second.gif"

        visualizer = RockSampleVisualizer(env)
        visualizer.visualize_path(states, actions, first)
        visualizer.visualize_path(states, actions, second)

        first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
        second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
        assert first_hash == second_hash

    def test_static_background_is_built_once_and_copied(self):
        env, states, actions = _fixed_case()
        visualizer = RockSampleVisualizer(env)

        with patch.object(
            visualizer,
            "_build_static_background",
            wraps=visualizer._build_static_background,
        ) as build:
            first = visualizer.render_frames(states, actions)
            second = visualizer.render_frames(states, actions)

        build.assert_called_once_with()
        assert first[0] is not first[1]
        assert first[0] is not second[0]

    def test_frames_preserve_scene_elements_and_events(self):
        env, states, actions = _fixed_case()
        visualizer = RockSampleVisualizer(env)

        frames = visualizer.render_frames(states, actions)
        assert visualizer._background is not None
        background = visualizer._background
        left, top, right, bottom = visualizer._plot_bounds
        danger_x, danger_y = visualizer._cell_center(1, 1)
        good_x, good_y = visualizer._cell_center(0, 1)
        bad_x, bad_y = visualizer._cell_center(2, 3)
        _, row_zero_y = visualizer._cell_center(0, 0)
        _, row_two_y = visualizer._cell_center(2, 0)

        assert row_zero_y < row_two_y
        env.dangerous_areas = []
        clear = RockSampleVisualizer(env)._build_static_background()
        assert background.getpixel((round(danger_x + 5), round(danger_y + 5))) != clear.getpixel(
            (round(danger_x + 5), round(danger_y + 5))
        )
        assert background.getpixel((right, (top + bottom) // 2)) == COLOR_EXIT
        badge_offset = visualizer._pixels_per_cell * 0.28
        rock_edge = (round(good_x), int(good_y + badge_offset))
        assert frames[2].getpixel(rock_edge) == COLOR_GOOD_ROCK
        assert frames[3].getpixel(rock_edge) == COLOR_BAD_ROCK
        assert frames[0].getpixel((round(bad_x), int(bad_y + badge_offset))) == COLOR_BAD_ROCK

        assert _color_count(frames[0], COLOR_ARROW) > _color_count(background, COLOR_ARROW)
        assert _color_count(frames[1], COLOR_SENSOR) > _color_count(background, COLOR_SENSOR)
        assert _color_count(frames[2], COLOR_SUCCESS) > 0
        assert _color_count(frames[3], COLOR_FAILURE) > 0
        x, y = visualizer._cell_center(0, 2)
        robot_box = (round(x - 20), round(y - 20), round(x + 20), round(y + 20))
        assert ImageChops.difference(frames[4].crop(robot_box), frames[5].crop(robot_box)).getbbox()
        terminal = visualizer.render_frames([states[5]], [])[0]
        assert (
            ImageChops.difference(terminal.crop(robot_box), background.crop(robot_box)).getbbox()
            is None
        )

    def test_packaged_sprites_have_transparent_edges_and_cache(self):
        from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_assets import sprite, terrain

        for name in ("rover", "ore-blue", "ore-green", "ore-red"):
            art = sprite(name, 80)
            assert art.mode == "RGBA"
            assert art.getchannel("A").getextrema() == (0, 255)
            assert sprite(name, 80) is art
        assert terrain(320, 240) is terrain(320, 240)

    def test_badges_stay_inside_dense_map_cells(self):
        env, _, _ = _fixed_case()
        env.map_size = (11, 11)
        visualizer = RockSampleVisualizer(env)
        state = create_rock_sample_state((-1, -1), (True, False))
        frame = Image.new("RGB", CANVAS_SIZE)
        from PIL import ImageDraw

        visualizer._draw_rock_badges(ImageDraw.Draw(frame), state)
        cell = visualizer._pixels_per_cell
        for row, col in env.rock_positions:
            x, y = visualizer._cell_center(row, col)
            outside = frame.crop(
                (round(x - cell / 2), round(y + cell / 2) + 1, round(x + cell / 2), round(y + cell))
            )
            assert outside.getbbox() is None
