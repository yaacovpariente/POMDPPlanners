# SPDX-License-Identifier: MIT

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from POMDPPlanners.core.simulation.history import StepData
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import RacetrackPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualizer import RacetrackVisualizer


def _state(x: float, opponent: bool = False) -> np.ndarray:
    state = np.zeros(12, dtype=float)
    state[:4] = [x, x / 2.0, 0.25, 8.0]
    if opponent:
        state[7:12] = [1.0, 5.0, 2.0, 0.0, 0.0]
    return state


def _step(state: np.ndarray, next_state: np.ndarray | None, action: int | None = 0) -> StepData:
    return StepData(state, action, next_state, None, None, None)  # type: ignore[arg-type]


def test_empty_history_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one recorded step"):
        RacetrackVisualizer(1).save([], tmp_path / "empty.gif")


def test_terminal_bookkeeping_keeps_final_state_once(tmp_path):
    start, terminal = _state(0.0), _state(3.0, opponent=True)
    history = [_step(start, terminal), _step(terminal, None, action=None)]
    output = tmp_path / "terminal.gif"

    RacetrackVisualizer(1).save(history, output)

    with Image.open(output) as decoded:
        assert getattr(decoded, "n_frames") == 2


def test_normal_history_adds_last_successor_and_can_save_still(tmp_path):
    first, second, final = _state(0.0), _state(2.0), _state(5.0, opponent=True)
    history = [_step(first, second), _step(second, final)]
    visualizer = RacetrackVisualizer(1)
    animation = tmp_path / "episode.gif"
    still = tmp_path / "episode.png"

    visualizer.save(history, animation)
    visualizer.save(history, still)

    with Image.open(animation) as decoded:
        assert getattr(decoded, "n_frames") == 3
        decoded.seek(2)
        final_animation = decoded.convert("RGB")
    with Image.open(still) as decoded_still:
        assert decoded_still.size == (1100, 760)
        assert decoded_still.convert("RGB").tobytes() == final_animation.tobytes()


def test_package_hook_uses_history_without_live_calls(tmp_path):
    env = RacetrackPOMDP(discount_factor=0.95, max_tracked_agents=1)
    env.render_frame = Mock(side_effect=AssertionError("live renderer called"))  # type: ignore[method-assign]
    state = _state(1.0)

    env.cache_visualization([_step(state, state)], tmp_path, 4)

    with Image.open(tmp_path / "agent_path_4.gif") as decoded:
        assert getattr(decoded, "n_frames") == 2
    env.render_frame.assert_not_called()


def test_live_render_frame_contract_is_preserved():
    env = RacetrackPOMDP(discount_factor=0.95, max_tracked_agents=1, render_mode="rgb_array")
    assert env.render_frame() is None

    expected = np.full((3, 4, 3), 17, dtype=np.uint8)
    session = Mock()
    session.render_frame.return_value = expected
    env._session = session  # pylint: disable=protected-access

    assert env.render_frame() is expected
    session.render_frame.assert_called_once_with()


def test_reference_map_matches_simulator_edges():
    """Check the displayed road against world geometry, not the planner model."""
    pytest.importorskip("highway_env")
    from highway_env.envs.racetrack_env import RacetrackEnv
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualizer_track import (
        reference_track_lanes,
    )

    world = RacetrackEnv()
    try:
        lanes = [
            lane
            for ends in world.road.network.graph.values()
            for parts in ends.values()
            for lane in parts
        ]
        rendered = reference_track_lanes()
        assert len(lanes) == len(rendered) == 18
        for lane, painted in zip(lanes, rendered):
            for side, edge in enumerate(painted["lines"]):
                distances = np.linspace(0, lane.length, len(edge["points"]))
                expected = np.array(
                    [lane.position(s, (side - 0.5) * lane.width_at(s)) for s in distances]
                )
                np.testing.assert_allclose(edge["points"], expected, atol=1e-12)
                assert edge["type"] == lane.line_types[side]
            np.testing.assert_array_equal(
                painted["polygon"],
                np.vstack([painted["lines"][0]["points"], painted["lines"][1]["points"][::-1]]),
            )
    finally:
        world.close()


def test_particle_projection_uses_each_particle_pose_and_preserves_weights():
    from POMDPPlanners.core.belief import WeightedParticleBelief

    first, second = _state(0, True), _state(10, True)
    first[:3] = [1, 2, 0]
    second[:3] = [10, 20, np.pi / 2]
    belief = WeightedParticleBelief([first, second], np.log([0.2, 0.8]))
    points, weights = RacetrackVisualizer(1)._belief_points(belief)
    np.testing.assert_allclose(points, [[1, 2], [6, 4], [10, 20], [8, 25]])
    np.testing.assert_allclose(weights, [0.2, 0.2, 0.8, 0.8])
    np.testing.assert_allclose(belief.normalized_weights, [0.2, 0.8])


def test_frames_preserve_poses_and_never_sample_belief():
    from unittest.mock import patch
    from POMDPPlanners.core.belief import WeightedParticleBelief

    first, final = _state(0), _state(4)
    belief = WeightedParticleBelief([first.copy(), final.copy()], np.log([0.3, 0.7]))
    belief.sample = Mock(side_effect=AssertionError("renderer sampled belief"))
    history = [StepData(first, 1, final, None, 0.25, belief)]
    original = np.stack([first.copy(), final.copy()])
    visualizer = RacetrackVisualizer(1)
    with patch.object(visualizer, "_car", wraps=visualizer._car) as car:
        frames = visualizer.render_frames(history)
    assert len(frames) == 2
    for call, state in zip(car.call_args_list, [first, final]):
        np.testing.assert_array_equal(call.args[1], state[:2])
        assert call.args[2] == state[2]
    np.testing.assert_array_equal(np.stack([first, final]), original)
    belief.sample.assert_not_called()


def test_gif_is_deterministic_with_readable_timing(tmp_path):
    import hashlib

    history = [_step(_state(0), _state(0)), _step(_state(0), _state(0))]
    visualizer = RacetrackVisualizer(1)
    one, two = tmp_path / "one.gif", tmp_path / "two.gif"
    visualizer.save(history, one)
    visualizer.save(history, two)
    assert hashlib.sha256(one.read_bytes()).digest() == hashlib.sha256(two.read_bytes()).digest()
    with Image.open(one) as gif:
        assert getattr(gif, "n_frames") == 3
        durations = []
        for index in range(3):
            gif.seek(index)
            durations.append(gif.info["duration"])
        assert durations == [500, 500, 1500]


def test_missing_map_and_terminal_metadata_are_explicit():
    from unittest.mock import patch

    history = [_step(_state(0), None, None)]
    visualizer = RacetrackVisualizer(1)
    with patch.object(visualizer, "_text", wraps=visualizer._text) as text:
        visualizer.render_frames(history)
    strings = [call.args[2] for call in text.call_args_list]
    assert "Road geometry unavailable" in strings
    assert "belief unavailable" in strings
    assert "Final recorded state" in strings
    assert any("Reward unavailable" in value for value in strings)


def test_history_gap_is_not_drawn_as_movement():
    from PIL import ImageColor
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualizer import TRAIL

    first, successor, disconnected = _state(0), _state(1), _state(10)
    history = [_step(first, successor), _step(disconnected, None, None)]
    visualizer = RacetrackVisualizer(1)
    frames = visualizer.render_frames(history)
    bounds = visualizer._bounds([first, disconnected])
    point = visualizer._project(5, 2.5, bounds)
    assert frames[-1].getpixel((round(point[0]), round(point[1]))) != ImageColor.getrgb(TRAIL)


def test_custom_scenario_omits_reference_map(tmp_path):
    from unittest.mock import patch

    env = RacetrackPOMDP(discount_factor=0.95, env_id="custom-track")
    with patch.object(RacetrackVisualizer, "save") as save:
        env.cache_visualization([_step(_state(0), None, None)], tmp_path, 0)
    save.assert_called_once()
    # The standard hook must work with no backend or optional simulator installed.
    assert env._session is None


def test_custom_controls_are_displayed_verbatim():
    visualizer = RacetrackVisualizer(1, action_presets=[(0.25, -0.125)])
    assert visualizer._action(0) == "0   accel +0.25 / steer -0.125"
    assert visualizer._action(None) == "none"


def test_terminal_frame_preserves_last_transition_event():
    from unittest.mock import patch

    first, final = _state(0), _state(4)
    history = [
        _step(first, final, 1)._replace(reward=0.25, info={"off_road": 1.0}),
        _step(final, None, None),
    ]
    visualizer = RacetrackVisualizer(1)
    with patch.object(visualizer, "_text", wraps=visualizer._text) as text:
        visualizer.render_frames(history)
    events = [
        call.args[2] for call in text.call_args_list if call.args[2] == "Recorded result: off road"
    ]
    assert len(events) == 2
