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
        assert decoded_still.size == (640, 640)
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
