# SPDX-License-Identifier: MIT
"""Replay semantics and saved-frame checks for the Tiger renderer."""

from pathlib import Path
import random
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from PIL import GifImagePlugin, Image
import pytest

from POMDPPlanners.core.belief import (
    WeightedParticleBelief,
    UnweightedParticleBelief,
    UnweightedParticleBeliefStateUpdate,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.tiger_visualizer import (
    CANVAS_SIZE,
    TigerVisualizer,
    _background,
    _belief_probabilities,
    _frame_data,
    _palette,
)


def step(**changes: Any) -> StepData:
    belief = WeightedParticleBelief(
        ["tiger_left", "tiger_right", "tiger_left"], np.log([0.1, 0.2, 0.7])
    )
    row = StepData("tiger_left", "listen", "tiger_left", "hear_right", -1.0, belief)
    return row._replace(**changes)


@pytest.mark.parametrize(
    "state,action,next_state,outcome,side",
    [
        ("tiger_left", "open_left", "tiger_right", "TIGER", 0),
        ("tiger_left", "open_right", "tiger_right", "TREASURE", 1),
        ("tiger_right", "open_left", "tiger_left", "TREASURE", 0),
        ("tiger_right", "open_right", "tiger_left", "TIGER", 1),
    ],
)
def test_open_outcome_uses_pre_action_state(state, action, next_state, outcome, side):
    data = _frame_data(
        step(
            state=state,
            action=action,
            next_state=next_state,
            observation="hear_nothing",
            reward=-100 if outcome == "TIGER" else 10,
        )
    )
    assert data.outcome == outcome
    assert data.opened_side == side
    assert data.hidden == state.removeprefix("tiger_").upper()
    assert data.observation == "hear_nothing"
    assert data.reward == ("-100" if outcome == "TIGER" else "+10")


def test_noisy_observation_is_not_hidden_truth():
    data = _frame_data(step())
    assert data.hidden == "LEFT"
    assert data.observation == "hear_right"
    assert data.outcome == ""
    assert data.opened_side is None
    assert data.probabilities == pytest.approx((0.8, 0.2))


def test_missing_belief_is_not_fabricated_uniform():
    assert _belief_probabilities(None) is None


class ReadOnlyUnweightedBelief(UnweightedParticleBelief):
    def _reinvigoration_pertubation(self, action, observation, pomdp):
        raise AssertionError("Rendering must not reinvigorate particles")


@pytest.mark.parametrize(
    "belief_type", [ReadOnlyUnweightedBelief, UnweightedParticleBeliefStateUpdate]
)
def test_unweighted_string_belief(belief_type):
    belief = belief_type(["tiger_left", "tiger_left", "tiger_right"])
    assert _belief_probabilities(belief) == pytest.approx((2 / 3, 1 / 3))


def test_unsupported_belief_is_explicit():
    with pytest.raises(TypeError, match="discrete particle belief"):
        _belief_probabilities(object())


def golden_history():
    return [
        step(),
        step(action="open_right", next_state="tiger_right", observation="hear_nothing", reward=10),
        step(state="tiger_right", action=None, next_state=None, observation=None, reward=None),
    ]


@pytest.mark.skipif(not Path("/.dockerenv").exists(), reason="Fixed pixels use the CI Docker image")
def test_tiger_golden_visualization(tmp_path):
    path = tmp_path / "tiger.gif"
    TigerVisualizer().create_visualization(golden_history(), path)
    golden = Path(__file__).parent / "golden_visualizations" / "tiger_visualization.gif"
    assert path.read_bytes() == golden.read_bytes()


def test_terminal_bookkeeping_does_not_invent_fields():
    data = _frame_data(step(action=None, next_state=None, observation=None, reward=None))
    assert data.action == "none (bookkeeping)"
    assert data.observation == data.reward == "none recorded"
    assert data.opened_side is None


def test_exact_decoded_frames_repeats_and_truncated_last_action(tmp_path):
    visualizer = TigerVisualizer()
    last = step(
        action="open_left", next_state="tiger_right", observation="hear_nothing", reward=-100
    )
    history = [step(), step(), last]
    path = tmp_path / "nested" / "replay.gif"
    visualizer.create_visualization(history, path)
    with Image.open(path) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.n_frames == len(history)
        assert gif.size == CANVAS_SIZE
        for index, row in enumerate(history):
            gif.seek(index)
            expected = (
                visualizer._render_frame(_frame_data(row), index, len(history))
                .quantize(palette=_palette(), dither=Image.Dither.NONE)
                .convert("RGB")
            )
            assert gif.info["duration"] == 500
            assert gif.convert("RGB").tobytes() == expected.tobytes()
    assert _frame_data(last).outcome == "TIGER"


def test_one_terminal_frame(tmp_path):
    path = tmp_path / "terminal.gif"
    TigerVisualizer().create_visualization(
        [step(action=None, next_state=None, observation=None, reward=None)], path
    )
    with Image.open(path) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.n_frames == 1
        assert gif.info["duration"] == 500


def test_hook_preserves_rng_and_never_steps_environment(tmp_path):
    env = TigerPOMDP(0.95)
    row = step()
    random_before = random.getstate()
    numpy_before = cast(tuple[Any, ...], np.random.get_state())
    with patch.object(env, "sample_next_state", side_effect=AssertionError("renderer sampled")):
        env.cache_visualization([row], tmp_path, 42)
    assert (tmp_path / "agent_path_42.gif").is_file()
    assert random.getstate() == random_before
    numpy_after = cast(tuple[Any, ...], np.random.get_state())
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_cache_loads_art_once_and_frames_do_not_mutate_background(tmp_path):
    _background.cache_clear()
    _palette.cache_clear()
    original_open = Image.open
    with patch(
        "POMDPPlanners.environments.tiger_visualizer.Image.open", wraps=original_open
    ) as opened:
        renderer = TigerVisualizer()
        history = [step(), step()]
        renderer.create_visualization(history, tmp_path / "first.gif")
        original = _background().tobytes()
        renderer.create_visualization(history, tmp_path / "second.gif")
        assert opened.call_count == 1
        assert _background().tobytes() == original
    assert (tmp_path / "first.gif").read_bytes() == (tmp_path / "second.gif").read_bytes()


@pytest.mark.parametrize(
    "history,error", [([], ValueError), (None, TypeError), ([None], TypeError)]
)
def test_bad_history_does_not_write(tmp_path, history, error):
    path = tmp_path / "bad.gif"
    with pytest.raises(error):
        TigerVisualizer().create_visualization(history, path)
    assert not path.exists()


@pytest.mark.parametrize(
    "changes",
    [{"state": "bad"}, {"action": "jump"}, {"observation": "true_left"}, {"reward": float("nan")}],
)
def test_bad_row_is_rejected_before_file_creation(tmp_path, changes):
    path = tmp_path / "bad.gif"
    with pytest.raises(ValueError):
        TigerVisualizer().create_visualization([step(**changes)], path)
    assert not path.exists()


@pytest.mark.parametrize("path,error", [("file.gif", TypeError), (Path("file.png"), ValueError)])
def test_bad_output_path(path, error):
    with pytest.raises(error):
        TigerVisualizer().create_visualization([step()], path)
