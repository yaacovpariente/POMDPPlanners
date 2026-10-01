# SPDX-License-Identifier: MIT

"""Tests for the Tiger episode trace.

The visualizer is the only thing standing between a recorded episode and a
browser page that claims to show it, so these check the two ways it could lie:
by writing a number the environment does not hold, and by writing a belief the
run did not have.
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.tiger_pomdp.tiger_visualization import (
    TIGER_PAYLOAD_KIND,
    TigerVisualizer,
)


def _belief(left_mass: float) -> WeightedParticleBelief:
    """A two-particle belief carrying ``left_mass`` on ``tiger_left``."""
    return WeightedParticleBelief(
        ["tiger_left", "tiger_right"], np.log([left_mass, 1.0 - left_mass])
    )


def _episode():
    """Listen, open the wrong door, then the step after the re-draw.

    The third record is what this environment is about: opening did not end the
    episode, the tiger's side was re-drawn, and the belief is back at even.
    """
    return [
        StepData("tiger_left", "listen", "tiger_left", "hear_left", -1.0, _belief(0.5)),
        StepData("tiger_left", "open_left", "tiger_right", "hear_nothing", -100.0, _belief(0.85)),
        StepData("tiger_right", None, None, None, None, _belief(0.5)),
    ]


@pytest.fixture(name="tiger")
def tiger_fixture() -> TigerPOMDP:
    return TigerPOMDP(discount_factor=0.95)


def test_payload_round_trips_through_json(tmp_path: Path, tiger: TigerPOMDP):
    """A written trace reads back as the same trace.

    Purpose: The file is the whole interface between Python and the browser
    viewer, so a field that does not survive the round trip is a field the
    viewer silently never sees.

    Given: A three-step Tiger episode.
    When: Its trace is written to disk and read back.
    Then: Envelope and payload match the object that was written.
    """
    trace = TigerVisualizer(tiger).build_trace(_episode(), episode_index=3, policy_name="POMCP")
    path = trace.write(tmp_path / "trace.json")
    back = EpisodeTrace.read(path)

    assert back.payload_kind == TIGER_PAYLOAD_KIND == "tiger.v1"
    assert back.environment == tiger.name
    assert back.episode_index == 3
    assert back.policy == "POMCP"
    assert back.discount_factor == pytest.approx(0.95)
    assert back.payload == trace.payload
    assert [step.action for step in back.steps] == ["listen", "open_left", None]
    assert [step.reward for step in back.steps] == [-1.0, -100.0, None]


def test_world_block_is_read_out_of_the_environment(tiger: TigerPOMDP):
    """The labels a viewer prints come from the models, not from literals.

    Purpose: A viewer that captions the room with this package's defaults would
    keep captioning it with them after a subclass changed the listen accuracy or
    the penalty, and a reader would believe the caption.

    Given: A Tiger environment.
    When: Its trace is built.
    Then: The world block agrees with what the environment's own models say.
    """
    world = TigerVisualizer(tiger).build_trace(_episode(), episode_index=0).payload["world"]

    assert world["states"] == ["tiger_left", "tiger_right"]
    assert world["actions"] == ["listen", "open_left", "open_right"]
    assert world["listen_accuracy"] == pytest.approx(0.85)
    assert world["listen_reward"] == pytest.approx(tiger.reward("tiger_left", "listen"))
    assert world["correct_door_reward"] == pytest.approx(10.0)
    assert world["wrong_door_reward"] == pytest.approx(-100.0)


def test_opening_a_door_is_not_terminal_and_the_side_is_re_drawn(tiger: TigerPOMDP):
    """The payload says what this environment actually does after an open.

    Purpose: A viewer that assumed the episode ends at the first open would
    stop drawing exactly where the interesting part starts — the belief being
    thrown away — and would never show the re-drawn state.

    Given: An episode whose second step opens a door.
    When: Its trace is built.
    Then: Nothing is marked terminal, and the re-draw is recorded as uniform.
    """
    trace = TigerVisualizer(tiger).build_trace(_episode(), episode_index=0)

    assert trace.reach_terminal_state is False
    assert trace.payload["world"]["open_is_terminal"] is False
    assert trace.payload["world"]["open_redraw_probabilities"] == pytest.approx([0.5, 0.5])
    # The state before the open and the state after it are both recorded, so a
    # reader can see the side change rather than infer it.
    assert trace.payload["states"] == ["tiger_left", "tiger_left", "tiger_right"]
    assert trace.payload["next_states"] == ["tiger_left", "tiger_right", None]


def test_belief_is_the_run_s_own_and_resets_after_the_open(tiger: TigerPOMDP):
    """The beliefs written are the recorded ones, serialized by core.

    Purpose: A belief regenerated from the true state looks convincing and says
    nothing. The reset to even after an open is the environment's teaching
    moment, and it has to come from the run rather than from this visualizer.

    Given: An episode that listens to 0.85, opens, and is back at 0.50.
    When: Its trace is built.
    Then: The particle payloads carry those masses in that order.
    """
    beliefs = TigerVisualizer(tiger).build_trace(_episode(), episode_index=0).payload["beliefs"]

    assert [b["kind"] for b in beliefs] == ["particles"] * 3
    assert [b["particles"] for b in beliefs] == [["tiger_left", "tiger_right"]] * 3
    left_mass = [b["weights"][b["particles"].index("tiger_left")] for b in beliefs]
    assert left_mass == pytest.approx([0.5, 0.85, 0.5])


def test_unweighted_belief_keeps_its_labels(tiger: TigerPOMDP):
    """An unweighted cloud of state labels survives serialization.

    Purpose: Tiger's particles are strings, not coordinates, and a viewer reads
    a hypothesis's probability as the mass its label carries. A payload that
    dropped or stringified the labels would leave it nothing to count.
    """
    history = [
        StepData(
            "tiger_left",
            "listen",
            "tiger_left",
            "hear_left",
            -1.0,
            UnweightedParticleBeliefStateUpdate(["tiger_left", "tiger_left", "tiger_right"]),
        )
    ]
    belief = TigerVisualizer(tiger).build_trace(history, episode_index=0).payload["beliefs"][0]

    assert belief["weighted"] is False
    assert belief["particles"] == ["tiger_left", "tiger_left", "tiger_right"]
    assert sum(belief["weights"]) == pytest.approx(1.0)


def test_empty_history_is_refused(tiger: TigerPOMDP):
    """There is no episode to write, so nothing is written."""
    with pytest.raises(ValueError, match="empty history"):
        TigerVisualizer(tiger).build_trace([], episode_index=0)


def test_the_environment_writes_a_readable_trace(tmp_path: Path, tiger: TigerPOMDP):
    """The environment's visualizer writes its trace where the reporting site looks.

    Purpose: The simulation layer calls ``episode_visualizer().write``, never
    the visualizer directly, so a visualizer nobody reaches writes nothing
    however correct it is.
    """
    written = tiger.episode_visualizer().write(
        history=_episode(), output_dir=tmp_path, episode_index=2, policy_name="POMCP"
    )

    assert written == tmp_path / "trace_2.json"
    assert EpisodeTrace.read(written).payload_kind == TIGER_PAYLOAD_KIND
