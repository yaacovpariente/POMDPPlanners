# SPDX-License-Identifier: MIT

"""Tests for the RockSample episode trace exporter and its visualizer package.

Two things are checked here, and they pull in opposite directions on purpose.

The exporter is new, so its payload is checked field by field: a trace is the
whole interface between a finished episode and the browser viewer, and a field
that does not survive the round trip is a field the viewer silently never sees.

The GIF is *not* new. Moving the renderer into a ``visualizer`` package must
not change a byte of it, so the pinned episode is rendered through the new
import path and compared against the golden file that was pinned before the
move.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    create_rock_sample_state,
)
from POMDPPlanners.environments.rock_sample_pomdp.visualizer import (
    ROCK_SAMPLE_PAYLOAD_KIND,
    RockSampleVisualizer,
    build_rock_sample_trace,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    GOLDEN_DIR,
    create_deterministic_rock_sample_episode,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import rock_sample_pinned_kwargs

# 0=sample, 1=north, 2=east, 3=south, 4=west, 5+=check_rock_i.
SAMPLE, EAST, CHECK_ROCK_0, CHECK_ROCK_1 = 0, 2, 5, 6

# The board the golden RockSample GIF is rendered for.
PINNED_BOARD = {
    "map_size": (5, 5),
    "rock_positions": [(1, 1), (2, 3), (4, 2)],
    "dangerous_areas": [(2, 2)],
    "dangerous_area_radius": 1.0,
}


@pytest.fixture(name="env")
def env_fixture():
    """A small RockSample with one hazard, so the world block has one to carry."""
    return RockSamplePOMDP(
        discount_factor=0.95,
        map_size=(5, 5),
        rock_positions=[(1, 1), (2, 3), (4, 2)],
        init_pos=(0, 0),
        dangerous_areas=[(2, 2)],
        dangerous_area_radius=1.0,
        dangerous_area_penalty=-5.0,
    )


def _belief(rock_configurations, weights=None):
    """A particle belief over whole RockSample states, as a real run holds one."""
    particles = [
        create_rock_sample_state((0, 0), configuration) for configuration in rock_configurations
    ]
    if weights is None:
        weights = np.ones(len(particles)) / len(particles)
    return WeightedParticleBelief(particles=particles, log_weights=np.log(weights))


def _step(state, action, observation, reward, belief, next_state=None):
    return StepData(
        state=state,
        action=action,
        next_state=next_state,
        observation=observation,
        reward=reward,
        belief=belief,
    )


def _episode():
    """A short hand-built episode: a check, a sample, a drive and the exit.

    Hand-built rather than planned, because these tests are about what the
    exporter writes, not about what a planner does. The shapes are the real
    ones: states are the environment's own arrays and the exit is its (-1, -1)
    sentinel.
    """
    belief = _belief([(True, True, False), (True, False, False), (False, True, True)])
    return [
        _step(
            create_rock_sample_state((0, 0), (True, True, False)), CHECK_ROCK_1, "bad", 0.0, belief
        ),
        _step(
            create_rock_sample_state((0, 0), (True, True, False)), CHECK_ROCK_1, "good", 0.0, belief
        ),
        _step(create_rock_sample_state((1, 1), (True, True, False)), SAMPLE, "none", 10.0, belief),
        _step(create_rock_sample_state((1, 4), (False, True, False)), EAST, "none", 10.0, belief),
        _step(create_rock_sample_state((-1, -1), (False, True, False)), None, None, None, belief),
    ]


def test_trace_round_trips_through_json(env, tmp_path: Path):
    """A trace written and read back is the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A RockSample episode.
    When: Its trace is written to disk and read back.
    Then: Envelope and payload match the original, field for field.

    Test type: unit
    """
    trace = build_rock_sample_trace(env, _episode(), episode_index=3, policy_name="PFT_DPW")
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_3.json"))

    assert restored.payload_kind == ROCK_SAMPLE_PAYLOAD_KIND
    assert restored.episode_index == 3
    assert restored.policy == "PFT_DPW"
    assert restored.discount_factor == pytest.approx(env.discount_factor)
    assert restored.payload == trace.payload
    assert [step.to_dict() for step in restored.steps] == [step.to_dict() for step in trace.steps]


def test_trace_takes_the_world_from_the_instance(env):
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured away from the defaults, and a viewer that
    drew the defaults would draw a different board from the one the planner
    solved.

    Test type: unit
    """
    world = build_rock_sample_trace(env, _episode(), 0).payload["world"]

    assert world["map_size"] == [5, 5]
    assert world["rock_positions"] == [[1, 1], [2, 3], [4, 2]]
    assert world["dangerous_areas"] == [[2, 2]]
    assert world["dangerous_area_radius"] == pytest.approx(1.0)
    assert world["dangerous_area_penalty"] == pytest.approx(-5.0)
    assert world["init_pos"] == [0, 0]
    assert world["action_names"] == env.action_names
    assert world["first_check_action"] == CHECK_ROCK_0


def test_world_names_the_sensor_contract_it_was_run_under(env):
    """The accuracy law is written down, not assumed.

    Purpose: RockSample's check accuracy was changed once already — version 1
    was a decaying exponential that fell below a coin flip, version 2 is Smith
    & Simmons. A viewer that quotes P(correct) has to know which law it is
    quoting, and the constructor arguments are identical across the two.

    Test type: unit
    """
    world = build_rock_sample_trace(env, _episode(), 0).payload["world"]

    assert world["sensor_contract_version"] == env.sensor_contract_version
    assert world["sensor_efficiency"] == pytest.approx(env.sensor_efficiency)


def test_trace_records_which_rock_each_check_hit(env):
    """A check step carries its rock and its reading; every other step carries none.

    Purpose: The belief here is per-rock and moves only on a check, so the
    viewer stages a check as an event. Reconstructing which rock was meant from
    a bare action integer needs the rock count, which is a property of the run —
    so it is resolved here, once.

    Test type: unit
    """
    checks = build_rock_sample_trace(env, _episode(), 0).payload["checks"]

    assert checks == [
        {"rock": 1, "observation": "bad"},
        {"rock": 1, "observation": "good"},
        None,
        None,
        None,
    ]


def test_check_lookup_ignores_actions_past_the_check_block(env):
    """An action beyond the last check is no check, not a check on rock 97.

    Purpose: The check actions are the tail of the action list, so an
    out-of-range index would silently address a rock the world does not have.

    Test type: unit
    """
    history = _episode()
    history[0] = _step(history[0].state, 99, "good", 0.0, history[0].belief)

    assert build_rock_sample_trace(env, history, 0).payload["checks"][0] is None


def test_trace_keeps_the_exit_sentinel(env):
    """The exit state is written as the environment recorded it.

    Purpose: Leaving east ends the episode and the environment records that as
    (-1, -1) — a state with no cell. Turning it into a cell here would invent a
    position the run never had; where to *draw* a rover that has left the board
    is the viewer's decision, made from this value.

    Test type: unit
    """
    payload = build_rock_sample_trace(env, _episode(), 0).payload

    assert payload["states"][0] == [0, 0]
    assert payload["states"][-1] == [-1, -1]
    assert len(payload["states"]) == len(_episode())


def test_rock_truth_comes_from_the_state(env):
    """The true rock qualities are read off the recorded states.

    Purpose: The viewer draws the truth beside the belief so the two can
    visibly disagree, and so a sampled-out rock visibly turns bad. Both of
    those are in the state; neither is in the belief.

    Test type: unit
    """
    payload = build_rock_sample_trace(env, _episode(), 0).payload

    assert payload["rock_truth"][0] == [True, True, False]
    assert payload["rock_truth"][-1] == [False, True, False]


def test_trace_delegates_belief_serialization_to_core(env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean fifteen copies of
    the same dispatch and fifteen chances to miss a class. Comparing against
    core's own output is what stops the two drifting.

    Test type: unit
    """
    history = _episode()
    trace = build_rock_sample_trace(env, history, 0)

    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_belief_particles_carry_the_rock_bits_the_viewer_marginalises(env):
    """Each written particle is a whole state, rocks included.

    Purpose: The per-rock gauge is the share of belief mass in which that rock
    is good, computed in the viewer from these particles. If the rock slots did
    not survive serialization there would be nothing to compute it from.

    Test type: unit
    """
    belief = build_rock_sample_trace(env, _episode(), 0).payload["beliefs"][0]

    assert belief["kind"] == "particles"
    assert [particle[2:] for particle in belief["particles"]] == [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    assert sum(belief["weights"]) == pytest.approx(1.0)


def test_trace_inherits_the_subsampling_cap(env):
    """A belief larger than core's cap is trimmed in the exporter's output too.

    Purpose: A vectorized RockSample belief carries far more particles than a
    browser scene can draw, and the trace file would be enormous.

    Test type: unit
    """
    count = MAX_PAYLOAD_PARTICLES * 3
    belief = _belief([(bool(i % 2), True, False) for i in range(count)])
    history = [
        _step(create_rock_sample_state((0, 0), (True, True, False)), EAST, "none", 0.0, belief)
    ]

    written = build_rock_sample_trace(env, history, 0).payload["beliefs"][0]

    assert written["num_particles"] == count
    assert written["num_written"] == MAX_PAYLOAD_PARTICLES


def test_environment_writes_a_trace_file(env, tmp_path: Path):
    """cache_trace writes a readable file under the episode's index.

    Purpose: This is the path the simulator actually calls, so it is the one
    that decides whether a run produces traces at all.

    Test type: integration
    """
    written = env.cache_trace(
        history=_episode(), output_dir=tmp_path, episode_index=2, policy_name="POMCPOW"
    )

    assert written == tmp_path / "trace_2.json"
    restored = EpisodeTrace.read(written)
    assert restored.payload_kind == ROCK_SAMPLE_PAYLOAD_KIND
    assert restored.policy == "POMCPOW"


def test_trace_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps.

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty history"):
        build_rock_sample_trace(env, [], 0)


def test_the_visualizer_package_renders_the_pinned_episode_deterministically(tmp_path: Path):
    """Two renders through the new import path are byte-for-byte identical.

    Purpose: The golden-hash comparison below only runs inside the project's
    Docker image, because the bytes are pinned to its PIL build. Determinism is
    not — and it is the property the move could plausibly have broken, by
    changing when a module-level cache is built.

    Test type: unit
    """
    env = RockSamplePOMDP(discount_factor=0.95, **rock_sample_pinned_kwargs(**PINNED_BOARD))
    history = create_deterministic_rock_sample_episode(seed=42)
    first, second = tmp_path / "first.gif", tmp_path / "second.gif"

    RockSampleVisualizer(env).create_visualization(history, first)
    RockSampleVisualizer(env).create_visualization(history, second)

    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        hashlib.sha256(second.read_bytes()).hexdigest()
    )


@pytest.mark.skipif(
    not Path("/.dockerenv").exists(),
    reason=(
        "Golden visualization hashes are pinned to the PIL version in the "
        "project's Docker CI image, so the byte-identical check only runs there."
    ),
)
def test_moving_the_visualizer_did_not_change_the_gif(tmp_path: Path):
    """The pinned episode still renders to the pinned golden bytes.

    Purpose: Adding the trace exporter meant putting the renderer in a
    ``visualizer`` package. The renderer's output is pinned by a golden hash, so
    the move has to be a move and nothing else. This renders the same
    deterministic episode through the new import path and compares it with the
    golden file that was pinned before the move.

    Given: The golden registry's own RockSample episode and renderer.
    When: It is rendered through ``visualizer.rock_sample_visualizer``.
    Then: The bytes hash to the golden file's hash.

    Test type: integration
    """
    golden = GOLDEN_DIR / "rock_sample_visualization.gif"
    if not golden.exists():
        pytest.skip("No golden RockSample GIF to compare against")

    env = RockSamplePOMDP(discount_factor=0.95, **rock_sample_pinned_kwargs(**PINNED_BOARD))
    output = tmp_path / "rock_sample.gif"
    RockSampleVisualizer(env).create_visualization(
        create_deterministic_rock_sample_episode(seed=42), output
    )

    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        hashlib.sha256(golden.read_bytes()).hexdigest()
    ), (
        "The RockSample GIF changed. Moving the renderer into its visualizer "
        "package must not alter a byte of its output."
    )
