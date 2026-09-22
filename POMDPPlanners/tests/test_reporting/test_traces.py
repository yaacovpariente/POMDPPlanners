# SPDX-License-Identifier: MIT

"""Tests for the episode trace schema and the Light-Dark exporter."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import (
    TRACE_SCHEMA_VERSION,
    EpisodeTrace,
    TraceStep,
    envelope_steps,
    to_jsonable,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_visualization.trace_exporter import (
    LIGHT_DARK_PAYLOAD_KIND,
    build_light_dark_trace,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_light_dark_pinned_kwargs,
)


def _belief(positions):
    particles = [np.asarray(p, dtype=float) for p in positions]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _episode(length: int = 4):
    history = []
    for step in range(length):
        state = np.array([float(step), 5.0])
        is_last = step == length - 1
        history.append(
            StepData(
                state=state,
                action=None if is_last else "right",
                next_state=None if is_last else np.array([float(step + 1), 5.0]),
                observation=None if is_last else np.array([float(step) + 0.1, 5.05]),
                reward=None if is_last else -2.0,
                belief=_belief([[float(step), 5.0], [float(step) + 0.2, 4.9]]),
                info={"in_hazard": 0.0} if not is_last else None,
            )
        )
    return history


@pytest.fixture(name="light_dark_env")
def light_dark_env_fixture():
    """A Light-Dark environment with the suite's pinned configuration."""
    return ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())


def test_to_jsonable_converts_numpy():
    """Numpy values become JSON data rather than reaching json.dump.

    Purpose: Traces are built from live environment state, which is numpy
    almost everywhere, and json refuses numpy scalars and arrays.

    Given: Numpy scalars, arrays and containers of them.
    When: to_jsonable converts them.
    Then: Only plain Python types come back, with the values preserved.
    """
    converted = to_jsonable(
        {"a": np.float64(1.5), "b": np.array([[1, 2], [3, 4]]), "c": (np.bool_(True), None)}
    )
    assert converted == {"a": 1.5, "b": [[1, 2], [3, 4]], "c": [True, None]}
    assert isinstance(converted["a"], float)


def test_envelope_steps_keeps_the_terminal_bookkeeping_step():
    """The final state-only step survives, because a viewer needs it.

    Purpose: An episode's last record carries a state but no action or
    reward; dropping it loses where the episode ended.

    Given: A four-step episode whose last record has no action.
    When: envelope_steps builds the envelope.
    Then: Four steps come back, the last with a null action and reward.
    """
    steps = envelope_steps(_episode(4))
    assert len(steps) == 4
    assert steps[-1].action is None
    assert steps[-1].reward is None
    assert steps[0].action == "right"


def test_envelope_steps_can_drop_the_terminal_step():
    """The terminal step is optional for a consumer that only wants decisions."""
    steps = envelope_steps(_episode(4), include_terminal_step=False)
    assert len(steps) == 3
    assert all(step.action is not None for step in steps)


def test_trace_round_trips_through_json(tmp_path: Path):
    """A trace written and read back is the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A trace with steps, a payload and metadata.
    When: It is written to disk and read back.
    Then: Every field matches the original.
    """
    original = EpisodeTrace(
        environment="ContinuousLightDarkPOMDP",
        payload_kind=LIGHT_DARK_PAYLOAD_KIND,
        episode_index=3,
        discount_factor=0.95,
        steps=[TraceStep(index=0, action="up", reward=-2.0, info={"x": 1.0})],
        payload={"states": [[0.0, 5.0]]},
        policy="PFT_DPW",
        reach_terminal_state=True,
        metadata={"seed": 7},
    )
    path = original.write(tmp_path / "nested" / "trace.json")
    restored = EpisodeTrace.read(path)

    assert restored == original
    assert restored.schema_version == TRACE_SCHEMA_VERSION


def test_trace_recomputes_its_summary_fields(tmp_path: Path):
    """Summary fields are derived on read, not trusted from the file.

    Purpose: num_steps, total_reward and discounted_return are written for a
    reader that only skims the header. A hand-edited file must not be able to
    make the object disagree with its own steps.

    Given: A trace file whose header claims a return the steps do not support.
    When: It is read back.
    Then: The recomputed values come from the steps.
    """
    trace = EpisodeTrace(
        environment="Env",
        payload_kind="x.v1",
        episode_index=0,
        discount_factor=1.0,
        steps=[TraceStep(index=0, reward=-1.0), TraceStep(index=1, reward=-3.0)],
    )
    path = tmp_path / "trace.json"
    data = trace.to_dict()
    data["total_reward"] = 999.0
    data["num_steps"] = 42
    (path).write_text(__import__("json").dumps(data), encoding="utf-8")

    restored = EpisodeTrace.read(path)
    assert restored.num_steps == 2
    assert restored.total_reward == pytest.approx(-4.0)


def test_trace_refuses_an_incompatible_major_version():
    """A file from a future schema is refused rather than misread."""
    data = EpisodeTrace(
        environment="Env", payload_kind="x.v1", episode_index=0, discount_factor=1.0
    ).to_dict()
    data["schema_version"] = "99.0"
    with pytest.raises(ValueError, match="Unsupported trace schema version"):
        EpisodeTrace.from_dict(data)


class _EnvironmentWithoutATraceExporter(Environment):
    """The minimum an Environment can be: it implements no trace exporter.

    Stands in for "any environment nobody has migrated yet", so this file does
    not have to be edited every time one is.
    """

    def __init__(self, discount_factor: float = 0.95):
        super().__init__(
            discount_factor=discount_factor,
            name="no-exporter",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
        )

    def sample_next_state(self, state, action, n_samples: int = 1):
        return state

    def sample_observation(self, next_state, action, n_samples: int = 1):
        return 0

    def reward(self, state, action, next_state=None):
        return 0.0

    def is_terminal(self, state) -> bool:
        return False

    def initial_state(self):
        return 0

    def actions(self, state=None):
        return [0]

    def initial_state_dist(self):
        return DiscreteDistribution({0: 1.0})

    def initial_observation_dist(self):
        return DiscreteDistribution({0: 1.0})

    def transition_log_probability(self, state, action, next_states) -> np.ndarray:
        return np.zeros(1)

    def observation_log_probability(self, next_state, action, observations) -> np.ndarray:
        return np.zeros(1)

    def is_equal_observation(self, observation1, observation2) -> bool:
        return observation1 == observation2

    def hash_action(self, action):
        return action


def test_environment_writes_no_trace_by_default(tmp_path: Path):
    """An environment that implements nothing writes nothing.

    Purpose: Adding the trace path must not change any existing environment's
    behaviour, which means the default has to be silence, not an empty file.

    Given: An environment that implements no trace exporter.
    When: cache_trace is called.
    Then: Nothing is returned and no file appears.

    Note: this deliberately uses a local stub rather than naming a real
    environment. It used to name Tiger, which made the test fail the day Tiger
    gained an exporter — the test would then have been asserting something
    about one environment's migration state instead of about the default.
    """
    env = _EnvironmentWithoutATraceExporter(discount_factor=0.95)
    assert env.cache_trace(history=[], output_dir=tmp_path, episode_index=0) is None
    assert list(tmp_path.iterdir()) == []


def test_light_dark_trace_carries_the_real_belief(light_dark_env):
    """The belief written is the episode's own particles and weights.

    Purpose: The belief is the field this package exists to show and the one
    most easily faked. A cloud regenerated in the viewer would look right and
    mean nothing.

    Given: An episode whose steps carry weighted particle beliefs.
    When: The trace is built.
    Then: Each step's payload holds that step's particle positions and its
        normalized weights, and the weights sum to one.
    """
    history = _episode(3)
    trace = build_light_dark_trace(light_dark_env, history, episode_index=1, policy_name="PFT_DPW")

    beliefs = trace.payload["beliefs"]
    assert len(beliefs) == len(history)
    for step, belief in zip(history, beliefs):
        assert belief["kind"] == "particles"
        expected = [[float(p[0]), float(p[1])] for p in step.belief.particles]
        assert belief["particles"] == expected
        assert sum(belief["weights"]) == pytest.approx(1.0)


def test_light_dark_trace_delegates_belief_serialization_to_core(light_dark_env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean fifteen copies
    of the same dispatch and fifteen chances to miss a class. Comparing against
    core's own output is what stops the two drifting.

    Given: An episode whose steps carry weighted particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that
        step's belief, field for field.
    """
    history = _episode(3)
    trace = build_light_dark_trace(light_dark_env, history, episode_index=0)

    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_light_dark_trace_inherits_the_subsampling_cap(light_dark_env):
    """A belief larger than core's cap is trimmed in the exporter's output too.

    Purpose: A vectorized Light-Dark belief carries far more particles than a
    browser scene can draw, and the trace file would be enormous. This checks
    the exporter inherits the cap rather than writing the whole cloud.
    """
    count = MAX_PAYLOAD_PARTICLES * 3
    positions = [[float(i) / count, 5.0] for i in range(count)]
    history = [
        StepData(
            state=np.array([0.0, 5.0]),
            action="right",
            next_state=np.array([1.0, 5.0]),
            observation=np.array([0.1, 5.0]),
            reward=-2.0,
            belief=_belief(positions),
        )
    ]
    belief = build_light_dark_trace(light_dark_env, history, episode_index=0).payload["beliefs"][0]

    assert belief["num_particles"] == count
    assert belief["num_written"] == MAX_PAYLOAD_PARTICLES
    assert len(belief["particles"]) == MAX_PAYLOAD_PARTICLES


def test_light_dark_trace_takes_the_world_from_the_instance(light_dark_env):
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured away from the defaults, and a viewer that
    drew the defaults would draw a different world from the one the planner
    solved.

    Given: An environment whose goal has been moved from the class default.
    When: The trace is built.
    Then: The payload's world reports the instance's values.
    """
    light_dark_env.goal_state = np.array([2.0, 3.0])
    trace = build_light_dark_trace(light_dark_env, _episode(2), 0)
    assert trace.payload["world"]["goal_state"] == [2.0, 3.0]
    assert trace.payload["world"]["grid_size"] == light_dark_env.grid_size


def test_light_dark_environment_writes_a_trace_file(light_dark_env, tmp_path: Path):
    """cache_trace writes a readable file under the episode's index."""
    written = light_dark_env.cache_trace(
        history=_episode(3), output_dir=tmp_path, episode_index=2, policy_name="POMCPOW"
    )
    assert written == tmp_path / "trace_2.json"

    restored = EpisodeTrace.read(written)
    assert restored.payload_kind == LIGHT_DARK_PAYLOAD_KIND
    assert restored.policy == "POMCPOW"
    assert restored.episode_index == 2


def test_light_dark_trace_rejects_an_empty_history(light_dark_env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        build_light_dark_trace(light_dark_env, [], 0)
