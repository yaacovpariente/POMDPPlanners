# SPDX-License-Identifier: MIT

"""Unit tests for the forward-only nuPlan world environment.

The tests drive a scripted ``FakeNuPlanSession`` (a tiny stand-in exposing reset/step over a
7-D ego + agent-slot state and an ``{ego, agents}`` observation) injected by monkeypatching
``NuPlanPOMDP._get_session``, so they run without the nuPlan devkit or dataset.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    NuPlanPOMDP,
    NuPlanPOMDPMetrics,
    assemble_state,
)

_MAX_AGENTS = 2


def _base_state(x: float = 0.0, agent_ahead: Optional[float] = None) -> np.ndarray:
    """Build a live-state vector with an optional single present agent slot."""
    ego = [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    agent_rows: List[List[float]] = []
    if agent_ahead is not None:
        agent_rows.append([1.0, agent_ahead, 0.0, 0.0, 0.0])
    return assemble_state(ego, agent_rows, _MAX_AGENTS)


class FakeNuPlanSession:
    """Scripted session: advances x by 1 m per step and tracks call counts / last control."""

    def __init__(self, collide: bool = False) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.last_control: Optional[Tuple[float, float]] = None
        self._collide = collide
        self._x = 0.0

    def _observation(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        agents_end = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH
        return {
            "ego": state[:EGO_STATE_WIDTH].copy(),
            "agents": state[EGO_STATE_WIDTH:agents_end].copy(),
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        del seed
        self.reset_calls += 1
        self._x = 0.0
        state = _base_state(self._x)
        return state, self._observation(state)

    def step(
        self, acceleration: float, steering_angle: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
        self.step_calls += 1
        self.last_control = (acceleration, steering_angle)
        self._x += 1.0
        state = _base_state(self._x)
        return state, self._observation(state), self._collide


@pytest.fixture(name="fake_session")
def _fake_session(monkeypatch: pytest.MonkeyPatch) -> FakeNuPlanSession:
    """Patch ``NuPlanPOMDP._get_session`` to return a shared FakeNuPlanSession."""
    session = FakeNuPlanSession()
    monkeypatch.setattr(NuPlanPOMDP, "_get_session", lambda self: session)
    return session


@pytest.fixture(name="world")
def _world(fake_session: FakeNuPlanSession) -> NuPlanPOMDP:
    """A NuPlanPOMDP wrapping the fake session, reset to its initial live state."""
    del fake_session
    env = NuPlanPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    env.initial_state_dist().sample()
    return env


def test_initial_state_dist_sample_triggers_reset(fake_session: FakeNuPlanSession) -> None:
    """Sampling the initial-state distribution resets the nuPlan session.

    Purpose: Validates initial_state_dist maps to session.reset.

    Given: A NuPlanPOMDP with a scripted fake session
    When: initial_state_dist().sample() is called
    Then: session.reset runs and the returned state is the reset ground truth

    Test type: unit
    """
    env = NuPlanPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    before = fake_session.reset_calls
    state = env.initial_state_dist().sample()[0]
    assert fake_session.reset_calls - before == 1
    assert np.array_equal(state, _base_state(0.0))


def test_initial_observation_is_reset_reading(world: NuPlanPOMDP) -> None:
    """The initial observation is the first post-reset ``{ego, agents}`` reading.

    Purpose: Validates initial_observation_dist returns the reset observation dict.

    Given: A NuPlanPOMDP world reset to its initial state
    When: initial_observation_dist().sample() is called
    Then: the returned observation carries the ego and agents channels

    Test type: unit
    """
    observation = world.initial_observation_dist().sample()[0]
    assert sorted(observation) == ["agents", "ego"]
    assert observation["ego"].shape == (EGO_STATE_WIDTH,)


def test_sample_next_state_advances_world_one_step(
    world: NuPlanPOMDP, fake_session: FakeNuPlanSession
) -> None:
    """sample_next_state advances the world exactly one iteration.

    Purpose: Validates one step maps to one session.step and returns the advanced state.

    Given: A freshly reset NuPlanPOMDP world at its live state
    When: sample_next_state is called from the live state
    Then: session.step runs once and x advances by 1 m

    Test type: unit
    """
    before = fake_session.step_calls
    next_state = world.sample_next_state(_base_state(0.0), 0)
    assert fake_session.step_calls - before == 1
    assert next_state[0] == pytest.approx(1.0)


def test_reward_and_next_state_share_one_step(
    world: NuPlanPOMDP, fake_session: FakeNuPlanSession
) -> None:
    """The reward and next-state getters of one step share a single world iteration.

    Purpose: Validates the tick-cache serves both getters from one session.step.

    Given: A freshly reset NuPlanPOMDP world at its live state
    When: sample_next_state then reward are queried for the same (state, action)
    Then: only one session.step is issued for the pair

    Test type: unit
    """
    before = fake_session.step_calls
    state = _base_state(0.0)
    world.sample_next_state(state, 0)
    world.reward(state, 0)
    assert fake_session.step_calls - before == 1


def test_action_preset_index_maps_to_control(
    world: NuPlanPOMDP, fake_session: FakeNuPlanSession
) -> None:
    """A discrete action index selects the matching control preset.

    Purpose: Validates the action-index to (acceleration, steering_angle) mapping.

    Given: A NuPlanPOMDP with the default control presets
    When: sample_next_state is called with the brake preset index (3)
    Then: the session receives the brake preset control

    Test type: unit
    """
    world.sample_next_state(_base_state(0.0), 3)
    assert fake_session.last_control == world.action_presets[3]


def test_sample_next_state_from_stale_state_raises(world: NuPlanPOMDP) -> None:
    """Stepping from a non-live state raises (forward-only world).

    Purpose: Validates the forward-only guard on arbitrary-state resampling.

    Given: A NuPlanPOMDP world at its live state
    When: sample_next_state is called with a state that is not the live one
    Then: a RuntimeError is raised

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="forward-only"):
        world.sample_next_state(_base_state(99.0), 0)


def test_transition_and_observation_density_unsupported(world: NuPlanPOMDP) -> None:
    """Density queries are unsupported on the forward-only world.

    Purpose: Validates transition/observation densities raise NotImplementedError.

    Given: A NuPlanPOMDP forward-only world
    When: transition_log_probability / observation_log_probability are called
    Then: both raise NotImplementedError

    Test type: unit
    """
    state = _base_state(0.0)
    with pytest.raises(NotImplementedError):
        world.transition_log_probability(state, 0, [state])
    with pytest.raises(NotImplementedError):
        world.observation_log_probability(state, 0, [{"ego": state[:EGO_STATE_WIDTH]}])


def test_collision_is_terminal_and_penalised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A colliding step marks the world terminal and applies the collision penalty.

    Purpose: Validates the terminal flag and collision-penalised reward on a crash.

    Given: A NuPlanPOMDP whose session reports a collision on step
    When: the world is stepped from its live state
    Then: is_terminal is True and the reward is below the collision penalty magnitude

    Test type: unit
    """
    session = FakeNuPlanSession(collide=True)
    monkeypatch.setattr(NuPlanPOMDP, "_get_session", lambda self: session)
    env = NuPlanPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS, collision_penalty=100.0)
    state = env.initial_state_dist().sample()[0]
    next_state, _, reward = env.sample_next_step(state, 0)
    assert env.is_terminal(next_state) is True
    assert reward < -50.0


def test_compute_metrics_reports_named_metrics(world: NuPlanPOMDP) -> None:
    """compute_metrics returns the declared nuPlan metric names.

    Purpose: Validates metric names and that a one-episode history is summarised.

    Given: A NuPlanPOMDP world and a one-step episode history
    When: compute_metrics is called on the history
    Then: the returned metric names match get_metric_names

    Test type: unit
    """
    belief = WeightedParticleBelief([np.zeros(1), np.zeros(1)], np.array([0.0, -1.0]))
    step = StepData(
        state=_base_state(0.0, agent_ahead=3.0),
        action=0,
        observation={"ego": np.zeros(EGO_STATE_WIDTH)},
        next_state=_base_state(1.0, agent_ahead=2.0),
        reward=0.5,
        belief=belief,
    )
    history = History(
        history=[step],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=1,
        reach_terminal_state=False,
        policy_run_data=[],
    )
    metrics = world.compute_metrics([history])
    assert [m.name for m in metrics] == world.get_metric_names()
    assert NuPlanPOMDPMetrics.COLLISION_RATE.value in {m.name for m in metrics}


def test_pickle_drops_live_session(world: NuPlanPOMDP) -> None:
    """Pickling the world drops the non-picklable live session handle.

    Purpose: Validates __getstate__ clears live-session fields for distributed runs.

    Given: A NuPlanPOMDP world with a live session and cached step
    When: __getstate__ is invoked
    Then: the serialized state carries no live session

    Test type: unit
    """
    world.sample_next_state(_base_state(0.0), 0)
    pickled = world.__getstate__()
    assert pickled["_session"] is None
    assert pickled["_pending"] is None


def test_observation_extractor_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured observation_extractor transforms the emitted observation.

    Purpose: Validates observation_extractor is applied to reset and step observations.

    Given: A NuPlanPOMDP with an extractor returning only the ego channel
    When: the initial observation is sampled
    Then: the emitted observation is the extractor's output

    Test type: unit
    """
    session = FakeNuPlanSession()
    monkeypatch.setattr(NuPlanPOMDP, "_get_session", lambda self: session)

    def _ego_only(observation: Dict[str, Any]) -> Any:
        return observation["ego"]

    env = NuPlanPOMDP(
        discount_factor=0.95,
        max_tracked_agents=_MAX_AGENTS,
        observation_extractor=_ego_only,
    )
    observation = env.initial_observation_dist().sample()[0]
    assert observation.shape == (EGO_STATE_WIDTH,)
