# SPDX-License-Identifier: MIT

"""Unit tests for the forward-only nuPlan world environment.

The tests drive a scripted ``FakeNuPlanSession`` (a tiny stand-in exposing reset/step over a
7-D ego + agent-slot state and an ``{ego, agents}`` observation) injected by monkeypatching
``NuPlanPOMDP._get_session``, so they run without the nuPlan devkit or dataset.
"""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    _NuPlanSession,
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


def test_compute_metrics_empty_histories_is_rejected(world: NuPlanPOMDP) -> None:
    """compute_metrics rejects an empty batch of episodes.

    Purpose: Validates that an empty batch is rejected rather than scored. A
        zero collision_rate over no episodes is indistinguishable from a run in
        which the ego never crashed.

    Given: A NuPlanPOMDP world and an empty history list
    When: compute_metrics is called
    Then: A ValueError naming the environment is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="received no episode histories"):
        world.compute_metrics([])


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


def test_near_miss_metrics_include_the_final_reached_state(world: NuPlanPOMDP) -> None:
    """The closest approach on the last transition is scored.

    Purpose: Validates the near-miss / minimum-distance metrics read the state the episode
        actually ended in, not just the states it acted from.

    Given: A one-transition episode starting 10 m from an agent and ending 1 m from it
    When: compute_metrics is called
    Then: min_vehicle_distance is 1 m and one near-miss event is counted

    Test type: unit
    """
    belief = WeightedParticleBelief([np.zeros(1), np.zeros(1)], np.array([0.0, -1.0]))
    step = StepData(
        state=_base_state(0.0, agent_ahead=10.0),
        action=0,
        observation={"ego": np.zeros(EGO_STATE_WIDTH)},
        next_state=_base_state(9.0, agent_ahead=1.0),
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
    metrics = {metric.name: metric.value for metric in world.compute_metrics([history])}
    assert metrics[NuPlanPOMDPMetrics.MIN_VEHICLE_DISTANCE.value] == pytest.approx(1.0)
    assert metrics[NuPlanPOMDPMetrics.NEAR_MISS_COUNT.value] == pytest.approx(1.0)


def test_compute_metrics_handles_terminal_step_without_next_state(world: NuPlanPOMDP) -> None:
    """compute_metrics summarises an episode whose last step is the terminal marker.

    Purpose: Validates the distance/speed metrics skip the terminal step the episode
        runner appends, which carries ``next_state=None``.

    Given: A history of one driven step followed by the runner's terminal StepData
    When: compute_metrics is called on that history
    Then: the declared metrics are returned and reflect only the driven step

    Test type: unit
    """
    belief = WeightedParticleBelief([np.zeros(1), np.zeros(1)], np.array([0.0, -1.0]))
    driven = StepData(
        state=_base_state(0.0, agent_ahead=3.0),
        action=0,
        observation={"ego": np.zeros(EGO_STATE_WIDTH)},
        next_state=_base_state(1.0, agent_ahead=2.0),
        reward=0.5,
        belief=belief,
    )
    # The terminal step EpisodeRunner appends when the world reports a collision.
    terminal = StepData(
        state=_base_state(1.0, agent_ahead=2.0),
        action=None,
        observation=None,
        next_state=None,
        reward=None,
        belief=belief,
    )
    history = History(
        history=[driven, terminal],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[],
    )
    metrics = {metric.name: metric.value for metric in world.compute_metrics([history])}
    assert set(metrics) == set(world.get_metric_names())
    assert metrics[NuPlanPOMDPMetrics.COLLISION_RATE.value] == 1.0
    assert metrics[NuPlanPOMDPMetrics.AVERAGE_PROGRESS.value] == pytest.approx(1.0)
    assert metrics[NuPlanPOMDPMetrics.AVERAGE_SPEED.value] == pytest.approx(1.0)


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


# ── Live-session control command (fake `nuplan` modules; no devkit needed) ────


class _FakeVector:
    """Minimal stand-in for nuPlan's StateVector2D."""

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


class _FakeDynamicCarState:
    """Records the control command built for one propagation step."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.rear_axle_velocity_2d = kwargs["rear_axle_velocity_2d"]
        self.rear_axle_acceleration_2d = kwargs["rear_axle_acceleration_2d"]
        self.tire_steering_rate = kwargs["tire_steering_rate"]

    @classmethod
    def build_from_rear_axle(cls, **kwargs: Any) -> "_FakeDynamicCarState":
        return cls(**kwargs)


class _FakeEgoState:
    """Ego state exposing only what the trajectory builder reads."""

    def __init__(self, tire_steering_angle: float) -> None:
        self.tire_steering_angle = tire_steering_angle
        self.car_footprint = SimpleNamespace(rear_axle_to_center_dist=1.5)
        self.dynamic_car_state = SimpleNamespace(
            rear_axle_velocity_2d=_FakeVector(3.0, 0.0),
            rear_axle_acceleration_2d=_FakeVector(0.0, 0.0),
        )


class _FakeMotionModel:
    """Integrates only the tire steering angle, so the commanded rate is observable."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.commands: List[Any] = []

    def propagate_state(self, state: Any, command: Any, sampling_time: Any) -> _FakeEgoState:
        del sampling_time
        self.commands.append(command)
        return _FakeEgoState(state.tire_steering_angle + command.tire_steering_rate * self.dt)


def _session_with_fake_nuplan(monkeypatch: pytest.MonkeyPatch, dt: float, horizon: float) -> Any:
    """Build a _NuPlanSession whose `nuplan` imports resolve to fakes."""
    modules = {
        "nuplan": ModuleType("nuplan"),
        "nuplan.common": ModuleType("nuplan.common"),
        "nuplan.common.actor_state": ModuleType("nuplan.common.actor_state"),
        "nuplan.common.actor_state.dynamic_car_state": ModuleType("dynamic_car_state"),
        "nuplan.common.actor_state.state_representation": ModuleType("state_representation"),
        "nuplan.planning": ModuleType("nuplan.planning"),
        "nuplan.planning.simulation": ModuleType("nuplan.planning.simulation"),
        "nuplan.planning.simulation.trajectory": ModuleType("trajectory"),
        "nuplan.planning.simulation.trajectory.interpolated_trajectory": ModuleType("interp"),
    }
    setattr(
        modules["nuplan.common.actor_state.dynamic_car_state"],
        "DynamicCarState",
        _FakeDynamicCarState,
    )
    setattr(modules["nuplan.common.actor_state.state_representation"], "StateVector2D", _FakeVector)
    setattr(
        modules["nuplan.planning.simulation.trajectory.interpolated_trajectory"],
        "InterpolatedTrajectory",
        list,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    session = _NuPlanSession(
        scenario_loader=lambda: None,
        max_tracked_agents=_MAX_AGENTS,
        simulation_horizon=horizon,
        fixed_delta_seconds=dt,
        reactive_agents=True,
        collision_distance=2.0,
    )
    current = _FakeEgoState(tire_steering_angle=0.0)
    planner_input = SimpleNamespace(history=SimpleNamespace(current_state=(current, None)))
    # pylint: disable=protected-access
    session._simulation = SimpleNamespace(get_planner_input=lambda: planner_input)
    session._motion_model = _FakeMotionModel(dt)
    return session


def test_control_trajectory_commands_the_requested_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The action preset's acceleration reaches the nuPlan motion model.

    Purpose: Validates the longitudinal command is applied, so "accelerate" and "brake"
        presets are not silently identical in the live world.

    Given: A live session driven with the (1.5, 0.0) accelerate preset
    When: the constant-control trajectory is rolled out
    Then: every propagation step carries a +1.5 m/s^2 rear-axle acceleration

    Test type: unit
    """
    session = _session_with_fake_nuplan(monkeypatch, dt=0.1, horizon=0.3)
    # pylint: disable=protected-access
    session._constant_control_trajectory(1.5, 0.0)

    commands = session._motion_model.commands
    assert commands
    assert all(command.rear_axle_acceleration_2d.x == pytest.approx(1.5) for command in commands)
    assert all(command.rear_axle_acceleration_2d.y == pytest.approx(0.0) for command in commands)


def test_control_trajectory_steers_to_the_requested_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The steering preset is a target angle, commanded as the rate that reaches it.

    Purpose: Validates the (acceleration, steering_angle) preset is converted into nuPlan's
        tire *steering rate*, instead of being passed through as if a rate were an angle.

    Given: A live session driven with a 0.3 rad steering preset from a straight wheel
    When: the constant-control trajectory is rolled out over several steps
    Then: the first step commands 0.3 / dt rad/s and later steps hold the angle (rate 0)

    Test type: unit
    """
    session = _session_with_fake_nuplan(monkeypatch, dt=0.1, horizon=0.3)
    # pylint: disable=protected-access
    session._constant_control_trajectory(0.0, 0.3)

    commands = session._motion_model.commands
    assert len(commands) == 3
    assert commands[0].tire_steering_rate == pytest.approx(3.0)  # (0.3 - 0.0) / 0.1
    assert commands[1].tire_steering_rate == pytest.approx(0.0)
    assert commands[2].tire_steering_rate == pytest.approx(0.0)
