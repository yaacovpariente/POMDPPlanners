# SPDX-License-Identifier: MIT

"""Tests for the RacetrackPOMDP forward-only world wrapper.

Most of the suite drives the world against a scripted ``FakeRacetrackSession`` injected by
monkeypatching :meth:`RacetrackPOMDP._get_session`, mirroring the CARLA suite. That keeps
the structural contracts -- one tick per interaction, the cache guards, pickling, the
per-step measurement channels -- testable without a simulator, and makes the outcomes
(crash, off-road, truncation) deterministic rather than something to wait for.

A larger group of tests runs the real HighwayEnv backend, because a fake session would be
asserting its own arithmetic. What those tests pin about the sensors is the *wiring*: that
the emitted reading describes the ego, the road and the traffic the world's own state
vector describes, at the same arclength along the same lane walk. That is the one thing a
stand-in cannot be asked. The reward parity and the matched pair are on the real backend
for the same reason. If the cross-mode trajectory test fails, the comparison the
environment exists to support is invalid, and no assertion here should be loosened to hide
that.

What each sensor *measures* -- the noise widths, the range gate, the occlusion rule, the
ordering, the wrapped heading -- lives in ``test_racetrack_world_sensors.py``, next to the
``WorldSensors`` code it moved out with. Those are properties of a chosen configuration of
vehicles rather than of a live road, and a coasting rollout visits whichever configurations
it happens to visit.
"""

# pylint: disable=protected-access,too-many-lines  # Tests inspect live-session internals

import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (
    RacetrackMetric,
    RacetrackPOMDP,
    RacetrackStepChannel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_POSE_ARCLENGTH,
    EGO_POSE_HEADING,
    EGO_POSE_X,
    EGO_POSE_Y,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    LANE_POSE_ANG,
    LANE_POSE_LAT,
    OBSERVED_EGO_POSE_WIDTH,
    OBSERVED_EGO_SPEED_WIDTH,
    OBSERVED_LANE_POSE_WIDTH,
    ObservationMode,
    RacetrackObservation,
    radial_velocities,
    state_agent_rows,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
    TrackGeometry,
    build_track_geometry,
)

# Index of the (0.0, 0.0) preset: coast, straight ahead. Looked up rather than written as a
# literal, because the shipped table is a 3-by-9 grid of accelerations by steering angles
# and every index moves whenever the steering resolution changes.
_COAST_STRAIGHT = DEFAULT_ACTION_PRESETS.index((0.0, 0.0))
_REPO_ROOT = Path(__file__).resolve().parents[4]

# A detection gate wide enough that nothing on this circuit is ever range-gated out, so a
# test about some other part of the radar is not silently testing the gate as well.
_UNLIMITED_RANGE_M = 1000.0


class FakeRacetrackSession:
    """Scripted stand-in for ``_RacetrackSession`` with the same two-method surface.

    The ego walks forward one metre per tick with its speed equal to the tick index and a
    lane offset of ``-0.5`` metres per tick, so every per-step measurement channel takes a
    distinct, hand-checkable value. One agent slot is filled at a configurable range so
    the near-miss channel can be driven from either side of its threshold. The three
    terminal outcomes fire on request rather than by chance.

    Attributes:
        reset_calls: How many times the world reset the session.
        step_calls: How many simulator ticks the world has requested.
        last_command: The most recent ``(acceleration, steering)`` pair.
        last_seed: The seed passed to the most recent reset, if any.
    """

    def __init__(
        self,
        *,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        freeze_state: bool = False,
        crash_after: Optional[int] = None,
        off_road_after: Optional[int] = None,
        truncate_after: Optional[int] = None,
        agent_range_m: float = 20.0,
        agent_present: bool = True,
    ) -> None:
        self.max_tracked_agents = max_tracked_agents
        self.freeze_state = freeze_state
        self.crash_after = crash_after
        self.off_road_after = off_road_after
        self.truncate_after = truncate_after
        self.agent_range_m = agent_range_m
        self.agent_present = agent_present
        self.reset_calls = 0
        self.step_calls = 0
        self.last_command: Optional[Tuple[float, float]] = None
        self.last_seed: Optional[int] = None
        self._tick = 0

    def state_at(self, tick: int) -> np.ndarray:
        """The scripted state after ``tick`` ticks, for tests to assert against."""
        ego = np.array(
            [float(tick), 0.0, 0.0, float(tick), -0.5 * float(tick), 0.0, 0.0], dtype=float
        )
        rows = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        if self.agent_present:
            rows[0] = [1.0, self.agent_range_m, 0.0, 0.0, 0.0]
        return np.concatenate([ego, rows.reshape(-1)])

    def observation_at(self, tick: int) -> RacetrackObservation:
        """The scripted reading after ``tick`` ticks: every channel filled with the tick.

        The values are not a plausible sensor output and are not meant to be -- these tests
        are about which reading is served for which tick, so what matters is that the five
        channels are distinguishable per tick and shaped as the world promises.
        """
        value = float(tick)
        return RacetrackObservation(
            ego_pose=np.full(OBSERVED_EGO_POSE_WIDTH, value, dtype=np.float32),
            ego_speed=np.full(OBSERVED_EGO_SPEED_WIDTH, value, dtype=np.float32),
            lane_pose=np.full(OBSERVED_LANE_POSE_WIDTH, value, dtype=np.float32),
            curvature_ahead=np.full(len(DEFAULT_CURVATURE_LOOKAHEAD_M), value, dtype=np.float32),
            detections=np.full(
                (self.max_tracked_agents, DETECTION_SLOT_WIDTH), value, dtype=np.float32
            ),
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, RacetrackObservation]:
        self.reset_calls += 1
        self.last_seed = seed
        self._tick = 0
        return self.state_at(0), self.observation_at(0)

    def step(self, acceleration: float, steering: float) -> Dict[str, Any]:
        self.step_calls += 1
        self.last_command = (acceleration, steering)
        self._tick += 1
        state_tick = 0 if self.freeze_state else self._tick
        return {
            "state": self.state_at(state_tick),
            "observation": self.observation_at(self._tick),
            "crashed": self._fired(self.crash_after),
            "off_road": self._fired(self.off_road_after),
            "truncated": self._fired(self.truncate_after),
        }

    def _fired(self, threshold: Optional[int]) -> bool:
        return threshold is not None and self._tick >= threshold


def _install(monkeypatch: pytest.MonkeyPatch, session: FakeRacetrackSession) -> None:
    """Point every RacetrackPOMDP at ``session`` instead of a live HighwayEnv."""
    monkeypatch.setattr(RacetrackPOMDP, "_get_session", lambda self: session)


def _reset_world(**kwargs: Any) -> RacetrackPOMDP:
    """A RacetrackPOMDP over the installed session, already reset to its initial state."""
    world = RacetrackPOMDP(discount_factor=0.95, **kwargs)
    world.initial_state_dist().sample()
    return world


def _quiet_world(**kwargs: Any) -> RacetrackPOMDP:
    """A live world with every sensor width set to zero, so its readings are the truth.

    A sensor test either asks what the reading measures or how wrong it is, and the two need
    opposite settings: the first needs the noise gone so a mismatch is a wrong quantity
    rather than a draw, the second sets its own widths. This builds the first.
    """
    silent: Dict[str, float] = {
        "ego_position_std_m": 0.0,
        "ego_heading_std_rad": 0.0,
        "ego_arclength_std_m": 0.0,
        "lane_lateral_std_m": 0.0,
        "lane_heading_std_rad": 0.0,
        "curvature_std_1pm": 0.0,
        "detection_position_std_m": 0.0,
        "detection_velocity_std": 0.0,
    }
    for name, width in silent.items():
        kwargs.setdefault(name, width)
    return RacetrackPOMDP(discount_factor=0.95, **kwargs)


def _drive(
    world: RacetrackPOMDP, start: np.ndarray, steps: int, action: int = _COAST_STRAIGHT
) -> Iterator[Tuple[np.ndarray, Any]]:
    """Step a live world, yielding each successor state and the reading it produced.

    Stops early at termination, which coasting on this circuit reaches in about twenty
    steps, so every caller has to tolerate a short rollout.
    """
    state = start
    for _ in range(steps):
        world.reward(state, action)
        state = world.sample_next_state(state, action)
        yield state, world.sample_observation(state, action)
        if world.is_terminal(state):
            return


def _ego_lane_geometry(world: RacetrackPOMDP) -> TrackGeometry:
    """The curvature profile of the lane the live ego is on, built independently.

    Built here from the road network rather than read off the session, so the curvature
    tests compare the emitted channel against the track and not against the same cached
    object the channel came from. Call it straight after the reset the world's own
    arclength base was fixed at.
    """
    unwrapped = world._get_session()._env.unwrapped
    return build_track_geometry(unwrapped.road.network, unwrapped.vehicle.lane_index)


def _present_agent_rows(state: np.ndarray, max_tracked_agents: int) -> np.ndarray:
    """The occupied agent slots of a state, as a ``(n, 5)`` block."""
    rows = state_agent_rows(state, max_tracked_agents)
    return rows[rows[:, AGENT_PRESENT] > 0.5]


@pytest.fixture(name="fake_session")
def _fake_session(monkeypatch: pytest.MonkeyPatch) -> FakeRacetrackSession:
    """A scripted session installed on RacetrackPOMDP for the duration of the test."""
    session = FakeRacetrackSession()
    _install(monkeypatch, session)
    return session


@pytest.fixture(name="world")
def _world(fake_session: FakeRacetrackSession) -> RacetrackPOMDP:
    """A RacetrackPOMDP over the scripted session, reset to its live initial state."""
    del fake_session  # Requested for its installation side effect; found via _get_session.
    return _reset_world()


@pytest.fixture(name="headless")
def _headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require the real backend and keep pygame off any display."""
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


# ── Construction and static surface ─────────────────────────────────────


def test_space_info_is_discrete_actions_over_continuous_observations(
    world: RacetrackPOMDP,
) -> None:
    """The world offers preset control indices and array observations.

    Purpose: Validates the fixed SpaceInfo, which decides which planners can be paired
        with this world at all.

    Given: A RacetrackPOMDP over the scripted session
    When: space_info is inspected
    Then: The action space is DISCRETE and the observation space is CONTINUOUS

    Test type: unit
    """
    assert world.space_info.action_space == SpaceType.DISCRETE
    assert world.space_info.observation_space == SpaceType.CONTINUOUS


@pytest.mark.parametrize("max_tracked_agents", [1, 2, 4, 7])
def test_state_width_is_the_ego_block_plus_one_slot_per_tracked_agent(
    monkeypatch: pytest.MonkeyPatch, max_tracked_agents: int
) -> None:
    """The state width follows the documented layout for any agent count.

    Purpose: Validates the state layout the world, the planner's model and the belief all
        reshape against; a width mismatch would scramble the agent slots silently.

    Given: Worlds configured for a range of tracked-agent counts
    When: state_width is read and a live state is produced
    Then: Both equal EGO_STATE_WIDTH + 5 * max_tracked_agents

    Test type: unit
    """
    session = FakeRacetrackSession(max_tracked_agents=max_tracked_agents)
    _install(monkeypatch, session)
    world = _reset_world(max_tracked_agents=max_tracked_agents)

    expected = EGO_STATE_WIDTH + AGENT_SLOT_WIDTH * max_tracked_agents
    assert world.state_width == expected
    assert world._live_state is not None
    assert world._live_state.shape == (expected,)


def test_zero_tracked_agents_is_rejected() -> None:
    """A world that tracks nobody is refused at construction.

    Purpose: Validates the agent-count guard. Zero slots would leave the near-miss channel
        permanently zero and the belief with nothing to track, which is a silent
        misconfiguration rather than a useful degenerate case.

    Given: max_tracked_agents of 0
    When: A RacetrackPOMDP is constructed
    Then: ValueError is raised naming the minimum

    Test type: unit
    """
    with pytest.raises(ValueError, match="max_tracked_agents must be at least 1"):
        RacetrackPOMDP(discount_factor=0.95, max_tracked_agents=0)


def test_get_actions_indexes_the_control_presets(world: RacetrackPOMDP) -> None:
    """Actions are indices into the preset table, not control values.

    Purpose: Validates the shared action vocabulary. The world and the planner's model
        agree by indexing one table, so an action must be a position in it.

    Given: A world with the default 3x3 throttle-by-steer preset grid
    When: get_actions is called
    Then: It returns 0..8, and every entry indexes a preset

    Test type: unit
    """
    actions = world.get_actions()

    assert actions == list(range(len(world.action_presets)))
    assert [world.action_presets[a] for a in actions] == list(world.action_presets)


def test_custom_action_presets_replace_the_default_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied preset table becomes the action vocabulary.

    Purpose: Validates that the preset table is configurable and that the chosen index is
        the command actually handed to the simulator.

    Given: A world built with two custom presets
    When: Action 1 is taken
    Then: get_actions has two entries and the session received the second preset

    Test type: unit
    """
    session = FakeRacetrackSession()
    _install(monkeypatch, session)
    world = _reset_world(action_presets=[(0.5, 0.0), (-0.5, 0.25)])

    world.sample_next_state(world._live_state, 1)

    assert world.get_actions() == [0, 1]
    assert session.last_command == (-0.5, 0.25)


def test_default_name_records_the_observation_mode() -> None:
    """The two arms get distinguishable default names.

    Purpose: Validates the naming, which is how the two arms of one comparison are told
        apart in results and caches.

    Given: A world in each observation mode, with no explicit name
    When: The names are read
    Then: Each carries its own mode and the two differ

    Test type: unit
    """
    mdp = RacetrackPOMDP(discount_factor=0.95, observation_mode=ObservationMode.MDP)
    pomdp = RacetrackPOMDP(discount_factor=0.95, observation_mode=ObservationMode.POMDP)

    assert mdp.name == "RacetrackPOMDP-mdp"
    assert pomdp.name == "RacetrackPOMDP-pomdp"


# ── One tick per interaction ────────────────────────────────────────────


def test_reward_and_next_state_share_a_single_simulator_tick(
    world: RacetrackPOMDP, fake_session: FakeRacetrackSession
) -> None:
    """The episode loop's separate queries advance the world exactly once.

    Purpose: Validates the step-once cache. The loop asks for the reward and the successor
        through two calls; a second tick would double the world's rate and desynchronise
        the reward from the state it is meant to score.

    Given: A world reset to its live state
    When: reward, then sample_next_state, then sample_observation are called for that step
    Then: The session stepped once, and all three describe the same transition

    Test type: unit
    """
    state = world._live_state
    before = fake_session.step_calls

    reward = world.reward(state, _COAST_STRAIGHT)
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    observation = world.sample_observation(next_state, _COAST_STRAIGHT)

    assert fake_session.step_calls - before == 1
    assert np.array_equal(next_state, fake_session.state_at(1))
    assert world.is_equal_observation(observation, fake_session.observation_at(1))
    # lateral -0.5 m, no control effort, no crash: (1/(1+4*0.25) + 1) / 2.
    assert reward == pytest.approx(0.75)


def test_next_state_before_reward_still_costs_one_tick(
    world: RacetrackPOMDP, fake_session: FakeRacetrackSession
) -> None:
    """Either query may come first; the cache is filled by whichever does.

    Purpose: Validates that the cache is not order-dependent, since the episode loop's
        call order is not part of this environment's contract.

    Given: A world reset to its live state
    When: sample_next_state is called before reward for the same (state, action)
    Then: The session still stepped once and both agree on the transition

    Test type: unit
    """
    state = world._live_state
    before = fake_session.step_calls

    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    reward = world.reward(state, _COAST_STRAIGHT)

    assert fake_session.step_calls - before == 1
    assert np.array_equal(next_state, fake_session.state_at(1))
    assert reward == pytest.approx(0.75)


def test_repeating_a_role_forces_a_new_tick_even_when_the_state_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two successor requests are two steps, however identical the states look.

    Purpose: Validates that the cache is keyed on the requesting role and not on the state
        alone. A state-only key would treat the second request as a repeat of the first
        and serve the stale tick. That is not a hypothetical here: a crashed vehicle is
        braked to rest, so ``next_state == state`` is a perfectly ordinary reading, and a
        world that stopped advancing on it would freeze the episode.

    Given: A session scripted to return an unchanged state on every tick
    When: sample_next_state is called twice for that same (state, action)
    Then: The session stepped twice, and the returned states are indeed identical

    Test type: unit
    """
    session = FakeRacetrackSession(freeze_state=True)
    _install(monkeypatch, session)
    world = _reset_world()
    state = world._live_state

    first = world.sample_next_state(state, _COAST_STRAIGHT)
    second = world.sample_next_state(state, _COAST_STRAIGHT)

    assert session.step_calls == 2
    assert np.array_equal(first, second)
    assert np.array_equal(first, session.state_at(0))


def test_observation_query_adds_no_tick(
    world: RacetrackPOMDP, fake_session: FakeRacetrackSession
) -> None:
    """Reading the observation is a cache lookup, never a simulator step.

    Purpose: Validates that the observation is served from the completed tick, so asking
        for it repeatedly cannot advance the world.

    Given: A world that has taken one step
    When: sample_observation is called several times for that successor
    Then: The step count is unchanged and every reading is identical

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    after_step = fake_session.step_calls

    readings = [world.sample_observation(next_state, _COAST_STRAIGHT) for _ in range(3)]

    assert fake_session.step_calls == after_step
    assert all(world.is_equal_observation(reading, readings[0]) for reading in readings)


# ── Forward-only guards ─────────────────────────────────────────────────


def test_sample_next_state_rejects_multiple_samples(world: RacetrackPOMDP) -> None:
    """A forward-only world cannot draw several successors.

    Purpose: Validates the n_samples guard on the transition. Silently returning one
        sample where several were asked for would corrupt a particle update.

    Given: A world at its live state
    When: sample_next_state is asked for two samples
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="only supports n_samples=1"):
        world.sample_next_state(world._live_state, _COAST_STRAIGHT, n_samples=2)


def test_sample_observation_rejects_multiple_samples(world: RacetrackPOMDP) -> None:
    """A forward-only world cannot draw several observations.

    Purpose: Validates the n_samples guard on the observation, for the same reason as the
        transition guard.

    Given: A world that has taken one step
    When: sample_observation is asked for two samples
    Then: ValueError is raised

    Test type: unit
    """
    next_state = world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    with pytest.raises(ValueError, match="only supports n_samples=1"):
        world.sample_observation(next_state, _COAST_STRAIGHT, n_samples=2)


def test_stepping_from_a_state_other_than_the_live_one_raises(world: RacetrackPOMDP) -> None:
    """The world refuses to be treated as a generative model.

    Purpose: Validates the central forward-only guard. Quietly stepping from the live
        state when an arbitrary one was requested would let a planner search against the
        real world and corrupt the episode it is supposed to be planning for.

    Given: A state that is not the world's live state
    When: sample_next_state is called from it
    Then: RuntimeError is raised, pointing at the separate model environment

    Test type: unit
    """
    imposter = np.zeros(world.state_width)
    imposter[0] = 99.0

    with pytest.raises(RuntimeError, match="forward-only world environment"):
        world.sample_next_state(imposter, _COAST_STRAIGHT)


def test_reward_from_a_state_other_than_the_live_one_raises(world: RacetrackPOMDP) -> None:
    """The reward is not queryable off-trajectory either.

    Purpose: Validates that the same guard covers the reward path, which shares the
        stepping helper with the transition path.

    Given: A state that is not the world's live state
    When: reward is called for it
    Then: RuntimeError is raised

    Test type: unit
    """
    imposter = np.full(world.state_width, 3.0)

    with pytest.raises(RuntimeError, match="forward-only world environment"):
        world.reward(imposter, _COAST_STRAIGHT)


def test_observation_for_a_stale_successor_raises(world: RacetrackPOMDP) -> None:
    """Only the reading just produced can be served.

    Purpose: Validates the observation guard. After the world has moved on, returning the
        latest reading for an older successor would silently mislabel it.

    Given: A world that has taken two steps
    When: sample_observation is asked for the first step's successor
    Then: RuntimeError is raised

    Test type: unit
    """
    first = world.sample_next_state(world._live_state, _COAST_STRAIGHT)
    world.sample_next_state(first, _COAST_STRAIGHT)

    with pytest.raises(RuntimeError, match="other than the live one"):
        world.sample_observation(first, _COAST_STRAIGHT)


def test_is_terminal_for_a_state_other_than_the_live_one_raises(world: RacetrackPOMDP) -> None:
    """Terminality is only known for the state the world is actually in.

    Purpose: Validates the terminality guard. A forward-only world has no way to decide
        whether some other state is terminal, and answering False would be a guess.

    Given: A state that is not the world's live state
    When: is_terminal is called for it
    Then: RuntimeError is raised

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="other than the live world state"):
        world.is_terminal(np.ones(world.state_width))


def test_transition_log_probability_is_not_implemented(world: RacetrackPOMDP) -> None:
    """The world exposes no transition density.

    Purpose: Validates that belief updates cannot be run against the world by accident;
        they belong on the planner's model environment.

    Given: A world at its live state
    When: transition_log_probability is called
    Then: NotImplementedError is raised, naming the model environment

    Test type: unit
    """
    with pytest.raises(NotImplementedError, match="no transition"):
        world.transition_log_probability(world._live_state, _COAST_STRAIGHT, world._live_state)


def test_observation_log_probability_is_not_implemented(world: RacetrackPOMDP) -> None:
    """The world exposes no observation density.

    Purpose: Validates the same separation for the observation side of a belief update.

    Given: A world at its live state
    When: observation_log_probability is called
    Then: NotImplementedError is raised, naming the model environment

    Test type: unit
    """
    with pytest.raises(NotImplementedError, match="no observation"):
        world.observation_log_probability(world._live_state, _COAST_STRAIGHT, np.zeros(3))


# ── Terminal outcomes ───────────────────────────────────────────────────


@pytest.mark.parametrize("outcome_kwarg", ["crash_after", "off_road_after", "truncate_after"])
def test_each_terminal_outcome_ends_the_episode(
    monkeypatch: pytest.MonkeyPatch, outcome_kwarg: str
) -> None:
    """Crashing, leaving the road and running out of time each terminate.

    Purpose: Validates all three terminal conditions separately. They are kept apart
        because the MDP-versus-POMDP comparison is about *why* a planner failed, so a
        world that ended the episode for only some of them would hide a failure mode.

    Given: A session scripted to produce exactly one of the three outcomes on tick 1
    When: One step is taken
    Then: is_terminal is True for the resulting live state

    Test type: unit
    """
    fires_on_tick_one: Dict[str, Optional[int]] = {
        "crash_after": None,
        "off_road_after": None,
        "truncate_after": None,
    }
    fires_on_tick_one[outcome_kwarg] = 1
    session = FakeRacetrackSession(
        crash_after=fires_on_tick_one["crash_after"],
        off_road_after=fires_on_tick_one["off_road_after"],
        truncate_after=fires_on_tick_one["truncate_after"],
    )
    _install(monkeypatch, session)
    world = _reset_world()

    next_state = world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    assert world.is_terminal(next_state) is True


def test_a_clean_step_is_not_terminal(world: RacetrackPOMDP) -> None:
    """A step with none of the three outcomes leaves the episode running.

    Purpose: Validates the negative case, so the terminal tests above are not passing for
        an unrelated reason.

    Given: A session scripted to produce no terminal outcome
    When: One step is taken
    Then: is_terminal is False for the resulting live state

    Test type: unit
    """
    next_state = world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    assert world.is_terminal(next_state) is False


def test_reset_clears_a_previous_terminal_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new episode starts non-terminal even after the last one crashed.

    Purpose: Validates that reset clears the terminal flag and the cache, so a reused
        world does not report the previous episode's ending on the new one's first state.

    Given: A world that has already reached a crash
    When: initial_state_dist is sampled again
    Then: The fresh live state is not terminal and no cached tick remains

    Test type: unit
    """
    session = FakeRacetrackSession(crash_after=1)
    _install(monkeypatch, session)
    world = _reset_world()
    world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    fresh = world.initial_state_dist().sample()[0]

    assert world.is_terminal(fresh) is False
    assert world._pending is None


@pytest.mark.parametrize("terminate_off_road,expected", [(True, True), (False, False)])
def test_terminate_off_road_decides_whether_leaving_the_road_ends_the_episode(
    monkeypatch: pytest.MonkeyPatch, terminate_off_road: bool, expected: bool
) -> None:
    """The off-road ending follows the constructor flag rather than being hard-coded.

    Purpose: Validates that ``terminate_off_road`` reaches this adapter's own termination
        decision and not only the simulator config. highway-env gates its ``_is_terminated``
        on the same flag, so a world that always ended the episode off-road would report the
        run over while the simulator kept driving -- and every metric downstream would be
        cut short at a step that was not terminal.

    Given: Two worlds scripted to leave the road on the first tick, one built with
        terminate_off_road True and one with it False
    When: One step is taken in each
    Then: Only the flagged world reports the resulting state as terminal, and both record
        the off-road event on its measurement channel either way

    Test type: unit
    """
    session = FakeRacetrackSession(off_road_after=1)
    _install(monkeypatch, session)
    world = _reset_world(terminate_off_road=terminate_off_road)
    state = world._live_state

    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert world.is_terminal(next_state) is expected
    assert info[RacetrackStepChannel.OFF_ROAD.value] == 1.0


def test_a_crash_still_ends_the_episode_when_off_road_termination_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling the off-road ending leaves the other two terminal conditions alone.

    Purpose: Guards the flag against being read as a blanket "never terminate". A crash and
        the time limit are separate endings that highway-env applies regardless, so a
        world that suppressed them along with the off-road ending would run every episode
        to its step budget and silently destroy the collision metrics.

    Given: A world with terminate_off_road False, scripted to crash on the first tick
    When: One step is taken
    Then: The resulting state is still terminal

    Test type: unit
    """
    session = FakeRacetrackSession(crash_after=1)
    _install(monkeypatch, session)
    world = _reset_world(terminate_off_road=False)

    next_state = world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    assert world.is_terminal(next_state) is True


# ── Reset, observations and hashing ─────────────────────────────────────


def test_initial_state_sampling_resets_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drawing an initial state is what starts an episode in the simulator.

    Purpose: Validates that the world's initial state comes from a real reset rather than
        a remembered one, which is what makes each episode independent.

    Given: A world whose session has never been reset
    When: initial_state_dist().sample() is called
    Then: The session reset once and the drawn state is the world's live state

    Test type: unit
    """
    session = FakeRacetrackSession()
    _install(monkeypatch, session)
    world = RacetrackPOMDP(discount_factor=0.95)

    state = world.initial_state_dist().sample()[0]

    assert session.reset_calls == 1
    assert np.array_equal(state, session.state_at(0))
    assert world._live_state is not None
    assert np.array_equal(state, world._live_state)


def test_the_configured_seed_is_passed_to_the_first_reset_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed opens the run; later episodes continue the same stream.

    Purpose: Validates the seeding policy. Re-seeding every episode would make every
        episode in a run identical, which would silently collapse the sample size.

    Given: A world constructed with a seed
    When: Two episodes are started
    Then: The first reset carried the seed and the second carried none

    Test type: unit
    """
    session = FakeRacetrackSession()
    _install(monkeypatch, session)
    world = RacetrackPOMDP(discount_factor=0.95, seed=17)

    world.initial_state_dist().sample()
    assert session.last_seed == 17

    world.initial_state_dist().sample()
    assert session.last_seed is None


def test_initial_observation_is_a_copy_the_caller_cannot_corrupt(
    world: RacetrackPOMDP,
) -> None:
    """Mutating a drawn observation leaves the world's own reading intact.

    Purpose: Validates the defensive copy. The world hands out its cached arrays, so
        without a copy a caller that normalised or scaled a channel in place would corrupt
        the world's state. The copy has to reach every channel of the named tuple, not just
        the first one, which is why each is mutated here.

    Given: A world reset to its live state
    When: An initial observation is drawn and every channel of it is mutated
    Then: A freshly drawn observation is unaffected

    Test type: unit
    """
    drawn = world.initial_observation_dist().sample()[0]
    for channel in drawn:
        channel[...] = 99.0

    redrawn = world.initial_observation_dist().sample()[0]

    assert world.is_equal_observation(drawn, redrawn) is False
    assert max(float(np.max(channel)) for channel in redrawn) == 0.0


def test_initial_observation_resets_when_no_episode_has_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for the first observation before the first state still works.

    Purpose: Validates the lazy reset on the observation path, so callers are not required
        to draw a state first.

    Given: A world whose session has never been reset
    When: initial_observation_dist().sample() is called
    Then: The session reset and the observation is the reset reading

    Test type: unit
    """
    session = FakeRacetrackSession()
    _install(monkeypatch, session)
    world = RacetrackPOMDP(discount_factor=0.95)

    observation = world.initial_observation_dist().sample()[0]

    assert session.reset_calls == 1
    assert world.is_equal_observation(observation, session.observation_at(0))


def test_observation_equality_and_hashing_agree_on_every_channel(
    world: RacetrackPOMDP, fake_session: FakeRacetrackSession
) -> None:
    """Equal readings compare equal and hash alike; a change in any channel splits them.

    Purpose: Validates the observation identity used to key planner tree nodes. The reading
        is four separate arrays now, so a hash built from only one of them -- or from a
        stacked view that quietly dropped a channel -- would merge branches that saw
        different roads or different traffic.

    Given: One scripted reading, an identical copy, and one copy perturbed in each channel
        in turn
    When: is_equal_observation and hash_observation are applied
    Then: The identical pair matches on both, and every perturbed copy on neither

    Test type: unit
    """
    reading = fake_session.observation_at(1)
    same = fake_session.observation_at(1)

    assert world.is_equal_observation(reading, same) is True
    assert world.hash_observation(reading) == world.hash_observation(same)

    for index in range(len(reading)):
        channels = [channel.copy() for channel in reading]
        channels[index].reshape(-1)[0] += 1.0
        other = RacetrackObservation(*channels)

        assert world.is_equal_observation(reading, other) is False
        assert world.hash_observation(reading) != world.hash_observation(other)


# ── Serialization ───────────────────────────────────────────────────────


def test_pickling_drops_the_live_session_and_its_caches(
    world: RacetrackPOMDP,
) -> None:
    """A pickled world carries configuration, never a live simulator handle.

    Purpose: Validates the serialization contract that lets a world be shipped to a worker
        process. The session is not picklable, and a restored cache would describe a tick
        that the restored process never took.

    Given: A world that has taken a step, so every cache is populated
    When: It is pickled and unpickled
    Then: The session, live state, latest observation and cached tick are all cleared

    Test type: unit
    """
    world.sample_next_state(world._live_state, _COAST_STRAIGHT)

    restored = pickle.loads(pickle.dumps(world))

    assert restored._session is None
    assert restored._live_state is None
    assert restored._latest_obs is None
    assert restored._pending is None
    assert restored._served_roles == set()
    assert restored._terminated is False


def test_pickling_preserves_the_configuration(world: RacetrackPOMDP) -> None:
    """The restored world is configured exactly as the original was.

    Purpose: Validates that what survives pickling is the whole public configuration, so a
        worker runs the arm the parent process intended rather than a default one. The
        sensor widths are checked alongside the simulator config because they live only on
        this object -- highway-env knows nothing about them -- so a worker that lost them
        would emit a differently-corrupted reading than the one the planner's model assumes.

    Given: A world in a known observation mode
    When: It is pickled and unpickled
    Then: name, observation_mode, the simulator config and every sensor width are unchanged

    Test type: unit
    """
    restored = pickle.loads(pickle.dumps(world))

    assert restored.name == world.name
    assert restored.observation_mode == world.observation_mode
    assert restored.simulator_config == world.simulator_config
    assert restored.ego_position_std_m == world.ego_position_std_m
    assert restored.ego_heading_std_rad == world.ego_heading_std_rad
    assert restored.ego_arclength_std_m == world.ego_arclength_std_m
    assert restored.curvature_lookahead_m == world.curvature_lookahead_m
    assert restored.curvature_std_1pm == world.curvature_std_1pm
    assert restored.max_detection_range_m == world.max_detection_range_m
    assert restored.detection_position_std_m == world.detection_position_std_m
    assert restored.detection_velocity_std == world.detection_velocity_std
    assert restored.blocker_half_width_m == world.blocker_half_width_m


def test_pickled_world_reopens_a_session_on_demand(
    world: RacetrackPOMDP, fake_session: FakeRacetrackSession
) -> None:
    """A restored world starts a fresh episode when it is next used.

    Purpose: Validates that clearing the live handle does not leave the world unusable,
        which is the whole point of clearing it lazily rather than refusing to pickle.

    Given: A world that has been pickled and unpickled
    When: An initial state is drawn from the restored world
    Then: The session is reset again and the state is the episode's first

    Test type: unit
    """
    restored = pickle.loads(pickle.dumps(world))
    before = fake_session.reset_calls

    state = restored.initial_state_dist().sample()[0]

    assert fake_session.reset_calls == before + 1
    assert np.array_equal(state, fake_session.state_at(0))


# ── Per-step measurement channels ───────────────────────────────────────


def test_step_info_is_empty_before_any_step(world: RacetrackPOMDP) -> None:
    """A world that has not stepped reports no measurements.

    Purpose: Validates that the channel reports nothing rather than inventing zeros, since
        a zero would be aggregated as a real observation of "no crash".

    Given: A freshly reset world
    When: step_info is queried for its live state
    Then: The mapping is empty

    Test type: unit
    """
    assert world.step_info(world._live_state, _COAST_STRAIGHT, world._live_state) == {}


def test_step_info_reports_every_channel_after_a_step(
    world: RacetrackPOMDP,
) -> None:
    """One tick fills every declared measurement channel with its own value.

    Purpose: Validates the measurement derivation end to end against a scripted tick whose
        expected values can be worked out by hand.

    Given: A clean step to a state 0.5 m off centre at 1 m/s with the nearest agent 20 m
        away, beyond the 5 m near-miss threshold
    When: step_info is queried for that transition
    Then: Every channel is present with the hand-computed values, and the collision-speed
        channel reads 0.0 because this step did not crash

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)

    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert info == {
        RacetrackStepChannel.CRASHED.value: 0.0,
        RacetrackStepChannel.OFF_ROAD.value: 0.0,
        RacetrackStepChannel.TIME_LIMIT.value: 0.0,
        RacetrackStepChannel.ABS_LANE_OFFSET_M.value: 0.5,
        RacetrackStepChannel.SPEED_MPS.value: 1.0,
        RacetrackStepChannel.COLLISION_SPEED_MPS.value: 0.0,
        RacetrackStepChannel.NEAR_MISS.value: 0.0,
    }


def test_collision_speed_channel_carries_the_speed_of_the_crashing_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash records how fast the ego was going, not merely that it happened.

    Purpose: Validates the severity channel that separates a scrape from a high-speed
        impact. A collision rate alone cannot distinguish them, and the mean speed over an
        episode averages the impact away entirely.

    Given: A session scripted to crash on its second tick, where the scripted ego speed
        equals the tick index
    When: step_info is read on the clean first step and again on the crashing step
    Then: The channel reads 0.0 while no crash has happened and the ego's speed on the
        step the crash is reported, so a MAX reduction recovers the impact speed

    Test type: unit
    """
    session = FakeRacetrackSession(crash_after=2)
    _install(monkeypatch, session)
    world = _reset_world()

    state = world._live_state
    clean_next = world.sample_next_state(state, _COAST_STRAIGHT)
    clean = world.step_info(state, _COAST_STRAIGHT, clean_next)

    crash_next = world.sample_next_state(clean_next, _COAST_STRAIGHT)
    crashed = world.step_info(clean_next, _COAST_STRAIGHT, crash_next)

    assert clean[RacetrackStepChannel.CRASHED.value] == 0.0
    assert clean[RacetrackStepChannel.COLLISION_SPEED_MPS.value] == 0.0
    assert crashed[RacetrackStepChannel.CRASHED.value] == 1.0
    assert (
        crashed[RacetrackStepChannel.COLLISION_SPEED_MPS.value]
        == crashed[RacetrackStepChannel.SPEED_MPS.value]
    )
    assert crashed[RacetrackStepChannel.COLLISION_SPEED_MPS.value] > 0.0


def test_collision_speed_metric_reduces_with_max_over_the_episode(
    world: RacetrackPOMDP,
) -> None:
    """The impact speed survives aggregation instead of being averaged away.

    Purpose: Validates the reduction choice. Most steps of a crashing episode report 0.0
        on this channel, so a MEAN would report a speed no collision ever occurred at;
        MAX picks the single impact out.

    Given: The declared metric specs
    When: The collision-speed spec is located
    Then: It reduces with MAX over the episode

    Test type: unit
    """
    spec = next(
        s for s in world.get_metric_specs() if s.name == RacetrackMetric.COLLISION_SPEED_MPS.value
    )

    assert spec.channel == RacetrackStepChannel.COLLISION_SPEED_MPS.value
    assert spec.per_episode is EpisodeReduction.MAX


def test_step_info_reports_a_crash_on_the_channel_that_measures_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed tick raises the crash channel and leaves the others alone.

    Purpose: Validates that the terminal outcomes reach the measurement channels
        separately, which is what keeps "hit a car" distinguishable from "left the road"
        in the reported metrics.

    Given: A session scripted to crash on the first tick
    When: step_info is queried for that transition
    Then: Only the crash channel is raised among the three outcome channels

    Test type: unit
    """
    session = FakeRacetrackSession(crash_after=1)
    _install(monkeypatch, session)
    world = _reset_world()
    state = world._live_state

    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert info[RacetrackStepChannel.CRASHED.value] == 1.0
    assert info[RacetrackStepChannel.OFF_ROAD.value] == 0.0
    assert info[RacetrackStepChannel.TIME_LIMIT.value] == 0.0


def test_step_info_is_empty_on_the_terminal_bookkeeping_call(
    world: RacetrackPOMDP,
) -> None:
    """The episode loop's final call reports nothing rather than raising.

    Purpose: Validates the terminal-step contract. The loop records the last state with no
        action and no successor; measuring a transition there would count an event that
        never happened.

    Given: A world that has taken a step
    When: step_info is called with action and next_state both None
    Then: The mapping is empty

    Test type: unit
    """
    state = world._live_state
    world.sample_next_state(state, _COAST_STRAIGHT)

    assert world.step_info(state, None, None) == {}


def test_step_info_is_empty_for_a_successor_other_than_the_cached_one(
    world: RacetrackPOMDP,
) -> None:
    """A request that does not refer to the cached tick is declined.

    Purpose: Validates that measurements are never mislabelled onto another transition,
        which a forward-only world cannot re-measure to check.

    Given: A world that has taken a step
    When: step_info is queried with an unrelated successor
    Then: The mapping is empty

    Test type: unit
    """
    state = world._live_state
    world.sample_next_state(state, _COAST_STRAIGHT)

    assert world.step_info(state, _COAST_STRAIGHT, np.full(world.state_width, 5.0)) == {}


def test_step_info_is_empty_for_a_predecessor_or_action_other_than_the_cached_one(
    world: RacetrackPOMDP,
) -> None:
    """The successor alone is not enough to identify the cached tick.

    Purpose: Validates the other two thirds of the cache check. Matching on the successor
        alone is too weak precisely where this world is unusual: a crashed vehicle is
        braked to rest, so a state can equal its own successor, and a later request
        carrying that array as its predecessor would then be handed the earlier step's
        measurements under a transition that never produced them.

    Given: A world that has taken one step
    When: step_info is queried for the cached successor but with a different predecessor,
        and again with a different action
    Then: Both requests report nothing, while the correct triple still reports the step

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    other_state = np.full(world.state_width, 5.0)
    other_action = _COAST_STRAIGHT + 1

    assert world.step_info(other_state, _COAST_STRAIGHT, next_state) == {}
    assert world.step_info(state, other_action, next_state) == {}
    assert world.step_info(state, _COAST_STRAIGHT, next_state)


def test_step_info_values_are_plain_floats(world: RacetrackPOMDP) -> None:
    """Every channel value is a Python float, not a NumPy scalar.

    Purpose: Validates the value type the shared step-info contract checks. NumPy scalars
        survive most assertions but change how the values pickle and compare, and the
        cross-environment contract suite asserts isinstance(value, float) directly.

    Given: A world that has taken a step
    When: step_info is queried for that transition
    Then: Every value is a float and none is a NumPy floating type

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)

    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert info
    for value in info.values():
        assert isinstance(value, float)
        assert not isinstance(value, np.floating)


def test_step_info_survives_a_pickle_round_trip(world: RacetrackPOMDP) -> None:
    """Measurements travel back from a worker process unchanged.

    Purpose: Validates that the reported mapping is plainly picklable, because it rides
        home inside a pickled History from a distributed run.

    Given: A world that has taken a step
    When: The reported mapping is pickled and unpickled
    Then: It is equal to the original

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert pickle.loads(pickle.dumps(info)) == info


def test_step_info_returns_a_detached_copy(world: RacetrackPOMDP) -> None:
    """A caller mutating the reported mapping cannot corrupt the cache.

    Purpose: Validates the defensive copy on the measurement path, so an aggregator that
        adds or rescales a key in place does not change what the world reports next.

    Given: A world that has taken a step
    When: The reported mapping is mutated and then re-queried
    Then: The second reading is unaffected

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)

    first = world.step_info(state, _COAST_STRAIGHT, next_state)
    first[RacetrackStepChannel.SPEED_MPS.value] = 99.0
    second = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert second[RacetrackStepChannel.SPEED_MPS.value] == 1.0


def test_step_info_consumes_no_randomness(world: RacetrackPOMDP) -> None:
    """Measuring a step leaves both global RNG streams untouched.

    Purpose: Validates the invariant that makes the hook safe to call from inside the
        episode loop. A single draw here would advance the stream and shift every later
        transition and observation, breaking seeded reproducibility far beyond metrics.

    Given: A world that has taken a step, with numpy and stdlib RNGs seeded
    When: step_info is called many times
    Then: Both RNG states are byte-identical to before

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    np.random.seed(4242)
    random.seed(4242)
    before = np.random.get_state(legacy=True)
    before_stdlib = random.getstate()

    for _ in range(100):
        world.step_info(state, _COAST_STRAIGHT, next_state)

    after = np.random.get_state(legacy=True)
    assert random.getstate() == before_stdlib
    assert isinstance(before, tuple) and isinstance(after, tuple)
    # The Mersenne Twister key is an ndarray, so the tuples cannot be compared
    # directly; the position counter is what a stray draw moves.
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "agent_range_m,agent_present,expected",
    [(3.0, True, 1.0), (5.0, True, 1.0), (5.5, True, 0.0), (20.0, True, 0.0), (1.0, False, 0.0)],
)
def test_near_miss_fires_only_for_a_present_agent_within_range(
    monkeypatch: pytest.MonkeyPatch, agent_range_m: float, agent_present: bool, expected: float
) -> None:
    """The near-miss channel tracks the closest occupied agent slot.

    Purpose: Validates the near-miss measurement on both sides of its threshold, including
        the boundary and the empty-slot case. An empty slot sits at the origin in the
        state vector, so a range check that ignored the presence flag would report a
        permanent near miss.

    Given: A session whose single agent slot is at a chosen range, present or not
    When: One step is taken and step_info is queried
    Then: The near-miss channel is 1.0 exactly when a present agent is within 5 m

    Test type: unit
    """
    session = FakeRacetrackSession(agent_range_m=agent_range_m, agent_present=agent_present)
    _install(monkeypatch, session)
    world = _reset_world()
    state = world._live_state

    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert info[RacetrackStepChannel.NEAR_MISS.value] == expected


def test_the_near_miss_channel_ignores_the_detection_range_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near miss is measured off the true state, not off what the radar reported.

    Purpose: The range gate belongs to the observation. Measuring the metric through it
        would make the reported near-miss rate a property of the sensor rather than of the
        driving, and the two arms of the comparison would then be scored on different
        events -- which is the one thing this environment may not do.

    Given: A world whose radar is gated at half a metre, scripted with an agent 3 m away
    When: One step is taken and step_info is queried
    Then: The near miss is still reported

    Test type: unit
    """
    session = FakeRacetrackSession(agent_range_m=3.0)
    _install(monkeypatch, session)
    world = _reset_world(max_detection_range_m=0.5)
    state = world._live_state

    next_state = world.sample_next_state(state, _COAST_STRAIGHT)
    info = world.step_info(state, _COAST_STRAIGHT, next_state)

    assert info[RacetrackStepChannel.NEAR_MISS.value] == 1.0


# ── Declared metrics versus produced channels ───────────────────────────


def test_declared_channels_match_the_channels_actually_produced(
    world: RacetrackPOMDP,
) -> None:
    """Every declared spec names a channel the world really emits, and vice versa.

    Purpose: Validates the invariant the aggregator cannot check for itself. A spec whose
        channel is never emitted yields a metric that is silently dropped, and a channel
        with no spec is measured and then thrown away.

    Given: A world that has taken a step
    When: The declared spec channels and the produced mapping's keys are compared
    Then: The two sets are equal

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, _COAST_STRAIGHT)

    declared = {spec.channel for spec in world.get_metric_specs()}
    produced = set(world.step_info(state, _COAST_STRAIGHT, next_state))

    assert declared == produced


def test_metric_names_follow_the_declared_enum_order(world: RacetrackPOMDP) -> None:
    """The reported metric names are the RacetrackMetric members, in order.

    Purpose: Validates that the enum is the single source of metric naming, so a metric
        renamed in one place cannot keep its old name in the other.

    Given: A world's metric specs
    When: get_metric_names is read
    Then: It equals the RacetrackMetric values in declaration order

    Test type: unit
    """
    assert world.get_metric_names() == [metric.value for metric in RacetrackMetric]
    assert [spec.name for spec in world.get_metric_specs()] == world.get_metric_names()


# ── Packaging ───────────────────────────────────────────────────────────


def _dev_extra_requirements() -> List[str]:
    """The `dev` extra's requirement strings, read without a TOML parser.

    Scanned as text rather than parsed: `tomllib` is standard library only from 3.11 and
    this project supports 3.10, where the CI image runs. Pulling in `tomli` to satisfy one
    assertion would add a dependency to check a dependency.
    """
    lines = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    requirements: List[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("dev = ["):
            inside = True
            continue
        if inside:
            if stripped.startswith("]"):
                break
            if stripped.startswith("#"):
                continue
            requirements.append(stripped.strip(",").strip('"').strip("'"))
    return requirements


def test_highway_env_is_declared_in_the_dev_extra() -> None:
    """The simulator backend is a declared test dependency.

    Purpose: Guards the gated tests below. They skip when highway-env is missing, so
        without this check a CI image that lost the dependency would report a green suite
        with the real-simulator assertions never run. This test never skips.

    Given: The project's pyproject.toml
    When: The dev extra is read
    Then: It contains a highway-env requirement

    Test type: configuration
    """
    assert any(requirement.startswith("highway-env") for requirement in _dev_extra_requirements())


# ── Real simulator: the matched pair ────────────────────────────────────


def test_live_pomdp_observation_channels_have_the_promised_shapes_and_dtype(
    headless: None,
) -> None:
    """The POMDP arm emits the five channels the module documents.

    Purpose: Validates the reading's structure against the real backend, which is the only
        place the promised shapes can actually be confirmed. The planner's model unflattens
        this reading by position, so a channel of the wrong width silently shifts every
        later channel into the wrong slot -- and the ego-pose channel now sits first, so a
        world that emitted the old four-channel reading would misalign every later one.

    Given: A live racetrack world in POMDP mode
    When: An initial observation is drawn
    Then: It is a five-channel RacetrackObservation with the documented widths, all float32

    Test type: integration
    """
    del headless
    world = RacetrackPOMDP(discount_factor=0.95, observation_mode=ObservationMode.POMDP, seed=0)

    observation = world.initial_observation_dist().sample()[0]

    assert isinstance(observation, RacetrackObservation)
    assert len(observation) == 5
    assert observation.ego_pose.shape == (OBSERVED_EGO_POSE_WIDTH,)
    assert observation.ego_speed.shape == (OBSERVED_EGO_SPEED_WIDTH,)
    assert observation.lane_pose.shape == (OBSERVED_LANE_POSE_WIDTH,)
    assert observation.curvature_ahead.shape == (len(DEFAULT_CURVATURE_LOOKAHEAD_M),)
    assert observation.detections.shape == (DEFAULT_MAX_TRACKED_AGENTS, DETECTION_SLOT_WIDTH)
    assert all(channel.dtype == np.float32 for channel in observation)


def test_live_mdp_observation_is_the_kinematics_table(headless: None) -> None:
    """The MDP arm still emits the ego-first kinematics table, unchanged.

    Purpose: Validates the baseline arm against the real backend. The POMDP arm's reading
        was redesigned; this pins that the fully-observed arm it is compared against was
        not touched, since a shifted baseline would move the measured gap on its own.

    Given: A live racetrack world in MDP mode
    When: An initial observation is drawn
    Then: It is a single float32 table with one row per tracked agent plus the ego

    Test type: integration
    """
    del headless
    world = RacetrackPOMDP(discount_factor=0.95, observation_mode=ObservationMode.MDP, seed=0)

    observation = world.initial_observation_dist().sample()[0]

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (DEFAULT_MAX_TRACKED_AGENTS + 1, 5)
    assert observation.dtype == np.float32


def test_live_simulator_configs_differ_only_in_the_observation(headless: None) -> None:
    """The two running simulators are configured identically apart from what they emit.

    Purpose: Validates the matched pair where it finally matters -- on the live
        HighwayEnv config, after highway-env has merged in its own defaults. Checking only
        the dictionary this package builds would miss a key the backend fills in
        differently for the two observation types.

    Given: A live world in each observation mode, both reset
    When: The two unwrapped configs are compared key by key
    Then: "observation" is the only key they disagree on

    Test type: integration
    """
    del headless
    worlds = {}
    for mode in (ObservationMode.MDP, ObservationMode.POMDP):
        world = RacetrackPOMDP(discount_factor=0.95, observation_mode=mode, seed=7)
        world.initial_state_dist().sample()
        worlds[mode] = world._get_session()._env.unwrapped.config

    mdp_config, pomdp_config = worlds[ObservationMode.MDP], worlds[ObservationMode.POMDP]
    differing = sorted(
        key
        for key in set(mdp_config) | set(pomdp_config)
        if mdp_config.get(key) != pomdp_config.get(key)
    )

    assert differing == ["observation"]


def test_shared_reward_reproduces_the_simulators_own_over_a_rollout(
    headless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reward this package computes matches highway-env's, step for step.

    Purpose: Validates the closed-form reward against the simulator that defines it. The
        planner optimises this function, so a drift from the scored reward would mean the
        planner is solving a different problem than the one being measured.

    Given: A live world with the simulator's own returned reward recorded per step
    When: Twenty steps are taken with a fixed pseudo-random action sequence
    Then: Every computed reward matches the simulator's to within 1e-6

    Test type: integration

    Note:
        The tolerance is 1e-6 rather than something tighter because the simulator works in
        float32 along the way; the observed discrepancy is around 1e-8, so 1e-9 would sit
        inside the numerical noise and fail for no real reason.
    """
    del headless
    world = RacetrackPOMDP(discount_factor=0.95, seed=0)
    state = world.initial_state_dist().sample()[0]
    simulator_rewards: List[float] = []
    raw_env = world._get_session()._env
    original_step = raw_env.step

    def recording_step(command: Any) -> Any:
        result = original_step(command)
        simulator_rewards.append(float(result[1]))
        return result

    monkeypatch.setattr(raw_env, "step", recording_step)

    rng = np.random.default_rng(3)
    differences: List[float] = []
    for _ in range(20):
        action = int(rng.integers(0, len(world.get_actions())))
        computed = world.reward(state, action)
        state = world.sample_next_state(state, action)
        differences.append(abs(computed - simulator_rewards[-1]))
        if world.is_terminal(state):
            break

    assert differences
    assert max(differences) < 1e-6, f"largest reward discrepancy was {max(differences)}"


def test_the_two_arms_produce_bit_identical_ego_trajectories(headless: None) -> None:
    """Same seed, same actions, both arms: the ego moves identically.

    Purpose: This is the matched pair, proved rather than assumed. If the two arms shared
        anything less than one dynamics path -- a different RNG draw, a different vehicle
        count, an observation that perturbs the simulation -- the trajectories would
        separate, and any performance gap measured between the arms would be
        uninterpretable. A failure here means the environment's premise is broken; the
        tolerance must not be loosened to make it pass.

    Given: One world per arm, same seed, and one fixed action sequence
    When: Both are driven through that sequence
    Then: The ego position, heading and speed agree exactly at every step

    Test type: integration
    """
    del headless
    actions = [4, 1, 4, 7, 4, 1, 3, 5, 4, 4, 1, 4]
    trajectories = {}
    for mode in (ObservationMode.MDP, ObservationMode.POMDP):
        world = RacetrackPOMDP(discount_factor=0.95, observation_mode=mode, seed=7)
        state = world.initial_state_dist().sample()[0]
        rows = [np.asarray(state[:4], dtype=float).copy()]
        for action in actions:
            state = world.sample_next_state(state, action)
            rows.append(np.asarray(state[:4], dtype=float).copy())
            if world.is_terminal(state):
                break
        trajectories[mode] = np.array(rows)

    mdp_rows, pomdp_rows = trajectories[ObservationMode.MDP], trajectories[ObservationMode.POMDP]

    assert mdp_rows.shape == pomdp_rows.shape
    assert len(mdp_rows) > 1
    assert np.array_equal(mdp_rows, pomdp_rows)


def test_a_live_episode_runs_to_termination_or_the_step_budget(headless: None) -> None:
    """A full rollout against the real backend completes without raising.

    Purpose: A smoke test over the whole live path -- reset, control conversion, state
        reading, reward, all five sensors and terminality -- which the scripted fake by
        construction cannot exercise.

    Given: A live world and a budget of thirty steps
    When: Actions are taken until termination or the budget runs out
    Then: Every step yields a finite reward, a correctly shaped state, and a reading whose
        every channel is finite and correctly shaped

    Test type: integration
    """
    del headless
    world = RacetrackPOMDP(discount_factor=0.95, seed=1)
    state = world.initial_state_dist().sample()[0]

    steps = 0
    for _ in range(30):
        reward = world.reward(state, _COAST_STRAIGHT)
        state = world.sample_next_state(state, _COAST_STRAIGHT)
        observation = world.sample_observation(state, _COAST_STRAIGHT)
        steps += 1
        assert np.isfinite(reward)
        assert state.shape == (world.state_width,)
        assert observation.ego_pose.shape == (OBSERVED_EGO_POSE_WIDTH,)
        assert observation.ego_speed.shape == (OBSERVED_EGO_SPEED_WIDTH,)
        assert observation.lane_pose.shape == (OBSERVED_LANE_POSE_WIDTH,)
        assert observation.curvature_ahead.shape == (len(DEFAULT_CURVATURE_LOOKAHEAD_M),)
        assert observation.detections.shape == (DEFAULT_MAX_TRACKED_AGENTS, DETECTION_SLOT_WIDTH)
        assert all(np.all(np.isfinite(channel)) for channel in observation)

    assert steps >= 1


# ── The ego pose channel ────────────────────────────────────────────────


def test_the_zero_noise_ego_pose_is_the_states_own_pose_and_arclength(headless: None) -> None:
    """What the arm reports as its pose is the pose the state vector holds.

    Purpose: This is the wiring of the channel the redesign added, and only the live world
        can be asked it. The sensors are handed a vehicle and an arclength; this pins that
        those are *the ego* and *the state's own distance round the lap*. The arclength
        matters most: it is numbered by walking lanes from the one the episode started on,
        and the state slot, the curvature channel and this reading all have to be numbered
        against the same walk. A channel re-basing at a segment boundary, or measuring the
        lane-local offset instead, would read plausibly and put the planner's mapped model
        a whole segment out.

    Given: A live POMDP world with every ego-pose width set to zero, driven a few steps
    When: Each reading's ego_pose is compared with the state it was emitted for
    Then: The four entries are that state's x, y, heading and arclength

    Test type: integration
    """
    del headless
    world = _quiet_world(seed=0, other_vehicles=0)
    start = world.initial_state_dist().sample()[0]

    steps = 0
    for state, observation in _drive(world, start, 12):
        steps += 1
        pose = observation.ego_pose
        assert float(pose[EGO_POSE_X]) == pytest.approx(float(state[EGO_X]), abs=1e-3)
        assert float(pose[EGO_POSE_Y]) == pytest.approx(float(state[EGO_Y]), abs=1e-3)
        assert float(pose[EGO_POSE_HEADING]) == pytest.approx(float(state[EGO_HEADING]), abs=1e-5)
        assert float(pose[EGO_POSE_ARCLENGTH]) == pytest.approx(
            float(state[EGO_ARCLENGTH_M]), abs=1e-3
        )

    assert steps > 5, "too few steps to have crossed a segment boundary"


# ── The ego speedometer ─────────────────────────────────────────────────


def test_the_observed_ego_speed_tracks_the_true_state_slot(headless: None) -> None:
    """What the POMDP arm reports as its speed is what the state slot holds.

    Purpose: Validates the whole point of the speedometer -- that it is a measurement of
        EGO_SPEED and not of something adjacent to it. The reduction runs through
        highway-env's kinematics block, so a wrong feature order, a normalised block or a
        relative frame would all show up here as a mismatch rather than downstream as a
        planner that mysteriously mis-times its braking.

    Given: A live POMDP world driven for 20 coasting steps
    When: Each observation's ego_speed is compared with the next state's EGO_SPEED slot
    Then: They agree to float32 precision at every step

    Test type: integration
    """
    del headless
    world = RacetrackPOMDP(discount_factor=0.95, seed=0)
    start = world.initial_state_dist().sample()[0]

    steps = 0
    for state, observation in _drive(world, start, 20):
        steps += 1
        assert float(observation.ego_speed[0]) == pytest.approx(float(state[EGO_SPEED]), abs=1e-4)

    assert steps > 1


def test_the_observed_ego_speed_goes_negative_when_the_ego_reverses(headless: None) -> None:
    """Braking through zero reports a negative speed, not its magnitude.

    Purpose: The state's EGO_SPEED is signed and the racetrack ego really does reverse
        under sustained braking. Reducing the velocity with a Euclidean norm would agree
        with the state everywhere the car drives forwards and silently report a reversing
        car as an accelerating one, which is exactly the reading a planner would act on.

    Given: A live POMDP world braking flat out from its 10 m/s start
    When: The episode is driven until the state's own speed slot goes negative
    Then: The observed speed is negative too and still matches the slot

    Test type: integration
    """
    del headless
    brake_straight = DEFAULT_ACTION_PRESETS.index((-1.0, 0.0))
    world = RacetrackPOMDP(discount_factor=0.95, seed=0)
    start = world.initial_state_dist().sample()[0]

    observed_negative = False
    for state, observation in _drive(world, start, 30, action=brake_straight):
        if float(state[EGO_SPEED]) < -0.5:
            observed_negative = True
            assert float(observation.ego_speed[0]) == pytest.approx(
                float(state[EGO_SPEED]), abs=1e-4
            )
            break

    assert observed_negative, "the ego never reversed, so the signed case went untested"


def test_the_reading_carries_the_whole_relative_velocity_and_not_only_its_radial_half(
    headless: None,
) -> None:
    """Both components of an opponent's relative motion are emitted, crossing rate included.

    Purpose: A detection row used to carry the projection onto the line of sight and
        nothing else, so the crossing rate had to be inferred -- a different estimation
        problem from the one this environment poses. This pins the replacement on the live
        world: the row carries the state's own relative velocity in full, and a step where
        the crossing rate differs from the closing rate is what tells the two apart. It also
        pins the trim of the kinematics block the speedometer is read out of -- highway-env
        1.12.1 returns a row per vehicle for the two it was asked about, and the second row
        must not survive.

    Given: A live noiseless POMDP world with an opponent and no range gate to speak of
    When: Each reading is compared with the true relative velocity in the state's slots, at
        steps where the crossing rate and the closing rate disagree
    Then: The row's two velocity entries are that slot's relative velocity, neither of them
        the radial projection alone, and the speedometer is still one number equal to
        EGO_SPEED

    Test type: integration
    """
    del headless
    world = _quiet_world(seed=2, other_vehicles=1, max_detection_range_m=_UNLIMITED_RANGE_M)
    start = world.initial_state_dist().sample()[0]

    checked = 0
    for state, observation in _drive(world, start, 8):
        assert len(observation) == 5
        assert observation.ego_speed.size == OBSERVED_EGO_SPEED_WIDTH
        assert float(observation.ego_speed[0]) == pytest.approx(float(state[EGO_SPEED]), abs=1e-4)

        present = _present_agent_rows(state, DEFAULT_MAX_TRACKED_AGENTS)
        if len(present) != 1 or observation.detections[0, DETECTION_PRESENT] < 0.5:
            continue
        offset = present[:, AGENT_REL_X : AGENT_REL_X + 2]
        velocity = present[:, AGENT_REL_VX : AGENT_REL_VX + 2]
        radial = float(radial_velocities(offset, velocity)[0])
        reported = np.asarray(
            observation.detections[0, DETECTION_REL_VX : DETECTION_REL_VX + 2], dtype=float
        )
        if abs(float(np.linalg.norm(velocity[0])) - abs(radial)) < 1.0:
            continue  # Almost head-on here, so the extra component proves nothing.

        assert reported == pytest.approx(velocity[0], abs=1e-3)
        # A radial-only row would have norm |radial| exactly; the guard above put the true
        # relative speed a whole metre per second above that.
        assert float(np.linalg.norm(reported)) - abs(radial) > 0.5
        checked += 1

    assert checked > 0, "no step had a crossing rate distinguishable from its closing rate"


# ── The lane camera ─────────────────────────────────────────────────────


def test_zero_lane_noise_gives_back_the_exact_lane_offset(headless: None) -> None:
    """Configuring both widths to zero recovers the simulator's own lane pose exactly.

    Purpose: Pins the reading to the state slots it is supposed to measure, on the live
        road. The camera's own arithmetic and its noise widths are covered against a
        stand-in vehicle in ``test_racetrack_world_sensors.py``; what only the live world
        can answer is whether the pose it measured is the *ego's* -- a channel fed the
        wrong vehicle or the wrong entries of ``lane_offset`` would still be a plausible
        two-vector, and the model's likelihood would score a residual against a mean it
        never converges on.

    Given: A live POMDP world with both lane-noise widths set to 0.0, driven a few steps
    When: Each reading is compared with the EGO_LAT and EGO_ANG of the state it saw
    Then: They agree to float32 precision

    Test type: integration
    """
    del headless
    world = _quiet_world(seed=5, other_vehicles=0)
    start = world.initial_state_dist().sample()[0]

    steps = 0
    for state, observation in _drive(world, start, 5):
        steps += 1
        assert float(observation.lane_pose[LANE_POSE_LAT]) == pytest.approx(
            float(state[EGO_LAT]), abs=1e-6
        )
        assert float(observation.lane_pose[LANE_POSE_ANG]) == pytest.approx(
            float(state[EGO_ANG]), abs=1e-6
        )

    assert steps > 1


# ── Curvature ahead ─────────────────────────────────────────────────────


def test_zero_noise_curvature_is_the_track_geometry_at_the_ego_arclength_plus_each_lookahead(
    headless: None,
) -> None:
    """Noiseless, the channel is the road: the profile read at ``arclength + d``.

    Purpose: Pins the channel to the geometry the planner's model indexes, and to the same
        lane walk the arclength state slot is numbered against. If the camera read a
        different walk -- or numbered its distances from a different origin -- the two would
        disagree about where a corner starts, and every mapped model built on this world
        would be scoring curvature at the wrong place along the track.

    Given: A live POMDP world with zero curvature noise, and the curvature profile of its
        starting lane built independently from the road network
    When: Each reading is compared against the profile at the state's own arclength plus
        each configured lookahead
    Then: They agree at every step

    Test type: integration
    """
    del headless
    lookaheads = (10.0, 20.0, 30.0)
    world = _quiet_world(seed=0, other_vehicles=0, curvature_lookahead_m=lookaheads)
    start = world.initial_state_dist().sample()[0]
    geometry = _ego_lane_geometry(world)

    steps = 0
    for state, observation in _drive(world, start, 20):
        steps += 1
        expected = geometry.curvature_at(float(state[EGO_ARCLENGTH_M]) + np.asarray(lookaheads))
        assert observation.curvature_ahead == pytest.approx(expected, abs=1e-6)

    assert steps > 5, "too few steps to have crossed a segment boundary"


def test_curvature_ahead_leads_the_ego_into_a_bend(headless: None) -> None:
    """Approaching a corner, the channel reports a bend the ego is not yet in.

    Purpose: This is the channel's reason to exist. A reading that only ever reported the
        curvature at the ego's own arclength would tell a planner nothing it could not read
        off its own state, and a rollout could not brake before a corner it cannot see. The
        test therefore pins the lead itself: a step where the road under the car is straight
        while the far sample already carries a curvature the car reaches later.

    Given: A live noiseless POMDP world coasting towards the circuit's first bend
    When: Each step's furthest lookahead is compared with the curvature at the ego's own
        arclength
    Then: Some step reports a bend ahead while the ego is still on a straight, and the ego
        later arrives at exactly that curvature

    Test type: integration
    """
    del headless
    world = _quiet_world(seed=0, other_vehicles=0)
    start = world.initial_state_dist().sample()[0]
    geometry = _ego_lane_geometry(world)

    profile: List[Tuple[float, float]] = []
    for state, observation in _drive(world, start, 25):
        here = float(geometry.curvature_at(float(state[EGO_ARCLENGTH_M])))
        profile.append((here, float(observation.curvature_ahead[-1])))

    leading = [index for index, (here, ahead) in enumerate(profile) if here == 0.0 and ahead != 0.0]
    assert leading, "the rollout never approached a bend, so the lead went untested"
    announced = profile[leading[0]][1]
    arrived = [here for here, _ in profile[leading[0] + 1 :] if here == pytest.approx(announced)]
    assert arrived, f"curvature {announced} was announced ahead but the ego never reached it"


def test_the_curvature_channel_is_noisy_at_the_width_it_was_configured_with(
    headless: None,
) -> None:
    """The camera's curvature reading is corrupted, by the amount it was told to be.

    Purpose: A mapless planner reading this channel is estimating the road rather than
        being told it, and that is only true if the reading is wrong. Emitting the true
        curvature would hand a POMDP planner a piece of ground truth the fully-observed arm
        does not even carry in its observation, which would invert the comparison.

    Given: Four live episodes with curvature noise of 0.02 1/m, an order above the default
        so the spread is measurable in the steps a coasting episode survives
    When: Each reading is compared with the track's own curvature at the same lookaheads
    Then: No reading is exact and the residuals' spread sits near the configured width

    Test type: integration
    """
    del headless
    np.random.seed(23)
    curvature_std = 0.02

    residuals: List[float] = []
    for seed in range(4):
        world = RacetrackPOMDP(
            discount_factor=0.95, seed=seed, other_vehicles=0, curvature_std_1pm=curvature_std
        )
        start = world.initial_state_dist().sample()[0]
        geometry = _ego_lane_geometry(world)
        for state, observation in _drive(world, start, 15):
            truth = geometry.curvature_at(
                float(state[EGO_ARCLENGTH_M]) + np.asarray(DEFAULT_CURVATURE_LOOKAHEAD_M)
            )
            residuals.extend((np.asarray(observation.curvature_ahead) - truth).tolist())

    errors = np.asarray(residuals)
    assert len(errors) > 30, "too few readings survived to measure a spread"
    assert not np.any(errors == 0.0), "an exact reading means the camera noise is missing"
    assert float(errors.std()) == pytest.approx(curvature_std, rel=0.5)


# ── Reproducibility of the sensor noise ─────────────────────────────────


def _seeded_readings(numpy_seed: int, steps: int = 4) -> np.ndarray:
    """Drive a live world under a given global NumPy seed and flatten what it saw."""
    np.random.seed(numpy_seed)
    world = RacetrackPOMDP(
        discount_factor=0.95, seed=0, other_vehicles=1, max_detection_range_m=_UNLIMITED_RANGE_M
    )
    start = world.initial_state_dist().sample()[0]
    rows = [
        np.concatenate([np.asarray(channel, dtype=float).reshape(-1) for channel in observation])
        for _, observation in _drive(world, start, steps)
    ]
    return np.asarray(rows)


def test_seeding_numpy_reproduces_a_reading(headless: None) -> None:
    """The same global seed replays the same sensor noise; a different one does not.

    Purpose: All four sensors draw from NumPy's global generator, which is what makes a
        seeded experiment repeatable -- and what makes an unnoticed extra draw anywhere in
        the world shift every later reading. Both halves are asserted: without the second,
        a world that had lost its noise entirely would pass the first trivially.

    Given: Two live POMDP episodes run under the same NumPy seed and one under another,
        all with the same simulator seed
    When: Their readings are compared
    Then: The matching pair is identical and the third differs

    Test type: integration
    """
    del headless
    first = _seeded_readings(12345)
    repeat = _seeded_readings(12345)
    different = _seeded_readings(999)

    assert first.size > 0
    assert np.array_equal(first, repeat)
    assert first.shape == different.shape
    assert not np.array_equal(first, different)
