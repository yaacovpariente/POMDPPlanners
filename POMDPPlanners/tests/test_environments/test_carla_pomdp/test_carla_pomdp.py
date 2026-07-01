# SPDX-License-Identifier: MIT

"""Unit tests for the CarlaPOMDP forward-only world wrapper.

These tests drive CarlaPOMDP against a controllable fake CARLA session (scripted
reset/step returning a 5-D ground-truth state and a 3-D GNSS observation) injected
by monkeypatching ``CarlaPOMDP._get_session``, so they run without a CARLA server
or the ``carla`` package installed. They mirror the GymPOMDP suite and add the
CARLA-specific assertion that the observation differs from the state.
"""

# pylint: disable=protected-access  # Tests inspect the live-session internals

import pickle
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.carla_pomdp import CarlaPOMDP, carla_video
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.simulations.episodes import EpisodeRunner


class FakeCarlaSession:
    """Scripted CARLA-like session: 5-D ground-truth state, 3-D GNSS observation.

    The state is ``[x, y, yaw, vx, vy]`` with ``vx`` advancing each tick so the
    ego speed equals the step index; the observation is a distinct 3-D GNSS
    ``[lat, lon, alt]`` vector. The episode terminates once three ticks have been
    taken, so terminal behavior is deterministic.
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.last_control: Optional[Tuple[float, float, float]] = None
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        del seed
        self.reset_calls += 1
        self._t = 0
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        observation = np.array([48.0, 2.0, 0.0])
        return state, observation

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        self.step_calls += 1
        self.last_control = (throttle, steer, brake)
        self._t += 1
        state = np.array([float(self._t), 0.0, 0.0, float(self._t), 0.0])
        observation = np.array([48.0 + self._t, 2.0, 0.0])
        terminated = self._t >= 3
        return state, observation, terminated


class FixedActionPolicy(Policy):
    """A trivial discrete policy that returns a single preset-index action."""

    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        name: str = "FixedActionPolicy",
        log_path: Optional[Path] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)

    def action(self, belief: Belief):
        del belief
        return ([0], PolicyRunData(info_variables=[]))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return []


@pytest.fixture(name="fake_session")
def _fake_session(monkeypatch: pytest.MonkeyPatch) -> FakeCarlaSession:
    """Patch ``CarlaPOMDP._get_session`` to return a shared FakeCarlaSession."""
    session = FakeCarlaSession()
    monkeypatch.setattr(CarlaPOMDP, "_get_session", lambda self: session)
    return session


@pytest.fixture(name="world")
def _world(fake_session: FakeCarlaSession) -> CarlaPOMDP:
    """A CarlaPOMDP wrapping the fake session, reset to its initial live state."""
    del fake_session
    env = CarlaPOMDP(discount_factor=0.95)
    env.initial_state_dist().sample()  # establish the live state
    return env


def test_space_info_is_discrete_actions_continuous_observations(world: CarlaPOMDP) -> None:
    """CarlaPOMDP exposes discrete actions and continuous observations.

    Purpose: Validates the fixed SpaceInfo for the CARLA world.

    Given: A CarlaPOMDP with preset discrete controls and a GNSS observation
    When: space_info is inspected
    Then: action space is DISCRETE and observation space is CONTINUOUS

    Test type: unit
    """
    assert world.space_info.action_space == SpaceType.DISCRETE
    assert world.space_info.observation_space == SpaceType.CONTINUOUS


def test_step_called_once_across_reward_next_state_observation(
    world: CarlaPOMDP, fake_session: FakeCarlaSession
) -> None:
    """The three per-step queries trigger exactly one underlying CARLA tick.

    Purpose: Validates the step-once cache serving reward/next-state/observation.

    Given: A freshly reset CarlaPOMDP world at its live state
    When: reward, sample_next_state and sample_observation are called in order
    Then: the session steps exactly once and the three agree on the transition

    Test type: unit
    """
    state = world._live_state
    before = fake_session.step_calls

    reward = world.reward(state, 0)
    next_state = world.sample_next_state(state, 0)
    observation = world.sample_observation(next_state, 0)

    assert fake_session.step_calls - before == 1
    assert reward == 1.0
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.0, 0.0]))
    assert np.array_equal(observation, np.array([49.0, 2.0, 0.0]))


def test_call_order_independence_next_state_before_reward(
    world: CarlaPOMDP, fake_session: FakeCarlaSession
) -> None:
    """The tick is served identically regardless of which query comes first.

    Purpose: Validates the cache trigger works from sample_next_state too.

    Given: A freshly reset CarlaPOMDP world at its live state
    When: sample_next_state is called before reward for the same (state, action)
    Then: still one session step, and reward reflects that same transition

    Test type: unit
    """
    state = world._live_state
    before = fake_session.step_calls

    next_state = world.sample_next_state(state, 0)
    reward = world.reward(state, 0)

    assert fake_session.step_calls - before == 1
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.0, 0.0]))
    assert reward == 1.0


def test_observation_differs_from_ground_truth_state(world: CarlaPOMDP) -> None:
    """The GNSS observation is a partial view, distinct from the true state.

    Purpose: Validates the CARLA state != observation modeling (partial observability).

    Given: A CarlaPOMDP world at its live state
    When: A step is sampled and both next state and observation are read
    Then: The observation is the 3-D GNSS vector, not the 5-D ground-truth state

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, 0)
    observation = world.sample_observation(next_state, 0)

    assert next_state.shape == (5,)
    assert observation.shape == (3,)
    assert not world.is_equal_observation(observation, next_state)


def test_is_terminal_reflects_last_terminated(world: CarlaPOMDP) -> None:
    """is_terminal returns the terminated flag of the current live state.

    Purpose: Validates terminal tracking across forward ticks.

    Given: A world whose fake session terminates after three ticks
    When: The world is advanced step by step from the live state
    Then: is_terminal is False until the third step, then True

    Test type: unit
    """
    state = world._live_state
    assert world.is_terminal(state) is False

    for expected_terminal in (False, False, True):
        next_state, _, _ = world.sample_next_step(state, 0)
        assert world.is_terminal(next_state) is expected_terminal
        state = next_state


def test_initial_state_dist_sample_triggers_reset(fake_session: FakeCarlaSession) -> None:
    """Sampling the initial-state distribution resets the CARLA session.

    Purpose: Validates initial_state_dist maps to session.reset.

    Given: A CarlaPOMDP wrapping a fake session
    When: initial_state_dist().sample() is called
    Then: session.reset runs and the returned state is the reset ground truth

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    before = fake_session.reset_calls

    samples = env.initial_state_dist().sample()

    assert fake_session.reset_calls - before == 1
    assert len(samples) == 1
    assert np.array_equal(samples[0], np.zeros(5))


def test_initial_observation_dist_returns_sensor_reading(world: CarlaPOMDP) -> None:
    """The initial observation is the first post-reset GNSS reading.

    Purpose: Validates initial_observation_dist serves the sensor payload.

    Given: A CarlaPOMDP world reset to its initial state
    When: initial_observation_dist().sample() is called
    Then: the returned observation is the reset GNSS vector

    Test type: unit
    """
    samples = world.initial_observation_dist().sample()
    assert len(samples) == 1
    assert np.array_equal(samples[0], np.array([48.0, 2.0, 0.0]))


def test_action_preset_index_maps_to_vehicle_control(world: CarlaPOMDP) -> None:
    """A discrete action index selects the matching control preset.

    Purpose: Validates the discrete action -> (throttle, steer, brake) mapping.

    Given: A CarlaPOMDP with the default control presets
    When: sample_next_state is called with the brake preset index (3)
    Then: the session receives the (0.0, 0.0, 1.0) brake control

    Test type: unit
    """
    world.sample_next_state(world._live_state, 3)
    assert world._get_session().last_control == (0.0, 0.0, 1.0)


def test_transition_log_probability_raises(world: CarlaPOMDP) -> None:
    """transition_log_probability is unsupported on a forward-only world.

    Purpose: Validates the density method raises instead of faking a value.

    Given: A CarlaPOMDP world
    When: transition_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    with pytest.raises(NotImplementedError):
        world.transition_log_probability(world._live_state, 0, [world._live_state])


def test_observation_log_probability_raises(world: CarlaPOMDP) -> None:
    """observation_log_probability is unsupported on a forward-only world.

    Purpose: Validates the density method raises instead of faking a value.

    Given: A CarlaPOMDP world
    When: observation_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    with pytest.raises(NotImplementedError):
        world.observation_log_probability(world._live_state, 0, [world._live_state])


def test_forward_only_guard_raises_on_mismatched_state(world: CarlaPOMDP) -> None:
    """Stepping from a non-live state is rejected loudly.

    Purpose: Validates the forward-only guard on arbitrary-state resampling.

    Given: A CarlaPOMDP world at a known live state
    When: sample_next_state is called with a different, arbitrary state
    Then: RuntimeError is raised naming the forward-only constraint

    Test type: unit
    """
    arbitrary_state = np.array([99.0, 99.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="forward-only"):
        world.sample_next_state(arbitrary_state, 0)


def test_sample_observation_rejects_mismatched_next_state(world: CarlaPOMDP) -> None:
    """A forward-only world cannot produce a sensor reading for an arbitrary state.

    Purpose: Validates sample_observation rejects a non-live next state.

    Given: A CarlaPOMDP world at its live state
    When: sample_observation is called for a next state it never stepped to
    Then: RuntimeError is raised

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="forward-only|live"):
        world.sample_observation(np.array([7.0, 7.0, 0.0, 0.0, 0.0]), 0)


def test_sample_next_state_rejects_multiple_samples(world: CarlaPOMDP) -> None:
    """A forward-only world cannot draw more than one next state.

    Purpose: Validates n_samples>1 is rejected.

    Given: A CarlaPOMDP world at its live state
    When: sample_next_state is called with n_samples=2
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="n_samples=1"):
        world.sample_next_state(world._live_state, 0, n_samples=2)


def test_getstate_setstate_round_trip_drops_and_rebuilds_handle(
    world: CarlaPOMDP, fake_session: FakeCarlaSession
) -> None:
    """Pickling drops the live handle and it is rebuilt lazily on use.

    Purpose: Validates the non-picklable session is not serialized.

    Given: A CarlaPOMDP world with a live fake session handle
    When: It is pickled and unpickled
    Then: The restored object carries no live handle until it is next needed

    Test type: unit
    """
    del fake_session
    restored = pickle.loads(pickle.dumps(world))

    assert restored._session is None
    assert restored._live_state is None
    assert restored._pending is None


def test_config_id_stable_across_pickling(world: CarlaPOMDP) -> None:
    """config_id depends only on public config, so it survives pickling.

    Purpose: Validates deterministic config identity for result caching.

    Given: A CarlaPOMDP world
    When: It is pickled and unpickled
    Then: config_id is unchanged (private live state is excluded)

    Test type: unit
    """
    restored = pickle.loads(pickle.dumps(world))
    assert restored.config_id == world.config_id


def test_two_env_initial_state_drawn_from_world(fake_session: FakeCarlaSession) -> None:
    """The ground-truth initial state comes from the CARLA world, not the model.

    Purpose: Validates the two-env initial-state sourcing with a CarlaPOMDP world.

    Given: A CarlaPOMDP world and a distinct TigerPOMDP planner model
    When: An EpisodeRunner is constructed around them
    Then: The runner's true state is the world's reset state and the world reset

    Test type: integration
    """
    world = CarlaPOMDP(discount_factor=0.95)
    model = TigerPOMDP(discount_factor=0.95)
    policy = FixedActionPolicy(environment=model, discount_factor=0.95)
    belief = get_initial_belief(model, n_particles=3)

    runner = EpisodeRunner(
        environment=world, policy=policy, initial_belief=belief, num_steps=2, logger=None
    )

    assert np.array_equal(runner.state, np.zeros(5))
    assert fake_session.reset_calls >= 1


class StationaryThenMovingSession:
    """CARLA-like session whose ego stays put for two ticks, then moves.

    While stationary the reported ``next_state`` equals the state stepped from,
    which is exactly the condition that collided the step-once cache across
    consecutive episode steps and froze the world.
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        del seed
        self.reset_calls += 1
        self._t = 0
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([48.0, 2.0, 0.0])

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        del throttle, steer, brake
        self.step_calls += 1
        self._t += 1
        position = 0.0 if self._t <= 2 else float(self._t - 2)
        state = np.array([position, 0.0, 0.0, 0.0, 0.0])
        return state, np.array([48.0, 2.0, 0.0]), False


def test_stationary_ego_does_not_freeze_the_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """A momentarily stationary ego must still advance the forward-only world.

    Purpose: Regression for the step-once cache colliding across episode steps
        when next_state equals state (a stationary vehicle), which previously
        served the stale cache and stopped CARLA from ticking after step one.

    Given: A CARLA-like session whose ego stays at the origin for the first ticks
    When: The world is driven forward several steps via sample_next_step
    Then: The session ticks exactly once per step and never freezes

    Test type: unit
    """
    session = StationaryThenMovingSession()
    monkeypatch.setattr(CarlaPOMDP, "_get_session", lambda self: session)
    env = CarlaPOMDP(discount_factor=0.95)
    state = env.initial_state_dist().sample()[0]

    for _ in range(5):
        before = session.step_calls
        next_state, _, _ = env.sample_next_step(state, 0)
        assert session.step_calls - before == 1
        state = next_state

    assert session.step_calls == 5


def test_two_env_discount_mismatch_raises(fake_session: FakeCarlaSession) -> None:
    """A world/model discount-factor mismatch is rejected at construction.

    Purpose: Validates the discount-consistency guard for the CARLA two-env case.

    Given: A CarlaPOMDP world discounted at 0.9 and a model discounted at 0.95
    When: An EpisodeRunner is constructed
    Then: ValueError is raised naming the discount mismatch

    Test type: integration
    """
    del fake_session
    world = CarlaPOMDP(discount_factor=0.9)
    model = TigerPOMDP(discount_factor=0.95)
    policy = FixedActionPolicy(environment=model, discount_factor=0.95)
    belief = get_initial_belief(model, n_particles=3)

    with pytest.raises(ValueError, match="discount_factor"):
        EpisodeRunner(
            environment=world, policy=policy, initial_belief=belief, num_steps=2, logger=None
        )


class _FakeCameraSession(FakeCarlaSession):
    """A FakeCarlaSession that also buffers RGB camera frames like the real session.

    Extends the scripted session with a ``frames`` buffer so the camera-video save
    path can be exercised without a CARLA server or a live camera sensor attached.
    """

    def __init__(self, frame_count: int = 0) -> None:
        super().__init__()
        self.frames: List[np.ndarray] = [
            np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(frame_count)
        ]


def test_hash_observation_matches_for_equal_and_differs_for_distinct(world: CarlaPOMDP) -> None:
    """hash_observation is stable for equal arrays and distinct for different ones.

    Purpose: Validates observation hashing used to key belief/tree observation nodes.

    Given: Two equal 3-D observation arrays and a third differing one
    When: hash_observation is computed for each
    Then: Equal arrays hash equally and the differing array hashes differently

    Test type: unit
    """
    observation = np.array([48.0, 2.0, 0.0])
    same = np.array([48.0, 2.0, 0.0])
    different = np.array([48.0, 2.0, 1.0])

    assert world.hash_observation(observation) == world.hash_observation(same)
    assert world.hash_observation(observation) != world.hash_observation(different)


def test_hash_action_hashes_arrays_and_passes_scalars_through(world: CarlaPOMDP) -> None:
    """hash_action returns bytes for ndarray actions and passes scalars unchanged.

    Purpose: Validates action hashing for both preset-index and array actions.

    Given: A scalar preset-index action and an ndarray action
    When: hash_action is computed for each
    Then: The scalar is returned unchanged and the array maps to its byte view

    Test type: unit
    """
    array_action = np.array([0.5, 0.0, 0.0])

    assert world.hash_action(1) == 1
    assert world.hash_action(array_action) == array_action.tobytes()


def test_save_camera_video_raises_when_recording_disabled() -> None:
    """save_camera_video refuses to run when camera recording was not enabled.

    Purpose: Validates the record_camera precondition guard.

    Given: A CarlaPOMDP constructed with record_camera=False
    When: save_camera_video is called
    Then: RuntimeError is raised explaining recording is disabled

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95, record_camera=False)

    with pytest.raises(RuntimeError, match="Camera recording is disabled"):
        env.save_camera_video(Path("unused.mp4"))


def test_save_camera_video_raises_when_no_frames_captured() -> None:
    """save_camera_video refuses to run when no frames were buffered.

    Purpose: Validates the non-empty-frames precondition guard.

    Given: A record_camera=True world whose session buffered zero frames
    When: save_camera_video is called
    Then: RuntimeError is raised explaining no frames were captured

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95, record_camera=True)
    env._session = _FakeCameraSession(frame_count=0)

    with pytest.raises(RuntimeError, match="No camera frames"):
        env.save_camera_video(Path("unused.mp4"))


def test_save_camera_video_forwards_session_frames_and_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """save_camera_video streams the buffered session frames to the encoder.

    Purpose: Validates the happy-path delegation to write_frames_to_mp4.

    Given: A record_camera=True world whose session buffered three frames
    When: save_camera_video is called with an explicit fps and output path
    Then: The encoder receives the session frames, the path, and the fps verbatim

    Test type: unit
    """
    written: Dict[str, Any] = {}

    def _fake_write(frames: List[np.ndarray], cache_path: Path, fps: int = 20) -> None:
        written["frames"] = frames
        written["path"] = cache_path
        written["fps"] = fps

    monkeypatch.setattr(carla_video, "write_frames_to_mp4", _fake_write)

    env = CarlaPOMDP(discount_factor=0.95, record_camera=True)
    session = _FakeCameraSession(frame_count=3)
    env._session = session
    output = tmp_path / "clip.mp4"

    env.save_camera_video(output, fps=30)

    assert written["frames"] is session.frames
    assert written["path"] == output
    assert written["fps"] == 30


def test_cache_visualization_writes_agent_path_named_mp4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cache_visualization names the camera clip agent_path_<episode>.mp4.

    Purpose: Validates the environment-owned visualization file-naming contract.

    Given: A record_camera=True world with buffered frames and an output directory
    When: cache_visualization is called for episode index 7
    Then: The encoder is asked to write <output_dir>/agent_path_7.mp4

    Test type: unit
    """
    written: Dict[str, Any] = {}

    def _fake_write(frames: List[np.ndarray], cache_path: Path, fps: int = 20) -> None:
        del frames, fps
        written["path"] = cache_path

    monkeypatch.setattr(carla_video, "write_frames_to_mp4", _fake_write)

    env = CarlaPOMDP(discount_factor=0.95, record_camera=True)
    env._session = _FakeCameraSession(frame_count=2)

    env.cache_visualization(history=[], output_dir=tmp_path, episode_index=7)

    assert written["path"] == tmp_path / "agent_path_7.mp4"
