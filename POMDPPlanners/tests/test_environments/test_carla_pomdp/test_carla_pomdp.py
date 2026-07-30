# SPDX-License-Identifier: MIT

"""Unit tests for the CarlaPOMDP forward-only world wrapper.

These tests drive CarlaPOMDP against a controllable fake CARLA session (scripted
reset/step returning a 7-D ground-truth state and a 3-D GNSS observation) injected
by monkeypatching ``CarlaPOMDP._get_session``, so they run without a CARLA server
or the ``carla`` package installed. They mirror the GymPOMDP suite and add the
CARLA-specific assertion that the observation differs from the state.
"""

# pylint: disable=protected-access,too-many-lines  # Tests inspect live-session internals

import json
import pickle
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.carla_pomdp import CarlaPOMDP, carla_pomdp, carla_video
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.simulations.episodes import EpisodeRunner


def _make_observation(latitude: float) -> Dict[str, np.ndarray]:
    """Build a multi-modal gnss/camera/lidar observation dict for the fakes."""
    return {
        "gnss": np.array([latitude, 2.0, 0.0]),
        "camera": np.zeros((128, 128, 3), dtype=np.uint8),
        "lidar": np.zeros((10, 4), dtype=np.float32),
    }


def _gnss_only_extractor(observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Observation extractor that keeps only the ``gnss`` modality."""
    return {"gnss": observation["gnss"]}


class FakeCarlaSession:
    """Scripted CARLA-like session: 7-D ground-truth state, multi-modal observation.

    The state is ``[x, y, yaw, vx, vy, lat, heading_err]`` with ``vx`` advancing
    each tick so the ego speed equals the step index and the ego kept perfectly on
    the lane centre (``lat == 0`` and ``heading_err == 0``); the observation is a
    distinct dict with a 3-D GNSS ``[lat, lon, alt]`` vector plus camera and lidar
    arrays. The episode terminates once three ticks have been taken, so terminal
    behavior is deterministic.
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.last_control: Optional[Tuple[float, float, float]] = None
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        del seed
        self.reset_calls += 1
        self._t = 0
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return state, _make_observation(48.0)

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool]:
        self.step_calls += 1
        self.last_control = (throttle, steer, brake)
        self._t += 1
        state = np.array([float(self._t), 0.0, 0.0, float(self._t), 0.0, 0.0, 0.0])
        collided = self._t >= 3
        return state, _make_observation(48.0 + self._t), collided, False


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
    assert reward == pytest.approx(0.9)
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(observation["gnss"], np.array([49.0, 2.0, 0.0]))


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
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert reward == pytest.approx(0.9)


def test_observation_differs_from_ground_truth_state(world: CarlaPOMDP) -> None:
    """The GNSS observation is a partial view, distinct from the true state.

    Purpose: Validates the CARLA state != observation modeling (partial observability).

    Given: A CarlaPOMDP world at its live state
    When: A step is sampled and both next state and observation are read
    Then: The observation is the multi-modal sensor dict, not the 5-D state

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, 0)
    observation = world.sample_observation(next_state, 0)

    assert next_state.shape == (7,)
    assert observation["gnss"].shape == (3,)
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
    assert np.array_equal(samples[0], np.zeros(7))


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
    assert np.array_equal(samples[0]["gnss"], np.array([48.0, 2.0, 0.0]))


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
    arbitrary_state = np.array([99.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
        world.sample_observation(np.array([7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0)


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

    assert np.array_equal(runner.state, np.zeros(7))
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

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        del seed
        self.reset_calls += 1
        self._t = 0
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), _make_observation(48.0)

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool]:
        del throttle, steer, brake
        self.step_calls += 1
        self._t += 1
        position = 0.0 if self._t <= 2 else float(self._t - 2)
        state = np.array([position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return state, _make_observation(48.0), False, False


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


def test_is_equal_observation_true_for_identical_false_for_distinct(world: CarlaPOMDP) -> None:
    """is_equal_observation compares GNSS observation vectors element-wise.

    Purpose: Validates the observation-equality predicate directly.

    Given: Two identical GNSS observations and a third differing one
    When: is_equal_observation compares the identical pair and the differing pair
    Then: The identical pair compares equal and the differing pair compares unequal

    Test type: unit
    """
    observation = np.array([48.0, 2.0, 0.0])
    same = np.array([48.0, 2.0, 0.0])
    different = np.array([48.0, 2.0, 1.0])

    assert world.is_equal_observation(observation, same) is True
    assert world.is_equal_observation(observation, different) is False


def test_reward_subtracts_collision_penalty_on_terminal_step(world: CarlaPOMDP) -> None:
    """The reward rewards along-lane progress and subtracts the collision penalty.

    Purpose: Validates the driving-quality reward and its collision branch.

    Given: A world whose fake session keeps the ego on the lane centre (lat and
        heading_err both zero), cruising straight (steer 0) at a speed equal to the
        step index, terminating on the third tick at speed 3
    When: The world is driven forward three steps, reading each reward
    Then: Each non-terminal reward is the along-lane speed minus the per-step cost
        (speed - 0.1), and the terminal reward additionally subtracts the default
        collision penalty (3.0 - 0.1 - 100.0 == -97.1)

    Test type: unit
    """
    assert world.collision_penalty == 100.0
    state = world._live_state
    rewards = []
    for _ in range(3):
        next_state, _, reward = world.sample_next_step(state, 0)
        rewards.append(reward)
        state = next_state

    assert rewards[0] == pytest.approx(1.0 - 0.1)
    assert rewards[1] == pytest.approx(2.0 - 0.1)
    assert rewards[2] == pytest.approx(3.0 - 0.1 - 100.0)


def test_reward_penalizes_leaving_the_lane(world: CarlaPOMDP) -> None:
    """A lateral offset beyond the threshold subtracts the out-of-lane penalty.

    Purpose: Validates the out-of-lane term of the driving-quality reward.

    Given: A transition cruising straight at 1 m/s but 3 m off the lane centre,
        beyond the default 2 m out_lane_thresh
    When: The reward for that transition is computed (non-terminal, no steering)
    Then: The reward is the along-lane progress minus the out-of-lane penalty and
        the per-step cost (1.0 - 1.0 - 0.1 == -0.1)

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0])

    reward = world._compute_reward(next_state, 0, collided=False, success=False)

    assert reward == pytest.approx(1.0 - 1.0 - 0.1)


def test_reward_penalizes_exceeding_desired_speed(world: CarlaPOMDP) -> None:
    """Longitudinal speed above desired_speed subtracts the overspeed penalty.

    Purpose: Validates the overspeed term of the driving-quality reward.

    Given: A transition cruising straight along the lane at 10 m/s, above the
        default 8 m/s desired_speed
    When: The reward for that transition is computed (non-terminal, no steering)
    Then: The reward is the along-lane progress minus the overspeed penalty and the
        per-step cost (10.0 - 10.0 - 0.1 == -0.1)

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    reward = world._compute_reward(next_state, 0, collided=False, success=False)

    assert reward == pytest.approx(10.0 - 10.0 - 0.1)


def test_reward_penalizes_steering(world: CarlaPOMDP) -> None:
    """A steering action subtracts the squared-steer and lateral-accel penalties.

    Purpose: Validates the steering smoothness terms of the driving-quality reward.

    Given: A transition at 1 m/s along the lane taken with the steer-left preset
        (index 1, steer -0.5)
    When: The reward for that transition is computed (non-terminal)
    Then: The reward is progress minus 5*steer**2, minus 0.2*|steer|*speed**2, minus
        the per-step cost (1.0 - 5*0.25 - 0.2*0.5 - 0.1 == -0.45)

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    reward = world._compute_reward(next_state, 1, collided=False, success=False)

    assert reward == pytest.approx(1.0 - 5 * 0.25 - 0.2 * 0.5 - 0.1)


def test_lane_geometry_returns_signed_offset_and_wrapped_heading_error() -> None:
    """_lane_geometry projects the ego onto the lane and wraps the heading error.

    Purpose: Validates the lane-relative geometry feeding lat/heading_err in state.

    Given: A session whose CARLA map reports a lane centred at the origin pointing
        along +x, with the ego 1.5 m to the lane's left
    When: _lane_geometry is queried for headings of 10 deg and 190 deg
    Then: The lateral offset is +1.5 m and the heading error equals the ego heading
        minus the lane direction, wrapped to [-pi, pi]

    Test type: unit
    """
    lane_transform = SimpleNamespace(
        location=SimpleNamespace(x=0.0, y=0.0), rotation=SimpleNamespace(yaw=0.0)
    )
    waypoint = SimpleNamespace(transform=lane_transform)
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._carla = SimpleNamespace(LaneType=SimpleNamespace(Driving="Driving"))
    session._map = SimpleNamespace(
        get_waypoint=lambda location, project_to_road, lane_type: waypoint
    )
    session._route_xy = None  # no route: fall back to the lane-based reference
    ego_location = SimpleNamespace(x=0.0, y=1.5)

    lateral, heading_err = session._lane_geometry(ego_location, 10.0)
    _, wrapped_heading_err = session._lane_geometry(ego_location, 190.0)

    assert lateral == pytest.approx(1.5)
    assert heading_err == pytest.approx(np.radians(10.0))
    assert wrapped_heading_err == pytest.approx(np.radians(190.0) - 2 * np.pi)


def test_relative_agent_row_expresses_pose_in_ego_frame() -> None:
    """_relative_agent_row rotates another agent's pose into the ego frame.

    Purpose: Validates the ego-frame transform feeding the agent state/obs slots.

    Given: An ego at the origin heading 90 degrees (north, +y) and another agent
        10 m north of it heading 90 degrees at 4 m/s
    When: _relative_agent_row builds the slot row
    Then: The agent maps to 10 m straight ahead (rel_x), 0 m lateral (rel_y), zero
        relative heading, its speed, and a set present flag

    Test type: unit
    """
    row = carla_pomdp._relative_agent_row(
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw_rad=np.radians(90.0),
        other_x=0.0,
        other_y=10.0,
        other_yaw_rad=np.radians(90.0),
        other_speed=4.0,
    )

    assert row[0] == pytest.approx(1.0)
    assert row[1] == pytest.approx(10.0)
    assert row[2] == pytest.approx(0.0, abs=1e-9)
    assert row[3] == pytest.approx(0.0)
    assert row[4] == pytest.approx(4.0)


def test_segment_occludes_only_for_blockers_on_the_sight_line() -> None:
    """_segment_occludes flags blockers between ego and target near the sight line.

    Purpose: Validates the geometric occlusion test for perception hiding.

    Given: An ego at the origin and a target 30 m ahead along +x
    When: A blocker sits on the sight line between them, off to the side, and behind
        the ego
    Then: Only the on-line, in-between blocker occludes the target

    Test type: unit
    """
    on_line = carla_pomdp._segment_occludes(0.0, 0.0, 30.0, 0.0, 15.0, 0.5, 1.5)
    off_to_side = carla_pomdp._segment_occludes(0.0, 0.0, 30.0, 0.0, 15.0, 5.0, 1.5)
    behind_ego = carla_pomdp._segment_occludes(0.0, 0.0, 30.0, 0.0, -5.0, 0.0, 1.5)

    assert on_line is True
    assert off_to_side is False
    assert behind_ego is False


def test_agent_rows_reports_every_nearest_agent_raw() -> None:
    """The world reports every nearest agent at its true pose; it never hides any.

    Purpose: Validates the world emits the raw ground-truth agents channel — range-gating
        and occlusion belong to the planner model, not the world.

    Given: An ego with three tracked neighbours, including one directly behind another on
        the same sight line and one far away that a perception model would drop
    When: _agent_rows builds the agent matrix
    Then: All three slots are present at their true ego-frame poses, with the nearest first;
        no slot is zeroed for range or occlusion

    Test type: unit
    """
    ego = _fake_actor(0, 0.0, 0.0)
    near = _fake_actor(1, 10.0, 0.0)  # 10 m ahead
    behind = _fake_actor(2, 30.0, 0.0)  # behind `near` on the same sight line
    far = _fake_actor(3, 60.0, 0.0)  # far away
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._vehicle = ego
    session._world = _fake_world([ego, near, behind, far])
    session._max_tracked_agents = 3

    rows = session._agent_rows()

    assert rows.shape == (3, carla_pomdp.AGENT_SLOT_WIDTH)
    assert list(rows[:, 0]) == [1.0, 1.0, 1.0]  # all present, none hidden
    assert rows[0][1] == pytest.approx(10.0)  # nearest first
    assert rows[1][1] == pytest.approx(30.0)
    assert rows[2][1] == pytest.approx(60.0)


def test_observation_includes_camera_and_lidar(world: CarlaPOMDP) -> None:
    """A stepped observation is a multi-modal gnss/camera/lidar dict.

    Purpose: Validates the multi-modal observation payload of the CARLA world.

    Given: A CarlaPOMDP world whose fake session returns a dict observation
    When: A step is sampled and the observation is read
    Then: The observation is a dict with gnss/camera/lidar; camera is (H, W, 3)
        uint8 and lidar is (N, 4) float32

    Test type: unit
    """
    state = world._live_state
    next_state = world.sample_next_state(state, 0)
    observation = world.sample_observation(next_state, 0)

    assert isinstance(observation, dict)
    assert set(observation) == {"gnss", "camera", "lidar"}
    assert np.array_equal(observation["gnss"], np.array([49.0, 2.0, 0.0]))
    assert observation["camera"].shape == (128, 128, 3)
    assert observation["camera"].dtype == np.uint8
    assert observation["lidar"].shape[1] == 4
    assert observation["lidar"].dtype == np.float32


def _fake_actor(actor_id: int, x: float, y: float, yaw: float = 0.0, speed: float = 0.0) -> Any:
    """A SimpleNamespace CARLA actor with transform/location/velocity accessors."""
    location = SimpleNamespace(
        x=x, y=y, distance=lambda other, _x=x, _y=y: float(np.hypot(other.x - _x, other.y - _y))
    )
    transform = SimpleNamespace(location=location, rotation=SimpleNamespace(yaw=yaw))
    velocity = SimpleNamespace(x=speed, y=0.0)
    return SimpleNamespace(
        id=actor_id,
        get_location=lambda _loc=location: _loc,
        get_transform=lambda _tf=transform: _tf,
        get_velocity=lambda _vel=velocity: _vel,
        get_traffic_light_state=lambda: "Green",
        get_traffic_light=lambda: None,
    )


def _fake_world(actors: List[Any]) -> Any:
    """A SimpleNamespace CARLA world whose get_actors().filter() returns ``actors``."""
    return SimpleNamespace(get_actors=lambda: SimpleNamespace(filter=lambda pattern: actors))


def _fake_carla_module() -> Any:
    """A minimal stand-in for the ``carla`` module exposing ``TrafficLightState``."""
    return SimpleNamespace(
        TrafficLightState=SimpleNamespace(Red="Red", Yellow="Yellow", Green="Green")
    )


def _bare_session(
    include_camera: bool,
    include_lidar: bool,
    observation_extractor: Optional[Any] = None,
    include_traffic_light: bool = True,
) -> Any:
    """Build a _CarlaSession bypassing __init__ so _read_observation runs sans carla."""
    session = object.__new__(carla_pomdp._CarlaSession)
    session._carla = _fake_carla_module()
    session._observation_extractor = observation_extractor
    session._include_camera = include_camera
    session._include_lidar = include_lidar
    session._include_traffic_light = include_traffic_light
    session._latest_gnss = None
    session._latest_camera = None
    session._latest_lidar = None
    session._camera_height = 128
    session._camera_width = 128
    ego = _fake_actor(0, 0.0, 0.0)
    session._vehicle = ego
    session._world = _fake_world([ego])
    session._max_tracked_agents = 5
    return session


def test_read_observation_omits_disabled_sensor_keys() -> None:
    """Disabled sensors drop their observation keys; gnss always stays.

    Purpose: Validates include_camera/include_lidar gate their observation keys.

    Given: Sessions configured with camera-only and lidar-only sensor sets
    When: _read_observation builds the observation dict
    Then: Only the enabled sensor key appears alongside the always-present gnss, agents
        and traffic_light keys

    Test type: unit
    """
    camera_only = _bare_session(include_camera=True, include_lidar=False)
    lidar_only = _bare_session(include_camera=False, include_lidar=True)

    assert set(camera_only._read_observation()) == {"gnss", "agents", "traffic_light", "camera"}
    assert set(lidar_only._read_observation()) == {"gnss", "agents", "traffic_light", "lidar"}


def test_read_observation_omits_traffic_light_when_disabled() -> None:
    """Disabling include_traffic_light drops the traffic_light observation key.

    Purpose: Validates the traffic-light ground-truth oracle can be withheld entirely.

    Given: A session with camera, lidar and traffic-light all disabled
    When: _read_observation builds the observation dict
    Then: Only gnss and agents remain — no traffic_light key

    Test type: unit
    """
    session = _bare_session(include_camera=False, include_lidar=False, include_traffic_light=False)

    assert set(session._read_observation()) == {"gnss", "agents"}


def test_read_observation_falls_back_to_zeros_before_sensor_data() -> None:
    """Before a sensor callback fires, observation values fall back to zeros.

    Purpose: Validates the zero-filled fallbacks for gnss/camera/lidar.

    Given: A session with both sensors enabled, no other vehicles, and no callback
        data yet
    When: _read_observation is called
    Then: gnss is zeros(3), the agents block is an all-empty K*slot vector, camera is
        (H, W, 3) uint8 zeros, and lidar is (0, 4) float32

    Test type: unit
    """
    session = _bare_session(include_camera=True, include_lidar=True)

    observation = session._read_observation()

    assert np.array_equal(observation["gnss"], np.zeros(3))
    assert np.array_equal(observation["agents"], np.zeros(5 * carla_pomdp.AGENT_SLOT_WIDTH))
    assert observation["camera"].shape == (128, 128, 3)
    assert observation["camera"].dtype == np.uint8
    assert not observation["camera"].any()
    assert observation["lidar"].shape == (0, 4)
    assert observation["lidar"].dtype == np.float32


def test_read_observation_applies_observation_extractor() -> None:
    """A supplied observation extractor replaces the emitted observation.

    Purpose: Validates that _read_observation routes the full dict through the
        injected observation_extractor and returns its result.

    Given: A bare session with both sensors enabled and a gnss-only extractor
    When: _read_observation builds the full dict and applies the extractor
    Then: The returned observation contains only the ``gnss`` key produced by the
        extractor, dropping agents/camera/lidar

    Test type: unit
    """
    session = _bare_session(
        include_camera=True, include_lidar=True, observation_extractor=_gnss_only_extractor
    )

    observation = session._read_observation()

    assert set(observation) == {"gnss"}
    assert np.array_equal(observation["gnss"], np.zeros(3))


def test_observation_extractor_forwarded_from_env_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CarlaPOMDP stores the extractor and threads it into the session build.

    Purpose: Validates the observation_extractor constructor argument is retained
        and forwarded to the underlying _CarlaSession by _get_session.

    Given: A CarlaPOMDP constructed with a gnss-only observation extractor and a
        recording _CarlaSession stub
    When: The environment builds its session via _get_session
    Then: The environment exposes the extractor and _CarlaSession receives the same
        callable in its keyword arguments

    Test type: unit
    """
    captured: Dict[str, Any] = {}

    def _recording_session(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "session"

    monkeypatch.setattr(carla_pomdp, "_CarlaSession", _recording_session)
    env = CarlaPOMDP(discount_factor=0.95, observation_extractor=_gnss_only_extractor)

    assert env.observation_extractor is _gnss_only_extractor
    assert env._get_session() == "session"
    assert captured["observation_extractor"] is _gnss_only_extractor


def test_include_flags_thread_into_session(fake_session: FakeCarlaSession) -> None:
    """The CarlaPOMDP include flags configure the underlying session build.

    Purpose: Validates include_camera/include_lidar reach _CarlaSession config.

    Given: A CarlaPOMDP constructed with camera enabled and lidar disabled
    When: Its public sensor configuration is inspected
    Then: The include flags and merged config defaults match the constructor args

    Test type: unit
    """
    del fake_session
    env = CarlaPOMDP(discount_factor=0.95, include_camera=True, include_lidar=False)

    assert env.include_camera is True
    assert env.include_lidar is False
    assert env.observation_camera_config["image_size_x"] == "128"
    assert env.lidar_config["channels"] == "32"


def _write_single_server_pool(pool_dir: Path, rpc_port: int, tm_port: int) -> None:
    """Hand-write a one-server pool spec + lease file (no server processes)."""
    (pool_dir / "server_0.lease").touch()
    spec = {
        "host": "127.0.0.1",
        "servers": [
            {
                "index": 0,
                "rpc_port": rpc_port,
                "traffic_manager_port": tm_port,
                "lease_file": "server_0.lease",
            }
        ],
    }
    (pool_dir / "pool.json").write_text(json.dumps(spec))


def test_server_pool_dir_survives_pickle_roundtrip(
    fake_session: FakeCarlaSession, tmp_path: Path
) -> None:
    """server_pool_dir is plain config, so it survives pickling to workers.

    Purpose: Validates that the pool wiring reaches joblib workers, which receive
        the environment via pickle and must re-resolve their lease lazily.

    Given: A CarlaPOMDP constructed with a server_pool_dir
    When: It is pickled and unpickled
    Then: server_pool_dir is preserved and the live session is dropped

    Test type: unit
    """
    del fake_session
    env = CarlaPOMDP(discount_factor=0.95, server_pool_dir=tmp_path)
    restored = pickle.loads(pickle.dumps(env))

    assert restored.server_pool_dir == str(tmp_path)
    assert restored._session is None


def test_get_session_resolves_ports_from_pool_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a pool configured, the session connects to the leased server's ports.

    Purpose: Validates that _get_session overrides the static host/port/
        traffic_manager_port with the per-process pool lease.

    Given: A hand-written one-server pool directory and a recording _CarlaSession
        stub, and a CarlaPOMDP constructed with server_pool_dir and default ports
    When: The environment builds its session via _get_session
    Then: _CarlaSession receives the leased rpc/traffic-manager ports and host,
        not the constructor defaults

    Test type: unit
    """
    captured: Dict[str, Any] = {}

    def _recording_session(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "session"

    monkeypatch.setattr(carla_pomdp, "_CarlaSession", _recording_session)
    _write_single_server_pool(tmp_path, rpc_port=2404, tm_port=8321)
    env = CarlaPOMDP(discount_factor=0.95, server_pool_dir=tmp_path)

    assert env._get_session() == "session"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 2404
    assert captured["traffic_manager_port"] == 8321


def test_get_session_uses_static_ports_without_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a pool, the session connects to the constructor's static ports.

    Purpose: Validates the default (no-pool) connection path forwards the
        configured host/port/traffic_manager_port to _CarlaSession.

    Given: A recording _CarlaSession stub and a CarlaPOMDP with explicit
        host/port/traffic_manager_port and no server_pool_dir
    When: The environment builds its session via _get_session
    Then: _CarlaSession receives exactly the configured endpoints

    Test type: unit
    """
    captured: Dict[str, Any] = {}

    def _recording_session(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "session"

    monkeypatch.setattr(carla_pomdp, "_CarlaSession", _recording_session)
    env = CarlaPOMDP(discount_factor=0.95, host="myhost", port=2345, traffic_manager_port=8123)

    assert env._get_session() == "session"
    assert captured["host"] == "myhost"
    assert captured["port"] == 2345
    assert captured["traffic_manager_port"] == 8123


def test_hash_observation_handles_dict_observations(world: CarlaPOMDP) -> None:
    """Dict observations hash equal when equal and differ on a changed modality.

    Purpose: Validates dict-aware hash_observation for multi-modal payloads.

    Given: Two equal sensor dicts and a third with a differing camera frame
    When: hash_observation is computed for each
    Then: Equal dicts hash equally and the differing dict hashes differently

    Test type: unit
    """
    observation = _make_observation(48.0)
    same = _make_observation(48.0)
    different = _make_observation(48.0)
    different["camera"] = np.ones((128, 128, 3), dtype=np.uint8)

    assert world.hash_observation(observation) == world.hash_observation(same)
    assert world.hash_observation(observation) != world.hash_observation(different)


def test_is_equal_observation_handles_dict_observations(world: CarlaPOMDP) -> None:
    """is_equal_observation compares dict observations modality by modality.

    Purpose: Validates dict-aware equality, including dict/non-dict mismatches.

    Given: Two equal sensor dicts, one with a differing lidar cloud, and an array
    When: is_equal_observation compares each pairing
    Then: Equal dicts compare equal; a changed modality and a dict/array mix differ

    Test type: unit
    """
    observation = _make_observation(48.0)
    same = _make_observation(48.0)
    different = _make_observation(48.0)
    different["lidar"] = np.ones((10, 4), dtype=np.float32)

    assert world.is_equal_observation(observation, same) is True
    assert world.is_equal_observation(observation, different) is False
    assert world.is_equal_observation(observation, np.zeros(3)) is False


# ── Evaluation metrics (compute_metrics / get_metric_names) ──────────────────


def _ego_state(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    """Build a minimal ego state with the given world position and velocity."""
    state = np.zeros(carla_pomdp.EGO_STATE_WIDTH)
    state[0], state[1] = x, y
    state[3], state[4] = vx, vy
    return state


def _metrics_history(
    transitions: List[Tuple[np.ndarray, np.ndarray]], reach_terminal: bool
) -> History:
    """Build a History from (state, next_state) ego transitions for metric tests."""
    belief: Belief = WeightedParticleBelief([np.zeros(1), np.zeros(1)], np.array([0.0, -1.0]))
    steps = [
        StepData(
            state=state,
            action=(0.5, 0.0, 0.0),
            next_state=next_state,
            observation={"gnss": np.zeros(3)},
            reward=0.0,
            belief=belief,
        )
        for state, next_state in transitions
    ]
    return History(
        history=steps,
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=len(steps),
        reach_terminal_state=reach_terminal,
        policy_run_data=[],
    )


def _metric_by_name(metrics: List[MetricValue], name: str) -> MetricValue:
    """Return the single MetricValue with the given name."""
    return next(metric for metric in metrics if metric.name == name)


def _light_state(
    x: float, vx: float, present: float, code: float, rel_x: float = 2.0
) -> np.ndarray:
    """Build a full ego+agents+light state with the given ego motion and light slot."""
    offset = carla_pomdp.EGO_STATE_WIDTH + carla_pomdp.DEFAULT_MAX_TRACKED_AGENTS * (
        carla_pomdp.AGENT_SLOT_WIDTH
    )
    state = np.zeros(offset + carla_pomdp.LIGHT_SLOT_WIDTH)
    state[0], state[3] = x, vx
    state[offset], state[offset + 1], state[offset + 3] = present, rel_x, code
    return state


def test_get_metric_names_lists_the_full_carla_metric_set() -> None:
    """Metric names advertise driving quality plus the traffic-light metrics.

    Purpose: Validates get_metric_names exposes exactly the CARLA metric set

    Given: A CarlaPOMDP world
    When: get_metric_names is called
    Then: It returns the driving-quality and traffic-light metric names

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)

    assert env.get_metric_names() == [
        "collision_rate",
        "success_rate",
        "route_completion",
        "average_progress",
        "average_speed",
        "red_light_violation_rate",
        "red_light_violation_count",
        "traffic_light_malfunction_count",
        "near_miss_count",
        "min_vehicle_distance",
    ]


def test_compute_metrics_empty_histories_returns_empty_list() -> None:
    """No histories yields no metrics.

    Purpose: Validates compute_metrics degrades gracefully on empty input

    Given: A CarlaPOMDP world
    When: compute_metrics is called with an empty history list
    Then: An empty list is returned (no division by zero, no CI on empty data)

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)

    assert not env.compute_metrics([])


def test_compute_metrics_emits_the_three_named_metrics() -> None:
    """A single episode produces all three named metrics.

    Purpose: Validates compute_metrics returns the full CARLA metric set

    Given: A CarlaPOMDP world and one one-step episode
    When: compute_metrics is called
    Then: Exactly the three CARLA metric names are emitted

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history([(_ego_state(0.0, 0.0), _ego_state(1.0, 0.0, 1.0))], False)

    metrics = env.compute_metrics([history])

    assert {metric.name for metric in metrics} == set(env.get_metric_names())


def test_collision_rate_is_fraction_of_episodes_ending_terminal() -> None:
    """Collision rate averages the per-episode terminal flag.

    Purpose: Validates collision_rate equals the fraction of terminal episodes

    Given: Four episodes, two of which reached a terminal (collision) state
    When: compute_metrics is called
    Then: collision_rate is 0.5

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    transition = [(_ego_state(0.0, 0.0), _ego_state(1.0, 0.0, 1.0))]
    histories = [
        _metrics_history(transition, reach_terminal=True),
        _metrics_history(transition, reach_terminal=False),
        _metrics_history(transition, reach_terminal=True),
        _metrics_history(transition, reach_terminal=False),
    ]

    metrics = env.compute_metrics(histories)

    assert _metric_by_name(metrics, "collision_rate").value == pytest.approx(0.5)


def test_average_progress_sums_euclidean_path_length() -> None:
    """Progress is the ground distance travelled along the ego path.

    Purpose: Validates average_progress integrates the ego's Euclidean path length

    Given: One episode moving (0,0)->(3,0)->(3,4)
    When: compute_metrics is called
    Then: average_progress is 7.0 metres (3 + 4)

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (_ego_state(0.0, 0.0), _ego_state(3.0, 0.0)),
            (_ego_state(3.0, 0.0), _ego_state(3.0, 4.0)),
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "average_progress").value == pytest.approx(7.0)


def test_average_speed_averages_ego_velocity_magnitude() -> None:
    """Speed is the mean magnitude of the ego velocity over the trajectory.

    Purpose: Validates average_speed averages sqrt(vx**2 + vy**2) over steps

    Given: One episode whose next-state velocities are (3,4) then (6,8)
    When: compute_metrics is called
    Then: average_speed is 7.5 m/s (mean of 5 and 10)

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (_ego_state(0.0, 0.0), _ego_state(1.0, 0.0, 3.0, 4.0)),
            (_ego_state(1.0, 0.0, 3.0, 4.0), _ego_state(2.0, 0.0, 6.0, 8.0)),
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "average_speed").value == pytest.approx(7.5)


def test_metric_confidence_bounds_bracket_value_across_episodes() -> None:
    """Confidence bounds bracket the reported mean for varied episodes.

    Purpose: Validates the reported value lies within its confidence interval

    Given: Two episodes with different path lengths (5 and 10 metres)
    When: compute_metrics is called
    Then: average_progress is 7.5 and its confidence bounds bracket that value

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    histories = [
        _metrics_history([(_ego_state(0.0, 0.0), _ego_state(5.0, 0.0))], reach_terminal=False),
        _metrics_history([(_ego_state(0.0, 0.0), _ego_state(10.0, 0.0))], reach_terminal=False),
    ]

    progress = _metric_by_name(env.compute_metrics(histories), "average_progress")

    assert progress.value == pytest.approx(7.5)
    assert progress.lower_confidence_bound <= progress.value <= progress.upper_confidence_bound


def test_red_light_crossing_counts_as_a_violation() -> None:
    """Crossing a stop line while the light is red is a red-light violation.

    Purpose: Validates the red-light metrics count a moving crossing on red

    Given: One episode where the ego is affiliated to a RED light and moving, then crosses
        (the light slot's present flag goes 1 -> 0)
    When: compute_metrics is called
    Then: red_light_violation_count is 1, its rate is 1.0, and malfunction count is 0

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (
                _light_state(0.0, 5.0, 1.0, carla_pomdp.TRAFFIC_LIGHT_RED),
                _light_state(1.0, 5.0, 0.0, 0.0),
            )
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "red_light_violation_count").value == pytest.approx(1.0)
    assert _metric_by_name(metrics, "red_light_violation_rate").value == pytest.approx(1.0)
    assert _metric_by_name(metrics, "traffic_light_malfunction_count").value == pytest.approx(0.0)


def test_green_crossing_is_not_a_violation() -> None:
    """Crossing on green is a legal functioning-light pass, not a violation.

    Purpose: Validates a green crossing lowers the violation rate to zero

    Given: One episode where the ego crosses while the light is GREEN
    When: compute_metrics is called
    Then: red_light_violation_count and rate are both 0

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (
                _light_state(0.0, 5.0, 1.0, carla_pomdp.TRAFFIC_LIGHT_GREEN),
                _light_state(1.0, 5.0, 0.0, 0.0),
            )
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "red_light_violation_count").value == pytest.approx(0.0)
    assert _metric_by_name(metrics, "red_light_violation_rate").value == pytest.approx(0.0)


def test_malfunctioning_light_crossing_recorded_separately() -> None:
    """Crossing an off/unknown light is recorded as a malfunction, never a violation.

    Purpose: Validates a non-operating light is scored separately from red-running

    Given: One episode where the ego crosses while the light is OFF
    When: compute_metrics is called
    Then: traffic_light_malfunction_count is 1 and red_light_violation_count is 0

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (
                _light_state(0.0, 5.0, 1.0, carla_pomdp.TRAFFIC_LIGHT_OFF),
                _light_state(1.0, 5.0, 0.0, 0.0),
            )
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "traffic_light_malfunction_count").value == pytest.approx(1.0)
    assert _metric_by_name(metrics, "red_light_violation_count").value == pytest.approx(0.0)


def test_stopping_at_a_red_light_is_not_a_violation() -> None:
    """Being at a red light while stopped is not counted as a crossing.

    Purpose: Validates the moving-speed gate excludes a car halted at the line

    Given: One episode where the ego is at a RED light but not moving when affiliation drops
    When: compute_metrics is called
    Then: no red-light violation is counted

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history(
        [
            (
                _light_state(0.0, 0.0, 1.0, carla_pomdp.TRAFFIC_LIGHT_RED),
                _light_state(0.0, 0.0, 0.0, 0.0),
            )
        ],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "red_light_violation_count").value == pytest.approx(0.0)


def _agent_state(rel_x: float, rel_y: float = 0.0) -> np.ndarray:
    """Full ego+agents+light state carrying one present agent at ``(rel_x, rel_y)``."""
    offset = carla_pomdp.EGO_STATE_WIDTH + carla_pomdp.DEFAULT_MAX_TRACKED_AGENTS * (
        carla_pomdp.AGENT_SLOT_WIDTH
    )
    state = np.zeros(offset + carla_pomdp.LIGHT_SLOT_WIDTH)
    agents = carla_pomdp.EGO_STATE_WIDTH
    state[agents], state[agents + 1], state[agents + 2] = 1.0, rel_x, rel_y
    return state


def test_near_miss_counted_when_a_vehicle_comes_within_threshold() -> None:
    """A close pass without a collision is counted as a near-miss.

    Purpose: Validates the near-miss metric flags a sub-threshold approach

    Given: One episode where the nearest vehicle is 1 m from the ego (no collision)
    When: compute_metrics is called
    Then: near_miss_count is 1 and min_vehicle_distance is 1 m

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history([(_agent_state(1.0), _agent_state(1.0))], reach_terminal=False)

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "near_miss_count").value == pytest.approx(1.0)
    assert _metric_by_name(metrics, "min_vehicle_distance").value == pytest.approx(1.0)


def test_no_near_miss_when_vehicles_stay_far() -> None:
    """A comfortable gap to other vehicles yields no near-miss.

    Purpose: Validates the near-miss metric ignores well-separated vehicles

    Given: One episode whose nearest vehicle stays 10 m away
    When: compute_metrics is called
    Then: near_miss_count is 0 and min_vehicle_distance is 10 m

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95)
    history = _metrics_history([(_agent_state(10.0), _agent_state(10.0))], reach_terminal=False)

    metrics = env.compute_metrics([history])

    assert _metric_by_name(metrics, "near_miss_count").value == pytest.approx(0.0)
    assert _metric_by_name(metrics, "min_vehicle_distance").value == pytest.approx(10.0)


def _lidar_vehicle_blob(center_x: float) -> np.ndarray:
    """A 5x5 grid of vehicle-height lidar returns centred ``center_x`` metres ahead."""
    grid = np.linspace(-0.4, 0.4, 5)
    xx, yy = np.meshgrid(grid, grid)
    points = np.zeros((xx.size, 4), dtype=np.float32)
    points[:, 0] = center_x + xx.ravel()
    points[:, 1] = yy.ravel()
    points[:, 2] = -1.0
    return points


class _LidarVehicleSession:
    """A fake session whose observation carries a lidar vehicle 8 m ahead (empty agents oracle)."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self._t = 0

    def _observation(self) -> Dict[str, np.ndarray]:
        return {
            "gnss": np.zeros(3),
            "agents": np.zeros(DEFAULT_MAX_TRACKED_AGENTS * AGENT_SLOT_WIDTH),
            "lidar": _lidar_vehicle_blob(8.0),
            "camera": np.zeros((32, 32, 3), dtype=np.uint8),
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        del seed
        self.reset_calls += 1
        self._t = 0
        return np.zeros(7), self._observation()

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool]:
        del throttle, steer, brake
        self.step_calls += 1
        self._t += 1
        state = np.array([float(self._t), 0.0, 0.0, float(self._t), 0.0, 0.0, 0.0])
        return state, self._observation(), self._t >= 3, False


def test_world_emits_raw_agents_never_perceived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The world emits the session's raw agents block unchanged; it never perceives.

    Purpose: Validates perception lives only on the planner's generative model, so the world
        passes its raw (ground-truth) agents channel through untouched even when the sensors
        would support a perceived detection.

    Given: A world and a session whose lidar shows a vehicle 8 m ahead but whose ground-truth
        agents channel is all-empty
    When: a step is taken and the observation is read
    Then: the emitted agents block is the raw (empty) channel, not a lidar-perceived one

    Test type: integration
    """
    session = _LidarVehicleSession()
    monkeypatch.setattr(CarlaPOMDP, "_get_session", lambda self: session)
    world = CarlaPOMDP(discount_factor=0.95)
    world.initial_state_dist().sample()  # reset to establish the live state
    state = world._live_state

    next_state = world.sample_next_state(state, 0)
    observation = world.sample_observation(next_state, 0)

    assert np.all(np.asarray(observation["agents"]) == 0.0)


# ── Destination / route support ──────────────────────────────────────────────


def _route_session(route_xy: np.ndarray, route_yaw: np.ndarray, goal_radius: float = 5.0) -> Any:
    """Bare _CarlaSession with a hand-set route polyline and no CARLA connection."""
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._route_xy = route_xy
    session._route_yaw = route_yaw
    segment_lengths = np.hypot(np.diff(route_xy[:, 0]), np.diff(route_xy[:, 1]))
    session._route_cumlen = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    session._route_index = 0
    session._goal_xy = route_xy[-1]
    session._goal_radius = goal_radius
    return session


def _fake_waypoint(x: float, y: float, yaw_deg: float = 0.0) -> SimpleNamespace:
    """A CARLA-like waypoint with transform.location / transform.rotation."""
    return SimpleNamespace(
        transform=SimpleNamespace(
            location=SimpleNamespace(x=x, y=y),
            rotation=SimpleNamespace(yaw=yaw_deg),
        )
    )


def test_route_geometry_measures_offset_against_route_not_lane() -> None:
    """With a route present, lat/heading_err are measured against the route line.

    Purpose: Validates the route-relative geometry that redefines the ego state's
        lat/heading_err fields when a destination route exists

    Given: A session with a straight route along +x and an ego 2 m to the route's
        left heading 30 degrees
    When: _lane_geometry is queried
    Then: The lateral offset is +2 m and the heading error is 30 degrees, without
        any CARLA map lane projection being consulted

    Test type: unit
    """
    route_xy = np.array([[float(x), 0.0] for x in range(0, 20, 2)])
    session = _route_session(route_xy, np.zeros(len(route_xy)))
    session._map = None  # would raise if the lane fallback were consulted

    lateral, heading_err = session._lane_geometry(SimpleNamespace(x=4.0, y=2.0), 30.0)

    assert lateral == pytest.approx(2.0)
    assert heading_err == pytest.approx(np.radians(30.0))


def test_route_geometry_index_advances_monotonically() -> None:
    """The nearest-route-waypoint index never moves backwards.

    Purpose: Validates the monotonic projection cache that keeps a self-crossing
        route from snapping the reference point back to an earlier segment

    Given: A straight route and an ego queried at x=10 and then back at x=2
    When: _lane_geometry is queried at both positions in order
    Then: The second query still references the x=10 waypoint (index unchanged)

    Test type: unit
    """
    route_xy = np.array([[float(x), 0.0] for x in range(0, 20, 2)])
    session = _route_session(route_xy, np.zeros(len(route_xy)))

    session._lane_geometry(SimpleNamespace(x=10.0, y=0.0), 0.0)
    index_after_forward = session._route_index
    session._lane_geometry(SimpleNamespace(x=2.0, y=0.0), 0.0)

    assert index_after_forward == 5
    assert session._route_index == 5


def test_read_state_goal_reports_destination_and_progress() -> None:
    """The goal slot carries the destination and the covered route fraction.

    Purpose: Validates the ground-truth goal slot appended to the world state

    Given: A session with a 18 m straight route whose ego has advanced to x=9
    When: _lane_geometry has projected the ego and _read_state_goal is read
    Then: The slot is [goal_x, goal_y, 0.5]

    Test type: unit
    """
    route_xy = np.array([[float(x), 0.0] for x in range(0, 20, 2)])
    session = _route_session(route_xy, np.zeros(len(route_xy)))

    session._lane_geometry(SimpleNamespace(x=9.0, y=0.0), 0.0)
    goal_slot = session._read_state_goal()

    assert goal_slot == pytest.approx(np.array([18.0, 0.0, 8.0 / 18.0]))


def test_reached_goal_is_true_within_goal_radius() -> None:
    """_reached_goal fires exactly when the ego is within goal_radius of the goal.

    Purpose: Validates the success terminal condition

    Given: A session with a route ending at (18, 0) and goal_radius 5
    When: _reached_goal is evaluated with the ego at x=10 and then x=14
    Then: It is False at 8 m from the goal and True at 4 m

    Test type: unit
    """
    route_xy = np.array([[float(x), 0.0] for x in range(0, 20, 2)])
    session = _route_session(route_xy, np.zeros(len(route_xy)), goal_radius=5.0)

    session._vehicle = SimpleNamespace(get_location=lambda: SimpleNamespace(x=10.0, y=0.0))
    far = session._reached_goal()
    session._vehicle = SimpleNamespace(get_location=lambda: SimpleNamespace(x=14.0, y=0.0))
    near = session._reached_goal()

    assert far is False
    assert near is True


def test_store_route_builds_polyline_and_cumulative_length() -> None:
    """_store_route extracts xy/yaw arrays and cumulative arc length from a route.

    Purpose: Validates the conversion of a GlobalRoutePlanner route into the
        session's numpy polyline representation

    Given: A traced route of three waypoints spanning two 3-4-5 segments
    When: _store_route stores it
    Then: The xy/yaw arrays, cumulative lengths, and goal point match the waypoints

    Test type: unit
    """
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    route = [
        (_fake_waypoint(0.0, 0.0, 0.0), "LANEFOLLOW"),
        (_fake_waypoint(3.0, 4.0, 45.0), "LANEFOLLOW"),
        (_fake_waypoint(6.0, 8.0, 90.0), "LANEFOLLOW"),
    ]

    session._store_route(route)

    assert session._route_xy == pytest.approx(np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]))
    assert session._route_yaw == pytest.approx(np.radians([0.0, 45.0, 90.0]))
    assert session._route_cumlen == pytest.approx(np.array([0.0, 5.0, 10.0]))
    assert session._route_index == 0
    assert session._goal_xy == pytest.approx(np.array([6.0, 8.0]))


def test_sample_destination_route_skips_too_short_routes() -> None:
    """Sampled destinations must yield a route of at least min_route_length.

    Purpose: Validates the min-route-length filter of destination sampling

    Given: A map with a near spawn point (10 m route) and a far one (200 m route)
    When: _sample_destination_route is called with min_route_length 100
    Then: The far spawn point's route is returned regardless of sampling order

    Test type: unit
    """
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._min_route_length = 100.0
    session._rng = np.random.default_rng(0)
    near = SimpleNamespace(location=SimpleNamespace(x=10.0, y=0.0))
    far = SimpleNamespace(location=SimpleNamespace(x=200.0, y=0.0))
    session._map = SimpleNamespace(get_spawn_points=lambda: [near, far])

    def _trace(start: Any, goal: Any) -> List[Tuple[SimpleNamespace, str]]:
        del start
        return [
            (_fake_waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_fake_waypoint(goal.x, goal.y), "LANEFOLLOW"),
        ]

    session._route_planner = SimpleNamespace(trace_route=_trace)

    route = session._sample_destination_route(SimpleNamespace(x=0.0, y=0.0))

    assert route[-1][0].transform.location.x == pytest.approx(200.0)


def test_sample_destination_route_raises_when_no_spawn_is_far_enough() -> None:
    """Destination sampling fails loudly when no route can satisfy the minimum.

    Purpose: Validates the descriptive error for unsatisfiable min_route_length

    Given: A map whose only spawn point is 10 m of route away
    When: _sample_destination_route is called with min_route_length 100
    Then: RuntimeError is raised naming min_route_length

    Test type: unit
    """
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._min_route_length = 100.0
    session._rng = np.random.default_rng(0)
    near = SimpleNamespace(location=SimpleNamespace(x=10.0, y=0.0))
    session._map = SimpleNamespace(get_spawn_points=lambda: [near])
    session._route_planner = SimpleNamespace(
        trace_route=lambda start, goal: [
            (_fake_waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_fake_waypoint(goal.x, goal.y), "LANEFOLLOW"),
        ]
    )

    with pytest.raises(RuntimeError, match="min_route_length"):
        session._sample_destination_route(SimpleNamespace(x=0.0, y=0.0))


def test_get_route_planner_requires_carla_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route planning without CARLA_ROOT fails with a descriptive error.

    Purpose: Validates the guarded lazy import of GlobalRoutePlanner

    Given: A session with no CARLA_ROOT in the environment
    When: _get_route_planner is called
    Then: RuntimeError is raised naming CARLA_ROOT

    Test type: unit
    """
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._route_planner = None
    monkeypatch.delenv("CARLA_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="CARLA_ROOT"):
        session._get_route_planner()


class _GoalReachingSession(FakeCarlaSession):
    """Scripted session whose third tick reaches the destination (no collision)."""

    def step(
        self, throttle: float, steer: float, brake: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool]:
        self.step_calls += 1
        self.last_control = (throttle, steer, brake)
        self._t += 1
        state = np.array([float(self._t), 0.0, 0.0, float(self._t), 0.0, 0.0, 0.0])
        return state, _make_observation(48.0 + self._t), False, self._t >= 3


def test_reaching_the_goal_terminates_with_success_bonus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrival terminates the episode and earns +success_reward, not -collision_penalty.

    Purpose: Validates that the world distinguishes the success terminal from the
        collision terminal in both the terminal flag and the reward

    Given: A world whose session reports reached_goal=True on the third step
    When: The world is stepped three times
    Then: The first two rewards carry no bonus, the third adds success_reward, and
        the world is terminal after the third step

    Test type: unit
    """
    session = _GoalReachingSession()
    monkeypatch.setattr(CarlaPOMDP, "_get_session", lambda self: session)
    world = CarlaPOMDP(discount_factor=0.95, success_reward=100.0)
    state = world.initial_state_dist().sample()[0]

    rewards = []
    for _ in range(3):
        state, _, reward = world.sample_next_step(state, 0)
        rewards.append(reward)

    assert rewards[0] == pytest.approx(1.0 - 0.1)
    assert rewards[1] == pytest.approx(2.0 - 0.1)
    assert rewards[2] == pytest.approx(3.0 - 0.1 + 100.0)
    assert world.is_terminal(state) is True


def test_destination_options_thread_into_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destination/goal constructor options reach the session build.

    Purpose: Validates the new goal kwargs are stored publicly (config) and
        forwarded to _CarlaSession

    Given: A recording _CarlaSession stub and a CarlaPOMDP with destination,
        goal_radius, and min_route_length set
    When: The environment builds its session via _get_session
    Then: _CarlaSession receives exactly those values

    Test type: unit
    """
    captured: Dict[str, Any] = {}

    def _recording_session(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "session"

    monkeypatch.setattr(carla_pomdp, "_CarlaSession", _recording_session)
    env = CarlaPOMDP(
        discount_factor=0.95,
        destination=(120.0, -40.0),
        goal_radius=3.0,
        min_route_length=50.0,
        success_reward=42.0,
    )

    assert env._get_session() == "session"
    assert captured["destination"] == (120.0, -40.0)
    assert captured["goal_radius"] == 3.0
    assert captured["min_route_length"] == 50.0
    assert env.success_reward == 42.0


def test_config_id_changes_with_destination(fake_session: FakeCarlaSession) -> None:
    """The destination is part of the environment configuration identity.

    Purpose: Validates destination participates in config_id so cached results
        from different goals never collide

    Given: Two CarlaPOMDPs differing only in destination
    When: Their config_id values are compared
    Then: They differ

    Test type: configuration
    """
    del fake_session
    env_a = CarlaPOMDP(discount_factor=0.95, destination=(10.0, 0.0))
    env_b = CarlaPOMDP(discount_factor=0.95, destination=(20.0, 0.0))

    assert env_a.config_id != env_b.config_id


def _goal_state(x: float, y: float, goal_x: float, goal_y: float, frac: float) -> np.ndarray:
    """Build a full ego+agents+light+goal state for the goal-metric tests."""
    offset = carla_pomdp.EGO_STATE_WIDTH + carla_pomdp.DEFAULT_MAX_TRACKED_AGENTS * (
        carla_pomdp.AGENT_SLOT_WIDTH
    )
    state = np.zeros(offset + carla_pomdp.LIGHT_SLOT_WIDTH + carla_pomdp.GOAL_SLOT_WIDTH)
    state[0], state[1] = x, y
    goal_offset = offset + carla_pomdp.LIGHT_SLOT_WIDTH
    state[goal_offset], state[goal_offset + 1], state[goal_offset + 2] = goal_x, goal_y, frac
    return state


def test_success_rate_and_route_completion_read_the_goal_slot() -> None:
    """success_rate and route_completion come from the final state's goal slot.

    Purpose: Validates the goal-slot-driven metrics

    Given: One episode ending 2 m from its goal at 90% completion and one ending
        50 m away at 40% completion, with goal_radius 5
    When: compute_metrics is called
    Then: success_rate is 0.5 and route_completion is 0.65

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95, goal_radius=5.0)
    reached = _metrics_history(
        [(_goal_state(0.0, 0.0, 18.0, 0.0, 0.0), _goal_state(16.0, 0.0, 18.0, 0.0, 0.9))],
        reach_terminal=True,
    )
    missed = _metrics_history(
        [(_goal_state(0.0, 0.0, 50.0, 0.0, 0.0), _goal_state(0.0, 0.0, 50.0, 0.0, 0.4))],
        reach_terminal=False,
    )

    metrics = env.compute_metrics([reached, missed])

    assert _metric_by_name(metrics, "success_rate").value == pytest.approx(0.5)
    assert _metric_by_name(metrics, "route_completion").value == pytest.approx(0.65)


def test_collision_rate_excludes_goal_reaching_terminals() -> None:
    """A terminal episode that reached its goal is a success, not a collision.

    Purpose: Validates collision_rate no longer counts every terminal episode now
        that reaching the destination also terminates

    Given: Two terminal episodes: one ending at its goal, one ending far from it
    When: compute_metrics is called
    Then: collision_rate is 0.5 and success_rate is 0.5

    Test type: unit
    """
    env = CarlaPOMDP(discount_factor=0.95, goal_radius=5.0)
    success = _metrics_history(
        [(_goal_state(0.0, 0.0, 18.0, 0.0, 0.0), _goal_state(18.0, 0.0, 18.0, 0.0, 1.0))],
        reach_terminal=True,
    )
    crash = _metrics_history(
        [(_goal_state(0.0, 0.0, 50.0, 0.0, 0.0), _goal_state(5.0, 0.0, 50.0, 0.0, 0.1))],
        reach_terminal=True,
    )

    metrics = env.compute_metrics([success, crash])

    assert _metric_by_name(metrics, "collision_rate").value == pytest.approx(0.5)
    assert _metric_by_name(metrics, "success_rate").value == pytest.approx(0.5)


def test_plan_route_handles_destination_at_spawn() -> None:
    """A spawn landing on the destination still yields a valid (trivial) route.

    Purpose: Regression for reset crashing when a random spawn coincides with the
        configured destination and the route planner returns a degenerate trace

    Given: A session whose destination equals the ego spawn and whose route planner
        returns a single-waypoint route
    When: _plan_route runs
    Then: A straight start->goal polyline is stored and the goal counts as reached

    Test type: unit
    """
    session: Any = object.__new__(carla_pomdp._CarlaSession)
    session._destination = (5.0, 5.0)
    session._goal_radius = 5.0
    session._carla = SimpleNamespace(Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z))
    spawn = SimpleNamespace(x=5.0, y=5.0, z=0.0)
    session._vehicle = SimpleNamespace(
        get_transform=lambda: SimpleNamespace(location=spawn),
        get_location=lambda: spawn,
    )
    session._route_planner = SimpleNamespace(trace_route=lambda start, goal: [])

    session._plan_route()

    assert session._route_xy == pytest.approx(np.array([[5.0, 5.0], [5.0, 5.0]]))
    assert session._route_cumlen == pytest.approx(np.array([0.0, 0.0]))
    assert session._reached_goal() is True


def test_route_geometry_ignores_later_pass_through_same_area() -> None:
    """The bounded search window keeps the index on the current route pass.

    Purpose: Regression for the nearest-waypoint index jumping to a later route
        segment when a route revisits the same area (loop / repeated junction)

    Given: An out-and-back route whose inbound leg passes nearer the ego than the
        outbound leg it is currently on
    When: _lane_geometry projects an ego early on the outbound leg
    Then: The reference index stays on the outbound leg instead of snapping to the
        geometrically closer inbound waypoint far ahead

    Test type: unit
    """
    outbound = [[float(x), 0.5] for x in range(0, 42, 2)]
    inbound = [[float(x), 0.0] for x in range(40, -2, -2)]
    route_xy = np.array(outbound + inbound)
    session = _route_session(route_xy, np.zeros(len(route_xy)))

    session._lane_geometry(SimpleNamespace(x=2.0, y=0.1), 0.0)

    assert session._route_index == 1  # outbound (2, 0.5), not inbound (2, 0.0)
