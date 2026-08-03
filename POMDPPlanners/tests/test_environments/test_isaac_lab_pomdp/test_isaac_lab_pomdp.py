# SPDX-License-Identifier: MIT

"""Unit tests for the IsaacLabPOMDP forward-only world wrapper.

These tests drive IsaacLabPOMDP against a controllable ``FakeIsaacEnv`` (torch
tensors, a scripted scene, deterministic reset/step) injected via the module-level
``_build_isaac_env`` seam, so they run without launching Isaac Sim or depending on
any registered IsaacLab task. A guarded ``smoke`` test exercises a real IsaacLab
task only when ``RUN_ISAAC_SMOKE`` is set (Isaac installed locally, not in CI).
"""

# pylint: disable=protected-access  # Tests inspect the live-simulator internals

import os
import pickle
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.isaac_lab_pomdp import IsaacLabPOMDP
from POMDPPlanners.environments.isaac_lab_pomdp import isaac_lab_pomdp as isaac_module
from POMDPPlanners.simulations.episodes import EpisodeRunner


class _FakeEntity:
    """A scene entry exposing a ``.data`` buffer namespace."""

    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeArticulationData:
    """Ground-truth articulation buffers as torch tensors of shape ``(1, dim)``."""

    def __init__(self, t: int) -> None:
        self.root_pos_w = torch.tensor([[float(t), 0.0, 0.0]])
        self.joint_pos = torch.tensor([[float(t) + 0.5, float(t) - 0.5]])


class _FakeSensorData:
    """Sensor buffer (LiDAR-like) as a torch tensor of shape ``(1, num_beams)``."""

    def __init__(self, t: int) -> None:
        self.ray_hits_w = torch.tensor([[float(t)] * 4])


class _FakeContactData:
    """Contact-sensor buffers as torch tensors, mirroring IsaacLab's layout.

    ``net_forces_w_history`` is shaped ``(1, history, bodies, 3)`` and is
    **newest-first**, matching IsaacLab's documented convention: "In the history
    dimension, the first index is the most recent and the last index is the
    oldest." Reading it from the wrong end yields stale forces that still look
    plausible, so the ordering is reproduced faithfully here.
    """

    def __init__(self, t: int, with_history: bool = True, stale: bool = False) -> None:
        self.net_forces_w = torch.tensor([[[0.0, 0.0, 3.0 * t]]])
        if not with_history:
            return
        # This control step's substeps, newest first. The 4t entry is the spike
        # the non-history buffer would miss.
        current = [[0.0, 0.0, 3.0 * t], [0.0, 4.0 * t, 0.0]]
        if stale:
            # Forces from *earlier* control steps live at the tail of the buffer.
            # A correct reader must not integrate them into this step.
            current = [*current, [0.0, 0.0, 500.0], [0.0, 900.0, 0.0]]
        self.net_forces_w_history = torch.tensor([current])


class _FakeTerminationManager:
    """Termination manager exposing named terms, as IsaacLab's manager does."""

    def __init__(self) -> None:
        self._terms: Dict[str, torch.Tensor] = {}

    def set_term(self, name: str, value: bool) -> None:
        self._terms[name] = torch.tensor([value])

    def get_term(self, name: str) -> torch.Tensor:
        if name not in self._terms:
            raise KeyError(name)
        return self._terms[name]


class FakeIsaacEnv:
    """Scripted IsaacLab-like env: torch tensors, a scene, deterministic steps.

    The scene exposes a ``robot`` articulation, a ``lidar`` sensor and a
    ``contact_forces`` sensor whose buffers advance with an internal step
    counter. The episode terminates once ``terminate_after`` steps have been
    taken.
    """

    def __init__(
        self,
        terminate_after: int = 3,
        contact_history: bool = True,
        stale_history: bool = False,
    ) -> None:
        self._t = 0
        self._terminate_after = terminate_after
        self._contact_history = contact_history
        self._stale_history = stale_history
        self.reset_calls = 0
        self.step_calls = 0
        self.render_calls = 0
        self.step_dt = 0.5
        self.action_space = SimpleNamespace(shape=(1, 2))
        self.physics_dt = 0.25
        self.termination_manager = _FakeTerminationManager()
        # A task that declares a success term, initially unmet. Tasks without one
        # are modelled by asking for a term name this manager does not have.
        self.termination_manager.set_term("success", False)
        self.step_info: Dict[str, Any] = {}
        self.truncate = False
        self.scene: Dict[str, _FakeEntity] = {}
        self._refresh()

    def render(self) -> np.ndarray:
        """Return a constant RGB frame tagged with the render-call count."""
        self.render_calls += 1
        return np.full((2, 3, 3), self.render_calls, dtype=np.uint8)

    @property
    def unwrapped(self) -> "FakeIsaacEnv":
        return self

    def _refresh(self) -> None:
        self.scene = {
            "robot": _FakeEntity(_FakeArticulationData(self._t)),
            "lidar": _FakeEntity(_FakeSensorData(self._t)),
            "contact_forces": _FakeEntity(
                _FakeContactData(
                    self._t, with_history=self._contact_history, stale=self._stale_history
                )
            ),
        }

    def reset(self, *, seed: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        del seed
        self.reset_calls += 1
        self._t = 0
        self._refresh()
        return {"policy": torch.zeros((1, 2))}, {}

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        del action
        self.step_calls += 1
        self._t += 1
        self._refresh()
        reward = torch.tensor([1.0])
        terminated = torch.tensor([self._t >= self._terminate_after])
        truncated = torch.tensor([self.truncate])
        return {"policy": torch.zeros((1, 2))}, reward, terminated, truncated, self.step_info


@pytest.fixture(name="fake_env")
def _fake_env(monkeypatch: pytest.MonkeyPatch) -> FakeIsaacEnv:
    """Patch the build seam to return a single shared FakeIsaacEnv instance."""
    env = FakeIsaacEnv()
    monkeypatch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
    return env


@pytest.fixture(name="world")
def _world(fake_env: FakeIsaacEnv) -> IsaacLabPOMDP:
    """An IsaacLabPOMDP wrapping the fake env, reset to its initial live state."""
    del fake_env
    world = IsaacLabPOMDP(task_id="Fake-Isaac-v0", discount_factor=0.99, device="cpu")
    world.initial_state_dist().sample()  # establish the live state
    return world


def test_space_info_uses_declared_space_types(world: IsaacLabPOMDP) -> None:
    """Space types come from the declared constructor arguments.

    Purpose: Validates SpaceInfo reflects the declared action/observation types.

    Given: An IsaacLabPOMDP built with default (continuous) space types
    When: Its space_info is inspected
    Then: Both action and observation spaces report CONTINUOUS

    Test type: unit
    """
    assert world.space_info.action_space == SpaceType.CONTINUOUS
    assert world.space_info.observation_space == SpaceType.CONTINUOUS


def test_step_called_once_across_reward_next_state_observation(
    world: IsaacLabPOMDP, fake_env: FakeIsaacEnv
) -> None:
    """The three per-step queries trigger exactly one underlying sim step.

    Purpose: Validates the step-once cache serving reward/next-state/observation.

    Given: A freshly reset IsaacLabPOMDP world at its live state
    When: reward, sample_next_state and sample_observation are called in order
    Then: env.step is invoked exactly once and the three agree on the transition

    Test type: unit
    """
    state = world._live_state
    action = np.zeros(2, dtype=np.float32)
    before = fake_env.step_calls

    reward = world.reward(state, action)
    next_state = world.sample_next_state(state, action)
    observation = world.sample_observation(next_state, action)

    assert fake_env.step_calls - before == 1
    assert reward == 1.0
    # State = root_pos_w (3) + joint_pos (2) at t=1; observation = 4 LiDAR beams.
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.5, 0.5]))
    assert np.array_equal(observation, np.array([1.0, 1.0, 1.0, 1.0]))


def test_state_and_observation_have_different_shapes(
    world: IsaacLabPOMDP, fake_env: FakeIsaacEnv
) -> None:
    """State (ground truth) and observation (sensor) are genuinely distinct.

    Purpose: Validates the state/observation split — obs is not equal to state.

    Given: A reset IsaacLabPOMDP world reading state from the articulation and
        observation from the LiDAR sensor
    When: One step is taken and both quantities are read
    Then: The state and observation differ in shape and content

    Test type: unit
    """
    del fake_env
    state = world._live_state
    action = np.zeros(2, dtype=np.float32)

    next_state = world.sample_next_state(state, action)
    observation = world.sample_observation(next_state, action)

    assert next_state.shape != observation.shape
    assert not np.array_equal(next_state, observation)


def test_call_order_independence_next_state_before_reward(
    world: IsaacLabPOMDP, fake_env: FakeIsaacEnv
) -> None:
    """The step is served identically regardless of which query comes first.

    Purpose: Validates the cache trigger works from sample_next_state too.

    Given: A freshly reset IsaacLabPOMDP world at its live state
    When: sample_next_state is called before reward for the same (state, action)
    Then: still one sim step, and reward reflects that same transition

    Test type: unit
    """
    state = world._live_state
    action = np.zeros(2, dtype=np.float32)
    before = fake_env.step_calls

    next_state = world.sample_next_state(state, action)
    reward = world.reward(state, action)

    assert fake_env.step_calls - before == 1
    assert np.array_equal(next_state, np.array([1.0, 0.0, 0.0, 1.5, 0.5]))
    assert reward == 1.0


def test_is_terminal_reflects_last_terminated(world: IsaacLabPOMDP) -> None:
    """is_terminal returns the terminated flag of the current live state.

    Purpose: Validates terminal tracking across forward steps.

    Given: A world whose fake env terminates after three steps
    When: The world is advanced step by step from the live state
    Then: is_terminal is False until the third step, then True

    Test type: unit
    """
    state = world._live_state
    action = np.zeros(2, dtype=np.float32)
    assert world.is_terminal(state) is False

    for expected_terminal in (False, False, True):
        next_state, _, _ = world.sample_next_step(state, action)
        assert world.is_terminal(next_state) is expected_terminal
        state = next_state


def test_initial_state_dist_sample_triggers_reset(fake_env: FakeIsaacEnv) -> None:
    """Sampling the initial-state distribution resets the wrapped env.

    Purpose: Validates initial_state_dist maps to env.reset and reads state.

    Given: An IsaacLabPOMDP wrapping a fake env
    When: initial_state_dist().sample() is called
    Then: env.reset runs and the returned state is read from the articulation

    Test type: unit
    """
    world = IsaacLabPOMDP(task_id="Fake-Isaac-v0", discount_factor=0.99, device="cpu")
    before = fake_env.reset_calls

    samples = world.initial_state_dist().sample()

    assert fake_env.reset_calls - before == 1
    assert len(samples) == 1
    assert np.array_equal(samples[0], np.array([0.0, 0.0, 0.0, 0.5, -0.5]))


def test_transition_log_probability_raises(world: IsaacLabPOMDP) -> None:
    """transition_log_probability is unsupported on a forward-only world.

    Purpose: Validates the density method raises instead of faking a value.

    Given: An IsaacLabPOMDP world
    When: transition_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    with pytest.raises(NotImplementedError):
        world.transition_log_probability(world._live_state, np.zeros(2), [world._live_state])


def test_observation_log_probability_raises(world: IsaacLabPOMDP) -> None:
    """observation_log_probability is unsupported on a forward-only world.

    Purpose: Validates the density method raises instead of faking a value.

    Given: An IsaacLabPOMDP world
    When: observation_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    with pytest.raises(NotImplementedError):
        world.observation_log_probability(world._live_state, np.zeros(2), [world._live_state])


def test_forward_only_guard_raises_on_mismatched_state(world: IsaacLabPOMDP) -> None:
    """Stepping from a non-live state is rejected loudly.

    Purpose: Validates the forward-only guard on arbitrary-state resampling.

    Given: An IsaacLabPOMDP world at a known live state
    When: sample_next_state is called with a different, arbitrary state
    Then: RuntimeError is raised naming the forward-only constraint

    Test type: unit
    """
    arbitrary_state = np.full(5, 99.0)
    with pytest.raises(RuntimeError, match="forward-only"):
        world.sample_next_state(arbitrary_state, np.zeros(2))


def test_sample_next_state_rejects_multiple_samples(world: IsaacLabPOMDP) -> None:
    """A forward-only world cannot draw more than one next state.

    Purpose: Validates n_samples>1 is rejected.

    Given: An IsaacLabPOMDP world at its live state
    When: sample_next_state is called with n_samples=2
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="n_samples=1"):
        world.sample_next_state(world._live_state, np.zeros(2), n_samples=2)


def test_num_envs_must_be_one() -> None:
    """A world adapter rejects a batched (num_envs > 1) configuration.

    Purpose: Validates the single-trajectory constraint at construction.

    Given: An attempt to build IsaacLabPOMDP with num_envs=4
    When: The constructor runs
    Then: ValueError is raised naming the num_envs=1 requirement

    Test type: unit
    """
    with pytest.raises(ValueError, match="num_envs=1"):
        IsaacLabPOMDP(task_id="Fake-Isaac-v0", discount_factor=0.99, num_envs=4, device="cpu")


def test_getstate_setstate_round_trip_drops_and_rebuilds_handle(
    world: IsaacLabPOMDP, fake_env: FakeIsaacEnv
) -> None:
    """Pickling drops the live handle and it is rebuilt lazily on use.

    Purpose: Validates the non-picklable handle is not serialized.

    Given: An IsaacLabPOMDP world with a live fake env handle
    When: It is pickled and unpickled
    Then: The restored object carries no live handle until it is next needed

    Test type: unit
    """
    del fake_env
    restored = pickle.loads(pickle.dumps(world))

    assert restored._env is None
    assert restored._live_state is None
    assert restored._pending is None
    # A fresh reset rebuilds the handle lazily via the patched build seam.
    restored.initial_state_dist().sample()
    assert restored._env is not None


def test_config_id_stable_across_pickling(world: IsaacLabPOMDP) -> None:
    """config_id depends only on public config, so it survives pickling.

    Purpose: Validates deterministic config identity for result caching.

    Given: An IsaacLabPOMDP world
    When: It is pickled and unpickled
    Then: config_id is unchanged (private live state is excluded)

    Test type: unit
    """
    restored = pickle.loads(pickle.dumps(world))
    assert restored.config_id == world.config_id


# ── End-to-end two-environment episode ──────────────────────────────────


class _StubBelief(Belief):
    """A trivial belief whose update is a no-op (decouples the model side)."""

    def update(
        self, action: Any, observation: Any, pomdp: Environment, state: Optional[Any] = None
    ) -> "Belief":
        del action, observation, pomdp, state
        return self

    def sample(self) -> Any:
        return np.zeros(3)


class _StubModelEnv(Environment):
    """A minimal generative-model stub; its methods are never exercised here."""

    def __init__(self) -> None:
        super().__init__(
            discount_factor=0.99,
            name="StubModel",
            space_info=SpaceInfo(SpaceType.CONTINUOUS, SpaceType.CONTINUOUS),
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del state, action, next_state
        return 0.0

    def is_terminal(self, state: Any) -> bool:
        del state
        return False

    def initial_state_dist(self) -> Distribution:
        class _Dist(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [np.zeros(3) for _ in range(n_samples)]

        return _Dist()

    def initial_observation_dist(self) -> Distribution:
        class _Dist(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [np.zeros(4) for _ in range(n_samples)]

        return _Dist()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)

    def hash_action(self, action: Any) -> Any:
        return np.asarray(action).tobytes()

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        del state, action, n_samples
        return np.zeros(3)

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del next_state, action, n_samples
        return np.zeros(4)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        return np.zeros(1)

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del next_state, action, observations
        return np.zeros(1)


class _StubContinuousPolicy(Policy):
    """A trivial policy returning a fixed continuous action; model != world."""

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        del belief
        return ([np.zeros(2, dtype=np.float32)], PolicyRunData(info_variables=[]))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return []


class _SpyIsaacWorld(IsaacLabPOMDP):
    """An IsaacLabPOMDP recording how many times each routed method is called."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls: Dict[str, int] = {
            "reward": 0,
            "sample_next_state": 0,
            "sample_observation": 0,
            "initial_state_dist": 0,
        }

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        self._calls["reward"] += 1
        return super().reward(state, action, next_state)

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        self._calls["sample_next_state"] += 1
        return super().sample_next_state(state, action, n_samples)

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        self._calls["sample_observation"] += 1
        return super().sample_observation(next_state, action, n_samples)

    def initial_state_dist(self) -> Distribution:
        self._calls["initial_state_dist"] += 1
        return super().initial_state_dist()


def test_episode_runner_routes_trajectory_through_isaac_world(fake_env: FakeIsaacEnv) -> None:
    """EpisodeRunner draws the executed trajectory from the IsaacLab world.

    Purpose: Validates the two-env split — reward/transition/observation and the
        initial state come from the IsaacLab world while the model stays separate.

    Given: A spied IsaacLabPOMDP world, a distinct stub model/policy/belief
    When: A two-step episode is run through EpisodeRunner
    Then: The world's reward/sample_next_state/sample_observation and
        initial_state_dist are exercised and a non-empty history is produced

    Test type: integration
    """
    del fake_env
    world = _SpyIsaacWorld(task_id="Fake-Isaac-v0", discount_factor=0.99, device="cpu")
    policy = _StubContinuousPolicy(
        environment=_StubModelEnv(), discount_factor=0.99, name="StubPolicy"
    )
    belief = _StubBelief()

    runner = EpisodeRunner(
        environment=world, policy=policy, initial_belief=belief, num_steps=2, logger=None
    )
    history = runner.run()

    assert world._calls["initial_state_dist"] >= 1
    assert world._calls["reward"] > 0
    assert world._calls["sample_next_state"] > 0
    assert world._calls["sample_observation"] > 0
    assert len(history.history) > 0


def test_render_requires_rgb_array_mode(world: IsaacLabPOMDP) -> None:
    """render raises when the world was not built with render_mode='rgb_array'.

    Purpose: Validates the render-mode guard on frame capture.

    Given: An IsaacLabPOMDP built with the default render_mode (None)
    When: render is called
    Then: RuntimeError is raised naming render_mode='rgb_array'

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="render_mode='rgb_array'"):
        world.render()


def test_render_returns_uint8_rgb_frame(fake_env: FakeIsaacEnv) -> None:
    """render returns the simulator's RGB frame as a uint8 array.

    Purpose: Validates render forwards to env.render and normalizes the dtype.

    Given: An IsaacLabPOMDP built with render_mode='rgb_array' over a fake env
    When: render is called
    Then: env.render is invoked and an (H, W, 3) uint8 frame is returned

    Test type: unit
    """
    world = IsaacLabPOMDP(
        task_id="Fake-Isaac-v0",
        discount_factor=0.99,
        device="cpu",
        render_mode="rgb_array",
    )
    before = fake_env.render_calls

    frame = world.render()

    assert fake_env.render_calls - before == 1
    assert frame.dtype == np.uint8
    assert frame.shape == (2, 3, 3)


@pytest.mark.smoke
@pytest.mark.skipif(
    not os.getenv("RUN_ISAAC_SMOKE"),
    reason="Requires a local Isaac Sim install; set RUN_ISAAC_SMOKE=1 to run.",
)
def test_real_isaac_task_state_observation_split() -> None:
    """A real IsaacLab task yields distinct ground-truth state and sensor obs.

    Purpose: Validates the wrapper drives a real registered task end to end.

    Given: A registered IsaacLab task with a configured sensor
    When: The world is reset and stepped once through sample_next_step
    Then: A finite reward is produced and state and observation shapes differ

    Test type: integration
    """
    task_id = os.environ.get("ISAAC_SMOKE_TASK", "Isaac-Velocity-Flat-Anymal-C-v0")
    world = IsaacLabPOMDP(task_id=task_id, discount_factor=0.99)

    state = world.initial_state_dist().sample()[0]
    action = np.zeros(world._get_env().action_space.shape[-1], dtype=np.float32)
    next_state, observation, reward = world.sample_next_step(state, action)

    assert np.isfinite(reward)
    assert np.asarray(next_state).shape != np.asarray(observation).shape
