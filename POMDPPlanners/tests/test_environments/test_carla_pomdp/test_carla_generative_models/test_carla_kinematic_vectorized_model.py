# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the scalar kinematic CARLA model.

These tests pin :class:`CarlaKinematicVectorizedModel` to the scalar
:class:`KinematicCarlaModelPOMDP` so the two implementations cannot drift. The
deterministic kernels (transition, reward, terminal, observation log-density)
are compared exactly in float64 over random batches spanning several supported
configurations (flat vs. obstacle-aware desired speed, custom perception); the
stochastic observation sampler is compared by empirical moments against the
scalar sampler. Geometric edge cases (occlusion, out-of-range agents) get
dedicated crafted-state checks.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_vectorized_model import (
    CarlaKinematicVectorizedModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models.gnss_models import (
    GnssObservationModel,
)

_CUSTOM_PRESETS = [
    (0.6, 0.0, 0.0),
    (0.2, -0.4, 0.0),
    (0.4, 0.3, 0.0),
    (0.0, 0.1, 0.9),
    (0.8, -0.2, 0.0),
]

# Each case is scalar-model kwargs; all are supported configurations.
_SUPPORTED_CASES = [
    {"discount_factor": 0.95, "dt": 0.05},
    {
        "discount_factor": 0.9,
        "dt": 0.05,
        "desired_speed": 5.0,
        "collision_gap": 5.0,
        "safe_distance": 15.0,
        "stop_gap": 7.0,
    },
    {
        "discount_factor": 0.95,
        "dt": 0.1,
        "action_presets": _CUSTOM_PRESETS,
        "wheelbase": 3.2,
        "max_steer_angle": 0.5,
        "accel": 2.5,
        "brake_decel": 6.0,
        "drag": 0.1,
        "pose_std": 0.8,
        "gnss_std": 1e-4,
        "detect_prob": 0.85,
        "perception_range": 30.0,
        "occlusion_radius": 2.0,
        "collision_halfwidth": 1.5,
    },
]
_CASE_IDS = ["flat-speed", "obstacle-aware", "custom-perception"]


@dataclass
class _Case:
    env: KinematicCarlaModelPOMDP
    model: CarlaKinematicVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = KinematicCarlaModelPOMDP(**request.param)
    model = CarlaKinematicVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _random_states(rng: np.random.Generator, env: KinematicCarlaModelPOMDP, n: int) -> np.ndarray:
    """Build random states with a mix of present/absent, near/far agent slots."""
    width = 7 + env.max_tracked_agents * 5
    states = np.zeros((n, width))
    states[:, 0:2] = rng.uniform(-20.0, 20.0, size=(n, 2))
    states[:, 2] = rng.uniform(-180.0, 180.0, size=n)
    states[:, 3:5] = rng.uniform(-5.0, 10.0, size=(n, 2))
    states[:, 5] = rng.uniform(-3.0, 3.0, size=n)
    states[:, 6] = rng.uniform(-0.5, 0.5, size=n)
    rows = states[:, 7:].reshape(n, env.max_tracked_agents, 5)
    rows[..., 0] = (rng.uniform(size=(n, env.max_tracked_agents)) < 0.6).astype(float)
    rows[..., 1] = rng.uniform(-60.0, 60.0, size=(n, env.max_tracked_agents))
    rows[..., 2] = rng.uniform(-8.0, 8.0, size=(n, env.max_tracked_agents))
    rows[..., 3] = rng.uniform(-np.pi, np.pi, size=(n, env.max_tracked_agents))
    rows[..., 4] = rng.uniform(0.0, 10.0, size=(n, env.max_tracked_agents))
    return states


def _random_actions(rng: np.random.Generator, env: KinematicCarlaModelPOMDP, n: int) -> np.ndarray:
    return rng.integers(0, len(env.action_presets), size=n)


def _flat_observations(
    rng: np.random.Generator, env: KinematicCarlaModelPOMDP, next_states: np.ndarray
) -> np.ndarray:
    """Random flat 27-vectors exercising present/absent observed agent slots."""
    n = next_states.shape[0]
    obs = np.zeros((n, 2 + env.max_tracked_agents * 5))
    obs[:, 0:2] = next_states[:, 0:2] + rng.normal(0.0, 1e-4, size=(n, 2))
    obs_rows = obs[:, 2:].reshape(n, env.max_tracked_agents, 5)
    true_rows = next_states[:, 7:].reshape(n, env.max_tracked_agents, 5)
    obs_rows[..., 0] = (rng.uniform(size=(n, env.max_tracked_agents)) < 0.6).astype(float)
    obs_rows[..., 1:] = true_rows[..., 1:] + rng.normal(
        0.0, 1.0, size=(n, env.max_tracked_agents, 4)
    )
    return obs


def _obs_dict(flat: np.ndarray) -> Dict[str, np.ndarray]:
    return {"gnss": flat[:2].copy(), "agents": flat[2:].copy()}


def test_model_conforms_to_protocol_across_configs(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance and reported dimensions

    Given: A model built from each supported scalar-model configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the dimensions match the schema

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == len(case.env.action_presets)
    assert case.model.state_dim == 7 + case.env.max_tracked_agents * 5
    assert case.model.observation_dim == 2 + case.env.max_tracked_agents * 5


def test_transition_matches_native_exactly(case: _Case) -> None:
    """Sampled next states equal the scalar kinematic propagation per row.

    Purpose: Validates the batched kinematic-bicycle transition kernel

    Given: Random states with per-row random action indices
    When: The vectorized transition is compared to sample_next_state per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(1)
    states = _random_states(rng, case.env, 256)
    actions = _random_actions(rng, case.env, 256)
    expected = np.stack(
        [case.env.sample_next_state(states[i], int(actions[i])) for i in range(states.shape[0])]
    )
    actual = case.model.sample_next_states(
        _tensor(states), torch.as_tensor(actions, dtype=torch.int64)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Rewards equal the scalar driving-quality reward per row.

    Purpose: Validates the reward kernel including obstacle-aware desired speed

    Given: Random states/actions and their vectorized next states
    When: The vectorized reward is compared to env.reward per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(2)
    states = _random_states(rng, case.env, 256)
    actions = _random_actions(rng, case.env, 256)
    action_t = torch.as_tensor(actions, dtype=torch.int64)
    next_states = case.model.sample_next_states(_tensor(states), action_t)
    expected = np.array(
        [
            case.env.reward(states[i], int(actions[i]), next_states[i].numpy())
            for i in range(states.shape[0])
        ]
    )
    actual = case.model.rewards(_tensor(states), action_t, next_states).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_terminal_matches_native_exactly(case: _Case) -> None:
    """Terminal flags equal the scalar predicted-collision check per row.

    Purpose: Validates the batched terminal mask

    Given: Random states with present/absent near/far agent slots
    When: The vectorized terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    rng = np.random.default_rng(3)
    states = _random_states(rng, case.env, 512)
    expected = np.array([case.env.is_terminal(states[i]) for i in range(states.shape[0])])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-densities equal the scalar per-channel density per row.

    Purpose: Validates the GNSS + factored-agent observation likelihood kernel

    Given: Random next states and random observations spanning all slot branches
    When: The vectorized log-density is compared to env.observation_log_probability
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(4)
    next_states = _random_states(rng, case.env, 256)
    observations = _flat_observations(rng, case.env, next_states)
    expected = np.array(
        [
            case.env.observation_log_probability(next_states[i], 0, _obs_dict(observations[i]))[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), torch.zeros(256, dtype=torch.int64), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def _occlusion_state() -> np.ndarray:
    """Two present agents where the blocker occludes the farther target."""
    state = np.zeros(32)
    state[7:12] = [1.0, 20.0, 0.0, 0.0, 0.0]  # slot 0: target ahead
    state[12:17] = [1.0, 10.0, 0.5, 0.0, 0.0]  # slot 1: blocker on the sight line
    return state


def test_occlusion_log_prob_matches_native() -> None:
    """An occluded agent is scored as invisible, matching the scalar density.

    Purpose: Validates the O(K^2) sight-line occlusion gate in the density

    Given: A crafted state whose slot-1 blocker occludes the slot-0 target,
        with an observation reporting the (unobservable) occluded target present
    When: The vectorized log-density is compared to env.observation_log_probability
    Then: The two agree to 1e-9 (the occluded-but-observed slot floors to _LOG_EPS)

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    model = CarlaKinematicVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    state = _occlusion_state()
    obs = np.zeros(27)
    obs[0:2] = state[0:2]
    obs[2:7] = [1.0, 20.0, 0.0, 0.0, 0.0]  # report the occluded target as present
    obs[7:12] = [1.0, 10.0, 0.5, 0.0, 0.0]  # blocker correctly observed present
    expected = env.observation_log_probability(state, 0, {"gnss": obs[:2], "agents": obs[2:]})[0]
    actual = float(
        model.observation_log_probs(
            _tensor(state[None]), torch.zeros(1, dtype=torch.int64), _tensor(obs[None])
        )[0]
    )
    assert abs(expected - actual) < 1e-9
    # Correctly reporting the occluded target as absent (a forced miss) must score
    # far higher, confirming the occlusion gate floored the present report.
    obs_absent = obs.copy()
    obs_absent[2:7] = 0.0
    log_prob_absent = float(
        model.observation_log_probs(
            _tensor(state[None]), torch.zeros(1, dtype=torch.int64), _tensor(obs_absent[None])
        )[0]
    )
    assert actual < log_prob_absent - 40.0


def test_none_perception_range_parity() -> None:
    """A far agent stays visible when the range gate is disabled (range=None).

    Purpose: Validates that perception_range=None disables the range gate

    Given: A state with a present agent well beyond the default 50 m range
    When: The vectorized density is compared to the scalar density
    Then: The two agree to 1e-9 and the far agent contributes a Gaussian term

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05, perception_range=None)
    model = CarlaKinematicVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    state = np.zeros(32)
    state[7:12] = [1.0, 200.0, 0.0, 0.0, 3.0]  # present agent far beyond 50 m
    obs = np.zeros(27)
    obs[2:7] = [1.0, 200.5, 0.1, 0.0, 3.0]  # detected with small pose noise
    expected = env.observation_log_probability(state, 0, {"gnss": obs[:2], "agents": obs[2:]})[0]
    actual = float(
        model.observation_log_probs(
            _tensor(state[None]), torch.zeros(1, dtype=torch.int64), _tensor(obs[None])
        )[0]
    )
    assert abs(expected - actual) < 1e-9


def _sample_agent_slot0(
    sampler: Callable[[np.ndarray], Dict[str, np.ndarray]],
    next_state: np.ndarray,
    n: int,
    extract: Callable[[Dict[str, np.ndarray]], np.ndarray],
) -> np.ndarray:
    return np.stack([extract(sampler(next_state)) for _ in range(n)])


def test_gnss_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled GNSS noise matches the scalar Gaussian in mean and covariance.

    Purpose: Validates the GNSS observation sampler's noise statistics

    Given: A fixed next state sampled many times by both implementations
    When: Empirical GNSS mean and covariance (normalised by gnss_std) are compared
    Then: The normalised mean is near zero and the covariance near the identity

    Test type: unit
    """
    torch.manual_seed(5)
    next_state = np.zeros(32)
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (60000, 1))), torch.zeros(60000, dtype=torch.int64)
    ).numpy()
    gnss_model = case.env.observation_models["gnss"] if case.env.observation_models else None
    assert isinstance(gnss_model, GnssObservationModel)
    normalised = (twin[:, 0:2] - next_state[0:2]) / gnss_model.gnss_std
    assert np.max(np.abs(normalised.mean(axis=0))) < 0.03
    assert np.max(np.abs(np.cov(normalised.T) - np.eye(2))) < 0.05


def test_agent_pose_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled detected-agent pose noise matches the scalar sampler's moments.

    Purpose: Validates the per-slot additive pose-noise sampler for a visible agent

    Given: A next state with one visible present agent, sampled many times by both
    When: Empirical per-column mean and covariance of the slot are compared
    Then: Means agree within 0.02 and covariances within a Monte Carlo tolerance

    Test type: unit
    """
    np.random.seed(6)
    torch.manual_seed(6)
    next_state = np.zeros(32)
    next_state[7:12] = [1.0, 10.0, 0.0, 0.3, 4.0]  # in range, unoccluded, visible
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (40000, 1))), torch.zeros(40000, dtype=torch.int64)
    ).numpy()[:, 3:7]
    native = _sample_agent_slot0(
        lambda s: case.env.sample_observation(s, 0),
        next_state,
        40000,
        lambda obs: np.asarray(obs["agents"], dtype=float)[1:5],
    )
    assert np.max(np.abs(twin.mean(axis=0) - native.mean(axis=0))) < 0.02
    assert np.max(np.abs(np.cov(twin.T) - np.cov(native.T))) < 0.02


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """All generative methods return the documented shapes and dtypes.

    Purpose: Validates the tensor contract of every public method

    Given: A small batch of states, actions, next states, and observations
    When: Each generative / key method is invoked
    Then: Shapes match [N, .] / [N] and integer keys are int64

    Test type: unit
    """
    rng = np.random.default_rng(8)
    states = _tensor(_random_states(rng, case.env, 16))
    actions = torch.zeros(16, dtype=torch.int64)
    next_states = case.model.sample_next_states(states, actions)
    observations = case.model.sample_observations(next_states, actions)
    assert next_states.shape == (16, case.model.state_dim)
    assert observations.shape == (16, case.model.observation_dim)
    assert case.model.rewards(states, actions, next_states).shape == (16,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_keys_are_deterministic_and_discriminating(case: _Case) -> None:
    """Action and observation keys are stable and separate distinct inputs.

    Purpose: Validates the integer tree-key mappings

    Given: Repeated actions and observations, some identical and some distinct
    When: The key methods are called twice
    Then: Identical inputs map to identical keys and distinct inputs differ

    Test type: unit
    """
    actions = torch.tensor([0, 1, 2, 3])
    assert torch.equal(case.model.action_keys(actions), actions.to(torch.int64))
    base = torch.zeros(3, case.model.observation_dim, dtype=torch.float64)
    base[1, 0] = 5.0
    base[2, 4] = 9.0
    keys_first = case.model.observation_keys(base)
    keys_second = case.model.observation_keys(base)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] != keys_first[1]
    assert keys_first[0] != keys_first[2]


def test_nonpositive_observation_resolution_raises() -> None:
    """A non-positive observation resolution is rejected at construction.

    Purpose: Validates the guard on the observation-resolution knob

    Given: A valid scalar model and a non-positive observation_resolution
    When: A vectorized model is constructed
    Then: ValueError is raised

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    with pytest.raises(ValueError):
        CarlaKinematicVectorizedModel(env, observation_resolution=0.0)


def test_unsupported_observation_models_raise() -> None:
    """Constructing without the factored/gaussian perception pair is rejected.

    Purpose: Validates the scope guard on the required observation models

    Given: A scalar model whose observation-model map has been cleared
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    env.observation_models = {}
    with pytest.raises(NotImplementedError):
        CarlaKinematicVectorizedModel(env)


def test_transition_batch_helper_matches_vectorized() -> None:
    """The scalar batch transition agrees with the vectorized kernel.

    Purpose: Cross-checks sample_next_state_batch against the vectorized model

    Given: Random states and a single shared action index
    When: The scalar batch transition and the vectorized transition are compared
    Then: The maximum absolute difference is below 1e-9

    Test type: integration
    """
    rng = np.random.default_rng(9)
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    model = CarlaKinematicVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    states = _random_states(rng, env, 128)
    expected = env.sample_next_state_batch(states, 2)
    actual = model.sample_next_states(
        _tensor(states), torch.full((128,), 2, dtype=torch.int64)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9
