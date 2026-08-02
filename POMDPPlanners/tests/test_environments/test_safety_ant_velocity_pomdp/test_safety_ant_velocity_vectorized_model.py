# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native Safety Ant Velocity env.

These tests pin :class:`SafetyAntVelocityVectorizedModel` to the environment's
native (C++) kernels so the two implementations cannot drift. The parity
checks run over a *sweep of supported configurations* (varying physics, noise,
and reward parameters) so a wiring bug that default parameters would hide is
caught. Observation log-densities and rewards are compared exactly in float64;
the stochastic transition and observation kernels are compared by empirical
moments over a large batch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
    SafeAntVelocityPOMDP,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_vectorized_model import (
    SafetyAntVelocityVectorizedModel,
)

# Each dict is a *supported* configuration: the default discrete force-level
# action set with physics / noise / reward parameters otherwise varying widely.
_SUPPORTED_CASES = [
    {},
    {"mass": 2.0, "damping": 0.3, "max_force": 2.0, "dt": 0.05},
    {"position_noise": 0.2, "velocity_noise": 0.1, "safe_velocity_threshold": 1.0},
    {"movement_reward_scale": 2.0, "safety_violation_penalty": -50.0},
]
_CASE_IDS = ["default", "heavy-damped", "noisy-tight-threshold", "scaled-reward"]


@dataclass
class _Case:
    env: SafeAntVelocityPOMDP
    model: SafetyAntVelocityVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = SafeAntVelocityPOMDP(discount_factor=0.99, **request.param)
    model = SafetyAntVelocityVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _action_indices(count: int, action: int = 0) -> torch.Tensor:
    return torch.full((count,), action, dtype=torch.int64)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the action count matches the env

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == len(case.env.get_actions())


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the diagonal-Gaussian observation likelihood kernel

    Given: (next_state, observation) pairs spanning the 4-D state per config
    When: The model's observation_log_probs is compared to the env per pair
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(3)
    next_states = rng.uniform(-2.0, 2.0, size=(128, 4))
    observations = next_states + rng.normal(scale=0.1, size=(128, 4))
    expected = np.array(
        [
            case.env.observation_log_probability_per_state(next_states[i], 0, observations[i])[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), _action_indices(128), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Rewards match the native reward kernel exactly across configs.

    Purpose: Validates the speed-based reward with the safety-violation penalty

    Given: Next states with velocities spanning safe and unsafe speeds
    When: The model reward is compared to env.reward_batch on those states
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(5)
    next_states = rng.uniform(-1.0, 1.0, size=(512, 4))
    next_states[:, 2:4] = rng.uniform(-3.0, 3.0, size=(512, 2))
    expected = case.env.reward_batch(next_states, 0)
    actual = case.model.rewards(
        _tensor(next_states), _action_indices(512), _tensor(next_states)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States with velocities below, just above, and far above the cutoff
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    cutoff = case.env.safe_velocity_threshold * 1.5
    speeds = np.array([0.0, cutoff * 0.5, cutoff - 0.01, cutoff + 0.01, cutoff * 2.0])
    states = np.zeros((speeds.shape[0], 4))
    states[:, 2] = speeds  # velocity_x carries the whole speed
    expected = np.array([case.env.is_terminal(s) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_transition_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled next-state moments match the native transition kernel per config.

    Purpose: Validates transition mean/covariance for each config's physics

    Given: A fixed moving state and a nonzero-force action sampled many times
    When: Empirical mean and covariance are compared for both implementations
    Then: Mean gap < 0.02 and covariance gap < 0.01

    Test type: unit
    """
    _native.set_seed(11)
    torch.manual_seed(11)
    state = np.array([0.5, -0.5, 0.4, 0.2])
    native = case.env.sample_next_state_batch(np.tile(state, (40000, 1)), 3)
    twin = case.model.sample_next_states(
        _tensor(np.tile(state, (40000, 1))), _action_indices(40000, action=3)
    ).numpy()
    assert np.max(np.abs(native.mean(0) - twin.mean(0))) < 0.02
    assert np.max(np.abs(np.cov(native.T) - np.cov(twin.T))) < 0.01


def test_observation_sampling_matches_native_covariance(case: _Case) -> None:
    """Observation sampling reproduces the diagonal observation covariance.

    Purpose: Validates the identity-mean diagonal-Gaussian observation noise

    Given: A fixed next state sampled many times by the model
    When: The empirical covariance is compared to the env's diagonal covariance
    Then: The covariance gap is below 0.01

    Test type: unit
    """
    torch.manual_seed(2)
    next_state = np.array([1.0, 2.0, 0.3, -0.4])
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (40000, 1))), _action_indices(40000)
    ).numpy()
    expected_cov = np.diag(
        [
            case.env.position_noise**2,
            case.env.position_noise**2,
            case.env.velocity_noise**2,
            case.env.velocity_noise**2,
        ]
    )
    assert np.max(np.abs(np.cov(twin.T) - expected_cov)) < 0.01


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every generative method returns the documented shape and dtype.

    Purpose: Validates the batched output contract of each protocol method

    Given: A random batch of states, actions, and observations
    When: Each generative / key method is invoked on the batch
    Then: Shapes match [N, ds]/[N, do]/[N] and key/mask dtypes are int64/bool

    Test type: unit
    """
    rng = np.random.default_rng(9)
    states = _tensor(rng.uniform(-1.0, 1.0, size=(32, 4)))
    next_states = _tensor(rng.uniform(-1.0, 1.0, size=(32, 4)))
    observations = _tensor(rng.uniform(-1.0, 1.0, size=(32, 4)))
    actions = _action_indices(32, action=1)
    assert case.model.sample_next_states(states, actions).shape == (32, 4)
    assert case.model.sample_observations(next_states, actions).shape == (32, 4)
    assert case.model.rewards(states, actions, next_states).shape == (32,)
    assert case.model.observation_log_probs(next_states, actions, observations).shape == (32,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_action_keys_pass_through_indices() -> None:
    """Action keys are the integer action indices unchanged.

    Purpose: Validates the discrete-action passthrough key mapping

    Given: A batch of discrete force-level action indices
    When: action_keys is called on the batch
    Then: The returned int64 keys equal the input indices

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.99)
    model = SafetyAntVelocityVectorizedModel(env, device=torch.device("cpu"))
    actions = torch.tensor([0, 1, 2, 3, 2, 0], dtype=torch.int64)
    keys = model.action_keys(actions)
    assert keys.dtype == torch.int64
    assert torch.equal(keys, actions)


def test_observation_keys_are_deterministic_and_discriminating() -> None:
    """Observation keys are stable per input and separate distinct cells.

    Purpose: Validates the continuous-observation to integer-key hashing

    Given: Observations, some identical and some in different grid cells
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct cells differ

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.99)
    model = SafetyAntVelocityVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    observations = torch.tensor(
        [
            [0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05],
            [3.2, 7.8, -1.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    keys_first = model.observation_keys(observations)
    keys_second = model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]
    assert keys_first[2] != keys_first[3]


def test_unsupported_action_set_raises() -> None:
    """Constructing on a non-default action set is rejected.

    Purpose: Validates the scope guard on the force-index action assumption

    Given: An env whose discrete action set is not range(len(force_scales))
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.99)
    env.actions = [0, 1, 2]
    with pytest.raises(NotImplementedError):
        SafetyAntVelocityVectorizedModel(env)


def test_non_positive_resolution_and_mass_raise() -> None:
    """Non-positive observation resolution or mass is rejected.

    Purpose: Validates the numeric-parameter guards in the constructor

    Given: A valid env plus, separately, an env with zero mass
    When: A model is built with a non-positive resolution / from the zero-mass env
    Then: ValueError is raised in both cases

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.99)
    with pytest.raises(ValueError):
        SafetyAntVelocityVectorizedModel(env, observation_resolution=0.0)
    zero_mass_env = SafeAntVelocityPOMDP(discount_factor=0.99, mass=0.0)
    with pytest.raises(ValueError):
        SafetyAntVelocityVectorizedModel(zero_mass_env)
