# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native CartPole POMDP env.

These tests pin :class:`CartPoleVectorizedModel` to the environment's native
(C++) kernels so the two implementations cannot drift. The parity checks run
over a sweep of supported configurations (varying the transition and
observation covariances) so a wiring bug that default parameters would hide is
caught. Observation log-densities, rewards, and terminal flags are compared
exactly; the stochastic transition / observation kernels are compared by
empirical moments over a large batch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_vectorized_model import (
    CartPoleVectorizedModel,
)

# Each case is (noise_cov, state_transition_cov | None). All are supported
# configurations (default "euler" integrator); the covariances vary widely.
_SUPPORTED_CASES = [
    (np.diag([0.1, 0.1, 0.1, 0.1]), None),
    (np.diag([0.02, 0.2, 0.05, 0.3]), np.diag([1e-3, 5e-4, 2e-4, 8e-4])),
    (
        np.array(
            [
                [0.10, 0.01, 0.00, 0.00],
                [0.01, 0.08, 0.00, 0.00],
                [0.00, 0.00, 0.05, 0.01],
                [0.00, 0.00, 0.01, 0.04],
            ]
        ),
        None,
    ),
]
_CASE_IDS = ["default", "anisotropic", "correlated-obs"]


@dataclass
class _Case:
    env: CartPolePOMDP
    model: CartPoleVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    noise_cov, state_transition_cov = request.param
    env = CartPolePOMDP(
        discount_factor=0.99,
        noise_cov=noise_cov,
        state_transition_cov=state_transition_cov,
    )
    model = CartPoleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _random_nonterminal_states(env: CartPolePOMDP, count: int, seed: int) -> np.ndarray:
    """Sample states comfortably inside the alive region (deterministic reward)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-env.x_threshold + 0.2, env.x_threshold - 0.2, size=count)
    x_dot = rng.uniform(-1.0, 1.0, size=count)
    theta = rng.uniform(
        -env.theta_threshold_radians + 0.02, env.theta_threshold_radians - 0.02, size=count
    )
    theta_dot = rng.uniform(-1.0, 1.0, size=count)
    return np.stack([x, x_dot, theta, theta_dot], axis=1)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the action count is two

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == 2


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the observation likelihood kernel across configs

    Given: (next_state, observation) pairs drawn around random next states
    When: The model's observation_log_probs is compared to the env per pair
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    next_states = _random_nonterminal_states(case.env, 256, seed=3)
    rng = np.random.default_rng(11)
    observations = next_states + rng.normal(scale=0.2, size=(256, 4))
    expected = np.array(
        [
            case.env.observation_log_probability(next_states[i], 0, observations[i].reshape(1, 4))[
                0
            ]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), torch.zeros(256, dtype=torch.int64), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


@pytest.mark.parametrize("action", [0, 1])
def test_reward_matches_native(case: _Case, action: int) -> None:
    """Rewards match the native +1-while-alive kernel exactly across configs.

    Purpose: Validates the deterministic reward path over alive/terminal states

    Given: A mix of clearly-alive and clearly-terminal states per config
    When: The model reward is compared to env.reward_batch on those states
    Then: Every entry agrees exactly

    Test type: unit
    """
    alive = _random_nonterminal_states(case.env, 200, seed=5)
    terminal = np.array(
        [
            [case.env.x_threshold + 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, case.env.theta_threshold_radians + 0.1, 0.0],
        ]
    )
    states = np.vstack([alive, terminal])
    expected = case.env.reward_batch(states, action, next_states=states)
    actual = case.model.rewards(
        _tensor(states), torch.full((states.shape[0],), action, dtype=torch.int64), _tensor(states)
    ).numpy()
    assert np.array_equal(expected, actual)


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States on both sides of the x and theta thresholds
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    x_thr = case.env.x_threshold
    theta_thr = case.env.theta_threshold_radians
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [x_thr + 0.1, 0.0, 0.0, 0.0],
            [-x_thr - 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, theta_thr + 0.01, 0.0],
            [0.0, 0.0, -theta_thr - 0.01, 0.0],
            [x_thr - 0.1, 5.0, theta_thr - 0.01, -5.0],
        ]
    )
    expected = np.array([case.env.is_terminal(s) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("action", [0, 1])
def test_transition_sampling_matches_native_moments(case: _Case, action: int) -> None:
    """Sampled next-state moments match the native transition kernel per config.

    Purpose: Validates the batched cart-pole physics + transition noise

    Given: A fixed non-trivial state and action sampled many times by both
    When: Empirical mean and covariance are compared
    Then: Mean gap < 1e-3 and covariance gap < 1e-3

    Test type: unit
    """
    torch.manual_seed(1)
    state = np.array([0.3, 0.5, 0.08, -0.4])
    native = case.env.sample_next_state_batch(np.tile(state, (60000, 1)), action)
    twin = case.model.sample_next_states(
        _tensor(np.tile(state, (60000, 1))),
        torch.full((60000,), action, dtype=torch.int64),
    ).numpy()
    assert np.max(np.abs(native.mean(0) - twin.mean(0))) < 1e-3
    assert np.max(np.abs(np.cov(native.T) - np.cov(twin.T))) < 1e-3


def test_observation_sampling_matches_noise_covariance(case: _Case) -> None:
    """Observation sampling reproduces the env's observation covariance.

    Purpose: Validates the observation noise kernel across configs

    Given: A fixed next state sampled many times by the model
    When: The empirical mean and covariance of the observations are computed
    Then: The mean matches the next state and the covariance matches noise_cov

    Test type: unit
    """
    torch.manual_seed(2)
    next_state = np.array([0.2, -0.3, 0.05, 0.4])
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (60000, 1))), torch.zeros(60000, dtype=torch.int64)
    ).numpy()
    assert np.max(np.abs(twin.mean(0) - next_state)) < 0.02
    assert np.max(np.abs(np.cov(twin.T) - case.env.noise_cov)) < 0.02


def test_action_keys_are_passthrough_indices(case: _Case) -> None:
    """Action keys are the integer action indices themselves.

    Purpose: Validates the discrete-action passthrough key mapping

    Given: A tensor of the two discrete action indices
    When: action_keys is called
    Then: The int64 keys equal the input indices

    Test type: unit
    """
    actions = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    keys = case.model.action_keys(actions)
    assert keys.dtype == torch.int64
    assert torch.equal(keys, actions)


def test_observation_keys_are_deterministic_and_discriminating(case: _Case) -> None:
    """Observation keys are stable per input and separate distinct cells.

    Purpose: Validates the continuous-observation to integer-key hashing

    Given: Observations, some identical and some in different grid cells
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct cells differ

    Test type: unit
    """
    observations = torch.tensor(
        [
            [0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01],
            [1.2, -0.7, 0.3, 2.4],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    keys_first = case.model.observation_keys(observations)
    keys_second = case.model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]
    assert keys_first[2] != keys_first[3]


def test_unsupported_integrator_raises() -> None:
    """Constructing on a non-euler integrator is rejected.

    Purpose: Validates the scope guard on the kinematics integrator

    Given: An env switched to the semi-implicit euler integrator
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    env.kinematics_integrator = "semi-implicit euler"
    with pytest.raises(NotImplementedError):
        CartPoleVectorizedModel(env)


def test_non_positive_observation_resolution_raises() -> None:
    """A non-positive observation resolution is rejected.

    Purpose: Validates the observation_resolution input guard

    Given: A default env and a zero observation resolution
    When: A vectorized model is constructed with that resolution
    Then: ValueError is raised

    Test type: unit
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    with pytest.raises(ValueError):
        CartPoleVectorizedModel(env, observation_resolution=0.0)
