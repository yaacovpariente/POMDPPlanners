# SPDX-License-Identifier: MIT

"""Parity tests for the Isaac Lab vectorized generative model.

The suite pins the torch kernels of :class:`IsaacLabVectorizedModel` to the
fitted numpy planner-side models (:class:`LinearGaussianTransition`,
:class:`GaussianObservationModel`, :class:`LinearRewardModel`) they are built
from. Deterministic kernels (transition mean, reward, observation
log-likelihood, terminal) are compared exactly; the stochastic transition and
observation kernels are compared by their empirical moments.
"""

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    GaussianObservationModel,
    LinearGaussianTransition,
    LinearRewardModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_vectorized_model import (
    IsaacLabVectorizedModel,
)

_STATE_DIM = 4
_ACTION_DIM = 3
_NUM_ACTIONS = 5


def _fitted_models(seed: int = 0):
    rng = np.random.default_rng(seed)
    weight_state = np.eye(_STATE_DIM) + 0.05 * rng.standard_normal((_STATE_DIM, _STATE_DIM))
    weight_action = 0.3 * rng.standard_normal((_STATE_DIM, _ACTION_DIM))
    bias = 0.1 * rng.standard_normal(_STATE_DIM)
    trans_cov = np.diag(0.01 + 0.02 * rng.random(_STATE_DIM))
    transition = LinearGaussianTransition(weight_state, weight_action, bias, trans_cov)
    observation = GaussianObservationModel(_STATE_DIM, noise_std=0.1)
    reward = LinearRewardModel(
        rng.standard_normal(_STATE_DIM),
        rng.standard_normal(_ACTION_DIM),
        rng.standard_normal(_STATE_DIM),
        float(rng.standard_normal()),
    )
    presets = rng.standard_normal((_NUM_ACTIONS, _ACTION_DIM))
    return transition, observation, reward, presets


def _build_model(seed: int = 0) -> IsaacLabVectorizedModel:
    transition, observation, reward, presets = _fitted_models(seed)
    return IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )


def _random_batch(num: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    states = torch.as_tensor(rng.standard_normal((num, _STATE_DIM)), dtype=torch.float64)
    actions = torch.as_tensor(rng.integers(0, _NUM_ACTIONS, size=num), dtype=torch.int64)
    return states, actions


def test_transition_mean_matches_native():
    """Test that the batched transition mean matches the native model.

    Purpose: Validates the linear-Gaussian transition mean kernel.

    Given: A fitted transition and a random batch of states and action indices
    When: The torch transition mean is compared to the native ``_mean`` per row
    Then: They agree to floating tolerance

    Test type: unit
    """
    transition, observation, reward, presets = _fitted_models()
    model = IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )
    states, actions = _random_batch(64)
    mean = model._transition_mean(states, actions)  # pylint: disable=protected-access
    for row in range(states.shape[0]):
        native = transition._mean(  # pylint: disable=protected-access
            states[row].numpy(), presets[int(actions[row])]
        )
        assert np.allclose(mean[row].numpy(), native, atol=1e-9)


def test_reward_matches_native():
    """Test that the batched reward matches the native linear reward model.

    Purpose: Validates the linear reward kernel.

    Given: A fitted reward model and random states / actions / next states
    When: The torch reward is compared to the native ``reward`` per row
    Then: They agree to floating tolerance

    Test type: unit
    """
    transition, observation, reward, presets = _fitted_models()
    model = IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )
    states, actions = _random_batch(64, seed=2)
    next_states, _ = _random_batch(64, seed=3)
    rewards = model.rewards(states, actions, next_states)
    for row in range(states.shape[0]):
        native = reward.reward(
            states[row].numpy(), presets[int(actions[row])], next_states[row].numpy()
        )
        assert np.isclose(float(rewards[row]), native, atol=1e-9)


def test_observation_log_probs_match_native():
    """Test that observation log-likelihoods match the native model.

    Purpose: Validates the additive-Gaussian observation likelihood kernel.

    Given: A fitted observation model and random next states / observations
    When: The torch ``observation_log_probs`` is compared to the native
        ``log_probability`` per row
    Then: They agree to floating tolerance

    Test type: unit
    """
    transition, observation, reward, presets = _fitted_models()
    model = IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )
    next_states, actions = _random_batch(48, seed=4)
    observations, _ = _random_batch(48, seed=5)
    log_probs = model.observation_log_probs(next_states, actions, observations)
    for row in range(next_states.shape[0]):
        native = observation.log_probability(next_states[row].numpy(), observations[row].numpy())
        assert np.isclose(float(log_probs[row]), float(np.asarray(native).ravel()[0]), atol=1e-9)


def test_terminal_mask_is_all_false():
    """Test that the terminal mask is always false.

    Purpose: Validates that the Isaac velocity model never terminates.

    Given: A vectorized model and a random batch of states
    When: ``terminal_mask`` is queried
    Then: Every entry is ``False``

    Test type: unit
    """
    model = _build_model()
    states, _ = _random_batch(32)
    assert not bool(model.terminal_mask(states).any())


def test_transition_sample_moments_match_native():
    """Test that sampled transitions reproduce the native mean and covariance.

    Purpose: Validates the stochastic transition kernel by empirical moments.

    Given: A fixed state and action repeated across a large batch
    When: The torch transition samples are summarised by mean and covariance
    Then: They match the native transition mean and covariance to tolerance

    Test type: integration
    """
    transition, observation, reward, presets = _fitted_models()
    model = IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )
    torch.manual_seed(0)
    state = torch.as_tensor(np.linspace(-1.0, 1.0, _STATE_DIM), dtype=torch.float64)
    states = state.repeat(200000, 1)
    actions = torch.full((200000,), 2, dtype=torch.int64)
    samples = model.sample_next_states(states, actions)
    native_mean = transition._mean(state.numpy(), presets[2])  # pylint: disable=protected-access
    assert np.allclose(samples.mean(dim=0).numpy(), native_mean, atol=2e-2)
    empirical_cov = np.cov(samples.numpy(), rowvar=False)
    native_cov = transition._normal.covariance  # pylint: disable=protected-access
    assert np.allclose(empirical_cov, native_cov, atol=2e-2)


def test_observation_sample_moments_match_noise():
    """Test that sampled observations reproduce the additive-noise covariance.

    Purpose: Validates the stochastic observation kernel by empirical moments.

    Given: A fixed next state repeated across a large batch
    When: The torch observation samples are summarised by mean and covariance
    Then: The mean matches the next state and the covariance matches the noise

    Test type: integration
    """
    transition, observation, reward, presets = _fitted_models()
    model = IsaacLabVectorizedModel(
        transition, observation, reward, presets, device=torch.device("cpu"), dtype=torch.float64
    )
    torch.manual_seed(1)
    next_state = torch.as_tensor(np.linspace(0.0, 2.0, _STATE_DIM), dtype=torch.float64)
    next_states = next_state.repeat(200000, 1)
    actions = torch.zeros(200000, dtype=torch.int64)
    samples = model.sample_observations(next_states, actions)
    assert np.allclose(samples.mean(dim=0).numpy(), next_state.numpy(), atol=2e-3)
    empirical_cov = np.cov(samples.numpy(), rowvar=False)
    native_cov = observation._normal.covariance  # pylint: disable=protected-access
    assert np.allclose(empirical_cov, native_cov, atol=5e-3)


def test_action_and_observation_keys_are_deterministic():
    """Test that action and observation keys are deterministic integer keys.

    Purpose: Validates the integer tree-key kernels.

    Given: A batch of actions and observations
    When: ``action_keys`` and ``observation_keys`` are computed twice
    Then: Both are int64 and identical across the two calls, and action keys
        equal the action indices

    Test type: unit
    """
    model = _build_model()
    _, actions = _random_batch(20)
    observations, _ = _random_batch(20, seed=9)
    action_keys = model.action_keys(actions)
    obs_keys_a = model.observation_keys(observations)
    obs_keys_b = model.observation_keys(observations)
    assert action_keys.dtype == torch.int64
    assert obs_keys_a.dtype == torch.int64
    assert torch.equal(action_keys, actions)
    assert torch.equal(obs_keys_a, obs_keys_b)


@pytest.mark.parametrize("observation_resolution", [0.0, -1.0])
def test_non_positive_observation_resolution_raises(observation_resolution):
    """Test that a non-positive observation resolution is rejected.

    Purpose: Validates the observation-resolution guard.

    Given: Fitted models and a non-positive observation resolution
    When: A vectorized model is constructed
    Then: ``ValueError`` is raised

    Test type: unit
    """
    transition, observation, reward, presets = _fitted_models()
    with pytest.raises(ValueError):
        IsaacLabVectorizedModel(
            transition,
            observation,
            reward,
            presets,
            device=torch.device("cpu"),
            observation_resolution=observation_resolution,
        )


def test_mismatched_action_preset_dim_raises():
    """Test that action presets of the wrong action dim are rejected.

    Purpose: Validates the action-preset dimension guard.

    Given: A transition with action dim 3 and presets with action dim 2
    When: A vectorized model is constructed
    Then: ``ValueError`` is raised

    Test type: unit
    """
    transition, observation, reward, _ = _fitted_models()
    bad_presets = np.zeros((_NUM_ACTIONS, _ACTION_DIM - 1))
    with pytest.raises(ValueError):
        IsaacLabVectorizedModel(
            transition, observation, reward, bad_presets, device=torch.device("cpu")
        )
