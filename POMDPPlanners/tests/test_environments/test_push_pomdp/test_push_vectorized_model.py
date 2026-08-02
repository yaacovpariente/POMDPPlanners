# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native discrete Push env.

These tests pin :class:`PushVectorizedModel` to the environment's own kernels
(the closed-form Python transition reference, the ``DiscretePushRewardModel``
reward, the Gaussian object-position observation likelihood, and the terminal
check) so the two implementations cannot drift. The checks run over a sweep of
supported configurations (varying grid size, push threshold, friction,
observation noise, obstacles, and dangerous areas) so a wiring bug that default
parameters would hide is caught. Deterministic quantities (transition, reward,
terminal, log-density) are compared exactly in float64; the stochastic
observation sample is checked by its exact (robot/target) slice and clamp
bounds.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)
from POMDPPlanners.environments.push_pomdp.push_vectorized_model import PushVectorizedModel

# Each case is a set of PushPOMDP kwargs. All use the supported
# CONSTANT_HAZARD_PENALTY reward model with deterministic (probability 1.0)
# penalties and no transition error, so every kernel compared is deterministic.
_SUPPORTED_CASES = [
    {"discount_factor": 0.99},
    {
        "discount_factor": 0.95,
        "obstacles": [(3.0, 3.0), (6.0, 6.0)],
        "obstacle_radius": 0.8,
        "obstacle_penalty": -12.0,
    },
    {
        "discount_factor": 0.9,
        "grid_size": 12,
        "push_threshold": 1.5,
        "friction_coefficient": 0.5,
        "observation_noise": 0.2,
    },
    {
        "discount_factor": 0.95,
        "dangerous_areas": [(4.0, 4.0)],
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": -8.0,
    },
]
_CASE_IDS = ["default", "obstacles", "custom-physics", "dangerous-areas"]


@dataclass
class _Case:
    env: PushPOMDP
    model: PushVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = PushPOMDP(**request.param)
    model = PushVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _random_states(env: PushPOMDP, count: int, rng: np.random.Generator) -> np.ndarray:
    """States with the robot near the object so the push path is exercised."""
    grid_max = float(env.grid_size - 1)
    obj = rng.uniform(0.0, grid_max, size=(count, 2))
    robot = np.clip(obj + rng.uniform(-1.5, 1.5, size=(count, 2)), 0.0, grid_max)
    target = rng.uniform(0.0, grid_max, size=(count, 2))
    return np.concatenate([robot, obj, target], axis=1)


def _random_actions(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 4, size=count)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the action count is four

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == 4


def test_transition_matches_native_exactly(case: _Case) -> None:
    """Deterministic transitions match the env's closed-form kernel per row.

    Purpose: Validates the batched push transition against the env reference

    Given: Random near-object states and per-row actions in each config
    When: The model transition is compared to the env's per-row Python kernel
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(11)
    states = _random_states(case.env, 512, rng)
    actions = _random_actions(512, rng)
    expected = np.array(
        [
            case.env._compute_next_state_for_action_python(  # pylint: disable=protected-access
                states[i], case.env.actions[int(actions[i])]
            )
            for i in range(states.shape[0])
        ]
    )
    actual = case.model.sample_next_states(
        _tensor(states), torch.as_tensor(actions, dtype=torch.int64)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Rewards match the deterministic native reward kernel per row.

    Purpose: Validates the reward kernel (distance, goal, obstacle, danger)

    Given: Random next states and per-row actions in each config
    When: The model reward is compared to the env reward model per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(13)
    next_states = _random_states(case.env, 512, rng)
    actions = _random_actions(512, rng)
    expected = np.array(
        [
            case.env.reward_model.compute_reward(
                next_states[i], case.env.actions[int(actions[i])], next_states[i]
            )
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.rewards(
        _tensor(next_states), torch.as_tensor(actions, dtype=torch.int64), _tensor(next_states)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the object-position Gaussian likelihood across configs

    Given: (next_state, observation) pairs with noisy object positions
    When: The model observation_log_probs is compared to the env per pair
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(17)
    next_states = _random_states(case.env, 256, rng)
    observations = next_states.copy()
    observations[:, 2:4] += rng.normal(scale=0.15, size=(256, 2))
    expected = np.array(
        [
            case.env.observation_log_probability_single(
                next_states[i], case.env.actions[0], observations[i]
            )
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), torch.zeros(256, dtype=torch.int64), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States with the object on, just inside, and outside the goal radius
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    target = np.array([case.env.grid_size - 1, case.env.grid_size - 1], dtype=np.float64)
    offsets = np.array([[0.0, 0.0], [0.4, 0.0], [0.6, 0.0], [3.0, 3.0]])
    states = np.array([[0.0, 0.0, *(target + off), *target] for off in offsets], dtype=np.float64)
    expected = np.array([case.env.is_terminal(s) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_observation_sample_preserves_exact_dims_and_clamps(case: _Case) -> None:
    """Observation sampling keeps robot/target exact and clamps the object slice.

    Purpose: Validates the stochastic observation sampler's structure

    Given: A batch of next states sampled many times
    When: sample_observations is drawn for the batch
    Then: Robot/target dims are exact and the object slice stays within grid

    Test type: unit
    """
    torch.manual_seed(3)
    rng = np.random.default_rng(19)
    next_states = _random_states(case.env, 4096, rng)
    grid_max = float(case.env.grid_size - 1)
    observations = case.model.sample_observations(
        _tensor(next_states), torch.zeros(4096, dtype=torch.int64)
    ).numpy()
    assert np.allclose(observations[:, [0, 1, 4, 5]], next_states[:, [0, 1, 4, 5]])
    assert observations[:, 2:4].min() >= 0.0
    assert observations[:, 2:4].max() <= grid_max


def test_observation_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled object-observation noise matches the native Gaussian in moments.

    Purpose: Validates the observation noise covariance against the env sigma

    Given: A fixed mid-grid next state sampled many times by the model
    When: The empirical object-position covariance is computed
    Then: It matches an isotropic observation_noise**2 covariance

    Test type: unit
    """
    torch.manual_seed(5)
    center = float(case.env.grid_size - 1) / 2.0
    next_state = np.array([center, center, center, center, center, center])
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (40000, 1))), torch.zeros(40000, dtype=torch.int64)
    ).numpy()
    empirical = np.cov(twin[:, 2:4].T)
    expected = np.eye(2) * (case.env.observation_noise**2)
    assert np.max(np.abs(empirical - expected)) < 0.01


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every kernel returns the documented shape and dtype.

    Purpose: Validates output shapes/dtypes of the full generative interface

    Given: A small batch of states, actions, and observations
    When: Each protocol method is invoked
    Then: Shapes match [N, ds]/[N, do]/[N] and key/mask dtypes are correct

    Test type: unit
    """
    rng = np.random.default_rng(23)
    states = _tensor(_random_states(case.env, 8, rng))
    actions = torch.as_tensor(_random_actions(8, rng), dtype=torch.int64)
    next_states = case.model.sample_next_states(states, actions)
    observations = case.model.sample_observations(next_states, actions)
    assert next_states.shape == (8, 6)
    assert observations.shape == (8, 6)
    assert case.model.rewards(states, actions, next_states).shape == (8,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_transition_error_prob_is_stochastic() -> None:
    """A positive transition-error probability makes the transition stochastic.

    Purpose: Validates the per-particle action-error branch in the transition

    Given: An env with transition_error_prob=0.5 and a fixed state/action batch
    When: Two transitions are sampled from the same inputs
    Then: The two batches differ on at least one row

    Test type: unit
    """
    torch.manual_seed(7)
    env = PushPOMDP(discount_factor=0.99, transition_error_prob=0.5)
    model = PushVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    states = _tensor(np.tile(np.array([2.0, 2.0, 2.5, 2.0, 9.0, 9.0]), (256, 1)))
    actions = torch.full((256,), 2, dtype=torch.int64)
    first = model.sample_next_states(states, actions).numpy()
    second = model.sample_next_states(states, actions).numpy()
    assert not np.array_equal(first, second)


def test_unsupported_reward_model_raises() -> None:
    """Constructing on an unsupported reward model is rejected.

    Purpose: Validates the scope guard on reward model type

    Given: An env configured with the zero-mean-shock hazard reward model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    other = PushPOMDP(
        discount_factor=0.99,
        reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
    )
    with pytest.raises(NotImplementedError):
        PushVectorizedModel(other)


def test_non_positive_observation_resolution_raises() -> None:
    """A non-positive observation resolution is rejected at construction.

    Purpose: Validates the observation-resolution argument guard

    Given: A default env and observation_resolution=0.0
    When: A vectorized model is constructed
    Then: ValueError is raised

    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.99)
    with pytest.raises(ValueError):
        PushVectorizedModel(env, observation_resolution=0.0)


def test_observation_keys_are_deterministic_and_discriminating() -> None:
    """Observation keys are stable per input and separate distinct cells.

    Purpose: Validates the continuous-observation to integer-key hashing

    Given: Observations, some identical and some with distinct object cells
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct cells differ

    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.99)
    model = PushVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    observations = torch.tensor(
        [
            [0.0, 0.0, 1.05, 2.05, 9.0, 9.0],
            [0.0, 0.0, 1.05, 2.05, 9.0, 9.0],
            [0.0, 0.0, 3.2, 7.8, 9.0, 9.0],
            [0.0, 0.0, 8.0, 1.0, 9.0, 9.0],
        ]
    )
    keys_first = model.observation_keys(observations)
    keys_second = model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]
    assert keys_first[2] != keys_first[3]


def test_action_keys_are_identity() -> None:
    """Action keys pass discrete action indices through as int64.

    Purpose: Validates the discrete-action passthrough keying

    Given: A tensor of the four discrete action indices
    When: action_keys is called
    Then: The result equals the input indices as int64

    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.99)
    model = PushVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    keys = model.action_keys(actions)
    assert keys.dtype == torch.int64
    assert torch.equal(keys, actions)
