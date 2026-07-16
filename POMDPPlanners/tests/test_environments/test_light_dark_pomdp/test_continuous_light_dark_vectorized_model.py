# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native Continuous Light-Dark env.

These tests pin :class:`ContinuousLightDarkVectorizedModel` to the
environment's native (C++/numba) kernels so the two implementations cannot
drift. The parity checks run over a *sweep of supported configurations*
(varying covariances, goal / obstacle / beacon layout, costs, grid size, and
the action table) so a wiring bug that default parameters would hide is
caught. Log-densities and free-space rewards are compared exactly in float64;
stochastic kernels (sampling, hazard reward) are compared by empirical
moments over a large batch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ObservationModelType,
    RewardModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
    ContinuousLightDarkVectorizedModel,
)

_DEFAULT_ACTIONS = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
_EIGHT_ACTIONS = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.7, 0.7],
        [-0.7, 0.7],
        [0.7, -0.7],
        [-0.7, -0.7],
    ]
)

# Each case is (env kwargs, optional action_vectors). All are *supported*
# configurations: constant-hazard reward, normal-noise observation,
# is_obstacle_hit_terminal=False. Parameters otherwise vary widely.
_SUPPORTED_CASES = [
    ({}, None),
    (
        {
            "state_transition_cov_matrix": np.array([[0.08, 0.02], [0.02, 0.06]]),
            "observation_cov_matrix": np.array([[0.10, 0.0], [0.0, 0.03]]),
            "goal_state": np.array([3, 8]),
            "obstacles": [(2, 2), (7, 7), (4, 9)],
            "obstacle_hit_probability": 0.35,
            "obstacle_reward": -15.0,
            "goal_reward": 25.0,
            "fuel_cost": 1.0,
            "grid_size": 12,
            "goal_state_radius": 1.0,
            "obstacle_radius": 1.0,
            "beacon_radius": 1.5,
        },
        None,
    ),
    (
        {
            "state_transition_cov_matrix": np.eye(2) * 0.2,
            "obstacles": [],
            "grid_size": 15,
        },
        None,
    ),
    (
        {
            "beacons": [(1, 1), (6, 6), (9, 2)],
            "beacon_radius": 2.0,
        },
        _EIGHT_ACTIONS,
    ),
]
_CASE_IDS = ["default", "anisotropic-shifted", "no-obstacles-noisy", "custom-actions-beacons"]


@dataclass
class _Case:
    env: ContinuousLightDarkPOMDP
    model: ContinuousLightDarkVectorizedModel
    action_vectors: np.ndarray


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env_kwargs, action_vectors = request.param
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95, is_obstacle_hit_terminal=False, **env_kwargs
    )
    model = ContinuousLightDarkVectorizedModel(
        env, action_vectors=action_vectors, device=torch.device("cpu"), dtype=torch.float64
    )
    vectors = _DEFAULT_ACTIONS if action_vectors is None else np.asarray(action_vectors)
    return _Case(env=env, model=model, action_vectors=vectors)


def _free_space_mask(env: ContinuousLightDarkPOMDP, points: np.ndarray) -> np.ndarray:
    """Points where the reward is deterministic (no goal/obstacle/edge draw)."""
    margin = 0.5
    goal = np.asarray(env.goal_state, dtype=np.float64)
    not_goal = np.linalg.norm(points - goal, axis=1) > env.goal_state_radius + margin
    obstacles = np.asarray(env.obstacles, dtype=np.float64)
    if obstacles.size:
        deltas = points[:, :, None] - obstacles[None, :, :]
        min_dist = np.linalg.norm(deltas, axis=1).min(axis=1)
        not_obstacle = min_dist > env.obstacle_radius + margin
    else:
        not_obstacle = np.ones(points.shape[0], dtype=bool)
    in_grid = (points > margin).all(axis=1) & (points < env.grid_size - margin).all(axis=1)
    return not_goal & not_obstacle & in_grid


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _action_indices(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.int64)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the action count matches

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == case.action_vectors.shape[0]


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the observation likelihood kernel across configs

    Given: (next_state, observation) pairs spanning near/far beacons per config
    When: The model's observation_log_probs is compared to the env per pair
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(3)
    next_states = rng.uniform(0.0, float(case.env.grid_size), size=(256, 2))
    observations = next_states + rng.normal(scale=0.1, size=(256, 2))
    action_vector = case.action_vectors[0]
    expected = np.array(
        [
            case.env.observation_log_probability_single(
                next_states[i], action_vector, observations[i]
            )
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), _action_indices(256), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_reward_matches_native_in_free_space(case: _Case) -> None:
    """Free-space rewards match the native kernel exactly across configs.

    Purpose: Validates the deterministic reward path (fuel + goal distance)

    Given: Next states filtered to the config's obstacle/goal/edge-free region
    When: The model reward is compared to env.reward_batch on those states
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(5)
    points = rng.uniform(0.0, float(case.env.grid_size), size=(1024, 2))
    free = points[_free_space_mask(case.env, points)]
    assert free.shape[0] > 20
    action_vector = case.action_vectors[0]
    expected = case.env.reward_batch(free, action_vector, next_states=free)
    actual = case.model.rewards(
        _tensor(free), _action_indices(free.shape[0]), _tensor(free)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_transition_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled next-state moments match the native transition kernel per config.

    Purpose: Validates transition mean/covariance for each config's covariance

    Given: A fixed mid-grid state and action sampled many times by both
    When: Empirical mean and covariance are compared
    Then: Mean gap < 0.05 and covariance gap < 0.03

    Test type: unit
    """
    torch.manual_seed(1)
    center = float(case.env.grid_size) / 2.0
    state = np.array([center, center])
    action_vector = case.action_vectors[0]
    native = case.env.sample_next_state_batch(np.tile(state, (40000, 1)), action_vector)
    twin = case.model.sample_next_states(
        _tensor(np.tile(state, (40000, 1))), _action_indices(40000)
    ).numpy()
    assert np.max(np.abs(native.mean(0) - twin.mean(0))) < 0.05
    assert np.max(np.abs(np.cov(native.T) - np.cov(twin.T))) < 0.03


def test_observation_sampling_uses_near_beacon_covariance(case: _Case) -> None:
    """Observation sampling reproduces the near-beacon (halved) covariance.

    Purpose: Validates the near/far covariance switch across configs

    Given: A next state on one of the config's beacons (near) sampled many times
    When: The empirical covariance is compared to obs_cov * 0.5
    Then: The covariance gap is below 0.02

    Test type: unit
    """
    torch.manual_seed(2)
    beacon_point = np.asarray(case.env.beacons, dtype=np.float64)[:, 0]
    twin = case.model.sample_observations(
        _tensor(np.tile(beacon_point, (40000, 1))), _action_indices(40000)
    ).numpy()
    expected_cov = case.env.observation_cov_matrix * 0.5
    assert np.max(np.abs(np.cov(twin.T) - expected_cov)) < 0.02


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States on the goal, just outside the goal radius, and far away
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    goal = np.asarray(case.env.goal_state, dtype=np.float64)
    radius = case.env.goal_state_radius
    states = np.array(
        [goal, goal + np.array([radius + 0.5, 0.0]), goal + np.array([radius - 0.2, 0.0])]
    )
    expected = np.array([case.env.is_terminal(s) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_reward_hazard_expectation_matches_native() -> None:
    """Mean hazard reward inside an obstacle matches the native kernel.

    Purpose: Validates the stochastic obstacle-hit contribution in expectation

    Given: Many repeats of a next state sitting on an obstacle centre
    When: Mean rewards from the env and the model are compared over the repeats
    Then: The means agree within a Monte Carlo tolerance of 0.1

    Test type: unit
    """
    np.random.seed(7)
    torch.manual_seed(7)
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    batch = np.tile(np.array([5.0, 5.0]), (20000, 1))
    env_mean = float(np.mean(env.reward_batch(batch, np.array([1.0, 0.0]), next_states=batch)))
    model_mean = float(model.rewards(_tensor(batch), _action_indices(20000), _tensor(batch)).mean())
    assert abs(env_mean - model_mean) < 0.1


def test_unsupported_reward_model_raises() -> None:
    """Constructing on an unsupported reward model is rejected.

    Purpose: Validates the scope guard on reward model type

    Given: An env configured with the distance-decayed hazard reward model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    other = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        is_obstacle_hit_terminal=False,
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
    )
    with pytest.raises(NotImplementedError):
        ContinuousLightDarkVectorizedModel(other)


def test_unsupported_observation_model_raises() -> None:
    """Constructing on an unsupported observation model is rejected.

    Purpose: Validates the scope guard on observation model type

    Given: An env configured with the no-observation-in-dark model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    other = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        is_obstacle_hit_terminal=False,
        observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
    )
    with pytest.raises(NotImplementedError):
        ContinuousLightDarkVectorizedModel(other)


def test_default_hazard_terminal_config_raises() -> None:
    """The default draw-coupled hazard-terminal config is rejected.

    Purpose: Validates the scope guard on the hazard-terminal absorbing slot

    Given: An env left at the default is_obstacle_hit_terminal=True
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    other = ContinuousLightDarkPOMDP(discount_factor=0.95)
    with pytest.raises(NotImplementedError):
        ContinuousLightDarkVectorizedModel(other)


def test_observation_keys_are_deterministic_and_discriminating() -> None:
    """Observation keys are stable per input and separate distinct cells.

    Purpose: Validates the continuous-observation to integer-key hashing

    Given: Observations, some identical and some in different grid cells
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct cells differ

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    observations = torch.tensor([[0.05, 0.05], [0.05, 0.05], [3.2, 7.8], [-1.0, -1.0]])
    keys_first = model.observation_keys(observations)
    keys_second = model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]
    assert keys_first[2] != keys_first[3]
