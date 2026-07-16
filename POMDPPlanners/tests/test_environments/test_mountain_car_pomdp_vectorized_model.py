# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native Mountain Car env.

These tests pin :class:`MountainCarVectorizedModel` to the environment's
native (C++) kernels so the two implementations cannot drift. The parity
checks run over a *sweep of supported configurations* (varying the state
transition covariance) so a wiring bug that default parameters would hide is
caught. Deterministic paths (the physics transition mean including clamping /
the min-position corner, observation log-densities, reward, and terminal) are
compared exactly in float64; the stochastic transition and observation kernels
are compared by empirical moments over a large batch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.mountain_car_pomdp import _native
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
    MountainCarPOMDP,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_vectorized_model import (
    MountainCarVectorizedModel,
)

# Each case is a state-transition covariance; ``None`` keeps the env default.
_SUPPORTED_CASES = [
    None,
    np.diag([1e-4, 1e-5]),
    np.array([[2.0e-4, 5.0e-5], [5.0e-5, 4.0e-5]]),
]
_CASE_IDS = ["default-cov", "diagonal-cov", "correlated-cov"]


@dataclass
class _Case:
    env: MountainCarPOMDP
    model: MountainCarVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    cov = request.param
    env = MountainCarPOMDP(discount_factor=0.99, state_transition_cov=cov)
    model = MountainCarVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _random_states(rng: np.random.Generator, count: int) -> np.ndarray:
    positions = rng.uniform(-1.2, 0.6, size=count)
    velocities = rng.uniform(-0.07, 0.07, size=count)
    return np.stack((positions, velocities), axis=1)


def _native_deterministic_next(env: MountainCarPOMDP, state: np.ndarray, action: int) -> np.ndarray:
    kernel = _native.MountainCarTransitionCpp(
        state=state,
        action=action,
        power=env.power,
        gravity=env.gravity,
        max_speed=env.max_speed,
        min_position=env.min_position,
        max_position=env.max_position,
        covariance=env.state_transition_cov,
    )
    return np.asarray(
        kernel._compute_deterministic_next_state()
    )  # pylint: disable=protected-access


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


def test_deterministic_transition_mean_matches_native(case: _Case) -> None:
    """The physics transition mean matches the native kernel exactly.

    Purpose: Validates the deterministic step (hill, gravity, force, clamping)

    Given: Random states spanning the full position/velocity range per config
    When: The model's deterministic next state is compared to the native kernel
    Then: The maximum absolute difference is below 1e-12 for every action

    Test type: unit
    """
    rng = np.random.default_rng(11)
    states = _random_states(rng, 256)
    for action_index, action_value in enumerate(case.env.get_actions()):
        expected = np.array([_native_deterministic_next(case.env, s, action_value) for s in states])
        actions = torch.full((states.shape[0],), action_index, dtype=torch.int64)
        actual = case.model._deterministic_next(  # pylint: disable=protected-access
            _tensor(states), actions
        ).numpy()
        assert np.max(np.abs(expected - actual)) < 1e-12


def test_min_position_corner_is_reproduced(case: _Case) -> None:
    """The min-position floor rule zeroes negative velocity exactly as native.

    Purpose: Validates the ``p == min_position and v < 0 -> v = 0`` corner

    Given: A state pinned at the minimum position with strong reverse action
    When: The model's deterministic next state is compared to the native kernel
    Then: The clamped position and zeroed velocity agree exactly

    Test type: unit
    """
    state = np.array([case.env.min_position, -0.05])
    action_value = -1
    action_index = case.env.get_actions().index(action_value)
    expected = _native_deterministic_next(case.env, state, action_value)
    actual = case.model._deterministic_next(  # pylint: disable=protected-access
        _tensor(state[None, :]), torch.tensor([action_index], dtype=torch.int64)
    ).numpy()[0]
    assert np.allclose(expected, actual, atol=1e-12)
    assert actual[1] == 0.0


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the Gaussian observation likelihood kernel

    Given: (next_state, observation) pairs spanning the state range per config
    When: The model's observation_log_probs is compared to the env per pair
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(3)
    next_states = _random_states(rng, 256)
    observations = next_states + rng.normal(scale=0.05, size=(256, 2))
    action_value = case.env.get_actions()[0]
    expected = np.array(
        [
            case.env.observation_log_probability(
                next_states[i], action_value, observations[i][None, :]
            )[0]
            for i in range(next_states.shape[0])
        ]
    )
    actions = torch.zeros(next_states.shape[0], dtype=torch.int64)
    actual = case.model.observation_log_probs(
        _tensor(next_states), actions, _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_reward_matches_native(case: _Case) -> None:
    """Sparse step rewards match the native reward across the state range.

    Purpose: Validates the -1 / 0 goal reward kernel

    Given: States on both sides of the goal position per config
    When: The model reward is compared to env.reward_batch on those states
    Then: Every reward agrees exactly

    Test type: unit
    """
    rng = np.random.default_rng(5)
    states = _random_states(rng, 512)
    states[:10, 0] = case.env.goal_position + 0.05  # force some goal-reaching rows
    action_value = case.env.get_actions()[0]
    expected = case.env.reward_batch(states, action_value)
    actions = torch.zeros(states.shape[0], dtype=torch.int64)
    actual = case.model.rewards(_tensor(states), actions, _tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States at, just below, and above the goal position
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    goal = case.env.goal_position
    states = np.array([[goal, 0.0], [goal - 0.01, 0.0], [goal + 0.1, 0.0], [-0.5, 0.0]])
    expected = np.array([case.env.is_terminal(tuple(s)) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_transition_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled next-state moments match the native transition kernel per config.

    Purpose: Validates transition mean/covariance for each config's covariance

    Given: A mid-range state and forward action sampled many times by both
    When: Empirical mean and covariance are compared
    Then: Mean gap < 1e-3 and covariance gap < 1e-4

    Test type: unit
    """
    torch.manual_seed(1)
    state = np.array([-0.5, 0.0])
    action_value = case.env.get_actions()[2]
    action_index = 2
    native = case.env.sample_next_state_batch(np.tile(state, (60000, 1)), action_value)
    twin = case.model.sample_next_states(
        _tensor(np.tile(state, (60000, 1))),
        torch.full((60000,), action_index, dtype=torch.int64),
    ).numpy()
    assert np.max(np.abs(native.mean(0) - twin.mean(0))) < 1e-3
    assert np.max(np.abs(np.cov(native.T) - np.cov(twin.T))) < 1e-4


def test_observation_sampling_matches_native_covariance(case: _Case) -> None:
    """Observation sampling reproduces the observation-noise covariance.

    Purpose: Validates the Gaussian observation sampler's covariance

    Given: A next state sampled many times through the observation kernel
    When: The empirical covariance is compared to env.cov_matrix
    Then: The covariance gap is below a Monte Carlo tolerance of 5e-4

    Test type: unit
    """
    torch.manual_seed(2)
    next_state = np.array([-0.3, 0.02])
    twin = case.model.sample_observations(
        _tensor(np.tile(next_state, (60000, 1))),
        torch.zeros(60000, dtype=torch.int64),
    ).numpy()
    empirical_mean = twin.mean(0)
    assert np.max(np.abs(empirical_mean - next_state)) < 1e-3
    assert np.max(np.abs(np.cov(twin.T) - case.env.cov_matrix)) < 5e-4


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every protocol method returns the documented shape and dtype.

    Purpose: Validates output shapes/dtypes of the batched kernels

    Given: A small random batch of states, actions, and observations
    When: Each generative method is invoked
    Then: Shapes are [N, .] / [N] and key dtypes are int64

    Test type: unit
    """
    rng = np.random.default_rng(9)
    states = _tensor(_random_states(rng, 7))
    actions = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)
    next_states = case.model.sample_next_states(states, actions)
    observations = case.model.sample_observations(next_states, actions)
    assert next_states.shape == (7, 2)
    assert observations.shape == (7, 2)
    assert case.model.rewards(states, actions, next_states).shape == (7,)
    assert case.model.terminal_mask(states).shape == (7,)
    assert case.model.observation_log_probs(next_states, actions, observations).shape == (7,)
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_action_and_observation_keys_are_deterministic() -> None:
    """Action and observation keys are stable and discriminating.

    Purpose: Validates integer-key mapping for actions and continuous obs

    Given: Repeated action indices and observations, some identical, some not
    When: action_keys / observation_keys are called twice
    Then: Identical inputs map to identical keys and distinct cells differ

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    model = MountainCarVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    actions = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    assert torch.equal(model.action_keys(actions), actions)
    observations = torch.tensor([[-0.3, 0.02], [-0.3, 0.02], [0.4, -0.05], [-1.1, 0.06]])
    keys_first = model.observation_keys(observations)
    keys_second = model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]
    assert keys_first[2] != keys_first[3]


def test_non_scalar_action_set_raises() -> None:
    """Constructing on a non-scalar action set is rejected.

    Purpose: Validates the scope guard on the scalar action-set assumption

    Given: An env whose action set has been replaced by non-scalar actions
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    env.actions = [(-1, 0), (0, 0), (1, 0)]  # type: ignore[assignment]  # force unsupported set
    with pytest.raises(NotImplementedError):
        MountainCarVectorizedModel(env)


def test_non_positive_observation_resolution_raises() -> None:
    """A non-positive observation resolution is rejected.

    Purpose: Validates the observation-resolution guard

    Given: A valid env and a zero observation resolution
    When: A vectorized model is constructed with it
    Then: ValueError is raised

    Test type: unit
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    with pytest.raises(ValueError):
        MountainCarVectorizedModel(env, observation_resolution=0.0)
