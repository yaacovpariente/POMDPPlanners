# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native discrete LaserTag env.

These tests pin :class:`LaserTagVectorizedModel` to the environment's native
scalar/numpy kernels so the two implementations cannot drift. The parity
checks run over a sweep of supported configurations (varying measurement
noise, walls, and dangerous areas) so a wiring bug that default parameters
would hide is caught. Log-densities, rewards, and terminal flags are compared
exactly; the stochastic transition kernel is compared against the env's
analytic ``transition_log_probability`` over the empirical support, and the
stochastic observation kernel is compared by empirical moments.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_vectorized_model import (
    LaserTagVectorizedModel,
)

# Each entry is a supported env-kwargs dict (constant-hazard reward, EVADE
# opponent, deterministic motion, non-terminal hazards).
_SUPPORTED_CASES = [
    {},
    {"measurement_noise": 2.0},
    {
        "walls": {(1, 1), (2, 3), (4, 2)},
        "dangerous_areas": set(),
        "measurement_noise": 0.5,
    },
]
_CASE_IDS = ["default", "noisy", "custom-walls-no-danger"]


@dataclass
class _Case:
    env: LaserTagPOMDP
    model: LaserTagVectorizedModel
    valid_cells: List[Tuple[int, int]]


def _valid_cells(env: LaserTagPOMDP) -> List[Tuple[int, int]]:
    rows, cols = env.floor_shape
    return [(row, col) for row in range(rows) for col in range(cols) if (row, col) not in env.walls]


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = LaserTagPOMDP(discount_factor=0.95, **request.param)
    model = LaserTagVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model, valid_cells=_valid_cells(env))


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _random_states(case: _Case, count: int, rng: np.random.Generator) -> np.ndarray:
    cells = np.asarray(case.valid_cells, dtype=np.float64)
    robots = cells[rng.integers(0, cells.shape[0], size=count)]
    opps = cells[rng.integers(0, cells.shape[0], size=count)]
    terminal = rng.integers(0, 2, size=(count, 1)).astype(np.float64)
    return np.concatenate([robots, opps, terminal], axis=1)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and the action count is five

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == 5


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Batched rewards match the env's scalar reward kernel exactly.

    Purpose: Validates the deterministic constant-hazard reward path

    Given: Random states, actions, and realised next states per config
    When: The model reward is compared to env.reward row by row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(1)
    states = _random_states(case, 300, rng)
    next_states = _random_states(case, 300, rng)
    actions = rng.integers(0, 5, size=300)
    expected = np.array(
        [
            case.env.reward(states[i], int(actions[i]), next_states[i])
            for i in range(states.shape[0])
        ]
    )
    actual = case.model.rewards(
        _tensor(states), torch.as_tensor(actions), _tensor(next_states)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the env's per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: Random states with mixed terminal flags
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    rng = np.random.default_rng(2)
    states = _random_states(case, 200, rng)
    expected = np.array([case.env.is_terminal(states[i]) for i in range(states.shape[0])])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the env kernel to float64 precision.

    Purpose: Validates the 8-direction laser likelihood kernel across configs

    Given: Non-terminal next states and realistic sampled observations
    When: The model's observation_log_probs is compared to the env per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(3)
    next_states = _random_states(case, 200, rng)
    next_states[:, 4] = 0.0  # non-terminal so lasers are emitted
    observations = np.array(
        [
            np.asarray(case.env.sample_observation(next_states[i], 0), dtype=np.float64)
            for i in range(next_states.shape[0])
        ]
    )
    expected = np.array(
        [
            case.env.observation_log_probability(next_states[i], 0, [observations[i]])[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), torch.zeros(200, dtype=torch.int64), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_terminal_observation_log_probs_match_native(case: _Case) -> None:
    """Terminal-state observation likelihoods honor the sentinel semantics.

    Purpose: Validates the terminal sentinel branch of the likelihood kernel

    Given: A terminal next state scored against the sentinel and a real reading
    When: observation_log_probs is evaluated for both observations
    Then: The sentinel scores 0.0 and a non-sentinel scores -inf

    Test type: unit
    """
    next_states = _tensor(np.array([[3.0, 3.0, 6.0, 5.0, 1.0], [3.0, 3.0, 6.0, 5.0, 1.0]]))
    sentinel = [-1.0] * 8
    reading = [3.0] * 8
    observations = _tensor(np.array([sentinel, reading]))
    logp = case.model.observation_log_probs(
        next_states, torch.zeros(2, dtype=torch.int64), observations
    ).numpy()
    assert logp[0] == pytest.approx(0.0)
    assert np.isneginf(logp[1])


@pytest.mark.parametrize(
    "policy, state, action",
    [
        # EVADE: opponent flees the robot's pre-move cell.
        (OpponentPolicy.EVADE, np.array([2.0, 2.0, 6.0, 5.0, 0.0]), 0),
        (OpponentPolicy.EVADE, np.array([4.0, 3.0, 4.0, 1.0, 0.0]), 2),
        (OpponentPolicy.EVADE, np.array([5.0, 2.0, 3.0, 2.0, 0.0]), 4),
        (OpponentPolicy.EVADE, np.array([3.0, 3.0, 3.0, 3.0, 0.0]), 4),
        # PURSUE: opponent chases the robot's post-move cell, so the move
        # action (which shifts the robot) changes the opponent conditioning.
        (OpponentPolicy.PURSUE, np.array([2.0, 2.0, 6.0, 5.0, 0.0]), 0),
        (OpponentPolicy.PURSUE, np.array([4.0, 3.0, 4.0, 1.0, 0.0]), 2),
        (OpponentPolicy.PURSUE, np.array([5.0, 2.0, 3.0, 2.0, 0.0]), 1),
        (OpponentPolicy.PURSUE, np.array([3.0, 3.0, 3.0, 3.0, 0.0]), 4),
        # EVADE_WHEN_SPOTTED, spotted branch: opponent shares the robot's row
        # (an unoccluded east ray) so it flees like EVADE.
        (OpponentPolicy.EVADE_WHEN_SPOTTED, np.array([2.0, 2.0, 2.0, 5.0, 0.0]), 4),
        # EVADE_WHEN_SPOTTED, unspotted branch: opponent off every laser ray so
        # it moves uniformly at random.
        (OpponentPolicy.EVADE_WHEN_SPOTTED, np.array([2.0, 2.0, 5.0, 4.0, 0.0]), 4),
        (OpponentPolicy.EVADE_WHEN_SPOTTED, np.array([1.0, 1.0, 8.0, 5.0, 0.0]), 0),
    ],
)
def test_transition_distribution_matches_native(
    policy: OpponentPolicy, state: np.ndarray, action: int
) -> None:
    """Sampled next-state frequencies match the env's analytic transition.

    Purpose: Validates the opponent-move + robot-move categorical for every
        supported opponent policy (EVADE, PURSUE, EVADE_WHEN_SPOTTED)

    Given: A fixed state/action and opponent policy sampled many times by the
        torch model
    When: Empirical frequencies of each distinct next state are compared to
        the env's transition_log_probability over the same support
    Then: Every outcome agrees within a Monte Carlo tolerance and the env
        support probabilities sum to one

    Test type: unit
    """
    torch.manual_seed(0)
    env = LaserTagPOMDP(discount_factor=0.95, opponent_policy=policy)
    model = LaserTagVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    count = 60000
    samples = model.sample_next_states(
        _tensor(np.tile(state, (count, 1))), torch.full((count,), action, dtype=torch.int64)
    ).numpy()
    unique, counts = np.unique(samples, axis=0, return_counts=True)
    freqs = counts / count
    total_prob = 0.0
    for row, freq in zip(unique, freqs):
        prob = float(np.exp(env.transition_log_probability(state, action, [row])[0]))
        total_prob += prob
        assert abs(freq - prob) < 0.01
    assert abs(total_prob - 1.0) < 1e-9


def test_observation_sampling_matches_native_moments(case: _Case) -> None:
    """Sampled laser observations match the env sampler's mean and noise level.

    Purpose: Validates the laser-geometry and Gaussian noise of the sampler

    Given: A next state whose lasers are far from walls and the opponent
    When: Many observations are drawn by both the model and the env
    Then: Per-direction means agree within 0.1 and the model std tracks sigma

    Test type: unit
    """
    torch.manual_seed(1)
    next_state = np.array([5.0, 3.0, 0.0, 0.0, 0.0])
    count = 40000
    env_obs = np.asarray(
        case.env.sample_observation(next_state, 0, n_samples=count), dtype=np.float64
    )
    model_obs = case.model.sample_observations(
        _tensor(np.tile(next_state, (count, 1))), torch.zeros(count, dtype=torch.int64)
    ).numpy()
    assert np.max(np.abs(env_obs.mean(axis=0) - model_obs.mean(axis=0))) < 0.1
    unclipped = model_obs.mean(axis=0) > 2.0 * case.env.measurement_noise
    if unclipped.any():
        std_gap = np.abs(model_obs.std(axis=0)[unclipped] - case.env.measurement_noise)
        assert np.max(std_gap) < 0.1


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every protocol method returns the documented shape and dtype.

    Purpose: Validates the tensor contract of all eight protocol methods

    Given: A random batch of states, actions, and next states
    When: Each generative method is invoked
    Then: Shapes match [N, ds]/[N, do]/[N] and key/mask dtypes are correct

    Test type: unit
    """
    rng = np.random.default_rng(4)
    states = _tensor(_random_states(case, 16, rng))
    next_states = _tensor(_random_states(case, 16, rng))
    actions = torch.as_tensor(rng.integers(0, 5, size=16))
    next_sampled = case.model.sample_next_states(states, actions)
    observations = case.model.sample_observations(next_states, actions)
    assert next_sampled.shape == (16, 5)
    assert observations.shape == (16, 8)
    assert case.model.rewards(states, actions, next_states).shape == (16,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.observation_log_probs(next_states, actions, observations).shape == (16,)
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_keys_are_deterministic_and_discriminating(case: _Case) -> None:
    """Action and observation keys are stable and separate distinct inputs.

    Purpose: Validates the integer tree-key mappings

    Given: Repeated actions and observations, some identical and some distinct
    When: action_keys and observation_keys are called twice
    Then: Identical inputs map to identical keys and distinct inputs differ

    Test type: unit
    """
    actions = torch.tensor([0, 4, 2])
    assert torch.equal(case.model.action_keys(actions), actions.to(torch.int64))
    observations = _tensor(
        np.array([[1.0] * 8, [1.0] * 8, [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    )
    keys_first = case.model.observation_keys(observations)
    keys_second = case.model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]


def test_unsupported_reward_model_raises() -> None:
    """Constructing on an unsupported reward model is rejected.

    Purpose: Validates the scope guard on reward model type

    Given: An env configured with the distance-decayed hazard reward model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = LaserTagPOMDP(
        discount_factor=0.95,
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
    )
    with pytest.raises(NotImplementedError):
        LaserTagVectorizedModel(env)


@pytest.mark.parametrize(
    "policy",
    [OpponentPolicy.EVADE, OpponentPolicy.PURSUE, OpponentPolicy.EVADE_WHEN_SPOTTED],
    ids=["evade", "pursue", "evade-when-spotted"],
)
def test_every_opponent_policy_constructs(policy: OpponentPolicy) -> None:
    """All three opponent policies build a conforming vectorized model.

    Purpose: Validates that PURSUE and EVADE_WHEN_SPOTTED are supported
        alongside EVADE (the guard no longer rejects them)

    Given: An env configured with each supported opponent policy
    When: A vectorized model is constructed from it
    Then: Construction succeeds and yields a conforming generative model

    Test type: unit
    """
    env = LaserTagPOMDP(discount_factor=0.95, opponent_policy=policy)
    model = LaserTagVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    assert isinstance(model, VectorizedGenerativeModel)
    assert model.num_actions == 5


def test_unsupported_transition_error_raises() -> None:
    """Constructing on stochastic robot motion is rejected.

    Purpose: Validates the scope guard on transition_error_prob

    Given: An env configured with a positive transition_error_prob
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = LaserTagPOMDP(discount_factor=0.95, transition_error_prob=0.1)
    with pytest.raises(NotImplementedError):
        LaserTagVectorizedModel(env)


def test_hazard_terminal_config_raises() -> None:
    """The draw-coupled hazard-terminal config is rejected.

    Purpose: Validates the scope guard on the hazard-terminal absorbing slot

    Given: An env configured with is_dangerous_area_hit_terminal=True
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    with pytest.raises(NotImplementedError):
        LaserTagVectorizedModel(env)
