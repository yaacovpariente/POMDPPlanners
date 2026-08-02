# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native RockSample env.

These tests pin :class:`RockSampleVectorizedModel` to the environment's
native (C++) transition / observation kernels and its numpy reward kernel so
the two implementations cannot drift. The parity checks run over a *sweep of
supported configurations* (varying grid size, rock layout, sensor efficiency,
costs, and dangerous-area geometry) so a wiring bug that default parameters
would hide is caught. The RockSample transition is deterministic and the
reward is deterministic in every swept config (no dangerous areas, or a
dangerous area with hit-probability 1.0), so those are compared exactly in
float64; the noisy Bernoulli sensor and the sub-unit hit-probability hazard
are compared by empirical moments over a large batch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RewardModelType,
    RockSamplePOMDP,
)
from POMDPPlanners.environments.rock_sample_pomdp.rocksample_vectorized_model import (
    RockSampleVectorizedModel,
)

_OBS_STRINGS = ("none", "good", "bad")

# Each case is a dict of RockSamplePOMDP kwargs. All are *supported*
# configurations: constant-hazard reward, is_dangerous_area_hit_terminal=False.
# Dangerous-area cases keep hit-probability 1.0 so the reward stays
# deterministic and can be pinned exactly.
_SUPPORTED_CASES = [
    {"map_size": (5, 5), "rock_positions": [(0, 0), (2, 2), (3, 3)]},
    {
        "map_size": (7, 6),
        "rock_positions": [(0, 1), (3, 3), (5, 4), (6, 0)],
        "sensor_efficiency": 4.0,
        "step_penalty": -0.1,
        "sensor_use_penalty": -0.3,
        "good_rock_reward": 12.0,
        "bad_rock_penalty": -8.0,
        "exit_reward": 15.0,
    },
    {
        "map_size": (6, 6),
        "rock_positions": [(1, 1), (4, 4)],
        "sensor_efficiency": 6.0,
        "dangerous_areas": [(2, 2), (3, 3)],
        "dangerous_area_radius": 1.5,
        "dangerous_area_penalty": -4.0,
        "dangerous_area_hit_probability": 1.0,
    },
]
_CASE_IDS = ["default", "large-costs", "danger-deterministic"]


@dataclass
class _Case:
    env: RockSamplePOMDP
    model: RockSampleVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = RockSamplePOMDP(discount_factor=0.95, **request.param)
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _random_states(env: RockSamplePOMDP, count: int, seed: int) -> np.ndarray:
    """Random valid states: in-grid robot position + random rock qualities."""
    rng = np.random.default_rng(seed)
    num_rocks = len(env.rock_positions)
    rows = rng.integers(0, env.map_size[0], count)
    cols = rng.integers(0, env.map_size[1], count)
    rocks = rng.integers(0, 2, (count, num_rocks)).astype(np.float64)
    return np.concatenate([rows[:, None], cols[:, None], rocks], axis=1).astype(np.float64)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _action_col(action: int, count: int) -> torch.Tensor:
    return torch.full((count,), action, dtype=torch.int64)


def test_model_satisfies_protocol_for_every_supported_config(case: _Case) -> None:
    """Every supported config yields a conforming VectorizedGenerativeModel.

    Purpose: Validates structural protocol conformance across configs

    Given: A model built from each supported environment configuration
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and num_actions equals 5 + num_rocks

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == 5 + len(case.env.rock_positions)


def test_transition_matches_native_exactly(case: _Case) -> None:
    """Sampled next states match the native transition kernel exactly.

    Purpose: Validates the deterministic grid / sample transition across actions

    Given: A random batch of valid states, transitioned under each action
    When: model.sample_next_states is compared to env.sample_next_state_batch
    Then: The two next-state arrays are identical for every action

    Test type: unit
    """
    states = _random_states(case.env, 256, seed=1)
    for action in range(case.model.num_actions):
        native = case.env.sample_next_state_batch(states, action)
        twin = case.model.sample_next_states(
            _tensor(states), _action_col(action, states.shape[0])
        ).numpy()
        assert np.array_equal(native, twin)


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the noisy-sensor likelihood kernel across actions / obs

    Given: Post-transition states and every (action, observation-code) pair
    When: model.observation_log_probs is compared to the env per-state kernel
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    states = _random_states(case.env, 256, seed=2)
    max_diff = 0.0
    for action in range(case.model.num_actions):
        next_states = case.env.sample_next_state_batch(states, action)
        for code, obs_str in enumerate(_OBS_STRINGS):
            expected = case.env.observation_log_probability_per_state(next_states, action, obs_str)
            actual = case.model.observation_log_probs(
                _tensor(next_states),
                _action_col(action, states.shape[0]),
                torch.full((states.shape[0], 1), float(code), dtype=torch.float64),
            ).numpy()
            max_diff = max(max_diff, float(np.max(np.abs(expected - actual))))
    assert max_diff < 1e-9


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Deterministic rewards match the native numpy reward kernel exactly.

    Purpose: Validates the base + deterministic-hazard reward across actions

    Given: A random batch of states with their realised next states per action
    When: model.rewards is compared to env.reward_model.compute_reward_batch
    Then: The two reward arrays are identical for every action

    Test type: unit
    """
    states = _random_states(case.env, 400, seed=3)
    for action in range(case.model.num_actions):
        next_states = case.env.sample_next_state_batch(states, action)
        expected = case.env.reward_model.compute_reward_batch(
            states, action, next_states=next_states
        )
        actual = case.model.rewards(
            _tensor(states), _action_col(action, states.shape[0]), _tensor(next_states)
        ).numpy()
        assert np.array_equal(expected, actual)


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: The exit sentinel row and several live in-grid rows
    When: model.terminal_mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    num_rocks = len(case.env.rock_positions)
    sentinel = np.concatenate([[-1.0, -1.0], np.ones(num_rocks)])
    live = _random_states(case.env, 8, seed=4)
    states = np.concatenate([sentinel[None, :], live], axis=0)
    expected = np.array([case.env.is_terminal(row) for row in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_observation_sampling_reproduces_sensor_efficiency() -> None:
    """Check-action sampling reproduces the exp(-dist/eff) sensor accuracy.

    Purpose: Validates the stochastic Bernoulli sensor in expectation

    Given: A robot one cell from a good rock, checked many times
    When: The empirical fraction of correct ("good") observations is measured
    Then: It matches exp(-distance / sensor_efficiency) within 0.02

    Test type: unit
    """
    torch.manual_seed(0)
    env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(2, 2)], sensor_efficiency=3.0)
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    next_state = np.array([2.0, 1.0, 1.0])  # one cell west of the good rock
    batch = np.tile(next_state, (40000, 1))
    observations = model.sample_observations(_tensor(batch), _action_col(5, 40000))
    frac_good = float((observations[:, 0] == 1).to(torch.float64).mean())
    expected = float(np.exp(-1.0 / 3.0))
    assert abs(frac_good - expected) < 0.02


def test_reward_hazard_expectation_matches_native() -> None:
    """Mean hazard reward with sub-unit hit-probability matches the env.

    Purpose: Validates the stochastic dangerous-area contribution in expectation

    Given: Next states sitting inside a dangerous area with hit-probability 0.4
    When: Mean rewards from the model and the numpy reward kernel are compared
    Then: The means agree within a Monte Carlo tolerance of 0.1

    Test type: unit
    """
    np.random.seed(7)
    torch.manual_seed(7)
    env = RockSamplePOMDP(
        map_size=(6, 6),
        rock_positions=[(0, 0)],
        dangerous_areas=[(3, 3)],
        dangerous_area_radius=1.0,
        dangerous_area_penalty=-5.0,
        dangerous_area_hit_probability=0.4,
    )
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    states = np.tile(np.array([3.0, 2.0, 1.0]), (20000, 1))
    next_states = np.tile(np.array([3.0, 3.0, 1.0]), (20000, 1))  # in-zone, action west
    action = 4
    env_mean = float(
        np.mean(env.reward_model.compute_reward_batch(states, action, next_states=next_states))
    )
    model_mean = float(
        model.rewards(_tensor(states), _action_col(action, 20000), _tensor(next_states)).mean()
    )
    assert abs(env_mean - model_mean) < 0.1


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every generative method returns the documented shape and dtype.

    Purpose: Validates the tensor contract of the protocol methods

    Given: A small batch of states, actions, next states, and observations
    When: Each vectorized method is invoked
    Then: Shapes are [N, ds] / [N, 1] / [N] and key dtypes are int64

    Test type: unit
    """
    states = _tensor(_random_states(case.env, 16, seed=5))
    actions = _action_col(5, 16)
    next_states = case.model.sample_next_states(states, actions)
    observations = case.model.sample_observations(next_states, actions)
    rewards = case.model.rewards(states, actions, next_states)
    log_probs = case.model.observation_log_probs(next_states, actions, observations)
    assert next_states.shape == states.shape
    assert observations.shape == (16, 1)
    assert rewards.shape == (16,)
    assert log_probs.shape == (16,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.action_keys(actions).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_action_and_observation_keys_are_deterministic() -> None:
    """Action and observation keys are stable and discriminating.

    Purpose: Validates the discrete key mappings used by the belief tree

    Given: A set of actions and categorical observation codes
    When: action_keys / observation_keys are each called twice
    Then: Repeated calls agree, keys equal the underlying integer codes, and
        distinct codes map to distinct keys

    Test type: unit
    """
    env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(0, 0), (2, 2)])
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    actions = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
    observations = torch.tensor([[0.0], [1.0], [2.0], [1.0]], dtype=torch.float64)
    assert torch.equal(model.action_keys(actions), model.action_keys(actions))
    assert torch.equal(model.action_keys(actions), actions)
    obs_keys = model.observation_keys(observations)
    assert torch.equal(obs_keys, model.observation_keys(observations))
    assert torch.equal(obs_keys, torch.tensor([0, 1, 2, 1], dtype=torch.int64))
    assert obs_keys[0] != obs_keys[1]


@pytest.mark.parametrize(
    "reward_model_type",
    [RewardModelType.ZERO_MEAN_HAZARD_SHOCK, RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY],
)
def test_unsupported_reward_model_raises(reward_model_type: RewardModelType) -> None:
    """Constructing on an unsupported reward model is rejected.

    Purpose: Validates the scope guard on reward model type

    Given: An env configured with a non-constant-hazard reward model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0), (2, 2)],
        dangerous_areas=[(1, 1)],
        reward_model_type=reward_model_type,
    )
    with pytest.raises(NotImplementedError):
        RockSampleVectorizedModel(env)


def test_hazard_terminal_config_raises() -> None:
    """The draw-coupled hazard-terminal config is rejected.

    Purpose: Validates the scope guard on the hazard-terminal absorbing slot

    Given: An env with is_dangerous_area_hit_terminal=True
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0), (2, 2)],
        dangerous_areas=[(1, 1)],
        is_dangerous_area_hit_terminal=True,
    )
    with pytest.raises(NotImplementedError):
        RockSampleVectorizedModel(env)
