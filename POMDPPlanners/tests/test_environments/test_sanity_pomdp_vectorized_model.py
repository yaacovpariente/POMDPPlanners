# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native Sanity POMDP env.

These tests pin :class:`SanityVectorizedModel` to the environment's native
scalar kernels so the two implementations cannot drift. The Sanity POMDP is
deterministic and perfectly observable, so every kernel (transition,
observation, reward, terminal, observation likelihood) is compared exactly
over random batches of the two states and two actions.
"""

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.sanity_pomdp_vectorized_model import (
    SanityVectorizedModel,
)


@pytest.fixture(name="env")
def env_fixture() -> SanityPOMDP:
    return SanityPOMDP(discount_factor=0.95)


@pytest.fixture(name="model")
def model_fixture(env: SanityPOMDP) -> SanityVectorizedModel:
    return SanityVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)


def _random_states(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(0, 2, size=count).astype(np.float64)


def _random_actions(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(0, 2, size=count)


def test_model_satisfies_protocol(model: SanityVectorizedModel) -> None:
    """The model conforms to the VectorizedGenerativeModel protocol.

    Purpose: Validates structural protocol conformance and the action count

    Given: A vectorized model built from the Sanity POMDP
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and num_actions is 2

    Test type: unit
    """
    assert isinstance(model, VectorizedGenerativeModel)
    assert model.num_actions == 2


def test_next_states_match_native(env: SanityPOMDP, model: SanityVectorizedModel) -> None:
    """Sampled next states match the native transition kernel exactly.

    Purpose: Validates the deterministic transition against env.sample_next_state

    Given: A random batch of states and actions in the two-state/two-action set
    When: The batched next states are compared to the env per row
    Then: Every entry agrees and the shape is [N, 1]

    Test type: unit
    """
    rng = np.random.default_rng(0)
    states = _random_states(rng, 128)
    actions = _random_actions(rng, 128)
    expected = np.array(
        [env.sample_next_state(int(states[i]), int(actions[i])) for i in range(states.shape[0])]
    )
    actual = model.sample_next_states(
        torch.as_tensor(states, dtype=torch.float64).unsqueeze(-1),
        torch.as_tensor(actions, dtype=torch.int64),
    )
    assert tuple(actual.shape) == (128, 1)
    assert np.array_equal(actual.squeeze(-1).numpy(), expected)


def test_observations_match_native(env: SanityPOMDP, model: SanityVectorizedModel) -> None:
    """Sampled observations match the native perfect-observation kernel exactly.

    Purpose: Validates the observation kernel against env.sample_observation

    Given: A random batch of next states and actions
    When: The batched observations are compared to the env per row
    Then: Every entry equals the next state and the shape is [N, 1]

    Test type: unit
    """
    rng = np.random.default_rng(1)
    next_states = _random_states(rng, 128)
    actions = _random_actions(rng, 128)
    expected = np.array(
        [
            env.sample_observation(int(next_states[i]), int(actions[i]))
            for i in range(next_states.shape[0])
        ]
    )
    actual = model.sample_observations(
        torch.as_tensor(next_states, dtype=torch.float64).unsqueeze(-1),
        torch.as_tensor(actions, dtype=torch.int64),
    )
    assert tuple(actual.shape) == (128, 1)
    assert np.array_equal(actual.squeeze(-1).numpy(), expected)


def test_rewards_match_native(env: SanityPOMDP, model: SanityVectorizedModel) -> None:
    """Rewards match the native reward kernel exactly across the batch.

    Purpose: Validates the reward kernel against env.reward

    Given: A random batch of states, actions, and next states
    When: The batched rewards are compared to the env per row
    Then: Every entry agrees and the shape is [N]

    Test type: unit
    """
    rng = np.random.default_rng(2)
    states = _random_states(rng, 128)
    actions = _random_actions(rng, 128)
    next_states = _random_states(rng, 128)
    expected = np.array(
        [env.reward(int(states[i]), int(actions[i])) for i in range(states.shape[0])]
    )
    actual = model.rewards(
        torch.as_tensor(states, dtype=torch.float64).unsqueeze(-1),
        torch.as_tensor(actions, dtype=torch.int64),
        torch.as_tensor(next_states, dtype=torch.float64).unsqueeze(-1),
    )
    assert tuple(actual.shape) == (128,)
    assert np.array_equal(actual.numpy(), expected)


def test_terminal_mask_matches_native(env: SanityPOMDP, model: SanityVectorizedModel) -> None:
    """Terminal flags match the native terminal check (always False).

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: States covering both state values
    When: The model terminal mask is compared to the env per row
    Then: Every entry agrees and the dtype is bool

    Test type: unit
    """
    states = np.array([0.0, 1.0, 0.0, 1.0])
    expected = np.array([env.is_terminal(int(s)) for s in states])
    actual = model.terminal_mask(torch.as_tensor(states, dtype=torch.float64).unsqueeze(-1))
    assert actual.dtype == torch.bool
    assert np.array_equal(actual.numpy(), expected)


def test_observation_log_probs_match_native(env: SanityPOMDP, model: SanityVectorizedModel) -> None:
    """Observation log-likelihoods match the native kernel exactly.

    Purpose: Validates observation_log_probs against env.observation_log_probability

    Given: Random next states and observations, some matching and some not
    When: The batched log-likelihoods are compared to the env per row
    Then: Every entry agrees (0.0 on a match, -inf otherwise)

    Test type: unit
    """
    rng = np.random.default_rng(3)
    next_states = _random_states(rng, 128)
    observations = _random_states(rng, 128)
    actions = _random_actions(rng, 128)
    expected = np.array(
        [
            env.observation_log_probability(
                int(next_states[i]), int(actions[i]), [int(observations[i])]
            )[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = model.observation_log_probs(
        torch.as_tensor(next_states, dtype=torch.float64).unsqueeze(-1),
        torch.as_tensor(actions, dtype=torch.int64),
        torch.as_tensor(observations, dtype=torch.float64).unsqueeze(-1),
    ).numpy()
    assert np.array_equal(actual, expected)


def test_action_keys_are_identity_int64(model: SanityVectorizedModel) -> None:
    """Action keys are the action indices as int64.

    Purpose: Validates the discrete action-to-key mapping

    Given: A batch of action indices
    When: action_keys is called
    Then: The keys equal the input indices with int64 dtype

    Test type: unit
    """
    actions = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    keys = model.action_keys(actions)
    assert keys.dtype == torch.int64
    assert torch.equal(keys, actions)


def test_observation_keys_are_deterministic_and_discriminating(
    model: SanityVectorizedModel,
) -> None:
    """Observation keys are stable per input and separate distinct states.

    Purpose: Validates the discrete-observation to integer-key mapping

    Given: Observations covering both state values, some repeated
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct states differ

    Test type: unit
    """
    observations = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    keys_first = model.observation_keys(observations)
    keys_second = model.observation_keys(observations)
    assert keys_first.dtype == torch.int64
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]


def test_unsupported_action_set_raises() -> None:
    """Constructing on a non-standard action set is rejected.

    Purpose: Validates the scope guard on the action set

    Given: A Sanity POMDP subclass exposing a three-action set
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """

    class _ThreeActionSanity(SanityPOMDP):
        def get_actions(self) -> list[int]:
            return [0, 1, 2]

    with pytest.raises(NotImplementedError):
        SanityVectorizedModel(_ThreeActionSanity(discount_factor=0.95))
