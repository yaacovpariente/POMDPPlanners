# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native Tiger POMDP env.

These tests pin :class:`TigerVectorizedModel` to the environment's native
scalar kernels so the two implementations cannot drift. Deterministic kernels
(reward, terminal, observation log-likelihood) are compared exactly in
float64; the stochastic kernels (transition and observation sampling) are
compared by empirical frequencies over a large batch, since the Tiger problem
is fully discrete.
"""

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.tiger_pomdp import (
    ACTIONS,
    OBSERVATIONS,
    STATES,
    TigerPOMDP,
)
from POMDPPlanners.environments.tiger_pomdp_vectorized_model import (
    TigerVectorizedModel,
)


@pytest.fixture(name="env")
def env_fixture() -> TigerPOMDP:
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture(name="model")
def model_fixture(env: TigerPOMDP) -> TigerVectorizedModel:
    return TigerVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)


def _coded(indices: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(indices.reshape(-1, 1), dtype=torch.int64)


def test_model_satisfies_protocol(model: TigerVectorizedModel) -> None:
    """The model conforms to the vectorized generative model protocol.

    Purpose: Validates structural protocol conformance and action count

    Given: A Tiger vectorized model on CPU
    When: It is checked against the runtime-checkable protocol
    Then: isinstance reports conformance and num_actions equals three

    Test type: unit
    """
    assert isinstance(model, VectorizedGenerativeModel)
    assert model.num_actions == len(ACTIONS)
    assert model.num_states == len(STATES)
    assert model.num_observations == len(OBSERVATIONS)


def test_reward_matches_native(env: TigerPOMDP, model: TigerVectorizedModel) -> None:
    """Rewards match the native scalar reward across all state/action pairs.

    Purpose: Validates the reward lookup table against env.reward

    Given: A random batch of integer-coded states and action indices
    When: The model reward is compared to env.reward per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(0)
    s_idx = rng.integers(0, len(STATES), size=256)
    a_idx = rng.integers(0, len(ACTIONS), size=256)
    expected = np.array([env.reward(STATES[s], ACTIONS[a]) for s, a in zip(s_idx, a_idx)])
    states = _coded(s_idx)
    actual = model.rewards(states, torch.as_tensor(a_idx), states).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_observation_log_probs_match_native(env: TigerPOMDP, model: TigerVectorizedModel) -> None:
    """Observation log-likelihoods match the native kernel exactly.

    Purpose: Validates the observation log-probability table over all combos

    Given: Every (action, next_state, observation) index combination
    When: The model log-prob is compared to env.observation_log_probability
    Then: The tensors are exactly equal (including -inf entries)

    Test type: unit
    """
    combos = [
        (a, ns, o)
        for a in range(len(ACTIONS))
        for ns in range(len(STATES))
        for o in range(len(OBSERVATIONS))
    ]
    a_idx = np.array([c[0] for c in combos])
    ns_idx = np.array([c[1] for c in combos])
    o_idx = np.array([c[2] for c in combos])
    expected = np.array(
        [
            env.observation_log_probability(STATES[ns], ACTIONS[a], [OBSERVATIONS[o]])[0]
            for a, ns, o in combos
        ]
    )
    actual = model.observation_log_probs(
        _coded(ns_idx), torch.as_tensor(a_idx), _coded(o_idx)
    ).numpy()
    assert np.array_equal(expected, actual)


def test_terminal_mask_matches_native(env: TigerPOMDP, model: TigerVectorizedModel) -> None:
    """Terminal flags match the native per-state terminal check.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: A random batch of integer-coded states
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees (all False for the Tiger problem)

    Test type: unit
    """
    rng = np.random.default_rng(1)
    s_idx = rng.integers(0, len(STATES), size=64)
    expected = np.array([env.is_terminal(STATES[s]) for s in s_idx])
    actual = model.terminal_mask(_coded(s_idx)).numpy()
    assert np.array_equal(expected, actual)


def test_transition_sampling_matches_native_frequencies(
    env: TigerPOMDP, model: TigerVectorizedModel
) -> None:
    """Sampled next-state frequencies match the native transition kernel.

    Purpose: Validates the stochastic transition for open and listen actions

    Given: A large batch in tiger_left sampled under open_left and under listen
    When: Empirical next-state frequencies are compared to the native kernel
    Then: The open reset frequency gap is below 0.02 and listen is a no-op

    Test type: unit
    """
    torch.manual_seed(1)
    n = 40000
    states = _coded(np.zeros(n, dtype=np.int64))  # tiger_left
    open_actions = torch.full((n,), ACTIONS.index("open_left"), dtype=torch.int64)
    twin = model.sample_next_states(states, open_actions).numpy().reshape(-1)
    native = env.sample_next_state_batch([STATES[0]] * n, "open_left")
    native_idx = np.array([STATES.index(s) for s in native])
    assert abs(twin.mean() - native_idx.mean()) < 0.02
    listen_actions = torch.full((n,), ACTIONS.index("listen"), dtype=torch.int64)
    kept = model.sample_next_states(states, listen_actions)
    assert torch.equal(kept, states)


def test_observation_sampling_matches_native_frequencies(
    env: TigerPOMDP, model: TigerVectorizedModel
) -> None:
    """Sampled observation frequencies match the native observation kernel.

    Purpose: Validates the stochastic observation model for listen and open

    Given: A large batch in tiger_left sampled under listen and under open_left
    When: Empirical observation frequencies are compared to the native kernel
    Then: The hear_left frequency gap is below 0.02 and open yields hear_nothing

    Test type: unit
    """
    torch.manual_seed(2)
    n = 40000
    next_states = _coded(np.zeros(n, dtype=np.int64))  # tiger_left
    listen_actions = torch.zeros(n, dtype=torch.int64)
    twin = model.sample_observations(next_states, listen_actions).numpy().reshape(-1)
    native = env.sample_observation(STATES[0], "listen", n_samples=n)
    native_freq = float(np.mean([o == "hear_left" for o in native]))
    twin_freq = float(np.mean(twin == OBSERVATIONS.index("hear_left")))
    assert abs(twin_freq - native_freq) < 0.02
    open_actions = torch.full((n,), ACTIONS.index("open_left"), dtype=torch.int64)
    silent = model.sample_observations(next_states, open_actions).numpy().reshape(-1)
    assert np.all(silent == OBSERVATIONS.index("hear_nothing"))


def test_method_shapes_and_dtypes(model: TigerVectorizedModel) -> None:
    """Every kernel returns the documented shape and dtype.

    Purpose: Validates output contracts of all vectorized methods

    Given: A small integer-coded batch of states, actions, and observations
    When: Each kernel is invoked on the batch
    Then: Shapes are [N,1] or [N] and dtypes are int64/float64/bool as specified

    Test type: unit
    """
    n = 5
    states = _coded(np.array([0, 1, 0, 1, 0]))
    actions = torch.as_tensor(np.array([0, 1, 2, 0, 1]))
    observations = _coded(np.array([0, 2, 2, 1, 2]))
    next_states = model.sample_next_states(states, actions)
    sampled_obs = model.sample_observations(next_states, actions)
    assert next_states.shape == (n, 1) and next_states.dtype == torch.int64
    assert sampled_obs.shape == (n, 1) and sampled_obs.dtype == torch.int64
    rewards = model.rewards(states, actions, next_states)
    assert rewards.shape == (n,) and rewards.dtype == torch.float64
    terminal = model.terminal_mask(states)
    assert terminal.shape == (n,) and terminal.dtype == torch.bool
    log_probs = model.observation_log_probs(next_states, actions, observations)
    assert log_probs.shape == (n,) and log_probs.dtype == torch.float64
    assert model.action_keys(actions).dtype == torch.int64
    assert model.observation_keys(observations).dtype == torch.int64


def test_action_and_observation_keys_are_deterministic(
    model: TigerVectorizedModel,
) -> None:
    """Action and observation keys are stable and discriminate distinct inputs.

    Purpose: Validates the discrete key mapping used by the belief tree

    Given: Action indices and integer-coded observations
    When: action_keys and observation_keys are called twice
    Then: Keys are stable, equal the coded index, and distinct inputs differ

    Test type: unit
    """
    actions = torch.as_tensor(np.array([0, 1, 2, 1]))
    observations = _coded(np.array([0, 1, 2, 1]))
    assert torch.equal(model.action_keys(actions), model.action_keys(actions))
    assert torch.equal(model.action_keys(actions), actions.to(torch.int64))
    obs_keys = model.observation_keys(observations)
    assert torch.equal(obs_keys, model.observation_keys(observations))
    assert obs_keys[0] != obs_keys[1]
    assert obs_keys[1] == obs_keys[3]


def test_unsupported_labels_raise(env: TigerPOMDP) -> None:
    """Constructing on a reconfigured label set is rejected.

    Purpose: Validates the scope guard on the state/action/observation labels

    Given: A Tiger environment whose state labels have been reconfigured
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env.states = ["den_left", "den_right"]
    with pytest.raises(NotImplementedError):
        TigerVectorizedModel(env)
