# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the native PacMan environment.

These tests pin :class:`PacManVectorizedModel` to the environment's native
(C++) kernels so the two implementations cannot drift. The parity checks run
over a sweep of supported configurations (varying maze walls, ghost count,
aggressiveness, and dangerous-area geometry) so a wiring bug that default
parameters would hide is caught. Rewards and observation log-likelihoods are
deterministic given (state, next_state, observation) and compared exactly in
float64; the deterministic PacMan fields of the stochastic transition are
compared exactly, while the stochastic ghost move and observation sampler are
compared by empirical frequencies over a large batch.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_vectorized_model import (
    PacManVectorizedModel,
)

_SUPPORTED_CASES = [
    {},
    {
        "maze_size": (6, 6),
        "walls": {(2, 2), (3, 3), (1, 4)},
        "num_ghosts": 2,
        "initial_ghost_positions": [(5, 5), (0, 5)],
        "initial_pellets": [(0, 0), (5, 0), (2, 5)],
        "initial_pacman_pos": (3, 0),
        "ghost_aggressiveness": 1.5,
    },
    {
        "dangerous_areas": {(2, 1), (4, 4)},
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": 7.0,
    },
]
_CASE_IDS = ["default", "walls-two-ghosts", "dangerous-areas"]


@dataclass
class _Case:
    env: PacManPOMDP
    model: PacManVectorizedModel


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    env = PacManPOMDP(**request.param)
    model = PacManVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return _Case(env=env, model=model)


def _state_dim(env: PacManPOMDP) -> int:
    return len(env.initial_state_dist().sample()[0])


def _valid_cells(env: PacManPOMDP) -> List[Tuple[int, int]]:
    rows, cols = env.maze_size
    return [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in env.walls]


def _random_states(env: PacManPOMDP, count: int, rng: np.random.Generator) -> np.ndarray:
    cells = _valid_cells(env)
    states = np.zeros((count, _state_dim(env)), dtype=np.float64)
    for i in range(count):
        pacman = cells[rng.integers(len(cells))]
        ghosts = tuple(cells[rng.integers(len(cells))] for _ in range(env.num_ghosts))
        states[i] = env.make_state(
            pacman_pos=pacman,
            ghost_positions=ghosts,
            pellets=None,
            score=float(rng.integers(0, 40)),
            terminal=False,
        )
    return states


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def _cell_frequencies(positions: np.ndarray) -> dict:
    unique, counts = np.unique(positions.astype(int), axis=0, return_counts=True)
    return {tuple(cell): count / positions.shape[0] for cell, count in zip(unique, counts)}


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
    """Rewards match the native reward kernel to float64 precision.

    Purpose: Validates the deterministic reward path across configs

    Given: Random non-terminal states and their realised native next states
    When: The model reward is compared to env.reward_batch on those pairs
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(5)
    states = _random_states(case.env, 600, rng)
    next_states = case.env.sample_next_state_batch(states, 1)
    expected = case.env.reward_batch(states, 1, next_states=next_states)
    actions = torch.ones(states.shape[0], dtype=torch.int64)
    actual = case.model.rewards(_tensor(states), actions, _tensor(next_states)).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_observation_log_probs_match_native_exactly(case: _Case) -> None:
    """Observation log-likelihoods match the native kernel to float64 precision.

    Purpose: Validates the observation likelihood kernel across configs

    Given: Realised next states and observations drawn from the native sampler
    When: The model's observation_log_probs is compared to the env per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(7)
    next_states = _random_states(case.env, 400, rng)
    observations = np.stack(
        [
            case.env.observation_to_array(case.env.sample_observation(next_states[i], 1))
            for i in range(next_states.shape[0])
        ]
    ).astype(np.float64)
    expected = np.array(
        [
            case.env.observation_log_probability_per_state(
                next_states[i : i + 1], 1, observations[i]
            )[0]
            for i in range(next_states.shape[0])
        ]
    )
    actions = torch.ones(next_states.shape[0], dtype=torch.int64)
    actual = case.model.observation_log_probs(
        _tensor(next_states), actions, _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < 1e-9


def test_transition_pacman_fields_are_deterministic_and_match_native(case: _Case) -> None:
    """Deterministic transition fields match the native kernel exactly.

    Purpose: Validates PacMan move, pellet collection, score, and terminal

    Given: States with all ghosts far from PacMan so no collision can occur
    When: Both kernels transition the batch under the same action
    Then: PacMan position, pellet mask, score, and terminal flag match exactly

    Test type: unit
    """
    cells = _valid_cells(case.env)
    far_ghost = max(cells, key=lambda cell: abs(cell[0]) + abs(cell[1]))
    pacman = min(cells, key=lambda cell: cell[0] + cell[1])
    ghosts = tuple([far_ghost] * case.env.num_ghosts)
    state = case.env.make_state(
        pacman_pos=pacman, ghost_positions=ghosts, pellets=None, score=0.0, terminal=False
    )
    batch = np.tile(state, (200, 1))
    native = case.env.sample_next_state_batch(batch, 1)
    twin = case.model.sample_next_states(_tensor(batch), torch.ones(200, dtype=torch.int64)).numpy()
    trans = case.env.get_transition_cpp_ctor_kwargs()
    pac_cols = [trans["idx_pac_row"], trans["idx_pac_col"]]
    pellet_slice = slice(trans["idx_pellets_start"], trans["idx_pellets_end"])
    assert np.array_equal(native[:, pac_cols], twin[:, pac_cols])
    assert np.array_equal(native[:, pellet_slice], twin[:, pellet_slice])
    assert np.array_equal(native[:, trans["idx_score"]], twin[:, trans["idx_score"]])
    assert np.array_equal(native[:, trans["idx_terminal"]], twin[:, trans["idx_terminal"]])


def test_ghost_move_distribution_matches_native(case: _Case) -> None:
    """Sampled ghost moves reproduce the native softmax move distribution.

    Purpose: Validates the aggressive-ghost softmax transition kernel

    Given: A fixed non-terminal state sampled many times under the stay action
    When: Empirical next-position frequencies of ghost 0 are compared
    Then: Every cell frequency agrees within a Monte Carlo tolerance of 0.02

    Test type: unit
    """
    torch.manual_seed(1)
    cells = _valid_cells(case.env)
    pacman = cells[0]
    ghosts = tuple(cells[min(3 + i, len(cells) - 1)] for i in range(case.env.num_ghosts))
    state = case.env.make_state(
        pacman_pos=pacman, ghost_positions=ghosts, pellets=None, score=0.0, terminal=False
    )
    batch = np.tile(state, (40000, 1))
    native = case.env.sample_next_state_batch(batch, 4)
    twin = case.model.sample_next_states(
        _tensor(batch), torch.full((40000,), 4, dtype=torch.int64)
    ).numpy()
    trans = case.env.get_transition_cpp_ctor_kwargs()
    start = trans["idx_ghosts_start"]
    native_freq = _cell_frequencies(native[:, start : start + 2])
    twin_freq = _cell_frequencies(twin[:, start : start + 2])
    keys = set(native_freq) | set(twin_freq)
    assert max(abs(native_freq.get(k, 0.0) - twin_freq.get(k, 0.0)) for k in keys) < 0.02


def test_observation_sampling_matches_native(case: _Case) -> None:
    """Sampled observations reproduce the native round-and-clamp distribution.

    Purpose: Validates the per-ghost Gaussian observation sampler

    Given: A fixed next state sampled many times by both observation kernels
    When: Empirical frequencies of ghost 0's observed cell are compared
    Then: Every cell frequency agrees within a Monte Carlo tolerance of 0.02

    Test type: unit
    """
    torch.manual_seed(2)
    cells = _valid_cells(case.env)
    pacman = cells[0]
    ghosts = tuple(cells[len(cells) // 2] for _ in range(case.env.num_ghosts))
    next_state = case.env.make_state(
        pacman_pos=pacman, ghost_positions=ghosts, pellets=None, score=0.0, terminal=False
    )
    native = np.stack(
        [
            case.env.observation_to_array(case.env.sample_observation(next_state, 0))
            for _ in range(20000)
        ]
    )
    batch = np.tile(next_state, (20000, 1))
    twin = case.model.sample_observations(
        _tensor(batch), torch.zeros(20000, dtype=torch.int64)
    ).numpy()
    native_freq = _cell_frequencies(native[:, 0:2])
    twin_freq = _cell_frequencies(twin[:, 0:2])
    keys = set(native_freq) | set(twin_freq)
    assert max(abs(native_freq.get(k, 0.0) - twin_freq.get(k, 0.0)) for k in keys) < 0.02


def test_terminal_mask_matches_native(case: _Case) -> None:
    """Terminal flags match the native per-state terminal check across configs.

    Purpose: Validates the batched terminal mask against env.is_terminal

    Given: A batch mixing terminal and non-terminal states
    When: The model terminal mask is compared to env.is_terminal per row
    Then: Every entry agrees

    Test type: unit
    """
    cells = _valid_cells(case.env)
    ghosts = tuple([cells[-1]] * case.env.num_ghosts)
    non_terminal = case.env.make_state(
        pacman_pos=cells[0], ghost_positions=ghosts, pellets=None, score=0.0, terminal=False
    )
    terminal = case.env.make_state(
        pacman_pos=cells[0], ghost_positions=ghosts, pellets=None, score=0.0, terminal=True
    )
    states = np.stack([non_terminal, terminal, non_terminal])
    expected = np.array([case.env.is_terminal(s) for s in states])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)


def test_terminal_state_produces_sentinel_observation(case: _Case) -> None:
    """A terminal next state yields the all-minus-one sentinel observation.

    Purpose: Validates the terminal-state branch of the observation sampler

    Given: A terminal next state
    When: The model samples an observation for it
    Then: Every observation coordinate is -1

    Test type: unit
    """
    cells = _valid_cells(case.env)
    ghosts = tuple([cells[-1]] * case.env.num_ghosts)
    terminal = case.env.make_state(
        pacman_pos=cells[0], ghost_positions=ghosts, pellets=None, score=0.0, terminal=True
    )
    observation = case.model.sample_observations(
        _tensor(terminal).unsqueeze(0), torch.zeros(1, dtype=torch.int64)
    )
    assert torch.all(observation == -1.0)


def test_action_keys_are_identity(case: _Case) -> None:
    """Action keys pass discrete action indices through as int64 keys.

    Purpose: Validates the discrete action-to-key mapping

    Given: The five discrete action indices
    When: action_keys is called
    Then: The output is the identical int64 indices

    Test type: unit
    """
    actions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    keys = case.model.action_keys(actions)
    assert keys.dtype == torch.int64
    assert torch.equal(keys, actions)


def test_observation_keys_are_deterministic_and_discriminating(case: _Case) -> None:
    """Observation keys are stable per input and separate distinct observations.

    Purpose: Validates the discrete-observation to integer-key hashing

    Given: Observations, some identical and some at different ghost positions
    When: observation_keys is called twice
    Then: Identical inputs map to identical keys and distinct ones differ

    Test type: unit
    """
    width = 2 * case.env.num_ghosts
    obs_a = torch.zeros(width, dtype=torch.float64)
    obs_b = torch.zeros(width, dtype=torch.float64)
    obs_b[0] = 3.0
    observations = torch.stack([obs_a, obs_a, obs_b])
    keys_first = case.model.observation_keys(observations)
    keys_second = case.model.observation_keys(observations)
    assert torch.equal(keys_first, keys_second)
    assert keys_first[0] == keys_first[1]
    assert keys_first[0] != keys_first[2]


def test_unsupported_ghost_coordination_raises() -> None:
    """Constructing on a non-independent coordination mode is rejected.

    Purpose: Validates the scope guard on ghost coordination

    Given: An env configured with coordinated ghost coordination
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = PacManPOMDP(ghost_coordination="coordinated")
    with pytest.raises(NotImplementedError):
        PacManVectorizedModel(env)


def test_unsupported_ghost_strategy_raises() -> None:
    """Constructing on a non-aggressive ghost strategy is rejected.

    Purpose: Validates the scope guard on ghost strategy

    Given: An env configured with the patrol ghost strategy
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = PacManPOMDP(num_ghosts=1, ghost_strategies=["patrol"])
    with pytest.raises(NotImplementedError):
        PacManVectorizedModel(env)


def test_unsupported_reward_model_raises() -> None:
    """Constructing on an unsupported reward model is rejected.

    Purpose: Validates the scope guard on reward model type

    Given: An env configured with the zero-mean hazard shock reward model
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = PacManPOMDP(reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
    with pytest.raises(NotImplementedError):
        PacManVectorizedModel(env)


def test_hazard_terminal_config_raises() -> None:
    """The draw-coupled hazard-terminal config is rejected.

    Purpose: Validates the scope guard on the hazard-terminal absorbing slot

    Given: An env configured with is_dangerous_area_hit_terminal=True
    When: A vectorized model is constructed from it
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = PacManPOMDP(dangerous_areas={(2, 2)}, is_dangerous_area_hit_terminal=True)
    with pytest.raises(NotImplementedError):
        PacManVectorizedModel(env)
