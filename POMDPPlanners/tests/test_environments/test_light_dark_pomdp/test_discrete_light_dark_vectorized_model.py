# SPDX-License-Identifier: MIT

"""Parity tests: torch vectorized model vs. the scalar Discrete Light-Dark env.

The registry conformance suite checks the model on the pinned configuration
only. These tests sweep configurations that move the goal, obstacles, beacons,
error probabilities and grid size, so a constant read from the wrong env
attribute shows up. Deterministic kernels (observation log-likelihood, terminal
mask, reward off obstacles) are compared exactly over every cell of a grid
padded by one cell outside it; stochastic ones by frequencies over a large
batch.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_vectorized_model import (
    DiscreteLightDarkVectorizedModel,
)

_CASES: List[Dict[str, Any]] = [
    {},
    {
        "transition_error_prob": 0.3,
        "observation_error_prob": 0.4,
        "goal_state": np.array([2, 7]),
        "obstacles": [(1, 1), (4, 4), (6, 2)],
        "obstacle_hit_probability": 0.6,
        "obstacle_reward": -7.0,
        "goal_reward": 20.0,
        "fuel_cost": 0.5,
        "grid_size": 8,
        "beacons": [(0, 0), (4, 6)],
        "beacon_radius": 2.0,
    },
    {"obstacles": [], "transition_error_prob": 0.0, "observation_error_prob": 0.0},
]
_IDS = ["defaults", "moved_layout", "no_obstacles_noise_free"]


def _build(case: Dict[str, Any]):
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, **case)
    model = DiscreteLightDarkVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    return env, model


def _cells(env: DiscreteLightDarkPOMDP) -> np.ndarray:
    span = np.arange(-1, env.grid_size + 2, dtype=np.float64)
    return np.array([[x, y] for x in span for y in span])


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_observation_log_probs_match_env_exactly(case: Dict[str, Any]) -> None:
    """Test that every candidate observation scores as the scalar env scores it.

    Purpose: Pins the near/far table choice and the candidate ordering.
    Given: Every cell of the padded grid, each paired with its five candidate
        observations and one observation two cells away.
    When: Both implementations score the pairs.
    Then: The log-likelihoods agree exactly, including -inf.

    Test type: unit
    """
    env, model = _build(case)
    offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0), (2, 0)]
    states, observations, expected = [], [], []
    for cell in _cells(env):
        for offset in offsets:
            observation = cell + np.asarray(offset, dtype=np.float64)
            states.append(cell)
            observations.append(observation)
            expected.append(env.observation_log_probability(cell, "up", observation[None, :])[0])
    actual = model.observation_log_probs(
        torch.tensor(np.array(states)),
        torch.zeros(len(states), dtype=torch.int64),
        torch.tensor(np.array(observations)),
    )
    np.testing.assert_array_equal(actual.numpy(), np.array(expected))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_terminal_mask_matches_env(case: Dict[str, Any]) -> None:
    """Test that terminal_mask agrees with is_terminal on every cell.

    Purpose: The model must stop planning where the env ends an episode.
    Given: Every cell of the padded grid.
    When: Both implementations test the cells.
    Then: The flags agree.

    Test type: unit
    """
    env, model = _build(case)
    cells = _cells(env)
    expected = np.array([env.is_terminal(cell) for cell in cells])
    np.testing.assert_array_equal(model.terminal_mask(torch.tensor(cells)).numpy(), expected)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_reward_matches_env_off_obstacles(case: Dict[str, Any]) -> None:
    """Test that the reward agrees exactly wherever it draws no hit.

    Purpose: Pins the fuel cost, goal distance, goal bonus and out-of-grid
        penalty.
    Given: Every padded-grid cell that is not an obstacle, as a next state.
    When: Both implementations score the step.
    Then: The rewards agree to float64 rounding.

    Test type: unit
    """
    env, model = _build(case)
    obstacles = {tuple(map(float, column)) for column in np.asarray(env.obstacles).T}
    cells = np.array([cell for cell in _cells(env) if tuple(cell) not in obstacles])
    expected = np.array([env.reward(cell, "up", next_state=cell) for cell in cells])
    tensor = torch.tensor(cells)
    actual = model.rewards(tensor, torch.zeros(len(cells), dtype=torch.int64), tensor)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)


def test_obstacle_hit_frequency_matches_hit_probability() -> None:
    """Test that the obstacle penalty fires at obstacle_hit_probability.

    Purpose: The hit is the reward's only random part.
    Given: The moved-layout config and 40000 steps onto an obstacle cell.
    When: The model scores them.
    Then: The penalty fires in 0.6 of them, within 0.01.

    Test type: unit
    """
    env, model = _build(_CASES[1])
    torch.manual_seed(0)
    n = 40_000
    cell = torch.tensor([[4.0, 4.0]]).double().expand(n, 2)
    rewards = model.rewards(cell, torch.zeros(n, dtype=torch.int64), cell)
    base = -env.fuel_cost - float(np.linalg.norm(np.array([4.0, 4.0]) - env.goal_state))
    hit_rate = float((rewards < base - 1e-9).double().mean())
    assert abs(hit_rate - env.obstacle_hit_probability) < 0.01


@pytest.mark.parametrize("case", _CASES[:2], ids=_IDS[:2])
def test_sampling_frequencies_match_env_probabilities(case: Dict[str, Any]) -> None:
    """Test that sampled moves and observations follow the env's probabilities.

    Purpose: Pins the inverse-CDF draw against the env's probability tables.
    Given: 40000 rows per action from a cell next to a beacon and one far
        from every beacon.
    When: The model samples next states, then observations at the start cell.
    Then: Each outcome's frequency is within 0.01 of exp(env log-probability).

    Test type: unit
    """
    env, model = _build(case)
    torch.manual_seed(1)
    n = 40_000
    near = np.asarray(env.beacons, dtype=np.float64)[:, 0]
    for start in (near, np.array([3.0, 3.0])):
        rows = torch.tensor(start).expand(n, 2)
        for action_index, action in enumerate(env.get_actions()):
            actions = torch.full((n,), action_index, dtype=torch.int64)
            moves, move_counts = np.unique(
                model.sample_next_states(rows, actions).numpy(), axis=0, return_counts=True
            )
            readings, reading_counts = np.unique(
                model.sample_observations(rows, actions).numpy(), axis=0, return_counts=True
            )
            for counts, expected in (
                (move_counts, np.exp(env.transition_log_probability(start, action, moves))),
                (reading_counts, np.exp(env.observation_log_probability(start, action, readings))),
            ):
                np.testing.assert_allclose(counts / n, expected, atol=0.01)
                assert abs(expected.sum() - 1.0) < 0.01, "a likely outcome was never drawn"


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_obstacle_hit_terminal": True},
        {"observation_model_type": ObservationModelType.NO_OBS_IN_DARK},
        {"observation_model_type": ObservationModelType.DISTANCE_BASED},
    ],
    ids=["hazard_terminal", "no_obs_in_dark", "distance_based"],
)
def test_unsupported_configs_are_declined(overrides: Dict[str, Any]) -> None:
    """Test that the model refuses configurations it does not implement.

    Purpose: A config the model silently got wrong would plan on wrong dynamics.
    Given: An env with the hazard-terminal slot or a non-NORMAL observation model.
    When: The model is built.
    Then: It raises NotImplementedError.

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, **overrides)
    with pytest.raises(NotImplementedError):
        DiscreteLightDarkVectorizedModel(env)
