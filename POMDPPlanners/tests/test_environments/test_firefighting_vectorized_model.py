# SPDX-License-Identifier: MIT

"""FirefightingVectorizedModel agrees with FirefightingPOMDP on hand-built fire fronts.

The cross-environment suite in ``test_vectorized_model_conformance.py`` checks
this model from random rollouts of six steps. Those rollouts start from one
burning cell far from the firefighters, so they seldom spray a fire, refill at
the depot, walk into a burning cell, or have two hoses cover one cell. The
states below put each of those events on the first step, so the suppression,
tank, spread, heat-damage and reward kernels are compared where they do
something.

Each check compares, per state-vector entry, the frequency of every value over
scalar draws and model draws, and compares rewards and observation
log-likelihoods exactly on the scalar draws.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.firefighting_pomdp import (
    FireCategory,
    FirefightingPOMDP,
    WindDirection,
    WindStrength,
    create_firefighting_state,
)
from POMDPPlanners.environments.firefighting_pomdp.firefighting_vectorized_model import (
    FirefightingVectorizedModel,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import firefighting_pinned_kwargs

_SCALAR_DRAWS = 3000
_MODEL_DRAWS = 30000

_U = int(FireCategory.UNBURNT)
_S = int(FireCategory.SMOLDERING)
_B = int(FireCategory.BURNING)
_X = int(FireCategory.BURNT)
_W = int(FireCategory.WET)

# Per-firefighter action codes: N, E, S, W, SUPPRESS.
_NORTH, _EAST, _SOUTH, _WEST, _SUPPRESS = 0, 1, 2, 3, 4


def _joint(*per_firefighter: int) -> int:
    return sum(code * 5**index for index, code in enumerate(per_firefighter))


def _fire_front() -> np.ndarray:
    fire = np.zeros((10, 10), dtype=np.int64)
    fire[2, 4] = _B
    fire[3, 4] = _B
    fire[4, 4] = _S
    fire[4, 3] = _X
    fire[3, 6] = _S
    fire[7, 7] = _W
    return fire


def _two_firefighter_cases() -> List[Tuple[str, FirefightingPOMDP, np.ndarray, int]]:
    env = FirefightingPOMDP(discount_factor=0.95, **firefighting_pinned_kwargs())
    wind = (int(WindDirection.EAST), int(WindStrength.HIGH))
    front = create_firefighting_state(env, [(3, 3, 2, 3), (3, 5, 0, 1)], wind, _fire_front(), 4)
    depot = create_firefighting_state(env, [(0, 1, 0, 3), (4, 5, 1, 2)], wind, _fire_front(), 9)
    return [
        ("both-spray-one-empty-tank", env, front, _joint(_SUPPRESS, _SUPPRESS)),
        ("spray-and-walk-into-fire", env, front, _joint(_SUPPRESS, _WEST)),
        ("walk-into-burnt-and-fire", env, front, _joint(_SOUTH, _NORTH)),
        ("depot-refill-and-spray", env, depot, _joint(_WEST, _SUPPRESS)),
    ]


def _three_firefighter_cases() -> List[Tuple[str, FirefightingPOMDP, np.ndarray, int]]:
    env = FirefightingPOMDP(
        discount_factor=0.95,
        **firefighting_pinned_kwargs(
            num_firefighters=3,
            firefighter_start_cells=[(2, 2), (2, 3), (3, 2)],
            is_all_firefighters_disabled_terminal=False,
        ),
    )
    wind = (int(WindDirection.SOUTH), int(WindStrength.LOW))
    state = create_firefighting_state(
        env, [(2, 3, 3, 3), (3, 5, 3, 3), (0, 0, 0, 0)], wind, _fire_front(), 1
    )
    # Two hoses overlap on (3, 4); the third firefighter is disabled.
    return [("three-overlapping-hoses", env, state, _joint(_SUPPRESS, _SUPPRESS, _SUPPRESS))]


_CASES = _two_firefighter_cases() + _three_firefighter_cases()


def _model(env: FirefightingPOMDP) -> FirefightingVectorizedModel:
    return FirefightingVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)


def _assert_same_column_frequencies(scalar: np.ndarray, model: np.ndarray, label: str) -> None:
    """Every column's value frequencies agree within five standard errors plus 0.01."""
    for column in range(scalar.shape[1]):
        values = np.union1d(scalar[:, column], model[:, column])
        for value in values:
            p_scalar = float(np.mean(scalar[:, column] == value))
            p_model = float(np.mean(model[:, column] == value))
            pooled = 0.5 * (p_scalar + p_model)
            tolerance = 5.0 * np.sqrt(pooled * (1.0 - pooled) / scalar.shape[0]) + 0.01
            assert abs(p_scalar - p_model) <= tolerance, (
                f"{label}: column {column} value {value}: scalar {p_scalar:.4f} "
                f"vs model {p_model:.4f}"
            )


def _scalar_successors(env: FirefightingPOMDP, state: np.ndarray, action: int) -> np.ndarray:
    np.random.seed(0)
    return np.stack(
        [env.sample_next_state(state=state, action=action) for _ in range(_SCALAR_DRAWS)]
    )


def _model_successors(
    model: FirefightingVectorizedModel, state: np.ndarray, action: int
) -> np.ndarray:
    torch.manual_seed(0)
    states = torch.as_tensor(state, dtype=torch.float64).repeat(_MODEL_DRAWS, 1)
    actions = torch.full((_MODEL_DRAWS,), action, dtype=torch.int64)
    return model.sample_next_states(states, actions).numpy()


@pytest.mark.parametrize("name,env,state,action", _CASES, ids=[case[0] for case in _CASES])
def test_next_state_frequencies_match_scalar_env(
    name: str, env: FirefightingPOMDP, state: np.ndarray, action: int
) -> None:
    """Every entry of the successor has the scalar environment's value frequencies.

    Purpose: VOPP searches on ``sample_next_states``. A suppression, spread or
        tank rule that differs from ``FirefightingPOMDP._transition`` makes the
        planner fight a different fire than the episode runs.

    Given: A state where hoses, the depot, burnt and burning cells are all
        reached on the first step.
    When: Both sides draw successors under the same joint action.
    Then: Each entry's value frequencies agree within five standard errors.

    Test type: integration
    """
    model = _model(env)
    _assert_same_column_frequencies(
        _scalar_successors(env, state, action), _model_successors(model, state, action), name
    )


@pytest.mark.parametrize("name,env,state,action", _CASES, ids=[case[0] for case in _CASES])
def test_rewards_and_terminal_flags_match_scalar_env(
    name: str, env: FirefightingPOMDP, state: np.ndarray, action: int
) -> None:
    """``rewards`` and ``terminal_mask`` equal the scalar values on every scalar successor.

    Purpose: The reward reads spray counts, health lost and newly burnt cells
        from the pair of states; any slip in that bookkeeping changes what
        VOPP optimizes.

    Given: Scalar successors of a hand-built fire front.
    When: The model scores the same (state, action, successor) rows.
    Then: Rewards agree to 1e-9 and terminal flags agree exactly.

    Test type: integration
    """
    model = _model(env)
    successors = _scalar_successors(env, state, action)[:300]
    states = torch.as_tensor(np.repeat(state[None, :], len(successors), axis=0))
    actions = torch.full((len(successors),), action, dtype=torch.int64)
    rewards = model.rewards(states, actions, torch.as_tensor(successors)).numpy()
    expected = np.array([env.reward(state, action, successor) for successor in successors])
    np.testing.assert_allclose(rewards, expected, atol=1e-9, err_msg=name)
    terminal = model.terminal_mask(torch.as_tensor(successors)).numpy()
    assert terminal.tolist() == [env.is_terminal(successor) for successor in successors]


def _observation_rows(env: FirefightingPOMDP, model: Any, successor: np.ndarray) -> np.ndarray:
    scalar = env.sample_observation(next_state=successor, action=0)
    vectorized = model.sample_observations(
        torch.as_tensor(successor[None, :]), torch.zeros(1, dtype=torch.int64)
    ).numpy()[0]
    return np.stack([scalar, vectorized])


@pytest.mark.parametrize("name,env,state,action", _CASES, ids=[case[0] for case in _CASES])
def test_observation_kernels_match_scalar_env(
    name: str, env: FirefightingPOMDP, state: np.ndarray, action: int
) -> None:
    """Readings have the scalar frequencies and score the scalar log-likelihood.

    Purpose: The belief inside VOPP is weighted by ``observation_log_probs``;
        a footprint or confusion rule that differs from the environment's
        tracks a different sensor.

    Given: Scalar successors of a hand-built fire front, including one with a
        disabled firefighter.
    When: Both sides draw readings, and the model scores readings from both
        sides against several successors.
    Then: Reading frequencies agree, and the model's log-likelihood equals the
        scalar one wherever the scalar one is finite, and is the floor where
        the scalar one is ``-inf``.

    Test type: integration
    """
    model = _model(env)
    successors = _scalar_successors(env, state, action)[:40]
    np.random.seed(1)
    torch.manual_seed(1)
    for successor in successors[:10]:
        readings = _observation_rows(env, model, successor)
        candidates = torch.as_tensor(successors)
        for reading in readings:
            scores = model.observation_log_probs(
                candidates,
                torch.zeros(len(successors), dtype=torch.int64),
                torch.as_tensor(np.repeat(reading[None, :], len(successors), axis=0)),
            ).numpy()
            expected = np.array(
                [float(env.observation_log_probability(s, 0, [reading])[0]) for s in successors]
            )
            finite = np.isfinite(expected)
            np.testing.assert_allclose(scores[finite], expected[finite], atol=1e-9, err_msg=name)
            assert np.all(scores[~finite] < -500.0), name

    successor = successors[0]
    np.random.seed(2)
    scalar = np.stack(
        [env.sample_observation(next_state=successor, action=0) for _ in range(_SCALAR_DRAWS)]
    )
    torch.manual_seed(2)
    vectorized = model.sample_observations(
        torch.as_tensor(successor).repeat(_MODEL_DRAWS, 1),
        torch.zeros(_MODEL_DRAWS, dtype=torch.int64),
    ).numpy()
    _assert_same_column_frequencies(scalar, vectorized, name)


def test_observation_keys_separate_readings_that_differ_in_one_cell() -> None:
    """Two readings that differ in one cell get different tree keys.

    Purpose: VOPP branches its belief tree on ``observation_keys``; a key that
        ignored a cell would merge two different readings into one branch.

    Given: One reading and copies that each change one entry.
    When: The model hashes them.
    Then: Every key differs from the original's.

    Test type: unit
    """
    _, env, state, _ = _CASES[0]
    model = _model(env)
    torch.manual_seed(0)
    reading = model.sample_observations(
        torch.as_tensor(state[None, :]), torch.zeros(1, dtype=torch.int64)
    )
    changed = reading.repeat(reading.shape[1], 1)
    for column in range(reading.shape[1]):
        changed[column, column] = changed[column, column] + 1.0
    keys = model.observation_keys(torch.cat([reading, changed]))
    assert torch.all(keys[1:] != keys[0])
