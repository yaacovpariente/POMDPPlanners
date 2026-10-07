# SPDX-License-Identifier: MIT

"""The native Battleship kernels return what the Python ones they replaced did.

Every Battleship kernel is deterministic, so each comparison is exact: the
environment's native-backed method against the Python body it replaced, kept
in ``_battleship_python_reference``. States are drawn at random, with each
occupancy and probe flag set independently, so they include boards no legal
fleet produces and probe flags on water and on every repeat; a kernel only has
to agree on the flags, not on whether the board is legal.
"""

# pylint: disable=missing-function-docstring

from typing import List

import numpy as np
import pytest

from POMDPPlanners.environments.battleship_pomdp import _native
from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import BattleshipPOMDP
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.tests.test_environments.test_battleship_pomdp import (
    _battleship_python_reference as ref,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs

_N_STATES = 300


def _make_env(**overrides) -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.95, **battleship_pinned_kwargs(**overrides))


@pytest.fixture(name="env", params=[{}, {"hit_reward": 2.5, "miss_penalty": 0.0}])
def _env_fixture(request) -> BattleshipPOMDP:
    return _make_env(**request.param)


def _random_states(env: BattleshipPOMDP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    occupancy = rng.random((n, env.num_cells)) < 0.3
    probed = rng.random((n, env.num_cells)) < rng.random((n, 1))
    return np.concatenate([occupancy, probed], axis=1).astype(np.float64)


def _actions(env: BattleshipPOMDP, n: int, seed: int) -> List[int]:
    return [int(a) for a in np.random.default_rng(seed).integers(0, env.num_cells, size=n)]


def test_sample_next_state_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=1)
    for state, action in zip(states, _actions(env, _N_STATES, seed=2)):
        got = env.sample_next_state(state, action)
        want = ref.sample_next_state(state, action, env.num_cells)
        assert got.dtype == want.dtype == np.float64
        np.testing.assert_array_equal(got, want)
        got_many = env.sample_next_state(state, action, n_samples=4)
        np.testing.assert_array_equal(
            got_many, ref.sample_next_state(state, action, env.num_cells, n_samples=4)
        )


def test_sample_next_state_does_not_mutate_input(env: BattleshipPOMDP) -> None:
    state = _random_states(env, 1, seed=3)[0]
    before = state.copy()
    env.sample_next_state(state, 0)
    env.sample_next_step(state, 0)
    np.testing.assert_array_equal(state, before)


def test_sample_next_state_accepts_lists_and_int_arrays(env: BattleshipPOMDP) -> None:
    state = _random_states(env, 1, seed=4)[0]
    want = ref.sample_next_state(state, 3, env.num_cells)
    np.testing.assert_array_equal(env.sample_next_state(list(state), 3), want)
    np.testing.assert_array_equal(env.sample_next_state(state.astype(np.int64), 3), want)
    np.testing.assert_array_equal(env.sample_next_state(state, np.int64(3)), want)


def test_sample_next_state_batch_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=5)
    for action in range(env.num_cells):
        got = env.sample_next_state_batch(states, action)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(
            got, ref.sample_next_state_batch(states, action, env.num_cells)
        )
    np.testing.assert_array_equal(
        env.sample_next_state_batch(states[0], 2),
        ref.sample_next_state_batch(states[0], 2, env.num_cells),
    )


def test_transition_log_probability_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, 60, seed=6)
    for state, action in zip(states, _actions(env, 60, seed=7)):
        successor = ref.sample_next_state(state, action, env.num_cells)
        nudged = successor + np.random.default_rng(action).uniform(-0.49, 0.49, successor.shape)
        flipped = successor.copy()
        flipped[action] = 1.0 - flipped[action]
        candidates = np.stack([successor, nudged, flipped, state])
        got = env.transition_log_probability(state, action, candidates)
        want = ref.transition_log_probability(state, action, candidates, env.num_cells)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(
            env.transition_log_probability(state, action, successor),
            ref.transition_log_probability(state, action, successor, env.num_cells),
        )


def test_sample_observation_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=8)
    for state, action in zip(states, _actions(env, _N_STATES, seed=9)):
        got = env.sample_observation(state, action)
        assert type(got) is int  # pylint: disable=unidiomatic-typecheck
        assert got == ref.sample_observation(state, action, env.num_cells)
        assert env.sample_observation(state, action, n_samples=3) == ref.sample_observation(
            state, action, env.num_cells, n_samples=3
        )


@pytest.mark.parametrize(
    "observations",
    [[0], [1], [0, 1, 1, 0], np.array([1.0, 0.0]), np.array([[0, 1], [1, 1]]), [], [True]],
)
def test_observation_log_probability_matches_python(env: BattleshipPOMDP, observations) -> None:
    states = _random_states(env, 40, seed=10)
    for state, action in zip(states, _actions(env, 40, seed=11)):
        got = env.observation_log_probability(state, action, observations)
        want = ref.observation_log_probability(state, action, observations, env.num_cells)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, want)


def test_reward_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=12)
    for state, action in zip(states, _actions(env, _N_STATES, seed=13)):
        got = env.reward(state, action)
        assert isinstance(got, float)
        assert got == ref.reward(state, action, env.num_cells, env.hit_reward, env.miss_penalty)


def test_reward_batch_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=14)
    for action in range(env.num_cells):
        got = env.reward_batch(states, action)
        want = ref.reward_batch(states, action, env.num_cells, env.hit_reward, env.miss_penalty)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, want)


def test_is_terminal_matches_python(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=15)
    # Add boards that are terminal: every occupied cell probed.
    terminal = states.copy()
    terminal[:, env.num_cells :] = np.maximum(
        terminal[:, env.num_cells :], terminal[:, : env.num_cells]
    )
    for state in np.concatenate([states, terminal]):
        assert env.is_terminal(state) is ref.is_terminal(state, env.num_cells)
    assert any(env.is_terminal(s) for s in terminal)


def test_sample_next_step_matches_its_three_kernels(env: BattleshipPOMDP) -> None:
    states = _random_states(env, _N_STATES, seed=16)
    for state, action in zip(states, _actions(env, _N_STATES, seed=17)):
        next_state, observation, reward = env.sample_next_step(state, action)
        np.testing.assert_array_equal(
            next_state, ref.sample_next_state(state, action, env.num_cells)
        )
        assert observation == ref.sample_observation(next_state, action, env.num_cells)
        assert reward == ref.reward(state, action, env.num_cells, env.hit_reward, env.miss_penalty)


@pytest.mark.parametrize("start_depth,max_depth", [(0, 30), (5, 12), (0, 1), (7, 7)])
def test_rollout_kernel_matches_python_rollout_exactly(
    env: BattleshipPOMDP, start_depth: int, max_depth: int
) -> None:
    """Same actions in, the same return out, to the last bit."""
    states = _random_states(env, 50, seed=18)
    for i, state in enumerate(states):
        actions = np.random.default_rng(100 + i).integers(
            0, env.num_cells, size=max(max_depth - start_depth, 0), dtype=np.int32
        )
        got = _native.simulate_rollout_discrete(
            state,
            actions,
            max_depth,
            start_depth,
            0.95,
            env.num_cells,
            env.hit_reward,
            env.miss_penalty,
        )
        want = ref.random_rollout(
            state,
            start_depth,
            ref.replay(actions),
            env.num_cells,
            env.hit_reward,
            env.miss_penalty,
            0.95,
            max_depth,
        )
        assert got == want


def test_simulate_random_rollout_matches_python_rollout_distribution() -> None:
    """The env's rollout and the generic Python one agree in mean.

    The native rollout draws its actions with ``np.random.randint`` and the
    Python one asks an action sampler, so the two consume the RNG differently
    and are compared as distributions: a uniform sampler over the cells
    should give the same mean return.
    """
    env = _make_env()

    class _UniformCells(ActionSampler):  # pylint: disable=too-few-public-methods
        def sample(self, belief_node=None):  # pylint: disable=unused-argument
            return int(np.random.randint(env.num_cells))

    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    n = 4000
    native = np.array(
        [
            env.simulate_random_rollout(
                state=state, action_sampler=None, max_depth=25, discount_factor=0.95
            )
            for _ in range(n)
        ]
    )
    python = np.array(
        [
            python_random_rollout(
                state=state,
                depth=0,
                action_sampler=_UniformCells(),
                environment=env,
                discount_factor=0.95,
                max_depth=25,
            )
            for _ in range(n)
        ]
    )
    standard_error = np.sqrt(native.var() / n + python.var() / n)
    assert abs(native.mean() - python.mean()) < 4.0 * standard_error


def test_simulate_random_rollout_is_zero_at_the_depth_limit() -> None:
    env = _make_env()
    state = _random_states(env, 1, seed=19)[0]
    assert (
        env.simulate_random_rollout(state, None, max_depth=5, discount_factor=0.9, depth=5) == 0.0
    )
    assert (
        env.simulate_random_rollout(state, None, max_depth=5, discount_factor=0.9, depth=9) == 0.0
    )


def test_out_of_board_action_raises_index_error() -> None:
    env = _make_env()
    state = _random_states(env, 1, seed=20)[0]
    for bad in (-1, env.num_cells):
        with pytest.raises(IndexError):
            env.sample_next_state(state, bad)
        with pytest.raises(IndexError):
            env.reward(state, bad)
        with pytest.raises(IndexError):
            env.sample_observation(state, bad)


def test_is_terminal_rejects_wrong_length() -> None:
    env = _make_env()
    with pytest.raises(ValueError):
        env.is_terminal(np.zeros(2 * env.num_cells + 1))
