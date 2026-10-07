# SPDX-License-Identifier: MIT

"""Native (C++) equivalence tests for :class:`FirefightingPOMDP`.

The public sampling, density, reward and termination methods call
``_native.FirefightingModelCpp``. The Python code they replaced stays on the
class as the ``_*_reference`` methods. These tests hold the two to the same
law:

* Deterministic parts -- reward, termination, both log-densities -- are
  compared value for value on random states, actions and candidates.
* With every probability at 0 or 1 the transition and the observation are
  deterministic, so native and reference successors must be equal arrays.
* With the draws on, the native sampler is checked against the reference
  density on a world small enough to enumerate, and against the reference
  sampler through per-cell marginals on the default world.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

import pickle
from typing import Any, Dict, List

import numpy as np
import pytest

from POMDPPlanners.environments.firefighting_pomdp import (
    FireCategory,
    FirefightingPOMDP,
    _native,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import firefighting_pinned_kwargs
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

# Configurations chosen to reach every branch the kernels have: the pinned
# default; three firefighters on an open grid; the error and slip endpoints
# (0 and 1, where a log term is -inf); clipped downwind rate; and the
# all-disabled flag off.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "pinned": {},
    "three_firefighters_open": {
        "num_rows": 6,
        "num_cols": 6,
        "num_firefighters": 3,
        "obstacle_cells": [],
        "firefighter_start_cells": [(1, 1), (1, 2), (4, 4)],
        "depot_cell": (0, 0),
        "sensing_radius": 1,
        "max_tank": 2,
        "max_health": 2,
    },
    "endpoints_zero": {
        "num_rows": 4,
        "num_cols": 5,
        "obstacle_cells": [(2, 2)],
        "firefighter_start_cells": [(0, 1), (3, 4)],
        "depot_cell": (0, 0),
        "observation_error_probability": 0.0,
        "slip_probability": 0.0,
        "growth_probability": 0.0,
        "suppression_probability_burning": 0.0,
        "is_all_firefighters_disabled_terminal": False,
    },
    "endpoints_one": {
        "num_rows": 5,
        "num_cols": 4,
        "obstacle_cells": [(1, 1)],
        "firefighter_start_cells": [(0, 0), (4, 3)],
        "depot_cell": (0, 3),
        "observation_error_probability": 1.0,
        "slip_probability": 1.0,
        "burnout_probability": 1.0,
        "crosswind_attenuation": 1.0,
        "spread_probability": 0.6,
        "wind_gain_high": 3.5,
        "sensing_radius": 0,
    },
}

# Every draw switched to 0 or 1, so a step and a reading have one outcome.
DETERMINISTIC = {
    "slip_probability": 0.0,
    "spread_probability": 0.0,
    "growth_probability": 0.0,
    "burnout_probability": 0.0,
    "suppression_probability_unburnt": 1.0,
    "suppression_probability_smoldering": 1.0,
    "suppression_probability_burning": 1.0,
    "observation_error_probability": 0.0,
}

# Every spread, growth and burnout draw at probability 1: an unburnt cell
# beside an alight one always catches, smoldering always grows and burning
# always burns out. Exercises the fire rules exactly, which DETERMINISTIC
# switches off.
DETERMINISTIC_FIRE = {
    **DETERMINISTIC,
    "spread_probability": 1.0,
    "crosswind_attenuation": 0.0,
    "growth_probability": 1.0,
    "burnout_probability": 1.0,
}


def build_env(**overrides: Any) -> FirefightingPOMDP:
    return FirefightingPOMDP(discount_factor=0.95, **firefighting_pinned_kwargs(**overrides))


@pytest.fixture(params=sorted(CONFIGS), name="env")
def fixture_env(request) -> FirefightingPOMDP:
    return build_env(**CONFIGS[request.param])


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)
    _native.set_seed(0)


def random_states(env: FirefightingPOMDP, rng: np.random.Generator, count: int) -> np.ndarray:
    """Draw states anywhere in the state space, not only reachable ones.

    Health and tank include zero, a quarter of the maps have no alight cell,
    and the step counter spans the budget, so the disabled, empty-tank,
    goal and timeout branches are all exercised.
    """
    free = [
        (row, col)
        for row in range(env.num_rows)
        for col in range(env.num_cols)
        if not env.obstacle_mask[row, col]
    ]
    states = np.zeros((count, env.state_size), dtype=np.float64)
    for index in range(count):
        state = states[index]
        state[0] = float(rng.integers(0, env.max_steps + 1))
        for firefighter in range(env.num_firefighters):
            row, col = free[int(rng.integers(0, len(free)))]
            base = 1 + 4 * firefighter
            state[base : base + 4] = (
                row,
                col,
                rng.integers(0, env.max_tank + 1),
                rng.integers(0, env.max_health + 1),
            )
        state[env.wind_direction_index] = float(rng.integers(0, 4))
        state[env.wind_strength_index] = float(rng.integers(0, 2))
        # Every fourth map holds only unburnt, burnt and wet cells: nothing alight.
        categories = [0, 1, 2, 3, 4] if index % 4 else [0, 3, 4]
        fire = rng.choice(categories, size=env.num_cells)
        fire[env.obstacle_mask.ravel()] = int(FireCategory.UNBURNT)
        state[env.fire_offset :] = fire
    return states


def perturbed(env: FirefightingPOMDP, successor: np.ndarray, rng: np.random.Generator) -> List:
    """Candidates one edit away from ``successor``, most of them unreachable."""
    out = []
    cell = env.fire_offset + int(rng.integers(0, env.num_cells))
    for category in range(5):
        candidate = successor.copy()
        candidate[cell] = float(category)
        out.append(candidate)
    for field in range(1, 1 + 4 * env.num_firefighters):
        candidate = successor.copy()
        candidate[field] += 1.0
        out.append(candidate)
    for index in (0, env.wind_direction_index):
        candidate = successor.copy()
        candidate[index] += 1.0
        out.append(candidate)
    candidate = successor.copy()
    candidate[cell] = np.nan
    out.append(candidate)
    return out


def assert_same_log(native: np.ndarray, reference: np.ndarray) -> None:
    native = np.asarray(native, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    # A density is a log-probability: NaN and +inf are bugs on either side.
    assert not np.any(np.isnan(native) | np.isposinf(native)), native
    assert not np.any(np.isnan(reference) | np.isposinf(reference)), reference
    np.testing.assert_array_equal(np.isneginf(native), np.isneginf(reference))
    finite = np.isfinite(reference)
    np.testing.assert_allclose(native[finite], reference[finite], rtol=1e-12, atol=1e-12)


class TestDeterministicParts:
    def test_reward_matches_reference(self, env):
        """Test that the native reward equals the Python reward on random transitions.

        Purpose: The reward is a pure function of (state, action, successor),
            so any difference is a port bug.

        Given: Random states, random joint actions and native successors.
        When: Both rewards and the batch reward are computed.
        Then: They are equal to the last bit.

        Test type: unit
        """
        rng = np.random.default_rng(1)
        states = random_states(env, rng, 60)
        for action in rng.integers(0, env.num_actions, size=8):
            successors = env.sample_next_state_batch(states, int(action))
            reference = np.array(
                [env._reward_reference(s, int(action), n) for s, n in zip(states, successors)]
            )
            single = np.array([env.reward(s, int(action), n) for s, n in zip(states, successors)])
            np.testing.assert_array_equal(single, reference)
            np.testing.assert_array_equal(
                env.reward_batch(states, int(action), successors), reference
            )

    def test_is_terminal_matches_reference(self, env):
        """Test that native termination equals the Python rule on random states.

        Purpose: Termination is read off the fire map, healths and step,
            and a disagreement would end episodes at different points.

        Given: Random states, including empty maps, all-disabled teams and
            exhausted step budgets.
        When: Both termination checks run.
        Then: They agree on every state.

        Test type: unit
        """
        rng = np.random.default_rng(2)
        for state in random_states(env, rng, 300):
            assert env.is_terminal(state) == env._is_terminal_reference(state)

    def test_transition_log_probability_matches_reference(self, env):
        """Test that the native transition density equals the Python density.

        Purpose: Belief reweighting reads this density, so the two must agree
            on reachable successors and on which candidates are impossible.

        Given: Random states and actions, native successors, and candidates
            one edit away from them.
        When: Both densities score every candidate.
        Then: The same candidates are -inf and the finite values agree to 1e-12.

        Test type: unit
        """
        rng = np.random.default_rng(3)
        states = random_states(env, rng, 25)
        for action in rng.integers(0, env.num_actions, size=6):
            for state in states:
                successor = env.sample_next_state(state, int(action))
                candidates = [successor] + perturbed(env, successor, rng)
                native = env.transition_log_probability(state, int(action), candidates)
                reference = [
                    env._successor_log_probability_reference(state, int(action), c)
                    for c in candidates
                ]
                assert np.isfinite(native[0])
                assert_same_log(native, reference)

    def test_observation_log_probability_matches_reference(self, env):
        """Test that the native observation likelihood equals the Python one.

        Purpose: Every particle weight is a product of these likelihoods.

        Given: Random states, native readings of them, and readings with one
            field or cell changed.
        When: Both likelihoods score every reading.
        Then: The same readings are -inf and the finite values agree to 1e-12.

        Test type: unit
        """
        rng = np.random.default_rng(4)
        for state in random_states(env, rng, 80):
            observation = env.sample_observation(state, 0)
            candidates = [observation]
            for position in (0, 4 * env.num_firefighters + int(rng.integers(0, env.num_cells))):
                for value in (-1.0, 0.0, 2.0, 4.0, 2.5, 7.0):
                    candidate = observation.copy()
                    candidate[position] = value
                    candidates.append(candidate)
            native = env.observation_log_probability(state, 0, candidates)
            reference = env._observation_log_probability_reference(state, candidates)
            assert np.isfinite(native[0])
            assert_same_log(native, reference)

    def test_wrongly_shaped_candidates_score_minus_infinity(self, env):
        state = random_states(env, np.random.default_rng(5), 1)[0]
        assert env.transition_log_probability(state, 0, [state[:-1]])[0] == -np.inf
        assert env.observation_log_probability(state, 0, [np.zeros(3)])[0] == -np.inf

    def test_an_action_outside_the_space_raises(self, env):
        state = random_states(env, np.random.default_rng(6), 1)[0]
        with pytest.raises(ValueError):
            env.sample_next_state(state, env.num_actions)
        with pytest.raises(ValueError):
            env.sample_next_state(state, -1)


class TestDeterministicConfiguration:
    @pytest.mark.parametrize("config", sorted(CONFIGS))
    @pytest.mark.parametrize(
        "draws", [DETERMINISTIC, DETERMINISTIC_FIRE], ids=["fire_off", "fire_certain"]
    )
    def test_successor_and_reading_equal_the_reference(self, config, draws):
        """Test that with every draw at 0 or 1 native and reference steps are equal.

        Purpose: Removes the randomness so the motion, suppression, tank,
            depot, heat and bookkeeping rules are compared exactly.

        Given: A configuration with every probability at 0 or 1.
        When: Native and reference successors and readings are drawn.
        Then: They are equal arrays.

        Test type: unit
        """
        env = build_env(**{**CONFIGS[config], **draws})
        rng = np.random.default_rng(7)
        states = random_states(env, rng, 40)
        for action in rng.integers(0, env.num_actions, size=10):
            for state in states:
                native = env.sample_next_state(state, int(action))
                np.testing.assert_array_equal(native, env._transition_reference(state, int(action)))
                np.testing.assert_array_equal(
                    env.sample_observation(native, int(action)),
                    env._draw_observation_reference(native),
                )

    def test_rollout_matches_a_python_loop_over_the_reference(self):
        """Test that the native rollout returns the Python recursion's value.

        Purpose: The rollout kernel replaces a Python loop of step, reward
            and terminal checks; with deterministic steps the returns must agree.

        Given: The deterministic configuration and a fixed action sequence.
        When: The native rollout and the recursion of python_random_rollout,
            run on the reference methods, use the same actions.
        Then: The returns agree to 1e-12.

        Test type: unit
        """
        env = build_env(**DETERMINISTIC)
        rng = np.random.default_rng(8)
        for state in random_states(env, rng, 30):
            depth, max_depth = 2, 25
            actions = rng.integers(0, env.num_actions, size=max_depth - depth)

            def recurse(current, d):
                if d >= max_depth or env._is_terminal_reference(current):
                    return 0.0
                action = int(actions[d - depth])
                successor = env._transition_reference(current, action)
                reward = env._reward_reference(current, action, successor)
                return reward + 0.9 * recurse(successor, d + 1)

            native = env._native_model.simulate_rollout(state, actions, max_depth, depth, 0.9)
            assert native == pytest.approx(recurse(state, depth), rel=1e-12, abs=1e-12)


class TestSampledLaw:
    @staticmethod
    def tiny_env() -> FirefightingPOMDP:
        return FirefightingPOMDP(
            discount_factor=0.95,
            num_rows=3,
            num_cols=3,
            num_firefighters=1,
            obstacle_cells=[(0, 2)],
            depot_cell=(2, 2),
            firefighter_start_cells=[(1, 1)],
            max_tank=2,
            max_health=2,
            sensing_radius=1,
            slip_probability=0.3,
        )

    def test_native_successor_frequencies_match_the_reference_density(self):
        """Test that native successors occur at the rate the Python density gives.

        Purpose: Checks the native sampler against the Python law directly,
            not against the native density.

        Given: A 3x3 world with a mixed fire map and three actions.
        When: 40000 native successors are drawn per action.
        Then: Each distinct successor's frequency is within 0.01 of its
            reference probability, and those probabilities sum to 1.

        Test type: integration
        """
        env = self.tiny_env()
        fire = np.zeros((3, 3))
        fire[0, 0] = FireCategory.BURNING
        fire[1, 0] = FireCategory.SMOLDERING
        fire[2, 0] = FireCategory.BURNT
        fire[2, 1] = FireCategory.WET
        state = np.concatenate([[3.0, 1, 1, 2, 2, 1, 1], fire.ravel()])
        trials = 40000
        for action in (0, 1, 4):
            draws = env.sample_next_state(state, action, n_samples=trials)
            keys, counts = np.unique(draws, axis=0, return_counts=True)
            mass = 0.0
            for key, count in zip(keys, counts):
                probability = np.exp(env._successor_log_probability_reference(state, action, key))
                mass += probability
                assert probability == pytest.approx(count / trials, abs=0.01)
            assert mass == pytest.approx(1.0, abs=0.01)

    def test_native_cell_marginals_match_the_reference_sampler(self):
        """Test that per-cell successor marginals agree between native and Python.

        Purpose: On the default 10x10 world the successor space is too large
            to enumerate, so the per-cell category rates are compared.

        Given: A mid-episode default-world state and a SUPPRESS joint action.
        When: 6000 successors are drawn by each sampler.
        Then: Every per-cell category rate agrees within five standard errors.

        Test type: integration
        """
        env = build_env()
        state = env.initial_state_dist().sample()[0]
        for action in np.random.randint(0, env.num_actions, size=8):
            state = env.sample_next_state(state, int(action))
        state[0] = 0.0
        action = 24  # both firefighters SUPPRESS
        trials = 6000
        native = env.sample_next_state(state, action, n_samples=trials)
        reference = np.stack([env._transition_reference(state, action) for _ in range(trials)])
        for category in range(5):
            p_native = (native == category).mean(axis=0)
            p_reference = (reference == category).mean(axis=0)
            pooled = 0.5 * (p_native + p_reference)
            bound = 5.0 * np.sqrt(2.0 * pooled * (1.0 - pooled) / trials) + 1e-9
            assert np.all(np.abs(p_native - p_reference) <= bound), category

    def test_native_reading_matches_the_confusion_matrix(self):
        """Test that native readings follow the confusion matrix of the reference.

        Purpose: A sampler whose wrong readings favour one category would
            still pass the support check but bias the belief.

        Given: A state with every category present inside the footprints.
        When: 20000 native readings are drawn.
        Then: Each visible cell reports its truth at 1 - e and each wrong
            category at e / 4 within 0.01, and unseen cells report -1.

        Test type: integration
        """
        env = build_env(observation_error_probability=0.2)
        state = random_states(env, np.random.default_rng(9), 1)[0]
        visible = env.visible_mask(state).ravel()
        truth = env.fire_map(state).ravel()
        readings = env.sample_observation(state, 0, n_samples=20000)
        cells = readings[:, 4 * env.num_firefighters :]
        assert np.all(cells[:, ~visible] == -1.0)
        for index in np.flatnonzero(visible):
            for category in range(5):
                expected = 0.8 if category == truth[index] else 0.05
                assert (cells[:, index] == category).mean() == pytest.approx(expected, abs=0.01)


class TestEnvironmentPlumbing:
    def test_sample_next_step_is_consistent(self, env):
        """Test that the one-call step returns a reward and reading of its own successor."""
        rng = np.random.default_rng(10)
        for state in random_states(env, rng, 40):
            action = int(rng.integers(0, env.num_actions))
            successor, observation, reward = env.sample_next_step(state, action)
            assert reward == env._reward_reference(state, action, successor)
            assert np.isfinite(env._successor_log_probability_reference(state, action, successor))
            assert np.isfinite(
                env._observation_log_probability_reference(successor, [observation])[0]
            )

    def test_an_off_grid_firefighter_raises_instead_of_reading_past_the_map(self, env):
        state = random_states(env, np.random.default_rng(12), 1)[0]
        for row in (env.num_rows, -1):
            bad = state.copy()
            bad[1] = float(row)
            with pytest.raises(ValueError):
                env.sample_next_state(bad, 4)  # firefighter 0 suppresses: no move
            candidate = bad.copy()
            candidate[0] += 1.0  # passes the step and wind checks that come first
            with pytest.raises(ValueError):
                env.transition_log_probability(bad, 4, [candidate])

    def test_empty_inputs_give_empty_outputs(self, env):
        state = random_states(env, np.random.default_rng(13), 1)[0]
        assert env.sample_next_state_batch([], 0).shape == (0, env.state_size)
        assert env.reward_batch([], 0, []).shape == (0,)
        assert env.sample_next_state(state, 0, n_samples=0).shape == (0, env.state_size)
        assert env.sample_observation(state, 0, n_samples=0).shape == (0, env.observation_size)

    def test_set_seed_reproduces_the_draws(self, env):
        state = random_states(env, np.random.default_rng(11), 1)[0]
        _native.set_seed(123)
        first = env.sample_next_state(state, 3, n_samples=20)
        _native.set_seed(123)
        np.testing.assert_array_equal(env.sample_next_state(state, 3, n_samples=20), first)

    def test_pickle_round_trip_keeps_identity_and_rebuilds_the_model(self, env):
        before = env.config_id
        state = env.initial_state_dist().sample()[0]
        env.sample_next_state(state, 0)
        clone = pickle.loads(pickle.dumps(env))
        assert clone.config_id == before == env.config_id
        assert clone == env
        assert clone.sample_next_state(state, 0).shape == (env.state_size,)

    def test_rollout_with_a_uniform_sampler_matches_the_reference_rollout_mean(self):
        """Test that the native rollout's return has the reference rollout's mean.

        Purpose: With the draws on, the two rollouts can only agree in law. The
            comparison loop steps the ``_reference`` methods, not the public
            ones, which now call the same C++ code as the rollout.

        Given: The default world, a mid-episode state and a uniform sampler.
        When: 1500 rollouts of depth 15 are run each way.
        Then: The means agree within five standard errors.

        Test type: integration
        """
        env = build_env()
        state = env.initial_state_dist().sample()[0]
        sampler = DiscreteActionSampler(env.get_actions())

        def reference_rollout(current):
            total, discount = 0.0, 1.0
            for _ in range(15):
                if env._is_terminal_reference(current):
                    break
                action = int(np.random.randint(env.num_actions))
                successor = env._transition_reference(current, action)
                total += discount * env._reward_reference(current, action, successor)
                discount *= 0.95
                current = successor
            return total

        native = np.array(
            [env.simulate_random_rollout(state, sampler, 15, 0.95) for _ in range(1500)]
        )
        python = np.array([reference_rollout(state) for _ in range(1500)])
        error = np.sqrt(native.var() / native.size + python.var() / python.size)
        assert abs(native.mean() - python.mean()) <= 5.0 * error + 1e-9

    def test_rollout_with_another_sampler_calls_the_sampler(self):
        """Test that a non-uniform sampler is not replaced by uniform draws."""

        class Always(ActionSampler):
            calls = 0

            def sample(self, belief_node=None):
                Always.calls += 1
                return 24

        env = build_env()
        state = env.initial_state_dist().sample()[0]
        env.simulate_random_rollout(state, Always(), 5, 0.95)
        assert Always.calls > 0
