# SPDX-License-Identifier: MIT

"""Native (C++) against Python reference for the Chicheck Invaders model.

The environment's public sampling, scoring, reward and terminal methods call
``_native``; the Python reference methods they replaced stay on the class
(``_step_once``, ``_python_transition_log_probability``,
``_sample_observation_once``, ``_observation_log_likelihood``,
``_python_reward``, ``_python_is_terminal``). These tests hold the two together
on many random states:

* deterministic parts must agree exactly -- the step given its dive coins, the
  transition log-probability, the reward and the terminal check;
* the observation log-likelihood must agree to rounding, because the native
  side calls the C++ ``erf`` where the Python side calls SciPy's ``ndtr``;
* the random parts -- dive coins, readings, rollouts -- draw from the native
  RNG rather than ``np.random``, so they are compared in distribution.
"""

# The Python reference methods are private on purpose: they exist for these
# comparisons, not for callers.
# pylint: disable=protected-access

import pickle
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pytest

from POMDPPlanners.environments.chicheck_invaders_pomdp import (  # pylint: disable=no-name-in-module
    _native,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    ChicheckInvadersAction,
    ChicheckInvadersPOMDP,
    ObservationMode,
    chicken_slots,
    create_chicheck_invaders_state,
    noiseless_preset,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
    IMPOSSIBLE_LOG_PROBABILITY,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

ACTIONS = [int(action) for action in ChicheckInvadersAction]

#: Configurations the exact-agreement tests sweep. Each one moves a branch of
#: the model: the degenerate dive probabilities, a longer cooldown, one and
#: many chickens, a small grid, the noiseless sensors and full observability.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "never_dives": {"dive_probability": 0.0},
    "always_dives": {"dive_probability": 1.0},
    "slow_gun": {"fire_cooldown": 2},
    "one_chicken": {"num_chickens": 1},
    "big_flock": {"num_columns": 10, "num_rows": 8, "num_chickens": 7},
    "small_grid": {"num_columns": 3, "num_rows": 3, "num_chickens": 2, "radar_radius": 1.5},
    "noiseless": noiseless_preset(),
    "full": {"observation_mode": ObservationMode.FULL},
    "odd_costs": {"shot_cost": 70.0, "kill_reward": 3.0, "step_cost": 0.0},
    "low_noise": {
        "num_chickens": 2,
        "ship_column_noise_std": 0.3,
        "camera_offset_noise_std": 0.4,
        "radar_range_noise_std": 0.4,
    },
}


def _env(name: str) -> ChicheckInvadersPOMDP:
    return ChicheckInvadersPOMDP(discount_factor=0.95, **CONFIGS[name])


def _random_states(env: ChicheckInvadersPOMDP, count: int, rng: np.random.Generator) -> np.ndarray:
    """States with every field drawn over its range, reachable or not.

    Chickens may be dead or diving, in any column and on any row above the
    ship, and the cooldown, hit flag and step counter cover the values that
    switch a branch. The two implementations must agree on all of them, not
    only on states an episode visits.

    Row 0 is left out. A live chicken there is reachable only as the one that
    hit the ship, and from it a dive and a pull-up leave the signature that
    ``_implied_dive_switches`` reads as "did not switch", so the replay oracle
    the exact tests rely on does not apply.
    """
    states = []
    for _ in range(count):
        chickens = np.column_stack(
            [
                rng.integers(0, env.num_columns, env.num_chickens),
                rng.integers(1, env.num_rows, env.num_chickens),
                rng.choice([-1.0, 1.0], env.num_chickens),
                rng.integers(0, 2, env.num_chickens),
                (rng.random(env.num_chickens) < 0.8).astype(float),
            ]
        ).astype(np.float64)
        states.append(
            create_chicheck_invaders_state(
                env,
                chickens=chickens,
                ship_column=int(rng.integers(0, env.num_columns)),
                cooldown=int(rng.integers(0, env.fire_cooldown + 2)),
                ship_hit=bool(rng.random() < 0.05),
                step=int(rng.integers(0, env.max_steps + 1)),
            )
        )
    return np.stack(states)


def _reachable_states(env: ChicheckInvadersPOMDP, count: int, seed: int) -> np.ndarray:
    """States reached from the start distribution by random actions, via the Python step."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    states = []
    for state in env.initial_state_dist().sample(n_samples=count):
        for _ in range(int(rng.integers(0, 12))):
            if env._python_is_terminal(state):
                break
            state = env._step_once(state, int(rng.integers(0, 4)))
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


def _test_states(env: ChicheckInvadersPOMDP, seed: int) -> np.ndarray:
    return np.concatenate(
        [_random_states(env, 150, np.random.default_rng(seed)), _reachable_states(env, 150, seed)]
    )


@pytest.mark.parametrize("config", sorted(CONFIGS))
class TestExactAgreement:
    def test_native_successor_replays_through_the_python_step(self, config):
        """Test that each native successor is the Python step under its implied coins.

        Purpose: The dive coins are the step's only randomness, and the
            environment can recover which coins a successor implies, so
            replaying them through the Python step makes the stochastic
            transition an exact comparison.

        Given: Random and reachable states under one configuration.
        When: A native successor is drawn for each state and action, both
            through ``sample_next_state`` and ``sample_next_state_batch``.
        Then: The Python step with the implied coins rebuilds it byte for byte,
            and the result is ``float64``.

        Test type: unit
        """
        env = _env(config)
        states = _test_states(env, seed=1)
        for action in ACTIONS:
            batch = env.sample_next_state_batch(states, action)
            assert batch.dtype == np.float64
            for state, batched in zip(states, batch):
                single = env.sample_next_state(state, action)
                assert single.dtype == np.float64
                for successor in (single, batched):
                    before = chicken_slots(state, env.num_chickens)
                    switches = env._implied_dive_switches(before, successor)
                    rebuilt = env._step_once(state, action, switches=switches)
                    np.testing.assert_array_equal(rebuilt, successor)

    def test_transition_log_probability_matches_python(self, config):
        """Test that the native transition log-probability equals the Python one.

        Purpose: The score decides importance weights in any planner that
            reweights by the transition, so it must be the same number.

        Given: Random and reachable states, native successors of them, and
            candidates made impossible by shifting one field.
        When: Both implementations score every candidate.
        Then: The scores are identical, including the impossible floor.

        Test type: unit
        """
        env = _env(config)
        states = _test_states(env, seed=2)
        rng = np.random.default_rng(2)
        for action in ACTIONS:
            for state in states[::3]:
                candidates = np.array(env.sample_next_state(state, action, n_samples=4))
                broken = candidates.copy()
                broken[np.arange(4), rng.integers(0, env.state_size, 4)] += 1.0
                every = np.concatenate([candidates, broken])
                native = env.transition_log_probability(state, action, every)
                python = env._python_transition_log_probability(state, action, every)
                np.testing.assert_array_equal(native, python)
                assert np.all(native[:4] > IMPOSSIBLE_LOG_PROBABILITY)

    def test_reward_and_terminal_match_python(self, config):
        """Test that reward, batched reward and the terminal check equal the Python ones.

        Purpose: The reward and terminal check are deterministic, so any
            difference is a porting error.

        Given: Random and reachable states and native successors of them.
        When: The reward is scored with and without the successor, singly and
            in a batch, and every state and successor is checked for
            terminality.
        Then: Every value equals the Python reference exactly.

        Test type: unit
        """
        env = _env(config)
        states = _test_states(env, seed=3)
        for action in ACTIONS:
            successors = env.sample_next_state_batch(states, action)
            with_next = [env._python_reward(s, action, n) for s, n in zip(states, successors)]
            without = [env._python_reward(s, action) for s in states]
            np.testing.assert_array_equal(
                env.reward_batch(states, action, successors), np.asarray(with_next)
            )
            np.testing.assert_array_equal(env.reward_batch(states, action), np.asarray(without))
            for state, successor, expected, bare in zip(states, successors, with_next, without):
                assert env.reward(state, action, successor) == expected
                assert env.reward(state, action) == bare
        for state in np.concatenate([states, env.sample_next_state_batch(states, 3)]):
            assert env.is_terminal(state) == env._python_is_terminal(state)

    def test_observation_log_likelihood_matches_python(self, config):
        """Test that the native observation log-likelihood equals the Python one to rounding.

        Purpose: The likelihood weights every particle in the belief update.
            The two sides use different standard-normal CDF code, so they may
            differ in the last bits but never more, and the impossible floor
            must match exactly.

        Given: Random and reachable successors, readings drawn from both
            samplers, and readings with one field shifted.
        When: Both implementations score each pair, through the scalar, the
            per-candidate and the per-state entry points.
        Then: The scores agree to a relative 1e-12 and impossible pairs score
            the floor on both sides.

        Test type: unit
        """
        env = _env(config)
        states = _test_states(env, seed=4)
        rng = np.random.default_rng(4)
        np.random.seed(4)
        for state in states[::2]:
            readings = [env.sample_observation(state, 0), env._sample_observation_once(state)]
            shifted = readings[0].copy()
            shifted[int(rng.integers(0, env.observation_size))] += 1.0
            readings.append(shifted)
            readings.append(np.asarray(env.sample_observation(states[0], 0)))
            native = env.observation_log_probability(state, 0, readings)
            python = np.array([env._observation_log_likelihood(state, r) for r in readings])
            np.testing.assert_allclose(native, python, rtol=1e-12, atol=0.0)
            np.testing.assert_array_equal(
                native <= IMPOSSIBLE_LOG_PROBABILITY, python <= IMPOSSIBLE_LOG_PROBABILITY
            )
            assert native[0] > IMPOSSIBLE_LOG_PROBABILITY
            for reading, expected in zip(readings, native):
                assert env.observation_log_probability_single(state, 0, reading) == expected
        reading = env.sample_observation(states[0], 0)
        per_state = env.observation_log_probability_per_state(states, 0, reading)
        python = np.array([env._observation_log_likelihood(s, reading) for s in states])
        np.testing.assert_allclose(per_state, python, rtol=1e-12, atol=0.0)


class TestMalformedInputs:
    def test_wrong_length_candidates_score_the_floor(self):
        """Test that a candidate of the wrong length is impossible, as in Python.

        Purpose: The Python reference returns the floor for a malformed
            candidate rather than raising; the native dispatch keeps that.

        Given: The default environment and a state.
        When: Successors and readings one field too long are scored.
        Then: Every score is the impossible floor.

        Test type: unit
        """
        env = _env("default")
        state = env.initial_state_dist().sample()[0]
        long_state = np.zeros((2, env.state_size + 1))
        long_reading = np.zeros(env.observation_size + 1)
        assert np.all(
            env.transition_log_probability(state, 0, long_state) == IMPOSSIBLE_LOG_PROBABILITY
        )
        assert np.all(
            env.observation_log_probability(state, 0, [long_reading]) == IMPOSSIBLE_LOG_PROBABILITY
        )
        assert env.observation_log_probability_single(state, 0, long_reading) == (
            IMPOSSIBLE_LOG_PROBABILITY
        )

    def test_action_outside_the_four_falls_back_to_python(self):
        """Test that an unknown action index behaves as the Python model does.

        Purpose: The Python step treats any action other than LEFT, RIGHT and
            FIRE as staying put. There is no kernel for such an action, so the
            dispatch falls back to Python rather than raising.

        Given: The never-dives configuration, so the step is deterministic.
        When: Action 7 is stepped, scored and rewarded.
        Then: The successor equals a STAY successor and the reward is the step cost.

        Test type: unit
        """
        env = _env("never_dives")
        state = env.initial_state_dist().sample()[0]
        stay = env.sample_next_state(state, int(ChicheckInvadersAction.STAY))
        odd = env.sample_next_state(state, 7)
        np.testing.assert_array_equal(odd, stay)
        assert env.transition_log_probability(state, 7, [odd])[0] == 0.0
        assert env.reward(state, 7, odd) == -env.step_cost


class TestDistributions:
    def test_dive_coin_frequency_is_the_dive_probability(self):
        """Test that each patrolling chicken switches to a dive at rate ``dive_probability``.

        Purpose: The native coins come from a different RNG, so their law is
            checked against the configured probability directly.

        Given: Four live patrolling chickens away from the floor and
            ``dive_probability = 0.3``.
        When: 20000 STAY successors are drawn.
        Then: Each chicken's dive frequency is within five standard errors of 0.3.

        Test type: unit
        """
        env = ChicheckInvadersPOMDP(discount_factor=0.95, dive_probability=0.3)
        state = create_chicheck_invaders_state(
            env, chickens=[[1, 4, 1, 0, 1], [3, 5, -1, 0, 1], [5, 3, 1, 0, 1], [6, 6, -1, 0, 1]]
        )
        draws = 20000
        successors = env.sample_next_state_batch(np.tile(state, (draws, 1)), 0)
        modes = successors[:, 4:].reshape(draws, env.num_chickens, 5)[:, :, 3]
        tolerance = 5.0 * np.sqrt(0.3 * 0.7 / draws)
        np.testing.assert_allclose(modes.mean(axis=0), 0.3, atol=tolerance)

    @pytest.mark.parametrize("config", ["low_noise", "small_grid"])
    def test_reading_frequencies_match_the_likelihood(self, config):
        """Test that native readings land on each value as often as the likelihood says.

        Purpose: Ties the native sampler to the likelihood both
            implementations agree on, and to the Python sampler, so a sampler
            that draws the wrong law fails even though its RNG differs.

        Given: A state with two live chickens within two cells of the ship,
            under a configuration whose noise is small enough that single
            readings recur often enough to count.
        When: 40000 readings are drawn from each sampler.
        Then: For every reading seen at least 1% of the time, the native
            frequency is within five standard errors of
            ``exp(log-likelihood)`` and of the Python frequency.

        Test type: integration
        """
        env = _env(config)
        chickens = [[env.ship_start_column, 2, 1, 1, 1], [env.ship_start_column - 1, 1, -1, 0, 1]]
        chickens += [[0, 1, 1, 0, 0]] * (env.num_chickens - 2)
        state = create_chicheck_invaders_state(env, chickens=chickens)
        draws = 40000
        np.random.seed(5)
        native = Counter(r.tobytes() for r in env.sample_observation(state, 0, n_samples=draws))
        python = Counter(env._sample_observation_once(state).tobytes() for _ in range(draws))
        frequent = [key for key, count in native.items() if count >= 0.01 * draws]
        assert frequent
        for key in frequent:
            reading = np.frombuffer(key, dtype=np.float64)
            expected = float(np.exp(env.observation_log_probability_single(state, 0, reading)))
            spread = 5.0 * np.sqrt(expected * (1.0 - expected) / draws)
            assert abs(native[key] / draws - expected) < spread
            assert abs(native[key] / draws - python[key] / draws) < np.sqrt(2.0) * spread

    def test_rollout_mean_matches_the_python_rollout(self):
        """Test that the native rollout's mean return equals the Python rollout's.

        Purpose: The native rollout draws its own dive coins, so it is
            compared with the Python reference loop in distribution.

        Given: The default environment and one start state.
        When: 4000 depth-20 rollouts are run each way with uniform actions.
        Then: The two mean returns are within five standard errors of each other.

        Test type: integration
        """
        env = _env("default")
        np.random.seed(6)
        state = env.initial_state_dist().sample()[0]
        sampler = DiscreteActionSampler(env.get_actions())
        runs = 4000
        native = np.array(
            [env.simulate_random_rollout(state, sampler, 20, 0.95) for _ in range(runs)]
        )
        python = np.array([_python_rollout(env, state, 20, 0.95) for _ in range(runs)])
        spread = 5.0 * np.sqrt(native.var() / runs + python.var() / runs)
        assert abs(native.mean() - python.mean()) < spread


def _python_rollout(env: ChicheckInvadersPOMDP, state: np.ndarray, depth: int, gamma: float):
    """Uniform random rollout through the Python reference methods only."""
    rewards: List[float] = []
    for _ in range(depth):
        if env._python_is_terminal(state):
            break
        action = int(np.random.randint(0, 4))
        successor = env._step_once(state, action)
        rewards.append(env._python_reward(state, action, successor))
        state = successor
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
    return total


class TestRolloutAndStep:
    def test_deterministic_rollout_equals_the_python_sum(self):
        """Test that a rollout with no randomness returns the Python sum exactly.

        Purpose: With ``dive_probability = 0`` the step is deterministic, so
            the native rollout over given actions must equal the Python loop
            over the same actions bit for bit, including the stop at a
            terminal state.

        Given: The never-dives configuration, random states, and pre-drawn actions.
        When: ``_native.simulate_rollout`` and a Python loop run the same actions.
        Then: The returns are identical.

        Test type: unit
        """
        env = _env("never_dives")
        rng = np.random.default_rng(7)
        for state in _test_states(env, seed=7)[::5]:
            actions = rng.integers(0, 4, 25).astype(np.int32)
            native = _native.simulate_rollout(env._native_config, state, actions, 0.9)
            rewards, current = [], state
            for action in actions:
                if env._python_is_terminal(current):
                    break
                successor = env._step_once(current, int(action))
                rewards.append(env._python_reward(current, int(action), successor))
                current = successor
            expected = 0.0
            for reward in reversed(rewards):
                expected = reward + 0.9 * expected
            assert native == expected

    def test_custom_sampler_is_followed(self):
        """Test that a rollout with a non-uniform sampler follows that sampler.

        Purpose: The native rollout draws uniform actions. A planner that
            passes its own rollout policy must get that policy, which the
            fallback to ``python_random_rollout`` provides.

        Given: The never-dives configuration and a sampler that always fires.
        When: A depth-10 rollout is run.
        Then: The return equals the Python loop that fires every step.

        Test type: unit
        """

        class AlwaysFire:  # pylint: disable=too-few-public-methods
            def sample(self, belief_node=None):  # pylint: disable=unused-argument
                return int(ChicheckInvadersAction.FIRE)

        env = _env("never_dives")
        state = env.initial_state_dist().sample()[0]
        rewards, current = [], state
        for _ in range(10):
            if env._python_is_terminal(current):
                break
            successor = env._step_once(current, 3)
            rewards.append(env._python_reward(current, 3, successor))
            current = successor
        expected = 0.0
        for reward in reversed(rewards):
            expected = reward + 0.95 * expected
        assert env.simulate_random_rollout(state, AlwaysFire(), 10, 0.95) == pytest.approx(
            expected, rel=1e-12
        )

    def test_sample_next_step_is_consistent(self):
        """Test that ``sample_next_step`` returns a reachable successor, its reading and reward.

        Purpose: The one-call native step must return the same triple the
            three separate calls would.

        Given: Reachable states of the default environment.
        When: ``sample_next_step`` is called for each action.
        Then: The successor is a possible transition, the reading is possible
            under the successor, and the reward equals ``reward`` on the
            successor.

        Test type: unit
        """
        env = _env("default")
        for state in _reachable_states(env, 60, seed=8):
            for action in ACTIONS:
                successor, reading, reward = env.sample_next_step(state, action)
                assert env.transition_log_probability(state, action, [successor])[0] > (
                    IMPOSSIBLE_LOG_PROBABILITY
                )
                assert env.observation_log_probability_single(successor, action, reading) > (
                    IMPOSSIBLE_LOG_PROBABILITY
                )
                assert isinstance(reward, float)
                assert reward == env.reward(state, action, successor)


class TestSeedingAndPickling:
    def test_set_seed_reproduces_samples(self):
        """Test that seeding the native RNG reproduces successors and readings.

        Purpose: ``np.random.seed`` does not reach the native RNG, so
            ``_native.set_seed`` is what makes a native run repeatable.

        Given: The default environment and a start state.
        When: The same draws are made twice after the same ``set_seed``.
        Then: Both runs return identical arrays.

        Test type: unit
        """
        env = _env("default")
        state = env.initial_state_dist().sample()[0]

        def draw():
            successors = env.sample_next_state(state, 3, n_samples=20)
            readings = env.sample_observation(successors[0], 3, n_samples=20)
            return successors, readings

        _native.set_seed(99)
        first = draw()
        _native.set_seed(99)
        second = draw()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_pickle_rebuilds_kernels_and_keeps_identity(self):
        """Test that a pickled environment works and keeps its config_id.

        Purpose: pybind11 kernels cannot be pickled, and the runner ships
            environments to worker processes, so the kernels are dropped and
            rebuilt; they must also stay out of ``config_id``.

        Given: An environment that has already sampled.
        When: It is pickled and unpickled.
        Then: The copy samples, its config_id equals the original's and a
            fresh environment's, and the two are equal.

        Test type: unit
        """
        env = _env("small_grid")
        state = env.initial_state_dist().sample()[0]
        env.sample_next_state(state, 0)
        copy = pickle.loads(pickle.dumps(env))
        assert copy.sample_next_state(state, 0).shape == (env.state_size,)
        assert copy.config_id == env.config_id == _env("small_grid").config_id
        assert copy == env
