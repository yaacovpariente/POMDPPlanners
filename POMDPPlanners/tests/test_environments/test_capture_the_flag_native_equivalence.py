# SPDX-License-Identifier: MIT

"""Native C++ CaptureTheFlag model against its Python reference.

The environment's public methods run ``_native.CaptureTheFlagModelCpp``; the
``_python_*`` methods and ``_successor_distribution`` are the original Python
code, kept as the reference. Deterministic quantities -- successor
distributions, transition and observation likelihoods, rewards -- must agree
bit for bit. Sampled quantities draw from a different RNG than the reference,
so they are checked against the exact distributions instead.
"""

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP, _native
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

# pylint: disable=protected-access

CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {},
    # Zero slip, zero range error and certain pursuit put exact zeros into the
    # weight tables, which both sides must skip the same way.
    "zero_weights": {
        "slip_probability": 0.0,
        "range_error_probability": 0.0,
        "red_pursuit_probability": 1.0,
    },
    # Three blue players, three red with two defenders, an open field, a
    # longer game and coefficients of unusual sign.
    "three_a_side": {
        "n_blue": 3,
        "n_red": 3,
        "n_red_defenders": 2,
        "trees": (),
        "score_to_win": 2,
        "freeze_steps": 2,
        "tagger_cooldown_steps": 0,
        "capture_reward": 37.5,
        "tag_reward": -3.25,
        "move_cost": 0.3,
        "scan_cost": -0.7,
    },
    # Certain slips and certain range errors.
    "all_noise": {
        "slip_probability": 1.0,
        "range_error_probability": 1.0,
        "red_pursuit_probability": 0.0,
        "red_alert_radius": 8,
    },
}


@pytest.fixture(name="env", params=list(CONFIGS), ids=list(CONFIGS))
def env_fixture(request: pytest.FixtureRequest) -> CaptureTheFlagPOMDP:
    """Return one environment per configuration under test."""
    return CaptureTheFlagPOMDP(**CONFIGS[request.param])


def random_state(env: CaptureTheFlagPOMDP, rng: np.random.Generator) -> np.ndarray:
    """Return a random well-formed state, including flag carriers and counters.

    Args:
        env: The environment whose layout and field are used.
        rng: Source of randomness.

    Returns:
        A state vector; some are terminal.
    """
    layout = env.layout
    free = env.free_cells()
    state = np.zeros(layout.size, dtype=np.float64)
    layout.write_blue_cells(state, [free[int(rng.integers(len(free)))] for _ in range(env.n_blue)])
    layout.write_red_cells(state, [free[int(rng.integers(len(free)))] for _ in range(env.n_red)])
    state[layout.flag_cell] = float(rng.integers(len(env.red_flag_candidates)))
    state[layout.carrier_red_flag] = float(rng.integers(0, env.n_blue + 1) * (rng.random() < 0.4))
    state[layout.carrier_blue_flag] = float(rng.integers(0, env.n_red + 1) * (rng.random() < 0.4))
    for i in range(env.n_blue):
        state[layout.freeze_blue + i] = float(
            rng.integers(0, env.freeze_steps + 1) * (rng.random() < 0.3)
        )
        state[layout.cooldown_blue + i] = float(rng.integers(0, env.tagger_cooldown_steps + 1))
    for j in range(env.n_red):
        state[layout.freeze_red + j] = float(
            rng.integers(0, env.freeze_steps + 1) * (rng.random() < 0.3)
        )
        state[layout.cooldown_red + j] = float(rng.integers(0, env.tagger_cooldown_steps + 1))
    state[layout.score_blue] = float(rng.integers(0, env.score_to_win + (rng.random() < 0.1)))
    state[layout.score_red] = float(rng.integers(0, env.score_to_win + (rng.random() < 0.1)))
    return state


def state_sample(env: CaptureTheFlagPOMDP, n_random: int = 60) -> List[np.ndarray]:
    """Return random states plus states reached by Python-reference rollouts.

    Args:
        env: The environment.
        n_random: Number of synthetic random states.

    Returns:
        The states to test on.
    """
    rng = np.random.default_rng(123)
    states = [random_state(env, rng) for _ in range(n_random)]
    np.random.seed(5)
    n_actions = len(env.get_actions())
    for opening in env.initial_state_dist().values:  # type: ignore[attr-defined]
        state = opening
        for _ in range(25):
            states.append(state)
            if env.is_terminal(state):
                break
            state = env._python_sample_next_state(state, int(rng.integers(n_actions)))
    return states


def action_sample(env: CaptureTheFlagPOMDP, rng: np.random.Generator, size: int) -> List[int]:
    """Return a few joint actions, always including stay-all and scan-all."""
    n_actions = len(env.get_actions())
    fixed = [0, n_actions - 1]
    return fixed + [int(a) for a in rng.integers(0, n_actions, size=size)]


def test_successor_distribution_is_identical(env: CaptureTheFlagPOMDP) -> None:
    """The enumerated successor distribution matches the reference exactly.

    Purpose: The enumeration is where pick-up, tagging and scoring order, the
        red role policy and the per-player move tables all meet; equal bytes
        and equal floats in the same order mean the two transition models are
        the same model.

    Given: Random and reachable states and a spread of joint actions
    When: Both enumerate the successor distribution
    Then: The successor arrays and the probabilities are identical, in order

    Test type: unit
    """
    model = env._native_model()
    rng = np.random.default_rng(0)
    for state in state_sample(env):
        if env.is_terminal(state):
            continue
        for action in action_sample(env, rng, 4):
            ref_states, ref_probs = env._successor_distribution(state, action)
            nat_states, nat_probs = model.successor_distribution(state, action)
            assert nat_states.shape == (len(ref_states), env.layout.size)
            np.testing.assert_array_equal(nat_states, np.asarray(ref_states))
            np.testing.assert_array_equal(nat_probs, ref_probs)


def test_transition_log_probability_is_identical(env: CaptureTheFlagPOMDP) -> None:
    """Transition log-likelihoods match the reference bit for bit, including -inf.

    Purpose: Particle filters weight by these values; any drift changes beliefs.

    Given: States, actions and candidates that are reachable, unreachable,
        or the terminal state itself
    When: Both score the candidates
    Then: The log-probabilities are identical

    Test type: unit
    """
    rng = np.random.default_rng(1)
    states = state_sample(env, n_random=30)
    for state in states:
        for action in action_sample(env, rng, 2):
            candidates = [env._python_sample_next_state(state, action) for _ in range(4)]
            candidates += [states[int(rng.integers(len(states)))], state.copy()]
            candidates.append(list(state))
            expected = env._python_transition_log_probability(state, action, candidates)
            actual = env.transition_log_probability(state, action, candidates)
            np.testing.assert_array_equal(actual, expected)


def observation_candidates(
    env: CaptureTheFlagPOMDP, next_state: np.ndarray, action: int, rng: np.random.Generator
) -> List[Any]:
    """Return possible, impossible and malformed observations for one successor."""
    sampled = env._python_sample_observation(next_state, action, 6)
    candidates: List[Any] = list(sampled)
    base = list(sampled[0])
    n_prefix = 2 * env.n_blue
    n_pairs = env.n_blue * env.n_red
    shifted = list(base)
    shifted[n_prefix] += 1.0
    candidates.append(tuple(shifted))
    shifted[n_prefix] = 2.5  # fractional reading
    candidates.append(tuple(shifted))
    shifted = list(base)
    shifted[n_prefix] = -1.0  # below every table
    candidates.append(tuple(shifted))
    shifted = list(base)
    shifted[n_prefix + n_pairs] = 0.5  # not a bit
    candidates.append(tuple(shifted))
    shifted = list(base)
    shifted[0] += 1.0  # wrong blue position
    candidates.append(tuple(shifted))
    shifted = list(base)
    shifted[-1] += 1.0  # wrong score
    candidates.append(tuple(shifted))
    candidates.append(tuple(base[:-1]))  # wrong size
    candidates.append(np.asarray(base))  # an ndarray, not a tuple
    candidates.append([int(v) for v in base])  # Python ints
    candidates.append(tuple([-1.0] * env.observation_size))  # terminal sentinel
    candidates.append(tuple(float(v) for v in rng.integers(0, 3, size=env.observation_size)))
    return candidates


def test_observation_log_probability_is_identical(env: CaptureTheFlagPOMDP) -> None:
    """Observation log-likelihoods match the reference bit for bit.

    Purpose: The likelihood is the hot path of every belief update and of
        POMCPOW/PFT-DPW weighting.

    Given: Successor states (terminal ones included) and candidate
        observations that are likely, impossible or malformed
    When: Both score the candidates, singly and per state
    Then: The log-likelihoods are identical, ``-inf`` included

    Test type: unit
    """
    rng = np.random.default_rng(2)
    states = state_sample(env, n_random=40)
    for next_state in states:
        for action in action_sample(env, rng, 2):
            candidates = observation_candidates(env, next_state, action, rng)
            expected = env._python_observation_log_probability(next_state, action, candidates)
            actual = env.observation_log_probability(next_state, action, candidates)
            np.testing.assert_array_equal(actual, expected)

    action = 7
    observation = env._python_sample_observation(states[-1], action)
    batch = np.asarray(states)
    expected = np.array(
        [env._python_observation_log_probability(s, action, [observation])[0] for s in batch]
    )
    np.testing.assert_array_equal(
        env.observation_log_probability_per_state(batch, action, observation), expected
    )


def test_reward_is_identical(env: CaptureTheFlagPOMDP) -> None:
    """Rewards match the reference exactly, singly and in batches.

    Purpose: The reward has six stacked terms; operation order decides the
        last bit, and the native file turns off fused multiply-add for this.

    Given: Transitions drawn by the reference, from terminal and live states
    When: Both compute the reward
    Then: The values are identical

    Test type: unit
    """
    rng = np.random.default_rng(3)
    for action in action_sample(env, rng, 6):
        states = state_sample(env, n_random=40)
        next_states = [env._python_sample_next_state(s, action) for s in states]
        expected = np.array(
            [env._python_reward(s, action, ns) for s, ns in zip(states, next_states)]
        )
        actual = np.array([env.reward(s, action, ns) for s, ns in zip(states, next_states)])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            env.reward_batch(np.asarray(states), action, np.asarray(next_states)), expected
        )


def empirical(samples: List[Any]) -> Dict[Any, float]:
    """Return the frequency of each distinct sample."""
    counts: Dict[Any, int] = {}
    for sample in samples:
        key = sample.tobytes() if isinstance(sample, np.ndarray) else tuple(sample)
        counts[key] = counts.get(key, 0) + 1
    return {key: count / len(samples) for key, count in counts.items()}


def interesting_states(env: CaptureTheFlagPOMDP) -> List[Tuple[np.ndarray, int]]:
    """Return a few live ``(state, action)`` pairs with wide successor support."""
    rng = np.random.default_rng(4)
    pairs = []
    for state in state_sample(env, n_random=40):
        if env.is_terminal(state):
            continue
        action = int(rng.integers(len(env.get_actions())))
        _, probs = env._successor_distribution(state, action)
        if len(probs) >= 4:
            pairs.append((state, action))
        if len(pairs) == 3:
            break
    return pairs


def test_sampled_successors_follow_the_exact_distribution(env: CaptureTheFlagPOMDP) -> None:
    """Native successor samples follow the reference's enumerated distribution.

    Purpose: The sampler draws per player while the enumeration is joint; this
        is the check that the native per-player draw is the same model.

    Given: States whose successor distribution has several outcomes
    When: 20000 successors are drawn natively, singly and in a batch
    Then: Every outcome is in the support and frequencies match within 0.012

    Test type: unit
    """
    _native.set_seed(11)
    draws = 20000
    for state, action in interesting_states(env):
        ref_states, ref_probs = env._successor_distribution(state, action)
        exact = {s.tobytes(): float(p) for s, p in zip(ref_states, ref_probs)}
        for samples in (
            env.sample_next_state(state, action, draws),
            list(env.sample_next_state_batch(np.tile(state, (draws, 1)), action)),
        ):
            freq = empirical(samples)
            assert set(freq) <= set(exact)
            for key, probability in exact.items():
                assert freq.get(key, 0.0) == pytest.approx(probability, abs=0.012)


def test_sampled_observations_follow_the_likelihood(env: CaptureTheFlagPOMDP) -> None:
    """Native observation samples follow the reference likelihood.

    Purpose: The sampler and the likelihood are two code paths of one model.

    Given: A few successor states
    When: 20000 observations are drawn natively
    Then: Each distinct observation's frequency matches ``exp`` of the
        reference log-likelihood within 0.012, and none is impossible

    Test type: unit
    """
    _native.set_seed(12)
    draws = 20000
    for state, action in interesting_states(env):
        samples = env.sample_observation(state, action, draws)
        freq = empirical(samples)
        keys = list(freq)
        expected = np.exp(env._python_observation_log_probability(state, action, keys))
        assert np.all(expected > 0.0)
        for key, probability in zip(keys, expected):
            assert freq[key] == pytest.approx(probability, abs=0.012)


def test_sample_next_step_is_one_consistent_transition(env: CaptureTheFlagPOMDP) -> None:
    """``sample_next_step`` returns a reachable successor, a possible observation,
    and the reference reward of that very transition.

    Purpose: The native step fuses three calls; it must not score a different
        draw than it returns.

    Given: Live and terminal states
    When: Steps are drawn natively
    Then: Each successor has positive probability, each observation positive
        likelihood, and the reward equals the reference reward of the pair

    Test type: unit
    """
    _native.set_seed(13)
    rng = np.random.default_rng(13)
    for state in state_sample(env, n_random=30):
        action = int(rng.integers(len(env.get_actions())))
        next_state, observation, reward = env.sample_next_step(state, action)
        assert isinstance(observation, tuple)
        assert env._python_transition_log_probability(state, action, [next_state])[0] > -np.inf
        assert (
            env._python_observation_log_probability(next_state, action, [observation])[0] > -np.inf
        )
        assert reward == env._python_reward(state, action, next_state)


def test_native_rollout_matches_a_stepwise_native_loop(env: CaptureTheFlagPOMDP) -> None:
    """The C++ rollout equals stepping the native model with the same seed.

    Purpose: Pins the rollout's bookkeeping -- stop at terminal, discount
        order, one RNG draw sequence per step -- exactly.

    Given: Fixed joint actions and a fixed native seed
    When: The rollout kernel runs, and separately the same steps are taken
        through ``sample_next_state`` and ``reward``
    Then: The discounted returns are identical

    Test type: unit
    """
    model = env._native_model()
    rng = np.random.default_rng(14)
    gamma = 0.95
    for state in state_sample(env, n_random=20):
        actions = rng.integers(0, len(env.get_actions()), size=15, dtype=np.int64)
        _native.set_seed(99)
        native = model.simulate_rollout(state, actions, gamma)
        _native.set_seed(99)
        total, discount, current = 0.0, 1.0, state
        for action in actions:
            if env.is_terminal(current):
                break
            nxt = env.sample_next_state(current, int(action))
            total += discount * env.reward(current, int(action), nxt)
            discount *= gamma
            current = nxt
        assert native == total


def test_native_rollout_mean_matches_the_python_rollout() -> None:
    """The native random rollout has the reference rollout's mean return.

    Purpose: Planners read the rollout's value; the native one must estimate
        the same quantity as ``python_random_rollout`` with uniform actions.

    Given: The default environment and an opening state
    When: 3000 rollouts of depth 12 run each way
    Then: The means agree within four standard errors

    Test type: unit
    """
    env = CaptureTheFlagPOMDP()
    state = env.initial_state_dist().values[0]  # type: ignore[attr-defined]
    np.random.seed(0)
    _native.set_seed(0)
    sampler = DiscreteActionSampler(env.get_actions())
    n = 3000
    native = np.array(
        [
            env.simulate_random_rollout(state, sampler, max_depth=12, discount_factor=0.95)
            for _ in range(n)
        ]
    )
    python = np.array(
        [
            python_random_rollout(state, 0, sampler, env, discount_factor=0.95, max_depth=12)
            for _ in range(n)
        ]
    )
    stderr = np.sqrt(native.var() / n + python.var() / n)
    assert abs(native.mean() - python.mean()) < 4 * stderr
    assert (
        env.simulate_random_rollout(state, sampler, max_depth=3, discount_factor=0.95, depth=3)
        == 0.0
    )


def test_set_seed_reproduces_native_draws() -> None:
    """Seeding the native RNG reproduces every native sampler.

    Purpose: Reproducible runs must seed this module as well as numpy.

    Given: One state and action
    When: The native RNG is seeded twice with the same value
    Then: Successors, observations and steps are identical

    Test type: unit
    """
    env = CaptureTheFlagPOMDP()
    state = env.initial_state_dist().values[1]  # type: ignore[attr-defined]

    def draw() -> Tuple[Any, ...]:
        next_state = env.sample_next_state(state, 13)
        return (
            next_state.tobytes(),
            env.sample_observation(next_state, 13),
            env.sample_next_step(next_state, 20)[1],
        )

    _native.set_seed(2024)
    first = draw()
    _native.set_seed(2024)
    assert draw() == first


def test_native_model_does_not_touch_identity_or_pickling() -> None:
    """Building the native model leaves ``config_id`` and pickling intact.

    Purpose: The model is cached on the instance; a cache that leaked into
        ``config_id`` would change result-cache keys, and a pybind11 object
        cannot be pickled for worker processes.

    Given: A fresh environment
    When: It is used through the native model, then pickled and reloaded
    Then: ``config_id`` is unchanged, the copy compares equal, and it samples

    Test type: unit
    """
    env = CaptureTheFlagPOMDP()
    fresh_id = env.config_id
    state = env.initial_state_dist().values[0]  # type: ignore[attr-defined]
    env.sample_next_step(state, 3)
    assert env.config_id == fresh_id
    copy = pickle.loads(pickle.dumps(env))
    assert copy == env
    assert copy.config_id == fresh_id
    assert copy.sample_next_state(state, 3).shape == state.shape


def test_large_teams_fall_back_to_the_python_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A team larger than the native buffers runs the Python reference.

    Purpose: The C++ scratch is fixed-size; above it the environment must
        still work, not fail.

    Given: The native team limit lowered below the team size
    When: The environment samples, scores and rolls out
    Then: No native model is built and the reference results come back

    Test type: unit
    """
    monkeypatch.setattr(_native, "MAX_PLAYERS_PER_TEAM", 1)
    env = CaptureTheFlagPOMDP()
    assert env._native_model() is None
    state = env.initial_state_dist().values[0]  # type: ignore[attr-defined]
    np.random.seed(3)
    next_state, observation, reward = env.sample_next_step(state, 5)
    assert reward == env._python_reward(state, 5, next_state)
    assert env.observation_log_probability(next_state, 5, [observation])[0] > -np.inf
    sampler = DiscreteActionSampler(env.get_actions())
    assert np.isfinite(env.simulate_random_rollout(state, sampler, 5, 0.95))
