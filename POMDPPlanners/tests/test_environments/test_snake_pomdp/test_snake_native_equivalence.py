# SPDX-License-Identifier: MIT

"""The C++ Snake kernels agree with the Python code they replaced.

``SnakePOMDP`` sends transition, observation, reward and rollout calls to
``snake_pomdp/_native``. ``SnakePOMDPPythonReference`` keeps the pre-port
Python code. The two draw from different RNGs (numpy versus the module's C++
engine), so:

* deterministic outputs -- rewards, terminal checks, non-eating transitions,
  transition and observation log-probabilities -- must match exactly, or to
  one ulp where a ``log`` is involved;
* sampled outputs -- the respawn cell and the reading -- are compared by
  frequency against the reference's own probabilities.

The states come from random episodes of the reference environment plus
hand-built states for each rule (eat, win, wall, self, starvation, the tail
cell), so both live and terminal states of every kind are covered.
"""

# pylint: disable=protected-access

import pickle
from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.snake_pomdp import _native
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    TERMINAL_OBSERVATION,
    SnakePOMDP,
    SnakeTermination,
    create_snake_state,
)
from POMDPPlanners.tests.test_environments.test_snake_pomdp.snake_python_reference import (
    SnakePOMDPPythonReference,
)

_KWARGS = {
    "grid_size": 7,
    "target_length": 6,
    "window_radius": 2,
    "detection_probability": 0.9,
    "scent_accuracy": 0.7,
    "starvation_limit": 12,
    "discount_factor": 0.98,
}
_ACTIONS = (0, 1, 2)


def _envs(**overrides) -> Tuple[SnakePOMDP, SnakePOMDPPythonReference]:
    kwargs = dict(_KWARGS)
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs), SnakePOMDPPythonReference(**kwargs)


def _hand_built_states(env: SnakePOMDP) -> List[np.ndarray]:
    """One state per rule the transition distinguishes."""

    def state(body, food, counter=0, status=0):
        return create_snake_state(
            body=body,
            food=food,
            steps_since_food=counter,
            target_length=env.target_length,
            status=status,
        )

    return [
        # Food straight ahead: going straight eats and respawns.
        state([(3, 3), (3, 2), (3, 1)], (3, 4)),
        # Length 5 eating reaches the target length 6: a win, no respawn.
        state([(3, 3), (3, 2), (3, 1), (3, 0), (2, 0)], (3, 4)),
        # Head on the east wall: going straight leaves the grid.
        state([(3, 6), (3, 5), (3, 4)], (0, 0)),
        # A coiled body: turning right enters the neck's neighbour.
        state([(2, 2), (2, 3), (3, 3), (3, 2), (2, 1)], (6, 6)),
        # Tail-follow: turning into the cell the tail is leaving is legal.
        state([(2, 2), (2, 3), (3, 3), (3, 2)], (6, 6)),
        # One step from starvation.
        state([(3, 3), (3, 2), (3, 1)], (0, 6), counter=11),
        # Food in the head's row and column (scent ties) and inside the window.
        state([(3, 3), (3, 2), (3, 1)], (3, 5)),
        state([(3, 3), (3, 2), (3, 1)], (1, 3)),
        # Terminal states of every kind.
        state([(3, 7), (3, 6), (3, 5)], (0, 0), status=int(SnakeTermination.WALL)),
        state([(3, 3), (3, 2), (3, 1)], (0, 0), status=int(SnakeTermination.SELF)),
        state([(3, 3), (3, 2), (3, 1)], (0, 0), status=int(SnakeTermination.STARVATION)),
        state(
            [(3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)],
            None,
            status=int(SnakeTermination.WIN),
        ),
    ]


def _episode_states(reference: SnakePOMDPPythonReference, n_states: int) -> List[np.ndarray]:
    """States visited by uniformly random episodes of the reference env."""
    rng_state = np.random.get_state()
    np.random.seed(1234)
    states: List[np.ndarray] = []
    try:
        while len(states) < n_states:
            current = reference.initial_state_dist().sample()[0]
            for _ in range(40):
                states.append(current)
                if reference.is_terminal(current):
                    break
                current = reference.sample_next_state(current, int(np.random.randint(3)))
    finally:
        np.random.set_state(rng_state)
    return states[:n_states]


@pytest.fixture(name="envs", scope="module")
def _envs_fixture() -> Tuple[SnakePOMDP, SnakePOMDPPythonReference]:
    return _envs()


@pytest.fixture(name="states", scope="module")
def _states_fixture(envs) -> List[np.ndarray]:
    native, reference = envs
    return _hand_built_states(native) + _episode_states(reference, 400)


def _assert_log_probs_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-14, atol=0.0)


def _respawns(reference: SnakePOMDPPythonReference, state, action) -> bool:
    if reference.is_terminal(state):
        return False
    _, eat, _, termination = reference.transition_outcome(state, action)
    return eat and termination is SnakeTermination.RUNNING


def test_the_state_sample_covers_every_outcome(envs, states):
    """The fixtures reach eat, win, wall, self and starvation, so a rule is not skipped."""
    _, reference = envs
    seen = set()
    for state in states:
        if reference.is_terminal(state):
            seen.add(("terminal", reference.termination(state)))
            continue
        for action in _ACTIONS:
            _, eat, _, termination = reference.transition_outcome(state, action)
            seen.add(("eat", eat))
            seen.add(("termination", termination))
    for termination in SnakeTermination:
        assert ("termination", termination) in seen
    assert ("eat", True) in seen


def test_reward_and_terminal_match_exactly(envs, states):
    native, reference = envs
    for state in states:
        assert native.is_terminal(state) == reference.is_terminal(state)
        for action in _ACTIONS:
            assert native.reward(state, action) == reference.reward(state, action)


def test_reward_batch_matches_the_reference(envs, states):
    native, reference = envs
    rows = np.stack(states)
    for action in _ACTIONS:
        actual = native.reward_batch(rows, action)
        expected = reference.reward_batch(rows, action)
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected)


def test_transitions_without_a_respawn_match_exactly(envs, states):
    """When nothing respawns the successor is a function of (state, action)."""
    native, reference = envs
    for state in states:
        for action in _ACTIONS:
            if _respawns(reference, state, action):
                continue
            actual = native.sample_next_state(state, action)
            expected = reference.sample_next_state(state, action)
            assert actual.dtype == np.float64
            np.testing.assert_array_equal(actual, expected)


def test_respawned_states_match_except_for_the_food_cell(envs, states):
    native, reference = envs
    checked = 0
    for state in states:
        for action in _ACTIONS:
            if not _respawns(reference, state, action):
                continue
            actual = native.sample_next_state(state, action)
            expected = reference.sample_next_state(state, action)
            mask = np.ones(actual.shape, dtype=bool)
            mask[[3, 4]] = False
            np.testing.assert_array_equal(actual[mask], expected[mask])
            new_body = reference.body(actual)
            cell = int(actual[3]) * native.grid_size + int(actual[4])
            assert cell in set(reference.free_cells(new_body).tolist())
            checked += 1
    assert checked > 0


def test_the_respawn_cell_is_uniform_over_the_free_cells(envs):
    native, reference = envs
    state = _hand_built_states(native)[0]
    _native.set_seed(7)
    n_draws = 40_000
    samples = native.sample_next_state(state, 1, n_samples=n_draws)
    assert samples.shape == (n_draws, native.state_size)
    cells = (samples[:, 3] * native.grid_size + samples[:, 4]).astype(int)
    new_body = reference.transition_outcome(state, 1)[0]
    free = reference.free_cells(new_body)
    counts = np.array([np.sum(cells == cell) for cell in free])
    assert counts.sum() == n_draws  # nothing lands outside the free cells
    expected = n_draws / free.size
    # Each count is Binomial(n, 1/k); 5 standard deviations bounds all k at once.
    sigma = np.sqrt(n_draws * (1 / free.size) * (1 - 1 / free.size))
    assert np.max(np.abs(counts - expected)) < 5 * sigma


def test_transition_log_probability_matches_the_reference(envs, states):
    native, reference = envs
    for state in states[:150]:
        for action in _ACTIONS:
            candidates = [reference.sample_next_state(state, action) for _ in range(3)]
            candidates.append(np.asarray(state, dtype=np.float64))
            shifted = np.array(candidates[0], copy=True)
            shifted[3] = (shifted[3] + 1) % native.grid_size
            candidates.append(shifted)
            off_grid = np.array(candidates[0], copy=True)
            off_grid[3] = 2.5
            candidates.append(off_grid)
            rows = np.stack(candidates)
            _assert_log_probs_equal(
                native.transition_log_probability(state, action, rows),
                reference.transition_log_probability(state, action, rows),
            )


def _candidate_readings(reference: SnakePOMDPPythonReference, next_state) -> List[tuple]:
    """Every reading the sensor can emit for ``next_state``, plus impossible ones."""
    readings: List[tuple] = [TERMINAL_OBSERVATION, (1,) + (0,) * 3]
    if reference.is_terminal(next_state):
        body = ((3, 3), (3, 2), (3, 1))
        readings.append(reference.encode_observation_tuple(body, None, 0))
        return readings
    body = reference.body(next_state)
    food = reference.food(next_state)
    other = (0, 0) if food != (0, 0) else (0, 1)
    for seen in (None, food, other):
        for scent in range(-1, 5):
            readings.append(reference.encode_observation_tuple(body, seen, scent))
    readings.append(reference.encode_observation_tuple(body[:-1], None, 0))
    return readings


def test_observation_log_probability_matches_the_reference(envs, states):
    native, reference = envs
    for state in states:
        readings = _candidate_readings(reference, state)
        _assert_log_probs_equal(
            native.observation_log_probability(state, 0, readings),
            reference.observation_log_probability(state, 0, readings),
        )
        # A bare reading is scored as one reading, not as a list of ints.
        _assert_log_probs_equal(
            native.observation_log_probability(state, 0, readings[-1]),
            reference.observation_log_probability(state, 0, readings[-1]),
        )


@pytest.mark.parametrize("state_index", [0, 2, 6, 7])
def test_sampled_readings_follow_the_reference_likelihood(envs, state_index):
    """Reading frequencies match exp(reference log-likelihood) within 5 sigma."""
    native, reference = envs
    # Turning left from these hand-built states leaves the food inside the
    # window (0, 6, 7; 7 is a scent tie) or outside it (2).
    state = _hand_built_states(native)[state_index]
    next_state = reference.sample_next_state(state, 0)
    _native.set_seed(11)
    n_draws = 40_000
    readings = native.sample_observation(next_state, 0, n_samples=n_draws)
    counts = {}
    for reading in readings:
        assert isinstance(reading, tuple)
        counts[reading] = counts.get(reading, 0) + 1
    support = list(counts)
    probabilities = np.exp(reference.observation_log_probability(next_state, 0, support))
    assert np.all(probabilities > 0.0)  # nothing impossible is drawn
    total = 0.0
    for reading, probability in zip(support, probabilities):
        sigma = np.sqrt(n_draws * probability * (1 - probability))
        assert abs(counts[reading] - n_draws * probability) < 5 * sigma + 1
        total += probability
    # Every reading with mass 1e-3 or more was drawn.
    assert total > 0.999


def test_terminal_states_emit_the_terminal_reading(envs, states):
    native, reference = envs
    for state in states:
        if reference.is_terminal(state):
            assert native.sample_observation(state, 1) == TERMINAL_OBSERVATION
            assert native.sample_observation(state, 1, n_samples=3) == [TERMINAL_OBSERVATION] * 3


def test_sample_next_step_makes_the_same_draws_as_the_separate_calls(envs, states):
    native, _ = envs
    for index, state in enumerate(states[:200]):
        for action in _ACTIONS:
            _native.set_seed(index)
            fused = native.sample_next_step(state, action)
            _native.set_seed(index)
            next_state = native.sample_next_state(state, action)
            observation = native.sample_observation(next_state, action)
            reward = native.reward(state, action)
            np.testing.assert_array_equal(fused[0], next_state)
            assert fused[1] == observation
            assert fused[2] == reward
            assert isinstance(fused[2], float)


def test_batch_sampling_matches_single_sampling(envs, states):
    native, _ = envs
    rows = np.stack(states)
    for action in _ACTIONS:
        _native.set_seed(3)
        batch = native.sample_next_state_batch(rows, action)
        _native.set_seed(3)
        single = np.stack([native.sample_next_state(row, action) for row in rows])
        assert batch.dtype == np.float64
        np.testing.assert_array_equal(batch, single)


def test_the_native_rollout_is_the_native_step_loop(envs, states):
    """simulate_rollout_discrete sums the same draws a step-by-step loop makes."""
    native, _ = envs
    discount = 0.9
    for index, state in enumerate(states[:120]):
        actions = np.random.RandomState(index).randint(0, 3, size=25).astype(np.int32)
        _native.set_seed(index)
        rollout = _native.simulate_rollout_discrete(
            state,
            actions,
            30,
            5,
            discount,
            native.grid_size,
            native.target_length,
            native.starvation_limit,
        )
        _native.set_seed(index)
        total, factor, current = 0.0, 1.0, np.asarray(state, dtype=np.float64)
        for action in actions:
            if native.is_terminal(current):
                break
            reward = native.reward(current, int(action))
            current = native.sample_next_state(current, int(action))
            total += factor * reward
            factor *= discount
        assert rollout == pytest.approx(total, rel=1e-12, abs=1e-12)


def test_the_rollout_mean_matches_the_python_rollout(envs):
    native, reference = envs

    class _Uniform:
        def sample(self):
            return int(np.random.randint(3))

    state = _hand_built_states(native)[0]
    np.random.seed(5)
    _native.set_seed(5)
    n_rollouts = 4_000
    native_values = [
        native.simulate_random_rollout(state, _Uniform(), 30, 0.95) for _ in range(n_rollouts)
    ]
    reference_values = [
        reference.simulate_random_rollout(state, _Uniform(), 30, 0.95) for _ in range(n_rollouts)
    ]
    standard_error = np.sqrt((np.var(native_values) + np.var(reference_values)) / n_rollouts)
    assert abs(np.mean(native_values) - np.mean(reference_values)) < 5 * standard_error


def test_the_rollout_respects_depth_and_terminal_states(envs):
    native, _ = envs
    states = _hand_built_states(native)
    assert native.simulate_random_rollout(states[0], None, 10, 0.9, depth=10) == 0.0
    assert native.simulate_random_rollout(states[-1], None, 10, 0.9) == 0.0
    # Going straight from the east wall dies on the first step.
    value = _native.simulate_rollout_discrete(
        states[2],
        np.array([1], dtype=np.int32),
        1,
        0,
        0.9,
        native.grid_size,
        native.target_length,
        native.starvation_limit,
    )
    assert value == -1.0


def test_invalid_inputs_raise_like_the_reference(envs):
    native, reference = envs
    state = _hand_built_states(native)[0]
    for env in (native, reference):
        with pytest.raises(ValueError):
            env.sample_next_state(state, 3)
        with pytest.raises(ValueError):
            env.reward(state, -1)
    no_heading = create_snake_state(body=[(3, 3), (3, 5)], food=(0, 0), target_length=6)
    for env in (native, reference):
        with pytest.raises(ValueError):
            env.sample_next_state(no_heading, 1)


def test_set_seed_makes_native_draws_repeat(envs):
    native, _ = envs
    state = _hand_built_states(native)[0]
    _native.set_seed(99)
    first = [native.sample_next_step(state, 1) for _ in range(20)]
    _native.set_seed(99)
    second = [native.sample_next_step(state, 1) for _ in range(20)]
    for one, two in zip(first, second):
        np.testing.assert_array_equal(one[0], two[0])
        assert one[1:] == two[1:]


def test_kernels_stay_out_of_identity_and_pickling():
    native, _ = _envs()
    config_id = native.config_id
    state = _hand_built_states(native)[0]
    native.sample_next_step(state, 1)
    native.sample_next_state(state, 0)
    native.observation_log_probability(state, 0, [TERMINAL_OBSERVATION])
    assert native._trans_kernel_cache and native._obs_kernel_cache
    assert native.config_id == config_id
    restored = pickle.loads(pickle.dumps(native))
    assert restored.config_id == config_id
    assert restored._trans_kernel_cache == {}
    restored.sample_next_state(state, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"detection_probability": 1.0, "scent_accuracy": 1.0},
        {"detection_probability": 0.0, "scent_accuracy": 0.25},
        {"window_radius": 0},
        {"grid_size": 4, "target_length": 5, "starvation_limit": 3},
    ],
)
def test_edge_sensor_settings_match_the_reference(overrides):
    native, reference = _envs(**overrides)
    states = _episode_states(reference, 120)
    for state in states:
        readings = _candidate_readings(reference, state)
        # A probability of 0 or 1 makes the reference take log(0), which numpy
        # reports as a divide warning; the value, -inf, is what is compared.
        with np.errstate(divide="ignore"):
            expected = reference.observation_log_probability(state, 0, readings)
        _assert_log_probs_equal(native.observation_log_probability(state, 0, readings), expected)
        for action in _ACTIONS:
            assert native.reward(state, action) == reference.reward(state, action)
            if not _respawns(reference, state, action):
                np.testing.assert_array_equal(
                    native.sample_next_state(state, action),
                    reference.sample_next_state(state, action),
                )
