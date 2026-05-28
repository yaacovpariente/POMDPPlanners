# SPDX-License-Identifier: MIT

"""Parity tests for the native discrete Push transition kernel.

Verifies that the C++ ``PushDiscreteTransitionCpp`` deterministic
transition (used by ``PushPOMDP._compute_next_state_for_action``) matches
the pure-Python reference (kept as ``_compute_next_state_for_action_python``)
bit-for-bit across the action set, the closed-form
``transition_log_probability``, and a pickle round-trip.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import push_pinned_kwargs


@pytest.fixture(name="env")
def _env_fixture() -> PushPOMDP:
    return PushPOMDP(
        discount_factor=0.95,
        **push_pinned_kwargs(
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            obstacles=[(5.0, 5.0), (3.0, 7.0)],
            obstacle_radius=0.5,
        ),
    )


@pytest.fixture(name="state_batch")
def _state_batch_fixture() -> np.ndarray:
    """Twelve-row state batch covering free-move, push, blocked, edge, corner.

    Layout: each row is [robot_x, robot_y, object_x, object_y, target_x, target_y].
    """
    return np.array(
        [
            # free-move (no push, no obstacle)
            [1.0, 1.0, 8.0, 8.0, 9.0, 9.0],
            # push: robot adjacent to object
            [2.0, 4.0, 3.0, 4.0, 9.0, 9.0],
            # push at fractional distance
            [2.5, 4.0, 3.0, 4.0, 9.0, 9.0],
            # robot blocked by obstacle moving right (obstacle at (5,5))
            [4.5, 5.0, 7.0, 7.0, 9.0, 9.0],
            # robot blocked by obstacle moving up (obstacle at (3,7))
            [3.0, 6.5, 8.0, 8.0, 9.0, 9.0],
            # push-blocked by obstacle at (5,5): object at (4,5), push right
            [3.0, 5.0, 4.0, 5.0, 9.0, 9.0],
            # near top-right grid edge
            [9.0, 9.0, 0.0, 0.0, 9.0, 9.0],
            # near bottom-left grid edge
            [0.0, 0.0, 5.0, 5.0, 9.0, 9.0],
            # near corner with obstacle nearby
            [0.0, 8.0, 5.0, 5.0, 9.0, 9.0],
            # robot far from object (no push possible)
            [0.5, 0.5, 7.0, 7.0, 9.0, 9.0],
            # push that bumps object into another obstacle (3,7): object at (3,6), push up
            [3.0, 5.5, 3.0, 6.0, 9.0, 9.0],
            # robot exactly on grid corner pushing into wall
            [9.0, 0.0, 5.0, 5.0, 9.0, 9.0],
        ],
        dtype=np.float64,
    )


def test_native_transition_matches_python_for_every_action(
    env: PushPOMDP, state_batch: np.ndarray
) -> None:
    """Native deterministic transition matches the Python reference.

    Purpose: Validates that ``_compute_next_state_for_action`` (native dispatch)
    returns the same next state as the kept Python reference for every
    (state, action) pair across a representative 12-row state batch.

    Given: A PushPOMDP with two obstacles and a 12-row state batch covering
        free-move, push, robot-blocked, push-blocked, edge and corner cases.
    When: The native and Python implementations are evaluated for every
        (row, action) combination.
    Then: The two next states are equal up to atol=1e-12 across all cases.

    Test type: integration
    """
    actions = env.get_actions()
    for row in state_batch:
        for action in actions:
            native_next = env._compute_next_state_for_action(  # pylint: disable=protected-access
                row, action
            )
            python_next = (
                env._compute_next_state_for_action_python(  # pylint: disable=protected-access
                    row, action
                )
            )
            assert np.allclose(native_next, python_next, atol=1e-12), (
                f"mismatch: action={action} state={row} "
                f"native={native_next} python={python_next}"
            )


def test_native_transition_log_probability_matches_python_reference() -> None:
    """transition_log_probability with native kernel matches the closed-form spec.

    Purpose: Validates that the native-routed
    ``PushPOMDP.transition_log_probability`` matches the analytic
    closed-form (1 - p_err) * 1[next == intended] + p_err * (1 / n_err) *
    sum_{a' != a} 1[next == intended(s, a')] for representative
    next-state candidates including the intended next, error candidates,
    and unrelated states.

    Given: A PushPOMDP with non-zero transition_error_prob and a state where
        each action produces a distinct next state.
    When: ``transition_log_probability`` is called with a candidate list
        containing the intended next, an error-action next, and an
        unrelated state.
    Then: The returned log-probabilities equal the closed-form values
        recomputed from the kept Python reference up to atol=1e-12.

    Test type: integration
    """
    env = PushPOMDP(
        discount_factor=0.95,
        **push_pinned_kwargs(
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            obstacles=[],
            transition_error_prob=0.2,
        ),
    )
    state = np.array([4.0, 4.0, 6.0, 6.0, 9.0, 9.0])
    intended_action = "right"

    # Build expected next-states using the kept Python reference so the
    # comparison does not pass through the kernel under test.
    intended_next = env._compute_next_state_for_action_python(  # pylint: disable=protected-access
        state, intended_action
    )
    error_actions = [a for a in env.get_actions() if a != intended_action]
    error_results = [
        env._compute_next_state_for_action_python(state, a)  # pylint: disable=protected-access
        for a in error_actions
    ]
    unrelated_state = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 9.0])

    candidates = [
        intended_next,
        error_results[0],
        error_results[1],
        error_results[2],
        unrelated_state,
    ]

    p_err = env.transition_error_prob
    n_err = len(error_actions)
    expected_probs = []
    for cand in candidates:
        intended_match = 1.0 if np.array_equal(cand, intended_next) else 0.0
        error_match_count = float(sum(1 for er in error_results if np.array_equal(cand, er)))
        prob = (1.0 - p_err) * intended_match + p_err * (error_match_count / n_err)
        expected_probs.append(prob)
    with np.errstate(divide="ignore"):
        expected_log = np.log(np.asarray(expected_probs, dtype=float))

    log_probs = env.transition_log_probability(state, intended_action, candidates)
    np.testing.assert_allclose(log_probs, expected_log, atol=1e-12, equal_nan=True)


def test_native_transition_log_probability_zero_error_matches_python(
    env: PushPOMDP, state_batch: np.ndarray
) -> None:
    """transition_log_probability with p_err=0 collapses to the intended branch.

    Purpose: Validates that with ``transition_error_prob == 0`` the native
    log-probability equals 0 on the intended next and -inf on any other
    candidate, across the 12-row state batch.

    Given: A PushPOMDP fixture with ``transition_error_prob == 0``.
    When: ``transition_log_probability`` is called with the intended next
        and an unrelated state for every action.
    Then: log-prob equals 0.0 on intended and -inf on the unrelated state.

    Test type: integration
    """
    actions = env.get_actions()
    unrelated = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 9.0])
    for row in state_batch:
        for action in actions:
            intended = (
                env._compute_next_state_for_action_python(  # pylint: disable=protected-access
                    row, action
                )
            )
            log_probs = env.transition_log_probability(row, action, [intended, unrelated])
            assert np.isclose(log_probs[0], 0.0, atol=1e-12)
            # Unrelated may coincidentally equal intended only if intended
            # collapses to it; skip strict -inf check in that pathological
            # case (no row in the batch hits this).
            if not np.array_equal(intended, unrelated):
                assert np.isneginf(log_probs[1])


def test_native_transition_pickle_round_trip(env: PushPOMDP, state_batch: np.ndarray) -> None:
    """Pickle round-trip drops kernel cache yet preserves transition parity.

    Purpose: Validates that ``__getstate__`` / ``__setstate__`` strip the
    pybind11 kernel cache so the env survives pickling, and that the
    restored env produces the same next states as before pickling.

    Given: A configured PushPOMDP whose kernel cache has been warmed up by
        evaluating one transition per action.
    When: The env is pickled and unpickled.
    Then: The kernel cache is empty after restore, and
        ``_compute_next_state_for_action`` agrees with the original env on
        every (row, action) up to atol=1e-12.

    Test type: integration
    """
    # Warm the cache: every action label gets a kernel built.
    for action in env.get_actions():
        env._compute_next_state_for_action(  # pylint: disable=protected-access
            state_batch[0], action
        )
    assert env._trans_kernel_cache  # pylint: disable=protected-access

    blob = pickle.dumps(env)
    restored = pickle.loads(blob)
    assert restored._trans_kernel_cache == {}  # pylint: disable=protected-access

    actions = env.get_actions()
    for row in state_batch:
        for action in actions:
            before = env._compute_next_state_for_action(  # pylint: disable=protected-access
                row, action
            )
            after = restored._compute_next_state_for_action(  # pylint: disable=protected-access
                row, action
            )
            assert before.shape == (6,)
            assert after.shape == (6,)
            np.testing.assert_allclose(after, before, atol=1e-12)
