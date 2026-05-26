# SPDX-License-Identifier: MIT

"""Native single-step parity tests for DiscreteLightDarkPOMDP.

Verifies that the C++ ``discrete_sample_next_state_step`` and
``discrete_sample_observation_step_normal`` entries:

1. Preserve byte-identical numpy RNG state vs the original Python path.
2. Produce empirical sampling distributions that match the cumulative
   probability tables they consume.
3. Actually fire on the ``n_samples == 1`` hot path (rather than silently
   falling back to the Python implementation).

The Python n_samples > 1 path always runs the original numpy implementation
and serves as the reference for the byte-identical RNG parity checks: the
same sequence of ``np.random.rand()`` draws must produce the same sequence
of sampled values.
"""

from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)


def _make_env() -> DiscreteLightDarkPOMDP:
    return DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        beacons=[(0, 0), (5, 5)],
        beacon_radius=1.5,
        observation_model_type=ObservationModelType.NORMAL,
    )


def _to_tuple(value: np.ndarray) -> Tuple[float, float]:
    return (float(value[0]), float(value[1]))


def test_native_entries_are_exposed() -> None:
    """The C++ extension exposes both new single-step entries.

    Purpose: Validates that the build picked up both new free functions
        and that the Python wrapper can actually take the native fast path
        (rather than silently falling back to Python).

    Given: The compiled ``_native`` extension module.
    When: We probe for the new function attributes by name.
    Then: Both ``discrete_sample_next_state_step`` and
        ``discrete_sample_observation_step_normal`` are callable attributes.

    Test type: unit
    """
    assert hasattr(_native, "discrete_sample_next_state_step")
    assert hasattr(_native, "discrete_sample_observation_step_normal")
    assert callable(_native.discrete_sample_next_state_step)
    assert callable(_native.discrete_sample_observation_step_normal)


@pytest.mark.parametrize("action", ["up", "down", "left", "right"])
def test_native_sample_next_state_rng_parity_vs_batch_path(action: str) -> None:
    """Single-sample native path matches the n_samples>1 numpy path under a fixed seed.

    Purpose: Validates byte-identical numpy RNG state across the new native
        ``n_samples == 1`` fast path and the unchanged ``n_samples > 1``
        numpy path. Both paths consume one ``np.random.rand()`` per sample
        in order; under a fixed seed the per-sample outputs must match.

    Given: A DiscreteLightDarkPOMDP env with the NORMAL observation model
        and a fixed start state.
    When: We draw 50 samples one-at-a-time via the native fast path under
        seed 42, then re-seed and draw 50 samples in a single batch call.
    Then: The two sequences are equal element-wise.

    Test type: unit
    """
    env = _make_env()
    state = np.array([2, 3])
    n_draws = 50

    np.random.seed(42)
    one_at_a_time: List[Tuple[float, float]] = [
        _to_tuple(env.sample_next_state(state, action, n_samples=1)) for _ in range(n_draws)
    ]

    np.random.seed(42)
    batch = env.sample_next_state(state, action, n_samples=n_draws)

    assert isinstance(batch, np.ndarray)
    assert batch.shape == (n_draws, 2)
    for i, draw in enumerate(one_at_a_time):
        assert draw == (float(batch[i, 0]), float(batch[i, 1]))


def test_native_sample_observation_rng_parity_vs_batch_path() -> None:
    """Single-sample native observation path matches n_samples>1 numpy path under fixed seed.

    Purpose: Same byte-identical RNG parity check as for sample_next_state,
        but for ``sample_observation`` under the NORMAL model.

    Given: A DiscreteLightDarkPOMDP env (NORMAL) and a near-beacon
        next_state at (5, 5).
    When: We draw 50 single samples via the native fast path under seed 42,
        then re-seed and draw 50 samples in one batch.
    Then: The per-sample outputs match.

    Test type: unit
    """
    env = _make_env()
    next_state = np.array([5, 5])
    action = "up"
    n_draws = 50

    np.random.seed(42)
    one_at_a_time: List[Tuple[float, float]] = [
        _to_tuple(env.sample_observation(next_state, action, n_samples=1)) for _ in range(n_draws)
    ]

    np.random.seed(42)
    batch = env.sample_observation(next_state, action, n_samples=n_draws)

    assert isinstance(batch, np.ndarray)
    assert batch.shape == (n_draws, 2)
    for i, draw in enumerate(one_at_a_time):
        assert draw == (float(batch[i, 0]), float(batch[i, 1]))


def test_native_sample_next_state_empirical_distribution_matches_cumprobs() -> None:
    """Empirical next-state distribution matches the action's transition cumprobs.

    Purpose: Validates that the native single-step sampler produces samples
        whose empirical frequencies converge to the same per-action
        candidate distribution that the Python implementation does.

    Given: A DiscreteLightDarkPOMDP env with transition_error_prob=0.1 and
        the action "up".
    When: We draw 5000 single samples from a non-edge starting state.
    Then: The "up"-direction success outcome (state + (0, 1)) appears with
        empirical frequency within 0.03 of (1 - transition_error_prob).

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.1,
        beacons=[(0, 0)],
        observation_model_type=ObservationModelType.NORMAL,
    )
    state = np.array([3, 4])
    success_offset = env.action_to_vector["up"]
    success_outcome = (
        float(state[0] + success_offset[0]),
        float(state[1] + success_offset[1]),
    )

    np.random.seed(123)
    n_draws = 5000
    counts: Counter[Tuple[float, float]] = Counter()
    for _ in range(n_draws):
        sample = env.sample_next_state(state, "up", n_samples=1)
        counts[_to_tuple(sample)] += 1

    success_freq = counts[success_outcome] / n_draws
    expected_success = 1.0 - env.transition_error_prob
    assert abs(success_freq - expected_success) < 0.03


def test_native_sample_observation_empirical_distribution_matches_cumprobs() -> None:
    """Empirical observation distribution matches near/far cumprobs.

    Purpose: Validates that the native single-step observation sampler
        produces empirical frequencies consistent with the precomputed
        near-beacon cumulative probability table.

    Given: A DiscreteLightDarkPOMDP env (NORMAL) with a near-beacon
        next_state and a small observation_error_prob.
    When: We draw 5000 single observations.
    Then: The "no noise" outcome (idx == n_actions, observation == next_state)
        appears with empirical frequency within 0.03 of
        ``1 - observation_error_prob * 0.2`` (the near-beacon factor baked
        into ``_obs_probs_near``).

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_error_prob=0.2,
        beacons=[(5, 5)],
        beacon_radius=1.5,
        observation_model_type=ObservationModelType.NORMAL,
    )
    next_state = np.array([5, 5])
    no_noise_outcome = (5.0, 5.0)

    np.random.seed(7)
    n_draws = 5000
    no_noise_count = 0
    for _ in range(n_draws):
        sample = env.sample_observation(next_state, "up", n_samples=1)
        if _to_tuple(sample) == no_noise_outcome:
            no_noise_count += 1

    no_noise_freq = no_noise_count / n_draws
    expected_no_noise = 1.0 - env.observation_error_prob * 0.2
    assert abs(no_noise_freq - expected_no_noise) < 0.03


def test_native_sample_next_state_path_is_invoked() -> None:
    """The native single-step entry actually fires on the n_samples == 1 path.

    Purpose: Guards against silent fallback to the Python implementation by
        wrapping the C++ entry with a counter and asserting the counter
        increments per single-sample call.

    Given: A DiscreteLightDarkPOMDP env and the native module patched so
        ``discrete_sample_next_state_step`` records each invocation.
    When: We call ``sample_next_state(state, action, n_samples=1)`` 7 times.
    Then: The native counter equals 7.

    Test type: unit
    """
    env = _make_env()
    state = np.array([2, 3])
    original = _native.discrete_sample_next_state_step
    counter = {"n": 0}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        counter["n"] += 1
        return original(*args, **kwargs)

    _native.discrete_sample_next_state_step = wrapped  # type: ignore[assignment]
    try:
        for _ in range(7):
            env.sample_next_state(state, "up", n_samples=1)
    finally:
        _native.discrete_sample_next_state_step = original  # type: ignore[assignment]

    assert counter["n"] == 7


def test_native_sample_observation_path_is_invoked() -> None:
    """The native single-step observation entry actually fires on n_samples == 1.

    Purpose: Same as the previous test but for sample_observation under the
        NORMAL observation model.

    Given: A DiscreteLightDarkPOMDP env (NORMAL) and the native module
        patched so ``discrete_sample_observation_step_normal`` records each
        invocation.
    When: We call ``sample_observation(next_state, action, n_samples=1)``
        9 times.
    Then: The native counter equals 9.

    Test type: unit
    """
    env = _make_env()
    next_state = np.array([5, 5])
    original = _native.discrete_sample_observation_step_normal
    counter = {"n": 0}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        counter["n"] += 1
        return original(*args, **kwargs)

    _native.discrete_sample_observation_step_normal = wrapped  # type: ignore[assignment]
    try:
        for _ in range(9):
            env.sample_observation(next_state, "up", n_samples=1)
    finally:
        _native.discrete_sample_observation_step_normal = original  # type: ignore[assignment]

    assert counter["n"] == 9
