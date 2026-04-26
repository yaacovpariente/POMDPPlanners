"""Numerical parity tests for the discrete LaserTag native reward kernel.

Verifies that ``LaserTagPOMDP.reward_batch`` (which routes through the
``_native.lasertag_discrete_reward_batch`` C++ free function) produces
results bit-for-bit equivalent to the legacy pure-Python implementation
(now retained as ``_compute_reward_batch_python``). Covers all five
discrete actions and both the default and explicit ``dangerous_areas``
configurations, plus the ``ndim`` normalisation paths.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP


def _build_state_batch() -> np.ndarray:
    """Return a 6-row state batch covering the LaserTag reward edge cases.

    Rows:
        0: robot at wall cell (1, 2) — flags wall hit when intended-pos = (1, 2).
        1: robot away from any wall / danger — interior cell (4, 3).
        2: terminal state (flag = 1.0) — reward must be exactly 0.
        3: robot == opponent at (3, 3) — tag-success row.
        4: robot at dangerous area centre (5, 3) — flags danger when
           intended-pos = (5, 3).
        5: interior cell (0, 0); used for boundary / OOB-intended-pos checks.
    """
    return np.array(
        [
            [1.0, 2.0, 6.0, 6.0, 0.0],
            [4.0, 3.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 1.0],
            [3.0, 3.0, 3.0, 3.0, 0.0],
            [5.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture(name="default_env")
def _default_env_fixture() -> LaserTagPOMDP:
    """Default LaserTag env (uses the built-in dangerous_areas + walls)."""
    return LaserTagPOMDP(discount_factor=0.95)


@pytest.fixture(name="explicit_env")
def _explicit_env_fixture() -> LaserTagPOMDP:
    """LaserTag env with an explicit dangerous_areas set ({(2,3),(5,5)})."""
    return LaserTagPOMDP(
        discount_factor=0.95,
        dangerous_areas={(2, 3), (5, 5)},
    )


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
def test_native_reward_batch_default_env_matches_python(
    default_env: LaserTagPOMDP, action: int
) -> None:
    """Native reward kernel matches the Python fallback on the default env.

    Purpose: Validates ``LaserTagPOMDP.reward_batch`` (native dispatch)
        is bit-for-bit equivalent to ``_compute_reward_batch_python``
        across all 5 discrete actions on the default-walls / default-
        dangerous-areas env.

    Given: A ``LaserTagPOMDP`` initialised with default walls and
        dangerous areas, and the 6-row edge-case state batch covering
        wall hits, interior cells, terminal flags, tag success, and
        danger-area hits.
    When: ``reward_batch`` (native) and ``_compute_reward_batch_python``
        (reference) are evaluated on the same batch and action.
    Then: The two reward vectors are equal (``np.allclose`` atol=1e-12)
        and have shape ``(6,)`` float64.

    Test type: integration
    """
    states = _build_state_batch()
    native = default_env.reward_batch(states, action)
    reference = default_env._compute_reward_batch_python(  # pylint: disable=protected-access
        np.ascontiguousarray(states), action
    )
    assert native.shape == (states.shape[0],)
    assert native.dtype == np.float64
    assert np.allclose(native, reference, atol=1e-12)


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
def test_native_reward_batch_explicit_dangerous_areas_matches_python(
    explicit_env: LaserTagPOMDP, action: int
) -> None:
    """Native reward kernel matches the Python fallback on an env with custom dangerous areas.

    Purpose: Validates that the native dispatch and the Python fallback
        agree when ``dangerous_areas`` is explicitly set rather than
        defaulted, exercising the (D, 2) dangerous-areas buffer in C++.

    Given: A ``LaserTagPOMDP`` initialised with an explicit
        ``dangerous_areas={(2, 3), (5, 5)}`` (default radius / penalty)
        and the 6-row edge-case state batch.
    When: ``reward_batch`` and ``_compute_reward_batch_python`` are
        evaluated on the same batch and action.
    Then: The two reward vectors are equal (``np.allclose`` atol=1e-12).

    Test type: integration
    """
    states = _build_state_batch()
    native = explicit_env.reward_batch(states, action)
    reference = explicit_env._compute_reward_batch_python(  # pylint: disable=protected-access
        np.ascontiguousarray(states), action
    )
    assert np.allclose(native, reference, atol=1e-12)


def test_native_reward_batch_single_row_returns_shape_one(
    default_env: LaserTagPOMDP,
) -> None:
    """Single-row (1, 5) input still yields a shape (1,) reward vector.

    Purpose: Validates that the native kernel and the reward_batch
        wrapper produce a length-1 reward array when handed a single-
        particle (1, 5) state batch.

    Given: A ``LaserTagPOMDP`` and a single-particle (1, 5) state at
        an interior cell.
    When: ``reward_batch`` is invoked with action 0 (North).
    Then: The output has shape (1,), is float64, and matches the
        reference Python implementation.

    Test type: unit
    """
    states = np.array([[4.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    native = default_env.reward_batch(states, 0)
    reference = default_env._compute_reward_batch_python(  # pylint: disable=protected-access
        states, 0
    )
    assert native.shape == (1,)
    assert native.dtype == np.float64
    assert np.allclose(native, reference, atol=1e-12)


def test_native_reward_batch_ndim_one_state_normalises_to_batch(
    default_env: LaserTagPOMDP,
) -> None:
    """A 1-D length-5 state is normalised to (1, 5) and yields a (1,) reward.

    Purpose: Validates that the ``reward_batch`` wrapper still
        normalises a flat 1-D state vector into a (1, 5) batch before
        dispatching to the native kernel, preserving the documented
        public API.

    Given: A flat 1-D state ``np.array([3.0, 3.0, 3.0, 3.0, 0.0])``
        (robot == opponent → tag success row).
    When: ``reward_batch`` is invoked with action 4 (Tag).
    Then: The output has shape (1,), float64, and equals
        ``[tag_reward]`` since the dangerous-area penalty is not
        triggered for the cell (3, 3) in the default env.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 3.0, 3.0, 0.0], dtype=np.float64)
    native = default_env.reward_batch(state, 4)
    reference = default_env._compute_reward_batch_python(  # pylint: disable=protected-access
        state.reshape(1, -1), 4
    )
    assert native.shape == (1,)
    assert native.dtype == np.float64
    assert np.allclose(native, reference, atol=1e-12)
    # Default env: (3, 3) is not a wall and not within radius 1 of any
    # default dangerous-area centre {(5, 3), (7, 1), (2, 5)}, so the
    # reward is exactly +tag_reward (10.0) with no extra penalty.
    assert native[0] == pytest.approx(default_env.tag_reward)
