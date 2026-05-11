"""Numerical parity tests for the discrete LaserTag native reward kernel.

Verifies that the ``_native.lasertag_discrete_reward_batch`` C++ free
function produces results bit-for-bit equivalent to the legacy pure-
Python implementation (now retained as ``_compute_reward_batch_python``
when invoked without a ``next_states`` argument — i.e., the legacy
intended-position branch). Covers all five discrete actions and both the
default and explicit ``dangerous_areas`` configurations, plus the
``ndim`` normalisation paths.

Note: ``LaserTagPOMDP.reward_batch`` no longer routes through this
native kernel when penalty terms exist (walls / dangerous areas) because
the public API now scores the wall / danger penalty against the
*realised* next-state position (matching ``Environment.sample_next_step``
semantics). The native kernel still encodes the legacy intended-position
formula, so these tests invoke it directly via
``_native.lasertag_discrete_reward_batch`` and compare to
``_compute_reward_batch_python(states, action)`` with no ``next_states``.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp import _native


def _native_reward_batch(env: LaserTagPOMDP, states: np.ndarray, action: int) -> np.ndarray:
    """Invoke the native legacy intended-position kernel directly."""
    return np.asarray(
        _native.lasertag_discrete_reward_batch(
            states=np.ascontiguousarray(states),
            action=int(action),
            rows=int(env.floor_shape[0]),
            cols=int(env.floor_shape[1]),
            walls_flat=env._reward_walls_flat,  # pylint: disable=protected-access
            n_walls=env._reward_n_walls,  # pylint: disable=protected-access
            dangerous_areas=env._dangerous_areas_arr,  # pylint: disable=protected-access
            n_dangerous=int(env._dangerous_areas_arr.shape[0]),  # pylint: disable=protected-access
            dangerous_area_radius=float(env.dangerous_area_radius),
            dangerous_area_penalty=float(env.dangerous_area_penalty),
            tag_reward=float(env.tag_reward),
            tag_penalty=float(env.tag_penalty),
            step_cost=float(env.step_cost),
            action_directions=env._action_directions_arr,  # pylint: disable=protected-access
        )
    )


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
    """Native reward kernel matches the Python intended-position reference.

    Purpose: Validates the native ``lasertag_discrete_reward_batch`` C++
        kernel is bit-for-bit equivalent to
        ``_compute_reward_batch_python(states, action)`` (legacy
        intended-position branch) across all 5 discrete actions on the
        default-walls / default-dangerous-areas env. The kernel is
        invoked directly here because ``LaserTagPOMDP.reward_batch`` now
        skips it whenever penalty terms exist, in order to honour the
        next-state-aware reward contract.

    Given: A ``LaserTagPOMDP`` initialised with default walls and
        dangerous areas, and the 6-row edge-case state batch covering
        wall hits, interior cells, terminal flags, tag success, and
        danger-area hits.
    When: The native kernel and ``_compute_reward_batch_python`` (no
        ``next_states``) are evaluated on the same batch and action.
    Then: The two reward vectors are equal (``np.allclose`` atol=1e-12)
        and have shape ``(6,)`` float64.

    Test type: integration
    """
    states = _build_state_batch()
    native = _native_reward_batch(default_env, states, action)
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
    """Native reward kernel matches the Python intended-position reference with custom dangerous areas.

    Purpose: Validates that the native kernel and the Python fallback
        agree when ``dangerous_areas`` is explicitly set rather than
        defaulted, exercising the (D, 2) dangerous-areas buffer in C++.

    Given: A ``LaserTagPOMDP`` initialised with an explicit
        ``dangerous_areas={(2, 3), (5, 5)}`` (default radius / penalty)
        and the 6-row edge-case state batch.
    When: The native kernel and ``_compute_reward_batch_python`` (no
        ``next_states``) are evaluated on the same batch and action.
    Then: The two reward vectors are equal (``np.allclose`` atol=1e-12).

    Test type: integration
    """
    states = _build_state_batch()
    native = _native_reward_batch(explicit_env, states, action)
    reference = explicit_env._compute_reward_batch_python(  # pylint: disable=protected-access
        np.ascontiguousarray(states), action
    )
    assert np.allclose(native, reference, atol=1e-12)


def test_native_reward_batch_single_row_returns_shape_one(
    default_env: LaserTagPOMDP,
) -> None:
    """Single-row (1, 5) input still yields a shape (1,) reward vector.

    Purpose: Validates that the native kernel produces a length-1 reward
        array when handed a single-particle (1, 5) state batch and that
        this matches the Python intended-position reference.

    Given: A ``LaserTagPOMDP`` and a single-particle (1, 5) state at
        an interior cell.
    When: The native kernel is invoked directly with action 0 (North).
    Then: The output has shape (1,), is float64, and matches the
        reference Python implementation.

    Test type: unit
    """
    states = np.array([[4.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    native = _native_reward_batch(default_env, states, 0)
    reference = default_env._compute_reward_batch_python(  # pylint: disable=protected-access
        states, 0
    )
    assert native.shape == (1,)
    assert native.dtype == np.float64
    assert np.allclose(native, reference, atol=1e-12)


def test_native_reward_batch_ndim_one_state_matches_python(
    default_env: LaserTagPOMDP,
) -> None:
    """The native kernel and Python reference agree on a single-particle batch.

    Purpose: Validates parity between the native kernel and the Python
        reference on a 1-row (reshaped from 1-D) state corresponding to
        a tag-success cell.

    Given: A flat 1-D state ``np.array([3.0, 3.0, 3.0, 3.0, 0.0])``
        (robot == opponent → tag success row).
    When: The native kernel is invoked with action 4 (Tag).
    Then: The output has shape (1,), float64, and equals
        ``[tag_reward]`` since the dangerous-area penalty is not
        triggered for the cell (3, 3) in the default env.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 3.0, 3.0, 0.0], dtype=np.float64)
    native = _native_reward_batch(default_env, state.reshape(1, -1), 4)
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
