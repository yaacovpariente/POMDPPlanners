# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for the Continuous Push POMDP.

Covers the ``hazard-terminates-episode v2`` redesign for
:class:`ContinuousPushPOMDP` and its discrete-action variant:

* flag-off (default) behaviour is unchanged and the state stays 6-D;
* enabling ``is_dangerous_area_hit_terminal`` appends a terminal slot to
  the state and couples termination to the (deterministic) hazard
  penalty;
* ``is_dangerous_area_hit_terminal=True`` with the zero-mean shock reward
  model raises at construction;
* the native rollout kernel agrees with a Python step-by-step rollout on
  a seeded flag-on trajectory;
* the terminal slot is absorbing.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_push_pinned_kwargs


def _danger_env(**overrides):
    """Flag-on env with a single reachable dangerous zone at (3, 3)."""
    kwargs = continuous_push_pinned_kwargs(
        dangerous_areas=[(3.0, 3.0)],
        dangerous_area_radius=0.5,
        dangerous_area_penalty=-10.0,
        dangerous_area_hit_probability=1.0,
        state_transition_cov_matrix=np.eye(2) * 1e-12,
    )
    kwargs.update(overrides)
    return ContinuousPushPOMDP(discount_factor=0.99, is_dangerous_area_hit_terminal=True, **kwargs)


# ---------------------------------------------------------------------------
# (a) flag-off is unchanged and stays 6-D
# ---------------------------------------------------------------------------


def test_flag_off_state_stays_six_dim():
    """Default (both flags off) env keeps the historical 6-D state layout.

    Purpose: Validates that the terminal slot is only introduced when a
        hazard-terminal flag is enabled, so the default behaviour is
        byte-compatible with develop.

    Given: A default ``ContinuousPushPOMDP``.
    When: An initial state is sampled and a transition is taken.
    Then: Both the initial state and the next state are 6-D and
        ``reward_requires_next_state`` is ``False``.

    Test type: unit
    """
    env = ContinuousPushPOMDP(discount_factor=0.99, **continuous_push_pinned_kwargs())
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (6,)
    next_state = env.sample_next_state(state, np.array([1.0, 0.0]))
    assert next_state.shape == (6,)
    assert env.reward_requires_next_state is False


def test_flag_off_transition_is_deterministic_under_seed():
    """Flag-off transitions reproduce identically under a fixed native seed.

    Purpose: Locks the flag-off transition RNG stream so the redesign does
        not perturb the untouched default path.

    Given: A default env and a fixed seed.
    When: A transition is sampled twice under the same seed.
    Then: The two next states are elementwise identical and 6-D.

    Test type: unit
    """
    env = ContinuousPushPOMDP(discount_factor=0.99, **continuous_push_pinned_kwargs())
    state = np.array([2.0, 3.0, 5.0, 5.0, 8.0, 8.0])
    _native.set_seed(4242)
    first = env.sample_next_state(state, np.array([1.0, 0.0]))
    _native.set_seed(4242)
    second = env.sample_next_state(state, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (6,)


# ---------------------------------------------------------------------------
# (b) flag-on draw-coupled termination + deterministic penalty
# ---------------------------------------------------------------------------


def test_flag_on_initial_state_is_seven_dim_with_zero_slot():
    """Enabling the flag appends a zero terminal slot to fresh states.

    Purpose: Validates the conditional 7-D layout.

    Given: A flag-on env.
    When: An initial state is sampled.
    Then: The state is 7-D and its last element is ``0.0``.

    Test type: unit
    """
    env = _danger_env()
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (7,)
    assert state[-1] == 0.0
    assert env.reward_requires_next_state is True


def test_dangerous_zone_landing_sets_terminal_and_applies_penalty():
    """Landing in a dangerous zone (not at goal) terminates and is penalised.

    Purpose: Validates draw-coupled termination at ``hit_probability=1.0``.

    Given: A flag-on env with one dangerous zone; a 7-D state whose robot
        sits inside the zone with the object far from the target.
    When: A (near-zero-noise) transition is sampled and its reward taken.
    Then: The next state's terminal slot is ``1.0`` and the reward includes
        the deterministic ``dangerous_area_penalty``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([3.0, 3.0, 5.0, 5.0, 9.0, 9.0, 0.0])
    _native.set_seed(1)
    next_state = env.sample_next_state(state, np.array([0.0, 0.0]))
    assert next_state.shape == (7,)
    assert next_state[-1] == 1.0

    reward = env.reward(state, np.array([0.0, 0.0]), next_state=next_state)
    base = -float(np.hypot(next_state[2] - next_state[4], next_state[3] - next_state[5]))
    np.testing.assert_allclose(reward, base + env.dangerous_area_penalty, atol=1e-9)


def test_reward_no_penalty_when_terminal_slot_unset():
    """Flag-on reward applies the hazard penalty only when the slot is set.

    Purpose: Validates the deterministic coupling in the reward: no penalty
        unless the realised next state is terminal.

    Given: A flag-on env and two identical next states differing only in the
        terminal slot (robot inside the zone, object away from target).
    When: The batch reward is evaluated on both.
    Then: The reward difference equals ``dangerous_area_penalty``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([3.0, 3.0, 5.0, 5.0, 9.0, 9.0, 0.0])
    live = np.array([[3.0, 3.0, 5.0, 5.0, 9.0, 9.0, 0.0]])
    dead = np.array([[3.0, 3.0, 5.0, 5.0, 9.0, 9.0, 1.0]])
    action = np.array([0.0, 0.0])
    r_live = env.reward_batch(np.array([state]), action, next_states=live)[0]
    r_dead = env.reward_batch(np.array([state]), action, next_states=dead)[0]
    np.testing.assert_allclose(r_dead - r_live, env.dangerous_area_penalty, atol=1e-9)


def test_obstacle_terminal_reward_couples_penalty_to_slot():
    """Obstacle-terminal reward penalises iff terminal and robot in the AABB.

    Purpose: Validates the obstacle-hazard reward coupling directly on a
        crafted next state (obstacle collisions are unreachable through the
        wall-resolved transition, so this is exercised at the reward level).

    Given: An ``is_obstacle_hit_terminal`` env with one AABB obstacle and a
        7-D next state whose robot centre sits inside the obstacle.
    When: The batch reward is evaluated with the terminal slot set vs unset.
    Then: The reward difference equals ``obstacle_penalty``.

    Test type: unit
    """
    env = ContinuousPushPOMDP(
        discount_factor=0.99,
        is_obstacle_hit_terminal=True,
        **continuous_push_pinned_kwargs(
            obstacles=[(5.0, 5.0, 1.0)],
            obstacle_penalty=-10.0,
            obstacle_hit_probability=1.0,
        ),
    )
    state = np.array([5.0, 5.0, 2.0, 2.0, 9.0, 9.0, 0.0])
    action = np.array([1.0, 0.0])
    live = np.array([[5.0, 5.0, 2.0, 2.0, 9.0, 9.0, 0.0]])
    dead = np.array([[5.0, 5.0, 2.0, 2.0, 9.0, 9.0, 1.0]])
    r_live = env.reward_batch(np.array([state]), action, next_states=live)[0]
    r_dead = env.reward_batch(np.array([state]), action, next_states=dead)[0]
    np.testing.assert_allclose(r_dead - r_live, env.obstacle_penalty, atol=1e-9)


# ---------------------------------------------------------------------------
# (c) shock + terminal flag is rejected at construction
# ---------------------------------------------------------------------------


def test_dangerous_terminal_with_shock_raises():
    """Zero-mean shock has no hit probability, so terminal coupling is invalid.

    Purpose: Validates the construction-time guard.

    Given: ``is_dangerous_area_hit_terminal=True`` and the shock reward model.
    When: The env is constructed.
    Then: ``ValueError`` is raised.

    Test type: unit
    """
    with pytest.raises(ValueError):
        ContinuousPushPOMDP(
            discount_factor=0.99,
            is_dangerous_area_hit_terminal=True,
            **continuous_push_pinned_kwargs(
                dangerous_areas=[(3.0, 3.0)],
                reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
            ),
        )


# ---------------------------------------------------------------------------
# (d) python <-> native rollout parity on a seeded flag-on trajectory
# ---------------------------------------------------------------------------


def test_native_rollout_matches_python_rollout_flag_on():
    """``cont_simulate_rollout`` matches a Python step rollout under one seed.

    Purpose: Validates that the native flag-on rollout draws the same C++ RNG
        stream (position noise then hazard uniform) and applies the same
        deterministic reward as the Python single-step path.

    Given: A flag-on env, a fixed action set, and a pre-drawn action-index
        sequence.
    When: ``cont_simulate_rollout`` and an equivalent Python loop are each run
        under the same native seed.
    Then: The two discounted returns agree to floating-point tolerance.

    Test type: integration
    """
    env = _danger_env(state_transition_cov_matrix=np.eye(2) * 0.05)
    actions = np.array([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0], [0.0, -0.2]], dtype=np.float64)
    initial = np.array([3.0, 2.6, 6.0, 6.0, 9.0, 9.0, 0.0])
    max_depth = 12
    rng = np.random.default_rng(7)
    action_indices = rng.integers(0, len(actions), size=max_depth).astype(np.int32)
    discount = 0.99

    _native.set_seed(20240712)
    native_return = _native.cont_simulate_rollout(
        initial_state=initial,
        action_array=actions,
        action_indices=action_indices,
        max_depth=max_depth,
        start_depth=0,
        discount_factor=discount,
        grid_size=float(env.grid_size),
        push_threshold=float(env.push_threshold),
        friction_coefficient=float(env.friction_coefficient),
        max_push=float(env.max_push),
        robot_radius=float(env.robot_radius),
        obstacle_penalty=float(env.obstacle_penalty),
        obstacles=np.ascontiguousarray(env.obstacles, dtype=np.float64),
        dangerous_areas=env._dangerous_areas_arr,  # pylint: disable=protected-access
        dangerous_area_radius=float(env.dangerous_area_radius),
        dangerous_area_penalty=float(env.dangerous_area_penalty),
        covariance=env._state_transition_dist.covariance_view(),  # pylint: disable=protected-access
        obstacle_hit_probability=float(env.obstacle_hit_probability),
        dangerous_area_hit_probability=float(env.dangerous_area_hit_probability),
        reward_variant_code=0,
        penalty_decay=float(env.penalty_decay),
        is_obstacle_hit_terminal=False,
        is_dangerous_area_hit_terminal=True,
    )

    _native.set_seed(20240712)
    total = 0.0
    gamma = 1.0
    state = initial.copy()
    for step in range(max_depth):
        if env.is_terminal(state):
            break
        action = actions[action_indices[step]]
        next_state = env.sample_next_state(state, action)
        total += gamma * env.reward(state, action, next_state=next_state)
        gamma *= discount
        state = next_state

    np.testing.assert_allclose(native_return, total, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# (e) terminal is absorbing
# ---------------------------------------------------------------------------


def test_terminal_state_is_absorbing():
    """A terminal state transitions to itself with the slot latched at 1.0.

    Purpose: Validates the absorbing contract.

    Given: A flag-on env and a 7-D state whose terminal slot is already set.
    When: A transition is sampled.
    Then: The next state equals the input (slot stays ``1.0``).

    Test type: unit
    """
    env = _danger_env()
    state = np.array([3.0, 3.0, 5.0, 5.0, 9.0, 9.0, 1.0])
    _native.set_seed(3)
    next_state = env.sample_next_state(state, np.array([1.0, 0.0]))
    assert next_state[-1] == 1.0
    np.testing.assert_array_equal(next_state, state)
    assert env.is_terminal(next_state)


# ---------------------------------------------------------------------------
# discrete-action variant carries the flags through
# ---------------------------------------------------------------------------


def test_discrete_actions_variant_supports_terminal_flag():
    """The discrete-action wrapper honours ``is_dangerous_area_hit_terminal``.

    Purpose: Validates flag plumbing through the discrete-action subclass.

    Given: A flag-on ``ContinuousPushPOMDPDiscreteActions`` with a dangerous
        zone under the robot and the object away from the target.
    When: A transition is sampled for a string action.
    Then: The next state is 7-D with the terminal slot set.

    Test type: unit
    """
    env = ContinuousPushPOMDPDiscreteActions(
        discount_factor=0.99,
        is_dangerous_area_hit_terminal=True,
        **continuous_push_pinned_kwargs(
            dangerous_areas=[(3.0, 3.0)],
            dangerous_area_radius=0.5,
            dangerous_area_hit_probability=1.0,
            state_transition_cov_matrix=np.eye(2) * 1e-12,
        ),
    )
    # Robot at (3, 2); action "up" = (0, +1) lands it at (3, 3) inside the zone.
    state = np.array([3.0, 2.0, 5.0, 5.0, 9.0, 9.0, 0.0])
    _native.set_seed(5)
    next_state = env.sample_next_state(state, "up")
    assert next_state.shape == (7,)
    assert next_state[-1] == 1.0
