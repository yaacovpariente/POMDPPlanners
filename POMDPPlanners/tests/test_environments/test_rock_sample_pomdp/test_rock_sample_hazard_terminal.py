# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for the RockSample POMDP.

Covers the ``hazard-terminates-episode v2`` redesign for
:class:`RockSamplePOMDP`:

* flag-off (default) behaviour is unchanged and the state keeps its
  historical ``2 + num_rocks`` width (float32);
* enabling ``is_dangerous_area_hit_terminal`` appends a terminal slot to
  the state and couples termination to the (now deterministic) hazard
  penalty;
* ``is_dangerous_area_hit_terminal=True`` with the zero-mean shock reward
  model raises at construction;
* the native rollout kernel agrees with a Python step-by-step rollout on
  a seeded flag-on trajectory;
* the terminal slot is absorbing and drives ``is_terminal``.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RewardModelType,
    RockSamplePOMDP,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import rock_sample_pinned_kwargs


def _danger_env(**overrides) -> RockSamplePOMDP:
    """Flag-on env: rock far from the reachable danger cell at ``(1, 0)``."""
    kwargs = rock_sample_pinned_kwargs(
        map_size=(5, 5),
        rock_positions=[(4, 4)],
        init_pos=(0, 0),
        dangerous_areas=[(1, 0)],
        dangerous_area_radius=0.5,
        dangerous_area_penalty=-5.0,
        dangerous_area_hit_probability=1.0,
    )
    kwargs.update(overrides)
    return RockSamplePOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True, **kwargs)


# ---------------------------------------------------------------------------
# (a) flag-off is unchanged and keeps the base width / float32 dtype
# ---------------------------------------------------------------------------


def test_flag_off_state_keeps_base_width_and_dtype():
    """Default env keeps the historical ``2 + num_rocks`` float32 layout.

    Purpose: Validates the CONDITIONAL slot — no terminal slot unless the
        hazard-terminal flag is enabled, so the default path is unchanged.

    Given: A default ``RockSamplePOMDP`` with one rock.
    When: An initial state is sampled.
    Then: The state has width ``3`` (2 + 1 rock), is float32, and
        ``reward_requires_next_state`` is ``False``.

    Test type: unit
    """
    env = RockSamplePOMDP(
        discount_factor=0.95, **rock_sample_pinned_kwargs(rock_positions=[(4, 4)])
    )
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (3,)
    assert state.dtype == np.float32
    assert env.reward_requires_next_state is False


# ---------------------------------------------------------------------------
# (b) flag-on draw-coupled termination + deterministic penalty
# ---------------------------------------------------------------------------


def test_flag_on_initial_state_has_zero_terminal_slot_float32():
    """Enabling the flag appends a zero float32 terminal slot to fresh states.

    Purpose: Validates the conditional widened layout and dtype.

    Given: A flag-on env with one rock.
    When: An initial state is sampled.
    Then: The state has width ``4``, dtype float32, its last element is
        ``0.0``, and ``reward_requires_next_state`` is ``True``.

    Test type: unit
    """
    env = _danger_env()
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (4,)
    assert state.dtype == np.float32
    assert state[-1] == 0.0
    assert env.reward_requires_next_state is True


def test_dangerous_landing_sets_terminal_and_applies_penalty():
    """Moving into a dangerous cell terminates and is penalised deterministically.

    Purpose: Validates draw-coupled termination at ``hit_probability=1.0``.

    Given: A flag-on env; a live state with the robot at ``(0, 0)``.
    When: A south move (into the danger cell ``(1, 0)``) is sampled and its
        reward taken with the realised next state.
    Then: The next state's terminal slot is ``1.0`` and the reward equals
        ``step_penalty + dangerous_area_penalty``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _native.set_seed(1)
    next_state = env.sample_next_state(state, 3)  # south -> (1, 0)
    assert next_state.shape == (4,)
    assert (int(next_state[0]), int(next_state[1])) == (1, 0)
    assert next_state[-1] == 1.0

    reward = env.reward(state, 3, next_state=next_state)
    np.testing.assert_allclose(reward, env.step_penalty + env.dangerous_area_penalty, atol=1e-9)


def test_reward_no_penalty_when_terminal_slot_unset():
    """Flag-on reward applies the hazard penalty only when the slot is set.

    Purpose: Validates the deterministic coupling in the reward path.

    Given: A flag-on env and two identical next states differing only in the
        terminal slot (robot inside the danger cell).
    When: The batch reward is evaluated on both.
    Then: The reward difference equals ``dangerous_area_penalty``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    live = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    dead = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    r_live = env.reward_batch(state, 3, next_states=live)[0]
    r_dead = env.reward_batch(state, 3, next_states=dead)[0]
    np.testing.assert_allclose(r_dead - r_live, env.dangerous_area_penalty, atol=1e-9)


def test_outside_zone_no_termination():
    """A move that stays out of every danger zone does not terminate.

    Purpose: Validates that the hazard draw only fires in-zone (constant model).

    Given: A flag-on env; a live state with the robot at ``(0, 0)``.
    When: An east move (to ``(0, 1)``, outside the ``(1, 0)`` zone) is sampled.
    Then: The terminal slot stays ``0.0``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _native.set_seed(2)
    next_state = env.sample_next_state(state, 2)  # east -> (0, 1)
    assert (int(next_state[0]), int(next_state[1])) == (0, 1)
    assert next_state[-1] == 0.0
    assert env.is_terminal(next_state) is False


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
        RockSamplePOMDP(
            discount_factor=0.95,
            is_dangerous_area_hit_terminal=True,
            **rock_sample_pinned_kwargs(
                dangerous_areas=[(1, 0)],
                reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
            ),
        )


# ---------------------------------------------------------------------------
# (d) python <-> native rollout parity on a seeded flag-on trajectory
# ---------------------------------------------------------------------------


def _run_python_rollout(env, initial, action_indices, discount):
    total = 0.0
    gamma = 1.0
    state = np.asarray(initial, dtype=np.float64).copy()
    for action in action_indices:
        if env.is_terminal(state):
            break
        next_state = env.sample_next_state(state, int(action))
        total += gamma * env.reward(state, int(action), next_state=next_state)
        gamma *= discount
        state = next_state
    return total


@pytest.mark.parametrize(
    "reward_model_type",
    [
        RewardModelType.CONSTANT_HAZARD_PENALTY,
        RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
    ],
)
def test_native_rollout_matches_python_rollout_flag_on(reward_model_type):
    """``simulate_rollout_discrete`` matches a Python step rollout under one seed.

    Purpose: Validates that the native flag-on rollout draws the same C++ RNG
        hazard stream and applies the same deterministic reward as the Python
        single-step transition + reward path.

    Given: A flag-on env whose whole small map is a danger zone (so the hazard
        uniform is drawn every non-exit step) with a sub-unit hit probability.
    When: The native rollout kernel and an equivalent Python loop are each run
        under the same native seed and identical pre-drawn actions.
    Then: The two discounted returns agree to floating-point tolerance.

    Test type: integration
    """
    env = _danger_env(
        map_size=(4, 4),
        rock_positions=[(3, 3)],
        dangerous_areas=[(1, 1), (2, 2)],
        dangerous_area_radius=10.0,
        dangerous_area_hit_probability=0.3,
        reward_model_type=reward_model_type,
        penalty_decay=2.0,
    )
    initial = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    max_depth = 15
    n_actions = len(env.action_names)
    rng = np.random.default_rng(7)
    # Bias toward movement (1..4) so the robot lingers in-zone before exiting.
    action_indices = rng.integers(1, 5, size=max_depth).astype(np.int32)
    discount = 0.95
    variant_code = {
        RewardModelType.CONSTANT_HAZARD_PENALTY: 0,
        RewardModelType.ZERO_MEAN_HAZARD_SHOCK: 1,
        RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY: 2,
    }[reward_model_type]

    _native.set_seed(20240712)
    native_return = _native.simulate_rollout_discrete(
        initial_state=initial,
        action_indices=action_indices,
        rock_positions_flat=env._rock_positions_flat,  # pylint: disable=protected-access
        max_depth=max_depth,
        start_depth=0,
        discount_factor=discount,
        map_rows=int(env.map_size[0]),
        map_cols=int(env.map_size[1]),
        n_actions=n_actions,
        step_penalty=float(env.step_penalty),
        exit_reward=float(env.exit_reward),
        good_rock_reward=float(env.good_rock_reward),
        bad_rock_penalty=float(env.bad_rock_penalty),
        sensor_use_penalty=float(env.sensor_use_penalty),
        dangerous_areas=env._dangerous_areas_arr,  # pylint: disable=protected-access
        dangerous_area_radius=float(env.dangerous_area_radius),
        dangerous_area_penalty=float(env.dangerous_area_penalty),
        dangerous_area_hit_probability=float(env.dangerous_area_hit_probability),
        reward_variant_code=variant_code,
        penalty_decay=float(env.penalty_decay),
        is_dangerous_area_hit_terminal=True,
    )

    _native.set_seed(20240712)
    python_return = _run_python_rollout(env, initial, action_indices, discount)

    np.testing.assert_allclose(native_return, python_return, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# (e) terminal is absorbing and drives is_terminal
# ---------------------------------------------------------------------------


def test_terminal_slot_is_absorbing():
    """A slot-terminal state transitions to itself with the slot latched at 1.0.

    Purpose: Validates the absorbing contract.

    Given: A flag-on env and a state whose terminal slot is already ``1.0``.
    When: A transition is sampled.
    Then: The next state equals the input and ``is_terminal`` is ``True``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    _native.set_seed(3)
    next_state = env.sample_next_state(state, 3)
    assert next_state[-1] == 1.0
    np.testing.assert_array_equal(next_state, state)
    assert env.is_terminal(next_state) is True


def test_is_terminal_reads_slot_and_sentinel():
    """``is_terminal`` fires on the exit sentinel and on the terminal slot.

    Purpose: Validates the ``is_terminal`` disjunction for a flag-on env.

    Given: A flag-on env.
    When: ``is_terminal`` is queried on a live state, a slot-terminal state,
        and the exit sentinel.
    Then: It returns ``False``, ``True`` and ``True`` respectively.

    Test type: unit
    """
    env = _danger_env()
    live = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    slot_terminal = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    sentinel = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float64)
    assert env.is_terminal(live) is False
    assert env.is_terminal(slot_terminal) is True
    assert env.is_terminal(sentinel) is True


def test_batch_transition_sets_terminal_slot():
    """The vectorized updater's batch transition sets the terminal slot in-zone.

    Purpose: Validates flag/param threading through the vectorized updater.

    Given: A flag-on env and a batch of two live 4-wide particles, both at the
        robot start ``(0, 0)``.
    When: A south move (into the danger cell) is applied via the updater.
    Then: Both next particles are at ``(1, 0)`` with the terminal slot ``1.0``.

    Test type: integration
    """
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (  # pylint: disable=import-outside-toplevel
        RockSampleVectorizedUpdater,
    )

    env = _danger_env()
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    _native.set_seed(9)
    nxt = updater.batch_transition(particles, np.asarray(3))
    assert nxt.shape == (2, 4)
    assert np.all(nxt[:, 0] == 1.0)
    assert np.all(nxt[:, 1] == 0.0)
    assert np.all(nxt[:, 3] == 1.0)
