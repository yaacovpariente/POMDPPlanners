# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for the discrete Push POMDP.

Covers the ``hazard-terminates-episode v2`` redesign for
:class:`PushPOMDP`:

* flag-off (default) behaviour is unchanged and the state stays 6-D;
* enabling ``is_dangerous_area_hit_terminal`` appends a terminal slot to
  the state and couples termination to the (now deterministic) hazard
  penalty drawn inside the C++ transition kernel;
* ``is_dangerous_area_hit_terminal=True`` with the zero-mean shock reward
  model raises at construction;
* the native rollout kernel agrees with the Python reference rollout on a
  seeded flag-on trajectory, for both hazard reward variants;
* the terminal slot is absorbing, drives ``is_terminal``, is carried by
  observations, is marginalised out of ``transition_log_probability``,
  and survives serialization.
"""

import math
import pickle

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import push_pinned_kwargs

# Live 7-D state: the robot sits one cell below the (3, 3) zone centre, so the
# action "up" lands it exactly on the centre; the object (8, 8) is out of push
# range of the robot and 1 diagonal cell away from the target (9, 9).
LIVE = np.array([3.0, 2.0, 8.0, 8.0, 9.0, 9.0, 0.0])

# Base (hazard-free) reward for any next state whose object is at (8, 8) and
# whose target is at (9, 9): -dist(object, target).
BASE = -math.sqrt(2.0)


def _danger_env(**overrides) -> PushPOMDP:
    """Flag-on env with a single reachable dangerous zone at (3, 3)."""
    kwargs = push_pinned_kwargs(
        dangerous_areas=[(3.0, 3.0)],
        dangerous_area_radius=0.5,
        dangerous_area_penalty=-10.0,
        dangerous_area_hit_probability=1.0,
        transition_error_prob=0.0,
        is_dangerous_area_hit_terminal=True,
    )
    kwargs.update(overrides)
    return PushPOMDP(discount_factor=0.95, **kwargs)


def _native_rollout(
    env: PushPOMDP,
    state: np.ndarray,
    action_indices: np.ndarray,
    max_depth: int,
    discount: float,
    is_dangerous_area_hit_terminal: bool,
) -> float:
    """Run ``simulate_rollout_discrete`` with ``env``'s configuration.

    Mirrors the kwargs :meth:`PushPOMDP.simulate_random_rollout` builds, but
    takes a pre-drawn action-index array so the native and Python paths can be
    compared step for step.
    """
    # pylint: disable=protected-access
    variant_code, _ = env._reward_variant_native_params()
    return float(
        _native.simulate_rollout_discrete(
            state=np.asarray(state, dtype=np.float64),
            action_indices=action_indices,
            max_depth=max_depth,
            depth=0,
            discount=discount,
            grid_size=float(env.grid_size),
            push_threshold=float(env.push_threshold),
            friction_coefficient=float(env.friction_coefficient),
            obstacles=env._get_native_rollout_obstacles(),
            obstacle_radius=float(env.obstacle_radius),
            obstacle_penalty=float(env.obstacle_penalty),
            dangerous_areas=env._dangerous_areas_arr,
            dangerous_area_radius=float(env.dangerous_area_radius),
            dangerous_area_penalty=float(env.dangerous_area_penalty),
            transition_error_prob=float(env.transition_error_prob),
            obstacle_hit_probability=float(env.obstacle_hit_probability),
            dangerous_area_hit_probability=float(env.dangerous_area_hit_probability),
            reward_variant_code=variant_code,
            penalty_decay=float(env.penalty_decay),
            is_dangerous_area_hit_terminal=is_dangerous_area_hit_terminal,
        )
    )


# ---------------------------------------------------------------------------
# (a) flag-off is unchanged and stays 6-D
# ---------------------------------------------------------------------------


def test_flag_off_state_stays_six_dim():
    """Default env keeps the historical 6-D state layout.

    Purpose: Validates that the terminal slot is only introduced when the
        hazard-terminal flag is enabled, so the default behaviour is
        byte-compatible with develop.

    Given: A default ``PushPOMDP`` with a dangerous zone but the flag off.
    When: An initial state is sampled and a transition is taken.
    Then: Both states are 6-D, ``reward_requires_next_state`` is ``False``
        and ``hazard_terminal_enabled`` is ``False``.

    Test type: unit
    """
    env = PushPOMDP(0.95, **push_pinned_kwargs(dangerous_areas=[(3.0, 3.0)]))
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (6,)
    next_state = env.sample_next_state(np.array([3.0, 2.0, 8.0, 8.0, 9.0, 9.0]), "up")
    assert next_state.shape == (6,)
    assert env.reward_requires_next_state is False
    assert env.hazard_terminal_enabled is False


def test_flag_off_native_rollout_matches_python_reference():
    """Flag-off native rollout still matches the Python reference rollout.

    Purpose: Locks the pre-existing flag-off rollout contract so the redesign
        does not perturb the untouched default path.

    Given: A flag-off env with a dangerous zone, deterministic transitions and
        both hit probabilities at ``1.0`` (the flag-off Python reference draws
        its hazard/obstacle Bernoulli on ``np.random`` while the native kernel
        draws on the C++ RNG, so parity only holds at probability ``1.0``).
    When: The native kernel and the Python reference rollout are each run under
        the same native seed with the same pre-drawn action sequence.
    Then: The two discounted returns agree to floating-point tolerance.

    Test type: integration
    """
    env = PushPOMDP(
        0.95,
        **push_pinned_kwargs(
            dangerous_areas=[(3.0, 3.0)],
            dangerous_area_radius=0.5,
            dangerous_area_penalty=-10.0,
            dangerous_area_hit_probability=1.0,
            obstacle_hit_probability=1.0,
            transition_error_prob=0.0,
        ),
    )
    initial = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])
    max_depth = 12
    action_indices = np.random.default_rng(11).integers(0, 4, size=max_depth).astype(np.int64)

    _native.set_seed(123)
    native_return = _native_rollout(env, initial, action_indices, max_depth, 0.95, False)

    _native.set_seed(123)
    python_return = env._python_simulate_random_rollout(  # pylint: disable=protected-access
        state=initial,
        actions=[env.actions[i] for i in action_indices],
        max_depth=max_depth,
        discount_factor=0.95,
    )

    np.testing.assert_allclose(native_return, python_return, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# (b) flag-on draw-coupled termination + deterministic penalty
# ---------------------------------------------------------------------------


def test_flag_on_initial_state_is_seven_dim_with_zero_slot():
    """Enabling the flag appends a live (0.0) terminal slot to fresh states.

    Purpose: Validates the conditional 7-D layout for both the random and the
        fixed initial-state distributions.

    Given: A flag-on env, and a flag-on env given a 6-D ``initial_state``.
    When: An initial state is sampled from each.
    Then: Both states are 7-D with a ``0.0`` terminal slot, and
        ``reward_requires_next_state`` is ``True``.

    Test type: unit
    """
    env = _danger_env()
    state = env.initial_state_dist().sample()[0]
    assert state.shape == (7,)
    assert state[-1] == 0.0
    assert env.reward_requires_next_state is True

    padded_env = _danger_env(initial_state=np.array([3.0, 2.0, 8.0, 8.0, 9.0, 9.0]))
    padded = padded_env.initial_state_dist().sample()[0]
    assert padded.shape == (7,)
    np.testing.assert_array_equal(padded, np.array([3.0, 2.0, 8.0, 8.0, 9.0, 9.0, 0.0]))


def test_dangerous_zone_landing_sets_terminal_and_applies_penalty():
    """Landing in a dangerous zone (not at goal) terminates and is penalised.

    Purpose: Validates draw-coupled termination at ``hit_probability=1.0`` and
        the deterministic penalty read off the realised terminal slot.

    Given: A flag-on env with one dangerous zone and a live 7-D state whose
        "up" action lands the robot on the zone centre.
    When: The transition, its reward, and a full ``sample_next_step`` are taken.
    Then: The next state's terminal slot is ``1.0``, the reward is
        ``-sqrt(2) - 10.0``, and ``is_terminal`` is ``True``.

    Test type: unit
    """
    env = _danger_env()
    next_state = env.sample_next_state(LIVE, "up")
    np.testing.assert_array_equal(next_state, np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0]))

    reward = env.reward(LIVE, "up", next_state=next_state)
    np.testing.assert_allclose(reward, BASE - 10.0, atol=1e-12, rtol=0.0)
    assert env.is_terminal(next_state) is True

    step_next_state, _, step_reward = env.sample_next_step(LIVE, "up")
    np.testing.assert_allclose(step_reward, BASE - 10.0, atol=1e-12, rtol=0.0)
    assert step_next_state[6] == 1.0


def test_outside_zone_no_termination():
    """A move that stays out of every danger zone does not terminate.

    Purpose: Validates that the constant-variant hazard draw only fires when
        the realised robot position is in-zone.

    Given: A flag-on env and a live state whose "up" action lands the robot at
        ``(1, 2)``, outside the ``(3, 3)`` zone.
    When: The transition is sampled and its reward taken.
    Then: The terminal slot stays ``0.0``, ``is_terminal`` is ``False``, and
        the reward is the unpenalised ``-sqrt(2)``.

    Test type: unit
    """
    env = _danger_env()
    state = np.array([1.0, 1.0, 8.0, 8.0, 9.0, 9.0, 0.0])
    next_state = env.sample_next_state(state, "up")
    np.testing.assert_array_equal(next_state, np.array([1.0, 2.0, 8.0, 8.0, 9.0, 9.0, 0.0]))
    assert env.is_terminal(next_state) is False
    np.testing.assert_allclose(
        env.reward(state, "up", next_state=next_state), BASE, atol=1e-12, rtol=0.0
    )


def test_reward_no_penalty_when_terminal_slot_unset():
    """Flag-on reward applies the hazard penalty only when the slot is set.

    Purpose: Validates the deterministic coupling in the reward path: no
        penalty unless the realised next state is terminal, even in-zone.

    Given: A flag-on env and two identical in-zone next states differing only
        in the terminal slot.
    When: The batch reward is evaluated on both.
    Then: The live reward is ``-sqrt(2)`` and the dead one ``-sqrt(2) - 10.0``.

    Test type: unit
    """
    env = _danger_env()
    live = np.array([[3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 0.0]])
    dead = np.array([[3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0]])
    r_live = env.reward_batch(live, "up", next_states=live)[0]
    r_dead = env.reward_batch(dead, "up", next_states=dead)[0]
    np.testing.assert_allclose(r_live, BASE, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(r_dead, BASE - 10.0, atol=1e-12, rtol=0.0)


def test_partial_hit_probability_terminates_statistically():
    """A sub-unit hit probability terminates at roughly that rate.

    Purpose: Validates that the in-zone hazard uniform is compared against
        ``dangerous_area_hit_probability`` rather than always firing.

    Given: A flag-on env with ``dangerous_area_hit_probability=0.3`` and a
        fixed native seed.
    When: 2000 in-zone transitions are sampled.
    Then: The empirical termination rate lies in ``[0.25, 0.35]``.

    Test type: statistical
    """
    env = _danger_env(dangerous_area_hit_probability=0.3)
    _native.set_seed(20240712)
    n_samples = 2000
    terminal_count = sum(1 for _ in range(n_samples) if env.sample_next_state(LIVE, "up")[6] == 1.0)
    rate = terminal_count / n_samples
    assert 0.25 <= rate <= 0.35


def test_distance_decayed_variant_terminates_at_zone_centre():
    """The distance-decayed variant draws every step, scaled by distance.

    Purpose: Validates the ``penalty_decay**2 * log(u)**2 > min_dist_sq``
        termination rule at both extremes.

    Given: A flag-on distance-decayed env with ``penalty_decay=1.0``, and one
        with a vanishing ``penalty_decay=1e-6``.
    When: The robot lands exactly on the zone centre (min distance ``0``), and
        far away from it, respectively.
    Then: The centre landing always terminates with reward
        ``-sqrt(2) - 10.0``, and the far landing never terminates.

    Test type: unit
    """
    env = _danger_env(
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        penalty_decay=1.0,
    )
    next_state = env.sample_next_state(LIVE, "up")
    np.testing.assert_array_equal(next_state, np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0]))
    np.testing.assert_allclose(
        env.reward(LIVE, "up", next_state=next_state), BASE - 10.0, atol=1e-12, rtol=0.0
    )

    far_env = _danger_env(
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        penalty_decay=1e-6,
    )
    far_state = np.array([0.0, 0.0, 8.0, 8.0, 9.0, 9.0, 0.0])
    far_next = far_env.sample_next_state(far_state, "up")
    np.testing.assert_array_equal(far_next, np.array([0.0, 1.0, 8.0, 8.0, 9.0, 9.0, 0.0]))


# ---------------------------------------------------------------------------
# (c) shock + terminal flag is rejected at construction
# ---------------------------------------------------------------------------


def test_dangerous_terminal_with_shock_raises():
    """Zero-mean shock has no hit probability, so terminal coupling is invalid.

    Purpose: Validates the construction-time guard and that it is scoped to
        the flag being on.

    Given: The shock reward model, with the hazard-terminal flag on and off.
    When: The env is constructed in each case.
    Then: The flag-on construction raises ``ValueError``; the flag-off one
        constructs a plain 6-D env.

    Test type: unit
    """
    with pytest.raises(ValueError):
        _danger_env(reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK)

    env = PushPOMDP(
        0.95,
        **push_pinned_kwargs(
            dangerous_areas=[(3.0, 3.0)],
            dangerous_area_radius=0.5,
            dangerous_area_penalty=-10.0,
            dangerous_area_hit_probability=1.0,
            transition_error_prob=0.0,
            is_dangerous_area_hit_terminal=False,
            reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
        ),
    )
    assert env.hazard_terminal_enabled is False


# ---------------------------------------------------------------------------
# (d) python <-> native rollout parity on a seeded flag-on trajectory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, hit_prob, decay",
    [
        (RewardModelType.CONSTANT_HAZARD_PENALTY, 0.3, 1.0),
        (RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, 1.0, 2.0),
    ],
)
def test_native_rollout_matches_python_rollout_flag_on(model, hit_prob, decay):
    """``simulate_rollout_discrete`` matches the Python reference under one seed.

    Purpose: Validates that the native flag-on rollout draws the same C++ RNG
        hazard stream in the same order and applies the same deterministic
        reward as the Python single-step transition + reward path.

    Given: A flag-on env whose whole grid is a danger zone (so the hazard
        uniform is drawn on every non-goal step) and a pre-drawn action index
        sequence, for both hazard reward variants.
    When: The native kernel and the Python reference rollout are each run under
        the same native seed.
    Then: The two discounted returns agree to floating-point tolerance and are
        not trivially zero.

    Test type: integration
    """
    env = _danger_env(
        dangerous_areas=[(4.0, 4.0), (6.0, 6.0)],
        dangerous_area_radius=10.0,
        dangerous_area_hit_probability=hit_prob,
        reward_model_type=model,
        penalty_decay=decay,
    )
    initial = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0, 0.0])
    max_depth = 15
    action_indices = np.random.default_rng(7).integers(0, 4, size=max_depth).astype(np.int64)

    _native.set_seed(20240712)
    native_return = _native_rollout(env, initial, action_indices, max_depth, 0.95, True)

    _native.set_seed(20240712)
    python_return = env._python_simulate_random_rollout(  # pylint: disable=protected-access
        state=initial,
        actions=[env.actions[i] for i in action_indices],
        max_depth=max_depth,
        discount_factor=0.95,
    )

    np.testing.assert_allclose(native_return, python_return, atol=1e-9, rtol=0.0)
    assert abs(native_return) > 0.0


# ---------------------------------------------------------------------------
# (e) terminal is absorbing and drives is_terminal
# ---------------------------------------------------------------------------


def test_terminal_state_is_absorbing():
    """A slot-terminal state echoes itself and yields a zero-return rollout.

    Purpose: Validates the absorbing contract on the scalar transition and the
        rollout entry point.

    Given: A flag-on env and a 7-D state whose terminal slot is already ``1.0``.
    When: A transition is sampled and a rollout is started from it.
    Then: The next state equals the input, ``is_terminal`` is ``True``, and the
        rollout return is exactly ``0.0``.

    Test type: unit
    """
    env = _danger_env()
    dead = np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0])
    _native.set_seed(3)
    next_state = env.sample_next_state(dead, "left")
    np.testing.assert_array_equal(next_state, dead)
    assert env.is_terminal(dead) is True
    assert env.simulate_random_rollout(dead, None, 10, 0.95) == 0.0


def test_batch_transition_sets_terminal_slot():
    """The batch transition sets, latches and skips the terminal slot per row.

    Purpose: Validates flag/param threading through the vectorized updater.

    Given: A flag-on env and a batch of a live in-zone-bound particle, an
        already-terminal particle, and a live particle that stays out of zone.
    When: The batch transition and the batch reward are evaluated for "up".
    Then: The first row terminates, the second is echoed verbatim, the third
        stays live, and the reward batch has one entry per particle.

    Test type: integration
    """
    env = _danger_env()
    particles = np.array(
        [
            [3.0, 2.0, 8.0, 8.0, 9.0, 9.0, 0.0],
            [3.0, 2.0, 8.0, 8.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 8.0, 8.0, 9.0, 9.0, 0.0],
        ]
    )
    next_particles = env.sample_next_state_batch(particles, "up")
    np.testing.assert_array_equal(
        next_particles,
        np.array(
            [
                [3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0],
                [3.0, 2.0, 8.0, 8.0, 9.0, 9.0, 1.0],
                [1.0, 2.0, 8.0, 8.0, 9.0, 9.0, 0.0],
            ]
        ),
    )
    assert env.reward_batch(particles, "up").shape == (3,)


def test_is_terminal_disjunction():
    """``is_terminal`` fires on the terminal slot and on the goal.

    Purpose: Validates the ``is_terminal`` disjunction for a flag-on env,
        including 6-D states handed in by callers.

    Given: A flag-on env, a live 7-D state, a slot-terminal state, a 6-D goal
        state and a 7-D live goal state.
    When: ``is_terminal`` is queried on each.
    Then: Only the live non-goal state is ``False``.

    Test type: unit
    """
    env = _danger_env()
    assert env.is_terminal(LIVE) is False
    assert env.is_terminal(np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0])) is True
    assert env.is_terminal(np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0])) is True
    assert env.is_terminal(np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 0.0])) is True


def test_step_info_goal_reached_excludes_hazard_termination():
    """``step_info`` reports a hazard death as NOT reaching the goal.

    Purpose: Validates that the GOAL_REACHED channel (and hence
        ``goal_reaching_rate``) is a goal-only predicate under the flag, so a
        hazard-terminated episode is not counted as a success.

    Given: A flag-on env, a hazard-terminated non-goal state, and a goal state.
    When: ``step_info`` is evaluated on each.
    Then: The hazard state reports ``0.0`` (although it is terminal) and the
        goal state reports ``1.0``.

    Test type: unit
    """
    env = _danger_env()
    hazard_dead = np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0])
    at_goal = np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 0.0])
    assert env.is_terminal(hazard_dead) is True
    assert env.step_info(hazard_dead, "up", hazard_dead)["goal_reached"] == 0.0
    assert env.step_info(at_goal, "up", at_goal)["goal_reached"] == 1.0


def test_observation_carries_terminal_slot():
    """Observations follow the state width and copy the terminal slot verbatim.

    Purpose: Validates the observation layout and that the extra slot does not
        change the (object-position-only) observation likelihood.

    Given: A flag-on env and a 7-D terminal next state.
    When: An observation is sampled and scored.
    Then: The observation is 7-D with the slot, robot and target entries copied
        exactly, its log-probability is finite and equals the flag-off 6-D
        value, and it hashes.

    Test type: unit
    """
    env = _danger_env()
    next_state = np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0])
    observation = env.sample_observation(next_state, "up")
    assert observation.shape == (7,)
    assert observation[6] == 1.0
    np.testing.assert_array_equal(observation[:2], next_state[:2])
    np.testing.assert_array_equal(observation[4:6], next_state[4:6])

    # Signature is ``observation_log_probability(next_state, action, observations)``.
    log_prob = env.observation_log_probability(next_state, "up", observation)
    assert np.all(np.isfinite(log_prob))
    flat_env = PushPOMDP(
        0.95,
        **push_pinned_kwargs(dangerous_areas=[(3.0, 3.0)], observation_noise=env.observation_noise),
    )
    np.testing.assert_allclose(
        log_prob,
        flat_env.observation_log_probability(next_state[:6], "up", observation[:6]),
        atol=1e-12,
        rtol=0.0,
    )
    assert isinstance(hash(env.hash_observation(observation)), int)


def test_vectorized_model_declines_hazard_terminal():
    """The torch vectorized model declines the draw-coupled hazard-terminal env.

    Purpose: Validates the scope guard: the vectorized model does not model the
        absorbing slot, so it must decline rather than silently mis-plan.

    Given: A flag-on env.
    When: ``PushVectorizedModel`` is constructed from it.
    Then: ``NotImplementedError`` is raised.

    Test type: unit
    """
    from POMDPPlanners.environments.push_pomdp.push_vectorized_model import (  # pylint: disable=import-outside-toplevel
        PushVectorizedModel,
    )

    with pytest.raises(NotImplementedError):
        PushVectorizedModel(_danger_env())


def test_transition_log_probability_marginalises_slot():
    """``transition_log_probability`` compares geometry only, without RNG draws.

    Purpose: Validates that the terminal slot is marginalised out of the
        closed-form transition probability and that a probability query does
        not perturb the C++ RNG stream.

    Given: A flag-on env with a sub-unit hit probability and candidate next
        states differing only in the terminal slot, plus a non-reachable one.
    When: The log-probabilities are queried, and a transition is sampled with
        and without an interleaved query under the same seed.
    Then: Both slot variants score ``log(1) == 0.0``, the unreachable candidate
        scores ``-inf``, and the interleaved query leaves the draw unchanged.

    Test type: unit
    """
    env = _danger_env(dangerous_area_hit_probability=0.3)
    next_slot_1 = np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 1.0])
    next_slot_0 = np.array([3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 0.0])
    log_probs = env.transition_log_probability(LIVE, "up", [next_slot_1, next_slot_0, LIVE])
    np.testing.assert_allclose(log_probs[:2], np.array([0.0, 0.0]), atol=1e-12, rtol=0.0)
    assert log_probs[2] == -np.inf

    _native.set_seed(5)
    without_query = env.sample_next_state(LIVE, "up")
    _native.set_seed(5)
    env.transition_log_probability(LIVE, "up", [LIVE])
    with_query = env.sample_next_state(LIVE, "up")
    np.testing.assert_array_equal(without_query, with_query)


def test_serialization_round_trip_preserves_flag():
    """Pickle and dict round trips preserve the hazard-terminal configuration.

    Purpose: Validates that the flag is part of the env's serialized config and
        of its configuration identity.

    Given: A flag-on env and an otherwise-equivalent flag-off env.
    When: The flag-on env is pickled and round-tripped through
        ``to_dict`` / ``from_dict``, and both ``config_id`` values compared.
    Then: Both round trips keep the flag (and the 7-D sampling), the dict round
        trip compares equal to the original, and the two config ids differ.

    Test type: unit
    """
    env = _danger_env()
    unpickled = pickle.loads(pickle.dumps(env))
    assert unpickled.is_dangerous_area_hit_terminal is True
    assert unpickled.initial_state_dist().sample()[0].shape == (7,)

    restored = PushPOMDP.from_dict(env.to_dict())
    assert isinstance(restored, PushPOMDP)
    assert restored.is_dangerous_area_hit_terminal is True
    assert restored == env

    flag_off = PushPOMDP(0.95, **push_pinned_kwargs(dangerous_areas=[(3.0, 3.0)]))
    assert env.config_id != flag_off.config_id
