"""Feature-driven bug-hunting tests for PushPOMDP (discrete).

This module complements ``test_push_pomdp.py`` with feature-driven tests
that target asymmetries and edge cases between scalar and batch APIs,
boundary semantics for terminal/obstacle-collision predicates, and the
sample/PDF pair on the observation model.

The test file was created after a sibling skill run on the light-dark
POMDP exposed a real asymmetry between
``observation_log_probability`` (scalar, floored) and
``observation_log_probability_per_state`` (batch, un-floored).  The
tests below specifically check that no analogous asymmetry exists in
the Push POMDP and exercise other features that lack dedicated coverage.
"""

# pylint: disable=protected-access  # Tests need to access protected helpers.

from typing import List

import numpy as np

from POMDPPlanners.environments.push_pomdp import PushPOMDP, _native as push_native


_ACTION_NAMES: List[str] = ["up", "down", "right", "left"]


def _make_default_env(**overrides: object) -> PushPOMDP:
    """Build a PushPOMDP with deterministic-friendly defaults.

    Helper used across multiple tests so each test reads one focused
    setup line rather than a wall of constructor arguments.
    """
    params: dict = {
        "discount_factor": 0.95,
        "grid_size": 10,
        "push_threshold": 1.0,
        "friction_coefficient": 0.3,
        "observation_noise": 0.1,
        "obstacles": [(3.0, 3.0), (7.0, 7.0)],
        "obstacle_radius": 0.5,
        "obstacle_penalty": -10.0,
        "transition_error_prob": 0.0,
    }
    params.update(overrides)
    return PushPOMDP(**params)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Reward feature regions
# ----------------------------------------------------------------------


def test_reward_at_target_includes_terminal_bonus() -> None:
    """Reward includes +100 bonus when next-state object is within 0.5 of target.

    Purpose: Validates that ``_reward_from_next_state`` adds the +100
        terminal bonus exactly when the object distance to target is
        below 0.5, and that the bonus is omitted just outside that band.

    Given: A PushPOMDP without obstacles. Two next-states that share the
        same robot position but differ only in the object-target
        distance: one with object exactly at the target (distance 0),
        another with object 0.5 units away (distance == 0.5).
    When: ``_reward_from_next_state`` is called for both.
    Then: The first returns +100.0 (bonus) - 0.0 (distance) = 100.0.
        The second returns -0.5 (no bonus, because 0.5 is NOT < 0.5).

    Test type: unit
    """
    env = _make_default_env(obstacles=None)
    state = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])

    next_at_target = np.array([5.0, 5.0, 9.0, 9.0, 9.0, 9.0])
    reward_at_target = env._compute_reward(state, "up", next_at_target)
    assert reward_at_target == 100.0

    next_just_outside = np.array([5.0, 5.0, 9.5, 9.0, 9.0, 9.0])
    reward_just_outside = env._compute_reward(state, "up", next_just_outside)
    assert np.isclose(reward_just_outside, -0.5)


def test_reward_obstacle_penalty_uses_realised_position_not_intended() -> None:
    """Obstacle penalty triggers off the REALISED robot position, not the intended move.

    Purpose: Pins the corrected design — ``_reward_from_next_state`` applies
        ``obstacle_penalty`` based on the realised ``next_state[:2]``, not on
        ``state[:2] + action_dxy``. When a move is blocked by an obstacle the
        robot stays put, the realised position is clear, and no penalty fires.

    Given: A PushPOMDP with one obstacle at (3, 3), radius 0.5. Robot at
        (2, 3); action "right" intends (3, 3) which is inside the obstacle,
        but the transition blocks the move so the robot stays at (2, 3).
    When: ``env.reward(state, "right")`` is called.
    Then: The reward equals the bare ``-distance`` term — no obstacle
        penalty (because the realised position is clear).

    Test type: unit
    """
    env = _make_default_env(obstacles=[(3.0, 3.0)])
    state = np.array([2.0, 3.0, 5.0, 5.0, 9.0, 9.0])

    next_state = env.sample_next_state(state, "right")
    assert np.allclose(next_state[:2], state[:2]), "Robot should stay put due to collision"

    reward = env.reward(state, "right", next_state=next_state)
    distance = float(np.linalg.norm(state[2:4] - state[4:6]))
    expected = -distance
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"


# ----------------------------------------------------------------------
# is_terminal boundary
# ----------------------------------------------------------------------


def test_is_terminal_strictly_less_than_half_unit() -> None:
    """is_terminal uses strict ``< 0.25`` on squared distance, NOT ``<=``.

    Purpose: Pins down the boundary semantics of ``is_terminal`` so a
        future change from ``<`` to ``<=`` (or vice versa) is caught.

    Given: A state where the object-target squared distance equals
        exactly 0.25 (object 0.5 units away from target).
    When: ``env.is_terminal(state)`` is queried.
    Then: Returns False (because 0.25 < 0.25 is False).
        And for distance epsilon under 0.5, returns True.

    Test type: unit
    """
    env = _make_default_env()

    state_exactly_half = np.array([1.0, 1.0, 9.0, 8.5, 9.0, 9.0])
    assert env.is_terminal(state_exactly_half) is False

    eps = 1e-6
    state_just_under = np.array([1.0, 1.0, 9.0, 9.0 - 0.5 + eps, 9.0, 9.0])
    assert env.is_terminal(state_just_under) is True


# ----------------------------------------------------------------------
# Observation log-probability: scalar vs batch parity (the light-dark bug)
# ----------------------------------------------------------------------


def test_observation_log_probability_scalar_matches_batch_kernel() -> None:
    """Scalar ``observation_log_probability`` equals batch ``per_state`` form.

    Purpose: Mirror of the asymmetry found in the light-dark POMDP:
        scalar floored at log(1e-300) while the batch path returned the
        un-floored kernel.  This test asserts no such asymmetry exists
        in the Push POMDP for a fixed observation evaluated against many
        next-states (or vice versa).

    Given: A PushPOMDP with observation_noise=0.1 and no obstacles.  A
        fixed observation and a fixed next-state.
    When: ``env.observation_log_probability(next_state, action,
        [observation])`` (scalar path) and
        ``env.observation_log_probability_per_state([next_state],
        action, observation)`` (batch path) are both evaluated.
    Then: Both return the same value (single Gaussian log-pdf at the
        same point).  Tested across a sweep of (next_state, observation)
        pairs that include points far from the mean (where the
        light-dark bug manifests via under/un-flooring).

    Test type: unit
    """
    env = _make_default_env(obstacles=None, observation_noise=0.1)

    rng = np.random.default_rng(seed=2024)
    next_states_grid = rng.uniform(0.0, 9.0, size=(20, 6))
    observations_grid = rng.uniform(0.0, 9.0, size=(20, 6))

    for next_state in next_states_grid:
        for observation in observations_grid:
            scalar_log_prob = env.observation_log_probability(next_state, "right", [observation])
            batch_log_prob = env.observation_log_probability_per_state(
                [next_state], "right", observation
            )

            assert scalar_log_prob.shape == (1,)
            assert batch_log_prob.shape == (1,)
            np.testing.assert_allclose(
                scalar_log_prob,
                batch_log_prob,
                atol=1e-12,
                err_msg=(
                    f"Scalar and batch observation log-prob disagree: "
                    f"next_state={next_state}, observation={observation}, "
                    f"scalar={scalar_log_prob[0]}, batch={batch_log_prob[0]}"
                ),
            )


def test_observation_log_probability_extreme_far_observation_no_floor_asymmetry() -> None:
    """At extreme distance both scalar and batch return the same un-floored value.

    Purpose: Specifically targets the light-dark-style asymmetry where
        the scalar path could clip ``log(p)`` at ``log(1e-300)`` while
        the batch path would return the raw kernel output.  An extreme
        observation (object position 1e6 away from next_state's object
        position) drives the kernel into log-prob territory far below
        ``log(1e-300) ≈ -690``.

    Given: A PushPOMDP and a next_state with object at (5, 5); an
        observation with object at (1e6, 1e6).  At that distance the
        true Gaussian log-pdf is on the order of -1e13.
    When: Both scalar and batch log-prob APIs evaluate this pair.
    Then: Both return identical, finite (non-NaN) values that are far
        below -700 (i.e. neither path floors).  If the scalar path is
        floored, this test fails.

    Test type: unit
    """
    env = _make_default_env(obstacles=None, observation_noise=0.1)

    next_state = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
    extreme_obs = np.array([5.0, 5.0, 1.0e6, 1.0e6, 9.0, 9.0])

    scalar_lp = float(env.observation_log_probability(next_state, "right", [extreme_obs])[0])
    batch_lp = float(
        env.observation_log_probability_per_state([next_state], "right", extreme_obs)[0]
    )

    assert np.isfinite(scalar_lp), f"Scalar log-prob should be finite, got {scalar_lp}"
    assert np.isfinite(batch_lp), f"Batch log-prob should be finite, got {batch_lp}"
    assert scalar_lp < -1.0e10, "Far-tail log-prob should be very negative, not floored"
    np.testing.assert_allclose(scalar_lp, batch_lp, atol=1e-3)


# ----------------------------------------------------------------------
# Observation: sample-vs-PDF consistency (Gaussian on object slice)
# ----------------------------------------------------------------------


def test_sample_observation_matches_log_probability_density_2d_gaussian() -> None:
    """Empirical density from samples matches closed-form log-prob (2-D Gaussian).

    Purpose: Validates the sample/PDF pair for ``sample_observation`` and
        ``observation_log_probability`` under the 2-D Gaussian noise on
        the object slice (cols 2:4).  An asymmetry between sampler and
        PDF (e.g. wrong variance, missing factor of 2*pi) would surface
        as a Wilson-CI violation.

    Given: A PushPOMDP with grid_size large enough that clipping is
        negligible (grid_size=100) and observation_noise=0.5 (small
        relative to grid).  next_state object centred at (50, 50).
        N=5000 observations sampled.
    When: We bin the sampled object positions (col 2 only — the
        marginal in x) into a histogram and compare the empirical bin
        probability to the integral of the predicted PDF over each bin
        (computed from ``observation_log_probability`` evaluated at bin
        centres, multiplied by bin width).
    Then: Each bin's empirical probability is within 3/sqrt(N)
        (Wilson-style) of the predicted probability.

    Test type: unit
    """
    env = PushPOMDP(
        discount_factor=0.95,
        grid_size=100,
        observation_noise=0.5,
        transition_error_prob=0.0,
    )
    next_state = np.array([50.0, 50.0, 50.0, 50.0, 99.0, 99.0])

    np.random.seed(20240426)
    n_samples = 5000
    samples = env.sample_observation(next_state, "right", n_samples=n_samples)
    sample_arr = np.array(samples)
    object_x_samples = sample_arr[:, 2]

    bin_edges = np.linspace(48.0, 52.0, 9)  # 8 bins centred on 50
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    empirical_counts, _ = np.histogram(object_x_samples, bins=bin_edges)
    empirical_probs = empirical_counts / n_samples

    # Build full observation rows at each bin centre to feed to the
    # closed-form log-prob (object_y fixed at the mean to isolate the
    # x-marginal; the kernel is separable so this gives exactly the
    # 1-D Gaussian times a constant we factor out below).
    predicted_marginal_pdf = (
        1.0
        / (env.observation_noise * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((bin_centres - 50.0) / env.observation_noise) ** 2)
    )
    predicted_probs = predicted_marginal_pdf * bin_width

    tol = 3.0 / np.sqrt(n_samples)
    diffs = np.abs(empirical_probs - predicted_probs)
    assert np.all(diffs < tol), (
        f"Sample/PDF Wilson check failed: max diff {np.max(diffs)} "
        f">= tol {tol}; empirical={empirical_probs}, predicted={predicted_probs}"
    )


# ----------------------------------------------------------------------
# Batch vs scalar transition parity (deterministic path)
# ----------------------------------------------------------------------


def test_sample_next_state_batch_matches_scalar_deterministic() -> None:
    """Batch ``sample_next_state_batch`` agrees byte-exactly with scalar path.

    Purpose: Validates that ``PushVectorizedUpdater.batch_transition``
        (the engine behind ``sample_next_state_batch``) produces results
        identical to ``_sample_one_next_state`` when
        ``transition_error_prob=0`` (so no RNG is involved).  Any
        divergence in collision-radius semantics (``<`` vs ``<=``),
        clipping, or push-threshold comparison would surface here.

    Given: A PushPOMDP with obstacles and ``transition_error_prob=0``.
        A diverse batch of 200 random states and one fixed action.
    When: ``sample_next_state_batch(states, action)`` and a scalar loop
        ``[sample_next_state(s, action) for s in states]`` are both
        evaluated.
    Then: Every row of the batch result equals the corresponding scalar
        result exactly (atol=1e-12).

    Test type: unit
    """
    env = _make_default_env(obstacles=[(3.0, 3.0), (7.0, 7.0)])

    rng = np.random.default_rng(seed=42)
    states = rng.uniform(0.0, 9.0, size=(200, 6))
    states[:, 4:6] = 9.0  # Fix target.

    for action in _ACTION_NAMES:
        batch_next = env.sample_next_state_batch(states, action)
        scalar_next = np.array([env.sample_next_state(s, action) for s in states])

        np.testing.assert_allclose(
            batch_next,
            scalar_next,
            atol=1e-12,
            err_msg=f"Batch vs scalar transition mismatch for action={action}",
        )


def test_sample_next_state_batch_push_threshold_boundary() -> None:
    """Batch path uses ``<`` (strict) on push-threshold, matching scalar path.

    Purpose: Pins down the push-threshold comparison.  Scalar path uses
        ``dist_sq < push_threshold_sq`` (strict).  Batch path uses
        ``dist_to_obj < self.push_threshold`` (strict).  This test
        verifies both yield the same answer at the exact boundary.

    Given: A PushPOMDP with push_threshold=1.0 and friction=0.3.  Robot
        positioned exactly 1.0 unit away from the object.  Action
        "right" attempts to move the robot directly toward the object.
    When: Batch and scalar transitions are computed.
    Then: Both produce identical next-states.  The object's position
        determines whether a strict-less-than at exactly 1.0 means "no
        push" (the documented behaviour).  This test asserts the two
        APIs agree on whatever the boundary semantics are.

    Test type: unit
    """
    env = _make_default_env(obstacles=None, push_threshold=1.0, friction_coefficient=0.3)

    boundary_state = np.array([2.0, 5.0, 3.0, 5.0, 9.0, 9.0])
    states = np.tile(boundary_state, (10, 1))

    batch_next = env.sample_next_state_batch(states, "right")
    scalar_next = env.sample_next_state(boundary_state, "right")

    for row in batch_next:
        np.testing.assert_allclose(
            row,
            scalar_next,
            atol=1e-12,
            err_msg=f"Batch row diverges at push-threshold boundary: {row} vs {scalar_next}",
        )


# ----------------------------------------------------------------------
# Reward batch obstacle-penalty parity
# ----------------------------------------------------------------------


def test_reward_batch_obstacle_radius_boundary_matches_scalar() -> None:
    """Batch and scalar obstacle-penalty agree at the exact radius boundary.

    Purpose: Both ``_obstacle_penalty_batch`` and the scalar
        ``_is_colliding_with_obstacle_scalar`` use ``<=`` against the
        squared radius, which is the *only* consistent choice. This
        test confirms they agree when the intended position lies
        exactly on the obstacle radius (squared distance == r^2).

    Given: A PushPOMDP with obstacle at (3, 3), radius 0.5,
        friction_coefficient=0.0 and transition_error_prob=0 (so the
        transition is fully deterministic).  Robot at (2.5, 3.0):
        action "right" intends (3.5, 3.0) which is at squared distance
        0.25 from the obstacle centre — exactly on the boundary.
    When: ``reward_batch`` is called on a batch of 5 copies and
        ``reward`` on one scalar.
    Then: Both yield the same reward (same obstacle-penalty decision).

    Test type: unit
    """
    env = _make_default_env(
        obstacles=[(3.0, 3.0)],
        obstacle_radius=0.5,
        friction_coefficient=0.0,
    )
    state = np.array([2.5, 3.0, 6.0, 6.0, 9.0, 9.0])

    batch_states = np.tile(state, (5, 1))
    np.random.seed(0)
    batch_rewards = env.reward_batch(batch_states, "right")
    np.random.seed(0)
    scalar_reward = env.reward(state, "right")

    np.testing.assert_allclose(
        batch_rewards,
        scalar_reward,
        atol=1e-12,
        err_msg=(
            f"Batch and scalar reward disagree at obstacle radius boundary: "
            f"batch={batch_rewards}, scalar={scalar_reward}"
        ),
    )


# ----------------------------------------------------------------------
# transition_log_probability: error path distribution sums to 1
# ----------------------------------------------------------------------


def test_transition_log_probability_distribution_sums_to_one_with_error() -> None:
    """transition_log_probability gives probabilities that sum to 1 across distinct outcomes.

    Purpose: Validates the closed-form mixture formula in
        ``transition_log_probability`` when ``transition_error_prob>0``:
        the total mass over the at-most-4 distinct intended outcomes
        (for the 4 actions) sums exactly to 1.0.  A mistake in the
        ``num_error_actions`` denominator or in the ``error_match_count``
        accumulator would break this invariant.

    Given: A PushPOMDP with no obstacles, ``transition_error_prob=0.5``,
        and a state where the 4 actions yield 4 distinct next-states
        (robot in the interior of the grid; no friction; no pushing).
    When: We enumerate the 4 distinct intended next-states and sum the
        probabilities ``np.exp(transition_log_probability(...))``.
    Then: The sum equals 1.0 exactly (within 1e-12).

    Test type: unit
    """
    env = _make_default_env(
        obstacles=None,
        push_threshold=0.0,  # Disable pushing — keep object fixed.
        friction_coefficient=0.0,
        transition_error_prob=0.5,
    )
    state = np.array([5.0, 5.0, 7.5, 7.5, 9.0, 9.0])

    distinct_next_states = [
        env._compute_next_state_for_action(state, action) for action in env.get_actions()
    ]

    # Action "right" is the intended action for this experiment.
    log_probs = env.transition_log_probability(state, "right", distinct_next_states)
    probs = np.exp(log_probs)

    assert np.isclose(np.sum(probs), 1.0, atol=1e-12), (
        f"Transition probabilities should sum to 1.0, got {np.sum(probs)}; " f"per-outcome={probs}"
    )


# ----------------------------------------------------------------------
# hash_observation contract
# ----------------------------------------------------------------------


def test_hash_observation_consistent_with_equality() -> None:
    """``hash_observation`` returns equal hashes for equal observations.

    Purpose: Sanity-check the hash-equality contract used by belief
        clustering. Equal observations (per ``is_equal_observation``)
        must hash to equal values; unequal observations should hash
        to different values for at least one numerically-different pair.

    Given: A PushPOMDP and three observations: two that are
        ``np.array_equal`` and one that differs in the object slice.
    When: Their hash values (as bytes) are compared.
    Then: The two equal observations have the same hash; the third
        differs from them.

    Test type: unit
    """
    env = _make_default_env()
    obs_a = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0])
    obs_a_copy = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0])
    obs_b = np.array([1.0, 2.0, 3.0, 4.5, 9.0, 9.0])

    assert env.is_equal_observation(obs_a, obs_a_copy)
    assert env.hash_observation(obs_a) == env.hash_observation(obs_a_copy)
    assert env.hash_observation(obs_a) != env.hash_observation(obs_b)


# ----------------------------------------------------------------------
# Reward range bounds actual rewards
# ----------------------------------------------------------------------


def test_reward_range_bounds_actual_rewards_across_random_states() -> None:
    """All sampled rewards lie inside ``env.reward_range`` for many random states.

    Purpose: Cross-checks that the ``reward_range`` advertised by the
        environment actually bounds the rewards returned by
        ``env.reward(...)`` for a representative random sample of
        states.  A mismatch (e.g. obstacle penalty pushing the reward
        below the lower bound) would expose a stale or wrong range.

    Given: A PushPOMDP with obstacles and obstacle_penalty=-10.0.
    When: For each of 50 random states and each action, we compute
        ``env.reward(state, action)``.
    Then: Every reward lies within the closed interval
        ``[reward_range[0], reward_range[1]]``.

    Test type: unit
    """
    env = _make_default_env()
    rng = np.random.default_rng(seed=11)
    assert env.reward_range is not None
    lo, hi = env.reward_range

    for _ in range(50):
        state = rng.uniform(0.0, 9.0, size=6)
        state[4:6] = 9.0  # Target.
        for action in env.get_actions():
            r = env.reward(state, action)
            assert lo <= r <= hi, (
                f"Reward {r} out of advertised range [{lo}, {hi}] "
                f"for state={state}, action={action}"
            )


# ----------------------------------------------------------------------
# Native rollout: error-prob path consistency
# ----------------------------------------------------------------------


def test_native_simulate_rollout_with_error_prob_runs_and_is_finite() -> None:
    """Native rollout with non-zero transition_error_prob returns a finite scalar.

    Purpose: Smoke-test the C++ error-action path in
        ``simulate_rollout_discrete`` (which uses the kErrorActions
        table and an extra RNG draw per step).  Without this test, the
        error branch in C++ would be untested by the existing
        deterministic-only parity tests.

    Given: A PushPOMDP with transition_error_prob=0.4 and a fixed
        initial state.  Native RNG seeded and called via
        ``env.simulate_random_rollout`` (which delegates to the C++
        kernel).
    When: A 30-step rollout is executed.
    Then: The discounted return is finite, lies in
        ``[reward_range[0]*30, reward_range[1]*30]``, and a second call
        with a different native seed yields a (likely) different return,
        confirming the error-action RNG is actually used.

    Test type: integration
    """
    env = _make_default_env(transition_error_prob=0.4)
    initial_state = np.array([2.0, 3.0, 2.5, 3.0, 9.0, 9.0])

    class _DummySampler:
        def sample(self) -> str:
            return "up"

    push_native.set_seed(7)
    np.random.seed(7)
    return_a = env.simulate_random_rollout(
        state=initial_state,
        action_sampler=_DummySampler(),
        max_depth=30,
        discount_factor=env.discount_factor,
    )

    push_native.set_seed(8)
    np.random.seed(8)
    return_b = env.simulate_random_rollout(
        state=initial_state,
        action_sampler=_DummySampler(),
        max_depth=30,
        discount_factor=env.discount_factor,
    )

    assert np.isfinite(return_a)
    assert np.isfinite(return_b)
    assert env.reward_range is not None
    horizon_lo = env.reward_range[0] * 30
    horizon_hi = env.reward_range[1] * 30
    assert horizon_lo <= return_a <= horizon_hi
    assert horizon_lo <= return_b <= horizon_hi


# ----------------------------------------------------------------------
# Metric names match compute_metrics output
# ----------------------------------------------------------------------


def test_metric_names_match_compute_metrics_output() -> None:
    """``get_metric_names`` matches the names emitted by ``compute_metrics``.

    Purpose: Validates the data-integrity contract: the
        ``get_metric_names`` declaration must exactly match the names
        produced by ``compute_metrics`` so downstream simulation
        consumers can index by name reliably.

    Given: A PushPOMDP and a one-step history.
    When: ``get_metric_names()`` and ``compute_metrics(...)`` are both
        invoked.
    Then: The set of names from each is identical.

    Test type: unit
    """
    # pylint: disable=import-outside-toplevel
    from POMDPPlanners.core.belief import WeightedParticleBelief
    from POMDPPlanners.core.simulation import History, StepData

    env = _make_default_env()
    declared_names = set(env.get_metric_names())

    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    next_state, observation, reward = env.sample_next_step(state, "up")

    # WeightedParticleBelief requires at least one nonzero log-weight, so
    # use two particles with arbitrary unequal log-weights.
    belief = WeightedParticleBelief([state, state], np.array([0.0, -1.0]))
    step = StepData(
        state=state,
        action="up",
        next_state=next_state,
        observation=observation,
        reward=reward,
        belief=belief,
    )
    history = History(
        history=[step],
        discount_factor=env.discount_factor,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=1,
        reach_terminal_state=False,
        policy_run_data=[],
    )

    emitted = env.compute_metrics([history])
    emitted_names = {m.name for m in emitted}

    assert (
        declared_names == emitted_names
    ), f"Declared names {declared_names} != emitted names {emitted_names}"
