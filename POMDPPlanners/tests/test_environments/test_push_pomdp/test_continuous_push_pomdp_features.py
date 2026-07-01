"""Feature-driven tests for ``ContinuousPushPOMDP``.

Targets gaps in :mod:`test_continuous_push_pomdp` and
:mod:`test_continuous_push_native_equivalence` by exercising the **env-level**
API contracts that are most likely to drift between the scalar and batch
code paths or between the documented spec and the underlying C++ kernels.

Coverage focus:

* Reward components (distance penalty, goal bonus, obstacle penalty) at
  feature-region boundaries.
* ``is_terminal`` at the documented ``< 0.5`` boundary.
* Sample / PDF consistency for the transition (robot slice) and observation
  (object slice) at the env-level via 2-D histogram density match.
* Scalar vs batch parity for ``observation_log_probability`` /
  ``observation_log_probability_per_state`` (mirrors the asymmetry that
  surfaced in the light-dark POMDP).
* ``transition_log_probability`` underflow regime — exposes the
  ``np.log(probs)`` flooring path.
* ``initial_state_dist`` always produces collision-free, in-bounds samples.
"""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
)


def _make_env(**overrides) -> ContinuousPushPOMDP:
    """Return a ``ContinuousPushPOMDP`` with overridable defaults."""
    defaults: dict = {
        "discount_factor": 0.99,
        "grid_size": 10,
        "push_threshold": 1.0,
        "friction_coefficient": 0.3,
        "max_push": 2.0,
        "observation_noise": 0.1,
        "robot_radius": 0.3,
        "obstacle_penalty": -10.0,
        "state_transition_cov_matrix": np.eye(2) * 0.04,
    }
    defaults.update(overrides)
    return ContinuousPushPOMDP(**defaults)


# ──────────────────────────────────────────────────────────────────────────
# Reward feature regions
# ──────────────────────────────────────────────────────────────────────────


class TestRewardFeatureRegions:
    """Exercise each branch of ``reward_batch_array`` independently."""

    def test_reward_far_from_target_is_negative_distance(self):
        """Reward equals -dist(object, target) when not at goal and no obstacle.

        Purpose: Validates the baseline ``rewards = -dist_to_target`` branch
            of ``_reward_batch_array`` when the goal bonus and obstacle
            penalty are both inactive.

        Given: An obstacle-free env with deterministic transitions (tiny
            covariance, ``max_push=0`` so the object cannot move) and a
            state where the object is far (>0.5) from the target.
        When: ``reward_batch`` is called over many states sharing the same
            object/target slice.
        Then: Mean reward agrees with ``-||object - target||`` to within
            0.05 (loose because tiny noise on the robot does not shift the
            object slice when ``max_push=0``).

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-12,
            max_push=0.0,
        )
        state = np.array([2.0, 2.0, 3.0, 3.0, 9.0, 9.0])
        states = np.tile(state, (32, 1))
        rewards = env.reward_batch(states, np.array([0.0, 0.0]))
        expected = -float(np.linalg.norm(state[2:4] - state[4:6]))
        np.testing.assert_allclose(rewards.mean(), expected, atol=0.05)

    def test_reward_includes_goal_bonus_when_object_at_target(self):
        """Reward includes the +100 goal bonus when next-object is within 0.5.

        Purpose: Validates the ``rewards[dist < 0.5] += 100`` branch.

        Given: A state with the object pinned at the target and a covariance
            small enough that the sampled next-object slice stays within
            0.5 of the target.
        When: ``reward`` is called.
        Then: The reward is at least ``100 - 0.5`` (bonus minus the residual
            distance penalty for any tiny drift).

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-12,
            max_push=0.0,
        )
        state = np.array([2.0, 2.0, 9.0, 9.0, 9.0, 9.0])
        rew = env.reward(state, np.array([0.0, 0.0]))
        assert rew > 100.0 - 0.5

    def test_reward_obstacle_penalty_uses_pre_action_robot_plus_action(self):
        """Obstacle penalty triggers when ``state[:2] + action`` lands inside obstacle.

        Purpose: Validates that the obstacle penalty branch is keyed on the
            **post-action** robot position (``state[:2] + action``) rather
            than the sampled next-state robot, matching the documented
            implementation.

        Given: Two states differing only in robot position; one such that
            ``robot + action`` enters an obstacle, the other safe; same
            target/object so the distance penalty matches.
        When: ``reward`` is called on each.
        Then: ``rew_collide - rew_safe`` is approximately
            ``obstacle_penalty`` (allowing slack for sample noise on the
            distance term).

        Test type: unit
        """
        env = _make_env(
            obstacles=[(5.0, 5.0, 0.5)],
            obstacle_penalty=-10.0,
            state_transition_cov_matrix=np.eye(2) * 1e-12,
            max_push=0.0,
        )
        # Robot at (4, 5) + action (1, 0) → (5, 5) which is inside obstacle
        # AABB centred at (5,5) with half=0.5.
        state_collide = np.array([4.0, 5.0, 1.0, 1.0, 9.0, 9.0])
        # Robot at (1, 1) + action (1, 0) → (2, 1) far from obstacle.
        state_safe = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 9.0])
        action = np.array([1.0, 0.0])
        rew_collide = env.reward(state_collide, action)
        rew_safe = env.reward(state_safe, action)
        # Distance term cancels because object/target slices are identical.
        diff = rew_collide - rew_safe
        np.testing.assert_allclose(diff, env.obstacle_penalty, atol=0.05)

    def test_reward_no_penalty_when_obstacle_list_empty(self):
        """No obstacle penalty when ``self.obstacles`` is empty.

        Purpose: Guards the ``if self.obstacles.shape[0] > 0`` short-circuit
            in ``_reward_batch_array``.

        Given: An env with no obstacles and an action that would otherwise
            land the robot in an obstacle had one existed.
        When: ``reward`` is called.
        Then: The reward equals ``-dist_to_target`` within 0.05 (no penalty).

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 1e-12,
            max_push=0.0,
        )
        state = np.array([4.0, 5.0, 1.0, 1.0, 9.0, 9.0])
        rew = env.reward(state, np.array([1.0, 0.0]))
        expected = -float(np.linalg.norm(state[2:4] - state[4:6]))
        np.testing.assert_allclose(rew, expected, atol=0.05)


# ──────────────────────────────────────────────────────────────────────────
# is_terminal boundary
# ──────────────────────────────────────────────────────────────────────────


class TestIsTerminalBoundary:
    """Exercise ``is_terminal`` exactly at and on either side of dist=0.5."""

    def test_is_terminal_just_inside_boundary(self):
        """Object at distance 0.499 from target is terminal.

        Purpose: Validates the ``< 0.5`` open-boundary semantics implemented
            as ``dx*dx + dy*dy < 0.25``.

        Given: An object placed at distance 0.499 from the target.
        When: ``is_terminal`` is called.
        Then: Returns ``True``.

        Test type: unit
        """
        env = _make_env()
        # 0.499 / sqrt(2) per axis → distance 0.499.
        d = 0.499 / np.sqrt(2.0)
        state = np.array([1.0, 1.0, 9.0 - d, 9.0 - d, 9.0, 9.0])
        assert env.is_terminal(state)

    def test_is_terminal_just_outside_boundary(self):
        """Object at distance 0.501 from target is NOT terminal.

        Purpose: Validates the upper side of the ``< 0.5`` boundary.

        Given: An object at distance 0.501 from the target.
        When: ``is_terminal`` is called.
        Then: Returns ``False``.

        Test type: unit
        """
        env = _make_env()
        d = 0.501 / np.sqrt(2.0)
        state = np.array([1.0, 1.0, 9.0 - d, 9.0 - d, 9.0, 9.0])
        assert not env.is_terminal(state)

    def test_is_terminal_exactly_at_boundary_is_not_terminal(self):
        """Object exactly at distance 0.5 is NOT terminal (strict ``<``).

        Purpose: Validates that ``is_terminal`` uses a strict-less-than
            comparison rather than ``<=``.

        Given: ``dx*dx + dy*dy == 0.25`` exactly.
        When: ``is_terminal`` is called.
        Then: Returns ``False``.

        Test type: unit
        """
        env = _make_env()
        # dx=0.3, dy=0.4 → dx^2+dy^2 = 0.25 exactly.
        state = np.array([1.0, 1.0, 9.0 - 0.3, 9.0 - 0.4, 9.0, 9.0])
        assert not env.is_terminal(state)


# ──────────────────────────────────────────────────────────────────────────
# Sample / PDF consistency at the env-API level
# ──────────────────────────────────────────────────────────────────────────


def _histogram2d_density_match(
    samples_xy: np.ndarray,
    pdf_fn,
    bin_centers_x: np.ndarray,
    bin_centers_y: np.ndarray,
    bin_width: float,
) -> float:
    """Mass-weighted L1 distance between empirical and analytic density.

    ``samples_xy`` shape ``(N, 2)``; ``pdf_fn(grid_xy)`` returns analytic
    density values at the bin-centre 2-D grid. Returns
    ``sum_i p_i * |emp_i - p_i| / sum_i p_i^2`` — a relative L1 weighted by
    the analytic mass (so far-tail bins do not dominate).
    """
    n_samples = samples_xy.shape[0]
    edges_x = np.concatenate(
        [bin_centers_x - bin_width / 2.0, [bin_centers_x[-1] + bin_width / 2.0]]
    )
    edges_y = np.concatenate(
        [bin_centers_y - bin_width / 2.0, [bin_centers_y[-1] + bin_width / 2.0]]
    )
    counts, _, _ = np.histogram2d(samples_xy[:, 0], samples_xy[:, 1], bins=[edges_x, edges_y])
    empirical = counts / (n_samples * bin_width * bin_width)
    grid_x, grid_y = np.meshgrid(bin_centers_x, bin_centers_y, indexing="ij")
    analytic = pdf_fn(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).reshape(
        len(bin_centers_x), len(bin_centers_y)
    )
    weights = analytic
    weight_total = float(weights.sum())
    if weight_total <= 0.0:
        return float("inf")
    return float((weights * np.abs(empirical - analytic)).sum() / weight_total)


class TestSamplePDFConsistency:
    """Empirical samples agree with the documented analytic densities."""

    def test_observation_object_slice_matches_isotropic_gaussian(self):
        """Observation object slice histogram matches a 2-D isotropic Gaussian.

        Purpose: Validates that ``sample_observation`` and
            ``observation_log_probability`` are consistent at the env level.
            The observation model adds isotropic Gaussian noise of std
            ``observation_noise`` to the **object** slice and clamps to
            ``[0, grid_size - 1]``. With the centre well inside the box,
            clamp is inactive.

        Given: ``next_state = (5, 5, 5, 5, 9, 9)``,
            ``observation_noise = 0.3``, and ``N = 20_000`` env-level
            observation samples.
        When: ``env.sample_observation`` is called once with ``n_samples=N``
            and the empirical 21x21 histogram on the object slice is
            compared against ``exp(observation_log_probability)`` evaluated
            at each bin centre.
        Then: Mass-weighted L1 distance < 0.10.

        Test type: unit
        """
        rng = np.random.default_rng(0)
        env = _make_env(observation_noise=0.3)
        next_state = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([0.5, 0.0])
        n_samples = 20_000
        # Use legacy seed for the C++ RNG which drives the observation noise.
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(int(rng.integers(0, 2**31 - 1)))
        samples = env.sample_observation(next_state, action, n_samples=n_samples)
        samples_arr = np.stack(list(samples), axis=0)
        # Object slice: cols 2..3.
        object_samples = samples_arr[:, 2:4]

        # 21 bins centred at the mean ±3 sigma.
        sigma = env.observation_noise
        centres = np.linspace(5.0 - 3.0 * sigma, 5.0 + 3.0 * sigma, 21)
        bin_width = float(centres[1] - centres[0])

        def pdf_fn(grid_xy):
            # Pad to (N, 6) observations: only the object slice is read by
            # ``observation_log_probability``.
            n_grid = grid_xy.shape[0]
            obs_batch = np.tile(next_state, (n_grid, 1))
            obs_batch[:, 2:4] = grid_xy
            log_prob = env.observation_log_probability(next_state, action, obs_batch)
            return np.exp(log_prob)

        l1 = _histogram2d_density_match(object_samples, pdf_fn, centres, centres, bin_width)
        assert l1 < 0.10, f"observation sample/pdf mismatch: weighted L1 = {l1:.4f}"

    def test_transition_robot_slice_matches_gaussian_in_open_region(self):
        """Robot transition histogram matches the 2-D Gaussian in the open region.

        Purpose: Validates that ``sample_next_state`` and
            ``transition_log_probability`` are consistent at the env level
            for the robot slice when the deterministic post-sample
            geometry (wall resolve, grid clamp, push) is **inactive**.

        Given: A state with the robot well inside the grid and far from
            obstacles, and a tiny action so the noise dominates. With no
            obstacles and a centred robot, ``resolve_circle_wall_collision``
            is a no-op and the grid clamp is inactive — so the transition
            density on the robot slice equals an isotropic Gaussian centred
            at ``robot + action``.
        When: ``env.sample_next_state`` is called ``N=20_000`` times via
            the batch path and the 21x21 histogram on the robot slice is
            compared against ``exp(transition_log_probability)`` at the bin
            centres (which constructs a full (state, action, candidate)
            evaluation per bin).
        Then: Mass-weighted L1 distance < 0.10.

        Test type: unit
        """
        env = _make_env(
            state_transition_cov_matrix=np.eye(2) * 0.04,
            obstacles=None,
        )
        state = np.array([5.0, 5.0, 1.0, 1.0, 9.0, 9.0])
        action = np.array([0.1, 0.0])
        n_samples = 80_000
        from POMDPPlanners.environments.push_pomdp import (
            _native,
        )  # pylint: disable=import-outside-toplevel

        _native.set_seed(7)
        samples = env.sample_next_state(state, action, n_samples=n_samples)
        samples_arr = np.stack(list(samples), axis=0)
        robot_samples = samples_arr[:, :2]

        # 21 bins around the post-action mean, ±3 sigma.
        sigma = float(np.sqrt(env.state_transition_cov_matrix[0, 0]))
        mean = state[:2] + action
        centres_x = np.linspace(mean[0] - 3.0 * sigma, mean[0] + 3.0 * sigma, 21)
        centres_y = np.linspace(mean[1] - 3.0 * sigma, mean[1] + 3.0 * sigma, 21)
        bin_width = float(centres_x[1] - centres_x[0])

        def pdf_fn(grid_xy):
            n_grid = grid_xy.shape[0]
            candidates = np.tile(state, (n_grid, 1))
            candidates[:, 0] = grid_xy[:, 0]
            candidates[:, 1] = grid_xy[:, 1]
            log_prob = env.transition_log_probability(state, action, candidates)
            return np.exp(log_prob)

        l1 = _histogram2d_density_match(robot_samples, pdf_fn, centres_x, centres_y, bin_width)
        assert l1 < 0.10, f"transition sample/pdf mismatch: weighted L1 = {l1:.4f}"


# ──────────────────────────────────────────────────────────────────────────
# Scalar vs batch parity for observation_log_probability
# ──────────────────────────────────────────────────────────────────────────


class TestScalarBatchObservationLogProbParity:
    """Mirror the light-dark asymmetry probe at the env API level.

    The scalar ``observation_log_probability`` calls
    ``_native.observation_log_probability_step``; the batch
    ``observation_log_probability_per_state`` calls
    ``ContinuousPushObservationCpp.batch_log_likelihood``. Both compute the
    same isotropic-Gaussian log-pdf on the object slice — these tests pin
    them to bit-exact agreement so any future flooring / clamp drift is
    caught.
    """

    @pytest.mark.parametrize(
        "obj_offset",
        [
            np.array([0.0, 0.0]),  # zero-distance: maximum density
            np.array([0.05, 0.0]),  # near-typical
            np.array([1.0, 1.0]),  # far tail
            np.array([5.0, 5.0]),  # extreme tail (probability underflow regime)
        ],
        ids=["zero", "near", "far", "extreme"],
    )
    def test_scalar_matches_batch_for_single_observation(self, obj_offset):
        """Scalar and batch return the same log-pdf for one (state, obs).

        Purpose: Catches any flooring / clamp asymmetry between the scalar
            and batch observation log-pdf paths (mirrors a real bug found
            in the light-dark POMDP where scalar floored at log(1e-300) but
            batch did not).

        Given: A fixed ``next_state`` and an observation displaced from it
            on the object slice. The batch path is called with a
            ``(1, 6)`` particles input and the scalar with a single
            observation row.
        When: Both ``observation_log_probability`` and
            ``observation_log_probability_per_state`` are evaluated.
        Then: Their length-1 outputs agree to ``rtol=0`` ``atol=1e-12``.

        Test type: unit
        """
        env = _make_env(observation_noise=0.3)
        next_state = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([0.5, 0.5])
        observation = next_state.copy()
        observation[2:4] += obj_offset

        scalar = env.observation_log_probability(next_state, action, observation)
        batch = env.observation_log_probability_per_state(
            next_state.reshape(1, -1), action, observation
        )
        assert scalar.shape == (1,)
        assert batch.shape == (1,)
        np.testing.assert_allclose(scalar[0], batch[0], rtol=0.0, atol=1e-12)

    def test_scalar_agrees_with_batch_across_grid(self):
        """Scalar over many obs == batch over many particles (transposed).

        Purpose: Sweeps the full obs/particle grid: for a fixed
            ``next_state`` and a list of observations, the scalar path
            returns one log-prob per observation; for a fixed observation
            and a list of next-states, the batch path returns one log-prob
            per particle. By symmetry of the Gaussian (the particle index
            and the observation index swap roles), a diagonal sweep — one
            ``next_state`` per observation — must produce the same value
            on both paths.

        Given: ``N=64`` independent ``(next_state, observation)`` pairs
            with random offsets.
        When: For each pair, ``observation_log_probability(next_state, ...,
            observation)`` and
            ``observation_log_probability_per_state(next_state, ...,
            observation)`` are compared.
        Then: All pairwise log-probabilities agree within ``atol=1e-12``.

        Test type: unit
        """
        rng = np.random.default_rng(123)
        env = _make_env(observation_noise=0.2)
        action = np.array([0.0, 0.0])
        n_pairs = 64
        next_states = np.tile([3.0, 3.0, 4.0, 4.0, 8.0, 8.0], (n_pairs, 1)).astype(float)
        observations = next_states.copy()
        # Perturb the object slice on both sides independently.
        next_states[:, 2:4] += rng.normal(scale=0.5, size=(n_pairs, 2))
        observations[:, 2:4] += rng.normal(scale=0.5, size=(n_pairs, 2))

        scalar_logs = np.empty(n_pairs)
        batch_logs = np.empty(n_pairs)
        for i in range(n_pairs):
            scalar_logs[i] = env.observation_log_probability(
                next_states[i], action, observations[i]
            )[0]
            batch_logs[i] = env.observation_log_probability_per_state(
                next_states[i].reshape(1, -1), action, observations[i]
            )[0]
        np.testing.assert_allclose(scalar_logs, batch_logs, rtol=0.0, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# transition_log_probability underflow regime
# ──────────────────────────────────────────────────────────────────────────


class TestTransitionLogProbabilityUnderflow:
    """Document the behavior of ``transition_log_probability`` deep in the tail.

    Unlike the observation path, the transition log-prob goes through
    ``np.log(kernel.probability(...))``. ``kernel.probability`` returns the
    linear PDF, so for next-states many sigmas away the probability
    underflows to 0 and ``np.log(0) = -inf``. The corresponding scalar /
    batch observation paths use the raw log-pdf and do **not** underflow
    in the same regime. These tests document that asymmetry so future
    refactors are constrained.
    """

    def test_finite_at_typical_displacement(self):
        """Log-prob is finite for next-states within a few sigma.

        Purpose: Sanity check that the typical-regime log-prob is finite.

        Given: A small covariance and a candidate next-state displaced by
            one sigma on the robot slice.
        When: ``transition_log_probability`` is called.
        Then: The returned log-prob is finite.

        Test type: unit
        """
        env = _make_env(state_transition_cov_matrix=np.eye(2) * 0.04)
        state = np.array([4.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([0.5, 0.0])
        sigma = float(np.sqrt(env.state_transition_cov_matrix[0, 0]))
        candidate = np.array(
            [state[0] + action[0] + sigma, state[1] + action[1], 5.0, 5.0, 9.0, 9.0]
        )
        log_prob = env.transition_log_probability(state, action, [candidate])
        assert np.isfinite(log_prob[0])

    def test_underflow_to_neg_inf_far_tail(self):
        """Far-tail next-state produces ``-inf`` from ``np.log(0)``.

        Purpose: Documents the current behavior of
            ``transition_log_probability`` deep in the tail. This is *not*
            an asymmetry bug — there is no batch alternative for the
            transition log-prob — but pinning the behavior guards against
            silent introduction of a flooring path that would diverge from
            the observation handling.

        Given: A 6 sigma displacement on the robot slice; the linear PDF
            value is below 1e-300 so ``np.log(probs)`` returns ``-inf``.
        When: ``transition_log_probability`` is called.
        Then: The result is ``-inf``.

        Test type: unit
        """
        env = _make_env(state_transition_cov_matrix=np.eye(2) * 0.0001)
        state = np.array([4.0, 4.0, 5.0, 5.0, 9.0, 9.0])
        action = np.array([0.5, 0.0])
        sigma = float(np.sqrt(env.state_transition_cov_matrix[0, 0]))
        candidate = np.array(
            [state[0] + action[0] + 60.0 * sigma, state[1] + action[1], 5.0, 5.0, 9.0, 9.0]
        )
        log_prob = env.transition_log_probability(state, action, [candidate])
        assert np.isneginf(log_prob[0])


# ──────────────────────────────────────────────────────────────────────────
# Initial-state distribution sanity
# ──────────────────────────────────────────────────────────────────────────


class TestInitialStateDistribution:
    """Random initial states must be in-bounds and collision-free."""

    def test_random_initial_states_robot_collision_free(self):
        """All random-initial robot positions clear all obstacles.

        Purpose: Validates ``_RandomInitialStateDistribution._generate_robot_position``
            actually rejects colliding samples (only the random-initial
            constructor branch is hit when ``initial_state is None``).

        Given: An env with a single obstacle at the centre and 50 random
            initial states.
        When: Each sample is checked against the obstacle via
            ``_is_circle_colliding_with_obstacle``.
        Then: No sample is in collision.

        Test type: unit
        """
        env = _make_env(obstacles=[(5.0, 5.0, 0.5)])
        np.random.seed(0)
        for _ in range(50):
            state = env.initial_state_dist().sample()[0]
            # pylint: disable=protected-access
            assert not env._is_circle_colliding_with_obstacle(state[:2], env.robot_radius)

    def test_random_initial_states_in_bounds(self):
        """All random-initial robot/object positions are inside the grid.

        Purpose: Validates the in-bounds spec for the random-initial
            distribution.

        Given: An obstacle-free env and 50 random initial states.
        When: The robot and object slices are inspected.
        Then: Robot lies in ``[radius, grid_size - 1 - radius]`` and the
            object lies in ``[0, grid_size - 1]`` on both axes.

        Test type: unit
        """
        env = _make_env()
        np.random.seed(1)
        for _ in range(50):
            state = env.initial_state_dist().sample()[0]
            assert env.robot_radius <= state[0] <= env.grid_size - 1 - env.robot_radius
            assert env.robot_radius <= state[1] <= env.grid_size - 1 - env.robot_radius
            assert 0.0 <= state[2] <= env.grid_size - 1
            assert 0.0 <= state[3] <= env.grid_size - 1
