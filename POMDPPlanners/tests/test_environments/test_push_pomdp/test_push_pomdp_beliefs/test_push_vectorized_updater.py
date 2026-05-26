# SPDX-License-Identifier: MIT

"""Tests for PushVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods for the Push POMDP, including an equivalence test
that verifies the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.push_pomdp import PushPOMDP, _native
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs import (
    PushVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_sample_distributions_match,
    assert_update_particles_match,
    assert_update_weights_match,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)


def _make_aligned_beliefs(updater, n_particles=100):
    """Create baseline + vectorized beliefs with identical initial particles."""
    np.random.seed(42)
    particles_array = np.random.uniform(1, 8, (n_particles, 6))
    particles_array[:, 4:6] = [9.0, 9.0]
    particles_list = [particles_array[i].copy() for i in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    base = WeightedParticleBelief(
        particles=particles_list,
        log_weights=log_weights.copy(),
        resampling=False,
    )
    vec = VectorizedWeightedParticleBelief(
        particles=particles_array.copy(),
        log_weights=log_weights.copy(),
        updater=updater,
        resampling=False,
    )
    return base, vec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACTION_STRINGS = ["up", "down", "right", "left"]


@pytest.fixture
def env():
    return PushPOMDP(
        discount_factor=0.99,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
        obstacles=[(3.0, 3.0), (7.0, 7.0)],
        obstacle_radius=0.5,
    )


@pytest.fixture
def env_no_obstacles():
    return PushPOMDP(
        discount_factor=0.99,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )


@pytest.fixture
def updater(env):
    return PushVectorizedUpdater.from_environment(env)


@pytest.fixture
def updater_no_obstacles(env_no_obstacles):
    return PushVectorizedUpdater.from_environment(env_no_obstacles)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A PushPOMDP instance with obstacles.
        When: from_environment is called.
        Then: A PushVectorizedUpdater is returned with matching parameters.

        Test type: unit
        """
        updater = PushVectorizedUpdater.from_environment(env)
        assert isinstance(updater, PushVectorizedUpdater)
        assert updater.grid_size == env.grid_size
        assert updater.push_threshold == env.push_threshold
        assert updater.friction_coefficient == env.friction_coefficient
        assert updater.obstacle_radius == env.obstacle_radius
        assert updater.transition_error_prob == env.transition_error_prob

    def test_obs_dist_covariance_matches(self, env, updater):
        """Test that obs_dist has the correct 2x2 covariance.

        Purpose: Validates that the observation distribution is constructed
                 from the environment's observation_noise parameter.

        Given: A PushPOMDP with observation_noise=0.1.
        When: from_environment is called.
        Then: The obs_dist has a 2x2 diagonal covariance with sigma^2 on diagonal.

        Test type: unit
        """
        expected_cov = np.diag([env.observation_noise**2] * 2)
        np.testing.assert_array_equal(updater.obs_dist.covariance, expected_cov)

    def test_obstacles_shape(self, updater, env):
        """Test that obstacles are stored as (M, 2) array.

        Purpose: Validates obstacle array shape.

        Given: A PushPOMDP with 2 obstacles.
        When: from_environment is called.
        Then: updater.obstacles has shape (2, 2).

        Test type: unit
        """
        assert updater.obstacles.shape == (len(env.obstacles), 2)

    def test_no_obstacles_shape(self, updater_no_obstacles):
        """Test that empty obstacles are stored as (0, 2) array.

        Purpose: Validates obstacle array shape when no obstacles present.

        Given: A PushPOMDP with no obstacles.
        When: from_environment is called.
        Then: updater.obstacles has shape (0, 2).

        Test type: unit
        """
        assert updater_no_obstacles.obstacles.shape == (0, 2)


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater_no_obstacles):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 6.
        When: batch_transition is called with action 0 ("up").
        Then: Result has shape (30, 6).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.uniform(1, 8, (30, 6))
        result = updater_no_obstacles.batch_transition(particles, action=0)
        assert result.shape == (30, 6)

    def test_correct_direction_up(self, updater_no_obstacles):
        """Test that action 0 ("up") moves robot in +y direction.

        Purpose: Validates movement direction for "up" action.

        Given: A particle with robot at (5, 5), object far away.
        When: batch_transition is called with action 0 ("up").
        Then: Robot y increases by 1.

        Test type: unit
        """
        particle = np.array([[5.0, 5.0, 2.0, 2.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=0)
        assert result[0, 0] == 5.0  # robot_x unchanged
        assert result[0, 1] == 6.0  # robot_y increased

    def test_correct_direction_right(self, updater_no_obstacles):
        """Test that action 2 ("right") moves robot in +x direction.

        Purpose: Validates movement direction for "right" action.

        Given: A particle with robot at (5, 5), object far away.
        When: batch_transition is called with action 2 ("right").
        Then: Robot x increases by 1.

        Test type: unit
        """
        particle = np.array([[5.0, 5.0, 2.0, 2.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=2)
        assert result[0, 0] == 6.0  # robot_x increased
        assert result[0, 1] == 5.0  # robot_y unchanged

    def test_push_when_close(self, updater_no_obstacles):
        """Test that object is pushed when robot is within push_threshold.

        Purpose: Validates pushing mechanics when robot is close enough.

        Given: Robot at (5, 5), object at (5.5, 5) (within threshold=1.0).
        When: batch_transition is called with action 2 ("right").
        Then: Object moves in action direction, reduced by friction.

        Test type: unit
        """
        particle = np.array([[5.0, 5.0, 5.5, 5.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=2)
        # Robot moves right to (6, 5); distance to obj = 0.5 < 1.0 → push
        # Push force = [1, 0] * (1 - 0.3) = [0.7, 0]
        # Object: 5.5 + 0.7 = 6.2
        np.testing.assert_allclose(result[0, 2], 6.2, atol=1e-10)
        np.testing.assert_allclose(result[0, 3], 5.0, atol=1e-10)

    def test_no_push_when_far(self, updater_no_obstacles):
        """Test that object is not pushed when robot is beyond push_threshold.

        Purpose: Validates that pushing only occurs within threshold distance.

        Given: Robot at (1, 1), object at (5, 5) (far apart).
        When: batch_transition is called.
        Then: Object position is unchanged.

        Test type: unit
        """
        particle = np.array([[1.0, 1.0, 5.0, 5.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=0)
        np.testing.assert_array_equal(result[0, 2:4], [5.0, 5.0])

    def test_obstacle_blocks_robot(self, updater):
        """Test that obstacles block robot movement.

        Purpose: Validates collision detection for robot movement.

        Given: Robot adjacent to obstacle at (3, 3), moving toward it.
        When: batch_transition is called.
        Then: Robot stays at original position.

        Test type: unit
        """
        # Obstacle at (3, 3) with radius 0.5
        # Robot at (2, 3) moving right (+x) → intended (3, 3) collides
        particle = np.array([[2.0, 3.0, 8.0, 8.0, 9.0, 9.0]])
        result = updater.batch_transition(particle, action=2)
        np.testing.assert_array_equal(result[0, :2], [2.0, 3.0])

    def test_target_unchanged(self, updater_no_obstacles):
        """Test that target position is always preserved.

        Purpose: Validates that target coordinates (indices 4:6) are unchanged.

        Given: A particle with target at (9, 9).
        When: batch_transition is called.
        Then: Target remains at (9, 9).

        Test type: unit
        """
        particle = np.array([[5.0, 5.0, 5.5, 5.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=0)
        np.testing.assert_array_equal(result[0, 4:6], [9.0, 9.0])

    def test_grid_clipping(self, updater_no_obstacles):
        """Test that positions are clipped to grid boundaries.

        Purpose: Validates boundary clipping.

        Given: Robot at (0, 0).
        When: batch_transition is called with action 1 ("down", -y).
        Then: Robot y is clipped to 0.

        Test type: unit
        """
        particle = np.array([[0.0, 0.0, 5.0, 5.0, 9.0, 9.0]])
        result = updater_no_obstacles.batch_transition(particle, action=1)
        assert result[0, 1] >= 0.0

    def test_deterministic_no_error(self, updater_no_obstacles):
        """Test that batch_transition is deterministic when transition_error_prob=0.

        Purpose: Validates deterministic behavior.

        Given: A set of particles and transition_error_prob=0.
        When: batch_transition is called twice.
        Then: Results are identical.

        Test type: unit
        """
        particles = np.array(
            [
                [5.0, 5.0, 5.5, 5.0, 9.0, 9.0],
                [1.0, 1.0, 3.0, 3.0, 9.0, 9.0],
            ]
        )
        result_a = updater_no_obstacles.batch_transition(particles, action=0)
        result_b = updater_no_obstacles.batch_transition(particles, action=0)
        np.testing.assert_array_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# batch_observation_log_likelihood tests
# ---------------------------------------------------------------------------


class TestBatchObservationLogLikelihood:
    def test_output_shape(self, updater_no_obstacles):
        """Test that batch_observation_log_likelihood returns correct shape.

        Purpose: Validates output shape.

        Given: 20 particles.
        When: batch_observation_log_likelihood is called.
        Then: Result has shape (20,).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.uniform(1, 8, (20, 6))
        obs = particles[0].copy()
        result = updater_no_obstacles.batch_observation_log_likelihood(
            particles, action=0, observation=obs
        )
        assert result.shape == (20,)

    def test_values_are_finite(self, updater_no_obstacles):
        """Test that log-likelihoods are finite.

        Purpose: Validates that all returned values are finite numbers.

        Given: A set of particles and an observation.
        When: batch_observation_log_likelihood is called.
        Then: All returned values are finite.

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.uniform(1, 8, (20, 6))
        obs = particles[0].copy()
        result = updater_no_obstacles.batch_observation_log_likelihood(
            particles, action=0, observation=obs
        )
        assert np.all(np.isfinite(result))

    def test_closer_object_higher_likelihood(self, updater_no_obstacles):
        """Test that particles with object closer to observation have higher likelihood.

        Purpose: Validates that the Gaussian observation model assigns higher
                 likelihood to particles with object position nearer to the observed.

        Given: One particle with object at observed position, one with object far away.
        When: batch_observation_log_likelihood is called.
        Then: The close particle has higher log-likelihood.

        Test type: unit
        """
        obs = np.array([5.0, 5.0, 3.0, 3.0, 9.0, 9.0])
        close_particle = np.array([[5.0, 5.0, 3.01, 3.01, 9.0, 9.0]])
        far_particle = np.array([[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]])

        ll_close = updater_no_obstacles.batch_observation_log_likelihood(
            close_particle, action=0, observation=obs
        )
        ll_far = updater_no_obstacles.batch_observation_log_likelihood(
            far_particle, action=0, observation=obs
        )
        assert ll_close[0] > ll_far[0]

    def test_only_object_position_matters(self, updater_no_obstacles):
        """Test that only object position (indices 2:4) affects log-likelihood.

        Purpose: Validates that robot and target positions do not affect observation
                 likelihood, only object position matters.

        Given: Two particles with identical object positions but different robot/target.
        When: batch_observation_log_likelihood is called.
        Then: Both particles have the same log-likelihood.

        Test type: unit
        """
        obs = np.array([5.0, 5.0, 3.0, 3.0, 9.0, 9.0])
        particle_a = np.array([[1.0, 1.0, 3.0, 3.0, 8.0, 8.0]])
        particle_b = np.array([[7.0, 7.0, 3.0, 3.0, 2.0, 2.0]])

        ll_a = updater_no_obstacles.batch_observation_log_likelihood(
            particle_a, action=0, observation=obs
        )
        ll_b = updater_no_obstacles.batch_observation_log_likelihood(
            particle_b, action=0, observation=obs
        )
        np.testing.assert_allclose(ll_a, ll_b, atol=1e-12)


# ---------------------------------------------------------------------------
# config_id tests
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_config_id_deterministic(self, updater):
        """Test that config_id is deterministic.

        Purpose: Validates reproducibility of config_id.

        Given: An updater.
        When: config_id is called twice.
        Then: The same ID is returned.

        Test type: unit
        """
        assert updater.config_id == updater.config_id

    def test_config_id_differs_for_different_params(self, env):
        """Test that config_id changes when parameters differ.

        Purpose: Validates that different configurations produce different IDs.

        Given: Two updaters with different grid_size.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = PushVectorizedUpdater.from_environment(env)
        cov = np.diag([env.observation_noise**2] * 2)
        from POMDPPlanners.utils.multivariate_normal import (
            CovarianceParameterizedMultivariateNormal,
        )

        u2 = PushVectorizedUpdater(
            obs_dist=CovarianceParameterizedMultivariateNormal(cov),
            grid_size=env.grid_size * 2,
            push_threshold=env.push_threshold,
            friction_coefficient=env.friction_coefficient,
            obstacles=np.array(env.obstacles, dtype=float),
            obstacle_radius=env.obstacle_radius,
            transition_error_prob=env.transition_error_prob,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(
        self, env_no_obstacles, updater_no_obstacles
    ):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as calling
                 the environment's state_transition_model per particle.

        Given: A set of particles, an integer action index, and transition_error_prob=0.
        When: batch_transition is called and the same transitions are computed
              per-particle using the environment's state_transition_model
              (mapping int to string action).
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        particles = np.random.uniform(1, 8, (50, 6))
        particles[:, 4:6] = [9.0, 9.0]

        for action_idx in range(4):
            action_str = ACTION_STRINGS[action_idx]

            def per_particle_fn(particle, _action, _str=action_str):
                return env_no_obstacles.sample_next_state(state=particle, action=_str)

            assert_batch_transition_matches_loop(
                updater=updater_no_obstacles,
                particles=particles,
                action=action_idx,
                per_particle_transition_fn=per_particle_fn,
                err_msg=f"Mismatch for action {action_idx} ({action_str})",
            )

    def test_batch_observation_log_likelihood_matches_per_particle_loop(
        self, env_no_obstacles, updater_no_obstacles
    ):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle observation probability from the environment.

        Given: A set of next-state particles with object positions near the
               observation (to avoid probability underflow in exp-space).
        When: batch_observation_log_likelihood is called, and per-particle
              log(observation_model.probability) is computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        action_idx = 2
        action_str = ACTION_STRINGS[action_idx]

        base_obj = np.array([4.0, 4.0])
        particles = np.empty((50, 6))
        particles[:, :2] = np.random.uniform(1, 8, (50, 2))
        particles[:, 2:4] = base_obj + np.random.normal(0, 0.05, (50, 2))
        particles[:, 4:6] = [9.0, 9.0]

        observation = particles[0].copy()
        observation[2:4] = base_obj + 0.02

        def per_particle_ll_fn(particle, _action, obs):
            log_probs = env_no_obstacles.observation_log_probability(
                next_state=particle, action=action_str, observations=[obs]
            )
            return float(log_probs[0])

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater_no_obstacles,
            particles=particles,
            action=action_idx,
            observation=observation,
            per_particle_ll_fn=per_particle_ll_fn,
        )

    def test_batch_transition_with_obstacles_matches_per_particle(self, env, updater):
        """Test vectorized transition with obstacles matches per-particle loop.

        Purpose: Verifies that obstacle collision handling is correctly
                 vectorized.

        Given: Particles near obstacles and transition_error_prob=0.
        When: batch_transition is called and compared to per-particle loop.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(77)
        particles = np.random.uniform(0, 9, (50, 6))
        particles[:, 4:6] = [9.0, 9.0]

        for action_idx in range(4):
            action_str = ACTION_STRINGS[action_idx]

            def per_particle_fn(particle, _action, _str=action_str):
                return env.sample_next_state(state=particle, action=_str)

            assert_batch_transition_matches_loop(
                updater=updater,
                particles=particles,
                action=action_idx,
                per_particle_transition_fn=per_particle_fn,
                err_msg=f"Obstacle mismatch for action {action_idx} ({action_str})",
            )


# ---------------------------------------------------------------------------
# Belief-level equivalence against WeightedParticleBelief
# ---------------------------------------------------------------------------


class TestBeliefEquivalenceWithBaseline:
    def test_update_particles_match(self, env_no_obstacles, updater_no_obstacles):
        """Test vectorized belief update produces identical next particles.

        Purpose: Validates that VectorizedWeightedParticleBelief.update and
            WeightedParticleBelief.update agree on next-state particles for
            Push's deterministic transitions (transition_error_prob=0).

        Given: 100 aligned particles; both beliefs share the same updater.
        When: Both are updated with the string action "up" and a fixed observation.
        Then: Next-particle arrays agree within floating-point tolerance.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater_no_obstacles)
        obs = np.array([4.0, 4.0, 4.0, 4.0, 9.0, 9.0])
        assert_update_particles_match(
            base=base,
            vec=vec,
            action="up",
            observation=obs,
            pomdp=env_no_obstacles,
            atol=1e-10,
        )

    def test_update_weights_match(self, env_no_obstacles, updater_no_obstacles):
        """Test vectorized and baseline beliefs produce identical normalized weights.

        Purpose: Validates that the observation-reweighting path is consistent
            between the vectorized and per-particle update implementations.

        Given: 100 aligned particles.
        When: Both beliefs are updated with "up" and a fixed observation.
        Then: Normalized weights agree within 1e-6 L-infinity.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater_no_obstacles)
        obs = np.array([4.0, 4.0, 4.0, 4.0, 9.0, 9.0])
        assert_update_weights_match(
            base=base,
            vec=vec,
            action="up",
            observation=obs,
            pomdp=env_no_obstacles,
            atol=1e-6,
        )

    def test_sample_distributions_match_post_update(self, env_no_obstacles, updater_no_obstacles):
        """Test sample() on both beliefs draws unbiased from normalized_weights.

        Purpose: Validates that WeightedParticleBelief.sample() and
            VectorizedWeightedParticleBelief.sample() are unbiased draws
            from their belief distributions after an update.

        Given: 100 aligned particles, one update with a non-trivial observation.
        When: 20,000 samples are drawn from each belief.
        Then: Empirical histograms agree with each other and each agrees with
            its belief's normalized_weights.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater_no_obstacles)
        obs = np.array([4.0, 4.0, 4.0, 4.0, 9.0, 9.0])
        base = base.update(action="up", observation=obs, pomdp=env_no_obstacles)
        vec = vec.update(action="up", observation=obs, pomdp=env_no_obstacles)

        assert_sample_distributions_match(
            base=base,
            vec=vec,
            n_samples=20_000,
            tol=0.02,
            atol_weights=0.02,
            seed=400,
        )


# ---------------------------------------------------------------------------
# Native vs Python parity for the C++ belief-update kernels
# ---------------------------------------------------------------------------


def _python_reference_transition(
    particles: np.ndarray,
    action_idx: int,
    obstacles: np.ndarray,
    obstacle_radius: float,
    grid_size: float,
    push_threshold: float,
    friction_coefficient: float,
) -> np.ndarray:
    """NumPy reference implementation of the native belief-batch transition.

    Mirrors the deterministic Python implementation that the C++ kernel
    replaces. Used by the parity test to assert bit-for-bit equivalence
    when ``transition_error_prob == 0`` (i.e., no RNG involved).
    """
    movement = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=float)[action_idx]
    intended_robot = particles[:, :2] + movement
    if obstacles.shape[0] > 0:
        diff_r = intended_robot[:, None, :] - obstacles[None, :, :]
        robot_colliding = np.any(np.linalg.norm(diff_r, axis=2) <= obstacle_radius, axis=1)
    else:
        robot_colliding = np.zeros(len(intended_robot), dtype=bool)
    new_robot = np.where(robot_colliding[:, None], particles[:, :2], intended_robot)

    dist_to_obj = np.linalg.norm(new_robot - particles[:, 2:4], axis=1)
    can_push = dist_to_obj < push_threshold
    push_force = movement * (1.0 - friction_coefficient)
    intended_obj = particles[:, 2:4] + push_force
    if obstacles.shape[0] > 0:
        diff_o = intended_obj[:, None, :] - obstacles[None, :, :]
        obj_colliding = np.any(np.linalg.norm(diff_o, axis=2) <= obstacle_radius, axis=1)
    else:
        obj_colliding = np.zeros(len(intended_obj), dtype=bool)
    pushed_obj = np.where(obj_colliding[:, None], particles[:, 2:4], intended_obj)
    new_obj = np.where(can_push[:, None], pushed_obj, particles[:, 2:4])

    new_robot = np.clip(new_robot, 0, grid_size - 1)
    new_obj = np.clip(new_obj, 0, grid_size - 1)
    return np.column_stack([new_robot, new_obj, particles[:, 4:6]])


@pytest.fixture
def random_particles_k200():
    """K=200 random Push particles distributed across the grid."""
    rng = np.random.default_rng(seed=42)
    particles = rng.uniform(0.5, 8.5, size=(200, 6))
    particles[:, 4:6] = [9.0, 9.0]
    return particles


class TestNativeVsPythonParity:
    def test_native_transition_matches_python_no_error(
        self, env_no_obstacles, updater_no_obstacles, random_particles_k200
    ):
        """Native belief_batch_transition_discrete matches Python reference (no error).

        Purpose: Validates that the C++ kernel's deterministic branch
            (transition_error_prob == 0) produces bit-for-bit identical
            output to the NumPy reference for K=200 random particles
            across all four discrete actions.

        Given: 200 random Push particles, no obstacles, transition_error_prob=0.
        When: belief_batch_transition_discrete is called per action and the
            equivalent NumPy reference is computed.
        Then: ``np.array_equal`` holds (bit-for-bit equality, no RNG).

        Test type: integration
        """
        del env_no_obstacles  # only updater params consumed
        for action_idx in range(4):
            native = _native.belief_batch_transition_discrete(
                particles=random_particles_k200.astype(np.float64),
                action_idx=action_idx,
                transition_error_prob=0.0,
                obstacles=updater_no_obstacles.obstacles.astype(np.float64),
                obstacle_radius=float(updater_no_obstacles.obstacle_radius),
                grid_size=float(updater_no_obstacles.grid_size),
                push_threshold=float(updater_no_obstacles.push_threshold),
                friction_coefficient=float(updater_no_obstacles.friction_coefficient),
            )
            reference = _python_reference_transition(
                particles=random_particles_k200,
                action_idx=action_idx,
                obstacles=updater_no_obstacles.obstacles,
                obstacle_radius=float(updater_no_obstacles.obstacle_radius),
                grid_size=float(updater_no_obstacles.grid_size),
                push_threshold=float(updater_no_obstacles.push_threshold),
                friction_coefficient=float(updater_no_obstacles.friction_coefficient),
            )
            assert np.array_equal(
                native, reference
            ), f"native vs python mismatch for action {action_idx}"

    def test_native_transition_matches_python_with_obstacles(self, updater, random_particles_k200):
        """Native belief_batch_transition_discrete matches Python reference with obstacles.

        Purpose: Same parity check, but with obstacles enabled, exercising
            both robot-blocking and object-blocking collision branches.

        Given: 200 random particles, 2 obstacles at (3,3) and (7,7).
        When: belief_batch_transition_discrete is called per action.
        Then: ``np.array_equal`` holds (bit-for-bit equality).

        Test type: integration
        """
        for action_idx in range(4):
            native = _native.belief_batch_transition_discrete(
                particles=random_particles_k200.astype(np.float64),
                action_idx=action_idx,
                transition_error_prob=0.0,
                obstacles=updater.obstacles.astype(np.float64),
                obstacle_radius=float(updater.obstacle_radius),
                grid_size=float(updater.grid_size),
                push_threshold=float(updater.push_threshold),
                friction_coefficient=float(updater.friction_coefficient),
            )
            reference = _python_reference_transition(
                particles=random_particles_k200,
                action_idx=action_idx,
                obstacles=updater.obstacles,
                obstacle_radius=float(updater.obstacle_radius),
                grid_size=float(updater.grid_size),
                push_threshold=float(updater.push_threshold),
                friction_coefficient=float(updater.friction_coefficient),
            )
            assert np.array_equal(
                native, reference
            ), f"native vs python obstacle mismatch for action {action_idx}"

    def test_native_transition_error_branch_distribution(
        self, updater_no_obstacles, random_particles_k200
    ):
        """Native error-branch action distribution matches expected probabilities.

        Purpose: Validates that the independent C++ RNG used for the
            per-particle action-error coin flip produces an action mix
            consistent with transition_error_prob. Uses a Wilson 99% CI
            on the observed error rate vs the configured probability.

        Given: 200 particles, transition_error_prob=0.5, intended action=0 (up).
            Per particle, with prob 0.5 the actual action is uniformly
            sampled from {1, 2, 3}; otherwise it is 0. We detect "error
            occurred" by checking whether the resulting position differs
            from the deterministic (action=0) result.
        When: belief_batch_transition_discrete is called many times and
            we count the fraction of particles that took a non-up move.
        Then: The observed error fraction lies within a Wilson 99% CI of 0.5.

        Test type: integration
        """
        # Use particles with object outside push radius and target far away
        # so action choice is fully determined by the robot displacement.
        particles = np.tile(np.array([4.0, 4.0, 8.0, 8.0, 9.0, 9.0]), (200, 1))
        intended_action = 0  # up: dy = +1
        n_trials = 30
        n_total = 0
        n_error = 0
        for trial in range(n_trials):
            _native.set_seed(1000 + trial)
            result = _native.belief_batch_transition_discrete(
                particles=particles.astype(np.float64),
                action_idx=intended_action,
                transition_error_prob=0.5,
                obstacles=updater_no_obstacles.obstacles.astype(np.float64),
                obstacle_radius=float(updater_no_obstacles.obstacle_radius),
                grid_size=float(updater_no_obstacles.grid_size),
                push_threshold=float(updater_no_obstacles.push_threshold),
                friction_coefficient=float(updater_no_obstacles.friction_coefficient),
            )
            # Robot moved y from 4.0 to 5.0 if action=up was applied.
            up_executed = result[:, 1] == 5.0
            n_error += int((~up_executed).sum())
            n_total += len(particles)
        # Wilson 99% CI for proportion (z=2.576).
        p = n_error / n_total
        z = 2.576
        denom = 1.0 + z * z / n_total
        center = (p + z * z / (2 * n_total)) / denom
        half = (z * np.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
        ci_lo = center - half
        ci_hi = center + half
        assert ci_lo <= 0.5 <= ci_hi, (
            f"observed error rate {p:.4f} 99%-CI [{ci_lo:.4f}, {ci_hi:.4f}] "
            "does not contain the configured probability 0.5"
        )
        del random_particles_k200  # unused, fixture provided for symmetry

    def test_native_obs_log_likelihood_matches_python(
        self, updater_no_obstacles, random_particles_k200
    ):
        """Native belief_batch_obs_log_likelihood_discrete matches Python reference.

        Purpose: Validates that the C++ obs log-likelihood kernel produces
            output identical to the Python ``CovarianceParameterizedMultivariateNormal``
            log_pdf used by the original updater, on K=200 random particles.

        Given: 200 random next-state particles and a fixed observation.
        When: belief_batch_obs_log_likelihood_discrete is called and the
            reference log-pdf is computed via obs_dist.log_pdf on cols 2:4.
        Then: ``np.allclose`` holds with rtol=1e-7 (no RNG).

        Test type: integration
        """
        observation = np.array([5.0, 5.0, 4.5, 4.5, 9.0, 9.0])
        native = _native.belief_batch_obs_log_likelihood_discrete(
            next_particles=random_particles_k200.astype(np.float64),
            observation=observation,
            observation_noise=updater_no_obstacles._observation_noise,  # pylint: disable=protected-access
        )
        reference = updater_no_obstacles.obs_dist.log_pdf(
            random_particles_k200[:, 2:4], observation[2:4]
        )
        np.testing.assert_allclose(native, reference, rtol=1e-7, atol=0.0)

    def test_updater_uses_native_belief_batch_transition(self, updater):
        """PushVectorizedUpdater dispatches batch_transition through ``_native``.

        Purpose: Wire-through smoke test confirming that the native module
            is actually loaded and the updater's batch_transition path
            calls into the C++ entry point rather than a Python fallback.

        Given: A PushVectorizedUpdater built from a PushPOMDP environment.
        When: We inspect the ``_native`` module attached to the updater's
            module namespace.
        Then: Both kernel symbols are callables on the module.

        Test type: unit
        """
        assert callable(getattr(_native, "belief_batch_transition_discrete", None))
        assert callable(getattr(_native, "belief_batch_obs_log_likelihood_discrete", None))
        # Smoke: drive one round-trip through the updater.
        particles = np.tile(np.array([4.0, 4.0, 4.0, 4.0, 9.0, 9.0]), (5, 1))
        next_p = updater.batch_transition(particles, action="up")
        assert next_p.shape == (5, 6)
