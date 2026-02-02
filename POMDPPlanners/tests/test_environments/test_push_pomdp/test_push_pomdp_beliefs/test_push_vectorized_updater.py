"""Tests for PushVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods for the Push POMDP, including an equivalence test
that verifies the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs import (
    PushVectorizedUpdater,
)


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
              (mapping int→string action).
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 50
        particles = np.random.uniform(1, 8, (n, 6))
        # Fix target position columns to be consistent
        particles[:, 4:6] = [9.0, 9.0]

        for action_idx in range(4):
            vectorized_result = updater_no_obstacles.batch_transition(particles, action=action_idx)

            per_particle_result = np.empty_like(particles)
            action_str = ACTION_STRINGS[action_idx]
            for i in range(n):
                next_state = env_no_obstacles.state_transition_model(
                    state=particles[i], action=action_str
                ).sample()[0]
                per_particle_result[i] = next_state

            np.testing.assert_allclose(
                vectorized_result,
                per_particle_result,
                atol=1e-10,
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
        n = 50
        action_idx = 2
        action_str = ACTION_STRINGS[action_idx]

        # Keep object positions close to avoid underflow: the env computes
        # exp(…) then we take log, which fails when exp underflows to 0.
        base_obj = np.array([4.0, 4.0])
        particles = np.empty((n, 6))
        particles[:, :2] = np.random.uniform(1, 8, (n, 2))
        particles[:, 2:4] = base_obj + np.random.normal(0, 0.05, (n, 2))
        particles[:, 4:6] = [9.0, 9.0]

        observation = particles[0].copy()
        observation[2:4] = base_obj + 0.02

        vectorized_ll = updater_no_obstacles.batch_observation_log_likelihood(
            particles, action=action_idx, observation=observation
        )

        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = env_no_obstacles.observation_model(
                next_state=particles[i], action=action_str
            )
            prob = obs_model.probability([observation])[0]
            per_particle_ll[i] = np.log(prob)

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

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
        n = 50
        particles = np.random.uniform(0, 9, (n, 6))
        particles[:, 4:6] = [9.0, 9.0]

        for action_idx in range(4):
            vectorized_result = updater.batch_transition(particles, action=action_idx)

            per_particle_result = np.empty_like(particles)
            action_str = ACTION_STRINGS[action_idx]
            for i in range(n):
                next_state = env.state_transition_model(
                    state=particles[i], action=action_str
                ).sample()[0]
                per_particle_result[i] = next_state

            np.testing.assert_allclose(
                vectorized_result,
                per_particle_result,
                atol=1e-10,
                err_msg=f"Obstacle mismatch for action {action_idx} ({action_str})",
            )
