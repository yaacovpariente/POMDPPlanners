"""Tests for the Continuous LaserTag vectorized belief updater.

Tests cover batch transition, batch observation log-likelihood,
from_environment construction, and config_id generation.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs.continuous_laser_tag_vectorized_updater import (
    ContinuousLaserTagVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_update_particles_match_per_particle_seeded,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
)


def _make_aligned_beliefs(updater, n_particles=20):
    """Create baseline + vectorized beliefs with identical initial particles."""
    np.random.seed(42)
    particles_array = np.column_stack(
        [
            np.random.rand(n_particles) * 10,
            np.random.rand(n_particles) * 6,
            np.random.rand(n_particles) * 10,
            np.random.rand(n_particles) * 6,
            np.zeros(n_particles),
        ]
    )
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


@pytest.fixture
def env():
    return ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[])


@pytest.fixture
def env_discrete():
    return ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95, walls=[])


@pytest.fixture
def updater(env):
    return ContinuousLaserTagVectorizedUpdater.from_environment(env)


@pytest.fixture
def updater_discrete(env_discrete):
    return ContinuousLaserTagVectorizedUpdater.from_environment(env_discrete)


class TestFromEnvironment:
    """Tests for the from_environment classmethod."""

    def test_creates_instance(self, updater):
        """Test that from_environment creates an updater.

        Purpose: Validates factory construction.

        Given: A ContinuousLaserTagPOMDP instance.
        When: from_environment is called.
        Then: Returns a ContinuousLaserTagVectorizedUpdater.

        Test type: unit
        """
        assert isinstance(updater, ContinuousLaserTagVectorizedUpdater)

    def test_creates_instance_discrete(self, updater_discrete):
        """Test from_environment with discrete action variant.

        Purpose: Validates factory with discrete action environment.

        Given: A ContinuousLaserTagPOMDPDiscreteActions instance.
        When: from_environment is called.
        Then: The updater has action_to_vector mapping.

        Test type: unit
        """
        assert isinstance(updater_discrete, ContinuousLaserTagVectorizedUpdater)
        assert updater_discrete._action_to_vector is not None


class TestBatchTransition:
    """Tests for the batch_transition method."""

    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates transition output shape.

        Given: N particles of shape (N, 5).
        When: batch_transition is called.
        Then: Output shape is (N, 5).

        Test type: unit
        """
        np.random.seed(42)
        n = 50
        particles = np.column_stack(
            [
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.zeros(n),
            ]
        )
        action = np.array([1.0, 0.0, 0.0])
        result = updater.batch_transition(particles, action)
        assert result.shape == (n, 5)

    def test_terminal_particles_unchanged(self, updater):
        """Test that terminal particles remain unchanged.

        Purpose: Validates terminal particle handling.

        Given: All-terminal particles.
        When: batch_transition is called.
        Then: All particles remain at terminal state.

        Test type: unit
        """
        np.random.seed(42)
        n = 10
        particles = np.column_stack(
            [
                np.full(n, 5.0),
                np.full(n, 3.0),
                np.full(n, 8.0),
                np.full(n, 5.0),
                np.ones(n),
            ]
        )
        action = np.array([1.0, 0.0, 0.0])
        result = updater.batch_transition(particles, action)
        np.testing.assert_array_equal(result[:, 4], 1.0)

    def test_tag_creates_terminal(self, updater):
        """Test that tag action at close range creates terminal particles.

        Purpose: Validates tag success in batch transition.

        Given: Particles with robot at opponent position.
        When: Tag action is applied.
        Then: Some particles become terminal.

        Test type: unit
        """
        np.random.seed(42)
        n = 50
        particles = np.column_stack(
            [
                np.full(n, 5.0),
                np.full(n, 3.0),
                np.full(n, 5.0),
                np.full(n, 3.0),
                np.zeros(n),
            ]
        )
        action = np.array([0.0, 0.0, 1.0])
        result = updater.batch_transition(particles, action)
        terminal_count = np.sum(result[:, 4] == 1.0)
        assert terminal_count > 0

    def test_discrete_action_string(self, updater_discrete):
        """Test batch_transition with string action.

        Purpose: Validates discrete action string conversion.

        Given: Particles and a string action.
        When: batch_transition is called.
        Then: Returns valid output.

        Test type: unit
        """
        np.random.seed(42)
        n = 20
        particles = np.column_stack(
            [
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.zeros(n),
            ]
        )
        result = updater_discrete.batch_transition(particles, "right")
        assert result.shape == (n, 5)


class TestBatchObservationLogLikelihood:
    """Tests for the batch_observation_log_likelihood method."""

    def test_output_shape(self, updater):
        """Test that log-likelihood returns correct shape.

        Purpose: Validates log-likelihood output shape.

        Given: N particles and a non-terminal observation.
        When: batch_observation_log_likelihood is called.
        Then: Output shape is (N,).

        Test type: unit
        """
        np.random.seed(42)
        n = 50
        particles = np.column_stack(
            [
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.zeros(n),
            ]
        )
        obs = np.random.rand(8) * 5
        action = np.array([1.0, 0.0, 0.0])
        ll = updater.batch_observation_log_likelihood(particles, action, obs)
        assert ll.shape == (n,)

    def test_terminal_observation_handling(self, updater):
        """Test that terminal observation gives zero log-likelihood for terminal particles.

        Purpose: Validates terminal observation handling.

        Given: Mixed terminal and non-terminal particles with terminal observation.
        When: batch_observation_log_likelihood is called.
        Then: Terminal particles get 0.0, non-terminal get -inf.

        Test type: unit
        """
        n = 10
        particles = np.column_stack(
            [
                np.full(n, 5.0),
                np.full(n, 3.0),
                np.full(n, 8.0),
                np.full(n, 5.0),
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float),
            ]
        )
        terminal_obs = np.full(8, -1.0)
        action = np.array([1.0, 0.0, 0.0])
        ll = updater.batch_observation_log_likelihood(particles, action, terminal_obs)
        # Terminal particles should have 0.0 log-likelihood
        assert np.all(ll[5:] == 0.0)
        # Non-terminal should have -inf
        assert np.all(ll[:5] == -np.inf)

    def test_non_terminal_finite_values(self, updater):
        """Test that non-terminal particles get finite log-likelihoods.

        Purpose: Validates finite log-likelihood values.

        Given: All non-terminal particles and a valid observation.
        When: batch_observation_log_likelihood is called.
        Then: All log-likelihoods are finite.

        Test type: unit
        """
        np.random.seed(42)
        n = 20
        particles = np.column_stack(
            [
                np.full(n, 5.0),
                np.full(n, 3.0),
                np.full(n, 8.0),
                np.full(n, 5.0),
                np.zeros(n),
            ]
        )
        obs = np.random.rand(8) * 5
        action = np.array([1.0, 0.0, 0.0])
        ll = updater.batch_observation_log_likelihood(particles, action, obs)
        assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as
                 calling the environment's state_transition_model per particle.
                 Because the vectorized path batches robot noise then opponent
                 noise (different RNG order from the per-particle path which
                 interleaves them), we compare each particle individually with
                 its own seed.

        Given: A set of non-terminal continuous-space particles.
        When: For each particle, batch_transition is called on a single
              particle, and the per-particle state_transition_model is called
              with the same seed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 30
        particles = np.column_stack(
            [
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.zeros(n),
            ]
        )
        action = np.array([1.0, 0.5, 0.0])

        for i in range(n):
            single = particles[i : i + 1]

            np.random.seed(1000 + i)
            vec_result = updater.batch_transition(single, action)

            np.random.seed(1000 + i)
            scalar_result = env.state_transition_model(state=particles[i], action=action).sample()[
                0
            ]

            np.testing.assert_allclose(
                vec_result[0],
                scalar_result,
                atol=1e-10,
                err_msg=f"Mismatch for particle {i}",
            )

    def test_batch_obs_log_likelihood_matches_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle log(observation_model.probability) from the
                 environment.

        Given: A set of non-terminal particles and a valid observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log-probabilities are computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        particles = np.column_stack(
            [
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.random.rand(n) * 10,
                np.random.rand(n) * 6,
                np.zeros(n),
            ]
        )
        obs = np.random.rand(8) * 5
        action = np.array([1.0, 0.0, 0.0])

        def per_particle_ll_fn(particle, act, observation):
            obs_model = env.observation_model(next_state=particle, action=act)
            prob = obs_model.probability([observation])[0]
            if prob > 0:
                return np.log(prob)
            return -np.inf

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=action,
            observation=obs,
            per_particle_ll_fn=per_particle_ll_fn,
        )


class TestConfigId:
    """Tests for the config_id property."""

    def test_config_id_is_string(self, updater):
        """Test that config_id returns a string.

        Purpose: Validates config_id type.

        Given: An updater instance.
        When: config_id is accessed.
        Then: Returns a non-empty string.

        Test type: unit
        """
        cid = updater.config_id
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_same_config_same_id(self, env):
        """Test that identical configurations produce the same id.

        Purpose: Validates deterministic config_id.

        Given: Two updaters from the same environment.
        When: config_id is compared.
        Then: Both IDs are equal.

        Test type: unit
        """
        u1 = ContinuousLaserTagVectorizedUpdater.from_environment(env)
        u2 = ContinuousLaserTagVectorizedUpdater.from_environment(env)
        assert u1.config_id == u2.config_id


class TestBeliefEquivalenceWithBaseline:
    def test_update_particles_match_per_particle_seeded(self, env, updater):
        """Test vectorized and baseline beliefs agree on particles under per-particle seeding.

        Purpose: Validates that VectorizedWeightedParticleBelief.update and
            WeightedParticleBelief.update produce identical next particles
            when each particle's update is seeded independently. The
            vectorized path samples robot and opponent continuous Gaussian
            noise in bulk, which differs in RNG order from the baseline
            per-particle loop. Per-particle seeding restores bit-for-bit
            equivalence on the particle array (mirroring the updater-level
            workaround in this file's TestEquivalenceWithPerParticleLoop).

            Weights are not compared here: for out-of-distribution
            observations (like the random one used here) every particle's
            Gaussian PDF underflows to zero, which the baseline floors via
            ``log(eps + prob)`` and the vectorized preserves exactly via
            ``log_pdf``. The resulting softmax distributions differ even
            when particles match.

        Given: 20 aligned continuous-space particles.
        When: Each particle is updated individually with a shared
            per-particle seed (base_seed + i) on both paths.
        Then: Aggregate next-particle arrays agree within floating-point
            tolerance.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater)
        obs = np.random.rand(8) * 5
        assert_update_particles_match_per_particle_seeded(
            base=base,
            vec=vec,
            action=np.array([1.0, 0.5, 0.0]),
            observation=obs,
            pomdp=env,
            atol=1e-10,
            base_seed=1000,
        )
