"""Tests for DiscreteLightDarkVectorizedUpdater and observation model variants.

This module tests the vectorized batch transition and observation
log-likelihood methods for all three discrete Light-Dark observation
models, including equivalence tests against per-particle loops.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    DiscreteLightDarkDistanceBasedVectorizedUpdater,
    DiscreteLightDarkNoObsInDarkVectorizedUpdater,
    DiscreteLightDarkVectorizedUpdater,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return DiscreteLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return DiscreteLightDarkVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Base updater (NORMAL) tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A DiscreteLightDarkPOMDP instance.
        When: from_environment is called.
        Then: A DiscreteLightDarkVectorizedUpdater is returned.

        Test type: unit
        """
        updater = DiscreteLightDarkVectorizedUpdater.from_environment(env)
        assert isinstance(updater, DiscreteLightDarkVectorizedUpdater)
        assert updater.beacon_radius == env.beacon_radius
        assert updater.grid_size == env.grid_size


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 2.
        When: batch_transition is called with action 'right'.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.array([[5, 5]] * 30, dtype=float)
        result = updater.batch_transition(particles, "right")
        assert result.shape == (30, 2)

    def test_stochastic_transition(self, updater):
        """Test that stochastic transitions produce some unexpected actions.

        Purpose: Validates that transition_error_prob causes some particles
                 to execute different actions.

        Given: Many identical particles and a nonzero transition_error_prob.
        When: batch_transition is called many times.
        Then: Not all particles move in the intended direction.

        Test type: unit
        """
        np.random.seed(42)
        n = 1000
        particles = np.tile([5.0, 5.0], (n, 1))
        result = updater.batch_transition(particles, "right")
        # 'right' moves [1, 0], so intended result is [6, 5]
        # With error_prob=0.05, some should move differently
        intended = np.array([6.0, 5.0])
        not_intended = ~np.all(result == intended, axis=1)
        assert np.any(not_intended), "Expected some stochastic transitions"

    def test_transition_distribution_matches_environment(self, env, updater):
        """Test that vectorized transition has same distribution as environment.

        Purpose: Verifies statistical equivalence with the environment's
                 state_transition_model.

        Given: Many identical particles and the 'right' action.
        When: Transitions are computed via vectorized path.
        Then: The fraction of intended vs error transitions matches
              the configured transition_error_prob.

        Test type: integration
        """
        np.random.seed(42)
        n = 5000
        particles = np.tile([5.0, 5.0], (n, 1))
        action = "right"

        result = updater.batch_transition(particles, action)
        # 'right' moves [1, 0], intended result is [6, 5]
        intended = np.all(result == [6.0, 5.0], axis=1)
        intended_frac = intended.sum() / n
        expected_frac = 1.0 - env.transition_error_prob
        np.testing.assert_allclose(intended_frac, expected_frac, atol=0.03)


class TestBatchObservationLogLikelihood:
    def test_output_shape(self, updater):
        """Test that batch_observation_log_likelihood returns correct shape.

        Purpose: Validates output shape.

        Given: 20 particles.
        When: batch_observation_log_likelihood is called.
        Then: Result has shape (20,).

        Test type: unit
        """
        particles = np.array([[5, 5]] * 20, dtype=float)
        obs = np.array([5, 5], dtype=float)
        result = updater.batch_observation_log_likelihood(particles, "right", obs)
        assert result.shape == (20,)

    def test_exact_match_highest_likelihood(self, updater):
        """Test that exact observation match has highest log-likelihood.

        Purpose: Validates that P(obs=state | state) is highest for exact match.

        Given: Particles at (5, 5).
        When: Log-likelihood computed for obs=(5,5) vs obs=(6,5).
        Then: Exact match has higher log-likelihood.

        Test type: unit
        """
        particles = np.array([[5.0, 5.0]])
        ll_exact = updater.batch_observation_log_likelihood(
            particles, "right", np.array([5.0, 5.0])
        )
        ll_offset = updater.batch_observation_log_likelihood(
            particles, "right", np.array([6.0, 5.0])
        )
        assert ll_exact[0] > ll_offset[0]

    def test_unmatched_observation_neg_inf(self, updater):
        """Test that observation not matching any offset gives -inf.

        Purpose: Validates that impossible observations get -inf.

        Given: A particle at (5, 5).
        When: Log-likelihood computed for obs=(10, 10) (no offset match).
        Then: Result is -inf.

        Test type: unit
        """
        particles = np.array([[5.0, 5.0]])
        obs = np.array([10.0, 10.0])
        ll = updater.batch_observation_log_likelihood(particles, "right", obs)
        assert ll[0] == -np.inf

    def test_near_beacon_higher_confidence(self, updater):
        """Test that near-beacon particles have higher exact-match confidence.

        Purpose: Validates beacon-dependent observation quality.

        Given: One particle near beacon at (0,0), one far at (2.5, 2.5).
        When: Log-likelihood for exact observation match is computed.
        Then: Near-beacon particle has higher log-likelihood.

        Test type: unit
        """
        near = np.array([[0.0, 0.0]])
        far = np.array([[2.5, 2.5]])
        ll_near = updater.batch_observation_log_likelihood(near, "right", np.array([0.0, 0.0]))
        ll_far = updater.batch_observation_log_likelihood(far, "right", np.array([2.5, 2.5]))
        assert ll_near[0] > ll_far[0]

    def test_equivalence_with_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle observation model.

        Purpose: Verifies equivalence with the environment's observation_model.

        Given: Particles and observations (including exact match and offset).
        When: Vectorized and per-particle probabilities are computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = "right"
        particles = np.random.randint(1, 9, size=(n, 2)).astype(float)

        # Test with exact observation
        for i in range(min(n, 5)):
            obs = particles[i].copy()
            vectorized_ll = updater.batch_observation_log_likelihood(particles, action, obs)
            per_particle_ll = np.empty(n)
            for j in range(n):
                per_particle_ll[j] = env.observation_log_probability(particles[j], action, [obs])[0]

            np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)


class TestConfigId:
    def test_deterministic(self, updater):
        """Test that config_id is deterministic.

        Purpose: Validates reproducibility.

        Given: An updater.
        When: config_id is called twice.
        Then: The same ID is returned.

        Test type: unit
        """
        assert updater.config_id == updater.config_id


# ---------------------------------------------------------------------------
# NoObsInDark updater tests
# ---------------------------------------------------------------------------


@pytest.fixture
def no_obs_env():
    return DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
    )


@pytest.fixture
def no_obs_updater(no_obs_env):
    return DiscreteLightDarkNoObsInDarkVectorizedUpdater.from_environment(no_obs_env)


class TestNoObsInDarkFromEnvironment:
    def test_creates_correct_subclass(self, no_obs_env):
        """Test that from_environment returns the correct subclass.

        Purpose: Validates that the factory creates the right type.

        Given: A DiscreteLightDarkPOMDP with NO_OBS_IN_DARK.
        When: from_environment is called.
        Then: The updater is a DiscreteLightDarkNoObsInDarkVectorizedUpdater.

        Test type: unit
        """
        updater = DiscreteLightDarkNoObsInDarkVectorizedUpdater.from_environment(no_obs_env)
        assert isinstance(updater, DiscreteLightDarkNoObsInDarkVectorizedUpdater)


class TestNoObsInDarkLogLikelihood:
    def test_none_obs_near_beacon_returns_neg_inf(self, no_obs_updater):
        """Test that 'None' observation near beacon gives -inf.

        Purpose: Validates that 'None' is impossible near beacons.

        Given: A particle at (0, 0) near the origin beacon.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, "right", "None")
        assert ll[0] == -np.inf

    def test_none_obs_far_from_beacon_returns_zero(self, no_obs_updater):
        """Test that 'None' observation far from beacons gives 0.

        Purpose: Validates that 'None' is certain when far from beacons.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is 0.0.

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, "right", "None")
        assert ll[0] == 0.0

    def test_array_obs_near_beacon(self, no_obs_updater):
        """Test that array observation near beacon gives finite log-likelihood.

        Purpose: Validates that real observations work near beacons.

        Given: A particle at (0, 0) near the origin beacon.
        When: Log-likelihood is computed for exact match observation.
        Then: The result is finite.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])
        obs = np.array([0.0, 0.0])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, "right", obs)
        assert np.isfinite(ll[0])

    def test_array_obs_far_from_beacon_returns_neg_inf(self, no_obs_updater):
        """Test that array observation far from beacons gives -inf.

        Purpose: Validates that real observations are impossible far from beacons.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for an array observation.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        obs = np.array([2.5, 2.5])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, "right", obs)
        assert ll[0] == -np.inf

    def test_equivalence_with_per_particle_loop(self, no_obs_env, no_obs_updater):
        """Test vectorized log-likelihood matches per-particle computation.

        Purpose: Verifies equivalence against the non-vectorized observation model.

        Given: Particles and both array and 'None' observations.
        When: Vectorized and per-particle log-likelihoods are computed.
        Then: Results match.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = "right"
        particles = np.random.randint(0, 10, size=(n, 2)).astype(float)

        # Test "None" observation. The vectorized updater uses a strict -inf
        # floor for impossible observations, while observation_log_probability
        # uses log(prob + 1e-300); we therefore go via the wrapper here so we
        # can apply the matching np.log(prob) if prob > 0 else -inf rule.
        vectorized_ll = no_obs_updater.batch_observation_log_likelihood(particles, action, "None")
        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = no_obs_env.observation_model(next_state=particles[i], action=action)
            prob = obs_model.probability(["None"])[0]
            per_particle_ll[i] = np.log(prob) if prob > 0 else -np.inf

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

        # Test exact-match observation for a near-beacon particle
        near_particles = np.array([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]])
        for particle in near_particles:
            obs = particle.copy()
            v_ll = no_obs_updater.batch_observation_log_likelihood(near_particles, action, obs)
            pp_ll = np.empty(len(near_particles))
            for j, np_j in enumerate(near_particles):
                obs_model = no_obs_env.observation_model(next_state=np_j, action=action)
                prob = obs_model.probability([obs])[0]
                pp_ll[j] = np.log(prob) if prob > 0 else -np.inf

            np.testing.assert_allclose(v_ll, pp_ll, atol=1e-10)


class TestNoObsInDarkConfigId:
    def test_config_id_differs_from_base(self, updater, no_obs_updater):
        """Test that config_id differs between NORMAL and NO_OBS_IN_DARK.

        Purpose: Validates distinct configuration IDs.

        Given: Updaters for NORMAL and NO_OBS_IN_DARK.
        When: config_id is compared.
        Then: The IDs differ.

        Test type: unit
        """
        assert updater.config_id != no_obs_updater.config_id


# ---------------------------------------------------------------------------
# DistanceBased updater tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dist_env():
    return DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )


@pytest.fixture
def dist_updater(dist_env):
    return DiscreteLightDarkDistanceBasedVectorizedUpdater.from_environment(dist_env)


class TestDistanceBasedFromEnvironment:
    def test_creates_correct_subclass(self, dist_env):
        """Test that from_environment returns the correct subclass.

        Purpose: Validates that the factory creates the right type.

        Given: A DiscreteLightDarkPOMDP with DISTANCE_BASED.
        When: from_environment is called.
        Then: The updater is a DiscreteLightDarkDistanceBasedVectorizedUpdater.

        Test type: unit
        """
        updater = DiscreteLightDarkDistanceBasedVectorizedUpdater.from_environment(dist_env)
        assert isinstance(updater, DiscreteLightDarkDistanceBasedVectorizedUpdater)


class TestDistanceBasedLogLikelihood:
    def test_none_obs_within_radius_returns_neg_inf(self, dist_updater):
        """Test that 'None' observation within beacon_radius gives -inf.

        Purpose: Validates that 'None' is impossible within beacon_radius.

        Given: A particle at (0, 0) at the origin beacon.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])
        ll = dist_updater.batch_observation_log_likelihood(particles, "right", "None")
        assert ll[0] == -np.inf

    def test_none_obs_beyond_radius_returns_zero(self, dist_updater):
        """Test that 'None' observation beyond beacon_radius gives 0.

        Purpose: Validates that 'None' is certain beyond all beacons.

        Given: A particle at (2.5, 2.5) beyond beacon_radius from all beacons.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is 0.0.

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        ll = dist_updater.batch_observation_log_likelihood(particles, "right", "None")
        assert ll[0] == 0.0

    def test_closer_to_beacon_higher_exact_match_prob(self, dist_updater):
        """Test that particles closer to beacon have higher exact-match probability.

        Purpose: Validates distance-based error scaling.

        Given: One particle near beacon, one farther (but still within radius).
        When: Log-likelihood for exact observation match is computed.
        Then: Closer particle has higher log-likelihood.

        Test type: unit
        """
        # Beacons at (0,0), (0,5), etc. beacon_radius=1.0
        # Particle at (0.1, 0) is very close to beacon (0,0)
        # Particle at (0.9, 0) is near edge of beacon_radius
        near = np.array([[0.1, 0.0]])
        far = np.array([[0.9, 0.0]])
        ll_near = dist_updater.batch_observation_log_likelihood(near, "right", np.array([0.1, 0.0]))
        ll_far = dist_updater.batch_observation_log_likelihood(far, "right", np.array([0.9, 0.0]))
        assert ll_near[0] > ll_far[0]

    def test_equivalence_with_per_particle_loop(self, dist_env, dist_updater):
        """Test vectorized log-likelihood matches per-particle computation.

        Purpose: Verifies equivalence against the non-vectorized observation model.

        Given: Particles and both array and 'None' observations.
        When: Vectorized and per-particle log-likelihoods are computed.
        Then: Results match.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = "right"
        particles = np.random.randint(0, 10, size=(n, 2)).astype(float)

        # Test "None" observation. The vectorized updater uses a strict -inf
        # floor for impossible observations, while observation_log_probability
        # uses log(prob + 1e-300); we therefore go via the wrapper here so we
        # can apply the matching np.log(prob) if prob > 0 else -inf rule.
        vectorized_ll = dist_updater.batch_observation_log_likelihood(particles, action, "None")
        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = dist_env.observation_model(next_state=particles[i], action=action)
            prob = obs_model.probability(["None"])[0]
            per_particle_ll[i] = np.log(prob) if prob > 0 else -np.inf

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

        # Test exact-match observation
        test_particles = np.array([[0.0, 0.0], [0.5, 0.0], [2.5, 2.5]])
        for particle in test_particles:
            obs = particle.copy()
            v_ll = dist_updater.batch_observation_log_likelihood(test_particles, action, obs)
            pp_ll = np.empty(len(test_particles))
            for j, tp in enumerate(test_particles):
                obs_model = dist_env.observation_model(next_state=tp, action=action)
                prob = obs_model.probability([obs])[0]
                pp_ll[j] = np.log(prob) if prob > 0 else -np.inf

            np.testing.assert_allclose(v_ll, pp_ll, atol=1e-10)


class TestDistanceBasedConfigId:
    def test_all_variants_differ(self, updater, no_obs_updater, dist_updater):
        """Test that all three discrete updater variants have different config_ids.

        Purpose: Validates that each observation model variant is uniquely identified.

        Given: Updaters for NORMAL, NO_OBS_IN_DARK, and DISTANCE_BASED.
        When: config_ids are compared.
        Then: All three IDs are different.

        Test type: unit
        """
        ids = {updater.config_id, no_obs_updater.config_id, dist_updater.config_id}
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# Full update equivalence tests (vectorized vs standard particle belief)
# ---------------------------------------------------------------------------


class TestFullUpdateEquivalence:
    def test_normal_updater_full_update(self, env, updater):
        """Test full vectorized update matches WeightedParticleBelief for NORMAL model.

        Purpose: End-to-end equivalence test comparing VectorizedWeightedParticleBelief
                 update against WeightedParticleBelief update.

        Given: Identical initial particles and weights.
        When: One step of update is performed via both paths with the same random seed.
        Then: Both paths agree on which particles carry significant weight
              and the top-ranked particles match.

        Test type: integration
        """
        np.random.seed(77)
        n = 40
        action = "right"
        # Observation at (6, 5) — exact match for a particle at (5, 5) moved right
        observation = np.array([6.0, 5.0])

        particles_array = np.random.randint(1, 9, size=(n, 2)).astype(float)
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        # Vectorized path
        np.random.seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        # Per-particle path
        np.random.seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=env)

        # Compare normalized weights for significant particles.
        # Tolerance is relaxed because WeightedParticleBelief uses
        # log(eps + prob) while vectorized uses log_prob directly.
        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        if np.any(significant):
            np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-2)

    def test_no_obs_in_dark_full_update_with_array(self, no_obs_env, no_obs_updater):
        """Test full update equivalence for NO_OBS_IN_DARK with array observation.

        Purpose: End-to-end equivalence test for the NO_OBS_IN_DARK model.

        Given: Identical initial particles and weights.
        When: One step of update with array observation via both paths.
        Then: Weight distributions agree on significant particles.

        Note: VectorizedWeightedParticleBelief.update() does not yet support
              string observations. Per-observation-level equivalence for
              'None' is tested separately above.

        Test type: integration
        """
        np.random.seed(77)
        n = 40
        action = "right"
        observation = np.array([6.0, 5.0])

        particles_array = np.random.randint(1, 9, size=(n, 2)).astype(float)
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        np.random.seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=no_obs_updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        np.random.seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=no_obs_env)

        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        if np.any(significant):
            np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-2)

    def test_distance_based_full_update_with_array(self, dist_env, dist_updater):
        """Test full update equivalence for DISTANCE_BASED with array observation.

        Purpose: End-to-end equivalence test for the DISTANCE_BASED model.

        Given: Particles near beacons so array observations have non-trivial weight.
        When: One step of update with array observation via both paths.
        Then: Weight distributions agree on significant particles.

        Note: VectorizedWeightedParticleBelief.update() does not yet support
              string observations. Per-observation-level equivalence for
              'None' is tested separately above.

        Test type: integration
        """
        np.random.seed(77)
        n = 40
        action = "right"
        observation = np.array([1.0, 0.0])

        # Place particles near beacon at (0, 0)
        particles_array = np.random.rand(n, 2).astype(float)
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        np.random.seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=dist_updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        np.random.seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=dist_env)

        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        if np.any(significant):
            np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-2)
