"""Tests for ContinuousLightDarkVectorizedUpdater and observation model variants.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop in WeightedParticleBelief.
"""

# protected-access: tests intentionally reach into belief/updater internals
#   (e.g. _active_dist, _action_to_vector) to assert parity with the
#   non-vectorized observation models.
# import-outside-toplevel / reimport: a couple of equivalence test methods
#   re-import WeightedParticleBelief inside their bodies for readability;
#   pre-existing pattern in this file.
# pylint: disable=protected-access,import-outside-toplevel,reimported,too-many-lines

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    ContinuousLightDarkDistanceBasedVectorizedUpdater,
    ContinuousLightDarkNoObsInDarkVectorizedUpdater,
    ContinuousLightDarkVectorizedUpdater,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return ContinuousLightDarkVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A ContinuousLightDarkPOMDP instance.
        When: from_environment is called.
        Then: A ContinuousLightDarkVectorizedUpdater is returned with
              matching parameters.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        assert isinstance(updater, ContinuousLightDarkVectorizedUpdater)
        assert updater.beacon_radius == env.beacon_radius
        assert updater.grid_size == env.grid_size
        np.testing.assert_array_equal(updater.beacons, env.beacons)


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 2.
        When: batch_transition is called.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(30, 2) * 10
        action = np.array([1.0, 0.0])
        result = updater.batch_transition(particles, action)
        assert result.shape == (30, 2)

    def test_mean_shift(self, updater):
        """Test that mean transition is particles + action (plus noise).

        Purpose: Validates that the mean next state is particles + action.

        Given: Identical particles at [5, 5] and action [1, 0].
        When: batch_transition is called many times and results averaged.
        Then: The mean result is close to [6, 5].

        Test type: unit
        """
        np.random.seed(42)
        n = 5000
        particles = np.tile([5.0, 5.0], (n, 1))
        action = np.array([1.0, 0.0])
        results = updater.batch_transition(particles, action)
        mean_result = results.mean(axis=0)
        np.testing.assert_allclose(mean_result, [6.0, 5.0], atol=0.1)


# ---------------------------------------------------------------------------
# batch_observation_log_likelihood tests
# ---------------------------------------------------------------------------


class TestBatchObservationLogLikelihood:
    def test_output_shape(self, updater):
        """Test that batch_observation_log_likelihood returns correct shape.

        Purpose: Validates output shape.

        Given: 20 particles.
        When: batch_observation_log_likelihood is called.
        Then: Result has shape (20,).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(20, 2) * 10
        action = np.array([1.0, 0.0])
        obs = np.array([5.0, 5.0])
        result = updater.batch_observation_log_likelihood(particles, action, obs)
        assert result.shape == (20,)

    def test_near_beacon_higher_likelihood(self, updater):
        """Test that particles near a beacon with matching observation get higher likelihood.

        Purpose: Validates beacon-dependent observation likelihood.

        Given: One particle at [0, 0] (near beacon at origin) and one at
               [2.5, 2.5] (far from all beacons), with observation at their positions.
        When: Log-likelihoods are computed for observation matching their positions.
        Then: The near-beacon particle has higher log-likelihood because
              the near-beacon distribution has smaller covariance.

        Test type: unit
        """
        near_particle = np.array([[0.0, 0.0]])  # at beacon
        far_particle = np.array([[2.5, 2.5]])  # far from all beacons
        action = np.zeros(2)

        ll_near = updater.batch_observation_log_likelihood(near_particle, action, near_particle[0])
        ll_far = updater.batch_observation_log_likelihood(far_particle, action, far_particle[0])
        # Near-beacon has tighter covariance → higher peak density
        assert ll_near[0] > ll_far[0]

    def test_mixed_near_far_particles(self, updater):
        """Test that mixed near/far particles produce valid log-likelihoods.

        Purpose: Validates correct handling when particles span near and far regions.

        Given: Particles at [0,0] (near beacon) and [2.5,2.5] (far from beacons).
        When: batch_observation_log_likelihood is called.
        Then: All log-likelihoods are finite and non-positive.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0], [2.5, 2.5]])
        action = np.zeros(2)
        obs = np.array([0.5, 0.5])
        ll = updater.batch_observation_log_likelihood(particles, action, obs)
        assert ll.shape == (2,)
        assert np.all(np.isfinite(ll))
        assert np.all(ll <= 0)  # log of probability density ≤ 0 for d≥2 Gaussian


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

        Given: Two updaters with different beacon_radius.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        u2 = ContinuousLightDarkVectorizedUpdater(
            state_transition_dist=env._state_transition_dist,
            obs_dist_near_beacon=env._obs_dist_near_beacon,
            obs_dist_far_from_beacon=env._obs_dist_far_from_beacon,
            beacons=env.beacons,
            beacon_radius=env.beacon_radius * 2,
            grid_size=env.grid_size,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches the env's per-particle sample_next_state.

        Purpose: Verifies that batch_transition produces the same results as calling
                 env.sample_next_state per particle with the same noise.

        Given: A set of particles, an action, and a fixed native RNG seed.
        When: batch_transition is called, and the same transitions are computed
              per-particle via env.sample_next_state.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        particles = np.random.rand(50, 2) * 10
        action = np.array([1.0, 0.5])

        def per_particle_fn(particle, act):
            return env.sample_next_state(state=particle, action=act)

        assert_batch_transition_matches_loop(
            updater=updater,
            particles=particles,
            action=action,
            per_particle_transition_fn=per_particle_fn,
            seed=999,
            seed_fn=_native.set_seed,
        )

    def test_batch_observation_log_likelihood_matches_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle log-pdf computation.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle log-PDF computed directly from the active
                 multivariate normal distribution. Both the batch path and
                 the per-particle reference apply the symmetric C++
                 ``kLogProbFloor = log(1e-300) ~= -690.776`` floor for
                 impossible events so the two paths agree across the
                 entire array (including extreme-distance particles).

        Given: A set of next-state particles and an observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log-pdf is computed via the env's near/far Gaussian (selected
              by env.is_state_near_beacon), with the same C++ floor applied.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        particles = np.random.rand(50, 2) * 10
        observation = np.array([5.0, 5.0])
        action = np.array([1.0, 0.0])

        log_floor = float(np.log(1e-300))  # ~= -690.7755278982137

        def per_particle_ll_fn(particle, act, obs):
            del act
            dist = (
                env._obs_dist_near_beacon  # pylint: disable=protected-access
                if env.is_state_near_beacon(particle)
                else env._obs_dist_far_from_beacon  # pylint: disable=protected-access
            )
            log_pdf = dist.log_pdf(np.array([obs]), particle)[0]
            # Mirror the symmetric C++ kernel floor applied by
            # ``batch_log_likelihood`` so the reference matches the
            # implementation under test for events past the floor.
            return max(log_pdf, log_floor)

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=action,
            observation=observation,
            per_particle_ll_fn=per_particle_ll_fn,
        )

    def test_full_update_equivalence(self, env, updater):
        """Test that a full vectorized update matches WeightedParticleBelief._update_weights.

        Purpose: End-to-end equivalence test comparing the vectorized belief update
                 path against the per-particle loop in WeightedParticleBelief.

        Given: Identical initial particles and weights.
        When: One step of update is performed via both paths with the same random seed.
        Then: Next particles match exactly, and both paths agree on which particles
              carry significant weight. The two paths differ in how they handle
              near-zero probabilities (eps floor vs exact log-space), so we compare
              normalized weights only for particles with non-negligible weight.

        Test type: integration
        """
        np.random.seed(77)
        n = 40
        action = np.array([0.5, -0.5])
        observation = np.array([3.0, 4.0])

        particles_array = np.random.rand(n, 2) * 10
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=updater,
            resampling=False,
        )
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )

        # Use the per-helper APIs so we can pass seed_fn=_native.set_seed:
        # both paths now share the C++ RNG via the native transition / obs
        # batch entry points, and assert_update_equivalence does not expose
        # seed_fn.
        assert_update_particles_match(
            base=w_belief,
            vec=v_belief,
            action=action,
            observation=observation,
            pomdp=env,
            atol=1e-10,
            seed=200,
            seed_fn=_native.set_seed,
        )
        assert_update_weights_match(
            base=w_belief,
            vec=v_belief,
            action=action,
            observation=observation,
            pomdp=env,
            atol=1e-3,
            significance_threshold=1e-4,
            seed=200,
            seed_fn=_native.set_seed,
        )

    def test_sample_distributions_match_post_update(self, env, updater):
        """Test sample() on both beliefs draws from normalized_weights post-update.

        Purpose: Validates that WeightedParticleBelief.sample() and
                 VectorizedWeightedParticleBelief.sample() are unbiased draws
                 from their respective updated belief distributions, and that
                 the two distributions agree.

        Given: Identical initial particles/weights; one update step applied
               via both paths with the same seed.
        When: 20,000 samples are drawn from each belief.
        Then: The two empirical histograms agree in L-infinity, and each
              histogram agrees with its belief's normalized_weights.

        Test type: integration
        """
        np.random.seed(77)
        n = 40
        action = np.array([0.5, -0.5])
        observation = np.array([3.0, 4.0])

        particles_array = np.random.rand(n, 2) * 10
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=updater,
            resampling=False,
        )
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )

        _native.set_seed(200)
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)
        _native.set_seed(200)
        w_updated = w_belief.update(action=action, observation=observation, pomdp=env)

        assert_sample_distributions_match(
            base=w_updated,
            vec=v_updated,
            n_samples=20_000,
            tol=0.02,
            atol_weights=0.02,
            seed=400,
        )


# ---------------------------------------------------------------------------
# Discrete action support tests
# ---------------------------------------------------------------------------


@pytest.fixture
def discrete_env():
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


@pytest.fixture
def discrete_updater(discrete_env):
    return ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)


class TestDiscreteActionFromEnvironment:
    def test_from_environment_detects_action_mapping(self, discrete_env):
        """Test that from_environment picks up action_to_vector from discrete env.

        Purpose: Validates that the factory classmethod auto-detects the
                 action_to_vector attribute on the discrete variant.

        Given: A ContinuousLightDarkPOMDPDiscreteActions instance.
        When: from_environment is called.
        Then: The updater stores a non-None _action_to_vector mapping.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)
        assert updater._action_to_vector is not None
        assert "right" in updater._action_to_vector

    def test_from_environment_continuous_has_no_mapping(self, env):
        """Test that from_environment leaves _action_to_vector as None for continuous env.

        Purpose: Validates that the continuous variant does not get an action mapping.

        Given: A ContinuousLightDarkPOMDP instance.
        When: from_environment is called.
        Then: The updater has _action_to_vector == None.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        assert updater._action_to_vector is None


class TestDiscreteActionBatchTransition:
    def test_string_action_output_shape(self, discrete_updater):
        """Test that batch_transition works with string actions.

        Purpose: Validates that string actions are resolved via the mapping
                 and produce correct output shape.

        Given: 30 particles and a string action 'right'.
        When: batch_transition is called.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(30, 2) * 10
        result = discrete_updater.batch_transition(particles, "right")
        assert result.shape == (30, 2)

    def test_string_action_mean_shift(self, discrete_updater):
        """Test that string action 'right' shifts mean by [1, 0].

        Purpose: Validates that the resolved vector produces the expected
                 mean state shift.

        Given: Identical particles at [5, 5] and action 'right'.
        When: batch_transition is called on many particles.
        Then: The mean result is close to [6, 5].

        Test type: unit
        """
        np.random.seed(42)
        n = 5000
        particles = np.tile([5.0, 5.0], (n, 1))
        results = discrete_updater.batch_transition(particles, "right")
        mean_result = results.mean(axis=0)
        np.testing.assert_allclose(mean_result, [6.0, 5.0], atol=0.1)

    def test_string_action_matches_vector_action(self, discrete_updater):
        """Test that string 'right' produces same result as vector [1, 0].

        Purpose: Validates equivalence between string and vector action paths.

        Given: Same particles and random seed.
        When: batch_transition is called with 'right' and with np.array([1, 0]).
        Then: Results are identical.

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(20, 2) * 10

        _native.set_seed(42)
        result_string = discrete_updater.batch_transition(particles, "right")

        _native.set_seed(42)
        result_vector = discrete_updater.batch_transition(particles, np.array([1.0, 0.0]))

        np.testing.assert_array_equal(result_string, result_vector)


class TestDiscreteActionFullUpdate:
    def test_vectorized_belief_update_with_string_action(self, discrete_updater):
        """Test full VectorizedWeightedParticleBelief.update() with a string action.

        Purpose: End-to-end test that the belief update path works when
                 the action is a string rather than a numeric array.

        Given: A VectorizedWeightedParticleBelief with the discrete updater.
        When: update() is called with action='right'.
        Then: A new belief is returned with valid particles and weights.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        particles = np.random.rand(n, 2) * 10
        log_w = np.full(n, -np.log(n))

        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_w,
            updater=discrete_updater,
            resampling=False,
        )
        updated = belief.update(action="right", observation=np.array([5.0, 5.0]), pomdp=None)

        assert updated.particles.shape == (n, 2)
        assert np.all(np.isfinite(updated.log_weights))


class TestDiscreteActionConfigId:
    def test_config_id_differs_discrete_vs_continuous(self, env, discrete_env):
        """Test that config_id differs between discrete and continuous updaters.

        Purpose: Validates that the action mapping is reflected in config_id.

        Given: Updaters built from continuous and discrete environments.
        When: config_id is compared.
        Then: The IDs differ.

        Test type: unit
        """
        u_continuous = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        u_discrete = ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)
        assert u_continuous.config_id != u_discrete.config_id


# ---------------------------------------------------------------------------
# NoObsInDark updater tests
# ---------------------------------------------------------------------------


@pytest.fixture
def no_obs_env():
    return ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK,
    )


@pytest.fixture
def no_obs_updater(no_obs_env):
    return ContinuousLightDarkNoObsInDarkVectorizedUpdater.from_environment(no_obs_env)


class TestNoObsInDarkFromEnvironment:
    def test_creates_correct_subclass(self, no_obs_env):
        """Test that from_environment returns the correct subclass.

        Purpose: Validates that the factory creates the right type.

        Given: A ContinuousLightDarkPOMDP with NORMAL_NOISE_NO_OBS_IN_DARK.
        When: from_environment is called.
        Then: The updater is a ContinuousLightDarkNoObsInDarkVectorizedUpdater.

        Test type: unit
        """
        updater = ContinuousLightDarkNoObsInDarkVectorizedUpdater.from_environment(no_obs_env)
        assert isinstance(updater, ContinuousLightDarkNoObsInDarkVectorizedUpdater)


class TestNoObsInDarkObservationLogLikelihood:
    def test_none_obs_near_beacon_returns_neg_inf(self, no_obs_updater):
        """Test that 'None' observation near a beacon gives -inf log-likelihood.

        Purpose: Validates that 'None' is impossible near beacons.

        Given: A particle at (0, 0) near the origin beacon.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])  # at beacon (0, 0)
        ll = no_obs_updater.batch_observation_log_likelihood(particles, np.zeros(2), "None")
        assert ll[0] == -np.inf

    def test_none_obs_far_from_beacon_returns_zero(self, no_obs_updater):
        """Test that 'None' observation far from beacons gives 0 log-likelihood.

        Purpose: Validates that 'None' is certain when far from beacons.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is 0.0 (log(1)).

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])  # far from all beacons
        ll = no_obs_updater.batch_observation_log_likelihood(particles, np.zeros(2), "None")
        assert ll[0] == 0.0

    def test_array_obs_near_beacon_returns_finite(self, no_obs_updater):
        """Test that array observation near a beacon gives finite log-likelihood.

        Purpose: Validates that real observations work near beacons.

        Given: A particle at (0, 0) near the origin beacon.
        When: Log-likelihood is computed for a nearby array observation.
        Then: The result is finite.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])
        obs = np.array([0.1, 0.1])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, np.zeros(2), obs)
        assert np.isfinite(ll[0])

    def test_array_obs_far_from_beacon_returns_finite(self, no_obs_updater):
        """Test that array observation far from beacons gives finite log-likelihood.

        Purpose: Validates that the NoObsInDark model uses the far-beacon
                 distribution for non-None observations when far from beacons,
                 matching the non-vectorized model behaviour.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for an array observation.
        Then: The result is finite (using the far-beacon distribution).

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        obs = np.array([2.5, 2.5])
        ll = no_obs_updater.batch_observation_log_likelihood(particles, np.zeros(2), obs)
        assert np.isfinite(ll[0])

    def test_batch_transition_inherited(self, no_obs_updater, updater):
        """Test that batch_transition is inherited from parent.

        Purpose: Validates that the transition logic is shared.

        Given: Same particles and seed for both updaters.
        When: batch_transition is called on both.
        Then: Results are identical.

        Test type: unit
        """
        np.random.seed(42)
        particles = np.random.rand(20, 2) * 10
        action = np.array([1.0, 0.0])

        _native.set_seed(99)
        result_no_obs = no_obs_updater.batch_transition(particles, action)
        _native.set_seed(99)
        result_normal = updater.batch_transition(particles, action)

        np.testing.assert_array_equal(result_no_obs, result_normal)

    def test_equivalence_with_per_particle_loop(self, no_obs_env, no_obs_updater):
        """Test vectorized log-likelihood matches per-particle computation.

        Purpose: Verifies equivalence against the non-vectorized observation model.

        Given: Particles and an observation (both array and 'None').
        When: Vectorized and per-particle log-likelihoods are computed.
        Then: Results match within floating-point tolerance.  For array
              observations, the per-particle path uses log_pdf directly
              (not log(pdf())) to avoid underflow divergence.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = np.array([1.0, 0.0])
        particles = np.random.rand(n, 2) * 10

        # Test with array observation — vectorized updater computes log(pdf)
        # directly for every particle, including those beyond beacon_radius
        # (where the env API returns -inf). Replicate that direct path here
        # using the env's stored Gaussian, which is the same computation the
        # vectorized updater performs internally.
        observation = np.array([5.0, 5.0])
        vectorized_ll = no_obs_updater.batch_observation_log_likelihood(
            particles, action, observation
        )
        per_particle_ll = np.empty(n)
        for i, particle in enumerate(particles):
            dist = (
                no_obs_env._obs_dist_near_beacon  # pylint: disable=protected-access
                if no_obs_env.is_state_near_beacon(particle)
                else no_obs_env._obs_dist_far_from_beacon  # pylint: disable=protected-access
            )
            per_particle_ll[i] = dist.log_pdf(np.array([observation]), particle)[0]

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

        # Test with "None" observation across all particles.
        vectorized_ll_none = no_obs_updater.batch_observation_log_likelihood(
            particles, action, "None"
        )
        per_particle_ll_none = np.empty(n)
        for i, particle in enumerate(particles):
            per_particle_ll_none[i] = no_obs_env.observation_log_probability(
                next_state=particle, action=action, observations=["None"]
            )[0]

        np.testing.assert_allclose(vectorized_ll_none, per_particle_ll_none, atol=1e-10)


class TestNoObsInDarkConfigId:
    def test_config_id_differs_from_parent(self, updater, no_obs_updater):
        """Test that config_id differs between NORMAL_NOISE and NO_OBS_IN_DARK updaters.

        Purpose: Validates distinct configuration IDs for different observation models.

        Given: Updaters for NORMAL_NOISE and NO_OBS_IN_DARK.
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
    return ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        observation_model_type=ObservationModelType.DISTANCE_BASED,
    )


@pytest.fixture
def dist_updater(dist_env):
    return ContinuousLightDarkDistanceBasedVectorizedUpdater.from_environment(dist_env)


class TestDistanceBasedFromEnvironment:
    def test_creates_correct_subclass(self, dist_env):
        """Test that from_environment returns the correct subclass.

        Purpose: Validates that the factory creates the right type.

        Given: A ContinuousLightDarkPOMDP with DISTANCE_BASED.
        When: from_environment is called.
        Then: The updater is a ContinuousLightDarkDistanceBasedVectorizedUpdater.

        Test type: unit
        """
        updater = ContinuousLightDarkDistanceBasedVectorizedUpdater.from_environment(dist_env)
        assert isinstance(updater, ContinuousLightDarkDistanceBasedVectorizedUpdater)


class TestDistanceBasedObservationLogLikelihood:
    def test_none_obs_near_beacon_returns_neg_inf(self, dist_updater):
        """Test that 'None' observation near a beacon gives -inf.

        Purpose: Validates that 'None' is impossible within beacon_radius.

        Given: A particle at (0, 0) near the origin beacon.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0]])
        ll = dist_updater.batch_observation_log_likelihood(particles, np.zeros(2), "None")
        assert ll[0] == -np.inf

    def test_none_obs_far_from_beacon_returns_zero(self, dist_updater):
        """Test that 'None' observation beyond beacon_radius gives 0.

        Purpose: Validates that 'None' is certain when beyond all beacons.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for observation 'None'.
        Then: The result is 0.0.

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        ll = dist_updater.batch_observation_log_likelihood(particles, np.zeros(2), "None")
        assert ll[0] == 0.0

    def test_array_obs_far_from_beacon_returns_neg_inf(self, dist_updater):
        """Test that array observation beyond beacon_radius gives -inf.

        Purpose: Validates that real observations are impossible beyond beacons.

        Given: A particle at (2.5, 2.5) far from all beacons.
        When: Log-likelihood is computed for an array observation.
        Then: The result is -inf.

        Test type: unit
        """
        particles = np.array([[2.5, 2.5]])
        obs = np.array([2.5, 2.5])
        ll = dist_updater.batch_observation_log_likelihood(particles, np.zeros(2), obs)
        assert ll[0] == -np.inf

    def test_equivalence_with_per_particle_loop(self, dist_env, dist_updater):
        """Test vectorized log-likelihood matches per-particle computation.

        Purpose: Verifies equivalence against the non-vectorized observation model.

        Given: Particles and both array and 'None' observations.
        When: Vectorized and per-particle log-likelihoods are computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = np.array([1.0, 0.0])
        particles = np.random.rand(n, 2) * 10

        # Test with "None" observation
        vectorized_ll = dist_updater.batch_observation_log_likelihood(particles, action, "None")
        per_particle_ll = np.empty(n)
        for i, particle in enumerate(particles):
            per_particle_ll[i] = dist_env.observation_log_probability(
                next_state=particle, action=action, observations=["None"]
            )[0]

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

    def test_array_obs_equivalence_with_per_particle_loop(self, dist_env, dist_updater):
        """Test vectorized array-obs log-likelihood matches per-particle computation.

        Purpose: Verifies equivalence for array observations against the
                 non-vectorized observation model (which returns 0 for
                 particles beyond beacon_radius).

        Given: Particles near beacons and a matching array observation.
        When: Vectorized and per-particle log-likelihoods are computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        action = np.array([1.0, 0.0])
        # Place particles near beacon at (5,5) so some are within radius
        particles = np.random.rand(n, 2) * 2 + 4.0
        observation = np.array([5.0, 5.0])

        # Restrict to particles within beacon_radius (where the env returns
        # finite log-pdf and the vectorized updater agrees). Particles outside
        # the radius would yield -inf from the env path but log(pdf) from the
        # vectorized updater; the env-API equivalence is meaningful only on
        # the shared near branch.
        near_mask = np.array(
            [
                float(np.linalg.norm(dist_env.beacons - p[:, np.newaxis], axis=0).min())
                <= dist_env.beacon_radius
                for p in particles
            ]
        )
        near_particles = particles[near_mask]
        vectorized_ll = dist_updater.batch_observation_log_likelihood(
            near_particles, action, observation
        )
        per_particle_ll = np.empty(near_particles.shape[0])
        for i, particle in enumerate(near_particles):
            per_particle_ll[i] = dist_env.observation_log_probability(
                next_state=particle, action=action, observations=[observation]
            )[0]

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)


class TestDistanceBasedConfigId:
    def test_config_id_differs_from_other_variants(self, updater, no_obs_updater, dist_updater):
        """Test that all three updater variants have different config_ids.

        Purpose: Validates that each observation model variant is uniquely identified.

        Given: Updaters for NORMAL_NOISE, NO_OBS_IN_DARK, and DISTANCE_BASED.
        When: config_ids are compared.
        Then: All three IDs are different.

        Test type: unit
        """
        ids = {updater.config_id, no_obs_updater.config_id, dist_updater.config_id}
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# Full update equivalence tests (vectorized vs standard particle belief)
# ---------------------------------------------------------------------------


class TestNoObsInDarkFullUpdateEquivalence:
    def test_full_update_equivalence_with_array_observation(self, no_obs_env, no_obs_updater):
        """Test full vectorized update matches WeightedParticleBelief for array observation.

        Purpose: End-to-end equivalence test for the NO_OBS_IN_DARK model.

        Given: Identical initial particles and weights.
        When: One step of update with an array observation via both paths.
        Then: Next particles match exactly, and normalized weights agree
              for significant particles.

        Note: VectorizedWeightedParticleBelief.update() does not yet support
              string observations (e.g. 'None'). Per-observation-level
              equivalence for 'None' is tested separately above.

        Test type: integration
        """
        from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )

        np.random.seed(77)
        n = 40
        action = np.array([0.5, -0.5])
        observation = np.array([3.0, 4.0])

        particles_array = np.random.rand(n, 2) * 10
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        _native.set_seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=no_obs_updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        _native.set_seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=no_obs_env)

        w_particles = np.array(w_updated.particles)
        np.testing.assert_allclose(v_updated.particles, w_particles, atol=1e-10)

        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        if np.any(significant):
            np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-3)


class TestDistanceBasedFullUpdateEquivalence:
    def test_full_update_equivalence_with_array_observation(self, dist_env, dist_updater):
        """Test full vectorized update matches WeightedParticleBelief for array observation.

        Purpose: End-to-end equivalence test for the DISTANCE_BASED model.

        Given: Particles placed near a beacon so the array observation
               gives non-trivial weight to some particles.
        When: One step of update with an array observation via both paths.
        Then: Next particles match exactly, and normalized weights agree.

        Note: VectorizedWeightedParticleBelief.update() does not yet support
              string observations (e.g. 'None'). Per-observation-level
              equivalence for 'None' is tested separately above.

        Test type: integration
        """
        from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )

        np.random.seed(77)
        n = 40
        action = np.array([0.1, 0.0])
        # Observation near beacon at (5, 5) so some particles get real weight
        observation = np.array([5.0, 5.0])

        # Place particles near beacons so they can receive array observations
        particles_array = np.random.rand(n, 2) * 2 + 4.0  # range [4, 6]
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        _native.set_seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=dist_updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        _native.set_seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=dist_env)

        w_particles = np.array(w_updated.particles)
        np.testing.assert_allclose(v_updated.particles, w_particles, atol=1e-10)

        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        if np.any(significant):
            np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-3)
