# SPDX-License-Identifier: MIT

"""Tests for OccupancyGridMappingVectorizedUpdater.

Every batched kernel is checked against the scalar environment method it
replaces, particle by particle. The generative transition draws from the global
NumPy stream in the same order as the scalar loop when moves cannot fail, so
that case is compared bit for bit under a shared seed; with move failure the
two paths interleave their draws differently and only the motion statistics
are compared.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridAction,
    OccupancyGridMappingPOMDP,
    OccupancyGridMappingVectorizedUpdater,
    RangeNoiseModel,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    occupancy_grid_mapping_pinned_kwargs,
)

ACTIONS = [int(action) for action in OccupancyGridAction]
RANGE_NOISE_MODELS = list(RangeNoiseModel)


def make_env(**overrides):
    """A small pinned world so a test runs in well under a second."""
    kwargs = occupancy_grid_mapping_pinned_kwargs(
        num_rows=8, num_cols=8, num_beams=12, start_row=4, start_col=4, **overrides
    )
    return OccupancyGridMappingPOMDP(discount_factor=0.95, **kwargs)


def diverse_particles(env, n_particles=24, seed=11):
    """Prior maps whose robots have been driven to different poses.

    Fresh prior particles all share the start pose, which would leave the
    per-particle pose handling untested. Driving disjoint subsets through
    different action strings spreads the poses, headings and step counts.
    """
    np.random.seed(seed)
    particles = np.asarray(env.initial_state_dist().sample(n_particles))
    scripts = ([0, 0], [1, 0], [2, 0, 0], [0, 1, 0], [])
    for index, script in enumerate(scripts):
        rows = np.arange(index, n_particles, len(scripts))
        for action in script:
            particles[rows] = np.asarray(
                [env.sample_next_state(particle, action) for particle in particles[rows]]
            )
    return particles


def observation_from(env, particle, action):
    successor = env.sample_next_state(particle, action)
    return env.sample_observation(successor, action)


@pytest.fixture
def env():
    return make_env()


@pytest.fixture
def updater(env):
    return OccupancyGridMappingVectorizedUpdater.from_environment(env)


@pytest.fixture
def particles(env):
    return diverse_particles(env)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_environment_copies_sensor_and_motion_settings(self, env, updater):
        """Purpose: the updater must describe the same world as the environment.

        Given: An environment with pinned settings.
        When: An updater is built from it.
        Then: Grid, beam, range, noise, log-odds and motion settings all match,
            and the derived state layout matches the environment's.

        Test type: unit
        """
        assert (updater.num_rows, updater.num_cols) == (env.num_rows, env.num_cols)
        assert updater.num_beams == env.num_beams
        assert updater.max_range_cells == env.max_range_cells
        assert updater.range_noise_std_cells == env.range_noise_std_cells
        assert updater.free_log_odds == env.free_log_odds
        assert updater.occupied_log_odds == env.occupied_log_odds
        assert updater.log_odds_clamp == env.log_odds_clamp
        assert updater.move_failure_probability == env.move_failure_probability
        assert updater.state_size == env.state_size
        assert updater.scan_offset == env.scan_offset
        assert updater.log_odds_offset == env.log_odds_offset

    def test_zero_noise_is_rejected(self):
        """Purpose: a zero-width Gaussian has no density to weight with.

        Test type: unit
        """
        with pytest.raises(ValueError, match="range_noise_std_cells"):
            OccupancyGridMappingVectorizedUpdater(
                num_rows=5,
                num_cols=5,
                num_beams=4,
                field_of_view_degrees=360.0,
                max_range_cells=2.0,
                range_noise_std_cells=0.0,
                free_log_odds=-1.0,
                occupied_log_odds=1.0,
                log_odds_clamp=6.0,
                move_failure_probability=0.0,
            )

    def test_config_id_tracks_every_setting(self, env, updater):
        """Purpose: two updaters that would score differently must cache differently.

        Given: An updater built from the environment.
        When: One sensor setting changes, and separately the contract version.
        Then: Each change gives a new identifier; an identical build gives the same.

        Test type: unit
        """
        same = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        assert same.config_id == updater.config_id
        noisier = OccupancyGridMappingVectorizedUpdater.from_environment(
            make_env(range_noise_std_cells=0.5)
        )
        assert noisier.config_id != updater.config_id
        env.sensor_contract_version += 1
        assert OccupancyGridMappingVectorizedUpdater.from_environment(env).config_id != (
            updater.config_id
        )


# ---------------------------------------------------------------------------
# Conditional kernels
# ---------------------------------------------------------------------------


class TestPredictiveLogLikelihood:
    @pytest.mark.parametrize("move_failure_probability", [0.0, 0.25])
    @pytest.mark.parametrize("action", ACTIONS)
    def test_matches_scalar_predictive_density(self, action, move_failure_probability):
        """Purpose: the batched score is the filter's importance weight; it must
        equal the scalar one for every particle, including the ``-inf`` entries
        for particles whose robot cannot have reached the observed pose.

        Given: Twenty-four particles at mixed poses, and an observation drawn
            from one of them.
        When: The batched and the scalar predictive scores are computed.
        Then: They agree to floating-point precision, and at least one particle
            has finite score (the one the observation was drawn from).

        Test type: unit
        """
        env = make_env(move_failure_probability=move_failure_probability)
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = diverse_particles(env)
        observation = observation_from(env, particles[7], action)

        batched = updater.batch_predictive_log_likelihood(particles, action, observation)
        scalar = np.array(
            [
                env.predictive_observation_log_probability(particle, action, observation)
                for particle in particles
            ]
        )
        np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-12)
        assert np.isfinite(batched[7])

    def test_motion_failure_weight_matches_scalar_ratio(self):
        """Purpose: a failed move must carry ``p_fail`` and a successful one
        ``1 - p_fail``, exactly as the scalar model integrates them.

        Given: The observation-contract world -- one east-facing beam over an
            empty 7x7 grid, so the nominal scan is the same at both poses --
            with move failure 0.25.
        When: The observed pose is the moved cell, then the start cell.
        Then: The score ratio is 3 for every particle, as in the scalar test.

        Test type: unit
        """
        env = OccupancyGridMappingPOMDP(
            num_rows=7,
            num_cols=7,
            num_beams=1,
            field_of_view_degrees=1,
            max_range_cells=2,
            has_boundary_wall=False,
            num_obstacles=0,
            move_failure_probability=0.25,
        )
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = np.asarray(env.initial_state_dist().sample(3))
        ratio = np.exp(
            updater.batch_predictive_log_likelihood(particles, 0, np.array([2, 3, 0, 2]))
            - updater.batch_predictive_log_likelihood(particles, 0, np.array([3, 3, 0, 2]))
        )
        np.testing.assert_allclose(ratio, 3.0)

    def test_unreachable_pose_and_malformed_observation_have_no_support(
        self, env, updater, particles
    ):
        """Purpose: no epsilon floor may sneak in through the batched path.

        Given: A valid observation.
        When: Its pose is shifted by a fraction, moved two cells away, or a
            range is made non-finite, or the vector has the wrong length.
        Then: Every particle scores ``-inf``.

        Test type: unit
        """
        observation = observation_from(env, particles[0], 0)
        fractional = observation.copy()
        fractional[0] += 0.1
        far = observation.copy()
        far[0] += 2
        non_finite = observation.copy()
        non_finite[5] = np.nan
        for bad in (fractional, far, non_finite, observation[:-1]):
            assert np.all(np.isneginf(updater.batch_predictive_log_likelihood(particles, 0, bad)))


class TestStateFromObservation:
    @pytest.mark.parametrize("action", ACTIONS)
    def test_matches_scalar_map_update(self, env, updater, particles, action):
        """Purpose: the shared inverse-sensor delta must produce exactly the
        scalar successor for every particle, whatever map it carries.

        Given: Particles at mixed poses and an observation from one of them.
        When: Both paths install the observation.
        Then: Step, pose, hidden map, log-odds and stored scan are identical.

        Test type: unit
        """
        observation = observation_from(env, particles[3], action)
        batched = updater.batch_state_from_observation(particles, observation)
        scalar = np.asarray(
            [env.state_from_observation(particle, observation) for particle in particles]
        )
        np.testing.assert_array_equal(batched, scalar)
        assert np.all(batched[:, 0] == particles[:, 0] + 1)
        assert np.all(
            np.abs(batched[:, env.log_odds_offset : env.scan_offset]) <= env.log_odds_clamp
        )

    def test_clamp_is_applied_after_accumulation(self, env, updater, particles):
        """Purpose: repeated evidence must saturate at the clamp, not grow past it.

        Given: The same observation installed twenty times over.
        When: The log-odds are read back.
        Then: No cell exceeds the clamp in magnitude, and the scalar path agrees.

        Test type: unit
        """
        observation = observation_from(env, particles[0], 0)
        batched = particles
        scalar = list(particles)
        for _ in range(20):
            batched = updater.batch_state_from_observation(batched, observation)
            scalar = [env.state_from_observation(particle, observation) for particle in scalar]
        np.testing.assert_array_equal(batched, np.asarray(scalar))
        assert np.max(np.abs(batched[:, env.log_odds_offset : env.scan_offset])) == pytest.approx(
            env.log_odds_clamp
        )

    def test_rejects_the_same_observations_the_scalar_path_rejects(self, env, updater, particles):
        """Purpose: validation must not be looser in the batched path.

        Test type: unit
        """
        observation = observation_from(env, particles[0], 0)
        fractional = observation.copy()
        fractional[2] += 0.5
        outside = observation.copy()
        outside[0] = env.num_rows
        with pytest.raises(ValueError, match="integer"):
            updater.batch_state_from_observation(particles, fractional)
        with pytest.raises(ValueError, match="in-grid"):
            updater.batch_state_from_observation(particles, outside)
        with pytest.raises(ValueError, match="finite"):
            updater.batch_state_from_observation(particles, observation[:-1])


# ---------------------------------------------------------------------------
# Shared updater interface (generative model)
# ---------------------------------------------------------------------------


class TestGenerativeInterface:
    @pytest.mark.parametrize("action", ACTIONS)
    def test_batch_transition_matches_scalar_under_shared_seed(
        self, env, updater, particles, action
    ):
        """Purpose: with no move failure both paths consume the Gaussian stream
        in the same order, so the successors must be bit-identical.

        Given: Particles at mixed poses, no move failure.
        When: Both paths transition from the same seed.
        Then: The successor arrays are equal.

        Test type: unit
        """
        assert_batch_transition_matches_loop(
            updater,
            particles,
            action,
            env.sample_next_state,
            atol=0.0,
            seed=5,
        )

    def test_batch_transition_motion_failure_rate(self):
        """Purpose: a failed move keeps the robot in place with ``p_fail``.

        Given: 4000 copies of one free-space map, move failure 0.4.
        When: All move forward once.
        Then: The fraction that stayed is 0.4 within sampling error, the rest
            are one cell north, and every scan differs (fresh noise per particle).

        Test type: unit
        """
        env = make_env(move_failure_probability=0.4, num_obstacles=0)
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        np.random.seed(2)
        particles = np.tile(env.initial_state_dist().sample(1)[0], (4000, 1))
        successors = updater.batch_transition(particles, 0)
        stayed = successors[:, 1] == 4
        assert np.mean(stayed) == pytest.approx(0.4, abs=0.03)
        assert np.all(successors[~stayed, 1] == 3)
        assert len(np.unique(successors[:, env.scan_offset :], axis=0)) == 4000

    @pytest.mark.parametrize("action", ACTIONS)
    def test_batch_observation_log_likelihood_is_a_point_mass(
        self, env, updater, particles, action
    ):
        """Purpose: the augmented observation kernel is a point mass on the
        stored scan; the batched form must say 0 for the matching successor
        and ``-inf`` for every other one.

        Test type: unit
        """
        np.random.seed(3)
        successors = updater.batch_transition(particles, action)
        observation = env.sample_observation(successors[5], action)
        assert_batch_obs_log_likelihood_matches_loop(
            updater,
            successors,
            action,
            observation,
            lambda particle, act, obs: env.observation_log_probability(particle, act, obs)[0],
            atol=0.0,
        )
        scores = updater.batch_observation_log_likelihood(successors, action, observation)
        assert scores[5] == 0.0
        assert np.count_nonzero(np.isfinite(scores)) == 1


# ---------------------------------------------------------------------------
# Truncated-normal range law
# ---------------------------------------------------------------------------


def truncated_env(**overrides):
    """The pinned world under the truncated law, noisy enough for it to bite.

    At ``sigma = 1.0`` a beam stopping one cell away sits one standard
    deviation above zero, so the per-particle normaliser is far from one and
    a batched path that hoisted it into a constant would fail every parity
    check below.
    """
    return make_env(
        range_noise_std_cells=1.0, range_noise_model=RangeNoiseModel.TRUNCATED_NORMAL, **overrides
    )


class TestTruncatedNormalMode:
    def test_from_environment_copies_the_law_and_the_identity_tracks_it(self, updater):
        """Purpose: the updater must score under the environment's law, and
        two updaters under different laws must cache differently.

        Test type: unit
        """
        assert updater.range_noise_model is RangeNoiseModel.GAUSSIAN
        truncated = OccupancyGridMappingVectorizedUpdater.from_environment(
            make_env(range_noise_model="truncated_normal")
        )
        assert truncated.range_noise_model is RangeNoiseModel.TRUNCATED_NORMAL
        assert truncated.config_id != updater.config_id
        with pytest.raises(ValueError, match="range_noise_model"):
            OccupancyGridMappingVectorizedUpdater(
                num_rows=5,
                num_cols=5,
                num_beams=4,
                field_of_view_degrees=360.0,
                max_range_cells=2.0,
                range_noise_std_cells=0.5,
                free_log_odds=-1.0,
                occupied_log_odds=1.0,
                log_odds_clamp=6.0,
                move_failure_probability=0.0,
                range_noise_model="clipped",
            )

    @pytest.mark.parametrize("move_failure_probability", [0.0, 0.25])
    @pytest.mark.parametrize("action", ACTIONS)
    def test_predictive_density_matches_scalar_with_per_particle_normalisers(
        self, action, move_failure_probability
    ):
        """Purpose: the batched importance weight must carry each particle's
        own ``Phi(rho / sigma)`` per beam, exactly as the scalar path does.

        Given: Particles at mixed poses with maps that put obstacles at
            different distances, under the truncated law with wide noise.
        When: Both paths score an observation drawn from one particle.
        Then: They agree to floating-point precision, the drawn-from particle
            is finite, and the scores are not what the Gaussian law gives.

        Test type: unit
        """
        env = truncated_env(move_failure_probability=move_failure_probability)
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = diverse_particles(env)
        observation = observation_from(env, particles[7], action)
        batched = updater.batch_predictive_log_likelihood(particles, action, observation)
        scalar = np.array(
            [
                env.predictive_observation_log_probability(particle, action, observation)
                for particle in particles
            ]
        )
        np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-12)
        assert np.isfinite(batched[7])
        gaussian = OccupancyGridMappingVectorizedUpdater.from_environment(
            make_env(
                range_noise_std_cells=1.0, move_failure_probability=move_failure_probability
            )
        ).batch_predictive_log_likelihood(particles, action, observation)
        finite = np.isfinite(batched)
        assert np.array_equal(finite, np.isfinite(gaussian))
        assert np.all(batched[finite] > gaussian[finite])

    def test_a_negative_reading_has_no_support_in_the_batched_path(self):
        """Test type: unit"""
        env = truncated_env()
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = diverse_particles(env)
        observation = observation_from(env, particles[0], 0)
        observation[5] = -0.01
        assert np.all(np.isneginf(updater.batch_predictive_log_likelihood(particles, 0, observation)))
        observation[5] = 0.0
        assert np.isfinite(updater.batch_predictive_log_likelihood(particles, 0, observation)[0])

    @pytest.mark.parametrize("action", ACTIONS)
    def test_batch_transition_matches_scalar_under_shared_seed_and_stays_nonnegative(
        self, action
    ):
        """Purpose: the truncated sampler consumes one uniform per beam in the
        same order on both paths, so the successors must be bit-identical, and
        no stored range may be negative.

        Test type: unit
        """
        env = truncated_env()
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = diverse_particles(env)
        assert_batch_transition_matches_loop(
            updater, particles, action, env.sample_next_state, atol=0.0, seed=5
        )
        np.random.seed(5)
        successors = updater.batch_transition(particles, action)
        assert successors[:, env.scan_offset :].min() >= 0.0
        np.random.seed(5)
        gaussian = OccupancyGridMappingVectorizedUpdater.from_environment(
            make_env(range_noise_std_cells=1.0)
        ).batch_transition(particles, action)
        assert gaussian[:, env.scan_offset :].min() < 0.0
