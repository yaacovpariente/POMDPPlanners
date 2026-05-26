# SPDX-License-Identifier: MIT

"""Tests for RockSample vectorized particle belief updater."""

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (
    OBS_BAD,
    OBS_GOOD,
    OBS_NONE,
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)


@pytest.fixture()
def simple_env():
    return RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(1, 1), (3, 3)],
        init_pos=(2, 2),
        sensor_efficiency=10.0,
        discount_factor=0.95,
    )


@pytest.fixture()
def updater(simple_env):
    return RockSampleVectorizedUpdater.from_environment(simple_env)


@pytest.fixture()
def sample_particles(simple_env):
    np.random.seed(42)
    states = simple_env.initial_state_dist().sample(n_samples=10)
    return np.stack(states)


class TestFromEnvironmentConstruction:
    def test_creates_without_error(self, simple_env):
        """Test that from_environment constructs an updater successfully.

        Purpose: Validates updater creation from environment.

        Given: A configured RockSamplePOMDP environment.
        When: RockSampleVectorizedUpdater.from_environment is called.
        Then: Updater has matching attributes.

        Test type: unit
        """
        upd = RockSampleVectorizedUpdater.from_environment(simple_env)
        assert upd.map_rows == 5
        assert upd.map_cols == 5
        assert upd.num_rocks == 2
        assert upd.rock_positions.shape == (2, 2)
        assert upd.sensor_efficiency == 10.0


class TestBatchTransition:
    def test_shape_preserved(self, updater, sample_particles):
        """Test that batch_transition preserves particle array shape.

        Purpose: Validates output shape matches input shape.

        Given: Particles of shape (N, d).
        When: batch_transition is called with any action.
        Then: Output has the same shape.

        Test type: unit
        """
        result = updater.batch_transition(sample_particles, np.array(2))
        assert result.shape == sample_particles.shape

    def test_terminal_unchanged(self, updater):
        """Test that terminal particles remain unchanged after transition.

        Purpose: Validates that terminal states are not modified.

        Given: A particle with terminal position (-1, -1).
        When: batch_transition is called.
        Then: The particle remains at (-1, -1).

        Test type: unit
        """
        terminal = np.array([[-1.0, -1.0, 1.0, 0.0]])
        for action in range(7):
            result = updater.batch_transition(terminal, np.array(action))
            assert result[0, 0] == -1.0
            assert result[0, 1] == -1.0

    def test_north_movement(self, updater):
        """Test northward movement decreases row by 1.

        Purpose: Validates North action (1) moves robot correctly.

        Given: Robot at position (2, 2).
        When: North action is applied.
        Then: Robot row decreases by 1.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(1))
        assert result[0, 0] == 1.0
        assert result[0, 1] == 2.0

    def test_north_clamps_at_boundary(self, updater):
        """Test that North movement clamps at row 0.

        Purpose: Validates boundary clamping for North action.

        Given: Robot at row 0.
        When: North action is applied.
        Then: Robot stays at row 0.

        Test type: unit
        """
        particles = np.array([[0.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(1))
        assert result[0, 0] == 0.0

    def test_east_movement(self, updater):
        """Test eastward movement increases column by 1.

        Purpose: Validates East action (2) moves robot correctly.

        Given: Robot at position (2, 2).
        When: East action is applied.
        Then: Robot column increases by 1.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(2))
        assert result[0, 0] == 2.0
        assert result[0, 1] == 3.0

    def test_south_movement(self, updater):
        """Test southward movement increases row by 1.

        Purpose: Validates South action (3) moves robot correctly.

        Given: Robot at position (2, 2).
        When: South action is applied.
        Then: Robot row increases by 1.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(3))
        assert result[0, 0] == 3.0
        assert result[0, 1] == 2.0

    def test_south_clamps_at_boundary(self, updater):
        """Test that South movement clamps at last row.

        Purpose: Validates boundary clamping for South action.

        Given: Robot at last row (4).
        When: South action is applied.
        Then: Robot stays at row 4.

        Test type: unit
        """
        particles = np.array([[4.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(3))
        assert result[0, 0] == 4.0

    def test_west_movement(self, updater):
        """Test westward movement decreases column by 1.

        Purpose: Validates West action (4) moves robot correctly.

        Given: Robot at position (2, 2).
        When: West action is applied.
        Then: Robot column decreases by 1.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(4))
        assert result[0, 0] == 2.0
        assert result[0, 1] == 1.0

    def test_west_clamps_at_boundary(self, updater):
        """Test that West movement clamps at column 0.

        Purpose: Validates boundary clamping for West action.

        Given: Robot at column 0.
        When: West action is applied.
        Then: Robot stays at column 0.

        Test type: unit
        """
        particles = np.array([[2.0, 0.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(4))
        assert result[0, 1] == 0.0

    def test_east_exit_becomes_terminal(self, updater):
        """Test that moving east from last column triggers exit.

        Purpose: Validates exit condition sets terminal state.

        Given: Robot at rightmost column (4) in a 5-wide map.
        When: East action is applied.
        Then: Robot position becomes (-1, -1).

        Test type: unit
        """
        particles = np.array([[2.0, 4.0, 1.0, 0.0]])
        result = updater.batch_transition(particles, np.array(2))
        assert result[0, 0] == -1.0
        assert result[0, 1] == -1.0

    def test_sample_at_rock_sets_quality_zero(self, updater):
        """Test that sampling at a rock position sets its quality to 0.

        Purpose: Validates Sample action (0) at a rock position.

        Given: Robot at rock position (1, 1) with rock quality 1.0.
        When: Sample action is applied.
        Then: Rock quality becomes 0.0.

        Test type: unit
        """
        particles = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = updater.batch_transition(particles, np.array(0))
        assert result[0, 2] == 0.0  # rock 0 at (1,1) sampled
        assert result[0, 3] == 1.0  # rock 1 at (3,3) unchanged

    def test_sample_not_at_rock_no_change(self, updater):
        """Test that sampling away from rocks doesn't change state.

        Purpose: Validates Sample action has no effect when not at a rock.

        Given: Robot at position (2, 2) with no rocks there.
        When: Sample action is applied.
        Then: All rock qualities unchanged.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 1.0]])
        result = updater.batch_transition(particles, np.array(0))
        assert result[0, 2] == 1.0
        assert result[0, 3] == 1.0

    def test_check_action_no_state_change(self, updater):
        """Test that check actions do not modify the state.

        Purpose: Validates Check actions (5+) leave state unchanged.

        Given: An arbitrary particle state.
        When: Check action is applied.
        Then: State is identical to input.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 1.0, 0.0]])
        for check_action in [5, 6]:
            result = updater.batch_transition(particles, np.array(check_action))
            np.testing.assert_array_equal(result, particles)

    def test_does_not_mutate_input(self, updater, sample_particles):
        """Test that batch_transition does not modify the input array.

        Purpose: Validates immutability of input particles.

        Given: A particle array.
        When: batch_transition is called.
        Then: The original array is unchanged.

        Test type: unit
        """
        original = sample_particles.copy()
        updater.batch_transition(sample_particles, np.array(2))
        np.testing.assert_array_equal(sample_particles, original)


class TestBatchObservationLogLikelihood:
    def test_shape(self, updater, sample_particles):
        """Test that output shape is (N,).

        Purpose: Validates output dimensionality.

        Given: Particles of shape (N, d).
        When: batch_observation_log_likelihood is called.
        Then: Output has shape (N,).

        Test type: unit
        """
        log_ll = updater.batch_observation_log_likelihood(
            sample_particles, np.array(5), np.array(OBS_GOOD)
        )
        assert log_ll.shape == (sample_particles.shape[0],)

    def test_movement_action_none_obs(self, updater, sample_particles):
        """Test that movement actions with 'none' obs give log-likelihood 0.

        Purpose: Validates deterministic 'none' observation for movement actions.

        Given: Particles with movement action (East).
        When: Observation is 'none' (0).
        Then: All log-likelihoods are 0.0.

        Test type: unit
        """
        for action in range(5):
            log_ll = updater.batch_observation_log_likelihood(
                sample_particles, np.array(action), np.array(OBS_NONE)
            )
            np.testing.assert_array_equal(log_ll, 0.0)

    def test_movement_action_non_none_obs(self, updater, sample_particles):
        """Test that movement actions with non-'none' obs give the symmetric C++ floor.

        Purpose: Validates impossible observations for movement actions.
            Post symmetric C++ floor, the historical -inf is now floored
            to ``log(1e-300) ~= -690.776`` so the batch and scalar API
            paths agree on the same value.

        Given: Particles with movement action.
        When: Observation is 'good' or 'bad'.
        Then: All log-likelihoods equal ``log(1e-300) ~= -690.776``.

        Test type: unit
        """
        floor = float(np.log(1e-300))  # ~= -690.7755278982137
        for action in range(5):
            for obs in [OBS_GOOD, OBS_BAD]:
                log_ll = updater.batch_observation_log_likelihood(
                    sample_particles, np.array(action), np.array(obs)
                )
                np.testing.assert_allclose(log_ll, np.full_like(log_ll, floor), atol=1e-6)

    def test_check_good_rock_close_good_obs(self, updater):
        """Test high log-likelihood for correct obs near a good rock.

        Purpose: Validates sensor model gives high probability for correct
            observation when close to a good rock.

        Given: Robot at rock 0 position (1, 1) with good rock quality.
        When: Check rock 0 with 'good' observation.
        Then: Log-likelihood is close to 0 (high probability).

        Test type: unit
        """
        particles = np.array([[1.0, 1.0, 1.0, 0.0]])
        log_ll = updater.batch_observation_log_likelihood(
            particles, np.array(5), np.array(OBS_GOOD)
        )
        # At distance 0, efficiency ≈ 1.0, so log(1.0) ≈ 0.0
        assert log_ll[0] > -0.1

    def test_check_good_rock_far_bad_obs(self, updater):
        """Test that bad obs has higher log-likelihood when far from good rock.

        Purpose: Validates sensor noise increases with distance.

        Given: Robot far from good rock 0.
        When: Check rock 0 with 'bad' observation (incorrect).
        Then: Log-likelihood for 'bad' is higher when far than when close.

        Test type: unit
        """
        # Close: robot at (1, 1), rock 0 at (1, 1), distance = 0
        close = np.array([[1.0, 1.0, 1.0, 0.0]])
        ll_close = updater.batch_observation_log_likelihood(close, np.array(5), np.array(OBS_BAD))

        # Far: robot at (4, 4), rock 0 at (1, 1), distance ≈ 4.24
        far = np.array([[4.0, 4.0, 1.0, 0.0]])
        ll_far = updater.batch_observation_log_likelihood(far, np.array(5), np.array(OBS_BAD))

        # Bad obs should be more likely when far (more noise)
        assert ll_far[0] > ll_close[0]

    def test_terminal_particles_neg_inf(self, updater):
        """Test that terminal particles get the symmetric C++ floor for check obs.

        Purpose: Validates terminal handling in observation model. Post
            symmetric C++ floor, the historical -inf is now floored to
            ``log(1e-300) ~= -690.776`` (kept the function name for
            historical continuity).

        Given: A terminal particle (-1, -1).
        When: Check action with 'good' observation.
        Then: Log-likelihood equals ``log(1e-300) ~= -690.776``.

        Test type: unit
        """
        terminal = np.array([[-1.0, -1.0, 1.0, 0.0]])
        log_ll = updater.batch_observation_log_likelihood(terminal, np.array(5), np.array(OBS_GOOD))
        floor = float(np.log(1e-300))  # ~= -690.7755278982137
        np.testing.assert_allclose(log_ll[0], floor, atol=1e-6)

    def test_check_none_obs_impossible(self, updater, sample_particles):
        """Test that 'none' observation is impossible for check actions.

        Purpose: Validates that check actions cannot produce 'none'.
            Post symmetric C++ floor, the historical -inf is now floored
            to ``log(1e-300) ~= -690.776`` so the batch and scalar API
            paths agree on the same value.

        Given: Particles with check action.
        When: Observation is 'none'.
        Then: All log-likelihoods equal ``log(1e-300) ~= -690.776``.

        Test type: unit
        """
        log_ll = updater.batch_observation_log_likelihood(
            sample_particles, np.array(5), np.array(OBS_NONE)
        )
        floor = float(np.log(1e-300))  # ~= -690.7755278982137
        np.testing.assert_allclose(log_ll, np.full_like(log_ll, floor), atol=1e-6)


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, simple_env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as
                 calling the environment's state_transition_model per particle.

        Given: A set of particles sampled from the initial distribution.
        When: batch_transition is called for movement, sample, and check actions,
              and the same transitions are computed per-particle.
        Then: Results match exactly (deterministic transitions).

        Test type: integration
        """
        np.random.seed(42)
        states = simple_env.initial_state_dist().sample(n_samples=50)
        particles = np.stack(states)

        def per_particle_fn(particle, action):
            return simple_env.sample_next_state(state=particle, action=action)

        for action in [0, 1, 2, 3, 4, 5, 6]:  # Sample, N, E, S, W, Check0, Check1
            assert_batch_transition_matches_loop(
                updater=updater,
                particles=particles,
                action=action,
                per_particle_transition_fn=per_particle_fn,
                err_msg=f"Mismatch for action {action}",
            )

    def test_batch_obs_log_likelihood_matches_per_particle_loop(self, simple_env, updater):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle observation probability from the environment.

        Given: A set of particles and a sensor observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log(observation_model.probability) is computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        states = simple_env.initial_state_dist().sample(n_samples=50)
        particles = np.stack(states)

        def per_particle_ll_fn(particle, action, obs):
            obs_str = {OBS_NONE: "none", OBS_GOOD: "good", OBS_BAD: "bad"}[int(obs)]
            log_prob = simple_env.observation_log_probability(particle, action, [obs_str])[0]
            return log_prob

        # Check rock 0 with "good" observation
        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=5,
            observation=np.array(OBS_GOOD),
            per_particle_ll_fn=per_particle_ll_fn,
        )

        # Check rock 1 with "bad" observation
        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=6,
            observation=np.array(OBS_BAD),
            per_particle_ll_fn=per_particle_ll_fn,
        )


class TestConfigId:
    def test_deterministic(self, simple_env):
        """Test that config_id is deterministic for same environment.

        Purpose: Validates reproducible config identification.

        Given: Two updaters created from the same environment.
        When: config_id is computed for each.
        Then: Both return the same string.

        Test type: unit
        """
        upd1 = RockSampleVectorizedUpdater.from_environment(simple_env)
        upd2 = RockSampleVectorizedUpdater.from_environment(simple_env)
        assert upd1.config_id == upd2.config_id
