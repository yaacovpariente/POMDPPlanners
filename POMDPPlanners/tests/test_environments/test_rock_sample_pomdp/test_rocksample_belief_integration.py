"""Integration tests for RockSample vectorized belief."""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_belief_factory import (
    RockSampleVectorizedWeightedParticleBelief,
    create_rocksample_belief,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_chained_update_equivalence,
    assert_sample_distributions_match,
    assert_update_particles_match,
    assert_update_weights_match,
)
from POMDPPlanners.utils.belief_factory import BeliefType


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
def vec_belief(simple_env):
    np.random.seed(42)
    return create_rocksample_belief(
        simple_env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=100
    )


class TestBeliefUpdateCycle:
    def test_update_with_movement_preserves_shapes(self, vec_belief):
        """Test that belief update with movement action preserves shapes.

        Purpose: Validates end-to-end update cycle for movement actions.

        Given: A vectorized belief with 100 particles.
        When: Updated with East action and 'none' observation.
        Then: Particle count and dimensionality are preserved.

        Test type: integration
        """
        updated = vec_belief.update(action=2, observation="none")
        assert isinstance(updated, RockSampleVectorizedWeightedParticleBelief)
        assert updated.particles.shape == vec_belief.particles.shape
        assert updated.log_weights.shape == vec_belief.log_weights.shape

    def test_update_with_check_action(self, vec_belief):
        """Test belief update with check action and sensor observation.

        Purpose: Validates end-to-end update for non-trivial observations.

        Given: A vectorized belief with 100 particles.
        When: Updated with check rock 0 and 'good' observation.
        Then: Shapes are preserved and weights are updated.

        Test type: integration
        """
        updated = vec_belief.update(action=5, observation="good")
        assert isinstance(updated, RockSampleVectorizedWeightedParticleBelief)
        assert updated.particles.shape == vec_belief.particles.shape

    def test_multiple_chained_updates(self, vec_belief):
        """Test that multiple sequential updates work correctly.

        Purpose: Validates chained updates maintain correct subclass.

        Given: A vectorized belief.
        When: Three sequential updates are applied.
        Then: Each returns RockSampleVectorizedWeightedParticleBelief
            with correct shapes.

        Test type: integration
        """
        b = vec_belief
        b = b.update(action=5, observation="good")
        b = b.update(action=2, observation="none")
        b = b.update(action=6, observation="bad")
        assert isinstance(b, RockSampleVectorizedWeightedParticleBelief)
        assert b.particles.shape == vec_belief.particles.shape


class TestBeliefSampleValidState:
    def test_sample_returns_valid_array(self, vec_belief):
        """Test that sampling from belief returns a valid state array.

        Purpose: Validates sampled state is a proper numpy array.

        Given: A vectorized belief.
        When: A state is sampled.
        Then: Result is a 1-D numpy array of correct dimensionality.

        Test type: integration
        """
        state = vec_belief.sample()
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)  # 2 pos + 2 rocks

    def test_sampled_state_has_valid_values(self, vec_belief):
        """Test that sampled states have physically valid values.

        Purpose: Validates that sampled state represents a valid RockSample state.

        Given: A vectorized belief for a 5x5 grid.
        When: Multiple states are sampled.
        Then: Robot positions are within bounds and rock qualities are binary.

        Test type: integration
        """
        for _ in range(20):
            state = vec_belief.sample()
            # Robot position within grid (or terminal)
            row, col = int(state[0]), int(state[1])
            assert (row == -1 and col == -1) or (0 <= row < 5 and 0 <= col < 5)
            # Rock qualities are 0 or 1
            for rock_q in state[2:]:
                assert rock_q in (0.0, 1.0)


class TestStringObservationEncoding:
    def test_all_observation_strings_accepted(self, vec_belief):
        """Test that all valid string observations are accepted.

        Purpose: Validates string-to-integer encoding for all observations.

        Given: A vectorized belief.
        When: Updated with each valid observation string.
        Then: No errors are raised.

        Test type: integration
        """
        vec_belief.update(action=2, observation="none")
        vec_belief.update(action=5, observation="good")
        vec_belief.update(action=5, observation="bad")

    def test_integer_observation_also_works(self, vec_belief):
        """Test that integer-encoded observations are also accepted.

        Purpose: Validates that pre-encoded integer observations pass through.

        Given: A vectorized belief.
        When: Updated with integer observation (1 for 'good').
        Then: No errors are raised.

        Test type: integration
        """
        updated = vec_belief.update(action=5, observation=1)
        assert isinstance(updated, RockSampleVectorizedWeightedParticleBelief)


def _make_aligned_beliefs(env, n_particles=200):
    """Create baseline and vectorized beliefs with identical particles."""
    np.random.seed(42)
    states = env.initial_state_dist().sample(n_samples=n_particles)
    particles_list = list(states)
    particles_array = np.stack(states)
    log_weights = np.log(np.ones(n_particles) / n_particles)

    base = WeightedParticleBelief(
        particles=particles_list,
        log_weights=log_weights.copy(),
        resampling=False,
    )
    vec = RockSampleVectorizedWeightedParticleBelief(
        particles=particles_array,
        log_weights=log_weights.copy(),
        updater=RockSampleVectorizedUpdater.from_environment(env),
        resampling=False,
    )
    return base, vec


class TestVectorizedMatchesBaseline:
    """Verify that vectorized and baseline beliefs produce equivalent results."""

    def test_transition_particles_match_for_movement(self, simple_env):
        """Test that transitioned particles match for movement actions.

        Purpose: Validates that vectorized batch_transition produces the same
            next states as the loop-based state_transition_model.

        Given: Identical particles in both belief types.
        When: Both are updated with East action and 'none' observation.
        Then: The transitioned particles are identical.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        assert_update_particles_match(
            base=base, vec=vec, action=2, observation="none", pomdp=simple_env, atol=0.0
        )

    def test_transition_particles_match_for_sample_action(self, simple_env):
        """Test that transitioned particles match for sample action.

        Purpose: Validates vectorized Sample action produces identical state
            changes as the loop-based approach.

        Given: Identical particles in both belief types.
        When: Both are updated with Sample action and 'none' observation.
        Then: The transitioned particles are identical.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        assert_update_particles_match(
            base=base, vec=vec, action=0, observation="none", pomdp=simple_env, atol=0.0
        )

    def test_transition_particles_match_for_check_action(self, simple_env):
        """Test that transitioned particles match for check actions.

        Purpose: Validates vectorized Check action (no state change) matches
            the loop-based approach.

        Given: Identical particles in both belief types.
        When: Both are updated with Check rock 0 and 'good' observation.
        Then: The transitioned particles are identical.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        assert_update_particles_match(
            base=base, vec=vec, action=5, observation="good", pomdp=simple_env, atol=0.0
        )

    def test_normalized_weights_match_for_movement(self, simple_env):
        """Test that normalized weights match for movement actions.

        Purpose: Validates that both belief types produce identical weight
            distributions for deterministic 'none' observations.

        Given: Identical particles in both belief types.
        When: Both are updated with North action and 'none' observation.
        Then: Normalized weights are equal.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        assert_update_weights_match(
            base=base, vec=vec, action=1, observation="none", pomdp=simple_env, atol=1e-6
        )

    def test_normalized_weights_close_for_check_action(self, simple_env):
        """Test that normalized weights are close for check actions.

        Purpose: Validates that both belief types produce nearly identical
            weight distributions for sensor observations. Small differences
            are expected due to different numerical clamping (eps vs max).

        Given: Identical particles in both belief types.
        When: Both are updated with Check rock 0 and 'good' observation.
        Then: Normalized weights are close (atol=1e-6).

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        assert_update_weights_match(
            base=base, vec=vec, action=5, observation="good", pomdp=simple_env, atol=1e-6
        )

    def test_weights_match_after_multiple_updates(self, simple_env):
        """Test that weights remain close after a sequence of updates.

        Purpose: Validates that numerical differences don't accumulate
            over multiple belief update steps.

        Given: Identical particles in both belief types.
        When: Both are updated with a sequence of actions/observations.
        Then: Normalized weights remain close after all updates.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env, n_particles=500)
        steps = [
            (5, "good"),  # check rock 0
            (2, "none"),  # move east
            (6, "bad"),  # check rock 1
            (1, "none"),  # move north
            (5, "good"),  # check rock 0 again
        ]
        assert_chained_update_equivalence(
            base=base,
            vec=vec,
            steps=steps,
            pomdp=simple_env,
            atol_particles=0.0,
            atol_weights=1e-5,
        )

    def test_sample_distributions_match_post_update(self, simple_env):
        """Test sample() on both beliefs draws from normalized_weights post-update.

        Purpose: Validates that WeightedParticleBelief.sample() and
            RockSampleVectorizedWeightedParticleBelief.sample() are unbiased
            draws from their updated belief distributions after a non-trivial
            check action that re-weights particles, and that the two
            distributions agree.

        Given: 200 aligned particles in both beliefs, updated with Check
            rock 0 and 'good' observation.
        When: 20,000 samples are drawn from each belief.
        Then: The two empirical histograms agree in L-infinity, and each
            histogram agrees with its belief's normalized_weights. Duplicate
            particles (common in RockSample) are aggregated by particle
            identity before comparison.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(simple_env)
        base = base.update(action=5, observation="good", pomdp=simple_env)
        vec = vec.update(action=5, observation="good")

        assert_sample_distributions_match(
            base=base,
            vec=vec,
            n_samples=20_000,
            tol=0.02,
            atol_weights=0.02,
            seed=400,
        )
