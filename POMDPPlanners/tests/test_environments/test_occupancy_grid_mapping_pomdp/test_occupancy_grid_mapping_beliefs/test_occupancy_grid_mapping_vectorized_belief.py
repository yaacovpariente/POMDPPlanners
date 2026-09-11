# SPDX-License-Identifier: MIT

"""Tests for OccupancyGridMappingVectorizedBelief.

The vectorized filter promises the scalar filter's results, not merely the
same distribution, so every test here drives both from the same seed and
compares particles, weights, history and restart counts directly.
"""

import pickle

import numpy as np
import pytest

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridMappingBelief,
    OccupancyGridMappingPOMDP,
    OccupancyGridMappingVectorizedBelief,
    OccupancyGridMappingVectorizedUpdater,
    create_occupancy_grid_state,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    occupancy_grid_mapping_pinned_kwargs,
)


def make_env(**overrides):
    kwargs = occupancy_grid_mapping_pinned_kwargs(
        num_rows=8, num_cols=8, num_beams=12, start_row=4, start_col=4, **overrides
    )
    return OccupancyGridMappingPOMDP(discount_factor=0.95, **kwargs)


def one_beam_world():
    """The observation-contract world: one east-facing beam, empty 7x7 grid."""
    env = OccupancyGridMappingPOMDP(
        num_rows=7,
        num_cols=7,
        num_beams=1,
        field_of_view_degrees=1,
        max_range_cells=2,
        has_boundary_wall=False,
        num_obstacles=0,
    )
    empty = np.zeros((7, 7))
    hit = empty.copy()
    hit[3, 5] = 1
    return env, create_occupancy_grid_state(env, empty), create_occupancy_grid_state(env, hit)


def paired_filters(env, n_particles=20, seed=3):
    """The scalar and the vectorized filter over the same prior draw."""
    np.random.seed(seed)
    scalar = OccupancyGridMappingBelief.initial(env, n_particles=n_particles)
    np.random.seed(seed)
    vectorized = OccupancyGridMappingVectorizedBelief.initial(env, n_particles=n_particles)
    return scalar, vectorized


def assert_same_filter_state(scalar, vectorized):
    np.testing.assert_array_equal(vectorized.particles, np.asarray(scalar.particles))
    np.testing.assert_array_equal(vectorized.log_weights, scalar.log_weights)
    np.testing.assert_allclose(vectorized.normalized_weights, scalar.normalized_weights)
    assert vectorized.support_restarts == scalar.support_restarts
    assert len(vectorized.history) == len(scalar.history)
    for (action_a, obs_a), (action_b, obs_b) in zip(vectorized.history, scalar.history):
        assert action_a == action_b
        np.testing.assert_array_equal(obs_a, obs_b)
    assert vectorized.pre_resample_ess == pytest.approx(scalar.pre_resample_ess)


class TestConstruction:
    def test_initial_draws_the_same_prior_as_the_scalar_filter(self):
        """Purpose: both filters must start from the same maps under one seed.

        Test type: unit
        """
        scalar, vectorized = paired_filters(make_env())
        assert isinstance(vectorized, VectorizedWeightedParticleBelief)
        assert_same_filter_state(scalar, vectorized)
        assert vectorized.n_particles == 20
        assert vectorized.replay_proposals == 0
        assert vectorized.update_seconds == 0.0

    def test_resampling_is_refused(self):
        """Purpose: routine resampling would drop the low-weight motion
        hypotheses the filter exists to keep.

        Test type: unit
        """
        env = make_env()
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = np.asarray(env.initial_state_dist().sample(2))
        with pytest.raises(ValueError, match="resampling"):
            OccupancyGridMappingVectorizedBelief(
                particles, np.log([0.5, 0.5]), updater, resampling=True
            )

    def test_weights_need_finite_support(self):
        """Purpose: a filter with no live particle cannot be sampled from.

        Test type: unit
        """
        env = make_env()
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        particles = np.asarray(env.initial_state_dist().sample(2))
        for weights in ([-np.inf, -np.inf], [0.0, np.nan], [0.0, np.inf]):
            with pytest.raises(ValueError, match="finite support"):
                OccupancyGridMappingVectorizedBelief(particles, weights, updater)
        kept = OccupancyGridMappingVectorizedBelief(particles, [0.0, -np.inf], updater)
        np.testing.assert_array_equal(kept.log_weights, [0.0, -np.inf])
        np.testing.assert_array_equal(kept.normalized_weights, [1.0, 0.0])


class TestUpdate:
    @pytest.mark.parametrize("move_failure_probability", [0.0, 0.25])
    def test_chained_updates_match_the_scalar_filter(self, move_failure_probability):
        """Purpose: the vectorized filter is a drop-in replacement, so an
        episode's worth of conditioning must give identical particles and
        weights, not just the same distribution.

        Given: Both filters over the same 20 prior maps and a true world.
        When: Twelve observed steps are conditioned on, in the same order.
        Then: After every step, particles, weights, history and restart
            counts are identical.

        Test type: integration
        """
        env = make_env(move_failure_probability=move_failure_probability)
        scalar, vectorized = paired_filters(env)
        np.random.seed(17)
        true_state = env.initial_state_dist().sample()[0]
        for action in [0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 2]:
            true_state, observation, _ = env.sample_next_step(true_state, action)
            np.random.seed(action + 100)
            scalar = scalar.update(action, observation, env)
            np.random.seed(action + 100)
            vectorized = vectorized.update(action, observation, env)
            assert isinstance(vectorized, OccupancyGridMappingVectorizedBelief)
            assert_same_filter_state(scalar, vectorized)
        assert len(vectorized.history) == 12
        assert vectorized.update_seconds > 0.0

    def test_bayes_weights_on_two_maps(self):
        """Purpose: posterior odds equal prior odds times the Gaussian
        likelihood ratio, as in the scalar observation-contract test.

        Test type: unit
        """
        env, empty, hit = one_beam_world()
        env.true_map(hit)[3, 4] = 1
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        belief = OccupancyGridMappingVectorizedBelief([empty, hit], np.log([0.4, 0.6]), updater)
        observation = np.array([3, 3, 1, 1.2])
        result = belief.update(2, observation, env, state=hit)
        scores = np.array(
            [env.predictive_observation_log_probability(s, 2, observation) for s in [empty, hit]]
        ) + np.log([0.4, 0.6])
        weights = np.exp(scores - np.max(scores))
        weights /= weights.sum()
        np.testing.assert_allclose(result.normalized_weights, weights)
        np.testing.assert_array_equal(
            env.log_odds(result.particles[0]), env.log_odds(result.particles[1])
        )
        assert result.support_restarts == 0

    def test_replay_matches_the_scalar_filter(self):
        """Purpose: on zero support both filters draw the same fresh maps from
        the same stream and must resample the same survivors.

        Given: Two copies of a map that blocks the observed move.
        When: The impossible observation is conditioned on under one seed.
        Then: Both filters restart once, test the same number of proposals,
            and hold identical particles, all consistent with the move.

        Test type: unit
        """
        env, empty, _ = one_beam_world()
        blocked = empty.copy()
        env.true_map(blocked)[2, 3] = 1
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        np.random.seed(5)
        scalar = OccupancyGridMappingBelief([blocked, blocked.copy()], np.log([0.5, 0.5])).update(
            0, [2, 3, 0, 2], env
        )
        np.random.seed(5)
        vectorized = OccupancyGridMappingVectorizedBelief(
            [blocked, blocked.copy()], np.log([0.5, 0.5]), updater
        ).update(0, [2, 3, 0, 2], env)
        assert vectorized.support_restarts == 1
        assert vectorized.replay_proposals == scalar.replay_proposals == 512
        assert_same_filter_state(scalar, vectorized)
        assert all(env.true_map(p)[2, 3] == 0 for p in vectorized.particles)
        assert all(env.pose(p) == (2, 3, 0) for p in vectorized.particles)

    def test_replay_needs_the_environment(self):
        """Purpose: fresh prior maps can only come from the environment, so a
        collapse without one must fail loudly rather than return an empty filter.

        Test type: unit
        """
        env, empty, _ = one_beam_world()
        blocked = empty.copy()
        env.true_map(blocked)[2, 3] = 1
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        belief = OccupancyGridMappingVectorizedBelief([blocked], np.log([1.0]), updater)
        with pytest.raises(RuntimeError, match="no environment"):
            belief.update(0, [2, 3, 0, 2])
        # With support, the environment is not needed.
        assert belief.update(2, [3, 3, 1, 2]).support_restarts == 0


class TestSerialization:
    def test_to_dict_and_pickle_round_trip(self):
        """Purpose: restored filters must keep the history that prior replay needs.

        Given: A filter after one update that left one particle at ``-inf``.
        When: It is rebuilt from ``to_dict`` and from a pickle.
        Then: Identity, weights and history survive, and a further update works.

        Test type: unit
        """
        env, empty, hit = one_beam_world()
        env.true_map(hit)[2, 3] = 1
        updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
        belief = OccupancyGridMappingVectorizedBelief([empty, hit], np.log([0.5, 0.5]), updater)
        result = belief.update(0, [2, 3, 0, 2], env)
        assert result.log_weights[1] == -np.inf
        serialized = result.to_dict()
        assert "updater" not in serialized
        for restored in [
            OccupancyGridMappingVectorizedBelief(updater=updater, **serialized),
            pickle.loads(pickle.dumps(result)),
        ]:
            assert restored.config_id == result.config_id
            assert len(restored.history) == 1
            np.testing.assert_array_equal(restored.normalized_weights, result.normalized_weights)
            assert len(restored.update(2, [2, 3, 1, 2], env).history) == 2

    def test_config_id_changes_with_history_and_particles(self):
        """Purpose: two filters that would condition differently must not share a cache key.

        Test type: unit
        """
        env = make_env()
        _, belief = paired_filters(env, n_particles=4)
        _, same = paired_filters(env, n_particles=4)
        assert belief.config_id == same.config_id
        _, observation, _ = env.sample_next_step(belief.sample(), 0)
        assert belief.update(0, observation, env).config_id != belief.config_id
        _, other = paired_filters(env, n_particles=4, seed=4)
        assert other.config_id != belief.config_id
