# SPDX-License-Identifier: MIT
"""Regressions for observation-conditioned mapping and mixed sensor support."""

import numpy as np
import pytest
from scipy.special import log_ndtr  # pylint: disable=no-name-in-module
from scipy.stats import truncnorm

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridMappingPOMDP,
    OccupancyGridMappingBelief,
    RangeNoiseModel,
    create_occupancy_grid_state,
)


def scenario(noise=0.35, range_noise_model=RangeNoiseModel.GAUSSIAN):
    """One east-facing beam whose endpoint is two cells from the robot."""
    env = OccupancyGridMappingPOMDP(
        num_rows=7,
        num_cols=7,
        num_beams=1,
        field_of_view_degrees=1,
        max_range_cells=2,
        has_boundary_wall=False,
        num_obstacles=0,
        range_noise_std_cells=noise,
        range_noise_model=range_noise_model,
    )
    empty = np.zeros((7, 7))
    hit = empty.copy()
    hit[3, 5] = 1
    return env, create_occupancy_grid_state(env, empty), create_occupancy_grid_state(env, hit)


def test_max_range_hit_and_miss_share_observation_and_map_update():
    """The original counterexample must have one observable mapping result."""
    env, empty, hit = scenario()
    for measured in (1.3, 2.0, 2.61741832):
        observation = np.array([3, 3, 1, measured])
        a = env.state_from_observation(empty, observation)
        b = env.state_from_observation(hit, observation)
        np.testing.assert_array_equal(env.log_odds(a), env.log_odds(b))
        assert env.predictive_observation_log_probability(empty, 2, observation) == pytest.approx(
            env.predictive_observation_log_probability(hit, 2, observation)
        )
        if measured >= 2:
            assert env.log_odds(a)[3, 5] < 0
    np.random.seed(19)
    a, observation_a, _ = env.sample_next_step(empty, 2)
    np.random.seed(19)
    b, observation_b, _ = env.sample_next_step(hit, 2)
    np.testing.assert_array_equal(observation_a, observation_b)
    np.testing.assert_array_equal(env.log_odds(a), env.log_odds(b))


def test_noise_moves_evidence_and_never_uses_cells_behind_measured_hit():
    """Short readings stop evidence early even when the hidden world is empty."""
    env, state, _ = scenario()
    close = env.state_from_observation(state, [3, 3, 1, 1.1])
    far = env.state_from_observation(state, [3, 3, 1, 2.1])
    assert env.log_odds(close)[3, 4] > 0
    assert env.log_odds(close)[3, 5] == 0
    assert env.log_odds(far)[3, 4] < 0
    assert env.log_odds(far)[3, 5] < 0
    # Contrary repeated evidence revises rather than freezing the first label.
    revised = env.state_from_observation(close, [3, 3, 1, 2.1])
    assert env.log_odds(revised)[3, 4] == pytest.approx(0)


def test_gaussian_is_sampled_once_and_predictive_density_matches_support():
    """Continuous Gaussian noise belongs to the transition, not a second draw."""
    env, state, _ = scenario(noise=0.7)
    np.random.seed(4)
    samples = env.sample_next_state(state, 2, 3000)
    ranges = samples[:, env.scan_offset]
    assert ranges.mean() == pytest.approx(2, abs=0.04)
    assert ranges.std() == pytest.approx(0.7, abs=0.04)
    assert ranges.max() > 2.7
    assert ranges.min() < 0
    observation = env.sample_observation(samples[0], 2)
    np.testing.assert_array_equal(env.sample_observation(samples[0], 2, 3), [observation] * 3)
    assert env.observation_log_probability(samples[0], 2, observation)[0] == 0
    changed = observation.copy()
    changed[-1] += 0.01
    assert env.observation_log_probability(samples[0], 2, changed)[0] == -np.inf
    expected = -0.5 * ((observation[-1] - 2) / 0.7) ** 2 - np.log(0.7 * np.sqrt(2 * np.pi))
    assert env.predictive_observation_log_probability(state, 2, observation) == pytest.approx(
        expected
    )
    assert env.transition_log_probability(state, 2, samples[:1])[0] == pytest.approx(expected)
    malformed = samples[0].copy()
    malformed[env.log_odds_offset] += 1
    assert env.transition_log_probability(state, 2, [malformed])[0] == -np.inf


@pytest.mark.parametrize("offset", [0.1, -0.1, 0.49])
def test_fractional_pose_has_zero_support(offset):
    """Exact pose never admits a fractional offset, however small."""
    env, state, _ = scenario()
    successor = env.sample_next_state(state, 2)
    observation = env.sample_observation(successor, 2)
    observation[0] += offset
    assert env.observation_log_probability(successor, 2, observation)[0] == -np.inf
    assert env.predictive_observation_log_probability(state, 2, observation) == -np.inf
    with pytest.raises(ValueError, match="integer"):
        env.state_from_observation(state, observation)


def test_motion_failure_probability_is_in_predictive_weight():
    """Conditioning known pose integrates both move outcomes without sampling one."""
    env, state, _ = scenario()
    env.move_failure_probability = 0.25
    success = np.array([2, 3, 0, 2])
    failure = np.array([3, 3, 0, 2])
    ratio = np.exp(
        env.predictive_observation_log_probability(state, 0, success)
        - env.predictive_observation_log_probability(state, 0, failure)
    )
    assert ratio == pytest.approx(3)


def test_filter_conditions_all_maps_on_same_reading_and_keeps_bayes_weights():
    """Posterior odds equal prior odds times the Gaussian likelihood ratio."""
    env, empty, hit = scenario()
    env.true_map(hit)[3, 4] = 1
    belief = OccupancyGridMappingBelief([empty, hit], np.log([0.4, 0.6]))
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
    assert isinstance(result, OccupancyGridMappingBelief)
    assert result.support_restarts == 0


def test_filter_replays_prior_when_exact_pose_exhausts_support():
    """All-wrong motion hypotheses trigger explicit recovery, never an eps floor."""
    env, empty, _ = scenario()
    blocked = empty.copy()
    env.true_map(blocked)[2, 3] = 1
    belief = OccupancyGridMappingBelief([blocked, blocked.copy()], np.log([0.5, 0.5]))
    result = belief.update(0, [2, 3, 0, 2], env)
    assert result.support_restarts == 1
    assert all(env.true_map(p)[2, 3] == 0 for p in result.particles)
    assert all(env.pose(p) == (2, 3, 0) for p in result.particles)


def test_successful_translation_metric_counts_a_revisit():
    """The metric names motion honestly: out and back counts as two."""
    env, state, _ = scenario()
    total = 0
    for action in [0, 2, 2, 0]:
        successor = env.sample_next_state(state, action)
        total += env.step_info(state, action, successor)["successful_translation"]
        state = successor
    assert env.pose(state)[:2] == (3, 3)
    assert total == 2
    assert "average_new_cells_visited" not in env.get_metric_names()


def test_expected_reward_is_deterministic_noise_sensitive_and_bounded():
    """The numerical expectation uses observed inverse maps and no global draws."""
    env, state, _ = scenario()
    np.random.seed(9)
    expected_next_random = np.random.random()
    np.random.seed(9)
    reward = env.reward(state, 2)
    assert np.random.random() == expected_next_random
    assert env.reward(state, 2) == reward
    assert env.reward_range[0] <= reward <= env.reward_range[1]
    noisy, same_state, _ = scenario(noise=2)
    assert noisy.reward(same_state, 2) != pytest.approx(reward)


def test_filter_configuration_and_pickle_preserve_history_and_zero_weights():
    """Restored filters must retain the observations needed for prior replay."""
    import pickle

    env, empty, hit = scenario()
    env.true_map(hit)[2, 3] = 1
    belief = OccupancyGridMappingBelief([empty, hit], np.log([0.5, 0.5]))
    result = belief.update(0, [2, 3, 0, 2], env)
    assert result.log_weights[1] == -np.inf
    for restored in [
        OccupancyGridMappingBelief(**result.to_dict()),
        pickle.loads(pickle.dumps(result)),
    ]:
        assert restored.config_id == result.config_id
        assert len(restored.history) == 1
        np.testing.assert_array_equal(restored.normalized_weights, result.normalized_weights)
        updated = restored.update(2, [2, 3, 1, 2], env)
        assert len(updated.history) == 2


def test_realised_reward_uses_the_same_observation_as_reported_mapping():
    """The runner scores the observed change; planning has an expected fallback."""
    env, state, _ = scenario()
    assert env.reward_requires_next_state
    for measured in (1.1, 2.1):
        successor = env.state_from_observation(state, [3, 3, 1, measured])
        assert env.reward(state, 2, successor) == pytest.approx(
            env.entropy_bits(state) - env.entropy_bits(successor) - env.step_cost
        )
    successor, observation, reward = env.sample_next_step(state, 2)
    reconstructed = env.state_from_observation(state, observation)
    assert reward == pytest.approx(env.entropy_bits(state) - env.entropy_bits(reconstructed))


# -- truncated-normal range law ---------------------------------------------

TRUNCATED = RangeNoiseModel.TRUNCATED_NORMAL


def test_truncated_normal_is_sampled_once_nonnegative_and_scored_with_its_normaliser():
    """The truncated law follows the same one-draw contract with a proper density.

    With ``sigma = 1.5`` and a nominal range of 2 the Gaussian would put about
    nine percent of its mass below zero, so the truncation is doing real work.
    """
    env, state, _ = scenario(noise=1.5, range_noise_model=TRUNCATED)
    np.random.seed(4)
    samples = env.sample_next_state(state, 2, 4000)
    ranges = samples[:, env.scan_offset]
    reference = truncnorm(a=-2 / 1.5, b=np.inf, loc=2, scale=1.5)
    assert ranges.min() >= 0
    assert ranges.mean() == pytest.approx(reference.mean(), abs=0.06)
    assert ranges.std() == pytest.approx(reference.std(), abs=0.06)
    observation = env.sample_observation(samples[0], 2)
    # The stored scan and the revealed observation are the same numbers.
    np.testing.assert_array_equal(observation[3:], samples[0, env.scan_offset :])
    np.testing.assert_array_equal(env.sample_observation(samples[0], 2, 3), [observation] * 3)
    assert env.observation_log_probability(samples[0], 2, observation)[0] == 0
    expected = (
        -0.5 * ((observation[-1] - 2) / 1.5) ** 2
        - np.log(1.5 * np.sqrt(2 * np.pi))
        - log_ndtr(2 / 1.5)
    )
    assert env.predictive_observation_log_probability(state, 2, observation) == pytest.approx(
        expected
    )
    assert env.transition_log_probability(state, 2, samples[:1])[0] == pytest.approx(expected)
    np.testing.assert_array_equal(env.state_from_observation(state, observation), samples[0])


def test_a_negative_reading_has_no_support_under_the_truncated_law_only():
    """The truncated law scores an impossible reading ``-inf``; the Gaussian does not."""
    truncated, state, _ = scenario(noise=1.5, range_noise_model=TRUNCATED)
    gaussian, _, _ = scenario(noise=1.5)
    negative = np.array([3, 3, 1, -0.2])
    assert truncated.predictive_observation_log_probability(state, 2, negative) == -np.inf
    assert np.isfinite(gaussian.predictive_observation_log_probability(state, 2, negative))
    forged = truncated.state_from_observation(state, negative)
    assert truncated.transition_log_probability(state, 2, [forged])[0] == -np.inf
    assert np.isfinite(gaussian.transition_log_probability(state, 2, [forged])[0])
    assert truncated.predictive_observation_log_probability(
        state, 2, np.array([3, 3, 1, 0.0])
    ) > -np.inf


def test_truncated_filter_weights_carry_each_particle_normaliser():
    """Posterior odds include the per-particle ``Phi(rho / sigma)``, and the
    correction favours the nearer hypothesis relative to the Gaussian."""
    env, empty, hit = scenario(noise=1.5, range_noise_model=TRUNCATED)
    env.true_map(hit)[3, 4] = 1  # nominal 1 for ``hit``, 2 for ``empty``
    belief = OccupancyGridMappingBelief([empty, hit], np.log([0.5, 0.5]))
    observation = np.array([3, 3, 1, 1.5])
    result = belief.update(2, observation, env, state=hit)
    gaussian_ratio = np.exp(
        -0.5 * ((1.5 - 1) / 1.5) ** 2 + 0.5 * ((1.5 - 2) / 1.5) ** 2
    )
    truncated_ratio = gaussian_ratio * np.exp(log_ndtr(2 / 1.5) - log_ndtr(1 / 1.5))
    weights = result.normalized_weights
    assert weights[1] / weights[0] == pytest.approx(truncated_ratio)
    assert truncated_ratio > gaussian_ratio
    scores = np.array(
        [env.predictive_observation_log_probability(s, 2, observation) for s in [empty, hit]]
    )
    assert scores[1] - scores[0] == pytest.approx(np.log(truncated_ratio))
    np.testing.assert_array_equal(
        env.log_odds(result.particles[0]), env.log_odds(result.particles[1])
    )


def test_truncated_expected_reward_is_deterministic_and_matches_gaussian_far_from_zero():
    """The quadrature represents the selected law without touching the RNG."""
    truncated, state, _ = scenario(noise=1.5, range_noise_model=TRUNCATED)
    gaussian, same_state, _ = scenario(noise=1.5)
    np.random.seed(9)
    expected_next_random = np.random.random()
    np.random.seed(9)
    reward = truncated.reward(state, 2)
    assert np.random.random() == expected_next_random
    assert truncated.reward(state, 2) == reward
    assert truncated.reward_range[0] <= reward <= truncated.reward_range[1]
    assert reward != pytest.approx(gaussian.reward(same_state, 2))
    # At the default noise the beam sits about six deviations above zero and
    # the two laws integrate over the same scans.
    narrow_truncated, narrow_state, _ = scenario(range_noise_model=TRUNCATED)
    narrow_gaussian, _, _ = scenario()
    assert narrow_truncated.reward(narrow_state, 2) == pytest.approx(
        narrow_gaussian.reward(narrow_state, 2), abs=1e-6
    )
