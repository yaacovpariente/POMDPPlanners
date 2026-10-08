# SPDX-License-Identifier: MIT

"""Unit tests for the two fitting procedures.

The property under test is recovery: given data from a system the learner's model
class can represent, the fit must find it. A learner that cannot recover a linear
system from clean linear data will not be diagnosable on a real one, because
every failure there is attributable to the environment.

The ensemble's second property is that it reports uncertainty at all. A
deterministic transition collapses a particle belief to a point, and a planner
grading a tail measure then has nothing to grade -- so "does it produce spread"
is not cosmetic, it is the reason the ensemble exists.
"""

from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.training.model_learning import (
    LinearGaussianLearner,
    ProbabilisticEnsembleLearner,
    TransitionDataset,
    held_out_log_likelihood,
)

WEIGHT_STATE = np.array([[0.9, 0.1], [0.0, 0.95]])
WEIGHT_ACTION = np.array([[0.5], [0.2]])
BIAS = np.array([0.01, -0.02])
NOISE_STD = 0.05


def _linear_world(num_rows: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rollouts from a known linear-Gaussian system."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(num_rows, 2))
    actions = rng.normal(size=(num_rows, 1))
    next_states = (
        states @ WEIGHT_STATE.T
        + actions @ WEIGHT_ACTION.T
        + BIAS
        + rng.normal(scale=NOISE_STD, size=(num_rows, 2))
    )
    return states, actions, next_states


def _dataset(num_rows: int = 2000, holdout_fraction: float = 0.2, seed: int = 0):
    states, actions, next_states = _linear_world(num_rows, seed)
    dataset = TransitionDataset(holdout_fraction=holdout_fraction, seed=seed + 1)
    for start in range(0, num_rows, 50):
        dataset.add_episode(
            states[start : start + 50],
            actions[start : start + 50],
            next_states[start : start + 50],
            source="exploration",
        )
    return dataset


def test_linear_learner_recovers_the_system_it_was_generated_from() -> None:
    """Purpose: Validates that the linear fit recovers known coefficients and noise

    Given: 2000 transitions from a linear-Gaussian system with a known A, B, b and
        a noise standard deviation of 0.05
    When: The linear learner fits them
    Then: The recovered mean prediction matches the true one to within the noise,
        and the recovered noise scale is close to 0.05
    """
    dataset = _dataset(holdout_fraction=0.0)
    model = LinearGaussianLearner().fit(dataset)

    state = np.array([0.3, -0.7])
    action = np.array([0.4])
    expected = WEIGHT_STATE @ state + WEIGHT_ACTION @ action + BIAS

    draws = np.atleast_2d(model.sample_next_state(state, action, 4000))
    assert np.allclose(draws.mean(axis=0), expected, atol=5e-3)
    assert np.allclose(draws.std(axis=0), NOISE_STD, rtol=0.15)


def test_ensemble_learner_recovers_the_same_system() -> None:
    """Purpose: Validates that the ensemble fit finds a linear system too

    Given: The same 2000 transitions, and an ensemble of three small networks
    When: The ensemble fits them
    Then: Its mean prediction matches the true one to a looser tolerance than the
        closed-form fit, which is the price of the larger model class
    """
    dataset = _dataset(holdout_fraction=0.0)
    model = ProbabilisticEnsembleLearner(
        num_members=3, hidden_sizes=(64, 64), epochs=60, seed=0
    ).fit(dataset)

    state = np.array([0.3, -0.7])
    action = np.array([0.4])
    expected = WEIGHT_STATE @ state + WEIGHT_ACTION @ action + BIAS

    draws = np.atleast_2d(model.sample_next_state(state, action, 4000))
    assert np.allclose(draws.mean(axis=0), expected, atol=0.05)


def test_ensemble_reports_uncertainty_rather_than_a_point() -> None:
    """Purpose: Validates that the fitted ensemble produces spread, not a point mass

    Given: A fitted ensemble
    When: Many next states are drawn from one (state, action)
    Then: They have non-zero spread in every dimension, so a particle belief built
        from them does not collapse
    """
    dataset = _dataset(holdout_fraction=0.0)
    model = ProbabilisticEnsembleLearner(
        num_members=3, hidden_sizes=(64, 64), epochs=40, seed=0
    ).fit(dataset)

    draws = np.atleast_2d(model.sample_next_state(np.array([0.1, 0.2]), np.array([0.0]), 500))

    assert np.all(draws.std(axis=0) > 1e-3)


def test_ensemble_density_is_the_mixture_it_actually_samples_from() -> None:
    """Purpose: Validates that log_probability describes the sampling distribution

    Given: A fitted ensemble and 20000 draws from one (state, action)
    When: The empirical log-density near the sample mean is compared to the
        reported one
    Then: They agree to within Monte-Carlo error, so a particle filter weighting
        with this density is weighting correctly
    """
    dataset = _dataset(holdout_fraction=0.0)
    model = ProbabilisticEnsembleLearner(
        num_members=3, hidden_sizes=(64, 64), epochs=40, seed=0
    ).fit(dataset)
    state = np.array([0.1, 0.2])
    action = np.array([0.0])

    draws = np.atleast_2d(model.sample_next_state(state, action, 20000))
    centre = draws.mean(axis=0)
    reported = float(model.log_probability(state, action, centre[None, :])[0])

    # Empirical density near the centre: the fraction of draws inside a small box,
    # divided by the box's volume.
    half_width = 0.02
    inside = np.all(np.abs(draws - centre) < half_width, axis=1)
    empirical = inside.mean() / ((2 * half_width) ** draws.shape[1])
    assert np.isclose(reported, np.log(empirical), atol=0.4)


def test_held_out_likelihood_prefers_the_fit_that_saw_the_data() -> None:
    """Purpose: Validates that held-out likelihood separates a good fit from a bad one

    Given: A model fitted on the system, and one fitted on shuffled successors
    When: Both are scored on the same held-out transitions
    Then: The correctly fitted model scores higher
    """
    dataset = _dataset()
    good = LinearGaussianLearner().fit(dataset)

    shuffled = TransitionDataset(holdout_fraction=0.0, seed=1)
    states, actions, next_states = _linear_world(1000, seed=5)
    rng = np.random.default_rng(0)
    shuffled.add_episode(states, actions, rng.permutation(next_states), source="exploration")
    bad = LinearGaussianLearner().fit(shuffled)

    holdout = dataset.holdout_batch()
    assert held_out_log_likelihood(good, holdout) > held_out_log_likelihood(bad, holdout)


def test_fitting_too_few_rows_is_refused() -> None:
    """Purpose: Validates that an under-determined fit raises instead of returning noise

    Given: A dataset holding a single transition
    When: Either learner fits it
    Then: A ValueError says how many rows were available
    """
    dataset = TransitionDataset(holdout_fraction=0.0)
    dataset.add_episode(np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 2)), source="x")

    with pytest.raises(ValueError, match="at least 2 rows"):
        LinearGaussianLearner().fit(dataset)
    with pytest.raises(ValueError, match="at least 2 rows"):
        ProbabilisticEnsembleLearner(num_members=2, epochs=1).fit(dataset)


def test_an_ensemble_needs_at_least_one_member() -> None:
    """Purpose: Validates that a zero-member ensemble is rejected at construction

    Given: num_members of 0
    When: The learner is constructed
    Then: A ValueError is raised, rather than a fit that returns an empty mixture
    """
    with pytest.raises(ValueError, match="num_members"):
        ProbabilisticEnsembleLearner(num_members=0)
