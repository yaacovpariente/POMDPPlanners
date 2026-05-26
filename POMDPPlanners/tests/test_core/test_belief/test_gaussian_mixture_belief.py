# SPDX-License-Identifier: MIT

"""Tests for GaussianMixtureBelief implementation.

This module tests the GaussianMixtureBelief class, covering:
- Construction with valid and invalid inputs
- Sampling correctness
- Update immutability and updater preservation
- inplace_update raising NotImplementedError
- config_id determinism and equality semantics
- Entropy MC estimation
- is_terminal_belief integration
- Cost dispatch integration
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    GaussianBelief,
    GaussianBeliefUpdater,
    GaussianMixtureBelief,
    GaussianMixtureBeliefUpdater,
    is_terminal_belief,
)
from POMDPPlanners.core.cost import (
    belief_expectation_cost,
    belief_expectation_cost_entropy_penalty,
    belief_expectation_cost_belief_information_gain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoopGMMUpdater(GaussianMixtureBeliefUpdater):
    """Test-only updater that shrinks covariances by 0.9."""

    def update(self, means, covariances, weights, action, observation):
        return means, [c * 0.9 for c in covariances], weights

    @property
    def config_id(self) -> str:
        return "noop_gmm_updater"


class _SplittingGMMUpdater(GaussianMixtureBeliefUpdater):
    """Test-only updater that splits the first component into two."""

    def update(self, means, covariances, weights, action, observation):
        return (
            [means[0], means[0] + 1.0],
            [covariances[0], covariances[0]],
            np.array([0.5, 0.5]),
        )

    @property
    def config_id(self) -> str:
        return "splitting_gmm_updater"


_noop_updater = _NoopGMMUpdater()


def _make_1comp_belief(**kwargs: Any) -> GaussianMixtureBelief:
    return GaussianMixtureBelief(
        means=kwargs.get("means", [np.array([0.0])]),
        covariances=kwargs.get("covariances", [np.array([[1.0]])]),
        weights=kwargs.get("weights", np.array([1.0])),
        updater=kwargs.get("updater", _noop_updater),
        **{k: v for k, v in kwargs.items() if k == "n_terminal_check_samples"},
    )


def _make_2comp_2d_belief(**kwargs: Any) -> GaussianMixtureBelief:
    return GaussianMixtureBelief(
        means=kwargs.get("means", [np.array([0.0, 0.0]), np.array([5.0, 5.0])]),
        covariances=kwargs.get("covariances", [np.eye(2), np.eye(2)]),
        weights=kwargs.get("weights", np.array([0.5, 0.5])),
        updater=kwargs.get("updater", _noop_updater),
        **{k: v for k, v in kwargs.items() if k == "n_terminal_check_samples"},
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefConstruction:
    def test_valid_single_component(self):
        """Test constructing a valid single-component GMM.

        Purpose: Validates basic single-component construction.

        Given: One mean vector, one covariance matrix, weight = [1.0].
        When: GaussianMixtureBelief is constructed.
        Then: n_components=1, dim=1, weights sum to 1.

        Test type: unit
        """
        belief = _make_1comp_belief()
        assert belief.n_components == 1
        assert belief.dim == 1
        np.testing.assert_allclose(belief.weights.sum(), 1.0)

    def test_valid_two_component_2d(self):
        """Test constructing a valid two-component 2D GMM.

        Purpose: Validates multi-component construction.

        Given: Two 2D mean vectors, two 2x2 covariance matrices, equal weights.
        When: GaussianMixtureBelief is constructed.
        Then: n_components=2, dim=2.

        Test type: unit
        """
        belief = _make_2comp_2d_belief()
        assert belief.n_components == 2
        assert belief.dim == 2

    def test_invalid_empty_components(self):
        """Test that zero components raises ValueError.

        Purpose: Validates input validation for empty component list.

        Given: Empty means list.
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="at least one"):
            GaussianMixtureBelief(
                means=[],
                covariances=[],
                weights=np.array([]),
                updater=_noop_updater,
            )

    def test_invalid_mismatched_counts(self):
        """Test that mismatched means/covariances counts raise ValueError.

        Purpose: Validates consistency check between means and covariances length.

        Given: 2 means but 1 covariance.
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not match"):
            GaussianMixtureBelief(
                means=[np.array([0.0]), np.array([1.0])],
                covariances=[np.array([[1.0]])],
                weights=np.array([0.5, 0.5]),
                updater=_noop_updater,
            )

    def test_invalid_weights_not_sum_to_one(self):
        """Test that weights not summing to 1 raises ValueError.

        Purpose: Validates weight normalization check.

        Given: Weights that sum to 0.8.
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="sum to 1"):
            GaussianMixtureBelief(
                means=[np.array([0.0]), np.array([1.0])],
                covariances=[np.array([[1.0]]), np.array([[1.0]])],
                weights=np.array([0.3, 0.5]),
                updater=_noop_updater,
            )

    def test_invalid_negative_weights(self):
        """Test that negative weights raise ValueError.

        Purpose: Validates non-negativity constraint on weights.

        Given: Weights with a negative value.
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="non-negative"):
            GaussianMixtureBelief(
                means=[np.array([0.0]), np.array([1.0])],
                covariances=[np.array([[1.0]]), np.array([[1.0]])],
                weights=np.array([-0.5, 1.5]),
                updater=_noop_updater,
            )

    def test_invalid_dimension_mismatch_across_components(self):
        """Test that mismatched dimensions across components raise ValueError.

        Purpose: Validates that all means must have the same dimension.

        Given: First mean is 1D, second mean is 2D.
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError):
            GaussianMixtureBelief(
                means=[np.array([0.0]), np.array([1.0, 2.0])],
                covariances=[np.array([[1.0]]), np.array([[1.0]])],
                weights=np.array([0.5, 0.5]),
                updater=_noop_updater,
            )

    def test_invalid_non_square_covariance(self):
        """Test that non-square covariance raises ValueError.

        Purpose: Validates covariance shape checking.

        Given: A covariance with shape (1, 2).
        When: GaussianMixtureBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError):
            GaussianMixtureBelief(
                means=[np.array([0.0])],
                covariances=[np.ones((1, 2))],
                weights=np.array([1.0]),
                updater=_noop_updater,
            )


# ---------------------------------------------------------------------------
# Sample tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefSample:
    def test_sample_shape_1d(self):
        """Test that sampling from 1D single-component GMM returns correct shape.

        Purpose: Validates sample output dimensionality.

        Given: A 1D single-component GaussianMixtureBelief.
        When: sample() is called.
        Then: Result is a 1D numpy array of length 1.

        Test type: unit
        """
        belief = _make_1comp_belief()
        s = belief.sample()
        assert isinstance(s, np.ndarray)
        assert s.shape == (1,)

    def test_sample_shape_2d(self):
        """Test that sampling from 2D GMM returns correct shape.

        Purpose: Validates sample output dimensionality for 2D case.

        Given: A 2D two-component GaussianMixtureBelief.
        When: sample() is called.
        Then: Result is a 1D numpy array of length 2.

        Test type: unit
        """
        belief = _make_2comp_2d_belief()
        s = belief.sample()
        assert isinstance(s, np.ndarray)
        assert s.shape == (2,)

    def test_sample_statistical_consistency(self):
        """Test that samples are consistent with the mixture distribution.

        Purpose: Validates that Monte Carlo samples reflect mixture component means.

        Given: A 1D two-component GMM with means at -10 and +10, small variance,
            and equal weights.
        When: A large number of samples are drawn.
        Then: Approximately half fall near -10 and half near +10.

        Test type: unit
        """
        np.random.seed(42)
        belief = GaussianMixtureBelief(
            means=[np.array([-10.0]), np.array([10.0])],
            covariances=[np.array([[0.01]]), np.array([[0.01]])],
            weights=np.array([0.5, 0.5]),
            updater=_noop_updater,
        )
        samples = np.array([belief.sample() for _ in range(2000)])
        near_neg = np.sum(samples[:, 0] < 0)
        near_pos = np.sum(samples[:, 0] > 0)
        assert 800 < near_neg < 1200
        assert 800 < near_pos < 1200


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefUpdate:
    def test_update_returns_new_belief(self):
        """Test that update() returns a new GaussianMixtureBelief instance.

        Purpose: Validates immutable update semantics.

        Given: A single-component GaussianMixtureBelief.
        When: update() is called.
        Then: A new GaussianMixtureBelief is returned, distinct from the original.

        Test type: unit
        """
        belief = _make_1comp_belief()
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert isinstance(new_belief, GaussianMixtureBelief)
        assert new_belief is not belief

    def test_update_uses_updater(self):
        """Test that update() delegates to the provided updater callable.

        Purpose: Validates correct covariance shrinkage after update.

        Given: A single-component GMM with noop updater (shrinks cov by 0.9).
        When: update() is called.
        Then: New covariance is 0.9 * original.

        Test type: unit
        """
        belief = _make_1comp_belief()
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        np.testing.assert_allclose(new_belief.covariances[0], [[0.9]])

    def test_update_preserves_updater(self):
        """Test that update() preserves the updater callable.

        Purpose: Validates that the same updater is passed to the child belief.

        Given: A GaussianMixtureBelief with a specific updater.
        When: update() is called.
        Then: new_belief.updater is the same object.

        Test type: unit
        """
        belief = _make_1comp_belief()
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert new_belief.updater is belief.updater

    def test_update_does_not_mutate_original(self):
        """Test that update() does not modify the original belief.

        Purpose: Validates immutability of original belief after update.

        Given: A single-component GMM.
        When: update() is called.
        Then: Original covariance is unchanged.

        Test type: unit
        """
        belief = _make_1comp_belief()
        original_cov = belief.covariances[0].copy()
        belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        np.testing.assert_array_equal(belief.covariances[0], original_cov)

    def test_update_can_change_component_count(self):
        """Test that updater can change the number of components.

        Purpose: Validates that the updater can return a different number of components.

        Given: A 1-component GMM with an updater that splits into 2 components.
        When: update() is called.
        Then: The new belief has 2 components.

        Test type: unit
        """
        belief = GaussianMixtureBelief(
            means=[np.array([0.0])],
            covariances=[np.array([[1.0]])],
            weights=np.array([1.0]),
            updater=_SplittingGMMUpdater(),
        )
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert new_belief.n_components == 2

    def test_update_preserves_n_terminal_check_samples(self):
        """Test that update() preserves n_terminal_check_samples.

        Purpose: Validates that terminal check sample count carries over.

        Given: A GMM with n_terminal_check_samples=100.
        When: update() is called.
        Then: New belief has n_terminal_check_samples=100.

        Test type: unit
        """
        belief = _make_1comp_belief(n_terminal_check_samples=100)
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert new_belief.n_terminal_check_samples == 100


# ---------------------------------------------------------------------------
# inplace_update test
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefInplaceUpdate:
    def test_inplace_update_raises(self):
        """Test that inplace_update() raises NotImplementedError.

        Purpose: Validates that GMM beliefs do not support incremental accumulation.

        Given: A GaussianMixtureBelief.
        When: inplace_update() is called.
        Then: NotImplementedError is raised.

        Test type: unit
        """
        belief = _make_1comp_belief()
        with pytest.raises(NotImplementedError):
            belief.inplace_update(action=0, observation=np.array([1.0]), pomdp=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# config_id tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefConfigId:
    def test_config_id_deterministic(self):
        """Test that config_id is deterministic for identical beliefs.

        Purpose: Validates that the same parameters produce the same config_id.

        Given: Two GaussianMixtureBelief instances with identical parameters.
        When: config_id is computed for both.
        Then: Both config_ids are equal.

        Test type: unit
        """
        b1 = _make_1comp_belief()
        b2 = _make_1comp_belief()
        assert b1.config_id == b2.config_id

    def test_config_id_differs_for_different_means(self):
        """Test that config_id differs when means differ.

        Purpose: Validates config_id sensitivity to mean values.

        Given: Two single-component GMMs with different means.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        b1 = _make_1comp_belief(means=[np.array([0.0])])
        b2 = _make_1comp_belief(means=[np.array([1.0])])
        assert b1.config_id != b2.config_id

    def test_config_id_differs_for_different_weights(self):
        """Test that config_id differs when weights differ.

        Purpose: Validates config_id sensitivity to weights.

        Given: Two two-component GMMs with different weight distributions.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        b1 = _make_2comp_2d_belief(weights=np.array([0.3, 0.7]))
        b2 = _make_2comp_2d_belief(weights=np.array([0.5, 0.5]))
        assert b1.config_id != b2.config_id

    def test_hash_and_equality(self):
        """Test that equal beliefs have equal hashes and are equal.

        Purpose: Validates __hash__ and __eq__ based on config_id.

        Given: Two identical GaussianMixtureBelief instances.
        When: hash() and == are compared.
        Then: Hashes match and beliefs are equal.

        Test type: unit
        """
        b1 = _make_1comp_belief()
        b2 = _make_1comp_belief()
        assert b1 == b2
        assert hash(b1) == hash(b2)


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefEntropy:
    def test_single_component_matches_gaussian_belief(self):
        """Test that single-component GMM entropy matches GaussianBelief entropy.

        Purpose: Validates that a single-component GMM has the same entropy
            as an equivalent GaussianBelief.

        Given: A single-component 2D GMM and an equivalent GaussianBelief.
        When: entropy() is called on both.
        Then: Entropies match within MC tolerance.

        Test type: unit
        """
        np.random.seed(42)
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        class _DummyUpdater(GaussianBeliefUpdater):
            def update(self, mean, covariance, action, observation):
                return np.asarray(observation, dtype=float).ravel(), covariance

            @property
            def config_id(self) -> str:
                return "dummy"

        gaussian_belief = GaussianBelief(mean=mean, covariance=cov, updater=_DummyUpdater())
        gmm_belief = GaussianMixtureBelief(
            means=[mean],
            covariances=[cov],
            weights=np.array([1.0]),
            updater=_noop_updater,
        )
        gaussian_entropy = gaussian_belief.entropy()
        gmm_entropy = gmm_belief.entropy(n_samples=5000)

        np.testing.assert_allclose(gmm_entropy, gaussian_entropy, atol=0.15)

    def test_mixture_entropy_greater_than_single_component(self):
        """Test that a mixture has higher entropy than its individual components.

        Purpose: Validates that mixing increases entropy.

        Given: A two-component 1D GMM with well-separated means, and a single
            component GMM with the same covariance.
        When: entropy() is called on both.
        Then: The mixture has higher entropy than the single component.

        Test type: unit
        """
        np.random.seed(42)
        single = GaussianMixtureBelief(
            means=[np.array([0.0])],
            covariances=[np.array([[1.0]])],
            weights=np.array([1.0]),
            updater=_noop_updater,
        )
        mixture = GaussianMixtureBelief(
            means=[np.array([-5.0]), np.array([5.0])],
            covariances=[np.array([[1.0]]), np.array([[1.0]])],
            weights=np.array([0.5, 0.5]),
            updater=_noop_updater,
        )
        assert mixture.entropy(n_samples=3000) > single.entropy(n_samples=3000)

    def test_entropy_is_finite(self):
        """Test that entropy returns a finite float.

        Purpose: Validates numerical stability of MC entropy estimate.

        Given: A 2D two-component GMM.
        When: entropy() is called.
        Then: Result is a finite float.

        Test type: unit
        """
        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        h = belief.entropy()
        assert isinstance(h, float)
        assert np.isfinite(h)


# ---------------------------------------------------------------------------
# is_terminal_belief tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefTerminal:
    def test_is_terminal_belief_non_terminal(self):
        """Test is_terminal_belief returns False for non-terminal GMM belief.

        Purpose: Validates terminal check integration with GaussianMixtureBelief.

        Given: A GMM belief and a mock environment where is_terminal returns False.
        When: is_terminal_belief is called.
        Then: Returns False.

        Test type: integration
        """

        class _MockEnv:
            def is_terminal(self, state):
                return False

        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        assert is_terminal_belief(belief, _MockEnv()) is False  # type: ignore[arg-type]

    def test_is_terminal_belief_terminal(self):
        """Test is_terminal_belief returns True when all samples are terminal.

        Purpose: Validates terminal check returns True when environment says terminal.

        Given: A GMM belief and a mock environment where is_terminal returns True.
        When: is_terminal_belief is called.
        Then: Returns True.

        Test type: integration
        """

        class _MockEnv:
            def is_terminal(self, state):
                return True

        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        assert is_terminal_belief(belief, _MockEnv()) is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cost dispatch tests
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefCostDispatch:
    def _make_mock_env(self):
        class _MockEnv:
            def reward(self, state, action, next_state=None):
                del action, next_state
                return float(np.sum(state))

            def reward_batch(self, states, action, next_states=None):
                del next_states
                return np.array([self.reward(states[i], action) for i in range(len(states))])

        return _MockEnv()

    def test_belief_expectation_cost(self):
        """Test belief_expectation_cost dispatches for GaussianMixtureBelief.

        Purpose: Validates cost dispatch returns a finite float.

        Given: A 2D GMM belief and a mock environment with sum-reward.
        When: belief_expectation_cost is called.
        Then: Returns a finite float.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        env = self._make_mock_env()
        cost = belief_expectation_cost(belief=belief, action=0, env=env)  # type: ignore[arg-type]
        assert isinstance(cost, float)
        assert np.isfinite(cost)

    def test_belief_expectation_cost_entropy_penalty(self):
        """Test entropy-penalized cost dispatches for GaussianMixtureBelief.

        Purpose: Validates entropy-penalized cost dispatch.

        Given: A 2D GMM belief and a mock environment.
        When: belief_expectation_cost_entropy_penalty is called with entropy_weight=1.0.
        Then: Returns a finite float, and includes entropy contribution.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        env = self._make_mock_env()
        cost_no = belief_expectation_cost_entropy_penalty(
            belief=belief, action=0, env=env, entropy_weight=0.0  # type: ignore[arg-type]
        )
        cost_yes = belief_expectation_cost_entropy_penalty(
            belief=belief, action=0, env=env, entropy_weight=1.0  # type: ignore[arg-type]
        )
        assert isinstance(cost_yes, float)
        assert np.isfinite(cost_yes)
        assert cost_yes != cost_no

    def test_belief_expectation_cost_information_gain(self):
        """Test information gain cost dispatches for GaussianMixtureBelief.

        Purpose: Validates information gain cost dispatch.

        Given: Two GMM beliefs (original and tighter) and a mock environment.
        When: belief_expectation_cost_belief_information_gain is called.
        Then: Returns a finite float.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2comp_2d_belief()
        next_belief = _make_2comp_2d_belief(covariances=[0.5 * np.eye(2), 0.5 * np.eye(2)])
        env = self._make_mock_env()
        cost = belief_expectation_cost_belief_information_gain(
            belief=belief,
            action=0,
            next_belief=next_belief,
            env=env,  # type: ignore[arg-type]
            entropy_weight=1.0,
        )
        assert isinstance(cost, float)
        assert np.isfinite(cost)
