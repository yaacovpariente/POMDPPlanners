"""Tests for GaussianBelief implementation.

This module tests the GaussianBelief class, covering:
- Construction with valid and invalid inputs
- Sampling correctness
- Update immutability and updater preservation
- inplace_update raising NotImplementedError
- config_id determinism and equality semantics
- Entropy closed-form computation
- is_terminal_belief integration
- Cost dispatch integration
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import GaussianBelief, is_terminal_belief
from POMDPPlanners.core.cost import (
    belief_expectation_cost,
    belief_expectation_cost_entropy_penalty,
    belief_expectation_cost_belief_information_gain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_updater(mean, cov, action, observation, pomdp):
    return observation, cov * 0.9


def _make_1d_belief(**kwargs: Any) -> GaussianBelief:
    gb_kwargs: dict[str, Any] = {
        "mean": kwargs.get("mean", np.array([0.0])),
        "covariance": kwargs.get("covariance", np.array([[1.0]])),
        "updater": kwargs.get("updater", _identity_updater),
    }
    if "n_terminal_check_samples" in kwargs:
        gb_kwargs["n_terminal_check_samples"] = kwargs["n_terminal_check_samples"]
    return GaussianBelief(**gb_kwargs)


def _make_2d_belief(**kwargs: Any) -> GaussianBelief:
    gb_kwargs: dict[str, Any] = {
        "mean": kwargs.get("mean", np.array([0.0, 0.0])),
        "covariance": kwargs.get("covariance", np.eye(2)),
        "updater": kwargs.get("updater", _identity_updater),
    }
    if "n_terminal_check_samples" in kwargs:
        gb_kwargs["n_terminal_check_samples"] = kwargs["n_terminal_check_samples"]
    return GaussianBelief(**gb_kwargs)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefConstruction:
    def test_valid_1d(self):
        """Test constructing a valid 1D Gaussian belief.

        Purpose: Validates basic 1D construction stores mean and covariance.

        Given: A scalar mean and 1x1 covariance matrix.
        When: GaussianBelief is constructed.
        Then: mean, covariance, dim are set correctly.

        Test type: unit
        """
        belief = _make_1d_belief()
        assert belief.dim == 1
        np.testing.assert_array_equal(belief.mean, [0.0])
        np.testing.assert_array_equal(belief.covariance, [[1.0]])

    def test_valid_2d(self):
        """Test constructing a valid 2D Gaussian belief.

        Purpose: Validates 2D construction with identity covariance.

        Given: A 2D mean vector and 2x2 identity covariance.
        When: GaussianBelief is constructed.
        Then: dim is 2, mean and covariance match inputs.

        Test type: unit
        """
        belief = _make_2d_belief()
        assert belief.dim == 2
        np.testing.assert_array_equal(belief.mean, [0.0, 0.0])

    def test_invalid_mean_not_1d(self):
        """Test that a 2D mean array raises ValueError.

        Purpose: Validates input validation for mean shape.

        Given: A 2D mean array (matrix instead of vector).
        When: GaussianBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="1D"):
            GaussianBelief(
                mean=np.array([[0.0, 0.0]]),
                covariance=np.eye(2),
                updater=_identity_updater,
            )

    def test_invalid_covariance_not_2d(self):
        """Test that a 1D covariance raises ValueError.

        Purpose: Validates input validation for covariance shape.

        Given: A 1D array as covariance.
        When: GaussianBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="2D"):
            GaussianBelief(
                mean=np.array([0.0]),
                covariance=np.array([1.0]),
                updater=_identity_updater,
            )

    def test_invalid_dimension_mismatch(self):
        """Test that mismatched mean/covariance dimensions raise ValueError.

        Purpose: Validates dimension consistency check.

        Given: A 1D mean and 2x2 covariance.
        When: GaussianBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="does not match"):
            GaussianBelief(
                mean=np.array([0.0]),
                covariance=np.eye(2),
                updater=_identity_updater,
            )

    def test_invalid_non_positive_definite(self):
        """Test that a non-positive-definite covariance raises an error.

        Purpose: Validates that CovarianceParameterizedMultivariateNormal rejects invalid matrices.

        Given: A covariance matrix that is not positive definite.
        When: GaussianBelief is constructed.
        Then: An error is raised (np.linalg.LinAlgError or ValueError).

        Test type: unit
        """
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            GaussianBelief(
                mean=np.array([0.0, 0.0]),
                covariance=np.array([[-1.0, 0.0], [0.0, 1.0]]),
                updater=_identity_updater,
            )

    def test_non_square_covariance(self):
        """Test that a non-square covariance raises ValueError.

        Purpose: Validates covariance must be square.

        Given: A 2x3 matrix as covariance.
        When: GaussianBelief is constructed.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="square"):
            GaussianBelief(
                mean=np.array([0.0, 0.0]),
                covariance=np.ones((2, 3)),
                updater=_identity_updater,
            )


# ---------------------------------------------------------------------------
# Sample tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefSample:
    def test_sample_shape_1d(self):
        """Test that sampling from 1D belief returns correct shape.

        Purpose: Validates sample output dimensionality for 1D case.

        Given: A 1D GaussianBelief.
        When: sample() is called.
        Then: Result is a 1D numpy array of length 1.

        Test type: unit
        """
        belief = _make_1d_belief()
        s = belief.sample()
        assert isinstance(s, np.ndarray)
        assert s.shape == (1,)

    def test_sample_shape_2d(self):
        """Test that sampling from 2D belief returns correct shape.

        Purpose: Validates sample output dimensionality for 2D case.

        Given: A 2D GaussianBelief.
        When: sample() is called.
        Then: Result is a 1D numpy array of length 2.

        Test type: unit
        """
        belief = _make_2d_belief()
        s = belief.sample()
        assert isinstance(s, np.ndarray)
        assert s.shape == (2,)

    def test_sample_statistical_consistency(self):
        """Test that samples are statistically consistent with the mean and covariance.

        Purpose: Validates Monte Carlo sample statistics converge to belief parameters.

        Given: A 2D GaussianBelief with known mean [3, -2] and covariance.
        When: A large number of samples are drawn.
        Then: Sample mean is close to belief mean and sample covariance is close
            to belief covariance (within Monte Carlo tolerance).

        Test type: unit
        """
        np.random.seed(42)
        mean = np.array([3.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        belief = GaussianBelief(mean=mean, covariance=cov, updater=_identity_updater)

        samples = np.array([belief.sample() for _ in range(5000)])
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.15)
        np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.25)


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefUpdate:
    def test_update_returns_new_belief(self):
        """Test that update() returns a new GaussianBelief instance.

        Purpose: Validates immutable update semantics.

        Given: A 1D GaussianBelief.
        When: update() is called.
        Then: A new GaussianBelief is returned, distinct from the original.

        Test type: unit
        """
        belief = _make_1d_belief()
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert isinstance(new_belief, GaussianBelief)
        assert new_belief is not belief

    def test_update_uses_updater(self):
        """Test that update() delegates to the provided updater callable.

        Purpose: Validates correct mean/covariance after update.

        Given: A 1D GaussianBelief with identity updater.
        When: update() is called with observation [5.0].
        Then: New mean equals [5.0] and covariance is 0.9 * original.

        Test type: unit
        """
        belief = _make_1d_belief()
        obs = np.array([5.0])
        new_belief = belief.update(action=0, observation=obs, pomdp=None)
        np.testing.assert_array_equal(new_belief.mean, obs)
        np.testing.assert_allclose(new_belief.covariance, [[0.9]])

    def test_update_preserves_updater(self):
        """Test that update() preserves the updater callable in the new belief.

        Purpose: Validates that the same updater is passed to the child belief.

        Given: A GaussianBelief with a specific updater.
        When: update() is called.
        Then: new_belief.updater is the same object as the original.

        Test type: unit
        """
        belief = _make_1d_belief()
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert new_belief.updater is belief.updater

    def test_update_preserves_n_terminal_check_samples(self):
        """Test that update() preserves the n_terminal_check_samples setting.

        Purpose: Validates that terminal check sample count carries over.

        Given: A GaussianBelief with n_terminal_check_samples=100.
        When: update() is called.
        Then: new_belief.n_terminal_check_samples equals 100.

        Test type: unit
        """
        belief = _make_1d_belief(n_terminal_check_samples=100)
        new_belief = belief.update(action=0, observation=np.array([1.0]), pomdp=None)
        assert new_belief.n_terminal_check_samples == 100

    def test_update_does_not_mutate_original(self):
        """Test that update() does not modify the original belief.

        Purpose: Validates immutability of the original belief after update.

        Given: A 1D GaussianBelief with mean [0.0].
        When: update() is called with observation [5.0].
        Then: Original belief mean is still [0.0].

        Test type: unit
        """
        belief = _make_1d_belief()
        original_mean = belief.mean.copy()
        belief.update(action=0, observation=np.array([5.0]), pomdp=None)
        np.testing.assert_array_equal(belief.mean, original_mean)


# ---------------------------------------------------------------------------
# inplace_update test
# ---------------------------------------------------------------------------


class TestGaussianBeliefInplaceUpdate:
    def test_inplace_update_raises(self):
        """Test that inplace_update() raises NotImplementedError.

        Purpose: Validates that Gaussian beliefs do not support incremental accumulation.

        Given: A GaussianBelief.
        When: inplace_update() is called.
        Then: NotImplementedError is raised.

        Test type: unit
        """
        belief = _make_1d_belief()
        with pytest.raises(NotImplementedError):
            belief.inplace_update(action=0, observation=np.array([1.0]), pomdp=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# config_id tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefConfigId:
    def test_config_id_deterministic(self):
        """Test that config_id is deterministic for identical beliefs.

        Purpose: Validates that the same parameters produce the same config_id.

        Given: Two GaussianBelief instances with identical parameters.
        When: config_id is computed for both.
        Then: Both config_ids are equal.

        Test type: unit
        """
        b1 = _make_1d_belief()
        b2 = _make_1d_belief()
        assert b1.config_id == b2.config_id

    def test_config_id_differs_for_different_mean(self):
        """Test that config_id differs when means differ.

        Purpose: Validates config_id sensitivity to mean values.

        Given: Two GaussianBelief instances with different means.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        b1 = _make_1d_belief(mean=np.array([0.0]))
        b2 = _make_1d_belief(mean=np.array([1.0]))
        assert b1.config_id != b2.config_id

    def test_config_id_differs_for_different_covariance(self):
        """Test that config_id differs when covariances differ.

        Purpose: Validates config_id sensitivity to covariance values.

        Given: Two 1D GaussianBelief instances with different covariances.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        b1 = _make_1d_belief(covariance=np.array([[1.0]]))
        b2 = _make_1d_belief(covariance=np.array([[2.0]]))
        assert b1.config_id != b2.config_id

    def test_hash_and_equality(self):
        """Test that equal beliefs have equal hashes and are equal.

        Purpose: Validates __hash__ and __eq__ based on config_id.

        Given: Two GaussianBelief instances with identical parameters.
        When: hash() and == are compared.
        Then: Hashes match and beliefs are equal.

        Test type: unit
        """
        b1 = _make_1d_belief()
        b2 = _make_1d_belief()
        assert b1 == b2
        assert hash(b1) == hash(b2)


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefEntropy:
    def test_entropy_1d(self):
        """Test 1D Gaussian entropy against analytical formula.

        Purpose: Validates closed-form entropy for 1D case.

        Given: A 1D GaussianBelief with variance sigma^2 = 1.
        When: entropy() is called.
        Then: Result matches 0.5 * ln(2 * pi * e * sigma^2).

        Test type: unit
        """
        belief = _make_1d_belief()
        expected = 0.5 * np.log(2.0 * np.pi * np.e * 1.0)
        np.testing.assert_allclose(belief.entropy(), expected, rtol=1e-10)

    def test_entropy_2d_identity(self):
        """Test 2D Gaussian entropy with identity covariance.

        Purpose: Validates closed-form entropy for 2D identity case.

        Given: A 2D GaussianBelief with identity covariance.
        When: entropy() is called.
        Then: Result matches d/2 * ln(2 * pi * e) + 0.5 * ln(det(I)) = d/2 * ln(2*pi*e).

        Test type: unit
        """
        belief = _make_2d_belief()
        d = 2
        expected = 0.5 * (d * np.log(2.0 * np.pi * np.e) + np.log(1.0))
        np.testing.assert_allclose(belief.entropy(), expected, rtol=1e-10)

    def test_entropy_2d_non_identity(self):
        """Test 2D Gaussian entropy with a non-identity covariance.

        Purpose: Validates entropy computation with non-trivial covariance.

        Given: A 2D GaussianBelief with covariance [[2, 0.5], [0.5, 1]].
        When: entropy() is called.
        Then: Result matches 0.5 * (2*ln(2*pi*e) + ln(det(cov))).

        Test type: unit
        """
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        belief = _make_2d_belief(covariance=cov)
        det = np.linalg.det(cov)
        expected = 0.5 * (2 * np.log(2.0 * np.pi * np.e) + np.log(det))
        np.testing.assert_allclose(belief.entropy(), expected, rtol=1e-10)

    def test_entropy_larger_covariance_has_higher_entropy(self):
        """Test that larger covariance leads to higher entropy.

        Purpose: Validates monotonicity of entropy w.r.t. covariance scale.

        Given: Two 1D beliefs with variance 1 and variance 4.
        When: entropy() is called on both.
        Then: The belief with variance 4 has higher entropy.

        Test type: unit
        """
        b_small = _make_1d_belief(covariance=np.array([[1.0]]))
        b_large = _make_1d_belief(covariance=np.array([[4.0]]))
        assert b_large.entropy() > b_small.entropy()


# ---------------------------------------------------------------------------
# is_terminal_belief tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefTerminal:
    def test_is_terminal_belief_non_terminal(self):
        """Test is_terminal_belief returns False for non-terminal Gaussian belief.

        Purpose: Validates terminal check integration with GaussianBelief via mock.

        Given: A GaussianBelief and a mock environment where is_terminal always returns False.
        When: is_terminal_belief is called.
        Then: Returns False.

        Test type: integration
        """

        class _MockEnv:
            def is_terminal(self, state):
                return False

        np.random.seed(42)
        belief = _make_2d_belief()
        assert is_terminal_belief(belief, _MockEnv()) is False  # type: ignore[arg-type]

    def test_is_terminal_belief_terminal(self):
        """Test is_terminal_belief returns True when all samples are terminal.

        Purpose: Validates that terminal check returns True when environment always says terminal.

        Given: A GaussianBelief and a mock environment where is_terminal always returns True.
        When: is_terminal_belief is called.
        Then: Returns True.

        Test type: integration
        """

        class _MockEnv:
            def is_terminal(self, state):
                return True

        np.random.seed(42)
        belief = _make_2d_belief()
        assert is_terminal_belief(belief, _MockEnv()) is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cost dispatch tests
# ---------------------------------------------------------------------------


class TestGaussianBeliefCostDispatch:
    def _make_mock_env(self):
        class _MockEnv:
            def reward(self, state, action):
                return float(np.sum(state))

        return _MockEnv()

    def test_belief_expectation_cost(self):
        """Test belief_expectation_cost dispatches correctly for GaussianBelief.

        Purpose: Validates cost dispatch returns a finite float.

        Given: A 2D GaussianBelief and a mock environment with sum-reward.
        When: belief_expectation_cost is called.
        Then: Returns a finite float.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2d_belief()
        env = self._make_mock_env()
        cost = belief_expectation_cost(belief=belief, action=0, env=env)  # type: ignore[arg-type]
        assert isinstance(cost, float)
        assert np.isfinite(cost)

    def test_belief_expectation_cost_entropy_penalty(self):
        """Test belief_expectation_cost_entropy_penalty dispatches for GaussianBelief.

        Purpose: Validates entropy-penalized cost dispatch.

        Given: A 2D GaussianBelief with identity covariance and a mock environment.
        When: belief_expectation_cost_entropy_penalty is called with entropy_weight=1.0.
        Then: Returns a finite float, and the result includes an entropy contribution.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2d_belief()
        env = self._make_mock_env()
        cost_no_entropy = belief_expectation_cost_entropy_penalty(
            belief=belief, action=0, env=env, entropy_weight=0.0  # type: ignore[arg-type]
        )
        cost_with_entropy = belief_expectation_cost_entropy_penalty(
            belief=belief, action=0, env=env, entropy_weight=1.0  # type: ignore[arg-type]
        )
        assert isinstance(cost_with_entropy, float)
        assert np.isfinite(cost_with_entropy)
        assert cost_with_entropy != cost_no_entropy

    def test_belief_expectation_cost_information_gain(self):
        """Test belief_expectation_cost_belief_information_gain dispatches for GaussianBelief.

        Purpose: Validates information gain cost dispatch.

        Given: Two GaussianBelief instances (original and updated) and a mock environment.
        When: belief_expectation_cost_belief_information_gain is called.
        Then: Returns a finite float.

        Test type: integration
        """
        np.random.seed(42)
        belief = _make_2d_belief()
        next_belief = _make_2d_belief(covariance=0.5 * np.eye(2))
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
