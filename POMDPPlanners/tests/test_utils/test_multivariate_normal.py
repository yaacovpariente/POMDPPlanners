"""Tests for CovarianceParameterizedMultivariateNormal class.

This module provides comprehensive tests for the multivariate normal distribution
with pre-computed Cholesky decomposition.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


class TestInitialization:
    """Tests for CovarianceParameterizedMultivariateNormal initialization."""

    def test_initialization_valid_covariance(self):
        """Test that valid positive definite matrix initializes correctly.

        Purpose: Validates that a valid positive definite covariance matrix
            results in successful initialization with correct pre-computed values.

        Given: A valid 2x2 positive definite covariance matrix.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: The object is created with correct dimension and stored covariance.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)

        assert mvn.dim == 2
        assert np.allclose(mvn.covariance, cov)

    def test_initialization_rejects_non_symmetric(self):
        """Test that non-symmetric matrix raises ValueError.

        Purpose: Validates that the validation correctly rejects non-symmetric matrices.

        Given: A non-symmetric 2x2 matrix.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: ValueError is raised with appropriate message.

        Test type: unit
        """
        non_symmetric = np.array([[1.0, 0.5], [0.3, 2.0]])

        with pytest.raises(ValueError, match="symmetric"):
            CovarianceParameterizedMultivariateNormal(non_symmetric)

    def test_initialization_rejects_non_positive_definite(self):
        """Test that non-positive definite matrix raises LinAlgError.

        Purpose: Validates that Cholesky decomposition fails for non-PD matrices.

        Given: A symmetric but non-positive definite matrix.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: np.linalg.LinAlgError is raised.

        Test type: unit
        """
        non_pd = np.array([[1.0, 2.0], [2.0, 1.0]])

        with pytest.raises(np.linalg.LinAlgError):
            CovarianceParameterizedMultivariateNormal(non_pd)

    def test_initialization_rejects_non_2d(self):
        """Test that non-2D array raises ValueError.

        Purpose: Validates that 1D arrays are rejected.

        Given: A 1D array instead of a 2D covariance matrix.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: ValueError is raised.

        Test type: unit
        """
        one_d = np.array([1.0, 0.5, 2.0])

        with pytest.raises(ValueError, match="2D"):
            CovarianceParameterizedMultivariateNormal(one_d)

    def test_initialization_rejects_non_square(self):
        """Test that non-square matrix raises ValueError.

        Purpose: Validates that non-square matrices are rejected.

        Given: A non-square 2D array.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: ValueError is raised.

        Test type: unit
        """
        non_square = np.array([[1.0, 0.5], [0.5, 2.0], [0.3, 0.4]])

        with pytest.raises(ValueError, match="square"):
            CovarianceParameterizedMultivariateNormal(non_square)

    def test_initialization_validation_disabled(self):
        """Test that validation can be disabled for performance.

        Purpose: Validates that validate=False skips validation checks.

        Given: A valid covariance matrix and validate=False.
        When: CovarianceParameterizedMultivariateNormal is initialized.
        Then: Initialization succeeds without validation overhead.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov, validate=False)

        assert mvn.dim == 2


class TestSampling:
    """Tests for the sample() method."""

    def test_sample_shape_single(self):
        """Test that n_samples=1 returns shape (1, dim).

        Purpose: Validates correct output shape for single sample.

        Given: A 2D covariance matrix.
        When: sample() is called with n_samples=1.
        Then: Output shape is (1, 2).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])

        samples = mvn.sample(mean, n_samples=1)

        assert samples.shape == (1, 2)

    def test_sample_shape_multiple(self):
        """Test that n_samples=100 returns shape (100, dim).

        Purpose: Validates correct output shape for multiple samples.

        Given: A 2D covariance matrix.
        When: sample() is called with n_samples=100.
        Then: Output shape is (100, 2).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])

        samples = mvn.sample(mean, n_samples=100)

        assert samples.shape == (100, 2)

    def test_sample_mean_converges(self):
        """Test that large sample mean approaches specified mean.

        Purpose: Validates that sampling generates samples with correct expected value.

        Given: A covariance matrix and a non-zero mean.
        When: Many samples are drawn.
        Then: Sample mean converges to the specified mean.

        Test type: unit
        """
        np.random.seed(42)
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([3.0, -2.0])

        samples = mvn.sample(mean, n_samples=10000)
        sample_mean = np.mean(samples, axis=0)

        assert np.allclose(sample_mean, mean, atol=0.1)

    def test_sample_covariance_converges(self):
        """Test that large sample covariance approaches specified covariance.

        Purpose: Validates that sampling generates samples with correct covariance.

        Given: A covariance matrix.
        When: Many samples are drawn.
        Then: Sample covariance converges to the specified covariance.

        Test type: unit
        """
        np.random.seed(42)
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])

        samples = mvn.sample(mean, n_samples=10000)
        sample_cov = np.cov(samples, rowvar=False)

        assert np.allclose(sample_cov, cov, atol=0.1)

    def test_dimension_mismatch_sample_raises(self):
        """Test that wrong mean dimension raises ValueError.

        Purpose: Validates that dimension mismatch is detected in sample().

        Given: A 2D covariance matrix.
        When: sample() is called with a 3D mean.
        Then: ValueError is raised.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        wrong_mean = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="dimension"):
            mvn.sample(wrong_mean)


class TestPDF:
    """Tests for the pdf() and log_pdf() methods."""

    def test_pdf_at_mean_is_maximum(self):
        """Test that PDF is highest at the mean.

        Purpose: Validates that the PDF has its maximum at the mean.

        Given: A covariance matrix and mean.
        When: PDF is computed at mean and at points away from mean.
        Then: PDF at mean is higher than at other points.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([1.0, 2.0])

        values = np.array(
            [
                [1.0, 2.0],  # at mean
                [0.0, 0.0],  # away from mean
                [3.0, 4.0],  # away from mean
            ]
        )

        pdf_values = mvn.pdf(values, mean)

        assert pdf_values[0] > pdf_values[1]
        assert pdf_values[0] > pdf_values[2]

    def test_pdf_matches_scipy(self):
        """Test that results match scipy.stats.multivariate_normal.pdf.

        Purpose: Validates correctness by comparing against scipy reference.

        Given: A covariance matrix, mean, and test points.
        When: PDF is computed using our implementation and scipy.
        Then: Results match within numerical tolerance.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mean = np.array([1.0, 2.0])
        values = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [-1.0, 3.0],
                [2.5, -0.5],
            ]
        )

        mvn = CovarianceParameterizedMultivariateNormal(cov)
        our_pdf = mvn.pdf(values, mean)

        scipy_pdf = multivariate_normal.pdf(values, mean=mean, cov=cov)

        assert np.allclose(our_pdf, scipy_pdf)

    def test_pdf_batch_shape(self):
        """Test that batch values returns correct shape.

        Purpose: Validates correct output shape for batch PDF evaluation.

        Given: A covariance matrix and batch of 5 test points.
        When: PDF is computed.
        Then: Output shape is (5,).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])
        values = np.random.randn(5, 2)

        pdf_values = mvn.pdf(values, mean)

        assert pdf_values.shape == (5,)

    def test_pdf_single_value(self):
        """Test that single value (1D array) works correctly.

        Purpose: Validates that 1D input is handled correctly.

        Given: A covariance matrix and a single point as 1D array.
        When: PDF is computed.
        Then: Output has shape (1,).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])
        single_value = np.array([1.0, 1.0])

        pdf_values = mvn.pdf(single_value, mean)

        assert pdf_values.shape == (1,)

    def test_log_pdf_numerical_stability(self):
        """Test that log_pdf is finite for extreme values.

        Purpose: Validates numerical stability of log_pdf for values far from mean.

        Given: A covariance matrix and points very far from mean.
        When: log_pdf is computed.
        Then: Results are finite (not -inf due to underflow in pdf).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])

        extreme_values = np.array(
            [
                [10.0, 10.0],
                [-10.0, -10.0],
                [20.0, -20.0],
            ]
        )

        log_pdf_values = mvn.log_pdf(extreme_values, mean)

        assert np.all(np.isfinite(log_pdf_values))
        assert np.all(log_pdf_values < 0)

    def test_log_pdf_matches_log_of_pdf(self):
        """Test that log_pdf equals log of pdf for moderate values.

        Purpose: Validates consistency between pdf and log_pdf methods.

        Given: A covariance matrix and moderate test points.
        When: pdf and log_pdf are computed.
        Then: log_pdf equals log(pdf).

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])
        values = np.array([[0.0, 0.0], [1.0, 1.0], [-0.5, 0.5]])

        pdf_values = mvn.pdf(values, mean)
        log_pdf_values = mvn.log_pdf(values, mean)

        assert np.allclose(log_pdf_values, np.log(pdf_values))

    def test_dimension_mismatch_pdf_raises(self):
        """Test that wrong values dimension raises ValueError.

        Purpose: Validates that dimension mismatch is detected in pdf().

        Given: A 2D covariance matrix.
        When: pdf() is called with 3D values.
        Then: ValueError is raised.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([0.0, 0.0])
        wrong_values = np.array([[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="dimension"):
            mvn.pdf(wrong_values, mean)


class TestSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_1d_case(self):
        """Test that 1x1 covariance (univariate) works correctly.

        Purpose: Validates that the implementation handles 1D case.

        Given: A 1x1 covariance matrix (variance).
        When: Sampling and PDF computation are performed.
        Then: Results are correct for univariate normal.

        Test type: unit
        """
        np.random.seed(42)
        variance = 4.0
        cov = np.array([[variance]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.array([5.0])

        samples = mvn.sample(mean, n_samples=10000)
        sample_mean = np.mean(samples)
        sample_var = np.var(samples, ddof=1)

        assert mvn.dim == 1
        assert samples.shape == (10000, 1)
        assert np.isclose(sample_mean, mean[0], atol=0.1)
        assert np.isclose(sample_var, variance, atol=0.2)

        values = np.array([[5.0], [7.0], [3.0]])
        pdf_values = mvn.pdf(values, mean)
        scipy_pdf = multivariate_normal.pdf(values, mean=mean, cov=cov)
        assert np.allclose(pdf_values, scipy_pdf)

    def test_identity_covariance(self):
        """Test that identity covariance (standard normal) works correctly.

        Purpose: Validates behavior with identity covariance matrix.

        Given: An identity covariance matrix.
        When: Sampling and PDF computation are performed.
        Then: Results match standard multivariate normal.

        Test type: unit
        """
        np.random.seed(42)
        dim = 3
        cov = np.eye(dim)
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.zeros(dim)

        samples = mvn.sample(mean, n_samples=10000)
        sample_cov = np.cov(samples, rowvar=False)

        assert np.allclose(sample_cov, cov, atol=0.1)

        values = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pdf_values = mvn.pdf(values, mean)
        scipy_pdf = multivariate_normal.pdf(values, mean=mean, cov=cov)
        assert np.allclose(pdf_values, scipy_pdf)

    def test_diagonal_covariance(self):
        """Test that diagonal covariance produces uncorrelated samples.

        Purpose: Validates that diagonal covariance results in uncorrelated dimensions.

        Given: A diagonal covariance matrix.
        When: Many samples are drawn.
        Then: Sample correlation between dimensions is near zero.

        Test type: unit
        """
        np.random.seed(42)
        cov = np.diag([1.0, 4.0, 9.0])
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.zeros(3)

        samples = mvn.sample(mean, n_samples=10000)
        sample_corr = np.corrcoef(samples, rowvar=False)

        off_diagonal = sample_corr[np.triu_indices(3, k=1)]
        assert np.allclose(off_diagonal, 0.0, atol=0.05)

        sample_vars = np.var(samples, axis=0, ddof=1)
        assert np.allclose(sample_vars, np.diag(cov), atol=0.3)

    def test_high_dimensional(self):
        """Test that high-dimensional case works correctly.

        Purpose: Validates that the implementation scales to higher dimensions.

        Given: A 10x10 positive definite covariance matrix.
        When: Sampling and PDF computation are performed.
        Then: Results have correct shapes and reasonable values.

        Test type: unit
        """
        np.random.seed(42)
        dim = 10
        A = np.random.randn(dim, dim)
        cov = A @ A.T + np.eye(dim)
        mvn = CovarianceParameterizedMultivariateNormal(cov)
        mean = np.random.randn(dim)

        samples = mvn.sample(mean, n_samples=100)
        assert samples.shape == (100, dim)

        values = samples[:5]
        pdf_values = mvn.pdf(values, mean)
        assert pdf_values.shape == (5,)
        assert np.all(pdf_values > 0)

        scipy_pdf = multivariate_normal.pdf(values, mean=mean, cov=cov)
        assert np.allclose(pdf_values, scipy_pdf)

    def test_covariance_is_copied(self):
        """Test that covariance is copied, not referenced.

        Purpose: Validates that modifying the input doesn't affect the distribution.

        Given: A covariance matrix.
        When: The original matrix is modified after initialization.
        Then: The distribution's covariance remains unchanged.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)

        original_cov = mvn.covariance.copy()
        cov[0, 0] = 100.0

        assert np.allclose(mvn.covariance, original_cov)

    def test_covariance_property_returns_copy(self):
        """Test that covariance property returns a copy.

        Purpose: Validates that the covariance property returns a copy, not a reference.

        Given: A CovarianceParameterizedMultivariateNormal instance.
        When: The returned covariance is modified.
        Then: The internal covariance remains unchanged.

        Test type: unit
        """
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        mvn = CovarianceParameterizedMultivariateNormal(cov)

        returned_cov = mvn.covariance
        returned_cov[0, 0] = 100.0

        assert mvn.covariance[0, 0] == 1.0


class TestScipyComparison:
    """Comprehensive comparison tests against scipy reference implementation."""

    def test_comprehensive_scipy_comparison(self):
        """Comprehensive test comparing against scipy across multiple scenarios.

        Purpose: Validates correctness across various covariance structures and means.

        Given: Multiple covariance matrices and means.
        When: PDF is computed for various test points.
        Then: All results match scipy within numerical tolerance.

        Test type: integration
        """
        test_cases = [
            {
                "cov": np.array([[1.0, 0.5], [0.5, 2.0]]),
                "mean": np.array([1.0, 2.0]),
            },
            {
                "cov": np.array([[0.1, 0.0], [0.0, 0.1]]),
                "mean": np.array([0.0, 0.0]),
            },
            {
                "cov": np.array([[10.0, -3.0], [-3.0, 5.0]]),
                "mean": np.array([-5.0, 10.0]),
            },
        ]

        for case in test_cases:
            cov = case["cov"]
            mean = case["mean"]

            mvn = CovarianceParameterizedMultivariateNormal(cov)

            np.random.seed(123)
            test_points = np.random.randn(20, 2) * 3 + mean

            our_pdf = mvn.pdf(test_points, mean)
            our_log_pdf = mvn.log_pdf(test_points, mean)

            scipy_pdf = multivariate_normal.pdf(test_points, mean=mean, cov=cov)
            scipy_log_pdf = multivariate_normal.logpdf(test_points, mean=mean, cov=cov)

            assert np.allclose(our_pdf, scipy_pdf), f"PDF mismatch for cov={cov}"
            assert np.allclose(our_log_pdf, scipy_log_pdf), f"Log PDF mismatch for cov={cov}"
