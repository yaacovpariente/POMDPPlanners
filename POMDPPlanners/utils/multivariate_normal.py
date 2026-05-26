# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Module for efficient multivariate normal distribution with pre-computed Cholesky decomposition.

This module provides a multivariate normal distribution implementation optimized for scenarios
where the covariance matrix is fixed but the mean varies (e.g., POMDP observation models).
The expensive Cholesky decomposition O(n^3) is computed once during initialization, making
subsequent sampling and PDF evaluations O(n^2).

Classes:
    CovarianceParameterizedMultivariateNormal: Efficient multivariate normal with pre-computed Cholesky
"""

import numpy as np
from scipy.linalg import solve_triangular


class CovarianceParameterizedMultivariateNormal:
    """Multivariate normal distribution with pre-computed Cholesky decomposition.

    This class provides efficient sampling and PDF computation for multivariate
    normal distributions where the covariance matrix is fixed but the mean can
    vary. The Cholesky decomposition is computed once during initialization,
    enabling O(n^2) operations for sampling and PDF evaluation.

    This class does NOT inherit from the Distribution base class because it has
    a different interface - the mean is passed to methods rather than the constructor.

    Attributes:
        covariance: The covariance matrix (read-only property).
        dim: Dimensionality of the distribution.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> # Create distribution with fixed covariance
        >>> cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        >>> mvn = CovarianceParameterizedMultivariateNormal(cov)
        >>>
        >>> # Sample with different means
        >>> mean1 = np.array([0.0, 0.0])
        >>> samples1 = mvn.sample(mean1, n_samples=3)
        >>> samples1.shape
        (3, 2)
        >>>
        >>> # Compute PDF
        >>> values = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> pdf_values = mvn.pdf(values, mean1)
        >>> pdf_values.shape
        (2,)
    """

    def __init__(self, covariance: np.ndarray, validate: bool = True):
        """Initialize the distribution with a covariance matrix.

        Args:
            covariance: Positive definite covariance matrix of shape (dim, dim).
            validate: Whether to validate the covariance matrix. Defaults to True.
                Set to False for performance if you're certain the matrix is valid.

        Raises:
            ValueError: If covariance is not 2D, not square, or not symmetric.
            np.linalg.LinAlgError: If covariance is not positive definite.
        """
        covariance = np.asarray(covariance)

        if validate:
            self._validate_covariance(covariance)

        self._covariance = covariance.copy()
        self._dim = covariance.shape[0]
        self._cholesky_L = np.linalg.cholesky(covariance)
        self._cholesky_L_T = self._cholesky_L.T.copy()
        self._log_det = 2.0 * np.sum(np.log(np.diag(self._cholesky_L)))
        self._log_normalization = -0.5 * (self._dim * np.log(2.0 * np.pi) + self._log_det)

    def _validate_covariance(self, covariance: np.ndarray) -> None:
        if covariance.ndim != 2:
            raise ValueError(f"Covariance must be a 2D array, got {covariance.ndim}D array")

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"Covariance must be square, got shape {covariance.shape}")

        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance matrix must be symmetric")

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the covariance matrix."""
        return self._covariance.copy()

    def covariance_view(self) -> np.ndarray:
        """Return the underlying covariance buffer without copying.

        Read-only fast path for hot-loop callers that need the covariance
        purely as input to a C++ Cholesky factorisation. Avoid mutating the
        returned array — it aliases the internal storage.
        """
        return self._covariance

    @property
    def dim(self) -> int:
        """Return the dimensionality of the distribution."""
        return self._dim

    def sample(self, mean: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Sample from the multivariate normal distribution.

        Generates samples using the transformation: x = mean + L @ z,
        where z ~ N(0, I) and L is the Cholesky factor of the covariance.

        Args:
            mean: Mean vector of shape (dim,).
            n_samples: Number of samples to generate. Defaults to 1.

        Returns:
            Array of shape (n_samples, dim) containing the samples.

        Raises:
            ValueError: If mean dimension doesn't match covariance dimension.
        """
        mean = np.asarray(mean)
        self._validate_mean_dimension(mean)

        z = np.random.standard_normal((n_samples, self._dim))
        samples = mean + z @ self._cholesky_L_T

        return samples

    def pdf(self, values: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """Compute the probability density function.

        Uses the pre-computed Cholesky decomposition for efficient computation:
        y = solve_triangular(L, (x - mean).T)
        mahalanobis_sq = sum(y^2)
        pdf = exp(log_normalization - 0.5 * mahalanobis_sq)

        Args:
            values: Array of shape (n, dim) or (dim,) containing points to evaluate.
            mean: Mean vector of shape (dim,).

        Returns:
            Array of shape (n,) containing PDF values at each point.

        Raises:
            ValueError: If dimensions don't match.
        """
        log_pdf_values = self.log_pdf(values, mean)
        return np.exp(log_pdf_values)

    def log_pdf(self, values: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """Compute the log probability density function.

        More numerically stable than pdf() for small probability values.

        Args:
            values: Array of shape (n, dim) or (dim,) containing points to evaluate.
            mean: Mean vector of shape (dim,).

        Returns:
            Array of shape (n,) containing log PDF values at each point.

        Raises:
            ValueError: If dimensions don't match.
        """
        values = np.asarray(values)
        mean = np.asarray(mean)

        self._validate_mean_dimension(mean)

        values_2d = values.reshape(-1, self._dim) if values.ndim == 1 else values

        if values_2d.shape[1] != self._dim:
            raise ValueError(
                f"Values dimension {values_2d.shape[1]} doesn't match "
                f"covariance dimension {self._dim}"
            )

        centered = values_2d - mean
        y = solve_triangular(self._cholesky_L, centered.T, lower=True)
        mahalanobis_sq = np.sum(y**2, axis=0)
        log_pdf_values = self._log_normalization - 0.5 * mahalanobis_sq

        return log_pdf_values

    def _validate_mean_dimension(self, mean: np.ndarray) -> None:
        if mean.shape != (self._dim,):
            raise ValueError(
                f"Mean shape {mean.shape} doesn't match covariance dimension ({self._dim},)"
            )
