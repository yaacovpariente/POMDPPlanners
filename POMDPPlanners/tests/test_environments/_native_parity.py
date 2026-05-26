# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Reusable parity-test helpers for native (C++) transition / observation models.

The helpers here encode the invariants every native model port must satisfy:

* The Python shim registers as a virtual subclass of the appropriate ABC and
  has no unresolved abstract methods.
* Samples drawn from the C++ path have empirical mean and covariance matching
  the declared Gaussian spec (since the C++ RNG cannot be compared
  bit-for-bit against numpy's).
* ``probability()`` agrees with ``CovarianceParameterizedMultivariateNormal``
  on a deterministic grid to the requested tolerance (default 1e-10).
* Seeding the module-level RNG yields reproducible sample sequences.
* ``sample()`` returns a list of 1-D ndarrays of the declared dimension.

The module name is underscore-prefixed so pytest does not collect it.
"""

from typing import Any, Callable, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


CovarianceSpec = Union[Sequence[float], NDArray[np.float64]]


def build_covariance(spec: CovarianceSpec) -> NDArray[np.float64]:
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 1:
        return np.diag(arr)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return arr.copy()
    raise ValueError(f"expected 1-D diagonal or square 2-D matrix, got shape {arr.shape}")


def assert_abc_registration(model: Any, abc_cls: type) -> None:
    assert isinstance(model, abc_cls), (
        f"{type(model).__name__} must be an instance of {abc_cls.__name__} "
        "(via ABC.register). The Python shim likely did not call .register()."
    )
    abstract_methods = getattr(type(model), "__abstractmethods__", frozenset())
    assert (
        abstract_methods == frozenset()
    ), f"{type(model).__name__} has unresolved abstract methods: {abstract_methods}"


def assert_sample_moments_match(
    *,
    model: Any,
    seed_fn: Callable[[int], None],
    expected_mean: NDArray[np.float64],
    expected_cov: NDArray[np.float64],
    n_samples: int = 200_000,
    seed: int = 12345,
    mean_atol: float = 5e-3,
    cov_frobenius_atol: float = 1e-3,
) -> None:
    seed_fn(seed)
    samples = np.asarray(model.sample(n_samples))
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    np.testing.assert_allclose(empirical_mean, expected_mean, atol=mean_atol)
    frob_err = np.linalg.norm(empirical_cov - expected_cov, ord="fro")
    assert (
        frob_err < cov_frobenius_atol
    ), f"Covariance Frobenius error {frob_err:.3e} >= {cov_frobenius_atol:.3e}"


def assert_logpdf_bitwise(
    *,
    model: Any,
    reference_dist: CovarianceParameterizedMultivariateNormal,
    mean: NDArray[np.float64],
    points: NDArray[np.float64],
    atol: float = 1e-10,
) -> None:
    cpp_pdf = model.probability(points)
    py_pdf = reference_dist.pdf(points, mean)
    np.testing.assert_allclose(cpp_pdf, py_pdf, atol=atol, rtol=0.0)


def assert_determinism_under_seed(
    *,
    model: Any,
    seed_fn: Callable[[int], None],
    n_samples: int = 50,
    seed: int = 2024,
) -> None:
    seed_fn(seed)
    first = np.asarray(model.sample(n_samples))
    seed_fn(seed)
    second = np.asarray(model.sample(n_samples))
    np.testing.assert_array_equal(first, second)


def assert_sample_shape_contract(
    *,
    model: Any,
    seed_fn: Callable[[int], None],
    n_samples: int,
    expected_dim: int,
    seed: int = 0,
) -> None:
    seed_fn(seed)
    out = model.sample(n_samples)
    assert isinstance(out, list), f"sample() must return a list, got {type(out).__name__}"
    assert len(out) == n_samples, f"sample({n_samples}) returned {len(out)} items"
    for idx, item in enumerate(out):
        assert isinstance(
            item, np.ndarray
        ), f"sample()[{idx}] must be an ndarray, got {type(item).__name__}"
        assert item.ndim == 1, f"sample()[{idx}] must be 1-D, got ndim={item.ndim}"
        assert item.shape == (
            expected_dim,
        ), f"sample()[{idx}] must have shape ({expected_dim},), got {item.shape}"
