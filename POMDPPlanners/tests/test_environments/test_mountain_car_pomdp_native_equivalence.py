"""Numerical equivalence tests for the native MountainCar sampling extension.

Verifies that the C++ ``_native`` sampling / probability code matches the
Python reference (``CovarianceParameterizedMultivariateNormal``) up to
statistical noise in the samples and up to floating-point tolerance in
the PDF. Complements ``test_mountain_car_pomdp.py`` which exercises the
shipped API; this module focuses on the port-specific invariants called
out in the implementation plan.
"""

import numpy as np
import pytest

from POMDPPlanners.core.environment import ObservationModel, StateTransitionModel
from POMDPPlanners.environments.mountain_car_pomdp import (
    MountainCarObservation,
    MountainCarPOMDP,
    MountainCarTransition,
    _native,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


@pytest.fixture(name="env")
def _env_fixture():
    return MountainCarPOMDP(discount_factor=0.99)


@pytest.fixture(name="transition")
def _transition_fixture(env):
    return env.state_transition_model(state=(-0.5, 0.0), action=1)


@pytest.fixture(name="observation")
def _observation_fixture(env):
    return env.observation_model(next_state=(-0.45, 0.02), action=1)


# ---------------------------------------------------------------------------
# ABC / typing contract
# ---------------------------------------------------------------------------


def test_transition_is_registered_as_state_transition_model(transition):
    """Shim passes isinstance check for StateTransitionModel after register().

    Purpose: Validates the ABC virtual-subclass registration applied in the
    Python shim so downstream polymorphic callers see the C++ class as a
    valid StateTransitionModel.

    Given: A MountainCarTransition produced by MountainCarPOMDP's factory.
    When: isinstance is checked against StateTransitionModel.
    Then: It returns True.

    Test type: unit
    """
    assert isinstance(transition, StateTransitionModel)


def test_observation_is_registered_as_observation_model(observation):
    """Shim passes isinstance check for ObservationModel after register().

    Purpose: Same as the transition case, for the observation model.

    Given: A MountainCarObservation produced by MountainCarPOMDP's factory.
    When: isinstance is checked against ObservationModel.
    Then: It returns True.

    Test type: unit
    """
    assert isinstance(observation, ObservationModel)


def test_transition_is_not_abstract():
    """MountainCarTransition is instantiable (no unresolved abstract methods).

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The MountainCarTransition class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(MountainCarTransition, "__abstractmethods__", frozenset())
    assert abstract_methods == frozenset()


def test_observation_is_not_abstract():
    """MountainCarObservation is instantiable (no unresolved abstract methods).

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The MountainCarObservation class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(MountainCarObservation, "__abstractmethods__", frozenset())
    assert abstract_methods == frozenset()


# ---------------------------------------------------------------------------
# Sample distribution matches the Gaussian spec
# ---------------------------------------------------------------------------


def test_transition_sample_empirical_moments_match_spec(env):
    """Native transition samples have the documented Gaussian distribution.

    Purpose: Validates that 200k samples from the C++ sampler have the
    theoretical mean (deterministic next state) and covariance (the
    declared process noise), since the C++ RNG cannot be compared bit-for-bit
    against numpy's.

    Given: A MountainCarTransition from state=(-0.5, 0.0) with action=0
        (no engine force, so the deterministic update is interior to the
        clipping bounds and no particles are clipped).
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches the deterministic next state within 5e-3,
        and the empirical covariance matches the configured process noise
        within a Frobenius-norm tolerance of 1e-5.

    Test type: integration
    """
    _native.set_seed(12345)
    transition = env.state_transition_model(state=(-0.5, 0.0), action=0)
    det = np.asarray(
        transition._compute_deterministic_next_state()
    )  # pylint: disable=protected-access

    samples = np.asarray(transition.sample(200_000))
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    np.testing.assert_allclose(empirical_mean, det, atol=5e-3)
    frob_err = np.linalg.norm(empirical_cov - env.state_transition_cov, ord="fro")
    assert frob_err < 1e-5, f"Covariance Frobenius error {frob_err:.3e} >= 1e-5"


def test_observation_sample_empirical_moments_match_spec(env):
    """Native observation samples match the documented observation noise.

    Purpose: Validates that 200k observation samples have the theoretical
    mean (the input next_state) and covariance (the declared observation
    noise).

    Given: A MountainCarObservation with next_state=(-0.2, 0.01).
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches the next_state within 5e-3, and the
        empirical covariance matches the observation covariance within
        Frobenius-norm 1e-3.

    Test type: integration
    """
    _native.set_seed(7777)
    next_state = (-0.2, 0.01)
    observation = env.observation_model(next_state=next_state, action=0)

    samples = np.asarray(observation.sample(200_000))
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    np.testing.assert_allclose(empirical_mean, np.array(next_state), atol=5e-3)
    frob_err = np.linalg.norm(empirical_cov - env.cov_matrix, ord="fro")
    assert frob_err < 1e-3, f"Covariance Frobenius error {frob_err:.3e} >= 1e-3"


# ---------------------------------------------------------------------------
# probability() bit-exactness vs the Python reference
# ---------------------------------------------------------------------------


def test_transition_probability_matches_python_reference_on_grid(env):
    """C++ probability() matches CovarianceParameterizedMultivariateNormal.pdf.

    Purpose: Validates that the C++ PDF computation agrees numerically with
    the Python reference ``CovarianceParameterizedMultivariateNormal.pdf``
    on a deterministic grid.

    Given: A 1000-point grid of candidate next-states around the deterministic
        transition target, plus the Python reference distribution constructed
        from the same covariance.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    transition = env.state_transition_model(state=(-0.5, 0.0), action=1)
    det = np.asarray(
        transition._compute_deterministic_next_state()
    )  # pylint: disable=protected-access

    rng = np.random.default_rng(42)
    offsets = rng.normal(size=(1000, 2)) * np.array([0.01, 0.005])
    grid = det[np.newaxis, :] + offsets

    cpp_pdf = transition.probability(grid)
    py_ref = CovarianceParameterizedMultivariateNormal(env.state_transition_cov)
    py_pdf = py_ref.pdf(grid, det)

    np.testing.assert_allclose(cpp_pdf, py_pdf, atol=1e-10, rtol=0.0)


def test_observation_probability_matches_python_reference_on_grid(env):
    """C++ observation probability() matches the Python reference on a grid.

    Purpose: Same as the transition case, for the observation model.

    Given: A 1000-point grid of candidate observations around the true state.
    When: probability(grid) is called on both C++ and Python reference.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([-0.3, 0.015])
    observation = env.observation_model(next_state=tuple(next_state), action=1)

    rng = np.random.default_rng(4242)
    offsets = rng.normal(size=(1000, 2)) * np.array([0.1, 0.01])
    grid = next_state[np.newaxis, :] + offsets

    cpp_pdf = observation.probability(grid)
    py_ref = CovarianceParameterizedMultivariateNormal(env.cov_matrix)
    py_pdf = py_ref.pdf(grid, next_state)

    np.testing.assert_allclose(cpp_pdf, py_pdf, atol=1e-10, rtol=0.0)


def test_transition_probability_accepts_list_and_ndarray_inputs(transition):
    """probability() accepts both a list of 1-D arrays and a 2-D ndarray.

    Purpose: Validates the Python-side input flexibility of the C++ binding
    (matches the pre-port duck-typed contract).

    Given: A list of three 1-D numpy arrays and the equivalent stacked 2-D
        array.
    When: probability() is called with each form.
    Then: Outputs are identical.

    Test type: unit
    """
    list_values = [
        np.array([-0.51, 0.001]),
        np.array([-0.495, -0.002]),
        np.array([-0.5, 0.0]),
    ]
    array_values = np.stack(list_values, axis=0)

    from_list = transition.probability(list_values)
    from_array = transition.probability(array_values)

    np.testing.assert_array_equal(from_list, from_array)


# ---------------------------------------------------------------------------
# Determinism under explicit seeding
# ---------------------------------------------------------------------------


def test_sample_is_deterministic_under_set_seed(env):
    """Repeated set_seed + sample yields identical samples.

    Purpose: Validates the determinism path used by reproducible tests.

    Given: Two MountainCarTransition instances sampled under the same seed.
    When: _native.set_seed(seed) is called before each, followed by sample(50).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    transition = env.state_transition_model(state=(-0.4, 0.01), action=1)

    _native.set_seed(2024)
    first = np.asarray(transition.sample(50))
    _native.set_seed(2024)
    second = np.asarray(transition.sample(50))

    np.testing.assert_array_equal(first, second)


def test_sample_returns_list_of_1d_ndarrays(transition):
    """sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift (many call sites index
    the returned list with ``[0]`` or iterate it).

    Given: A MountainCarTransition.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 2.

    Test type: unit
    """
    _native.set_seed(0)
    out = transition.sample(3)
    assert isinstance(out, list)
    assert len(out) == 3
    for item in out:
        assert isinstance(item, np.ndarray)
        assert item.ndim == 1
        assert item.shape == (2,)
