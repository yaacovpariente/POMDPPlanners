"""Numerical equivalence tests for the native Continuous Light-Dark extension.

Verifies that the C++ ``_native`` sampling / probability code matches the
Python reference (``CovarianceParameterizedMultivariateNormal``) up to
statistical noise in the samples and up to floating-point tolerance in
the PDF. Complements ``test_continuous_light_dark_pomdp.py`` which
exercises the shipped API; this module focuses on the port-specific
invariants (ABC registration, empirical moments, bit-exact probability,
determinism under seeding, batch-vs-per-particle parity). Generic
assertion mechanics live in ``_native_parity.py`` so other native env
ports can reuse them.
"""

import numpy as np
import pytest

from POMDPPlanners.core.environment import ObservationModel, StateTransitionModel
from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkStateTransitionModel,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
    ContinuousLightDarkNormalNoiseObservationModel,
)
from POMDPPlanners.tests.test_environments import _native_parity
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


@pytest.fixture(name="env")
def _env_fixture():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture(name="transition")
def _transition_fixture(env):
    return env.state_transition_model(state=np.array([3.0, 4.0]), action=np.array([1.0, 0.5]))


@pytest.fixture(name="observation_near_beacon")
def _observation_near_fixture(env):
    # (5, 5) is an env beacon -> near-beacon Gaussian (half covariance),
    # away from the grid boundary so clip-induced bias doesn't leak into
    # empirical-moments tests.
    return env.observation_model(next_state=np.array([5.0, 5.0]), action=np.array([0.0, 0.0]))


@pytest.fixture(name="observation_far_from_beacon")
def _observation_far_fixture(env):
    # Mid-grid, far from any beacon -> far Gaussian.
    return env.observation_model(next_state=np.array([2.5, 2.5]), action=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# ABC / typing contract
# ---------------------------------------------------------------------------


def test_transition_is_registered_as_state_transition_model(transition):
    """Shim passes isinstance check for StateTransitionModel after register().

    Purpose: Validates the ABC virtual-subclass registration applied in the
    Python shim so downstream polymorphic callers see the C++ class as a
    valid StateTransitionModel.

    Given: A ContinuousLightDarkStateTransitionModel from the POMDP factory.
    When: isinstance is checked against StateTransitionModel.
    Then: It returns True.

    Test type: unit
    """
    _native_parity.assert_abc_registration(transition, StateTransitionModel)


def test_observation_is_registered_as_observation_model(observation_near_beacon):
    """Shim passes isinstance check for ObservationModel after register().

    Purpose: Same as the transition case, for the observation model.

    Given: A ContinuousLightDarkNormalNoiseObservationModel from the factory.
    When: isinstance is checked against ObservationModel.
    Then: It returns True.

    Test type: unit
    """
    _native_parity.assert_abc_registration(observation_near_beacon, ObservationModel)


def test_transition_is_not_abstract():
    """ContinuousLightDarkStateTransitionModel is instantiable.

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The ContinuousLightDarkStateTransitionModel class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(
        ContinuousLightDarkStateTransitionModel, "__abstractmethods__", frozenset()
    )
    assert abstract_methods == frozenset()


def test_observation_is_not_abstract():
    """ContinuousLightDarkNormalNoiseObservationModel is instantiable.

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The ContinuousLightDarkNormalNoiseObservationModel class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(
        ContinuousLightDarkNormalNoiseObservationModel, "__abstractmethods__", frozenset()
    )
    assert abstract_methods == frozenset()


# ---------------------------------------------------------------------------
# Sample distribution matches the Gaussian spec
# ---------------------------------------------------------------------------


def test_transition_sample_empirical_moments_match_spec(env):
    """Native transition samples have the documented Gaussian distribution.

    Purpose: Validates that 200k samples from the C++ sampler have the
    theoretical mean (``state + action``) and covariance (the declared
    process noise), since the C++ RNG cannot be compared bit-for-bit
    against numpy's.

    Given: A transition from state=(3.0, 4.0) with action=(0.5, -0.3).
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches ``state + action`` within 5e-3,
        and the empirical covariance matches the configured process noise
        within Frobenius-norm 1e-3.

    Test type: integration
    """
    state = np.array([3.0, 4.0])
    action = np.array([0.5, -0.3])
    transition = env.state_transition_model(state=state, action=action)

    _native_parity.assert_sample_moments_match(
        model=transition,
        seed_fn=_native.set_seed,
        expected_mean=state + action,
        expected_cov=env.state_transition_cov_matrix,
        n_samples=200_000,
        seed=12345,
        mean_atol=5e-3,
        cov_frobenius_atol=1e-3,
    )


def test_observation_sample_moments_match_far_from_beacon(env, observation_far_from_beacon):
    """Native far-from-beacon observation samples match the declared wide noise.

    Purpose: Validates that 200k observation samples (at a point far from
    all beacons, so the far-beacon Gaussian is active) have the theoretical
    mean (next_state) and covariance (observation_cov_matrix).

    Given: An observation with next_state=(2.5, 2.5) — far from every beacon.
    When: 200,000 samples are drawn via the C++ path.
    Then: Empirical mean ≈ next_state (atol 5e-3) and empirical covariance
        ≈ observation_cov_matrix (Frobenius 5e-3).

    Test type: integration
    """
    next_state = np.array([2.5, 2.5])
    _native_parity.assert_sample_moments_match(
        model=observation_far_from_beacon,
        seed_fn=_native.set_seed,
        expected_mean=next_state,
        expected_cov=env.observation_cov_matrix,
        n_samples=200_000,
        seed=7777,
        mean_atol=5e-3,
        cov_frobenius_atol=5e-3,
    )


def test_observation_sample_moments_match_near_beacon(env, observation_near_beacon):
    """Native near-beacon observation samples match the declared tighter noise.

    Purpose: Validates the state-dependent covariance branch of the C++
    observation sampler: the effective covariance near a beacon is
    ``observation_cov_matrix * 0.5``.

    Given: An observation with next_state=(5.0, 5.0) — sitting on a beacon,
        well away from the grid boundary so clipping does not bias the
        sample moments.
    When: 200,000 samples are drawn via the C++ path.
    Then: Empirical mean ≈ next_state (atol 5e-3) and empirical covariance
        ≈ 0.5 * observation_cov_matrix (Frobenius 5e-3).

    Test type: integration
    """
    next_state = np.array([5.0, 5.0])
    _native_parity.assert_sample_moments_match(
        model=observation_near_beacon,
        seed_fn=_native.set_seed,
        expected_mean=next_state,
        expected_cov=env.observation_cov_matrix * 0.5,
        n_samples=200_000,
        seed=4242,
        mean_atol=5e-3,
        cov_frobenius_atol=5e-3,
    )


# ---------------------------------------------------------------------------
# probability() bit-exactness vs the Python reference
# ---------------------------------------------------------------------------


def test_transition_probability_matches_python_reference_on_grid(env):
    """C++ probability() matches CovarianceParameterizedMultivariateNormal.pdf.

    Purpose: Validates that the C++ PDF computation agrees numerically with
    the Python reference ``CovarianceParameterizedMultivariateNormal.pdf``
    on a deterministic grid.

    Given: A 1000-point grid of candidate next-states around the
        deterministic transition target, plus the Python reference
        distribution constructed from the same covariance.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    state = np.array([3.0, 4.0])
    action = np.array([1.0, 0.5])
    transition = env.state_transition_model(state=state, action=action)
    mean = state + action

    rng = np.random.default_rng(42)
    offsets = rng.normal(size=(1000, 2)) * 0.2
    grid = mean[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(env.state_transition_cov_matrix)
    _native_parity.assert_logpdf_bitwise(
        model=transition,
        reference_dist=py_ref,
        mean=mean,
        points=grid,
        atol=1e-10,
    )


def test_observation_probability_matches_python_reference_far_from_beacon(
    env, observation_far_from_beacon
):
    """C++ observation probability() matches the wide-noise Python reference.

    Purpose: Same as the transition case, for the far-beacon branch.

    Given: A 1000-point grid of candidate observations around next_state,
        plus the Python reference built from ``observation_cov_matrix``.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([2.5, 2.5])
    rng = np.random.default_rng(4242)
    offsets = rng.normal(size=(1000, 2)) * 0.3
    grid = next_state[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(env.observation_cov_matrix)
    _native_parity.assert_logpdf_bitwise(
        model=observation_far_from_beacon,
        reference_dist=py_ref,
        mean=next_state,
        points=grid,
        atol=1e-10,
    )


def test_observation_probability_matches_python_reference_near_beacon(env, observation_near_beacon):
    """C++ observation probability() matches the near-beacon (tight) reference.

    Purpose: Validates the near-beacon Gaussian (halved covariance) branch.

    Given: A 1000-point grid of candidate observations around a next_state
        sitting on the beacon at (5, 5), plus the Python reference built
        from ``0.5 * observation_cov_matrix``.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([5.0, 5.0])
    rng = np.random.default_rng(54321)
    offsets = rng.normal(size=(1000, 2)) * 0.2
    grid = next_state[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(env.observation_cov_matrix * 0.5)
    _native_parity.assert_logpdf_bitwise(
        model=observation_near_beacon,
        reference_dist=py_ref,
        mean=next_state,
        points=grid,
        atol=1e-10,
    )


def test_transition_probability_accepts_list_and_ndarray_inputs(transition):
    """probability() accepts both a list of 1-D arrays and a 2-D ndarray.

    Purpose: Validates Python-side input flexibility of the C++ binding
    (matches the pre-port duck-typed contract).

    Given: A list of three 1-D numpy arrays and the equivalent stacked 2-D
        array.
    When: probability() is called with each form.
    Then: Outputs are identical.

    Test type: unit
    """
    list_values = [
        np.array([4.0, 4.5]),
        np.array([4.1, 4.6]),
        np.array([3.9, 4.4]),
    ]
    array_values = np.stack(list_values, axis=0)

    from_list = transition.probability(list_values)
    from_array = transition.probability(array_values)

    np.testing.assert_array_equal(from_list, from_array)


# ---------------------------------------------------------------------------
# Determinism under explicit seeding
# ---------------------------------------------------------------------------


def test_sample_is_deterministic_under_set_seed(transition):
    """Repeated set_seed + sample yields identical samples.

    Purpose: Validates the determinism path used by reproducible tests.

    Given: A ContinuousLightDarkStateTransitionModel.
    When: _native.set_seed(seed) is called before each sample(50).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    _native_parity.assert_determinism_under_seed(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=50,
        seed=2024,
    )


def test_sample_returns_list_of_1d_ndarrays(transition):
    """sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift (many call sites index
    the returned list with ``[0]`` or iterate it).

    Given: A ContinuousLightDarkStateTransitionModel.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 2.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=2,
    )


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------


def test_transition_batch_sample_matches_per_particle_sample(env):
    """batch_sample(P) is bit-identical to sample(1) per row under fixed seed.

    Purpose: Validates that TransitionModelCpp<2>.batch_sample produces the
    same noise sequence as N per-particle sample(1) calls when both consume
    the same module-level C++ RNG seeded once.

    Given: 64 particles scattered over the grid and a single action vector.
    When: batch_sample runs under seed=777; then for each particle a fresh
        ContinuousLightDarkStateTransitionModel is built and sample(1) is
        called in the same order under the same seed.
    Then: The two (64, 2) arrays are array_equal.

    Test type: unit
    """
    rng = np.random.default_rng(123)
    particles = rng.uniform(0.0, 10.0, size=(64, 2))
    action = np.array([0.5, 0.2])

    _native.set_seed(777)
    transition = env.state_transition_model(state=np.array([5.0, 5.0]), action=action)
    batch_result = transition.batch_sample(particles)

    _native.set_seed(777)
    per_particle_rows = []
    for row in particles:
        model = env.state_transition_model(state=row, action=action)
        per_particle_rows.append(model.sample(1)[0])
    per_particle_result = np.stack(per_particle_rows, axis=0)

    np.testing.assert_array_equal(batch_result, per_particle_result)


def test_observation_batch_log_likelihood_matches_per_particle(env):
    """batch_log_likelihood matches per-particle log_pdf on the active noise.

    Purpose: Validates the observation batch path against the per-particle
    reference. The comparison is done in log-space against the active
    distribution's ``log_pdf`` (rather than ``log(probability())``) so
    rows where the probability underflows to 0 still have a well-defined
    reference value.

    Given: 64 random next-state particles spanning positions close to and
        far from beacons, and a single observation. The per-row near/far
        decision is exercised by mixing both regions.
    When: batch_log_likelihood(next_particles, observation) is called, and
        for each particle a fresh
        ContinuousLightDarkNormalNoiseObservationModel is built so that
        ``_active_dist.log_pdf`` selects the correct near/far Gaussian.
    Then: The batch log-likelihoods equal the per-particle log_pdf values
        within atol=1e-12.

    Test type: unit
    """
    rng = np.random.default_rng(456)
    particles = rng.uniform(0.0, 10.0, size=(64, 2))
    observation = np.array([5.0, 5.0])
    action = np.array([0.0, 0.0])

    obs_model = env.observation_model(next_state=np.array([5.0, 5.0]), action=action)
    batch_log_ll = obs_model.batch_log_likelihood(particles, observation)

    per_particle_log_ll = np.empty(len(particles))
    for i, next_state in enumerate(particles):
        model = env.observation_model(next_state=next_state, action=action)
        # log_pdf directly on the active distribution avoids underflow
        # where the PDF is numerically zero.
        per_particle_log_ll[i] = model._active_dist.log_pdf(  # pylint: disable=protected-access
            np.array([observation]), next_state
        )[0]

    np.testing.assert_allclose(batch_log_ll, per_particle_log_ll, atol=1e-12, rtol=0.0)


def test_transition_batch_sample_shape_contract(env):
    """batch_sample returns a 2-D ndarray matching the input shape.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A ContinuousLightDarkStateTransitionModel and a (37, 2)
        particles ndarray.
    When: batch_sample is called.
    Then: The result is an ndarray of shape (37, 2) and dtype float64.

    Test type: unit
    """
    transition = env.state_transition_model(state=np.array([5.0, 5.0]), action=np.array([0.5, 0.0]))
    particles = np.zeros((37, 2), dtype=np.float64)
    _native.set_seed(0)
    result = transition.batch_sample(particles)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37, 2)
    assert result.dtype == np.float64


def test_observation_batch_log_likelihood_shape_contract(env):
    """batch_log_likelihood returns a 1-D ndarray of length N.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A ContinuousLightDarkNormalNoiseObservationModel, 37 next-particles,
        and one observation.
    When: batch_log_likelihood is called.
    Then: The result is an ndarray of shape (37,) and dtype float64.

    Test type: unit
    """
    obs_model = env.observation_model(next_state=np.array([5.0, 5.0]), action=np.array([0.0, 0.0]))
    next_particles = np.zeros((37, 2), dtype=np.float64)
    observation = np.array([5.0, 5.0])
    result = obs_model.batch_log_likelihood(next_particles, observation)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37,)
    assert result.dtype == np.float64
