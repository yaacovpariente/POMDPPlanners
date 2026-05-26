# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Numerical equivalence tests for the native CartPole sampling extension.

Verifies that the C++ ``_native`` sampling / probability code matches the
Python reference (``CovarianceParameterizedMultivariateNormal``) up to
statistical noise in the samples and up to floating-point tolerance in
the PDF. Complements ``test_cartpole_pomdp.py`` which exercises the
shipped API; this module focuses on the port-specific invariants.
Generic assertion mechanics live in ``_native_parity.py`` so other native
env ports can reuse them.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP, _native
from POMDPPlanners.tests.test_environments import _native_parity
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


@pytest.fixture(name="env")
def _env_fixture():
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    return CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)


def _build_transition(env: CartPolePOMDP, state: np.ndarray, action: int):
    return _native.CartPoleTransitionCpp(
        state=state,
        action=action,
        force_mag=env.force_mag,
        total_mass=env.total_mass,
        polemass_length=env.polemass_length,
        gravity=env.gravity,
        length=env.length,
        kinematics_integrator=env.kinematics_integrator,
        tau=env.tau,
        masspole=env.masspole,
        covariance=env.state_transition_cov,
    )


def _build_observation(env: CartPolePOMDP, next_state: np.ndarray, action: int):
    return _native.CartPoleObservationCpp(
        next_state=next_state,
        action=action,
        covariance=env.noise_cov,
    )


@pytest.fixture(name="transition")
def _transition_fixture(env):
    state = np.array([0.0, 0.0, 0.05, 0.0])
    return _build_transition(env, state, action=1)


@pytest.fixture(name="observation")
def _observation_fixture(env):
    next_state = np.array([0.01, 0.0, 0.05, 0.0])
    return _build_observation(env, next_state, action=1)


# ---------------------------------------------------------------------------
# Sample distribution matches the Gaussian spec
# ---------------------------------------------------------------------------


def test_transition_sample_empirical_moments_match_spec(env):
    """Native transition samples have the documented Gaussian distribution.

    Purpose: Validates that 200k samples from the C++ sampler have the
    theoretical mean (deterministic next state) and covariance (the
    declared process noise), since the C++ RNG cannot be compared
    bit-for-bit against numpy's.

    Given: A native CartPoleTransitionCpp from an interior state with
        action=1 (no boundary effects).
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches the deterministic next state within
        5e-3, and the empirical covariance matches the configured process
        noise within a Frobenius-norm tolerance of 1e-5.

    Test type: integration
    """
    state = np.array([0.0, 0.0, 0.05, 0.0])
    transition = _build_transition(env, state, action=1)
    det = np.asarray(
        transition._compute_deterministic_next_state()  # pylint: disable=protected-access
    )

    _native_parity.assert_sample_moments_match(
        model=transition,
        seed_fn=_native.set_seed,
        expected_mean=det,
        expected_cov=env.state_transition_cov,
        n_samples=200_000,
        seed=12345,
        mean_atol=5e-3,
        cov_frobenius_atol=1e-5,
    )


def test_observation_sample_empirical_moments_match_spec(env):
    """Native observation samples match the documented observation noise.

    Purpose: Validates that 200k observation samples have the theoretical
    mean (the input next_state) and covariance (the declared observation
    noise).

    Given: A native CartPoleObservationCpp with a fixed next_state.
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches the next_state within 5e-3, and the
        empirical covariance matches the observation covariance within
        Frobenius-norm 1e-2.

    Test type: integration
    """
    next_state = np.array([0.05, 0.1, -0.02, 0.0])
    observation_model = _build_observation(env, next_state, action=0)

    _native_parity.assert_sample_moments_match(
        model=observation_model,
        seed_fn=_native.set_seed,
        expected_mean=next_state,
        expected_cov=env.noise_cov,
        n_samples=200_000,
        seed=7777,
        mean_atol=5e-3,
        cov_frobenius_atol=1e-2,
    )


# ---------------------------------------------------------------------------
# probability() bit-exactness vs the Python reference
# ---------------------------------------------------------------------------


def test_transition_probability_matches_python_reference_on_grid(env):
    """C++ probability() matches CovarianceParameterizedMultivariateNormal.pdf.

    Purpose: Validates that the C++ PDF computation agrees numerically with
    the Python reference on a deterministic grid.

    Given: A 1000-point grid of candidate next-states around the
        deterministic transition target, plus the Python reference
        distribution constructed from the same covariance.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-6 (PDF values are large
        relative to the log-pdf difference tolerance).

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.05, 0.0])
    transition = _build_transition(env, state, action=1)
    det = np.asarray(
        transition._compute_deterministic_next_state()  # pylint: disable=protected-access
    )

    rng = np.random.default_rng(42)
    offsets = rng.normal(size=(1000, 4)) * np.array([0.005, 0.005, 0.002, 0.005])
    grid = det[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(env.state_transition_cov)
    _native_parity.assert_logpdf_bitwise(
        model=transition,
        reference_dist=py_ref,
        mean=det,
        points=grid,
        atol=1e-6,
    )


def test_observation_probability_matches_python_reference_on_grid(env):
    """C++ observation probability() matches the Python reference on a grid.

    Purpose: Same as the transition case, for the observation model.

    Given: A 1000-point grid of candidate observations around the true state.
    When: probability(grid) is called on both C++ and Python reference.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([0.02, 0.05, -0.01, 0.1])
    observation_model = _build_observation(env, next_state, action=1)

    rng = np.random.default_rng(4242)
    offsets = rng.normal(size=(1000, 4)) * np.array([0.1, 0.1, 0.05, 0.1])
    grid = next_state[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(env.noise_cov)
    _native_parity.assert_logpdf_bitwise(
        model=observation_model,
        reference_dist=py_ref,
        mean=next_state,
        points=grid,
        atol=1e-10,
    )


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
        np.array([0.0, 0.0, 0.05, 0.0]),
        np.array([0.001, -0.002, 0.048, 0.003]),
        np.array([-0.001, 0.003, 0.052, -0.002]),
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

    Given: Two CartPoleTransitionCpp instances sampled under the same seed.
    When: _native.set_seed(seed) is called before each, followed by sample(50).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    state = np.array([0.01, 0.005, 0.03, -0.01])
    transition = _build_transition(env, state, action=1)
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

    Given: A native CartPoleTransitionCpp.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 4.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=4,
    )


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------


def test_transition_batch_sample_matches_per_particle_sample(env):
    """batch_sample(P) is bit-identical to sample(1) per row under fixed seed.

    Purpose: Validates that the TransitionModelCpp<Dim>.batch_sample produces
    the same noise sequence as N per-particle sample(1) calls when both
    consume the same module-level C++ RNG seeded once.

    Given: 64 particles spanning a range of CartPole states.
    When: batch_sample runs under seed=777; then for each particle a fresh
        CartPoleTransitionCpp is built and sample(1) is called in the same
        order under the same seed.
    Then: The two (64, 4) arrays are array_equal.

    Test type: unit
    """
    rng = np.random.default_rng(123)
    particles = np.column_stack(
        [
            rng.uniform(-0.05, 0.05, 64),
            rng.uniform(-0.05, 0.05, 64),
            rng.uniform(-0.1, 0.1, 64),
            rng.uniform(-0.05, 0.05, 64),
        ]
    )
    action = 1

    _native.set_seed(777)
    transition = _build_transition(env, particles[0], action=action)
    batch_result = transition.batch_sample(particles)

    _native.set_seed(777)
    per_particle_rows = []
    for row in particles:
        model = _build_transition(env, row, action=action)
        per_particle_rows.append(model.sample(1)[0])
    per_particle_result = np.stack(per_particle_rows, axis=0)

    np.testing.assert_array_equal(batch_result, per_particle_result)


def test_observation_batch_log_likelihood_matches_per_particle(env):
    """batch_log_likelihood matches np.log(probability([observation])) per row.

    Purpose: Validates the observation batch path against the per-particle
    reference (C++ probability() uses the same log_pdf internally, then
    exp's it; we take log to invert).

    Given: 64 random next-state particles and a single observation.
    When: batch_log_likelihood(next_particles, observation) is called, and
        for each particle a fresh CartPoleObservationCpp is built and
        probability([observation]) is computed.
    Then: The batch log-likelihoods equal np.log of the per-particle
        probabilities within atol=1e-12.

    Test type: unit
    """
    rng = np.random.default_rng(456)
    next_particles = np.column_stack(
        [
            rng.uniform(-0.1, 0.1, 64),
            rng.uniform(-0.1, 0.1, 64),
            rng.uniform(-0.15, 0.15, 64),
            rng.uniform(-0.1, 0.1, 64),
        ]
    )
    observation = np.array([0.02, 0.0, -0.01, 0.0])
    action = 0

    obs_model = _build_observation(env, next_particles[0], action=action)
    batch_log_ll = obs_model.batch_log_likelihood(next_particles, observation)

    per_particle_log_ll = np.empty(len(next_particles))
    for i, next_state in enumerate(next_particles):
        model = _build_observation(env, next_state, action=action)
        per_particle_log_ll[i] = np.log(model.probability([observation])[0])

    np.testing.assert_allclose(batch_log_ll, per_particle_log_ll, atol=1e-12, rtol=0.0)


def test_transition_batch_sample_shape_contract(env):
    """batch_sample returns a 2-D ndarray matching the input shape.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A native CartPoleTransitionCpp and a (37, 4) particles ndarray.
    When: batch_sample is called.
    Then: The result is an ndarray of shape (37, 4) and dtype float64.

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.05, 0.0])
    transition = _build_transition(env, state, action=1)
    particles = np.zeros((37, 4), dtype=np.float64)
    _native.set_seed(0)
    result = transition.batch_sample(particles)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37, 4)
    assert result.dtype == np.float64


def test_observation_batch_log_likelihood_shape_contract(env):
    """batch_log_likelihood returns a 1-D ndarray of length N.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A native CartPoleObservationCpp, 37 next-particles, and one observation.
    When: batch_log_likelihood is called.
    Then: The result is an ndarray of shape (37,) and dtype float64.

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.05, 0.0])
    obs_model = _build_observation(env, next_state, action=1)
    next_particles = np.zeros((37, 4), dtype=np.float64)
    observation = np.array([0.0, 0.0, 0.05, 0.0])
    result = obs_model.batch_log_likelihood(next_particles, observation)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37,)
    assert result.dtype == np.float64
