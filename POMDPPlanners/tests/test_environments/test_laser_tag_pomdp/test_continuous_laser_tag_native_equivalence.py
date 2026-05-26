# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Numerical equivalence tests for the native Continuous LaserTag extension.

Verifies that the C++ ``_native`` sampling / probability code matches the
Python reference (geometry helpers in ``continuous_laser_tag_geometry`` +
``CovarianceParameterizedMultivariateNormal``) up to statistical noise in
the samples and up to floating-point tolerance in the PDF / log-likelihood
paths. Generic assertion helpers live in ``_native_parity.py``; this module
wires the laser-tag-specific C++ kernels through them and adds the batch-
entry parity tests called out in the implementation plan.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    compute_laser_measurements,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
)
from POMDPPlanners.tests.test_environments import _native_parity
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


@pytest.fixture(name="env")
def _env_fixture():
    # Use an empty-walls environment so the geometry is purely grid-bounded;
    # per-test fixtures below cover the populated-walls case separately.
    return ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[])


def _build_transition_kernel(
    env: ContinuousLaserTagPOMDP,
    state: np.ndarray,
    action: np.ndarray,
) -> "_native.ContinuousLaserTagTransitionCpp":
    """Construct a fresh C++ transition kernel matching the env's parameters."""
    return _native.ContinuousLaserTagTransitionCpp(
        state=np.asarray(state, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        robot_covariance=env.robot_transition_cov_matrix,
        opponent_covariance=env.opponent_transition_cov_matrix,
        pursuit_speed=env.pursuit_speed,
        walls=np.asarray(env.walls, dtype=np.float64).reshape(-1, 4),
        grid_size=np.asarray(env.grid_size, dtype=np.float64),
        robot_radius=env.robot_radius,
        opponent_radius=env.opponent_radius,
        tag_radius=env.tag_radius,
    )


def _build_observation_kernel(
    env: ContinuousLaserTagPOMDP,
    next_state: np.ndarray,
    action: np.ndarray,
) -> "_native.ContinuousLaserTagObservationCpp":
    """Construct a fresh C++ observation kernel matching the env's parameters."""
    return _native.ContinuousLaserTagObservationCpp(
        next_state=np.asarray(next_state, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        measurement_noise=env.measurement_noise,
        walls=np.asarray(env.walls, dtype=np.float64).reshape(-1, 4),
        grid_size=np.asarray(env.grid_size, dtype=np.float64),
        opponent_radius=env.opponent_radius,
    )


@pytest.fixture(name="transition")
def _transition_fixture(env):
    state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    return _build_transition_kernel(env, state, action)


@pytest.fixture(name="observation")
def _observation_fixture(env):
    next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    return _build_observation_kernel(env, next_state, action)


# ---------------------------------------------------------------------------
# Sample distribution matches the Gaussian spec
# ---------------------------------------------------------------------------


def test_transition_robot_move_moments_match_spec(env):
    """Empirical mean / cov of the robot-position slice match the declared spec.

    Purpose: Validates that the robot Gaussian step inside the native
    transition sampler has the documented mean (robot_pos + action[:2])
    and covariance (env.robot_transition_cov_matrix), since the C++ RNG
    cannot be compared bit-for-bit against numpy's.

    Given: A non-terminal state with robot far from walls and the
        opponent placed on top of the robot (which forces a random-unit
        pursuit direction but does NOT move the robot). The empty-walls
        fixture ensures no collision push-out skews the moments.
    When: 200,000 transition samples are drawn via the C++ path with the
        module-level RNG seeded once.
    Then: The empirical mean of the robot-position slice (columns 0:2) is
        within 5e-3 of (robot_pos + action[:2]), and the empirical
        covariance is within Frobenius-norm 5e-3 of the configured
        robot_transition_cov_matrix.

    Test type: integration
    """
    state = np.array([3.0, 3.0, 3.0, 3.0, 0.0])
    action = np.array([0.5, -0.2, 0.0])
    transition = _build_transition_kernel(env, state, action)

    _native.set_seed(12345)
    samples = np.asarray(transition.sample(200_000))
    robot_samples = samples[:, :2]

    expected_mean = np.array([3.5, 2.8])
    np.testing.assert_allclose(robot_samples.mean(axis=0), expected_mean, atol=5e-3)
    empirical_cov = np.cov(robot_samples, rowvar=False)
    frob_err = np.linalg.norm(empirical_cov - env.robot_transition_cov_matrix, ord="fro")
    assert frob_err < 5e-3, f"Covariance Frobenius error {frob_err:.3e}"


def test_observation_mean_matches_python_geometry(env):
    """Native observation mean matches the Python laser geometry routine.

    Purpose: Validates that the C++ 8-direction laser scan agrees with the
    Python reference in ``continuous_laser_tag_geometry`` for a non-terminal
    state. A matching scan guarantees the observation model's Gaussian
    centre is bit-identical across both paths.

    Given: A non-terminal state with the robot in the arena interior.
    When: ``observation.mean`` is read from the native kernel and
        ``compute_laser_measurements`` is called on the same state in Python.
    Then: The two 8-vectors agree to absolute tolerance 1e-12.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    obs_model = _build_observation_kernel(env, state, action)

    native_mean = np.asarray(obs_model.mean)
    python_mean = compute_laser_measurements(
        state[:2], state[2:4], env.opponent_radius, env.walls, env.grid_size
    )
    np.testing.assert_allclose(native_mean, python_mean, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# probability() bit-exactness vs the Python reference
# ---------------------------------------------------------------------------


def test_observation_probability_matches_python_reference_on_grid(env):
    """C++ observation probability matches an isotropic-Gaussian reference on a grid.

    Purpose: Validates that the C++ observation PDF agrees numerically with
    a Python reference ``CovarianceParameterizedMultivariateNormal`` centred
    on the same laser-geometry mean, which is the model the environment's
    observation noise implements.

    Given: A 500-point grid of candidate observations around the true laser
        measurement.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    obs_model = _build_observation_kernel(env, next_state, action)

    mean = compute_laser_measurements(
        next_state[:2], next_state[2:4], env.opponent_radius, env.walls, env.grid_size
    )
    rng = np.random.default_rng(42)
    offsets = rng.normal(size=(500, 8)) * 0.5
    grid = mean[np.newaxis, :] + offsets

    py_ref = CovarianceParameterizedMultivariateNormal(np.eye(8) * env.measurement_noise**2)
    _native_parity.assert_logpdf_bitwise(
        model=obs_model,
        reference_dist=py_ref,
        mean=mean,
        points=grid,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Determinism under explicit seeding
# ---------------------------------------------------------------------------


def test_transition_sample_is_deterministic_under_set_seed(env):
    """Repeated set_seed + sample yields identical samples.

    Purpose: Validates the determinism path used by reproducible tests.

    Given: Two transitions sampled under the same ``_native.set_seed`` seed.
    When: _native.set_seed(seed) is called before each, followed by sample(32).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.5, 0.0])
    transition = _build_transition_kernel(env, state, action)
    _native_parity.assert_determinism_under_seed(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=32,
        seed=2024,
    )


def test_observation_sample_is_deterministic_under_set_seed(env):
    """Repeated set_seed + sample yields identical observation samples.

    Purpose: Validates the determinism path for observations.

    Given: Two observations sampled under the same seed.
    When: _native.set_seed(seed) is called before each, followed by sample(32).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    obs_model = _build_observation_kernel(env, state, action)
    _native_parity.assert_determinism_under_seed(
        model=obs_model,
        seed_fn=_native.set_seed,
        n_samples=32,
        seed=7777,
    )


def test_transition_sample_returns_list_of_1d_ndarrays(transition):
    """transition.sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift on the transition kernel.

    Given: A native ContinuousLaserTagTransitionCpp kernel.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 5.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=5,
    )


def test_observation_sample_returns_list_of_1d_ndarrays(observation):
    """observation.sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift on the observation kernel.

    Given: A native ContinuousLaserTagObservationCpp kernel.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 8.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=observation,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=8,
    )


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------


def test_transition_batch_sample_matches_per_particle_sample(env):
    """batch_sample(P) is bit-identical to sample(1) per row under fixed seed.

    Purpose: Validates that the native batch_sample produces the same noise
    sequence as N per-particle sample(1) calls when both consume the same
    module-level C++ RNG seeded once. This is the post-port guarantee
    that closes the cross-path RNG divergence the pre-port tests had to
    work around with per-particle seeding.

    Given: 32 particles spanning the continuous arena.
    When: batch_sample runs under _native.set_seed(777); then for each
        particle a fresh transition kernel is built and sample(1) is
        called in the same order under the same seed.
    Then: The two (32, 5) arrays are bit-exact under np.testing.assert_array_equal.

    Test type: unit
    """
    rng = np.random.default_rng(321)
    particles = np.column_stack(
        [
            rng.uniform(1.0, 9.0, 32),
            rng.uniform(1.0, 5.0, 32),
            rng.uniform(1.0, 9.0, 32),
            rng.uniform(1.0, 5.0, 32),
            np.zeros(32),
        ]
    )
    action = np.array([1.0, 0.5, 0.0])

    _native.set_seed(777)
    transition = _build_transition_kernel(env, particles[0], action)
    batch_result = transition.batch_sample(particles)

    _native.set_seed(777)
    per_particle_rows = []
    for row in particles:
        model = _build_transition_kernel(env, row, action)
        per_particle_rows.append(model.sample(1)[0])
    per_particle_result = np.stack(per_particle_rows, axis=0)

    np.testing.assert_array_equal(batch_result, per_particle_result)


def test_observation_batch_log_likelihood_matches_per_particle(env):
    """batch_log_likelihood matches np.log(probability([observation])) per row.

    Purpose: Validates the observation batch path against the per-particle
    reference (C++ probability() uses the same log_pdf internally, then
    exp's it; we take log to invert).

    Given: 32 random non-terminal next-state particles and a single
        non-terminal observation.
    When: batch_log_likelihood(next_particles, observation) is called, and
        for each particle a fresh observation kernel is built and
        probability([observation]) is computed.
    Then: The batch log-likelihoods equal np.log of the per-particle
        probabilities within atol=1e-10.

    Test type: unit
    """
    rng = np.random.default_rng(654)
    next_particles = np.column_stack(
        [
            rng.uniform(1.0, 9.0, 32),
            rng.uniform(1.0, 5.0, 32),
            rng.uniform(1.0, 9.0, 32),
            rng.uniform(1.0, 5.0, 32),
            np.zeros(32),
        ]
    )
    observation = rng.uniform(0.1, 5.0, 8)
    action = np.array([1.0, 0.0, 0.0])

    obs_model = _build_observation_kernel(env, next_particles[0], action)
    batch_log_ll = obs_model.batch_log_likelihood(next_particles, observation)

    per_particle_log_ll = np.empty(len(next_particles))
    for i, next_state in enumerate(next_particles):
        model = _build_observation_kernel(env, next_state, action)
        per_particle_log_ll[i] = np.log(model.probability([observation])[0])

    np.testing.assert_allclose(batch_log_ll, per_particle_log_ll, atol=1e-10, rtol=0.0)


def test_transition_batch_sample_shape_contract(env):
    """batch_sample returns a 2-D ndarray matching the input shape.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A transition kernel and a (41, 5) particles ndarray.
    When: batch_sample is called.
    Then: The result is an ndarray of shape (41, 5) and dtype float64.

    Test type: unit
    """
    state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    transition = _build_transition_kernel(env, state, action)
    particles = np.tile(state, (41, 1))
    _native.set_seed(0)
    result = transition.batch_sample(particles)
    assert isinstance(result, np.ndarray)
    assert result.shape == (41, 5)
    assert result.dtype == np.float64


def test_observation_batch_log_likelihood_shape_contract(env):
    """batch_log_likelihood returns a 1-D ndarray of length N.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: An observation kernel, 41 next-particles, and one observation.
    When: batch_log_likelihood is called.
    Then: The result is an ndarray of shape (41,) and dtype float64.

    Test type: unit
    """
    next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    obs_model = _build_observation_kernel(env, next_state, action)
    next_particles = np.tile(next_state, (41, 1))
    observation = np.full(8, 2.5)
    result = obs_model.batch_log_likelihood(next_particles, observation)
    assert isinstance(result, np.ndarray)
    assert result.shape == (41,)
    assert result.dtype == np.float64


def test_batch_log_likelihood_terminal_semantics(env):
    """Terminal observations / particles get the documented log-likelihoods.

    Purpose: Validates that the native batch_log_likelihood reproduces the
    pre-port terminal-observation special case: terminal particles match
    the all--1 terminal observation with log-likelihood 0.0; non-terminal
    particles against the terminal observation, and terminal particles
    against any other observation, get -inf.

    Given: A mix of terminal (flag=1.0) and non-terminal particles, plus
        both a terminal and a non-terminal observation.
    When: batch_log_likelihood is called with each observation.
    Then: Terminal ↔ terminal pairings give 0.0; all other mixed pairings
        give -inf.

    Test type: unit
    """
    next_state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    obs_model = _build_observation_kernel(env, next_state, action)

    particles = np.array(
        [
            [3.0, 3.0, 8.0, 5.0, 0.0],
            [3.0, 3.0, 8.0, 5.0, 1.0],
            [4.0, 4.0, 6.0, 2.0, 1.0],
        ]
    )

    terminal_obs = np.full(8, -1.0)
    ll_terminal = obs_model.batch_log_likelihood(particles, terminal_obs)
    assert ll_terminal[0] == -np.inf
    assert ll_terminal[1] == 0.0
    assert ll_terminal[2] == 0.0

    nonterminal_obs = np.full(8, 2.0)
    ll_nonterminal = obs_model.batch_log_likelihood(particles, nonterminal_obs)
    assert np.isfinite(ll_nonterminal[0])
    assert ll_nonterminal[1] == -np.inf
    assert ll_nonterminal[2] == -np.inf
