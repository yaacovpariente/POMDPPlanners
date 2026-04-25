"""Numerical equivalence tests for the native Safety Ant Velocity extension.

Verifies that the C++ ``_native`` sampling / log-likelihood code matches the
Python reference (``CovarianceParameterizedMultivariateNormal`` for the
observation model; analytic damped-force integration for the transition)
up to statistical noise in the samples and up to floating-point tolerance.

Complements ``test_safety_ant_velocity_pomdp.py`` which exercises the
shipped API; this module focuses on the port-specific invariants.
Generic assertion mechanics live in ``_native_parity.py``.
"""

import numpy as np
import pytest

from POMDPPlanners.core.environment import ObservationModel, StateTransitionModel
from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
    SafeAntVelocityObservation,
    SafeAntVelocityPOMDP,
    SafeAntVelocityStateTransition,
    _native,
)
from POMDPPlanners.tests.test_environments import _native_parity
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


@pytest.fixture(name="env")
def _env_fixture():
    return SafeAntVelocityPOMDP(discount_factor=0.99)


@pytest.fixture(name="transition")
def _transition_fixture(env):
    state = np.array([0.0, 0.0, 0.5, -0.3])
    return env.state_transition_model(state=state, action=2)


@pytest.fixture(name="observation")
def _observation_fixture(env):
    next_state = np.array([0.1, -0.1, 0.5, 0.3])
    return env.observation_model(next_state=next_state, action=1)


# ---------------------------------------------------------------------------
# ABC / typing contract
# ---------------------------------------------------------------------------


def test_transition_is_registered_as_state_transition_model(transition):
    """Shim passes isinstance check for StateTransitionModel after register().

    Purpose: Validates the ABC virtual-subclass registration applied in the
    Python shim so downstream polymorphic callers see the C++ class as a
    valid StateTransitionModel.

    Given: A SafeAntVelocityStateTransition produced by the env factory.
    When: isinstance is checked against StateTransitionModel.
    Then: It returns True.

    Test type: unit
    """
    _native_parity.assert_abc_registration(transition, StateTransitionModel)


def test_observation_is_registered_as_observation_model(observation):
    """Shim passes isinstance check for ObservationModel after register().

    Purpose: Same as the transition case, for the observation model.

    Given: A SafeAntVelocityObservation produced by the env factory.
    When: isinstance is checked against ObservationModel.
    Then: It returns True.

    Test type: unit
    """
    _native_parity.assert_abc_registration(observation, ObservationModel)


def test_transition_is_not_abstract():
    """SafeAntVelocityStateTransition is instantiable (no unresolved abstract methods).

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The SafeAntVelocityStateTransition class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(SafeAntVelocityStateTransition, "__abstractmethods__", frozenset())
    assert abstract_methods == frozenset()


def test_observation_is_not_abstract():
    """SafeAntVelocityObservation is instantiable (no unresolved abstract methods).

    Purpose: Guards against ABC.register masking missing slot implementations
    on the C++ side.

    Given: The SafeAntVelocityObservation class object.
    When: __abstractmethods__ is inspected.
    Then: The set is empty.

    Test type: unit
    """
    abstract_methods = getattr(SafeAntVelocityObservation, "__abstractmethods__", frozenset())
    assert abstract_methods == frozenset()


# ---------------------------------------------------------------------------
# Transition: shape, determinism, zero-force degeneracy, ring statistics
# ---------------------------------------------------------------------------


def test_transition_sample_returns_list_of_1d_ndarrays(transition):
    """sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift (call sites index with [0]
    or iterate the list).

    Given: A SafeAntVelocityStateTransition.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of length 4.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=4,
    )


def test_transition_sample_is_deterministic_under_set_seed(transition):
    """Repeated set_seed + sample yields identical samples.

    Purpose: Validates the determinism path used by reproducible tests.

    Given: A SafeAntVelocityStateTransition.
    When: _native.set_seed(seed) is called before each of two sample(50) draws.
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    _native_parity.assert_determinism_under_seed(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=50,
        seed=2024,
    )


def test_transition_batch_sample_shape_contract(env):
    """batch_sample returns a 2-D ndarray of shape (N, 4) with dtype float64.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A SafeAntVelocityStateTransition and a (37, 4) particles ndarray.
    When: batch_sample is called.
    Then: The result is an ndarray of shape (37, 4) and dtype float64.

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.0, 0.0])
    transition = env.state_transition_model(state=state, action=1)
    particles = np.zeros((37, 4), dtype=np.float64)
    _native.set_seed(0)
    result = transition.batch_sample(particles)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37, 4)
    assert result.dtype == np.float64


def test_transition_zero_force_is_deterministic(env):
    """action=0 yields a damping-only deterministic transition (no random draw).

    Purpose: Validates that with zero force magnitude the sampler degenerates
    to a deterministic damped-integration step, independent of the RNG.

    Given: A particle with non-zero velocity and action=0.
    When: batch_sample is called with 100 copies of that particle.
    Then: All rows equal the analytic damped-integration result.

    Test type: unit
    """
    state = np.array([0.1, -0.2, 1.0, 0.5])
    transition = env.state_transition_model(state=state, action=0)
    particles = np.tile(state, (100, 1))
    _native.set_seed(99)
    result = transition.batch_sample(particles)

    damping_accel = -env.damping * state[2:4] / env.mass
    expected_vel = state[2:4] + damping_accel * env.dt
    expected_pos = state[:2] + expected_vel * env.dt
    expected_row = np.concatenate([expected_pos, expected_vel])

    np.testing.assert_allclose(result, np.tile(expected_row, (100, 1)), atol=1e-12)


def test_transition_batch_sample_velocity_magnitude_matches_ring_radius(env):
    """Non-zero-action batch_sample velocity magnitudes concentrate on the ring radius.

    Purpose: Validates the uniform-on-a-ring distribution: every sample's
    delta-velocity magnitude should equal ``force_magnitude * dt / mass``
    (exactly, up to floating point) regardless of the sampled force angle.

    Given: Starting from zero velocity with action=3 (max force), drawing
        100k transitions via batch_sample.
    When: We take the L2 norm of (new_velocity - old_velocity) per row.
    Then: Every norm equals force_magnitude * dt / mass within 1e-10.

    Test type: integration
    """
    state = np.zeros(4)
    transition = env.state_transition_model(state=state, action=3)
    particles = np.tile(state, (100_000, 1))

    _native.set_seed(123)
    result = transition.batch_sample(particles)
    delta_velocity = result[:, 2:4] - particles[:, 2:4]
    magnitudes = np.linalg.norm(delta_velocity, axis=1)

    force_scales = np.array([0.0, 0.33, 0.67, 1.0])
    expected_magnitude = force_scales[3] * env.max_force * env.dt / env.mass
    np.testing.assert_allclose(magnitudes, expected_magnitude, atol=1e-10)


def test_transition_batch_sample_force_direction_is_uniform_on_circle(env):
    """Force directions drawn by batch_sample are uniformly distributed on [-pi, pi].

    Purpose: Validates the uniform-angle RNG draw: histogramming the angles
    of the sampled delta-velocity vectors should produce an approximately
    uniform distribution (flat chi-square against uniform).

    Given: 100k batch_sample draws from zero state with action=3.
    When: We compute the angle of each delta-velocity vector.
    Then: A 20-bin histogram of angles is flat within ~3% of uniform.

    Test type: integration
    """
    state = np.zeros(4)
    transition = env.state_transition_model(state=state, action=3)
    particles = np.tile(state, (100_000, 1))

    _native.set_seed(456)
    result = transition.batch_sample(particles)
    delta_velocity = result[:, 2:4] - particles[:, 2:4]
    angles = np.arctan2(delta_velocity[:, 1], delta_velocity[:, 0])

    n_bins = 20
    counts, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    expected_per_bin = 100_000 / n_bins
    np.testing.assert_allclose(counts, expected_per_bin, rtol=0.04)


def test_transition_batch_sample_matches_per_particle_loop_under_shared_seed(env):
    """batch_sample is bit-identical to a per-row sample(1) loop under a single seed.

    Purpose: Validates that SafeAntVelocityTransitionCpp.batch_sample consumes
    the module-level C++ RNG in the same order as N sequential sample(1) calls
    (one per particle, with a fresh transition rebuilt per row).

    Given: 64 randomly-placed particles.
    When: batch_sample runs under seed=777; then for each particle a fresh
        SafeAntVelocityStateTransition is built and sample(1) is called in
        the same order under the same seed.
    Then: The two (64, 4) arrays are array_equal.

    Test type: unit
    """
    rng = np.random.default_rng(123)
    particles = np.column_stack(
        [
            rng.uniform(-1, 1, 64),
            rng.uniform(-1, 1, 64),
            rng.uniform(-0.5, 0.5, 64),
            rng.uniform(-0.5, 0.5, 64),
        ]
    )
    action = 2

    _native.set_seed(777)
    transition = env.state_transition_model(state=particles[0], action=action)
    batch_result = transition.batch_sample(particles)

    _native.set_seed(777)
    per_particle_rows = []
    for row in particles:
        model = env.state_transition_model(state=row, action=action)
        per_particle_rows.append(model.sample(1)[0])
    per_particle_result = np.stack(per_particle_rows, axis=0)

    np.testing.assert_array_equal(batch_result, per_particle_result)


# ---------------------------------------------------------------------------
# Observation: moments, bit-exact PDF, batch parity
# ---------------------------------------------------------------------------


def test_observation_sample_empirical_moments_match_spec(env):
    """Native observation samples match the documented Gaussian noise.

    Purpose: Validates that 200k observation samples have the theoretical
    mean (the input next_state) and covariance (diag of pos/vel noises).

    Given: A SafeAntVelocityObservation with a fixed next_state.
    When: 200,000 samples are drawn via the C++ path.
    Then: The empirical mean matches the next_state within 5e-3, and the
        empirical covariance matches the observation covariance within
        Frobenius-norm 1e-2.

    Test type: integration
    """
    next_state = np.array([0.05, 0.1, -0.3, 0.4])
    obs_model = env.observation_model(next_state=next_state, action=0)
    expected_cov = np.diag(
        [
            env.position_noise**2,
            env.position_noise**2,
            env.velocity_noise**2,
            env.velocity_noise**2,
        ]
    )

    _native_parity.assert_sample_moments_match(
        model=obs_model,
        seed_fn=_native.set_seed,
        expected_mean=next_state,
        expected_cov=expected_cov,
        n_samples=200_000,
        seed=7777,
        mean_atol=5e-3,
        cov_frobenius_atol=1e-2,
    )


def test_observation_probability_matches_python_reference_on_grid(env):
    """C++ observation probability() matches CovarianceParameterizedMultivariateNormal.pdf.

    Purpose: Validates that the C++ PDF computation agrees numerically with
    the Python reference on a deterministic grid.

    Given: A 1000-point grid of candidate observations around the true
        next_state, plus the Python reference distribution constructed from
        the same diagonal covariance.
    When: probability(grid) is called on both implementations.
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    next_state = np.array([0.02, 0.05, -0.01, 0.1])
    obs_model = env.observation_model(next_state=next_state, action=1)
    cov = np.diag(
        [
            env.position_noise**2,
            env.position_noise**2,
            env.velocity_noise**2,
            env.velocity_noise**2,
        ]
    )
    py_ref = CovarianceParameterizedMultivariateNormal(cov)

    rng = np.random.default_rng(4242)
    offsets = rng.normal(size=(1000, 4)) * np.array(
        [env.position_noise, env.position_noise, env.velocity_noise, env.velocity_noise]
    )
    grid = next_state[np.newaxis, :] + offsets

    _native_parity.assert_logpdf_bitwise(
        model=obs_model,
        reference_dist=py_ref,
        mean=next_state,
        points=grid,
        atol=1e-10,
    )


def test_observation_batch_log_likelihood_matches_per_particle(env):
    """batch_log_likelihood matches np.log(probability([observation])) per row.

    Purpose: Validates the observation batch path against the per-particle
    reference (C++ probability() uses the same log_pdf internally, then
    exp's it; we take log to invert).

    Given: 64 random next-state particles and a single observation.
    When: batch_log_likelihood(next_particles, observation) is called, and
        for each particle a fresh SafeAntVelocityObservation is built and
        probability([observation]) is computed.
    Then: The batch log-likelihoods equal np.log of the per-particle
        probabilities within atol=1e-12.

    Test type: unit
    """
    rng = np.random.default_rng(456)
    next_particles = np.column_stack(
        [
            rng.uniform(-1, 1, 64),
            rng.uniform(-1, 1, 64),
            rng.uniform(-0.5, 0.5, 64),
            rng.uniform(-0.5, 0.5, 64),
        ]
    )
    observation = np.array([0.02, 0.0, -0.01, 0.0])
    action = 0

    obs_model = env.observation_model(next_state=next_particles[0], action=action)
    batch_log_ll = obs_model.batch_log_likelihood(next_particles, observation)

    per_particle_log_ll = np.empty(len(next_particles))
    for i, next_state in enumerate(next_particles):
        model = env.observation_model(next_state=next_state, action=action)
        per_particle_log_ll[i] = np.log(model.probability([observation])[0])

    np.testing.assert_allclose(batch_log_ll, per_particle_log_ll, atol=1e-12, rtol=0.0)


def test_observation_batch_log_likelihood_shape_contract(env):
    """batch_log_likelihood returns a 1-D ndarray of length N with dtype float64.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A SafeAntVelocityObservation, 37 next-particles, and one observation.
    When: batch_log_likelihood is called.
    Then: The result is an ndarray of shape (37,) and dtype float64.

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.0, 0.0])
    obs_model = env.observation_model(next_state=next_state, action=1)
    next_particles = np.zeros((37, 4), dtype=np.float64)
    observation = np.array([0.0, 0.0, 0.0, 0.0])
    result = obs_model.batch_log_likelihood(next_particles, observation)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37,)
    assert result.dtype == np.float64


def test_observation_probability_accepts_list_and_ndarray_inputs(observation):
    """probability() accepts both a list of 1-D arrays and a 2-D ndarray.

    Purpose: Validates the Python-side input flexibility of the C++ binding
    (matches the pre-port duck-typed contract).

    Given: A list of three 1-D numpy arrays and the equivalent stacked 2-D array.
    When: probability() is called with each form.
    Then: Outputs are identical.

    Test type: unit
    """
    list_values = [
        np.array([0.1, -0.1, 0.5, 0.3]),
        np.array([0.11, -0.09, 0.51, 0.29]),
        np.array([0.09, -0.11, 0.49, 0.31]),
    ]
    array_values = np.stack(list_values, axis=0)

    from_list = observation.probability(list_values)
    from_array = observation.probability(array_values)

    np.testing.assert_array_equal(from_list, from_array)


# ---------------------------------------------------------------------------
# Hot-path sample_next_state / sample_observation overrides
# (skip Python wrapper, route directly to native kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,action",
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),  # zero force -> deterministic
        (np.array([0.5, -0.2, 1.0, 0.5]), 1),
        (np.array([-0.3, 0.4, 0.7, -0.6]), 2),
        (np.array([0.1, 0.1, 1.5, 0.8]), 3),
    ],
)
def test_sample_next_state_override_matches_wrapper(env, state, action):
    """Override sample_next_state matches the per-call wrapper bit-exactly.

    Purpose: Validates that ``SafeAntVelocityPOMDP.sample_next_state``
    (which constructs the native transition kernel directly, skipping the
    Python wrapper) produces the exact same next state as the legacy
    ``state_transition_model(state, action).sample()[0]`` path under a
    fixed C++ RNG seed.

    Given: The shared SafeAnt env fixture and a parametrized
        ``(state, action)`` pair covering zero-force and varied-force
        regimes.
    When: Both paths are invoked with ``_native.set_seed`` reset to the
        same value before each call.
    Then: ``np.array_equal`` holds elementwise on the returned arrays.

    Test type: integration
    """
    _native.set_seed(2024)
    via_wrapper = env.state_transition_model(state=state, action=action).sample()[0]

    _native.set_seed(2024)
    via_override = env.sample_next_state(state=state, action=action)

    np.testing.assert_array_equal(via_wrapper, via_override)


@pytest.mark.parametrize(
    "next_state,action",
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([0.5, -0.2, 1.0, 0.5]), 1),
        (np.array([-0.3, 0.4, 0.7, -0.6]), 2),
        (np.array([0.1, 0.1, 1.5, 0.8]), 3),
    ],
)
def test_sample_observation_override_matches_wrapper(env, next_state, action):
    """Override sample_observation matches the per-call wrapper bit-exactly.

    Purpose: Validates that ``SafeAntVelocityPOMDP.sample_observation``
    (which constructs the native observation kernel directly, skipping
    the Python wrapper) produces the exact same observation as the legacy
    ``observation_model(...).sample()[0]`` path under a fixed C++ RNG seed.

    Given: The shared env fixture plus a parametrized
        ``(next_state, action)`` pair.
    When: Both paths are invoked with ``_native.set_seed`` reset to the
        same value before each call.
    Then: ``np.array_equal`` holds elementwise.

    Test type: integration
    """
    _native.set_seed(7777)
    via_wrapper = env.observation_model(next_state=next_state, action=action).sample()[0]

    _native.set_seed(7777)
    via_override = env.sample_observation(next_state=next_state, action=action)

    np.testing.assert_array_equal(via_wrapper, via_override)
