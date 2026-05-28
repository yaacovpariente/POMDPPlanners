# SPDX-License-Identifier: MIT

"""Numerical equivalence tests for the native Continuous Push sampling extension.

Verifies that the C++ ``_native`` sampling / probability code matches the
Python reference (the pre-port numpy implementation) up to statistical
noise in the samples and floating-point tolerance in the PDF. Complements
``test_continuous_push_pomdp.py`` which exercises the shipped env API;
this module focuses on the port-specific invariants (empirical moments,
bit-exact PDF, determinism under seeding, and batch vs per-particle
parity for both transition and observation).

The Python wrapper classes ``ContinuousPushStateTransitionModel`` and
``ContinuousPushObservationModel`` no longer exist; this module
constructs the C++ kernels directly via ``_native.ContinuousPush*Cpp``,
seeded with the same parameters the env would have passed.

Generic assertion mechanics live in ``_native_parity.py`` so the helpers
are shared with MountainCar.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import ContinuousPushPOMDP
from POMDPPlanners.tests.test_environments import _native_parity
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_push_pinned_kwargs
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal


def _make_transition_kernel(env: ContinuousPushPOMDP, state, action):
    """Construct a ``_native.ContinuousPushTransitionCpp`` matching ``env``.

    Replaces the deleted ``env.state_transition_model(...)`` factory; the
    arguments mirror what the wrapper passed to its C++ base.
    """
    return _native.ContinuousPushTransitionCpp(
        state=np.asarray(state, dtype=float).ravel(),
        action=np.asarray(action, dtype=float).ravel(),
        grid_size=float(env.grid_size),
        push_threshold=float(env.push_threshold),
        friction_coefficient=float(env.friction_coefficient),
        max_push=float(env.max_push),
        robot_radius=float(env.robot_radius),
        obstacles=np.asarray(env.obstacles, dtype=float),
        covariance=env._state_transition_dist.covariance,  # pylint: disable=protected-access
    )


def _make_observation_kernel(env: ContinuousPushPOMDP, next_state, action):
    """Construct a ``_native.ContinuousPushObservationCpp`` matching ``env``."""
    return _native.ContinuousPushObservationCpp(
        next_state=np.asarray(next_state, dtype=float).ravel(),
        action=np.asarray(action, dtype=float).ravel(),
        observation_noise=float(env.observation_noise),
        grid_size=float(env.grid_size),
    )


@pytest.fixture(name="env")
def _env_fixture():
    return ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        ),
    )


@pytest.fixture(name="transition")
def _transition_fixture(env):
    state = np.array([2.0, 3.0, 5.0, 5.0, 8.0, 8.0])
    return _make_transition_kernel(env, state=state, action=np.array([1.0, 0.0]))


@pytest.fixture(name="observation")
def _observation_fixture(env):
    state = np.array([2.0, 3.0, 2.8, 3.2, 8.0, 8.0])
    return _make_observation_kernel(env, next_state=state, action=np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Sample distribution matches the Gaussian spec (robot noise only)
# ---------------------------------------------------------------------------


def test_transition_sample_empirical_robot_moments_match_spec(env):
    """Native transition samples: robot slice matches the declared 2-D Gaussian.

    Purpose: Validates that 200k samples drawn via the C++ batch path have the
    theoretical mean (robot_pos + action) and covariance (the declared process
    noise) on the robot slice, since the C++ RNG cannot be compared
    bit-for-bit against numpy's.

    Given: A transition from a state where robot+action stays in-bounds and
        the object is out of push range (so object / target coordinates stay
        deterministic).
    When: 200,000 samples are drawn via the native sampler.
    Then: Robot-slice empirical mean matches robot_pos+action within 5e-3,
        and empirical covariance matches the configured process noise within
        Frobenius-norm 1e-3.

    Test type: integration
    """
    state = np.array([2.0, 3.0, 7.0, 7.0, 8.0, 8.0])  # object far from robot
    action = np.array([1.0, 0.0])
    transition = _make_transition_kernel(env, state=state, action=action)
    expected_robot_mean = state[:2] + action

    _native.set_seed(12345)
    samples = np.asarray(transition.sample(200_000))
    robot_samples = samples[:, :2]
    empirical_mean = robot_samples.mean(axis=0)
    empirical_cov = np.cov(robot_samples, rowvar=False)

    np.testing.assert_allclose(empirical_mean, expected_robot_mean, atol=5e-3)
    frob_err = np.linalg.norm(empirical_cov - env.state_transition_cov_matrix, ord="fro")
    assert frob_err < 1e-3, f"robot-slice cov Frobenius error {frob_err:.3e}"


def test_transition_sample_object_and_target_deterministic(env):
    """Out-of-range transitions leave object and target unchanged.

    Purpose: Validates that the deterministic-outside-push branch of
    ``apply_push`` is preserved under sampling (no noise leaks into object
    or target coordinates when robot-object distance exceeds threshold).

    Given: A state with robot far from the object (beyond push_threshold).
    When: A batch of 500 samples is drawn.
    Then: object (cols 2:4) and target (cols 4:6) slices are elementwise
        equal across all samples.

    Test type: unit
    """
    state = np.array([2.0, 3.0, 7.0, 7.0, 8.0, 8.0])
    transition = _make_transition_kernel(env, state=state, action=np.array([1.0, 0.0]))
    _native.set_seed(999)
    samples = np.asarray(transition.sample(500))
    np.testing.assert_array_equal(samples[:, 2:4], np.tile(state[2:4], (500, 1)))
    np.testing.assert_array_equal(samples[:, 4:6], np.tile(state[4:6], (500, 1)))


# ---------------------------------------------------------------------------
# probability() agrees with the Python reference on a grid
# ---------------------------------------------------------------------------


def test_transition_probability_matches_python_reference_on_grid(env):
    """C++ transition probability matches the Python reference on a grid.

    Purpose: Validates that the C++ PDF (2-D Gaussian on the robot slice)
    agrees numerically with the Python reference
    ``CovarianceParameterizedMultivariateNormal.pdf`` on a deterministic grid.

    Given: A 1000-point grid of candidate 6-D next-states whose robot slice
        is perturbed around the deterministic robot target. Object and target
        coordinates do not affect the probability.
    When: probability(grid) is called on both implementations (C++ uses the
        full 6-D values; the Python reference uses only the robot slice).
    Then: Outputs agree to absolute tolerance 1e-10.

    Test type: unit
    """
    state = np.array([2.0, 3.0, 5.0, 5.0, 8.0, 8.0])
    action = np.array([1.0, 0.0])
    transition = _make_transition_kernel(env, state=state, action=action)
    robot_mean = state[:2] + action

    rng = np.random.default_rng(42)
    offsets = rng.normal(size=(1000, 2)) * np.array([0.1, 0.1])
    robot_grid = robot_mean[np.newaxis, :] + offsets
    full_grid = np.zeros((1000, 6))
    full_grid[:, :2] = robot_grid
    full_grid[:, 2:4] = state[2:4]
    full_grid[:, 4:6] = state[4:6]

    cpp_pdf = transition.probability(full_grid)
    py_ref = CovarianceParameterizedMultivariateNormal(env.state_transition_cov_matrix)
    py_pdf = py_ref.pdf(robot_grid, robot_mean)
    np.testing.assert_allclose(cpp_pdf, py_pdf, atol=1e-10, rtol=0.0)


def test_observation_probability_matches_python_reference_on_grid(env):
    """C++ observation probability matches the isotropic-Gaussian reference.

    Purpose: Validates that the C++ observation PDF agrees with the pre-port
    formula ``(1 / (2 pi sigma^2)) exp(-0.5 * |x-mu|^2 / sigma^2)``.

    Given: A 1000-point grid of candidate observations whose object slice is
        perturbed around the true object position.
    When: probability(grid) is called on both C++ and Python reference.
    Then: Outputs agree to absolute tolerance 1e-12.

    Test type: unit
    """
    next_state = np.array([2.0, 3.0, 2.8, 3.2, 8.0, 8.0])
    observation = _make_observation_kernel(env, next_state=next_state, action=np.array([1.0, 0.0]))

    rng = np.random.default_rng(4242)
    offsets = rng.normal(size=(1000, 2)) * np.array([0.1, 0.1])
    obj_grid = next_state[2:4][np.newaxis, :] + offsets
    full_grid = np.zeros((1000, 6))
    full_grid[:, :2] = next_state[:2]
    full_grid[:, 2:4] = obj_grid
    full_grid[:, 4:6] = next_state[4:6]

    cpp_pdf = observation.probability(full_grid)
    # Python reference: per pre-port ContinuousPushObservationModel.probability
    sigma = env.observation_noise
    variance = sigma * sigma
    normalization = 1.0 / (2.0 * np.pi * variance)
    diff = obj_grid - next_state[2:4][np.newaxis, :]
    py_pdf = normalization * np.exp(-0.5 * np.sum(diff * diff, axis=1) / variance)
    np.testing.assert_allclose(cpp_pdf, py_pdf, atol=1e-12, rtol=0.0)


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
        np.array([3.01, 3.0, 5.0, 5.0, 8.0, 8.0]),
        np.array([2.99, 3.0, 5.0, 5.0, 8.0, 8.0]),
        np.array([3.0, 3.01, 5.0, 5.0, 8.0, 8.0]),
    ]
    array_values = np.stack(list_values, axis=0)
    from_list = transition.probability(list_values)
    from_array = transition.probability(array_values)
    np.testing.assert_array_equal(from_list, from_array)


# ---------------------------------------------------------------------------
# Determinism under explicit seeding
# ---------------------------------------------------------------------------


def test_sample_is_deterministic_under_set_seed(env):
    """Repeated set_seed + sample yields identical transition samples.

    Purpose: Validates the determinism path used by reproducible tests.

    Given: Two ``_native.ContinuousPushTransitionCpp`` instances sampled
        under the same seed.
    When: _native.set_seed(seed) is called before each, followed by sample(50).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    state = np.array([2.0, 3.0, 5.0, 5.0, 8.0, 8.0])
    transition = _make_transition_kernel(env, state=state, action=np.array([1.0, 0.0]))
    _native_parity.assert_determinism_under_seed(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=50,
        seed=2024,
    )


def test_observation_sample_is_deterministic_under_set_seed(env):
    """Repeated set_seed + sample yields identical observation samples.

    Purpose: Validates the determinism path for the observation model.

    Given: Two ``_native.ContinuousPushObservationCpp`` instances sampled
        under the same seed.
    When: _native.set_seed(seed) is called before each, followed by sample(50).
    Then: The two sample sequences are elementwise identical.

    Test type: unit
    """
    next_state = np.array([2.0, 3.0, 2.8, 3.2, 8.0, 8.0])
    observation = _make_observation_kernel(env, next_state=next_state, action=np.array([1.0, 0.0]))
    _native_parity.assert_determinism_under_seed(
        model=observation,
        seed_fn=_native.set_seed,
        n_samples=50,
        seed=1111,
    )


def test_sample_returns_list_of_1d_ndarrays(transition):
    """sample() preserves the pre-port List[np.ndarray] contract.

    Purpose: Guards against accidental API drift (many call sites index
    the returned list with ``[0]`` or iterate it).

    Given: A ``_native.ContinuousPushTransitionCpp`` kernel.
    When: sample(3) is called.
    Then: The return value is a list of exactly three 1-D ndarrays of
        length 6.

    Test type: unit
    """
    _native_parity.assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=6,
    )


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------


def test_transition_batch_sample_matches_per_particle_sample(env):
    """batch_sample(P) is bit-identical to sample(1) per row under fixed seed.

    Purpose: Validates that ContinuousPushTransitionCpp.batch_sample produces
    the same noise sequence as N per-particle sample(1) calls when both
    consume the same module-level C++ RNG seeded once.

    Given: 64 particles spanning the grid with objects both inside and
        outside push range.
    When: batch_sample runs under seed=777; then for each particle a fresh
        ``_native.ContinuousPushTransitionCpp`` kernel is built and sample(1)
        is called in the same order under the same seed.
    Then: The two (64, 6) arrays are array_equal.

    Test type: unit
    """
    rng = np.random.default_rng(123)
    particles = np.column_stack(
        [
            rng.uniform(1.0, 8.0, 64),  # robot x
            rng.uniform(1.0, 8.0, 64),  # robot y
            rng.uniform(1.0, 8.0, 64),  # obj x
            rng.uniform(1.0, 8.0, 64),  # obj y
            np.full(64, 8.0),  # target x
            np.full(64, 8.0),  # target y
        ]
    )
    action = np.array([1.0, 0.0])

    _native.set_seed(777)
    transition = _make_transition_kernel(env, state=particles[0], action=action)
    batch_result = transition.batch_sample(particles)

    _native.set_seed(777)
    per_particle_rows = []
    for row in particles:
        model = _make_transition_kernel(env, state=row, action=action)
        per_particle_rows.append(model.sample(1)[0])
    per_particle_result = np.stack(per_particle_rows, axis=0)

    np.testing.assert_array_equal(batch_result, per_particle_result)


def test_observation_batch_log_likelihood_matches_per_particle(env):
    """batch_log_likelihood matches np.log(probability([observation])) per row.

    Purpose: Validates the observation batch path against the per-particle
    reference (C++ probability() uses the same formula internally; we take
    log to invert).

    Given: 64 random next-state particles and a single observation.
    When: batch_log_likelihood(next_particles, observation) is called, and
        for each particle a fresh ``_native.ContinuousPushObservationCpp``
        kernel is built and probability([observation]) is computed.
    Then: The batch log-likelihoods equal np.log of the per-particle
        probabilities within atol=1e-12.

    Test type: unit
    """
    rng = np.random.default_rng(456)
    # Keep object positions near the observation so ``probability`` doesn't
    # underflow to 0 (which would produce -inf after np.log and defeat the
    # bit-exact compare against the C++ log_pdf path).
    observation_obj = np.array([2.5, 3.1])
    next_particles = np.column_stack(
        [
            rng.uniform(1.0, 8.0, 64),
            rng.uniform(1.0, 8.0, 64),
            observation_obj[0] + rng.normal(scale=0.1, size=64),
            observation_obj[1] + rng.normal(scale=0.1, size=64),
            np.full(64, 8.0),
            np.full(64, 8.0),
        ]
    )
    observation = np.array([2.0, 3.0, observation_obj[0], observation_obj[1], 8.0, 8.0])
    action = np.array([1.0, 0.0])

    obs_model = _make_observation_kernel(env, next_state=next_particles[0], action=action)
    batch_log_ll = obs_model.batch_log_likelihood(next_particles, observation)

    per_particle_log_ll = np.empty(len(next_particles))
    for i, next_state in enumerate(next_particles):
        model = _make_observation_kernel(env, next_state=next_state, action=action)
        per_particle_log_ll[i] = np.log(model.probability([observation])[0])

    assert np.all(
        np.isfinite(per_particle_log_ll)
    ), "reference log-likelihoods must be finite for bit-exact compare"
    np.testing.assert_allclose(batch_log_ll, per_particle_log_ll, atol=1e-12, rtol=0.0)


def test_transition_batch_sample_shape_contract(env):
    """batch_sample returns a 2-D ndarray matching the input shape.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A ``_native.ContinuousPushTransitionCpp`` kernel and a (37, 6)
        particles ndarray.
    When: batch_sample is called.
    Then: The result is an ndarray of shape (37, 6) and dtype float64.

    Test type: unit
    """
    state = np.array([2.0, 3.0, 5.0, 5.0, 8.0, 8.0])
    transition = _make_transition_kernel(env, state=state, action=np.array([1.0, 0.0]))
    particles = np.tile(state, (37, 1))
    _native.set_seed(0)
    result = transition.batch_sample(particles)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37, 6)
    assert result.dtype == np.float64


def test_observation_batch_log_likelihood_shape_contract(env):
    """batch_log_likelihood returns a 1-D ndarray of length N.

    Purpose: Guards the shape contract used by belief-level callers.

    Given: A ``_native.ContinuousPushObservationCpp`` kernel, 37
        next-particles, and one observation.
    When: batch_log_likelihood is called.
    Then: The result is an ndarray of shape (37,) and dtype float64.

    Test type: unit
    """
    next_state = np.array([2.0, 3.0, 2.8, 3.2, 8.0, 8.0])
    obs_model = _make_observation_kernel(env, next_state=next_state, action=np.array([1.0, 0.0]))
    next_particles = np.tile(next_state, (37, 1))
    observation = next_state.copy()
    result = obs_model.batch_log_likelihood(next_particles, observation)
    assert isinstance(result, np.ndarray)
    assert result.shape == (37,)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Contact-geometry parity (push mechanics and wall collisions)
# ---------------------------------------------------------------------------


def test_transition_push_mechanics_match_python_reference():
    """C++ push-mechanics branch produces the same object position as Python.

    Purpose: Guards against drift in the hand-translated contact-mechanics
    code (apply_push + point_inside_aabb). Uses tiny covariance so additive
    noise stays well below the grid/clamp thresholds; the deterministic push
    branch dominates.

    Given: A state where the robot is within push_threshold of the object so
        ``_apply_push`` is exercised.
    When: The native transition samples and the Python reference
        ``_apply_push`` are compared on the (deterministic, zero-noise limit)
        object output column.
    Then: The object slice matches within 1e-6 (margin allows the tiny
        noise on the robot position to shift the push distance calculation).

    Test type: unit
    """
    # Use essentially zero noise so we can compare the push output directly.
    tiny_env = ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            state_transition_cov_matrix=np.eye(2) * 1e-14,
        ),
    )
    state = np.array([2.5, 3.1, 3.0, 3.0, 8.0, 8.0])  # dist ~0.51 < 1.0 threshold
    action = np.array([1.0, 0.0])
    transition = _make_transition_kernel(tiny_env, state=state, action=action)
    _native.set_seed(0)
    samples = np.asarray(transition.sample(10))

    # Python reference: robot moves by action (+ ~0 noise), then push.
    expected_robot = state[:2] + action  # (3.5, 3.1)
    # dist_to_obj = sqrt((3.5-3.0)^2 + (3.1-3.0)^2) = sqrt(0.26) ~= 0.51
    # < push_threshold (1.0), so push is applied.
    # direction = (1, 0), force_mag = min(1, max_push=2) * (1-0.3) = 0.7
    expected_obj = state[2:4] + np.array([0.7, 0.0])
    np.testing.assert_allclose(samples[:, :2], np.tile(expected_robot, (10, 1)), atol=1e-6)
    np.testing.assert_allclose(samples[:, 2:4], np.tile(expected_obj, (10, 1)), atol=1e-6)


def test_transition_wall_collision_clamps_to_grid(env):
    """Robot samples stay within the in-bounds circle when pushed against walls.

    Purpose: Guards the grid-clamp branch ``clamp_circle_to_grid`` in the
    C++ post-sample transform.

    Given: A state at the lower-left grid corner with a large action pointing
        into the wall, and a small noise covariance.
    When: 100 samples are drawn.
    Then: Every sample's robot position is within the feasible rectangle
        ``[radius, grid_size-1-radius]`` on both axes.

    Test type: unit
    """
    state = np.array([0.3, 0.3, 5.0, 5.0, 8.0, 8.0])  # robot at lower corner
    action = np.array([-5.0, -5.0])
    transition = _make_transition_kernel(env, state=state, action=action)
    _native.set_seed(0)
    samples = np.asarray(transition.sample(100))
    radius = env.robot_radius
    hi = env.grid_size - 1 - radius
    assert np.all(samples[:, 0] >= radius - 1e-9)
    assert np.all(samples[:, 1] >= radius - 1e-9)
    assert np.all(samples[:, 0] <= hi + 1e-9)
    assert np.all(samples[:, 1] <= hi + 1e-9)
