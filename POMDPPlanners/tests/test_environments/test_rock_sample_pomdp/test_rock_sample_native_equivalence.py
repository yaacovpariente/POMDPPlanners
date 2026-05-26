# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Native (C++) equivalence tests for the RockSample POMDP.

Mirrors the structure of
``test_continuous_laser_tag_native_equivalence.py`` but adapted to
RockSample's discrete-action, categorical-observation contract:

* Transitions are deterministic, so there are no Gaussian moment tests
  or grid PDF parity tests. Instead we assert the deterministic next
  state is reproduced bit-exactly by both the per-particle path and the
  batch path, and that ``transition_log_probability`` is an indicator
  (0.0 for the deterministic next state, -inf for perturbed states).
* Observations are categorical over ``{none, good, bad}``. The C++
  boundary uses integer codes; the env-API translates to strings.
  We assert the per-check Bernoulli probability matches the closed-form
  ``exp(-distance / sensor_efficiency)`` expression and that 200K-sample
  empirical frequencies match the analytic efficiency within 5e-3.
* The batch equivalence tests compare the native batch entry points
  against a per-particle loop over the public ``env.sample_next_state``
  / ``env.observation_log_probability`` env-API.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
    create_rock_sample_state,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (
    OBS_BAD,
    OBS_GOOD,
    OBS_NONE,
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_transition_matches_loop,
)

# Shared small-env fixture values used across tests. Rock 0 lives at the
# robot's starting position so the "distance-zero sensor" case is trivial.
_ROCK_POSITIONS: List[Tuple[int, int]] = [(1, 1), (3, 3), (5, 5), (6, 2)]
_MAP_SIZE: Tuple[int, int] = (7, 7)
_INIT_POS: Tuple[int, int] = (0, 0)
_SENSOR_EFFICIENCY: float = 10.0
_NUM_ROCKS: int = len(_ROCK_POSITIONS)
_STATE_DIM: int = 2 + _NUM_ROCKS


def _make_env() -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=_MAP_SIZE,
        rock_positions=list(_ROCK_POSITIONS),
        init_pos=_INIT_POS,
        sensor_efficiency=_SENSOR_EFFICIENCY,
    )


def _state(row: int, col: int, rocks: Tuple[bool, ...]) -> np.ndarray:
    assert len(rocks) == _NUM_ROCKS
    return create_rock_sample_state((row, col), rocks)


@pytest.fixture(name="env")
def _env_fixture() -> RockSamplePOMDP:
    return _make_env()


# ---------------------------------------------------------------------------
# Deterministic transition contract
# ---------------------------------------------------------------------------


def test_transition_sample_is_deterministic(env: RockSamplePOMDP) -> None:
    """Purpose: Validates that env.sample_next_state(n) returns n identical states.

    Given: A live state and a movement action.
    When: env.sample_next_state(state, action, n_samples=5) is called.
    Then: All 5 returned rows are equal.

    Test type: unit
    """
    state = _state(2, 2, (True, True, False, True))
    action = 2  # East
    rows = env.sample_next_state(state=state, action=action, n_samples=5)
    assert len(rows) == 5
    reference = rows[0]
    for row in rows[1:]:
        np.testing.assert_array_equal(row, reference)


def test_transition_log_probability_is_indicator(env: RockSamplePOMDP) -> None:
    """Purpose: Validates env.transition_log_probability returns 0.0 for the
    deterministic next state and a large negative value (effectively -inf
    after the +1e-300 floor) for perturbed states.

    Given: A live state with a known deterministic next state.
    When: env.transition_log_probability is called for [next_state, perturbed].
    Then: probs[0] is approximately 0.0; probs[1] is large-negative.

    Test type: unit
    """
    state = _state(2, 2, (True, True, False, True))
    action = 2  # East
    next_state = env.sample_next_state(state=state, action=action)
    perturbed = next_state.copy()
    perturbed[0] = perturbed[0] + 1  # shift robot_row by 1

    log_p = env.transition_log_probability(state, action, [next_state, perturbed])
    np.testing.assert_allclose(log_p[0], 0.0, atol=1e-12)
    assert log_p[1] < -100  # log(0 + 1e-300) ≈ -690


# ---------------------------------------------------------------------------
# Observation movement-only branch
# ---------------------------------------------------------------------------


def test_observation_movement_returns_none(env: RockSamplePOMDP) -> None:
    """Purpose: Movement actions must deterministically observe "none".

    Given: action=2 (East), any state.
    When: env.sample_observation(n=100) and env.observation_log_probability are
        called for the canonical observation alphabet.
    Then: All samples are "none"; log_probs are [0.0, -inf-ish, -inf-ish].

    Test type: unit
    """
    next_state = _state(2, 2, (True, True, False, False))
    samples = env.sample_observation(next_state=next_state, action=2, n_samples=100)
    assert all(s == "none" for s in samples)

    log_probs = env.observation_log_probability(next_state, 2, ["none", "good", "bad"])
    np.testing.assert_allclose(log_probs[0], 0.0, atol=1e-12)
    assert log_probs[1] < -100
    assert log_probs[2] < -100


# ---------------------------------------------------------------------------
# Check probability matches the closed-form Bernoulli
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "robot_pos,rock_quality,expected_efficiency_close",
    [
        ((1, 1), True, 1.0),  # distance 0 -> efficiency 1.0
        ((1, 1), False, 1.0),
        ((5, 5), True, float(np.exp(-np.sqrt(32.0) / _SENSOR_EFFICIENCY))),  # distance to (1,1)
        ((5, 5), False, float(np.exp(-np.sqrt(32.0) / _SENSOR_EFFICIENCY))),
    ],
)
def test_observation_check_probability_matches_python_formula(
    env: RockSamplePOMDP,
    robot_pos: Tuple[int, int],
    rock_quality: bool,
    expected_efficiency_close: float,
) -> None:
    """Purpose: Validates the check-action Bernoulli probability matches
    the analytic expression ``exp(-d / sensor_efficiency)``.

    Given: A next_state with known robot position and rock 0 quality.
    When: env.observation_log_probability for ["good", "bad", "none"] is
        evaluated for action=5 (check rock 0 at (1,1)).
    Then: Returned probabilities equal efficiency or 1-efficiency
        depending on rock quality, to 1e-12 absolute tolerance.

    Test type: unit
    """
    rocks_tuple = (rock_quality, True, False, True)  # only rock 0 matters here
    next_state = _state(robot_pos[0], robot_pos[1], rocks_tuple)

    eff = expected_efficiency_close
    expected_good = eff if rock_quality else (1.0 - eff)
    expected_bad = (1.0 - eff) if rock_quality else eff

    log_probs = env.observation_log_probability(next_state, 5, ["good", "bad", "none"])
    probs = np.exp(log_probs)
    np.testing.assert_allclose(probs[0], expected_good, atol=1e-12)
    np.testing.assert_allclose(probs[1], expected_bad, atol=1e-12)
    assert probs[2] < 1e-200


# ---------------------------------------------------------------------------
# Check-action empirical sample frequency matches analytic efficiency
# ---------------------------------------------------------------------------


def test_observation_check_sample_empirical_matches_probability(
    env: RockSamplePOMDP,
) -> None:
    """Purpose: Validates the C++ Bernoulli sampler's long-run frequency
    matches the analytic efficiency to statistical tolerance.

    Given: A check action at a known distance with a known rock quality.
    When: 200_000 observations are sampled (seeded) via env.sample_observation.
    Then: The empirical fraction of "correct" observations (good for a
        good rock) is within 5e-3 of the analytic efficiency.

    Test type: unit
    """
    rocks = (True, False, True, False)  # rock 0 good
    next_state = _state(5, 5, rocks)

    expected_efficiency = float(np.exp(-np.sqrt(32.0) / _SENSOR_EFFICIENCY))

    _native.set_seed(12345)
    samples = env.sample_observation(next_state=next_state, action=5, n_samples=200_000)
    fraction_good = sum(1 for s in samples if s == "good") / len(samples)

    np.testing.assert_allclose(fraction_good, expected_efficiency, atol=5e-3)


# ---------------------------------------------------------------------------
# Determinism under seed (observations)
# ---------------------------------------------------------------------------


def test_observation_sample_is_deterministic_under_set_seed(env: RockSamplePOMDP) -> None:
    """Purpose: Seeding _native.set_seed reproduces identical samples.

    Given: A check-action observation invocation.
    When: set_seed(s) is called twice, interleaved with env.sample_observation(N).
    Then: Both sample sequences are exactly equal.

    Test type: unit
    """
    next_state = _state(1, 1, (True, False, True, False))
    _native.set_seed(7777)
    first = env.sample_observation(next_state=next_state, action=5, n_samples=32)
    _native.set_seed(7777)
    second = env.sample_observation(next_state=next_state, action=5, n_samples=32)
    assert first == second


# ---------------------------------------------------------------------------
# Observation sample returns strings
# ---------------------------------------------------------------------------


def test_observation_sample_returns_list_of_strings(env: RockSamplePOMDP) -> None:
    """Purpose: Validates the env-API string-translation contract.

    Given: A check-action and a movement-action observation invocation.
    When: env.sample_observation(N) is called.
    Then: Returns a list of strings, each in {none, good, bad}.

    Test type: unit
    """
    cases = [
        (_state(1, 1, (True, False, True, False)), 5),  # check rock 0 colocated
        (_state(2, 2, (True, True, False, False)), 2),  # movement
    ]
    for next_state, action in cases:
        samples = env.sample_observation(next_state=next_state, action=action, n_samples=8)
        assert isinstance(samples, list)
        assert len(samples) == 8
        for s in samples:
            assert isinstance(s, str)
            assert s in {"none", "good", "bad"}


# ---------------------------------------------------------------------------
# batch_sample vs per-particle sample bit-exact equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_transition_batch_sample_matches_per_particle_sample(
    env: RockSamplePOMDP, action: int
) -> None:
    """Purpose: The native batch_sample must produce bit-identical output
    to N per-particle env.sample_next_state calls (transitions are deterministic).

    Given: 32 particles covering varied robot positions and rock configs.
    When: batch_sample(particles) is compared to a per-particle loop
        over env.sample_next_state(state=row, action=action).
    Then: np.testing.assert_array_equal passes (exact match).

    Test type: integration
    """
    rng = np.random.default_rng(12345 + action)
    particles = np.zeros((32, _STATE_DIM), dtype=float)
    for i in range(32):
        row = int(rng.integers(0, _MAP_SIZE[0]))
        col = int(rng.integers(0, _MAP_SIZE[1]))
        rocks = rng.integers(0, 2, size=_NUM_ROCKS).astype(float)
        particles[i, 0] = row
        particles[i, 1] = col
        particles[i, 2:] = rocks

    updater = RockSampleVectorizedUpdater.from_environment(env)
    batch_out = updater.batch_transition(particles, np.array(action))

    loop_out = np.stack([env.sample_next_state(state=row, action=action) for row in particles])
    np.testing.assert_array_equal(batch_out, loop_out.astype(float))


# ---------------------------------------------------------------------------
# batch_log_likelihood vs per-particle probability equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,obs_str,obs_code",
    [
        (5, "good", OBS_GOOD),
        (5, "bad", OBS_BAD),
        (6, "good", OBS_GOOD),
        (7, "bad", OBS_BAD),
    ],
)
def test_observation_batch_log_likelihood_matches_per_particle(
    env: RockSamplePOMDP, action: int, obs_str: str, obs_code: int
) -> None:
    """Purpose: For check actions on live (non-terminal) particles, the
    native batch_log_likelihood agrees with the per-particle
    ``env.observation_log_probability([obs_str])[0]`` across the entire
    array (including impossible events).

    Post symmetric C++ floor: both
    ``batch_log_likelihood`` (log-floor) and the scalar path
    ``np.log(kernel.probability(...))`` (linear-floor then log) clamp
    impossible events to ``log(kProbFloor) = log(1e-300) ~= -690.776``,
    so the two paths produce bit-exact values everywhere — including
    the impossible-event subset. The earlier relaxation that compared
    only the finite-probability subset is no longer needed.

    Given: 32 random live next-particles (no terminal sentinel).
    When: batch_log_likelihood is compared to the per-particle loop.
    Then: Arrays agree within 1e-10 across all entries.

    Test type: integration
    """
    rng = np.random.default_rng(24680 + action)
    particles = np.zeros((32, _STATE_DIM), dtype=float)
    for i in range(32):
        particles[i, 0] = int(rng.integers(0, _MAP_SIZE[0]))
        particles[i, 1] = int(rng.integers(0, _MAP_SIZE[1]))
        particles[i, 2:] = rng.integers(0, 2, size=_NUM_ROCKS).astype(float)

    updater = RockSampleVectorizedUpdater.from_environment(env)
    batch_ll = updater.batch_observation_log_likelihood(
        particles, np.array(action), np.array(obs_code)
    )

    expected = np.zeros(32)
    for i, row in enumerate(particles):
        expected[i] = env.observation_log_probability(row, action, [obs_str])[0]

    np.testing.assert_allclose(batch_ll, expected, atol=1e-10)


def test_observation_batch_log_likelihood_matches_per_particle_movement(
    env: RockSamplePOMDP,
) -> None:
    """Purpose: For movement actions with obs=none, the batch path
    returns 0 on every particle. The per-particle path agrees for
    non-terminal particles (``observation_log_probability(["none"])[0] == 0``).
    This test verifies that specific agreement.

    Given: 32 random live next-particles.
    When: batch_log_likelihood(particles, movement_action, OBS_NONE)
        is compared to the per-particle loop.
    Then: Both return the all-zero array.

    Test type: integration
    """
    rng = np.random.default_rng(13579)
    particles = np.zeros((32, _STATE_DIM), dtype=float)
    for i in range(32):
        particles[i, 0] = int(rng.integers(0, _MAP_SIZE[0]))
        particles[i, 1] = int(rng.integers(0, _MAP_SIZE[1]))
        particles[i, 2:] = rng.integers(0, 2, size=_NUM_ROCKS).astype(float)

    updater = RockSampleVectorizedUpdater.from_environment(env)
    batch_ll = updater.batch_observation_log_likelihood(particles, np.array(2), np.array(OBS_NONE))

    expected = np.array([env.observation_log_probability(row, 2, ["none"])[0] for row in particles])
    np.testing.assert_array_equal(batch_ll, np.zeros(32))
    np.testing.assert_array_equal(expected, np.zeros(32))


# ---------------------------------------------------------------------------
# Batch output shape / dtype contracts
# ---------------------------------------------------------------------------


def test_transition_batch_sample_shape_contract(env: RockSamplePOMDP) -> None:
    """Purpose: batch_sample of shape (N, state_dim) must return
    shape (N, state_dim) float64.

    Given: A (41, state_dim) input particle array.
    When: batch_sample is invoked.
    Then: Output is (41, state_dim) float64.

    Test type: unit
    """
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.zeros((41, _STATE_DIM), dtype=float)
    particles[:, 0] = np.arange(41) % _MAP_SIZE[0]
    particles[:, 1] = (np.arange(41) // 2) % _MAP_SIZE[1]
    out = updater.batch_transition(particles, np.array(2))
    assert out.shape == (41, _STATE_DIM)
    assert out.dtype == np.float64


def test_observation_batch_log_likelihood_shape_contract(env: RockSamplePOMDP) -> None:
    """Purpose: batch_log_likelihood returns shape (N,) float64.

    Given: A (41, state_dim) input, scalar action, scalar obs code.
    When: batch_observation_log_likelihood is invoked.
    Then: Output is shape (41,) float64.

    Test type: unit
    """
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.zeros((41, _STATE_DIM), dtype=float)
    particles[:, 0] = 1.0
    particles[:, 1] = 1.0
    particles[:, 2] = 1.0  # rock 0 good
    out = updater.batch_observation_log_likelihood(particles, np.array(5), np.array(OBS_GOOD))
    assert out.shape == (41,)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Terminal semantics on batch_log_likelihood
# ---------------------------------------------------------------------------


def test_batch_log_likelihood_terminal_semantics(env: RockSamplePOMDP) -> None:
    """Purpose: Terminal particles get the symmetric ``kLogProbFloor``
    floor (~ -690.776) for check actions regardless of the observation;
    for movement actions + obs=none they get 0.0.

    Post symmetric C++ floor: the historical -inf for impossible events
    is now floored to ``log(kProbFloor) = log(1e-300) ~= -690.776`` so
    the C++ ``batch_log_likelihood`` and the env-API scalar
    ``np.log(kernel.probability(...))`` paths agree on the same value.

    Given: A batch of particles: [live_at_rock, terminal].
    When: batch_log_likelihood is invoked for (check action, good obs)
        and (movement action, none obs).
    Then: For check: [finite, ~ -690.776]. For movement + none: [0, 0].

    Test type: unit
    """
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # live at rock 0, good
            [-1.0, -1.0, 1.0, 0.0, 0.0, 0.0],  # terminal
        ],
        dtype=float,
    )
    floor = float(np.log(1e-300))  # ~= -690.7755278982137
    # Check action + good obs: live gets ~0 (log(efficiency)), terminal = floor.
    ll_check = updater.batch_observation_log_likelihood(particles, np.array(5), np.array(OBS_GOOD))
    assert np.isfinite(ll_check[0])
    np.testing.assert_allclose(ll_check[1], floor, atol=1e-6)

    # Movement action + none obs: both rows get 0.
    ll_move = updater.batch_observation_log_likelihood(particles, np.array(2), np.array(OBS_NONE))
    np.testing.assert_array_equal(ll_move, np.zeros(2))


# ---------------------------------------------------------------------------
# Movement-action observation semantics on batch_log_likelihood
# ---------------------------------------------------------------------------


def test_batch_log_likelihood_movement_action_semantics(env: RockSamplePOMDP) -> None:
    """Purpose: For any movement action, batch_log_likelihood is 0 for
    obs=none and floored to ``log(1e-300) ~= -690.776`` for obs in
    {good, bad}.

    Post symmetric C++ floor: the historical -inf for impossible events
    is now floored to ``kLogProbFloor`` so the batch and scalar API
    paths agree on the same value.

    Given: Arbitrary particles.
    When: batch_log_likelihood is invoked with movement actions 0-4.
    Then: All particles get 0 for none, ~ -690.776 for good/bad.

    Test type: unit
    """
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.zeros((5, _STATE_DIM), dtype=float)
    particles[:, 0] = np.arange(5)
    particles[:, 1] = np.arange(5)

    floor = float(np.log(1e-300))  # ~= -690.7755278982137
    for action in (0, 1, 2, 3, 4):
        ll_none = updater.batch_observation_log_likelihood(
            particles, np.array(action), np.array(OBS_NONE)
        )
        np.testing.assert_array_equal(ll_none, np.zeros(5))

        for obs in (OBS_GOOD, OBS_BAD):
            ll = updater.batch_observation_log_likelihood(
                particles, np.array(action), np.array(obs)
            )
            np.testing.assert_allclose(ll, np.full(5, floor), atol=1e-6)


# ---------------------------------------------------------------------------
# Full-parity sanity vs updater helper (preserves the pre-port contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [0, 2, 5])
def test_batch_transition_full_parity_against_updater(env: RockSamplePOMDP, action: int) -> None:
    """Purpose: Exercise the reusable ``assert_batch_transition_matches_loop``
    helper on the native-backed updater to guarantee the pre-port
    equivalence contract survives the rewrite.

    Given: 16 particles covering varied robot positions and rock configs.
    When: assert_batch_transition_matches_loop compares updater
        batch_transition to the per-particle env.sample_next_state.
    Then: No assertion fires (bit-exact agreement).

    Test type: integration
    """
    rng = np.random.default_rng(98765 + action)
    particles = np.zeros((16, _STATE_DIM), dtype=float)
    for i in range(16):
        particles[i, 0] = int(rng.integers(0, _MAP_SIZE[0]))
        particles[i, 1] = int(rng.integers(0, _MAP_SIZE[1]))
        particles[i, 2:] = rng.integers(0, 2, size=_NUM_ROCKS).astype(float)

    updater = RockSampleVectorizedUpdater.from_environment(env)

    def per_particle(particle: np.ndarray, act: int) -> np.ndarray:
        return env.sample_next_state(state=particle, action=act)

    assert_batch_transition_matches_loop(
        updater=updater,
        particles=particles,
        action=action,
        per_particle_transition_fn=per_particle,
    )
