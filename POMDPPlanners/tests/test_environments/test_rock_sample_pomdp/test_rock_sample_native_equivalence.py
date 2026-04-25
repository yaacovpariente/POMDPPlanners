"""Native (C++) equivalence tests for the RockSample POMDP.

Mirrors the structure of
``test_continuous_laser_tag_native_equivalence.py`` but adapted to
RockSample's discrete-action, categorical-observation contract:

* Transitions are deterministic, so there are no Gaussian moment tests
  or grid PDF parity tests. Instead we assert the deterministic next
  state is reproduced bit-exactly by both the per-particle path and the
  batch path, and that ``probability(values)`` is an indicator.
* Observations are categorical over ``{none, good, bad}``. The C++
  boundary uses integer codes; the public shim translates to strings.
  We assert the per-check Bernoulli probability matches the closed-form
  ``exp(-distance / sensor_efficiency)`` expression and that 200K-sample
  empirical frequencies match the analytic efficiency within 5e-3.
* The batch equivalence tests compare the native batch entry points
  against a per-particle loop over the public ``state_transition_model``
  / ``observation_model`` APIs, exactly like the pre-port
  ``test_rocksample_vectorized_updater.py::TestEquivalenceWithPerParticleLoop``
  suite.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.environment import ObservationModel, StateTransitionModel
from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
    RockSampleObservationModel,
    RockSampleStateTransitionModel,
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
from POMDPPlanners.tests.test_environments._native_parity import (
    assert_abc_registration,
    assert_sample_shape_contract,
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


@pytest.fixture(name="transition")
def _transition_fixture(env: RockSamplePOMDP) -> RockSampleStateTransitionModel:
    state = _state(2, 2, (True, True, False, True))
    # action=2 (East)
    return RockSampleStateTransitionModel(state, 2, env)


@pytest.fixture(name="observation_check")
def _observation_check_fixture(env: RockSamplePOMDP) -> RockSampleObservationModel:
    # Robot at (1, 1) queries rock 0 (also at (1, 1)) — distance zero,
    # efficiency ~1.0, deterministic correct observation.
    next_state = _state(1, 1, (True, False, True, False))
    return RockSampleObservationModel(next_state, 5, env)


@pytest.fixture(name="observation_move")
def _observation_move_fixture(env: RockSamplePOMDP) -> RockSampleObservationModel:
    next_state = _state(2, 2, (True, True, False, False))
    return RockSampleObservationModel(next_state, 2, env)


# ---------------------------------------------------------------------------
# 1-2. ABC registration
# ---------------------------------------------------------------------------


def test_transition_is_registered_as_state_transition_model(
    transition: RockSampleStateTransitionModel,
) -> None:
    """Purpose: Validates StateTransitionModel.register() wired the shim.

    Given: A RockSampleStateTransitionModel instance.
    When: isinstance(..., StateTransitionModel) is evaluated.
    Then: Returns True.

    Test type: unit
    """
    assert_abc_registration(transition, StateTransitionModel)


def test_observation_is_registered_as_observation_model(
    observation_check: RockSampleObservationModel,
) -> None:
    """Purpose: Validates ObservationModel subclassing for the shim.

    Given: A RockSampleObservationModel instance (inherits from
        ObservationModel directly; uses composition for the C++ backend).
    When: isinstance(..., ObservationModel) is evaluated.
    Then: Returns True.

    Test type: unit
    """
    assert_abc_registration(observation_check, ObservationModel)


# ---------------------------------------------------------------------------
# 3-4. Not-abstract
# ---------------------------------------------------------------------------


def test_transition_is_not_abstract() -> None:
    """Purpose: Ensures the shim has no unresolved abstract methods.

    Given: The RockSampleStateTransitionModel class.
    When: __abstractmethods__ is inspected.
    Then: It is an empty frozenset.

    Test type: unit
    """
    abstract = getattr(RockSampleStateTransitionModel, "__abstractmethods__", frozenset())
    assert abstract == frozenset()


def test_observation_is_not_abstract() -> None:
    """Purpose: Ensures the shim has no unresolved abstract methods.

    Given: The RockSampleObservationModel class.
    When: __abstractmethods__ is inspected.
    Then: It is an empty frozenset.

    Test type: unit
    """
    abstract = getattr(RockSampleObservationModel, "__abstractmethods__", frozenset())
    assert abstract == frozenset()


# ---------------------------------------------------------------------------
# 5-6. Deterministic transition contract
# ---------------------------------------------------------------------------


def test_transition_sample_is_deterministic(
    transition: RockSampleStateTransitionModel,
) -> None:
    """Purpose: Validates that sample(n) returns n identical states.

    Given: A transition with any action.
    When: sample(5) is called.
    Then: All 5 returned rows are equal.

    Test type: unit
    """
    rows = transition.sample(5)
    assert len(rows) == 5
    reference = rows[0]
    for row in rows[1:]:
        np.testing.assert_array_equal(row, reference)


def test_transition_probability_is_indicator(
    transition: RockSampleStateTransitionModel,
) -> None:
    """Purpose: Validates probability() is 1.0 for the deterministic next
    state and 0.0 for perturbed states.

    Given: A transition whose next state is known.
    When: probability([next_state, perturbed]) is called.
    Then: Returns [1.0, 0.0] exactly.

    Test type: unit
    """
    next_state = transition.sample()[0]
    perturbed = next_state.copy()
    perturbed[0] = perturbed[0] + 1  # shift robot_row by 1

    probs = transition.probability([next_state, perturbed])
    np.testing.assert_array_equal(probs, np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# 7. Observation movement-only branch
# ---------------------------------------------------------------------------


def test_observation_movement_returns_none(
    observation_move: RockSampleObservationModel,
) -> None:
    """Purpose: Movement actions must deterministically observe "none".

    Given: action=2 (East), any state.
    When: sample(100) and probability(["none","good","bad"]) are called.
    Then: All samples are "none"; probs are [1.0, 0.0, 0.0].

    Test type: unit
    """
    samples = observation_move.sample(100)
    assert all(s == "none" for s in samples)
    probs = observation_move.probability(["none", "good", "bad"])
    np.testing.assert_array_equal(probs, np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# 8. Check probability matches the closed-form Bernoulli
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
    When: probability(["good"]) and probability(["bad"]) are called for
        action=5 (check rock 0 at (1,1)).
    Then: Returned probabilities equal efficiency or 1-efficiency
        depending on rock quality, to 1e-12 absolute tolerance.

    Test type: unit
    """
    rocks_tuple = (rock_quality, True, False, True)  # only rock 0 matters here
    next_state = _state(robot_pos[0], robot_pos[1], rocks_tuple)
    obs_model = RockSampleObservationModel(next_state, 5, env)

    eff = expected_efficiency_close
    expected_good = eff if rock_quality else (1.0 - eff)
    expected_bad = (1.0 - eff) if rock_quality else eff

    probs = obs_model.probability(["good", "bad", "none"])
    np.testing.assert_allclose(probs[0], expected_good, atol=1e-12)
    np.testing.assert_allclose(probs[1], expected_bad, atol=1e-12)
    assert probs[2] == 0.0


# ---------------------------------------------------------------------------
# 9. Check-action empirical sample frequency matches analytic efficiency
# ---------------------------------------------------------------------------


def test_observation_check_sample_empirical_matches_probability(
    env: RockSamplePOMDP,
) -> None:
    """Purpose: Validates the C++ Bernoulli sampler's long-run frequency
    matches the analytic efficiency to statistical tolerance.

    Given: A check action at a known distance with a known rock quality.
    When: 200_000 observations are sampled (seeded).
    Then: The empirical fraction of "correct" observations (good for a
        good rock) is within 5e-3 of the analytic efficiency.

    Test type: unit
    """
    rocks = (True, False, True, False)  # rock 0 good
    next_state = _state(5, 5, rocks)
    obs_model = RockSampleObservationModel(next_state, 5, env)

    expected_efficiency = float(np.exp(-np.sqrt(32.0) / _SENSOR_EFFICIENCY))

    _native.set_seed(12345)
    samples = obs_model.sample(200_000)
    fraction_good = sum(1 for s in samples if s == "good") / len(samples)

    np.testing.assert_allclose(fraction_good, expected_efficiency, atol=5e-3)


# ---------------------------------------------------------------------------
# 10. Determinism under seed (observations)
# ---------------------------------------------------------------------------


def test_observation_sample_is_deterministic_under_set_seed(
    observation_check: RockSampleObservationModel,
) -> None:
    """Purpose: Seeding _native.set_seed reproduces identical samples.

    Given: A check-action observation model.
    When: set_seed(s) is called twice, interleaved with sample(N).
    Then: Both sample sequences are exactly equal.

    Test type: unit
    """
    # The per-particle observation shim uses composition: its .sample()
    # delegates to the stored _native instance, so we seed the module RNG
    # and then call .sample() on the shim (which reads from the same RNG).
    _native.set_seed(7777)
    first = observation_check.sample(32)
    _native.set_seed(7777)
    second = observation_check.sample(32)
    assert first == second


# ---------------------------------------------------------------------------
# 11. Transition sample shape contract
# ---------------------------------------------------------------------------


def test_transition_sample_shape_contract(
    transition: RockSampleStateTransitionModel,
) -> None:
    """Purpose: sample(k) must return a List[ndarray] of k rows, each 1-D
    with length 2+num_rocks.

    Given: A transition with any action.
    When: sample(3) is called.
    Then: Returns a Python list of 3 ndarrays, each shape (state_dim,).

    Test type: unit
    """
    assert_sample_shape_contract(
        model=transition,
        seed_fn=_native.set_seed,
        n_samples=3,
        expected_dim=_STATE_DIM,
    )


# ---------------------------------------------------------------------------
# 12. Observation sample returns strings
# ---------------------------------------------------------------------------


def test_observation_sample_returns_list_of_strings(
    observation_check: RockSampleObservationModel,
    observation_move: RockSampleObservationModel,
) -> None:
    """Purpose: Validates the shim's string-translation contract.

    Given: A check-action and a movement-action observation model.
    When: sample(N) is called.
    Then: Returns a list of strings, each in {none, good, bad}.

    Test type: unit
    """
    for model in (observation_check, observation_move):
        samples = model.sample(8)
        assert isinstance(samples, list)
        assert len(samples) == 8
        for s in samples:
            assert isinstance(s, str)
            assert s in {"none", "good", "bad"}


# ---------------------------------------------------------------------------
# 13. batch_sample vs per-particle sample bit-exact equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_transition_batch_sample_matches_per_particle_sample(
    env: RockSamplePOMDP, action: int
) -> None:
    """Purpose: The native batch_sample must produce bit-identical output
    to N per-particle sample(1) calls (transitions are deterministic).

    Given: 32 particles covering varied robot positions and rock configs.
    When: batch_sample(particles) is compared to a per-particle loop
        over env.state_transition_model(...).sample(1)[0].
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

    loop_out = np.stack([env.state_transition_model(row, action).sample()[0] for row in particles])
    np.testing.assert_array_equal(batch_out, loop_out.astype(float))


# ---------------------------------------------------------------------------
# 14. batch_log_likelihood vs per-particle probability equivalence
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
    native batch_log_likelihood must agree with per-particle
    ``np.log(probability([obs_str])[0])`` to 1e-10 atol.

    The batch path treats terminal particles as -inf by design (same
    as the pre-port Python vectorized updater), while the per-particle
    path computes a finite value using the terminal sentinel as a real
    position. That pre-existing divergence is tested separately in
    ``test_batch_log_likelihood_terminal_semantics``; this test
    intentionally uses only non-terminal particles so the two paths
    are in the regime where they are contractually equivalent.

    Given: 32 random live next-particles (no terminal sentinel).
    When: batch_log_likelihood is compared to the per-particle loop.
    Then: Arrays agree within 1e-10.

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
        prob = env.observation_model(row, action).probability([obs_str])[0]
        expected[i] = np.log(max(prob, 1e-300))

    np.testing.assert_allclose(batch_ll, expected, atol=1e-10)


def test_observation_batch_log_likelihood_matches_per_particle_movement(
    env: RockSamplePOMDP,
) -> None:
    """Purpose: For movement actions with obs=none, the batch path
    returns 0 on every particle. The per-particle path agrees for
    non-terminal particles (``probability(["none"])[0] == 1.0`` so
    ``log 1 = 0``). This test verifies that specific agreement.

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

    expected = np.array(
        [
            np.log(max(env.observation_model(row, 2).probability(["none"])[0], 1e-300))
            for row in particles
        ]
    )
    np.testing.assert_array_equal(batch_ll, np.zeros(32))
    np.testing.assert_array_equal(expected, np.zeros(32))


# ---------------------------------------------------------------------------
# 15-16. Batch output shape / dtype contracts
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
# 17. Terminal semantics on batch_log_likelihood
# ---------------------------------------------------------------------------


def test_batch_log_likelihood_terminal_semantics(env: RockSamplePOMDP) -> None:
    """Purpose: Terminal particles get -inf for check actions regardless
    of the observation; for movement actions + obs=none they get 0.0.

    Given: A batch of particles: [live_at_rock, terminal].
    When: batch_log_likelihood is invoked for (check action, good obs)
        and (movement action, none obs).
    Then: For check: [finite, -inf]. For movement + none: [0, 0].

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
    # Check action + good obs: live gets ~0 (log(efficiency)), terminal = -inf.
    ll_check = updater.batch_observation_log_likelihood(particles, np.array(5), np.array(OBS_GOOD))
    assert np.isfinite(ll_check[0])
    assert ll_check[1] == -np.inf

    # Movement action + none obs: both rows get 0.
    ll_move = updater.batch_observation_log_likelihood(particles, np.array(2), np.array(OBS_NONE))
    np.testing.assert_array_equal(ll_move, np.zeros(2))


# ---------------------------------------------------------------------------
# 18. Movement-action observation semantics on batch_log_likelihood
# ---------------------------------------------------------------------------


def test_batch_log_likelihood_movement_action_semantics(env: RockSamplePOMDP) -> None:
    """Purpose: For any movement action, batch_log_likelihood is 0 for
    obs=none and -inf for obs in {good, bad}.

    Given: Arbitrary particles.
    When: batch_log_likelihood is invoked with movement actions 0-4.
    Then: All particles get 0 for none, -inf for good/bad.

    Test type: unit
    """
    updater = RockSampleVectorizedUpdater.from_environment(env)
    particles = np.zeros((5, _STATE_DIM), dtype=float)
    particles[:, 0] = np.arange(5)
    particles[:, 1] = np.arange(5)

    for action in (0, 1, 2, 3, 4):
        ll_none = updater.batch_observation_log_likelihood(
            particles, np.array(action), np.array(OBS_NONE)
        )
        np.testing.assert_array_equal(ll_none, np.zeros(5))

        for obs in (OBS_GOOD, OBS_BAD):
            ll = updater.batch_observation_log_likelihood(
                particles, np.array(action), np.array(obs)
            )
            assert np.all(ll == -np.inf)


# ---------------------------------------------------------------------------
# 19. Full-parity sanity vs updater helper (preserves the pre-port contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [0, 2, 5])
def test_batch_transition_full_parity_against_updater(env: RockSamplePOMDP, action: int) -> None:
    """Purpose: Exercise the reusable ``assert_batch_transition_matches_loop``
    helper on the native-backed updater to guarantee the pre-port
    equivalence contract survives the rewrite.

    Given: 16 particles covering varied robot positions and rock configs.
    When: assert_batch_transition_matches_loop compares updater
        batch_transition to the per-particle state_transition_model sample.
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
        return env.state_transition_model(particle, act).sample()[0]

    assert_batch_transition_matches_loop(
        updater=updater,
        particles=particles,
        action=action,
        per_particle_transition_fn=per_particle,
    )


# ---------------------------------------------------------------------------
# 20. Hot-path sample_next_state / sample_observation overrides
#     (skip Python wrapper, route directly to native kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "robot_pos,action,rocks",
    [
        ((0, 0), 1, (True, True, False, True)),  # north move
        ((2, 2), 2, (False, True, True, False)),  # east move
        ((3, 3), 5, (True, False, False, True)),  # check rock 0 from far
        ((1, 1), 5, (True, True, True, True)),  # check rock 0 colocated
        ((4, 4), 0, (False, True, False, True)),  # sample at empty cell
    ],
)
def test_sample_next_state_override_matches_wrapper(
    env: RockSamplePOMDP,
    robot_pos: Tuple[int, int],
    action: int,
    rocks: Tuple[bool, ...],
) -> None:
    """Override sample_next_state matches the per-call wrapper bit-exactly.

    Purpose: Validates that ``RockSamplePOMDP.sample_next_state`` (which
    constructs the native kernel directly, skipping the Python wrapper)
    produces the exact same next state as the legacy
    ``state_transition_model(state, action).sample()[0]`` path under a
    fixed C++ RNG seed.

    Given: The shared RockSample env fixture and a parametrized
        ``(robot_pos, action, rocks)`` triple covering movement, check,
        and sample actions.
    When: Both paths are invoked with ``_native.set_seed`` reset to the
        same value before each call.
    Then: ``np.array_equal`` holds elementwise on the returned arrays.

    Test type: integration
    """
    state = _state(robot_pos[0], robot_pos[1], rocks)

    _native.set_seed(2024)
    via_wrapper = env.state_transition_model(state, action).sample()[0]

    _native.set_seed(2024)
    via_override = env.sample_next_state(state=state, action=action)

    assert np.array_equal(via_wrapper, via_override)


@pytest.mark.parametrize(
    "robot_pos,action,rocks",
    [
        ((1, 1), 5, (True, True, False, True)),  # check rock 0 colocated
        ((0, 0), 5, (True, False, True, False)),  # check rock 0 from corner
        ((3, 3), 6, (False, True, False, True)),  # check rock 1 colocated
        ((6, 2), 8, (True, True, True, False)),  # check rock 3 colocated
        ((2, 2), 2, (True, True, True, True)),  # movement -> none
    ],
)
def test_sample_observation_override_matches_wrapper(
    env: RockSamplePOMDP,
    robot_pos: Tuple[int, int],
    action: int,
    rocks: Tuple[bool, ...],
) -> None:
    """Override sample_observation matches the per-call wrapper bit-exactly.

    Purpose: Validates that ``RockSamplePOMDP.sample_observation`` (which
    constructs the native observation kernel directly, skipping the
    Python wrapper) produces the exact same string observation as the
    legacy ``observation_model(...).sample()[0]`` path under a fixed
    C++ RNG seed.

    Given: The shared env fixture plus a parametrized
        ``(robot_pos, action, rocks)`` triple covering check actions
        across distances and a movement action.
    When: Both paths are invoked with ``_native.set_seed`` reset to the
        same value before each call.
    Then: The two string observations are equal.

    Test type: integration
    """
    next_state = _state(robot_pos[0], robot_pos[1], rocks)

    _native.set_seed(7777)
    via_wrapper = env.observation_model(next_state, action).sample()[0]

    _native.set_seed(7777)
    via_override = env.sample_observation(next_state=next_state, action=action)

    assert via_wrapper == via_override
