# SPDX-License-Identifier: MIT

"""Parity tests: the torch vectorized racetrack model vs. the scalar racetrack model.

These tests pin :class:`RacetrackVectorizedModel` to
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`
so the two implementations cannot drift apart. The deterministic kernels — the
kinematic-bicycle transition with its Frenet pair and agent-slot drift, the highway-env
reward, the collision terminal check, and both observation log-densities — are compared in
float64 against the scalar model row by row, over several supported configurations. The
scalar model is built with ``process_noise_std=0.0`` wherever the transition itself is under
test, so the comparison is against a deterministic propagation rather than two independent
noise draws; the noise is then checked separately by its empirical moments.

The occupancy rasteriser gets a dedicated crafted-state test, because its axis order
(along-track first, across-track second) and its always-marked centre cell were established
empirically against highway-env 1.12.1 and are the easiest thing in the module to silently
invert.

**Curvature is the part that has to be watched.** The scalar model asks its subclass where
the road bends once per substep, and the torch model re-implements that lookup rather than
calling back into NumPy. So the parity tests deliberately put particles on a lap short
enough that a rollout crosses several segment boundaries, and both arms of the lookup — a
track map and a per-step scalar estimate — are compared against the scalar model over a
multi-decision rollout, not just one step. A single-step comparison would agree even if the
torch side froze the curvature at the value it started with, which is the exact bug the
arclength slot exists to prevent.
"""

# pylint: disable=too-many-lines  # One test module per source module, as the test layout requires.

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    ObservedTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    EGO_ARCLENGTH_M,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    GRID_CELLS,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    ON_ROAD_LAYER,
    ObservationMode,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry
from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
    ObservedCurvature,
    RacetrackVectorizedModel,
    TrackMapCurvature,
)

_TOLERANCE = 1e-9

_CUSTOM_PRESETS = [(1.0, -1.0), (0.5, 0.25), (0.0, 0.0), (-1.0, 0.75)]

# Index of the (0.0, 0.0) preset: coast, straight ahead. Looked up rather than written as a
# literal, because the shipped table is a 3-by-9 grid whose indices move whenever the
# steering resolution changes. Only valid for cases built on the default table.
_COAST_STRAIGHT = DEFAULT_ACTION_PRESETS.index((0.0, 0.0))

# A deliberately short lap -- two straights and two arcs of opposite sign over 60 m -- so a
# rollout of a few decisions at track speed crosses several boundaries and wraps. A profile
# the length of the real circuit would let a frozen-curvature bug pass every test here.
_TEST_GEOMETRY = TrackGeometry(
    segment_starts=np.array([0.0, 12.0, 30.0, 45.0]),
    segment_curvatures=np.array([0.0, 0.05, -0.03, 0.0]),
    total_length_m=60.0,
)

# Every case is a supported scalar-model configuration. Noise is off so the transition
# comparison is deterministic on both sides; the samplers are tested separately.
_SUPPORTED_CASES: List[Dict[str, Any]] = [
    {"discount_factor": 0.95, "process_noise_std": 0.0},
    {
        "discount_factor": 0.9,
        "process_noise_std": 0.0,
        "dt": 0.5,
        "substeps": 7,
        "vehicle_length": 4.2,
        "max_tracked_agents": 2,
        "lane_half_width": 3.5,
        "collision_distance": 2.0,
    },
    {
        "discount_factor": 0.95,
        "process_noise_std": 0.0,
        "action_presets": _CUSTOM_PRESETS,
        "collision_reward": -3.0,
        "lane_centering_cost": 1.5,
        "lane_centering_reward": 2.0,
        "action_reward": -0.1,
        "cell_flip_prob": 0.2,
    },
]
_CASE_IDS = ["defaults", "coarse-steps", "custom-weights"]


class _FixedCurvatureModel(RacetrackModelPOMDP):
    """A scalar model whose curvature is one number, replaced from outside between steps.

    Stands in for the shipped model that estimates curvature from its observations rather
    than from a map. What the vectorized twin has to reproduce about such a model is only
    that the estimate is read *live* — it changes every real step while the torch model is
    built once — so a settable attribute is the whole of the contract worth testing here.

    Attributes:
        curvature_estimate: The current estimate in 1/m, shared by every particle.
    """

    def __init__(self, discount_factor: float, curvature_estimate: Any, **kwargs: Any) -> None:
        # Stored as handed over, not coerced to float: a real estimator computes this off an
        # array and would hand over a NumPy scalar, and whether the twin recognises that is
        # part of what these tests check.
        self.curvature_estimate = curvature_estimate
        super().__init__(discount_factor=discount_factor, **kwargs)

    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        return np.full(len(ego), float(self.curvature_estimate), dtype=float)


@dataclass
class _Case:
    """A scalar model paired with the vectorized twin built from it."""

    env: RacetrackModelPOMDP
    model: RacetrackVectorizedModel


def _build_case(**kwargs: Any) -> _Case:
    env = KnownTrackModel(track_geometry=_TEST_GEOMETRY, **kwargs)
    return _Case(env=env, model=_twin(env))


def _twin(env: RacetrackModelPOMDP) -> RacetrackVectorizedModel:
    return RacetrackVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)


@pytest.fixture(params=_SUPPORTED_CASES, ids=_CASE_IDS, name="case")
def case_fixture(request: pytest.FixtureRequest) -> _Case:
    return _build_case(**request.param)


@pytest.fixture(name="pomdp_case")
def pomdp_case_fixture() -> _Case:
    return _build_case(discount_factor=0.95, process_noise_std=0.0)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float64)


def _actions(indices: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(indices), dtype=torch.int64)


def _random_states(rng: np.random.Generator, env: RacetrackModelPOMDP, count: int) -> np.ndarray:
    """Random states mixing present/absent slots, and every segment of the test lap.

    Arclengths span several laps in both directions, so the wrap and every segment of the
    profile are exercised by an ordinary parity run rather than only by the tests that
    target the lookup directly.
    """
    states = np.zeros((count, env.state_width))
    states[:, 0:2] = rng.uniform(-40.0, 40.0, size=(count, 2))
    states[:, 2] = rng.uniform(-np.pi, np.pi, size=count)
    states[:, 3] = rng.uniform(0.0, 15.0, size=count)
    states[:, 4] = rng.uniform(-3.0, 3.0, size=count)
    states[:, 5] = rng.uniform(-0.6, 0.6, size=count)
    states[:, EGO_ARCLENGTH_M] = rng.uniform(-150.0, 150.0, size=count)
    slots = states[:, EGO_STATE_WIDTH:].reshape(count, env.max_tracked_agents, AGENT_SLOT_WIDTH)
    slots[..., 0] = (rng.uniform(size=slots.shape[:2]) < 0.6).astype(float)
    slots[..., 1] = rng.uniform(-25.0, 25.0, size=slots.shape[:2])
    slots[..., 2] = rng.uniform(-12.0, 12.0, size=slots.shape[:2])
    slots[..., 3] = rng.uniform(-6.0, 6.0, size=slots.shape[:2])
    slots[..., 4] = rng.uniform(-6.0, 6.0, size=slots.shape[:2])
    return states


def _random_action_indices(
    rng: np.random.Generator, env: RacetrackModelPOMDP, count: int
) -> np.ndarray:
    return rng.integers(0, len(env.action_presets), size=count)


def _scalar_next_states(
    env: RacetrackModelPOMDP, states: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    return np.stack(
        [env.sample_next_state(states[i], int(indices[i])) for i in range(states.shape[0])]
    )


def _occupancy_dict(flat: torch.Tensor) -> Dict[str, np.ndarray]:
    return {"occupancy": flat.numpy().reshape(2, GRID_CELLS, GRID_CELLS)}


def _kinematics_dict(flat: torch.Tensor, num_agents: int) -> Dict[str, np.ndarray]:
    array = flat.numpy()
    return {
        "ego": array[:4].copy(),
        "agents": array[4:].reshape(num_agents, AGENT_SLOT_WIDTH).copy(),
    }


def test_model_conforms_to_protocol_across_configs(case: _Case) -> None:
    """Every supported configuration yields a conforming vectorized model.

    Purpose: Validates structural protocol conformance and the reported dimensions

    Given: A vectorized model built from each supported scalar configuration
    When: It is checked against the runtime-checkable VectorizedGenerativeModel protocol
    Then: isinstance reports conformance and the dimensions follow the schema

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == len(case.env.action_presets)
    assert case.model.state_dim == case.env.state_width
    assert case.model.observation_dim == 2 * GRID_CELLS * GRID_CELLS


def test_transition_matches_native_exactly(case: _Case) -> None:
    """Sampled next states equal the scalar propagation row by row.

    Purpose: Validates the batched bicycle + Frenet + agent-drift transition kernel

    Given: 256 random states with per-row random action indices and no process noise
    When: sample_next_states is compared to scalar sample_next_state per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(1)
    states = _random_states(rng, case.env, 256)
    indices = _random_action_indices(rng, case.env, 256)
    expected = _scalar_next_states(case.env, states, indices)
    actual = case.model.sample_next_states(_tensor(states), _actions(indices)).numpy()
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_transition_matches_native_for_every_action(case: _Case) -> None:
    """Each control preset, including full-lock steer and braking, propagates identically.

    Purpose: Validates the per-action slip and acceleration table against the scalar model

    Given: A fixed batch of random states replayed once under every action index
    When: The vectorized transition is compared to the scalar one for that action
    Then: Every action agrees to within 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(2)
    states = _random_states(rng, case.env, 32)
    for index in range(len(case.env.action_presets)):
        indices = np.full(states.shape[0], index)
        expected = _scalar_next_states(case.env, states, indices)
        actual = case.model.sample_next_states(_tensor(states), _actions(indices)).numpy()
        assert np.max(np.abs(expected - actual)) < _TOLERANCE, f"action {index} diverged"


def test_transition_matches_scalar_batch_helper(pomdp_case: _Case) -> None:
    """The scalar batch transition agrees with the vectorized kernel under one action.

    Purpose: Cross-checks sample_next_state_batch, the filter's hot path, against the twin

    Given: 128 random states and a single shared action index
    When: The scalar batch propagation and the vectorized transition are compared
    Then: The maximum absolute difference is below 1e-9

    Test type: integration
    """
    rng = np.random.default_rng(3)
    states = _random_states(rng, pomdp_case.env, 128)
    expected = pomdp_case.env.sample_next_state_batch(states, _COAST_STRAIGHT)
    actual = pomdp_case.model.sample_next_states(
        _tensor(states), _actions(np.full(128, _COAST_STRAIGHT))
    ).numpy()
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def _rollout_states(count: int, geometry: TrackGeometry) -> np.ndarray:
    """Particles on the lane centreline at track speed, spread around the whole lap."""
    states = np.zeros((count, EGO_STATE_WIDTH + 4 * AGENT_SLOT_WIDTH))
    states[:, EGO_SPEED] = 10.0
    states[:, EGO_ARCLENGTH_M] = np.linspace(0.0, geometry.total_length_m, count, endpoint=False)
    return states


def _scalar_rollout(
    env: RacetrackModelPOMDP, states: np.ndarray, action: int, steps: int
) -> np.ndarray:
    current = states
    for _ in range(steps):
        current = env.sample_next_state_batch(current, action)
    return current


def _vector_rollout(
    model: RacetrackVectorizedModel, states: np.ndarray, action: int, steps: int
) -> np.ndarray:
    current = _tensor(states)
    indices = _actions(np.full(states.shape[0], action))
    for _ in range(steps):
        current = model.sample_next_states(current, indices)
    return current.numpy()


def test_map_rollout_crossing_segment_boundaries_matches_native() -> None:
    """A multi-decision rollout over a whole lap tracks the scalar model step for step.

    Purpose: This is the test the curvature change exists for. One step agrees even if the
        torch side freezes the curvature it started with; only a rollout that drives through
        corners can tell a live lookup from a frozen one

    Given: 32 particles spread around a four-segment lap at 10 m/s, no process noise, and
        40 coasting decisions -- enough to cross every boundary and wrap past the finish
    When: The scalar and vectorized models are each rolled forward the same 40 steps
    Then: The full state agrees to within 1e-9, and the run really did visit more than one
        curvature rather than sitting on a straight the whole way

    Test type: integration
    """
    case = _build_case(discount_factor=0.95, process_noise_std=0.0)
    states = _rollout_states(32, _TEST_GEOMETRY)
    steps = 40

    expected = _scalar_rollout(case.env, states, _COAST_STRAIGHT, steps)
    actual = _vector_rollout(case.model, states, _COAST_STRAIGHT, steps)
    assert np.max(np.abs(expected - actual)) < _TOLERANCE

    # The rollout is only meaningful if it moved: arclength has to have advanced past at
    # least one boundary, and the particles must have seen more than one curvature.
    travelled = expected[:, EGO_ARCLENGTH_M] - states[:, EGO_ARCLENGTH_M]
    assert np.min(travelled) > _TEST_GEOMETRY.segment_starts[1]
    assert len(np.unique(_TEST_GEOMETRY.curvature_at(expected[:, EGO_ARCLENGTH_M]))) > 1


def test_torch_curvature_lookup_matches_the_numpy_profile() -> None:
    """The torch table lookup returns exactly what TrackGeometry.curvature_at returns.

    Purpose: The torch lookup is a deliberate re-implementation rather than a call into
        NumPy, so the two have to be pinned together or the map silently means two
        different things on the two sides

    Given: 4000 arclengths spanning several laps in both directions, plus every exact
        segment boundary and the lap length itself
    When: TrackMapCurvature and TrackGeometry.curvature_at are both evaluated on them
    Then: The results are bit-identical, boundaries included

    Test type: unit
    """
    rng = np.random.default_rng(13)
    arclengths = np.concatenate(
        [
            rng.uniform(-200.0, 200.0, size=4000),
            _TEST_GEOMETRY.segment_starts,
            np.array([_TEST_GEOMETRY.total_length_m, -_TEST_GEOMETRY.total_length_m]),
        ]
    )
    ego = np.zeros((arclengths.size, EGO_STATE_WIDTH))
    ego[:, EGO_ARCLENGTH_M] = arclengths

    lookup = TrackMapCurvature(_TEST_GEOMETRY, torch.device("cpu"), torch.float64)
    actual = lookup(_tensor(ego)).numpy()
    np.testing.assert_array_equal(actual, _TEST_GEOMETRY.curvature_at(arclengths))


def _real_profile() -> TrackGeometry:
    """The shipped circuit's measured profile: awkward boundaries over an awkward lap."""
    return TrackGeometry(
        segment_starts=np.array(
            [0.0, 58.0, 97.7062, 107.7062, 170.8872, 206.7537, 235.0379, 313.5778, 372.2208]
        ),
        segment_curvatures=np.array(
            [0.0, -0.04, 0.0, -0.05, 1.0 / 15.0, 0.0, -1.0 / 30.0, -1.0 / 30.0, 2.0 / 37.0]
        ),
        total_length_m=381.9074033363339,
    )


def _float32_lookup(geometry: TrackGeometry, arclengths: np.ndarray) -> np.ndarray:
    ego = np.zeros((arclengths.size, EGO_STATE_WIDTH))
    ego[:, EGO_ARCLENGTH_M] = arclengths
    lookup = TrackMapCurvature(geometry, torch.device("cpu"), torch.float32)
    return lookup(torch.as_tensor(ego, dtype=torch.float32)).numpy()


def test_float32_curvature_lookup_agrees_with_the_numpy_profile() -> None:
    """In float32 the lookup still picks the segment the float64 profile picks.

    Purpose: Every other parity test runs in float64, and the shipped default is float32.
        Segment boundaries on the real circuit are awkward numbers like 372.2208 m over a
        381.9074 m lap, so this is where the two dtypes would diverge if they were going to

    Given: The measured profile of the shipped circuit, its nine exact boundaries, and
        20000 arclengths spread over three laps in both directions
    When: The float32 lookup is compared to TrackGeometry.curvature_at
    Then: Every one agrees

    Test type: unit
    """
    geometry = _real_profile()
    rng = np.random.default_rng(17)
    arclengths = np.concatenate(
        [
            geometry.segment_starts,
            rng.uniform(-3.0, 3.0, size=20000) * geometry.total_length_m,
        ]
    )
    actual = _float32_lookup(geometry, arclengths)
    np.testing.assert_array_equal(actual, geometry.curvature_at(arclengths).astype(np.float32))


def test_float32_boundaries_a_whole_lap_away_are_off_by_at_most_one_segment() -> None:
    """A boundary reached after a full lap can read the neighbouring segment in float32.

    Purpose: Records a real limit rather than papering over it. A float32 arclength is up to
        1.1e-5 m from the boundary it means, so which side of that boundary it falls on is
        not recoverable -- not by holding the profile in float64, which measurably makes it
        worse, and not by any other lookup. What can be pinned is that the disagreement
        stays confined to boundaries and to one segment either way

    Given: The shipped profile's boundaries offset by exactly one and three laps, looked up
        through a probe profile whose "curvatures" are the segment indices, so the two sides
        can be compared by which segment they chose rather than by a curvature value two
        segments happen to share
    When: The float32 lookup is compared to the float64 profile
    Then: Any row that disagrees is within 1e-4 m of a boundary and lands one segment away,
        never further

    Test type: unit
    """
    geometry = _real_profile()
    starts, lap = geometry.segment_starts, geometry.total_length_m
    probe = TrackGeometry(
        segment_starts=starts,
        segment_curvatures=np.arange(len(starts), dtype=float),
        total_length_m=lap,
    )
    arclengths = np.concatenate([starts + lap, starts + 3.0 * lap, starts - lap])

    chosen = _float32_lookup(probe, arclengths)
    reference = probe.curvature_at(arclengths)
    disagreed = np.flatnonzero(chosen != reference)
    assert disagreed.size > 0, "the float32 limit this test documents no longer reproduces"

    distance = np.mod(arclengths[disagreed], lap)
    assert np.all(np.min(np.abs(distance[:, None] - starts[None, :]), axis=1) < 1e-4)
    assert np.all(np.abs(chosen[disagreed] - reference[disagreed]) == 1.0)


def test_curvature_estimate_is_read_live_and_matches_the_scalar_model() -> None:
    """A per-step curvature estimate reaches the torch rollout the step it changes.

    Purpose: Validates the other arm of the curvature source. Caching the estimate at
        construction would leave a mapless planner frozen on whichever corner it started
        in, and every single-step test would still pass

    Given: A scalar model holding one settable curvature, its vectorized twin, and a
        20-decision rollout run at two different curvature values
    When: The estimate is changed on the scalar model between the two runs
    Then: Both runs match the scalar model to 1e-9, and the two runs differ from each other

    Test type: unit
    """
    env = _FixedCurvatureModel(discount_factor=0.95, curvature_estimate=0.0, process_noise_std=0.0)
    model = _twin(env)
    states = _rollout_states(8, _TEST_GEOMETRY)

    outcomes = []
    for curvature in (0.02, -0.04):
        env.curvature_estimate = curvature
        expected = _scalar_rollout(env, states, _COAST_STRAIGHT, 20)
        actual = _vector_rollout(model, states, _COAST_STRAIGHT, 20)
        assert np.max(np.abs(expected - actual)) < _TOLERANCE, f"curvature {curvature} diverged"
        outcomes.append(actual)
    assert np.max(np.abs(outcomes[0] - outcomes[1])) > 1e-3


def test_a_numpy_scalar_curvature_estimate_is_recognised() -> None:
    """An estimate held as a NumPy scalar resolves the same as a Python float.

    Purpose: A curvature estimated from the on-road layer falls out of an array computation
        as a np.float32, which is not a Python float. Rejecting it would send a working
        model down the "cannot infer a curvature source" path for a reason that has nothing
        to do with the model

    Given: A scalar model whose curvature_estimate is a np.float32
    When: A vectorized twin is built from it and a five-decision rollout is run
    Then: Construction succeeds and the rollout matches the scalar model to 1e-9

    Test type: unit
    """
    env = _FixedCurvatureModel(
        discount_factor=0.95, curvature_estimate=np.float32(0.03), process_noise_std=0.0
    )
    states = _rollout_states(8, _TEST_GEOMETRY)
    expected = _scalar_rollout(env, states, _COAST_STRAIGHT, 5)
    actual = _vector_rollout(_twin(env), states, _COAST_STRAIGHT, 5)
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def _curved_on_road_observation(curvature: float) -> np.ndarray:
    """A raw occupancy grid whose on-road corridor bends, so a fit reads a real curvature."""
    grid = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    for row in range(GRID_CELLS):
        along = row * GRID_STEP_M - GRID_HALF_EXTENT_M + 0.5 * GRID_STEP_M
        across = 0.5 * curvature * along**2
        column = int(np.floor((across + GRID_HALF_EXTENT_M) / GRID_STEP_M))
        if 0 <= column < GRID_CELLS:
            grid[ON_ROAD_LAYER, row, column] = 1.0
    return grid


def _observed_case() -> _Case:
    """A live ObservedTrackModel that has already read one bending corridor, and its twin.

    The corridor is drawn at 0.08 1/m, well beyond anything on the real circuit. It has to
    be: the fit runs on 3 m cells over a 12 m window, so a realistic 0.04 bends the corridor
    by less than one cell and quantises back to a straight road. What is under test here is
    the *layer*, not the fit, and the layer needs a curvature that survives rasterisation.
    """
    env = ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0, cell_flip_prob=0.05)
    env.encode_observation(_curved_on_road_observation(0.08))
    assert abs(env.curvature_estimate) > 1e-3, "the fixture failed to produce a bending road"
    return _Case(env=env, model=_twin(env))


def test_observed_model_on_road_layer_matches_native() -> None:
    """The torch corridor render equals the mapless model's, cell for cell.

    Purpose: The mapless model predicts the road back in order to score the road it reads,
        so the layer is state-dependent and separates particles. A twin that kept the base
        model's all-ones layer would still return the right shape and the wrong weights

    Given: A live ObservedTrackModel that has fitted a bending corridor, and 64 random
        states spanning the lateral offsets and lane angles the ego actually takes
    When: Both models draw a noise-free on-road layer for each state
    Then: The two layers are identical, and the rendered corridor is neither empty nor the
        whole grid -- an all-ones or all-zeros layer would match trivially

    Test type: unit
    """
    case = _observed_case()
    rng = np.random.default_rng(14)
    states = _random_states(rng, case.env, 64)

    drawn = case.model.sample_observations(_tensor(states), _actions(np.zeros(64, dtype=int)))
    actual = drawn.numpy().reshape(64, 2, GRID_CELLS, GRID_CELLS)[:, ON_ROAD_LAYER]
    expected = np.stack(
        [
            np.asarray(case.env.sample_observation(states[i], None)["occupancy"])[ON_ROAD_LAYER]
            for i in range(states.shape[0])
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    marked = expected.reshape(64, -1).sum(axis=1)
    assert np.all(marked > 0) and np.all(marked < GRID_CELLS * GRID_CELLS)


def test_observed_model_observation_log_probs_match_native() -> None:
    """Occupancy log-densities match the mapless model, on-road term included.

    Purpose: Validates the full POMDP likelihood for the mapless arm -- the presence
        Bernoulli plus the on-road Bernoulli the base model declines to score

    Given: A live ObservedTrackModel with a fitted corridor, 64 random states, and
        observations drawn from the vectorized sampler
    When: observation_log_probs is compared to env.observation_log_probability per row
    Then: The maximum absolute difference is below 1e-9, every value is finite, and the
        on-road term is actually contributing rather than silently zero

    Test type: unit
    """
    torch.manual_seed(15)
    case = _observed_case()
    rng = np.random.default_rng(15)
    states = _random_states(rng, case.env, 64)
    indices = _actions(np.zeros(64, dtype=int))

    observations = case.model.sample_observations(_tensor(states), indices)
    expected = np.array(
        [
            case.env.observation_log_probability(states[i], 0, _occupancy_dict(observations[i]))[0]
            for i in range(states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(_tensor(states), indices, observations).numpy()
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected - actual)) < _TOLERANCE

    # The on-road term must be doing work: a zero contribution would make this test pass
    # while the mapless arm scored its road exactly as the mapped arm does.
    on_road = case.model.road_layer.log_probs(_tensor(states), observations).numpy()
    assert np.all(on_road < 0.0)


def test_a_mapped_model_keeps_the_all_ones_road_layer() -> None:
    """A model holding a track map renders road everywhere and scores it at zero.

    Purpose: Pins the other side of the dispatch. Rendering a corridor for a mapped model
        would cost 144 cells of arithmetic per particle to add a constant every particle
        shares, and the scalar model declines to for that reason

    Given: A vectorized twin of a KnownTrackModel and 16 random states
    When: The road layer is rendered and scored
    Then: Every cell is on-road and every log-probability is exactly zero

    Test type: unit
    """
    case = _build_case(discount_factor=0.95, process_noise_std=0.0)
    rng = np.random.default_rng(16)
    states = _tensor(_random_states(rng, case.env, 16))
    observations = case.model.sample_observations(states, _actions(np.zeros(16, dtype=int)))
    assert torch.all(case.model.road_layer.render(states) == 1.0)
    assert torch.all(case.model.road_layer.log_probs(states, observations) == 0.0)


def test_explicit_curvature_source_overrides_the_env() -> None:
    """A caller-supplied curvature source is used in place of the one read off the model.

    Purpose: Validates the escape hatch that keeps this module from needing an edit for
        every new subclass

    Given: A model carrying a track map, and a vectorized twin built with an explicit
        constant-curvature source instead
    When: One decision is propagated
    Then: The result matches a scalar model whose curvature is that same constant, not the
        one the map would have given

    Test type: unit
    """
    mapped = KnownTrackModel(
        discount_factor=0.95, track_geometry=_TEST_GEOMETRY, process_noise_std=0.0
    )
    fixed = _FixedCurvatureModel(
        discount_factor=0.95, curvature_estimate=0.05, process_noise_std=0.0
    )
    overridden = RacetrackVectorizedModel(
        mapped,
        device=torch.device("cpu"),
        dtype=torch.float64,
        curvature_source=ObservedCurvature(lambda: 0.05),
    )
    states = _rollout_states(8, _TEST_GEOMETRY)
    expected = _scalar_rollout(fixed, states, _COAST_STRAIGHT, 5)
    actual = _vector_rollout(overridden, states, _COAST_STRAIGHT, 5)
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_model_without_a_curvature_source_is_rejected() -> None:
    """A scalar model exposing neither a map nor an estimate cannot be twinned silently.

    Purpose: Guards the one failure that would not look like a failure -- defaulting to zero
        curvature gives a model that runs, scores, and drives straight through every corner

    Given: A scalar model with neither a track_geometry nor an curvature_estimate
    When: A vectorized twin is built from it with no explicit curvature source
    Then: ValueError is raised naming both attributes and the explicit argument

    Test type: unit
    """

    class _NoCurvature(RacetrackModelPOMDP):
        def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
            return np.zeros(len(ego))

    with pytest.raises(ValueError, match="Cannot infer a curvature source"):
        RacetrackVectorizedModel(_NoCurvature(discount_factor=0.95))


def _reward_states(env: RacetrackModelPOMDP) -> np.ndarray:
    """Four states spanning centred, off-centre, off-road, and crashed."""
    states = np.zeros((4, env.state_width))
    states[1, EGO_LAT] = 1.2
    states[2, EGO_LAT] = env.lane_half_width + 0.5
    states[3, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 1.0, 0.5, 0.0, 0.0]
    return states


def test_reward_matches_native_exactly(case: _Case) -> None:
    """Rewards equal the scalar racetrack reward across every branch.

    Purpose: Validates the centering, effort, collision, normalisation and on-road terms

    Given: Crafted next states spanning centred, off-centre, off-road and crashed, plus a
        batch of random ones, each scored under every action preset
    When: The vectorized reward is compared to env.reward per row
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    rng = np.random.default_rng(4)
    next_states = np.concatenate(
        [_reward_states(case.env), _random_states(rng, case.env, 64)], axis=0
    )
    states = np.zeros_like(next_states)
    for index in range(len(case.env.action_presets)):
        indices = np.full(next_states.shape[0], index)
        expected = np.array(
            [case.env.reward(states[i], index, next_states[i]) for i in range(next_states.shape[0])]
        )
        actual = case.model.rewards(
            _tensor(states), _actions(indices), _tensor(next_states)
        ).numpy()
        assert np.max(np.abs(expected - actual)) < _TOLERANCE, f"action {index} diverged"


def test_terminal_matches_native_exactly(case: _Case) -> None:
    """Terminal flags equal the scalar off-lane and collision check row by row.

    Purpose: Validates the batched terminal mask

    Given: Crafted off-road and crashed states plus 512 random ones
    When: terminal_mask is compared to env.is_terminal per row
    Then: Every entry agrees, and both terminal branches are actually exercised

    Test type: unit
    """
    rng = np.random.default_rng(5)
    states = np.concatenate([_reward_states(case.env), _random_states(rng, case.env, 512)], axis=0)
    expected = np.array([case.env.is_terminal(states[i]) for i in range(states.shape[0])])
    actual = case.model.terminal_mask(_tensor(states)).numpy()
    assert np.array_equal(expected, actual)
    assert bool(expected.any()) and not bool(expected.all())


def test_occupancy_log_probs_match_native_exactly(case: _Case) -> None:
    """Occupancy log-densities equal the scalar Bernoulli grid density row by row.

    Purpose: Validates the presence-grid likelihood, including the excluded on-road layer

    Given: Random next states and observations drawn from the vectorized sampler
    When: observation_log_probs is compared to env.observation_log_probability
    Then: The maximum absolute difference is below 1e-9 and every value is finite

    Test type: unit
    """
    torch.manual_seed(6)
    rng = np.random.default_rng(6)
    next_states = _random_states(rng, case.env, 128)
    indices = _actions(np.zeros(128, dtype=int))
    observations = case.model.sample_observations(_tensor(next_states), indices)
    expected = np.array(
        [
            case.env.observation_log_probability(
                next_states[i], 0, _occupancy_dict(observations[i])
            )[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(_tensor(next_states), indices, observations).numpy()
    assert actual.shape == (128,)
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_occupancy_log_prob_prefers_the_matching_grid(pomdp_case: _Case) -> None:
    """A grid that matches the state scores strictly above a corrupted one.

    Purpose: Validates that the presence likelihood actually discriminates particles

    Given: A state with one visible agent and its noise-free rendered observation
    When: That observation is scored, then scored again with twenty cells inverted
    Then: The matching grid scores strictly higher, and both scores are finite

    Test type: unit
    """
    state = np.zeros((1, pomdp_case.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 6.0, 0.0, 0.0, 0.0]
    clean = _build_case(discount_factor=0.95, process_noise_std=0.0, cell_flip_prob=0.0)
    matching = clean.model.sample_observations(_tensor(state), _actions(np.zeros(1, dtype=int)))
    mismatched = matching.clone()
    mismatched[0, 100:120] = 1.0 - mismatched[0, 100:120]
    indices = _actions(np.zeros(1, dtype=int))
    scores = [
        float(pomdp_case.model.observation_log_probs(_tensor(state), indices, grid)[0])
        for grid in (matching, mismatched)
    ]
    assert all(np.isfinite(scores))
    assert scores[0] > scores[1]


def test_rasteriser_axis_order_places_agent_along_then_across() -> None:
    """An agent 9 m ahead and 3 m to the right lands in cell (9, 5), not (5, 9).

    Purpose: Pins the empirically established axis order and the always-marked centre cell

    Given: A state with one present agent at rel_x = +9 m, rel_y = -3 m, and no cell flips
    When: A noise-free occupancy observation is sampled and reshaped to (2, 12, 12)
    Then: Exactly the ego centre cell (6, 6) and the cell (9, 5) are marked present

    Test type: unit
    """
    clean = _build_case(discount_factor=0.95, process_noise_std=0.0, cell_flip_prob=0.0)
    state = np.zeros((1, clean.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 9.0, -3.0, 0.0, 0.0]
    observation = clean.model.sample_observations(_tensor(state), _actions(np.zeros(1, dtype=int)))
    presence = observation[0].numpy().reshape(2, GRID_CELLS, GRID_CELLS)[0]
    assert sorted(map(tuple, np.argwhere(presence > 0.5))) == [(6, 6), (9, 5)]


def test_occupancy_sampler_flips_at_the_configured_rate(pomdp_case: _Case) -> None:
    """The sampled presence layer disagrees with the truth at cell_flip_prob.

    Purpose: Validates the Bernoulli flip sampler against the parameter it reads

    Given: One fixed state observed 4000 times by the vectorized sampler
    When: The fraction of cells differing from the noise-free grid is measured
    Then: It is within 0.005 of the model's cell_flip_prob, and on_road stays all ones

    Test type: unit
    """
    torch.manual_seed(7)
    clean = _build_case(discount_factor=0.95, process_noise_std=0.0, cell_flip_prob=0.0)
    state = np.zeros((1, pomdp_case.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 6.0, 3.0, 0.0, 0.0]
    truth = clean.model.sample_observations(_tensor(state), _actions(np.zeros(1, dtype=int)))
    batch = _tensor(np.tile(state, (4000, 1)))
    drawn = pomdp_case.model.sample_observations(batch, _actions(np.zeros(4000, dtype=int)))
    cells = GRID_CELLS * GRID_CELLS
    flipped = (drawn[:, :cells] != truth[0, :cells]).to(torch.float64).mean().item()
    assert abs(flipped - pomdp_case.env.cell_flip_prob) < 0.005
    assert torch.all(drawn[:, cells:] == 1.0)


def _mdp_case(**kwargs: Any) -> _Case:
    return _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        observation_mode=ObservationMode.MDP,
        **kwargs,
    )


def test_mdp_log_probs_match_native_exactly() -> None:
    """MDP log-densities equal the scalar diagonal-Gaussian density row by row.

    Purpose: Validates the MDP arm's ego and present-slot Gaussian likelihood

    Given: Random next states and random observations, so present and absent slots and
        both matching and mismatching readings all occur
    When: observation_log_probs is compared to env.observation_log_probability
    Then: The maximum absolute difference is below 1e-9

    Test type: unit
    """
    case = _mdp_case()
    assert isinstance(case.model, VectorizedGenerativeModel)
    rng = np.random.default_rng(8)
    next_states = _random_states(rng, case.env, 128)
    observations = rng.uniform(-8.0, 8.0, size=(128, case.model.observation_dim))
    expected = np.array(
        [
            case.env.observation_log_probability(
                next_states[i], 0, _kinematics_dict(_tensor(observations[i]), 4)
            )[0]
            for i in range(next_states.shape[0])
        ]
    )
    actual = case.model.observation_log_probs(
        _tensor(next_states), _actions(np.zeros(128, dtype=int)), _tensor(observations)
    ).numpy()
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_mdp_observation_dimension_and_sampler_moments() -> None:
    """MDP observations have the documented width and the scalar sampler's noise scale.

    Purpose: Validates the MDP observation layout and its per-channel noise standard
        deviations against the parameters read off the scalar model

    Given: One state with a single present agent, observed 20000 times
    When: The per-column standard deviation of the observation is measured
    Then: The width is 4 + 5K, and the ego, pose and velocity channels each match their
        configured standard deviation to within 3%

    Test type: unit
    """
    torch.manual_seed(9)
    case = _mdp_case()
    assert case.model.observation_dim == 4 + case.env.max_tracked_agents * AGENT_SLOT_WIDTH
    state = np.zeros((1, case.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 8.0, 1.0, 2.0, -1.0]
    drawn = case.model.sample_observations(
        _tensor(np.tile(state, (20000, 1))), _actions(np.zeros(20000, dtype=int))
    ).numpy()
    deviations = drawn.std(axis=0)
    assert np.max(np.abs(deviations[:4] - case.env.ego_pose_std)) < 0.03 * case.env.ego_pose_std
    assert (
        np.max(np.abs(deviations[5:7] - case.env.agent_pose_std)) < 0.03 * case.env.agent_pose_std
    )
    assert (
        np.max(np.abs(deviations[7:9] - case.env.agent_velocity_std))
        < 0.03 * case.env.agent_velocity_std
    )
    assert np.max(deviations[9:]) < _TOLERANCE  # absent slots carry no measurement


def test_process_noise_matches_the_configured_scale() -> None:
    """Transition noise perturbs the first six ego entries at process_noise_std.

    Purpose: Validates that sample_next_states reproduces the scalar model's process noise,
        including which entries it leaves alone

    Given: One state propagated 20000 times by a model with process_noise_std = 0.3
    When: The per-column standard deviation of the next states is measured
    Then: The six noisy ego entries match 0.3 within 3%, while the arclength and the agent
        slots stay deterministic to floating-point precision

    Test type: unit
    """
    torch.manual_seed(10)
    case = _build_case(discount_factor=0.95, process_noise_std=0.3)
    state = np.zeros((1, case.env.state_width))
    state[0, EGO_SPEED] = 8.0
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 12.0, 2.0, -1.0, 0.5]
    drawn = case.model.sample_next_states(
        _tensor(np.tile(state, (20000, 1))), _actions(np.zeros(20000, dtype=int))
    ).numpy()
    deviations = drawn.std(axis=0)
    assert np.max(np.abs(deviations[:6] - 0.3)) < 0.03 * 0.3
    # Not exactly zero: torch's elementwise kernels can differ in the last bit between the
    # vectorised body of a batch and its remainder, even on identical inputs.
    assert np.max(deviations[6:]) < _TOLERANCE


def test_method_shapes_and_dtypes(case: _Case) -> None:
    """Every generative method returns the documented shape and dtype.

    Purpose: Validates the tensor contract of the whole public surface

    Given: A batch of 16 random states and zero actions
    When: Each generative and key method is invoked
    Then: Shapes are [N, .] or [N], the terminal mask is bool and the keys are int64

    Test type: unit
    """
    rng = np.random.default_rng(11)
    states = _tensor(_random_states(rng, case.env, 16))
    indices = _actions(np.zeros(16, dtype=int))
    next_states = case.model.sample_next_states(states, indices)
    observations = case.model.sample_observations(next_states, indices)
    assert next_states.shape == (16, case.model.state_dim)
    assert next_states.dtype == torch.float64
    assert observations.shape == (16, case.model.observation_dim)
    assert case.model.rewards(states, indices, next_states).shape == (16,)
    assert case.model.terminal_mask(states).dtype == torch.bool
    assert case.model.observation_log_probs(next_states, indices, observations).shape == (16,)
    assert case.model.action_keys(indices).dtype == torch.int64
    assert case.model.observation_keys(observations).dtype == torch.int64


def test_defaults_to_cpu_and_float32() -> None:
    """A model built with no device runs on the CPU in float32.

    Purpose: Validates that the model is usable on a machine with no GPU

    Given: A model constructed without a device or dtype argument
    When: A short batch is propagated, observed, scored and hashed
    Then: Every tensor is a CPU float32 tensor and no CUDA call is made

    Test type: unit
    """
    model = RacetrackVectorizedModel(
        KnownTrackModel(discount_factor=0.95, track_geometry=_TEST_GEOMETRY)
    )
    assert model.device == torch.device("cpu")
    states = torch.zeros(5, model.state_dim)
    indices = torch.zeros(5, dtype=torch.int64)
    next_states = model.sample_next_states(states, indices)
    observations = model.sample_observations(next_states, indices)
    assert next_states.dtype == torch.float32
    assert observations.device.type == "cpu"
    assert model.rewards(states, indices, next_states).dtype == torch.float32
    assert model.observation_keys(observations).shape == (5,)


def test_keys_are_deterministic_and_discriminating(case: _Case) -> None:
    """Action and observation keys are stable and separate distinct inputs.

    Purpose: Validates the integer belief-tree key mappings

    Given: A fixed action vector and three observations, two of them distinct
    When: The key methods are called twice on the same inputs
    Then: Repeated calls agree and distinct observations receive distinct keys

    Test type: unit
    """
    indices = torch.tensor([0, 1, 2, 3])
    assert torch.equal(case.model.action_keys(indices), indices.to(torch.int64))
    base = torch.zeros(3, case.model.observation_dim, dtype=torch.float64)
    base[1, 0] = 1.0
    base[2, 5] = 1.0
    first = case.model.observation_keys(base)
    assert torch.equal(first, case.model.observation_keys(base))
    assert first[0] != first[1]
    assert first[0] != first[2]
    assert first[1] != first[2]


def test_observation_keys_separate_sparse_binary_grids(pomdp_case: _Case) -> None:
    """Two thousand distinct sparse occupancy grids hash to two thousand distinct keys.

    Purpose: Guards the hash weights against the collisions a small-prime weighting causes
        on binary vectors, where a handful of set cells sum into a range of a few thousand

    Given: 2000 grids that each set a different random subset of six presence cells
    When: They are hashed into belief-tree keys
    Then: Every key is distinct, so no two genuinely different observations merge

    Test type: unit
    """
    rng = np.random.default_rng(12)
    cells = GRID_CELLS * GRID_CELLS
    grids = np.zeros((2000, pomdp_case.model.observation_dim))
    for row in range(grids.shape[0]):
        grids[row, rng.choice(cells, size=6, replace=False)] = 1.0
    unique_grids = np.unique(grids, axis=0).shape[0]
    keys = pomdp_case.model.observation_keys(_tensor(grids)).numpy()
    assert len(np.unique(keys)) == unique_grids


def test_nonpositive_observation_resolution_raises() -> None:
    """A non-positive observation resolution is rejected at construction.

    Purpose: Validates the guard on the only knob the model does not read off the env

    Given: A valid scalar model and observation_resolution = 0.0
    When: A vectorized model is constructed from it
    Then: ValueError is raised

    Test type: unit
    """
    env = KnownTrackModel(discount_factor=0.95, track_geometry=_TEST_GEOMETRY)
    with pytest.raises(ValueError, match="observation_resolution must be positive"):
        RacetrackVectorizedModel(env, observation_resolution=0.0)
