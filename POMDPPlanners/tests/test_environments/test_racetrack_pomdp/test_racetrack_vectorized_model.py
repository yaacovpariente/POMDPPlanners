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

**The POMDP observation is five sensors flattened**, ``[ego_pose(4) | ego_speed(1) |
lane_pose(2) | curvature_ahead(L) | detections(K * 5)]``, and the scalar model works in a
dictionary. So the parity tests unflatten a torch row back into that dictionary with
:func:`_sensor_dict` and score the two sides against the same reading. Every offset comes off
the schema, never off a literal, because the two widths that move — the agent-slot count K
and the curvature-sample count L — are exactly what a hardcoded offset would silently
mis-slice.

The detection block is where the two are easiest to break apart silently: association is by
range rank, so a sampler that emitted its detections in slot order rather than packed
nearest-first would still produce a well-shaped observation that the density scores against
the wrong pairs. That property gets its own test against the scalar :func:`pack_detections`,
and the range gate and occlusion rule behind it get another. A detection row now carries
**both** components of relative velocity rather than a single closing rate, so the tests that
move one of them deliberately give ``rel_vx`` and ``rel_vy`` different values: a swapped
column pair scores identically to a correct one on any row where the two happen to agree.

**The ego-pose channel is the other place a term can go missing without anything looking
wrong.** Four numbers scored by three Gaussians of three different widths, one of them
wrapped, and a likelihood that dropped any of them would still be finite, still be smooth,
and still rank particles plausibly. So it is pinned three ways: against the scalar model with
all four entries displaced at once, entry by entry against each width's own penalty, and
across the ``+pi`` branch cut in both directions.

**Curvature is the other part that has to be watched.** The scalar model asks its subclass
where the road bends once per substep, and the torch model re-implements that lookup rather
than calling back into NumPy. So the parity tests deliberately put particles on a lap short
enough that a rollout crosses several segment boundaries, and both arms of the lookup — a
track map and a per-step scalar estimate — are compared against the scalar model over a
multi-decision rollout, not just one step. A single-step comparison would agree even if the
torch side froze the curvature at the value it started with, which is the exact bug the
arclength slot exists to prevent. The camera's curvature channel is a second, independent
use of the same map, so it is pinned separately.
"""

# pylint: disable=too-many-lines  # One test module per source module, as the test layout requires.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import pack_detections
from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    ObservedTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_VY,
    DETECTION_REL_X,
    DETECTION_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_POSE_ARCLENGTH,
    EGO_POSE_HEADING,
    EGO_POSE_X,
    EGO_POSE_Y,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    LANE_POSE_ANG,
    LANE_POSE_LAT,
    OBSERVED_EGO_POSE_WIDTH,
    OBSERVED_EGO_SPEED_WIDTH,
    OBSERVED_LANE_POSE_WIDTH,
    POMDP_OBS_CURVATURE_INDEX,
    POMDP_OBS_EGO_POSE_INDEX,
    POMDP_OBS_EGO_SPEED_INDEX,
    POMDP_OBS_LANE_POSE_INDEX,
    ObservationMode,
    pomdp_observation_width,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model import (
    CURVATURE_AHEAD_KEY,
    DETECTIONS_KEY,
    EGO_POSE_KEY,
    EGO_SPEED_KEY,
    LANE_POSE_KEY,
    SensorObservationModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry
from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
    ObservedCurvature,
    RacetrackVectorizedModel,
    TrackMapCurvature,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_road import (
    curvature_ahead_of,
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
# comparison is deterministic on both sides; the samplers are tested separately. The three
# differ in the two widths that reshape the flat observation -- the number of agent slots K
# and the number of curvature samples L -- so no test can pass by assuming the shipped 4 and 3.
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
        "curvature_lookahead_m": (5.0, 25.0),
        "max_detection_range_m": 25.0,
        "detection_position_std_m": 0.8,
        "detection_velocity_std": 0.5,
        "ego_position_std_m": 0.3,
        "ego_heading_std_rad": 0.02,
        "ego_arclength_std_m": 0.25,
        "presence_miss_prob": 0.1,
        "presence_false_alarm_prob": 0.08,
    },
]
_CASE_IDS = ["defaults", "coarse-steps", "custom-weights"]


class _FixedCurvatureModel(RacetrackModelPOMDP):
    """A scalar model whose curvature is one number, replaced from outside between steps.

    Stands in for the shipped model that reads curvature off its observations rather than
    from a map. What the vectorized twin has to reproduce about such a model is only that the
    estimate is read *live* — it changes every real step while the torch model is built once
    — so a settable attribute is the whole of the contract worth testing here.

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


def _noise_free_detection_case(**overrides: Any) -> _Case:
    """A case whose radar is deterministic: nothing missed, invented, or measured wrong.

    All four knobs are needed. Zero detection rates alone still leave the position and
    velocity widths blurring each report, so the draw stays random; zeroing those as well
    collapses a reading onto exactly the detections the sensor geometry predicts, which is
    the noise-free reading these tests compare against.

    Args:
        **overrides: Extra scalar-model arguments, so a caller can turn the range dial
            without losing the deterministic radar this builds.
    """
    return _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        presence_miss_prob=0.0,
        presence_false_alarm_prob=0.0,
        detection_position_std_m=0.0,
        detection_velocity_std=0.0,
        **overrides,
    )


def _sensor_arm(env: RacetrackModelPOMDP) -> SensorObservationModel:
    """The POMDP arm of a scalar model, narrowed from the protocol it is held behind."""
    arm = env.observation_model
    assert isinstance(arm, SensorObservationModel)
    return arm


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


def _zeros(count: int) -> torch.Tensor:
    return _actions(np.zeros(count, dtype=int))


def _random_states(rng: np.random.Generator, env: RacetrackModelPOMDP, count: int) -> np.ndarray:
    """Random states mixing present/absent slots, and every segment of the test lap.

    Arclengths span several laps in both directions, so the wrap and every segment of the
    profile are exercised by an ordinary parity run rather than only by the tests that
    target the lookup directly. Agent slots reach past the default 40 m detection range on
    purpose, so the range gate is exercised too.
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
    slots[..., 1] = rng.uniform(-45.0, 45.0, size=slots.shape[:2])
    slots[..., 2] = rng.uniform(-30.0, 30.0, size=slots.shape[:2])
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


def _sensor_dict(flat: torch.Tensor, env: RacetrackModelPOMDP) -> Dict[str, np.ndarray]:
    """Unflatten one POMDP observation row back into the scalar model's encoded form.

    The inverse of the layout the torch model documents, written against the schema's own
    offsets so the two cannot disagree about where a channel starts.
    """
    array = flat.numpy()
    lookahead = len(env.curvature_lookahead_m)
    detections = POMDP_OBS_CURVATURE_INDEX + lookahead
    return {
        EGO_POSE_KEY: array[
            POMDP_OBS_EGO_POSE_INDEX : POMDP_OBS_EGO_POSE_INDEX + OBSERVED_EGO_POSE_WIDTH
        ].copy(),
        EGO_SPEED_KEY: array[
            POMDP_OBS_EGO_SPEED_INDEX : POMDP_OBS_EGO_SPEED_INDEX + OBSERVED_EGO_SPEED_WIDTH
        ].copy(),
        LANE_POSE_KEY: array[
            POMDP_OBS_LANE_POSE_INDEX : POMDP_OBS_LANE_POSE_INDEX + OBSERVED_LANE_POSE_WIDTH
        ].copy(),
        CURVATURE_AHEAD_KEY: array[POMDP_OBS_CURVATURE_INDEX:detections].copy(),
        DETECTIONS_KEY: array[detections:].reshape(-1, DETECTION_SLOT_WIDTH).copy(),
    }


def _kinematics_dict(flat: torch.Tensor, num_agents: int) -> Dict[str, np.ndarray]:
    array = flat.numpy()
    return {
        "ego": array[:4].copy(),
        "agents": array[4:].reshape(num_agents, AGENT_SLOT_WIDTH).copy(),
    }


def _scalar_log_probs(case: _Case, states: np.ndarray, observations: torch.Tensor) -> np.ndarray:
    """Score every row through the scalar POMDP arm, one state against its own reading."""
    return np.array(
        [
            case.env.observation_log_probability(
                states[i], None, _sensor_dict(observations[i], case.env)
            )[0]
            for i in range(states.shape[0])
        ]
    )


def test_model_conforms_to_protocol_across_configs(case: _Case) -> None:
    """Every supported configuration yields a conforming vectorized model.

    Purpose: Validates structural protocol conformance and the reported dimensions

    Given: A vectorized model built from each supported scalar configuration
    When: It is checked against the runtime-checkable VectorizedGenerativeModel protocol
    Then: isinstance reports conformance and the observation width is the schema's own
        pomdp_observation_width rather than a literal

    Test type: unit
    """
    assert isinstance(case.model, VectorizedGenerativeModel)
    assert case.model.num_actions == len(case.env.action_presets)
    assert case.model.state_dim == case.env.state_width
    assert case.model.observation_dim == pomdp_observation_width(
        case.env.max_tracked_agents, len(case.env.curvature_lookahead_m)
    )


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


def test_track_map_curvature_ahead_matches_the_map_and_the_scalar_model() -> None:
    """Looking ahead down the map agrees with the profile and with KnownTrackModel.

    Purpose: The camera's curvature channel is a second, independent use of the track map,
        and it is the one term in the likelihood that scores a particle's arclength. A
        lookahead that wrapped differently, or that dropped the offset entirely, would leave
        every transition test passing while the channel scored the wrong place on the lap

    Given: 3000 arclengths spanning several laps in both directions, plus every exact
        segment boundary, and the three shipped lookahead distances
    When: TrackMapCurvature.curvature_ahead is compared to curvature_at(arclength + d) and
        to KnownTrackModel.curvature_ahead
    Then: All three agree exactly, and the channel really does report more than one
        curvature rather than repeating the value under the ego

    Test type: unit
    """
    rng = np.random.default_rng(31)
    arclengths = np.concatenate(
        [rng.uniform(-200.0, 200.0, size=3000), _TEST_GEOMETRY.segment_starts]
    )
    ego = np.zeros((arclengths.size, EGO_STATE_WIDTH))
    ego[:, EGO_ARCLENGTH_M] = arclengths
    lookahead = np.asarray(DEFAULT_CURVATURE_LOOKAHEAD_M, dtype=float)

    lookup = TrackMapCurvature(
        _TEST_GEOMETRY, torch.device("cpu"), torch.float64, lookahead_m=lookahead
    )
    actual = lookup.curvature_ahead(_tensor(ego)).numpy()
    expected = _TEST_GEOMETRY.curvature_at(arclengths[:, None] + lookahead[None, :])
    np.testing.assert_array_equal(actual, expected)

    env = KnownTrackModel(
        discount_factor=0.95, track_geometry=_TEST_GEOMETRY, process_noise_std=0.0
    )
    np.testing.assert_array_equal(actual, env.curvature_ahead(ego))
    # A channel that just repeated the curvature under the ego would satisfy the shape.
    assert np.any(actual[:, 0] != actual[:, -1])


def test_observed_curvature_ahead_holds_one_value_across_the_channel() -> None:
    """A mapless source reports its single live estimate at every lookahead distance.

    Purpose: Mirrors the scalar base class exactly, and the two properties are opposite
        sides of the same design: the value is *held* across the channel because a mapless
        planner has nothing further to say, and it is read *live* because the estimate is
        replaced every real step while the torch model is built once

    Given: An ObservedCurvature reading a mutable estimate, over four lookahead samples and
        six ego rows at different arclengths
    When: curvature_ahead is evaluated, the estimate is changed, and it is evaluated again
    Then: Every entry of every row holds the current estimate, and it follows the change

    Test type: unit
    """
    estimate = {"value": 0.02}
    source = ObservedCurvature(lambda: estimate["value"], lookahead_count=4)
    ego = np.zeros((6, EGO_STATE_WIDTH))
    ego[:, EGO_ARCLENGTH_M] = np.linspace(0.0, 120.0, 6)

    ahead = source.curvature_ahead(_tensor(ego))
    assert tuple(ahead.shape) == (6, 4)
    assert torch.all(ahead == 0.02)

    estimate["value"] = -0.031
    assert torch.all(source.curvature_ahead(_tensor(ego)) == -0.031)


def test_curvature_ahead_of_broadcasts_for_a_plain_callable_source() -> None:
    """A source with no curvature_ahead method still answers the camera's channel.

    Purpose: The lookahead is found by name so a caller may pass a bare lambda as the
        transition's curvature source. Without the fallback that caller would crash at the
        first observation rather than at construction, which is the worst place to find out

    Given: A plain callable returning the arclength-dependent curvature of each row, and a
        TrackMapCurvature that does expose curvature_ahead
    When: curvature_ahead_of is asked for a three-sample channel from each
    Then: The callable's answer is its own curvature held across the channel, and the map's
        answer is its lookahead rather than that broadcast

    Test type: unit
    """
    ego = np.zeros((5, EGO_STATE_WIDTH))
    ego[:, EGO_ARCLENGTH_M] = np.array([0.0, 13.0, 31.0, 46.0, 59.0])
    tensors = _tensor(ego)

    plain = curvature_ahead_of(lambda block: block[:, EGO_ARCLENGTH_M] * 0.001, tensors, 3)
    assert tuple(plain.shape) == (5, 3)
    np.testing.assert_array_equal(
        plain.numpy(), np.repeat((ego[:, EGO_ARCLENGTH_M] * 0.001)[:, None], 3, axis=1)
    )

    mapped = TrackMapCurvature(
        _TEST_GEOMETRY, torch.device("cpu"), torch.float64, lookahead_m=(10.0, 20.0, 30.0)
    )
    np.testing.assert_array_equal(
        curvature_ahead_of(mapped, tensors, 3).numpy(), mapped.curvature_ahead(tensors).numpy()
    )


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

    Purpose: A curvature read off the observation's curvature channel falls out of an array
        computation as a np.float32, which is not a Python float. Rejecting it would send a
        working model down the "cannot infer a curvature source" path for a reason that has
        nothing to do with the model

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


def _camera_reading(curvature: float) -> Tuple[np.ndarray, ...]:
    """A raw five-part reading whose camera reports one curvature at every distance."""
    return (
        np.zeros(OBSERVED_EGO_POSE_WIDTH, dtype=np.float32),
        np.array([10.0], dtype=np.float32),
        np.zeros(OBSERVED_LANE_POSE_WIDTH, dtype=np.float32),
        np.full(len(DEFAULT_CURVATURE_LOOKAHEAD_M), curvature, dtype=np.float32),
        np.zeros((DEFAULT_MAX_TRACKED_AGENTS, DETECTION_SLOT_WIDTH), dtype=np.float32),
    )


def _observed_case() -> _Case:
    """A live ObservedTrackModel that has already read one bending road, and its twin."""
    env = ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0)
    env.encode_observation(_camera_reading(0.04))
    assert abs(env.curvature_estimate) > 1e-3, "the fixture failed to produce a bending road"
    return _Case(env=env, model=_twin(env))


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


def test_pomdp_log_probs_match_native_exactly(case: _Case) -> None:
    """POMDP log-densities equal the scalar sensor density row by row.

    Purpose: Validates the whole POMDP likelihood -- the speedometer, both lane-camera
        terms, the curvature channel against the map, and the detection Bernoulli with its
        matched Gaussians and its clutter density

    Given: 128 random states spanning several laps and both lane-offset directions, and
        observations drawn from the vectorized sampler at float64
    When: observation_log_probs is compared to env.observation_log_probability
    Then: The maximum absolute difference is below 1e-9 and every value is finite

    Test type: unit
    """
    torch.manual_seed(6)
    rng = np.random.default_rng(6)
    next_states = _random_states(rng, case.env, 128)
    indices = _zeros(128)
    observations = case.model.sample_observations(_tensor(next_states), indices)
    expected = _scalar_log_probs(case, next_states, observations)
    actual = case.model.observation_log_probs(_tensor(next_states), indices, observations).numpy()
    assert actual.shape == (128,)
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_observed_model_log_probs_match_native() -> None:
    """The mapless arm's log-densities match too, with its held-constant curvature channel.

    Purpose: The mapless model predicts the curvature channel from the one estimate it read
        off its last observation, so the term is identical across particles. A twin that
        looked the road up in a map it does not have would score a different number for
        every particle and still look plausible

    Given: A live ObservedTrackModel that has read a bending road, 64 random states, and
        observations drawn from the vectorized sampler
    When: observation_log_probs is compared to env.observation_log_probability per row
    Then: The maximum absolute difference is below 1e-9 and every value is finite

    Test type: unit
    """
    torch.manual_seed(15)
    case = _observed_case()
    rng = np.random.default_rng(15)
    states = _random_states(rng, case.env, 64)
    indices = _zeros(64)

    observations = case.model.sample_observations(_tensor(states), indices)
    expected = _scalar_log_probs(case, states, observations)
    actual = case.model.observation_log_probs(_tensor(states), indices, observations).numpy()
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def _flatten_encoded(encoded: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the scalar model's five encoded keys in the flat layout's own order."""
    return np.concatenate(
        [
            np.asarray(encoded[EGO_POSE_KEY]).reshape(-1),
            np.asarray(encoded[EGO_SPEED_KEY]).reshape(-1),
            np.asarray(encoded[LANE_POSE_KEY]).reshape(-1),
            np.asarray(encoded[CURVATURE_AHEAD_KEY]).reshape(-1),
            np.asarray(encoded[DETECTIONS_KEY]).reshape(-1),
        ]
    ).astype(float)


def test_pomdp_observation_layout_round_trips_through_the_scalar_encoder(case: _Case) -> None:
    """A flat torch row survives a trip through the world's reading and back unchanged.

    Purpose: The flat layout and the scalar dictionary are two spellings of one observation,
        and every parity test in this module reads one as the other. A transposed channel --
        the lane pose before the speedometer, say, or detections in Fortran order -- would
        leave both sides internally consistent and the comparison meaningless

    Given: Observations drawn from the vectorized sampler for each supported configuration,
        which vary the agent-slot count K and the curvature-sample count L
    When: Each row is unflattened, handed to the scalar encoder as the world's five-part
        reading, and flattened again
    Then: The width is the schema's pomdp_observation_width, the encoded keys carry the
        documented shapes, and the round trip reproduces the row to float32 precision

    Test type: unit
    """
    torch.manual_seed(23)
    rng = np.random.default_rng(23)
    agents, lookahead = case.env.max_tracked_agents, len(case.env.curvature_lookahead_m)
    states = _random_states(rng, case.env, 16)
    observations = case.model.sample_observations(_tensor(states), _zeros(16))
    assert case.model.observation_dim == pomdp_observation_width(agents, lookahead)

    for row in range(states.shape[0]):
        unflattened = _sensor_dict(observations[row], case.env)
        encoded = case.env.encode_observation(
            (
                unflattened[EGO_POSE_KEY],
                unflattened[EGO_SPEED_KEY],
                unflattened[LANE_POSE_KEY],
                unflattened[CURVATURE_AHEAD_KEY],
                unflattened[DETECTIONS_KEY],
            )
        )
        assert encoded[EGO_POSE_KEY].shape == (OBSERVED_EGO_POSE_WIDTH,)
        assert encoded[EGO_SPEED_KEY].shape == (OBSERVED_EGO_SPEED_WIDTH,)
        assert encoded[LANE_POSE_KEY].shape == (OBSERVED_LANE_POSE_WIDTH,)
        assert encoded[CURVATURE_AHEAD_KEY].shape == (lookahead,)
        assert encoded[DETECTIONS_KEY].shape == (agents, DETECTION_SLOT_WIDTH)
        np.testing.assert_allclose(
            _flatten_encoded(encoded), observations[row].numpy(), rtol=1e-6, atol=1e-6
        )


def test_sampled_detections_are_packed_like_the_scalar_pack_detections() -> None:
    """Drawn detections come out nearest-first with the empty slots trailing.

    Purpose: The density associates the i-th predicted detection with the i-th reported one,
        so the packing *is* the association. A sampler that left its reports in slot order
        would still produce a correctly shaped reading, and every residual in the likelihood
        would then be taken against the wrong pair

    Given: 512 random states drawn once each at the default detection rates, so misses,
        false alarms and heavy-tailed clutter reports all occur
    When: Each drawn detection block is stripped back to its reported rows and re-packed by
        the scalar pack_detections
    Then: The drawn reading is the schema's own width, the re-pack reproduces the drawn block
        exactly -- which holds only if the draw was already ordered by measured range with its
        empty slots trailing -- and the fixture really did produce blocks that are neither
        always full nor always empty

    Test type: unit
    """
    torch.manual_seed(25)
    case = _build_case(discount_factor=0.95, process_noise_std=0.0)
    rng = np.random.default_rng(25)
    states = _random_states(rng, case.env, 512)
    agents, lookahead = case.env.max_tracked_agents, len(case.env.curvature_lookahead_m)

    drawn = case.model.sample_observations(_tensor(states), _zeros(512)).numpy()
    assert drawn.shape == (512, pomdp_observation_width(agents, lookahead))
    blocks = drawn[:, POMDP_OBS_CURVATURE_INDEX + lookahead :].reshape(
        512, agents, DETECTION_SLOT_WIDTH
    )
    counts = []
    for row in range(blocks.shape[0]):
        reported = blocks[row][blocks[row][:, DETECTION_PRESENT] > 0.5]
        counts.append(len(reported))
        # Everything past the presence flag is the report pack_detections takes: position and
        # both velocity components, sliced off the schema so the widening from 4 to 5 lands.
        repacked = pack_detections(reported[:, DETECTION_REL_X:], agents)
        np.testing.assert_array_equal(repacked, blocks[row])
    assert min(counts) == 0 and max(counts) > 1


def _occlusion_state(env: RacetrackModelPOMDP) -> np.ndarray:
    """One state whose four slots are: visible, occluded, out of range, visible.

    The second slot sits directly behind the first at twice the range, well inside the
    angular half-width a 1 m-wide blocker subtends at 10 m; the third is beyond the 40 m
    gate as well as behind both; the fourth is off to the side and clear of everything.

    Both visible slots carry ``rel_vx != rel_vy`` on purpose, so a reading that transposed
    the two velocity columns would not reproduce them.
    """
    state = np.zeros((1, env.state_width))
    slots = [
        [1.0, 10.0, 0.0, -3.0, 1.0],
        [1.0, 20.0, 0.0, 1.0, 0.0],
        [1.0, 50.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 15.0, 2.0, -4.0],
    ]
    state[0, EGO_STATE_WIDTH:] = np.asarray(slots, dtype=float).reshape(-1)
    return state


def test_occlusion_and_range_gate_agree_with_the_scalar_sensor() -> None:
    """Torch and NumPy hide the same vehicles, and report the same ones the same way.

    Purpose: Whether a slot should have produced a detection decides the Bernoulli's
        predicted count, which shifts every rank behind it. Two implementations of the
        occlusion geometry that disagreed by one vehicle would mis-associate the whole
        reading while both still returned finite, plausible weights

    Given: One state whose four filled slots are a visible car, a car directly behind it, a
        car past the range gate, and a clear car to the side -- observed by a model with the
        detection rates and both measurement widths at zero
    When: The drawn reading is compared to the scalar model's predicted_detections
    Then: Both report exactly the two visible cars, nearest first, with the same relative
        positions and the same *two-component* relative velocities, and the two hidden slots
        appear in neither

    Test type: unit
    """
    case = _noise_free_detection_case()
    state = _occlusion_state(case.env)
    predicted = _sensor_arm(case.env).predicted_detections(state[0])

    drawn = case.model.sample_observations(_tensor(state), _zeros(1))
    detections = _sensor_dict(drawn[0], case.env)[DETECTIONS_KEY]
    reported = detections[detections[:, DETECTION_PRESENT] > 0.5]

    assert len(predicted) == 2, "the fixture no longer hides two of its four vehicles"
    assert predicted.shape[1] == DETECTION_SLOT_WIDTH - 1
    assert len(reported) == len(predicted)
    np.testing.assert_allclose(reported[:, DETECTION_REL_X:], predicted, atol=_TOLERANCE)
    np.testing.assert_allclose(predicted[:, :2], np.array([[10.0, 0.0], [0.0, 15.0]]))
    # The whole relative velocity, not its projection onto the line of sight. Slot 3 crosses
    # the ego abeam at (2, -4); a radial-only reading would have reported -4 alone, and the
    # tangential 2 m/s -- the component that says which way it is crossing -- would be gone.
    np.testing.assert_allclose(
        predicted[:, 2:], np.array([[-3.0, 1.0], [2.0, -4.0]]), atol=_TOLERANCE
    )


def test_detection_log_prob_prefers_the_matching_reading(pomdp_case: _Case) -> None:
    """A reading that matches the state's traffic scores strictly above a moved one.

    Purpose: Validates that the detection likelihood actually discriminates particles,
        rather than returning a finite number that happens to agree between the two models

    Given: A state with one visible agent and its noise-free drawn reading
    When: That reading is scored, then scored again with the detection displaced by 6 m
    Then: The matching reading scores strictly higher, and both scores are finite

    Test type: unit
    """
    clean = _noise_free_detection_case()
    state = np.zeros((1, clean.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 12.0, 0.0, -2.0, 0.0]
    matching = clean.model.sample_observations(_tensor(state), _zeros(1))
    moved = matching.clone()
    moved[
        0, POMDP_OBS_CURVATURE_INDEX + len(clean.env.curvature_lookahead_m) + DETECTION_REL_X
    ] += 6.0

    scores = [
        float(pomdp_case.model.observation_log_probs(_tensor(state), _zeros(1), reading)[0])
        for reading in (matching, moved)
    ]
    assert all(np.isfinite(scores))
    assert scores[0] > scores[1]


# The ego-pose channel's four entries and the four state slots each one reports, paired in
# the channel's own order. Written as schema constants rather than as 0, 1, 2, 6 so a test
# that shifts "the arclength entry" cannot end up shifting the heading one.
_POSE_CHANNEL = (EGO_POSE_X, EGO_POSE_Y, EGO_POSE_HEADING, EGO_POSE_ARCLENGTH)
_POSE_SLOTS = (EGO_X, EGO_Y, EGO_HEADING, EGO_ARCLENGTH_M)


def _ego_pose_case() -> _Case:
    """A case whose three ego-pose widths are all different from each other.

    Deliberate: the channel is four numbers scored by three Gaussians, and at equal widths a
    likelihood that scored the arclength residual at the heading's sigma would be
    indistinguishable from a correct one.
    """
    return _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        ego_position_std_m=0.2,
        ego_heading_std_rad=0.05,
        ego_arclength_std_m=0.4,
    )


def _pose_index(entry: int) -> int:
    """Flat index of one ego-pose entry, derived from the schema rather than counted."""
    return POMDP_OBS_EGO_POSE_INDEX + entry


def _observation_with_exact_pose(case: _Case, states: np.ndarray, seed: int) -> np.ndarray:
    """A drawn reading whose ego-pose block is replaced by the state's own pose exactly.

    Zero residual on all four entries, so a later per-entry shift of ``d`` costs exactly the
    Gaussian's own ``0.5 * (d / sigma)^2`` and nothing has to be solved for.
    """
    torch.manual_seed(seed)
    count = states.shape[0]
    flat = case.model.sample_observations(_tensor(states), _zeros(count)).numpy().copy()
    block = slice(POMDP_OBS_EGO_POSE_INDEX, POMDP_OBS_EGO_POSE_INDEX + OBSERVED_EGO_POSE_WIDTH)
    flat[:, block] = states[:, list(_POSE_SLOTS)]
    return flat


def _torch_scores(case: _Case, states: np.ndarray, flat: np.ndarray) -> np.ndarray:
    return case.model.observation_log_probs(
        _tensor(states), _zeros(states.shape[0]), _tensor(flat)
    ).numpy()


def test_the_ego_pose_channel_matches_the_scalar_model_entry_by_entry() -> None:
    """All four ego-pose entries are scored, each against its own state slot and width.

    Purpose: The ego pose is the channel this redesign added, and it is the one that pins a
        particle's arclength. Four numbers, three widths, one wrapped residual -- a
        likelihood that dropped one of them, or took it against the wrong slot, would still
        be finite and smooth and would still rank particles plausibly. The parity batch
        displaces all four at once so no term can cancel against another, and the per-entry
        shifts then charge each width separately so a transposed pair fails

    Given: 32 random states spanning several laps, a model whose position, heading and
        arclength widths are 0.2 m, 0.05 rad and 0.4 m, and readings whose pose block is
        first displaced on every entry and then set exactly to the truth
    When: Both models score the displaced reading, and the torch model then scores four more
        readings, each shifting one pose entry by 0.3
    Then: The displaced reading matches the scalar model, and each shift costs exactly that
        entry's own Gaussian penalty -- three distinct numbers, so no width is standing in
        for another

    Test type: unit
    """
    case = _ego_pose_case()
    rng = np.random.default_rng(41)
    states = _random_states(rng, case.env, 32)
    exact = _observation_with_exact_pose(case, states, seed=41)

    # Every entry off by a different amount, and by a different amount in every row: a term
    # taken against the wrong state slot cannot come out equal to one taken against the right
    # one, and a missing term cannot be masked by a neighbour's.
    displaced = exact.copy()
    displaced[
        :, POMDP_OBS_EGO_POSE_INDEX : POMDP_OBS_EGO_POSE_INDEX + OBSERVED_EGO_POSE_WIDTH
    ] += rng.uniform(-0.6, 0.6, size=(32, OBSERVED_EGO_POSE_WIDTH))
    np.testing.assert_allclose(
        _torch_scores(case, states, displaced),
        _scalar_log_probs(case, states, _tensor(displaced)),
        atol=_TOLERANCE,
    )

    base = _torch_scores(case, states, exact)
    np.testing.assert_allclose(
        base, _scalar_log_probs(case, states, _tensor(exact)), atol=_TOLERANCE
    )
    shift = 0.3
    widths = (
        case.env.ego_position_std_m,
        case.env.ego_position_std_m,
        case.env.ego_heading_std_rad,
        case.env.ego_arclength_std_m,
    )
    for entry, width in zip(_POSE_CHANNEL, widths):
        moved = exact.copy()
        moved[:, _pose_index(entry)] += shift
        penalty = 0.5 * (shift / width) ** 2
        assert np.all(
            base - _torch_scores(case, states, moved) == pytest.approx(penalty)
        ), f"ego-pose entry {entry} is not scored at its own width"


def test_the_ego_pose_heading_residual_wraps_across_the_branch_cut() -> None:
    """A heading read on the far side of +pi costs what its small geometric error costs.

    Purpose: Both models wrap this residual, so a parity test that never crosses the cut
        would not notice if one of them stopped -- the two would still agree everywhere it
        looked. Crossing it is also the case that matters: the racetrack ego drives a closed
        loop, so it passes through +pi every lap, and an unwrapped residual would charge it
        6.28 rad of error for a hundredth of a radian and annihilate the particle

    Given: 24 states, alternating rows pointing just short of +pi and just past -pi, and
        three readings of each -- the heading exactly right, 0.012 rad away on the *far* side
        of the cut, and 0.012 rad away on the near side
    When: Both models score all three
    Then: The torch scores match the scalar model's on every reading, the crossing reading
        scores identically to the non-crossing one at the same geometric error, and both are
        below the exact reading by exactly the heading Gaussian's own penalty -- which is
        what rules out the two sides agreeing by both ignoring the term

    Test type: unit
    """
    case = _ego_pose_case()
    rng = np.random.default_rng(43)
    states = _random_states(rng, case.env, 24)
    states[0::2, EGO_HEADING] = np.pi - 0.004
    states[1::2, EGO_HEADING] = -np.pi + 0.004
    exact = _observation_with_exact_pose(case, states, seed=43)
    heading = _pose_index(EGO_POSE_HEADING)
    error = 0.012

    # 0.012 rad from every state, reached the long way round: a row at +pi - 0.004 is scored
    # against a reading at -pi + 0.008, and a row at -pi + 0.004 against one at +pi - 0.008.
    across = exact.copy()
    across[0::2, heading] = -np.pi + 0.008
    across[1::2, heading] = np.pi - 0.008
    # The same 0.012 rad, staying on the state's own side of the cut.
    near = exact.copy()
    near[0::2, heading] = np.pi - 0.016
    near[1::2, heading] = -np.pi + 0.016

    scores = {}
    for name, flat in (("exact", exact), ("across", across), ("near", near)):
        scores[name] = _torch_scores(case, states, flat)
        np.testing.assert_allclose(
            scores[name], _scalar_log_probs(case, states, _tensor(flat)), atol=_TOLERANCE
        )
    assert np.all(np.isfinite(scores["across"]))
    np.testing.assert_allclose(scores["across"], scores["near"], atol=_TOLERANCE)
    penalty = 0.5 * (error / case.env.ego_heading_std_rad) ** 2
    assert np.all(scores["exact"] - scores["across"] == pytest.approx(penalty))


def _crossing_traffic_state(env: RacetrackModelPOMDP) -> np.ndarray:
    """Two visible cars whose rel_vx and rel_vy differ, and differ from each other's.

    One ahead closing while sliding left, one abeam to the right crossing the ego's path.
    Every velocity component is a distinct number on purpose: a likelihood that transposed
    the two velocity columns, or scored one of them twice, comes out identical on any row
    where ``rel_vx == rel_vy``, and that row proves nothing.
    """
    state = np.zeros((1, env.state_width))
    slots = [
        [1.0, 8.0, 0.0, -3.0, 1.5],
        [1.0, 0.0, -14.0, 2.5, -4.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    state[0, EGO_STATE_WIDTH:] = np.asarray(slots, dtype=float).reshape(-1)
    return state


def _detection_column(env: RacetrackModelPOMDP, channel: int) -> np.ndarray:
    """Flat indices of one detection-row column, one per slot, derived from the schema."""
    start = POMDP_OBS_CURVATURE_INDEX + len(env.curvature_lookahead_m)
    return start + np.arange(env.max_tracked_agents) * DETECTION_SLOT_WIDTH + channel


def test_both_detection_velocity_components_are_scored_and_are_not_interchangeable() -> None:
    """A detection's full relative velocity is scored, both components, one width each.

    Purpose: Replaces the invariant this redesign removed. The velocity half of a detection
        used to be a single closing rate, so the crossing component was unobservable and a
        test asserted as much; now both components are reported and both are scored, and
        perturbing rel_vy has to move the likelihood. The swap check is the other half:
        two components sharing one width are symmetric, so only a row whose rel_vx and
        rel_vy differ can tell a correct pairing from a transposed one

    Given: A state with two visible cars whose four velocity components are four different
        numbers, a noise-free radar that reports them exactly, and a scoring model whose
        detection velocity width is 0.4 m/s
    When: The exact reading is scored, then one with rel_vx shifted, one with rel_vy shifted
        by the same amount, and one with the two columns transposed
    Then: Every reading matches the scalar model, each single-component shift costs the same
        strictly positive penalty at the shared width, and the transposed reading scores
        strictly below the exact one

    Test type: unit
    """
    torch.manual_seed(45)
    clean = _noise_free_detection_case()
    case = _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        detection_position_std_m=0.6,
        detection_velocity_std=0.4,
    )
    state = _crossing_traffic_state(case.env)
    exact = clean.model.sample_observations(_tensor(state), _zeros(1)).numpy().copy()
    rows = _sensor_dict(_tensor(exact[0]), case.env)[DETECTIONS_KEY]
    assert int(np.count_nonzero(rows[:, DETECTION_PRESENT] > 0.5)) == 2

    lateral = _detection_column(case.env, DETECTION_REL_VX)
    vertical = _detection_column(case.env, DETECTION_REL_VY)
    shift = 0.5
    readings = {"exact": exact}
    for name, columns in (("moved_vx", lateral), ("moved_vy", vertical)):
        readings[name] = exact.copy()
        readings[name][:, columns] += shift
    readings["swapped"] = exact.copy()
    readings["swapped"][:, lateral] = exact[:, vertical]
    readings["swapped"][:, vertical] = exact[:, lateral]

    scores = {}
    for name, flat in readings.items():
        scores[name] = _torch_scores(case, state, flat)
        np.testing.assert_allclose(
            scores[name], _scalar_log_probs(case, state, _tensor(flat)), atol=_TOLERANCE
        )

    # Both reported rows move, so each shift costs two entries at the shared velocity width.
    penalty = 2.0 * 0.5 * (shift / case.env.detection_velocity_std) ** 2
    assert penalty > 0.0
    for name in ("moved_vx", "moved_vy"):
        assert scores["exact"] - scores[name] == pytest.approx(penalty), f"{name} is unscored"
    assert float(scores["swapped"][0]) < float(scores["exact"][0])


def _spread_traffic_state(env: RacetrackModelPOMDP) -> np.ndarray:
    """Four cars at four bearings and four ranges, three of them past the default gate."""
    state = np.zeros((1, env.state_width))
    slots = [
        [1.0, 30.0, 0.0, -2.0, 1.0],
        [1.0, 0.0, 60.0, 3.0, -1.5],
        [1.0, -90.0, 0.0, 0.5, 4.0],
        [1.0, 0.0, -120.0, -1.5, -3.5],
    ]
    state[0, EGO_STATE_WIDTH:] = np.asarray(slots, dtype=float).reshape(-1)
    return state


def test_the_range_dial_hides_far_traffic_and_still_charges_a_slot_inside_the_gate() -> None:
    """Shrinking max_detection_range_m removes cars from the reading and from its price.

    Purpose: This is the environment's headline dial, and the torch model is what VOPP turns
        it on. It has to act in two places at once: on the *sampler*, where a car past the
        gate produces no row, and on the *density*, where a particle placing a car inside the
        gate must still pay a finite price for a reading that does not show it. That second
        half is the inference mechanism -- an empty reading is what kills particles that put
        a car in front of you -- and a model that only gated the sampler would look right

    Given: One car 30 m ahead, observed by noise-free radars gated at 10 m and at 1e9 m; and
        an all-empty detection block scored by two models identical but for a 10 m and a 40 m
        gate
    When: The readings are drawn and the empty one is scored by both models
    Then: The 10 m radar reports nothing while the 1e9 m one reports every car exactly and
        nearest-first, and the empty reading costs the 40 m model strictly more than the 10 m
        model -- finitely, in both cases, because the car leaves the gate every few steps

    Test type: unit
    """
    torch.manual_seed(47)
    narrow = _noise_free_detection_case(max_detection_range_m=10.0)
    wide = _noise_free_detection_case(max_detection_range_m=1.0e9)
    state = _spread_traffic_state(narrow.env)

    hidden = _sensor_dict(
        narrow.model.sample_observations(_tensor(state), _zeros(1))[0], narrow.env
    )
    assert not np.any(hidden[DETECTIONS_KEY][:, DETECTION_PRESENT] > 0.5)

    drawn = wide.model.sample_observations(_tensor(state), _zeros(1))
    seen = _sensor_dict(drawn[0], wide.env)[DETECTIONS_KEY]
    reported = seen[seen[:, DETECTION_PRESENT] > 0.5]
    assert len(reported) == wide.env.max_tracked_agents
    np.testing.assert_allclose(
        reported[:, DETECTION_REL_X:],
        np.array(
            [
                [30.0, 0.0, -2.0, 1.0],
                [0.0, 60.0, 3.0, -1.5],
                [-90.0, 0.0, 0.5, 4.0],
                [0.0, -120.0, -1.5, -3.5],
            ]
        ),
        atol=_TOLERANCE,
    )

    # Two models differing in the gate and in nothing else, so every other term of the
    # likelihood cancels and the difference below is the detection Bernoulli alone.
    near_sighted = _build_case(
        discount_factor=0.95, process_noise_std=0.0, max_detection_range_m=10.0
    )
    far_sighted = _build_case(discount_factor=0.95, process_noise_std=0.0)
    empty = drawn.numpy().copy()
    empty[:, POMDP_OBS_CURVATURE_INDEX + len(far_sighted.env.curvature_lookahead_m) :] = 0.0

    prices = {}
    for name, gated in (("near", near_sighted), ("far", far_sighted)):
        prices[name] = _torch_scores(gated, state, empty)
        np.testing.assert_allclose(
            prices[name], _scalar_log_probs(gated, state, _tensor(empty)), atol=_TOLERANCE
        )
        assert np.all(np.isfinite(prices[name])), f"{name} charged an empty reading -inf"
    assert float(prices["far"][0]) < float(prices["near"][0])

    # The reading the 40 m model expects: the one car inside its gate and nothing else. It
    # must beat the empty one, or the missing row is costing the particle nothing.
    block = POMDP_OBS_CURVATURE_INDEX + len(far_sighted.env.curvature_lookahead_m)
    single = empty.copy()
    single[0, block : block + DETECTION_SLOT_WIDTH] = [1.0, 30.0, 0.0, -2.0, 1.0]
    matched = _torch_scores(far_sighted, state, single)
    np.testing.assert_allclose(
        matched, _scalar_log_probs(far_sighted, state, _tensor(single)), atol=_TOLERANCE
    )
    assert float(matched[0]) > float(prices["far"][0])


def test_the_ego_speed_entry_matches_the_scalar_model(pomdp_case: _Case) -> None:
    """The torch observation's first entry is the scalar model's ego_speed, and is scored.

    Purpose: The torch model is what VOPP runs. If it omitted the speedometer, or carried it
        without scoring it, the whole channel would be invisible to the planner it matters
        most for while every scalar test still passed

    Given: 32 random states and observations drawn from them, with the speed entry
        deliberately offset from each state's own speed
    When: Both models score them
    Then: The width is the schema's pomdp_observation_width, the log-densities agree with
        the scalar model's, and moving only the speed entry moves the torch score

    Test type: unit
    """
    torch.manual_seed(21)
    rng = np.random.default_rng(21)
    states = _random_states(rng, pomdp_case.env, 32)
    assert pomdp_case.model.observation_dim == pomdp_observation_width(
        pomdp_case.env.max_tracked_agents, len(pomdp_case.env.curvature_lookahead_m)
    )

    drawn = pomdp_case.model.sample_observations(_tensor(states), _zeros(32))
    flat = drawn.numpy().copy()
    flat[:, POMDP_OBS_EGO_SPEED_INDEX] = states[:, EGO_SPEED] + rng.uniform(-3.0, 3.0, size=32)

    torch_scores = pomdp_case.model.observation_log_probs(
        _tensor(states), _zeros(32), _tensor(flat)
    ).numpy()
    np.testing.assert_allclose(
        torch_scores, _scalar_log_probs(pomdp_case, states, _tensor(flat)), atol=_TOLERANCE
    )

    # Moving only the speedometer moves the score, so the entry is genuinely read.
    shifted = flat.copy()
    shifted[:, POMDP_OBS_EGO_SPEED_INDEX] += 1.0
    shifted_scores = pomdp_case.model.observation_log_probs(
        _tensor(states), _zeros(32), _tensor(shifted)
    ).numpy()
    assert np.all(np.abs(shifted_scores - torch_scores) > 1e-6)


def test_the_sampled_ego_speed_is_centred_on_the_state_at_the_configured_width() -> None:
    """The torch speedometer sampler draws around the true speed at ego_speed_std.

    Purpose: Pins the sampler against the parameter it reads. A sampler centred on zero, or
        one that ignored the width, would still produce a plausible-looking observation of
        the documented width

    Given: One state at 10 m/s observed 4000 times by a model with ego_speed_std = 0.4
    When: The mean and standard deviation of the ego-speed entry are measured
    Then: They match 10.0 and 0.4 within sampling error

    Test type: unit
    """
    torch.manual_seed(27)
    case = _build_case(discount_factor=0.95, process_noise_std=0.0, ego_speed_std=0.4)
    state = np.zeros((1, case.env.state_width))
    state[0, EGO_SPEED] = 10.0
    batch = _tensor(np.tile(state, (4000, 1)))

    drawn = case.model.sample_observations(batch, _zeros(4000))
    speeds = drawn[:, POMDP_OBS_EGO_SPEED_INDEX].numpy()

    assert abs(float(speeds.mean()) - 10.0) < 0.03
    assert abs(float(speeds.std()) - 0.4) < 0.03


def test_the_lane_pose_entries_are_scored_and_wrap_like_the_scalar_model() -> None:
    """The torch model reads the lane camera's two entries, and wraps its angle residual.

    Purpose: The torch model is what VOPP runs, so a lane channel carried there but not
        scored would leave the planner exactly as lane-blind as before while every scalar
        test passed. The wrap is checked in the same test because an unwrapped torch
        residual is the one way the two sides can disagree while both look right

    Given: 24 random states pinned just inside the +pi branch cut, observations whose lane
        entries match them exactly, and two perturbations -- one sliding the lateral entry,
        one moving the angle entry across the cut
    When: Both models score all three
    Then: The torch scores match the scalar model's, the lateral slide moves the score by
        exactly the Gaussian's own penalty, and the wrapped angle scores as the small
        residual it geometrically is

    Test type: unit
    """
    torch.manual_seed(77)
    case = _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        lane_lateral_std_m=0.05,
        lane_heading_std_rad=0.1,
    )
    rng = np.random.default_rng(77)
    states = _random_states(rng, case.env, 24)
    states[:, EGO_ANG] = np.pi - 0.01
    lateral_index = POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_LAT
    angle_index = POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_ANG

    flat = case.model.sample_observations(_tensor(states), _zeros(24)).numpy().copy()
    flat[:, lateral_index] = states[:, EGO_LAT]
    flat[:, angle_index] = states[:, EGO_ANG]

    torch_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(flat)
    ).numpy()
    np.testing.assert_allclose(
        torch_scores, _scalar_log_probs(case, states, _tensor(flat)), atol=_TOLERANCE
    )

    slid = flat.copy()
    slid[:, lateral_index] += 0.5
    slid_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(slid)
    ).numpy()
    assert np.all(torch_scores - slid_scores == pytest.approx(0.5 * (0.5 / 0.05) ** 2))

    # 0.02 rad away from every state, but on the far side of the branch cut.
    across = flat.copy()
    across[:, angle_index] = -np.pi + 0.01
    near = flat.copy()
    near[:, angle_index] = np.pi - 0.03
    wrapped_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(across)
    ).numpy()
    near_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(near)
    ).numpy()
    np.testing.assert_allclose(wrapped_scores, near_scores, atol=_TOLERANCE)


def test_the_sampled_lane_pose_is_centred_on_the_state_at_the_configured_widths() -> None:
    """The torch lane-camera sampler draws around the true pose at its two widths.

    Purpose: Pins the sampler against the parameters it reads. The two widths are a metre
        and a radian and are easy to transpose; a sampler using one for both would still
        produce a plausible-looking observation

    Given: One state 0.7 m off centre and 0.2 rad across the lane, observed 4000 times by
        a model with widths of 0.3 m and 0.05 rad
    When: The mean and standard deviation of the two lane-pose entries are measured
    Then: They match the state's own pose and the two configured widths, each to its own

    Test type: unit
    """
    torch.manual_seed(29)
    case = _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        lane_lateral_std_m=0.3,
        lane_heading_std_rad=0.05,
    )
    state = np.zeros((1, case.env.state_width))
    state[0, EGO_LAT], state[0, EGO_ANG] = 0.7, 0.2
    batch = _tensor(np.tile(state, (4000, 1)))

    drawn = case.model.sample_observations(batch, _zeros(4000))
    lateral = drawn[:, POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_LAT].numpy()
    angle = drawn[:, POMDP_OBS_LANE_POSE_INDEX + LANE_POSE_ANG].numpy()

    assert abs(float(lateral.mean()) - 0.7) < 0.03
    assert abs(float(lateral.std()) - 0.3) < 0.03
    assert abs(float(angle.mean()) - 0.2) < 0.01
    assert abs(float(angle.std()) - 0.05) < 0.01


def test_the_curvature_channel_is_scored_against_the_map() -> None:
    """The torch model scores the camera's curvature against its own map lookup.

    Purpose: This is the only term in the whole reading that says where along the lap a
        particle is. A torch model that held the curvature under the ego across the channel
        -- the mapless default -- would score the same number for every particle and quietly
        lose the arclength from the filter

    Given: 24 random states spanning several laps, and observations whose curvature entries
        are exactly what the map predicts for each state
    When: Both models score them, and the whole channel is then shifted by 0.01 1/m
    Then: The scores match the scalar model's, the shift costs exactly the Gaussian's own
        penalty over the L samples, and the predictions differ between particles

    Test type: unit
    """
    torch.manual_seed(33)
    case = _build_case(discount_factor=0.95, process_noise_std=0.0, curvature_std_1pm=0.002)
    rng = np.random.default_rng(33)
    states = _random_states(rng, case.env, 24)
    lookahead = len(case.env.curvature_lookahead_m)
    channel = slice(POMDP_OBS_CURVATURE_INDEX, POMDP_OBS_CURVATURE_INDEX + lookahead)

    predicted = case.env.curvature_ahead(states[:, :EGO_STATE_WIDTH])
    flat = case.model.sample_observations(_tensor(states), _zeros(24)).numpy().copy()
    flat[:, channel] = predicted

    torch_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(flat)
    ).numpy()
    np.testing.assert_allclose(
        torch_scores, _scalar_log_probs(case, states, _tensor(flat)), atol=_TOLERANCE
    )

    shifted = flat.copy()
    shifted[:, channel] += 0.01
    shifted_scores = case.model.observation_log_probs(
        _tensor(states), _zeros(24), _tensor(shifted)
    ).numpy()
    penalty = 0.5 * lookahead * (0.01 / 0.002) ** 2
    assert np.all(torch_scores - shifted_scores == pytest.approx(penalty))
    # A held-constant channel would predict one number per particle, not a lap-dependent row.
    assert len(np.unique(predicted, axis=0)) > 1


def _mdp_case(**kwargs: Any) -> _Case:
    return _build_case(
        discount_factor=0.95,
        process_noise_std=0.0,
        observation_mode=ObservationMode.MDP,
        **kwargs,
    )


def test_mdp_log_probs_match_native_exactly() -> None:
    """MDP log-densities equal the scalar diagonal-Gaussian density row by row.

    Purpose: Validates the MDP arm's ego and present-slot Gaussian likelihood, which this
        redesign deliberately left alone -- it is the control the POMDP arm is measured
        against, so a drift here would move the yardstick

    Given: Random next states and random observations, so present and absent slots and
        both matching and mismatching readings all occur
    When: observation_log_probs is compared to env.observation_log_probability
    Then: The maximum absolute difference is below 1e-9 and every value is finite

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
        _tensor(next_states), _zeros(128), _tensor(observations)
    ).numpy()
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected - actual)) < _TOLERANCE


def test_mdp_observation_dimension_and_sampler_moments() -> None:
    """MDP observations have the documented width and the scalar sampler's noise scale.

    Purpose: Validates the MDP observation layout and its per-channel noise standard
        deviations against the parameters read off the scalar model

    Given: One state with a single present agent and the detection model switched off,
        observed 20000 times
    When: The per-column standard deviation of the observation is measured
    Then: The width is 4 + 5K, and the ego, pose and velocity channels each match their
        configured standard deviation to within 3%

    Test type: unit
    """
    torch.manual_seed(9)
    # Detection off: a dropped slot or a Cauchy phantom would swamp the moments this pins,
    # and the moments in question are the measurement noise, not the detection rates.
    case = _mdp_case(presence_miss_prob=0.0, presence_false_alarm_prob=0.0)
    assert case.model.observation_dim == 4 + case.env.max_tracked_agents * AGENT_SLOT_WIDTH
    state = np.zeros((1, case.env.state_width))
    state[0, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [1.0, 8.0, 1.0, 2.0, -1.0]
    drawn = case.model.sample_observations(
        _tensor(np.tile(state, (20000, 1))), _zeros(20000)
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
        _tensor(np.tile(state, (20000, 1))), _zeros(20000)
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
    indices = _zeros(16)
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


def test_observation_keys_separate_small_integer_readings(pomdp_case: _Case) -> None:
    """Two thousand distinct small-integer readings hash to two thousand distinct keys.

    Purpose: Guards the hash weights against the collisions a small-prime weighting causes.
        Much of a POMDP reading is binary detection flags and quantized single-digit metres,
        so a prime-weighted sum lands in a range of a few thousand and merges genuinely
        different observations onto one belief-tree node

    Given: 2000 readings whose every entry is an independent integer in [0, 8)
    When: They are hashed into belief-tree keys
    Then: The number of distinct keys equals the number of distinct readings

    Test type: unit
    """
    rng = np.random.default_rng(12)
    readings = rng.integers(0, 8, size=(2000, pomdp_case.model.observation_dim)).astype(float)
    unique_readings = np.unique(readings, axis=0).shape[0]
    keys = pomdp_case.model.observation_keys(_tensor(readings)).numpy()
    assert len(np.unique(keys)) == unique_readings


def test_observation_keys_read_every_entry_of_the_widened_reading(case: _Case) -> None:
    """One weight per entry, across the whole observation width, none of them shared.

    Purpose: The hash weights are sized from observation_dim, and observation_dim moved when
        the ego pose was added and the detection rows widened. A weight vector left at the
        old width would either raise on the shape or, worse, silently stop separating the
        entries past it -- and a tail of entries that never reach the key merges genuinely
        different readings onto one belief-tree node

    Given: An all-zero reading and, for every entry index of every supported configuration,
        a reading with that one entry raised into the next quantization bucket
    When: All of them are hashed
    Then: Every key is distinct, so no entry is unweighted and no two entries share a weight

    Test type: unit
    """
    width = case.model.observation_dim
    readings = np.concatenate([np.zeros((1, width)), np.eye(width)], axis=0)
    keys = case.model.observation_keys(_tensor(readings)).numpy()
    assert len(np.unique(keys)) == width + 1


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
