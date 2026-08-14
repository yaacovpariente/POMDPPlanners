# SPDX-License-Identifier: MIT
# pylint: disable=too-many-lines  # Mirrors the module under test, which carries the same waiver.

"""Tests for the abstract planner-side racetrack generative model.

The class under test cannot be instantiated — it is abstract precisely because it does not
know where the road bends — so most of what follows is exercised through the thinnest
concrete subclass available, a
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
over a one-segment map. What is being checked is the *base* behaviour: the bicycle
integration, the Frenet coupling, the arclength slot, and the densities the belief weights
particles with. Behaviour specific to either subclass lives in its own file.

The observation half of this file is about the five-channel sensor reading — the ego's own
pose, a speedometer, a lane camera, the road's curvature at fixed distances ahead, and
unlabeled radar detections — and the closed-form likelihood that scores it. There is no
occupancy grid and no rasterisation any more, so the tests that pinned cell edges and soft
mass are gone with it. What replaces them is a ranking argument: the true state must beat
every single-coordinate perturbation of itself, and nothing may ever score ``-inf``.

The arm's partial observability now sits in **one** place: vehicles the sensor cannot see.
``max_detection_range_m`` is the dial, and it acts through the likelihood as much as through
the world — a particle holding a car beyond it pays nothing for the reading not showing one,
and a particle holding a car inside it pays a finite miss penalty. Both halves are pinned
below, because the second is the inference mechanism and the first is what stops it firing on
cars nobody could have seen.

Every test but the two live-simulator ones is pure NumPy and never touches highway-env: the
model exists so a planner can run without the simulator, and a test suite that needed the
simulator to check the model would not be checking that. Those two are the exception on
purpose — they are the only things that can show the reproduced bicycle really tracks the one
the world integrates.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import (
    PROBABILITY_EPS,
    pack_detections,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model import (
    AGENTS_KEY,
    CURVATURE_AHEAD_KEY,
    DETECTIONS_KEY,
    EGO_KEY,
    EGO_POSE_KEY,
    EGO_SPEED_KEY,
    LANE_POSE_KEY,
    KinematicsObservationModel,
    SensorObservationModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_VY,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_VY,
    DETECTION_REL_X,
    DETECTION_REL_Y,
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
    MAX_ACCELERATION_MPS2,
    MAX_STEERING_RAD,
    OBSERVED_EGO_POSE_WIDTH,
    ObservationMode,
    RacetrackObservation,
    racetrack_reward,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

# Indices into DEFAULT_ACTION_PRESETS, named so the tests read as driving commands. The
# table is the 3 accelerations crossed with the 9 steering angles, acceleration major.
COAST_STRAIGHT = 13
ACCELERATE_STRAIGHT = 4
COAST_FULL_LEFT = 17

DT = 0.2

# What one visibility disagreement costs at the shipped detection rates of zero. The world's
# detector drops nothing it can see and invents nothing, so a particle whose visibility
# prediction contradicts the reading has probability zero and is *excluded*; the probability
# floor is what turns that into 27.63 nats instead of -inf, and this is that floor written the
# way `bernoulli_log_prob` computes it — a slot the reading is silent about, scored as free
# against scored as certain.
_CONTRADICTION_NATS = float(np.log1p(-PROBABILITY_EPS) - np.log1p(-(1.0 - PROBABILITY_EPS)))

# A map with one segment and no bends, so the base class's dynamics are exercised without
# a subclass's curvature source doing anything interesting.
_STRAIGHT_LAP_M = 1000.0

# The mapped-curvature tests need a lap that actually bends somewhere: 100 m of straight, then
# a constant left-hand arc back to the seam. At the default 10/20/30 m lookaheads a particle
# at 80 m reads [0, 0.05, 0.05] and one at 0 m reads [0, 0, 0], which is what makes the
# curvature channel say where along the lap a particle thinks it is.
_BEND_START_M = 100.0
_BEND_CURVATURE_1PM = 0.05
_BEND_LAP_M = 200.0


def _flat_geometry(curvature: float = 0.0) -> TrackGeometry:
    """A single-segment lap of constant curvature."""
    return TrackGeometry(
        segment_starts=np.array([0.0]),
        segment_curvatures=np.array([curvature]),
        total_length_m=_STRAIGHT_LAP_M,
    )


def _bent_geometry() -> TrackGeometry:
    """A lap of one straight followed by one constant-radius left-hand arc."""
    return TrackGeometry(
        segment_starts=np.array([0.0, _BEND_START_M]),
        segment_curvatures=np.array([0.0, _BEND_CURVATURE_1PM]),
        total_length_m=_BEND_LAP_M,
    )


def _model(**overrides: Any) -> KnownTrackModel:
    """Build a model with deterministic dynamics on a straight map, unless asked otherwise."""
    settings: Dict[str, Any] = {
        "discount_factor": 0.95,
        "process_noise_std": 0.0,
        "track_geometry": _flat_geometry(),
    }
    settings.update(overrides)
    return KnownTrackModel(**settings)


class _FixedCurvatureModel(RacetrackModelPOMDP):
    """The thinnest possible concrete model: one curvature, wherever the particle is.

    Exists so the *base class's* own :meth:`RacetrackModelPOMDP.curvature_ahead` can be
    reached. Both shipped subclasses override it — one from a map, one from the reading — so
    nothing else in the suite exercises the default that holds a single value across the
    whole channel.
    """

    def __init__(self, discount_factor: float, curvature: float = 0.0, **model_kwargs: Any) -> None:
        """Build a model whose road bends at ``curvature`` everywhere."""
        self._curvature = float(curvature)
        super().__init__(discount_factor=discount_factor, **model_kwargs)

    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        return np.full(len(ego), self._curvature, dtype=float)


def _ego_state(
    model: RacetrackModelPOMDP,
    *,
    speed: float = 10.0,
    heading: float = 0.0,
    lateral: float = 0.0,
    angle: float = 0.0,
    arclength: float = 0.0,
) -> np.ndarray:
    """A state with the given ego block and every agent slot empty."""
    state = np.zeros(model.state_width, dtype=float)
    state[EGO_SPEED] = speed
    state[EGO_HEADING] = heading
    state[EGO_LAT] = lateral
    state[EGO_ANG] = angle
    state[EGO_ARCLENGTH_M] = arclength
    return state


def _place_agent(
    state: np.ndarray,
    slot: int,
    rel_x: float,
    rel_y: float,
    rel_vx: float = 0.0,
    rel_vy: float = 0.0,
) -> np.ndarray:
    """Return a copy of ``state`` with one agent slot filled in the ego body frame."""
    filled = state.copy()
    base = EGO_STATE_WIDTH + slot * AGENT_SLOT_WIDTH
    filled[base + AGENT_PRESENT] = 1.0
    filled[base + AGENT_REL_X] = rel_x
    filled[base + AGENT_REL_Y] = rel_y
    filled[base + AGENT_REL_VX] = rel_vx
    filled[base + AGENT_REL_VY] = rel_vy
    return filled


def _sensor_arm(model: RacetrackModelPOMDP) -> SensorObservationModel:
    """The POMDP arm of a model, narrowed from the protocol the model declares."""
    arm = model.observation_model
    assert isinstance(arm, SensorObservationModel)
    return arm


def _kinematics_arm(model: RacetrackModelPOMDP) -> KinematicsObservationModel:
    """The MDP arm of a model, narrowed from the protocol the model declares."""
    arm = model.observation_model
    assert isinstance(arm, KinematicsObservationModel)
    return arm


def _ego_pose_of(state: np.ndarray) -> np.ndarray:
    """The four state slots the ego-pose channel reports, in the channel's own order."""
    array = np.asarray(state, dtype=float)
    pose = np.zeros(OBSERVED_EGO_POSE_WIDTH, dtype=float)
    pose[EGO_POSE_X] = array[EGO_X]
    pose[EGO_POSE_Y] = array[EGO_Y]
    pose[EGO_POSE_HEADING] = array[EGO_HEADING]
    pose[EGO_POSE_ARCLENGTH] = array[EGO_ARCLENGTH_M]
    return pose


def _reading(
    model: RacetrackModelPOMDP,
    state: np.ndarray,
    *,
    ego_pose: Optional[np.ndarray] = None,
    ego_speed: Optional[float] = None,
    lane_pose: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """The noise-free five-channel reading a state would produce, as the encoder hands it on.

    Built term for term rather than drawn, because the sampler applies the miss and
    false-alarm rates and would make every comparison here depend on which draws came up.
    ``ego_pose``, ``ego_speed`` and ``lane_pose`` override their channels so a test can move
    one sensor with the state and every other channel held fixed.
    """
    sensor = _sensor_arm(model)
    array = np.asarray(state, dtype=float)
    speed = float(array[EGO_SPEED]) if ego_speed is None else ego_speed
    pose = (float(array[EGO_LAT]), float(array[EGO_ANG])) if lane_pose is None else lane_pose
    return {
        EGO_POSE_KEY: np.array(
            _ego_pose_of(array) if ego_pose is None else ego_pose, dtype=np.float32
        ),
        EGO_SPEED_KEY: np.array([speed], dtype=np.float32),
        LANE_POSE_KEY: np.array(pose, dtype=np.float32),
        CURVATURE_AHEAD_KEY: sensor.curvature_ahead(array).astype(np.float32),
        DETECTIONS_KEY: pack_detections(
            sensor.predicted_detections(array), model.max_tracked_agents
        ).astype(np.float32),
    }


def _kinematics(model: RacetrackModelPOMDP, state: np.ndarray) -> Dict[str, np.ndarray]:
    """The noise-free MDP reading of a state, as the encoder hands it to the density."""
    return _kinematics_arm(model).clean(state)


def _score(model: RacetrackModelPOMDP, state: np.ndarray, observation: Any) -> float:
    """One state scored against one observation, as a plain float."""
    return float(model.observation_log_probability(state, COAST_STRAIGHT, observation)[0])


def test_the_base_model_cannot_be_instantiated_without_a_curvature_source():
    """Test that the base class is abstract on the one method that matters.

    Purpose: The base class deliberately does not know the track. Leaving it concrete with
        a zero-curvature default would let a caller build a model that silently drives
        straight through every corner, which is the exact bug the split exists to prevent

    Given: The abstract RacetrackModelPOMDP class
    When: It is constructed directly with otherwise valid arguments
    Then: TypeError names the unimplemented curvature hook

    Test type: unit
    """
    # pylint: disable=abstract-class-instantiated
    with pytest.raises(TypeError, match="_curvature_for"):
        RacetrackModelPOMDP(discount_factor=0.95)  # type: ignore[abstract]


def test_actions_index_the_presets_and_hash_action_round_trips():
    """Test that the discrete action set enumerates the shared control presets.

    Purpose: Validates that the planner's action vocabulary is exactly the indices into
        the preset table the world also indexes, and that hashing an action is reversible

    Given: A model built with the default 27-entry preset grid
    When: get_actions() is enumerated and each action is hashed
    Then: The actions are 0..26 in order and each hash equals the action itself

    Test type: unit
    """
    model = _model()

    actions = model.get_actions()

    assert actions == list(range(len(DEFAULT_ACTION_PRESETS)))
    assert [model.hash_action(action) for action in actions] == actions


def test_coasting_straight_advances_x_by_speed_times_dt():
    """Test that a coasting step moves the ego exactly one dt of travel down the track.

    Purpose: Validates the closed form of the bicycle integration in its simplest case,
        where the substeps must sum to exactly speed * dt with no drift in speed or offset

    Given: A model at 10 m/s, heading 0, on the centreline of a straight lane
    When: The coast-straight preset is applied for one 0.2 s decision
    Then: x advances by exactly 2.0 m and speed, y, and lateral offset are unchanged

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    next_state = model.sample_next_state(state, COAST_STRAIGHT)

    assert next_state[EGO_X] == pytest.approx(2.0, abs=1e-12)
    assert next_state[EGO_Y] == pytest.approx(0.0, abs=1e-12)
    assert next_state[EGO_SPEED] == pytest.approx(10.0, abs=1e-12)
    assert next_state[EGO_LAT] == pytest.approx(0.0, abs=1e-12)


def test_full_lock_steering_turns_heading_by_the_analytic_bicycle_rate():
    """Test that a full-lock steering step turns the heading by the bicycle model's rate.

    Purpose: Validates that the slip angle and the LENGTH/2 wheelbase term are the ones
        highway-env uses, since a wrong wheelbase is invisible until the model is compared
        against the simulator

    Given: A model at 10 m/s with the steering command saturated at +1
    When: One 0.2 s decision is propagated
    Then: The heading equals speed * sin(arctan(tan(pi/4)/2)) / 2.5 * dt, computed here
        from the same constants rather than hardcoded

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)
    slip = np.arctan(0.5 * np.tan(MAX_STEERING_RAD))
    expected = 10.0 * np.sin(slip) / (model.vehicle_length / 2.0) * DT

    next_state = model.sample_next_state(state, COAST_FULL_LEFT)

    assert next_state[EGO_HEADING] == pytest.approx(float(expected), abs=1e-12)


def test_acceleration_preset_changes_speed_by_accel_times_dt():
    """Test that the throttle preset changes speed by the full commanded acceleration.

    Purpose: Validates that the normalised acceleration is scaled by the shared
        MAX_ACCELERATION_MPS2 and integrated over the whole decision, not one substep

    Given: A model at 10 m/s and the accelerate-straight preset (normalised +1)
    When: One 0.2 s decision is propagated
    Then: Speed rises by MAX_ACCELERATION_MPS2 * dt

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    next_state = model.sample_next_state(state, ACCELERATE_STRAIGHT)

    assert next_state[EGO_SPEED] == pytest.approx(10.0 + MAX_ACCELERATION_MPS2 * DT, abs=1e-12)


def test_non_zero_curvature_bends_the_lane_relative_angle():
    """Test that lane curvature rotates the lane frame under a straight-driving ego.

    Purpose: Validates the Frenet coupling: on an arc the lane direction turns even when
        the ego does not, so the heading-versus-lane angle must drift negative on a
        left-curving lane

    Given: Two identical coasting states, propagated by two models whose maps differ only
        in curvature, 0.0 and 0.05 1/m
    When: Each is propagated one decision with no steering
    Then: The straight lane leaves the angle at zero while the curved lane makes it
        negative, at roughly -curvature * speed * dt

    Test type: unit
    """
    curvature = 0.05
    flat = _model()
    curved_model = _model(track_geometry=_flat_geometry(curvature))

    straight = flat.sample_next_state(_ego_state(flat), COAST_STRAIGHT)
    curved = curved_model.sample_next_state(_ego_state(curved_model), COAST_STRAIGHT)

    assert straight[EGO_ANG] == pytest.approx(0.0, abs=1e-12)
    assert curved[EGO_ANG] < 0.0
    assert curved[EGO_ANG] == pytest.approx(-curvature * 10.0 * DT, rel=0.05)


def test_arclength_advances_monotonically_and_its_lookup_wraps_at_the_lap_seam():
    """Test the arclength slot the curvature hook is indexed by.

    Purpose: The slot replaced a frozen curvature value, and the whole point is that it
        moves: a rollout has to walk along the track for a lookup by arclength to ever see
        a different segment. It also has to survive crossing the finish line, which a
        planning horizon on the last segment routinely does

    Given: A one-segment 1000 m map, and a state seeded 4 m short of the lap seam
    When: Ten coasting decisions are propagated from each of arclength 0 and 996
    Then: The slot strictly increases on every step, advances by speed * dt per step on a
        straight, and the map's own lookup wraps a past-the-seam arclength back onto the
        first segment rather than clamping to the last

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)
    history = [float(state[EGO_ARCLENGTH_M])]
    for _ in range(10):
        state = model.sample_next_state(state, COAST_STRAIGHT)
        history.append(float(state[EGO_ARCLENGTH_M]))

    assert all(later > earlier for earlier, later in zip(history, history[1:]))
    assert history[-1] == pytest.approx(10 * 10.0 * DT, abs=1e-9)

    past_seam = _ego_state(model, arclength=_STRAIGHT_LAP_M - 4.0)
    for _ in range(10):
        past_seam = model.sample_next_state(past_seam, COAST_STRAIGHT)
    assert past_seam[EGO_ARCLENGTH_M] > _STRAIGHT_LAP_M
    assert float(model.track_geometry.curvature_at(past_seam[EGO_ARCLENGTH_M])) == 0.0


def test_sample_next_state_batch_matches_a_seeded_loop_over_sample_next_state():
    """Test that the vectorised particle path draws the same states as the scalar path.

    Purpose: Validates that the loop-free batch propagation used by the belief is the same
        kernel as the single-particle one, noise included, so the filter and the planner
        cannot silently disagree

    Given: A noisy model and five distinct particles
    When: The batch is propagated once and the same particles are propagated one at a time
        from the same random seed
    Then: Every entry of the two results matches exactly

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    particles = np.stack(
        [_ego_state(model, speed=8.0 + index, lateral=0.1 * index) for index in range(5)]
    )

    np.random.seed(7)
    batched = model.sample_next_state_batch(particles, COAST_FULL_LEFT)
    np.random.seed(7)
    looped = np.stack(
        [model.sample_next_state(particle, COAST_FULL_LEFT) for particle in particles]
    )

    assert batched.shape == (5, model.state_width)
    np.testing.assert_array_equal(batched, looped)


def test_process_noise_leaves_the_arclength_slot_alone():
    """Test that the transition's noise stops short of the arclength slot.

    Purpose: Arclength is not an independent coordinate but the integral of the ego's own
        motion, and a known-track model indexes its curvature profile with it. Jittering it
        would scatter particles onto different parts of the circuit for no modelling reason

    Given: A noisy model and one particle propagated many times from the same state
    When: The successors' arclength slots are compared
    Then: Every draw carries exactly the same arclength, while the ego position does vary

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    state = _ego_state(model, arclength=25.0)

    draws = model.sample_next_state(state, COAST_STRAIGHT, n_samples=32)

    assert len(np.unique(draws[:, EGO_ARCLENGTH_M])) == 1
    assert draws[0, EGO_ARCLENGTH_M] == pytest.approx(25.0 + 10.0 * DT, abs=1e-9)
    assert len(np.unique(draws[:, EGO_X])) > 1


def test_transition_log_probability_is_finite_and_peaks_at_the_deterministic_successor():
    """Test that the transition density is a proper Gaussian centred on the propagation.

    Purpose: Validates the shape contract the belief relies on and that the mode of the
        density is the noise-free successor, which is what makes the weights meaningful

    Given: A noisy model, a coasting state, and three candidate successors — the exact
        deterministic propagation and two increasingly displaced copies of it
    When: transition_log_probability scores all three
    Then: The result has shape (3,), is finite, and decreases with displacement

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    state = _ego_state(model)
    exact = _model().sample_next_state(state, COAST_STRAIGHT)
    near, far = exact.copy(), exact.copy()
    near[EGO_X] += 0.05
    far[EGO_X] += 0.5

    log_probs = model.transition_log_probability(state, COAST_STRAIGHT, [exact, near, far])

    assert log_probs.shape == (3,)
    assert np.all(np.isfinite(log_probs))
    assert log_probs[0] > log_probs[1] > log_probs[2]


def test_transition_log_probability_rejects_a_noise_free_model():
    """Test that scoring a point-mass transition is refused rather than returning NaN.

    Purpose: A model built without process noise has a degenerate transition; dividing by
        its zero variance would hand the belief a silent NaN weight, which is far harder to
        diagnose than an exception at the call site

    Given: A model with process_noise_std left at zero
    When: transition_log_probability is called
    Then: A ValueError naming the point mass is raised

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    with pytest.raises(ValueError, match="point mass"):
        model.transition_log_probability(state, COAST_STRAIGHT, [state])


def test_reward_matches_the_shared_racetrack_reward_on_the_resulting_state():
    """Test that the model scores a transition with the world's own reward function.

    Purpose: The world and the planner call one reward function so the planner cannot be
        optimising a different objective; that only holds if the model feeds it the same
        arguments the world does — the *resulting* lateral offset and the applied command

    Given: A coasting state and its deterministic successor
    When: reward is called with the successor and, separately, with the successor left out
    Then: Both equal racetrack_reward evaluated on the successor's lateral offset with this
        model's own weights, and the off-road case scores exactly zero

    Test type: unit
    """
    model = _model()
    state = _ego_state(model, lateral=0.4)
    successor = model.sample_next_state(state, COAST_FULL_LEFT)
    expected = racetrack_reward(
        float(successor[EGO_LAT]),
        model.action_presets[COAST_FULL_LEFT],
        False,
        True,
        collision_reward=model.collision_reward,
        lane_centering_cost=model.lane_centering_cost,
        lane_centering_reward=model.lane_centering_reward,
        action_reward=model.action_reward,
    )

    assert model.reward(state, COAST_FULL_LEFT, successor) == pytest.approx(expected, abs=1e-12)
    assert model.reward(state, COAST_FULL_LEFT) == pytest.approx(expected, abs=1e-12)

    off_road = _ego_state(model, lateral=model.lane_half_width + 1.0)
    assert model.reward(off_road, COAST_STRAIGHT, off_road) == pytest.approx(0.0, abs=1e-12)


def test_is_terminal_fires_off_road_and_on_a_near_agent_but_not_in_the_clear():
    """Test the two ways the model predicts an episode ending.

    Purpose: Validates that the model can foresee both failure modes the world reports —
        leaving the lane and hitting a vehicle — since a planner blind to either has no
        reason to avoid it

    Given: Three states: on the centreline with no agents, past the lane half-width, and
        centred with an agent 2 m ahead
    When: is_terminal is queried on each
    Then: Only the clear state is non-terminal

    Test type: unit
    """
    model = _model()
    clear = _ego_state(model)
    off_road = _ego_state(model, lateral=model.lane_half_width + 0.1)
    collision = _place_agent(clear, slot=0, rel_x=2.0, rel_y=0.0)

    assert model.is_terminal(clear) is False
    assert model.is_terminal(off_road) is True
    assert model.is_terminal(collision) is True


# ── Encoding the world's reading ────────────────────────────────────────


def test_encoding_splits_the_worlds_reading_into_its_five_sensor_keys():
    """An ego pose, a speedometer, a lane camera, a curvature channel and a radar.

    Purpose: encode_observation is the single raw-observation seam, so if it dropped or
        reshaped an entry every consumer downstream would be wrong at once. The detections in
        particular arrive ragged-looking and must land as a fixed (K, 5) block, and the ego
        pose is the channel the redesign added — if the seam dropped it the filter would go
        back to inferring where round the lap it is from curvature alone

    Given: A POMDP model and the world's own RacetrackObservation: an ego at (12, -4) heading
        0.2 rad, 61 m round the lap, at 7.5 m/s, 0.3 m off centre and 0.05 rad across the
        lane, reading curvature at three distances and one detection
    When: It is encoded
    Then: Each channel lands under its own key at its own shape, ego pose first, and the
        detection row survives as [detected, rel_x, rel_y, rel_vx, rel_vy]

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    detections = np.zeros((model.max_tracked_agents, DETECTION_SLOT_WIDTH), dtype=np.float32)
    detections[0] = [1.0, 14.0, -2.0, -3.5, 0.75]

    encoded = model.encode_observation(
        RacetrackObservation(
            ego_pose=np.array([12.0, -4.0, 0.2, 61.0], dtype=np.float32),
            ego_speed=np.array([7.5], dtype=np.float32),
            lane_pose=np.array([0.3, 0.05], dtype=np.float32),
            curvature_ahead=np.array([0.0, 0.01, 0.02], dtype=np.float32),
            detections=detections,
        )
    )

    assert set(encoded) == {
        EGO_POSE_KEY,
        EGO_SPEED_KEY,
        LANE_POSE_KEY,
        CURVATURE_AHEAD_KEY,
        DETECTIONS_KEY,
    }
    assert encoded[EGO_POSE_KEY].shape == (OBSERVED_EGO_POSE_WIDTH,)
    np.testing.assert_allclose(encoded[EGO_POSE_KEY], [12.0, -4.0, 0.2, 61.0], atol=1e-6)
    assert encoded[EGO_SPEED_KEY].shape == (1,)
    assert float(encoded[EGO_SPEED_KEY][0]) == pytest.approx(7.5)
    assert encoded[LANE_POSE_KEY].shape == (2,)
    assert float(encoded[LANE_POSE_KEY][LANE_POSE_LAT]) == pytest.approx(0.3)
    assert float(encoded[LANE_POSE_KEY][LANE_POSE_ANG]) == pytest.approx(0.05)
    assert encoded[CURVATURE_AHEAD_KEY].shape == (len(DEFAULT_CURVATURE_LOOKAHEAD_M),)
    np.testing.assert_allclose(encoded[CURVATURE_AHEAD_KEY], [0.0, 0.01, 0.02], atol=1e-7)
    assert encoded[DETECTIONS_KEY].shape == (model.max_tracked_agents, DETECTION_SLOT_WIDTH)
    np.testing.assert_allclose(encoded[DETECTIONS_KEY][0], [1.0, 14.0, -2.0, -3.5, 0.75], atol=1e-6)


def test_encoding_an_already_encoded_dictionary_round_trips_it():
    """A reading that has been through the seam once survives going through it again.

    Purpose: The episode loop encodes once, but a planner scoring its own drawn observation
        re-encodes, and the belief pickles and restores encoded readings. Any of those paths
        hitting the tuple branch would raise; silently returning a different dictionary would
        be worse

    Given: A POMDP model and the noise-free reading of a state carrying one opponent
    When: That dictionary is passed back through encode_observation
    Then: All five keys come back with the same values and the same float32 dtype

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model, lateral=0.2), slot=0, rel_x=11.0, rel_y=1.0, rel_vx=-2.0)
    reading = _reading(model, state)

    encoded = model.encode_observation(reading)

    assert set(encoded) == set(reading)
    for key, value in reading.items():
        assert encoded[key].dtype == np.float32
        np.testing.assert_array_equal(encoded[key], value)


def test_encoding_refuses_a_four_part_reading_that_has_lost_a_sensor():
    """A reading missing one of its five channels is rejected at the seam.

    Purpose: Four parts is the *exact* shape this arm emitted before the ego-pose channel
        existed, so it is what a stale caller or a pickled episode hands over. Unpacking it
        into five names raises somewhere deep instead of here, and defaulting the missing
        channel would produce a planner reading a sensor that never reported — a silently
        dropped sensor looks like success, which is the whole reason this guard exists

    Given: A POMDP model and a (speed, lane_pose, curvature, detections) four-tuple with the
        ego-pose channel missing
    When: encode_observation is called with it
    Then: ValueError names the five-part reading and says a whole sensor would otherwise go
        silently unobserved

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    stale = (
        np.array([7.5], dtype=np.float32),
        np.array([0.3, 0.05], dtype=np.float32),
        np.array([0.0, 0.01, 0.02], dtype=np.float32),
        np.zeros((model.max_tracked_agents, DETECTION_SLOT_WIDTH), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="five-part") as raised:
        model.encode_observation(stale)

    assert "silently unobserved" in str(raised.value)


def test_mdp_encode_observation_rotates_absolute_rows_into_the_ego_body_frame():
    """Test that the MDP encoder converts absolute vehicle rows to ego-relative ones.

    Purpose: Validates the single raw-observation seam for the MDP arm — the state's agent
        slots are body-frame, so an encoder that forgot the rotation would compare
        world-frame observations against body-frame means and score nonsense

    Given: A raw table whose ego drives due north at (10, 5) and a second vehicle 10 m due
        north of it at the same velocity, plus three empty rows
    When: encode_observation is applied
    Then: The ego row is passed through as [x, y, vx, vy], the other vehicle lands 10 m
        straight ahead in the body frame with zero relative velocity, and the empty rows
        stay zero

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.MDP)
    raw = np.zeros((model.max_tracked_agents + 1, 5), dtype=np.float32)
    raw[0] = [1.0, 10.0, 5.0, 0.0, 4.0]
    raw[1] = [1.0, 10.0, 15.0, 0.0, 4.0]

    encoded = model.encode_observation(raw)

    np.testing.assert_allclose(encoded[EGO_KEY], [10.0, 5.0, 0.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(encoded[AGENTS_KEY][0], [1.0, 10.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(encoded[AGENTS_KEY][1:], 0.0, atol=1e-12)


# ── Sampling and scoring the sensor reading ─────────────────────────────


def test_sampled_readings_and_their_scores_agree_in_shape_and_stay_finite():
    """Every drawn reading has the five channels the density expects, and scores finitely.

    Purpose: The sampler and the scorer are one model, so a draw the density cannot score is
        a contradiction — and the density must never return -inf, because an opponent leaves
        sensor range or slips behind a closer car every few steps and the filter would lose
        the particle each time

    Given: A POMDP model, a state carrying two opponents, and 20 sampled readings of it
    When: The draws are inspected and then scored back against the state
    Then: Each draw carries (4,), (1,), (2,), (3,) and (4, 5) channels, the scores come back
        as a (20,) array, and every one of them is finite

    Test type: unit
    """
    np.random.seed(3)
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(
        _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=1.5, rel_vx=-2.0),
        slot=1,
        rel_x=25.0,
        rel_y=-6.0,
        rel_vx=1.0,
    )

    draws = model.sample_observation(state, COAST_STRAIGHT, n_samples=20)

    assert len(draws) == 20
    for draw in draws:
        assert draw[EGO_POSE_KEY].shape == (OBSERVED_EGO_POSE_WIDTH,)
        assert draw[EGO_SPEED_KEY].shape == (1,)
        assert draw[LANE_POSE_KEY].shape == (2,)
        assert draw[CURVATURE_AHEAD_KEY].shape == (len(DEFAULT_CURVATURE_LOOKAHEAD_M),)
        assert draw[DETECTIONS_KEY].shape == (model.max_tracked_agents, DETECTION_SLOT_WIDTH)

    scores = model.observation_log_probability(state, COAST_STRAIGHT, draws)

    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))


def test_a_single_drawn_reading_scores_as_a_one_entry_array():
    """One observation in, one score out — not a bare float and not a ragged list.

    Purpose: The belief indexes the returned array, so an arm that special-cased the single
        observation into a scalar would break the filter's weight update while every
        multi-observation test still passed

    Given: A POMDP model, a state with one opponent, and a single drawn reading of it
    When: It is scored without being wrapped in a list
    Then: The result is a (1,) array holding one finite score

    Test type: unit
    """
    np.random.seed(5)
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=1.5)

    scores = model.observation_log_probability(
        state, COAST_STRAIGHT, model.sample_observation(state, COAST_STRAIGHT)
    )

    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_the_likelihood_ranks_the_true_state_above_every_single_perturbation():
    """Every channel of the reading pulls the weight towards the state that produced it.

    Purpose: This is the property the whole redesign exists to deliver, and the one a
        product of closed-form terms can lose quietly: a channel written into the observation
        but never read when scoring is dead weight, and the belief would carry it around
        while weighting every particle as though it had never arrived

    Given: A POMDP model, a true state 0.3 m off centre at 10 m/s carrying two opponents, its
        own noise-free reading, and eight copies of it each perturbed in one coordinate — the
        ego's x and y by +0.2 m, its heading by +0.02 rad, its arclength by +0.3 m, the
        lateral offset by +0.20 m, the ego speed by +0.5 m/s, the near slot's position by
        +1.0 m along-track, and each component of that slot's relative velocity by +1.0 m/s
    When: All nine are scored against the one reading
    Then: Every score is finite and the true state wins each pairing, by exactly the Gaussian
        each channel's own width implies: 2.00 nats on the ego x, y and heading, 4.50 on the
        arclength, 8.00 on the lateral offset, 12.50 on the speed, 2.00 on the slot position
        and 5.56 on *each* component of the slot's relative velocity — the crossing component
        included, which a radial-only reading charged 0.09 nats for

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    truth = _place_agent(
        _place_agent(
            _ego_state(model, lateral=0.3, angle=0.02),
            slot=0,
            rel_x=12.0,
            rel_y=1.5,
            rel_vx=-2.0,
            rel_vy=0.5,
        ),
        slot=1,
        rel_x=25.0,
        rel_y=-6.0,
        rel_vx=1.0,
    )
    reading = _reading(model, truth)
    slot_zero = EGO_STATE_WIDTH

    perturbed: Dict[str, np.ndarray] = {}
    for label, index, delta in (
        ("ego x", EGO_X, 0.2),
        ("ego y", EGO_Y, 0.2),
        ("ego heading", EGO_HEADING, 0.02),
        ("arclength", EGO_ARCLENGTH_M, 0.3),
        ("lateral", EGO_LAT, 0.2),
        ("speed", EGO_SPEED, 0.5),
        ("slot position", slot_zero + AGENT_REL_X, 1.0),
        ("slot along-track velocity", slot_zero + AGENT_REL_VX, 1.0),
        ("slot crossing velocity", slot_zero + AGENT_REL_VY, 1.0),
    ):
        moved = truth.copy()
        moved[index] += delta
        perturbed[label] = moved

    best = _score(model, truth, reading)
    gaps = {label: best - _score(model, state, reading) for label, state in perturbed.items()}

    assert np.isfinite(best)
    assert all(np.isfinite(best - gap) for gap in gaps.values())
    assert gaps["ego x"] == pytest.approx(2.0, rel=1e-6)
    assert gaps["ego y"] == pytest.approx(2.0, rel=1e-6)
    assert gaps["ego heading"] == pytest.approx(2.0, rel=1e-6)
    assert gaps["arclength"] == pytest.approx(4.5, rel=1e-6)
    assert gaps["lateral"] == pytest.approx(8.0, rel=1e-6)
    assert gaps["speed"] == pytest.approx(12.5, rel=1e-6)
    assert gaps["slot position"] == pytest.approx(2.0, abs=0.01)
    assert gaps["slot along-track velocity"] == pytest.approx(5.5556, abs=0.01)
    assert gaps["slot crossing velocity"] == pytest.approx(5.5556, abs=0.01)


def test_the_speed_channel_moves_the_score_with_every_other_channel_held_fixed():
    """Two readings differing only in their speed entry score differently.

    Purpose: Ranking particles could in principle come out right while the speed term was a
        constant. Holding the state and every other channel fixed and moving only the
        speedometer is what isolates it, so nothing else can account for the difference

    Given: One state at 10 m/s, and two readings of it identical apart from reading 10 and
        14 m/s
    When: Both are scored against that state
    Then: The scores differ by exactly the Gaussian the residuals imply, 800 nats at the
        default 0.1 m/s width

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _ego_state(model, speed=10.0)

    matching, off_by_four = model.observation_log_probability(
        state,
        COAST_STRAIGHT,
        [_reading(model, state, ego_speed=10.0), _reading(model, state, ego_speed=14.0)],
    )

    # A 4 m/s residual at the default 0.1 m/s width costs 4^2 / (2 * 0.1^2) = 800 nats.
    assert matching - off_by_four == pytest.approx(800.0, rel=1e-9)


def test_the_lane_camera_channel_moves_the_score_with_everything_else_held_fixed():
    """Two readings differing only in lane pose score differently against one particle.

    Purpose: An observation channel the likelihood ignores is a channel that does nothing.
        The racetrack reward is built from lateral offset, so a filter blind to the lane
        reading is blind to the quantity the agent is scored on

    Given: A particle centred in its lane, and two readings of 0.0 m and 0.5 m off centre
    When: Both are scored against it at a lane width of 0.05 m
    Then: The matching reading wins, by exactly the Gaussian's margin at that width

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP, lane_lateral_std_m=0.05)
    state = _ego_state(model, lateral=0.0)

    scores = [
        _score(model, state, _reading(model, state, lane_pose=(offset, 0.0)))
        for offset in (0.0, 0.5)
    ]

    assert scores[0] > scores[1]
    assert scores[0] - scores[1] == pytest.approx(0.5 * (0.5 / 0.05) ** 2, rel=1e-9)


def test_the_lane_heading_residual_wraps_rather_than_counting_a_full_turn():
    """A lane angle read across the +/-pi cut is a small error, not a 2 pi one.

    Purpose: EGO_ANG is wrapped to [-pi, pi), so a particle just below the cut and a reading
        just above it are geometrically 0.02 rad apart. Subtracting them raw makes that 6.26
        rad and annihilates a correct particle

    Given: A particle at +pi - 0.01 rad, a reading at -pi + 0.01 rad across the cut, and a
        second reading 0.02 rad away on the same side of it
    When: Both readings are scored against the particle
    Then: The two scores are equal, so the wrapped residual was 0.02 rad and not 6.26

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP, lane_heading_std_rad=0.1)
    state = _ego_state(model, angle=np.pi - 0.01)
    across = _reading(model, state, lane_pose=(0.0, -np.pi + 0.01))
    near = _reading(model, state, lane_pose=(0.0, np.pi - 0.03))

    assert _score(model, state, across) == pytest.approx(_score(model, state, near), abs=1e-6)


# ── The ego-pose channel ────────────────────────────────────────────────


def test_the_ego_pose_channel_is_drawn_around_the_truth_at_its_configured_widths():
    """The ego reports its own pose, near-exactly, at the widths the density scores with.

    Purpose: This channel is the redesign's answer to "what is actually hidden here?" — the
        ego's own pose is not, so it is reported. A sampler that centred it anywhere but the
        truth, or spread it at some other width, would make the filter confidently wrong
        about the one thing it is not supposed to be guessing at

    Given: A POMDP model at the shipped widths and a state at (3, -2), heading 0.4 rad, 61 m
        round the lap, drawn 4000 times
    When: The ego-pose channel of every draw is collected
    Then: Each draw carries four entries, their means sit on the true [x, y, heading,
        arclength] to within 0.02, and their spreads match ego_position_std_m twice over,
        then ego_heading_std_rad and ego_arclength_std_m

    Test type: unit
    """
    np.random.seed(11)
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _ego_state(model, heading=0.4, arclength=61.0)
    state[EGO_X], state[EGO_Y] = 3.0, -2.0

    poses = np.array(
        [
            np.asarray(draw[EGO_POSE_KEY], dtype=float)
            for draw in model.sample_observation(state, COAST_STRAIGHT, n_samples=4000)
        ]
    )

    assert poses.shape == (4000, OBSERVED_EGO_POSE_WIDTH)
    for channel, truth, width in (
        (EGO_POSE_X, 3.0, model.ego_position_std_m),
        (EGO_POSE_Y, -2.0, model.ego_position_std_m),
        (EGO_POSE_HEADING, 0.4, model.ego_heading_std_rad),
        (EGO_POSE_ARCLENGTH, 61.0, model.ego_arclength_std_m),
    ):
        assert float(poses[:, channel].mean()) == pytest.approx(truth, abs=0.02)
        assert float(poses[:, channel].std()) == pytest.approx(width, rel=0.1)


def test_the_ego_pose_heading_residual_wraps_across_the_branch_cut():
    """A pose read just the other side of +/-pi is a hundredth of a radian out, not 6.28.

    Purpose: This is the term a naive implementation gets wrong, and it fails loudly rather
        than subtly: at the shipped 0.01 rad width an unwrapped 6.26 rad residual is nearly
        two hundred thousand nats, so the first particle to sit near the cut is annihilated
        and the filter loses the car pointing the way it really points

    Given: A particle heading +pi - 0.01 rad, a reading placing it at -pi + 0.01 rad across
        the cut, a second reading 0.02 rad away on the same side, and the exact reading
    When: All three are scored against the particle
    Then: The two 0.02 rad readings score identically, and both fall short of the exact one by
        exactly the Gaussian a 0.02 rad residual implies at the configured width

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP, ego_heading_std_rad=0.1)
    state = _ego_state(model, heading=np.pi - 0.01)
    across, same_side = _ego_pose_of(state), _ego_pose_of(state)
    across[EGO_POSE_HEADING] = -np.pi + 0.01
    same_side[EGO_POSE_HEADING] = np.pi - 0.03

    wrapped = _score(model, state, _reading(model, state, ego_pose=across))

    assert wrapped == pytest.approx(
        _score(model, state, _reading(model, state, ego_pose=same_side)), abs=1e-5
    )
    assert _score(model, state, _reading(model, state)) - wrapped == pytest.approx(
        0.5 * (0.02 / model.ego_heading_std_rad) ** 2, abs=1e-4
    )


def test_the_arclength_term_pins_a_particle_round_a_lap_the_curvature_channel_cannot():
    """Where round the lap a particle sits is now scored even on a road with no bends.

    Purpose: Arclength used to be the curvature channel's job alone, and on a straight it had
        no way to do it — every particle read the same flat channel, the residual cancelled at
        normalisation, and the filter's estimate of lap position was free to drift. The
        ego-pose channel's odometer is what closed that

    Given: A model over a single straight segment, a true particle 80 m round the lap, its own
        noise-free reading, and a second particle half a metre further on
    When: The curvature the two particles predict is compared, and both are scored against
        that one reading
    Then: The two predict the identical curvature channel, so it cannot be doing the work, and
        the true particle still wins by exactly the Gaussian 0.5 m implies at
        ego_arclength_std_m

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    truth = _ego_state(model, arclength=80.0)
    further = _ego_state(model, arclength=80.5)
    reading = _reading(model, truth)

    ahead = model.curvature_ahead(np.stack([truth, further])[:, :EGO_STATE_WIDTH])
    gap = _score(model, truth, reading) - _score(model, further, reading)

    np.testing.assert_allclose(ahead[0], ahead[1], atol=1e-12)
    assert gap == pytest.approx(0.5 * (0.5 / model.ego_arclength_std_m) ** 2, rel=1e-6)


# ── The curvature channel ───────────────────────────────────────────────


def test_the_mapped_curvature_channel_separates_particles_by_arclength():
    """A model holding the track scores where along the lap a particle thinks it is.

    Purpose: This is the only term in the reading that says anything about arclength. Without
        it a particle 80 m down the track and one at the start line score identically, and the
        filter has no way to place the car on the circuit at all

    Given: A lap of 100 m of straight then a 0.05 1/m left-hand arc, a true particle at 80 m
        whose camera therefore reads [0, 0.05, 0.05] at the 10/20/30 m lookaheads, and two
        wrong particles at 0 m (reading all-straight) and 95 m (reading all-bend)
    When: The true particle's reading is scored against all three
    Then: The lookups differ across the three particles, every score is finite, and the true
        arclength wins — by over 600 nats against the start line and over 300 against 95 m,
        at the camera's own 0.002 1/m width

    Test type: unit
    """
    model = KnownTrackModel(
        discount_factor=0.95,
        track_geometry=_bent_geometry(),
        process_noise_std=0.0,
        observation_mode=ObservationMode.POMDP,
    )
    truth = _ego_state(model, arclength=80.0)
    ahead = model.curvature_ahead(np.stack([truth, _ego_state(model)])[:, :EGO_STATE_WIDTH])
    reading = _reading(model, truth)

    scores = {
        arclength: _score(model, _ego_state(model, arclength=arclength), reading)
        for arclength in (0.0, 80.0, 95.0)
    }

    np.testing.assert_allclose(ahead[0], [0.0, _BEND_CURVATURE_1PM, _BEND_CURVATURE_1PM])
    np.testing.assert_allclose(ahead[1], [0.0, 0.0, 0.0])
    assert all(np.isfinite(score) for score in scores.values())
    assert scores[80.0] - scores[0.0] > 600.0
    assert scores[80.0] - scores[95.0] > 300.0


def test_the_base_class_holds_one_curvature_across_the_whole_channel():
    """A model with no source beyond the road under the ego predicts one value everywhere.

    Purpose: That is the honest prediction for a planner whose only curvature source is the
        reading it is scoring — it has nothing to say about 30 m ahead that it did not read
        off that reading. Making it up would separate particles on a difference the model did
        not actually know, which is worse than the term dropping out at normalisation

    Given: A concrete model whose curvature is a fixed 0.03 1/m, and two particles at
        different arclengths
    When: curvature_ahead is asked for both
    Then: Every entry of both rows is 0.03, so the channel is flat and identical across the
        two particles

    Test type: unit
    """
    model = _FixedCurvatureModel(
        discount_factor=0.95,
        curvature=0.03,
        process_noise_std=0.0,
        observation_mode=ObservationMode.POMDP,
    )
    particles = np.stack([_ego_state(model), _ego_state(model, arclength=80.0)])

    ahead = model.curvature_ahead(particles[:, :EGO_STATE_WIDTH])

    assert ahead.shape == (2, len(DEFAULT_CURVATURE_LOOKAHEAD_M))
    np.testing.assert_allclose(ahead, 0.03)


def test_a_curvature_channel_of_the_wrong_length_is_refused():
    """A reading reporting curvature at a different number of distances raises.

    Purpose: The residuals are taken pairwise, so a reading with two samples scored against a
        three-sample model would either broadcast — comparing the curvature at 10 m against
        the curvature at 20 m — or fail deep inside the density. Both are worse than saying so

    Given: A POMDP model at the default three lookaheads, and a reading carrying two
    When: The reading is scored against a state
    Then: ValueError names both counts and says the two must agree

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _ego_state(model)
    reading = _reading(model, state)
    reading[CURVATURE_AHEAD_KEY] = np.zeros(2, dtype=np.float32)

    with pytest.raises(ValueError, match="curvature at 2 distances"):
        model.observation_log_probability(state, COAST_STRAIGHT, reading)


def test_an_empty_curvature_lookahead_is_refused_at_construction():
    """A model asked to read curvature at no distances at all is rejected.

    Purpose: An empty channel scores no residual, so the one reading that says where the road
        bends would be silently dropped and a mapless model would drive straight through every
        corner while every value in the likelihood stayed finite

    Given: A model constructed with curvature_lookahead_m set to an empty tuple
    When: The constructor runs
    Then: ValueError says at least one distance must be named

    Test type: unit
    """
    with pytest.raises(ValueError, match="at least one distance"):
        _model(curvature_lookahead_m=())


# ── Detections: whether, where, and which rank ──────────────────────────


def test_predicted_detections_report_position_and_both_relative_velocity_components():
    """What a particle expects the radar to say is a full kinematic row, nearest first.

    Purpose: This used to be a position and a closing rate, which threw away the component of
        an opponent's motion across the ego's path — the component that decides whether it is
        cutting in front or peeling away. Widening the row to the whole relative velocity is
        what the likelihood's new velocity term has to score against

    Given: A POMDP model and a state carrying two visible opponents, the further one filed in
        the earlier slot: one 23.4 m away moving out at (1.5, -0.5), one 10.4 m away closing
        at (-3.0, 2.0)
    When: predicted_detections is asked what that state should produce
    Then: Two four-wide rows come back, ordered by range rather than by slot, each carrying
        [rel_x, rel_y, rel_vx, rel_vy] verbatim

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(
        _place_agent(_ego_state(model), slot=0, rel_x=22.0, rel_y=-8.0, rel_vx=1.5, rel_vy=-0.5),
        slot=1,
        rel_x=10.0,
        rel_y=3.0,
        rel_vx=-3.0,
        rel_vy=2.0,
    )

    predicted = _sensor_arm(model).predicted_detections(state)

    assert predicted.shape == (2, DETECTION_SLOT_WIDTH - 1)
    np.testing.assert_allclose(predicted[0], [10.0, 3.0, -3.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(predicted[1], [22.0, -8.0, 1.5, -0.5], atol=1e-12)


def test_detections_are_associated_to_slots_by_range_rank():
    """The nearest detection is scored against the nearest slot, and swapping them costs.

    Purpose: Detections carry no identity, so some association rule is needed and this one is
        rank by range. A likelihood that paired them in arrival order instead would take every
        residual against the wrong vehicle while still returning finite, plausible scores

    Given: A state with opponents at 10 m closing at 3 m/s and at 20.6 m opening at 1.9 m/s,
        its own noise-free reading, and that reading with its two detection rows swapped
    When: Both are scored against the state
    Then: Both are finite, the in-order reading wins by over 700 nats, and re-ordering the
        state's own slots changes nothing — only the range rank is used

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    near_first = _place_agent(
        _place_agent(_ego_state(model), slot=0, rel_x=10.0, rel_y=0.0, rel_vx=-3.0),
        slot=1,
        rel_x=20.0,
        rel_y=5.0,
        rel_vx=2.0,
    )
    far_first = _place_agent(
        _place_agent(_ego_state(model), slot=0, rel_x=20.0, rel_y=5.0, rel_vx=2.0),
        slot=1,
        rel_x=10.0,
        rel_y=0.0,
        rel_vx=-3.0,
    )
    reading = _reading(model, near_first)
    scrambled = dict(reading)
    scrambled[DETECTIONS_KEY] = reading[DETECTIONS_KEY][[1, 0, 2, 3]]

    in_order = _score(model, near_first, reading)
    swapped = _score(model, near_first, scrambled)

    assert np.isfinite(in_order)
    assert np.isfinite(swapped)
    assert in_order - swapped > 700.0
    assert _score(model, far_first, reading) == pytest.approx(in_order, rel=1e-12)


def test_an_opponent_hidden_behind_a_closer_one_is_not_charged_for_going_unreported():
    """A particle that predicts an occlusion pays nothing for the reading not showing it.

    Purpose: This is what sharing the occlusion rule with the world buys. The model runs the
        same range gate and the same angular blocking test the world ran, so *whether* a slot
        should have been reported is decided by one definition rather than two — otherwise
        every particle hypothesising a car behind another car would be punished for being
        right

    Given: A reading showing one detection at 12 m, and three particles: one holding that car
        alone, one holding it plus a second directly behind it at 24 m, and one holding it
        plus a second at 24 m off to the side where nothing blocks the line of sight
    When: All three are scored against that reading
    Then: The occluded pair scores exactly what the lone car scores, while the visible pair is
        charged 27.63 nats — the probability floor, because at the shipped rate of zero a
        detection it predicted and the reading did not show is not unlikely but impossible

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    lead = _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=0.0)
    hidden = _place_agent(lead, slot=1, rel_x=24.0, rel_y=0.0)
    beside = _place_agent(lead, slot=1, rel_x=24.0, rel_y=12.0)
    reading = _reading(model, lead)

    alone = _score(model, lead, reading)
    behind = _score(model, hidden, reading)
    visible = _score(model, beside, reading)

    assert behind == pytest.approx(alone, rel=1e-12)
    assert np.isfinite(visible)
    assert alone - visible == pytest.approx(_CONTRADICTION_NATS, rel=1e-9)


def test_the_matched_velocity_term_now_prices_motion_across_the_line_of_sight():
    """Moving only the reported crossing rate moves the score, where once it barely did.

    Purpose: Regression for exactly what the redesign bought. The matched-detection term used
        to score the *radial* component alone, so an opponent's motion across the ego's path
        — the component that says whether it is cutting in or peeling away — was all but
        unobserved. The magnitude is the point: "the score changed" would have passed on the
        old reading too, because the projection of a crossing velocity onto a line of sight
        1.5 m off the nose is small rather than zero

    Given: A POMDP model, a state carrying one opponent 12 m ahead and 1.5 m to the left
        moving at (-2.0, 0.5) m/s, its own noise-free reading, and a copy of that reading with
        the detection's rel_vy moved 1.0 m/s and nothing else touched
    When: Both are scored against the one state
    Then: The exact reading wins by 5.56 nats — the full Gaussian a 1 m/s residual implies at
        detection_velocity_std — which is over fifty times the 0.085 nats the same change
        would have cost when only its radial projection was scored

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=1.5, rel_vx=-2.0, rel_vy=0.5)
    reading = _reading(model, state)
    crossing = {key: value.copy() for key, value in reading.items()}
    crossing[DETECTIONS_KEY][0, DETECTION_REL_VY] += 1.0

    gap = _score(model, state, reading) - _score(model, state, crossing)

    line_of_sight = np.array([12.0, 1.5]) / float(np.linalg.norm([12.0, 1.5]))
    radial_only = 0.5 * (line_of_sight[1] / model.detection_velocity_std) ** 2
    assert gap == pytest.approx(0.5 * (1.0 / model.detection_velocity_std) ** 2, rel=1e-9)
    assert radial_only == pytest.approx(0.085, abs=0.005)
    assert gap > 50.0 * radial_only


def test_the_clutter_velocity_density_is_two_dimensional():
    """A phantom's reported velocity is priced in both components, not just one.

    Purpose: The clutter density has to be a density over everything a false alarm reports, or
        the two branches of the detection model are not comparable — a phantom whose crossing
        rate went unpriced would be cheaper than the truth by a constant that depends on
        nothing, and the likelihood would drift towards explaining traffic away

    Given: A POMDP model, an empty particle, and three readings holding one detection at the
        same place: at rest, moving 25 m/s along the line of sight, and 25 m/s across it
    When: All three are scored against that particle, which explains none of them
    Then: Both moving phantoms cost the same as each other and exactly log1p((25/scale)^2)
        more than the still one, so each component is scored at the same Cauchy scale

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    empty = _ego_state(model)
    base = _reading(model, empty)

    def _phantom(rel_vx: float, rel_vy: float) -> Dict[str, np.ndarray]:
        reading = {key: value.copy() for key, value in base.items()}
        row = reading[DETECTIONS_KEY][0]
        row[DETECTION_PRESENT] = 1.0
        row[DETECTION_REL_X], row[DETECTION_REL_Y] = 15.0, -4.0
        row[DETECTION_REL_VX], row[DETECTION_REL_VY] = rel_vx, rel_vy
        return reading

    at_rest = _score(model, empty, _phantom(0.0, 0.0))
    along = _score(model, empty, _phantom(25.0, 0.0))
    across = _score(model, empty, _phantom(0.0, 25.0))

    expected = float(np.log1p((25.0 / model.clutter_velocity_scale) ** 2))
    assert at_rest - along == pytest.approx(expected, rel=1e-6)
    assert at_rest - across == pytest.approx(expected, rel=1e-6)


def test_the_range_dial_decides_whether_an_unreported_car_costs_a_particle_anything():
    """Both halves of the dial: nothing for a car nobody could see, a finite price otherwise.

    Purpose: ``max_detection_range_m`` is this arm's headline dial, and it acts through the
        likelihood as much as through the world. The half that does the inference is a car
        *inside* range going unreported — that is what kills particles putting traffic where
        the reading shows none. The half that keeps it honest is a car outside range costing
        nothing, without which the filter would punish particles for hypothesising vehicles no
        sensor could ever have reported. The penalty must also stay finite, though only just:
        the world's detector is deterministic, so an unreported car inside range is a ruled-out
        hypothesis, and the probability floor is what keeps the filter's normalisation alive

    Given: Two models differing only in range, 40 m and 1e9 m; an empty reading; and three
        particles — an empty one, one holding a car at 60 m, and one holding a car at 20 m
    When: Each is scored against that empty reading
    Then: At 40 m the far car is free and the near one costs the 27.63-nat floor; wind the
        dial out to 1e9 m and the far car costs exactly the same 27.63 nats, because it is now
        a car the sensor should have seen

    Test type: unit
    """
    ranged = _model(observation_mode=ObservationMode.POMDP)
    unbounded = _model(observation_mode=ObservationMode.POMDP, max_detection_range_m=1e9)
    empty = _ego_state(ranged)
    beyond = _place_agent(empty, slot=0, rel_x=60.0, rel_y=0.0)
    inside = _place_agent(empty, slot=0, rel_x=20.0, rel_y=0.0)
    reading = _reading(ranged, empty)
    penalty = _CONTRADICTION_NATS

    baseline = _score(ranged, empty, reading)
    out_of_range = _score(ranged, beyond, reading)
    in_range = _score(ranged, inside, reading)
    dialled_out = _score(unbounded, beyond, reading)

    assert np.isfinite(in_range)
    assert np.isfinite(dialled_out)
    assert baseline - out_of_range == pytest.approx(0.0, abs=1e-12)
    assert baseline - in_range == pytest.approx(penalty, rel=1e-9)
    assert _score(unbounded, empty, reading) - dialled_out == pytest.approx(penalty, rel=1e-9)


def test_a_drawn_reading_beats_every_clearly_different_state_it_is_scored_against():
    """What draw() writes, log_prob() reads — one channel at a time.

    Purpose: The sampler and the density are one model, and the failure they can hide is a
        channel drawn and then never scored: every self-consistency check still passes, the
        reading still has the right shape, and the filter simply ignores a sensor. Displacing
        the state one coordinate at a time is what catches that, because a dead channel shows
        up as its own coordinate no longer separating anything

    Given: A POMDP model, a true state carrying one opponent, 32 readings drawn from it, and
        five copies of that state each displaced far in one coordinate — the ego's x, its
        arclength, its speed, its lateral offset, and the opponent's range
    When: The drawn readings are scored against the true state and against each displaced one
    Then: Every score is finite, and the true state's mean beats each displaced state's mean.
        Means rather than per-draw, because a draw that missed the detection or invented a
        phantom genuinely carries no information about where the opponent is

    Test type: unit
    """
    np.random.seed(17)
    model = _model(observation_mode=ObservationMode.POMDP)
    truth = _place_agent(
        _ego_state(model, lateral=0.2, arclength=40.0), slot=0, rel_x=12.0, rel_y=1.5, rel_vx=-2.0
    )

    draws = model.sample_observation(truth, COAST_STRAIGHT, n_samples=32)
    own = model.observation_log_probability(truth, COAST_STRAIGHT, draws)

    assert np.all(np.isfinite(own))
    for label, index, delta in (
        ("ego x", EGO_X, 2.0),
        ("arclength", EGO_ARCLENGTH_M, 2.0),
        ("speed", EGO_SPEED, 2.0),
        ("lateral", EGO_LAT, 1.0),
        ("slot range", EGO_STATE_WIDTH + AGENT_REL_X, 8.0),
    ):
        moved = truth.copy()
        moved[index] += delta
        displaced = model.observation_log_probability(moved, COAST_STRAIGHT, draws)
        assert np.all(np.isfinite(displaced)), label
        assert float(np.mean(own)) > float(np.mean(displaced)), label


@pytest.mark.parametrize("mode", [ObservationMode.POMDP, ObservationMode.MDP])
def test_a_particle_with_empty_slots_is_finitely_penalised_against_a_reading_with_traffic(mode):
    """Both arms score whether a vehicle is there, not only where it is.

    Purpose: This is the whole point of the detection model. Before it, presence was read
        from the state and never from the observation, so a particle with empty slots was
        scored on its ego row alone and paid nothing for a reading full of traffic

    Given: A state with one opponent 10.5 m ahead and 1.5 m to the left, an otherwise
        identical state with empty slots, and the noise-free reading of the first
    When: Both are scored against that reading
    Then: Both scores are finite and the state holding the vehicle wins — by 43.01 nats in the
        POMDP arm, where the empty particle pays the 27.63-nat floor for a detection it cannot
        explain plus the clutter density of where it landed *and* of the velocity it reported,
        and by 37.83 nats in the MDP arm, whose empty slot is the same ruled-out hypothesis

    Test type: unit
    """
    model = _model(observation_mode=mode)
    filled = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)
    empty = _ego_state(model)
    observation = (
        _reading(model, filled) if mode is ObservationMode.POMDP else _kinematics(model, filled)
    )

    with_vehicle = _score(model, filled, observation)
    without = _score(model, empty, observation)

    assert np.isfinite(with_vehicle)
    assert np.isfinite(without)
    assert with_vehicle > without
    expected = 43.01 if mode is ObservationMode.POMDP else 37.83
    assert with_vehicle - without == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("mode", [ObservationMode.POMDP, ObservationMode.MDP])
def test_a_state_holding_a_vehicle_the_observation_does_not_show_is_charged_and_survives(mode):
    """The converse direction is ruled out, at the floor rather than at -inf.

    Purpose: Silence about a vehicle the particle says is visible is not weak evidence here —
        the world's detector reports everything it can see, so the particle is wrong. The
        floor is what keeps that from arriving as a -inf and an all-zero weight vector, which
        crashes the filter's normalisation rather than telling it anything

    Given: A state with one opponent and an otherwise identical empty state, and the
        noise-free reading of the *empty* one
    When: Both are scored against that reading
    Then: The empty state wins by exactly the 27.63-nat floor in both arms, and the loser's
        score is finite

    Test type: unit
    """
    model = _model(observation_mode=mode)
    filled = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)
    empty = _ego_state(model)
    observation = (
        _reading(model, empty) if mode is ObservationMode.POMDP else _kinematics(model, empty)
    )

    with_vehicle = _score(model, filled, observation)
    without = _score(model, empty, observation)

    assert np.isfinite(with_vehicle)
    assert without - with_vehicle == pytest.approx(_CONTRADICTION_NATS, rel=1e-9)


# ── The MDP arm, left alone by the sensor redesign ──────────────────────


def test_the_mdp_likelihood_is_the_hand_written_sum_of_its_four_terms():
    """The MDP arm's density is pinned to a literal and to the arithmetic behind it.

    Purpose: The MDP arm is the control the POMDP arm is measured against, so an accidental
        re-pricing would move the yardstick and make every comparison incomparable. Pinning
        the literal values is what catches one. The literals moved once, deliberately, when
        the detection rates went to zero: the Bernoulli term that was log(0.95) + 3 log(0.98)
        is now four agreements at a probability of one, which is free

    Given: An MDP model at the shipped widths, a state with one opponent 10.5 m ahead and
        1.5 m left, its own noise-free reading, and an otherwise identical empty state
    When: Both states are scored against that reading
    Then: The matching state scores -5.965213904521491 and the empty one -43.798817103861936,
        and the first equals the hand-written sum of its four terms: a 4-D ego Gaussian at
        0.5, a Bernoulli of log(1) + 3 log(1), a 2-D position Gaussian at 1.0 and a 2-D
        velocity Gaussian at 2.0

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.MDP)
    filled = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)
    observation = _kinematics(model, filled)
    by_hand = (
        -0.5 * 4 * np.log(2.0 * np.pi * model.ego_pose_std**2)
        + np.log(1.0 - model.presence_miss_prob)
        + 3 * np.log(1.0 - model.presence_false_alarm_prob)
        - 0.5 * 2 * np.log(2.0 * np.pi * model.agent_pose_std**2)
        - 0.5 * 2 * np.log(2.0 * np.pi * model.agent_velocity_std**2)
    )

    assert _score(model, filled, observation) == pytest.approx(-5.965213904521491, rel=1e-12)
    # The floor leaves 4e-12 nats on the four agreeing flags, which the hand-written form
    # writes as an exact log(1); rel=1e-9 is what lets the two agree without restating it.
    assert _score(model, filled, observation) == pytest.approx(float(by_hand), rel=1e-9)
    assert _score(model, _ego_state(model), observation) == pytest.approx(
        -43.798817103861936, rel=1e-12
    )


def test_the_mdp_clutter_density_is_what_keeps_a_lossy_radars_likelihood_the_right_way_up():
    """Without a density over what a false alarm reports, a lossy MDP likelihood inverts.

    Purpose: Regression for a real inversion, and for the configuration it can happen in. The
        shipped rates are zero, so a detection no slot explains is a hypothesis already ruled
        out; configure a lossy radar and the two branches have to be compared for real. A
        matched slot scores a 4-D Gaussian whose peak at the default widths is exp(-5.06) =
        0.0063, *below* a 0.02 false-alarm rate. Score the flag alone and "no vehicle,
        spurious detection" beats a perfect match every time, so the model explains away every
        vehicle it can see

    Given: An MDP model configured as a lossy radar at 0.05 miss and 0.02 false alarm, a state
        holding the observed vehicle, and an empty one
    When: The clutter term and the detection term are compared against the Gaussian peak
    Then: The clutter term is more negative than the Gaussian peak by enough to reverse the
        ordering, and the full likelihood puts the matching state ahead

    Test type: unit
    """
    model = _model(
        observation_mode=ObservationMode.MDP,
        presence_miss_prob=0.05,
        presence_false_alarm_prob=0.02,
    )
    filled = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)
    observation = _kinematics(model, filled)
    phantom = np.asarray(observation[AGENTS_KEY], dtype=float)[:1]

    clutter = _kinematics_arm(model).clutter_log_prob(phantom)
    gaussian_peak = -0.5 * 2 * np.log(2.0 * np.pi * model.agent_pose_std**2) - 0.5 * 2 * np.log(
        2.0 * np.pi * model.agent_velocity_std**2
    )

    assert clutter < gaussian_peak
    assert np.exp(gaussian_peak) < model.presence_false_alarm_prob
    assert clutter + np.log(model.presence_false_alarm_prob) < gaussian_peak + np.log(
        1.0 - model.presence_miss_prob
    )


def test_the_mdp_sampler_drops_and_invents_slots_at_the_configured_rates():
    """The sampler applies the detection model the density scores, not a subset of it.

    Purpose: This package has already shipped a layer that was rendered and then not scored.
        A sampler that never drops or never invents makes the density it is paired with the
        density of a different model, and every self-consistency check silently measures the
        wrong thing

    Given: A two-slot state with the first filled and the second empty, drawn 20000 times at
        exaggerated rates so the counts are unambiguous
    When: The fraction of draws reporting each slot is measured
    Then: The filled slot is reported 1 - miss of the time and the empty one false_alarm of
        the time, each within 1.5 percentage points

    Test type: unit
    """
    np.random.seed(4)
    model = _model(
        observation_mode=ObservationMode.MDP,
        max_tracked_agents=2,
        presence_miss_prob=0.25,
        presence_false_alarm_prob=0.10,
    )
    state = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)

    reported = np.array(
        [
            np.asarray(model.sample_observation(state, COAST_STRAIGHT)[AGENTS_KEY], dtype=float)[
                :, AGENT_PRESENT
            ]
            for _ in range(20000)
        ]
    )

    assert abs(reported[:, 0].mean() - 0.75) < 0.015
    assert abs(reported[:, 1].mean() - 0.10) < 0.015


def test_the_pomdp_sampler_drops_and_invents_detections_at_the_configured_rates():
    """The radar's own sampler applies the detection model the density scores.

    Purpose: The same argument as the MDP sampler above, on the arm the redesign rewrote. A
        draw and a score that disagree about how often a visible car goes unreported make
        every likelihood measurement on this arm the measurement of a different model.
        Counting *how many* detections come back rather than which rank each landed in,
        because the reading is packed nearest-first: a phantom slides into rank 0 whenever
        the real vehicle was dropped

    Given: A state with one visible opponent at 12 m and three empty ranks, drawn 20000 times
        at an exaggerated 0.25 miss and 0.10 false-alarm rate
    When: The number of detections per draw is counted
    Then: The mean count is 1 - miss + 3 * false_alarm = 1.05, and the fraction of draws
        carrying nothing at all is miss * (1 - false_alarm)^3 = 0.182, each within 0.02

    Test type: unit
    """
    np.random.seed(6)
    miss, false_alarm = 0.25, 0.10
    model = _model(
        observation_mode=ObservationMode.POMDP,
        presence_miss_prob=miss,
        presence_false_alarm_prob=false_alarm,
    )
    state = _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=0.0)

    counts = np.array(
        [
            np.asarray(
                model.sample_observation(state, COAST_STRAIGHT)[DETECTIONS_KEY], dtype=float
            )[:, DETECTION_PRESENT].sum()
            for _ in range(20000)
        ]
    )

    assert abs(counts.mean() - (1.0 - miss + 3.0 * false_alarm)) < 0.02
    assert abs(float(np.mean(counts == 0.0)) - miss * (1.0 - false_alarm) ** 3) < 0.02


def test_at_the_shipped_rates_the_pomdp_sampler_reports_the_visible_set_and_nothing_else():
    """Zero rates mean the draw is the predicted-visible set, every single time.

    Purpose: The density excludes a particle that disagrees with the reading about
        visibility, so the sampler has to be the model that justifies it. One dropped row or
        one phantom in a long run and the two halves are different models, and the exclusion
        is charging particles for the sampler's noise

    Given: A state with two opponents inside the 40 m gate and clear of each other, a third
        beyond it, and a fourth hidden directly behind the nearest, drawn 2000 times at the
        shipped rates of zero
    When: The detections in each draw are counted and their flags read
    Then: Every draw carries exactly the two visible opponents, in ranks 0 and 1, and no draw
        carries anything in the ranks behind them

    Test type: unit
    """
    np.random.seed(11)
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model), slot=0, rel_x=12.0, rel_y=0.0)
    state = _place_agent(state, slot=1, rel_x=20.0, rel_y=15.0)
    state = _place_agent(state, slot=2, rel_x=60.0, rel_y=0.0)
    state = _place_agent(state, slot=3, rel_x=24.0, rel_y=0.0)
    visible = len(_sensor_arm(model).predicted_detections(state))

    flags = np.array(
        [
            np.asarray(
                model.sample_observation(state, COAST_STRAIGHT)[DETECTIONS_KEY], dtype=float
            )[:, DETECTION_PRESENT]
            for _ in range(2000)
        ]
    )

    assert visible == 2
    assert np.all(flags[:, :visible] == 1.0)
    assert not np.any(flags[:, visible:])


def test_at_the_shipped_rates_the_mdp_sampler_reports_every_filled_slot_and_no_empty_one():
    """The MDP arm's sampler is deterministic in presence for the same reason.

    Purpose: The MDP arm's slot flags run through the same detection composition as the
        radar's ranks, so zero rates have to land on both. A sampler that still dropped a
        filled slot here would make the MDP baseline's own likelihood the likelihood of a
        different model, and the baseline is the yardstick

    Given: A two-slot state with the first filled and the second empty, drawn 2000 times at
        the shipped rates of zero
    When: The presence flags of both slots are read across the draws
    Then: The filled slot is reported in every draw and the empty one in none

    Test type: unit
    """
    np.random.seed(12)
    model = _model(observation_mode=ObservationMode.MDP, max_tracked_agents=2)
    state = _place_agent(_ego_state(model), slot=0, rel_x=10.5, rel_y=1.5)

    reported = np.array(
        [
            np.asarray(model.sample_observation(state, COAST_STRAIGHT)[AGENTS_KEY], dtype=float)[
                :, AGENT_PRESENT
            ]
            for _ in range(2000)
        ]
    )

    assert np.all(reported[:, 0] == 1.0)
    assert not np.any(reported[:, 1])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"presence_miss_prob": -0.01}, "presence_miss_prob must be in"),
        ({"presence_false_alarm_prob": 1.0}, "presence_false_alarm_prob must be in"),
        (
            {"presence_miss_prob": 0.6, "presence_false_alarm_prob": 0.4},
            "must stay below 1",
        ),
        ({"clutter_position_scale_m": 0.0}, "clutter_position_scale_m must be positive"),
        ({"clutter_velocity_scale": -1.0}, "clutter_velocity_scale must be positive"),
    ],
)
def test_unusable_detection_settings_are_rejected_at_construction(overrides, message):
    """Rates the composition cannot use are refused rather than producing a silent inversion.

    Purpose: At miss + false_alarm >= 1 a tracked vehicle makes a detection no *more* likely
        to be reported than an empty rank, so the likelihood runs backwards while every value
        stays finite and plausible — the kind of error a filter absorbs without complaint

    Given: A rate outside [0, 1), a pair summing to 1, or a non-positive clutter scale
    When: A model is constructed with it
    Then: ValueError is raised, naming the parameter

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _model(**overrides)


# ── Against the live simulator ──────────────────────────────────────────


def _track_live_world(action: int, steps: int = 10) -> tuple:
    """Run one action through the live world and the model, returning both end states."""
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (  # pylint: disable=import-outside-toplevel
        geometry_from_world,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    model = _model(track_geometry=geometry_from_world(world)[0])
    predicted = truth.copy()
    for _ in range(steps):
        predicted = model.sample_next_state(predicted, action)
        truth = np.asarray(world.sample_next_state(truth, action), dtype=float)
    return predicted, truth


def test_model_tracks_the_live_simulator_over_ten_coasting_steps(monkeypatch):
    """Test that the reproduced bicycle matches highway-env's own integration.

    Purpose: This is the only check that the model is a model *of this world*. Every closed
        form above could be self-consistently wrong; running the same action through both
        and comparing the ego pose is what rules that out

    Given: A live racetrack world seeded at 0 with no extra opponents, and a noise-free
        model started from the world's true state and carrying the world's own track map
    When: Ten coast-straight decisions are applied to both, the model propagating from its
        own prediction rather than being re-synchronised
    Then: The predicted (x, y, heading, lateral offset, arclength) stays within 1e-6 of the
        simulator's. The tolerance is tight on purpose: the ego update is reproduced term
        for term, so the only expected difference is float accumulation over thirty
        substeps — the measured error at this seed is in fact exactly zero, and anything
        near 1e-6 would already be a real disagreement rather than drift

    Note:
        The arclength comparison is only valid because ten steps from the spawn point do
        not reach the lap seam. The world derives arclength from lane offsets, so it wraps
        into ``[0, lap)``; the model integrates it and leaves it unbounded, exactly as
        ``curvature_at`` expects. Extend this drive past a full lap and the two must be
        compared modulo the lap length, the same caveat heading already carries

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")

    predicted, truth = _track_live_world(COAST_STRAIGHT)

    assert abs(predicted[EGO_X] - truth[EGO_X]) < 1e-6
    assert abs(predicted[EGO_Y] - truth[EGO_Y]) < 1e-6
    assert abs(predicted[EGO_HEADING] - truth[EGO_HEADING]) < 1e-6
    assert abs(predicted[EGO_LAT] - truth[EGO_LAT]) < 1e-6
    assert abs(predicted[EGO_ARCLENGTH_M] - truth[EGO_ARCLENGTH_M]) < 1e-6


def test_model_tracks_the_live_simulator_through_a_full_lock_arc(monkeypatch):
    """Test that the model tracks the simulator when steering, not just when coasting.

    Purpose: Coasting straight leaves the slip angle at zero, so it cannot catch a wrong
        Frenet rate. Under full lock the slip angle is 0.46 rad and the lateral offset must
        be integrated along the *velocity* direction rather than the heading; getting that
        wrong costs about 2 m over this manoeuvre while everything else still looks right

    Given: The same seeded world and noise-free model
    When: Ten full-lock left decisions are applied to both
    Then: Position, heading and lateral offset all match to 1e-6. Heading is compared
        modulo 2*pi because the model wraps to [-pi, pi) and highway-env does not

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")

    predicted, truth = _track_live_world(COAST_FULL_LEFT)

    heading_error = (predicted[EGO_HEADING] - truth[EGO_HEADING] + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(predicted[EGO_X] - truth[EGO_X]) < 1e-6
    assert abs(predicted[EGO_Y] - truth[EGO_Y]) < 1e-6
    assert abs(heading_error) < 1e-6
    assert abs(predicted[EGO_LAT] - truth[EGO_LAT]) < 1e-6
