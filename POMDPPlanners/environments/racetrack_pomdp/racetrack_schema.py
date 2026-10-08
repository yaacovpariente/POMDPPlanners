# SPDX-License-Identifier: MIT

"""Shared schema for the racetrack POMDP: state layout, config, and reward.

This module is the single source of truth shared by the three racetrack pieces — the
forward-only world, the planner-side generative model, and the belief. It deliberately
imports **nothing** from ``highway_env``, so the model and the belief stay pure NumPy and
can be constructed, tested and pickled on a machine where the simulator is not installed.
Only :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp` touches the
backend.

**The matched pair.** :func:`build_racetrack_config` assembles the whole highway-env
configuration once and attaches an ``"observation"`` block last. Every dynamics key — the
action type, the step rates, the reward weights, the vehicle counts — comes off the same
code path before the observation is chosen, so the MDP baseline and the POMDP differ in
what the agent *sees* and in nothing else. That property is what lets a planner's
performance gap be attributed to partial observability, so it is asserted by a test rather
than trusted.

The guarantee is asserted on the *dynamics* config rather than on the observation key,
because the POMDP arm's reading is no longer something highway-env produces. The simulator
supplies only the ego's own kinematics; the curvature-ahead channel and the detections are
measured off the road network and the vehicle list by
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp` and corrupted there. So
the test compares the two configs with ``"observation"`` removed and requires them to be
byte-identical, which is the same guarantee stated against what it actually protects.

**Longitudinal control is enabled**, unlike the racetrack defaults. Under the shipped
configuration ``ContinuousAction`` is lateral-only: acceleration is pinned at zero and the
``target_speeds`` key is inert (it belongs to ``DiscreteMetaAction``). The ego could then
never brake for an opponent, which removes most of what partial observability costs. The
flag is a dynamics key applied identically to both arms, so the matched pair is preserved.

**State layout** (identical in the world and in the model, on purpose — a wider world
state is what makes CARLA's agent-slot reshape unsafe against a world vector)::

    [x, y, heading, speed, lat, ang, s] + max_tracked_agents * [present, rel_x, rel_y, rel_vx, rel_vy]

The ego block is world-frame position in metres, heading in radians, scalar speed in m/s,
then the Frenet terms: signed lateral offset from the lane centreline in metres, the angle
between the heading and the lane direction in radians, and the distance travelled along the
track centreline in metres.

The last slot is deliberately the car's **arclength**, not the road's curvature. Curvature is
a property of the road, so freezing it in the state encodes a prediction about the future
rather than a fact about the present, and a rollout that reuses it drives straight through
every corner. Where the road bends is the transition model's business: see
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry`.

Agent slots are in the ego body frame — ``rel_x`` forward, ``rel_y`` left — and hold
**relative** velocity. A slot the sensor can see is reported in full; a slot outside the
range gate or behind a closer vehicle is not reported at all, and *that* is the hidden state.

Classes:
    ObservationMode: Which observation the world emits; dynamics are unaffected.
    RacetrackObservation: The POMDP arm's reading — ego pose, speedometer, lane camera,
        curvature ahead, and unlabeled detections.
"""

from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

# ── State layout ────────────────────────────────────────────────────────
EGO_STATE_WIDTH = 7
AGENT_SLOT_WIDTH = 5
DEFAULT_MAX_TRACKED_AGENTS = 4

EGO_X = 0
EGO_Y = 1
EGO_HEADING = 2
EGO_SPEED = 3
EGO_LAT = 4
EGO_ANG = 5
EGO_ARCLENGTH_M = 6

AGENT_PRESENT = 0
AGENT_REL_X = 1
AGENT_REL_Y = 2
AGENT_REL_VX = 3
AGENT_REL_VY = 4

# ── Detections ──────────────────────────────────────────────────────────
# One detection: whether the slot holds one at all, where the vehicle is in the ego body
# frame, and its full relative velocity in that frame. A detection reports a tracked
# vehicle's whole kinematic row, not a degraded projection of it.
#
# The fourth and fifth entries used to be a single Doppler closing rate, which is what a
# radar alone measures. That was dropped on purpose: the crossing rate then had to be
# inferred, and inferring it is a *different* estimation problem from the one this
# environment exists to pose. What stays hidden here is a vehicle the sensor cannot see at
# all — beyond ``max_detection_range_m``, or behind a closer one — and nothing else. A
# production stack fusing radar with a camera or a lidar reports both components anyway, so
# this is also the more honest sensor.
DETECTION_PRESENT = 0
DETECTION_REL_X = 1
DETECTION_REL_Y = 2
DETECTION_REL_VX = 3
DETECTION_REL_VY = 4
DETECTION_SLOT_WIDTH = 5

# ── Sensor widths ───────────────────────────────────────────────────────
# These are sensor noise in the literal sense: the world corrupts its own readings at these
# widths before emitting them, and the planner's model must assume the same numbers or its
# filter is confidently wrong.
#
# The position and velocity widths are the accuracy band an automotive radar in this class is
# specified at — decimetre-scale range and cross-range accuracy at the tens of metres this
# sensor reaches, and a few tenths of a metre per second on the velocity, which is a far
# tighter measurement than position because it comes off a frequency shift rather than a time
# of flight. The velocity width is the Doppler figure the closing rate used to carry, now
# applied to both components: it is the sensor's velocity accuracy, and the axis it happens
# to be quoted along is not what makes it small.
DEFAULT_DETECTION_POSITION_STD_M = 0.5
DEFAULT_DETECTION_VELOCITY_STD = 0.3

# How far the sensor sees, and **the dial this environment turns**. Beyond this a vehicle is
# simply absent from the observation, which is a real and frequent event on this circuit: at
# the shipped speed limit an opponent crosses the boundary every few seconds. Everything else
# in the state is reported, so the MDP and POMDP arms are the two ends of a continuum in this
# one number — at ``R -> inf`` the reading is the state to within the sensor widths, and as R
# shrinks the traffic disappears from it first.
DEFAULT_MAX_DETECTION_RANGE_M = 40.0

# Occlusion is deterministic and geometric. A vehicle is masked when a *closer* vehicle lies
# within the angular half-width its own body subtends from the ego, treating a blocker as a
# disc of this radius — 1 m, so a 2 m-wide car. The half-width at range r is
# ``arcsin(min(1, DEFAULT_BLOCKER_HALF_WIDTH_M / r))``, which is exact for a disc and degrades
# to "everything behind a vehicle you are touching is hidden" as r goes to zero rather than to
# a domain error. A disc and not the true oriented rectangle because the state carries no
# heading for the other vehicles, so a rectangle would need a quantity the model cannot
# represent; the disc is the inscribed approximation and it under-occludes a car seen
# broadside, which is the conservative direction.
DEFAULT_BLOCKER_HALF_WIDTH_M = 1.0

# ── Curvature ahead ─────────────────────────────────────────────────────
# What a lane-detection camera reports about the road in front of the bumper: the curvature of
# the lane at fixed distances along it. The world reads these off the true track geometry and
# adds Gaussian noise.
#
# The distances are the band this circuit's corners live in. At the shipped 10 m/s speed limit
# a five-decision rollout covers about 10 m, so the nearest sample is the curvature the rollout
# is actually driving through and the two further ones say what is coming.
DEFAULT_CURVATURE_LOOKAHEAD_M: Tuple[float, ...] = (10.0, 20.0, 30.0)

# The noise width, and it is derived rather than picked. A camera does not measure curvature;
# it fits a polynomial through lane points whose *lateral* position it measures, so the
# curvature error follows from the lateral error and the distance it is fitted over. For a
# parabola through points at distance d with lateral error sigma_y, the quadratic coefficient
# has error of order sigma_y / d^2 and the curvature is twice that. At the nearest lookahead —
# d = 10 m, sigma_y = 0.1 m, the decimetre band the same detector's lateral offset is quoted
# at a few metres out — that is 2e-3 1/m. The shipped circuit's arcs run 0.01 to 0.05 1/m, so
# this is 4% to 20% of the signal: a real measurement with real error, not a formality.
DEFAULT_CURVATURE_STD_1PM = 2.0e-3

# ── Detection rates ─────────────────────────────────────────────────────
# Distinct from the widths above, and worth keeping apart from them: those say how wrong a
# reported number is, these say whether the report happens at all. They are what lets the
# likelihood score *whether* a vehicle is there and not only where. The predicted probability
# that a slot is reported is
#
#     p = q * (1 - miss) + (1 - q) * false_alarm
#
# for an occupancy q, which is the textbook detection composition. `false_alarm` is what a
# particle carrying no vehicle pays for a detection the observation reports; `miss` is what a
# particle carrying one pays for a detection the observation does not.
#
# **Both are zero, because this world's detection decision is deterministic.** The range gate
# and the occlusion rule are applied to the vehicles' true positions, and the radar neither
# drops a vehicle it can see nor invents one it cannot. A nonzero rate here would be a sensor
# that does not exist and that no measurement of this world could fit. At zero the likelihood
# does what Bayes says it should: a particle whose visibility prediction contradicts the
# reading is *excluded*, not discounted.
#
# They remain parameters because a lossy radar is a legitimate thing to configure — set either
# above zero and both arms model one, in the sampler as well as in the density. What they are
# not is a property of the shipped world.
#
# Scoring floors a probability at `racetrack_detection.PROBABILITY_EPS`, so a contradiction
# costs about 27.6 nats instead of -inf. That floor is a NUMERICAL guard: on a finite particle
# set an all-zero weight vector is a crash rather than an inference. It is not a claim that
# the sensor misses one vehicle in 1e12.
DEFAULT_PRESENCE_MISS_PROB = 0.0
DEFAULT_PRESENCE_FALSE_ALARM_PROB = 0.0

# ── Clutter model ───────────────────────────────────────────────────────
# A bare false-alarm rate is not comparable with a matched detection's density: one is a
# probability and the other a probability *density*, and subtracting the two inverts the
# likelihood. Measured on the MDP arm before this term existed, a state holding the observed
# vehicle scored 1.20 nats *worse* than one holding nothing.
#
# So a false alarm reports a phantom drawn from a clutter distribution, as PDA and JPDA have
# done since the 1970s. This is part of the lossy-radar configuration above: at the shipped
# `DEFAULT_PRESENCE_FALSE_ALARM_PROB` of 0 no sampler ever draws a phantom, and the density's
# clutter term only reaches a detection the particle already cannot explain. Turn the rate up
# and the term is what keeps the two branches comparable.
#
# Cauchy rather than the usual uniform-over-the-field-of-view, because
# the slots are ranked by range with no window to be uniform over: the heavy tail keeps a
# genuinely distant report at a log cost instead of a quadratic one, and the support is all of
# R so no observation can be impossible.
#
# The scales are the range over which a spurious report is plausible, not fitted widths. 18 m
# was the occupancy grid's half-extent, which is what the MDP arm was calibrated against; it
# is kept at that number so the MDP arm's likelihood is bit-identical across this redesign,
# and it remains a defensible plausible-range scale for a 40 m sensor given the Cauchy's tail.
# The velocity scale is `DEFAULT_SPEED_LIMIT`, set further down beside the other simulator
# defaults.
DEFAULT_CLUTTER_POSITION_SCALE_M = 18.0

# ── Observed ego pose ───────────────────────────────────────────────────
# Where the car is, which way it points, and how far round the lap it has come: GPS/IMU for
# the first three, a wheel odometer against the lane graph for the last. Four numbers, and
# they are near-exact because that is what the hardware is — a production stack localises to
# decimetres and its heading to a fraction of a degree.
#
# Withholding them was the old design's second source of partial observability, and it was
# never the one this environment exists to study. The car not knowing where it is on a
# circuit it can see is a localisation problem; the car not knowing where the other cars are
# is a tracking problem, and only the second is what a range gate controls. Observing the
# pose leaves exactly one thing hidden, so the range dial is the only variable in play.
#
# The widths are small but not zero: a zero-width channel is a delta in the likelihood, and
# the first particle whose dead reckoning misses by a hair would be annihilated.
EGO_POSE_X = 0
EGO_POSE_Y = 1
EGO_POSE_HEADING = 2
EGO_POSE_ARCLENGTH = 3
OBSERVED_EGO_POSE_WIDTH = 4
DEFAULT_EGO_POSITION_STD_M = 0.1
DEFAULT_EGO_HEADING_STD_RAD = 0.01
DEFAULT_EGO_ARCLENGTH_STD_M = 0.1

# ── Observed ego speed ──────────────────────────────────────────────────
# The POMDP arm reads its own speedometer. Every real car has one, and withholding it would
# add a second source of partial observability on top of the one this environment exists to
# measure.
#
# `vehicles_count` is 2 and not 1, which is the reading everyone reaches for and the one
# that silently breaks. In highway-env 1.12.1 `KinematicObservation.observe` asks the road
# for `count=vehicles_count - 1`, and `Road.close_objects_to` guards its truncation with
# `if count:` -- so `count=0` means *no limit* rather than *none*. It then slices
# `close_vehicles[-vehicles_count + 1:]`, which at `vehicles_count=1` is `[0:]`: every
# vehicle in range. Measured on a three-vehicle track, `vehicles_count=1` returns a (3, 2)
# block carrying two opponents' velocities -- exactly the hidden state the POMDP arm exists
# to withhold. At 2 the block is the (2, n) it claims to be, and the world drops row 1
# before the observation leaves it. There is no config that restricts the block to the ego
# alone, so the trim happens in the adapter and a test pins it.
EGO_KINEMATICS_VEHICLES_COUNT = 2
EGO_KINEMATICS_FEATURES: Tuple[str, ...] = ("vx", "vy", "cos_h", "sin_h")
OBSERVED_EGO_SPEED_WIDTH = 1

# ── Observed lane pose ──────────────────────────────────────────────────
# The lane-keeping camera: signed lateral offset from the lane centreline in metres and
# the angle between the ego's heading and the lane direction in radians, the two numbers
# every production lane-keeper measures. They are read out of the same `lane_offset` the
# EGO_LAT and EGO_ANG state slots come from, so the observation lines up one-to-one with
# two state slots the way the speedometer lines up with EGO_SPEED.
#
# The racetrack reward is *built* from the lateral offset, so withholding it would score the
# agent on lane-centering while never telling it its lane offset.
#
# Unlike the speedometer, these are emitted with noise, because `lane_offset` is exact and
# a camera is not. The defaults are the accuracy a production mono-camera lane detector is
# specified at: centimetre-scale lateral error and sub-degree heading error over the few
# metres ahead of the bumper. 0.05 m is the conservative end of the centimetre band; 0.01 rad
# is 0.57 degrees. They come off the same camera as `DEFAULT_CURVATURE_STD_1PM`, which is
# derived from this lateral accuracy carried out to the lookahead distances.
LANE_POSE_LAT = 0
LANE_POSE_ANG = 1
OBSERVED_LANE_POSE_WIDTH = 2
DEFAULT_LANE_LATERAL_STD_M = 0.05
DEFAULT_LANE_HEADING_STD_RAD = 0.01

# ── Flattened observation layout ────────────────────────────────────────
# Where each channel starts in the flat vector the torch model trades in. See
# :func:`pomdp_observation_width`; the detections start after a variable number of
# curvature samples, so their offset is a function rather than a constant.
POMDP_OBS_EGO_POSE_INDEX = 0
POMDP_OBS_EGO_SPEED_INDEX = POMDP_OBS_EGO_POSE_INDEX + OBSERVED_EGO_POSE_WIDTH
POMDP_OBS_LANE_POSE_INDEX = POMDP_OBS_EGO_SPEED_INDEX + OBSERVED_EGO_SPEED_WIDTH
POMDP_OBS_CURVATURE_INDEX = POMDP_OBS_LANE_POSE_INDEX + OBSERVED_LANE_POSE_WIDTH

# ── Action presets ──────────────────────────────────────────────────────
# highway-env maps a normalised action in [-1, 1] onto these ranges.
MAX_ACCELERATION_MPS2 = 5.0
MAX_STEERING_RAD = np.pi / 4

# Every acceleration crossed with every steering angle, both normalised to [-1, 1].
# The planner selects an index into this tuple, so the world and the model share
# one action vocabulary by construction rather than by convention.
# Steering is sampled finely near zero and coarsely at the extremes. Full lock is
# pi/4 = 45 degrees, which on this track is a spin rather than a correction: sweeping
# constant steering through the first bend, -1.0 survives 5 steps while -0.05 survives
# 29. A bang-bang set of {-1, 0, +1} simply does not contain the manoeuvre the track
# needs, so no amount of planning can select it.
STEERING_PRESETS: Tuple[float, ...] = (-1.0, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 1.0)
ACCELERATION_PRESETS: Tuple[float, ...] = (1.0, 0.0, -1.0)

DEFAULT_ACTION_PRESETS: Tuple[Tuple[float, float], ...] = tuple(
    (acceleration, steering)
    for acceleration in ACCELERATION_PRESETS
    for steering in STEERING_PRESETS
)

# ── Simulator defaults (racetrack-v0, verified against highway-env 1.12.1) ──
DEFAULT_ENV_ID = "racetrack-v0"
DEFAULT_DURATION = 300
DEFAULT_POLICY_FREQUENCY = 5
DEFAULT_SIMULATION_FREQUENCY = 15
DEFAULT_OTHER_VEHICLES = 1
DEFAULT_SPEED_LIMIT = 10.0
DEFAULT_COLLISION_REWARD = -1.0
DEFAULT_LANE_CENTERING_COST = 4.0
DEFAULT_LANE_CENTERING_REWARD = 1.0
DEFAULT_ACTION_REWARD = -0.3
DEFAULT_NEAR_MISS_DISTANCE_M = 5.0

# The velocity half of the MDP arm's clutter model; see `DEFAULT_CLUTTER_POSITION_SCALE_M`
# above. Relative velocities between vehicles on this track are of the order of the speed
# limit, so that is the scale over which a spurious velocity report is plausible.
DEFAULT_CLUTTER_VELOCITY_SCALE = DEFAULT_SPEED_LIMIT

_CONFIG_OBSERVATION_KEY = "observation"


class ObservationMode(Enum):
    """Which observation the racetrack world emits.

    The two modes share one dynamics path and differ only in the observation, so a
    planner's performance gap between them measures partial observability alone.

    Attributes:
        MDP: Absolute position and velocity for the ego and nearby vehicles. Only the
            other vehicles' driver policy stays hidden, so this is a near-MDP baseline
            rather than a true MDP.
        POMDP: **The full state, minus the vehicles the sensor cannot see.** The ego's
            own pose, speed and lane-relative pose are all reported, as is the curvature
            of the road ahead; the other vehicles are reported in full when they are
            within ``max_detection_range_m`` and not behind a closer one, and are absent
            from the reading entirely when they are not.

    Note:
        The POMDP arm withholds **one** thing on purpose: vehicles out of sensor range or
        line of sight. Everything else a sensor could fetch is in the reading, at
        near-exact widths, so a planner's loss can be attributed to the traffic it could
        not see rather than to a pose it had to estimate at the same time. What remains
        hidden is therefore:

        * **Vehicles outside the range gate.** The headline dial. Nothing in the reading
          says whether the road beyond ``max_detection_range_m`` is empty or full.
        * **Occlusion.** A vehicle behind a closer one is not in the reading at all.
        * **Driver intent.** No sensor reports what another car is about to do.
        * **Identity.** Detections are unlabeled, emitted sorted by range, so nothing
          carries across a step except through the filter.

        That makes the two modes a **continuum in one number** rather than a pair of
        unrelated readings: at ``max_detection_range_m -> inf`` the POMDP arm reports
        every vehicle and the reading is the state to within the sensor widths, and as the
        range shrinks the traffic drops out of it first. A planner that fails here fails
        on cars it genuinely could not see.
    """

    MDP = "mdp"
    POMDP = "pomdp"


class RacetrackObservation(NamedTuple):
    """What the POMDP arm emits: the whole state, minus the vehicles it cannot see.

    A tuple rather than a stacked array because the parts have nothing in common — a pose,
    metres per second, a metre-and-radian pair, reciprocal metres, and a ragged list of
    returns — and a named tuple rather than a bare one so a reader of
    ``observation.detections`` does not have to remember which index that is.
    ``np.asarray`` on it raises instead of quietly producing something wrong.

    Each channel is one field per *sensor*, not per number: a camera that loses the lane
    loses both its readings together, and anything that identified a channel by its size
    would otherwise mistake a lone lateral offset for a second speedometer.

    Attributes:
        ego_pose: A ``(4,)`` float32 array of world-frame ``x``, ``y`` (m), heading (rad)
            and arclength along the lap (m), measured near-exactly — GPS/IMU and an
            odometer against the lane graph.
        ego_speed: A ``(1,)`` float32 array holding the ego's signed speed in m/s, exact.
        lane_pose: A ``(2,)`` float32 array of lateral offset (m) and lane-relative
            heading (rad), measured with noise; see the note beside its defaults.
        curvature_ahead: A ``(L,)`` float32 array of the lane's signed curvature in 1/m at
            each of the camera's fixed lookahead distances, measured with noise.
        detections: A ``(K, 5)`` float32 array of
            ``[detected, rel_x, rel_y, rel_vx, rel_vy]`` rows in the ego body frame,
            ordered by measured range, with undetected slots left at zero. A visible
            vehicle's whole kinematic row is reported; a vehicle out of range or behind a
            closer one produces no row at all, which is the arm's only hidden state.
            Unlabeled: row ``k`` on one step and row ``k`` on the next need not be the same
            vehicle.
    """

    ego_pose: np.ndarray
    ego_speed: np.ndarray
    lane_pose: np.ndarray
    curvature_ahead: np.ndarray
    detections: np.ndarray


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi)``.

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in ``[-pi, pi)``.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def state_agent_rows(state: np.ndarray, max_tracked_agents: int) -> np.ndarray:
    """View a state vector's agent slots as a ``(max_tracked_agents, 5)`` block.

    Args:
        state: A state vector of width ``EGO_STATE_WIDTH + max_tracked_agents * 5``.
        max_tracked_agents: Number of fixed agent slots in the state.

    Returns:
        The agent slots reshaped to one row per slot. This is a reshape of a slice, so
        it may share memory with ``state``; copy it before mutating.

    Raises:
        ValueError: If the state width does not match ``max_tracked_agents``.
    """
    array = np.asarray(state, dtype=float)
    expected = EGO_STATE_WIDTH + max_tracked_agents * AGENT_SLOT_WIDTH
    if array.shape[-1] != expected:
        raise ValueError(
            f"State width {array.shape[-1]} does not match max_tracked_agents="
            f"{max_tracked_agents} (expected {expected})."
        )
    return array[..., EGO_STATE_WIDTH:].reshape(
        *array.shape[:-1], max_tracked_agents, AGENT_SLOT_WIDTH
    )


def rotate(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D row vectors counter-clockwise by ``angle`` radians.

    Args:
        vectors: Array whose last axis has length 2.
        angle: Rotation angle in radians.

    Returns:
        The rotated vectors, same shape as the input.
    """
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    return np.asarray(vectors, dtype=float) @ matrix.T


def radial_velocities(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Project relative velocities onto their own lines of sight.

    A geometry helper, and no longer part of the observation path: detections now carry
    both components of relative velocity, so nothing in the world or the likelihood
    projects one out. Kept because the closing rate is the quantity a time-to-collision or
    a gap-acceptance rule is written in, and deriving it from a detection row is a
    one-liner nobody should write twice.

    Args:
        positions: ``(K, 2)`` relative positions in the ego body frame, in metres.
        velocities: ``(K, 2)`` relative velocities in the same frame, in m/s.

    Returns:
        ``(K,)`` signed closing rates in m/s, negative when the range is shrinking. Zero
        for a row sitting exactly on the ego, where the line of sight is undefined.
    """
    offsets = np.asarray(positions, dtype=float)
    ranges = np.linalg.norm(offsets, axis=1)
    safe = np.where(ranges > 0.0, ranges, 1.0)
    line_of_sight = offsets / safe[:, None]
    projected = np.sum(np.asarray(velocities, dtype=float) * line_of_sight, axis=1)
    return np.where(ranges > 0.0, projected, 0.0)


def detection_visibility(
    positions: np.ndarray,
    present: np.ndarray,
    max_range_m: float,
    blocker_half_width_m: float = DEFAULT_BLOCKER_HALF_WIDTH_M,
) -> np.ndarray:
    """Which vehicles the sensor can actually see: in range and not behind another one.

    Occlusion is deterministic and geometric. Every vehicle is treated as a disc of radius
    ``blocker_half_width_m``, which subtends a half-angle of ``arcsin(w / r)`` at range
    ``r``; a vehicle is masked when a **closer** one lies within that half-angle of its own
    line of sight. Discs and not the oriented rectangles highway-env collides with, because
    the state carries no heading for the other vehicles — a rectangle would need a quantity
    neither the world's state nor the planner's model represents. The disc is the inscribed
    approximation, so it under-occludes a car seen broadside, which leaves the planner
    seeing slightly more than it should rather than less.

    Blockers are counted whether or not they are themselves in range, because a car at 45 m
    genuinely hides what is behind it; the thing behind it is out of range anyway.

    Both the world and the planner's model call this one function, on the true vehicle list
    and on a particle's agent slots respectively, so the model's prediction of *whether* it
    should have seen something is the same rule the world applied.

    Args:
        positions: ``(K, 2)`` relative positions in the ego body frame, in metres.
        present: ``(K,)`` mask of which rows hold a vehicle at all.
        max_range_m: Range beyond which a vehicle is absent from the reading.
        blocker_half_width_m: Half-width of a blocking vehicle in metres. Defaults to 1.0,
            a 2 m-wide car.

    Returns:
        ``(K,)`` boolean mask of the rows the sensor reports.
    """
    offsets = np.asarray(positions, dtype=float)
    here = np.asarray(present, dtype=bool)
    ranges = np.linalg.norm(offsets, axis=1)
    bearings = np.arctan2(offsets[:, 1], offsets[:, 0])
    # arcsin of a clipped ratio: a blocker closer than its own half-width would otherwise
    # leave the domain, and the clip turns that into "everything behind it is hidden".
    half_width = np.arcsin(np.clip(blocker_half_width_m / np.maximum(ranges, 1e-9), -1.0, 1.0))
    separation = np.abs(_wrap_to_pi_array(bearings[:, None] - bearings[None, :]))
    blocked_by = (
        here[None, :] & (ranges[None, :] < ranges[:, None]) & (separation < half_width[None, :])
    )
    return here & (ranges <= max_range_m) & ~np.any(blocked_by, axis=1)


def _wrap_to_pi_array(angles: np.ndarray) -> np.ndarray:
    """Array form of :func:`wrap_to_pi`, for the geometry helpers above."""
    return (np.asarray(angles, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def pomdp_observation_width(max_detections: int, lookahead_count: int) -> int:
    """Width of one flattened POMDP observation.

    The flat layout is ``[ego_pose(4) | ego_speed(1) | lane_pose(2) | curvature_ahead(L) |
    detections(K * 5)]``, which is what the torch model's ``[N, do]`` tensors hold and what
    the parity test unflattens back into the scalar model's dictionary. Declared here rather
    than in either model so the two cannot disagree about where a channel starts.

    Args:
        max_detections: Number of detection slots the reading carries.
        lookahead_count: Number of curvature-ahead samples.

    Returns:
        The flattened width in entries.
    """
    return (
        OBSERVED_EGO_POSE_WIDTH
        + OBSERVED_EGO_SPEED_WIDTH
        + OBSERVED_LANE_POSE_WIDTH
        + lookahead_count
        + max_detections * DETECTION_SLOT_WIDTH
    )


def racetrack_reward(
    lateral: float,
    action: Tuple[float, float],
    crashed: bool,
    on_road: bool,
    *,
    collision_reward: float = DEFAULT_COLLISION_REWARD,
    lane_centering_cost: float = DEFAULT_LANE_CENTERING_COST,
    lane_centering_reward: float = DEFAULT_LANE_CENTERING_REWARD,
    action_reward: float = DEFAULT_ACTION_REWARD,
) -> float:
    """Score a racetrack transition, reproducing highway-env's own reward.

    Both the world and the planner's model call this one function, so the planner can
    never be optimising a different objective than the one being scored. It reproduces
    ``RacetrackEnv._reward`` in closed form; a test pins the two together against a live
    simulator, which is what makes that a fact rather than an intention.

    Args:
        lateral: Signed lateral offset from the lane centreline, in metres.
        action: The normalised ``(acceleration, steering)`` command actually applied.
        crashed: Whether the ego collided on this step.
        on_road: Whether the ego is still on the road.
        collision_reward: Weight applied to a collision. Defaults to -1.0.
        lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
        lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
        action_reward: Weight on the control-effort penalty. Defaults to -0.3.

    Returns:
        The scalar reward, zero whenever the ego is off the road.

    Note:
        Two details of the upstream formula are reproduced rather than tidied: the
        normalisation maps from ``[collision_reward, 1]`` using the literal ``1`` and not
        ``lane_centering_reward``, and it does not clip, so unusual weights can push the
        result outside ``[0, 1]``.
    """
    centering = lane_centering_reward / (1.0 + lane_centering_cost * float(lateral) ** 2)
    effort = float(np.linalg.norm(np.asarray(action, dtype=float)))
    raw = centering + action_reward * effort + collision_reward * float(crashed)
    scaled = (raw - collision_reward) / (1.0 - collision_reward)
    return float(scaled * float(on_road))


def _ego_kinematics_config() -> Dict[str, Any]:
    """The block the ego's speedometer is read out of; see the note beside its constants.

    ``absolute`` and ``normalize`` are pinned because the defaults are relative and
    normalised, and a normalised row cannot be turned back into metres per second.
    ``order`` is ``"sorted"`` for the same reason the MDP arm pins it: ``"shuffled"`` draws
    from the environment's generator, which would make the two arms consume different
    randomness and break the shared-dynamics guarantee.
    """
    return {
        "type": "Kinematics",
        "features": list(EGO_KINEMATICS_FEATURES),
        "vehicles_count": EGO_KINEMATICS_VEHICLES_COUNT,
        "absolute": True,
        "normalize": False,
        "order": "sorted",
        "see_behind": True,
    }


def _pomdp_observation_config() -> Dict[str, Any]:
    """The ego's own kinematics, which is all the POMDP arm asks the simulator for.

    Everything else the POMDP arm emits — the ego pose, the lane camera, the
    curvature-ahead channel and the detections — is measured off the ego vehicle, the road
    network and the vehicle list by the world adapter and corrupted there, because
    highway-env has no observation type that reports arclength, occlusion or range gating.
    This block exists only so the ego's own speed arrives in metres per second rather than
    being read off the vehicle object twice.
    """
    return _ego_kinematics_config()


def ego_speed_from_kinematics_row(row: Any) -> float:
    """Reduce one ``[vx, vy, cos_h, sin_h]`` row to the ego's signed scalar speed.

    Scalar speed, not the ``(vx, vy)`` pair, because the pair also reveals the ego's
    heading through ``atan2(vy, vx)`` and the ego-aligned occupancy grid deliberately
    withholds heading. A signed scalar is what a speedometer reads and what the
    ``EGO_SPEED`` state slot holds, so it maps one-to-one onto that slot and adds exactly
    one number to the observation.

    The projection onto the heading, rather than ``hypot(vx, vy)``, because the speed is
    **signed** and the norm is not: braking through zero takes the racetrack ego to -30 m/s,
    where ``hypot`` reports +30 and this returns -30. ``velocity == speed * (cos_h, sin_h)``
    with a unit heading vector, so the projection recovers the speed exactly.

    Args:
        row: A four-element ``[vx, vy, cos_h, sin_h]`` row in world-frame metres per second.

    Returns:
        The signed scalar speed in metres per second.

    Raises:
        ValueError: If ``row`` does not hold exactly four values.
    """
    values = np.asarray(row, dtype=float).reshape(-1)
    if values.size != len(EGO_KINEMATICS_FEATURES):
        raise ValueError(
            f"Expected a {len(EGO_KINEMATICS_FEATURES)}-element "
            f"{list(EGO_KINEMATICS_FEATURES)} row, got {values.size} values."
        )
    velocity_x, velocity_y, cos_heading, sin_heading = values
    return float(velocity_x * cos_heading + velocity_y * sin_heading)


def _mdp_observation_config(max_tracked_agents: int) -> Dict[str, Any]:
    """Absolute kinematics for the ego and the nearest ``max_tracked_agents`` vehicles.

    ``absolute`` and ``normalize`` are set explicitly because highway-env defaults to
    relative, normalised rows, which is not the "absolute x, y, vx, vy" baseline this
    comparison needs. ``order`` is pinned to ``"sorted"`` for a second reason beyond
    determinism: ``"shuffled"`` draws from the environment's random generator, which
    would make the two arms consume different randomness and break the shared-dynamics
    guarantee.
    """
    return {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "vehicles_count": max_tracked_agents + 1,
        "absolute": True,
        "normalize": False,
        "order": "sorted",
        "see_behind": True,
    }


def observation_config(mode: ObservationMode, max_tracked_agents: int) -> Dict[str, Any]:
    """Build the highway-env observation block for one arm of the matched pair.

    Args:
        mode: Which arm to build.
        max_tracked_agents: Number of other vehicles the MDP arm reports.

    Returns:
        The ``"observation"`` sub-dictionary for a highway-env config.
    """
    if mode is ObservationMode.POMDP:
        return _pomdp_observation_config()
    return _mdp_observation_config(max_tracked_agents)


def build_racetrack_config(
    mode: ObservationMode,
    *,
    max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
    other_vehicles: int = DEFAULT_OTHER_VEHICLES,
    duration: int = DEFAULT_DURATION,
    policy_frequency: int = DEFAULT_POLICY_FREQUENCY,
    simulation_frequency: int = DEFAULT_SIMULATION_FREQUENCY,
    collision_reward: float = DEFAULT_COLLISION_REWARD,
    lane_centering_cost: float = DEFAULT_LANE_CENTERING_COST,
    lane_centering_reward: float = DEFAULT_LANE_CENTERING_REWARD,
    action_reward: float = DEFAULT_ACTION_REWARD,
    speed_limit: float = DEFAULT_SPEED_LIMIT,
    terminate_off_road: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the highway-env configuration for one arm of the matched pair.

    Every key except ``"observation"`` is built from the same arguments regardless of
    ``mode``, which is what makes the two arms a controlled comparison. ``overrides`` is
    applied last and is deliberately shared by both arms too — an override that differed
    between them would silently break the guarantee.

    Args:
        mode: Which observation the environment should emit.
        max_tracked_agents: Fixed number of agent slots in the state and the MDP
            observation. Defaults to 4.
        other_vehicles: Extra vehicles beyond the one the racetrack always spawns.
            Defaults to 1.
        duration: Episode length limit in simulator time. Defaults to 300.
        policy_frequency: Decisions per second. Defaults to 5.
        simulation_frequency: Physics steps per second. Defaults to 15.
        collision_reward: Weight applied to a collision. Defaults to -1.0.
        lane_centering_cost: Sharpness of the lane-centering falloff. Defaults to 4.0.
        lane_centering_reward: Weight on the lane-centering term. Defaults to 1.0.
        action_reward: Weight on the control-effort penalty. Defaults to -0.3.
        speed_limit: Track speed limit in m/s. Defaults to 10.0.
        terminate_off_road: Whether leaving the road ends the episode. Defaults to True.
        overrides: Extra highway-env keys merged in last. Defaults to None.

    Returns:
        A configuration dictionary ready for ``gymnasium.make(..., config=...)``.

    Raises:
        ValueError: If ``overrides`` tries to set the observation, which would defeat the
            matched pair, or if the substep ratio is not a positive integer.
    """
    _require_integral_substeps(simulation_frequency, policy_frequency)
    config: Dict[str, Any] = {
        # Longitudinal control is enabled on purpose; see the module docstring.
        "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
        "simulation_frequency": simulation_frequency,
        "policy_frequency": policy_frequency,
        "duration": duration,
        "collision_reward": collision_reward,
        "lane_centering_cost": lane_centering_cost,
        "lane_centering_reward": lane_centering_reward,
        "action_reward": action_reward,
        "controlled_vehicles": 1,
        "other_vehicles": other_vehicles,
        "speed_limit": speed_limit,
        "terminate_off_road": terminate_off_road,
    }
    if overrides:
        if _CONFIG_OBSERVATION_KEY in overrides:
            raise ValueError(
                "Overriding 'observation' would break the matched MDP/POMDP pair, whose "
                "whole purpose is that the two arms differ in that key alone. Select an "
                "arm with ObservationMode instead."
            )
        config.update(overrides)
    config[_CONFIG_OBSERVATION_KEY] = observation_config(mode, max_tracked_agents)
    return config


def _require_integral_substeps(simulation_frequency: int, policy_frequency: int) -> None:
    """Reject a step-rate pair the planner's model could not reproduce exactly."""
    if policy_frequency <= 0 or simulation_frequency <= 0:
        raise ValueError(
            f"simulation_frequency and policy_frequency must be positive, got "
            f"{simulation_frequency} and {policy_frequency}."
        )
    if simulation_frequency % policy_frequency != 0:
        raise ValueError(
            f"simulation_frequency ({simulation_frequency}) must be an integer multiple "
            f"of policy_frequency ({policy_frequency}); otherwise the planner's model "
            f"integrates a different number of substeps than the world and the two "
            f"diverge silently."
        )
