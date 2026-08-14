# SPDX-License-Identifier: MIT

"""Tests for the world-side sensor suite the racetrack POMDP emits its reading from.

These cover :mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_world_sensors`:
the ego-pose channel, the lane camera, the curvature-ahead channel, the radar with its
range gate and its occlusion rule, and the body-frame transform they all sit on. They used
to live in ``test_racetrack_pomdp.py`` and moved here with the code they exercise; what
stayed behind is what genuinely needs a live simulator -- episode stepping, termination,
metrics, pickling, and the matched pair of configs.

Every test drives ``WorldSensors`` with a ``StandInVehicle`` rather than booting
highway-env. That is not a shortcut around the real thing: ``WorldSensors`` never imports
the simulator and reads only ``position``, ``velocity``, ``heading`` and ``lane_offset``
off whatever it is handed, so a stand-in is the *whole* input it has. It is also sharper,
because occlusion, the range gate and the ordering are properties of a chosen configuration
of vehicles, and a coasting rollout visits whichever ones it happens to visit. The stand-in
is a real dataclass and not a mock: a mock would answer any attribute at all, so a test
against one would keep passing after the sensors started reading a field that does not
exist.

What the sensors cannot be asked here is whether the world wires the *right* vehicle, road
and arclength into them. That is the live suite's job, and it pins it there.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    DEFAULT_BLOCKER_HALF_WIDTH_M,
    DEFAULT_CURVATURE_LOOKAHEAD_M,
    DEFAULT_MAX_TRACKED_AGENTS,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_X,
    DETECTION_SLOT_WIDTH,
    EGO_POSE_ARCLENGTH,
    EGO_POSE_HEADING,
    EGO_POSE_X,
    EGO_POSE_Y,
    LANE_POSE_ANG,
    LANE_POSE_LAT,
    OBSERVED_EGO_POSE_WIDTH,
    OBSERVED_EGO_SPEED_WIDTH,
    OBSERVED_LANE_POSE_WIDTH,
    RacetrackObservation,
    radial_velocities,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry
from POMDPPlanners.environments.racetrack_pomdp.racetrack_world_sensors import (
    SensorConfig,
    WorldSensors,
    relative_vehicles,
)

# A gate wide enough that nothing any test here places is ever range-gated out, so a test
# about some other part of the radar is not silently testing the gate as well.
_UNLIMITED_RANGE_M = 1000.0

# The far end of the range dial: with this gate every vehicle on any circuit is reported,
# which is the limit in which this arm's reading becomes the state to within its widths.
_OPEN_RANGE_M = 1e9

# Draws per noise test. Enough that a sample standard deviation lands within 20% of the
# width it was drawn at (its own relative error is 1/sqrt(2N), about 3.5% here), so the
# tolerances below fail on a missing or mis-scaled corruption rather than on sampling luck.
_NOISE_DRAWS = 400


@dataclass
class StandInVehicle:
    """A vehicle-shaped object carrying exactly what ``WorldSensors`` reads off one.

    Attributes:
        position: World-frame ``(x, y)`` position in metres.
        velocity: World-frame ``(vx, vy)`` velocity in m/s.
        heading: World-frame heading in radians.
        lane_offset: highway-env's own triple of longitudinal offset along the lane,
            signed lateral offset from its centreline, and heading relative to it.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    lane_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _vehicle(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    heading: float = 0.0,
    lane_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> StandInVehicle:
    """A stand-in vehicle at a world-frame pose, written one scalar at a time."""
    return StandInVehicle(
        position=np.array([x, y], dtype=float),
        velocity=np.array([vx, vy], dtype=float),
        heading=heading,
        lane_offset=np.asarray(lane_offset, dtype=float),
    )


def _sensor(**overrides: Any) -> SensorConfig:
    """A SensorConfig with every width at zero and no gate worth speaking of.

    A sensor test either asks what a channel measures or how wrong it is, and the two need
    opposite settings: the first needs the noise gone so a mismatch is a wrong quantity
    rather than a draw, the second names its own widths. This builds the first and lets a
    caller override its way to the second.
    """
    widths: Dict[str, Any] = {
        "ego_position_std_m": 0.0,
        "ego_heading_std_rad": 0.0,
        "ego_arclength_std_m": 0.0,
        "lane_lateral_std_m": 0.0,
        "lane_heading_std_rad": 0.0,
        "curvature_lookahead_m": DEFAULT_CURVATURE_LOOKAHEAD_M,
        "curvature_std_1pm": 0.0,
        "max_detection_range_m": _UNLIMITED_RANGE_M,
        "detection_position_std_m": 0.0,
        "detection_velocity_std": 0.0,
        "blocker_half_width_m": DEFAULT_BLOCKER_HALF_WIDTH_M,
    }
    widths.update(overrides)
    return SensorConfig(**widths)


def _geometry() -> TrackGeometry:
    """A three-segment lap: a straight, a left arc, then a tighter right one.

    The curvatures are distinct and the segments long enough that a lookahead list can put
    each of its entries in a different one, which is what makes "entry i is read at
    ``arclength + d_i``" testable rather than merely plausible.
    """
    return TrackGeometry(
        segment_starts=np.array([0.0, 100.0, 160.0]),
        segment_curvatures=np.array([0.0, 0.02, -0.05]),
        total_length_m=240.0,
    )


def _read(
    sensor: SensorConfig,
    ego: StandInVehicle,
    others: Sequence[StandInVehicle] = (),
    *,
    arclength: float = 0.0,
    ego_speed: float = 10.0,
    max_detections: int = DEFAULT_MAX_TRACKED_AGENTS,
) -> RacetrackObservation:
    """One reading of a hand-built scene, through the sensors' own public entry point."""
    sensors = WorldSensors(sensor, max_detections)
    return sensors.read(
        ego_speed=ego_speed,
        ego=ego,
        arclength=arclength,
        geometry=_geometry(),
        others=list(others),
    )


def _reported_rows(detections: np.ndarray) -> np.ndarray:
    """The occupied rows of a detection block, in the order the sensor emitted them."""
    return detections[detections[:, DETECTION_PRESENT] > 0.5]


def _ahead_of(
    ego: StandInVehicle, forward_m: float, left_m: float, **motion: float
) -> StandInVehicle:
    """A vehicle placed at a body-frame offset from an ego whose heading is zero.

    Only valid for a zero-heading ego, which every scene that cares about the *arrangement*
    of vehicles uses: the body frame is then the world frame shifted, so the offsets in the
    test read as the geometry the test is about. The rotation itself is pinned separately.
    """
    assert ego.heading == 0.0, "body-frame placement assumes a zero-heading ego"
    return _vehicle(
        x=float(ego.position[0]) + forward_m,
        y=float(ego.position[1]) + left_m,
        vx=float(ego.velocity[0]) + motion.get("closing_mps", 0.0),
        vy=float(ego.velocity[1]) + motion.get("crossing_mps", 0.0),
    )


# ── The reading's surface ───────────────────────────────────────────────


def test_read_emits_the_five_documented_channels_at_their_promised_widths() -> None:
    """One read produces the five channels the observation declares, all float32.

    Purpose: Validates the reading's structure at its source. The planner's model
        unflattens this reading by position, so a channel of the wrong width silently
        shifts every later channel into the wrong slot -- and the speedometer is checked
        here too, because it is the one number the sensors pass straight through and a
        channel that re-derived it would be measuring rather than reporting.

    Given: A zero-noise sensor suite, an ego with a stand-in pose and one other vehicle
    When: A reading is taken
    Then: It is a five-channel RacetrackObservation at the documented widths, every channel
        float32, and its ego_speed is the speed it was handed

    Test type: unit
    """
    ego = _vehicle(x=12.0, y=-3.0, heading=0.4)

    reading = _read(_sensor(), ego, [_vehicle(x=22.0, y=-3.0)], ego_speed=7.25)

    assert isinstance(reading, RacetrackObservation)
    assert len(reading) == 5
    assert reading.ego_pose.shape == (OBSERVED_EGO_POSE_WIDTH,)
    assert reading.ego_speed.shape == (OBSERVED_EGO_SPEED_WIDTH,)
    assert reading.lane_pose.shape == (OBSERVED_LANE_POSE_WIDTH,)
    assert reading.curvature_ahead.shape == (len(DEFAULT_CURVATURE_LOOKAHEAD_M),)
    assert reading.detections.shape == (DEFAULT_MAX_TRACKED_AGENTS, DETECTION_SLOT_WIDTH)
    assert all(channel.dtype == np.float32 for channel in reading)
    assert float(reading.ego_speed[0]) == pytest.approx(7.25)


# ── The ego pose channel ────────────────────────────────────────────────


def test_the_zero_noise_ego_pose_is_the_ego_s_own_position_heading_and_arclength() -> None:
    """Noiseless, the pose channel is the four numbers it was handed, in order.

    Purpose: Pins what the channel *is*. This arm withholds vehicles and nothing else, so
        the pose has to be the ego's real one; a channel reporting a lane-relative pose, a
        body-frame zero, or the arclength and heading transposed would still look like a
        plausible four-vector and would still have the right spread under the noise test
        below.

    Given: A zero-width sensor suite and an ego at a known world pose, read at a known
        arclength
    When: A reading is taken
    Then: The four entries are that position, that heading and that arclength

    Test type: unit
    """
    ego = _vehicle(x=31.5, y=-8.25, heading=0.6)

    pose = _read(_sensor(), ego, arclength=142.0).ego_pose

    assert float(pose[EGO_POSE_X]) == pytest.approx(31.5, abs=1e-5)
    assert float(pose[EGO_POSE_Y]) == pytest.approx(-8.25, abs=1e-5)
    assert float(pose[EGO_POSE_HEADING]) == pytest.approx(0.6, abs=1e-5)
    assert float(pose[EGO_POSE_ARCLENGTH]) == pytest.approx(142.0, abs=1e-4)


def test_the_ego_pose_channel_is_noisy_at_the_widths_it_was_configured_with() -> None:
    """Position, heading and arclength are each corrupted at their own configured width.

    Purpose: The pose is near-exact by design, but near-exact is not exact: a zero-width
        channel is a delta in the planner's likelihood, and the first particle whose dead
        reckoning misses by a hair would be annihilated. The three widths are separate
        because the hardware is -- GPS, an IMU and a wheel odometer -- and the model scores
        a residual against each, so emitting them at one width would make it confidently
        wrong about how much disagreement is normal.

    Given: A sensor suite with deliberately distinguishable widths of 0.5 m, 0.05 rad and
        2.0 m, and one fixed ego pose read many times
    When: The residuals against that pose are collected
    Then: No reading is exact, each residual is centred on zero, and the four spreads sit
        near their own configured widths

    Test type: unit
    """
    np.random.seed(31)
    position_std, heading_std, arclength_std = 0.5, 0.05, 2.0
    sensor = _sensor(
        ego_position_std_m=position_std,
        ego_heading_std_rad=heading_std,
        ego_arclength_std_m=arclength_std,
    )
    ego = _vehicle(x=4.0, y=9.0, heading=0.2)
    truth = np.array([4.0, 9.0, 0.2, 60.0])

    poses = np.asarray(
        [_read(sensor, ego, arclength=60.0).ego_pose for _ in range(_NOISE_DRAWS)], dtype=float
    )

    errors = poses - truth
    assert not np.any(errors == 0.0), "an exact reading means the localisation noise is missing"
    assert np.all(np.abs(errors.mean(axis=0)) < 0.5 * errors.std(axis=0))
    assert float(errors[:, EGO_POSE_X].std()) == pytest.approx(position_std, rel=0.2)
    assert float(errors[:, EGO_POSE_Y].std()) == pytest.approx(position_std, rel=0.2)
    assert float(errors[:, EGO_POSE_HEADING].std()) == pytest.approx(heading_std, rel=0.2)
    assert float(errors[:, EGO_POSE_ARCLENGTH].std()) == pytest.approx(arclength_std, rel=0.2)


def test_a_measured_heading_near_pi_is_wrapped_rather_than_reported_past_it() -> None:
    """Noise that pushes the heading past pi comes back at the other end of the range.

    Purpose: The wrap has to happen *after* the corruption, and only a heading already near
        the boundary can tell. A car pointing just short of pi that reads as pointing just
        past it is, to any consumer differencing two angles, a car that has spun through
        half a turn between one step and the next -- and the planner's likelihood scores
        exactly that difference.

    Given: An ego heading 3.10 rad, a hair under pi, measured many times at 0.25 rad
    When: The reported headings are collected
    Then: Every one lies in the wrapped range, and some sit just above -pi rather than just
        above pi, which is where an unwrapped channel would have put them

    Test type: unit
    """
    np.random.seed(5)
    sensor = _sensor(ego_heading_std_rad=0.25)
    ego = _vehicle(heading=3.10)

    headings = np.asarray(
        [float(_read(sensor, ego).ego_pose[EGO_POSE_HEADING]) for _ in range(_NOISE_DRAWS)]
    )

    # float32 rounding can nudge a value a hair past pi, which is a cast and not a missing
    # wrap; an unwrapped reading would sit whole tenths of a radian beyond it.
    assert np.all(headings <= np.pi + 1e-6)
    assert np.all(headings >= -np.pi - 1e-6)
    assert np.any(headings < -3.0), "no draw crossed pi, so the wrap went untested"


# ── The lane camera ─────────────────────────────────────────────────────


def test_the_lane_camera_is_noisy_at_the_widths_it_was_configured_with() -> None:
    """The lane pose is the true lane offset, corrupted by the amount it was told to be.

    Purpose: highway-env's ``lane_offset`` is exact. Emitting it unchanged would hand the
        planner a lane-relative pose no camera delivers, which is smuggling ground truth
        into the arm this environment exists to compare against the fully-observed one. The
        residuals are checked for their centre as well as their spread, so a channel that
        reported the wrong entries of the triple -- the longitudinal offset for the lateral
        one, say -- fails here rather than passing as noise.

    Given: A camera at 0.2 m and 0.08 rad, well above the defaults, reading one fixed lane
        offset many times
    When: The residuals against that offset are collected
    Then: No reading is exact, both residuals are centred on zero, and both spreads sit
        near their own configured widths

    Test type: unit
    """
    np.random.seed(11)
    lateral_std, heading_std = 0.2, 0.08
    sensor = _sensor(lane_lateral_std_m=lateral_std, lane_heading_std_rad=heading_std)
    ego = _vehicle(lane_offset=(37.0, -1.25, 0.09))
    truth = np.array([-1.25, 0.09])

    poses = np.asarray([_read(sensor, ego).lane_pose for _ in range(_NOISE_DRAWS)], dtype=float)

    errors = poses - truth
    assert not np.any(errors == 0.0), "an exact reading means the camera noise is missing"
    assert np.all(np.abs(errors.mean(axis=0)) < 0.5 * errors.std(axis=0))
    assert float(errors[:, LANE_POSE_LAT].std()) == pytest.approx(lateral_std, rel=0.2)
    assert float(errors[:, LANE_POSE_ANG].std()) == pytest.approx(heading_std, rel=0.2)


# ── Curvature ahead ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lookaheads", [(5.0,), (5.0, 20.0), (5.0, 20.0, 80.0, 120.0), DEFAULT_CURVATURE_LOOKAHEAD_M]
)
def test_curvature_ahead_reports_the_profile_at_the_arclength_plus_each_lookahead(
    lookaheads: Tuple[float, ...],
) -> None:
    """One entry per configured distance, each read at its own distance ahead.

    Purpose: The planner's model scores one Gaussian per entry, so it has to be built with
        the same distances in the same order. A channel whose width did not follow the
        configuration, or whose entries were read at a common distance or in reverse, would
        line each residual up against the wrong distance -- and the mismatch shows up as a
        filter that is confidently wrong rather than as an error.

    Given: Zero-noise cameras built with lookahead lists of several lengths, read at an
        arclength where the lookaheads straddle three segments of different curvature
    When: A reading is taken
    Then: The channel has one entry per distance, each equal to the track's curvature there

    Test type: unit
    """
    sensor = _sensor(curvature_lookahead_m=lookaheads)

    curvature = _read(sensor, _vehicle(), arclength=90.0).curvature_ahead

    expected = _geometry().curvature_at(90.0 + np.asarray(lookaheads))
    assert curvature.shape == (len(lookaheads),)
    assert curvature == pytest.approx(expected, abs=1e-6)


def test_the_curvature_channel_is_noisy_at_the_width_it_was_configured_with() -> None:
    """The camera's curvature reading is corrupted, by the amount it was told to be.

    Purpose: A mapless planner reading this channel is estimating the road rather than
        being told it, and that is only true if the reading is wrong. Emitting the true
        curvature would hand a POMDP planner a piece of ground truth the fully-observed arm
        does not even carry in its observation, which would invert the comparison.

    Given: A camera at 0.02 1/m, an order above the default, reading one fixed arclength
        many times
    When: The residuals against the track's own curvature there are collected
    Then: No reading is exact and the residuals' spread sits near the configured width

    Test type: unit
    """
    np.random.seed(23)
    curvature_std = 0.02
    sensor = _sensor(curvature_std_1pm=curvature_std)

    readings = np.asarray(
        [_read(sensor, _vehicle(), arclength=90.0).curvature_ahead for _ in range(_NOISE_DRAWS)],
        dtype=float,
    )

    errors = readings - _geometry().curvature_at(90.0 + np.asarray(DEFAULT_CURVATURE_LOOKAHEAD_M))
    assert not np.any(errors == 0.0), "an exact reading means the camera noise is missing"
    assert float(errors.mean()) == pytest.approx(0.0, abs=0.2 * curvature_std)
    assert float(errors.std()) == pytest.approx(curvature_std, rel=0.2)


# ── The radar: what a detection carries ─────────────────────────────────


def test_zero_noise_detections_report_the_true_relative_position_and_velocity() -> None:
    """Noiseless, a row is a real vehicle's whole relative kinematic row.

    Purpose: Pins what the four measured numbers of a detection row *are*. A row reporting
        a world-frame offset, an absolute velocity, or a projection of one would still look
        plausible in a rollout and would still have the right spread under the noise test;
        only a comparison against the true relative geometry catches it.

    Given: A zero-noise radar, a moving ego and two other vehicles at known offsets and
        velocities
    When: A reading is taken
    Then: Each reported row holds that vehicle's offset from the ego and its velocity
        relative to the ego's, both in the body frame

    Test type: unit
    """
    ego = _vehicle(vx=10.0, vy=0.0)
    near = _ahead_of(ego, 12.0, 3.0, closing_mps=-4.0, crossing_mps=1.5)
    far = _ahead_of(ego, 40.0, -6.0, closing_mps=2.0, crossing_mps=-3.0)

    rows = _reported_rows(_read(_sensor(), ego, [near, far]).detections)

    assert len(rows) == 2
    assert rows[0, DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx([12.0, 3.0], abs=1e-4)
    assert rows[0, DETECTION_REL_VX : DETECTION_REL_VX + 2] == pytest.approx([-4.0, 1.5], abs=1e-4)
    assert rows[1, DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx([40.0, -6.0], abs=1e-4)
    assert rows[1, DETECTION_REL_VX : DETECTION_REL_VX + 2] == pytest.approx([2.0, -3.0], abs=1e-4)


def test_a_vehicle_crossing_the_ego_s_path_abeam_is_reported_with_its_real_crossing_rate() -> None:
    """A car moving straight across the line of sight is reported as moving, not as still.

    Purpose: This is what the row gained when the closing rate became a full relative
        velocity. A vehicle crossing the ego's path directly ahead has *no* component along
        the line of sight, so the old radial-only reading described it as a stationary
        obstacle -- the one geometry where the missing component mattered most, since it is
        the car about to be in front of the ego. Inferring it back was a different
        estimation problem from the one this environment poses, so the sensor reports it.

    Given: A zero-noise radar and one vehicle 15 m directly ahead moving purely sideways
    When: A reading is taken
    Then: The row carries the true crossing rate, while the projection onto the line of
        sight -- what the old channel would have reported -- is zero

    Test type: unit
    """
    ego = _vehicle(vx=10.0)
    crossing = _ahead_of(ego, 15.0, 0.0, crossing_mps=6.0)

    row = _reported_rows(_read(_sensor(), ego, [crossing]).detections)[0]

    offset = np.asarray(row[DETECTION_REL_X : DETECTION_REL_X + 2], dtype=float).reshape(1, 2)
    velocity = np.asarray(row[DETECTION_REL_VX : DETECTION_REL_VX + 2], dtype=float).reshape(1, 2)
    assert offset[0] == pytest.approx([15.0, 0.0], abs=1e-4)
    assert velocity[0] == pytest.approx([0.0, 6.0], abs=1e-4)
    assert float(radial_velocities(offset, velocity)[0]) == pytest.approx(0.0, abs=1e-4)


def test_no_other_vehicles_leaves_every_detection_slot_empty() -> None:
    """An empty road produces a correctly shaped block of zeros, not a ragged one.

    Purpose: The reading is fixed-width, and an empty road is the case where a shortcut is
        tempting. A block of the wrong shape would break the flattening the planner's model
        does; a block of anything but zeros would be read as phantom traffic, which the
        likelihood then charges every particle for.

    Given: A radar with no other vehicles on the road
    When: A reading is taken
    Then: Every slot is zero and the block still has one row per tracked agent

    Test type: unit
    """
    detections = _read(_sensor(), _vehicle(), []).detections

    assert detections.shape == (DEFAULT_MAX_TRACKED_AGENTS, DETECTION_SLOT_WIDTH)
    assert not np.any(detections)


# ── The radar: the range dial ───────────────────────────────────────────


def test_a_vehicle_beyond_the_detection_range_produces_no_row_at_all() -> None:
    """Turn the dial down and a distant car leaves the reading entirely.

    Purpose: Range gating is the dial this whole environment turns, and the near half of it
        is what makes the arm partially observed: a vehicle that genuinely exists produces
        no row, and the empty slot inside the gate is what the planner's filter has to
        reason from. A gate that did not follow the configuration would leave the model
        expecting reports the world never makes.

    Given: A radar gated at 25 m, with one vehicle 10 m ahead and another 60 m ahead
    When: A reading is taken
    Then: Only the near vehicle is reported, and the far one is nowhere in the block

    Test type: unit
    """
    ego = _vehicle()
    near, far = _ahead_of(ego, 10.0, 0.0), _ahead_of(ego, 60.0, 8.0)

    detections = _read(_sensor(max_detection_range_m=25.0), ego, [near, far]).detections

    rows = _reported_rows(detections)
    assert len(rows) == 1
    assert rows[0, DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx([10.0, 0.0], abs=1e-4)
    assert not np.any(np.isclose(detections[:, DETECTION_REL_X], 60.0, atol=1.0))


def test_an_open_detection_range_reports_every_vehicle_at_its_true_relative_state() -> None:
    """Turn the dial up and the reading is the traffic, exactly.

    Purpose: The far half of the same dial, and the claim the two arms are a continuum in
        it: as the range goes to infinity the POMDP reading becomes the state to within the
        sensor widths, so nothing is hidden but the widths. Zero widths here make that
        equality exact and the assertion tight -- with noise on, a gate that quietly
        dropped the furthest vehicle could hide inside the residuals.

    Given: A zero-noise radar with the gate opened to 1e9 m, four slots, and four vehicles
        spread from 10 m to 900 m on bearings far enough apart that none shadows another
    When: A reading is taken
    Then: All four are reported, each at its true relative offset and velocity

    Test type: unit
    """
    ego = _vehicle(vx=10.0, vy=-2.0)
    offsets = [(10.0, 0.0), (0.0, 60.0), (-300.0, 0.0), (0.0, -900.0)]
    motions = [(-3.0, 0.5), (1.0, -2.0), (5.0, 4.0), (-7.0, 1.0)]
    scene = [
        _ahead_of(ego, forward, left, closing_mps=closing, crossing_mps=crossing)
        for (forward, left), (closing, crossing) in zip(offsets, motions)
    ]

    detections = _read(_sensor(max_detection_range_m=_OPEN_RANGE_M), ego, scene).detections

    rows = _reported_rows(detections)
    assert len(rows) == len(scene)
    # The scene is written nearest first, which is the order the sensor reports in.
    for row, offset, motion in zip(rows, offsets, motions):
        assert row[DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx(offset, abs=1e-3)
        assert row[DETECTION_REL_VX : DETECTION_REL_VX + 2] == pytest.approx(motion, abs=1e-3)


# ── The radar: ordering, slots and occlusion ────────────────────────────


def test_detections_are_ordered_by_range_with_the_empty_slots_trailing() -> None:
    """Nearest first, then the rest, then zeros -- whatever order the vehicles arrived in.

    Purpose: The rows are unlabeled, so their order is the only structure a filter can rely
        on. It must be range and nothing else: an order that leaked the vehicle list's own
        order would hand the planner an identity across steps that a radar does not give,
        and a hole among the reported rows would make "row k is empty" stop meaning "there
        were fewer than k+1 returns".

    Given: A zero-noise radar with three vehicles at 32 m, 11 m and 20 m, in that order in
        the list handed to it
    When: A reading is taken
    Then: The rows come out nearest first and the unused slot is left at zero

    Test type: unit
    """
    ego = _vehicle()
    scene = [
        _ahead_of(ego, 30.0, -12.0),
        _ahead_of(ego, 10.0, 4.0),
        _ahead_of(ego, 20.0, -3.0),
    ]

    detections = _read(_sensor(), ego, scene).detections

    ranges = np.linalg.norm(detections[:, DETECTION_REL_X : DETECTION_REL_X + 2], axis=1)
    present = detections[:, DETECTION_PRESENT] > 0.5
    assert present.tolist() == [True, True, True, False]
    assert np.all(np.diff(ranges[present]) > 0.0)
    assert np.array_equal(detections[~present], np.zeros((1, DETECTION_SLOT_WIDTH)))


def test_more_vehicles_than_slots_drops_the_furthest() -> None:
    """When the returns outnumber the slots, it is the far ones that are lost.

    Purpose: The reading is a fixed-width block, so something has to go when more vehicles
        are visible than there are slots. Dropping the nearest -- or dropping by list order
        -- would delete exactly the vehicle the planner most needs, and the loss would be
        invisible because the block would still be full.

    Given: A zero-noise radar with a single detection slot and three vehicles at 11 m,
        20 m and 32 m
    When: A reading is taken
    Then: The single row holds the nearest of the three

    Test type: unit
    """
    ego = _vehicle()
    scene = [
        _ahead_of(ego, 30.0, -12.0),
        _ahead_of(ego, 10.0, 4.0),
        _ahead_of(ego, 20.0, -3.0),
    ]

    detections = _read(_sensor(), ego, scene, max_detections=1).detections

    assert detections.shape == (1, DETECTION_SLOT_WIDTH)
    assert float(detections[0, DETECTION_PRESENT]) == 1.0
    assert detections[0, DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx(
        [10.0, 4.0], abs=1e-3
    )


@pytest.mark.parametrize("far_lateral_m,expected_returns", [(0.0, 1), (10.0, 2)])
def test_a_vehicle_hidden_behind_a_closer_one_is_not_reported(
    far_lateral_m: float, expected_returns: int
) -> None:
    """A vehicle in another's shadow is absent; move it aside and it comes back.

    Purpose: Occlusion is the second of the two ways this world hides a vehicle that
        exists, and it is the one the planner's belief has to reason about, since a
        particle that insists on a vehicle the reading does not contain must not be
        penalised for it. The negative case is in the same test because "reported nothing"
        is also what a broken radar does.

    Given: A zero-noise radar with one vehicle 10 m directly ahead and a second 25 m ahead,
        either exactly behind it or 10 m to one side
    When: A reading is taken
    Then: The shadowed vehicle is missing and the one beside it is reported, and the near
        vehicle is reported either way

    Test type: unit
    """
    ego = _vehicle()
    scene = [_ahead_of(ego, 10.0, 0.0), _ahead_of(ego, 25.0, far_lateral_m)]

    detections = _read(_sensor(max_detection_range_m=100.0), ego, scene).detections

    assert int(np.sum(detections[:, DETECTION_PRESENT] > 0.5)) == expected_returns
    assert detections[0, DETECTION_REL_X : DETECTION_REL_X + 2] == pytest.approx(
        [10.0, 0.0], abs=1e-3
    )


@pytest.mark.parametrize("blocker_half_width_m,expected_returns", [(1.0, 2), (1.3, 1)])
def test_the_blocker_half_width_decides_where_the_shadow_ends(
    blocker_half_width_m: float, expected_returns: int
) -> None:
    """How wide a car is treated as being is what decides whether it hides the next one.

    Purpose: The shadow is ``arcsin(w / r)`` wide, so ``w`` is the whole occlusion model and
        the planner's belief must use the same number. This straddles the boundary with one
        fixed geometry: a vehicle at 25 m and 3 m to the side sits 0.119 rad off the bearing
        of one at 10 m, which a half-width of 1.0 m does not cover (0.100 rad) and 1.3 m does
        (0.130 rad). A model built with a different width would disagree with the world about
        exactly the marginal cases a filter is decided by.

    Given: A zero-noise radar with vehicles at 10 m ahead and at 25 m, 3 m to the side
    When: A reading is taken at each of two blocker half-widths
    Then: The far vehicle is reported at 1.0 m and hidden at 1.3 m

    Test type: unit
    """
    ego = _vehicle()
    scene = [_ahead_of(ego, 10.0, 0.0), _ahead_of(ego, 25.0, 3.0)]
    sensor = _sensor(max_detection_range_m=100.0, blocker_half_width_m=blocker_half_width_m)

    detections = _read(sensor, ego, scene).detections

    assert int(np.sum(detections[:, DETECTION_PRESENT] > 0.5)) == expected_returns


def test_the_radar_is_noisy_at_the_widths_it_was_configured_with() -> None:
    """Position and velocity are each corrupted at their own configured width.

    Purpose: The two widths are separate on purpose -- a velocity comes off a frequency
        shift and is measured far more tightly than a time of flight -- and the planner's
        model scores a residual against each. Emitting either exactly, or emitting both at
        one width, would make the model's likelihood confidently wrong about how much
        disagreement is normal. Both velocity components are checked, because the width now
        applies to a vector and a corruption that reached only the first would leave the
        crossing rate exact.

    Given: A radar at 2.0 m and 1.5 m/s, an order above the defaults, reading one fixed
        vehicle many times
    When: The residuals against its true offset and relative velocity are collected
    Then: All four residual spreads sit near their own configured widths

    Test type: unit
    """
    np.random.seed(7)
    position_std, velocity_std = 2.0, 1.5
    sensor = _sensor(detection_position_std_m=position_std, detection_velocity_std=velocity_std)
    ego = _vehicle(vx=10.0)
    other = _ahead_of(ego, 18.0, -2.0, closing_mps=-3.0, crossing_mps=4.0)
    truth = np.array([18.0, -2.0, -3.0, 4.0])

    rows = np.asarray(
        [_read(sensor, ego, [other]).detections[0, DETECTION_REL_X:] for _ in range(_NOISE_DRAWS)],
        dtype=float,
    )

    errors = rows - truth
    assert np.all(np.abs(errors.mean(axis=0)) < 0.5 * errors.std(axis=0))
    assert float(errors[:, 0].std()) == pytest.approx(position_std, rel=0.2)
    assert float(errors[:, 1].std()) == pytest.approx(position_std, rel=0.2)
    assert float(errors[:, 2].std()) == pytest.approx(velocity_std, rel=0.2)
    assert float(errors[:, 3].std()) == pytest.approx(velocity_std, rel=0.2)


# ── The body-frame transform underneath all of it ───────────────────────


def test_relative_vehicles_reports_offsets_and_velocities_in_the_ego_body_frame() -> None:
    """Both the offset and the relative velocity are rotated out of the world frame.

    Purpose: Every detection row and every agent slot in the state is built on this one
        transform, so a rotation applied to the position and not to the velocity -- or
        applied by ``+heading`` instead of ``-heading`` -- would put a car on the correct
        side of the ego while having it drive the wrong way. The expected values here are
        written out in trigonometry rather than taken from the same helper, so the test
        cannot agree with a wrong implementation.

    Given: An ego heading 0.7 rad and moving, and one other vehicle elsewhere and moving
    When: relative_vehicles is called
    Then: Both rows are the world-frame differences rotated by minus the ego's heading

    Test type: unit
    """
    heading = 0.7
    ego = _vehicle(x=5.0, y=-2.0, vx=8.0, vy=1.0, heading=heading)
    other = _vehicle(x=17.0, y=6.0, vx=-3.0, vy=4.0)

    positions, velocities = relative_vehicles(ego, [other])

    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    for reported, (dx, dy) in ((positions, (12.0, 8.0)), (velocities, (-11.0, 3.0))):
        assert reported.shape == (1, 2)
        assert reported[0] == pytest.approx(
            [cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy], abs=1e-9
        )


def test_relative_vehicles_with_no_other_vehicles_returns_two_empty_arrays() -> None:
    """An empty road returns empty ``(0, 2)`` blocks rather than raising or broadcasting.

    Purpose: The empty case is reached on every single-vehicle episode, and the shape it
        returns is what the detection path branches on. A ``(0,)`` array or a raise here
        would take out the whole reading rather than merely the detections.

    Given: An ego and no other vehicles
    When: relative_vehicles is called
    Then: Both returned arrays are empty and two-dimensional

    Test type: unit
    """
    positions, velocities = relative_vehicles(_vehicle(), [])

    assert positions.shape == (0, 2)
    assert velocities.shape == (0, 2)
