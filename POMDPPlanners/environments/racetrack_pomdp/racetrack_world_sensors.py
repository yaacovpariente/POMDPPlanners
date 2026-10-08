# SPDX-License-Identifier: MIT

"""What the racetrack world measures, before its reading leaves the simulator.

The world-side counterpart of
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model`: that module says
what a reading is *worth* to a particle, this one produces the reading in the first place.
Keeping them apart keeps the planner's model importable without highway-env, and keeping
this one out of
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp` keeps the adapter to the
episode-loop contract it implements.

Almost none of a POMDP reading comes from highway-env. The simulator supplies the ego's own
kinematics and nothing else — it has no observation type reporting arclength around a lap,
lane-relative pose, curvature ahead, a range gate or occlusion — so every other channel is
measured here off the live ego vehicle, the road network and the vehicle list, and corrupted
here at the configured widths.

**What is withheld is a whole vehicle, and nothing else.** A vehicle inside
``max_detection_range_m`` and not behind a closer one is reported in full: relative position
*and* both components of relative velocity. One that fails either test produces no row at
all. Everything about the ego is reported at near-exact widths. That is what makes the range
gate the single dial separating this arm from the fully-observed one.

The vehicle objects are typed :class:`~typing.Any` throughout, deliberately: highway-env
ships no stubs, and this module never imports it — it reads ``position``, ``velocity``,
``heading`` and ``lane_offset`` off whatever it is handed, which is also what lets the tests
drive it with a stand-in.

Classes:
    SensorConfig: Every width and limit the POMDP arm's sensors are simulated at.
    WorldSensors: Measures one POMDP reading off a live vehicle list and road.

Functions:
    relative_vehicles: Other vehicles' position and velocity in the ego body frame.
"""

from typing import Any, NamedTuple, Sequence, Tuple

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import pack_detections
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    DETECTION_SLOT_WIDTH,
    EGO_POSE_HEADING,
    RacetrackObservation,
    detection_visibility,
    rotate,
    wrap_to_pi,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry


class SensorConfig(NamedTuple):
    """Every width and limit the POMDP arm's sensors are simulated at.

    Gathered into one object rather than passed as eleven arguments because they travel
    together: the world configures them, the sensors measure with them, and the planner's
    model has to be built with the same numbers or its filter is confidently wrong.

    Attributes:
        ego_position_std_m: Localisation noise on the reported ``x`` and ``y``, in metres.
        ego_heading_std_rad: Localisation noise on the reported heading, in radians.
        ego_arclength_std_m: Odometry noise on the reported arclength, in metres.
        lane_lateral_std_m: Lane camera's lateral-offset noise, in metres.
        lane_heading_std_rad: Lane camera's heading noise, in radians.
        curvature_lookahead_m: Distances along the lane curvature is reported at.
        curvature_std_1pm: Lane camera's curvature noise, in 1/m.
        max_detection_range_m: Range beyond which no vehicle is reported at all.
        detection_position_std_m: Per-axis position noise on a detection, in metres.
        detection_velocity_std: Per-axis relative-velocity noise on a detection, in m/s.
        blocker_half_width_m: Half-width of an occluding vehicle, in metres.
    """

    ego_position_std_m: float
    ego_heading_std_rad: float
    ego_arclength_std_m: float
    lane_lateral_std_m: float
    lane_heading_std_rad: float
    curvature_lookahead_m: Tuple[float, ...]
    curvature_std_1pm: float
    max_detection_range_m: float
    detection_position_std_m: float
    detection_velocity_std: float
    blocker_half_width_m: float


def relative_vehicles(ego: Any, others: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Every other vehicle's position and velocity relative to the ego, in its body frame.

    Args:
        ego: The controlled vehicle.
        others: The vehicles to measure against it, with the ego already excluded.

    Returns:
        ``((V, 2)`` positions in metres, ``(V, 2)`` velocities in m/s``)``, both rotated
        into the ego body frame — ``x`` forward, ``y`` left. Two empty arrays when there
        are no other vehicles.
    """
    if not others:
        return np.zeros((0, 2)), np.zeros((0, 2))
    heading = float(ego.heading)
    positions = np.stack([np.asarray(v.position, dtype=float) for v in others]) - np.asarray(
        ego.position, dtype=float
    )
    velocities = np.stack([np.asarray(v.velocity, dtype=float) for v in others]) - np.asarray(
        ego.velocity, dtype=float
    )
    return rotate(positions, -heading), rotate(velocities, -heading)


class WorldSensors:
    """Measures one POMDP reading off the live ego vehicle, road network and vehicle list.

    Attributes:
        max_detections: Number of detection slots the emitted reading carries.

    Example:
        Reading a two-vehicle scene with a stand-in for the simulator's vehicles::

            sensors = WorldSensors(sensor_config, max_detections=4)
            reading = sensors.read(
                ego_speed=10.0,
                ego=ego_vehicle,
                arclength=240.0,
                geometry=track_geometry,
                others=[opponent],
            )
    """

    def __init__(self, sensor: SensorConfig, max_detections: int) -> None:
        """Build the world's sensor suite.

        Args:
            sensor: The widths and limits every channel is simulated at.
            max_detections: Number of detection slots the emitted reading carries.
        """
        self._sensor = sensor
        self.max_detections = int(max_detections)

    def read(
        self,
        *,
        ego_speed: float,
        ego: Any,
        arclength: float,
        geometry: TrackGeometry,
        others: Sequence[Any],
    ) -> RacetrackObservation:
        """Measure one complete POMDP reading.

        Args:
            ego_speed: The ego's signed scalar speed in m/s, taken from the simulator's own
                reading rather than re-derived, and emitted exactly.
            ego: The controlled vehicle.
            arclength: The ego's true distance around the lap in metres, numbered against
                the same lane walk ``geometry`` is built from.
            geometry: The lap's curvature profile, for the lookahead channel.
            others: Every other vehicle on the road.

        Returns:
            The five-channel reading the world emits in ``ObservationMode.POMDP``.
        """
        return RacetrackObservation(
            ego_pose=self._measure_ego_pose(ego, arclength),
            ego_speed=np.array([ego_speed], dtype=np.float32),
            lane_pose=self._measure_lane_pose(ego),
            curvature_ahead=self._measure_curvature_ahead(arclength, geometry),
            detections=self._measure_detections(ego, others),
        )

    def _measure_ego_pose(self, ego: Any, arclength: float) -> np.ndarray:
        # GPS/IMU and a wheel odometer: where the car is, which way it points, and how far
        # round the lap it has come. Near-exact by design -- this arm withholds vehicles,
        # not the ego's own pose, so a wide width here would add a localisation problem on
        # top of the tracking one the range gate poses. Heading is wrapped after the noise,
        # so a car pointing just short of pi does not read as pointing just past -pi.
        sensor = self._sensor
        truth = (
            float(ego.position[0]),
            float(ego.position[1]),
            float(ego.heading),
            float(arclength),
        )
        scales = (
            sensor.ego_position_std_m,
            sensor.ego_position_std_m,
            sensor.ego_heading_std_rad,
            sensor.ego_arclength_std_m,
        )
        measured = [
            value + float(np.random.normal(scale=scale)) for value, scale in zip(truth, scales)
        ]
        measured[EGO_POSE_HEADING] = wrap_to_pi(measured[EGO_POSE_HEADING])
        return np.array(measured, dtype=np.float32)

    def _measure_lane_pose(self, ego: Any) -> np.ndarray:
        # The lane camera's reading: true lane-relative pose plus Gaussian sensor noise.
        # highway-env's own `lane_offset` is exact and no camera is.
        sensor = self._sensor
        offset = np.asarray(ego.lane_offset, dtype=float)
        lateral = float(offset[1]) + float(np.random.normal(scale=sensor.lane_lateral_std_m))
        angle = float(offset[2]) + float(np.random.normal(scale=sensor.lane_heading_std_rad))
        return np.array([lateral, wrap_to_pi(angle)], dtype=np.float32)

    def _measure_curvature_ahead(self, arclength: float, geometry: TrackGeometry) -> np.ndarray:
        # The lane camera's other product: signed curvature at each lookahead distance, read
        # off the same lane walk the arclength is numbered against, so "20 m ahead" means 20 m
        # further along the profile the planner's mapped model indexes -- the two cannot
        # disagree about where a corner starts. The noise is the camera's, and it is why a
        # mapless planner reading this channel is estimating rather than being told.
        sensor = self._sensor
        true = np.asarray(
            geometry.curvature_at(arclength + np.asarray(sensor.curvature_lookahead_m)),
            dtype=float,
        ).reshape(-1)
        noise = np.random.normal(scale=sensor.curvature_std_1pm, size=true.shape)
        return (true + noise).astype(np.float32)

    def _measure_detections(self, ego: Any, others: Sequence[Any]) -> np.ndarray:
        """The visible vehicles, in full, noisily, nearest first.

        The range gate and the occlusion rule decide *which* vehicles appear; a vehicle that
        appears at all appears whole, position and both components of relative velocity. So
        the only thing this channel withholds is an entire vehicle, which is what makes
        ``max_detection_range_m`` the single dial the arm turns on.

        Ordering is by *measured* range rather than true range, because that is the only
        range the sensor has. The rows carry no identity, so slot ``k`` on two consecutive
        steps need not be the same vehicle; recovering the correspondence is the filter's
        problem, which is the point.

        The ranking and padding go through the same ``pack_detections`` the planner's own
        sampler uses, so a reading the world emits and one the model draws have the same
        shape by construction rather than by two implementations agreeing.
        """
        sensor = self._sensor
        positions, velocities = relative_vehicles(ego, others)
        if len(positions) == 0:
            return np.zeros((self.max_detections, DETECTION_SLOT_WIDTH), dtype=np.float32)

        visible = detection_visibility(
            positions,
            np.ones(len(positions), dtype=bool),
            sensor.max_detection_range_m,
            sensor.blocker_half_width_m,
        )
        seen, moving = positions[visible], velocities[visible]
        measured = seen + np.random.normal(scale=sensor.detection_position_std_m, size=seen.shape)
        reported = moving + np.random.normal(scale=sensor.detection_velocity_std, size=moving.shape)
        reports = np.column_stack([measured, reported])
        return pack_detections(reports, self.max_detections).astype(np.float32)


__all__ = ["SensorConfig", "WorldSensors", "relative_vehicles"]
