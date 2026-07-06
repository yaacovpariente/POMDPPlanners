# SPDX-License-Identifier: MIT

"""Forward-obstacle perception from the CARLA lidar and front camera.

The planner-side models carry other vehicles as fixed ego-frame *agent slots*, but those
slots are a nearest-K-vehicle abstraction: they miss non-vehicle geometry, vehicles beyond
the tracked count, and — because the kinematic model never propagates agent lateral motion —
a car cutting into the lane until it is already dead ahead. The raycast **lidar** does not
share those blind spots: it measures the true distance to whatever solid is in front of the
ego right now, so it is the reliable signal for "is something about to be hit."

This module turns the raw sensors into a single scalar — the forward clearance (metres to the
nearest in-corridor obstacle, or the sensor range when clear) — with pure, testable helpers:

* :func:`lidar_forward_clearance` — the authoritative geometric range from the ``(N, 4)``
  point cloud, gated to a forward driving corridor and a vehicle-height band (so ground and
  overhead returns are ignored).
* :func:`camera_looming_cue` — a lightweight secondary cue in ``[0, 1]`` from the front RGB
  frame that rises when a large near object fills the lower-centre view; it corroborates a
  lidar detection and can flag a near obstacle when the lidar returns are sparse.
* :func:`fuse_forward_obstacle` — combines the two into the reported obstacle distance, lidar
  taking precedence and the camera only shortening the range when the lidar sees nothing.
* :func:`traffic_light_stop_distance` — turns the ``traffic_light`` observation into a stop
  distance, so a red light can be injected as a virtual obstacle just like a stopped car.

The distance this produces is consumed by
:class:`~POMDPPlanners.environments.carla_pomdp.carla_belief.CarlaAgentReinvigoration`, which
injects it as an agent slot so the planner's terminal-collision and headway logic brake for
it — i.e. the sensors are used *by the planner*, not by a hidden actuation override.
"""

from typing import Optional

import numpy as np

# Lidar corridor / height gating defaults (sensor frame: x forward, y right, z up, roof-mounted).
DEFAULT_CORRIDOR_HALFWIDTH = 1.5  # metres either side of the ego centreline to watch
DEFAULT_Z_MIN = -2.0  # drop ground returns (roof sensor sits ~2.4 m above the road)
DEFAULT_Z_MAX = 1.0  # drop high overhead returns (gantries, foliage)
DEFAULT_LIDAR_RANGE = 50.0  # sensor range; reported clearance when the corridor is clear

# Camera looming-cue defaults (front RGB frame).
DEFAULT_CAMERA_TRIGGER = 0.35  # looming fraction above which the camera flags a near obstacle
DEFAULT_CAMERA_RANGE = 12.0  # conservative distance assumed for a camera-only detection


def lidar_forward_clearance(
    points: Optional[np.ndarray],
    corridor_halfwidth: float = DEFAULT_CORRIDOR_HALFWIDTH,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    max_range: float = DEFAULT_LIDAR_RANGE,
) -> float:
    """Distance (m) to the nearest forward in-corridor lidar return, or ``max_range`` if clear.

    Args:
        points: ``(N, 4)`` lidar cloud ``[x, y, z, intensity]`` in the sensor frame
            (``x`` forward, ``y`` right, ``z`` up), or ``None``/empty when no scan is available.
        corridor_halfwidth: Half-width (m) of the forward corridor watched for obstacles.
        z_min: Lower height bound (m); returns below it (the ground) are ignored.
        z_max: Upper height bound (m); returns above it (overhead structure) are ignored.
        max_range: Value returned when the corridor holds no qualifying return.

    Returns:
        The minimum forward ``x`` among corridor returns, else ``max_range``.
    """
    if points is None:
        return max_range
    cloud = np.asarray(points, dtype=float)
    if cloud.size == 0:
        return max_range
    forward, lateral, height = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    in_corridor = (
        (forward > 0.0)
        & (np.abs(lateral) < corridor_halfwidth)
        & (height > z_min)
        & (height < z_max)
    )
    if not np.any(in_corridor):
        return max_range
    return float(np.min(forward[in_corridor]))


def camera_looming_cue(image: Optional[np.ndarray]) -> float:
    """Fraction in ``[0, 1]`` of the lower-centre view filled by a large, dark near object.

    A close vehicle or wall directly ahead darkens and flattens the lower-centre of the front
    camera. This coarse cue corroborates a lidar detection; it is deliberately conservative so
    it cannot, on its own, manufacture phantom braking (see :func:`fuse_forward_obstacle`).

    Args:
        image: ``(H, W, 3)`` RGB frame, or ``None`` when no frame is available.

    Returns:
        The fraction of lower-centre pixels below a darkness threshold.
    """
    if image is None:
        return 0.0
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        return 0.0
    height, width = frame.shape[:2]
    roi = frame[height // 2 :, (3 * width) // 10 : (7 * width) // 10, :]
    if roi.size == 0:
        return 0.0
    intensity = roi.mean(axis=2)
    return float(np.mean(intensity < 60.0))


def traffic_light_stop_distance(
    traffic_light: Optional[np.ndarray], max_range: float = DEFAULT_LIDAR_RANGE
) -> float:
    """Forward distance (m) to a red/yellow stop line, or ``max_range`` if clear.

    Turns the world's ``traffic_light`` observation into an obstacle distance so a red
    light can be injected as a virtual stop-obstacle, exactly like a stopped vehicle.

    Args:
        traffic_light: ``[should_stop, distance_m]`` from the observation, or ``None``.
        max_range: Value returned when there is no active stop signal.

    Returns:
        The stop-line distance when ``should_stop`` is set and positive, else ``max_range``.
    """
    if traffic_light is None:
        return max_range
    signal = np.asarray(traffic_light, dtype=float).reshape(-1)
    if signal.size < 2 or signal[0] < 0.5:
        return max_range
    distance = float(signal[1])
    return distance if distance > 0.0 else max_range


def fuse_forward_obstacle(
    lidar_clearance: float,
    camera_cue: float,
    max_range: float = DEFAULT_LIDAR_RANGE,
    camera_trigger: float = DEFAULT_CAMERA_TRIGGER,
    camera_range: float = DEFAULT_CAMERA_RANGE,
) -> float:
    """Fuse the lidar clearance and camera cue into a single forward-obstacle distance.

    Lidar is authoritative: whenever it reports an in-range obstacle that distance is used.
    The camera only acts when the lidar corridor is clear (``lidar_clearance >= max_range``)
    yet the looming cue is strong, in which case a conservative ``camera_range`` obstacle is
    assumed so a lidar miss (sparse returns on a dark, close surface) still triggers caution.

    Args:
        lidar_clearance: Forward clearance from :func:`lidar_forward_clearance`.
        camera_cue: Looming fraction from :func:`camera_looming_cue`.
        max_range: Sensor range that denotes "lidar corridor clear".
        camera_trigger: Cue value at/above which the camera flags a near obstacle.
        camera_range: Distance assumed for a camera-only detection.

    Returns:
        Distance (m) to the nearest forward obstacle; ``max_range`` when nothing is detected.
    """
    if lidar_clearance < max_range:
        return lidar_clearance
    if camera_cue >= camera_trigger:
        return camera_range
    return max_range
