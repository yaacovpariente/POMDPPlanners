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

These helpers are composed by
:class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.pipeline.CarlaPerceptionPipeline`,
whose output the belief stamps as agent slots so the planner's terminal-collision and headway
logic brake for them — i.e. the sensors are used *by the planner*, not by a hidden actuation
override.
"""

from typing import Optional

import numpy as np
from scipy import ndimage

# Lidar corridor / height gating defaults (sensor frame: x forward, y right, z up, roof-mounted).
DEFAULT_CORRIDOR_HALFWIDTH = 1.5  # metres either side of the ego centreline to watch
DEFAULT_Z_MIN = -2.0  # drop ground returns (roof sensor sits ~2.4 m above the road)
DEFAULT_Z_MAX = 1.0  # drop high overhead returns (gantries, foliage)
DEFAULT_LIDAR_RANGE = 50.0  # sensor range; reported clearance when the corridor is clear

# Camera looming-cue defaults (front RGB frame).
DEFAULT_CAMERA_TRIGGER = 0.35  # looming fraction above which the camera flags a near obstacle
DEFAULT_CAMERA_RANGE = 12.0  # conservative distance assumed for a camera-only detection

# Camera traffic-light-inference defaults (front RGB frame).
DEFAULT_TL_ROI_FRACTION = 0.6  # only the upper this-fraction of the frame is searched for a bulb
DEFAULT_TL_MIN_BULB_PIXELS = 4  # red/amber blobs smaller than this many pixels are treated as noise
DEFAULT_TL_FOCAL_PIXELS = 200.0  # camera focal length (px) for the pinhole distance estimate
DEFAULT_TL_BULB_DIAMETER = 0.3  # m; physical diameter of a traffic-light bulb
DEFAULT_TL_MIN_DISTANCE = 3.0  # m; clamp floor for the estimated stop distance
DEFAULT_TL_MAX_DISTANCE = DEFAULT_LIDAR_RANGE  # m; clamp ceiling for the estimated stop distance

# Lidar vehicle-clustering defaults (a BEV connected-components detector).
DEFAULT_CLUSTER_CELL = 0.5  # BEV grid cell size (m); adjacent occupied cells form one cluster
DEFAULT_MIN_CLUSTER_POINTS = 4  # clusters with fewer returns are treated as noise
DEFAULT_MAX_VEHICLE_EXTENT = 8.0  # m; a cluster wider than this is a wall/building, not a vehicle
DEFAULT_CONFIDENT_POINTS = 40  # return count at/above which detection confidence saturates to 1.0
DEFAULT_DETECTION_RANGE = 50.0  # m; clusters beyond this are dropped


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


def traffic_light_from_camera(
    image: Optional[np.ndarray],
    focal_pixels: float = DEFAULT_TL_FOCAL_PIXELS,
    bulb_diameter: float = DEFAULT_TL_BULB_DIAMETER,
    roi_fraction: float = DEFAULT_TL_ROI_FRACTION,
    min_bulb_pixels: int = DEFAULT_TL_MIN_BULB_PIXELS,
    min_distance: float = DEFAULT_TL_MIN_DISTANCE,
    max_distance: float = DEFAULT_TL_MAX_DISTANCE,
) -> np.ndarray:
    """Infer ``[should_stop, distance_m]`` from a red/amber bulb in the upper camera frame.

    A lightweight classic-vision stand-in for a learned traffic-light detector, so the light
    is *perceived from the image* rather than read from a ground-truth channel. It thresholds
    the upper region of interest for red and amber bulbs, takes the largest qualifying blob,
    and estimates its forward distance from the pinhole relation ``distance = focal_pixels *
    bulb_diameter / bulb_pixel_size``. A green or absent bulb yields ``[0, 0]`` (no stop).

    Note:
        Distance from a single low-resolution frame is inherently coarse; ``focal_pixels`` is a
        calibration constant that should match the camera intrinsics, and the result is clamped
        to ``[min_distance, max_distance]``. The output matches the world's ``traffic_light``
        channel layout so it is a drop-in replacement for :func:`traffic_light_stop_distance`.

    Args:
        image: ``(H, W, 3)`` RGB frame, or ``None`` when no frame is available.
        focal_pixels: Camera focal length (px) used by the pinhole distance estimate.
        bulb_diameter: Physical traffic-light bulb diameter (m).
        roi_fraction: Fraction of the frame height (from the top) searched for a bulb.
        min_bulb_pixels: Blobs with fewer pixels than this are rejected as noise.
        min_distance: Lower clamp (m) on the estimated stop distance.
        max_distance: Upper clamp (m) on the estimated stop distance.

    Returns:
        ``[1.0, distance_m]`` when a red/amber bulb is detected, else ``[0.0, 0.0]``.
    """
    if image is None:
        return np.array([0.0, 0.0])
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        return np.array([0.0, 0.0])
    roi = frame[: max(1, int(frame.shape[0] * roi_fraction)), :, :3].astype(float)
    bulb_pixels = _largest_stop_bulb_size(_red_or_amber_mask(roi), min_bulb_pixels)
    if bulb_pixels <= 0.0:
        return np.array([0.0, 0.0])
    distance = float(
        np.clip(focal_pixels * bulb_diameter / bulb_pixels, min_distance, max_distance)
    )
    return np.array([1.0, distance])


def _red_or_amber_mask(roi: np.ndarray) -> np.ndarray:
    """Boolean mask of red or amber (stop-signalling) pixels in an RGB ``roi``."""
    red_ch, green_ch, blue_ch = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    red = (red_ch > 120.0) & (red_ch > 1.6 * green_ch) & (red_ch > 1.6 * blue_ch)
    amber = (
        (red_ch > 120.0)
        & (green_ch > 120.0)
        & (blue_ch < 0.6 * red_ch)
        & (blue_ch < 0.6 * green_ch)
    )
    return red | amber


def _largest_stop_bulb_size(mask: np.ndarray, min_pixels: int) -> float:
    """Largest bounding-box extent (px) among mask blobs with at least ``min_pixels`` pixels."""
    if not mask.any():
        return 0.0
    result = ndimage.label(mask)
    assert isinstance(result, tuple)  # output=None always yields (labels, n_labels)
    labelled, n_labels = result
    best = 0.0
    for label in range(1, n_labels + 1):
        rows, cols = np.where(labelled == label)
        if rows.size < min_pixels:
            continue
        extent = max(float(rows.max() - rows.min() + 1), float(cols.max() - cols.min() + 1))
        best = max(best, extent)
    return best


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


def lidar_vehicle_detections(
    points: Optional[np.ndarray],
    cell_size: float = DEFAULT_CLUSTER_CELL,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    detection_range: float = DEFAULT_DETECTION_RANGE,
    min_points: int = DEFAULT_MIN_CLUSTER_POINTS,
    max_extent: float = DEFAULT_MAX_VEHICLE_EXTENT,
    confident_points: int = DEFAULT_CONFIDENT_POINTS,
) -> np.ndarray:
    """Cluster a lidar cloud into ego-frame vehicle detections ``[rel_x, rel_y, confidence]``.

    A single-frame perception stage — the first step of a real perception pipeline that
    replaces the ground-truth ``agents`` oracle. It drops ground / overhead returns, groups
    the rest into BEV clusters (8-connected occupied cells), and keeps the vehicle-sized ones
    (rejecting walls / buildings by extent and noise by point count). Confidence rises with the
    return count. Positions are in the ego frame (``rel_x`` forward, ``rel_y`` left); velocity
    and heading are left to a downstream tracker.

    Args:
        points: ``(N, 4)`` cloud ``[x, y, z, intensity]`` in the sensor frame, or ``None``.
        cell_size: BEV grid cell size (m); adjacent occupied cells merge into one cluster.
        z_min: Lower height bound (m); returns below it (the ground) are dropped.
        z_max: Upper height bound (m); returns above it (overhead structure) are dropped.
        detection_range: Clusters beyond this range (m) are dropped.
        min_points: Clusters with fewer returns are discarded as noise.
        max_extent: Clusters wider than this (m) are discarded as walls / buildings.
        confident_points: Return count at/above which confidence saturates to 1.0.

    Returns:
        An ``(M, 3)`` array of ``[rel_x, rel_y, confidence]`` detections (empty if none).
    """
    if points is None:
        return np.zeros((0, 3))
    cloud = np.asarray(points, dtype=float)
    if cloud.size == 0:
        return np.zeros((0, 3))
    forward, lateral, height = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    keep = (height > z_min) & (height < z_max) & (np.hypot(forward, lateral) <= detection_range)
    forward, lateral = forward[keep], lateral[keep]
    if forward.size == 0:
        return np.zeros((0, 3))
    n_labels, point_label = _bev_cluster_labels(forward, lateral, cell_size, detection_range)
    detections = [
        detection
        for label in range(1, n_labels + 1)
        if (
            detection := _cluster_detection(
                forward, lateral, point_label == label, min_points, max_extent, confident_points
            )
        )
        is not None
    ]
    return np.array(detections) if detections else np.zeros((0, 3))


def _bev_cluster_labels(forward, lateral, cell_size, detection_range):
    """Label the BEV occupancy grid; return ``(n_labels, per-point label)`` (8-connectivity)."""
    dim = int(np.ceil(2 * detection_range / cell_size)) + 1
    col = np.clip(((forward + detection_range) / cell_size).astype(int), 0, dim - 1)
    row = np.clip(((lateral + detection_range) / cell_size).astype(int), 0, dim - 1)
    grid = np.zeros((dim, dim), dtype=bool)
    grid[col, row] = True
    result = ndimage.label(grid, structure=np.ones((3, 3)))
    assert isinstance(result, tuple)  # output=None always yields (labels, n_labels)
    labelled, n_labels = result
    return int(n_labels), labelled[col, row]


def _cluster_detection(forward, lateral, member, min_points, max_extent, confident_points):
    """One cluster -> ``[rel_x, rel_y, confidence]`` (ego frame), or ``None`` if not a vehicle."""
    count = int(member.sum())
    if count < min_points:
        return None
    cluster_x, cluster_y = forward[member], lateral[member]
    span = max(float(cluster_x.max() - cluster_x.min()), float(cluster_y.max() - cluster_y.min()))
    if span > max_extent:
        return None
    # Sensor y points right; the ego frame's rel_y points left, hence the sign flip.
    return [float(cluster_x.mean()), -float(cluster_y.mean()), min(1.0, count / confident_points)]
