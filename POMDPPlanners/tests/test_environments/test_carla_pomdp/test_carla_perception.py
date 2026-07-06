# SPDX-License-Identifier: MIT

"""Tests for the CARLA forward-obstacle perception helpers.

Covers the pure lidar/camera helpers that turn raw sensors into a forward-obstacle distance:
:func:`lidar_forward_clearance` gating a point cloud to a forward, vehicle-height corridor,
:func:`camera_looming_cue` scoring the lower-centre view, and :func:`fuse_forward_obstacle`
combining them with lidar taking precedence. All tests run on hand-built arrays.
"""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception import (
    DEFAULT_LIDAR_RANGE,
    camera_looming_cue,
    fuse_forward_obstacle,
    lidar_forward_clearance,
    lidar_vehicle_detections,
    traffic_light_from_camera,
    traffic_light_stop_distance,
)


def _camera_with_bulb(color: tuple, size: int, top: int = 4, left: int = 60) -> np.ndarray:
    """A 128x128 dark RGB frame with one ``size``x``size`` coloured bulb in the upper region."""
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[top : top + size, left : left + size, :] = np.array(color, dtype=np.uint8)
    return frame


def _blob(center_x: float, center_y: float, height: float = -1.0, half: float = 0.4) -> np.ndarray:
    """A 5x5 grid of vehicle-height lidar points centred at ``(center_x, center_y)``."""
    grid = np.linspace(-half, half, 5)
    xx, yy = np.meshgrid(grid, grid)
    points = np.zeros((xx.size, 4))
    points[:, 0] = center_x + xx.ravel()
    points[:, 1] = center_y + yy.ravel()
    points[:, 2] = height
    return points


def test_lidar_clearance_returns_nearest_forward_return():
    """The clearance is the nearest in-corridor forward return.

    Purpose: Validates the corridor gate reports the closest obstacle distance.

    Given: A cloud with forward returns at x = 9 m and x = 6 m in the corridor
    When: lidar_forward_clearance is computed
    Then: It returns the nearer 6 m

    Test type: unit
    """
    points = np.array([[9.0, 0.0, -1.0, 0.5], [6.0, 0.3, -1.0, 0.5]])

    assert lidar_forward_clearance(points) == 6.0


def test_lidar_clearance_ignores_ground_and_out_of_corridor_returns():
    """Ground-height and off-corridor returns do not count as obstacles.

    Purpose: Validates the height band and lateral gate exclude non-obstacles.

    Given: A near ground return (z below the band) and a near return 3 m to the side
    When: lidar_forward_clearance is computed
    Then: Both are ignored and the corridor reads clear (max range)

    Test type: unit
    """
    points = np.array([[3.0, 0.0, -2.4, 0.5], [4.0, 3.0, -1.0, 0.5]])

    assert lidar_forward_clearance(points) == DEFAULT_LIDAR_RANGE


def test_lidar_clearance_empty_or_none_is_clear():
    """A missing or empty scan reports a clear corridor.

    Purpose: Validates the no-data path returns the sensor range, not an error.

    Given: ``None`` and an empty array as the point cloud
    When: lidar_forward_clearance is computed
    Then: Both return the max range

    Test type: unit
    """
    assert lidar_forward_clearance(None) == DEFAULT_LIDAR_RANGE
    assert lidar_forward_clearance(np.zeros((0, 4))) == DEFAULT_LIDAR_RANGE


def test_camera_looming_cue_high_for_dark_near_object():
    """A dark lower-centre view yields a high looming cue.

    Purpose: Validates the cue rises when a large near object fills the view.

    Given: An all-dark RGB frame and an all-bright RGB frame
    When: camera_looming_cue is computed for each
    Then: The dark frame scores near 1.0 and the bright frame near 0.0

    Test type: unit
    """
    dark = np.zeros((128, 128, 3), dtype=np.uint8)
    bright = np.full((128, 128, 3), 255, dtype=np.uint8)

    assert camera_looming_cue(dark) > 0.9
    assert camera_looming_cue(bright) < 0.1


def test_camera_looming_cue_none_is_zero():
    """A missing frame yields no looming cue.

    Purpose: Validates the no-data path returns 0.0.

    Given: ``None`` as the camera frame
    When: camera_looming_cue is computed
    Then: It returns 0.0

    Test type: unit
    """
    assert camera_looming_cue(None) == 0.0


def test_traffic_light_stop_distance_reports_red_light_gap():
    """A red/yellow stop signal reports its stop-line distance; otherwise the way is clear.

    Purpose: Validates the traffic-light observation becomes an obstacle distance.

    Given: a stop signal [1, 12] , a no-stop signal [0, 0] , and None
    When: traffic_light_stop_distance is computed
    Then: the stop signal yields 12 m; the others yield the max range

    Test type: unit
    """
    assert traffic_light_stop_distance(np.array([1.0, 12.0])) == 12.0
    assert traffic_light_stop_distance(np.array([0.0, 0.0])) == DEFAULT_LIDAR_RANGE
    assert traffic_light_stop_distance(None) == DEFAULT_LIDAR_RANGE


def test_fuse_prefers_lidar_when_it_detects():
    """A lidar detection overrides the camera cue.

    Purpose: Validates lidar is authoritative on obstacle distance.

    Given: A lidar clearance of 6 m and a strong camera cue
    When: fuse_forward_obstacle combines them
    Then: The lidar 6 m is returned

    Test type: unit
    """
    assert fuse_forward_obstacle(6.0, camera_cue=0.9) == 6.0


def test_lidar_detects_separate_vehicle_clusters_in_ego_frame():
    """Two vehicle-sized point clusters become two ego-frame detections.

    Purpose: Validates clustering finds each vehicle and converts to the ego frame.

    Given: A cloud with two vehicle-height blobs at sensor (10, -3) and (10, 3)
    When: lidar_vehicle_detections clusters it
    Then: Two detections appear at rel_x ~ 10 with rel_y ~ +3 and -3 (sensor-y sign flipped)

    Test type: unit
    """
    cloud = np.vstack([_blob(10.0, -3.0), _blob(10.0, 3.0)])

    detections = lidar_vehicle_detections(cloud)

    assert detections.shape[0] == 2
    by_rel_y = detections[np.argsort(detections[:, 1])]
    np.testing.assert_allclose(by_rel_y[:, 0], [10.0, 10.0], atol=0.2)  # both ~10 m ahead
    np.testing.assert_allclose(by_rel_y[:, 1], [-3.0, 3.0], atol=0.2)  # ego-frame left/right
    assert np.all((by_rel_y[:, 2] > 0.0) & (by_rel_y[:, 2] <= 1.0))  # confidence in (0, 1]


def test_lidar_detection_ignores_ground_and_none():
    """Ground-height returns and a missing cloud yield no detections.

    Purpose: Validates the height gate and the no-data path.

    Given: A blob at ground height (below the vehicle band) and ``None``
    When: lidar_vehicle_detections is computed for each
    Then: Both return an empty (0, 3) detection array

    Test type: unit
    """
    ground = _blob(10.0, 0.0, height=-2.4)

    assert lidar_vehicle_detections(ground).shape == (0, 3)
    assert lidar_vehicle_detections(None).shape == (0, 3)


def test_lidar_detection_rejects_wall_sized_clusters():
    """A long wall-sized cluster is rejected as not a vehicle.

    Purpose: Validates the extent filter drops buildings / walls.

    Given: A 20 m-long line of vehicle-height returns (extent far exceeds a vehicle)
    When: lidar_vehicle_detections clusters it
    Then: No detection is returned

    Test type: unit
    """
    wall = np.zeros((60, 4))
    wall[:, 0] = 10.0
    wall[:, 1] = np.linspace(-10.0, 10.0, 60)
    wall[:, 2] = -1.0

    assert lidar_vehicle_detections(wall).shape == (0, 3)


def test_fuse_uses_camera_only_when_lidar_clear():
    """The camera flags a near obstacle only when the lidar sees nothing.

    Purpose: Validates the conservative camera fallback and the all-clear case.

    Given: A clear lidar corridor with a strong cue, and the same with no cue
    When: fuse_forward_obstacle combines them
    Then: The strong cue yields the conservative camera range; no cue yields max range

    Test type: unit
    """
    assert fuse_forward_obstacle(DEFAULT_LIDAR_RANGE, camera_cue=0.9) < DEFAULT_LIDAR_RANGE
    assert fuse_forward_obstacle(DEFAULT_LIDAR_RANGE, camera_cue=0.0) == DEFAULT_LIDAR_RANGE


def test_traffic_light_from_camera_detects_red_bulb_as_stop():
    """A red bulb high in the frame is read as a stop signal at a finite distance.

    Purpose: Validates camera-based traffic-light inference flags a red light to stop.

    Given: A dark frame with a red bulb in the upper region
    When: traffic_light_from_camera reads it
    Then: should_stop is 1.0 and the estimated distance is finite and in range

    Test type: unit
    """
    signal = traffic_light_from_camera(_camera_with_bulb((255, 0, 0), size=10))

    assert signal[0] == 1.0
    assert 0.0 < signal[1] <= DEFAULT_LIDAR_RANGE


def test_traffic_light_from_camera_detects_amber_bulb_as_stop():
    """An amber (yellow) bulb is also treated as a stop signal.

    Purpose: Validates amber lights are perceived as stop, matching the world's stop states.

    Given: A dark frame with a yellow bulb in the upper region
    When: traffic_light_from_camera reads it
    Then: should_stop is 1.0

    Test type: unit
    """
    assert traffic_light_from_camera(_camera_with_bulb((255, 220, 0), size=10))[0] == 1.0


def test_traffic_light_from_camera_ignores_green_and_missing_frame():
    """A green bulb or an absent frame yields no stop signal.

    Purpose: Validates green lights and missing frames produce [0, 0] (no stop).

    Given: A frame with a green bulb, and separately None
    When: traffic_light_from_camera reads each
    Then: Both return [0.0, 0.0]

    Test type: unit
    """
    assert list(traffic_light_from_camera(_camera_with_bulb((0, 255, 0), size=10))) == [0.0, 0.0]
    assert list(traffic_light_from_camera(None)) == [0.0, 0.0]


def test_traffic_light_from_camera_larger_bulb_reads_nearer():
    """A larger red bulb estimates a nearer stop distance than a smaller one.

    Purpose: Validates the pinhole distance estimate is monotonic in bulb pixel size.

    Given: Two red bulbs, one 20 px and one 8 px across
    When: traffic_light_from_camera estimates each distance
    Then: The larger bulb yields the smaller (nearer) distance

    Test type: unit
    """
    near = traffic_light_from_camera(_camera_with_bulb((255, 0, 0), size=20))
    far = traffic_light_from_camera(_camera_with_bulb((255, 0, 0), size=8))

    assert near[1] < far[1]


def test_traffic_light_from_camera_rejects_sub_threshold_noise():
    """A single stray red pixel is rejected as noise, not a bulb.

    Purpose: Validates the minimum-blob-size gate suppresses spurious detections.

    Given: A frame with a 1x1 red speck (below min_bulb_pixels)
    When: traffic_light_from_camera reads it
    Then: No stop signal is produced

    Test type: unit
    """
    assert traffic_light_from_camera(_camera_with_bulb((255, 0, 0), size=1))[0] == 0.0
