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
    traffic_light_stop_distance,
)


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
