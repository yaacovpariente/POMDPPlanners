# SPDX-License-Identifier: MIT

"""Tests for the constant-velocity lidar multi-object tracker.

Covers :func:`update_tracks` spawning tracks from detections, estimating an ego-frame
velocity from consecutive detections (alpha-beta filter), coasting and evicting missed
tracks, and keeping distinct detections as separate tracks. All tests use hand-built
detection arrays; no CARLA server is needed.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.carla_pomdp.carla_perception.tracking import update_tracks

_EMPTY = np.zeros((0, 5))


def _det(rel_x: float, rel_y: float, confidence: float = 0.9) -> np.ndarray:
    return np.array([[rel_x, rel_y, confidence]])


def test_new_detection_spawns_a_track():
    """An unmatched detection becomes a fresh zero-velocity track.

    Purpose: Validates track spawning from a detection.

    Given: An empty track set and one detection 10 m ahead
    When: update_tracks runs
    Then: One track appears at that position with zero velocity

    Test type: unit
    """
    tracks = update_tracks(_EMPTY, _det(10.0, 0.0), dt=1.0)

    assert tracks.shape[0] == 1
    np.testing.assert_allclose(tracks[0, :4], [10.0, 0.0, 0.0, 0.0])


def test_consecutive_detections_estimate_approaching_velocity():
    """A lead vehicle drawing nearer each frame yields a negative rel_x velocity.

    Purpose: Validates the alpha-beta filter recovers the relative velocity.

    Given: A detection that moves from 10 m to 5 m over five 1 s steps
    When: update_tracks is run each step
    Then: A single track survives with an estimated approaching (negative) rel_x velocity

    Test type: unit
    """
    tracks = update_tracks(_EMPTY, _det(10.0, 0.0), dt=1.0)
    for rel_x in (9.0, 8.0, 7.0, 6.0, 5.0):
        tracks = update_tracks(tracks, _det(rel_x, 0.0), dt=1.0)

    assert tracks.shape[0] == 1
    assert tracks[0, 2] < -0.5  # closing on the ego


def test_missed_track_coasts_then_evicts():
    """An undetected track decays in confidence and is dropped.

    Purpose: Validates coasting and confidence-based eviction.

    Given: A spawned track and then two updates with no detections
    When: update_tracks is run for the misses
    Then: The track survives the first miss and is evicted by the second

    Test type: unit
    """
    tracks = update_tracks(_EMPTY, _det(10.0, 0.0), dt=1.0)

    tracks = update_tracks(tracks, None, dt=1.0)
    assert tracks.shape[0] == 1  # coasted, still confident enough

    tracks = update_tracks(tracks, None, dt=1.0)
    assert tracks.shape[0] == 0  # confidence exhausted -> evicted


def test_distinct_detections_stay_separate_tracks():
    """Two well-separated detections produce two tracks.

    Purpose: Validates association does not merge distinct vehicles.

    Given: An empty track set and two detections 10 m apart laterally
    When: update_tracks runs
    Then: Two tracks are returned

    Test type: unit
    """
    detections = np.array([[10.0, 0.0, 0.9], [10.0, 10.0, 0.9]])

    tracks = update_tracks(_EMPTY, detections, dt=1.0)

    assert tracks.shape[0] == 2


def test_repeated_match_saturates_confidence():
    """A track matched each frame rises to full confidence.

    Purpose: Validates the confidence gain on matched detections.

    Given: A track re-detected at the same spot on three frames
    When: update_tracks is run each frame
    Then: Its confidence saturates at 1.0

    Test type: unit
    """
    tracks = update_tracks(_EMPTY, _det(10.0, 0.0), dt=1.0)
    tracks = update_tracks(tracks, _det(10.0, 0.0), dt=1.0)
    tracks = update_tracks(tracks, _det(10.0, 0.0), dt=1.0)

    assert tracks[0, 4] == pytest.approx(1.0)
