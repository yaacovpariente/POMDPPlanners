# SPDX-License-Identifier: MIT

"""Tests for the racetrack occupancy-grid velocity tracker.

Every grid here is built by hand, so each test states exactly which cells are occupied and
what the tracker is therefore obliged to report. No simulator is involved.
"""

import math
from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_occupancy_tracker import (
    OccupancyVelocityTracker,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    GRID_CELLS,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    PRESENCE_LAYER,
)

EGO_CELL = (6, 6)
DT_S = 0.2
ONE_CELL_SPEED = GRID_STEP_M / DT_S  # 15 m/s: the smallest non-zero reading possible


def empty_grid() -> np.ndarray:
    """A grid with nothing in it, not even the ego."""
    return np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)


def grid_with(*cells: Tuple[int, int], include_ego: bool = True) -> np.ndarray:
    """A presence grid marking ``cells``, plus the ego at the centre as the simulator does."""
    grid = empty_grid()
    if include_ego:
        grid[PRESENCE_LAYER][EGO_CELL] = 1.0
    for cell in cells:
        grid[PRESENCE_LAYER][cell] = 1.0
    return grid


def cell_centre(along: int, across: int) -> np.ndarray:
    """The ego-frame metre position of a cell's centre."""
    return -GRID_HALF_EXTENT_M + (np.array([along, across], dtype=float) + 0.5) * GRID_STEP_M


def test_single_blob_translated_two_cells_reports_two_cell_speed():
    """Test that a blob moved two cells along-track differences into the matching speed.

    Purpose: Validates the core measurement — cell displacement divided by the frame interval

    Given: A tracker with a gate wide enough to admit a two-cell jump, and one opponent blob
        that moves from along-index 5 to along-index 7 with the across-index unchanged
    When: track is called on the two frames with no ego yaw change
    Then: The single reported cluster is matched and its velocity is (2 * 3.0 / 0.2, 0) = (30, 0)

    Test type: unit
    """
    tracker = OccupancyVelocityTracker(gate_radius_m=9.0)
    clusters = tracker.track(grid_with((5, 9)), grid_with((7, 9)), ego_yaw_delta=0.0)

    assert len(clusters) == 1
    assert clusters[0].matched
    np.testing.assert_allclose(clusters[0].velocity, [2 * ONE_CELL_SPEED, 0.0], atol=1e-9)


def test_ego_centre_cell_alone_yields_no_clusters():
    """Test that the ego's own mark is never reported as a tracked vehicle.

    Purpose: Validates that the permanently occupied centre cell is excluded

    Given: Two identical frames in which the only occupied cell is the ego's centre cell
    When: track is called on them
    Then: No clusters are reported, so the ego never becomes a stationary phantom opponent

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()

    assert tracker.track(grid_with(), grid_with(), ego_yaw_delta=0.0) == []


def test_opponent_beside_the_ego_is_not_merged_into_the_ego_blob():
    """Test that a vehicle one cell from the ego survives as its own cluster.

    Purpose: Validates the merge hazard — 8-connectivity would otherwise join an adjacent
        opponent to the ego's permanent centre blob, and dropping that whole component would
        blind the tracker at the closest and most safety-critical range

    Given: A frame with the ego at the centre cell and an opponent in the immediately
        adjacent along-track cell, about 3 m ahead
    When: detect_clusters is called on that frame
    Then: Exactly one cluster is reported, centred on the opponent's cell and not on a blob
        that has absorbed the ego

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()
    clusters = tracker.detect_clusters(grid_with((7, 6)))

    assert len(clusters) == 1
    np.testing.assert_allclose(clusters[0].centre, cell_centre(7, 6))


def test_two_blobs_match_their_nearer_partner_rather_than_crossing():
    """Test that greedy global nearest-neighbour matching does not swap two blobs.

    Purpose: Validates that each blob is paired with its closest previous-frame partner even
        when the crossing assignment is also inside the gate

    Given: Two well-separated previous blobs and two current blobs that each moved one cell
        along-track in opposite directions, with a gate wide enough that the crossing pairing
        would also be admitted
    When: track is called with no ego yaw change
    Then: Each cluster's velocity has the sign of its own blob's motion, which the crossing
        assignment could not produce

    Test type: unit
    """
    tracker = OccupancyVelocityTracker(gate_radius_m=12.0)
    previous = grid_with((8, 4), (8, 7))
    current = grid_with((9, 4), (7, 7))

    clusters = tracker.track(previous, current, ego_yaw_delta=0.0)
    by_across = sorted(clusters, key=lambda cluster: float(cluster.centre[1]))

    assert [cluster.matched for cluster in by_across] == [True, True]
    np.testing.assert_allclose(by_across[0].velocity, [ONE_CELL_SPEED, 0.0], atol=1e-9)
    np.testing.assert_allclose(by_across[1].velocity, [-ONE_CELL_SPEED, 0.0], atol=1e-9)


def test_blob_displaced_beyond_the_gate_is_reported_unmatched():
    """Test that an implausibly large jump is refused rather than turned into a velocity.

    Purpose: Validates the association gate, which is what stops a vehicle leaving the window
        and an unrelated one entering it from being differenced into a huge fake velocity

    Given: A previous blob and a current blob separated by more than the gate radius
    When: track is called with no ego yaw change
    Then: The current cluster is reported with matched False and zero velocity

    Test type: unit
    """
    tracker = OccupancyVelocityTracker(gate_radius_m=6.0)
    clusters = tracker.track(grid_with((2, 2)), grid_with((10, 10)), ego_yaw_delta=0.0)

    assert len(clusters) == 1
    assert not clusters[0].matched
    np.testing.assert_array_equal(clusters[0].velocity, np.zeros(2))


def test_static_blob_under_ego_yaw_needs_the_de_rotation():
    """Test that de-rotating the previous frame cancels the apparent motion of a static blob.

    Purpose: Validates that the grid's rotation with the ego is removed before differencing.
        The observation is aligned to the vehicle axes, so a turning ego sweeps every blob
        across the grid; without de-rotation the tracker manufactures velocity for a world
        that did not move.

    Given: A blob that stays fixed in the world while the ego yaws by the angle that carries
        cell (11, 6) exactly onto cell (11, 5) -- roughly 0.18 rad, a plausible one-step turn
    When: track is called once with that yaw delta and once with a yaw delta of zero
    Then: The de-rotated call reports essentially zero velocity, while the naive call reports
        a full one-cell speed of 15 m/s that is entirely an artefact of the frame turning

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()
    previous = grid_with((11, 6))
    current = grid_with((11, 5))
    yaw_delta = 2.0 * math.atan2(1.5, 16.5)

    de_rotated = tracker.track(previous, current, ego_yaw_delta=yaw_delta)[0]
    naive = tracker.track(previous, current, ego_yaw_delta=0.0)[0]

    np.testing.assert_allclose(de_rotated.velocity, np.zeros(2), atol=1e-9)
    np.testing.assert_allclose(naive.velocity, [0.0, -ONE_CELL_SPEED], atol=1e-9)


def test_diagonally_touching_cells_form_one_cluster():
    """Test that 8-connectivity merges a corner-touching cell pair into one blob.

    Purpose: Pins the connectivity structure and, with it, the trade it makes. A vehicle
        marks one cell, so a corner-touching pair is either two real opponents 4 m apart or a
        vehicle plus one flipped cell from the model's observation noise. Eight-connectivity
        reads both as a single obstacle at the midpoint; four-connectivity would read both as
        two, inventing a phantom opponent with a fabricated velocity in the noise case. The
        merge is the conservative reading, and this test is what makes the choice visible if
        someone later changes the structure.

    Given: A frame in which two diagonally adjacent cells are occupied
    When: detect_clusters is called on that frame
    Then: One cluster is reported, centred midway between the two cells

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()
    clusters = tracker.detect_clusters(grid_with((9, 4), (10, 5)))

    assert len(clusters) == 1
    expected = (cell_centre(9, 4) + cell_centre(10, 5)) / 2.0
    np.testing.assert_allclose(clusters[0].centre, expected)


def test_empty_grid_reports_no_clusters():
    """Test that a frame with no occupancy at all yields an empty list.

    Purpose: Validates the degenerate case, which the belief hits before any vehicle enters
        the window and which must not raise or fabricate a slot

    Given: Two frames with no occupied cells, not even the ego's
    When: track is called on them
    Then: An empty list is returned

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()

    assert tracker.track(empty_grid(), empty_grid(), ego_yaw_delta=0.0) == []


def test_frame_stride_divides_the_velocity_quantum():
    """Test that a wider measurement baseline scales the reported speed down.

    Purpose: Validates the frame_stride knob, the only way to get below the 15 m/s
        quantisation floor that a 3 m cell and a 0.2 s step impose

    Given: The same one-cell displacement measured by a stride-1 tracker and a stride-3
        tracker, the latter being told the two frames are three decision steps apart
    When: track is called on both
    Then: The stride-3 tracker reports one third of the stride-1 speed

    Test type: unit
    """
    previous, current = grid_with((8, 6)), grid_with((9, 6))

    single = OccupancyVelocityTracker(frame_stride=1)
    strided = OccupancyVelocityTracker(frame_stride=3)

    fast = single.track(previous, current, ego_yaw_delta=0.0)[0].velocity
    slow = strided.track(previous, current, ego_yaw_delta=0.0)[0].velocity
    np.testing.assert_allclose(fast, [ONE_CELL_SPEED, 0.0], atol=1e-9)
    np.testing.assert_allclose(slow, [ONE_CELL_SPEED / 3.0, 0.0], atol=1e-9)


def test_grid_of_the_wrong_shape_is_rejected():
    """Test that a grid whose geometry does not match the tracker's raises.

    Purpose: Validates that a mis-configured window fails loudly. Silently indexing a
        differently sized grid would put every cluster at the wrong metre position.

    Given: A tracker configured for the shipped 12x12 window and a 10x10 grid
    When: detect_clusters is called on it
    Then: ValueError is raised naming the expected shape

    Test type: unit
    """
    tracker = OccupancyVelocityTracker()

    with pytest.raises(ValueError, match="shape"):
        tracker.detect_clusters(np.zeros((2, 10, 10), dtype=np.float32))
