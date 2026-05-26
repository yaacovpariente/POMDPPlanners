# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the continuous LaserTag geometry utilities module.

Tests cover ray-AABB intersection, ray-circle intersection,
wall collision resolution, grid clamping, and batch variants.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    LASER_DIRECTIONS,
    batch_clamp_to_grid,
    batch_laser_measurements,
    batch_resolve_wall_collision,
    clamp_to_grid,
    compute_laser_measurements,
    ray_aabb_distances,
    ray_circle_distance,
    resolve_wall_collision,
)


class TestRayAABBDistances:
    """Tests for the ray_aabb_distances function."""

    def test_no_walls_returns_large_distance(self):
        """Test that no walls results in very large distances.

        Purpose: Validates behavior when there are no walls.

        Given: An origin point and directions with no walls.
        When: ray_aabb_distances is called with an empty wall array.
        Then: All returned distances should be very large (RAY_MAX).

        Test type: unit
        """
        origin = np.array([5.0, 3.0])
        walls = np.empty((0, 4))
        dists = ray_aabb_distances(origin, LASER_DIRECTIONS, walls)
        assert dists.shape == (8,)
        assert np.all(dists > 100.0)

    def test_ray_hits_wall_in_front(self):
        """Test that a ray hitting a wall returns the correct distance.

        Purpose: Validates that a ray correctly intersects a wall AABB.

        Given: A robot at (1, 3) and a wall centered at (5, 3).
        When: Casting a ray East (direction index 2).
        Then: The distance should be approximately 3.5 (wall edge minus robot x).

        Test type: unit
        """
        origin = np.array([1.0, 3.0])
        walls = np.array([[5.0, 3.0, 0.5, 0.5]])
        dists = ray_aabb_distances(origin, LASER_DIRECTIONS, walls)
        # East direction (index 2) should hit the wall
        east_dist = dists[2]
        expected = 5.0 - 0.5 - 1.0  # wall left edge minus origin x
        assert abs(east_dist - expected) < 0.1

    def test_ray_behind_wall_not_hit(self):
        """Test that a ray in the opposite direction does not hit a wall behind.

        Purpose: Validates that backward rays don't intersect walls behind the origin.

        Given: A robot at (6, 3) with a wall at (5, 3) behind it (West).
        When: Casting a ray East (away from the wall).
        Then: East ray should not hit the wall (very large distance).

        Test type: unit
        """
        origin = np.array([6.0, 3.0])
        walls = np.array([[5.0, 3.0, 0.5, 0.5]])
        dists = ray_aabb_distances(origin, LASER_DIRECTIONS, walls)
        # East ray should not hit (wall is to the West)
        assert dists[2] > 100.0


class TestRayCircleDistance:
    """Tests for the ray_circle_distance function."""

    def test_ray_hits_circle(self):
        """Test that a ray directed at a circle hits it.

        Purpose: Validates basic ray-circle intersection.

        Given: A ray from (0, 0) going East and a circle at (5, 0) with radius 0.3.
        When: ray_circle_distance is called.
        Then: Distance should be approximately 4.7 (5 - 0.3).

        Test type: unit
        """
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        center = np.array([5.0, 0.0])
        radius = 0.3
        dist = ray_circle_distance(origin, direction, center, radius)
        assert abs(dist - 4.7) < 0.1

    def test_ray_misses_circle(self):
        """Test that a perpendicular ray misses the circle.

        Purpose: Validates that non-intersecting rays return infinity.

        Given: A ray from (0, 0) going North and a circle at (5, 0).
        When: ray_circle_distance is called.
        Then: Distance should be infinity.

        Test type: unit
        """
        origin = np.array([0.0, 0.0])
        direction = np.array([0.0, 1.0])
        center = np.array([5.0, 0.0])
        radius = 0.3
        dist = ray_circle_distance(origin, direction, center, radius)
        assert dist == np.inf

    def test_ray_origin_inside_circle(self):
        """Test ray-circle with origin inside the circle.

        Purpose: Validates behavior when ray starts inside the circle.

        Given: A ray starting inside a circle.
        When: ray_circle_distance is called.
        Then: Should return the exit distance (positive).

        Test type: unit
        """
        origin = np.array([5.0, 0.0])
        direction = np.array([1.0, 0.0])
        center = np.array([5.0, 0.0])
        radius = 1.0
        dist = ray_circle_distance(origin, direction, center, radius)
        assert dist > 0
        assert abs(dist - 1.0) < 0.1


class TestComputeLaserMeasurements:
    """Tests for the compute_laser_measurements function."""

    def test_measurements_shape(self):
        """Test that laser measurements return 8 values.

        Purpose: Validates output shape of laser measurements.

        Given: A robot position, opponent position, and grid setup.
        When: compute_laser_measurements is called.
        Then: Returns array of shape (8,).

        Test type: unit
        """
        robot = np.array([5.0, 3.0])
        opponent = np.array([8.0, 3.0])
        walls = np.empty((0, 4))
        grid = np.array([11.0, 7.0])
        m = compute_laser_measurements(robot, opponent, 0.3, walls, grid)
        assert m.shape == (8,)

    def test_measurements_positive(self):
        """Test that all measurements are non-negative.

        Purpose: Validates that distances are non-negative.

        Given: Valid robot and opponent positions.
        When: compute_laser_measurements is called.
        Then: All measurements should be >= 0.

        Test type: unit
        """
        robot = np.array([5.0, 3.0])
        opponent = np.array([8.0, 5.0])
        walls = np.array([[3.0, 3.0, 0.5, 0.5]])
        grid = np.array([11.0, 7.0])
        m = compute_laser_measurements(robot, opponent, 0.3, walls, grid)
        assert np.all(m >= 0)


class TestResolveWallCollision:
    """Tests for the resolve_wall_collision function."""

    def test_no_overlap_returns_same_position(self):
        """Test that non-overlapping entity is unchanged.

        Purpose: Validates that positions far from walls are unchanged.

        Given: A position far from any wall.
        When: resolve_wall_collision is called.
        Then: Position should be unchanged.

        Test type: unit
        """
        pos = np.array([5.0, 3.0])
        walls = np.array([[1.0, 1.0, 0.5, 0.5]])
        result = resolve_wall_collision(pos, 0.3, walls)
        np.testing.assert_array_almost_equal(result, pos)

    def test_overlap_pushes_entity_out(self):
        """Test that overlapping entity is pushed out of wall.

        Purpose: Validates collision resolution pushes entity outside wall.

        Given: A position overlapping a wall AABB.
        When: resolve_wall_collision is called.
        Then: The resolved position should not overlap the wall.

        Test type: unit
        """
        # Position right next to wall center
        pos = np.array([1.3, 1.0])
        walls = np.array([[1.0, 1.0, 0.5, 0.5]])
        result = resolve_wall_collision(pos, 0.3, walls)
        # Should be pushed to the right of the wall
        assert result[0] >= 1.5 + 0.3 - 0.01  # wall edge + radius

    def test_no_walls_returns_same(self):
        """Test with no walls.

        Purpose: Validates empty walls case.

        Given: No walls.
        When: resolve_wall_collision is called.
        Then: Position is unchanged.

        Test type: unit
        """
        pos = np.array([5.0, 3.0])
        walls = np.empty((0, 4))
        result = resolve_wall_collision(pos, 0.3, walls)
        np.testing.assert_array_almost_equal(result, pos)


class TestClampToGrid:
    """Tests for the clamp_to_grid function."""

    def test_position_within_grid_unchanged(self):
        """Test that a position inside the grid is unchanged.

        Purpose: Validates that interior positions pass through.

        Given: A position well within the grid.
        When: clamp_to_grid is called.
        Then: Position is unchanged.

        Test type: unit
        """
        pos = np.array([5.0, 3.0])
        grid = np.array([11.0, 7.0])
        result = clamp_to_grid(pos, 0.3, grid)
        np.testing.assert_array_almost_equal(result, pos)

    def test_position_below_zero_clamped(self):
        """Test that a position below grid boundary is clamped.

        Purpose: Validates lower boundary clamping.

        Given: A position with negative coordinates.
        When: clamp_to_grid is called.
        Then: Position is clamped to entity_radius.

        Test type: unit
        """
        pos = np.array([-1.0, -2.0])
        grid = np.array([11.0, 7.0])
        result = clamp_to_grid(pos, 0.3, grid)
        assert result[0] >= 0.3
        assert result[1] >= 0.3

    def test_position_above_grid_clamped(self):
        """Test that a position above grid boundary is clamped.

        Purpose: Validates upper boundary clamping.

        Given: A position beyond the grid.
        When: clamp_to_grid is called.
        Then: Position is clamped to grid_size - entity_radius.

        Test type: unit
        """
        pos = np.array([12.0, 8.0])
        grid = np.array([11.0, 7.0])
        result = clamp_to_grid(pos, 0.3, grid)
        assert result[0] <= 11.0 - 0.3
        assert result[1] <= 7.0 - 0.3


class TestBatchOperations:
    """Tests for batch geometry operations."""

    def test_batch_resolve_wall_collision_shape(self):
        """Test batch wall collision output shape.

        Purpose: Validates shape of batch collision resolution.

        Given: N positions and M walls.
        When: batch_resolve_wall_collision is called.
        Then: Output shape is (N, 2).

        Test type: unit
        """
        positions = np.random.rand(20, 2) * 10
        walls = np.array([[5.0, 3.0, 0.5, 0.5], [2.0, 2.0, 0.5, 0.5]])
        result = batch_resolve_wall_collision(positions, 0.3, walls)
        assert result.shape == (20, 2)

    def test_batch_clamp_to_grid_shape(self):
        """Test batch grid clamping output shape.

        Purpose: Validates shape of batch clamping.

        Given: N positions and grid size.
        When: batch_clamp_to_grid is called.
        Then: Output shape is (N, 2).

        Test type: unit
        """
        positions = np.random.rand(20, 2) * 15 - 2
        grid = np.array([11.0, 7.0])
        result = batch_clamp_to_grid(positions, 0.3, grid)
        assert result.shape == (20, 2)
        assert np.all(result[:, 0] >= 0.3)
        assert np.all(result[:, 0] <= 11.0 - 0.3)

    def test_batch_laser_measurements_shape(self):
        """Test batch laser measurements output shape.

        Purpose: Validates shape of batch laser measurements.

        Given: N robot/opponent position pairs.
        When: batch_laser_measurements is called.
        Then: Output shape is (N, 8).

        Test type: unit
        """
        n = 10
        robot = np.random.rand(n, 2) * 5 + 2
        opponent = np.random.rand(n, 2) * 5 + 2
        walls = np.array([[5.0, 3.0, 0.5, 0.5]])
        grid = np.array([11.0, 7.0])
        result = batch_laser_measurements(robot, opponent, 0.3, walls, grid)
        assert result.shape == (n, 8)
        assert np.all(result >= 0)
