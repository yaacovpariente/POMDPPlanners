# SPDX-License-Identifier: MIT

"""Tests for Continuous Push POMDP geometry utilities.

This module tests circle-AABB overlap, point-inside-AABB, collision resolution,
and grid clamping functions, including their batch variants.
"""

import numpy as np

from POMDPPlanners.environments.push_pomdp.continuous_push_geometry import (
    batch_clamp_circle_to_grid,
    batch_clamp_point_to_grid,
    batch_point_inside_aabb,
    batch_resolve_circle_wall_collision,
    circle_aabb_overlap,
    clamp_circle_to_grid,
    clamp_point_to_grid,
    point_inside_aabb,
    resolve_circle_wall_collision,
)


class TestContinuousPushGeometry:
    """Test circle-AABB and point-AABB geometry utilities."""

    def setup_method(self):
        """Set up shared test fixtures."""
        # wall: center (5, 5), half-size 1  ->  AABB [4, 6] x [4, 6]
        self.wall = np.array([5.0, 5.0, 1.0, 1.0])
        self.walls = self.wall.reshape(1, 4)

    def test_circle_aabb_overlap_true(self):
        """Test that overlapping circle-AABB returns True.

        Purpose: Validates circle-AABB overlap detection.

        Given: A circle centered at (3.5, 5.0) with radius 1.0 and
               an AABB centered at (5, 5) with half-extent 1.
        When: circle_aabb_overlap is called.
        Then: Returns True because the circle reaches into the AABB.

        Test type: unit
        """
        assert circle_aabb_overlap(np.array([3.5, 5.0]), 1.0, self.wall)

    def test_circle_aabb_overlap_false(self):
        """Test that non-overlapping circle-AABB returns False.

        Purpose: Validates that distant circles are not detected as overlapping.

        Given: A circle centered at (1.0, 1.0) with radius 0.3 and an AABB
               centered at (5, 5) with half-extent 1.
        When: circle_aabb_overlap is called.
        Then: Returns False.

        Test type: unit
        """
        assert not circle_aabb_overlap(np.array([1.0, 1.0]), 0.3, self.wall)

    def test_point_inside_aabb_true(self):
        """Test that a point inside an AABB is detected.

        Purpose: Validates point-inside-AABB test.

        Given: A point at (5.0, 5.0) inside the AABB [4, 6] x [4, 6].
        When: point_inside_aabb is called.
        Then: Returns True.

        Test type: unit
        """
        assert point_inside_aabb(np.array([5.0, 5.0]), self.wall)

    def test_point_inside_aabb_false(self):
        """Test that a point outside an AABB is not detected.

        Purpose: Validates point-outside-AABB.

        Given: A point at (1.0, 1.0) outside the AABB [4, 6] x [4, 6].
        When: point_inside_aabb is called.
        Then: Returns False.

        Test type: unit
        """
        assert not point_inside_aabb(np.array([1.0, 1.0]), self.wall)

    def test_resolve_circle_wall_collision_pushes_out(self):
        """Test that collision resolution pushes circle away from AABB.

        Purpose: Validates that an overlapping circle is pushed out.

        Given: A circle centered at (4.5, 5.0) with radius 0.3 overlapping
               the AABB [4, 6] x [4, 6].
        When: resolve_circle_wall_collision is called.
        Then: The returned position moves the circle center away from
              the original overlap.

        Test type: unit
        """
        pos = np.array([4.5, 5.0])
        radius = 0.3
        # Circle center is inside the AABB, so it will be pushed out
        resolved = resolve_circle_wall_collision(pos, radius, self.walls)
        # Resolved position should differ from original
        assert not np.array_equal(resolved, pos)
        # The circle should be pushed to the left (nearest edge)
        assert resolved[0] < pos[0]

    def test_resolve_circle_wall_collision_no_walls(self):
        """Test collision resolution with no walls returns copy.

        Purpose: Validates no-op when there are no walls.

        Given: Empty walls array.
        When: resolve_circle_wall_collision is called.
        Then: Returns a copy of the original position.

        Test type: unit
        """
        pos = np.array([3.0, 3.0])
        result = resolve_circle_wall_collision(pos, 0.3, np.empty((0, 4)))
        np.testing.assert_array_equal(result, pos)

    def test_clamp_circle_to_grid(self):
        """Test that circle clamping keeps circle within grid.

        Purpose: Validates grid clamping for circles.

        Given: A circle centered at (-1, 12) with radius 0.3 on a grid of size 10.
        When: clamp_circle_to_grid is called.
        Then: Center is clamped to [radius, grid_size - 1 - radius].

        Test type: unit
        """
        pos = np.array([-1.0, 12.0])
        result = clamp_circle_to_grid(pos, 0.3, 10)
        assert result[0] >= 0.3
        assert result[1] <= 10 - 1 - 0.3

    def test_clamp_point_to_grid(self):
        """Test that point clamping keeps point within [0, grid_size-1].

        Purpose: Validates grid clamping for points.

        Given: A point at (-2, 15) on a grid of size 10.
        When: clamp_point_to_grid is called.
        Then: Point is clamped to [0, 9].

        Test type: unit
        """
        pos = np.array([-2.0, 15.0])
        result = clamp_point_to_grid(pos, 10)
        np.testing.assert_array_equal(result, np.array([0.0, 9.0]))

    # -- Batch variants --

    def test_batch_resolve_circle_wall_collision_shapes(self):
        """Test batch circle-wall collision resolution output shape.

        Purpose: Validates output shape of batch collision resolution.

        Given: 10 random positions with radius 0.5 and one wall.
        When: batch_resolve_circle_wall_collision is called.
        Then: Output shape matches input shape (10, 2).

        Test type: unit
        """
        positions = np.random.uniform(0, 9, (10, 2))
        result = batch_resolve_circle_wall_collision(positions, 0.5, self.walls)
        assert result.shape == (10, 2)

    def test_batch_clamp_circle_to_grid_shapes(self):
        """Test batch circle grid clamping output shape.

        Purpose: Validates output shape of batch circle clamping.

        Given: 10 random positions.
        When: batch_clamp_circle_to_grid is called.
        Then: Output shape matches (10, 2) and values are in range.

        Test type: unit
        """
        positions = np.random.uniform(-5, 15, (10, 2))
        result = batch_clamp_circle_to_grid(positions, 0.3, 10)
        assert result.shape == (10, 2)
        assert np.all(result >= 0.3)
        assert np.all(result <= 10 - 1 - 0.3)

    def test_batch_point_inside_aabb(self):
        """Test batch point-inside-AABB detection.

        Purpose: Validates batch point-inside-AABB returns correct booleans.

        Given: Points at (5, 5) inside and (0, 0) outside the AABB.
        When: batch_point_inside_aabb is called.
        Then: First is True, second is False.

        Test type: unit
        """
        points = np.array([[5.0, 5.0], [0.0, 0.0]])
        result = batch_point_inside_aabb(points, self.walls)
        assert result[0] is np.True_
        assert result[1] is np.False_

    def test_batch_clamp_point_to_grid(self):
        """Test batch point grid clamping.

        Purpose: Validates batch point clamping bounds.

        Given: Points at (-1, 20) and (5, 5).
        When: batch_clamp_point_to_grid is called with grid_size=10.
        Then: First is clamped to (0, 9), second unchanged.

        Test type: unit
        """
        points = np.array([[-1.0, 20.0], [5.0, 5.0]])
        result = batch_clamp_point_to_grid(points, 10)
        np.testing.assert_array_equal(result[0], [0.0, 9.0])
        np.testing.assert_array_equal(result[1], [5.0, 5.0])
