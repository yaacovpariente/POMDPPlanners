"""Geometry utilities for the Continuous Push POMDP.

Provides circle-AABB overlap tests, point-in-AABB tests, collision
resolution and grid clamping used by the continuous push environment
and its vectorized belief updater.

Wall AABBs (obstacles) are stored as rows ``(cx, cy, hx, hy)`` where
``(cx, cy)`` is the center and ``(hx, hy)`` the half-extents.  For
square obstacles ``hx == hy``.

Functions:
    circle_aabb_overlap: Boolean circle-AABB overlap test.
    point_inside_aabb: Boolean point-inside-AABB test.
    resolve_circle_wall_collision: Push a circle out of overlapping AABBs.
    clamp_circle_to_grid: Clamp a circle center so the circle stays in-bounds.
    clamp_point_to_grid: Clamp a point to ``[0, grid_size-1]``.
    batch_resolve_circle_wall_collision: Vectorized circle-AABB resolution.
    batch_clamp_circle_to_grid: Vectorized circle grid clamping.
    batch_point_inside_aabb: Vectorized point-inside-AABB test.
    batch_clamp_point_to_grid: Vectorized point grid clamping.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Single-entity helpers
# ------------------------------------------------------------------


def circle_aabb_overlap(
    center: np.ndarray,
    radius: float,
    wall: np.ndarray,
) -> bool:
    """Test whether a circle overlaps an axis-aligned bounding box.

    Args:
        center: Shape ``(2,)`` -- circle center ``(x, y)``.
        radius: Circle radius.
        wall: Shape ``(4,)`` -- AABB ``(cx, cy, hx, hy)``.

    Returns:
        ``True`` if the circle and AABB overlap.
    """
    cx, cy, hx, hy = wall
    closest_x = np.clip(center[0], cx - hx, cx + hx)
    closest_y = np.clip(center[1], cy - hy, cy + hy)
    dx = center[0] - closest_x
    dy = center[1] - closest_y
    return bool(dx * dx + dy * dy < radius * radius)


def point_inside_aabb(point: np.ndarray, wall: np.ndarray) -> bool:
    """Test whether a point lies inside an axis-aligned bounding box.

    Args:
        point: Shape ``(2,)`` -- point ``(x, y)``.
        wall: Shape ``(4,)`` -- AABB ``(cx, cy, hx, hy)``.

    Returns:
        ``True`` if the point is inside the AABB.
    """
    cx, cy, hx, hy = wall
    return bool((cx - hx) <= point[0] <= (cx + hx) and (cy - hy) <= point[1] <= (cy + hy))


def resolve_circle_wall_collision(
    pos: np.ndarray,
    radius: float,
    walls: np.ndarray,
) -> np.ndarray:
    """Push a circular entity out of any overlapping wall AABBs.

    For each wall, if the entity circle overlaps the AABB, the entity is
    pushed along the axis of minimum penetration.

    Args:
        pos: Shape ``(2,)`` -- entity center.
        radius: Entity body radius.
        walls: Shape ``(M, 4)`` -- wall AABBs.

    Returns:
        Resolved position as shape ``(2,)`` array.
    """
    if walls.shape[0] == 0:
        return pos.copy()
    result = pos.copy()
    for i in range(walls.shape[0]):
        result = _resolve_single_circle_wall(result, radius, walls[i])
    return result


def _resolve_single_circle_wall(pos: np.ndarray, radius: float, wall: np.ndarray) -> np.ndarray:
    cx, cy, hx, hy = wall
    closest_x = np.clip(pos[0], cx - hx, cx + hx)
    closest_y = np.clip(pos[1], cy - hy, cy + hy)

    dx = pos[0] - closest_x
    dy = pos[1] - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return pos  # no overlap

    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
    result = pos.copy()

    if dist < 1e-12:
        pen_left = pos[0] - (cx - hx)
        pen_right = (cx + hx) - pos[0]
        pen_down = pos[1] - (cy - hy)
        pen_up = (cy + hy) - pos[1]
        min_pen = min(pen_left, pen_right, pen_down, pen_up)
        if min_pen == pen_left:
            result[0] = cx - hx - radius
        elif min_pen == pen_right:
            result[0] = cx + hx + radius
        elif min_pen == pen_down:
            result[1] = cy - hy - radius
        else:
            result[1] = cy + hy + radius
    else:
        overlap = radius - dist
        result[0] += (dx / dist) * overlap
        result[1] += (dy / dist) * overlap

    return result


def clamp_circle_to_grid(
    pos: np.ndarray,
    radius: float,
    grid_size: float,
) -> np.ndarray:
    """Clamp a circle center so the full circle stays within ``[0, grid_size-1]``.

    Args:
        pos: Shape ``(2,)`` -- circle center.
        radius: Circle radius.
        grid_size: Grid dimension (positions valid in ``[0, grid_size-1]``).

    Returns:
        Clamped position as shape ``(2,)`` array.
    """
    result = pos.copy()
    lo = radius
    hi = grid_size - 1 - radius
    result[0] = np.clip(result[0], lo, hi)
    result[1] = np.clip(result[1], lo, hi)
    return result


def clamp_point_to_grid(pos: np.ndarray, grid_size: float) -> np.ndarray:
    """Clamp a point to ``[0, grid_size-1]``.

    Args:
        pos: Shape ``(2,)`` -- point position.
        grid_size: Grid dimension.

    Returns:
        Clamped position as shape ``(2,)`` array.
    """
    return np.clip(pos, 0.0, grid_size - 1)


# ------------------------------------------------------------------
# Batch helpers (for vectorized belief updater)
# ------------------------------------------------------------------


def batch_resolve_circle_wall_collision(
    positions: np.ndarray,
    radius: float,
    walls: np.ndarray,
) -> np.ndarray:
    """Resolve circle-AABB collisions for an array of positions.

    Args:
        positions: Shape ``(N, 2)``.
        radius: Circle radius.
        walls: Shape ``(M, 4)`` -- wall AABBs.

    Returns:
        Shape ``(N, 2)`` resolved positions.
    """
    if walls.shape[0] == 0:
        return positions.copy()
    result = positions.copy()
    for i in range(walls.shape[0]):
        result = _batch_resolve_single_circle_wall(result, radius, walls[i])
    return result


def _batch_resolve_single_circle_wall(
    pos: np.ndarray, radius: float, wall: np.ndarray
) -> np.ndarray:
    cx, cy, hx, hy = wall
    closest_x = np.clip(pos[:, 0], cx - hx, cx + hx)
    closest_y = np.clip(pos[:, 1], cy - hy, cy + hy)

    dx = pos[:, 0] - closest_x
    dy = pos[:, 1] - closest_y
    dist_sq = dx * dx + dy * dy
    r_sq = radius * radius

    overlap_mask = dist_sq < r_sq
    if not np.any(overlap_mask):
        return pos

    result = pos.copy()
    ov = np.where(overlap_mask)[0]
    dist = np.sqrt(np.maximum(dist_sq[ov], 1e-24))
    inside = dist < 1e-12

    outside_idx = ov[~inside]
    if outside_idx.size > 0:
        d = dist[~inside]
        o = radius - d
        result[outside_idx, 0] += (dx[outside_idx] / d) * o
        result[outside_idx, 1] += (dy[outside_idx] / d) * o

    inside_idx = ov[inside]
    if inside_idx.size > 0:
        for j in inside_idx:
            pen_left = pos[j, 0] - (cx - hx)
            pen_right = (cx + hx) - pos[j, 0]
            pen_down = pos[j, 1] - (cy - hy)
            pen_up = (cy + hy) - pos[j, 1]
            min_pen = min(pen_left, pen_right, pen_down, pen_up)
            if min_pen == pen_left:
                result[j, 0] = cx - hx - radius
            elif min_pen == pen_right:
                result[j, 0] = cx + hx + radius
            elif min_pen == pen_down:
                result[j, 1] = cy - hy - radius
            else:
                result[j, 1] = cy + hy + radius

    return result


def batch_clamp_circle_to_grid(
    positions: np.ndarray,
    radius: float,
    grid_size: float,
) -> np.ndarray:
    """Clamp an array of circle centers so circles stay in-bounds.

    Args:
        positions: Shape ``(N, 2)``.
        radius: Circle radius.
        grid_size: Grid dimension.

    Returns:
        Shape ``(N, 2)`` clamped positions.
    """
    lo = radius
    hi = grid_size - 1 - radius
    result = positions.copy()
    result[:, 0] = np.clip(result[:, 0], lo, hi)
    result[:, 1] = np.clip(result[:, 1], lo, hi)
    return result


def batch_point_inside_aabb(
    points: np.ndarray,
    walls: np.ndarray,
) -> np.ndarray:
    """Test whether each point lies inside any AABB.

    Args:
        points: Shape ``(N, 2)``.
        walls: Shape ``(M, 4)`` -- AABBs.

    Returns:
        Shape ``(N,)`` boolean array -- ``True`` where the point is inside
        at least one AABB.
    """
    if walls.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    cx = walls[:, 0]
    cy = walls[:, 1]
    hx = walls[:, 2]
    hy = walls[:, 3]
    # (N, 1) vs (M,) -> (N, M)
    in_x = (points[:, 0:1] >= (cx - hx)) & (points[:, 0:1] <= (cx + hx))
    in_y = (points[:, 1:2] >= (cy - hy)) & (points[:, 1:2] <= (cy + hy))
    return np.any(in_x & in_y, axis=1)


def batch_clamp_point_to_grid(
    positions: np.ndarray,
    grid_size: float,
) -> np.ndarray:
    """Clamp an array of points to ``[0, grid_size-1]``.

    Args:
        positions: Shape ``(N, 2)``.
        grid_size: Grid dimension.

    Returns:
        Shape ``(N, 2)`` clamped positions.
    """
    return np.clip(positions, 0.0, grid_size - 1)
