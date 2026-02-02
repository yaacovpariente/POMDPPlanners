"""Geometry utilities for the Continuous LaserTag POMDP.

Provides ray-AABB intersection, ray-circle intersection, wall collision
resolution and grid clamping used by the continuous laser-tag environment
and its vectorized belief updater.

Wall AABBs are stored as rows ``(cx, cy, hx, hy)`` where ``(cx, cy)`` is
the center and ``(hx, hy)`` the half-extents.  Entity radii are used for
circle-AABB overlap tests during collision resolution.

Functions:
    ray_aabb_distances: Vectorized ray-AABB slab intersection for multiple
        rays originating from a single point against an array of AABBs.
    ray_circle_distance: Distance along a ray to the nearest intersection
        with a circle.
    compute_laser_measurements: Full 8-direction laser scan from a position.
    resolve_wall_collision: Push a circular entity out of overlapping AABBs.
    clamp_to_grid: Clamp a 2-D position to the grid boundaries.
"""

from __future__ import annotations

import numpy as np

# 8 laser ray unit-direction vectors: N, NE, E, SE, S, SW, W, NW
LASER_DIRECTIONS = np.array(
    [
        [0.0, 1.0],  # N
        [1.0, 1.0],  # NE
        [1.0, 0.0],  # E
        [1.0, -1.0],  # SE
        [0.0, -1.0],  # S
        [-1.0, -1.0],  # SW
        [-1.0, 0.0],  # W
        [-1.0, 1.0],  # NW
    ],
    dtype=np.float64,
)
# Normalize diagonal directions
_norms = np.linalg.norm(LASER_DIRECTIONS, axis=1, keepdims=True)
LASER_DIRECTIONS = LASER_DIRECTIONS / _norms

# Grid boundary for clipping rays (large enough for any practical grid)
_RAY_MAX = 1e4


def ray_aabb_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    walls: np.ndarray,
) -> np.ndarray:
    """Compute distances from *origin* along each ray to the nearest wall AABB.

    Uses the slab method.  For each of the *D* directions the minimum
    positive intersection distance across all *M* walls is returned.  If a
    ray does not hit any wall before ``_RAY_MAX`` the returned distance is
    ``_RAY_MAX``.

    Args:
        origin: Shape ``(2,)`` – ray origin ``(x, y)``.
        directions: Shape ``(D, 2)`` – unit direction vectors.
        walls: Shape ``(M, 4)`` – AABBs ``(cx, cy, hx, hy)``.

    Returns:
        Shape ``(D,)`` array of nearest intersection distances (positive).
    """
    if walls.shape[0] == 0:
        return np.full(directions.shape[0], _RAY_MAX)

    cx = walls[:, 0]
    cy = walls[:, 1]
    hx = walls[:, 2]
    hy = walls[:, 3]

    min_x = cx - hx  # (M,)
    max_x = cx + hx
    min_y = cy - hy
    max_y = cy + hy

    n_dirs = directions.shape[0]
    result = np.full(n_dirs, _RAY_MAX)

    for d in range(n_dirs):
        dx, dy = directions[d]
        t_min, t_max = _slab_intersect(origin[0], origin[1], dx, dy, min_x, max_x, min_y, max_y)
        # Valid hits: t_max > max(t_min, 0)
        hit_t = np.where(t_min > 0, t_min, t_max)
        valid = (t_max > np.maximum(t_min, 0.0)) & (hit_t > 1e-9)
        if np.any(valid):
            result[d] = np.min(hit_t[valid])
    return result


def _slab_intersect(ox, oy, dx, dy, min_x, max_x, min_y, max_y):
    m = len(min_x)

    # X slab
    if abs(dx) > 1e-12:
        inv_dx = 1.0 / dx
        tx1 = (min_x - ox) * inv_dx
        tx2 = (max_x - ox) * inv_dx
        t_enter_x = np.minimum(tx1, tx2)
        t_exit_x = np.maximum(tx1, tx2)
    else:
        # Ray parallel to Y axis: check if origin is within x range
        inside_x = (ox >= min_x) & (ox <= max_x)
        t_enter_x = np.where(inside_x, -np.inf, np.inf)
        t_exit_x = np.where(inside_x, np.inf, -np.inf)

    # Y slab
    if abs(dy) > 1e-12:
        inv_dy = 1.0 / dy
        ty1 = (min_y - oy) * inv_dy
        ty2 = (max_y - oy) * inv_dy
        t_enter_y = np.minimum(ty1, ty2)
        t_exit_y = np.maximum(ty1, ty2)
    else:
        # Ray parallel to X axis: check if origin is within y range
        inside_y = (oy >= min_y) & (oy <= max_y)
        t_enter_y = np.where(inside_y, -np.inf, np.inf)
        t_exit_y = np.where(inside_y, np.inf, -np.inf)

    t_min = np.maximum(t_enter_x, t_enter_y)
    t_max = np.minimum(t_exit_x, t_exit_y)
    return t_min, t_max


def ray_circle_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    """Distance along a ray to the nearest intersection with a circle.

    Args:
        origin: Shape ``(2,)`` – ray origin.
        direction: Shape ``(2,)`` – unit direction.
        center: Shape ``(2,)`` – circle center.
        radius: Circle radius.

    Returns:
        Positive intersection distance, or ``np.inf`` if no hit.
    """
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return np.inf
    sqrt_disc = np.sqrt(disc)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc
    if t1 > 1e-9:
        return t1
    if t2 > 1e-9:
        return t2
    return np.inf


def compute_laser_measurements(
    robot_pos: np.ndarray,
    opponent_pos: np.ndarray,
    opponent_radius: float,
    walls: np.ndarray,
    grid_size: np.ndarray,
) -> np.ndarray:
    """Compute 8-direction laser measurements from the robot.

    Each measurement is the distance to the nearest obstacle (wall AABB,
    opponent circle, or grid boundary) along the corresponding ray in
    :data:`LASER_DIRECTIONS`.

    Args:
        robot_pos: Shape ``(2,)`` – robot ``(x, y)``.
        opponent_pos: Shape ``(2,)`` – opponent ``(x, y)``.
        opponent_radius: Opponent body radius.
        walls: Shape ``(M, 4)`` – wall AABBs.
        grid_size: Shape ``(2,)`` – ``(width, height)`` of the arena.

    Returns:
        Shape ``(8,)`` array of distances.
    """
    # Wall distances
    wall_dists = ray_aabb_distances(robot_pos, LASER_DIRECTIONS, walls)

    # Grid boundary distances
    boundary_dists = _grid_boundary_distances(robot_pos, LASER_DIRECTIONS, grid_size)
    wall_dists = np.minimum(wall_dists, boundary_dists)

    # Opponent distances
    for d in range(8):
        opp_d = ray_circle_distance(robot_pos, LASER_DIRECTIONS[d], opponent_pos, opponent_radius)
        wall_dists[d] = min(wall_dists[d], opp_d)

    return wall_dists


def _grid_boundary_distances(
    origin: np.ndarray, directions: np.ndarray, grid_size: np.ndarray
) -> np.ndarray:
    n_dirs = directions.shape[0]
    result = np.full(n_dirs, _RAY_MAX)
    for d in range(n_dirs):
        dx, dy = directions[d]
        ts = []
        if abs(dx) > 1e-12:
            t_left = -origin[0] / dx
            t_right = (grid_size[0] - origin[0]) / dx
            if t_left > 1e-9:
                ts.append(t_left)
            if t_right > 1e-9:
                ts.append(t_right)
        if abs(dy) > 1e-12:
            t_bottom = -origin[1] / dy
            t_top = (grid_size[1] - origin[1]) / dy
            if t_bottom > 1e-9:
                ts.append(t_bottom)
            if t_top > 1e-9:
                ts.append(t_top)
        if ts:
            result[d] = min(ts)
    return result


def resolve_wall_collision(
    position: np.ndarray,
    entity_radius: float,
    walls: np.ndarray,
) -> np.ndarray:
    """Push a circular entity out of any overlapping wall AABBs.

    For each wall, if the entity circle overlaps the AABB, the entity is
    pushed along the axis of minimum penetration.

    Args:
        position: Shape ``(2,)`` – entity center.
        entity_radius: Entity body radius.
        walls: Shape ``(M, 4)`` – wall AABBs.

    Returns:
        Resolved position as shape ``(2,)`` array.
    """
    if walls.shape[0] == 0:
        return position.copy()

    pos = position.copy()
    for i in range(walls.shape[0]):
        pos = _resolve_single_wall(pos, entity_radius, walls[i])
    return pos


def _resolve_single_wall(pos: np.ndarray, radius: float, wall: np.ndarray) -> np.ndarray:
    cx, cy, hx, hy = wall
    # Closest point on AABB to circle center
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
        # Center is inside the AABB – push out along min-penetration axis
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


def clamp_to_grid(
    position: np.ndarray,
    entity_radius: float,
    grid_size: np.ndarray,
) -> np.ndarray:
    """Clamp a position so the entity circle stays within ``[0, w] x [0, h]``.

    Args:
        position: Shape ``(2,)`` – entity center.
        entity_radius: Entity body radius.
        grid_size: Shape ``(2,)`` – ``(width, height)`` of the arena.

    Returns:
        Clamped position as shape ``(2,)`` array.
    """
    result = position.copy()
    result[0] = np.clip(result[0], entity_radius, grid_size[0] - entity_radius)
    result[1] = np.clip(result[1], entity_radius, grid_size[1] - entity_radius)
    return result


# ---------------------------------------------------------------------------
# Vectorized helpers for the belief updater
# ---------------------------------------------------------------------------


def batch_resolve_wall_collision(
    positions: np.ndarray,
    entity_radius: float,
    walls: np.ndarray,
) -> np.ndarray:
    """Resolve wall collisions for an array of positions.

    Args:
        positions: Shape ``(N, 2)``.
        entity_radius: Entity body radius.
        walls: Shape ``(M, 4)``.

    Returns:
        Shape ``(N, 2)`` resolved positions.
    """
    if walls.shape[0] == 0:
        return positions.copy()

    result = positions.copy()
    for i in range(walls.shape[0]):
        result = _batch_resolve_single_wall(result, entity_radius, walls[i])
    return result


def _batch_resolve_single_wall(pos: np.ndarray, radius: float, wall: np.ndarray) -> np.ndarray:
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

    # Entities whose center is outside the wall AABB but within radius
    outside_idx = ov[~inside]
    if outside_idx.size > 0:
        d = dist[~inside]
        o = radius - d
        result[outside_idx, 0] += (dx[outside_idx] / d) * o
        result[outside_idx, 1] += (dy[outside_idx] / d) * o

    # Entities whose center is inside the wall AABB
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


def batch_clamp_to_grid(
    positions: np.ndarray,
    entity_radius: float,
    grid_size: np.ndarray,
) -> np.ndarray:
    """Clamp an array of positions to the grid.

    Args:
        positions: Shape ``(N, 2)``.
        entity_radius: Entity body radius.
        grid_size: Shape ``(2,)`` – ``(width, height)``.

    Returns:
        Shape ``(N, 2)`` clamped positions.
    """
    result = positions.copy()
    result[:, 0] = np.clip(result[:, 0], entity_radius, grid_size[0] - entity_radius)
    result[:, 1] = np.clip(result[:, 1], entity_radius, grid_size[1] - entity_radius)
    return result


def batch_laser_measurements(
    robot_positions: np.ndarray,
    opponent_positions: np.ndarray,
    opponent_radius: float,
    walls: np.ndarray,
    grid_size: np.ndarray,
) -> np.ndarray:
    """Compute 8-direction laser measurements for many particles.

    Args:
        robot_positions: Shape ``(N, 2)``.
        opponent_positions: Shape ``(N, 2)``.
        opponent_radius: Opponent body radius.
        walls: Shape ``(M, 4)`` – wall AABBs.
        grid_size: Shape ``(2,)`` – ``(width, height)``.

    Returns:
        Shape ``(N, 8)`` measurement array.
    """
    n = robot_positions.shape[0]
    measurements = np.empty((n, 8))
    for i in range(n):
        measurements[i] = compute_laser_measurements(
            robot_positions[i],
            opponent_positions[i],
            opponent_radius,
            walls,
            grid_size,
        )
    return measurements
