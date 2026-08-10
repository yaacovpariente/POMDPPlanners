# SPDX-License-Identifier: MIT

"""Exteroceptive observation models over IsaacLab's ``RayCaster`` sensors.

IsaacLab ships two ray-cast sensors that dominate legged-robot tasks: a planar LiDAR ring and a
downward height scanner. Both are functions of the robot's planar pose and the scene geometry, so
both are modelled here against the same obstacle description — a set of discs on the floor with a
height. That is deliberately the *same* geometry a hazard zone uses, so a study that turns its
zones into real obstacle prims gets a sensor that can actually see them.

The obstacle set is injected rather than read from the simulator: a planner-side generative model
must be able to cast a ray from a hypothetical state with no live Isaac process attached, and
these models are used inside the search tree where there is none.

Functions:
    grid_scan_pattern: Body-frame grid offsets in IsaacLab's own cell order.

Classes:
    RayCasterObservationModel: Planar LiDAR ranges to a disc obstacle set, with Gaussian noise.
    HeightScanObservationModel: Downward height samples on a body-frame grid, with Gaussian noise.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    register_observation_model,
)


def _planar_pose(
    clean_state: Mapping[str, np.ndarray], channel: str, indices: Sequence[int]
) -> Tuple[np.ndarray, float]:
    """Read ``(x, y)`` and the heading from a pose block."""
    block = np.asarray(clean_state[channel], dtype=float).reshape(-1)
    return block[list(indices[:2])], float(block[indices[2]])


def _disc_array(values: Optional[ArrayLike], width: int) -> np.ndarray:
    """Normalize an obstacle description to a ``(n, width)`` (or ``(n,)``) float array."""
    if values is None:
        return np.zeros((0, width)) if width > 1 else np.zeros(0)
    array = np.asarray(values, dtype=float)
    if width > 1:
        return array.reshape(-1, width)
    return array.reshape(-1)


def _ray_disc_ranges(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Distance along each unit ray to the nearest disc, or ``max_range`` when nothing is hit.

    Args:
        origin: Ray origin, shape ``(2,)``.
        directions: Unit ray directions, shape ``(R, 2)``.
        centers: Disc centres, shape ``(D, 2)``.
        radii: Disc radii, shape ``(D,)``.
        max_range: Range reported when a ray hits nothing within it.

    Returns:
        Shape ``(R,)`` of ranges in ``[0, max_range]``.
    """
    if centers.shape[0] == 0:
        return np.full(directions.shape[0], float(max_range))

    # Solve |origin + t * direction - centre|^2 = radius^2 for every (ray, disc) pair.
    offset = origin[np.newaxis, :] - centers  # (D, 2), disc centre -> ray origin
    projection = np.asarray(directions @ offset.T, dtype=float)  # (R, D)
    squared_gap = (np.sum(offset**2, axis=-1) - radii**2)[np.newaxis, :]  # (1, D)
    discriminant = projection**2 - squared_gap

    hit = discriminant >= 0.0
    root = np.sqrt(np.where(hit, discriminant, 0.0))
    near = -projection - root
    far = -projection + root
    # A ray starting inside a disc has near < 0; the first surface ahead is then ``far``.
    distance = np.where(near >= 0.0, near, far)
    valid = hit & (distance >= 0.0)
    distance = np.where(valid, distance, np.inf)
    return np.minimum(distance.min(axis=1), float(max_range))


def grid_scan_pattern(size: Tuple[float, float], resolution: float) -> np.ndarray:
    """Body-frame grid offsets matching IsaacLab's ``GridPatternCfg`` cell order.

    The order matters and is not obvious. IsaacLab builds the grid with ``meshgrid(x, y,
    indexing="xy")`` and flattens it, so the reading runs ``x`` fastest within each ``y`` row. A
    model that lays its own grid out ``y``-fastest produces a permuted prediction, every particle
    is scored against the wrong cells, and the belief degrades in a way that looks like sensor
    noise rather than an indexing bug.

    Args:
        size: Grid extent ``(length, width)`` in metres, the sensor's own ``size``.
        resolution: Grid spacing in metres.

    Returns:
        Shape ``(M, 2)`` of ``(x, y)`` offsets, in the sensor's cell order.

    Raises:
        ValueError: If ``resolution`` is not positive.

    Example:
        >>> grid_scan_pattern((0.2, 0.1), 0.1).tolist()
        [[-0.1, -0.05], [0.0, -0.05], [0.1, -0.05], [-0.1, 0.05], [0.0, 0.05], [0.1, 0.05]]
    """
    if resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    axis_x = np.arange(-0.5 * size[0], 0.5 * size[0] + 1e-9, resolution)
    axis_y = np.arange(-0.5 * size[1], 0.5 * size[1] + 1e-9, resolution)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y, indexing="xy")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)


@register_observation_model("ray_caster")
class RayCasterObservationModel(IsaacObservationModel):
    """Planar LiDAR ranges to a disc obstacle set, corrupted by Gaussian range noise.

    A ring of rays is cast from the robot's planar position, evenly spaced over a field of view
    centred on its heading. Each ray reports the distance to the nearest obstacle disc, saturating
    at ``max_range`` when it hits nothing.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The single pose block read, as a one-tuple.
        num_rays: Number of rays in the ring.
        max_range: Range reported by a ray that hits nothing.
        range_std: Std of the Gaussian noise added to each range.

    Note:
        Ranges are clipped to ``[0, max_range]`` after the noise is added, so the density is the
        unclipped Gaussian and is therefore approximate for a reading sitting exactly on a bound.
        With ``range_std`` small against ``max_range`` — the regime a usable LiDAR is in — the
        misallocated mass is negligible; the alternative, a censored likelihood, would put an atom
        at the bound that a particle filter then has to resolve.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = RayCasterObservationModel(
        ...     channel="lidar", pose_channel="base_pose", num_rays=4,
        ...     max_range=5.0, obstacle_centers=[(1.0, 0.0)], obstacle_radii=[0.5])
        >>> ranges = model.perceive({"base_pose": np.array([0.0, 0.0, 0.0])})
        >>> ranges.shape
        (4,)
        >>> bool(ranges[0] < 1.0)  # the forward ray sees the disc at 0.5 m
        True
    """

    supports_density = True

    def __init__(
        self,
        channel: str,
        pose_channel: str = "base_pose",
        pose_indices: Tuple[int, int, int] = (0, 1, 2),
        num_rays: int = 16,
        max_range: float = 5.0,
        range_std: float = 0.05,
        field_of_view: float = 2.0 * np.pi,
        obstacle_centers: Optional[ArrayLike] = None,
        obstacle_radii: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize the ray-caster model.

        Args:
            channel: The observation-dict key this model produces.
            pose_channel: The state block holding the planar pose.
            pose_indices: Positions of ``(x, y, yaw)`` within that block.
            num_rays: Number of rays in the ring.
            max_range: Range reported by a ray that hits nothing.
            range_std: Std of the Gaussian noise on each range.
            field_of_view: Angular extent of the ring, in radians, centred on the heading. The
                default is a full circle.
            obstacle_centers: Disc centres, shape ``(D, 2)``. Defaults to no obstacles.
            obstacle_radii: Disc radii, shape ``(D,)``. Defaults to no obstacles.

        Raises:
            ValueError: If the obstacle centres and radii disagree in count, or if ``num_rays``,
                ``max_range`` or ``range_std`` is not positive.
        """
        if num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {num_rays}")
        if max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {max_range}")
        if range_std <= 0.0:
            raise ValueError(f"range_std must be positive, got {range_std}")

        self.channel = channel
        self.pose_channel = pose_channel
        self.state_channels = (pose_channel,)
        self.pose_indices = tuple(int(index) for index in pose_indices)
        self.num_rays = int(num_rays)
        self.max_range = float(max_range)
        self.range_std = float(range_std)
        self.field_of_view = float(field_of_view)
        self.obstacle_centers = _disc_array(obstacle_centers, 2)
        self.obstacle_radii = _disc_array(obstacle_radii, 1)
        if self.obstacle_centers.shape[0] != self.obstacle_radii.shape[0]:
            raise ValueError(
                f"obstacle_centers has {self.obstacle_centers.shape[0]} entries but "
                f"obstacle_radii has {self.obstacle_radii.shape[0]}"
            )
        # Ray bearings relative to the heading; the heading is added per query.
        self._bearings = self._ring_bearings()

    def _ring_bearings(self) -> np.ndarray:
        if self.field_of_view >= 2.0 * np.pi:
            return np.linspace(0.0, 2.0 * np.pi, self.num_rays, endpoint=False)
        half = 0.5 * self.field_of_view
        return np.linspace(-half, half, self.num_rays)

    def clean_ranges(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Noise-free ranges from the pose in ``clean_state``.

        Args:
            clean_state: The state's named blocks; the pose block is read.

        Returns:
            Shape ``(num_rays,)`` of noise-free ranges.
        """
        position, heading = _planar_pose(clean_state, self.pose_channel, self.pose_indices)
        angles = self._bearings + heading
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        return _ray_disc_ranges(
            position, directions, self.obstacle_centers, self.obstacle_radii, self.max_range
        )

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        noisy = self.clean_ranges(clean_state) + np.random.normal(
            0.0, self.range_std, size=self.num_rays
        )
        return np.clip(noisy, 0.0, self.max_range)

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        truth = self.clean_ranges(clean_state)
        reading = np.asarray(channel_observation, dtype=float).reshape(-1)
        if reading.shape != truth.shape:
            return float("-inf")
        residual = (reading - truth) / self.range_std
        return float(
            -0.5 * np.sum(residual**2)
            - self.num_rays * (np.log(self.range_std) + 0.5 * np.log(2.0 * np.pi))
        )


@register_observation_model("height_scan")
class HeightScanObservationModel(IsaacObservationModel):
    """Downward height samples on a body-frame grid, corrupted by Gaussian noise.

    The scanner reports, at each grid point, the height of whatever is underneath it: an
    obstacle's height when the point falls inside that obstacle's disc, and the floor otherwise.
    Unlike the LiDAR it is unaffected by occlusion, which makes it the cheaper channel for telling
    "am I standing on the hazard" apart from "am I near it".

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The single pose block read, as a one-tuple.
        pattern: Body-frame grid offsets, shape ``(M, 2)``.
        height_std: Std of the Gaussian noise added to each height sample.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = HeightScanObservationModel(
        ...     channel="height_scan", pose_channel="base_pose",
        ...     pattern=[(0.0, 0.0)], obstacle_centers=[(0.0, 0.0)],
        ...     obstacle_radii=[0.5], obstacle_heights=[0.3])
        >>> heights = model.perceive({"base_pose": np.zeros(3)})
        >>> heights.shape
        (1,)
        >>> bool(abs(heights[0] - 0.3) < 0.2)
        True
    """

    supports_density = True

    def __init__(
        self,
        channel: str,
        pose_channel: str = "base_pose",
        pose_indices: Tuple[int, int, int] = (0, 1, 2),
        pattern: Optional[ArrayLike] = None,
        grid_extent: float = 0.8,
        grid_size: int = 3,
        height_std: float = 0.02,
        obstacle_centers: Optional[ArrayLike] = None,
        obstacle_radii: Optional[ArrayLike] = None,
        obstacle_heights: Optional[ArrayLike] = None,
        floor_height: float = 0.0,
    ) -> None:
        """Initialize the height-scan model.

        Args:
            channel: The observation-dict key this model produces.
            pose_channel: The state block holding the planar pose.
            pose_indices: Positions of ``(x, y, yaw)`` within that block.
            pattern: Explicit body-frame grid offsets, shape ``(M, 2)``. Defaults to a square
                ``grid_size`` x ``grid_size`` grid spanning ``grid_extent`` metres.
            grid_extent: Side length of the default grid, in metres.
            grid_size: Points per side of the default grid.
            height_std: Std of the Gaussian noise on each height sample.
            obstacle_centers: Disc centres, shape ``(D, 2)``. Defaults to no obstacles.
            obstacle_radii: Disc radii, shape ``(D,)``. Defaults to no obstacles.
            obstacle_heights: Disc heights, shape ``(D,)``. Defaults to 1.0 for every disc.
            floor_height: Height reported where no obstacle covers the sample.

        Raises:
            ValueError: If the obstacle arrays disagree in count or ``height_std`` is not positive.
        """
        if height_std <= 0.0:
            raise ValueError(f"height_std must be positive, got {height_std}")

        self.channel = channel
        self.pose_channel = pose_channel
        self.state_channels = (pose_channel,)
        self.pose_indices = tuple(int(index) for index in pose_indices)
        self.height_std = float(height_std)
        self.floor_height = float(floor_height)
        self.pattern = (
            _disc_array(pattern, 2)
            if pattern is not None
            else self._square_grid(grid_extent, grid_size)
        )
        self.obstacle_centers = _disc_array(obstacle_centers, 2)
        self.obstacle_radii = _disc_array(obstacle_radii, 1)
        self.obstacle_heights = (
            _disc_array(obstacle_heights, 1)
            if obstacle_heights is not None
            else np.ones(self.obstacle_centers.shape[0])
        )
        counts = {
            self.obstacle_centers.shape[0],
            self.obstacle_radii.shape[0],
            self.obstacle_heights.shape[0],
        }
        if len(counts) > 1:
            raise ValueError(
                "obstacle_centers, obstacle_radii and obstacle_heights must have equal length; "
                f"got {sorted(counts)}"
            )

    @staticmethod
    def _square_grid(extent: float, size: int) -> np.ndarray:
        axis = np.linspace(-0.5 * extent, 0.5 * extent, max(1, int(size)))
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
        return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)

    def clean_heights(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Noise-free height samples from the pose in ``clean_state``.

        Args:
            clean_state: The state's named blocks; the pose block is read.

        Returns:
            Shape ``(M,)`` of noise-free heights, one per grid point.
        """
        position, heading = _planar_pose(clean_state, self.pose_channel, self.pose_indices)
        rotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        points = position[np.newaxis, :] + self.pattern @ rotation.T  # (M, 2)
        heights = np.full(points.shape[0], self.floor_height)
        for center, radius, height in zip(
            self.obstacle_centers, self.obstacle_radii, self.obstacle_heights
        ):
            inside = np.linalg.norm(points - center[np.newaxis, :], axis=-1) <= radius
            heights = np.where(inside, np.maximum(heights, height), heights)
        return heights

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        clean = self.clean_heights(clean_state)
        return clean + np.random.normal(0.0, self.height_std, size=clean.shape)

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        truth = self.clean_heights(clean_state)
        reading = np.asarray(channel_observation, dtype=float).reshape(-1)
        if reading.shape != truth.shape:
            return float("-inf")
        residual = (reading - truth) / self.height_std
        return float(
            -0.5 * np.sum(residual**2)
            - truth.size * (np.log(self.height_std) + 0.5 * np.log(2.0 * np.pi))
        )
