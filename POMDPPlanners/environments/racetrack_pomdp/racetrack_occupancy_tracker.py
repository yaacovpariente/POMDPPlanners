# SPDX-License-Identifier: MIT

"""Recover other vehicles' relative motion by differencing two ego-aligned occupancy grids.

The racetrack POMDP observation is a presence/on-road occupancy grid. It says *where* things
are and nothing about how fast they move, so a planner that only reads one frame cannot tell
an opponent closing at 8 m/s from one drifting away. This module supplies the missing channel
the only way a grid allows: label the occupied cells in two consecutive frames, match the
blobs, and difference their centres.

**Read this before trusting a number that comes out of here.** Three properties of the
highway-env grid, all verified against 1.12.1 rather than read off the source, decide what
this tracker can and cannot measure.

*A vehicle marks exactly one cell, not its footprint.* So a blob is normally a single cell and
its centre can only ever sit at a cell centre. Centre motion is therefore quantised at a full
``grid_step_m``. At the shipped 3 m step and a 0.2 s decision period, the smallest non-zero
speed this tracker can report is ``3.0 / 0.2 = 15 m/s`` — above the track's own 10 m/s speed
limit. **The output is the sign and rough magnitude of relative motion, not a calibrated
velocity.** Treat it as "closing / opening / roughly stationary" and give the belief a
velocity jitter wide enough to cover the quantisation. Raising ``frame_stride`` divides the
quantum by the stride (a stride of 3 brings it to 5 m/s) at the cost of that many steps of
latency, which is the knob a study should turn if the coarse reading proves too blunt.

*The ego is always written into the grid at the centre cell.* Left alone it becomes a
permanent stationary blob in every frame, and — because components are 8-connected — it
swallows any opponent that comes within one cell. That is a blind spot at exactly the range
that matters, so the centre cell is cleared from the mask *before* labelling rather than its
whole component being discarded afterwards: clearing keeps a neighbouring opponent as a
cluster of its own, discarding the component would delete it along with the ego.

*Touching cells are read as one vehicle.* Components are 8-connected, so two opponents in
neighbouring cells are reported as a single blob at their midpoint. Under the one-cell-per-
vehicle rule that is a real loss of resolution, accepted for what it buys on the other side:
the planner's generative model perturbs its rendered grids with per-cell flips, and under
4-connectivity a single spurious cell beside a vehicle becomes a *second* phantom opponent
with its own fabricated velocity. Both readings are wrong; merging is the one where the
obstacle does not disappear and no vehicle is invented. Widen the grid resolution rather than
the connectivity if two-vehicle separation at 3 m starts to matter.

*The grid rotates with the ego* (``align_to_vehicle_axes``). Under steering lock, one 0.2 s
step turns the frame enough to shift a blob at the grid edge by several metres with nothing
having moved. The previous frame's centres are therefore de-rotated by the ego's yaw change
before matching; skipping that step does not add noise, it manufactures velocity.

What comes out is velocity **relative to a moving, turning car**, which is what the racetrack
state's agent slots hold, so the two are consistent.

Classes:
    TrackedCluster: One occupied blob in the current frame with its relative velocity.
    OccupancyVelocityTracker: Matches blobs across two grids and differences their centres.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from scipy import ndimage

from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    PRESENCE_LAYER,
    rotate,
)

DEFAULT_DT_S = 0.2
DEFAULT_GATE_RADIUS_M = 6.0
DEFAULT_FRAME_STRIDE = 1

# 8-connectivity: a diagonal neighbour belongs to the same blob. See the module docstring --
# with one cell per vehicle this is a deliberate trade, not a footprint-stitching device.
_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class TrackedCluster:
    """One occupied blob in the current frame.

    Attributes:
        centre: ``(2,)`` ego-frame position in metres, ``[forward, across]``.
        velocity: ``(2,)`` ego-frame velocity in m/s, **relative** to the ego. Zeros when the
            blob could not be matched to the previous frame, which is not the same claim as
            "it is stationary" — check ``matched`` before reading it as a measurement.
        matched: Whether a previous-frame blob was found inside the gate.
    """

    centre: np.ndarray
    velocity: np.ndarray
    matched: bool


class OccupancyVelocityTracker:
    """Differences two ego-aligned occupancy grids into per-blob relative velocities.

    The tracker is pure NumPy and holds no state between calls: both frames are passed in, so
    the caller owns the history and the tracker stays trivially picklable and reusable across
    particles. See the module docstring for the quantisation limit that governs how the
    reported velocities should be read.

    Attributes:
        grid_half_extent_m: Half-width of the grid window in metres.
        grid_step_m: Cell size in metres.
        dt: Seconds between two consecutive frames.
        gate_radius_m: Largest blob displacement accepted as the same vehicle.
        frame_stride: How many decision steps separate the two frames handed to
            :meth:`track`. Raising it widens the measurement baseline and divides the velocity
            quantum by the same factor, at the cost of that many steps of latency.

    Example:
        >>> import numpy as np
        >>> tracker = OccupancyVelocityTracker()
        >>> previous = np.zeros((2, 12, 12), dtype=np.float32)
        >>> current = np.zeros((2, 12, 12), dtype=np.float32)
        >>> previous[0, 6, 6] = current[0, 6, 6] = 1.0  # the ego, always at the centre cell
        >>> previous[0, 8, 6] = 1.0                     # an opponent, one cell further ahead
        >>> current[0, 7, 6] = 1.0                      # ... now one cell closer
        >>> clusters = tracker.track(previous, current, ego_yaw_delta=0.0)
        >>> len(clusters), clusters[0].matched
        (1, True)
        >>> clusters[0].velocity.round(1)  # closing at one cell per step
        array([-15.,   0.])
    """

    def __init__(
        self,
        grid_half_extent_m: float = GRID_HALF_EXTENT_M,
        grid_step_m: float = GRID_STEP_M,
        dt: float = DEFAULT_DT_S,
        gate_radius_m: float = DEFAULT_GATE_RADIUS_M,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
    ):
        """Initialize the tracker.

        Args:
            grid_half_extent_m: Half-width of the grid window in metres. Defaults to 18.0.
            grid_step_m: Cell size in metres. Defaults to 3.0.
            dt: Seconds between two consecutive decision steps. Defaults to 0.2.
            gate_radius_m: Largest blob displacement accepted as the same vehicle. Defaults to
                6.0, which at a 3 m step admits a one-cell move on each axis and rejects the
                two-cell jumps that no vehicle on a 10 m/s track can make.
            frame_stride: Decision steps between the two frames passed to :meth:`track`.
                Defaults to 1.

        Raises:
            ValueError: If any geometry or timing argument is non-positive, or if the window
                is not an integer number of cells across.
        """
        self.grid_half_extent_m = float(grid_half_extent_m)
        self.grid_step_m = float(grid_step_m)
        self.dt = float(dt)
        self.gate_radius_m = float(gate_radius_m)
        self.frame_stride = int(frame_stride)
        self._validate()
        self._cells = int(round(2.0 * self.grid_half_extent_m / self.grid_step_m))
        # The ego sits at the origin, so its cell is whichever one contains (0, 0). Derived
        # from the geometry rather than written as 6, so a re-sized grid stays correct.
        self._centre_index = int(np.floor(self.grid_half_extent_m / self.grid_step_m))
        self._baseline_s = self.dt * self.frame_stride

    def track(
        self,
        previous_grid: np.ndarray,
        current_grid: np.ndarray,
        ego_yaw_delta: float,
    ) -> List[TrackedCluster]:
        """Match blobs across two frames and difference their centres.

        Args:
            previous_grid: The earlier occupancy observation, ``(layers, cells, cells)``.
            current_grid: The later occupancy observation, same shape.
            ego_yaw_delta: The ego's heading change from the earlier frame to the later one,
                in radians. The previous frame's centres are de-rotated by this before
                matching; passing 0.0 while the ego is turning invents velocity out of the
                frame rotation.

        Returns:
            One :class:`TrackedCluster` per blob in the *current* frame, ego excluded. Blobs
            with no partner inside the gate come back with zero velocity and ``matched=False``.

        Raises:
            ValueError: If either grid does not match the configured geometry.
        """
        current_centres = self._cluster_centres(current_grid)
        previous_centres = rotate(self._cluster_centres(previous_grid), -float(ego_yaw_delta))
        matches = self._match(current_centres, previous_centres)
        return [
            self._to_cluster(current_centres[index], previous_centres, int(match))
            for index, match in enumerate(matches)
        ]

    def detect_clusters(self, grid: np.ndarray) -> List[TrackedCluster]:
        """Label one frame's blobs without measuring velocity.

        Used on the first step of an episode, when there is no previous frame to difference
        against and reporting a fabricated velocity would be worse than reporting none.

        Args:
            grid: An occupancy observation, ``(layers, cells, cells)``.

        Returns:
            One cluster per blob, ego excluded, each with zero velocity and ``matched=False``.

        Raises:
            ValueError: If the grid does not match the configured geometry.
        """
        return [
            TrackedCluster(centre=centre, velocity=np.zeros(2), matched=False)
            for centre in self._cluster_centres(grid)
        ]

    def _validate(self) -> None:
        positive = {
            "grid_half_extent_m": self.grid_half_extent_m,
            "grid_step_m": self.grid_step_m,
            "dt": self.dt,
            "gate_radius_m": self.gate_radius_m,
            "frame_stride": float(self.frame_stride),
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")
        span = 2.0 * self.grid_half_extent_m / self.grid_step_m
        if abs(span - round(span)) > 1e-9:
            raise ValueError(
                f"grid_half_extent_m={self.grid_half_extent_m} is not an integer number of "
                f"grid_step_m={self.grid_step_m} cells across; the cell-to-metre map would "
                f"not line up with the simulator's grid."
            )

    def _cluster_centres(self, grid: np.ndarray) -> np.ndarray:
        mask = self._presence_mask(grid)
        # ndimage.label returns either a bare count or a (labels, count) pair depending on
        # whether an output array was passed, so its inferred return type is a union the
        # checker cannot narrow. Without `output=` it is always the pair.
        labelled: Any = ndimage.label(mask, structure=_EIGHT_CONNECTIVITY)
        labels, count = labelled[0], int(labelled[1])
        if count == 0:
            return np.zeros((0, 2), dtype=float)
        indices = np.asarray(
            ndimage.center_of_mass(mask, labels, list(range(1, count + 1))), dtype=float
        ).reshape(count, 2)
        return self._to_metres(indices)

    def _presence_mask(self, grid: np.ndarray) -> np.ndarray:
        array = np.asarray(grid)
        self._require_geometry(array)
        mask = array[PRESENCE_LAYER] > 0.5
        # Clear the ego rather than discarding its whole component; see the module docstring.
        mask[self._centre_index, self._centre_index] = False
        return mask

    def _require_geometry(self, array: np.ndarray) -> None:
        expected = (self._cells, self._cells)
        if array.ndim != 3 or array.shape[1:] != expected:
            raise ValueError(
                f"Occupancy grid must have shape (layers, {self._cells}, {self._cells}) for "
                f"grid_half_extent_m={self.grid_half_extent_m} and "
                f"grid_step_m={self.grid_step_m}, got {array.shape}."
            )
        if array.shape[0] <= PRESENCE_LAYER:
            raise ValueError(
                f"Occupancy grid has {array.shape[0]} layer(s) but the presence layer is at "
                f"index {PRESENCE_LAYER}."
            )

    def _to_metres(self, indices: np.ndarray) -> np.ndarray:
        # Axis 0 of the grid is along-track, axis 1 across-track (verified against
        # highway-env 1.12.1), so an index pair maps straight onto [forward, across].
        return -self.grid_half_extent_m + (indices + 0.5) * self.grid_step_m

    def _match(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        matches = np.full(len(current), -1, dtype=int)
        if len(current) == 0 or len(previous) == 0:
            return matches
        remaining = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=-1)
        for _ in range(min(len(current), len(previous))):
            row, column, distance = _smallest_entry(remaining)
            if distance >= self.gate_radius_m:
                break
            matches[row] = column
            remaining[row, :] = np.inf
            remaining[:, column] = np.inf
        return matches

    def _to_cluster(self, centre: np.ndarray, previous: np.ndarray, match: int) -> TrackedCluster:
        if match < 0:
            return TrackedCluster(centre=centre, velocity=np.zeros(2), matched=False)
        velocity = (centre - previous[match]) / self._baseline_s
        return TrackedCluster(centre=centre, velocity=velocity, matched=True)


def _smallest_entry(distances: np.ndarray) -> Tuple[int, int, float]:
    # Greedy global nearest neighbour: the single closest surviving pair, whatever row and
    # column it sits in. Struck-out rows and columns are +inf, so they never win.
    flat = int(np.argmin(distances))
    row, column = divmod(flat, distances.shape[1])
    return row, column, float(distances[row, column])
