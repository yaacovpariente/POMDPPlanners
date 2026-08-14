# SPDX-License-Identifier: MIT

"""Shape of the racetrack as a function of distance along it.

The planner's model needs to know where the road bends. Putting that in the *state* would
be wrong: curvature is a property of the road, so a state slot holding it encodes a
prediction about the future rather than a fact about the present, and a rollout that
reuses one frozen value drives straight through every corner. Where the road bends belongs
to the transition model, and this is how the transition model holds it.

The circuit is exactly representable, which makes this cheap and precise rather than an
approximation. Walking ``racetrack-v0``'s lane graph gives nine segments over 381.9 m, and
every one is either a straight (curvature ``0``) or a circular arc (curvature
``±1/radius``). So the curvature profile is **piecewise constant** — three short arrays,
no polyline sampling and no numerical differentiation of a fitted spline.

**The road is more than the ego's own lane.** A model that has to predict the observation's
on-road layer needs to know where the *neighbouring* lanes run too, so the walk also records
each segment's lane width and the signed lateral offset of every lane sharing that segment.
On ``racetrack-v0`` that is a second lane 5.00 m to one side on eight of the nine segments
and 5.09 m on the slant, and the width is 5 m throughout — the two agree because the
circuit's parallel arcs are built one lane-width apart.

Width is recorded because it is the road property callers ask for, but note that it plays
no part in predicting the on-road layer, and the reason is worth stating: highway-env's
``fill_road_layer_by_lanes`` walks each lane's **centreline** at ``min(grid_step)`` spacing
and marks the cell each waypoint lands in. It does not fill a corridor of the lane's width.
Measured on the shipped 12x12 grid, about 29 of 144 cells are set — two thin curves, not a
band — so it is the parallel lane's *offset* that a renderer needs and its width that it
does not.

:meth:`TrackGeometry.centreline_pose` turns the curvature profile back into positions and
headings by arclength, which is what a renderer draws. It is here rather than in the model
so the NumPy and torch renderers derive their sample points from one function instead of
two implementations that have to be kept in step.

Only :func:`build_track_geometry` touches ``highway_env``, and it is called once when a
model is constructed. :class:`TrackGeometry` itself is plain NumPy, so a model carrying one
still pickles for Ray or Dask and still constructs on a machine with no simulator
installed.

Classes:
    TrackGeometry: Piecewise-constant curvature and lane layout indexed by arclength.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

# Absent lane slots are padded with an offset far outside any plausible grid window rather
# than with NaN. A batched renderer indexes one padded row per particle and cannot drop
# columns per row, so the padding has to be a lateral offset that rasterises to nothing;
# NaN would instead reach ``floor`` and cast to an undefined integer cell.
ABSENT_LANE_OFFSET_M = 1.0e4

# Sub-intervals per output sample when integrating the curvature profile into a pose table.
# The profile is piecewise constant, so a boundary landing inside an output step would
# otherwise be integrated at the wrong curvature for up to a whole step; subdividing pushes
# that error down by the same factor at a cost of one short cumulative sum.
_POSE_SUBSTEPS = 16

# Below this the segment is integrated as a straight. A circular arc's closed form divides
# by the curvature, so it has to hand over near zero, and 1e-12 1/m is a radius of 1e12 m.
_STRAIGHT_CURVATURE = 1e-12


@dataclass(frozen=True)
class TrackGeometry:
    """Shape and lane layout of a closed track, as a function of distance along it.

    Attributes:
        segment_starts: Cumulative arclength in metres at the start of each segment,
            ascending, beginning at ``0.0``.
        segment_curvatures: Signed curvature of each segment in 1/m; ``0.0`` on a
            straight, ``±1/radius`` on an arc.
        total_length_m: Length of one lap in metres.
        lane_width_m: Width of the walked lane in metres, or NaN when the walk could not
            read one. Recorded as a road property; the on-road layer does not use it,
            because highway-env rasterises lane centrelines and not filled corridors.
        segment_lane_offsets: Signed lateral offset in metres of every lane sharing each
            segment, measured in the walked lane's own frame so its own entry is ``0.0``.
            Shape ``(segments, lanes)``, padded with :data:`ABSENT_LANE_OFFSET_M` where a
            segment carries fewer lanes than the widest one. ``None`` when the layout is
            unknown, which reads as "the walked lane and nothing beside it".

    Example:
        A square-ish loop of one straight and one arc, with a second lane 5 m to the left::

            >>> import numpy as np
            >>> geometry = TrackGeometry(
            ...     segment_starts=np.array([0.0, 10.0]),
            ...     segment_curvatures=np.array([0.0, 0.05]),
            ...     total_length_m=20.0,
            ...     lane_width_m=5.0,
            ...     segment_lane_offsets=np.array([[0.0, 5.0], [0.0, 5.0]]),
            ... )
            >>> float(geometry.curvature_at(4.0))
            0.0
            >>> float(geometry.curvature_at(12.0))
            0.05
            >>> float(geometry.curvature_at(24.0))  # wraps around the lap
            0.0
            >>> geometry.lane_offsets_at(4.0)
            array([0., 5.])
    """

    segment_starts: np.ndarray
    segment_curvatures: np.ndarray
    total_length_m: float
    lane_width_m: float = float("nan")
    segment_lane_offsets: Optional[np.ndarray] = None

    def curvature_at(self, arclength_m: Any) -> np.ndarray:
        """Signed curvature at one or many distances along the track.

        Args:
            arclength_m: Distance along the centreline in metres. Scalar or array; values
                outside one lap wrap around, so a rollout may run past the finish line.

        Returns:
            Curvature in 1/m, shaped like the input.
        """
        return self.segment_curvatures[self.segment_index_at(arclength_m)]

    def segment_index_at(self, arclength_m: Any) -> np.ndarray:
        """Index of the segment containing one or many distances along the track.

        Args:
            arclength_m: Distance along the centreline in metres, wrapping around the lap.

        Returns:
            Integer indices shaped like the input, clipped into the recorded segments.
        """
        distance = np.mod(np.asarray(arclength_m, dtype=float), self.total_length_m)
        index = np.searchsorted(self.segment_starts, distance, side="right") - 1
        return np.clip(index, 0, len(self.segment_curvatures) - 1)

    def lane_offsets_at(self, arclength_m: float) -> np.ndarray:
        """Lateral offsets of the lanes sharing the segment at one distance along the track.

        Args:
            arclength_m: Distance along the centreline in metres, wrapping around the lap.

        Returns:
            Signed offsets in metres in the walked lane's frame, padding dropped, so the
            walked lane's own ``0.0`` is always present. ``[0.0]`` when the layout is
            unknown.
        """
        if self.segment_lane_offsets is None:
            return np.zeros(1, dtype=float)
        row = np.asarray(self.segment_lane_offsets, dtype=float)[
            int(self.segment_index_at(float(arclength_m)))
        ]
        return row[np.abs(row) < ABSENT_LANE_OFFSET_M]

    def padded_lane_offsets(self) -> np.ndarray:
        """Lane offsets per segment as one rectangular array, padding included.

        The batched form of :meth:`lane_offsets_at`, for a renderer that scores many
        particles at once and so cannot drop a different set of columns per row.

        Returns:
            ``(segments, lanes)`` offsets in metres, absent lanes held at
            :data:`ABSENT_LANE_OFFSET_M`.
        """
        if self.segment_lane_offsets is None:
            return np.zeros((len(self.segment_curvatures), 1), dtype=float)
        return np.asarray(self.segment_lane_offsets, dtype=float)

    def centreline_pose(self, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """Positions and headings along the walked lane, sampled by arclength.

        Integrates the piecewise-constant curvature profile into a closed-form arc per
        sub-interval, in a track-local frame whose origin and heading are those of
        arclength ``0``. A renderer only ever takes *differences* within a short window, so
        the frame's absolute placement never matters.

        Args:
            step_m: Spacing between samples in metres. Must be positive.

        Returns:
            ``(positions, headings)``: an ``(M, 2)`` array of metres and an ``(M,)`` array
            of radians, where ``M = ceil(total_length_m / step_m)``. Index ``k`` is
            arclength ``k * step_m``, so a lookup is arithmetic rather than a search.

        Raises:
            ValueError: If ``step_m`` is not positive.

        Note:
            The lap does not close exactly. The profile records one curvature per lane
            segment, and on ``racetrack-v0`` re-integrating it returns 1.9 m and 0.28 rad
            from where it started. Inside any one window that is a fraction of a cell, but
            a window straddling arclength ``0`` sees the whole discrepancy at once.
        """
        if step_m <= 0.0:
            raise ValueError(f"step_m must be positive, got {step_m}.")
        samples = int(np.ceil(self.total_length_m / step_m)) * _POSE_SUBSTEPS
        fine = step_m / _POSE_SUBSTEPS
        # Curvature at each sub-interval's midpoint, which is the value that integrates the
        # interval correctly whenever a segment boundary does not fall inside it.
        curvature = self.curvature_at((np.arange(samples) + 0.5) * fine)
        positions, headings = _integrate_arcs(curvature, fine)
        return positions[::_POSE_SUBSTEPS], headings[::_POSE_SUBSTEPS]


def _integrate_arcs(curvature: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    # Each sub-interval is a circular arc of constant curvature, so its displacement in its
    # own start frame is closed form. Euler stepping would instead chord every arc and lose
    # radius steadily around the lap, which is the one error a pose table cannot absorb.
    turn = curvature * step
    heading = np.concatenate([[0.0], np.cumsum(turn)[:-1]])
    straight = np.abs(curvature) < _STRAIGHT_CURVATURE
    safe = np.where(straight, 1.0, curvature)
    forward = np.where(straight, step, np.sin(turn) / safe)
    sideways = np.where(straight, 0.0, (1.0 - np.cos(turn)) / safe)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    steps = np.column_stack(
        [cos_h * forward - sin_h * sideways, sin_h * forward + cos_h * sideways]
    )
    positions = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)[:-1]])
    return positions, heading


def lane_curvature(lane: Any) -> float:
    """Signed curvature of a highway-env lane, in 1/m.

    Args:
        lane: A highway-env lane object.

    Returns:
        ``0.0`` for a straight lane, otherwise ``direction / radius`` where ``direction``
        is ``+1`` clockwise and ``-1`` counter-clockwise.
    """
    radius = getattr(lane, "radius", None)
    if radius is None or float(radius) == 0.0:
        return 0.0
    return float(getattr(lane, "direction", 1.0)) / float(radius)


def build_track_geometry(
    road_network: Any, lane_index: Any, max_segments: int = 64
) -> TrackGeometry:
    """Walk a closed lane loop and record its shape and lane layout against distance.

    Follows ``next_lane`` from ``lane_index`` until the walk returns to where it started,
    accumulating each segment's length, curvature, width and neighbouring-lane offsets.

    Args:
        road_network: A highway-env ``RoadNetwork``.
        lane_index: The lane to start from, normally the ego's current lane.
        max_segments: Guard against a network that never closes. Defaults to 64.

    Returns:
        The profile of that lap.

    Raises:
        ValueError: If the walk does not return to its starting lane, which means the lane
            sequence is not a closed loop and arclength would not wrap correctly.

    Note:
        The offsets are recorded **per segment**, in the frame of whichever lane the walk
        is on at that segment, and that is not cosmetic. ``next_lane`` picks the lane
        closest to the handover position, so a walk starting on lane 0 can continue on
        lane 1: measured on ``racetrack-v0``, three of the eight seeded starts do exactly
        that. Collecting one offset set for the whole lap would then mix ``+5`` and ``-5``
        and invent a third lane that is not there.

    Note:
        Still built for one lane at a time. Parallel lanes on the same segment have
        different radii, so a lane change invalidates the curvature profile and re-bases
        the arclength the caller indexes it with. The planner's model does not represent
        lane changes, so this is consistent today, but it is a trap if that ever changes.
    """
    starts: List[float] = []
    curvatures: List[float] = []
    offsets: List[List[float]] = []
    widths: List[float] = []
    total = 0.0
    current = lane_index
    for _ in range(max_segments):
        lane = road_network.get_lane(current)
        starts.append(total)
        curvatures.append(lane_curvature(lane))
        offsets.append(_sibling_offsets(road_network, current, lane))
        widths.append(lane_width(lane))
        total += float(lane.length)
        current = road_network.next_lane(current, position=lane.position(lane.length, 0))
        if current == lane_index:
            return TrackGeometry(
                segment_starts=np.asarray(starts, dtype=float),
                segment_curvatures=np.asarray(curvatures, dtype=float),
                total_length_m=total,
                lane_width_m=_median_width(widths),
                segment_lane_offsets=_pad_offsets(offsets),
            )
    raise ValueError(
        f"Lane walk from {lane_index} did not close into a loop within {max_segments} "
        "segments; arclength cannot wrap on an open lane sequence."
    )


def lane_width(lane: Any) -> float:
    """Width of a highway-env lane in metres, or NaN when it does not report one.

    Args:
        lane: A highway-env lane object.

    Returns:
        The lane's width at its start, or NaN if the object has neither ``width_at`` nor
        ``width``.
    """
    # Annotated Any rather than left to inference: narrowing an unknown attribute with
    # ``callable`` gives pyright a ``() -> object``, and the float conversion below then has
    # to be silenced. Declaring the intent once is cheaper than a type: ignore per call.
    width_at: Any = getattr(lane, "width_at", None)
    if width_at is not None:
        return float(width_at(0.0))
    width: Any = getattr(lane, "width", None)
    return float("nan") if width is None else float(width)


def _sibling_offsets(road_network: Any, lane_index: Any, lane: Any) -> List[float]:
    # Every lane sharing this segment, projected onto the walked lane to get a signed
    # lateral offset. Read through ``all_side_lanes`` rather than the raw graph so a network
    # that does not expose one -- a test double, say -- degrades to "this lane alone"
    # instead of raising, which keeps the walk usable without a simulator.
    side_lanes: Any = getattr(road_network, "all_side_lanes", None)
    if side_lanes is None or getattr(lane, "local_coordinates", None) is None:
        return [0.0]
    measured = []
    for index in side_lanes(lane_index):
        other = road_network.get_lane(index)
        midpoint = other.position(float(other.length) * 0.5, 0)
        measured.append(float(lane.local_coordinates(midpoint)[1]))
    return sorted(measured, key=abs) if measured else [0.0]


def _median_width(widths: List[float]) -> float:
    # Median rather than the first lane's width, so one oddly-built segment does not decide
    # the lap's. Guarded against an all-NaN walk -- a network of lanes that report no width
    # at all -- because ``nanmedian`` warns and still returns NaN, and the warning would be
    # noise rather than news.
    measured = [width for width in widths if not np.isnan(width)]
    return float(np.median(measured)) if measured else float("nan")


def _pad_offsets(offsets: List[List[float]]) -> np.ndarray:
    lanes = max(len(row) for row in offsets)
    table = np.full((len(offsets), lanes), ABSENT_LANE_OFFSET_M, dtype=float)
    for index, row in enumerate(offsets):
        table[index, : len(row)] = row
    return table


def geometry_from_world(world: Any) -> Tuple[TrackGeometry, Any]:
    """Build the curvature profile of the lane the world's ego currently occupies.

    Args:
        world: A live :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp.RacetrackPOMDP`
            that has been reset at least once.

    Returns:
        The geometry and the lane index it was built for.
    """
    # pylint: disable=protected-access  # The session is the world's only backend handle.
    unwrapped = world._get_session()._env.unwrapped
    lane_index = unwrapped.vehicle.lane_index
    return build_track_geometry(unwrapped.road.network, lane_index), lane_index


__all__ = [
    "ABSENT_LANE_OFFSET_M",
    "TrackGeometry",
    "build_track_geometry",
    "geometry_from_world",
    "lane_curvature",
    "lane_width",
]
