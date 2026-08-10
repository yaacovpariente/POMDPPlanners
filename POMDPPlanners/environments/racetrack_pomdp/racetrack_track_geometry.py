# SPDX-License-Identifier: MIT

"""Curvature of the racetrack as a function of distance along it.

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

Only :func:`build_track_geometry` touches ``highway_env``, and it is called once when a
model is constructed. :class:`TrackGeometry` itself is plain NumPy, so a model carrying one
still pickles for Ray or Dask and still constructs on a machine with no simulator
installed.

Classes:
    TrackGeometry: Piecewise-constant curvature indexed by arclength along the track.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackGeometry:
    """Signed curvature of a closed track, as a function of distance along it.

    Attributes:
        segment_starts: Cumulative arclength in metres at the start of each segment,
            ascending, beginning at ``0.0``.
        segment_curvatures: Signed curvature of each segment in 1/m; ``0.0`` on a
            straight, ``±1/radius`` on an arc.
        total_length_m: Length of one lap in metres.

    Example:
        A square-ish loop of one straight and one arc::

            >>> import numpy as np
            >>> geometry = TrackGeometry(
            ...     segment_starts=np.array([0.0, 10.0]),
            ...     segment_curvatures=np.array([0.0, 0.05]),
            ...     total_length_m=20.0,
            ... )
            >>> float(geometry.curvature_at(4.0))
            0.0
            >>> float(geometry.curvature_at(12.0))
            0.05
            >>> float(geometry.curvature_at(24.0))  # wraps around the lap
            0.0
    """

    segment_starts: np.ndarray
    segment_curvatures: np.ndarray
    total_length_m: float

    def curvature_at(self, arclength_m: Any) -> np.ndarray:
        """Signed curvature at one or many distances along the track.

        Args:
            arclength_m: Distance along the centreline in metres. Scalar or array; values
                outside one lap wrap around, so a rollout may run past the finish line.

        Returns:
            Curvature in 1/m, shaped like the input.
        """
        distance = np.mod(np.asarray(arclength_m, dtype=float), self.total_length_m)
        index = np.searchsorted(self.segment_starts, distance, side="right") - 1
        return self.segment_curvatures[np.clip(index, 0, len(self.segment_curvatures) - 1)]


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
    """Walk a closed lane loop and record its curvature against distance.

    Follows ``next_lane`` from ``lane_index`` until the walk returns to where it started,
    accumulating each segment's length and curvature.

    Args:
        road_network: A highway-env ``RoadNetwork``.
        lane_index: The lane to start from, normally the ego's current lane.
        max_segments: Guard against a network that never closes. Defaults to 64.

    Returns:
        The curvature profile of that lap.

    Raises:
        ValueError: If the walk does not return to its starting lane, which means the lane
            sequence is not a closed loop and arclength would not wrap correctly.

    Note:
        Built for one lane. Parallel lanes on the same segment have different radii, so a
        lane change invalidates the profile. The planner's model does not represent lane
        changes, so this is consistent today, but it is a trap if that ever changes.
    """
    starts: List[float] = []
    curvatures: List[float] = []
    total = 0.0
    current = lane_index
    for _ in range(max_segments):
        lane = road_network.get_lane(current)
        starts.append(total)
        curvatures.append(lane_curvature(lane))
        total += float(lane.length)
        current = road_network.next_lane(current, position=lane.position(lane.length, 0))
        if current == lane_index:
            return TrackGeometry(
                segment_starts=np.asarray(starts, dtype=float),
                segment_curvatures=np.asarray(curvatures, dtype=float),
                total_length_m=total,
            )
    raise ValueError(
        f"Lane walk from {lane_index} did not close into a loop within {max_segments} "
        "segments; arclength cannot wrap on an open lane sequence."
    )


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
    "TrackGeometry",
    "build_track_geometry",
    "geometry_from_world",
    "lane_curvature",
]
