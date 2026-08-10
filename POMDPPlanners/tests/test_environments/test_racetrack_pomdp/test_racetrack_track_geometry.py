# SPDX-License-Identifier: MIT

"""Tests for the racetrack's piecewise-constant curvature profile.

:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.TrackGeometry`
is what replaced the state's frozen curvature slot, so the lookup has to be right in the
places a rollout actually lands: inside a segment, exactly on a boundary, and past the
finish line where the arclength has to wrap. Those are checked against hand-built profiles
in pure NumPy, with no simulator involved.

:func:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.build_track_geometry`
gets two kinds of test. A synthetic lane network pins the walk and the closed-loop guard
without highway-env; a live ``racetrack-v0`` pins the numbers that matter in practice --
nine segments over 381.9 m, straights at exactly zero and arcs not -- because a walk that
silently visited a different set of lanes would still look like a valid profile.
"""

from typing import Any, Optional, Sequence

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
    TrackGeometry,
    build_track_geometry,
    lane_curvature,
)

# The real circuit, measured against highway-env 1.12.1. A walk that wandered onto the
# parallel lane or stopped early would still produce an ascending profile, so the totals
# are what makes the live test an assertion rather than a smoke test.
_REAL_SEGMENT_COUNT = 9
_REAL_STRAIGHT_COUNT = 3
_REAL_LAP_LENGTH_M = 381.9


@pytest.fixture(name="geometry")
def geometry_fixture() -> TrackGeometry:
    """A three-segment lap: a 10 m straight, a 15 m left arc, then a 15 m right arc."""
    return TrackGeometry(
        segment_starts=np.array([0.0, 10.0, 25.0]),
        segment_curvatures=np.array([0.0, 0.04, -0.02]),
        total_length_m=40.0,
    )


class _FakeLane:
    """Minimal stand-in for a highway-env lane: a length and an optional arc radius."""

    def __init__(self, length: float, radius: Optional[float] = None, direction: float = 1.0):
        self.length = length
        self.direction = direction
        if radius is not None:
            self.radius = radius

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        # The walk only forwards this to ``next_lane``, which ignores it here.
        return np.array([longitudinal, lateral], dtype=float)


class _FakeNetwork:
    """A chain of lanes indexed ``0, 1, 2, ...`` that either closes into a lap or does not.

    ``get_lane`` takes the index modulo the chain length so an open walk keeps returning
    real lanes instead of raising, which is what lets the closed-loop guard be the thing
    under test rather than an incidental KeyError.
    """

    def __init__(self, lanes: Sequence[_FakeLane], closed: bool = True):
        self._lanes = list(lanes)
        self._closed = closed

    def get_lane(self, index: Any) -> _FakeLane:
        return self._lanes[int(index) % len(self._lanes)]

    def next_lane(self, index: Any, position: Any = None) -> int:
        del position
        following = int(index) + 1
        return following % len(self._lanes) if self._closed else following


def test_curvature_lookup_lands_in_the_containing_segment(geometry: TrackGeometry) -> None:
    """A distance inside a segment reads that segment's curvature.

    Purpose: Validates the basic piecewise-constant lookup for every segment

    Given: A three-segment lap of a straight, a left arc and a right arc
    When: curvature_at is called at a point strictly inside each segment
    Then: Each lookup returns that segment's own curvature

    Test type: unit
    """
    assert float(geometry.curvature_at(4.0)) == 0.0
    assert float(geometry.curvature_at(17.5)) == 0.04
    assert float(geometry.curvature_at(31.0)) == -0.02


def test_curvature_exactly_on_a_boundary_takes_the_following_segment(
    geometry: TrackGeometry,
) -> None:
    """A distance sitting exactly on a segment start belongs to the segment it starts.

    Purpose: Pins the half-open convention at the one input where the two segments meet,
        which a searchsorted ``side`` flip would silently invert

    Given: The same three-segment lap
    When: curvature_at is called at 0.0, 10.0 and 25.0 -- each an exact segment start
    Then: Each returns the curvature of the segment beginning there, not the one before it

    Test type: unit
    """
    assert float(geometry.curvature_at(0.0)) == 0.0
    assert float(geometry.curvature_at(10.0)) == 0.04
    assert float(geometry.curvature_at(25.0)) == -0.02


def test_arclength_beyond_the_lap_wraps_to_the_start(geometry: TrackGeometry) -> None:
    """Driving past the finish line reads the curvature of the next lap, not the last arc.

    Purpose: Validates the wrap, without which every rollout that completes a lap would be
        pinned to the final segment's curvature for the rest of the horizon

    Given: A 40 m lap
    When: curvature_at is called at exactly 40 m, and at 52 m and 97.5 m
    Then: Each equals the lookup at the same distance modulo the lap length

    Test type: unit
    """
    assert float(geometry.curvature_at(40.0)) == float(geometry.curvature_at(0.0))
    assert float(geometry.curvature_at(52.0)) == float(geometry.curvature_at(12.0))
    assert float(geometry.curvature_at(97.5)) == float(geometry.curvature_at(17.5))


def test_negative_arclength_wraps_backwards_from_the_end(geometry: TrackGeometry) -> None:
    """A negative distance reads from the end of the lap, not from the first segment.

    Purpose: Validates the wrap in the reverse direction, which a truncating modulo (C-style
        remainder rather than NumPy's floored ``mod``) would get wrong by a whole lap

    Given: A 40 m lap whose last segment starts at 25 m
    When: curvature_at is called at -5 m and -32.5 m
    Then: They read the last and the middle segment respectively, matching +35 m and +7.5 m

    Test type: unit
    """
    assert float(geometry.curvature_at(-5.0)) == -0.02
    assert float(geometry.curvature_at(-5.0)) == float(geometry.curvature_at(35.0))
    assert float(geometry.curvature_at(-32.5)) == float(geometry.curvature_at(7.5))


def test_batch_lookup_matches_scalar_lookups_one_by_one(geometry: TrackGeometry) -> None:
    """A batch of arclengths gives exactly what a loop of scalar lookups gives.

    Purpose: Validates the vectorised path the torch and NumPy models both call per substep,
        where a broadcasting mistake would go unnoticed because the shape would still fit

    Given: 500 arclengths spanning several laps in both directions, including exact
        boundaries and exact multiples of the lap length
    When: curvature_at is called once on the whole array and once per element
    Then: The results are identical and the array is shaped like the input

    Test type: unit
    """
    rng = np.random.default_rng(0)
    arclengths = np.concatenate(
        [
            rng.uniform(-200.0, 200.0, size=497),
            np.array([0.0, 10.0, 40.0]),
        ]
    )
    batch = geometry.curvature_at(arclengths)
    expected = np.array([float(geometry.curvature_at(float(s))) for s in arclengths])
    assert batch.shape == arclengths.shape
    np.testing.assert_array_equal(batch, expected)


def test_lane_curvature_is_zero_on_a_straight_and_signed_on_an_arc() -> None:
    """Curvature is direction over radius, and zero when there is no radius at all.

    Purpose: Validates the single place a highway-env lane object is read for its geometry

    Given: A straight lane with no radius attribute, a degenerate lane with radius 0, and
        two arcs of radius 25 m running in opposite directions
    When: lane_curvature is called on each
    Then: The straights give exactly 0.0 and the arcs give +-1/25

    Test type: unit
    """
    assert lane_curvature(_FakeLane(length=10.0)) == 0.0
    assert lane_curvature(_FakeLane(length=10.0, radius=0.0)) == 0.0
    assert lane_curvature(_FakeLane(length=10.0, radius=25.0, direction=1.0)) == 0.04
    assert lane_curvature(_FakeLane(length=10.0, radius=25.0, direction=-1.0)) == -0.04


def test_build_track_geometry_accumulates_a_synthetic_closed_loop() -> None:
    """Walking a closed chain records each lane's start distance and curvature in order.

    Purpose: Validates the walk itself -- cumulative starts, per-lane curvature, and the lap
        total -- without needing a simulator

    Given: A four-lane loop of a 10 m straight, a 20 m left arc, a 10 m straight and a 20 m
        right arc
    When: build_track_geometry walks it from lane 0
    Then: The starts are the running totals, the curvatures follow the lanes, and the total
        length is the sum

    Test type: unit
    """
    network = _FakeNetwork(
        [
            _FakeLane(length=10.0),
            _FakeLane(length=20.0, radius=50.0, direction=1.0),
            _FakeLane(length=10.0),
            _FakeLane(length=20.0, radius=40.0, direction=-1.0),
        ]
    )
    geometry = build_track_geometry(network, 0)
    np.testing.assert_array_equal(geometry.segment_starts, [0.0, 10.0, 30.0, 40.0])
    np.testing.assert_allclose(geometry.segment_curvatures, [0.0, 0.02, 0.0, -0.025])
    assert geometry.total_length_m == 60.0


def test_open_lane_sequence_raises_value_error() -> None:
    """A lane chain that never returns to its start is rejected rather than truncated.

    Purpose: Validates the guard that keeps arclength wrapping meaningful -- on an open
        sequence the lap length is whatever the walk happened to stop at, so every
        past-the-end lookup would read a curvature from the wrong part of the track

    Given: A network whose next_lane always advances and never closes
    When: build_track_geometry walks it with a small max_segments budget
    Then: ValueError is raised naming the starting lane

    Test type: unit
    """
    network = _FakeNetwork([_FakeLane(length=10.0)] * 3, closed=False)
    with pytest.raises(ValueError, match="did not close into a loop"):
        build_track_geometry(network, 0, max_segments=5)


def _real_track_geometry() -> TrackGeometry:
    """Profile of the lane a freshly reset ``racetrack-v0`` puts its ego on, or skip."""
    # Importing highway_env is also what registers racetrack-v0 with gymnasium, so the
    # skip and the registration are the same line.
    pytest.importorskip("highway_env")
    gymnasium = pytest.importorskip("gymnasium")
    env = gymnasium.make("racetrack-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        return build_track_geometry(unwrapped.road.network, unwrapped.vehicle.lane_index)
    finally:
        env.close()


def test_real_racetrack_is_a_closed_nine_segment_lap() -> None:
    """The live circuit walks to nine segments over 381.9 m: three straights and six arcs.

    Purpose: Pins the measured shape of the real track, so a walk that wandered onto the
        parallel lane or closed early is caught -- it would still produce an ascending
        profile and pass every structural check

    Given: A freshly reset racetrack-v0 and the lane its ego starts on
    When: build_track_geometry walks that lane's loop
    Then: There are nine segments starting at 0.0 and ascending, the lap is 381.9 m to
        within 0.1 m, and exactly three of them are straights at an exact zero

    Test type: integration
    """
    geometry = _real_track_geometry()

    starts = geometry.segment_starts
    curvatures = geometry.segment_curvatures
    assert len(starts) == _REAL_SEGMENT_COUNT
    assert len(curvatures) == _REAL_SEGMENT_COUNT
    assert starts[0] == 0.0
    assert np.all(np.diff(starts) > 0.0)
    assert starts[-1] < geometry.total_length_m
    assert abs(geometry.total_length_m - _REAL_LAP_LENGTH_M) < 0.1
    # Exactly zero, not almost: a straight lane has no radius at all, so anything else here
    # means the walk read a curvature off a lane it should not have been on.
    assert int(np.count_nonzero(curvatures == 0.0)) == _REAL_STRAIGHT_COUNT
    assert int(np.count_nonzero(curvatures != 0.0)) == _REAL_SEGMENT_COUNT - _REAL_STRAIGHT_COUNT


def test_real_racetrack_lookup_reaches_every_segment_of_the_lap() -> None:
    """Sweeping the live lap reads every distinct curvature the profile records.

    Purpose: Guards the lookup against an off-by-one that would clip the last segment away
        or index past it -- the three-segment synthetic profiles are too short to expose it

    Given: The live circuit's profile, 2000 points spread over four laps in both
        directions, and a 1 m sweep of a single lap
    When: curvature_at is evaluated on both
    Then: Every sampled value is one of the recorded curvatures, and the sweep reaches all
        of the distinct ones

    Test type: integration
    """
    geometry = _real_track_geometry()

    rng = np.random.default_rng(1)
    sampled = geometry.curvature_at(rng.uniform(-2.0, 2.0, size=2000) * geometry.total_length_m)
    assert np.all(np.isin(sampled, geometry.segment_curvatures))

    swept = geometry.curvature_at(np.arange(0.0, geometry.total_length_m, 1.0))
    np.testing.assert_array_equal(np.unique(swept), np.unique(geometry.segment_curvatures))
