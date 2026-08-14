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
    ABSENT_LANE_OFFSET_M,
    TrackGeometry,
    build_track_geometry,
    lane_curvature,
    lane_width,
)

# The real circuit, measured against highway-env 1.12.1. A walk that wandered onto the
# parallel lane or stopped early would still produce an ascending profile, so the totals
# are what makes the live test an assertion rather than a smoke test.
_REAL_SEGMENT_COUNT = 9
_REAL_STRAIGHT_COUNT = 3
_REAL_LAP_LENGTH_M = 381.9
# Every segment of the circuit is two lanes of 5 m, built one width apart. The slant is
# laid out from hand-written endpoints rather than a shared centre, so its parallel lane
# measures 5.09 m away instead of 5.00 -- a third of a grid cell, and the reason the offset
# check below is a tolerance rather than an equality.
_REAL_LANE_COUNT = 2
_REAL_LANE_WIDTH_M = 5.0
_REAL_LANE_OFFSET_TOLERANCE_M = 0.1


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

    def __init__(
        self,
        length: float,
        radius: Optional[float] = None,
        direction: float = 1.0,
        width: Optional[float] = None,
    ):
        self.length = length
        self.direction = direction
        if radius is not None:
            self.radius = radius
        if width is not None:
            self.width = width

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


def test_lane_offsets_drop_the_padding_of_a_narrower_segment() -> None:
    """A segment carrying fewer lanes reports only the lanes it has.

    Purpose: The offsets are stored as one rectangular array so a batched renderer can index
        a row per particle, which means short segments carry filler. A caller reading one
        segment must never see that filler as a real lane metres off to the side

    Given: A two-segment lap whose first segment has two lanes and whose second has one,
        padded to the same width
    When: lane_offsets_at is called inside each segment
    Then: The first reports both offsets and the second reports the walked lane alone

    Test type: unit
    """
    geometry = TrackGeometry(
        segment_starts=np.array([0.0, 10.0]),
        segment_curvatures=np.array([0.0, 0.0]),
        total_length_m=20.0,
        lane_width_m=5.0,
        segment_lane_offsets=np.array([[0.0, -5.0], [0.0, ABSENT_LANE_OFFSET_M]]),
    )

    np.testing.assert_array_equal(geometry.lane_offsets_at(4.0), [0.0, -5.0])
    np.testing.assert_array_equal(geometry.lane_offsets_at(14.0), [0.0])
    assert geometry.padded_lane_offsets().shape == (2, 2)


def test_a_geometry_with_no_lane_layout_reports_the_walked_lane_alone(
    geometry: TrackGeometry,
) -> None:
    """A profile built without a lane graph behaves as a single-lane road.

    Purpose: ``TrackGeometry`` is constructed directly by tests, doctests and any caller
        holding a curvature profile with no network to walk. Those must keep working rather
        than raise, and must not invent a neighbouring lane nobody supplied

    Given: The three-segment fixture, built with no lane offsets at all
    When: lane_offsets_at and padded_lane_offsets are called
    Then: Both report exactly one lane, at offset zero, and the width reads NaN

    Test type: unit
    """
    np.testing.assert_array_equal(geometry.lane_offsets_at(17.5), [0.0])
    np.testing.assert_array_equal(geometry.padded_lane_offsets(), np.zeros((3, 1)))
    assert np.isnan(geometry.lane_width_m)


def test_centreline_pose_traces_a_circle_of_the_profiles_own_radius() -> None:
    """Re-integrating a constant curvature gives back the circle it came from.

    Purpose: The pose table is what a renderer draws, so an error here misplaces the whole
        road. A circle is the case that catches the mistake a straight cannot -- chording
        each step instead of integrating the arc loses radius steadily and still looks like
        a closed curve

    Given: A single-segment lap of curvature 1/25 whose length is one full circumference
    When: centreline_pose samples it every 0.5 m
    Then: Every sample lies 25 m from the centre implied by the starting pose, to 1 mm, and
        the heading turns through exactly 2*pi over the lap

    Test type: unit
    """
    radius = 25.0
    geometry = TrackGeometry(
        segment_starts=np.array([0.0]),
        segment_curvatures=np.array([1.0 / radius]),
        total_length_m=2.0 * np.pi * radius,
    )

    positions, headings = geometry.centreline_pose(0.5)

    # Starting at the origin heading +x and curving left puts the centre at (0, radius).
    distances = np.linalg.norm(positions - np.array([0.0, radius]), axis=1)
    np.testing.assert_allclose(distances, radius, atol=1e-3)
    assert headings[0] == 0.0
    assert float(headings[-1]) == pytest.approx(2.0 * np.pi, abs=1e-2)


def test_centreline_pose_of_a_straight_lap_advances_along_one_axis() -> None:
    """A profile with no curvature integrates to evenly spaced points on a line.

    Purpose: Pins the degenerate branch. The arc formula divides by the curvature, so a
        straight has to be handled separately, and a mistake there would be a NaN or an
        infinity rather than a subtly wrong shape

    Given: A 30 m lap of a single straight segment
    When: centreline_pose samples it every 1.5 m
    Then: There are 20 samples, all headings are zero, the across-track coordinate never
        leaves zero, and the along-track coordinate steps by exactly 1.5 m

    Test type: unit
    """
    geometry = TrackGeometry(
        segment_starts=np.array([0.0]),
        segment_curvatures=np.array([0.0]),
        total_length_m=30.0,
    )

    positions, headings = geometry.centreline_pose(1.5)

    assert positions.shape == (20, 2)
    np.testing.assert_allclose(headings, 0.0, atol=1e-12)
    np.testing.assert_allclose(positions[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diff(positions[:, 0]), 1.5, atol=1e-9)


def test_centreline_pose_rejects_a_non_positive_step(geometry: TrackGeometry) -> None:
    """A zero or negative sample spacing is refused rather than looped on forever.

    Purpose: The sample count is a division by the step, so zero would raise deep inside
        NumPy and a negative one would produce an empty table that renders an empty road --
        a model that silently believes it is off-track everywhere

    Given: The three-segment fixture
    When: centreline_pose is called with 0.0 and with -1.5
    Then: ValueError is raised naming the argument

    Test type: unit
    """
    for step in (0.0, -1.5):
        with pytest.raises(ValueError, match="step_m must be positive"):
            geometry.centreline_pose(step)


def test_lane_width_reads_either_accessor_and_falls_back_to_nan() -> None:
    """Width comes from ``width_at`` when there is one, then ``width``, then NaN.

    Purpose: highway-env lanes expose ``width_at``; hand-built stand-ins and any lane object
        that predates it may expose only a plain attribute or neither. Raising on the last
        case would make the whole walk unusable without a simulator

    Given: A lane with a plain width, a lane with neither, and an object whose ``width_at``
        is callable
    When: lane_width is called on each
    Then: The first two give 5.0 and NaN, and the callable one is preferred over any
        attribute beside it

    Test type: unit
    """

    class _WidthAtLane:
        width = 3.0

        def width_at(self, longitudinal: float) -> float:
            del longitudinal
            return 7.0

    assert lane_width(_FakeLane(length=10.0, width=5.0)) == 5.0
    assert np.isnan(lane_width(_FakeLane(length=10.0)))
    assert lane_width(_WidthAtLane()) == 7.0


def test_a_network_without_side_lanes_records_one_lane_per_segment() -> None:
    """A walk over a network that cannot report neighbours still produces a usable profile.

    Purpose: ``all_side_lanes`` is a highway-env method, and the walk has to survive without
        it -- a synthetic network, or any caller holding lanes but no graph. Degrading to
        "this lane alone" is right; raising would make the profile unbuildable

    Given: The synthetic four-lane loop, whose network exposes no all_side_lanes
    When: build_track_geometry walks it
    Then: Every segment records exactly one lane at offset zero, and the width is NaN

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

    np.testing.assert_array_equal(geometry.padded_lane_offsets(), np.zeros((4, 1)))
    assert np.isnan(geometry.lane_width_m)


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


def test_real_racetrack_records_a_second_lane_on_one_consistent_side() -> None:
    """The live circuit walks to two 5 m lanes per segment, the neighbour always one side.

    Purpose: The offsets are recorded per segment because ``next_lane`` can hand the walk
        onto the parallel lane partway round, which flips the neighbour from +5 to -5. If
        that ever produced a mixed set the renderer would draw a third lane that is not
        there, and every check short of this one would still pass

    Given: A freshly reset racetrack-v0 and the lane its ego starts on
    When: build_track_geometry walks that lane's loop
    Then: Every segment records exactly two lanes, the walked one at zero and a neighbour
        5 m away to within 0.1 m, all on the same side, and the lap width is 5 m

    Test type: integration
    """
    geometry = _real_track_geometry()
    offsets = geometry.padded_lane_offsets()

    assert offsets.shape == (_REAL_SEGMENT_COUNT, _REAL_LANE_COUNT)
    assert geometry.lane_width_m == pytest.approx(_REAL_LANE_WIDTH_M)
    # Sorted by absolute offset, so column 0 is the walked lane itself.
    np.testing.assert_allclose(offsets[:, 0], 0.0, atol=1e-9)

    neighbour = offsets[:, 1]
    np.testing.assert_allclose(
        np.abs(neighbour), _REAL_LANE_WIDTH_M, atol=_REAL_LANE_OFFSET_TOLERANCE_M
    )
    assert len(np.unique(np.sign(neighbour))) == 1


def test_real_racetrack_centreline_pose_stays_on_the_lap_it_came_from() -> None:
    """Re-integrating the live profile reproduces a lap of the right length and turn.

    Purpose: The pose table is derived from the curvature profile rather than from the lane
        objects, so it is only as good as that profile. Pinning how far it drifts is what
        keeps a future change from making it quietly worse -- and what justifies the
        renderer treating the residual as acceptable

    Given: The live circuit's profile, sampled every 1.5 m
    When: The samples are compared against the lap they should trace
    Then: The table covers the lap at the requested spacing, consecutive samples are 1.5 m
        apart, and the heading turns through one full revolution to within 0.3 rad

    Test type: integration
    """
    geometry = _real_track_geometry()
    step = 1.5

    positions, headings = geometry.centreline_pose(step)

    assert len(positions) == int(np.ceil(geometry.total_length_m / step))
    np.testing.assert_allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1), step, atol=1e-3)
    # One lap of a closed circuit is one full turn. It does not land exactly on 2*pi: the
    # profile records one curvature per lane segment, and re-integrating those leaves 0.28
    # rad and 1.9 m on the table. Inside the 27 m window a renderer uses that is a fraction
    # of a cell; only a window straddling the start line sees the whole of it.
    assert abs(float(headings[-1])) == pytest.approx(2.0 * np.pi, abs=0.3)


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
