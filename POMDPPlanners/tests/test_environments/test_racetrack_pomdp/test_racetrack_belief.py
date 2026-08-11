# SPDX-License-Identifier: MIT

"""Tests for the racetrack tracked-agents belief.

Grids are built by hand and ``reinvigorate`` is called directly, the same way the base class
calls it after its weight update, so no simulator or generative model is needed here.
"""

# The carried frames are private on purpose -- config_id must not see them -- so the tests
# that pin that behaviour have to reach for the private attributes.
# pylint: disable=protected-access

import pickle
from typing import Any, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_occupancy_tracker import (
    OccupancyVelocityTracker,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    GRID_CELLS,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    ObservationMode,
    PRESENCE_LAYER,
)

EGO_CELL = (6, 6)
ONE_AGENT = 1
ONE_CELL_SPEED = GRID_STEP_M / 0.2


def grid_with(*cells: Tuple[int, int]) -> np.ndarray:
    """A presence grid marking ``cells``, plus the ego at the centre as the simulator does."""
    grid = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    grid[PRESENCE_LAYER][EGO_CELL] = 1.0
    for cell in cells:
        grid[PRESENCE_LAYER][cell] = 1.0
    return grid


def cell_centre(along: int, across: int) -> np.ndarray:
    """The ego-frame metre position of a cell's centre."""
    return -GRID_HALF_EXTENT_M + (np.array([along, across], dtype=float) + 0.5) * GRID_STEP_M


def make_particles(count: int = 4, max_tracked_agents: int = ONE_AGENT) -> np.ndarray:
    """``count`` all-zero particles of the right width for ``max_tracked_agents``."""
    width = EGO_STATE_WIDTH + max_tracked_agents * AGENT_SLOT_WIDTH
    return np.zeros((count, width), dtype=float)


def make_belief(particles: np.ndarray, **kwargs) -> TrackedAgentsBelief:
    """A belief over ``particles`` with jitter off unless a test asks for it."""
    kwargs.setdefault("max_tracked_agents", ONE_AGENT)
    kwargs.setdefault("agent_pose_jitter", 0.0)
    kwargs.setdefault("agent_velocity_jitter", 0.0)
    return TrackedAgentsBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
        **kwargs,
    )


def as_base(particles: np.ndarray) -> WeightedParticleBelief:
    """The plain belief the base class hands to reinvigorate after its weight update."""
    # WeightedParticleBelief annotates `particles` as a list but documents and accepts a 2-D
    # ndarray, which is what the production path passes.
    particle_arg: Any = particles
    return WeightedParticleBelief(
        particles=particle_arg,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def stamp(
    belief: TrackedAgentsBelief, observation: Any, particles: np.ndarray
) -> TrackedAgentsBelief:
    """Call reinvigorate the way the base class does. No environment is read, so None is fine."""
    pomdp: Any = None
    return belief.reinvigorate("noop", observation, pomdp, as_base(particles))


def agent_rows(belief: WeightedParticleBelief) -> np.ndarray:
    """Every particle's agent block, reshaped to one row per slot."""
    particles = np.asarray(belief.particles, dtype=float)
    return particles[:, EGO_STATE_WIDTH:].reshape(len(particles), -1, AGENT_SLOT_WIDTH)


def test_first_step_stamps_positions_with_zero_velocity_and_second_step_measures_motion():
    """Test that velocity appears only once there are two frames to difference.

    Purpose: Validates the episode's first step, where inventing a velocity from a single
        frame would be worse than admitting none, and validates that the carried frame makes
        the following step a real measurement

    Given: A belief with one agent slot, an opponent at along-index 9 on the first frame and
        at along-index 8 on the second, and jitter switched off
    When: reinvigorate is called on the first frame and then again on the second
    Then: The first stamp carries the opponent's position with zero velocity, and the second
        carries the closing speed of one cell per step

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)

    first = stamp(belief, {"occupancy": grid_with((9, 6))}, particles)
    np.testing.assert_allclose(agent_rows(first)[0, 0], [1.0, *cell_centre(9, 6), 0.0, 0.0])

    second = stamp(first, {"occupancy": grid_with((8, 6))}, np.asarray(first.particles))
    np.testing.assert_allclose(
        agent_rows(second)[0, 0], [1.0, *cell_centre(8, 6), -ONE_CELL_SPEED, 0.0], atol=1e-9
    )


def test_reinvigorate_returns_a_tracked_belief_carrying_the_grid_forward():
    """Test that the returned belief is set up to track again on the next step.

    Purpose: Validates the carry-forward. A plain particle belief coming back here would stop
        the stamping after one step and silently leave every later agent slot stale.

    Given: A belief and one occupancy observation
    When: reinvigorate is called
    Then: The result is a TrackedAgentsBelief whose carried frame equals the observed grid
        but is a copy rather than the caller's array

    Test type: unit
    """
    particles = make_particles()
    grid = grid_with((9, 6))

    refreshed = stamp(make_belief(particles), {"occupancy": grid}, particles)

    assert isinstance(refreshed, TrackedAgentsBelief)
    carried_grid, carried_heading = refreshed._previous_frames[0]
    np.testing.assert_array_equal(carried_grid, grid)
    assert carried_grid is not grid
    assert carried_heading == 0.0


def test_mdp_mode_stamps_the_observed_rows_without_a_tracker():
    """Test that the MDP arm uses the observation's agent rows directly.

    Purpose: Validates that the near-MDP baseline is not degraded through the tracker. Its
        rows already carry exact velocities, so re-deriving them from a grid would hand the
        baseline the POMDP arm's quantisation error and destroy the matched comparison.

    Given: A belief constructed in MDP mode and an observation whose agents block holds one
        present vehicle 12 m ahead closing at 4 m/s
    When: reinvigorate is called
    Then: No tracker was ever constructed, and every particle carries the observed row exactly

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles, observation_mode=ObservationMode.MDP)
    observation = {"agents": np.array([[1.0, 12.0, 0.0, -4.0, 0.5]])}

    refreshed = stamp(belief, observation, particles)

    assert belief.tracker is None
    assert refreshed.tracker is None
    np.testing.assert_allclose(agent_rows(refreshed)[:, 0], np.tile(observation["agents"], (4, 1)))


@pytest.mark.parametrize(
    "mode, observation, expected",
    [
        (ObservationMode.POMDP, {"agents": np.zeros((1, AGENT_SLOT_WIDTH))}, "occupancy"),
        (ObservationMode.MDP, {"occupancy": np.zeros((2, GRID_CELLS, GRID_CELLS))}, "agents"),
    ],
)
def test_observation_from_the_wrong_arm_raises(mode, observation, expected):
    """Test that a belief handed the other arm's observation fails loudly.

    Purpose: Validates that a mode mismatch cannot degrade silently. A POMDP belief that
        quietly found no grid would stamp empty agent slots forever and produce a
        velocity-blind planner whose episodes still run to completion.

    Given: A belief in one observation mode and an observation carrying only the other mode's
        key
    When: reinvigorate is called
    Then: ValueError is raised naming the key that was expected

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles, observation_mode=mode)

    with pytest.raises(ValueError, match=expected):
        stamp(belief, observation, particles)


def test_config_id_ignores_the_carried_frame_but_tracks_the_jitter_width():
    """Test that the belief's identity covers configuration and not the observation history.

    Purpose: Validates the reason the carried frames are private. config_id feeds __hash__ and
        __eq__, so folding a per-step grid into it would make two otherwise identical beliefs
        unequal; the jitter widths, by contrast, change how the next step behaves and do
        belong in the identity.

    Given: Three beliefs over identical particles -- one untouched, one carrying a previous
        frame, and one with a wider pose jitter
    When: Their config_id values are compared
    Then: The carried frame leaves config_id unchanged while the jitter width changes it

    Test type: unit
    """
    particles = make_particles()
    plain = make_belief(particles)
    carrying = make_belief(particles, previous_frames=((grid_with((9, 6)), 0.7),))
    jittered = make_belief(particles, agent_pose_jitter=2.0)

    assert carrying.config_id == plain.config_id
    assert hash(carrying) == hash(plain)
    assert jittered.config_id != plain.config_id


def test_pickle_round_trip_preserves_the_carried_frame():
    """Test that a belief shipped to a worker process keeps its tracking history.

    Purpose: Validates that the private carry-forward survives serialization. The simulator
        pickles beliefs across processes, and a frame lost in transit would silently reset
        every opponent's velocity to zero on the step after the hand-off.

    Given: A belief carrying one previous frame and the ego heading that went with it
    When: It is pickled and unpickled
    Then: The carried grid and heading come back unchanged

    Test type: unit
    """
    belief = make_belief(make_particles(), previous_frames=((grid_with((9, 6)), 0.25),))

    restored = pickle.loads(pickle.dumps(belief))

    np.testing.assert_array_equal(restored._previous_frames[0][0], belief._previous_frames[0][0])
    assert restored._previous_frames[0][1] == 0.25


def test_pose_and_velocity_jitter_are_zero_mean_and_use_their_own_widths():
    """Test that the two jitter widths are applied to their own halves of the agent row.

    Purpose: Validates the split. The grid fixes an opponent's position to about a cell but
        quantises its velocity at 15 m/s, so a single shared width would either wash out the
        position or claim a precision the velocity does not have.

    Given: 20000 particles, one present agent, a pose jitter of 0.5 m and a velocity jitter
        of 4.0 m/s
    When: reinvigorate stamps them
    Then: Both jittered pairs are centred on the tracked row, the position spread matches the
        pose width, the velocity spread matches the velocity width, and the presence flag is
        left exactly 1.0

    Test type: unit
    """
    np.random.seed(7)
    particles = make_particles(count=20000)
    belief = make_belief(particles, agent_pose_jitter=0.5, agent_velocity_jitter=4.0)

    rows = agent_rows(stamp(belief, {"occupancy": grid_with((9, 6))}, particles))[:, 0]

    assert np.all(rows[:, 0] == 1.0)
    np.testing.assert_allclose(rows[:, 1:3].mean(axis=0), cell_centre(9, 6), atol=0.02)
    np.testing.assert_allclose(rows[:, 3:5].mean(axis=0), [0.0, 0.0], atol=0.15)
    np.testing.assert_allclose(rows[:, 1:3].std(axis=0), [0.5, 0.5], rtol=0.05)
    np.testing.assert_allclose(rows[:, 3:5].std(axis=0), [4.0, 4.0], rtol=0.05)


def test_absent_slots_stay_empty_and_the_nearest_agents_win_the_slots():
    """Test slot allocation when the frame holds more vehicles than there are slots.

    Purpose: Validates that the fixed-width agent block is filled nearest-first and zero
        padded. A far vehicle displacing a near one would hide the collision risk that the
        slots exist to represent.

    Given: A belief with two agent slots and a frame holding three opponents at increasing
        range
    When: reinvigorate is called
    Then: The two nearest opponents occupy the slots in range order and no third row exists

    Test type: unit
    """
    particles = make_particles(count=2, max_tracked_agents=2)
    belief = make_belief(particles, max_tracked_agents=2)
    grid = grid_with((7, 6), (9, 6), (11, 6))

    rows = agent_rows(stamp(belief, {"occupancy": grid}, particles))

    assert rows.shape[1] == 2
    np.testing.assert_allclose(rows[0, 0], [1.0, *cell_centre(7, 6), 0.0, 0.0])
    np.testing.assert_allclose(rows[0, 1], [1.0, *cell_centre(9, 6), 0.0, 0.0])


def test_empty_frame_leaves_every_slot_zeroed_and_unjittered():
    """Test that an empty window produces empty slots rather than jittered phantoms.

    Purpose: Validates that jitter is applied to present slots only. Noise on an absent slot
        would give the planner a vehicle at a random position that nothing observed.

    Given: A belief with a wide jitter and a frame holding nothing but the ego
    When: reinvigorate is called
    Then: Every particle's agent block is exactly zero

    Test type: unit
    """
    np.random.seed(1)
    particles = make_particles()
    belief = make_belief(particles, agent_pose_jitter=5.0, agent_velocity_jitter=5.0)

    refreshed = stamp(belief, {"occupancy": grid_with()}, particles)

    np.testing.assert_array_equal(agent_rows(refreshed), np.zeros((4, 1, AGENT_SLOT_WIDTH)))


def test_ego_yaw_change_between_frames_is_taken_from_the_particles():
    """Test that the belief de-rotates using the heading change its own particles imply.

    Purpose: Validates that the yaw delta is read from the particle set rather than plumbed in
        from outside. The occupancy grid turns with the ego, so a belief that could not
        recover its own heading change would report a static world as moving.

    Given: A world-static opponent seen at cell (11, 6) and then, after the ego yaws by about
        0.18 rad, at cell (11, 5) -- with the particles' ego heading advanced by that same
        angle between the two calls
    When: reinvigorate is called on both frames in turn
    Then: The second stamp reports essentially zero relative velocity, not the 15 m/s the
        frame rotation alone would suggest

    Test type: unit
    """
    yaw_delta = 2.0 * np.arctan2(1.5, 16.5)
    particles = make_particles()
    first = stamp(make_belief(particles), {"occupancy": grid_with((11, 6))}, particles)

    turned = np.asarray(first.particles, dtype=float).copy()
    turned[:, 2] += yaw_delta  # EGO_HEADING
    second = stamp(first, {"occupancy": grid_with((11, 5))}, turned)

    np.testing.assert_allclose(agent_rows(second)[0, 0, 3:5], [0.0, 0.0], atol=1e-9)


def test_an_explicit_tracker_is_reused_by_the_successor_belief():
    """Test that a caller-supplied tracker is carried into the next belief.

    Purpose: Validates that a tuned tracker -- a wider frame_stride, say, chosen to beat the
        velocity quantisation -- is not silently replaced by the default one step later

    Given: A belief constructed with a tracker whose frame_stride is 3
    When: reinvigorate returns a successor
    Then: The successor holds the very same tracker object

    Test type: unit
    """
    tracker = OccupancyVelocityTracker(frame_stride=3)
    particles = make_particles()
    belief = make_belief(particles, tracker=tracker)

    refreshed = stamp(belief, {"occupancy": grid_with((9, 6))}, particles)

    assert refreshed.tracker is tracker


def test_particles_narrower_than_the_agent_slots_are_rejected():
    """Test that a belief whose slot count disagrees with the particle width raises.

    Purpose: Validates a loud failure on the one configuration mistake this class invites.
        The agent rows are written straight into fixed slots, so a mismatch is either a
        cryptic broadcast error or, worse, a silent misalignment of every field.

    Given: Particles sized for one agent slot and a belief configured for two
    When: reinvigorate is called
    Then: ValueError is raised naming both widths

    Test type: unit
    """
    particles = make_particles(max_tracked_agents=1)
    belief = make_belief(particles, max_tracked_agents=2)

    with pytest.raises(ValueError, match="max_tracked_agents"):
        stamp(belief, {"occupancy": grid_with((9, 6))}, particles)


def test_frame_stride_widens_the_baseline_across_successive_updates():
    """Test that a strided tracker differences frames that are really that far apart.

    Purpose: Validates that frame_stride buys a wider measurement baseline rather than just a
        larger divisor. A belief that kept only the immediately preceding frame would divide a
        one-step displacement by three steps and under-report every velocity by that factor --
        a silently wrong number, which is the worst outcome for a knob whose whole purpose is
        to make the coarse velocities more trustworthy.

    Given: A belief whose tracker has frame_stride 2, and an opponent closing by one cell on
        each of three successive frames
    When: reinvigorate is called on each frame in turn
    Then: The first two steps report position only, and the third reports the true closing
        speed of 15 m/s measured over the two-step baseline -- not the 7.5 m/s a
        consecutive-frame difference divided by two steps would give

    Test type: unit
    """
    tracker = OccupancyVelocityTracker(frame_stride=2, gate_radius_m=9.0)
    particles = make_particles()
    belief = make_belief(particles, tracker=tracker)

    speeds = []
    for along in (10, 9, 8):
        belief = stamp(belief, {"occupancy": grid_with((along, 6))}, particles)
        speeds.append(float(agent_rows(belief)[0, 0, 3]))

    assert speeds[:2] == [0.0, 0.0]
    assert speeds[2] == pytest.approx(-ONE_CELL_SPEED)


def test_yaw_delta_follows_the_particle_weights_not_the_particle_count():
    """Test that the de-rotation is driven by the weighted mean heading.

    Purpose: Validates that low-probability particles cannot steer the de-rotation. The base
        filter resamples only when the effective sample size drops, so between resamples a
        cloud can be dominated by a few heavy particles; an unweighted mean would let the
        light majority rotate the reference frame and manufacture velocity.

    Given: A world-static opponent, and a second frame whose particles hold the true turned
        heading on one heavy particle and the untouched heading on three light ones
    When: reinvigorate is called on both frames in turn
    Then: The de-rotation follows the heavy particle and the reported relative velocity is
        essentially zero, which the unweighted mean of the four headings could not produce

    Test type: unit
    """
    yaw_delta = 2.0 * np.arctan2(1.5, 16.5)
    particles = make_particles()
    first = stamp(make_belief(particles), {"occupancy": grid_with((11, 6))}, particles)

    turned = np.asarray(first.particles, dtype=float).copy()
    turned[0, 2] = yaw_delta  # EGO_HEADING on the one heavy particle
    weighted: Any = turned
    base = WeightedParticleBelief(
        particles=weighted, log_weights=np.log(np.array([1e6, 1.0, 1.0, 1.0]))
    )
    pomdp: Any = None
    second = first.reinvigorate("noop", {"occupancy": grid_with((11, 5))}, pomdp, base)

    np.testing.assert_allclose(agent_rows(second)[0, 0, 3:5], [0.0, 0.0], atol=1e-3)
