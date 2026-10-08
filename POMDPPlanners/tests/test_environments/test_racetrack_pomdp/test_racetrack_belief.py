# SPDX-License-Identifier: MIT

"""Tests for the racetrack tracked-agents belief.

Detection blocks are built by hand and ``reinvigorate`` is called directly, the same way the
base class calls it after its weight update, so no simulator or generative model is needed
here.

The tests that matter most are the ones about **what a detection carries**. A detection now
reports a vehicle's whole relative velocity, so both components are copied straight into the
slot and nothing is resolved along a line of sight. A vehicle crossing the ego's path abeam
used to be stamped at zero velocity because nothing radial was measured; it is now stamped at
the rate it is really crossing at, and several tests below exist to hold that flip in place.

What stays hidden is a vehicle that produced *no* detection at all — outside
``max_detection_range_m``, or behind a closer one. That is the weight update's business, not
the stamp's, so these tests say nothing about it; ``agent_velocity_jitter`` is now ordinary
measurement noise plus slack for the constant-velocity drift, not a stand-in for a component
the sensor never reported.
"""

import pickle
import warnings
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    DETECTION_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    ObservationMode,
)

ONE_AGENT = 1

# A vehicle 10 m dead ahead, closing at 2 m/s and not crossing at all: the simplest row there
# is, and the one the sign convention is fixed against.
AHEAD_CLOSING = [1.0, 10.0, 0.0, -2.0, 0.0]
# A vehicle 10 m directly abeam, moving at 6 m/s along the ego's forward axis. That velocity
# is entirely perpendicular to the line of sight, so it is exactly what the old radial
# projection threw away and what the current stamp has to keep.
ABEAM_CROSSING = [1.0, 0.0, 10.0, 6.0, 0.0]
# A vehicle off the nose whose four numbers are all different in size and two of them in sign,
# so a swapped column pair cannot pass by symmetry.
OFF_THE_NOSE = [1.0, 8.0, -3.0, -2.0, 5.0]


def detections(*rows: Any) -> np.ndarray:
    """A ``(D, 5)`` detection block, which is what the sensor emits ordered by range."""
    if not rows:
        return np.zeros((0, DETECTION_SLOT_WIDTH), dtype=float)
    return np.array(rows, dtype=float)


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


def test_a_detection_dead_ahead_stamps_its_reported_velocity_onto_the_forward_axis():
    """Test the simplest geometry there is, which is where the conventions are fixed.

    Purpose: Validates the stamp in the case with nothing to argue about, which fixes the
        sign convention and the units before any of the harder rows are read. A closing
        vehicle must arrive with a *negative* forward rate, because the range is shrinking;
        the opposite sign would have every rollout predict opponents running away

    Given: A belief with one agent slot, jitter off, and one detection 10 m dead ahead
        closing at 2 m/s
    When: reinvigorate is called on it
    Then: Every particle carries [1, 10, 0, -2, 0] exactly

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)

    refreshed = stamp(belief, {"detections": detections(AHEAD_CLOSING)}, particles)

    np.testing.assert_allclose(
        agent_rows(refreshed)[:, 0], np.tile([1.0, 10.0, 0.0, -2.0, 0.0], (4, 1))
    )


def test_a_detection_directly_abeam_stamps_the_crossing_rate_the_projection_used_to_discard():
    """Test the geometry the previous design deliberately got wrong.

    Purpose: This is the flip that the detection redesign is for, and the case it is sharpest
        on. A vehicle abeam is moving perpendicular to its own line of sight, so a Doppler
        closing rate measured nothing about it and the belief used to stamp it as stationary
        -- a car crossing at 6 m/s predicted, by every rollout, to still be there next step.
        The detection now carries both components, so the stamp has to be the real crossing
        rate. Stamping zero here again would be the old blind spot coming back

    Given: A belief with jitter off and one detection 10 m directly abeam of the ego, moving
        at 6 m/s along the ego's forward axis -- entirely perpendicular to the line of sight
    When: reinvigorate is called on it
    Then: The slot carries the full 6 m/s, not the zero the radial projection produced

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)

    refreshed = stamp(belief, {"detections": detections(ABEAM_CROSSING)}, particles)

    np.testing.assert_allclose(agent_rows(refreshed)[0, 0], [1.0, 0.0, 10.0, 6.0, 0.0])


def test_a_detection_whose_components_all_differ_keeps_each_one_in_its_own_column():
    """Test the copy column by column, against a row that cannot pass by symmetry.

    Purpose: The stamp is now two slice assignments, and the failure mode a slice invites is
        a swapped or shifted pair -- position written into the velocity columns, or rel_vx and
        rel_vy exchanged. A row with equal components, or with velocity lying along the line
        of sight, passes every such mistake; this one fails all of them, because no two of its
        four numbers agree

    Given: A belief with jitter off and one detection at (8, -3) m moving at (-2, 5) m/s
    When: reinvigorate is called on it
    Then: Every column arrives where it was sent, signs included

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)

    refreshed = stamp(belief, {"detections": detections(OFF_THE_NOSE)}, particles)

    np.testing.assert_allclose(agent_rows(refreshed)[0, 0], [1.0, 8.0, -3.0, -2.0, 5.0])


def test_a_detection_at_zero_range_is_stamped_like_any_other_and_stays_finite():
    """Test the bearing that used to need a special case, now that nothing is divided by it.

    Purpose: A detection on the ego's own origin has no bearing, so the old projection
        divided by a zero range and needed a guard that zeroed the velocity to avoid emitting
        a NaN into every rollout. The copy has no such singularity, and the guard is gone with
        it -- so the reading that reaches the planner is now the vehicle's real velocity. The
        finiteness check is kept because a NaN here still poisons a whole rollout, and it is
        the degenerate row that would produce one

    Given: A belief with jitter off and one detection at the origin of the ego body frame
        closing at 3 m/s
    When: reinvigorate is called on it
    Then: The slot is present at the origin carrying that 3 m/s, and every field is finite

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)

    refreshed = stamp(belief, {"detections": detections([1.0, 0.0, 0.0, -3.0, 0.0])}, particles)

    row = agent_rows(refreshed)[0, 0]
    assert np.all(np.isfinite(row))
    np.testing.assert_allclose(row, [1.0, 0.0, 0.0, -3.0, 0.0])


def test_undetected_slots_are_dropped_rather_than_stamped_as_vehicles_at_the_origin():
    """Test that the zero rows a real reading pads with are filtered on the presence flag.

    Purpose: The sensor emits a fixed-width block with undetected slots left at zero. Read
        without the presence filter, each of those becomes a vehicle sitting on the ego's own
        bumper -- which the model would score as an immediate collision on every step

    Given: A belief with two agent slots and a four-row detection block holding one real
        detection followed by three all-zero rows
    When: reinvigorate is called on it
    Then: Slot 0 carries the detection and slot 1 stays entirely zero

    Test type: unit
    """
    particles = make_particles(count=2, max_tracked_agents=2)
    belief = make_belief(particles, max_tracked_agents=2)
    block = np.zeros((4, DETECTION_SLOT_WIDTH), dtype=float)
    block[0] = AHEAD_CLOSING

    rows = agent_rows(stamp(belief, {"detections": block}, particles))

    np.testing.assert_allclose(rows[0, 0], [1.0, 10.0, 0.0, -2.0, 0.0])
    np.testing.assert_array_equal(rows[0, 1], np.zeros(AGENT_SLOT_WIDTH))


def test_more_detections_than_slots_warns_and_drops_the_furthest():
    """Test that overflowing the fixed agent slots is reported rather than absorbed.

    Purpose: The truncation is intended -- the state carries a fixed number of slots and the
        nearest vehicles are the ones that matter -- but doing it quietly turns a visible
        opponent into empty road, which is a false negative in the only channel the POMDP arm
        has and looks exactly like success

    Given: A belief with two agent slots and a three-detection block ordered by range, as the
        sensor emits it, at 5 m, 10 m and 20 m ahead
    When: reinvigorate is called on it
    Then: A UserWarning names both counts and how many were dropped, and the two nearest
        detections are the ones occupying the slots

    Test type: unit
    """
    particles = make_particles(count=2, max_tracked_agents=2)
    belief = make_belief(particles, max_tracked_agents=2)
    block = detections(
        [1.0, 5.0, 0.0, -1.0, 0.0],
        [1.0, 10.0, 0.0, -2.0, 0.0],
        [1.0, 20.0, 0.0, -3.0, 0.0],
    )

    with pytest.warns(
        UserWarning, match=r"Observed 3 detections.*max_tracked_agents=2.*1 furthest"
    ):
        rows = agent_rows(stamp(belief, {"detections": block}, particles))

    assert rows.shape[1] == 2
    np.testing.assert_allclose(rows[0, 0], [1.0, 5.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(rows[0, 1], [1.0, 10.0, 0.0, -2.0, 0.0])


def test_exactly_filling_the_agent_slots_does_not_warn():
    """Test that the overflow guard fires on overflow only.

    Purpose: A warning that fired whenever the slots were merely full would be ignored within
        one episode, and the real overflow would then be ignored along with it

    Given: A belief with two agent slots and a block holding exactly two detections
    When: reinvigorate is called with warnings promoted to errors
    Then: Nothing is raised and both detections are stamped

    Test type: unit
    """
    particles = make_particles(count=2, max_tracked_agents=2)
    belief = make_belief(particles, max_tracked_agents=2)
    block = detections([1.0, 5.0, 0.0, -1.0, 0.0], [1.0, 10.0, 0.0, -2.0, 0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rows = agent_rows(stamp(belief, {"detections": block}, particles))

    np.testing.assert_allclose(rows[0, 0], [1.0, 5.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(rows[0, 1], [1.0, 10.0, 0.0, -2.0, 0.0])


def test_an_empty_reading_leaves_every_slot_zeroed_and_unjittered():
    """Test that empty road produces empty slots rather than jittered phantoms.

    Purpose: Validates that jitter is applied to present slots only. Noise on an absent slot
        would hand the planner a vehicle at a random position that nothing observed, and with
        a wide jitter that phantom lands close enough to matter

    Given: A belief with a 5 m and 5 m/s jitter, and a reading holding no detections at all
    When: reinvigorate is called
    Then: Every particle's agent block is exactly zero

    Test type: unit
    """
    np.random.seed(1)
    particles = make_particles()
    belief = make_belief(particles, agent_pose_jitter=5.0, agent_velocity_jitter=5.0)

    refreshed = stamp(belief, {"detections": detections()}, particles)

    np.testing.assert_array_equal(agent_rows(refreshed), np.zeros((4, 1, AGENT_SLOT_WIDTH)))


def test_a_missing_detections_block_raises_instead_of_stamping_empty_road():
    """Test that a POMDP belief handed a reading with no detections key fails loudly.

    Purpose: Validates that the one configuration mistake this class invites cannot degrade
        quietly. A belief that found no detections and stamped zeros would produce a planner
        that sees no traffic at all, runs every episode to completion, and looks like it works

    Given: A belief in POMDP mode and an observation carrying only the MDP arm's agents block
    When: reinvigorate is called
    Then: ValueError names the missing key and says what silence would have cost

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles)
    observation = {"agents": np.zeros((ONE_AGENT, AGENT_SLOT_WIDTH))}

    with pytest.raises(ValueError, match="sees no traffic at all and still looks like it works"):
        stamp(belief, observation, particles)


def test_the_readings_ego_pose_channel_is_ignored_and_the_ego_block_is_left_to_the_filter():
    """Test that the new ego-pose channel does not get stamped into the particles.

    Purpose: The observation now reports where the ego is, and stamping it would be the
        obvious thing to do with it. It is also wrong: the particles are then scored on
        agreeing with a number every one of them was just handed, which is double-counting,
        and it collapses the ego spread to zero so the likelihood becomes identical across
        the belief and stops discriminating at all. The ego block is the weight update's job
        -- unlike an opponent, the ego cannot fail to be acquired -- so this class must leave
        it exactly as the filter produced it

    Given: A belief in POMDP mode over particles whose ego blocks all differ, and a reading
        carrying both an ego pose that matches none of them and one detection
    When: reinvigorate is called
    Then: Every ego block comes back bit-for-bit unchanged, spread intact, while the agent
        slot is stamped from the detection

    Test type: unit
    """
    particles = make_particles(count=4)
    particles[:, :EGO_STATE_WIDTH] = np.arange(4 * EGO_STATE_WIDTH).reshape(4, EGO_STATE_WIDTH)
    before = particles.copy()
    belief = make_belief(particles)
    observation = {
        "ego_pose": np.array([12.0, -4.0, 0.3, 88.0]),
        "detections": detections(AHEAD_CLOSING),
    }

    refreshed = stamp(belief, observation, particles)

    ego = np.asarray(refreshed.particles, dtype=float)[:, :EGO_STATE_WIDTH]
    np.testing.assert_array_equal(ego, before[:, :EGO_STATE_WIDTH])
    np.testing.assert_allclose(agent_rows(refreshed)[0, 0], AHEAD_CLOSING)


def test_mdp_mode_stamps_the_observed_agent_rows_exactly_as_given():
    """Test that the MDP arm reads its agents block instead of any detection geometry.

    Purpose: The MDP arm's block lists every vehicle on the road, in range or not. Deriving
        the rows from the detections instead would hand the baseline the POMDP arm's range
        gate and its occlusions -- which, now that both arms stamp the same five numbers per
        vehicle, is the *only* thing left separating them and the whole content of the
        comparison

    Given: A belief constructed in MDP mode and an agents block holding one vehicle 12 m ahead
        with a relative velocity of (-4, 0.5)
    When: reinvigorate is called
    Then: Every particle carries that row unchanged

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles, observation_mode=ObservationMode.MDP)
    observation = {"agents": np.array([[1.0, 12.0, 0.0, -4.0, 0.5]])}

    refreshed = stamp(belief, observation, particles)

    np.testing.assert_allclose(agent_rows(refreshed)[:, 0], np.tile(observation["agents"], (4, 1)))


def test_mdp_mode_without_an_agents_block_raises_rather_than_reading_the_detections():
    """Test that the MDP arm refuses to fall back on the POMDP arm's reading.

    Purpose: Falling back would leave the MDP baseline quietly consuming a degraded
        observation -- the same detections the other arm reads -- so the two arms would no
        longer differ in the one thing the comparison controls for

    Given: A belief in MDP mode and an observation carrying only a detections block
    When: reinvigorate is called
    Then: ValueError names the missing agents block

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles, observation_mode=ObservationMode.MDP)

    with pytest.raises(ValueError, match="expects an 'agents' block"):
        stamp(belief, {"detections": detections(AHEAD_CLOSING)}, particles)


def test_an_mdp_agents_block_of_the_wrong_size_is_rejected():
    """Test that an agents block sized for a different slot count raises.

    Purpose: The block is reshaped straight into the fixed slots, so a size mismatch is
        either a cryptic reshape error or, worse, a silent misalignment that shifts every
        field of every row by one

    Given: A belief configured for one agent slot and an agents block holding two vehicles
    When: reinvigorate is called
    Then: ValueError names the value count and the slot count

    Test type: unit
    """
    particles = make_particles()
    belief = make_belief(particles, observation_mode=ObservationMode.MDP)
    observation = {"agents": np.zeros((2, AGENT_SLOT_WIDTH))}

    with pytest.raises(ValueError, match="max_tracked_agents=1"):
        stamp(belief, observation, particles)


def test_particles_narrower_than_the_agent_slots_are_rejected():
    """Test that a belief whose slot count disagrees with the particle width raises.

    Purpose: Validates a loud failure on a configuration mistake whose quiet form is far
        worse. The agent rows are written straight into fixed slots, so a mismatch is either
        a cryptic broadcast error or a silent misalignment of every field

    Given: Particles sized for one agent slot and a belief configured for two
    When: reinvigorate is called
    Then: ValueError is raised naming both widths

    Test type: unit
    """
    particles = make_particles(max_tracked_agents=1)
    belief = make_belief(particles, max_tracked_agents=2)

    with pytest.raises(ValueError, match="max_tracked_agents"):
        stamp(belief, {"detections": detections(AHEAD_CLOSING)}, particles)


def test_zero_jitter_stamps_the_exact_row_and_each_width_spreads_only_its_own_block():
    """Test the two jitter widths against the two different things they stand for.

    Purpose: Validates the split, and the reason there are two widths rather than one. The
        position and the velocity are measured to different accuracies and propagated with
        different amounts of model error -- the velocity width also has to carry the drift of
        a constant-velocity rollout, which the position width does not -- so a single shared
        std would either erase the position information or under-spread the velocity. Both
        blocks must stay centred on what was reported: jitter is noise around a measurement,
        not a correction to it

    Given: 20000 particles and one detection off the nose at (8, -3) m moving at (-2, 5) m/s,
        stamped twice: once with both widths at zero and once at 0.5 m and 4.0 m/s
    When: reinvigorate stamps them
    Then: The zero-width stamp is identical on every particle; the jittered one is centred on
        the same four numbers, with each block spread at its own width and neither at the
        other's

    Test type: unit
    """
    np.random.seed(7)
    particles = make_particles(count=20000)
    exact = agent_rows(
        stamp(make_belief(particles), {"detections": detections(OFF_THE_NOSE)}, particles)
    )
    belief = make_belief(particles, agent_pose_jitter=0.5, agent_velocity_jitter=4.0)

    rows = agent_rows(stamp(belief, {"detections": detections(OFF_THE_NOSE)}, particles))[:, 0]

    np.testing.assert_allclose(exact[:, 0], np.tile(OFF_THE_NOSE, (20000, 1)))
    assert np.all(rows[:, 0] == 1.0)
    np.testing.assert_allclose(rows[:, 1:3].mean(axis=0), [8.0, -3.0], atol=0.02)
    np.testing.assert_allclose(rows[:, 3:5].mean(axis=0), [-2.0, 5.0], atol=0.15)
    np.testing.assert_allclose(rows[:, 1:3].std(axis=0), [0.5, 0.5], rtol=0.05)
    np.testing.assert_allclose(rows[:, 3:5].std(axis=0), [4.0, 4.0], rtol=0.05)


def test_config_id_covers_the_jitter_widths_and_is_stable_across_identical_beliefs():
    """Test that the belief's identity covers configuration and only configuration.

    Purpose: config_id feeds __hash__ and __eq__ and the simulator's result caching, so it has
        to be stable across two beliefs configured the same way. The jitter widths belong in
        it because they change how the next step behaves; this step's reading does not, and
        folding it in would give an identity that changed every frame

    Given: Two beliefs built separately over identical particles with the same jitter widths,
        and two more differing only in one width apiece
    When: Their config_id values and hashes are compared
    Then: The two identically configured beliefs agree on both, and each changed width
        produces a different identity

    Test type: unit
    """
    particles = make_particles()
    one = make_belief(particles, agent_pose_jitter=0.5, agent_velocity_jitter=1.0)
    same = make_belief(particles, agent_pose_jitter=0.5, agent_velocity_jitter=1.0)
    wider_pose = make_belief(particles, agent_pose_jitter=2.0, agent_velocity_jitter=1.0)
    wider_velocity = make_belief(particles, agent_pose_jitter=0.5, agent_velocity_jitter=3.0)

    assert one.config_id == same.config_id
    assert hash(one) == hash(same)
    assert wider_pose.config_id != one.config_id
    assert wider_velocity.config_id != one.config_id


def test_reinvigorate_returns_a_belief_configured_to_stamp_again_next_step():
    """Test that the successor keeps stamping instead of going inert after one step.

    Purpose: A plain particle belief coming back here would stop the stamping after one step
        and leave every later agent slot holding whatever the transition model drifted it to,
        with no observation correcting it -- and the episode would still run to completion

    Given: A belief with two agent slots and non-default jitter widths, and one reading
    When: reinvigorate returns a successor
    Then: The successor is a TrackedAgentsBelief carrying the same mode, slot count and jitter
        widths as the belief it came from

    Test type: unit
    """
    particles = make_particles(count=2, max_tracked_agents=2)
    belief = make_belief(
        particles, max_tracked_agents=2, agent_pose_jitter=0.25, agent_velocity_jitter=1.5
    )

    refreshed = stamp(belief, {"detections": detections(AHEAD_CLOSING)}, particles)

    assert isinstance(refreshed, TrackedAgentsBelief)
    assert refreshed.observation_mode is belief.observation_mode
    assert refreshed.max_tracked_agents == 2
    assert refreshed.agent_pose_jitter == 0.25
    assert refreshed.agent_velocity_jitter == 1.5


def test_pickle_round_trip_preserves_the_stamping_configuration():
    """Test that a belief shipped to a worker process keeps stamping the way it was built.

    Purpose: The simulator pickles beliefs across processes. A jitter width lost in transit
        would come back as the default, so a run tuned for a wide velocity spread would
        silently narrow it at the process boundary and the change would show up only as
        slightly worse returns

    Given: A belief with a non-default slot count and both jitter widths set
    When: It is pickled and unpickled
    Then: The restored belief carries the same configuration and the same identity

    Test type: unit
    """
    belief = make_belief(
        make_particles(count=2, max_tracked_agents=2),
        max_tracked_agents=2,
        agent_pose_jitter=0.25,
        agent_velocity_jitter=1.5,
    )

    restored = pickle.loads(pickle.dumps(belief))

    assert restored.max_tracked_agents == 2
    assert restored.agent_pose_jitter == 0.25
    assert restored.agent_velocity_jitter == 1.5
    assert restored.config_id == belief.config_id


def test_the_class_docstring_example_stamps_the_crossing_rate_it_advertises():
    """Test the usage example in the TrackedAgentsBelief docstring.

    Purpose: The example is what a reader copies, and it is also the class's own claim about
        the redesign: it shows a vehicle closing at 2 m/s *and* crossing left at 3 m/s and
        promises both numbers reach the slot. An example that drifted from the code would
        advertise the old radial behaviour long after it was removed

    Given: The example's own four zero particles, one agent slot and both jitter widths off,
        and its detection 10 m ahead closing at 2 m/s and crossing left at 3 m/s
    When: reinvigorate is called exactly as the example calls it
    Then: The agent block is [1, 10, 0, -2, 3], the crossing rate included, and the successor
        is another TrackedAgentsBelief so the stamping repeats next step

    Test type: example
    """
    np.random.seed(0)
    width = EGO_STATE_WIDTH + 1 * 5
    particles = np.zeros((4, width))
    belief = TrackedAgentsBelief(
        particles=particles,
        log_weights=np.log(np.ones(4) / 4),
        max_tracked_agents=1,
        agent_pose_jitter=0.0,
        agent_velocity_jitter=0.0,
    )
    # The example names this `detections`; renamed here only to keep the module-level helper
    # of that name visible. The reinvigorate call below is the example's, verbatim.
    reported = np.array([[1.0, 10.0, 0.0, -2.0, 3.0]])
    pomdp: Any = None

    refreshed = belief.reinvigorate("noop", {"detections": reported}, pomdp, as_base(particles))

    np.testing.assert_array_equal(
        np.asarray(refreshed.particles)[0, EGO_STATE_WIDTH:], [1.0, 10.0, 0.0, -2.0, 3.0]
    )
    assert isinstance(refreshed, TrackedAgentsBelief)
