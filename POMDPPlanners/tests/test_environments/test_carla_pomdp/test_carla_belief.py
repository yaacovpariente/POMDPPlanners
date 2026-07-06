# SPDX-License-Identifier: MIT

"""Tests for the traffic-reinjecting CARLA particle belief.

Covers :class:`CarlaAgentReinvigoration` overwriting each particle's ego-frame agent block
with the current observation's ``agents`` block: present slots take the measured pose plus
jitter, empty slots are cleared, the ego block is preserved, and the returned belief is
itself a :class:`CarlaAgentReinvigoration` so re-injection persists across every update. All
tests run on hand-built states with no CARLA server.
"""

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.carla_pomdp.carla_belief import CarlaAgentReinvigoration
from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
)

_MAX_AGENTS = 2
_WIDTH = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH

# A concrete Environment to satisfy reinvigorate's typed ``pomdp`` argument; the
# reinvigoration step ignores it (traffic is re-injected from the observation alone).
_POMDP = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05, max_tracked_agents=_MAX_AGENTS)


def _empty_particles(n_particles: int) -> list:
    return [np.zeros(_WIDTH) for _ in range(n_particles)]


def _belief(n_particles: int) -> CarlaAgentReinvigoration:
    return CarlaAgentReinvigoration(
        particles=_empty_particles(n_particles),
        log_weights=np.log(np.ones(n_particles) / n_particles),
        max_tracked_agents=_MAX_AGENTS,
        agent_pose_jitter=0.3,
    )


def _base_belief(particles: list) -> WeightedParticleBelief:
    weights = np.log(np.ones(len(particles)) / len(particles))
    return WeightedParticleBelief(particles=particles, log_weights=weights)


def _agents_block(rows: list) -> np.ndarray:
    """Flatten ``_MAX_AGENTS`` slot rows into an ``agents`` observation block."""
    block = np.zeros((_MAX_AGENTS, AGENT_SLOT_WIDTH))
    for slot, row in enumerate(rows):
        block[slot] = row
    return block.reshape(-1)


def test_reinvigorate_marks_observed_agent_present():
    """A present observed agent turns on the matching particle slot.

    Purpose: Validates re-injection acquires traffic the particles were not seeded with.

    Given: A belief of empty-agent particles and an observation with one present agent
    When: reinvigorate overwrites the agent block from that observation
    Then: Every particle's first slot is marked present (== 1.0)

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(6)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(6)))

    particles = np.asarray(refreshed.particles)
    assert np.all(particles[:, EGO_STATE_WIDTH] == 1.0)


def test_reinvigorate_places_agent_near_observed_pose():
    """Re-injected agents cluster around the observed pose.

    Purpose: Validates the measured pose (not zero) is written into the slot.

    Given: A belief and an observation with an agent 8 m ahead, 1 m to the left
    When: reinvigorate overwrites the agent block over many particles
    Then: The mean re-injected slot pose is close to the observed pose

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(2000)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 1.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(2000))
    )

    particles = np.asarray(refreshed.particles)
    mean_pose = particles[:, EGO_STATE_WIDTH + 1 : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH].mean(axis=0)
    np.testing.assert_allclose(mean_pose, [8.0, 1.0, 0.0, 5.0], atol=0.05)


def _particles_with_track(n_particles: int, rel_x: float, rel_y: float = 0.0) -> list:
    """Particles each carrying one present propagated track at ``(rel_x, rel_y)``."""
    seeded = _empty_particles(n_particles)
    for particle in seeded:
        particle[EGO_STATE_WIDTH] = 1.0
        particle[EGO_STATE_WIDTH + 1] = rel_x
        particle[EGO_STATE_WIDTH + 2] = rel_y
    return seeded


def test_unmatched_in_range_track_coasts_when_undetected():
    """An in-range track missing from the observation is carried forward, not dropped.

    Purpose: Validates a briefly occluded (undetected) agent survives via its motion model.

    Given: Particles with a present track 4 m ahead and an all-empty observation
    When: reinvigorate reconciles tracks with the (empty) detections
    Then: Slot 0 stays present at its propagated pose (the track coasts)

    Test type: unit
    """
    np.random.seed(0)
    seeded = _particles_with_track(5, rel_x=4.0)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([])}

    refreshed = _belief(5).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    particles = np.asarray(refreshed.particles)
    assert np.all(particles[:, EGO_STATE_WIDTH] == 1.0)
    np.testing.assert_allclose(particles[:, EGO_STATE_WIDTH + 1], 4.0)


def test_unmatched_out_of_range_track_is_evicted():
    """A track that has drifted beyond persistence_range is cleared when undetected.

    Purpose: Validates coasting does not accumulate ghosts that can never be re-confirmed.

    Given: Particles with a present track 60 m ahead (beyond the 50 m default) and empty obs
    When: reinvigorate reconciles tracks with the (empty) detections
    Then: Slot 0 is cleared on every particle

    Test type: unit
    """
    np.random.seed(0)
    seeded = _particles_with_track(5, rel_x=60.0)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([])}

    refreshed = _belief(5).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    particles = np.asarray(refreshed.particles)
    assert np.all(particles[:, EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] == 0.0)


def test_detection_near_propagated_track_associates_without_duplicating():
    """A detection near a propagated track corrects it rather than adding a second slot.

    Purpose: Validates proximity association deduplicates a re-detected track.

    Given: Particles with a track at 8 m and a detection 0.3 m away (within association_radius)
    When: reinvigorate reconciles the detection with the track
    Then: Exactly one slot is present (the corrected track), not two

    Test type: unit
    """
    np.random.seed(0)
    seeded = _particles_with_track(20, rel_x=8.0)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.3, 0.0, 0.0, 5.0]])}

    refreshed = _belief(20).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    present = np.asarray(refreshed.particles)[
        :, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH  # present flag of each slot
    ]
    np.testing.assert_array_equal(present.sum(axis=1), np.ones(20))


def test_distant_detection_and_track_produce_two_tracks():
    """A detection beyond association_radius is a new track alongside the coasted one.

    Purpose: Validates a genuinely different agent is not merged into an existing track.

    Given: Particles with an in-range track at 8 m and a detection 30 m ahead (gap > radius)
    When: reinvigorate reconciles the detection with the track
    Then: Two slots are present (the new detection plus the coasted track)

    Test type: unit
    """
    np.random.seed(0)
    seeded = _particles_with_track(20, rel_x=8.0)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 30.0, 0.0, 0.0, 5.0]])}

    refreshed = _belief(20).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    present = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH]
    np.testing.assert_array_equal(present.sum(axis=1), np.full(20, 2.0))


def test_reinvigorate_preserves_ego_block():
    """Re-injection leaves the ego state block untouched.

    Purpose: Validates only agent slots are overwritten, not the filtered ego estimate.

    Given: Particles with distinct non-zero ego blocks and an observation with an agent
    When: reinvigorate overwrites the agent block
    Then: The ego block of every particle is unchanged

    Test type: unit
    """
    np.random.seed(0)
    seeded = _empty_particles(4)
    for index, particle in enumerate(seeded):
        particle[:EGO_STATE_WIDTH] = float(index) + 1.0
    ego_before = np.asarray([p[:EGO_STATE_WIDTH].copy() for p in seeded])
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = _belief(4).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    ego_after = np.asarray(refreshed.particles)[:, :EGO_STATE_WIDTH]
    np.testing.assert_array_equal(ego_after, ego_before)


def test_reinvigorate_returns_same_type_for_persistent_reinjection():
    """Reinvigorate returns a CarlaAgentReinvigoration so re-injection repeats.

    Purpose: Validates the belief type persists across steps (not downgraded to base).

    Given: A CarlaAgentReinvigoration belief and any observation
    When: reinvigorate produces the next belief
    Then: The result is itself a CarlaAgentReinvigoration carrying the same config

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(4)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(4)))

    assert isinstance(refreshed, CarlaAgentReinvigoration)
    assert refreshed.max_tracked_agents == _MAX_AGENTS


def test_injected_agents_have_particle_diversity():
    """Per-particle jitter keeps re-injected agent poses diverse.

    Purpose: Validates the filter can still discriminate agent poses after re-injection.

    Given: A belief and an observation with one present agent, non-zero jitter
    When: reinvigorate overwrites the agent block
    Then: The re-injected slot poses are not all identical across particles

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(50)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(50)))

    rel_x = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH + 1]
    assert float(np.std(rel_x)) > 0.0


def _fusion_belief(n_particles: int) -> CarlaAgentReinvigoration:
    return CarlaAgentReinvigoration(
        particles=_empty_particles(n_particles),
        log_weights=np.log(np.ones(n_particles) / n_particles),
        max_tracked_agents=_MAX_AGENTS,
        agent_pose_jitter=0.3,
        sensor_fusion=True,
        obstacle_detection_range=30.0,
    )


def _lidar_obstacle(rel_x: float) -> np.ndarray:
    """A lidar cloud with one in-corridor return at ``rel_x`` metres ahead."""
    return np.array([[rel_x, 0.0, -1.0, 0.5]])


def test_sensor_fusion_off_ignores_lidar_obstacle():
    """With fusion disabled, a lidar obstacle is not injected.

    Purpose: Validates lidar/camera are consulted only when sensor_fusion is enabled.

    Given: A non-fusion belief, empty particles, and an observation with a 6 m lidar return
    When: reinvigorate runs
    Then: No agent slot is added (particles stay empty)

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([]),
        "lidar": _lidar_obstacle(6.0),
    }

    refreshed = _belief(4).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(4))
    )

    present = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH]
    np.testing.assert_array_equal(present.sum(axis=1), np.zeros(4))


def test_sensor_fusion_injects_lidar_obstacle_the_tracks_missed():
    """A lidar obstacle with no tracked vehicle becomes a forward-obstacle slot.

    Purpose: Validates the planner is handed hazards the agent slots miss.

    Given: A fusion belief, empty agent tracks, and a 6 m lidar return ahead
    When: reinvigorate runs
    Then: Every particle gains one present slot at ~6 m directly ahead

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([]),
        "lidar": _lidar_obstacle(6.0),
    }

    refreshed = _fusion_belief(4).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(4))
    )

    particles = np.asarray(refreshed.particles)
    np.testing.assert_array_equal(
        particles[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH].sum(axis=1), np.ones(4)
    )
    np.testing.assert_allclose(particles[:, EGO_STATE_WIDTH + 1], 6.0)


def test_sensor_fusion_does_not_duplicate_a_tracked_obstacle():
    """A lidar return already covered by a tracked vehicle adds no extra slot.

    Purpose: Validates fusion deduplicates against the observed traffic.

    Given: A fusion belief, an observed agent 6 m ahead, and a coincident 6 m lidar return
    When: reinvigorate runs
    Then: Exactly one slot is present (no phantom duplicate)

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([[1.0, 6.0, 0.0, 0.0, 3.0]]),
        "lidar": _lidar_obstacle(6.0),
    }

    refreshed = _fusion_belief(6).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(6))
    )

    present = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH]
    np.testing.assert_array_equal(present.sum(axis=1), np.ones(6))


def test_sensor_fusion_ignores_obstacle_beyond_detection_range():
    """A distant lidar return past the detection range is not injected.

    Purpose: Validates only near hazards are fused (no far phantom braking).

    Given: A fusion belief (30 m range) and a 40 m lidar return
    When: reinvigorate runs
    Then: No slot is injected

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([]),
        "lidar": _lidar_obstacle(40.0),
    }

    refreshed = _fusion_belief(4).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(4))
    )

    present = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH]
    np.testing.assert_array_equal(present.sum(axis=1), np.zeros(4))


def _traffic_light_belief(n_particles: int) -> CarlaAgentReinvigoration:
    return CarlaAgentReinvigoration(
        particles=_empty_particles(n_particles),
        log_weights=np.log(np.ones(n_particles) / n_particles),
        max_tracked_agents=_MAX_AGENTS,
        agent_pose_jitter=0.3,
        stop_for_traffic_lights=True,
        obstacle_detection_range=30.0,
    )


def test_red_light_injected_as_virtual_stop_obstacle():
    """A red light becomes a forward-obstacle slot at its stop-line distance.

    Purpose: Validates a red light is handled by the same virtual-obstacle machinery.

    Given: a light-stopping belief, empty tracks, and a red light 10 m ahead
    When: reinvigorate runs
    Then: every particle gains one present slot at ~10 m directly ahead

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([]),
        "traffic_light": np.array([1.0, 10.0]),
    }

    refreshed = _traffic_light_belief(4).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(4))
    )

    particles = np.asarray(refreshed.particles)
    np.testing.assert_array_equal(
        particles[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH].sum(axis=1), np.ones(4)
    )
    np.testing.assert_allclose(particles[:, EGO_STATE_WIDTH + 1], 10.0)


def test_green_light_injects_nothing():
    """A green (no-stop) light adds no obstacle.

    Purpose: Validates only a red/yellow signal injects a stop-obstacle.

    Given: a light-stopping belief, empty tracks, and a no-stop signal [0, 0]
    When: reinvigorate runs
    Then: no slot is injected

    Test type: unit
    """
    np.random.seed(0)
    observation = {
        "gnss": np.zeros(2),
        "agents": _agents_block([]),
        "traffic_light": np.array([0.0, 0.0]),
    }

    refreshed = _traffic_light_belief(4).reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(4))
    )

    present = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH::AGENT_SLOT_WIDTH]
    np.testing.assert_array_equal(present.sum(axis=1), np.zeros(4))


def test_update_persists_reinjection_through_full_filter_step():
    """A full belief.update re-injects traffic and stays a CarlaAgentReinvigoration.

    Purpose: Validates the reinvigoration hook fires through the whole PF update path.

    Given: A belief of empty-agent particles and a kinematic model as the transition/obs POMDP
    When: update is called with an observation carrying a present agent
    Then: The updated belief is a CarlaAgentReinvigoration whose particles carry that agent

    Test type: integration
    """
    np.random.seed(0)
    model = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05, max_tracked_agents=_MAX_AGENTS)
    belief = _belief(16)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    updated = belief.update(action=0, observation=observation, pomdp=model)

    assert isinstance(updated, CarlaAgentReinvigoration)
    assert np.all(np.asarray(updated.particles)[:, EGO_STATE_WIDTH] == 1.0)
