# SPDX-License-Identifier: MIT

"""Tests for the perceived-agents CARLA particle belief.

Covers :class:`PerceivedAgentsBelief` stamping the observation's ``agents`` block onto every
particle: perceived agents are written into the slots (with jitter), empty slots are cleared,
the ego block is preserved, and the returned belief is itself a :class:`PerceivedAgentsBelief`
so the stamping repeats each step. Perception itself lives in the world's observation model, so
these tests feed a perceived ``agents`` block directly. All tests run on hand-built states.
"""

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.carla_pomdp.carla_belief import PerceivedAgentsBelief
from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
)

_MAX_AGENTS = 2
_WIDTH = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH

# A concrete Environment to satisfy reinvigorate's typed ``pomdp`` argument; the reinvigoration
# step ignores it (the agent block comes from the observation alone).
_POMDP = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05, max_tracked_agents=_MAX_AGENTS)


def _empty_particles(n_particles: int) -> list:
    return [np.zeros(_WIDTH) for _ in range(n_particles)]


def _belief(n_particles: int, agent_pose_jitter: float = 0.3) -> PerceivedAgentsBelief:
    return PerceivedAgentsBelief(
        particles=_empty_particles(n_particles),
        log_weights=np.log(np.ones(n_particles) / n_particles),
        max_tracked_agents=_MAX_AGENTS,
        agent_pose_jitter=agent_pose_jitter,
    )


def _base_belief(particles: list) -> WeightedParticleBelief:
    weights = np.log(np.ones(len(particles)) / len(particles))
    return WeightedParticleBelief(particles=particles, log_weights=weights)


def _agents_block(rows: list) -> np.ndarray:
    """Flatten ``_MAX_AGENTS`` slot rows into a perceived ``agents`` observation block."""
    block = np.zeros((_MAX_AGENTS, AGENT_SLOT_WIDTH))
    for slot, row in enumerate(rows):
        block[slot] = row
    return block.reshape(-1)


def test_reinvigorate_marks_perceived_agent_present():
    """A perceived agent turns on the matching particle slot on every particle.

    Purpose: Validates stamping acquires traffic the particles were not seeded with.

    Given: A belief of empty-agent particles and an observation with one perceived agent
    When: reinvigorate stamps the observation's agent block onto the particles
    Then: Every particle's first slot is marked present (== 1.0)

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(6)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(6)))

    particles = np.asarray(refreshed.particles)
    assert np.all(particles[:, EGO_STATE_WIDTH] == 1.0)


def test_reinvigorate_places_agent_near_perceived_pose():
    """Stamped agents cluster around the perceived pose.

    Purpose: Validates the perceived pose (not zero) is written into the slot.

    Given: A belief and an observation with an agent 8 m ahead, 1 m to the left
    When: reinvigorate stamps the agent block over many particles
    Then: The mean stamped slot position is close to the perceived position

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(2000)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 1.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate(
        "noop", observation, _POMDP, _base_belief(_empty_particles(2000))
    )

    particles = np.asarray(refreshed.particles)
    mean_pose = particles[:, EGO_STATE_WIDTH + 1 : EGO_STATE_WIDTH + 3].mean(axis=0)
    np.testing.assert_allclose(mean_pose, [8.0, 1.0], atol=0.05)


def test_reinvigorate_clears_slots_when_nothing_perceived():
    """An observation with no perceived agents clears the stamped agent block.

    Purpose: Validates stamping overwrites (does not accumulate) stale slots.

    Given: Particles pre-seeded with a present slot and an observation with no agents
    When: reinvigorate stamps the empty perceived block
    Then: Every particle's agent slots are cleared to empty

    Test type: unit
    """
    np.random.seed(0)
    seeded = _empty_particles(5)
    for particle in seeded:
        particle[EGO_STATE_WIDTH] = 1.0
        particle[EGO_STATE_WIDTH + 1] = 4.0
    observation = {"gnss": np.zeros(2), "agents": _agents_block([])}

    refreshed = _belief(5).reinvigorate("noop", observation, _POMDP, _base_belief(seeded))

    particles = np.asarray(refreshed.particles)
    assert np.all(particles[:, EGO_STATE_WIDTH:] == 0.0)


def test_reinvigorate_preserves_ego_block():
    """Stamping leaves the ego state block untouched.

    Purpose: Validates only agent slots are overwritten, not the filtered ego estimate.

    Given: Particles with distinct non-zero ego blocks and an observation with an agent
    When: reinvigorate stamps the agent block
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


def test_reinvigorate_returns_same_type_for_persistent_stamping():
    """Reinvigorate returns a PerceivedAgentsBelief so stamping repeats across steps.

    Purpose: Validates the belief type persists (not downgraded to base).

    Given: A PerceivedAgentsBelief and any observation
    When: reinvigorate produces the next belief
    Then: The result is a PerceivedAgentsBelief carrying the same config

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(4)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(4)))

    assert isinstance(refreshed, PerceivedAgentsBelief)
    assert refreshed.max_tracked_agents == _MAX_AGENTS


def test_stamped_agents_have_particle_diversity():
    """Per-particle jitter keeps stamped agent poses diverse.

    Purpose: Validates the filter can still discriminate agent poses after stamping.

    Given: A belief and an observation with one perceived agent, non-zero jitter
    When: reinvigorate stamps the agent block
    Then: The stamped slot poses are not all identical across particles

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(50)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(50)))

    rel_x = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH + 1]
    assert float(np.std(rel_x)) > 0.0


def test_zero_jitter_stamps_identical_blocks():
    """With jitter disabled every particle receives the exact perceived block.

    Purpose: Validates the jitter guard so a study can stamp the perceived estimate verbatim.

    Given: A zero-jitter belief and an observation with one perceived agent
    When: reinvigorate stamps the agent block
    Then: Every particle's agent block is identical

    Test type: unit
    """
    np.random.seed(0)
    belief = _belief(8, agent_pose_jitter=0.0)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    refreshed = belief.reinvigorate("noop", observation, _POMDP, _base_belief(_empty_particles(8)))

    blocks = np.asarray(refreshed.particles)[:, EGO_STATE_WIDTH:]
    np.testing.assert_array_equal(blocks, np.broadcast_to(blocks[0], blocks.shape))


def test_update_persists_stamping_through_full_filter_step():
    """A full belief.update stamps the perceived agents and stays a PerceivedAgentsBelief.

    Purpose: Validates the reinvigoration hook fires through the whole PF update path.

    Given: A belief of empty-agent particles and a kinematic model as the transition/obs POMDP
    When: update is called with an observation carrying a perceived agent
    Then: The updated belief is a PerceivedAgentsBelief whose particles carry that agent

    Test type: integration
    """
    np.random.seed(0)
    model = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05, max_tracked_agents=_MAX_AGENTS)
    belief = _belief(16)
    observation = {"gnss": np.zeros(2), "agents": _agents_block([[1.0, 8.0, 0.0, 0.0, 5.0]])}

    updated = belief.update(action=0, observation=observation, pomdp=model)

    assert isinstance(updated, PerceivedAgentsBelief)
    assert np.all(np.asarray(updated.particles)[:, EGO_STATE_WIDTH] == 1.0)
