# SPDX-License-Identifier: MIT

"""Unit tests for the perceived-agents nuPlan belief."""

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.environments.nuplan_pomdp.nuplan_belief import PerceivedAgentsBelief
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    LIGHT_SLOT_WIDTH,
)

_MAX_AGENTS = 1


def _belief(width: int, count: int = 4) -> PerceivedAgentsBelief:
    particles = [np.zeros(width) for _ in range(count)]
    return PerceivedAgentsBelief(
        particles=particles,
        log_weights=np.log(np.ones(count) / count),
        max_tracked_agents=_MAX_AGENTS,
        agent_pose_jitter=0.0,
    )


def test_reinvigorate_stamps_observed_agents_onto_particles() -> None:
    """Reinvigoration writes the observed agent block onto every particle.

    Purpose: Validates the belief acquires a mid-episode agent from the observation.

    Given: Particles with an empty agent slot and an observation with a present agent
    When: reinvigorate stamps the observation
    Then: every particle's agent slot becomes present at the observed pose

    Test type: unit
    """
    np.random.seed(0)
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH
    belief = _belief(width)
    particles = [np.zeros(width) for _ in range(4)]
    base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
    observation = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0])}
    refreshed = belief.reinvigorate("noop", observation, None, base)
    stamped = np.asarray(refreshed.particles)
    assert np.all(stamped[:, EGO_STATE_WIDTH] == 1.0)
    assert stamped[0, EGO_STATE_WIDTH + 1] == 8.0


def test_reinvigorate_preserves_trailing_light_slot() -> None:
    """Stamping leaves any trailing traffic-light slot untouched.

    Purpose: Validates the stamp is bounded to the agents block.

    Given: Particles carrying a trailing light slot with a marker value
    When: reinvigorate stamps the observed agents
    Then: the trailing light slot keeps its marker value

    Test type: unit
    """
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH
    belief = _belief(width)
    particles = [np.zeros(width) for _ in range(4)]
    for particle in particles:
        particle[-1] = 7.0  # marker in the trailing light slot
    base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
    observation = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0])}
    refreshed = belief.reinvigorate("noop", observation, None, base)
    assert np.all(np.asarray(refreshed.particles)[:, -1] == 7.0)


def test_reinvigorate_returns_same_belief_type() -> None:
    """Reinvigoration returns a PerceivedAgentsBelief so stamping repeats each step.

    Purpose: Validates the returned belief type persists across the episode.

    Given: A perceived-agents belief and a base particle belief
    When: reinvigorate is applied
    Then: the result is itself a PerceivedAgentsBelief

    Test type: unit
    """
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH
    belief = _belief(width)
    particles = [np.zeros(width) for _ in range(4)]
    base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
    observation = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.zeros(AGENT_SLOT_WIDTH)}
    refreshed = belief.reinvigorate("noop", observation, None, base)
    assert isinstance(refreshed, PerceivedAgentsBelief)
