# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Belief factory for the Safety Ant Velocity POMDP.

Functions:
    create_safety_ant_velocity_belief: Factory producing a configured belief
        for SafeAntVelocityPOMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp_beliefs.safety_ant_velocity_vectorized_updater import (
    SafetyAntVelocityVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
        SafeAntVelocityPOMDP,
    )

_SUPPORTED_TYPES = {BeliefType.PARTICLE, BeliefType.VECTORIZED_PARTICLE}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_safety_ant_velocity_belief(
    env: "SafeAntVelocityPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
) -> "Belief":
    """Create a ready-to-use belief for the Safety Ant Velocity POMDP.

    Args:
        env: SafeAntVelocityPOMDP environment instance.
        belief_type: Desired belief representation.
            Defaults to ``BeliefType.VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"SafeAntVelocityPOMDP does not support {belief_type}. "
            f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "SafeAntVelocityPOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_vectorized_belief(env: "SafeAntVelocityPOMDP", n_particles: int) -> "Belief":
    updater = SafetyAntVelocityVectorizedUpdater.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
