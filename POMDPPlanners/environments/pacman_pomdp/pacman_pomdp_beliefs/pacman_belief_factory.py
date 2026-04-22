"""Belief factory for the PacMan POMDP.

This module provides a factory function that creates ready-to-use belief
objects for the PacMan POMDP, supporting both standard weighted particle
beliefs and vectorized particle beliefs.

Functions:
    create_pacman_belief: Factory producing a configured belief for PacManPOMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP

_SUPPORTED_TYPES = {BeliefType.PARTICLE, BeliefType.VECTORIZED_PARTICLE}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE


def create_pacman_belief(
    env: "PacManPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> "Belief":
    """Create a ready-to-use belief for the PacMan POMDP.

    Args:
        env: PacManPOMDP environment instance.
        belief_type: Desired belief representation.
            Defaults to ``BeliefType.VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.
        **kwargs: Extra arguments (reserved for future use).

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
        >>> env = PacManPOMDP(discount_factor=0.95)
        >>> belief = create_pacman_belief(env, n_particles=50)
        >>> belief.sample().shape[0] > 0
        True
    """
    del kwargs  # reserved for future use
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"PacManPOMDP does not support {belief_type}. " f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "PacManPOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_vectorized_belief(env: "PacManPOMDP", n_particles: int) -> "Belief":
    updater = PacManVectorizedUpdater.from_environment(env)
    initial_states = env.initial_state_dist().sample(n_samples=n_particles)
    particles = np.stack(initial_states)
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
