# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Belief factory for the Discrete Light-Dark POMDP.

Functions:
    create_discrete_light_dark_belief: Factory producing a configured belief
        for DiscreteLightDarkPOMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    ObservationModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs.discrete_light_dark_vectorized_updater import (
    DiscreteLightDarkDistanceBasedVectorizedUpdater,
    DiscreteLightDarkNoObsInDarkVectorizedUpdater,
    DiscreteLightDarkVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        DiscreteLightDarkPOMDP,
    )

_SUPPORTED_TYPES = {
    BeliefType.PARTICLE,
    BeliefType.VECTORIZED_PARTICLE,
}
_DEFAULT_TYPE = BeliefType.VECTORIZED_PARTICLE

_DISCRETE_UPDATER_MAP = {
    ObservationModelType.NORMAL: DiscreteLightDarkVectorizedUpdater,
    ObservationModelType.NO_OBS_IN_DARK: DiscreteLightDarkNoObsInDarkVectorizedUpdater,
    ObservationModelType.DISTANCE_BASED: DiscreteLightDarkDistanceBasedVectorizedUpdater,
}


def create_discrete_light_dark_belief(  # pylint: disable=unused-argument
    env: "DiscreteLightDarkPOMDP",
    belief_type: BeliefType = _DEFAULT_TYPE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a ready-to-use belief for the Discrete Light-Dark POMDP.

    Args:
        env: DiscreteLightDarkPOMDP environment instance.
        belief_type: Desired belief representation.
            Defaults to ``BeliefType.VECTORIZED_PARTICLE``.
        n_particles: Number of particles.  Defaults to 200.
        **kwargs: Reserved for future use; kept for interface compatibility.

    Returns:
        A configured :class:`Belief` object.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    if belief_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"DiscreteLightDarkPOMDP does not support {belief_type}. "
            f"Supported: {_SUPPORTED_TYPES}"
        )
    if belief_type == BeliefType.PARTICLE:
        return _create_particle_belief(env, n_particles)
    return _create_vectorized_belief(env, n_particles)


def _create_particle_belief(env: "DiscreteLightDarkPOMDP", n_particles: int) -> "Belief":
    return get_initial_belief(env, n_particles)


def _create_vectorized_belief(env: "DiscreteLightDarkPOMDP", n_particles: int) -> "Belief":
    updater_cls = _DISCRETE_UPDATER_MAP[env.observation_model_type]
    updater = updater_cls.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=n_particles), dtype=float)
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)
