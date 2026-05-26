# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Belief factory for RockSample POMDP.

This module provides a factory function for creating belief objects for the
RockSample environment, supporting both standard weighted particle beliefs
and vectorized particle beliefs.

Classes:
    RockSampleVectorizedWeightedParticleBelief: Thin subclass that handles
        string-to-integer observation encoding for RockSample.

Functions:
    create_rocksample_belief: Factory returning a configured Belief.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
        VectorizedParticleBeliefUpdater,
    )
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
        RockSamplePOMDP,
    )

_OBS_ENCODING = {"none": 0, "good": 1, "bad": 2}


class RockSampleVectorizedWeightedParticleBelief(VectorizedWeightedParticleBelief):
    """Vectorized weighted particle belief with string observation encoding.

    RockSample observations are strings (``"none"``, ``"good"``, ``"bad"``),
    but the parent class expects numeric observations.  This subclass
    transparently encodes string observations to integers before delegating
    to the vectorized updater.
    """

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional[Environment] = None,
        state: Optional[Any] = None,
    ) -> RockSampleVectorizedWeightedParticleBelief:
        """Update belief, encoding string observations to integers."""
        del pomdp, state  # unused — kept for interface compatibility
        if isinstance(observation, str):
            observation = _OBS_ENCODING[observation]

        encoded_obs = np.asarray(observation, dtype=float)

        next_particles = self.updater.batch_transition(self.particles, action)
        log_likelihoods = self.updater.batch_observation_log_likelihood(
            next_particles, action, encoded_obs
        )
        next_log_weights = self.log_weights + log_likelihoods

        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        return RockSampleVectorizedWeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )


def create_rocksample_belief(
    env: RockSamplePOMDP,
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> Belief:
    """Create a belief object for the RockSample POMDP.

    Args:
        env: RockSample environment instance.
        belief_type: Desired belief representation.  Supports
            ``PARTICLE`` and ``VECTORIZED_PARTICLE``.
        n_particles: Number of particles.  Defaults to 200.
        **kwargs: Reserved for future use.

    Returns:
        A configured belief object.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    del kwargs  # reserved for future use
    if belief_type == BeliefType.PARTICLE:
        return get_initial_belief(env, n_particles)
    if belief_type == BeliefType.VECTORIZED_PARTICLE:
        return _create_vectorized_belief(env, n_particles)
    raise ValueError(f"RockSamplePOMDP does not support belief type {belief_type!r}")


def _create_vectorized_belief(
    env: RockSamplePOMDP, n_particles: int
) -> RockSampleVectorizedWeightedParticleBelief:
    updater: VectorizedParticleBeliefUpdater = RockSampleVectorizedUpdater.from_environment(env)
    initial_states = env.initial_state_dist().sample(n_samples=n_particles)
    particles = np.stack(initial_states)
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return RockSampleVectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
        resampling=True,
    )
