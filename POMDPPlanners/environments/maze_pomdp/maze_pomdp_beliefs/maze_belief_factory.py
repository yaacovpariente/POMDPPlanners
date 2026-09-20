# SPDX-License-Identifier: MIT

"""Belief factory for the generated-maze POMDPs.

Maze observations are the labels ``"left_cue"``, ``"right_cue"`` and
``"empty"``, but :class:`VectorizedWeightedParticleBelief` hands the updater a
float array. :class:`MazeVectorizedWeightedParticleBelief` encodes the label
first, so the vectorized path takes the environment's own observations
unchanged.

Classes:
    MazeVectorizedWeightedParticleBelief: Vectorized belief with label encoding.

Functions:
    create_discrete_maze_belief: Belief for :class:`DiscreteMazePOMDP`.
    create_continuous_maze_belief: Belief for :class:`ContinuousMazePOMDP`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import OBSERVATIONS
from POMDPPlanners.environments.maze_pomdp.maze_pomdp_beliefs.maze_vectorized_updater import (
    ContinuousMazeVectorizedUpdater,
    DiscreteMazeVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
        VectorizedParticleBeliefUpdater,
    )
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
        BaseMazePOMDP,
        ContinuousMazePOMDP,
        DiscreteMazePOMDP,
    )

_OBS_ENCODING = {name: index for index, name in enumerate(OBSERVATIONS)}

# Any label the environment cannot emit is impossible under every particle, and
# encoding it to a sentinel keeps that an ``-inf`` likelihood rather than a
# KeyError deep inside a tree search.
_UNKNOWN_OBSERVATION = -1


class MazeVectorizedWeightedParticleBelief(VectorizedWeightedParticleBelief):
    """Vectorized weighted particle belief that speaks the maze's observation labels."""

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional["Environment"] = None,
        state: Optional[Any] = None,
    ) -> "MazeVectorizedWeightedParticleBelief":
        """Update the belief, encoding the observation label to its index.

        Args:
            action: The action that was executed.
            observation: The label the environment reported.
            pomdp: Unused; kept for interface compatibility.
            state: Ignored, so the true state cannot leak into the belief.

        Returns:
            The posterior belief.
        """
        del pomdp, state
        if isinstance(observation, str):
            observation = _OBS_ENCODING.get(observation, _UNKNOWN_OBSERVATION)
        encoded = np.asarray(observation, dtype=np.float64)

        next_particles = self.updater.batch_transition(self.particles, action)
        log_likelihoods = self.updater.batch_observation_log_likelihood(
            next_particles, action, encoded
        )
        next_log_weights = self.log_weights + log_likelihoods

        if self.resampling:
            next_particles, next_log_weights = self._resample(next_particles, next_log_weights)

        return MazeVectorizedWeightedParticleBelief(
            particles=next_particles,
            log_weights=next_log_weights,
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )


def create_discrete_maze_belief(
    env: "DiscreteMazePOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the discrete maze.

    Args:
        env: The environment.
        belief_type: ``PARTICLE`` or ``VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.
        **kwargs: Reserved for future use.

    Returns:
        A configured belief.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    del kwargs
    return _create(env, belief_type, n_particles, DiscreteMazeVectorizedUpdater)


def create_continuous_maze_belief(
    env: "ContinuousMazePOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the continuous maze.

    Args:
        env: The environment.
        belief_type: ``PARTICLE`` or ``VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.
        **kwargs: Reserved for future use.

    Returns:
        A configured belief.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    del kwargs
    return _create(env, belief_type, n_particles, ContinuousMazeVectorizedUpdater)


def _create(
    env: "BaseMazePOMDP",
    belief_type: BeliefType,
    n_particles: int,
    updater_class: Any,
) -> "Belief":
    if belief_type == BeliefType.PARTICLE:
        return get_initial_belief(env, n_particles)
    if belief_type != BeliefType.VECTORIZED_PARTICLE:
        raise ValueError(f"{type(env).__name__} does not support belief type {belief_type!r}")

    updater: "VectorizedParticleBeliefUpdater" = updater_class.from_environment(env)
    particles = np.stack(env.initial_state_dist().sample(n_samples=n_particles))
    log_weights = np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)
    return MazeVectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
        resampling=True,
    )


__all__ = [
    "MazeVectorizedWeightedParticleBelief",
    "create_continuous_maze_belief",
    "create_discrete_maze_belief",
]
