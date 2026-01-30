"""Module-level helper functions for belief state operations.

This module provides utility functions for common belief operations such as
sampling the next belief, creating initial beliefs, and checking terminal
conditions across different belief types.

Functions:
    sample_next_belief: Simulate one step of belief evolution
    get_initial_belief: Create initial belief from environment's initial distribution
    is_terminal_particle_belief: Check if a particle belief is terminal
    is_terminal_belief: Check if any belief type is terminal
"""

from typing import Any, Tuple, Union

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_mixture_belief import GaussianMixtureBelief
from POMDPPlanners.core.belief.particle_beliefs import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.environment import Environment


def sample_next_belief(belief: Belief, action: Any, pomdp: "Environment") -> Tuple[Belief, Any]:
    """Simulate one step of belief evolution.

    This function samples a state from the current belief, simulates the
    environment dynamics, and updates the belief with the resulting observation.

    Args:
        belief: Current belief state
        action: Action to execute
        pomdp: Environment providing dynamics models

    Returns:
        Tuple containing:
            - Updated belief after incorporating the observation
            - Observation that was generated
    """
    state = belief.sample()
    next_state = pomdp.state_transition_model(state=state, action=action).sample()[0]
    observation = pomdp.observation_model(next_state=next_state, action=action).sample()[0]

    next_belief = belief.update(action=action, observation=observation, pomdp=pomdp)

    return next_belief, observation


def get_initial_belief(
    pomdp: Environment, n_particles: int, resampling: bool = True
) -> WeightedParticleBelief:
    """Create initial belief from environment's initial state distribution.

    Args:
        pomdp: Environment to get initial distribution from
        n_particles: Number of particles to generate for the belief
        resampling: Enable resampling in the created belief. Defaults to True.

    Returns:
        WeightedParticleBelief with uniform weights over initial states

    Raises:
        TypeError: If n_particles is not an integer
        ValueError: If n_particles is not positive
    """
    if not isinstance(n_particles, int):
        raise TypeError("n_particles must be an integer")
    if n_particles <= 0:
        raise ValueError("n_particles must be greater than 0")

    particles = pomdp.initial_state_dist().sample(n_samples=n_particles)
    log_weights = np.log(np.ones(n_particles) / n_particles)

    return WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=resampling
    )


def is_terminal_particle_belief(
    belief: Union[
        WeightedParticleBelief,
        WeightedParticleBeliefStateUpdate,
        UnweightedParticleBeliefStateUpdate,
    ],
    env: Environment,
) -> bool:
    """Check if the belief is terminal."""
    return all(env.is_terminal(particle) for particle in belief.particles)


def is_terminal_belief(belief: Belief, env: Environment) -> bool:
    """Check if the belief is terminal."""
    if isinstance(
        belief,
        (
            WeightedParticleBelief,
            WeightedParticleBeliefStateUpdate,
            UnweightedParticleBeliefStateUpdate,
        ),
    ):
        return is_terminal_particle_belief(belief=belief, env=env)
    elif isinstance(belief, VectorizedWeightedParticleBelief):
        return all(env.is_terminal(belief.particles[i]) for i in range(belief.n_particles))
    elif isinstance(belief, GaussianBelief):
        samples = [belief.sample() for _ in range(belief.n_terminal_check_samples)]
        return all(env.is_terminal(s) for s in samples)
    elif isinstance(belief, GaussianMixtureBelief):
        samples = [belief.sample() for _ in range(belief.n_terminal_check_samples)]
        return all(env.is_terminal(s) for s in samples)
    else:
        raise NotImplementedError("is_terminal_belief is not implemented for this belief type")
