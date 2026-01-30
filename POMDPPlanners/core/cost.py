"""Module for cost and reward calculation utilities.

This module provides utility functions for calculating expected costs and
rewards from belief states, particularly for weighted particle beliefs.

Functions:
    belief_expectation_cost: Calculate expected cost from weighted particle belief
    belief_expectation_reward: Calculate expected reward from weighted particle belief
"""

from typing import Any, Union

import numpy as np

from POMDPPlanners.core.belief import (
    WeightedParticleBelief,
    Belief,
    WeightedParticleBeliefStateUpdate,
    GaussianBelief,
    GaussianMixtureBelief,
)
from POMDPPlanners.core.environment import Environment


def particle_belief_entropy(
    belief: Union[WeightedParticleBelief, WeightedParticleBeliefStateUpdate],
) -> float:
    unique_belief = belief.to_unique_support_distribution()
    entropy = -float(np.sum(unique_belief.probs * np.log(unique_belief.probs)))
    return entropy


def belief_expectation_cost_particle_belief(
    belief: WeightedParticleBelief, action: Any, env: Environment
) -> float:
    """Calculate expected cost for an action given a weighted particle belief.

    This function computes the expected immediate cost (negative reward) by
    taking the weighted average over all particles in the belief state.

    Args:
        belief: Weighted particle belief representing state uncertainty
        action: Action to evaluate
        env: Environment providing reward function

    Returns:
        Expected immediate cost (negative of expected reward)
    """
    costs = np.array(
        [-env.reward(belief.particles[i], action) for i in range(len(belief.particles))]
    )
    cost_: float = float(np.sum(costs * belief.normalized_weights))

    return cost_


def particle_belief_expectation_cost_entropy_penalty(
    belief: WeightedParticleBelief,
    action: Any,
    env: Environment,
    entropy_weight: float = 0.0,
    lower_clip: float = -np.inf,
    upper_clip: float = np.inf,
) -> float:
    cost_ = belief_expectation_cost_particle_belief(belief=belief, action=action, env=env)
    if entropy_weight > 0.0:
        entropy_value = particle_belief_entropy(belief=belief)
        cost_ += entropy_weight * entropy_value

    return np.clip(cost_, lower_clip, upper_clip)


def particle_belief_expectation_cost_information_gain(
    belief: WeightedParticleBelief,
    action: Any,
    next_belief: WeightedParticleBelief,
    env: Environment,
    entropy_weight: float = 0.0,
    lower_clip: float = -np.inf,
    upper_clip: float = np.inf,
) -> float:
    cost_ = belief_expectation_cost_particle_belief(belief=belief, action=action, env=env)
    information_gain = particle_belief_entropy(belief=next_belief) - particle_belief_entropy(
        belief=belief
    )
    total_cost = cost_ + entropy_weight * information_gain
    return np.clip(total_cost, lower_clip, upper_clip)


def belief_expectation_reward_particle_belief(
    belief: WeightedParticleBelief, action: Any, env: Environment
) -> float:
    """Calculate expected reward for an action given a weighted particle belief.

    This function computes the expected immediate reward by taking the weighted
    average over all particles in the belief state.

    Args:
        belief: Weighted particle belief representing state uncertainty
        action: Action to evaluate
        env: Environment providing reward function

    Returns:
        Expected immediate reward
    """
    return -belief_expectation_cost_particle_belief(belief=belief, env=env, action=action)


def belief_expectation_cost_gaussian_belief(
    belief: GaussianBelief, action: Any, env: Environment, n_samples: int = 100
) -> float:
    samples = belief._mvn.sample(belief.mean, n_samples=n_samples)
    costs = np.array([-env.reward(samples[i], action) for i in range(n_samples)])
    return float(np.mean(costs))


def belief_expectation_cost_gaussian_mixture_belief(
    belief: GaussianMixtureBelief, action: Any, env: Environment, n_samples: int = 100
) -> float:
    samples = np.array([belief.sample() for _ in range(n_samples)])
    costs = np.array([-env.reward(samples[i], action) for i in range(n_samples)])
    return float(np.mean(costs))


def belief_expectation_cost(belief: Belief, action: Any, env: Environment) -> float:
    """Calculate expected cost for an action given a belief.

    This function computes the expected immediate cost (negative reward) by
    taking the weighted average over all particles in the belief state.

    Args:
        belief: Belief representing state uncertainty
        action: Action to evaluate
        env: Environment providing cost function

    Returns:
        Expected immediate cost
    """
    if isinstance(belief, WeightedParticleBelief):
        return belief_expectation_cost_particle_belief(belief=belief, action=action, env=env)
    elif isinstance(belief, GaussianBelief):
        return belief_expectation_cost_gaussian_belief(belief=belief, action=action, env=env)
    elif isinstance(belief, GaussianMixtureBelief):
        return belief_expectation_cost_gaussian_mixture_belief(
            belief=belief, action=action, env=env
        )
    else:
        raise NotImplementedError("Belief expectation cost is not implemented for this belief type")


def belief_expectation_reward(belief: Belief, action: Any, env: Environment) -> float:
    """Calculate expected reward for an action given a belief.

    This function computes the expected immediate reward by taking the weighted
    average over all particles in the belief state.

    Args:
        belief: Belief representing state uncertainty
        action: Action to evaluate
        env: Environment providing reward function

    Returns:
        Expected immediate reward
    """
    return -belief_expectation_cost(belief=belief, action=action, env=env)


def belief_expectation_cost_entropy_penalty(
    belief: Belief,
    action: Any,
    env: Environment,
    entropy_weight: float = 0.0,
    lower_clip: float = -np.inf,
    upper_clip: float = np.inf,
) -> float:
    if isinstance(belief, WeightedParticleBelief):
        return particle_belief_expectation_cost_entropy_penalty(
            belief=belief,
            action=action,
            env=env,
            entropy_weight=entropy_weight,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
    elif isinstance(belief, GaussianBelief):
        cost_ = belief_expectation_cost_gaussian_belief(belief=belief, action=action, env=env)
        if entropy_weight > 0.0:
            cost_ += entropy_weight * belief.entropy()
        return float(np.clip(cost_, lower_clip, upper_clip))
    elif isinstance(belief, GaussianMixtureBelief):
        cost_ = belief_expectation_cost_gaussian_mixture_belief(
            belief=belief, action=action, env=env
        )
        if entropy_weight > 0.0:
            cost_ += entropy_weight * belief.entropy()
        return float(np.clip(cost_, lower_clip, upper_clip))
    else:
        raise NotImplementedError(
            "Belief expectation cost entropy penalty is not implemented for this belief type"
        )


def belief_expectation_cost_belief_information_gain(
    belief: Belief,
    action: Any,
    next_belief: Belief,
    env: Environment,
    entropy_weight: float = 0.0,
    lower_clip: float = -np.inf,
    upper_clip: float = np.inf,
) -> float:
    if isinstance(belief, WeightedParticleBelief) and isinstance(
        next_belief, WeightedParticleBelief
    ):
        return particle_belief_expectation_cost_information_gain(
            belief=belief,
            action=action,
            next_belief=next_belief,
            env=env,
            entropy_weight=entropy_weight,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )
    elif isinstance(belief, GaussianBelief) and isinstance(next_belief, GaussianBelief):
        cost_ = belief_expectation_cost_gaussian_belief(belief=belief, action=action, env=env)
        information_gain = next_belief.entropy() - belief.entropy()
        total_cost = cost_ + entropy_weight * information_gain
        return float(np.clip(total_cost, lower_clip, upper_clip))
    elif isinstance(belief, GaussianMixtureBelief) and isinstance(
        next_belief, GaussianMixtureBelief
    ):
        cost_ = belief_expectation_cost_gaussian_mixture_belief(
            belief=belief, action=action, env=env
        )
        information_gain = next_belief.entropy() - belief.entropy()
        total_cost = cost_ + entropy_weight * information_gain
        return float(np.clip(total_cost, lower_clip, upper_clip))
    else:
        raise NotImplementedError(
            "Belief expectation cost information gain is not implemented for this belief type"
        )
