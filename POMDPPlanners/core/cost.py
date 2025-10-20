"""Module for cost and reward calculation utilities.

This module provides utility functions for calculating expected costs and
rewards from belief states, particularly for weighted particle beliefs.

Functions:
    belief_expectation_cost: Calculate expected cost from weighted particle belief
    belief_expectation_reward: Calculate expected reward from weighted particle belief
"""

from typing import Any

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief, Belief
from POMDPPlanners.core.environment import Environment


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
