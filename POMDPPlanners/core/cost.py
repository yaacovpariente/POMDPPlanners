from typing import Any

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment


def belief_expectation_cost(belief: WeightedParticleBelief, action: Any, env: Environment):
    costs = np.array(
        [-env.reward(belief.particles[i], action) for i in range(len(belief.particles))]
    )
    cost_ = np.sum(costs * belief.normalized_weights)

    return cost_

def belief_expectation_reward(belief: WeightedParticleBelief, action: Any, env: Environment):
    return -belief_expectation_cost(belief=belief, env=env, action=action)
