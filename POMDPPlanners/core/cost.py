from typing import Any

import numpy as np

from POMDPPlanners.core.belief import ParticleBelief
from POMDPPlanners.core.environment import Environment


def belief_expectation_cost(belief: ParticleBelief, action: Any, env: Environment):
    costs = np.array(
        [-env.reward(belief.particles[i], action) for i in range(len(belief.particles))]
    )
    cost_ = np.sum(costs * belief.log_weights)

    return cost_
