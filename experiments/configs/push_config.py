"""Push POMDP environment instance."""

import numpy as np
import random
from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
push_env = PushPOMDP(
    discount_factor=0.99,
    name="PushPOMDP"
)

# Belief instance
push_belief = get_initial_belief(
    pomdp=push_env,
    n_particles=20,  # Small number of particles for testing
    resampling=True
) 

policies = [
    SparsePFT(
        environment=push_env,
        discount_factor=0.99,
        gamma=0.99,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=8,
        n_simulations=1000,
        name="SparsePFT_Push"
    ),
    StandardSparseSamplingDiscreteActionsPlanner(
        environment=push_env,
        branching_factor=8,
        depth=3,
        name="StandardSparseSampling_Push"
    )
] 

# Experiment configuration
push_experiment_config = ExperimentConfig(
    environment=push_env,
    policies=policies,
    belief=push_belief,
    num_episodes=100,
    num_steps=200
)

