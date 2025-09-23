"""Mountain Car POMDP environment instance."""

import numpy as np
import random
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
mountain_car_env = MountainCarPOMDP(discount_factor=0.99, name="MountainCarPOMDP")

# Belief instance
mountain_car_belief = get_initial_belief(
    pomdp=mountain_car_env, n_particles=20, resampling=True  # Small number of particles for testing
)

policies = [
    SparsePFT(
        environment=mountain_car_env,
        discount_factor=0.99,
        gamma=0.99,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=8,
        n_simulations=1000,
        name="SparsePFT_MountainCar",
    ),
    StandardSparseSamplingDiscreteActionsPlanner(
        environment=mountain_car_env,
        branching_factor=8,
        depth=3,
        name="StandardSparseSampling_MountainCar",
    ),
]

# Experiment configuration
mountain_car_experiment_config = ExperimentConfig(
    environment=mountain_car_env,
    policies=policies,
    belief=mountain_car_belief,
    num_episodes=100,
    num_steps=200,
)
