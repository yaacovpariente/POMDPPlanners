"""CartPole POMDP environment instance."""
import random
import numpy as np
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
cartpole_env = CartPolePOMDP(
    discount_factor=0.99,
    noise_cov=np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]]),
    name="CartPolePOMDP",
)

# Belief instance
cartpole_belief = get_initial_belief(
    pomdp=cartpole_env, n_particles=20, resampling=True  # Small number of particles for testing
)

# Policy instances
policies = [
    SparsePFT(
        environment=cartpole_env,
        discount_factor=0.99,
        gamma=0.99,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=8,
        n_simulations=1000,
        name="SparsePFT_CartPole",
    ),
    StandardSparseSamplingDiscreteActionsPlanner(
        environment=cartpole_env,
        branching_factor=8,
        depth=3,
        name="StandardSparseSampling_CartPole",
    ),
]

# Experiment configuration
cartpole_experiment_config = ExperimentConfig(
    environment=cartpole_env,
    policies=policies,
    belief=cartpole_belief,
    num_episodes=100,
    num_steps=200,
)
