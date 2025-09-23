"""Discrete Light-Dark POMDP environment instance."""

import numpy as np
import random
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
discrete_light_dark_env = DiscreteLightDarkPOMDP(
    discount_factor=0.99,
    name="DiscreteLightDarkPOMDP",
    transition_error_prob=0.05,
    observation_error_prob=0.05,
    beacons=np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
    goal_state=np.array([10, 5]),
    start_state=np.array([0, 5]),
    obstacles=np.array([[3, 7], [5, 5]]),
    obstacle_hit_probability=0.2,
    obstacle_reward=-10.0,
    goal_reward=10.0,
    beacon_radius=1.0,
    fuel_cost=2.0,
    grid_size=11,
    is_stochastic_reward=True,
)

# Belief instance
discrete_light_dark_belief = get_initial_belief(
    pomdp=discrete_light_dark_env,
    n_particles=20,  # Small number of particles for testing
    resampling=True,
)

# POMCP policy instances
pomcp_policies = [
    POMCP(
        environment=discrete_light_dark_env,
        discount_factor=0.99,
        depth=5,
        exploration_constant=1.0,
        name="POMCP_Depth5",
        n_simulations=1000,
    ),
    POMCP(
        environment=discrete_light_dark_env,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        name="POMCP_Depth10",
        n_simulations=1000,
    ),
]

# Experiment configuration
discrete_light_dark_experiment_config = ExperimentConfig(
    environment=discrete_light_dark_env,
    policies=pomcp_policies,
    belief=discrete_light_dark_belief,
    num_episodes=100,
    num_steps=200,
)
