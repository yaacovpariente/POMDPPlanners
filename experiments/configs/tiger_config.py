"""Tiger POMDP environment instance."""

from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.config_types import ExperimentConfig
import numpy as np
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
tiger_env = TigerPOMDP(
    discount_factor=0.95,
    name="TigerPOMDP"
)

# Belief instance
tiger_belief = get_initial_belief(
    pomdp=tiger_env,
    n_particles=20,  # Small number of particles for testing
    resampling=True
) 

pomcp_policies = [
    POMCP(
        environment=tiger_env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="POMCP_Depth3_Tiger",
        n_simulations=1000
    ),
    POMCP(
        environment=tiger_env,
        discount_factor=0.95,
        depth=7,
        exploration_constant=1.0,
        name="POMCP_Depth7_Tiger",
        n_simulations=1000
    )
] 

# Experiment configuration
tiger_experiment_config = ExperimentConfig(
    environment=tiger_env,
    policies=pomcp_policies,
    belief=tiger_belief,
    num_episodes=100,
    num_steps=200
) 