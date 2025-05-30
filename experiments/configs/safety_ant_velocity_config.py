"""Safety Ant Velocity POMDP environment instance."""

import numpy as np
import random
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Environment instance
safety_ant_velocity_env = SafeAntVelocityPOMDP(
    discount_factor=0.99,
    safe_velocity_threshold=2.0,
    max_force=1.0,
    dt=0.1,
    mass=1.0,
    damping=0.1,
    position_noise=0.1,
    velocity_noise=0.2,
    safety_violation_penalty=-100.0,
    movement_reward_scale=1.0,
    name="SafeVelocityPOMDP"
)

# Belief instance
safety_ant_velocity_belief = get_initial_belief(
    pomdp=safety_ant_velocity_env,
    n_particles=20,  # Small number of particles for testing
    resampling=True
) 

policies = [
    SparsePFT(
        environment=safety_ant_velocity_env,
        discount_factor=0.99,
        gamma=0.99,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=8,
        n_simulations=1000,
        name="SparsePFT_SafetyAntVelocity"
    ),
    StandardSparseSamplingDiscreteActionsPlanner(
        environment=safety_ant_velocity_env,
        branching_factor=8,
        depth=3,
        name="StandardSparseSampling_SafetyAntVelocity"
    )
] 

# Experiment configuration
safety_ant_velocity_experiment_config = ExperimentConfig(
    environment=safety_ant_velocity_env,
    policies=policies,
    belief=safety_ant_velocity_belief,
    num_episodes=100,
    num_steps=200
)

