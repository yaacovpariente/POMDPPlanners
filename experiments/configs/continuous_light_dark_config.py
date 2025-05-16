"""Continuous Light-Dark POMDP environment instance."""

import numpy as np
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
from POMDPPlanners.core.config_types import ExperimentConfig

# Environment instance
continuous_light_dark_env = ContinuousLightDarkPOMDP(
    discount_factor=0.99,
    name="ContinuousLightDarkPOMDP",
    state_transition_cov_matrix=np.eye(2),
    observation_cov_matrix=np.eye(2),
    beacons=np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], 
                     [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
    goal_state=np.array([10, 5]),
    start_state=np.array([0, 5]),
    obstacles=np.array([[3, 7], [5, 5]]),
    obstacle_hit_probability=0.2,
    obstacle_reward=-10.0,
    goal_reward=10.0,
    fuel_cost=2.0,
    grid_size=11,
    goal_state_radius=1.5,
    beacon_radius=1.0,
    obstacle_radius=1.5
)

# Belief instance
continuous_light_dark_belief = get_initial_belief(
    pomdp=continuous_light_dark_env,
    n_particles=20,  # Small number of particles for testing
    resampling=True
) 

# Policy instances
policies = [
    PFT_DPW(
        environment=continuous_light_dark_env,
        discount_factor=0.99,
        depth=3,
        name="PFT_DPW_LightDark",
        action_sampler=UnitCircleActionSampler(max_action_magnitude=1.0),
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=1000,
        min_samples_per_node=10,
        min_visit_count_per_action=1
    )
] 

# Experiment configuration
continuous_light_dark_experiment_config = ExperimentConfig(
    environment=continuous_light_dark_env,
    policies=policies,
    belief=continuous_light_dark_belief,
    num_episodes=100,
    num_steps=200
) 