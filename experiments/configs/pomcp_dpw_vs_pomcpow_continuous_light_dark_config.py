"""Configuration for comparing POMCP_DPW and POMCPOW on Continuous Light-Dark POMDP."""

import numpy as np
import random
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    RewardModelType,
)
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
from POMDPPlanners.core.config_types import ExperimentConfig

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

DEPTH = 5
NUM_EPISODES = 5
K_a = 2
K_o = 5
ALPHA_a = 0.1
ALPHA_o = 0.1
EXPLORATION_CONSTANT = DEPTH * 10
TIMEOUT = 6
OBSTACLE_RADIUS = 0.5

# Environment instance
continuous_light_dark_env = ContinuousLightDarkPOMDP(
    discount_factor=0.95,
    name="ContinuousLightDarkPOMDP_ContinuousActions",
    state_transition_cov_matrix=np.eye(2) * 0.05,
    observation_cov_matrix=np.array([[0.075, 0.01], [0.01, 0.075]]),
    beacons=[(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5)],
    goal_state=np.array([5, 5]),
    start_state=np.array([1, 1]),
    obstacles=[(5, 3), (3, 1), (5, 1)],
    obstacle_hit_probability=0.2,
    obstacle_reward=-10.0,
    goal_reward=10.0,
    fuel_cost=2.0,
    grid_size=6,
    goal_state_radius=1.5,
    beacon_radius=0.5,
    obstacle_radius=OBSTACLE_RADIUS,
    reward_model_type=RewardModelType.STANDARD,
    is_obstacle_hit_terminal=True,
)

# Belief instance
initial_belief = get_initial_belief(
    pomdp=continuous_light_dark_env, n_particles=100, resampling=True
)

# Action sampler
action_sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

# Policy instances
policies = [
    # POMCP_DPW(
    #     environment=continuous_light_dark_env,
    #     discount_factor=0.95,
    #     depth=DEPTH,
    #     exploration_constant=EXPLORATION_CONSTANT,
    #     k_a=K_a,
    #     alpha_a=ALPHA_a,
    #     k_o=K_o,
    #     alpha_o=ALPHA_o,
    #     name="POMCP_DPW_Conservative",
    #     action_sampler=action_sampler,
    #     n_simulations=10000,
    #     min_samples_per_node=5
    # ),
    POMCPOW(
        environment=continuous_light_dark_env,
        discount_factor=0.95,
        depth=DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        k_a=K_a,
        alpha_a=ALPHA_a,
        k_o=K_o,
        alpha_o=ALPHA_o,
        name="POMCPOW_Conservative",
        action_sampler=action_sampler,
        time_out_in_seconds=TIMEOUT,
        # n_simulations=10000,
        min_samples_per_node=1,
    )
]

# Experiment configuration
pomcp_dpw_vs_pomcpow_experiment_config = ExperimentConfig(
    environment=continuous_light_dark_env,
    policies=policies,
    belief=initial_belief,
    num_episodes=NUM_EPISODES,
    num_steps=30,
)
