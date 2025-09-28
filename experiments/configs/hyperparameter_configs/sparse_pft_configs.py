"""SparsePFT hyperparameter optimization configurations.

This module contains HyperParameterRunParams configurations for the SparsePFT planner
across different POMDP environments. Each configuration defines the hyperparameter
search space and optimization settings for the Sparse Progressive Function Transfer algorithm.

SparsePFT combines sparse sampling with progressive function transfer and Monte Carlo Tree Search,
using enhanced UCB exploration and controlled tree growth through belief child limits.
"""

import numpy as np
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterRunParams,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterFeature,
)
from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.core.belief import get_initial_belief
from typing import List, cast
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT


# SparsePFT hyperparameter ranges
SPARSE_PFT_HYPERPARAMETERS = [
    NumericalHyperParameter(low=0.8, high=1.0, name="gamma"),  # Discount for recursive calls
    NumericalHyperParameter(low=5, high=25, name="depth"),  # Maximum search depth
    NumericalHyperParameter(low=0.5, high=3.0, name="c_ucb"),  # Base exploration constant
    NumericalHyperParameter(low=1.0, high=5.0, name="beta_ucb"),  # Enhanced exploration parameter
    NumericalHyperParameter(
        low=3, high=10, name="belief_child_num"
    ),  # Max belief children per action
    NumericalHyperParameter(low=500, high=3000, name="n_simulations"),  # Number of MCTS simulations
]


# Tiger POMDP Configuration
class SparsePFTTigerConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Tiger POMDP."""

    def __new__(cls):
        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=50,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# CartPole POMDP Configuration
class SparsePFTCartPoleConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for CartPole POMDP."""

    def __new__(cls):
        env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Mountain Car POMDP Configuration
class SparsePFTMountainCarConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Mountain Car POMDP."""

    def __new__(cls):
        env = MountainCarPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Push POMDP Configuration
class SparsePFTPushConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Push POMDP."""

    def __new__(cls):
        env = PushPOMDP(discount_factor=0.95)  # Uses default parameters
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=150,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Sanity POMDP Configuration
class SparsePFTSanityConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Sanity POMDP."""

    def __new__(cls):
        env = SanityPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=20,
            num_steps=50,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Laser Tag POMDP Configuration
class SparsePFTLaserTagConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Laser Tag POMDP."""

    def __new__(cls):
        env = LaserTagPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Safety Ant Velocity POMDP Configuration
class SparsePFTSafetyAntVelocityConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Safety Ant Velocity POMDP."""

    def __new__(cls):
        env = SafeAntVelocityPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Discrete Light Dark POMDP Configuration
class SparsePFTDiscreteLightDarkConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Discrete Light Dark POMDP."""

    def __new__(cls):
        env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Continuous Light Dark POMDP with Discrete Actions Configuration
class SparsePFTContinuousLightDarkDiscreteActionsConfig(HyperParameterRunParams):
    """SparsePFT hyperparameter optimization configuration for Continuous Light Dark POMDP with Discrete Actions."""

    def __new__(cls):
        env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=SparsePFT,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_PFT_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# All SparsePFT configurations for easy access
ALL_SPARSE_PFT_CONFIGS = [
    SparsePFTTigerConfig(),
    SparsePFTCartPoleConfig(),
    SparsePFTMountainCarConfig(),
    SparsePFTPushConfig(),
    SparsePFTSanityConfig(),
    SparsePFTLaserTagConfig(),
    SparsePFTSafetyAntVelocityConfig(),
    SparsePFTDiscreteLightDarkConfig(),
    SparsePFTContinuousLightDarkDiscreteActionsConfig(),
]
