"""StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configurations.

This module contains HyperParameterRunParams configurations for the StandardSparseSamplingDiscreteActionsPlanner
across different POMDP environments. Each configuration defines the hyperparameter
search space and optimization settings for the sparse sampling algorithm.

StandardSparseSamplingDiscreteActionsPlanner builds finite-depth lookahead trees
with limited branching and requires parameters for tree depth and branching factor.
"""

import numpy as np
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
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
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)


# StandardSparseSamplingDiscreteActionsPlanner hyperparameter ranges
SPARSE_SAMPLING_HYPERPARAMETERS = [
    NumericalHyperParameter(low=2, high=10, name="branching_factor"),
    NumericalHyperParameter(low=2, high=15, name="depth"),
]


# Tiger POMDP Configuration
class SparseSamplingTigerConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Tiger POMDP."""

    def __new__(cls):
        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=50,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# CartPole POMDP Configuration
class SparseSamplingCartPoleConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for CartPole POMDP."""

    def __new__(cls):
        env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Mountain Car POMDP Configuration
class SparseSamplingMountainCarConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Mountain Car POMDP."""

    def __new__(cls):
        env = MountainCarPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Push POMDP Configuration
class SparseSamplingPushConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Push POMDP."""

    def __new__(cls):
        env = PushPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=150,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Sanity POMDP Configuration
class SparseSamplingSanityConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Sanity POMDP."""

    def __new__(cls):
        env = SanityPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=20,
            num_steps=50,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Laser Tag POMDP Configuration
class SparseSamplingLaserTagConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Laser Tag POMDP."""

    def __new__(cls):
        env = LaserTagPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Safety Ant Velocity POMDP Configuration
class SparseSamplingSafetyAntVelocityConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Safety Ant Velocity POMDP."""

    def __new__(cls):
        env = SafeAntVelocityPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Discrete Light Dark POMDP Configuration
class SparseSamplingDiscreteLightDarkConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Discrete Light Dark POMDP."""

    def __new__(cls):
        env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Continuous Light Dark POMDP with Discrete Actions Configuration
class SparseSamplingContinuousLightDarkDiscreteActionsConfig(HyperParameterRunParams):
    """StandardSparseSamplingDiscreteActionsPlanner hyperparameter optimization configuration for Continuous Light Dark POMDP with Discrete Actions."""

    def __new__(cls):
        env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        return super().__new__(
            cls,
            environment=env,
            belief=belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], SPARSE_SAMPLING_HYPERPARAMETERS),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# All StandardSparseSamplingDiscreteActionsPlanner configurations for easy access
ALL_SPARSE_SAMPLING_CONFIGS = [
    SparseSamplingTigerConfig(),
    SparseSamplingCartPoleConfig(),
    SparseSamplingMountainCarConfig(),
    SparseSamplingPushConfig(),
    SparseSamplingSanityConfig(),
    SparseSamplingLaserTagConfig(),
    SparseSamplingSafetyAntVelocityConfig(),
    SparseSamplingDiscreteLightDarkConfig(),
    SparseSamplingContinuousLightDarkDiscreteActionsConfig(),
]
