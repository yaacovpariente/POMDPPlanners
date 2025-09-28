"""DiscreteActionSequencesPlanner hyperparameter optimization configurations.

This module contains HyperParameterRunParams configurations for the DiscreteActionSequencesPlanner
across different POMDP environments. Each configuration defines the hyperparameter
search space and optimization settings for the open-loop discrete action sequences algorithm.

DiscreteActionSequencesPlanner is an exhaustive open-loop planner that enumerates
all possible action sequences and requires parameters for planning depth and sampling.
"""

import numpy as np
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterRunParams,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterFeature
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
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)


# DiscreteActionSequencesPlanner hyperparameter ranges
DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS = [
    NumericalHyperParameter(low=2, high=6, name="depth"),
    NumericalHyperParameter(low=50, high=500, name="n_return_samples"),
]


# Tiger POMDP Configuration
class DiscreteActionSequencesTigerConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Tiger POMDP."""

    def __new__(cls):
        env = TigerPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=50,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# CartPole POMDP Configuration
class DiscreteActionSequencesCartPoleConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for CartPole POMDP."""

    def __new__(cls):
        env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Mountain Car POMDP Configuration
class DiscreteActionSequencesMountainCarConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Mountain Car POMDP."""

    def __new__(cls):
        env = MountainCarPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Push POMDP Configuration
class DiscreteActionSequencesPushConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Push POMDP."""

    def __new__(cls):
        env = PushPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=40,
            num_steps=150,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Sanity POMDP Configuration
class DiscreteActionSequencesSanityConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Sanity POMDP."""

    def __new__(cls, belief=None, constant_parameters=None, n_trials=None):
        env = SanityPOMDP(discount_factor=0.95)
        if belief is None:
            belief = get_initial_belief(env, n_particles=100)
        if constant_parameters is None:
            constant_parameters = {}
        if n_trials is None:
            n_trials = 50
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            num_episodes=20,
            num_steps=50,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            belief=belief,
            constant_parameters=constant_parameters,
            n_trials=n_trials,
        )


# Laser Tag POMDP Configuration
class DiscreteActionSequencesLaserTagConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Laser Tag POMDP."""

    def __new__(cls):
        env = LaserTagPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Safety Ant Velocity POMDP Configuration
class DiscreteActionSequencesSafetyAntVelocityConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Safety Ant Velocity POMDP."""

    def __new__(cls):
        env = SafeAntVelocityPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=30,
            num_steps=200,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Discrete Light Dark POMDP Configuration
class DiscreteActionSequencesDiscreteLightDarkConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Discrete Light Dark POMDP."""

    def __new__(cls):
        env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# Continuous Light Dark POMDP with Discrete Actions Configuration
class DiscreteActionSequencesContinuousLightDarkDiscreteActionsConfig(HyperParameterRunParams):
    """DiscreteActionSequencesPlanner hyperparameter optimization configuration for Continuous Light Dark POMDP with Discrete Actions."""

    def __new__(cls):
        env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
        return super().__new__(
            cls,
            environment=env,
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=cast(
                List[HyperParameterFeature], DISCRETE_ACTION_SEQUENCES_HYPERPARAMETERS
            ),
            belief=get_initial_belief(env, n_particles=100),
            constant_parameters={},
            num_episodes=40,
            num_steps=100,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=50,
        )


# All DiscreteActionSequencesPlanner configurations for easy access
ALL_DISCRETE_ACTION_SEQUENCES_CONFIGS = [
    DiscreteActionSequencesTigerConfig(),
    DiscreteActionSequencesCartPoleConfig(),
    DiscreteActionSequencesMountainCarConfig(),
    DiscreteActionSequencesPushConfig(),
    DiscreteActionSequencesSanityConfig(),
    DiscreteActionSequencesLaserTagConfig(),
    DiscreteActionSequencesSafetyAntVelocityConfig(),
    DiscreteActionSequencesDiscreteLightDarkConfig(),
    DiscreteActionSequencesContinuousLightDarkDiscreteActionsConfig(),
]
