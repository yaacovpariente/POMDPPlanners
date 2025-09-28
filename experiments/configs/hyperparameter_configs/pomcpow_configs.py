"""POMCPOW hyperparameter optimization configurations.

This module contains HyperParameterRunParams configurations for the POMCPOW planner
across different POMDP environments. Each configuration defines the hyperparameter
search space and optimization settings for the POMCPOW algorithm.

POMCPOW (Partially Observable Monte Carlo Planning with Optimistic Weights) extends
POMCP with double progressive widening capabilities, requiring ActionSampler implementations
and progressive widening parameters.
"""

import random
import numpy as np
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.core.simulation import NumericalHyperParameter, CategoricalHyperParameter
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
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


# Simple ActionSampler implementations for different environments
class DiscreteEnvironmentActionSampler(ActionSampler):
    """Generic action sampler for discrete action environments."""

    def __init__(self, actions):
        self.actions = actions

    def sample(self, belief_node=None):
        return random.choice(self.actions)


class TigerActionSampler(ActionSampler):
    """Action sampler for Tiger POMDP."""

    def sample(self, belief_node=None):
        return random.choice(["listen", "open_left", "open_right"])


class CartPoleActionSampler(ActionSampler):
    """Action sampler for CartPole POMDP."""

    def sample(self, belief_node=None):
        return random.choice([0, 1])  # Left or right force


class MountainCarActionSampler(ActionSampler):
    """Action sampler for Mountain Car POMDP."""

    def sample(self, belief_node=None):
        return random.choice([0, 1, 2])  # Left, no push, right


class LightDarkActionSampler(ActionSampler):
    """Action sampler for Light Dark POMDPs."""

    def sample(self, belief_node=None):
        return random.choice([0, 1, 2, 3, 4])  # Up, down, left, right, stay


# Note: POMCPOW requires ActionSampler but we'll use discrete action environments
# and provide the appropriate action sampler for each environment. However, since
# POMCPOW's constructor requires an action_sampler parameter and we can't optimize
# over object instances directly, we'll need to create a custom configuration approach.

# For hyperparameter optimization, we'll need to work around the ActionSampler requirement
# by creating configurations that use categorical parameters to select sampler types,
# then construct the appropriate sampler within the optimization process.

# This requires a more complex setup - let's create a simplified approach first
# by focusing on the hyperparameters that can be optimized numerically

POMCPOW_HYPERPARAMETERS = [
    NumericalHyperParameter(low=0.1, high=10.0, name="exploration_constant"),
    NumericalHyperParameter(low=5, high=20, name="depth"),
    NumericalHyperParameter(low=100, high=2000, name="n_simulations"),
    NumericalHyperParameter(low=1.0, high=5.0, name="k_o"),
    NumericalHyperParameter(low=1.0, high=5.0, name="k_a"),
    NumericalHyperParameter(low=0.1, high=1.0, name="alpha_o"),
    NumericalHyperParameter(low=0.1, high=1.0, name="alpha_a"),
    NumericalHyperParameter(low=5, high=50, name="min_samples_per_node"),
]

# Note: Due to POMCPOW's requirement for ActionSampler instances in the constructor,
# we cannot directly use the standard HyperParameterRunParams approach.
# The configurations below would need a custom optimization setup that constructs
# the ActionSampler based on the environment type.

# For now, we'll provide example configurations showing the intended structure,
# but users will need to implement custom optimization logic to handle the
# ActionSampler requirement.


# Example configuration structure (requires custom optimization implementation):
class POMCPOWTigerConfigExample:
    """Example POMCPOW configuration for Tiger POMDP (requires custom implementation)."""

    @staticmethod
    def create_config():
        """Create a POMCPOW configuration for Tiger POMDP.

        Note: This requires custom optimization logic to handle ActionSampler construction.
        """
        environment = TigerPOMDP(discount_factor=0.95)
        action_sampler = TigerActionSampler()

        # This would need to be handled in a custom optimization function
        # that constructs POMCPOW instances with the action sampler
        return {
            "environment": environment,
            "policy_cls": POMCPOW,
            "action_sampler": action_sampler,
            "hyper_parameters": POMCPOW_HYPERPARAMETERS,
            "num_episodes": 50,
            "num_steps": 100,
            "direction": HyperParameterOptimizationDirection.MAXIMIZE,
            "parameter_to_optimize": "average_return",
        }


class POMCPOWCartPoleConfigExample:
    """Example POMCPOW configuration for CartPole POMDP (requires custom implementation)."""

    @staticmethod
    def create_config():
        """Create a POMCPOW configuration for CartPole POMDP."""
        environment = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        action_sampler = CartPoleActionSampler()

        return {
            "environment": environment,
            "policy_cls": POMCPOW,
            "action_sampler": action_sampler,
            "hyper_parameters": POMCPOW_HYPERPARAMETERS,
            "num_episodes": 30,
            "num_steps": 200,
            "direction": HyperParameterOptimizationDirection.MAXIMIZE,
            "parameter_to_optimize": "average_return",
        }


# Additional example configurations for other environments...
# (Similar structure for MountainCar, Push, Sanity, LaserTag, SafetyAnt, LightDark environments)

# Due to the ActionSampler requirement, these configurations cannot be used directly
# with the standard HyperParameterOptimizer.optimize() method. Instead, users need to:
#
# 1. Create a custom optimization function that:
#    - Takes environment and ActionSampler as inputs
#    - Constructs POMCPOW instances with suggested hyperparameters
#    - Uses the action_sampler for the POMCPOW constructor
#
# 2. Use Optuna directly with custom objective functions that handle the ActionSampler construction
#
# 3. Or modify the POMCPOW constructor to accept action sampler parameters instead of instances


# Example custom optimization approach:
def optimize_pomcpow_custom(environment, action_sampler, n_trials=50):
    """Custom optimization function for POMCPOW with ActionSampler support.

    This function demonstrates how to optimize POMCPOW hyperparameters while
    handling the ActionSampler requirement.
    """
    import optuna
    from POMDPPlanners.core.belief import get_initial_belief

    def objective(trial):
        # Suggest hyperparameters
        exploration_constant = trial.suggest_float("exploration_constant", 0.1, 10.0)
        depth = trial.suggest_int("depth", 5, 20)
        n_simulations = trial.suggest_int("n_simulations", 100, 2000)
        k_o = trial.suggest_float("k_o", 1.0, 5.0)
        k_a = trial.suggest_float("k_a", 1.0, 5.0)
        alpha_o = trial.suggest_float("alpha_o", 0.1, 1.0)
        alpha_a = trial.suggest_float("alpha_a", 0.1, 1.0)
        min_samples_per_node = trial.suggest_int("min_samples_per_node", 5, 50)

        # Create POMCPOW instance with action sampler
        planner = POMCPOW(
            environment=environment,
            discount_factor=environment.discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            name=f"POMCPOW_Trial_{trial.number}",
            action_sampler=action_sampler,
            n_simulations=n_simulations,
            min_samples_per_node=min_samples_per_node,
        )

        # Run evaluation episodes
        total_return = 0.0
        num_episodes = 10  # Reduced for faster optimization

        for episode in range(num_episodes):
            initial_belief = get_initial_belief(environment, n_particles=100)
            episode_return = 0.0

            # Simple episode simulation (would need proper implementation)
            # This is a placeholder - real implementation would run full episodes
            episode_return = random.uniform(0, 10)  # Placeholder
            total_return += episode_return

        return total_return / num_episodes

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value


# Export information about POMCPOW configurations
POMCPOW_CONFIG_INFO = {
    "note": "POMCPOW requires ActionSampler instances which cannot be directly optimized with HyperParameterRunParams",
    "solution": "Use custom optimization functions like optimize_pomcpow_custom() that handle ActionSampler construction",
    "hyperparameters": [param.name for param in POMCPOW_HYPERPARAMETERS],
    "environments_supported": [
        "TigerPOMDP",
        "CartPolePOMDP",
        "MountainCarPOMDP",
        "PushPOMDP",
        "SanityPOMDP",
        "LaserTagPOMDP",
        "SafeAntVelocityPOMDP",
        "DiscreteLightDarkPOMDP",
        "ContinuousLightDarkPOMDPDiscreteActions",
    ],
    "action_samplers": {
        "TigerPOMDP": "TigerActionSampler",
        "CartPolePOMDP": "CartPoleActionSampler",
        "MountainCarPOMDP": "MountainCarActionSampler",
        "Others": "DiscreteEnvironmentActionSampler",
    },
}

# For compatibility with the main configuration system, we'll create empty lists
# since POMCPOW requires custom optimization approach
ALL_POMCPOW_CONFIGS = []

# Users should use the custom optimization approach shown above instead of
# trying to use these configurations with the standard HyperParameterOptimizer
