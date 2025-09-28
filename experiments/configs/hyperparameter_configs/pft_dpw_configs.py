"""PFT_DPW hyperparameter optimization configurations.

This module contains HyperParameterRunParams configurations for the PFT_DPW planner
across different POMDP environments. Each configuration defines the hyperparameter
search space and optimization settings for the Progressive Function Transfer with
Double Progressive Widening algorithm.

PFT_DPW is designed for continuous action spaces but can work with discrete actions.
It requires ActionSampler implementations for progressive widening capabilities.
"""

import random
import numpy as np
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.core.simulation import NumericalHyperParameter
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
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


# ActionSampler implementations for different environments
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


class PushActionSampler(ActionSampler):
    """Action sampler for Push POMDP."""

    def sample(self, belief_node=None):
        # Push POMDP typically has directional actions
        return random.choice([0, 1, 2, 3])  # North, South, East, West


class SanityActionSampler(ActionSampler):
    """Action sampler for Sanity POMDP."""

    def sample(self, belief_node=None):
        return random.choice([0, 1])  # Simple binary actions


class LaserTagActionSampler(ActionSampler):
    """Action sampler for Laser Tag POMDP."""

    def sample(self, belief_node=None):
        # Laser tag typically has movement and shooting actions
        return random.choice([0, 1, 2, 3, 4, 5])  # Move directions + shoot actions


class SafetyAntActionSampler(ActionSampler):
    """Action sampler for Safety Ant Velocity POMDP."""

    def sample(self, belief_node=None):
        # Ant locomotion typically has multiple joint actions
        return random.choice([0, 1, 2, 3, 4, 5, 6, 7])  # Multiple movement actions


class LightDarkActionSampler(ActionSampler):
    """Action sampler for Light Dark POMDPs."""

    def sample(self, belief_node=None):
        return random.choice([0, 1, 2, 3, 4])  # Up, down, left, right, stay


# PFT_DPW hyperparameter ranges
PFT_DPW_HYPERPARAMETERS = [
    NumericalHyperParameter(low=5, high=25, name="depth"),  # Maximum search depth
    NumericalHyperParameter(
        low=1.0, high=5.0, name="k_a"
    ),  # Action progressive widening coefficient
    NumericalHyperParameter(
        low=0.2, high=1.0, name="alpha_a"
    ),  # Action progressive widening exponent
    NumericalHyperParameter(
        low=1.0, high=5.0, name="k_o"
    ),  # Observation progressive widening coefficient
    NumericalHyperParameter(
        low=0.2, high=1.0, name="alpha_o"
    ),  # Observation progressive widening exponent
    NumericalHyperParameter(
        low=0.5, high=3.0, name="exploration_constant"
    ),  # UCB1 exploration parameter
    NumericalHyperParameter(low=500, high=3000, name="n_simulations"),  # Number of MCTS simulations
    NumericalHyperParameter(
        low=5, high=50, name="min_samples_per_node"
    ),  # Minimum samples per node
    NumericalHyperParameter(
        low=1, high=10, name="min_visit_count_per_action"
    ),  # Minimum visits per action
]

# Note: Like POMCPOW, PFT_DPW requires ActionSampler instances which cannot be directly
# optimized with HyperParameterRunParams. The configurations below show the intended
# structure but require custom optimization implementation.


# Example configuration structure (requires custom optimization implementation):
class PFTDPWTigerConfigExample:
    """Example PFT_DPW configuration for Tiger POMDP (requires custom implementation)."""

    @staticmethod
    def create_config():
        """Create a PFT_DPW configuration for Tiger POMDP.

        Note: This requires custom optimization logic to handle ActionSampler construction.
        """
        environment = TigerPOMDP(discount_factor=0.95)
        action_sampler = TigerActionSampler()

        return {
            "environment": environment,
            "policy_cls": PFT_DPW,
            "action_sampler": action_sampler,
            "hyper_parameters": PFT_DPW_HYPERPARAMETERS,
            "num_episodes": 50,
            "num_steps": 100,
            "direction": HyperParameterOptimizationDirection.MAXIMIZE,
            "parameter_to_optimize": "average_return",
        }


class PFTDPWCartPoleConfigExample:
    """Example PFT_DPW configuration for CartPole POMDP (requires custom implementation)."""

    @staticmethod
    def create_config():
        """Create a PFT_DPW configuration for CartPole POMDP."""
        environment = CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        action_sampler = CartPoleActionSampler()

        return {
            "environment": environment,
            "policy_cls": PFT_DPW,
            "action_sampler": action_sampler,
            "hyper_parameters": PFT_DPW_HYPERPARAMETERS,
            "num_episodes": 30,
            "num_steps": 200,
            "direction": HyperParameterOptimizationDirection.MAXIMIZE,
            "parameter_to_optimize": "average_return",
        }


# Custom optimization function for PFT_DPW with ActionSampler support
def optimize_pft_dpw_custom(environment, action_sampler, n_trials=50):
    """Custom optimization function for PFT_DPW with ActionSampler support.

    This function demonstrates how to optimize PFT_DPW hyperparameters while
    handling the ActionSampler requirement.
    """
    import optuna
    from POMDPPlanners.core.belief import get_initial_belief

    def objective(trial):
        # Suggest hyperparameters
        depth = trial.suggest_int("depth", 5, 25)
        k_a = trial.suggest_float("k_a", 1.0, 5.0)
        alpha_a = trial.suggest_float("alpha_a", 0.2, 1.0)
        k_o = trial.suggest_float("k_o", 1.0, 5.0)
        alpha_o = trial.suggest_float("alpha_o", 0.2, 1.0)
        exploration_constant = trial.suggest_float("exploration_constant", 0.5, 3.0)
        n_simulations = trial.suggest_int("n_simulations", 500, 3000)
        min_samples_per_node = trial.suggest_int("min_samples_per_node", 5, 50)
        min_visit_count_per_action = trial.suggest_int("min_visit_count_per_action", 1, 10)

        # Create PFT_DPW instance with action sampler
        planner = PFT_DPW(
            environment=environment,
            discount_factor=environment.discount_factor,
            depth=depth,
            name=f"PFT_DPW_Trial_{trial.number}",
            action_sampler=action_sampler,
            k_a=k_a,
            alpha_a=alpha_a,
            k_o=k_o,
            alpha_o=alpha_o,
            exploration_constant=exploration_constant,
            n_simulations=n_simulations,
            min_samples_per_node=min_samples_per_node,
            min_visit_count_per_action=min_visit_count_per_action,
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


# Example usage for different environments
def create_pft_dpw_optimization_configs():
    """Create PFT_DPW optimization configurations for all environments.

    Returns a dictionary mapping environment names to their optimization configs.
    """
    configs = {}

    # Tiger POMDP
    configs["TigerPOMDP"] = {
        "environment": TigerPOMDP(discount_factor=0.95),
        "action_sampler": TigerActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # CartPole POMDP
    configs["CartPolePOMDP"] = {
        "environment": CartPolePOMDP(discount_factor=0.95, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])),
        "action_sampler": CartPoleActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Mountain Car POMDP
    configs["MountainCarPOMDP"] = {
        "environment": MountainCarPOMDP(discount_factor=0.95),
        "action_sampler": MountainCarActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Push POMDP
    configs["PushPOMDP"] = {
        "environment": PushPOMDP(discount_factor=0.95),
        "action_sampler": PushActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Sanity POMDP
    configs["SanityPOMDP"] = {
        "environment": SanityPOMDP(discount_factor=0.95),
        "action_sampler": SanityActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Laser Tag POMDP
    configs["LaserTagPOMDP"] = {
        "environment": LaserTagPOMDP(discount_factor=0.95),
        "action_sampler": LaserTagActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Safety Ant Velocity POMDP
    configs["SafeAntVelocityPOMDP"] = {
        "environment": SafeAntVelocityPOMDP(discount_factor=0.95),
        "action_sampler": SafetyAntActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Discrete Light Dark POMDP
    configs["DiscreteLightDarkPOMDP"] = {
        "environment": DiscreteLightDarkPOMDP(discount_factor=0.95),
        "action_sampler": LightDarkActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    # Continuous Light Dark POMDP with Discrete Actions
    configs["ContinuousLightDarkPOMDPDiscreteActions"] = {
        "environment": ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95),
        "action_sampler": LightDarkActionSampler(),
        "optimizer_function": optimize_pft_dpw_custom,
    }

    return configs


# Export information about PFT_DPW configurations
PFT_DPW_CONFIG_INFO = {
    "note": "PFT_DPW requires ActionSampler instances which cannot be directly optimized with HyperParameterRunParams",
    "solution": "Use custom optimization functions like optimize_pft_dpw_custom() that handle ActionSampler construction",
    "hyperparameters": [param.name for param in PFT_DPW_HYPERPARAMETERS],
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
        "PushPOMDP": "PushActionSampler",
        "SanityPOMDP": "SanityActionSampler",
        "LaserTagPOMDP": "LaserTagActionSampler",
        "SafeAntVelocityPOMDP": "SafetyAntActionSampler",
        "DiscreteLightDarkPOMDP": "LightDarkActionSampler",
        "ContinuousLightDarkPOMDPDiscreteActions": "LightDarkActionSampler",
    },
}

# For compatibility with the main configuration system, we'll create empty lists
# since PFT_DPW requires custom optimization approach
ALL_PFT_DPW_CONFIGS = []

# Users should use the custom optimization approach shown above instead of
# trying to use these configurations with the standard HyperParameterOptimizer
