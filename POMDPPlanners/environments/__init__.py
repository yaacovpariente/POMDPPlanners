"""POMDP Environment Implementations.

This package contains concrete implementations of various POMDP environments
used for testing and benchmarking planning algorithms. Each environment
implements the core Environment interface with specific state spaces,
action spaces, observation models, and reward functions.

Available Environments:
    TigerPOMDP: Classic tiger problem with discrete states and observations
    CartPolePOMDP: Pole balancing task with continuous states, discrete actions
    MountainCarPOMDP: Car climbing hill task with continuous state space
    PushPOMDP: Object manipulation task with spatial reasoning
    SafeAntVelocityPOMDP: Safety-constrained ant navigation
    SanityPOMDP: Simple test environment for debugging
    DiscreteLightDarkPOMDP: Grid-based light-dark navigation
    ContinuousLightDarkPOMDP: Continuous light-dark navigation problem
    LaserTagPOMDP: Pursuit-evasion problem with robot tagging opponent
    RockSamplePOMDP: Rock sampling problem with sensor-based rock quality evaluation

Factory Functions:
    get_environment: Create environment instances by name with parameters
"""

from typing import Dict, Any, Type

from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP

from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)

__all__ = [
    "PushPOMDP",
    "SafeAntVelocityPOMDP",
    "DiscreteLightDarkPOMDP",
    "ContinuousLightDarkPOMDP",
    "ContinuousLightDarkPOMDPDiscreteActions",
    "TigerPOMDP",
    "SanityPOMDP",
    "CartPolePOMDP",
    "MountainCarPOMDP",
    "LaserTagPOMDP",
    "RockSamplePOMDP",
    "PacManPOMDP",
]

# Registry of available environments
ENVIRONMENT_REGISTRY: Dict[str, Type] = {
    "CartPolePOMDP": CartPolePOMDP,
    "MountainCarPOMDP": MountainCarPOMDP,
    "TigerPOMDP": TigerPOMDP,
    "PushPOMDP": PushPOMDP,
    "SanityPOMDP": SanityPOMDP,
    "SafeAntVelocityPOMDP": SafeAntVelocityPOMDP,
    "DiscreteLightDarkPOMDP": DiscreteLightDarkPOMDP,
    "ContinuousLightDarkPOMDP": ContinuousLightDarkPOMDP,
    "ContinuousLightDarkPOMDPDiscreteActions": ContinuousLightDarkPOMDPDiscreteActions,
    "LaserTagPOMDP": LaserTagPOMDP,
    "RockSamplePOMDP": RockSamplePOMDP,
    "PacManPOMDP": PacManPOMDP,
}


def get_environment(env_type: str, **kwargs) -> Any:
    """Factory function to create environment instances by name.

    This function provides a convenient way to create environment instances
    using string identifiers, enabling configuration-driven environment creation.

    Args:
        env_type: Name of the environment type to create
        **kwargs: Additional arguments to pass to the environment constructor

    Returns:
        An instance of the requested environment

    Raises:
        ValueError: If the environment type is not registered

    Example:
        Creating different environments::

            # Create Tiger POMDP
            tiger = get_environment("TigerPOMDP", discount_factor=0.95)

            # Create Light-Dark POMDP with custom parameters
            lightdark = get_environment(
                "ContinuousLightDarkPOMDP",
                discount_factor=0.99,
                light_loc=[5.0, 0.0]
            )

            # Create CartPole POMDP
            cartpole = get_environment("CartPolePOMDP", discount_factor=0.95)
    """
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unsupported environment type: {env_type}. "
            f"Available types: {list(ENVIRONMENT_REGISTRY.keys())}"
        )

    return ENVIRONMENT_REGISTRY[env_type](**kwargs)
