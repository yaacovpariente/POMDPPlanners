from typing import Dict, Any, Type

from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDP, ContinuousLightDarkPOMDPDiscreteActions
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP

__all__ = [
    "PushPOMDP",
    "SafeAntVelocityPOMDP",
    "DiscreteLightDarkPOMDP",
    "ContinuousLightDarkPOMDP",
    "ContinuousLightDarkPOMDPDiscreteActions",
    "TigerPOMDP",
    "SanityPOMDP",
    "CartPolePOMDP",
    "MountainCarPOMDP"
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
}

def get_environment(env_type: str, **kwargs) -> Any:
    """
    Factory function to create environment instances.
    
    Args:
        env_type: Type of environment to create
        **kwargs: Additional arguments to pass to the environment constructor
        
    Returns:
        An instance of the requested environment
        
    Raises:
        ValueError: If the environment type is not supported
    """
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unsupported environment type: {env_type}. "
                       f"Available types: {list(ENVIRONMENT_REGISTRY.keys())}")
    
    return ENVIRONMENT_REGISTRY[env_type](**kwargs)