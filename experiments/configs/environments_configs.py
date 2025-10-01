"""Environment instances for all POMDP environments."""

from typing import Dict, List

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo

from experiments.configs.tiger_config import tiger_env, tiger_belief
from experiments.configs.cartpole_config import cartpole_env, cartpole_belief
from experiments.configs.mountain_car_config import mountain_car_env, mountain_car_belief
from experiments.configs.push_config import push_env, push_belief
from experiments.configs.safety_ant_velocity_config import (
    safety_ant_velocity_env,
    safety_ant_velocity_belief,
)
from experiments.configs.discrete_light_dark_config import (
    discrete_light_dark_env,
    discrete_light_dark_belief,
)
from experiments.configs.continuous_light_dark_config import (
    continuous_light_dark_env,
    continuous_light_dark_belief,
)

# Dictionary mapping environment names to their instances
environment_instances = {
    "tiger": tiger_env,
    "cartpole": cartpole_env,
    "mountain_car": mountain_car_env,
    "push": push_env,
    "safety_ant_velocity": safety_ant_velocity_env,
    "discrete_light_dark": discrete_light_dark_env,
    "continuous_light_dark": continuous_light_dark_env,
}

# Dictionary mapping environment names to their initial beliefs
belief_instances = {
    "tiger": tiger_belief,
    "cartpole": cartpole_belief,
    "mountain_car": mountain_car_belief,
    "push": push_belief,
    "safety_ant_velocity": safety_ant_velocity_belief,
    "discrete_light_dark": discrete_light_dark_belief,
    "continuous_light_dark": continuous_light_dark_belief,
}


class EnvironmentConfigsAPI:
    """API for querying environment configurations and compatibility."""

    @staticmethod
    def get_compatible_environments(policy_space_info: PolicySpaceInfo) -> List[str]:
        """Get list of environment names compatible with the given policy space info.

        Args:
            policy_space_info: Policy space information containing action and observation space types

        Returns:
            List of environment names that are compatible with the policy
        """
        compatible_envs = []

        for env_name, env_instance in environment_instances.items():
            if EnvironmentConfigsAPI._is_compatible(policy_space_info, env_instance.space_info):
                compatible_envs.append(env_name)

        return compatible_envs

    @staticmethod
    def _is_compatible(policy_space_info: PolicySpaceInfo, env_space_info) -> bool:
        """Check if policy and environment space types are compatible."""
        # Check action space compatibility
        if (
            policy_space_info.action_space == SpaceType.DISCRETE
            and env_space_info.action_space in [SpaceType.CONTINUOUS, SpaceType.MIXED]
        ):
            return False

        # Check observation space compatibility
        if (
            policy_space_info.observation_space == SpaceType.DISCRETE
            and env_space_info.observation_space in [SpaceType.CONTINUOUS, SpaceType.MIXED]
        ):
            return False

        return True
