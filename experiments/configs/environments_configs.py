"""Environment instances for all POMDP environments."""

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
