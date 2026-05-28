# SPDX-License-Identifier: MIT

"""Pinned (snapshotted) constructor defaults for every instantiable env.

Each ``<env_slug>_pinned_kwargs`` function returns a fresh dict of all the
environment's *optional* (defaulted) config kwargs, with each value pinned to a
literal snapshot of the env's *current* default. ``overrides`` are merged on top
(overrides win).

Excluded from every dict:

* ``discount_factor`` — callers always pass it explicitly.
* any required (no-default) constructor argument (e.g. ``CartPolePOMDP``'s
  ``noise_cov``) — callers pass those too.
* the framework kwargs ``name``, ``output_dir``, ``debug``, ``use_queue_logger``
  — they never affect transitions / observations / rewards / shape.

For ``None``-sentinel kwargs that ``__init__`` substitutes with a concrete value
(e.g. continuous laser-tag ``walls`` / ``dangerous_areas``, push / pacman /
rock-sample obstacle / pellet / ghost defaults, cartpole / mountain-car
``state_transition_cov``), the concrete substituted value is pinned rather than
``None``. ``None`` is pinned only when it stays the meaningful default (e.g.
``initial_state``).

All mutable values are materialized as fresh objects on every call so two
constructions never share the same backing object, and the snapshot stays fixed
even if the env source default later changes.
"""

from typing import Any, Dict

import numpy as np

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    RewardModelType as LaserTagRewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ObservationModelType as ContinuousLightDarkObservationModelType,
    RewardModelType as ContinuousLightDarkRewardModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    ObservationModelType as DiscreteLightDarkObservationModelType,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    RewardModelType as PacManRewardModelType,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType as PushRewardModelType,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RewardModelType as RockSampleRewardModelType,
)


def tiger_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``TigerPOMDP`` (no non-framework optionals)."""
    pinned: Dict[str, Any] = {}
    pinned.update(overrides)
    return pinned


def sanity_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``SanityPOMDP`` (no non-framework optionals)."""
    pinned: Dict[str, Any] = {}
    pinned.update(overrides)
    return pinned


def laser_tag_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``LaserTagPOMDP``."""
    pinned: Dict[str, Any] = {
        "floor_shape": (11, 7),
        "walls": {
            (1, 2),
            (3, 0),
            (3, 4),
            (5, 0),
            (6, 4),
            (9, 1),
            (9, 4),
            (10, 6),
        },
        "tag_reward": 10.0,
        "tag_penalty": 10.0,
        "step_cost": 1.0,
        "measurement_noise": 1.0,
        "dangerous_areas": {(5, 3), (7, 1), (2, 5)},
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": 5.0,
        "initial_state": None,
        "transition_error_prob": 0.0,
        "reward_model_type": LaserTagRewardModelType.CONSTANT_HAZARD_PENALTY,
        "penalty_decay": 1.0,
        "opponent_policy": OpponentPolicy.EVADE,
    }
    pinned.update(overrides)
    return pinned


def _continuous_laser_tag_default_walls() -> list:
    half = 0.5
    cells = [
        (1, 2),
        (3, 0),
        (3, 4),
        (5, 0),
        (6, 4),
        (9, 1),
        (9, 4),
        (10, 6),
    ]
    return [(float(r), float(c), half, half) for r, c in cells]


def continuous_laser_tag_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousLaserTagPOMDP``."""
    pinned: Dict[str, Any] = {
        "grid_size": (11.0, 7.0),
        "walls": _continuous_laser_tag_default_walls(),
        "robot_radius": 0.3,
        "opponent_radius": 0.3,
        "tag_radius": 0.5,
        "tag_reward": 10.0,
        "tag_penalty": 10.0,
        "step_cost": 1.0,
        "measurement_noise": 1.0,
        "robot_transition_cov_matrix": np.eye(2) * 0.1,
        "opponent_transition_cov_matrix": np.eye(2) * 0.05,
        "evasion_speed": 0.6,
        "dangerous_areas": [(5.0, 3.0), (7.0, 1.0), (2.0, 5.0)],
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": 5.0,
        "dangerous_area_hit_probability": 1.0,
        "initial_state": None,
        "opponent_policy": OpponentPolicy.EVADE,
    }
    pinned.update(overrides)
    return pinned


def continuous_laser_tag_discrete_actions_pinned_kwargs(
    **overrides: Any,
) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousLaserTagPOMDPDiscreteActions``."""
    pinned = continuous_laser_tag_pinned_kwargs()
    pinned.update(overrides)
    return pinned


def cartpole_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``CartPolePOMDP`` (``noise_cov`` is required)."""
    pinned: Dict[str, Any] = {
        "state_transition_cov": np.diag([1e-4, 1e-4, 2.5e-5, 1e-4]),
    }
    pinned.update(overrides)
    return pinned


def mountain_car_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``MountainCarPOMDP``."""
    pinned: Dict[str, Any] = {
        "state_transition_cov": np.diag([2.5e-5, 1e-6]),
    }
    pinned.update(overrides)
    return pinned


def _light_dark_default_beacons() -> list:
    return [
        (0, 0),
        (0, 5),
        (0, 10),
        (5, 0),
        (5, 5),
        (5, 10),
        (10, 0),
        (10, 5),
        (10, 10),
    ]


def discrete_light_dark_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``DiscreteLightDarkPOMDP``."""
    pinned: Dict[str, Any] = {
        "transition_error_prob": 0.05,
        "observation_error_prob": 0.05,
        "beacons": _light_dark_default_beacons(),
        "goal_state": np.array([10, 5]),
        "start_state": np.array([0, 5]),
        "obstacles": [(3, 7), (5, 5)],
        "obstacle_hit_probability": 0.2,
        "obstacle_reward": -10.0,
        "goal_reward": 10.0,
        "beacon_radius": 1.0,
        "fuel_cost": 2.0,
        "grid_size": 11,
        "is_stochastic_reward": True,
        "observation_model_type": DiscreteLightDarkObservationModelType.NORMAL,
    }
    pinned.update(overrides)
    return pinned


def continuous_light_dark_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousLightDarkPOMDP``."""
    pinned: Dict[str, Any] = {
        "state_transition_cov_matrix": np.eye(2) * 0.05,
        "observation_cov_matrix": np.eye(2) * 0.05,
        "beacons": _light_dark_default_beacons(),
        "goal_state": np.array([10, 5]),
        "start_state": np.array([0, 5]),
        "obstacles": [(3, 7), (5, 5)],
        "obstacle_hit_probability": 0.2,
        "obstacle_reward": -10.0,
        "goal_reward": 10.0,
        "fuel_cost": 2.0,
        "grid_size": 11,
        "goal_state_radius": 1.5,
        "beacon_radius": 1.0,
        "obstacle_radius": 1.5,
        "reward_model_type": ContinuousLightDarkRewardModelType.CONSTANT_HAZARD_PENALTY,
        "observation_model_type": ContinuousLightDarkObservationModelType.NORMAL_NOISE,
        "penalty_decay": 1.0,
        "is_obstacle_hit_terminal": True,
    }
    pinned.update(overrides)
    return pinned


def continuous_light_dark_discrete_actions_pinned_kwargs(
    **overrides: Any,
) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousLightDarkPOMDPDiscreteActions``.

    Note: this variant's covariance defaults are ``np.eye(2)`` (unscaled),
    which differ from the continuous parent's ``np.eye(2) * 0.05``.
    """
    pinned: Dict[str, Any] = {
        "state_transition_cov_matrix": np.eye(2),
        "observation_cov_matrix": np.eye(2),
        "obstacle_hit_probability": 0.2,
        "obstacle_reward": -10.0,
        "goal_reward": 10.0,
        "fuel_cost": 2.0,
        "grid_size": 11,
        "goal_state_radius": 1.5,
        "beacon_radius": 1.0,
        "obstacle_radius": 1.5,
        "beacons": _light_dark_default_beacons(),
        "goal_state": np.array([10, 5]),
        "start_state": np.array([0, 5]),
        "obstacles": [(3, 7), (5, 5)],
        "reward_model_type": ContinuousLightDarkRewardModelType.CONSTANT_HAZARD_PENALTY,
        "observation_model_type": ContinuousLightDarkObservationModelType.NORMAL_NOISE,
        "penalty_decay": 1.0,
        "is_obstacle_hit_terminal": True,
    }
    pinned.update(overrides)
    return pinned


def push_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``PushPOMDP``."""
    pinned: Dict[str, Any] = {
        "grid_size": 10,
        "push_threshold": 1.0,
        "friction_coefficient": 0.3,
        "observation_noise": 0.1,
        "obstacles": [],
        "obstacle_radius": 0.5,
        "obstacle_penalty": -10.0,
        "obstacle_hit_probability": 1.0,
        "dangerous_areas": [],
        "dangerous_area_radius": 0.5,
        "dangerous_area_penalty": -10.0,
        "dangerous_area_hit_probability": 1.0,
        "reward_model_type": PushRewardModelType.CONSTANT_HAZARD_PENALTY,
        "penalty_decay": 1.0,
        "initial_state": None,
        "transition_error_prob": 0.0,
    }
    pinned.update(overrides)
    return pinned


def continuous_push_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousPushPOMDP``."""
    pinned: Dict[str, Any] = {
        "grid_size": 10,
        "push_threshold": 1.0,
        "friction_coefficient": 0.3,
        "max_push": 2.0,
        "observation_noise": 0.1,
        "obstacles": [],
        "obstacle_penalty": -10.0,
        "obstacle_hit_probability": 1.0,
        "dangerous_areas": [],
        "dangerous_area_radius": 0.5,
        "dangerous_area_penalty": -10.0,
        "dangerous_area_hit_probability": 1.0,
        "reward_model_type": PushRewardModelType.CONSTANT_HAZARD_PENALTY,
        "penalty_decay": 1.0,
        "robot_radius": 0.3,
        "state_transition_cov_matrix": np.eye(2) * 0.1,
        "initial_state": None,
    }
    pinned.update(overrides)
    return pinned


def continuous_push_discrete_actions_pinned_kwargs(
    **overrides: Any,
) -> Dict[str, Any]:
    """Pinned optional defaults for ``ContinuousPushPOMDPDiscreteActions``."""
    pinned = continuous_push_pinned_kwargs()
    pinned.update(overrides)
    return pinned


def pacman_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``PacManPOMDP``.

    The ghost / pellet / wall ``None`` sentinels are pinned to the concrete
    values ``__init__`` substitutes for the default ``maze_size=(7, 7)`` and
    ``num_ghosts=1`` (auto-generated ghost corner ``(6, 6)``, corner-adjacent
    pellets, and the predefined wall set).
    """
    pinned: Dict[str, Any] = {
        "maze_size": (7, 7),
        "walls": {(2, 2), (2, 3), (3, 2), (4, 4), (3, 5)},
        "initial_pellets": [(1, 1), (1, 5), (5, 1), (5, 5)],
        "initial_pacman_pos": (0, 0),
        "num_ghosts": 1,
        "initial_ghost_positions": [(6, 6)],
        "pellet_reward": 10.0,
        "ghost_collision_penalty": -100.0,
        "step_penalty": -1.0,
        "win_reward": 100.0,
        "ghost_aggressiveness": 2.0,
        "ghost_coordination": "independent",
        "ghost_strategies": ["aggressive"],
        "observation_noise_factor": 0.3,
        "max_observation_noise": 1.5,
        "dangerous_areas": None,
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": 5.0,
        "reward_model_type": PacManRewardModelType.CONSTANT_HAZARD_PENALTY,
        "penalty_decay": 1.0,
    }
    pinned.update(overrides)
    return pinned


def rock_sample_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``RockSamplePOMDP``."""
    pinned: Dict[str, Any] = {
        "map_size": (5, 5),
        "rock_positions": [(0, 0), (2, 2), (3, 3)],
        "init_pos": (0, 0),
        "sensor_efficiency": 10.0,
        "bad_rock_penalty": -10.0,
        "good_rock_reward": 10.0,
        "step_penalty": 0.0,
        "sensor_use_penalty": 0.0,
        "exit_reward": 10.0,
        "dangerous_areas": [],
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": -5.0,
        "dangerous_area_hit_probability": 1.0,
        "reward_model_type": RockSampleRewardModelType.CONSTANT_HAZARD_PENALTY,
        "penalty_decay": 1.0,
    }
    pinned.update(overrides)
    return pinned


def safety_ant_velocity_pinned_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Pinned optional defaults for ``SafeAntVelocityPOMDP``."""
    pinned: Dict[str, Any] = {
        "safe_velocity_threshold": 2.0,
        "max_force": 1.0,
        "dt": 0.1,
        "mass": 1.0,
        "damping": 0.1,
        "position_noise": 0.1,
        "velocity_noise": 0.2,
        "safety_violation_penalty": -100.0,
        "movement_reward_scale": 1.0,
    }
    pinned.update(overrides)
    return pinned
