# SPDX-License-Identifier: MIT

"""Frozen ``compute_metrics`` baseline for every instantiable environment.

Generated on branch ``feat/step-info-metrics-channel`` **before** the per-step
``StepData.info`` channel was introduced, so it records the pre-change behaviour
of every environment's metrics.

Its job is to make "the shared metrics plumbing changed, but no existing
environment moved" a provable statement rather than an assumption. ``VALUES``
catches a changed computation; ``NAMES`` catches a rename, which is the more
dangerous failure because it breaks saved MLflow runs and Optuna objective
configs silently instead of crashing.

Both are asserted against
:mod:`POMDPPlanners.tests.test_utils.golden_metric_snapshot`, which replays the
committed history fixture. Regenerate only when a metric change is intended, via
that module's ``generate_snapshot_histories_fixture``.

Note:
    The spread of names for a single concept — ``success_rate`` (tiger),
    ``tag_success_rate`` (laser tag), ``goal_reaching_rate`` (most), ``win_rate``
    (pacman), ``exit_success_rate`` (rock sample) — is recorded here as-is.
    Unifying them is deliberately out of scope: renaming would invalidate saved
    runs, so it needs its own migration.

Attributes:
    GOLDEN_METRIC_VALUES: env slug -> metric name -> frozen point estimate.
    GOLDEN_METRIC_VALUES_WITH_TERMINAL: the same, over the terminated-episode
        shape of the fixture (see ``append_terminal_step``).
    GOLDEN_METRIC_NAMES: env slug -> sorted ``get_metric_names()`` output.
    GOLDEN_METRIC_ORDER: env slug -> unsorted metric names, in emission order.
    GOLDEN_METRIC_BOUNDS: env slug -> metric name -> frozen (lower, upper)
        confidence bounds.
    GOLDEN_METRIC_BOUNDS_WITH_TERMINAL: the same, over the terminated-episode
        shape of the fixture.
"""

from typing import Dict, List, Tuple

GOLDEN_METRIC_VALUES: Dict[str, Dict[str, float]] = {
    "tiger": {"success_rate": 0.6666666666666666, "average_listens": 2.0},
    "cartpole": {"goal_reaching_rate": 1.0},
    "mountain_car": {"goal_reaching_rate": 0.0},
    "laser_tag": {
        "tag_success_rate": 0.0,
        "goal_reaching_rate": 0.0,
        "average_episode_length": 5.0,
        "average_failed_tag_attempts": 1.0,
        "average_obstacle_collisions": 0.0,
        "average_dangerous_area_steps": 1.3333333333333333,
        "average_all_dangerous_encounters": 1.3333333333333333,
    },
    "continuous_laser_tag": {
        "tag_success_rate": 0.0,
        "goal_reaching_rate": 0.0,
        "average_episode_length": 5.0,
        "average_failed_tag_attempts": 0.0,
        "average_wall_collisions": 0.0,
        "average_dangerous_area_steps": 0.0,
        "average_all_dangerous_encounters": 0.0,
    },
    "discrete_light_dark": {
        "goal_reaching_rate": 0.0,
        "obstacle_hit_rate": 0.0,
        "avg_obstacle_hit_counter": 0.0,
        "out_of_grid_rate": 0.3333333333333333,
        "avg_high_variance_states_counter": 0.6666666666666666,
    },
    "continuous_light_dark": {
        "goal_reaching_rate": 0.0,
        "obstacle_hit_rate": 0.0,
        "avg_obstacle_hit_counter": 0.0,
        "out_of_grid_rate": 0.6666666666666666,
        "avg_high_variance_states_counter": 0.6666666666666666,
    },
    "push": {
        "goal_reaching_rate": 0.0,
        "robot_obstacle_collision_rate": 0.0,
        "object_obstacle_collision_rate": 0.0,
        "total_obstacle_collision_rate": 0.0,
        "total_robot_obstacle_collisions": 0.0,
        "total_object_obstacle_collisions": 0.0,
        "total_all_obstacle_collisions": 0.0,
        "dangerous_area_rate": 0.0,
        "total_dangerous_area_steps": 0.0,
    },
    "continuous_push": {
        "goal_reaching_rate": 0.0,
        "robot_obstacle_collision_rate": 0.0,
        "object_obstacle_collision_rate": 0.0,
        "total_obstacle_collision_rate": 0.0,
        "total_robot_obstacle_collisions": 0.0,
        "total_object_obstacle_collisions": 0.0,
        "total_all_obstacle_collisions": 0.0,
        "dangerous_area_rate": 0.0,
        "total_dangerous_area_steps": 0.0,
    },
    "pacman": {
        "win_rate": 0.0,
        "avg_pellets_collected": 1.0,
        "avg_episode_length": 5.0,
        "avg_pacman_closest_ghost_distance": 10.333333333333334,
        "avg_collision_encounters": 0.0,
        "avg_dangerous_area_steps": 0.0,
        "avg_all_dangerous_encounters": 0.0,
    },
    "rock_sample": {
        "avg_rocks_sampled": 1.0,
        "exit_success_rate": 0.0,
        "average_dangerous_area_steps": 0.0,
    },
}

# Same environments and same frozen histories, but with the terminal bookkeeping
# step appended -- the shape a real episode has when it reaches a terminal state,
# and which the raw fixture structurally cannot represent (it stops after the last
# recorded transition). Captured pre-migration, like GOLDEN_METRIC_VALUES.
#
# Several entries deliberately differ from GOLDEN_METRIC_VALUES, and those
# differences are the point of this baseline. The terminal step contributes one
# more state to every whole-history scan (laser tag's average_episode_length goes
# 5.0 -> 6.0, its dangerous-area count 1.33 -> 1.67), and it carries
# ``action is None`` / ``reward is None``, which flips every metric that reads
# ``history.history[-1]``: tiger's success_rate goes 0.667 -> 0.0 because the last
# step is no longer a door-opening action. That is pre-existing behaviour, and
# pinning it here is what stops a migration from silently "fixing" it.
GOLDEN_METRIC_VALUES_WITH_TERMINAL: Dict[str, Dict[str, float]] = {
    "tiger": {"success_rate": 0.0, "average_listens": 2.0},
    "cartpole": {"goal_reaching_rate": 1.0},
    "mountain_car": {"goal_reaching_rate": 0.0},
    "laser_tag": {
        "tag_success_rate": 0.0,
        "goal_reaching_rate": 0.0,
        "average_episode_length": 6.0,
        "average_failed_tag_attempts": 1.0,
        "average_obstacle_collisions": 0.0,
        "average_dangerous_area_steps": 1.6666666666666667,
        "average_all_dangerous_encounters": 1.6666666666666667,
    },
    "continuous_laser_tag": {
        "tag_success_rate": 0.0,
        "goal_reaching_rate": 0.0,
        "average_episode_length": 6.0,
        "average_failed_tag_attempts": 0.0,
        "average_wall_collisions": 0.0,
        "average_dangerous_area_steps": 0.0,
        "average_all_dangerous_encounters": 0.0,
    },
    "discrete_light_dark": {
        "goal_reaching_rate": 0.0,
        "obstacle_hit_rate": 0.0,
        "avg_obstacle_hit_counter": 0.0,
        "out_of_grid_rate": 0.3333333333333333,
        "avg_high_variance_states_counter": 1.0,
    },
    "continuous_light_dark": {
        "goal_reaching_rate": 0.0,
        "obstacle_hit_rate": 0.0,
        "avg_obstacle_hit_counter": 0.0,
        "out_of_grid_rate": 0.6666666666666666,
        "avg_high_variance_states_counter": 0.6666666666666666,
    },
    "push": {
        "goal_reaching_rate": 0.0,
        "robot_obstacle_collision_rate": 0.0,
        "object_obstacle_collision_rate": 0.0,
        "total_obstacle_collision_rate": 0.0,
        "total_robot_obstacle_collisions": 0.0,
        "total_object_obstacle_collisions": 0.0,
        "total_all_obstacle_collisions": 0.0,
        "dangerous_area_rate": 0.0,
        "total_dangerous_area_steps": 0.0,
    },
    "continuous_push": {
        "goal_reaching_rate": 0.0,
        "robot_obstacle_collision_rate": 0.0,
        "object_obstacle_collision_rate": 0.0,
        "total_obstacle_collision_rate": 0.0,
        "total_robot_obstacle_collisions": 0.0,
        "total_object_obstacle_collisions": 0.0,
        "total_all_obstacle_collisions": 0.0,
        "dangerous_area_rate": 0.0,
        "total_dangerous_area_steps": 0.0,
    },
    "pacman": {
        "win_rate": 0.0,
        "avg_pellets_collected": 1.0,
        "avg_episode_length": 6.0,
        "avg_pacman_closest_ghost_distance": 10.055555555555555,
        "avg_collision_encounters": 0.0,
        "avg_dangerous_area_steps": 0.0,
        "avg_all_dangerous_encounters": 0.0,
    },
    "rock_sample": {
        "avg_rocks_sampled": 1.0,
        "exit_success_rate": 0.0,
        "average_dangerous_area_steps": 0.0,
    },
}

GOLDEN_METRIC_NAMES: Dict[str, List[str]] = {
    "tiger": ["average_listens", "success_rate"],
    "cartpole": ["goal_reaching_rate"],
    "mountain_car": ["goal_reaching_rate"],
    "laser_tag": [
        "average_all_dangerous_encounters",
        "average_dangerous_area_steps",
        "average_episode_length",
        "average_failed_tag_attempts",
        "average_obstacle_collisions",
        "goal_reaching_rate",
        "tag_success_rate",
    ],
    "continuous_laser_tag": [
        "average_all_dangerous_encounters",
        "average_dangerous_area_steps",
        "average_episode_length",
        "average_failed_tag_attempts",
        "average_wall_collisions",
        "goal_reaching_rate",
        "tag_success_rate",
    ],
    "discrete_light_dark": [
        "avg_high_variance_states_counter",
        "avg_obstacle_hit_counter",
        "goal_reaching_rate",
        "obstacle_hit_rate",
        "out_of_grid_rate",
    ],
    "continuous_light_dark": [
        "avg_high_variance_states_counter",
        "avg_obstacle_hit_counter",
        "goal_reaching_rate",
        "obstacle_hit_rate",
        "out_of_grid_rate",
    ],
    "push": [
        "dangerous_area_rate",
        "goal_reaching_rate",
        "object_obstacle_collision_rate",
        "robot_obstacle_collision_rate",
        "total_all_obstacle_collisions",
        "total_dangerous_area_steps",
        "total_object_obstacle_collisions",
        "total_obstacle_collision_rate",
        "total_robot_obstacle_collisions",
    ],
    "continuous_push": [
        "dangerous_area_rate",
        "goal_reaching_rate",
        "object_obstacle_collision_rate",
        "robot_obstacle_collision_rate",
        "total_all_obstacle_collisions",
        "total_dangerous_area_steps",
        "total_object_obstacle_collisions",
        "total_obstacle_collision_rate",
        "total_robot_obstacle_collisions",
    ],
    "pacman": [
        "avg_all_dangerous_encounters",
        "avg_collision_encounters",
        "avg_dangerous_area_steps",
        "avg_episode_length",
        "avg_pacman_closest_ghost_distance",
        "avg_pellets_collected",
        "win_rate",
    ],
    "rock_sample": ["average_dangerous_area_steps", "avg_rocks_sampled", "exit_success_rate"],
}

# Unsorted, unlike GOLDEN_METRIC_NAMES. Order is part of the contract:
# get_metric_names_from_environment_policy_pair feeds hyperparameter-tuning
# objective selection, and both of the assertions above are order-blind --
# GOLDEN_METRIC_VALUES is compared as a dict and GOLDEN_METRIC_NAMES is sorted.
#
# For every environment here, compute_metrics emits its metrics in exactly the
# order get_metric_names declares them, so one list pins both. A migration that
# concatenates spec-driven metrics ahead of bespoke ones would break that
# correspondence without changing either of the older baselines.
GOLDEN_METRIC_ORDER: Dict[str, List[str]] = {
    "tiger": ["success_rate", "average_listens"],
    "cartpole": ["goal_reaching_rate"],
    "mountain_car": ["goal_reaching_rate"],
    "laser_tag": [
        "tag_success_rate",
        "goal_reaching_rate",
        "average_episode_length",
        "average_failed_tag_attempts",
        "average_obstacle_collisions",
        "average_dangerous_area_steps",
        "average_all_dangerous_encounters",
    ],
    "continuous_laser_tag": [
        "tag_success_rate",
        "goal_reaching_rate",
        "average_episode_length",
        "average_failed_tag_attempts",
        "average_wall_collisions",
        "average_dangerous_area_steps",
        "average_all_dangerous_encounters",
    ],
    "discrete_light_dark": [
        "goal_reaching_rate",
        "obstacle_hit_rate",
        "avg_obstacle_hit_counter",
        "out_of_grid_rate",
        "avg_high_variance_states_counter",
    ],
    "continuous_light_dark": [
        "goal_reaching_rate",
        "obstacle_hit_rate",
        "avg_obstacle_hit_counter",
        "out_of_grid_rate",
        "avg_high_variance_states_counter",
    ],
    "push": [
        "goal_reaching_rate",
        "robot_obstacle_collision_rate",
        "object_obstacle_collision_rate",
        "total_obstacle_collision_rate",
        "total_robot_obstacle_collisions",
        "total_object_obstacle_collisions",
        "total_all_obstacle_collisions",
        "dangerous_area_rate",
        "total_dangerous_area_steps",
    ],
    "continuous_push": [
        "goal_reaching_rate",
        "robot_obstacle_collision_rate",
        "object_obstacle_collision_rate",
        "total_obstacle_collision_rate",
        "total_robot_obstacle_collisions",
        "total_object_obstacle_collisions",
        "total_all_obstacle_collisions",
        "dangerous_area_rate",
        "total_dangerous_area_steps",
    ],
    "pacman": [
        "win_rate",
        "avg_pellets_collected",
        "avg_episode_length",
        "avg_pacman_closest_ghost_distance",
        "avg_collision_encounters",
        "avg_dangerous_area_steps",
        "avg_all_dangerous_encounters",
    ],
    "rock_sample": ["avg_rocks_sampled", "exit_success_rate", "average_dangerous_area_steps"],
}

# Captured the same way as GOLDEN_METRIC_VALUES, by running the pre-migration
# compute_metrics over the same frozen fixture. A MetricValue is a point estimate
# *and* an interval, and the value assertions above see only half of it: a
# reduction that produced the right mean from the wrong per-episode samples would
# pass them and move the interval. These pin the other half.
GOLDEN_METRIC_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cartpole": {
        "goal_reaching_rate": (1.0, 1.0),
    },
    "continuous_laser_tag": {
        "average_all_dangerous_encounters": (0.0, 0.0),
        "average_dangerous_area_steps": (0.0, 0.0),
        "average_episode_length": (5.0, 5.0),
        "average_failed_tag_attempts": (0.0, 0.0),
        "average_wall_collisions": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "tag_success_rate": (0.0, 0.0),
    },
    "continuous_light_dark": {
        "avg_high_variance_states_counter": (-0.7675509098987142, 2.1008842432320476),
        "avg_obstacle_hit_counter": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "obstacle_hit_rate": (0.0, 0.0),
        "out_of_grid_rate": (-0.7675509098987142, 2.1008842432320476),
    },
    "continuous_push": {
        "dangerous_area_rate": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "object_obstacle_collision_rate": (0.0, 0.0),
        "robot_obstacle_collision_rate": (0.0, 0.0),
        "total_all_obstacle_collisions": (0.0, 0.0),
        "total_dangerous_area_steps": (0.0, 0.0),
        "total_object_obstacle_collisions": (0.0, 0.0),
        "total_obstacle_collision_rate": (0.0, 0.0),
        "total_robot_obstacle_collisions": (0.0, 0.0),
    },
    "discrete_light_dark": {
        "avg_high_variance_states_counter": (-2.201768486464095, 3.535101819797428),
        "avg_obstacle_hit_counter": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "obstacle_hit_rate": (0.0, 0.0),
        "out_of_grid_rate": (-1.1008842432320476, 1.767550909898714),
    },
    "laser_tag": {
        "average_all_dangerous_encounters": (-4.40353697292819, 7.070203639594856),
        "average_dangerous_area_steps": (-4.40353697292819, 7.070203639594856),
        "average_episode_length": (5.0, 5.0),
        "average_failed_tag_attempts": (1.0, 1.0),
        "average_obstacle_collisions": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "tag_success_rate": (0.0, 0.0),
    },
    "mountain_car": {
        "goal_reaching_rate": (0.0, 0.0),
    },
    "pacman": {
        "avg_all_dangerous_encounters": (0.0, 0.0),
        "avg_collision_encounters": (0.0, 0.0),
        "avg_dangerous_area_steps": (0.0, 0.0),
        "avg_episode_length": (5.0, 5.0),
        "avg_pacman_closest_ghost_distance": (9.574416726623387, 11.092249940043281),
        "avg_pellets_collected": (1.0, 1.0),
        "win_rate": (0.0, 0.0),
    },
    "push": {
        "dangerous_area_rate": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "object_obstacle_collision_rate": (0.0, 0.0),
        "robot_obstacle_collision_rate": (0.0, 0.0),
        "total_all_obstacle_collisions": (0.0, 0.0),
        "total_dangerous_area_steps": (0.0, 0.0),
        "total_object_obstacle_collisions": (0.0, 0.0),
        "total_obstacle_collision_rate": (0.0, 0.0),
        "total_robot_obstacle_collisions": (0.0, 0.0),
    },
    "rock_sample": {
        "average_dangerous_area_steps": (0.0, 0.0),
        "avg_rocks_sampled": (1.0, 1.0),
        "exit_success_rate": (0.0, 0.0),
    },
    "tiger": {
        "average_listens": (2.0, 2.0),
        "success_rate": (0.6666666666666666, 0.6666666666666666),
    },
}

GOLDEN_METRIC_BOUNDS_WITH_TERMINAL: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cartpole": {
        "goal_reaching_rate": (1.0, 1.0),
    },
    "continuous_laser_tag": {
        "average_all_dangerous_encounters": (0.0, 0.0),
        "average_dangerous_area_steps": (0.0, 0.0),
        "average_episode_length": (6.0, 6.0),
        "average_failed_tag_attempts": (0.0, 0.0),
        "average_wall_collisions": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "tag_success_rate": (0.0, 0.0),
    },
    "continuous_light_dark": {
        "avg_high_variance_states_counter": (-0.7675509098987142, 2.1008842432320476),
        "avg_obstacle_hit_counter": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "obstacle_hit_rate": (0.0, 0.0),
        "out_of_grid_rate": (-0.7675509098987142, 2.1008842432320476),
    },
    "continuous_push": {
        "dangerous_area_rate": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "object_obstacle_collision_rate": (0.0, 0.0),
        "robot_obstacle_collision_rate": (0.0, 0.0),
        "total_all_obstacle_collisions": (0.0, 0.0),
        "total_dangerous_area_steps": (0.0, 0.0),
        "total_object_obstacle_collisions": (0.0, 0.0),
        "total_obstacle_collision_rate": (0.0, 0.0),
        "total_robot_obstacle_collisions": (0.0, 0.0),
    },
    "discrete_light_dark": {
        "avg_high_variance_states_counter": (-3.302652729696142, 5.302652729696142),
        "avg_obstacle_hit_counter": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "obstacle_hit_rate": (0.0, 0.0),
        "out_of_grid_rate": (-1.1008842432320476, 1.767550909898714),
    },
    "laser_tag": {
        "average_all_dangerous_encounters": (-5.504421216160236, 8.83775454949357),
        "average_dangerous_area_steps": (-5.504421216160236, 8.83775454949357),
        "average_episode_length": (6.0, 6.0),
        "average_failed_tag_attempts": (1.0, 1.0),
        "average_obstacle_collisions": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "tag_success_rate": (0.0, 0.0),
    },
    "mountain_car": {
        "goal_reaching_rate": (0.0, 0.0),
    },
    "pacman": {
        "avg_all_dangerous_encounters": (0.0, 0.0),
        "avg_collision_encounters": (0.0, 0.0),
        "avg_dangerous_area_steps": (0.0, 0.0),
        "avg_episode_length": (6.0, 6.0),
        "avg_pacman_closest_ghost_distance": (8.601554733051053, 11.509556378060058),
        "avg_pellets_collected": (1.0, 1.0),
        "win_rate": (0.0, 0.0),
    },
    "push": {
        "dangerous_area_rate": (0.0, 0.0),
        "goal_reaching_rate": (0.0, 0.0),
        "object_obstacle_collision_rate": (0.0, 0.0),
        "robot_obstacle_collision_rate": (0.0, 0.0),
        "total_all_obstacle_collisions": (0.0, 0.0),
        "total_dangerous_area_steps": (0.0, 0.0),
        "total_object_obstacle_collisions": (0.0, 0.0),
        "total_obstacle_collision_rate": (0.0, 0.0),
        "total_robot_obstacle_collisions": (0.0, 0.0),
    },
    "rock_sample": {
        "average_dangerous_area_steps": (0.0, 0.0),
        "avg_rocks_sampled": (1.0, 1.0),
        "exit_success_rate": (0.0, 0.0),
    },
    "tiger": {
        "average_listens": (2.0, 2.0),
        "success_rate": (0.0, 0.0),
    },
}
