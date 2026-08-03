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
    GOLDEN_METRIC_NAMES: env slug -> sorted ``get_metric_names()`` output.
"""

from typing import Dict, List

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
