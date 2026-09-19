# SPDX-License-Identifier: MIT

"""Pinned QA inputs for the CaptureTheFlag POMDP.

Only the *inputs* live here. No completion rate, no baseline margin, no
verdict: a committed measurement goes stale silently and starts lying, while a
committed input makes the measurement reproducible. Recompute the numbers by
running the configurations below.

The reference environment is not solved by ``PFT_DPW`` at any budget tried
(see :data:`QA_BUDGET_SECONDS_PER_DECISION`), so the ladder here exists to
separate "the planner cannot do this" from "the task cannot be done": the
oracle policy in :func:`oracle_courier_action` reaches the goal on the easy
configuration, which is the goal-reachability evidence that planner failure
alone cannot supply.

Functions:
    capture_the_flag_qa_configs: The environment ladder, easiest first.
    capture_the_flag_planner_kwargs: The planner settings used against it.
    oracle_courier_action: The reachability policy, which reads the true state.
"""

from typing import Any, Dict, Tuple

import numpy as np

from POMDPPlanners.environments.capture_the_flag_pomdp import encode_joint_action
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_EAST,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_STAY,
    ACTION_WEST,
    manhattan,
)

# Wall-clock budget per decision. Pinning seconds rather than a simulation
# count is the repo's calibration convention: a fixed count is too weak on a
# hard environment, needlessly slow on an easy one, and silently becomes wrong
# as the environment changes.
QA_BUDGET_SECONDS_PER_DECISION = 1

QA_HORIZON = 80
QA_BELIEF_PARTICLES = 60


def capture_the_flag_qa_configs() -> Dict[str, Dict[str, Any]]:
    """Return the QA environment ladder, easiest first.

    Returns:
        Constructor kwargs per rung. ``easy`` removes the hidden-flag search
        and shrinks the joint action space to six; ``hidden`` restores the
        search on the same field; ``reference`` is the shipped default.
    """
    field = {
        "grid_size": (7, 5),
        "midline": 3,
        "trees": [(1, 1), (5, 3)],
        "n_blue": 1,
        "n_red": 1,
        "n_red_defenders": 1,
        "blue_base": (0, 2),
        "red_base": (6, 2),
        "blue_flag_cell": (1, 2),
        "score_to_win": 1,
    }
    return {
        "easy": {**field, "red_flag_candidates": [(5, 2)]},
        "hidden": {**field, "red_flag_candidates": [(5, 1), (5, 2), (4, 4)]},
        "reference": {},
    }


def capture_the_flag_planner_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Return the ``PFT_DPW`` settings QA was run with.

    Args:
        **overrides: Settings to replace.

    Returns:
        Planner kwargs, excluding ``environment`` and ``action_sampler``.
    """
    pinned: Dict[str, Any] = {
        "depth": 15,
        "k_a": 8.0,
        "alpha_a": 0.2,
        "k_o": 3.0,
        "alpha_o": 0.2,
        "exploration_constant": 80.0,
        "time_out_in_seconds": QA_BUDGET_SECONDS_PER_DECISION,
    }
    pinned.update(overrides)
    return pinned


def oracle_courier_action(env: Any, state: np.ndarray) -> int:
    """Return a greedy joint action for a courier that can see the true state.

    This is deliberately not a planner. It reads the hidden flag cell straight
    out of the state and walks at it, then walks home, ignoring the defender
    entirely. Its completion rate is therefore evidence that the goal is
    *reachable* -- which planner failure cannot establish on its own -- and a
    floor rather than a ceiling, since it never evades.

    Args:
        env: The environment being driven.
        state: The true state.

    Returns:
        A joint action moving blue player 0 toward its current objective and
        holding every other player.
    """
    layout = env.layout
    cell = layout.blue_cells(state)[0]
    carrying = int(state[layout.carrier_red_flag]) == 1
    target = env.blue_base if carrying else env.red_flag_cell(state)
    best_action, best_distance = ACTION_STAY, manhattan(cell, target)
    steps: Tuple[Tuple[int, Tuple[int, int]], ...] = (
        (ACTION_EAST, (1, 0)),
        (ACTION_WEST, (-1, 0)),
        (ACTION_NORTH, (0, 1)),
        (ACTION_SOUTH, (0, -1)),
    )
    for action, (step_x, step_y) in steps:
        candidate = (cell[0] + step_x, cell[1] + step_y)
        if env.is_free(candidate) and manhattan(candidate, target) < best_distance:
            best_action, best_distance = action, manhattan(candidate, target)
    return encode_joint_action([best_action] + [ACTION_STAY] * (env.n_blue - 1))
