# SPDX-License-Identifier: MIT

"""Compatibility alias for :mod:`POMDPPlanners.environments.maze_pomdp.t_maze_pomdp`.

Kept because a configuration saved before the move records this module path and
:meth:`Environment.from_dict` imports it verbatim.
"""

from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import (  # noqa: F401
    ACTIONS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_OFFSETS,
    ACTION_RIGHT,
    ACTION_UP,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_ROW,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATIONS,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_WIDTH,
    STATE_X,
    STATE_Y,
    TMazeMetric,
    TMazePOMDP,
    TMazeStepChannel,
    create_t_maze_state,
)

__all__ = [
    "ACTIONS",
    "CUE_CONSUMED",
    "CUE_EMITTING",
    "CUE_UNSEEN",
    "GOAL_LEFT",
    "GOAL_RIGHT",
    "OBSERVATIONS",
    "OBSERVATION_EMPTY",
    "OBSERVATION_LEFT_CUE",
    "OBSERVATION_RIGHT_CUE",
    "STATE_CUE_PHASE",
    "STATE_GOAL",
    "STATE_WIDTH",
    "STATE_X",
    "STATE_Y",
    "TMazeMetric",
    "TMazePOMDP",
    "TMazeStepChannel",
    "create_t_maze_state",
]
