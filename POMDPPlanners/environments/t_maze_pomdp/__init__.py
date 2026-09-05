# SPDX-License-Identifier: MIT

"""Maze POMDP family, including the legacy T-shaped compatibility class.

Classes:
    DiscreteMazePOMDP: Generated maze with cell actions.
    ContinuousMazePOMDP: The same generated maze with real displacement actions.
    TMazePOMDP: Compatibility class preserving the original T-shaped geometry.
"""

from POMDPPlanners.environments.t_maze_pomdp.maze_geometry import MazeGeometry
from POMDPPlanners.environments.t_maze_pomdp.maze_pomdp import (
    BaseMazePOMDP,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeMetric,
    MazeStepChannel,
    StepOutcome,
    create_maze_state,
)

from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import (
    ACTIONS,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    OBSERVATIONS,
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
    "BaseMazePOMDP",
    "CUE_CONSUMED",
    "CUE_EMITTING",
    "CUE_UNSEEN",
    "GOAL_LEFT",
    "GOAL_RIGHT",
    "ContinuousMazePOMDP",
    "DiscreteMazePOMDP",
    "MazeGeometry",
    "MazeMetric",
    "MazeStepChannel",
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
    "StepOutcome",
    "create_maze_state",
    "create_t_maze_state",
]
