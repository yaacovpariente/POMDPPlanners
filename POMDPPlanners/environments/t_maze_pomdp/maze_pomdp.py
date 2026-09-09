# SPDX-License-Identifier: MIT

"""Compatibility alias for :mod:`POMDPPlanners.environments.maze_pomdp.maze_pomdp`.

Kept because a configuration saved before the move records this module path and
:meth:`Environment.from_dict` imports it verbatim.
"""

# ``Cell`` and ``MazeGeometry`` were reachable here before the move, because the
# implementation imported them at module level.
from POMDPPlanners.environments.maze_pomdp.maze_geometry import (  # noqa: F401
    Cell,
    MazeGeometry,
)
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (  # noqa: F401
    ACTIONS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_OFFSETS,
    ACTION_RIGHT,
    ACTION_UP,
    BaseMazePOMDP,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    GOAL_LEFT,
    GOAL_RIGHT,
    MazeMetric,
    MazeStepChannel,
    OBSERVATIONS,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_WIDTH,
    STATE_X,
    STATE_Y,
    StepOutcome,
    create_maze_state,
)

__all__ = [
    "ACTIONS",
    "BaseMazePOMDP",
    "CUE_CONSUMED",
    "CUE_EMITTING",
    "CUE_UNSEEN",
    "ContinuousMazePOMDP",
    "DiscreteMazePOMDP",
    "GOAL_LEFT",
    "GOAL_RIGHT",
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
    "StepOutcome",
    "create_maze_state",
]
