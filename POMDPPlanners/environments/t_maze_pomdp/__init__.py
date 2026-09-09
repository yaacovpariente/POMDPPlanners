# SPDX-License-Identifier: MIT

"""Compatibility alias for :mod:`POMDPPlanners.environments.maze_pomdp`.

The Maze family moved to the ``maze_pomdp`` package, because nothing here was
T-specific: ``DiscreteMazePOMDP``, ``ContinuousMazePOMDP`` and the shared renderer
all lived under the special case's name. This package and its submodules stay
importable because :meth:`Environment.from_dict` imports the module path recorded
in a saved configuration, so dropping them would break every Maze or T-Maze
configuration saved before the move.

New code should import from ``POMDPPlanners.environments.maze_pomdp``.
"""

# The submodules are imported for their side effect of binding themselves as
# attributes of this package. Before the move the package imported from its own
# submodules, so ``t_maze_pomdp.maze_pomdp`` resolved by attribute access and not
# only by ``import``; keeping that working costs five re-export modules.
from POMDPPlanners.environments.t_maze_pomdp import (  # noqa: F401
    maze_geometry,
    maze_pomdp,
    maze_visualizer,
    t_maze_pomdp,
    t_maze_visualizer,
)

from POMDPPlanners.environments.maze_pomdp import (  # noqa: F401
    ACTIONS,
    BaseMazePOMDP,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeGeometry,
    MazeMetric,
    MazeStepChannel,
    MazeVisualizer,
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
    TMazeMetric,
    TMazePOMDP,
    TMazeStepChannel,
    create_maze_state,
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
    "MazeVisualizer",
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
    "TMazeMetric",
    "TMazePOMDP",
    "TMazeStepChannel",
    "create_maze_state",
    "create_t_maze_state",
]
