# SPDX-License-Identifier: MIT

"""Maze POMDP family, including the T-shaped compatibility class.

Classes:
    DiscreteMazePOMDP: Generated maze with cell actions.
    ContinuousMazePOMDP: The same generated maze with real displacement actions.
    TMazePOMDP: Compatibility class preserving the original T-shaped geometry.
    MazeVisualizer: Writes Maze episodes, discrete or continuous, as traces.
    TMazeVisualizer: Writes T-Maze episodes as traces.

The torch models VOPP plans on, ``DiscreteMazeVectorizedModel`` in
``maze_vectorized_model`` and ``TMazeVectorizedModel`` in
``t_maze_vectorized_model``, are not re-exported: they import torch, which
this package must not load on every ``import POMDPPlanners.environments``.
``ContinuousMazePOMDP`` has no torch model, because its actions are real
2-vectors and VOPP needs a finite action set.

The package was previously called ``t_maze_pomdp``. That name still imports, so
saved configurations keep loading, but new code should import from here.
"""

from POMDPPlanners.environments.maze_pomdp.maze_geometry import MazeGeometry
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    BaseMazePOMDP,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeMetric,
    MazeStepChannel,
    StepOutcome,
    create_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualization import (
    MazeVisualizer,
    TMazeVisualizer,
)
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import (
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
    "TMazeVisualizer",
    "create_maze_state",
    "create_t_maze_state",
]
