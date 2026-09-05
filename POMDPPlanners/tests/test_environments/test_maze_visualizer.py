# SPDX-License-Identifier: MIT

"""Renderer checks specific to the generated Maze variants."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.t_maze_pomdp.maze_pomdp import (
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_LEFT_CUE,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    create_maze_state,
)
from POMDPPlanners.environments.t_maze_pomdp.t_maze_visualizer import MazeVisualizer


def _belief(state):
    return WeightedParticleBelief(
        particles=[
            create_maze_state(state[:2], GOAL_LEFT, state[3]),
            create_maze_state(state[:2], GOAL_RIGHT, state[3]),
        ],
        log_weights=np.log(np.array([0.8, 0.2])),
    )


def _history(env, action):
    state = create_maze_state(env.start_cell, GOAL_LEFT)
    next_state = env.sample_next_state(state, action)
    return [
        StepData(
            state=state,
            action=action,
            next_state=next_state,
            observation=OBSERVATION_LEFT_CUE,
            reward=env.reward(state, action, next_state),
            belief=_belief(state),
        ),
        StepData(
            state=next_state,
            action=None,
            next_state=next_state,
            observation=None,
            reward=0.0,
            belief=_belief(next_state),
        ),
    ]


@pytest.mark.parametrize("env_type", [DiscreteMazePOMDP, ContinuousMazePOMDP])
def test_renderer_draws_every_walkable_cell_and_no_background_grid(env_type):
    env = env_type(maze_width=11, maze_height=13, maze_seed=7)
    visualizer = MazeVisualizer(env)
    with plt.style.context(["default"]):
        figure, map_axes, _, _ = visualizer._setup_figure()
        visualizer._draw_corridor(map_axes)
        # Exactly one filled Rectangle per walkable cell. A ">=" count passes even
        # when corridor cells are dropped, because _draw_cue_glow adds Circles.
        rectangles = [patch for patch in map_axes.patches if isinstance(patch, Rectangle)]
        assert len(rectangles) == len(env.walkable_cells)
        drawn = {
            (round(patch.get_x() + 0.5), round(patch.get_y() + 0.5)) for patch in rectangles
        }
        assert drawn == set(env.walkable_cells)
        assert not any(line.get_visible() for line in map_axes.get_xgridlines())
        assert "Maze" in map_axes.get_title()
        assert "T-Maze" not in map_axes.get_title()
        plt.close(figure)


def test_discrete_has_cell_guides_and_continuous_does_not():
    discrete = DiscreteMazePOMDP()
    continuous = ContinuousMazePOMDP()
    with plt.style.context(["default"]):
        d_figure, d_axes, _, _ = MazeVisualizer(discrete)._setup_figure()
        c_figure, c_axes, _, _ = MazeVisualizer(continuous)._setup_figure()
        MazeVisualizer(discrete)._draw_corridor(d_axes)
        MazeVisualizer(continuous)._draw_corridor(c_axes)
        discrete_edge = d_axes.patches[0].get_edgecolor()
        continuous_edge = c_axes.patches[0].get_edgecolor()
        assert not np.allclose(discrete_edge, d_axes.patches[0].get_facecolor())
        assert np.allclose(continuous_edge, c_axes.patches[0].get_facecolor())
        plt.close(d_figure)
        plt.close(c_figure)


def test_continuous_frame_keeps_actual_unrounded_position():
    env = ContinuousMazePOMDP()
    history = _history(env, np.array([0.0, 0.4]))
    visualizer = MazeVisualizer(env)
    states = [np.asarray(step.state) for step in history]
    with plt.style.context(["default"]):
        figure, map_axes, panel_axes, bar_axes = visualizer._setup_figure()
        artists = visualizer._create_animated_artists(map_axes, panel_axes, bar_axes)
        visualizer._draw_frame(
            1,
            states,
            [step.action for step in history],
            [None, history[0].observation],
            [step.belief for step in history],
            artists,
        )
        x_data, y_data = artists["agent"].get_data()
        assert x_data[0] == pytest.approx(env.start_cell[0])
        assert y_data[0] == pytest.approx(env.start_cell[1] + 0.4)
        assert "Last observation: left cue" in artists["readout"].get_text()
        plt.close(figure)
