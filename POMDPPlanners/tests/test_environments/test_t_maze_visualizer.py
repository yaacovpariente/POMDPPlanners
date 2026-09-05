# SPDX-License-Identifier: MIT

"""Tests for the T-Maze visualizer's style isolation and its readout timing.

The golden-file test in ``test_environment_visualizations_golden_files`` pins the
*bytes* of one render, but it renders only under matplotlib's defaults. A real
simulation does not: ``utils.visualization.returns_plots`` calls
``sns.set_style("whitegrid")`` and ``sns.set_context("notebook", font_scale=1.2)``
globally, so by the time the per-episode GIFs are written the process is carrying
a seaborn theme. That is how the shipped planner GIFs came out with a grid ruled
across the maze while the golden GIF had none — the picture a human reviewed and
the picture a test checked were not the same picture.

So the checks here are about the *caller's* state, not about pixels:

* the same history renders to the same bytes whether or not the caller has set a
  seaborn theme, and
* the caller's theme is still installed after the render.
"""

import hashlib
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import (
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    TMazePOMDP,
    create_t_maze_state,
)
from POMDPPlanners.environments.t_maze_pomdp.t_maze_visualizer import TMazeVisualizer
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import t_maze_pinned_kwargs

# rcParams the seaborn theme moves and the visualizer must not inherit.
_WATCHED_RC_KEYS = ("axes.grid", "axes.facecolor", "font.size", "figure.facecolor")


def _episode(env: TMazePOMDP) -> List[StepData]:
    """A short scripted episode: up to the cue, on to the junction, into the left arm."""
    state = create_t_maze_state(env.start_cell, GOAL_LEFT)
    actions = ["up", "up", "up", "up", "left"]
    observations = [OBSERVATION_LEFT_CUE] + [OBSERVATION_EMPTY] * 4
    left_weights = [0.5, 0.9, 0.9, 0.9, 0.9]

    history: List[StepData] = []
    for action, observation, left_weight in zip(actions, observations, left_weights):
        next_state = env.sample_next_state(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=env.reward(state, action, next_state),
                belief=_belief_at(state, left_weight),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_belief_at(state, left_weights[-1]),
        )
    )
    return history


def _belief_at(state: np.ndarray, left_weight: float) -> WeightedParticleBelief:
    """Two particles on the agent's own cell, split ``left_weight`` over the goal side."""
    position = (int(state[0]), int(state[1]))
    cue_phase = float(state[3])
    return WeightedParticleBelief(
        particles=[
            create_t_maze_state(position, GOAL_LEFT, cue_phase),
            create_t_maze_state(position, GOAL_RIGHT, cue_phase),
        ],
        log_weights=np.log(np.array([left_weight, 1.0 - left_weight])),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(name="restore_rc")
def _restore_rc():
    """Put the process's rcParams back, so one test cannot theme the next one."""
    saved = matplotlib.rcParams.copy()
    yield
    matplotlib.rcParams.update(saved)


@pytest.fixture(name="env")
def _env() -> TMazePOMDP:
    return TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs())


class TestTMazeVisualizerStyleIsolation:
    """The render must not read, and must not write, the caller's matplotlib theme."""

    def test_seaborn_theme_does_not_change_the_output(self, env, tmp_path, restore_rc):
        """Test the GIF is identical with and without a caller-installed seaborn theme.

        Purpose: This is the real defect — a simulation plots returns first, which
            sets ``whitegrid`` globally, so the episode GIFs shipped with a grid
            over the maze that the golden file never had.

        Given: One episode history
        When: It is rendered under matplotlib defaults and again after
            ``sns.set_style("whitegrid")`` / ``sns.set_context("notebook")``
        Then: The two GIFs are byte-identical

        Test type: integration
        """
        history = _episode(env)
        visualizer = TMazeVisualizer(env)

        plt.style.use("default")
        default_gif = tmp_path / "default.gif"
        visualizer.create_visualization(history, default_gif)

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        themed_gif = tmp_path / "themed.gif"
        visualizer.create_visualization(history, themed_gif)

        assert _digest(default_gif) == _digest(themed_gif), (
            "The T-Maze GIF changed with the caller's matplotlib theme. The render "
            "must run inside its own style context so the artifact a human reviews "
            "is the artifact the golden file pins."
        )

    def test_caller_style_survives_the_render(self, env, tmp_path, restore_rc):
        """Test the visualizer restores the caller's rcParams when it is done.

        Purpose: Isolation has to run both ways — a visualizer that reset the
            process theme would silently strip the grid off every plot a run draws
            after it.

        Given: A caller that has installed the seaborn whitegrid / notebook theme
        When: One episode is rendered
        Then: The watched rcParams are exactly what they were before the render

        Test type: unit
        """
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        before = {key: matplotlib.rcParams[key] for key in _WATCHED_RC_KEYS}
        assert before["axes.grid"] is True, "Precondition: whitegrid turns the grid on."

        TMazeVisualizer(env).create_visualization(_episode(env), tmp_path / "themed.gif")

        after = {key: matplotlib.rcParams[key] for key in _WATCHED_RC_KEYS}
        assert after == before, (
            "The T-Maze render leaked its own styling back to the caller: "
            f"{before} became {after}."
        )


class TestTMazeVisualizerReadout:
    """The panel must report what the episode recorded, at the step it was recorded."""

    def test_observation_shown_is_the_one_received_on_arrival(self, env, restore_rc):
        """Test the readout lags the observation by one step, and never names a phase.

        Purpose: ``StepData.observation`` is what the world returned *after* the
            step's action, so a frame that printed its own step's observation would
            show the agent a reading it has not received yet — and the old readout
            printed the state's cue-delivery phase instead, which the agent never
            observes at all.

        Given: An episode whose first action moves onto the cue cell and returns
            ``left_cue``
        When: Frames 0, 1 and the terminal frame are drawn
        Then: Frame 0 shows no observation, frame 1 shows the cue read on arrival,
            and the terminal frame shows no action

        Test type: unit
        """
        history = _episode(env)
        visualizer = TMazeVisualizer(env)
        states = [np.asarray(step.state, dtype=np.float64) for step in history]
        actions = [step.action for step in history]
        beliefs = [step.belief for step in history]
        observations = [None] + [step.observation for step in history[:-1]]

        with plt.style.context(["default"]):
            figure, map_axes, panel_axes, bar_axes = visualizer._setup_figure()
            visualizer._draw_corridor(map_axes)
            artists = visualizer._create_animated_artists(map_axes, panel_axes, bar_axes)

            visualizer._draw_frame(0, states, actions, observations, beliefs, artists)
            first = artists["readout"].get_text()

            visualizer._draw_frame(1, states, actions, observations, beliefs, artists)
            second = artists["readout"].get_text()

            visualizer._draw_frame(
                len(states) - 1, states, actions, observations, beliefs, artists
            )
            terminal = artists["readout"].get_text()
            plt.close(figure)

        assert "Last observation: —" in first
        assert "Action: up" in first
        assert "Last observation: left cue" in second
        assert "Action: —" in terminal
        for text in (first, second, terminal):
            for phase_name in ("CUE_", "cue_phase", "emitting", "consumed", "unseen"):
                assert phase_name not in text, (
                    f"The readout printed the internal cue phase {phase_name!r}; the "
                    "agent observes a cue, not a delivery phase."
                )

    def test_belief_ring_is_drawn_over_the_agent(self, env, restore_rc):
        """Test a particle sharing the agent's cell still produces a visible marker.

        Purpose: Position is observable in this maze, so every particle sits on the
            agent. A filled belief marker disappeared under the agent's disc, which
            made the belief overlay useless exactly where it is always drawn.

        Given: A belief whose particles are all on the agent's cell
        When: A frame is drawn
        Then: The belief marker is on the agent's position, unfilled, and above the
            agent in draw order

        Test type: unit
        """
        history = _episode(env)
        visualizer = TMazeVisualizer(env)
        states = [np.asarray(step.state, dtype=np.float64) for step in history]

        with plt.style.context(["default"]):
            figure, map_axes, panel_axes, bar_axes = visualizer._setup_figure()
            artists = visualizer._create_animated_artists(map_axes, panel_axes, bar_axes)
            visualizer._draw_frame(
                1,
                states,
                [step.action for step in history],
                [None] + [step.observation for step in history[:-1]],
                [step.belief for step in history],
                artists,
            )
            offsets = np.asarray(artists["belief_ring"].get_offsets())
            face_colors = artists["belief_ring"].get_facecolors()
            ring_zorder = artists["belief_ring"].get_zorder()
            agent_zorder = artists["agent"].get_zorder()
            agent_x, agent_y = artists["agent"].get_data()
            plt.close(figure)

        assert offsets.shape[0] > 0, "The belief overlay drew nothing."
        assert np.allclose(offsets[:, 0], float(agent_x[0]))
        assert np.allclose(offsets[:, 1], float(agent_y[0]))
        assert face_colors.size == 0 or np.allclose(face_colors[:, 3], 0.0), (
            "The belief marker is filled, so it is hidden by the agent it sits on."
        )
        assert ring_zorder > agent_zorder


class TestTMazeVisualizerGeometry:
    """The drawn map must cover the walkable cells and nothing else."""

    @pytest.mark.parametrize(
        "stem_length,arm_length", [(2, 1), (4, 1), (9, 4)]
    )
    def test_outline_traces_only_the_walkable_region(self, stem_length, arm_length):
        """Test the heavy boundary is drawn on cell edges that face the exterior.

        Purpose: The map's readability rests on one closed outline around the
            corridor; an outline computed from a bounding box instead of from the
            cells would draw walls across the arms on any maze but the default one.

        Given: Mazes of three shapes
        When: The boundary segments are computed
        Then: Every segment separates a walkable cell from a non-walkable one, and
            there are as many segments as there are such edges

        Test type: unit
        """
        env = TMazePOMDP(
            discount_factor=0.95,
            **t_maze_pinned_kwargs(stem_length=stem_length, arm_length=arm_length),
        )
        cells = env.valid_cells
        expected = sum(
            1
            for cell in cells
            for step in ((0, 1), (0, -1), (-1, 0), (1, 0))
            if (cell[0] + step[0], cell[1] + step[1]) not in cells
        )

        segments = TMazeVisualizer(env)._boundary_segments()

        assert len(segments) == expected
        for x0, y0, x1, y1 in segments:
            midpoint_x, midpoint_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            # A segment lies on a cell edge, so stepping half a cell along its
            # normal lands on the cell centre either side of it.
            normal = (0.0, 0.5) if y0 == y1 else (0.5, 0.0)
            neighbours = [
                (round(midpoint_x + sign * normal[0]), round(midpoint_y + sign * normal[1]))
                for sign in (1, -1)
            ]
            inside = [cell in cells for cell in neighbours]
            assert any(inside) and not all(inside), (
                f"Segment {(x0, y0, x1, y1)} does not separate corridor from exterior."
            )
