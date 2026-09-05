# SPDX-License-Identifier: MIT

"""T-Maze POMDP visualization.

Renders one episode as an animated GIF: the T-shaped corridor, the agent's path,
the cue's delivery phase, and — the part a POMDP visualization exists for — the
belief.

Two belief views are drawn, because the T-Maze hides two different things:

* **Position particles.** Scattered over the corridor, sized by weight, following
  the convention ``laser_tag_visualizer`` and ``pacman_visualizer`` set. Position
  is in principle observable here, so this cloud is normally a single point; when
  it is not, the belief has gone wrong and that is worth seeing.
* **Goal-side posterior.** The hidden variable the task actually turns on. Each
  arm endpoint is shaded in proportion to the belief mass on that side, and the
  two numbers are printed. Watching this jump at the cue cell and then hold flat
  through the corridor is how a human checks that the observation model and the
  belief update do what the environment says they do.

Nothing here draws from an RNG and nothing is iterated out of order, so two
renders of one history are byte-identical — which is what the golden-file test
compares.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from POMDPPlanners.core.simulation import StepData  # noqa: E402
from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import (  # noqa: E402
    CUE_CONSUMED,
    CUE_EMITTING,
    GOAL_LEFT,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
    TMazePOMDP,
)

_CUE_PHASE_LABELS = {0.0: "unseen", CUE_EMITTING: "emitting", CUE_CONSUMED: "consumed"}


class TMazeVisualizer:
    """Renders a :class:`~POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp.TMazePOMDP` episode.

    Attributes:
        environment: The maze whose geometry is drawn.
    """

    def __init__(self, environment: TMazePOMDP) -> None:
        """Initialize the visualizer.

        Args:
            environment: The environment supplying the corridor geometry.
        """
        self.environment = environment

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Write an animated GIF of ``history``.

        Args:
            history: The episode's step records, in order.
            cache_path: Destination path; must end in ``.gif``.

        Raises:
            ValueError: If ``history`` is empty or ``cache_path`` is not a ``.gif``.
        """
        if not history:
            raise ValueError("Cannot visualize an empty episode history.")
        if cache_path.suffix != ".gif":
            raise ValueError(f"cache_path must end in .gif, got {cache_path}.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        states = [np.asarray(step.state, dtype=np.float64) for step in history]
        actions = [step.action for step in history]
        beliefs = [getattr(step, "belief", None) for step in history]

        figure, axes = self._setup_figure()
        self._draw_corridor(axes)
        artists = self._create_animated_artists(axes)

        def draw_frame(frame: int):
            return self._draw_frame(frame, states, actions, beliefs, artists)

        anim = animation.FuncAnimation(
            figure, draw_frame, frames=len(states), blit=False, repeat=False
        )
        anim.save(cache_path, writer="pillow", fps=1)
        plt.close(figure)

    # ── Static scene ────────────────────────────────────────────────────
    def _setup_figure(self):
        env = self.environment
        figure, axes = plt.subplots(figsize=(6.0, 6.0))
        axes.set_xlim(-env.arm_length - 1, env.arm_length + 1)
        axes.set_ylim(-1, env.stem_length + 1)
        axes.set_aspect("equal")
        axes.set_xlabel("x (0 = stem, negative = left arm)")
        axes.set_ylabel("y (0 = start, stem_length = junction)")
        axes.set_title(f"{env.name}: cue accuracy {env.cue_accuracy:.2f}")
        return figure, axes

    def _draw_corridor(self, axes) -> None:
        env = self.environment
        for x, y in sorted(env.valid_cells):
            axes.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5), 1.0, 1.0, facecolor="whitesmoke", edgecolor="lightgray"
                )
            )
        self._label_cell(axes, env.start_cell, "start", "tab:green")
        self._label_cell(axes, env.cue_cell, "cue", "tab:orange")
        self._label_cell(axes, env.junction, "junction", "tab:gray")
        self._label_cell(axes, env.left_endpoint, "L", "tab:blue")
        self._label_cell(axes, env.right_endpoint, "R", "tab:blue")

    @staticmethod
    def _label_cell(axes, cell: Tuple[int, int], text: str, color: str) -> None:
        axes.text(
            cell[0],
            cell[1] - 0.42,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )

    def _create_animated_artists(self, axes) -> dict:
        env = self.environment
        left_shade = Rectangle(
            (env.left_endpoint[0] - 0.5, env.left_endpoint[1] - 0.5),
            1.0,
            1.0,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.0,
        )
        right_shade = Rectangle(
            (env.right_endpoint[0] - 0.5, env.right_endpoint[1] - 0.5),
            1.0,
            1.0,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.0,
        )
        axes.add_patch(left_shade)
        axes.add_patch(right_shade)
        return {
            "belief_scatter": axes.scatter([], [], c="tab:purple", alpha=0.5, s=30, zorder=2),
            "agent": axes.plot([], [], "o", color="tab:red", markersize=14, zorder=3)[0],
            "trail": axes.plot([], [], "-", color="tab:red", alpha=0.4, zorder=1)[0],
            "true_goal": axes.plot([], [], "*", color="gold", markersize=18, zorder=2)[0],
            "left_shade": left_shade,
            "right_shade": right_shade,
            "status": axes.text(
                0.02,
                0.98,
                "",
                transform=axes.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            ),
        }

    # ── Per-frame ───────────────────────────────────────────────────────
    def _draw_frame(
        self,
        frame: int,
        states: List[np.ndarray],
        actions: List[Any],
        beliefs: List[Any],
        artists: dict,
    ):
        state = states[frame]
        artists["agent"].set_data([state[STATE_X]], [state[STATE_Y]])
        artists["trail"].set_data(
            [row[STATE_X] for row in states[: frame + 1]],
            [row[STATE_Y] for row in states[: frame + 1]],
        )
        goal_cell = self.environment.goal_endpoint(float(state[STATE_GOAL]))
        artists["true_goal"].set_data([goal_cell[0]], [goal_cell[1]])

        positions, weights, left_probability = self._read_belief(beliefs[frame])
        if positions.size:
            artists["belief_scatter"].set_offsets(positions)
            artists["belief_scatter"].set_sizes(weights * 300.0)
        else:
            artists["belief_scatter"].set_offsets(np.empty((0, 2)))

        if left_probability is None:
            artists["left_shade"].set_alpha(0.0)
            artists["right_shade"].set_alpha(0.0)
            belief_text = "belief over goal side: unavailable"
        else:
            artists["left_shade"].set_alpha(0.65 * float(left_probability))
            artists["right_shade"].set_alpha(0.65 * float(1.0 - left_probability))
            belief_text = (
                f"belief over goal side: L {left_probability:.2f} / R {1 - left_probability:.2f}"
            )

        artists["status"].set_text(
            f"step {frame}  action {actions[frame]}\n"
            f"true goal: {'L' if float(state[STATE_GOAL]) == GOAL_LEFT else 'R'}   "
            f"cue: {_CUE_PHASE_LABELS.get(float(state[STATE_CUE_PHASE]), '?')}\n"
            f"{belief_text}"
        )
        return list(artists.values())

    @staticmethod
    def _read_belief(belief: Any) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Return belief positions, their weights, and the mass on the left goal.

        Reads the unique-support view when the belief offers one — that is the
        collapsed distribution the other visualizers draw — and falls back to raw
        particles and normalized weights otherwise.

        Args:
            belief: The belief recorded on this step, or ``None``.

        Returns:
            An ``(N, 2)`` array of positions, an ``(N,)`` array of weights, and the
            probability the goal is on the left, or ``None`` when no belief is
            available.
        """
        if belief is None:
            return np.empty((0, 2)), np.empty(0), None

        values: List[Any]
        probs: np.ndarray
        if hasattr(belief, "to_unique_support_distribution"):
            support = belief.to_unique_support_distribution()
            values = list(support.values)
            probs = np.asarray(support.probs, dtype=np.float64)
        else:
            values = list(getattr(belief, "particles", []))
            probs = np.asarray(getattr(belief, "normalized_weights", []), dtype=np.float64)
        if not values or probs.size != len(values):
            return np.empty((0, 2)), np.empty(0), None

        total = float(np.sum(probs))
        if total <= 0.0:
            return np.empty((0, 2)), np.empty(0), None
        probs = probs / total

        positions = np.array(
            [[float(np.asarray(v)[STATE_X]), float(np.asarray(v)[STATE_Y])] for v in values],
            dtype=np.float64,
        )
        left_mass = float(
            np.sum(
                [
                    probs[index]
                    for index, value in enumerate(values)
                    if float(np.asarray(value)[STATE_GOAL]) == GOAL_LEFT
                ]
            )
        )
        return positions, probs, left_mass


__all__ = ["TMazeVisualizer"]
