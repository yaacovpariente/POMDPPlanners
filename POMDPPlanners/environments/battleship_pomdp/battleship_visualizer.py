# SPDX-License-Identifier: MIT

"""Visualization for the Battleship POMDP.

Renders an episode as an animated GIF with three panels per frame:

* **Agent view** — what the agent has actually learned: probed cells shown as
  hit or miss, everything else unknown. The hidden fleet is *not* drawn here,
  and nothing in this panel is derived from it. Leaking it would make the panel
  useless for the thing it exists for, namely telling apart a planner that
  reasoned about the board from one that guessed well.
* **Belief** — the posterior probability that each cell holds a ship. This is
  the belief view the environment contract asks for. A cloud of particles is
  the usual choice, but a Battleship particle is a whole 25-cell board, and
  twenty overlaid boards are not a picture of anything; the per-cell marginal is
  the projection the task actually turns on, and for
  :class:`~POMDPPlanners.environments.battleship_pomdp.battleship_belief.BattleshipBelief`
  it is exact rather than a Monte Carlo summary. It is still the belief and not
  a fit to it: no distributional shape is assumed anywhere.
* **Ground truth** — the real fleet, drawn only here and labelled as hidden from
  the agent. It exists so a reviewer can check the other two panels against what
  was really on the board.

Every frame is drawn *before* its own action resolves.
:class:`~POMDPPlanners.core.simulation.StepData` records the state a step was
taken *from*, so the boards show the situation the agent chose from, while the
step's observation and reward are the outcome of the probe it is about to make.
The labels say exactly that: the ring marks the cell being probed now, the
crosses mark cells probed earlier, and the caption reports this probe's result
separately from the tally of what was already known. Labelling the ring as a
resolved hit or miss would claim the panels show something they do not.

Classes:
    BattleshipVisualizer: Renders Battleship episodes.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
        BattleshipPOMDP,
    )


# Agent-view cell codes, in the order the colormap lists them.
_UNKNOWN = 0
_MISS = 1
_HIT = 2

_AGENT_VIEW_COLORS = ("#c9ccd1", "#2f6fb5", "#c0392b")
_AGENT_VIEW_LABELS = (
    "not probed yet",
    "probed - water (miss)",
    "probed - ship (hit)",
)

# Marks drawn on top of the boards.
_PROBE_RING_COLOR = "#ffd21f"
_PROBED_CROSS_COLOR = "#c0392b"

# Panel headings. Each is a short name plus one plain sentence, because the
# name alone ("Belief") does not tell a reader what the picture is of.
_PANEL_HEADINGS = (
    ("Agent view", "what the agent has observed so far"),
    ("Belief", "estimated chance a cell contains a ship"),
    ("Ground truth", "actual ship layout, hidden from the agent"),
)

_TITLE_FONTSIZE = 12
_SUBTITLE_FONTSIZE = 9.5
_LEGEND_FONTSIZE = 9
_CAPTION_FONTSIZE = 11
_AXIS_LABEL_FONTSIZE = 9.5

# Legend swatches sit on a light panel, so every one carries a thin dark edge;
# without it the pale "not probed yet" grey and the yellow ring vanish.
_LEGEND_EDGE = "#333333"
_LEGEND_FACE = "#f2f3f5"


class BattleshipVisualizer:
    """Renders a Battleship episode as an animated GIF.

    Attributes:
        env: The environment the episode was run in, used for board geometry
            and to read the belief's marginal.
    """

    def __init__(self, env: "BattleshipPOMDP"):
        """Initialize the visualizer.

        Args:
            env: The Battleship environment instance to visualize.
        """
        self.env = env
        self.board_size = env.board_size
        self.num_cells = env.num_cells

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Render ``history`` to ``cache_path``.

        Args:
            history: Episode history. One frame is drawn per recorded step, so
                the frame count equals the recorded episode length.
            cache_path: Destination path, which must end in ``.gif``.

        Raises:
            TypeError: If ``history`` is not a list of :class:`StepData`, or
                ``cache_path`` is not a :class:`Path`.
            ValueError: If ``history`` is empty or ``cache_path`` is not a GIF.
        """
        self._validate(history, cache_path)
        frames = self._build_frames(history)
        fig, axes, artists = self._setup_figure()
        animate = self._animation_function(frames, axes, artists)
        self._save(fig, animate, len(frames), cache_path)

    def _validate(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, list):
            raise TypeError("history must be a list object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a list of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _belief_marginal(self, belief: Any) -> np.ndarray:
        """Per-cell ship probability under ``belief``.

        Uses the exact marginal when the belief can supply one, and otherwise
        the particle mean, so an episode recorded with a generic particle filter
        still renders. Both are deterministic functions of the belief, which the
        golden-file hash depends on.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(num_cells,)`` array of probabilities; all ``NaN`` when the belief
            carries nothing usable, which the colormap renders as blank.
        """
        if belief is None:
            return np.full(self.num_cells, np.nan)
        if hasattr(belief, "occupancy_marginal"):
            return np.asarray(belief.occupancy_marginal(self.env), dtype=np.float64)
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return np.full(self.num_cells, np.nan)
        weights = np.asarray(
            getattr(belief, "normalized_weights", np.full(len(particles), 1.0 / len(particles))),
            dtype=np.float64,
        )
        occupancy = np.asarray(particles, dtype=np.float64)[:, : self.num_cells] > 0.5
        return (weights[:, None] * occupancy).sum(axis=0) / weights.sum()

    def _build_frames(self, history: List[StepData]) -> List[Dict[str, Any]]:
        """Turn the history into one self-contained drawing record per step."""
        frames: List[Dict[str, Any]] = []
        for index, step in enumerate(history):
            state = np.asarray(step.state, dtype=np.float64)
            occupancy = state[: self.num_cells] > 0.5
            probed = state[self.num_cells :] > 0.5

            agent_view = np.full(self.num_cells, _UNKNOWN, dtype=np.int64)
            agent_view[probed & ~occupancy] = _MISS
            agent_view[probed & occupancy] = _HIT

            frames.append(
                {
                    "index": index,
                    "total": len(history),
                    "agent_view": agent_view.reshape(self.board_size, self.board_size),
                    "belief": self._belief_marginal(step.belief).reshape(
                        self.board_size, self.board_size
                    ),
                    "truth": occupancy.astype(np.float64).reshape(
                        self.board_size, self.board_size
                    ),
                    "probed": probed.reshape(self.board_size, self.board_size),
                    "action": step.action,
                    "observation": step.observation,
                    "reward": step.reward,
                    "hits": int(np.count_nonzero(occupancy & probed)),
                }
            )
        return frames

    def _square_handle(self, color: str, label: str) -> Any:
        """A filled square legend key, outlined so pale fills stay visible."""
        return plt.Line2D(
            [0], [0], marker="s", linestyle="", markersize=10, color=color,
            markeredgecolor=_LEGEND_EDGE, markeredgewidth=0.6, label=label,
        )

    def _panel_legend(self, ax: Axes, handles: List[Any], ncol: int) -> None:
        """Put a legend under ``ax``, below the axis label."""
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=ncol, fontsize=_LEGEND_FONTSIZE, frameon=True,
            facecolor=_LEGEND_FACE, edgecolor="#d0d3d8", framealpha=1.0,
            handletextpad=0.5, columnspacing=1.2, borderpad=0.6,
        )

    def _setup_figure(self) -> Tuple[Figure, List[Axes], Dict[str, Any]]:
        fig_temp, axes_temp = plt.subplots(1, 3, figsize=(15.0, 6.6))
        fig = cast(Figure, fig_temp)
        axes = [cast(Axes, ax) for ax in np.atleast_1d(axes_temp).ravel()]

        agent_cmap = plt.matplotlib.colors.ListedColormap(_AGENT_VIEW_COLORS)
        blank = np.zeros((self.board_size, self.board_size))

        artists: Dict[str, Any] = {}
        artists["agent"] = axes[0].imshow(
            blank, cmap=agent_cmap, vmin=_UNKNOWN, vmax=_HIT, interpolation="nearest"
        )
        artists["belief"] = axes[1].imshow(
            blank, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        artists["truth"] = axes[2].imshow(
            blank, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest"
        )

        # White cell separators read well on the coloured panels, but the truth
        # panel draws water as white, where a white grid disappears.
        grid_colors = ("white", "white", "#b6bac0")
        for ax, (name, description), grid_color in zip(axes, _PANEL_HEADINGS, grid_colors):
            ax.set_title(name, fontsize=_TITLE_FONTSIZE, fontweight="bold", pad=22)
            ax.text(
                0.5, 1.035, description, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=_SUBTITLE_FONTSIZE, color="#444444",
            )
            ax.set_xlabel("column", fontsize=_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("row", fontsize=_AXIS_LABEL_FONTSIZE)
            ax.set_xticks(range(self.board_size))
            ax.set_yticks(range(self.board_size))
            ax.set_xticks(np.arange(-0.5, self.board_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.board_size, 1), minor=True)
            ax.grid(which="minor", color=grid_color, linewidth=1.0)
            ax.tick_params(which="minor", length=0)

        agent_handles = [
            self._square_handle(color, label)
            for color, label in zip(_AGENT_VIEW_COLORS, _AGENT_VIEW_LABELS)
        ]
        agent_handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="", markersize=11,
                markerfacecolor="none", markeredgecolor=_PROBE_RING_COLOR,
                markeredgewidth=2.5,
                label="probing now - result not on the board yet",
            )
        )
        self._panel_legend(axes[0], agent_handles, ncol=2)

        colorbar = fig.colorbar(artists["belief"], ax=axes[1], fraction=0.046, pad=0.04)
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        colorbar.set_label("chance the cell contains a ship", fontsize=_LEGEND_FONTSIZE)
        colorbar.ax.tick_params(labelsize=_LEGEND_FONTSIZE)
        self._panel_legend(
            axes[1],
            [
                self._square_handle("#fde725", "100% - almost certainly a ship"),
                self._square_handle("#440154", "0% - almost certainly water"),
            ],
            ncol=2,
        )

        self._panel_legend(
            axes[2],
            [
                self._square_handle("#000000", "ship cell"),
                self._square_handle("#ffffff", "water"),
                plt.Line2D(
                    [0], [0], marker="x", linestyle="", markersize=8,
                    color=_PROBED_CROSS_COLOR, markeredgewidth=1.8,
                    label="already probed",
                ),
            ],
            ncol=3,
        )

        artists["probe_marker"] = axes[0].scatter(
            [], [], s=180, facecolors="none", edgecolors=_PROBE_RING_COLOR,
            linewidths=2.5, zorder=5,
        )
        artists["truth_probe"] = axes[2].scatter(
            [], [], s=30, marker="x", c=_PROBED_CROSS_COLOR, zorder=5
        )
        artists["caption"] = fig.text(
            0.5, 0.055, "", ha="center", va="center", fontsize=_CAPTION_FONTSIZE,
            linespacing=1.6,
        )
        return fig, axes, artists

    def _animation_function(
        self, frames: List[Dict[str, Any]], axes: List[Axes], artists: Dict[str, Any]
    ) -> Any:
        board_size = self.board_size
        num_ship_cells = self.env.num_ship_cells

        def animate(frame_index: int) -> Tuple[Any, ...]:
            frame = frames[min(frame_index, len(frames) - 1)]
            artists["agent"].set_data(frame["agent_view"])
            artists["belief"].set_data(frame["belief"])
            artists["truth"].set_data(frame["truth"])

            step_text = f"Step {frame['index'] + 1} of {frame['total']}"
            if frame["action"] is None:
                # The last recorded step carries no action: the boards are the
                # final situation and there is no probe to describe.
                artists["probe_marker"].set_offsets(np.empty((0, 2)))
                headline = f"{step_text} - episode over, no further probe"
                detail = f"Ship cells found: {frame['hits']} of {num_ship_cells}"
            else:
                row, col = divmod(int(frame["action"]), board_size)
                artists["probe_marker"].set_offsets(np.array([[col, row]]))
                headline = (
                    f"{step_text} - the agent is about to probe "
                    f"row {row}, column {col} (yellow ring)"
                )
                observation_text = "-" if frame["observation"] is None else (
                    "HIT - a ship is there"
                    if int(frame["observation"])
                    else "MISS - water"
                )
                reward_text = (
                    "-" if frame["reward"] is None else f"{float(frame['reward']):+.2f}"
                )
                detail = (
                    f"Result of this probe: {observation_text}.   "
                    f"Reward: {reward_text}.   "
                    f"Ship cells found before this probe: "
                    f"{frame['hits']} of {num_ship_cells}"
                )
            artists["caption"].set_text(f"{headline}\n{detail}")

            probed_rows, probed_cols = np.nonzero(frame["probed"])
            if probed_rows.size:
                artists["truth_probe"].set_offsets(np.column_stack([probed_cols, probed_rows]))
            else:
                artists["truth_probe"].set_offsets(np.empty((0, 2)))

            return (
                artists["agent"],
                artists["belief"],
                artists["truth"],
                artists["probe_marker"],
                artists["truth_probe"],
                artists["caption"],
            )

        del axes
        return animate

    def _save(self, fig: Figure, animate: Any, num_frames: int, cache_path: Path) -> None:
        ani = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=1000, blit=False, repeat=False
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Reserved margins, not tight_layout: the legends and the two-line
        # caption live outside the axes, and a layout solver that ignored
        # them would clip whichever frame happened to have the longest text.
        fig.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.30, wspace=0.30)
        ani.save(cache_path, writer="pillow", fps=1)
        plt.close(fig)
