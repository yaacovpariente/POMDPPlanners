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
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image
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

_AGENT_VIEW_COLORS = ("#e5edf1", "#c5e1e9", "#bf4b36")
_AGENT_VIEW_LABELS = (
    "not probed yet",
    "probed - water (miss)",
    "probed - ship (hit)",
)

# Marks drawn on top of the boards.
_PROBE_RING_COLOR = "#e7a624"
_PROBED_CROSS_COLOR = "#f5ad76"

# Panel headings. Each is a short name plus one plain sentence, because the
# name alone ("Belief") does not tell a reader what the picture is of.
_PANEL_HEADINGS = (
    ("Agent view", "what the agent has observed so far"),
    ("Belief", "estimated chance a cell contains a ship"),
    ("Ground truth", "actual ship layout, hidden from the agent"),
)

_LEGEND_FONTSIZE = 9

# Legend swatches sit on a light panel, so every one carries a thin dark edge;
# without it the pale "not probed yet" grey and the yellow ring vanish.
_LEGEND_EDGE = "#333333"


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
        return Line2D(
            [0], [0], marker="s", linestyle="", markersize=10, color=color,
            markeredgecolor=_LEGEND_EDGE, markeredgewidth=0.6, label=label,
        )

    def _panel_legend(self, ax: Axes, handles: List[Any]) -> None:
        """Put a legend under ``ax``, below the axis label.

        One key per row. A multi-column legend grows sideways with its longest
        label, so the agent panel's legend ran wider than its own panel and
        collided with the belief panel's. Stacked in a single column, each
        legend is at most as wide as its longest key, which keeps every legend
        inside the panel it explains.
        """
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
            ncol=1, fontsize=_LEGEND_FONTSIZE, frameon=True,
            facecolor="white", edgecolor="white", framealpha=1.0,
            handletextpad=0.5, labelspacing=0.45, borderpad=0.6,
        )

    def _setup_figure(self) -> Tuple[Figure, List[Axes], Dict[str, Any]]:
        """Build equal-sized boards with a separate header, keys and outcome strip."""
        fig = plt.figure(figsize=(15, 8.5), facecolor="#f3f5f4")
        axes = []
        ink, muted = "#193746", "#526c78"
        fig.text(0.045, 0.945, "BATTLESHIP", fontsize=25, weight="bold", color=ink)
        fig.text(0.045, 0.902, "Search a hidden fleet  /  compare evidence, belief and reality",
                 fontsize=12, color=muted)
        artists: Dict[str, Any] = {}
        artists["status"] = fig.text(0.955, 0.943, "", ha="right", fontsize=15,
                                      weight="bold", color=ink)
        artists["phase"] = fig.text(0.955, 0.903, "Boards show the state before this probe",
                 ha="right", fontsize=11, color=muted)
        for i, (name, description) in enumerate(_PANEL_HEADINGS):
            left = 0.025 + i * 0.325
            card = FancyBboxPatch((left, 0.20), 0.31, 0.65,
                                 boxstyle="round,pad=0.005,rounding_size=0.015",
                                 transform=fig.transFigure, facecolor="white",
                                 edgecolor="#d8e2e5", linewidth=1, zorder=-1)
            fig.add_artist(card)
            ax = fig.add_axes((left + 0.036, 0.395, 0.24, 0.424))
            axes.append(ax)
            ax.set_title(name, fontsize=15, fontweight="bold", color=ink, pad=32)
            ax.text(0.5, 1.035, description, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=9, color=muted)
            ax.set_xlabel("column", fontsize=9, color=muted, labelpad=2)
            ax.set_ylabel("row", fontsize=9, color=muted, labelpad=2)
            ax.set_xticks(range(self.board_size))
            ax.set_yticks(range(self.board_size))
            ax.set_xticks(np.arange(-0.5, self.board_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.board_size, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="both", length=0, labelsize=9, colors=muted)
            for spine in ax.spines.values():
                spine.set_color("#9bb1bd")
                spine.set_linewidth(1.2)

        blank = np.zeros((self.board_size, self.board_size))
        belief_cmap = LinearSegmentedColormap.from_list(
            "battleship_probability", ["#edf4f2", "#77bcb3", "#125650"]).with_extremes(bad="#dce0e4")
        for key, ax, cmap, vmax in zip(
            ("agent", "belief", "truth"), axes,
            (ListedColormap(_AGENT_VIEW_COLORS), belief_cmap,
             ListedColormap(("#e4f0f4", "#35576a"))), (2, 1, 1),
        ):
            artists[key] = ax.imshow(blank, cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation="nearest")

        agent_handles = [self._square_handle(color, label)
                         for color, label in zip(_AGENT_VIEW_COLORS, _AGENT_VIEW_LABELS)]
        agent_handles.append(Line2D(
            [0], [0], marker="o", linestyle="", markersize=11,
            markerfacecolor="none", markeredgecolor=_PROBE_RING_COLOR,
            markeredgewidth=2.5, label="probing now - result not on the board yet"))
        self._panel_legend(axes[0], agent_handles)
        self._panel_legend(axes[1], [
            self._square_handle("#125650", "100% - certainly a ship"),
            self._square_handle("#edf4f2", "0% - certainly water"),
        ])
        self._panel_legend(axes[2], [
            self._square_handle("#35576a", "ship cell"),
            self._square_handle("#e4f0f4", "water"),
            Line2D([0], [0], marker="x", linestyle="", markersize=8,
                   color=_PROBED_CROSS_COLOR, markeredgewidth=1.8, label="already probed"),
        ])
        for ax, note in zip(axes, (
            "Cross = hit; dot = miss",
            "Cell labels rounded; — = belief unavailable",
            "Each shape marks one occupied cell",
        )):
            ax.text(.5, -.145, note, transform=ax.transAxes, ha="center",
                    fontsize=9, color=muted)
        colorbar = fig.colorbar(artists["belief"], cax=fig.add_axes((0.393, 0.205, 0.20, 0.012)),
                                orientation="horizontal")
        colorbar.set_ticks([0, .25, .5, .75, 1])
        colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        colorbar.ax.tick_params(labelsize=8, length=0, colors=muted)
        colorbar.set_label("chance the cell contains a ship", fontsize=9, color=muted)
        colorbar.outline.set_visible(False)

        artists["probe_marker"] = axes[0].scatter(
            [], [], s=380, facecolors="none", edgecolors=_PROBE_RING_COLOR,
            linewidths=3, zorder=6)
        artists["truth_probe"] = axes[2].scatter(
            [], [], s=90, marker="x", c=_PROBED_CROSS_COLOR, linewidths=2, zorder=5)
        artists["hit_marks"] = axes[0].scatter(
            [], [], s=85, marker="x", c="white", linewidths=2, zorder=5)
        artists["miss_marks"] = axes[0].scatter(
            [], [], s=28, marker="o", c="#376d80", zorder=5)
        artists["ship_cells"] = []
        artists["probabilities"] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                # Occupancy stores ship cells, not identities or orientations.
                patch = FancyBboxPatch((col-.33, row-.33), .66, .66,
                                      boxstyle="round,pad=0,rounding_size=0.13",
                                      facecolor="#35576a", edgecolor="#8facbb",
                                      linewidth=1.5, zorder=3, visible=False)
                axes[2].add_patch(patch)
                artists["ship_cells"].append(patch)
                text = axes[1].text(col, row, "", ha="center", va="center",
                                    fontsize=max(5, min(12, 60/self.board_size)), zorder=3)
                artists["probabilities"].append(text)
        artists["caption"] = fig.text(
            0.5, 0.075, "", ha="center", va="center", fontsize=11,
            color=ink, linespacing=1.8)
        return fig, axes, artists

    @staticmethod
    def _apply_layout(fig: Figure) -> None:
        """Axes use fixed equal-sized slots so a colorbar cannot shrink a board."""
        del fig

    def _animation_function(
        self, frames: List[Dict[str, Any]], axes: List[Axes], artists: Dict[str, Any]
    ) -> Any:
        board_size = self.board_size
        num_ship_cells = self.env.num_ship_cells

        def animate(frame_index: int) -> Tuple[Any, ...]:
            frame = frames[min(frame_index, len(frames) - 1)]
            artists["agent"].set_data(frame["agent_view"])
            artists["belief"].set_data(frame["belief"])
            # Water is the base; inset shapes show precisely the occupied cells.
            artists["truth"].set_data(np.zeros_like(frame["truth"]))
            for patch, occupied in zip(artists["ship_cells"], frame["truth"].flat):
                patch.set_visible(bool(occupied))
            for text, probability in zip(artists["probabilities"], frame["belief"].flat):
                text.set_text("—" if not np.isfinite(probability) else f"{probability:.0%}")
                text.set_color("white" if probability > .6 else "#193746")
                text.set_visible(self.board_size <= 10)
            for key, code in (("hit_marks", _HIT), ("miss_marks", _MISS)):
                rows, cols = np.nonzero(frame["agent_view"] == code)
                artists[key].set_offsets(np.column_stack((cols, rows)))
            artists["status"].set_text(
                f"{frame['index'] + 1:02d} / {frame['total']:02d}   |   "
                f"{frame['hits']} of {num_ship_cells} ship cells found")

            artists["phase"].set_text(
                "Final recorded state" if frame["action"] is None
                else "Boards show the state before this probe")
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
                    f"row {row}, column {col} (amber ring)"
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
        self._apply_layout(fig)
        images = []
        try:
            for index in range(num_frames):
                animate(index)
                fig.canvas.draw()
                images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(cache_path, save_all=True, append_images=images[1:],
                           duration=[1400] * (num_frames - 1) + [2400],
                           loop=0, disposal=2)
        finally:
            plt.close(fig)
