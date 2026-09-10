# SPDX-License-Identifier: MIT

"""Visualization for the occupancy-grid mapping POMDP.

Renders an episode as an animated GIF with three panels per frame:

* **Occupancy grid** -- the map the robot has built, as the occupancy
  probability of each cell. This is the thing the task is *about*: the reward is
  the rate at which the grey in this panel turns black or white.
* **Belief over the true map** -- the per-cell posterior probability that the
  cell is occupied, averaged over the belief's particles with their weights.
  This is the belief view the environment contract asks for. Drawing the
  particles themselves is the usual choice here, but one particle is a whole
  hundred-cell world and a hundred overlaid worlds are not a picture of
  anything; the per-cell marginal is the projection the task turns on. It is
  still the belief and not a fit to it -- no distributional shape is assumed.
* **Ground truth** -- the real map, the robot's pose and its heading, drawn only
  here and labelled as hidden from the robot. It exists so a reviewer can check
  the other two panels against what was really there.

The two map panels look alike on purpose and mean different things, so both are
labelled: the left panel is what the robot *believes about the cells it has
swept*, the middle is what the *belief over whole worlds* implies per cell. A
planner that is exploring well drives grey out of the left panel; a filter that
is tracking well makes the middle panel approach the right one.

Every frame is drawn *before* its own action resolves.
:class:`~POMDPPlanners.core.simulation.StepData` records the state a step was
taken *from*, so each frame shows the situation the robot chose from while the
caption reports the action it is about to take.

Classes:
    OccupancyGridMappingVisualizer: Renders occupancy-grid mapping episodes.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
import matplotlib.pyplot as plt

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_LABELS,
    HEADING_STEPS,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (  # noqa: E501
        OccupancyGridMappingPOMDP,
    )


_ACTION_LABELS = ("move forward", "turn left", "turn right")

_PANEL_HEADINGS = (
    ("Observed inverse map", "approximate occupancy from measured scans"),
    ("Belief over the map", "chance a cell is occupied, over the belief's worlds"),
    ("Ground truth", "hidden occupancy; the robot knows its pose"),
)

_INK = "#193746"
_MUTED = "#526c78"
_ROBOT_COLOR = "#e7a624"
_LEGEND_EDGE = "#333333"
_LEGEND_FONTSIZE = 9


class OccupancyGridMappingVisualizer:
    """Renders an occupancy-grid mapping episode as an animated GIF.

    Attributes:
        env: The environment the episode was run in, used for grid geometry and
            to read state and belief in its own layout.
    """

    def __init__(self, env: "OccupancyGridMappingPOMDP"):
        """Initialize the visualizer.

        Args:
            env: The environment instance to visualize.
        """
        self.env = env
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self.num_cells = env.num_cells

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Render ``history`` to ``cache_path``.

        Args:
            history: Episode history. One frame per recorded step, so the frame
                count equals the recorded episode length.
            cache_path: Destination path, which must end in ``.gif``.

        Raises:
            TypeError: If ``history`` is not a list of :class:`StepData`, or
                ``cache_path`` is not a :class:`Path`.
            ValueError: If ``history`` is empty or ``cache_path`` is not a GIF.
        """
        self._validate(history, cache_path)
        frames = self._build_frames(history)
        fig, axes, artists = self._setup_figure()
        animate = self._animation_function(frames, artists)
        del axes
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
        """Per-cell occupancy probability of the true map under ``belief``.

        The weighted mean over particles, which is the exact marginal of a
        weighted particle belief rather than a summary of it. Deterministic in
        the belief, which the golden-file hash depends on.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(num_rows, num_cols)`` array of probabilities, all ``NaN`` when
            the belief carries nothing usable, which the colormap draws blank.
        """
        blank = np.full((self.num_rows, self.num_cols), np.nan)
        if belief is None:
            return blank
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return blank
        weights = np.asarray(
            getattr(belief, "normalized_weights", np.full(len(particles), 1.0 / len(particles))),
            dtype=np.float64,
        )
        offset = self.env.map_offset
        occupancy = (
            np.asarray(particles, dtype=np.float64)[:, offset : offset + self.num_cells] > 0.5
        )
        marginal = (weights[:, None] * occupancy).sum(axis=0) / weights.sum()
        return marginal.reshape(self.num_rows, self.num_cols)

    def _build_frames(self, history: List[StepData]) -> List[Dict[str, Any]]:
        """Turn the history into one self-contained drawing record per step."""
        frames: List[Dict[str, Any]] = []
        for index, step in enumerate(history):
            state = np.asarray(step.state, dtype=np.float64)
            row, col, heading = self.env.pose(state)
            frames.append(
                {
                    "index": index,
                    "total": len(history),
                    "grid": self.env.occupancy_probabilities(state),
                    "belief": self._belief_marginal(step.belief),
                    "truth": np.asarray(self.env.true_map(state), dtype=np.float64),
                    "row": row,
                    "col": col,
                    "heading": heading,
                    "entropy": self.env.entropy_bits(state),
                    "action": step.action,
                    "reward": step.reward,
                }
            )
        return frames

    def _square_handle(self, color: str, label: str) -> Any:
        """A filled square legend key, outlined so pale fills stay visible."""
        return Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            color=color,
            markeredgecolor=_LEGEND_EDGE,
            markeredgewidth=0.6,
            label=label,
        )

    def _panel_legend(self, ax: Axes, handles: List[Any]) -> None:
        """Put a single-column legend under ``ax`` so it stays inside its panel."""
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=1,
            fontsize=_LEGEND_FONTSIZE,
            frameon=True,
            facecolor="white",
            edgecolor="white",
            framealpha=1.0,
            handletextpad=0.5,
            labelspacing=0.4,
            borderpad=0.5,
        )

    def _setup_figure(self) -> Tuple[Figure, List[Axes], Dict[str, Any]]:
        """Build three equal-sized grids with a header, keys and a caption strip."""
        fig = plt.figure(figsize=(15, 8.5), facecolor="#f3f5f4")
        fig.text(0.045, 0.945, "OCCUPANCY GRID MAPPING", fontsize=25, weight="bold", color=_INK)
        fig.text(
            0.045,
            0.902,
            "Explore an unknown world  /  paid for the entropy it drives out of the map",
            fontsize=12,
            color=_MUTED,
        )
        artists: Dict[str, Any] = {}
        artists["status"] = fig.text(
            0.955, 0.943, "", ha="right", fontsize=15, weight="bold", color=_INK
        )
        artists["phase"] = fig.text(
            0.955,
            0.903,
            "Panels show the state before this action",
            ha="right",
            fontsize=11,
            color=_MUTED,
        )

        occupancy_cmap = LinearSegmentedColormap.from_list(
            "occupancy_probability", ["#f5f9fa", "#b9c6cc", "#1d3b4a"]
        ).with_extremes(bad="#dce0e4")

        axes: List[Axes] = []
        for i, (name, description) in enumerate(_PANEL_HEADINGS):
            left = 0.025 + i * 0.325
            ax = fig.add_axes((left + 0.036, 0.395, 0.24, 0.424))
            axes.append(ax)
            ax.set_title(name, fontsize=15, fontweight="bold", color=_INK, pad=32)
            ax.text(
                0.5,
                1.035,
                description,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color=_MUTED,
            )
            ax.set_xlabel("column", fontsize=9, color=_MUTED, labelpad=2)
            ax.set_ylabel("row", fontsize=9, color=_MUTED, labelpad=2)
            ax.set_xticks(range(self.num_cols))
            ax.set_yticks(range(self.num_rows))
            ax.set_xticks(np.arange(-0.5, self.num_cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.num_rows, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="both", length=0, labelsize=8, colors=_MUTED)
            for spine in ax.spines.values():
                spine.set_color("#9bb1bd")
                spine.set_linewidth(1.2)

        blank = np.zeros((self.num_rows, self.num_cols))
        for key, ax, cmap in zip(
            ("grid", "belief", "truth"),
            axes,
            (occupancy_cmap, occupancy_cmap, ListedColormap(("#f5f9fa", "#1d3b4a"))),
        ):
            artists[key] = ax.imshow(blank, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

        self._panel_legend(
            axes[0],
            [
                self._square_handle("#1d3b4a", "believed occupied (p = 1)"),
                self._square_handle("#b9c6cc", "unknown (p = 0.5)"),
                self._square_handle("#f5f9fa", "believed free (p = 0)"),
            ],
        )
        self._panel_legend(
            axes[1],
            [
                self._square_handle("#1d3b4a", "weighted occupancy probability = 1"),
                self._square_handle("#f5f9fa", "weighted occupancy probability = 0"),
                self._square_handle("#dce0e4", "no belief recorded"),
            ],
        )
        self._panel_legend(
            axes[2],
            [
                self._square_handle("#1d3b4a", "wall or obstacle"),
                self._square_handle("#f5f9fa", "free space"),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="",
                    markersize=11,
                    color=_ROBOT_COLOR,
                    markeredgecolor=_LEGEND_EDGE,
                    markeredgewidth=0.6,
                    label="robot, pointing along its heading",
                ),
            ],
        )

        colorbar = fig.colorbar(
            artists["grid"],
            cax=fig.add_axes((0.393, 0.205, 0.20, 0.012)),
            orientation="horizontal",
        )
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        colorbar.ax.tick_params(labelsize=8, length=0, colors=_MUTED)
        colorbar.set_label("chance the cell is occupied", fontsize=9, color=_MUTED)
        colorbar.outline.set_visible(False)

        artists["robot_grid"] = axes[0].scatter(
            [],
            [],
            s=150,
            marker="o",
            facecolors="none",
            edgecolors=_ROBOT_COLOR,
            linewidths=2.5,
            zorder=6,
        )
        artists["robot_truth"] = axes[2].scatter(
            [],
            [],
            s=190,
            marker="^",
            c=_ROBOT_COLOR,
            edgecolors="#7a5300",
            linewidths=1.0,
            zorder=6,
        )
        artists["heading_arrow"] = axes[2].annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": _ROBOT_COLOR, "linewidth": 2.2},
            zorder=7,
        )
        artists["caption"] = fig.text(
            0.5, 0.075, "", ha="center", va="center", fontsize=11, color=_INK, linespacing=1.8
        )
        return fig, axes, artists

    def _animation_function(self, frames: List[Dict[str, Any]], artists: Dict[str, Any]) -> Any:
        initial_entropy = self.env.initial_entropy_bits
        threshold = self.env.entropy_threshold_bits

        def animate(frame_index: int) -> Tuple[Any, ...]:
            frame = frames[min(frame_index, len(frames) - 1)]
            artists["grid"].set_data(frame["grid"])
            artists["belief"].set_data(frame["belief"])
            artists["truth"].set_data(frame["truth"])

            position = np.array([[frame["col"], frame["row"]]], dtype=np.float64)
            artists["robot_grid"].set_offsets(position)
            artists["robot_truth"].set_offsets(position)
            delta_row, delta_col = HEADING_STEPS[frame["heading"]]
            artists["heading_arrow"].set_position((frame["col"], frame["row"]))
            artists["heading_arrow"].xy = (
                frame["col"] + 0.9 * delta_col,
                frame["row"] + 0.9 * delta_row,
            )

            artists["status"].set_text(
                f"{frame['index'] + 1:02d} / {frame['total']:02d}   |   "
                f"map entropy {frame['entropy']:.1f} of {initial_entropy:.0f} bits"
            )
            artists["phase"].set_text(
                "Final recorded state"
                if frame["action"] is None
                else "Panels show the state before this action"
            )
            step_text = f"Step {frame['index'] + 1} of {frame['total']}"
            heading_text = HEADING_LABELS[frame["heading"]]
            if frame["action"] is None:
                headline = f"{step_text} - episode over, no further action"
                detail = (
                    f"Final map entropy {frame['entropy']:.2f} bits; "
                    f"the map counts as resolved at {threshold:.2f} bits or below"
                )
            else:
                action_text = _ACTION_LABELS[int(frame["action"])]
                reward_text = "-" if frame["reward"] is None else f"{float(frame['reward']):+.2f}"
                headline = (
                    f"{step_text} - robot at row {frame['row']}, column {frame['col']}, "
                    f"facing {heading_text}; about to {action_text}"
                )
                detail = (
                    f"Observed map entropy reduction: {reward_text} bits.   "
                    f"Map entropy before it: {frame['entropy']:.2f} bits "
                    f"(resolved at {threshold:.2f} or below)"
                )
            artists["caption"].set_text(f"{headline}\n{detail}")

            return (
                artists["grid"],
                artists["belief"],
                artists["truth"],
                artists["robot_grid"],
                artists["robot_truth"],
                artists["caption"],
            )

        return animate

    def _save(self, fig: Figure, animate: Any, num_frames: int, cache_path: Path) -> None:
        images = []
        canvas = FigureCanvasAgg(fig)
        try:
            for index in range(num_frames):
                animate(index)
                canvas.draw()
                images.append(
                    Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB")
                )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(
                cache_path,
                save_all=True,
                append_images=images[1:],
                duration=[900] * (num_frames - 1) + [2400],
                loop=0,
                disposal=2,
            )
        finally:
            plt.close(fig)
