# SPDX-License-Identifier: MIT

"""The learning curve a model-fitting run is judged by.

One plot: return of the planner searching each round's model, against the
transitions that round cost, one line per method, averaged over shared seeds.
It is the figure Ross & Bagnell (ICML 2012) report and the only thing that
answers whether iterating on the data collection was worth it.

Three choices in here are the plot rather than decoration.

**The x-axis is transitions, not rounds.** Two methods' rounds are not the same
amount of data, so a round-indexed plot compares different budgets and flatters
whichever collects more per round.

**The band is across seeds.** Spread across the episodes inside one repetition
says how noisy an evaluation was; spread across repetitions says whether the
methods actually differed. Only the second belongs next to a claim.

**A reference line for the starting model.** Without it a curve that rises says
nothing about whether the fitted models ever beat the hand-built one they
started from, which is usually the decision being made.

Unlike its neighbours this module is not re-exported from the package's
``__init__``. That import is eager, so exporting it would make every importer of
``utils.visualization`` -- including ones with no interest in model fitting --
pull in the training layer, and would leave a cycle waiting for the first time
training wants a plot. Import it by module path.

Functions:
    plot_learning_curves: Return against data, one line per method.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.training.model_learning.control_evaluation import (
    AggregatedCurve,
    LearningCurve,
    aggregate_curves,
)

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)

# Enough distinct colors for the method families a comparison plausibly carries;
# beyond this the lines stop being separable and the plot should be split.
_METHOD_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")


def plot_learning_curves(
    curves: Sequence[LearningCurve],
    output_path: Path,
    baseline_return: Optional[float] = None,
    baseline_label: str = "initial model",
    log_scale: bool = False,
    title: str = "Control performance vs collected data",
) -> Optional[Path]:
    """Plot mean return against cumulative transitions, one line per method.

    Args:
        curves: Curves across methods and seeds, as returned by
            :func:`~POMDPPlanners.training.model_learning.curve_comparison.run_learning_curves`.
            Curves sharing a method are averaged together.
        output_path: Where to write the PNG.
        baseline_return: Return of the model the loop started from, drawn as a
            horizontal reference. ``None`` omits it.
        baseline_label: Label for that reference line.
        log_scale: Log-scale the y-axis. The paper does, because its quantity is
            a cost spanning five orders of magnitude; a bounded return usually
            reads better linear, so this is off by default.
        title: Plot title.

    Returns:
        The written path, or ``None`` if no curve had any points.
    """
    grouped: Dict[str, list] = {}
    for curve in curves:
        if curve.points:
            grouped.setdefault(curve.method, []).append(curve)
    if not grouped:
        logger.warning("plot_learning_curves: no evaluated rounds to plot")
        return None

    fig, axis = plt.subplots(figsize=(9, 5))
    for index, (method, method_curves) in enumerate(sorted(grouped.items())):
        aggregated = aggregate_curves(method_curves)
        _draw(axis, aggregated, _METHOD_COLORS[index % len(_METHOD_COLORS)])

    if baseline_return is not None:
        axis.axhline(
            baseline_return,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=baseline_label,
        )

    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("Transitions collected")
    axis.set_ylabel("Mean return of the planner in the true world")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _draw(axis, curve: AggregatedCurve, color: str) -> None:
    """One method's mean line, with a band of one standard error across seeds."""
    transitions = np.asarray(curve.cumulative_transitions, dtype=float)
    means = np.asarray(curve.mean_returns, dtype=float)
    errors = np.asarray(curve.standard_errors, dtype=float)
    axis.plot(
        transitions,
        means,
        color=color,
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"{curve.method} ({curve.num_seeds} seeds)",
    )
    if np.isfinite(errors).any():
        band = np.nan_to_num(errors, nan=0.0)
        axis.fill_between(transitions, means - band, means + band, color=color, alpha=0.2)
