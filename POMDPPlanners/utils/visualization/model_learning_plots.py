# SPDX-License-Identifier: MIT

"""The plots a model-fitting run is judged and debugged by.

The headline plot: return of the planner searching each round's model, against the
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

Beside it are the fit's own diagnostics, which the headline plot cannot give.
A learning curve that fails to rise has two very different causes -- the network
never fitted the data, or it fitted it and the model still does not help the
planner -- and only the training curves separate them. So the same run also
produces:

**Train and holdout loss per epoch, per round.** The gap between them is the
number to read: a holdout curve that turns up while the training curve keeps
falling is a fit spending its epochs memorizing, and more data will not fix it.
A round is one line pair, so a fit that degraded as the dataset grew is visible
as the curves separating round over round.

**Per-member training curves.** The ensemble's spread *is* its uncertainty
estimate, so a single member that diverged silently poisons every belief the
planner grades -- and it is invisible in the mean the ensemble reports.

**The model diagnostics per round.** Held-out likelihood says whether the fit
improved; the horizon drift ratio says whether the model's error over a planning
horizon stays inside the error bars it claims. A drift ratio above one is the
overconfidence a risk-sensitive planner turns into confident bad decisions, and
it can grow while the likelihood also improves.

Functions:
    plot_learning_curves: Return against data, one line per method.
    plot_round_returns: One run's per-round return with its confidence interval.
    plot_training_curves: Train and holdout loss per epoch, one panel per round.
    plot_member_training_curves: Every ensemble member's training curve.
    plot_round_diagnostics: Held-out likelihood and drift ratio per round.
    plot_model_learning_report: Every plot above, into one directory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from POMDPPlanners.training.model_learning.control_evaluation import (
    AggregatedCurve,
    LearningCurve,
    aggregate_curves,
)

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)

#: Confidence level of every band and error bar here -- the same one the tables
#: report, so a figure and a table cannot disagree.
CONFIDENCE_LEVEL = 0.95

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
        _draw(axis, aggregated, _METHOD_COLORS[index % len(_METHOD_COLORS)], method_curves)

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


def _draw(axis, curve: AggregatedCurve, color: str, episode_curves=None) -> None:
    """One method's mean line, with a 95% t-interval band.

    Across seeds when there is more than one, because that is the spread that
    answers whether two methods differed. With a single seed there is no such
    spread, so the band falls back to the interval across that run's evaluation
    episodes and says so in the label -- a plot with no band at all reads as a
    measurement without noise, and one whose band silently changes meaning is
    worse than either.
    """
    transitions = np.asarray(curve.cumulative_transitions, dtype=float)
    means = np.asarray(curve.mean_returns, dtype=float)
    across_seeds = curve.num_seeds > 1
    if across_seeds:
        errors = np.asarray(curve.standard_errors, dtype=float)
        half = _t_half_width(errors, curve.num_seeds)
        band_label = f"{curve.num_seeds} seeds"
    else:
        half = _episode_half_widths(episode_curves)
        band_label = "episodes, 1 seed"
    axis.plot(
        transitions,
        means,
        color=color,
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"{curve.method} ({band_label}, {int(CONFIDENCE_LEVEL * 100)}% CI)",
    )
    if half is not None and np.isfinite(half).any():
        band = np.nan_to_num(half, nan=0.0)[: means.size]
        axis.fill_between(transitions, means - band, means + band, color=color, alpha=0.2)


def _t_half_width(standard_errors: np.ndarray, sample_size: int) -> Optional[np.ndarray]:
    """Half-width of the t-interval implied by a standard error."""
    if sample_size < 2:
        return None
    return np.asarray(standard_errors, dtype=float) * float(
        stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, sample_size - 1)
    )


def _episode_half_widths(curves) -> Optional[np.ndarray]:
    """Per-round half-widths across a single run's evaluation episodes."""
    if not curves:
        return None
    points = curves[0].points
    half = []
    for point in points:
        returns = np.asarray(point.returns, dtype=float)
        if returns.size < 2:
            half.append(np.nan)
            continue
        error = float(np.std(returns, ddof=1) / np.sqrt(returns.size))
        half.append(error * float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, returns.size - 1)))
    return np.asarray(half, dtype=float)


def plot_round_returns(
    rounds: Sequence[Any],
    output_path: Path,
    baseline_return: Optional[float] = None,
    title: str = "Return per round, with 95% confidence intervals",
) -> Optional[Path]:
    """Plot each round's return with its interval, against the data it cost.

    The learning curve compares methods; this reads one run. An interval that
    spans the whole plot is the answer to "did round 3 beat round 2" -- no -- and
    it is the plot that stops a noise gap being read as progress.

    Args:
        rounds: The run's rounds, as results or as the tracker's dictionaries.
        output_path: Where to write the PNG.
        baseline_return: Return of the model the loop started from, drawn as a
            horizontal reference.
        title: Plot title.

    Returns:
        The written path, or ``None`` if no round was evaluated.
    """
    transitions, means, lows, highs = [], [], [], []
    for entry in rounds:
        control = entry.get("control") if isinstance(entry, dict) else getattr(entry, "control", None)
        if control is None:
            continue
        if not isinstance(control, dict):
            control = control.to_dict()
        returns = np.asarray(control["returns"], dtype=float)
        if returns.size == 0:
            continue
        transitions.append(float(control["cumulative_transitions"]))
        means.append(float(np.mean(returns)))
        if returns.size < 2:
            lows.append(0.0)
            highs.append(0.0)
        else:
            error = float(np.std(returns, ddof=1) / np.sqrt(returns.size))
            half = error * float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, returns.size - 1))
            lows.append(half)
            highs.append(half)
    if not means:
        logger.warning("plot_round_returns: no round was evaluated")
        return None

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(
        transitions,
        means,
        yerr=[lows, highs],
        color="tab:blue",
        linewidth=2,
        marker="o",
        capsize=4,
        label="round return",
    )
    best = int(np.argmax(means))
    axis.scatter(
        [transitions[best]],
        [means[best]],
        color="tab:green",
        zorder=5,
        s=90,
        label="chosen round",
    )
    if baseline_return is not None:
        axis.axhline(baseline_return, color="black", linestyle="--", linewidth=1.5,
                     label="initial model")
    axis.set_xlabel("Transitions collected")
    axis.set_ylabel("Return in the true world")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _finish(fig, output_path)


def _finish(fig, output_path: Path) -> Path:
    """Write a figure and close it, creating the parent directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _training_metrics(rounds: Sequence[Any]) -> List[Dict[str, List[float]]]:
    """Per-round training metrics, from round results or from the tracker's JSON."""
    extracted: List[Dict[str, List[float]]] = []
    for entry in rounds:
        metrics = (
            entry.get("training_metrics", {})
            if isinstance(entry, dict)
            else getattr(entry, "training_metrics", {})
        )
        extracted.append({key: list(values) for key, values in (metrics or {}).items()})
    return extracted


def plot_training_curves(
    rounds: Sequence[Any],
    output_path: Path,
    title: str = "Fit per round: training and held-out loss",
) -> Optional[Path]:
    """Plot training and holdout loss against epoch, one panel per round.

    Args:
        rounds: The run's rounds -- ``RoundResult`` objects, or the dictionaries
            the tracker writes to ``evaluation/rounds.json``, so a plot can be
            rebuilt from a finished run without re-fitting anything.
        output_path: Where to write the PNG.
        title: Figure title.

    Returns:
        The written path, or ``None`` if no round recorded a training curve.
    """
    metrics = _training_metrics(rounds)
    drawable = [(index, entry) for index, entry in enumerate(metrics, start=1) if entry.get("train_nll")]
    if not drawable:
        logger.warning("plot_training_curves: no round recorded a training curve")
        return None

    fig, axes = plt.subplots(
        1, len(drawable), figsize=(4.5 * len(drawable), 4), squeeze=False, sharey=True
    )
    for axis, (round_index, entry) in zip(axes[0], drawable):
        epochs = np.arange(1, len(entry["train_nll"]) + 1)
        axis.plot(epochs, entry["train_nll"], color="tab:blue", linewidth=2, label="train")
        members = [
            entry[key] for key in sorted(entry) if key.startswith("train_nll_member_")
        ]
        if len(members) > 1:
            # Across members, at the same confidence as every other interval
            # here: a mean line hides the ensemble disagreeing with itself,
            # which is the quantity the planner treats as uncertainty.
            stacked = np.asarray(members, dtype=float)
            mean = stacked.mean(axis=0)
            error = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            half = error * float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, stacked.shape[0] - 1))
            axis.fill_between(
                np.arange(1, stacked.shape[1] + 1),
                mean - half,
                mean + half,
                color="tab:blue",
                alpha=0.2,
                label=f"members, {int(CONFIDENCE_LEVEL * 100)}% CI",
            )
        holdout = entry.get("holdout_nll")
        if holdout:
            axis.plot(
                np.arange(1, len(holdout) + 1),
                holdout,
                color="tab:red",
                linewidth=2,
                label="held out",
            )
            best = int(np.argmin(holdout))
            # The epoch the fit should have stopped at: past it the network is
            # buying training loss with held-out loss.
            axis.axvline(best + 1, color="tab:red", linestyle=":", linewidth=1)
        axis.set_title(f"round {round_index}")
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Gaussian NLL")
    axes[0][0].legend()
    fig.suptitle(title)
    return _finish(fig, output_path)


def plot_member_training_curves(
    rounds: Sequence[Any],
    output_path: Path,
    title: str = "Ensemble members, training loss per epoch",
) -> Optional[Path]:
    """Plot every member's training curve, one panel per round.

    Args:
        rounds: The run's rounds, as for :func:`plot_training_curves`.
        output_path: Where to write the PNG.
        title: Figure title.

    Returns:
        The written path, or ``None`` if no per-member curve was recorded.
    """
    metrics = _training_metrics(rounds)
    drawable = [
        (index, entry)
        for index, entry in enumerate(metrics, start=1)
        if any(key.startswith("train_nll_member_") for key in entry)
    ]
    if not drawable:
        logger.warning("plot_member_training_curves: no per-member curve was recorded")
        return None

    fig, axes = plt.subplots(
        1, len(drawable), figsize=(4.5 * len(drawable), 4), squeeze=False, sharey=True
    )
    for axis, (round_index, entry) in zip(axes[0], drawable):
        members = sorted(key for key in entry if key.startswith("train_nll_member_"))
        for member in members:
            curve = entry[member]
            axis.plot(
                np.arange(1, len(curve) + 1),
                curve,
                linewidth=1.2,
                alpha=0.8,
                label=member.rsplit("_", 1)[-1],
            )
        axis.set_title(f"round {round_index}")
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Gaussian NLL")
    axes[0][0].legend(title="member", fontsize="small")
    fig.suptitle(title)
    return _finish(fig, output_path)


def plot_round_diagnostics(
    rounds: Sequence[Any],
    output_path: Path,
    title: str = "Model quality per round",
) -> Optional[Path]:
    """Plot the per-round model diagnostics against round index.

    Args:
        rounds: The run's rounds, as for :func:`plot_training_curves`.
        output_path: Where to write the PNG.
        title: Figure title.

    Returns:
        The written path, or ``None`` if no diagnostics were recorded.
    """
    series: Dict[str, List[float]] = {}
    indices: List[int] = []
    for position, entry in enumerate(rounds, start=1):
        diagnostics = (
            entry.get("diagnostics", {})
            if isinstance(entry, dict)
            else getattr(entry, "diagnostics", {})
        )
        if not diagnostics:
            continue
        indices.append(position)
        for name, value in diagnostics.items():
            series.setdefault(name, []).append(float(value))
    if not series:
        logger.warning("plot_round_diagnostics: no round recorded diagnostics")
        return None

    names = sorted(series)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), squeeze=False)
    for axis, name in zip(axes[0], names):
        values = series[name]
        axis.plot(indices[: len(values)], values, color="tab:green", linewidth=2, marker="o")
        if name == "horizon_drift_ratio":
            # One is the line the measure is calibrated against: above it the
            # model's error over the horizon is outside its own error bars.
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="claimed spread")
            axis.legend()
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Round")
        axis.grid(True, alpha=0.3)
    fig.suptitle(title)
    return _finish(fig, output_path)


def plot_model_learning_report(
    rounds: Sequence[Any],
    output_dir: Path,
    curves: Optional[Sequence[LearningCurve]] = None,
    baseline_return: Optional[float] = None,
) -> Dict[str, Path]:
    """Write every plot a fitting run should be read with, into one directory.

    Args:
        rounds: The run's rounds, as for :func:`plot_training_curves`.
        output_dir: Directory to write the PNGs into.
        curves: Control curves across methods and seeds. ``None`` skips the
            headline plot, which needs more than one run to be worth drawing.
        baseline_return: Return of the model the loop started from.

    Returns:
        Plot name to written path, omitting the plots that had no data.
    """
    output_dir = Path(output_dir)
    written: Dict[str, Path] = {}
    candidates = {
        "round_returns": lambda: plot_round_returns(
            rounds, output_dir / "round_returns.png", baseline_return=baseline_return
        ),
        "training_curves": lambda: plot_training_curves(
            rounds, output_dir / "training_curves.png"
        ),
        "member_training_curves": lambda: plot_member_training_curves(
            rounds, output_dir / "member_training_curves.png"
        ),
        "round_diagnostics": lambda: plot_round_diagnostics(
            rounds, output_dir / "round_diagnostics.png"
        ),
    }
    if curves:
        candidates["learning_curves"] = lambda: plot_learning_curves(
            curves, output_dir / "learning_curves.png", baseline_return=baseline_return
        )
    for name, draw in candidates.items():
        path = draw()
        if path is not None:
            written[name] = path
    return written
