# SPDX-License-Identifier: MIT

"""Diagnostic plots for a hyperparameter tuning study.

These answer two questions a tuning run should not be trusted without:

1. Did the search converge, or did it just run out of budget? The front-quality
   curve is the evidence -- it is the same number the early-stopping callback
   watches, so a flat tail is exactly what made the study stop.
2. What did the search give up? A study told to maximize return will happily
   trade away collision rate, so metrics that were recorded but *not* optimized
   are plotted too.

The plots are built from :class:`TrialRecord` rather than from an Optuna study,
so they can be regenerated later from the JSON dump without re-running
anything. The study object itself is in-memory only: its ``user_attrs`` hold
episode histories, which no Optuna storage backend can serialize.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend

logger = logging.getLogger(__name__)

# Objectives are plotted pairwise; beyond this many the grid stops being readable.
MAX_OBJECTIVE_PAIRS = 6


@dataclass(frozen=True)
class TrialRecord:
    """One trial, reduced to what the diagnostic plots need.

    Attributes:
        number: Optuna trial number.
        state: Optuna trial state name, e.g. ``"COMPLETE"``.
        params: The hyperparameter values the sampler suggested.
        objective_values: Optimized metric name to its value for this trial.
        metric_statistics: Every recorded metric, optimized or not, mapped to
            ``(value, lower_confidence_bound, upper_confidence_bound)``. The
            bounds are what tell you whether one trial really beat another or
            just got a luckier draw of episodes.
        duration_seconds: Wall-clock time of the trial, or None if unknown.
        is_pareto: Whether the trial is on the study's final Pareto front.
        confidence_interval_level: The level the bounds in ``metric_statistics``
            were computed at, carried alongside them so a plot can label its
            intervals without the caller having to remember the setting.
    """

    number: int
    state: str
    params: Dict[str, Any] = field(default_factory=dict)
    objective_values: Dict[str, float] = field(default_factory=dict)
    metric_statistics: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    is_pareto: bool = False
    confidence_interval_level: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the record."""
        return {
            "number": self.number,
            "state": self.state,
            "params": self.params,
            "objective_values": self.objective_values,
            "metric_statistics": {
                name: list(bounds) for name, bounds in self.metric_statistics.items()
            },
            "duration_seconds": self.duration_seconds,
            "is_pareto": self.is_pareto,
            "confidence_interval_level": self.confidence_interval_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialRecord":
        """Rebuild a record from :meth:`to_dict` output."""
        return cls(
            number=data["number"],
            state=data["state"],
            params=data.get("params", {}),
            objective_values=data.get("objective_values", {}),
            metric_statistics={
                name: tuple(bounds)  # type: ignore[misc]
                for name, bounds in data.get("metric_statistics", {}).items()
            },
            duration_seconds=data.get("duration_seconds"),
            is_pareto=data.get("is_pareto", False),
            confidence_interval_level=data.get("confidence_interval_level"),
        )


def extract_trial_records(
    study, parameters_to_optimize, confidence_interval_level: Optional[float] = None
) -> List[TrialRecord]:
    """Reduce an Optuna study to plain records.

    Args:
        study: The completed Optuna study.
        parameters_to_optimize: The ``(metric_name, direction)`` pairs the study
            optimized, in the same order as the study's directions.
        confidence_interval_level: The level the trial statistics were computed
            at, so the plots can state it.

    Returns:
        One record per trial, in trial order.
    """
    import optuna  # pylint: disable=import-outside-toplevel

    pareto_numbers = set()
    try:
        pareto_numbers = {trial.number for trial in study.best_trials}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Could not read Pareto trials from study: %s", exc)

    records = []
    for trial in study.trials:
        objective_values = {}
        for metric_name, _ in parameters_to_optimize:
            value = trial.user_attrs.get(f"metric_{metric_name}")
            if value is not None:
                objective_values[metric_name] = float(value)

        metric_statistics = {}
        for statistic in trial.user_attrs.get("statistics", []) or []:
            try:
                metric_statistics[statistic["name"]] = (
                    float(statistic["value"]),
                    float(statistic["lower_confidence_bound"]),
                    float(statistic["upper_confidence_bound"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

        duration = None
        if trial.datetime_start is not None and trial.datetime_complete is not None:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()

        records.append(
            TrialRecord(
                number=trial.number,
                state=trial.state.name
                if isinstance(trial.state, optuna.trial.TrialState)
                else str(trial.state),
                params=dict(trial.params),
                objective_values=objective_values,
                metric_statistics=metric_statistics,
                duration_seconds=duration,
                is_pareto=trial.number in pareto_numbers,
                confidence_interval_level=confidence_interval_level,
            )
        )
    return records


def save_trial_records(records: Sequence[TrialRecord], path: Path) -> Path:
    """Write records as JSON so the plots can be rebuilt without re-running the study."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([record.to_dict() for record in records], indent=2)
    path.write_text(payload, encoding="utf-8")
    return path


def load_trial_records(path: Path) -> List[TrialRecord]:
    """Read records written by :func:`save_trial_records`."""
    return [
        TrialRecord.from_dict(item)
        for item in json.loads(Path(path).read_text(encoding="utf-8"))
    ]


def _completed(records: Sequence[TrialRecord]) -> List[TrialRecord]:
    return [record for record in records if record.state == "COMPLETE"]


def _confidence_suffix(records: Sequence[TrialRecord]) -> str:
    """Render the confidence level as a short parenthetical, e.g. " (95% CI)"."""
    levels = {
        record.confidence_interval_level
        for record in records
        if record.confidence_interval_level is not None
    }
    if len(levels) != 1:
        # No level recorded, or records from runs with different levels: saying
        # nothing beats stating a number the bounds may not have come from.
        return " with confidence intervals"
    return f" ({levels.pop():.0%} CI)"


def _finish(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_front_quality_history(
    history: Sequence[Tuple[int, float]],
    output_path: Path,
    stopped_at_trial: Optional[int] = None,
) -> Optional[Path]:
    """Plot normalized front quality against completed trials.

    This is the convergence evidence: the curve is monotone by construction, so
    a flat tail means later trials added nothing the front did not already have.

    Args:
        history: ``(completed_trials, front_quality)`` pairs from the
            early-stopping callback.
        output_path: Where to write the PNG.
        stopped_at_trial: Trial count at which early stopping fired, marked with
            a vertical line. None if the study spent its whole budget.

    Returns:
        The written path, or None if there was nothing to plot.
    """
    if not history:
        return None

    trials = [point[0] for point in history]
    quality = [point[1] for point in history]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(trials, quality, color="tab:blue", linewidth=2)
    if stopped_at_trial is not None:
        axis.axvline(
            stopped_at_trial,
            color="tab:red",
            linestyle="--",
            label=f"early stop at {stopped_at_trial} trials",
        )
        axis.legend()
    axis.set_xlabel("Completed trials")
    axis.set_ylabel("Pareto front hypervolume (normalized)")
    axis.set_title("Front quality vs trials")
    axis.grid(True, alpha=0.3)
    return _finish(fig, output_path)


def plot_objective_history(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_path: Path,
) -> Optional[Path]:
    """Plot each optimized metric per trial, with its running best.

    The scatter shows how noisy the objective is. If its spread is as wide as
    the gain the running-best line makes, the improvement is not real.
    """
    completed = _completed(records)
    if not completed:
        return None

    names = [name for name, _ in parameters_to_optimize]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 3.2 * len(names)), squeeze=False)
    for axis, (name, direction) in zip(axes[:, 0], parameters_to_optimize):
        points = [
            (r.number, r.objective_values[name])
            for r in completed
            if name in r.objective_values
        ]
        if not points:
            axis.set_visible(False)
            continue
        numbers = [p[0] for p in points]
        values = [p[1] for p in points]
        maximize = getattr(direction, "value", str(direction)) == "maximize"
        running = np.maximum.accumulate(values) if maximize else np.minimum.accumulate(values)

        axis.scatter(numbers, values, s=14, alpha=0.5, color="tab:blue", label="trial")
        axis.plot(numbers, running, color="tab:red", linewidth=2, label="running best")
        axis.set_ylabel(name)
        axis.set_title(f"{name} ({getattr(direction, 'value', direction)})")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    axes[-1, 0].set_xlabel("Trial")
    return _finish(fig, output_path)


def plot_objective_confidence_intervals(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_path: Path,
    top_k: int = 20,
) -> Optional[Path]:
    """Plot the best trials' objectives with their confidence intervals.

    Answers whether the winner actually beat the runners-up. Overlapping
    intervals mean the ranking is within episode-sampling noise, and the fix is
    more episodes per trial rather than more trials.

    The x-axis is ranked by the metric, best on the left, so the trial numbers
    along it are not in order -- and each panel ranks by its own metric, so a
    trial sits in a different place in each.

    Args:
        records: Trial records.
        parameters_to_optimize: The optimized ``(metric_name, direction)`` pairs.
        output_path: Where to write the PNG.
        top_k: How many of the best trials to show per metric. Studies with
            fewer completed trials than this show all of them.
    """
    completed = _completed(records)
    if not completed:
        return None

    names = [name for name, _ in parameters_to_optimize]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 3.2 * len(names)), squeeze=False)
    plotted_any = False
    for axis, (name, direction) in zip(axes[:, 0], parameters_to_optimize):
        with_stats = [r for r in completed if name in r.metric_statistics]
        if not with_stats:
            axis.set_visible(False)
            continue
        maximize = getattr(direction, "value", str(direction)) == "maximize"
        best = sorted(
            with_stats,
            key=lambda r, n=name: r.metric_statistics[n][0],
            reverse=maximize,
        )[:top_k]

        positions = np.arange(len(best))
        values = np.array([r.metric_statistics[name][0] for r in best])
        lower = values - np.array([r.metric_statistics[name][1] for r in best])
        upper = np.array([r.metric_statistics[name][2] for r in best]) - values

        axis.errorbar(
            positions,
            values,
            yerr=np.vstack([np.maximum(lower, 0), np.maximum(upper, 0)]),
            fmt="o",
            color="tab:blue",
            ecolor="tab:gray",
            capsize=3,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([str(r.number) for r in best], rotation=90, fontsize="x-small")
        axis.set_ylabel(name)
        # Each panel ranks by its own metric, so a trial sits at a different
        # position in each -- and "top k" only applies when k actually bit.
        trimmed = f" (top {len(best)})" if len(best) < len(with_stats) else ""
        axis.set_title(f"Ranked by {name}{trimmed}", fontsize="medium")
        axis.grid(True, alpha=0.3)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return None
    axes[-1, 0].set_xlabel("Trial number, best first")
    fig.suptitle(f"Objective values{_confidence_suffix(completed)}")
    return _finish(fig, output_path)


def plot_pareto_front(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_path: Path,
) -> Optional[Path]:
    """Scatter every objective pair, highlighting the front.

    Shows the trade-off the study settled on, which a single best-value curve
    hides. Returns None for single-objective studies, where there is no front.
    """
    completed = _completed(records)
    names = [name for name, _ in parameters_to_optimize]
    if len(names) < 2 or not completed:
        return None

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]][:MAX_OBJECTIVE_PAIRS]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.5), squeeze=False)
    for axis, (x_name, y_name) in zip(axes[0], pairs):
        usable = [
            r for r in completed if x_name in r.objective_values and y_name in r.objective_values
        ]
        axis.scatter(
            [r.objective_values[x_name] for r in usable if not r.is_pareto],
            [r.objective_values[y_name] for r in usable if not r.is_pareto],
            s=16,
            alpha=0.4,
            color="tab:gray",
            label="trial",
        )
        front = [r for r in usable if r.is_pareto]
        axis.scatter(
            [r.objective_values[x_name] for r in front],
            [r.objective_values[y_name] for r in front],
            s=45,
            color="tab:red",
            label="Pareto front",
        )
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.suptitle("Pareto front")
    return _finish(fig, output_path)


def plot_parameter_history(
    records: Sequence[TrialRecord],
    output_path: Path,
) -> Optional[Path]:
    """Plot each searched hyperparameter against trial number.

    This is convergence of the *parameters* rather than the objective: as the
    sampler homes in, the cloud should collapse toward a region. A cloud that
    stays uniform to the last trial means the parameter never mattered, or the
    budget was too small to tell.
    """
    completed = _completed(records)
    if not completed:
        return None

    names = sorted({name for record in completed for name in record.params})
    if not names:
        return None

    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.2 * rows), squeeze=False)
    flat_axes = axes.flatten()
    for axis, name in zip(flat_axes, names):
        usable = [r for r in completed if name in r.params]
        values = [r.params[name] for r in usable]
        numbers = [r.number for r in usable]
        is_pareto = [r.is_pareto for r in usable]

        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            y_values: Sequence[Any] = values
            tick_labels = None
        else:
            # Categorical parameters get a stable integer encoding so they can
            # share the same "value vs trial" plot shape as numerical ones.
            categories = sorted({str(value) for value in values})
            index = {category: position for position, category in enumerate(categories)}
            y_values = [index[str(value)] for value in values]
            tick_labels = categories

        axis.scatter(
            [n for n, pareto in zip(numbers, is_pareto) if not pareto],
            [v for v, pareto in zip(y_values, is_pareto) if not pareto],
            s=14,
            alpha=0.4,
            color="tab:gray",
        )
        axis.scatter(
            [n for n, pareto in zip(numbers, is_pareto) if pareto],
            [v for v, pareto in zip(y_values, is_pareto) if pareto],
            s=36,
            color="tab:red",
            label="Pareto front",
        )
        if tick_labels is not None:
            axis.set_yticks(range(len(tick_labels)))
            axis.set_yticklabels(tick_labels, fontsize="x-small")
        axis.set_title(name)
        axis.set_xlabel("Trial")
        axis.grid(True, alpha=0.3)
    for axis in flat_axes[len(names) :]:
        axis.set_visible(False)
    fig.suptitle("Sampled hyperparameters vs trial")
    return _finish(fig, output_path)


def plot_parameter_slices(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_path: Path,
) -> Optional[Path]:
    """Plot each searched hyperparameter against each optimized metric.

    The companion to :func:`plot_parameter_history`: that one shows *where* the
    search settled, this one shows whether it settled somewhere good. Points are
    shaded by trial number, so a good region that only late trials reach reads
    as the search working rather than as luck.
    """
    completed = _completed(records)
    if not completed:
        return None

    param_names = sorted({name for record in completed for name in record.params})
    metric_names = [name for name, _ in parameters_to_optimize]
    if not param_names or not metric_names:
        return None

    fig, axes = plt.subplots(
        len(metric_names),
        len(param_names),
        figsize=(4.2 * len(param_names), 3.2 * len(metric_names)),
        squeeze=False,
    )
    for row, metric_name in enumerate(metric_names):
        for column, param_name in enumerate(param_names):
            axis = axes[row][column]
            usable = [
                r
                for r in completed
                if param_name in r.params and metric_name in r.objective_values
            ]
            if not usable:
                axis.set_visible(False)
                continue

            values = [r.params[param_name] for r in usable]
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                x_values: Sequence[Any] = values
                tick_labels = None
            else:
                categories = sorted({str(value) for value in values})
                index = {category: position for position, category in enumerate(categories)}
                x_values = [index[str(value)] for value in values]
                tick_labels = categories

            scatter = axis.scatter(
                x_values,
                [r.objective_values[metric_name] for r in usable],
                c=[r.number for r in usable],
                cmap="viridis",
                s=18,
                alpha=0.75,
            )
            if tick_labels is not None:
                axis.set_xticks(range(len(tick_labels)))
                axis.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize="x-small")
            if row == len(metric_names) - 1:
                axis.set_xlabel(param_name)
            if column == 0:
                axis.set_ylabel(metric_name)
            axis.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes, label="Trial", fraction=0.02, pad=0.02)
    fig.suptitle("Hyperparameter value vs objective")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_secondary_metrics(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_path: Path,
) -> Optional[Path]:
    """Plot metrics that were recorded but not optimized, against trial number.

    These show the price of the objective. A study told only to maximize return
    can drive collision rate up for a hundred trials and report success.
    """
    completed = _completed(records)
    if not completed:
        return None

    optimized = {name for name, _ in parameters_to_optimize}
    names = sorted({name for record in completed for name in record.metric_statistics} - optimized)
    if not names:
        return None

    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.2 * rows), squeeze=False)
    flat_axes = axes.flatten()
    for axis, name in zip(flat_axes, names):
        usable = [r for r in completed if name in r.metric_statistics]
        numbers = [r.number for r in usable]
        values = np.array([r.metric_statistics[name][0] for r in usable])
        lower = np.array([r.metric_statistics[name][1] for r in usable])
        upper = np.array([r.metric_statistics[name][2] for r in usable])

        axis.plot(numbers, values, color="tab:blue", linewidth=1, alpha=0.8)
        axis.fill_between(numbers, lower, upper, color="tab:blue", alpha=0.15)
        pareto = [r for r in usable if r.is_pareto]
        if pareto:
            axis.scatter(
                [r.number for r in pareto],
                [r.metric_statistics[name][0] for r in pareto],
                s=30,
                color="tab:red",
                label="Pareto front",
                zorder=3,
            )
            axis.legend(fontsize="small")
        axis.set_title(name)
        axis.set_xlabel("Trial")
        axis.grid(True, alpha=0.3)
    for axis in flat_axes[len(names) :]:
        axis.set_visible(False)
    fig.suptitle("Metrics recorded but not optimized")
    return _finish(fig, output_path)


def plot_trial_durations(
    records: Sequence[TrialRecord],
    output_path: Path,
) -> Optional[Path]:
    """Plot per-trial wall-clock time.

    Catches budget drift: if later trials take longer per decision, the search
    has been comparing compute rather than parameters, which breaks the
    calibration the whole comparison rests on.
    """
    usable = [r for r in _completed(records) if r.duration_seconds is not None]
    if not usable:
        return None

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(
        [r.number for r in usable],
        # Narrowed to float by the filter above, but only for a reader; spell
        # it out so the type checker sees the same thing.
        [float(r.duration_seconds) for r in usable if r.duration_seconds is not None],
        marker="o",
        markersize=3,
        linewidth=1,
        color="tab:purple",
    )
    axis.set_xlabel("Trial")
    axis.set_ylabel("Trial duration (s)")
    axis.set_title("Trial duration vs trial")
    axis.grid(True, alpha=0.3)
    return _finish(fig, output_path)


def plot_parameter_importances(
    study,
    parameters_to_optimize,
    output_path: Path,
) -> Optional[Path]:
    """Plot fANOVA parameter importance, one bar group per optimized metric.

    Needs the live study because Optuna computes importance from its internal
    search space. A near-zero importance usually means the search range was too
    narrow to matter, not that the parameter is unimportant in general.
    """
    import optuna  # pylint: disable=import-outside-toplevel

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return None

    importances: Dict[str, Dict[str, float]] = {}
    for index, (name, _) in enumerate(parameters_to_optimize):
        try:
            importances[name] = optuna.importance.get_param_importances(
                study, target=lambda trial, i=index: trial.values[i]
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # fANOVA needs scikit-learn and at least two distinct parameter
            # values; neither is worth failing a finished study over.
            logger.debug("Parameter importance unavailable for %s: %s", name, exc)
    if not importances:
        return None

    param_names = sorted({param for values in importances.values() for param in values})
    positions = np.arange(len(param_names))
    width = 0.8 / max(1, len(importances))

    fig, axis = plt.subplots(figsize=(max(7, 1.2 * len(param_names)), 4.5))
    for offset, (metric_name, values) in enumerate(importances.items()):
        axis.bar(
            positions + offset * width,
            [values.get(param, 0.0) for param in param_names],
            width=width,
            label=metric_name,
        )
    axis.set_xticks(positions + width * (len(importances) - 1) / 2)
    axis.set_xticklabels(param_names, rotation=30, ha="right")
    axis.set_ylabel("fANOVA importance")
    axis.set_title("Hyperparameter importance per objective")
    axis.legend(fontsize="small")
    axis.grid(True, axis="y", alpha=0.3)
    return _finish(fig, output_path)


def plot_tuning_diagnostics(
    records: Sequence[TrialRecord],
    parameters_to_optimize,
    output_dir: Path,
    front_quality_history: Optional[Sequence[Tuple[int, float]]] = None,
    stopped_at_trial: Optional[int] = None,
    study=None,
) -> List[Path]:
    """Write the full diagnostic set for a tuning study.

    Args:
        records: Trial records, from :func:`extract_trial_records` or
            :func:`load_trial_records`.
        parameters_to_optimize: The optimized ``(metric_name, direction)`` pairs.
        output_dir: Directory to write the PNGs into.
        front_quality_history: Convergence curve from the early-stopping
            callback. Omitted when early stopping was disabled.
        stopped_at_trial: Where early stopping fired, if it did.
        study: The live Optuna study, needed only for parameter importance.

    Returns:
        Paths actually written, skipping plots with no data behind them.
    """
    output_dir = Path(output_dir)
    written: List[Optional[Path]] = [
        plot_front_quality_history(
            front_quality_history or [], output_dir / "front_quality_history.png", stopped_at_trial
        ),
        plot_objective_history(
            records, parameters_to_optimize, output_dir / "objective_history.png"
        ),
        plot_objective_confidence_intervals(
            records, parameters_to_optimize, output_dir / "objective_confidence_intervals.png"
        ),
        plot_pareto_front(records, parameters_to_optimize, output_dir / "pareto_front.png"),
        plot_parameter_history(records, output_dir / "parameter_history.png"),
        plot_parameter_slices(
            records, parameters_to_optimize, output_dir / "parameter_slices.png"
        ),
        plot_secondary_metrics(
            records, parameters_to_optimize, output_dir / "secondary_metrics.png"
        ),
        plot_trial_durations(records, output_dir / "trial_durations.png"),
    ]
    if study is not None:
        written.append(
            plot_parameter_importances(
                study, parameters_to_optimize, output_dir / "parameter_importances.png"
            )
        )
    return [path for path in written if path is not None]
