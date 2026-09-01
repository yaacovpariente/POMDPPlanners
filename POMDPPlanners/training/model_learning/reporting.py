# SPDX-License-Identifier: MIT

"""The tables a person reads a fitting run from.

JSON is what a plot is rebuilt from; it is not what anyone reads to decide
whether a model is worth planning with. That decision needs four numbers side by
side per round -- what the round cost, what the planner scored with it, whether
the fit improved, and whether the model's confidence is honest -- and reading
them out of a nested record is work that gets skipped.

The tables are derived from the same records the plots are, so a table and a
figure never disagree.

Two columns here are judgements rather than measurements.

**The best round is marked, not the last.** The guarantee is on the best model of
the sequence, and the sequence is not monotone, so a table that reads bottom-up
reports the wrong model.

**The drift ratio is flagged above one.** Above one the model's error over a
planning horizon is outside the spread it claims, which is the failure that makes
a risk-sensitive planner confidently wrong. A table without the flag leaves the
reader to remember the threshold.

These are text, not files. Where they land is the tracker's business, and it
puts them in one place -- the run's own artifacts -- rather than mirroring them
into a second tree that then has to be kept in step.

A rounds table belongs to one run, and the method table to the study, so the two
are separate functions: a rounds table published at study level is a single
run's numbers wearing the study's name, which is how the wrong method gets
reported.

Functions:
    metrics_table_markdown: The run reduced to the numbers a decision needs.
    metrics_table_csv: The same numbers, for a spreadsheet.
    round_table_markdown: Per-round table of cost, return and model quality.
    round_table_csv: The same rows, for a spreadsheet.
    curve_table_markdown: Best round per method and seed.
    write_reports: Write both tables into a directory.
"""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.training.model_learning.control_evaluation import (
    LearningCurve,
    aggregate_curves,
    best_point,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval

#: Drift above this means the model's horizon error exceeds its claimed spread.
DRIFT_WARNING_THRESHOLD = 1.0

#: Confidence level for every interval reported here -- the repo's t-interval,
#: the same one the simulation statistics and the paper's tables use, so a
#: model-learning number and a planner number can sit in one table.
CONFIDENCE_LEVEL = 0.95

_COLUMNS = (
    "round",
    "transitions",
    "exploration",
    "planner",
    "mean_return",
    "standard_error",
    "return_ci_low",
    "return_ci_high",
    "held_out_log_likelihood",
    "horizon_drift_ratio",
    "best_holdout_epoch",
    "model",
)


def _field(entry: Any, name: str, default: Any = None) -> Any:
    """Read a field from a round result or from the tracker's dictionary form."""
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _rows(rounds: Sequence[Any]) -> List[Dict[str, Any]]:
    """Flatten each round into the columns the tables share."""
    rows: List[Dict[str, Any]] = []
    for position, entry in enumerate(rounds, start=1):
        diagnostics = _field(entry, "diagnostics", {}) or {}
        sources = _field(entry, "source_counts", {}) or {}
        training = _field(entry, "training_metrics", {}) or {}
        control = _field(entry, "control")
        if control is not None and not isinstance(control, dict):
            control = control.to_dict()
        returns = list(control["returns"]) if control else []
        holdout = list(training.get("holdout_nll", []))
        rows.append(
            {
                "round": int(_field(entry, "round_index", position) or position),
                "transitions": int(_field(entry, "dataset_size", 0) or 0),
                "exploration": int(sources.get("exploration", 0)),
                "planner": int(sources.get("planner", 0)),
                "mean_return": float(np.mean(returns)) if returns else float("nan"),
                "standard_error": (
                    float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
                    if len(returns) > 1
                    else float("nan")
                ),
                "return_ci_low": _interval(returns)[0],
                "return_ci_high": _interval(returns)[1],
                "held_out_log_likelihood": float(
                    diagnostics.get("held_out_log_likelihood", float("nan"))
                ),
                "horizon_drift_ratio": float(
                    diagnostics.get("horizon_drift_ratio", float("nan"))
                ),
                # Where the holdout loss bottomed out: past it the epochs bought
                # training loss with generalization.
                "best_holdout_epoch": (
                    int(np.argmin(holdout)) + 1 if holdout else None
                ),
                # The tracker's records name the saved file; a live round result
                # has the model itself, and its fingerprint is what ties the row
                # to a set of parameters either way.
                "model": _field(entry, "model_artifact") or _short_fingerprint(entry),
            }
        )
    return rows


def _short_fingerprint(entry: Any) -> str:
    """First characters of the round model's fingerprint, or an empty string."""
    fingerprint = _field(entry, "model_fingerprint")
    if fingerprint is None:
        model = _field(entry, "model")
        fingerprint = getattr(model, "fingerprint", None)
    return str(fingerprint)[:10] if fingerprint else ""


def _interval(values: Sequence[float]) -> Tuple[float, float]:
    """The repo's t-interval for a mean, or ``(nan, nan)`` when it is undefined.

    Below two episodes there is no interval, and reporting the point estimate as
    though it were one is the way a single lucky episode becomes a result.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    low, high = confidence_interval(list(values), confidence=CONFIDENCE_LEVEL)
    return float(low), float(high)


def _range(low: float, high: float, digits: int = 3) -> str:
    """Render an interval, or an em dash when there was not enough data for one."""
    if not (np.isfinite(low) and np.isfinite(high)):
        return "--"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _number(value: Any, digits: int = 3) -> str:
    """Render a number for a table, or an em dash when it was never measured."""
    if value is None:
        return "--"
    if isinstance(value, float) and not np.isfinite(value):
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def round_table_markdown(rounds: Sequence[Any], title: str = "Rounds") -> str:
    """Per-round table of what the round cost and what it bought.

    Args:
        rounds: Round results, or the dictionaries written to ``rounds.json``.
        title: Heading above the table.

    Returns:
        A Markdown document. The best round by mean return is marked, and a
        drift ratio above one is flagged.
    """
    rows = _rows(rounds)
    if not rows:
        return f"# {title}\n\nNo rounds recorded.\n"

    scored = [row for row in rows if np.isfinite(row["mean_return"])]
    best_round = max(scored, key=lambda row: row["mean_return"])["round"] if scored else None

    lines = [
        f"# {title}",
        "",
        f"| Round | Transitions | Explore / Planner | Mean return | {int(CONFIDENCE_LEVEL * 100)}% CI | "
        "Held-out LL | Drift ratio | Best holdout epoch | Model |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = " **(best)**" if row["round"] == best_round else ""
        error = _number(row["standard_error"])
        drift = _number(row["horizon_drift_ratio"])
        if np.isfinite(row["horizon_drift_ratio"]) and (
            row["horizon_drift_ratio"] > DRIFT_WARNING_THRESHOLD
        ):
            drift += " ⚠"
        lines.append(
            f"| {row['round']}{mark} | {row['transitions']} | "
            f"{row['exploration']} / {row['planner']} | "
            f"{_number(row['mean_return'])} ± {error} | "
            f"{_range(row['return_ci_low'], row['return_ci_high'])} | "
            f"{_number(row['held_out_log_likelihood'], 1)} | {drift} | "
            f"{_number(row['best_holdout_epoch'])} | {row['model'] or '--'} |"
        )
    lines += [
        "",
        f"Intervals are the repo's {int(CONFIDENCE_LEVEL * 100)}% t-intervals across the "
        "round's evaluation episodes -- the same ones the planner tables use. A round with "
        "fewer than two episodes has no interval and shows a dash.",
        "",
        f"Drift ratio above {DRIFT_WARNING_THRESHOLD:.0f} (⚠) means the model's error over the "
        "planning horizon is larger than the spread it predicts -- it is confident and wrong, "
        "which is the failure a risk-sensitive planner cannot absorb.",
        "",
        "The best round is the one to report: the guarantee covers the best model of the "
        "sequence, and the sequence is not monotone.",
        "",
    ]
    return "\n".join(lines)


def round_table_csv(rounds: Sequence[Any]) -> str:
    """The same rows as :func:`round_table_markdown`, as CSV.

    Args:
        rounds: Round results, or the dictionaries written to ``rounds.json``.

    Returns:
        CSV text with a header row.
    """
    buffer = io.StringIO()
    writer: csv.DictWriter = csv.DictWriter(buffer, fieldnames=list(_COLUMNS))
    writer.writeheader()
    for row in _rows(rounds):
        writer.writerow(row)
    return buffer.getvalue()


def curve_table_markdown(curves: Sequence[LearningCurve]) -> str:
    """Best round per method and seed, and each method's mean over seeds.

    Args:
        curves: Curves across methods and seeds.

    Returns:
        A Markdown document, empty-handed if no curve had an evaluated round.
    """
    scored = [curve for curve in curves if curve.points]
    if not scored:
        return "# Methods\n\nNo evaluated rounds.\n"

    lines = [
        "# Methods",
        "",
        f"| Method | Seed | Best round | Transitions | Mean return | {int(CONFIDENCE_LEVEL * 100)}% CI |",
        "|---|---|---|---|---|---|",
    ]
    for curve in scored:
        best = best_point(curve)
        if best is None:
            continue
        low, high = _interval(best.returns)
        lines.append(
            f"| {curve.method} | {curve.seed} | {best.round_index} | "
            f"{best.cumulative_transitions} | {_number(best.mean_return)} ± "
            f"{_number(best.standard_error)} | {_range(low, high)} |"
        )

    grouped: Dict[str, List[LearningCurve]] = {}
    for curve in scored:
        grouped.setdefault(curve.method, []).append(curve)
    lines += [
        "",
        "## Final round, averaged over seeds",
        "",
        "| Method | Seeds | Transitions | Mean return | SE across seeds |",
        "|---|---|---|---|---|",
    ]
    for method, method_curves in sorted(grouped.items()):
        aggregated = aggregate_curves(method_curves)
        lines.append(
            f"| {method} | {aggregated.num_seeds} | "
            f"{_number(aggregated.cumulative_transitions[-1], 0)} | "
            f"{_number(aggregated.mean_returns[-1])} | "
            f"{_number(aggregated.standard_errors[-1])} |"
        )
    lines += [
        "",
        f"Intervals are the repo's {int(CONFIDENCE_LEVEL * 100)}% t-intervals. Per method "
        "and seed they are across that run's evaluation episodes; in the table above they "
        "are across seeds. Only the second answers whether two methods differed -- a "
        "method can be very consistent within one repetition and swing between them.",
        "",
    ]
    return "\n".join(lines)


def write_reports(
    rounds: Sequence[Any],
    output_dir: Path,
    curves: Optional[Sequence[LearningCurve]] = None,
) -> Dict[str, Path]:
    """Write the readable summary of a run beside its JSON.

    Args:
        rounds: Round results, or the dictionaries written to ``rounds.json``.
        output_dir: Directory to write into; created if missing.
        curves: Control curves across methods and seeds. ``None`` writes the
            per-round table only.

    Returns:
        Report name to written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    summary = round_table_markdown(rounds)
    if curves:
        summary = f"{summary}\n{curve_table_markdown(curves)}"
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    written["summary"] = summary_path

    csv_path = output_dir / "rounds.csv"
    csv_path.write_text(round_table_csv(rounds), encoding="utf-8")
    written["rounds_csv"] = csv_path

    curve = curves[0] if curves and len(curves) == 1 else None
    metrics_path = output_dir / "metrics.md"
    metrics_path.write_text(metrics_table_markdown(rounds, curve), encoding="utf-8")
    written["metrics"] = metrics_path

    metrics_csv = output_dir / "metrics.csv"
    metrics_csv.write_text(metrics_table_csv(rounds, curve), encoding="utf-8")
    written["metrics_csv"] = metrics_csv
    return written


def run_metrics(rounds: Sequence[Any], curve: Optional[LearningCurve] = None) -> Dict[str, Any]:
    """The whole run reduced to the numbers a decision is made on.

    The per-round table says what happened; this says what it came to. Both are
    kept because a single row cannot show a curve that rose and then collapsed,
    and a table of rounds does not answer "so which model, and is it any good".

    Args:
        rounds: The run's rounds, as results or as the tracker's dictionaries.
        curve: The run's control curve, if it was evaluated.

    Returns:
        Metric name to value.
    """
    rows = _rows(rounds)
    scored = [row for row in rows if np.isfinite(row["mean_return"])]
    best_row = max(scored, key=lambda row: row["mean_return"]) if scored else None
    last = rows[-1] if rows else None
    drifts = [row["horizon_drift_ratio"] for row in rows if np.isfinite(row["horizon_drift_ratio"])]
    epochs = [row["best_holdout_epoch"] for row in rows if row["best_holdout_epoch"]]

    metrics: Dict[str, Any] = {
        "rounds": len(rows),
        "transitions_total": last["transitions"] if last else 0,
        "transitions_exploration": last["exploration"] if last else 0,
        "transitions_planner": last["planner"] if last else 0,
        "chosen_round": best_row["round"] if best_row else None,
        "chosen_return": best_row["mean_return"] if best_row else float("nan"),
        "chosen_return_standard_error": best_row["standard_error"] if best_row else float("nan"),
        "chosen_return_ci_low": best_row["return_ci_low"] if best_row else float("nan"),
        "chosen_return_ci_high": best_row["return_ci_high"] if best_row else float("nan"),
        "final_round_return": last["mean_return"] if last else float("nan"),
        "final_round_ci_low": last["return_ci_low"] if last else float("nan"),
        "final_round_ci_high": last["return_ci_high"] if last else float("nan"),
        # The gap the loop is judged on: a chosen round far above the last one
        # means the sequence went backwards, which a final-round report hides.
        "gain_over_final_round": (
            best_row["mean_return"] - last["mean_return"]
            if best_row and last and np.isfinite(last["mean_return"])
            else float("nan")
        ),
        "final_held_out_log_likelihood": last["held_out_log_likelihood"] if last else float("nan"),
        "max_horizon_drift_ratio": max(drifts) if drifts else float("nan"),
        # Above one at any round means the model was overconfident somewhere in
        # the sequence, which is the failure that does not announce itself.
        "rounds_overconfident": sum(1 for value in drifts if value > DRIFT_WARNING_THRESHOLD),
        "median_best_holdout_epoch": float(np.median(epochs)) if epochs else float("nan"),
    }
    if curve is not None:
        metrics["evaluation_episodes_per_round"] = (
            len(curve.points[0].returns) if curve.points else 0
        )
    return metrics


def metrics_table_markdown(
    rounds: Sequence[Any],
    curve: Optional[LearningCurve] = None,
    title: str = "Metrics",
) -> str:
    """The run's headline metrics as a Markdown table.

    Args:
        rounds: The run's rounds.
        curve: The run's control curve, if it was evaluated.
        title: Heading above the table.

    Returns:
        A Markdown document.
    """
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---|"]
    for name, value in run_metrics(rounds, curve).items():
        lines.append(f"| {name.replace('_', ' ')} | {_number(value)} |")
    lines += [
        "",
        "`chosen round` is the round to load -- `models/chosen`. `gain over final "
        "round` is how much reporting the last round instead would have cost.",
        "",
        "`rounds overconfident` counts rounds whose horizon drift exceeded the "
        "spread the model claimed. Any at all is a reason not to trust the model "
        "in a risk-sensitive planner, however good the return looks.",
        "",
    ]
    return "\n".join(lines)


def metrics_table_csv(rounds: Sequence[Any], curve: Optional[LearningCurve] = None) -> str:
    """The same metrics as one CSV row.

    Args:
        rounds: The run's rounds.
        curve: The run's control curve, if it was evaluated.

    Returns:
        CSV text with a header row.
    """
    metrics = run_metrics(rounds, curve)
    buffer = io.StringIO()
    writer: csv.DictWriter = csv.DictWriter(buffer, fieldnames=list(metrics))
    writer.writeheader()
    writer.writerow(metrics)
    return buffer.getvalue()
