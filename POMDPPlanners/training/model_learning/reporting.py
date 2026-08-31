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

Functions:
    round_table_markdown: Per-round table of cost, return and model quality.
    round_table_csv: The same rows, for a spreadsheet.
    curve_table_markdown: Best round per method and seed.
    write_reports: Write both tables into a directory.
"""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.training.model_learning.control_evaluation import (
    LearningCurve,
    aggregate_curves,
    best_point,
)

#: Drift above this means the model's horizon error exceeds its claimed spread.
DRIFT_WARNING_THRESHOLD = 1.0

_COLUMNS = (
    "round",
    "transitions",
    "exploration",
    "planner",
    "mean_return",
    "standard_error",
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
        "| Round | Transitions | Explore / Planner | Mean return | Held-out LL | Drift ratio | Best holdout epoch | Model |",
        "|---|---|---|---|---|---|---|---|",
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
            f"{_number(row['held_out_log_likelihood'], 1)} | {drift} | "
            f"{_number(row['best_holdout_epoch'])} | {row['model'] or '--'} |"
        )
    lines += [
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
        "| Method | Seed | Best round | Transitions | Mean return |",
        "|---|---|---|---|---|",
    ]
    for curve in scored:
        best = best_point(curve)
        if best is None:
            continue
        lines.append(
            f"| {curve.method} | {curve.seed} | {best.round_index} | "
            f"{best.cumulative_transitions} | {_number(best.mean_return)} ± "
            f"{_number(best.standard_error)} |"
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
        "The spread is across seeds, not across the episodes inside one: only the "
        "first says whether two methods actually differed.",
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
    return written
