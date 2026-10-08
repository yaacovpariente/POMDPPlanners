# SPDX-License-Identifier: MIT

"""A tuning study's diagnostic charts, drawn as inline SVG from its records.

The optimizer also draws these as PNGs with matplotlib, and they stay in the
run as artifacts for reading offline. The site draws its own from
``tuning/trial_records.json`` and ``tuning/study_summary.json`` instead, for
the reasons :mod:`POMDPPlanners.reporting.charts` gives: inline SVG follows the
page's theme, stays sharp, and needs nothing vendored. And unlike a PNG, a
point can say which trial it is: every mark carries a ``<title>``, which the
browser shows on hover.

What is not drawn: parameter importances. Optuna computes them with fANOVA
over the live study object, which the run does not keep and the records
cannot rebuild; the PNG it drew is still among the run's artifacts.
"""

import math
from dataclasses import dataclass, field
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from POMDPPlanners.reporting import tuning

# Drawn at about the width a card gets, for the reason charts.comparison_svg
# gives: a viewBox far wider than its card shrinks the text with it.
WIDTH = 430
HEIGHT = 270
_LEFT, _RIGHT, _TOP, _BOTTOM = 66, 16, 30, 58


@dataclass(frozen=True)
class Point:
    """One mark on a chart.

    Attributes:
        x: Horizontal position, in data units (a category index for a
            categorical axis).
        y: Vertical position, in data units.
        tip: What hovering the mark says.
        kind: ``"trial"``, ``"pareto"`` or ``"best"``; picks the mark's style.
        low: Lower end of an error bar, or ``None``.
        high: Upper end of an error bar, or ``None``.
    """

    x: float
    y: float
    tip: str
    kind: str = "trial"
    low: Optional[float] = None
    high: Optional[float] = None


@dataclass(frozen=True)
class Line:
    """A polyline through data points, such as a best-so-far curve."""

    points: Sequence[Tuple[float, float]]
    css: str = "chart-line"
    step: bool = False
    label: str = ""


@dataclass(frozen=True)
class Chart:
    """Everything one chart draws.

    Attributes:
        title: Chart title.
        x_label: Horizontal axis name.
        y_label: Vertical axis name.
        points: The marks.
        lines: Curves drawn under the marks.
        x_categories: Names of the categories when the horizontal axis is
            categorical; ``None`` for a numeric axis.
        x_marker: A vertical rule at this x, such as where early stopping fired.
        x_marker_label: The rule's label.
        legend: Extra legend entries as ``(css_class, label)``.
        x_integer: The horizontal axis counts something -- trials, ranks --
            so its ticks fall on whole numbers.
        x_span: Data range the horizontal axis must cover, such as a
            parameter's search range, so a value at its edge looks it.
        y_span: The same for the vertical axis.
        y_categories: Category names when the vertical axis is categorical.
    """

    title: str
    x_label: str
    y_label: str
    points: Sequence[Point] = ()
    lines: Sequence[Line] = ()
    x_categories: Optional[Sequence[str]] = None
    x_marker: Optional[float] = None
    x_marker_label: str = ""
    legend: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    x_integer: bool = False
    x_span: Optional[Tuple[float, float]] = None
    y_span: Optional[Tuple[float, float]] = None
    y_categories: Optional[Sequence[str]] = None


def _bounds(values: Sequence[float]) -> Tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _ticks(low: float, high: float, count: int = 5) -> List[float]:
    step = (high - low) / (count - 1)
    return [low + step * i for i in range(count)]


def _integer_ticks(low: float, high: float, count: int = 6) -> List[float]:
    first, last = int(-(-low // 1)), int(high // 1)
    if last < first:
        return []
    step = max(1, -(-(last - first) // (count - 1)))
    return [float(v) for v in range(first, last + 1, step)]


def _tick_labels(values: Sequence[float]) -> Tuple[List[str], str]:
    """Labels for one axis's ticks, short and all different.

    Very small or very large values are divided by a power of ten that the
    axis title then names, because ``6.1e-05`` does not fit in the margin and
    a column of them is hard to compare. Significant digits are added until
    no two ticks read the same, so a narrow range does not print one value
    twice.

    Args:
        values: The tick values.

    Returns:
        The labels, and a suffix such as ``" (×1e-5)"`` for the axis title.
    """
    largest = max((abs(v) for v in values), default=0.0)
    exponent = 0
    if largest and (largest < 1e-2 or largest >= 1e5):
        exponent = int(math.floor(math.log10(largest)))
    scale = 10.0**exponent
    labels: List[str] = []
    for digits in range(3, 8):
        labels = [f"{v / scale:.{digits}g}" for v in values]
        if len(set(labels)) == len(labels):
            break
    return labels, (f" (×1e{exponent})" if exponent else "")


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def chart_svg(chart: Chart, width: int = WIDTH, height: int = HEIGHT) -> str:
    """Draw one chart.

    Args:
        chart: What to draw.
        width: Drawing width in user units.
        height: Drawing height in user units.

    Returns:
        An ``<svg>`` element, or an empty string when there is nothing to draw.
    """
    if not chart.points and not any(line.points for line in chart.lines):
        return ""
    plot_w = width - _LEFT - _RIGHT
    plot_h = height - _TOP - _BOTTOM

    xs = [p.x for p in chart.points] + [x for line in chart.lines for x, _ in line.points]
    ys = (
        [p.y for p in chart.points]
        + [v for p in chart.points for v in (p.low, p.high) if v is not None]
        + [y for line in chart.lines for _, y in line.points]
    )
    if chart.x_marker is not None:
        xs.append(chart.x_marker)
    if chart.x_span is not None:
        xs.extend(chart.x_span)
    if chart.y_span is not None:
        ys.extend(chart.y_span)
    if chart.x_categories is not None:
        x_low, x_high = -0.5, len(chart.x_categories) - 0.5
    elif chart.x_integer:
        # Half a step of margin, so the first and last trial are not drawn on
        # the axis lines.
        x_low, x_high = min(xs) - 0.5, max(xs) + 0.5
    else:
        x_low, x_high = _bounds(xs)
    if chart.y_categories is not None:
        y_low, y_high = -0.5, len(chart.y_categories) - 0.5
    else:
        y_low, y_high = _bounds(ys)

    def to_x(value: float) -> float:
        return _LEFT + (value - x_low) / (x_high - x_low) * plot_w

    def to_y(value: float) -> float:
        return _TOP + plot_h - (value - y_low) / (y_high - y_low) * plot_h

    parts: List[str] = [
        f'<svg class="chart tuning-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(chart.title)}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{_LEFT}" y="18" class="chart-title">{escape(chart.title)}</text>',
    ]

    # Horizontal rules with their values: the reader compares heights here.
    y_suffix = ""
    if chart.y_categories is not None:
        y_ticks = [(float(i), name) for i, name in enumerate(chart.y_categories)]
    else:
        values = _ticks(y_low, y_high)
        labels, y_suffix = _tick_labels(values)
        y_ticks = list(zip(values, labels))
    for value, text in y_ticks:
        y = to_y(value)
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + plot_w}" y2="{y:.1f}" '
            'class="chart-grid"/>'
            f'<text x="{_LEFT - 6}" y="{y + 4:.1f}" class="chart-label" '
            f'text-anchor="end">{escape(text)}</text>'
        )
    axis_y = _TOP + plot_h
    x_suffix = ""
    parts.append(
        f'<line x1="{_LEFT}" y1="{axis_y}" x2="{_LEFT + plot_w}" y2="{axis_y}" '
        'class="chart-axis"/>'
    )
    if chart.x_categories is not None:
        for index, name in enumerate(chart.x_categories):
            parts.append(
                f'<text x="{to_x(index):.1f}" y="{axis_y + 16}" class="chart-label" '
                f'text-anchor="middle">{escape(name)}</text>'
            )
    else:
        x_ticks = _integer_ticks(x_low, x_high) if chart.x_integer else _ticks(x_low, x_high)
        x_labels, x_suffix = _tick_labels(x_ticks)
        for value, text in zip(x_ticks, x_labels):
            parts.append(
                f'<text x="{to_x(value):.1f}" y="{axis_y + 16}" class="chart-label" '
                f'text-anchor="middle">{text}</text>'
            )
    parts.append(
        f'<text x="{_LEFT + plot_w / 2:.1f}" y="{axis_y + 34}" class="chart-label" '
        f'text-anchor="middle">{escape(chart.x_label + x_suffix)}</text>'
        f'<text x="12" y="{_TOP + plot_h / 2:.1f}" class="chart-label" text-anchor="middle" '
        f'transform="rotate(-90 12 {_TOP + plot_h / 2:.1f})">'
        f"{escape(chart.y_label + y_suffix)}</text>"
    )

    if chart.x_marker is not None:
        x = to_x(chart.x_marker)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_TOP}" x2="{x:.1f}" y2="{axis_y}" class="chart-marker"/>'
            f'<text x="{x - 4:.1f}" y="{_TOP + 12}" class="chart-label" text-anchor="end">'
            f"{escape(chart.x_marker_label)}</text>"
        )

    for line in chart.lines:
        if not line.points:
            continue
        coords: List[str] = []
        previous_y: Optional[float] = None
        for x_value, y_value in line.points:
            x, y = to_x(x_value), to_y(y_value)
            if line.step and previous_y is not None:
                coords.append(f"{x:.1f},{previous_y:.1f}")
            coords.append(f"{x:.1f},{y:.1f}")
            previous_y = y
        parts.append(f'<polyline points="{" ".join(coords)}" class="{escape(line.css)}"/>')

    # Best and Pareto marks are drawn last, so a crowd of trials never hides them.
    order = {"trial": 0, "pareto": 1, "best": 2}
    for point in sorted(chart.points, key=lambda p: order.get(p.kind, 0)):
        x, y = to_x(point.x), to_y(point.y)
        whisker = ""
        if point.low is not None and point.high is not None and point.high > point.low:
            whisker = (
                f'<line x1="{x:.1f}" y1="{to_y(point.low):.1f}" x2="{x:.1f}" '
                f'y2="{to_y(point.high):.1f}" class="chart-ci"/>'
            )
        parts.append(
            f'<g class="pt pt-{escape(point.kind)}"><title>{escape(point.tip)}</title>'
            f'{whisker}<circle cx="{x:.1f}" cy="{y:.1f}" r="4"/></g>'
        )

    legend = _legend(chart)
    if legend:
        x = _LEFT
        y = height - 6
        for css, label in legend:
            if css.startswith("line:"):
                # A curve is keyed by a stroke, not a dot, so it reads as a line.
                parts.append(
                    f'<line x1="{x:.1f}" y1="{y - 4:.1f}" x2="{x + 12:.1f}" y2="{y - 4:.1f}" '
                    f'class="legend-line {escape(css[len("line:"):])}"/>'
                    f'<text x="{x + 16:.1f}" y="{y:.1f}" class="chart-label">{escape(label)}</text>'
                )
                x += 26 + len(label) * 6.6
                continue
            parts.append(
                f'<circle cx="{x + 4:.1f}" cy="{y - 4:.1f}" r="4" class="legend-dot {css}"/>'
                f'<text x="{x + 12:.1f}" y="{y:.1f}" class="chart-label">{escape(label)}</text>'
            )
            x += 22 + len(label) * 6.6
    parts.append("</svg>")
    return "".join(parts)


def _legend(chart: Chart) -> List[Tuple[str, str]]:
    kinds = {p.kind for p in chart.points}
    entries: List[Tuple[str, str]] = []
    if "trial" in kinds:
        entries.append(("pt-trial", "trial"))
    if "pareto" in kinds:
        entries.append(("pt-pareto", "Pareto"))
    if "best" in kinds:
        entries.append(("pt-best", "chosen"))
    entries += [(f"line:{line.css}", line.label) for line in chart.lines if line.label]
    return entries + list(chart.legend)


# -- the study's charts --------------------------------------------------------


def _completed(study: tuning.TuningStudy) -> List[tuning.Trial]:
    return [t for t in study.trials if t.state.upper() == "COMPLETE"]


def _kind(study: tuning.TuningStudy, trial: tuning.Trial) -> str:
    if trial.number == study.best_trial_number:
        return "best"
    if trial.is_pareto or trial.number in study.pareto_trial_numbers:
        return "pareto"
    return "trial"


def _tip(trial: tuning.Trial, headline: str) -> str:
    params = ", ".join(f"{k}={_value_text(v)}" for k, v in trial.params.items())
    return f"Trial #{trial.number} · {headline}" + (f" · {params}" if params else "")


def _value_text(value: object) -> str:
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _direction(study: tuning.TuningStudy, name: str) -> str:
    return next((str(d) for n, d in study.objectives if n == name), "maximize")


def _objective_names(study: tuning.TuningStudy) -> List[str]:
    names = [str(n) for n, _ in study.objectives]
    return names or sorted({k for t in study.trials for k in t.objective_values})


def _stop_trial_number(study: tuning.TuningStudy) -> Optional[float]:
    """The trial number at which early stopping fired.

    ``stopped_at_trial`` counts completed trials, while the history charts
    are drawn against trial numbers; a failed or pruned trial before the stop
    makes the two differ, so the stop is placed at the number of the trial
    that completed the count.
    """
    if not study.stopped_at_trial:
        return None
    completed = sorted(t.number for t in _completed(study))
    if len(completed) >= study.stopped_at_trial:
        return float(completed[study.stopped_at_trial - 1])
    return float(study.stopped_at_trial - 1)


def objective_history(study: tuning.TuningStudy) -> List[str]:
    """Each objective against trial number, with the best value so far.

    The best-so-far curve going flat is what a converged search looks like;
    one still climbing at the last trial says the budget ran out first.
    """
    charts = []
    trials = _completed(study)
    for name in _objective_names(study):
        maximize = _direction(study, name) != "minimize"
        points, best_line = [], []
        best: Optional[float] = None
        for trial in trials:
            value = trial.objective_values.get(name)
            if value is None:
                continue
            best = value if best is None else (max(best, value) if maximize else min(best, value))
            best_line.append((float(trial.number), best))
            points.append(
                Point(
                    trial.number, value, _tip(trial, f"{name} {_fmt(value)}"), _kind(study, trial)
                )
            )
        charts.append(
            chart_svg(
                Chart(
                    title=f"{name} ({'maximized' if maximize else 'minimized'})",
                    x_label="Trial",
                    x_integer=True,
                    y_label=name,
                    points=points,
                    lines=[Line(best_line, "chart-line", step=True, label="best so far")],
                    x_marker=_stop_trial_number(study),
                    x_marker_label="early stop" if study.stopped_at_trial else "",
                )
            )
        )
    return [c for c in charts if c]


def objective_confidence_intervals(study: tuning.TuningStudy) -> List[str]:
    """Each objective's trials ranked best first, with their intervals.

    Whether the top trial really beat the next is a question of overlapping
    intervals, not of which mean is larger.
    """
    charts = []
    trials = _completed(study)
    for name in _objective_names(study):
        maximize = _direction(study, name) != "minimize"
        rows = []
        for trial in trials:
            stats = trial.metric_statistics.get(name)
            value = trial.objective_values.get(name, stats[0] if stats else None)
            if value is None:
                continue
            low, high = (stats[1], stats[2]) if stats and len(stats) == 3 else (None, None)
            rows.append((trial, value, low, high))
        rows.sort(key=lambda r: r[1], reverse=maximize)
        points = [
            Point(
                rank + 1,
                value,
                _tip(
                    trial,
                    f"rank {rank + 1} · {name} {_fmt(value)}"
                    + (
                        f" [{_fmt(low)}, {_fmt(high)}]"
                        if low is not None and high is not None
                        else ""
                    ),
                ),
                _kind(study, trial),
                low,
                high,
            )
            for rank, (trial, value, low, high) in enumerate(rows)
        ]
        charts.append(
            chart_svg(
                Chart(
                    title=f"{name}, ranked, with intervals",
                    x_label="Rank (best first)",
                    y_label=name,
                    points=points,
                    x_integer=True,
                )
            )
        )
    return [c for c in charts if c]


def pareto_front(study: tuning.TuningStudy) -> str:
    """The two objectives against each other, with the Pareto front joined.

    Drawn only for a study with exactly two objectives; with one there is no
    front, and with three a plane cannot show it.
    """
    names = _objective_names(study)
    if len(names) != 2:
        return ""
    first, second = names
    points, front = [], []
    for trial in _completed(study):
        a, b = trial.objective_values.get(first), trial.objective_values.get(second)
        if a is None or b is None:
            continue
        kind = _kind(study, trial)
        points.append(Point(a, b, _tip(trial, f"{first} {_fmt(a)}, {second} {_fmt(b)}"), kind))
        if kind != "trial":
            front.append((a, b))
    return chart_svg(
        Chart(
            title="Pareto front",
            x_label=first,
            y_label=second,
            points=points,
            lines=[Line(sorted(front), "chart-line chart-front", label="front")],
        )
    )


def front_quality(study: tuning.TuningStudy) -> str:
    """The front-quality curve early stopping watched, and where it stopped."""
    if not study.front_quality_history:
        return ""
    curve = [(float(n), float(q)) for n, q in study.front_quality_history]
    points = [Point(n, q, f"after {int(n)} completed trials · quality {_fmt(q)}") for n, q in curve]
    return chart_svg(
        Chart(
            title="Pareto-front quality (early stopping)",
            x_label="Completed trials",
            y_label="normalized hypervolume",
            points=points,
            x_integer=True,
            lines=[Line(curve, "chart-line")],
            x_marker=float(study.stopped_at_trial) if study.stopped_at_trial else None,
            x_marker_label="stopped" if study.stopped_at_trial else "",
        )
    )


def _parameter_axis(
    parameter: tuning.SearchParameter, values: Sequence[object]
) -> Tuple[Optional[List[str]], Dict[str, int]]:
    """A categorical axis for a parameter that needs one, else ``None``."""
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if parameter.choices is None and numeric:
        return None, {}
    names = [str(c) for c in (parameter.choices or [])]
    for value in values:
        if str(value) not in names:
            names.append(str(value))
    return names, {name: index for index, name in enumerate(names)}


def _span(parameter: tuning.SearchParameter) -> Optional[Tuple[float, float]]:
    if parameter.low is None or parameter.high is None:
        return None
    try:
        return float(parameter.low), float(parameter.high)
    except (TypeError, ValueError):
        return None


def _parameters(study: tuning.TuningStudy) -> List[tuning.SearchParameter]:
    if study.parameters:
        return list(study.parameters)
    names = sorted({k for t in study.trials for k in t.params})
    return [tuning.SearchParameter(name=n) for n in names]


def parameter_history(study: tuning.TuningStudy) -> List[str]:
    """Each parameter's sampled value against trial number.

    A sampler that has found something narrows its draws over time; one still
    spraying the whole range late in the study has not.
    """
    charts = []
    trials = study.trials
    for parameter in _parameters(study):
        values = [t.params[parameter.name] for t in trials if parameter.name in t.params]
        if not values:
            continue
        categories, index = _parameter_axis(parameter, values)
        points = []
        for trial in trials:
            if parameter.name not in trial.params:
                continue
            value = trial.params[parameter.name]
            y = index[str(value)] if categories is not None else float(value)
            points.append(
                Point(
                    trial.number,
                    y,
                    f"Trial #{trial.number} · {parameter.name}={_value_text(value)}",
                    _kind(study, trial),
                )
            )
        svg = chart_svg(
            Chart(
                title=parameter.name,
                x_label="Trial",
                x_integer=True,
                y_label=parameter.name,
                y_categories=categories,
                points=points,
                y_span=_span(parameter) if categories is None else None,
            )
        )
        if svg:
            charts.append(svg)
    return charts


def parameter_slices(study: tuning.TuningStudy) -> List[str]:
    """Each objective against each parameter, one chart per pair.

    The shape of a slice says whether the range was right: a peak inside it
    is a range that brackets the optimum, a slope running into its edge is one
    that stops short.
    """
    charts = []
    trials = _completed(study)
    for name in _objective_names(study):
        for parameter in _parameters(study):
            values = [t.params[parameter.name] for t in trials if parameter.name in t.params]
            if not values:
                continue
            categories, index = _parameter_axis(parameter, values)
            points = []
            for trial in trials:
                value = trial.params.get(parameter.name)
                objective = trial.objective_values.get(name)
                if value is None or objective is None:
                    continue
                x = index[str(value)] if categories is not None else float(value)
                points.append(
                    Point(
                        x, objective, _tip(trial, f"{name} {_fmt(objective)}"), _kind(study, trial)
                    )
                )
            svg = chart_svg(
                Chart(
                    title=f"{name} by {parameter.name}",
                    x_label=parameter.name,
                    y_label=name,
                    points=points,
                    x_categories=categories,
                    x_span=_span(parameter) if categories is None else None,
                )
            )
            if svg:
                charts.append(svg)
    return charts


def secondary_metrics(study: tuning.TuningStudy) -> List[str]:
    """Every metric the trials recorded but the study did not optimize.

    A study told to maximize return will trade away anything it was not told
    about; these are where that shows.
    """
    objectives = set(_objective_names(study))
    trials = _completed(study)
    names = sorted({k for t in trials for k in t.metric_statistics} - objectives)
    charts = []
    for name in names:
        points = []
        for trial in trials:
            stats = trial.metric_statistics.get(name)
            if not stats:
                continue
            value = stats[0]
            low, high = (stats[1], stats[2]) if len(stats) == 3 else (None, None)
            points.append(
                Point(
                    trial.number,
                    value,
                    f"Trial #{trial.number} · {name} {_fmt(value)}"
                    + (
                        f" [{_fmt(low)}, {_fmt(high)}]"
                        if low is not None and high is not None
                        else ""
                    ),
                    _kind(study, trial),
                    low,
                    high,
                )
            )
        svg = chart_svg(
            Chart(title=name, x_label="Trial", y_label=name, points=points, x_integer=True)
        )
        if svg:
            charts.append(svg)
    return charts


def trial_durations(study: tuning.TuningStudy) -> str:
    """How long each trial took, which is where a study's budget actually goes."""
    points = [
        Point(
            t.number,
            t.duration_seconds,
            f"Trial #{t.number} · {t.duration_seconds:.1f} s · {t.state.title()}",
            _kind(study, t),
        )
        for t in study.trials
        if t.duration_seconds is not None
    ]
    return chart_svg(
        Chart(
            title="Trial duration",
            x_label="Trial",
            y_label="seconds",
            points=points,
            x_integer=True,
        )
    )
