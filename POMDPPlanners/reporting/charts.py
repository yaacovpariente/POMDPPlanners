# SPDX-License-Identifier: MIT

"""Comparison plots, drawn as inline SVG.

The MLflow UI's comparison view is a bar chart of one metric across runs, with
its confidence interval. This draws the same thing from the metrics a run
already logged, as SVG written straight into the page.

SVG rather than a plotting library because the site must work offline and stay
dependency-light: an inline ``<svg>`` needs no vendored JavaScript, no image
files on disk, and it stays sharp and selectable. It also inherits the page's
colours, so it follows the viewer's theme without a second palette.
"""

from html import escape
from typing import Dict, List, Sequence, Tuple

# Confidence intervals arrive as three sibling metrics next to the value.
CI_LOWER_SUFFIX = "_ci_lower"
CI_UPPER_SUFFIX = "_ci_upper"
CI_WIDTH_SUFFIX = "_ci_width"
_CI_SUFFIXES = (CI_LOWER_SUFFIX, CI_UPPER_SUFFIX, CI_WIDTH_SUFFIX)


def base_metric_names(metrics: Dict[str, float]) -> List[str]:
    """List the metrics that are values rather than interval bounds.

    Args:
        metrics: One policy's metrics, keyed by bare metric name.

    Returns:
        The metric names with the ``_ci_*`` siblings removed, sorted.
    """
    return sorted(name for name in metrics if not name.endswith(_CI_SUFFIXES))


def _nice_bounds(values: Sequence[float]) -> Tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        # A flat series still needs a range, or every bar is zero-height.
        pad = abs(low) * 0.1 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.12
    return low - pad, high + pad


#: One row per planner. A chart is as tall as it has bars, rather than a fixed
#: height divided among them, so two planners do not get a row each the depth
#: of a paragraph.
ROW_HEIGHT = 44


def comparison_svg(
    metric: str,
    series: Sequence[Tuple[str, float, float, float]],
    width: int = 430,
    height: int = 0,
) -> str:
    """Draw one metric across policies as a bar chart with error bars.

    The drawing size matters: an SVG with a ``viewBox`` is scaled to its
    container, and its text with it. Drawn at 640 units in a 340-pixel card,
    every label came out at half the size it asked for and the charts stopped
    being readable. The default width is close to the width a card actually
    gets, so a 13-unit label is about 13 pixels on screen.

    Args:
        metric: Metric name, used as the chart's title.
        series: One ``(policy_name, value, ci_lower, ci_upper)`` per bar. A
            policy with no interval passes its value for both bounds.
        width: Chart width in user units, which is about CSS pixels.
        height: Chart height; 0 means one row per planner.

    Returns:
        An ``<svg>`` element as a string, or an empty string when there is
        nothing to plot.
    """
    if not series:
        return ""

    left, right, top, bottom = 124, 62, 34, 16
    height = height or top + bottom + ROW_HEIGHT * len(series)
    plot_w = width - left - right
    plot_h = height - top - bottom

    spread = [v for _, value, lo, hi in series for v in (value, lo, hi)]
    low, high = _nice_bounds(spread)
    span = high - low

    def to_x(value: float) -> float:
        return left + (value - low) / span * plot_w

    zero_x = to_x(0.0) if low <= 0.0 <= high else left

    row_h = plot_h / len(series)
    bar_h = min(18.0, row_h * 0.45)

    parts: List[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(metric)} by planner" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{left}" y="20" class="chart-title">{escape(metric)}</text>',
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" '
        'class="chart-axis"/>',
    ]

    for index, (name, value, ci_low, ci_high) in enumerate(series):
        cy = top + row_h * index + row_h / 2
        x_value = to_x(value)
        bar_left = min(zero_x, x_value)
        bar_width = max(1.0, abs(x_value - zero_x))
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{cy - bar_h / 2:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_h:.1f}" class="chart-bar"/>'
        )
        if ci_high > ci_low:
            x_lo, x_hi = to_x(ci_low), to_x(ci_high)
            parts.append(
                f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" '
                'class="chart-ci"/>'
                f'<line x1="{x_lo:.1f}" y1="{cy - 5:.1f}" x2="{x_lo:.1f}" y2="{cy + 5:.1f}" '
                'class="chart-ci"/>'
                f'<line x1="{x_hi:.1f}" y1="{cy - 5:.1f}" x2="{x_hi:.1f}" y2="{cy + 5:.1f}" '
                'class="chart-ci"/>'
            )
        parts.append(
            f'<text x="{left - 10}" y="{cy + 4:.1f}" class="chart-label" '
            f'text-anchor="end">{escape(name)}</text>'
        )
        # The value sits in the right margin rather than at the end of its bar:
        # against the bar it lands on top of the interval whisker, which runs
        # past the bar whenever the estimate is uncertain — which is the case
        # the chart is drawn for.
        parts.append(
            f'<text x="{width - 6}" y="{cy + 4:.1f}" class="chart-value" '
            f'text-anchor="end">{value:.3g}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def returns_svg(
    series: Sequence[Tuple[str, Sequence[float]]],
    width: int = 700,
) -> str:
    """Draw each planner's episode returns as a strip, with its mean marked.

    A histogram of a handful of episodes is mostly empty bins, and the run
    writes one as a PNG anyway. What a reader wants from this plot is where
    the episodes fell and how far apart they are, so every episode is a mark
    on the planner's own line and the mean sits under them as a rule. It stays
    honest at three episodes and still reads at fifty.

    Args:
        series: One ``(policy_name, returns)`` per planner, in display order.
        width: Drawing width in user units, about CSS pixels.

    Returns:
        An ``<svg>`` element as a string, or an empty string when no planner
        has a return to plot.
    """
    rows = [(name, [float(v) for v in values]) for name, values in series if len(values)]
    if not rows:
        return ""

    left, right, top, bottom = 150, 30, 30, 42
    row_h = 52
    height = top + bottom + row_h * len(rows)
    plot_w = width - left - right

    spread = [value for _, values in rows for value in values]
    low, high = _nice_bounds(spread)
    span = high - low or 1.0

    def to_x(value: float) -> float:
        return left + (value - low) / span * plot_w

    parts: List[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Discounted return of each episode, by planner" '
        'xmlns="http://www.w3.org/2000/svg">'
    ]

    # A tick at each end and at the middle: enough to read the scale without
    # drawing an axis a strip plot does not need.
    for value in (low, (low + high) / 2, high):
        x = to_x(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_h * len(rows)}" '
            'class="chart-axis"/>'
            f'<text x="{x:.1f}" y="{height - 14}" class="chart-label" '
            f'text-anchor="middle">{value:.3g}</text>'
        )

    for index, (name, values) in enumerate(rows):
        cy = top + row_h * index + row_h / 2
        mean = sum(values) / len(values)
        parts.append(
            f'<line x1="{left}" y1="{cy:.1f}" x2="{left + plot_w}" y2="{cy:.1f}" '
            'class="chart-strip"/>'
            f'<text x="{left - 12}" y="{cy + 4:.1f}" class="chart-label" '
            f'text-anchor="end">{escape(name)}</text>'
        )
        if len(values) > 1:
            parts.append(
                f'<line x1="{to_x(min(values)):.1f}" y1="{cy:.1f}" '
                f'x2="{to_x(max(values)):.1f}" y2="{cy:.1f}" class="chart-ci"/>'
            )
        for value in values:
            parts.append(
                f'<circle cx="{to_x(value):.1f}" cy="{cy:.1f}" r="5" class="chart-dot"/>'
            )
        parts.append(
            f'<line x1="{to_x(mean):.1f}" y1="{cy - 13:.1f}" x2="{to_x(mean):.1f}" '
            f'y2="{cy + 13:.1f}" class="chart-mean"/>'
            f'<text x="{to_x(mean):.1f}" y="{cy - 18:.1f}" class="chart-value" '
            f'text-anchor="middle">{mean:.3g}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
