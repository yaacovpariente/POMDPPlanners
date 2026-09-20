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

    # The bottom margin holds the axis: a scale and a line saying what the bar
    # and the whisker are. Without it a reader cannot tell a mean from a max.
    left, right, top, bottom = 124, 62, 34, 52
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

    # The value axis: a scale under the plot, then a line naming what is drawn.
    axis_y = top + plot_h + 6
    parts.append(
        f'<line x1="{left}" y1="{axis_y:.1f}" x2="{left + plot_w}" y2="{axis_y:.1f}" '
        'class="chart-axis"/>'
    )
    for value, anchor in ((low, "start"), ((low + high) / 2, "middle"), (high, "end")):
        x = to_x(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y:.1f}" x2="{x:.1f}" y2="{axis_y + 4:.1f}" '
            'class="chart-axis"/>'
            f'<text x="{x:.1f}" y="{axis_y + 17:.1f}" class="chart-label" '
            f'text-anchor="{anchor}">{value:.3g}</text>'
        )
    # The metric already titles the chart, so the axis only has to say what the
    # two marks are. Repeating the name here ran the line past the card's edge.
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{axis_y + 33:.1f}" class="chart-label" '
        'text-anchor="middle">bar: value · line: confidence interval</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def _bin_edges(values: Sequence[float], count: int) -> List[float]:
    """Split a range into ``count`` equal bins.

    Args:
        values: Every value that must fall inside the bins.
        count: How many bins.

    Returns:
        ``count + 1`` edges, lowest first.
    """
    low, high = min(values), max(values)
    if low == high:
        # One distinct value still needs a bin with width, or it draws nothing.
        pad = abs(low) * 0.05 or 0.5
        low, high = low - pad, high + pad
    step = (high - low) / count
    return [low + step * i for i in range(count + 1)]


def histogram_svg(
    series: Sequence[Tuple[str, Sequence[float]]],
    value_label: str = "Discounted return",
    count_label: str = "Episodes",
    width: int = 720,
    height: int = 320,
) -> str:
    """Draw each planner's values as a histogram over shared bins.

    The bins are shared across planners on purpose: histograms drawn over
    their own ranges cannot be compared, which is the whole point of putting
    them on one plot. Both axes are named, because a bar whose axis is unnamed
    could be a count, a mean or a maximum, and a reader should not have to
    guess which.

    Args:
        series: One ``(policy_name, values)`` per planner.
        value_label: Name of the binned quantity, for the horizontal axis.
        count_label: Name of the count, for the vertical axis.
        width: Drawing width in user units.
        height: Drawing height in user units.

    Returns:
        An ``<svg>`` element as a string, or an empty string when no planner
        has a value to bin.
    """
    rows = [(name, [float(v) for v in values]) for name, values in series if len(values)]
    if not rows:
        return ""

    pooled = [value for _, values in rows for value in values]
    # Square root of the sample size, which is the usual default, kept between
    # three and twelve so that a handful of episodes still shows shape and a
    # long run does not turn into a comb.
    bins = min(12, max(3, int(len(pooled) ** 0.5 + 0.5)))
    edges = _bin_edges(pooled, bins)

    counts: List[List[int]] = []
    for _, values in rows:
        row = [0] * bins
        for value in values:
            # The last bin is closed at the top, so the maximum lands in it
            # rather than falling off the end.
            index = min(bins - 1, int((value - edges[0]) / (edges[-1] - edges[0]) * bins))
            row[max(0, index)] += 1
        counts.append(row)

    tallest = max(max(row) for row in counts) or 1
    left, right, top, bottom = 62, 20, 30, 64
    legend_h = 22 if len(rows) > 1 else 0
    height += legend_h
    plot_w = width - left - right
    plot_h = height - top - bottom - legend_h

    bin_w = plot_w / bins
    group_w = bin_w * 0.82
    bar_w = group_w / len(rows)

    parts: List[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Histogram of {escape(value_label)} per planner" '
        'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Horizontal rules at whole counts: a histogram's vertical axis counts
    # episodes, so a tick at 1.5 would be meaningless.
    step = max(1, tallest // 5)
    for count in range(0, tallest + 1, step):
        y = top + plot_h - (count / tallest) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'class="chart-axis"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="chart-label" '
            f'text-anchor="end">{count}</text>'
        )

    for row_index, row in enumerate(counts):
        for bin_index, count in enumerate(row):
            if not count:
                continue
            bar_h = (count / tallest) * plot_h
            x = left + bin_w * bin_index + (bin_w - group_w) / 2 + bar_w * row_index
            parts.append(
                f'<rect x="{x:.1f}" y="{top + plot_h - bar_h:.1f}" '
                f'width="{max(1.0, bar_w - 2):.1f}" height="{bar_h:.1f}" '
                f'class="chart-bar series-{row_index % 6}"/>'
            )

    # Every bin edge is labelled when there are few, else every other one, so
    # the axis says what range each bar covers rather than only where it sits.
    stride = 1 if bins <= 6 else 2
    for index in range(0, bins + 1, stride):
        x = left + bin_w * index
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" class="chart-label" '
            f'text-anchor="middle">{edges[index]:.3g}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_h:.1f}" x2="{left + plot_w}" '
        f'y2="{top + plot_h:.1f}" class="chart-axis"/>'
        f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 40:.1f}" '
        f'class="chart-label" text-anchor="middle">{escape(value_label)}</text>'
        f'<text x="16" y="{top + plot_h / 2:.1f}" class="chart-label" '
        f'text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2:.1f})">'
        f"{escape(count_label)}</text>"
    )

    if legend_h:
        x = left
        y = height - 8
        for index, (name, values) in enumerate(rows):
            parts.append(
                f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="11" height="11" '
                f'class="chart-bar series-{index % 6}"/>'
                f'<text x="{x + 16:.1f}" y="{y:.1f}" class="chart-label">'
                f"{escape(name)} ({len(values)})</text>"
            )
            x += 20 + len(name) * 7.2 + 34
    parts.append("</svg>")
    return "".join(parts)
