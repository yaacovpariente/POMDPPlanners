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


def comparison_svg(
    metric: str,
    series: Sequence[Tuple[str, float, float, float]],
    width: int = 640,
    height: int = 260,
) -> str:
    """Draw one metric across policies as a bar chart with error bars.

    Args:
        metric: Metric name, used as the chart's title.
        series: One ``(policy_name, value, ci_lower, ci_upper)`` per bar. A
            policy with no interval passes its value for both bounds.
        width: Chart width in CSS pixels.
        height: Chart height in CSS pixels.

    Returns:
        An ``<svg>`` element as a string, or an empty string when there is
        nothing to plot.
    """
    if not series:
        return ""

    left, right, top, bottom = 132, 16, 34, 28
    plot_w = width - left - right
    plot_h = height - top - bottom

    spread = [v for _, value, lo, hi in series for v in (value, lo, hi)]
    low, high = _nice_bounds(spread)
    span = high - low

    def to_x(value: float) -> float:
        return left + (value - low) / span * plot_w

    zero_x = to_x(0.0) if low <= 0.0 <= high else left

    row_h = plot_h / len(series)
    bar_h = min(26.0, row_h * 0.55)

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
        label_x = x_value + (6 if x_value >= zero_x else -6)
        anchor = "start" if x_value >= zero_x else "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{cy + 4:.1f}" class="chart-value" '
            f'text-anchor="{anchor}">{value:.3g}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
