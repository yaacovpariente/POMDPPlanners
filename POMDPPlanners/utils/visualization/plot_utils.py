"""Shared plotting utilities for visualization modules."""

import logging
from typing import Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend

# Set up logger
logger = logging.getLogger(__name__)


def _safe_histplot(
    data: Sequence[float],
    *,
    max_bins: int = 15,
    color: Union[str, Tuple[float, float, float]] = "skyblue",
    alpha: float = 0.7,
    edgecolor: str = "black",
    linewidth: float = 0.5,
    label: Optional[str] = None,
):
    """Safely render a histogram for possibly degenerate data.

    Handles cases with very small or zero data range by reducing the number of
    bins and/or explicitly setting a padded bin range to avoid numpy/seaborn
    errors like "Too many bins for data range".

    Returns True if a plot was rendered, False otherwise.
    """
    values = np.asarray(list(data), dtype=float)
    if values.size == 0:
        return False

    values = values[np.isfinite(values)]
    if values.size == 0:
        return False

    unique_values = np.unique(values)
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_range = data_max - data_min

    # Determine a safe bin count
    bins = min(max_bins, max(1, int(values.size)))
    # Do not request more bins than unique values for tiny datasets
    bins = min(bins, max(1, int(unique_values.size)))

    binrange = None
    # If the data range is zero or extremely tiny, pad the range and use a single bin
    if not np.isfinite(data_range) or data_range == 0.0 or data_range < 1e-12:
        bins = 1
        pad = 0.5 if data_range == 0.0 else max(1e-6, data_range * 0.5)
        binrange = (data_min - pad, data_max + pad)

    try:
        sns.histplot(
            data=values,
            bins=bins,
            binrange=binrange,
            edgecolor=edgecolor,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _log_or_print(logger: Optional[logging.Logger], message: str, level: str = "warning") -> None:
    """Log message or print if logger is None."""
    if logger:
        if level == "warning":
            logger.warning(message)
        elif level == "info":
            logger.info(message)
    else:
        print(f"Warning: {message}" if level == "warning" else message)
