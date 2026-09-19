# SPDX-License-Identifier: MIT

"""The ship's two sensors: what each can see, what it reports, and how likely that is.

The camera gives a chicken's *column* offset from the ship and nothing about its
row; the radar gives its *row* distance and whether it is dropping, and nothing
about its column. One alone pins a chicken to a whole column or a whole row;
together they pin it to a cell. Splitting the reading this way is what makes the
belief interesting: a chicken seen by one sensor only is genuinely half-located,
and a chicken seen by neither is still constrained, because silence inside a
sensor's reach is evidence that it is somewhere else.

Every integer reading is drawn from a rounded Gaussian,

.. math::

    G_\\sigma(k; \\mu) = \\Phi\\!\\big((k + 1/2 - \\mu) / \\sigma\\big)
                       - \\Phi\\!\\big((k - 1/2 - \\mu) / \\sigma\\big),

which is exactly the law of ``round(mu + sigma * xi)`` for a standard normal
``xi``. It is **not** truncated at the grid edge, so its masses sum to one over
all of ``Z`` without renormalising and a reading may name a column that does not
exist. Truncating would make the normaliser depend on ``mu``, which is the
hidden quantity being inferred, and would therefore change every likelihood
ratio in the belief update for no gain in realism -- a sensor that cannot report
off-grid is a sensor that has already solved part of the problem.

``sigma = 0`` collapses the law to a point mass on ``mu``. That is the noiseless
preset, and it is handled as its own branch rather than as a limit because the
Gaussian expression divides by ``sigma``.

Functions:
    rounded_normal_pmf: The integer noise law above.
    sample_rounded_normal: One draw from it.
    camera_sees: Whether a chicken is inside the camera cone.
    radar_sees: Whether a chicken is inside the radar disc.
"""

from typing import Union

import numpy as np

# pylint: disable-next=no-name-in-module  # ndtr lives in SciPy's C extension
from scipy.special import ndtr


def _standard_normal_cdf(values: np.ndarray) -> np.ndarray:
    """Standard normal CDF, evaluated elementwise.

    ``scipy.special.ndtr`` rather than an ``erf`` written out by hand: this runs
    once per chicken per particle per planning step, and ``ndtr`` is both the
    faster route and the more accurate one in the far tail, where the hand-built
    version loses the digits that decide whether an unlikely reading is unlikely
    or impossible.

    Args:
        values: Points to evaluate at.

    Returns:
        ``Phi(values)``, the same shape.
    """
    return ndtr(values)


def rounded_normal_pmf(
    readings: Union[float, np.ndarray], mean: Union[float, np.ndarray], std: float
) -> np.ndarray:
    """Probability that a rounded Gaussian reading lands on each integer.

    Args:
        readings: Integer reading(s) ``k``.
        mean: True value(s) ``mu``.
        std: Noise standard deviation ``sigma``. Zero gives a point mass.

    Returns:
        ``G_sigma(readings; mean)``, broadcast over the inputs.
    """
    readings = np.asarray(readings, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if std <= 0.0:
        return np.asarray(readings == mean, dtype=np.float64)
    upper = (readings + 0.5 - mean) / std
    lower = (readings - 0.5 - mean) / std
    # Clipped at zero rather than left to float error: the two CDFs are
    # subtracted, and for a reading many standard deviations away the
    # difference can come out as a tiny negative, whose logarithm is a NaN that
    # then poisons a particle's whole weight.
    return np.maximum(_standard_normal_cdf(upper) - _standard_normal_cdf(lower), 0.0)


def sample_rounded_normal(mean: float, std: float) -> int:
    """Draw one integer reading from :func:`rounded_normal_pmf`.

    Draws from the global ``np.random`` stream, which is what makes a seeded
    episode reproduce.

    Args:
        mean: True value ``mu``.
        std: Noise standard deviation. Zero returns ``round(mean)``.

    Returns:
        The integer reading.
    """
    if std <= 0.0:
        return int(round(float(mean)))
    return int(round(float(mean) + float(std) * float(np.random.normal())))


def camera_sees(column_offset: np.ndarray, row_distance: np.ndarray, slope: float) -> np.ndarray:
    """Whether each chicken lies inside the camera's upward cone.

    The cone opens from the ship with half-slope ``slope``: a chicken ``dy``
    rows up is visible while it is no more than ``slope * dy`` columns to either
    side. At ``dy = 0`` the cone is one cell wide, which is the ship's own cell.

    Args:
        column_offset: Signed column offsets ``dx``.
        row_distance: Row distances ``dy``, non-negative.
        slope: Cone half-slope ``alpha``.

    Returns:
        Boolean array, ``True`` where the chicken is inside the cone.
    """
    return np.abs(column_offset) <= float(slope) * row_distance


def radar_sees(column_offset: np.ndarray, row_distance: np.ndarray, radius: float) -> np.ndarray:
    """Whether each chicken lies inside the radar's disc of radius ``rho``.

    Args:
        column_offset: Signed column offsets ``dx``.
        row_distance: Row distances ``dy``.
        radius: Radar radius ``rho``, in cells.

    Returns:
        Boolean array, ``True`` where the chicken is within range.
    """
    return column_offset**2 + row_distance**2 <= float(radius) ** 2
