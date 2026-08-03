# SPDX-License-Identifier: MIT

"""Metric value container.

Classes:
    MetricValue: A metric point estimate with its confidence interval.

Note:
    There is deliberately no shared cross-environment vocabulary of channel or
    metric names here. A shared name implies a shared *definition*, and for
    physical quantities that implication is false: "impact severity" could
    reasonably be a peak force (N), a contact impulse (N*s), a lost kinetic
    energy (J) or a collision count, and averaging those into one column would
    launder incomparable numbers. Each environment therefore names its own
    channels and metrics, ideally after the quantity and unit it actually
    measures, and still gets the shared aggregation from
    :mod:`POMDPPlanners.core.simulation.step_info_metrics` for free.
"""

from typing import NamedTuple


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float
