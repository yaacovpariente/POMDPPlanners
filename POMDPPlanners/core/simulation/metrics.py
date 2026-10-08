# SPDX-License-Identifier: MIT

"""Metric value container and the metric names shared across environments.

Classes:
    MetricValue: A metric point estimate with its confidence interval.
    CommonMetricName: Metric names every environment must spell the same way.

Note:
    The shared vocabulary is kept small on purpose. A shared name implies a
    shared *definition*, so only quantities that mean the same thing in every
    environment belong in :class:`CommonMetricName`: whether the task was
    completed, why the episode ended, how long it was, and plain event counts
    such as collisions. Physical quantities are shared only with their unit in
    the name (``average_speed_mps``): "impact severity" could be a peak force
    (N), a contact impulse (N*s) or a lost kinetic energy (J), and averaging
    those into one column would launder incomparable numbers. Everything else
    stays in each environment's own metrics enum, named after the quantity and
    unit it measures.

    Naming rules for every metric name, shared or not:

    - ``_rate``: fraction of episodes in which something happened, in [0, 1].
    - ``average_``: a per-episode count or value averaged over episodes. Never
      ``avg_``, and never ``total_`` for a per-episode average.
    - ``max_`` / ``min_`` / ``final_``: the worst, best or last value within an
      episode.
    - Units as a suffix where the quantity is physical: ``_m``, ``_mps``.
"""

from enum import Enum
from typing import NamedTuple


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float


class CommonMetricName(str, Enum):
    """Metric names that mean the same thing in every environment.

    Environments take these names from here instead of writing the string, so
    one concept cannot drift into several spellings again. It had: task
    completion alone was ``success_rate``, ``goal_reaching_rate``, ``win_rate``
    and ``exit_success_rate`` depending on the environment. An environment's
    own metrics enum points its member at the shared value, e.g.
    ``TASK_COMPLETION_RATE = CommonMetricName.TASK_COMPLETION_RATE.value``.

    Attributes:
        TASK_COMPLETION_RATE: Fraction of episodes in which the task was
            completed.
        ENDED_BY_GOAL_RATE: Fraction of episodes that ended because the task
            was completed.
        ENDED_BY_FAILURE_RATE: Fraction of episodes that ended in a failure.
        ENDED_BY_TIMEOUT_RATE: Fraction of episodes that ran out of steps.
        AVERAGE_EPISODE_LENGTH: Steps per episode, averaged over episodes.
        COLLISION_RATE: Fraction of episodes with a collision.
        AVERAGE_COLLISIONS: Collisions per episode, averaged over episodes.
        AVERAGE_DANGEROUS_AREA_STEPS: Steps spent in a dangerous area per
            episode, averaged over episodes.
        AVERAGE_DANGEROUS_ENCOUNTERS: Dangerous events of every kind per
            episode, averaged over episodes.
        AVERAGE_NEAR_MISSES: Near-miss events per episode, averaged over
            episodes.
        AVERAGE_SPEED_MPS: Mean speed within an episode in m/s, averaged over
            episodes.
    """

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL_RATE = "ended_by_goal_rate"
    ENDED_BY_FAILURE_RATE = "ended_by_failure_rate"
    ENDED_BY_TIMEOUT_RATE = "ended_by_timeout_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    COLLISION_RATE = "collision_rate"
    AVERAGE_COLLISIONS = "average_collisions"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"
    AVERAGE_DANGEROUS_ENCOUNTERS = "average_dangerous_encounters"
    AVERAGE_NEAR_MISSES = "average_near_misses"
    AVERAGE_SPEED_MPS = "average_speed_mps"
