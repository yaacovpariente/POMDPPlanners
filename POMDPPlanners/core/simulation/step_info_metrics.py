# SPDX-License-Identifier: MIT

"""Shared aggregation of per-step measurements into metrics with confidence intervals.

Environments report auxiliary per-step measurements through
:meth:`~POMDPPlanners.core.environment.environment.Environment.step_info`, which
the episode loop stores on
:attr:`~POMDPPlanners.core.simulation.history.StepData.info`. This module turns
those raw channels into :class:`~POMDPPlanners.core.simulation.metrics.MetricValue`
entries: reduce a channel within each episode, then average across episodes and
attach a confidence interval.

The reduction is the only part that differs between metrics, so it is declared
rather than reimplemented. Without this, every environment hand-rolls the same
counting / list-accumulation / CI-fallback code — which is how the existing
environments each ended up with a bespoke ``compute_metrics`` body.

The input is a list of per-episode lists of info mappings rather than a list of
``History`` objects, because not every runner produces a ``History``: the
vectorized VOPP episode runner returns its own result type, and both must feed
the same aggregator.

Classes:
    EpisodeReduction: Named within-episode reductions over a channel's values.
    StepInfoMetric: Declarative spec mapping one channel to one metric.

Functions:
    aggregate_step_info_metrics: Reduce declared specs into MetricValues.
    extract_episode_step_infos: Pull per-step info mappings out of histories.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.simulation.metrics import MetricValue
from POMDPPlanners.utils.statistics_utils import confidence_interval

if TYPE_CHECKING:
    from POMDPPlanners.core.simulation.history import History

# Episode counts below this make a t-interval undefined; mirror the unbounded
# interval the existing environment metrics report in that case.
_MIN_EPISODES_FOR_CONFIDENCE_INTERVAL = 2
_UNBOUNDED_INTERVAL = (-np.inf, np.inf)


class EpisodeReduction(Enum):
    """How a channel's per-step values collapse to one number per episode.

    Attributes:
        ANY: 1.0 if any step reported a non-zero value, else 0.0. Across
            episodes this yields a rate — the natural shape for "did it ever
            happen", e.g. reaching a goal.
        ALL: 1.0 only if *every* step reported a non-zero value. The correct
            shape for a "never failed" predicate such as a legged robot staying
            upright: under ANY a robot that falls at the last step would still
            count as a success because earlier steps were fine.
        MAX: The largest value seen. The natural shape for severity, where the
            worst moment characterizes the episode rather than the average.
        SUM: The total over the episode, for counting occurrences.
        MEAN: The average over the episode's steps.
        LAST: The final reported value, for a terminal condition.
    """

    ANY = "any"
    ALL = "all"
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"
    LAST = "last"


def _reduce_episode(reduction: EpisodeReduction, values: List[float]) -> float:
    if reduction is EpisodeReduction.ANY:
        return float(any(value != 0.0 for value in values))
    if reduction is EpisodeReduction.ALL:
        return float(all(value != 0.0 for value in values))
    if reduction is EpisodeReduction.MAX:
        return float(max(values))
    if reduction is EpisodeReduction.SUM:
        return float(sum(values))
    if reduction is EpisodeReduction.MEAN:
        return float(np.mean(values))
    return float(values[-1])


@dataclass(frozen=True)
class StepInfoMetric:
    """Declares one metric derived from one per-step channel.

    Attributes:
        name: The metric name, as it will appear in results and tuning configs.
            Name it after what is measured, including the unit where the
            quantity is physical, so two environments cannot silently report
            different quantities under one name.
        channel: The ``step_info`` key to read.
        per_episode: How the channel's values collapse within one episode.
        scale: Multiplied into the per-episode value before averaging, for unit
            conversion. Defaults to 1.0.
        default: Value contributed by a step that reported no such channel.
            ``None`` (the default) means such steps are skipped entirely. Use
            0.0 only when a missing channel genuinely means zero — treating
            "not measured" as "did not happen" silently biases the metric.
    """

    name: str
    channel: str
    per_episode: EpisodeReduction
    scale: float = 1.0
    default: Optional[float] = None


def extract_episode_step_infos(histories: Sequence["History"]) -> List[List[Dict[str, float]]]:
    """Pull the per-step info mappings out of episode histories.

    Args:
        histories: Episode histories recorded by the standard episode runner.

    Returns:
        One list of info mappings per episode. Steps that carry no info (such as
        the terminal bookkeeping step) contribute an empty mapping.
    """
    return [[step.info or {} for step in history.history] for history in histories]


def _episode_values(
    episode: Sequence[Dict[str, float]], spec: StepInfoMetric
) -> Optional[List[float]]:
    values: List[float] = []
    for step_info in episode:
        if spec.channel in step_info:
            values.append(float(step_info[spec.channel]))
        elif spec.default is not None:
            values.append(spec.default)
    return values or None


def _metric_from_episode_values(spec: StepInfoMetric, per_episode: List[float]) -> MetricValue:
    mean_value = float(np.mean(per_episode))
    if len(per_episode) >= _MIN_EPISODES_FOR_CONFIDENCE_INTERVAL:
        lower, upper = confidence_interval(data=per_episode, confidence=0.95)
    else:
        lower, upper = _UNBOUNDED_INTERVAL
    return MetricValue(
        name=spec.name,
        value=mean_value,
        lower_confidence_bound=float(lower),
        upper_confidence_bound=float(upper),
    )


def aggregate_step_info_metrics(
    episodes: Sequence[Sequence[Dict[str, float]]],
    specs: Sequence[StepInfoMetric],
) -> List[MetricValue]:
    """Reduce per-step channels into metrics with confidence intervals.

    Each spec is applied independently: its channel is reduced within every
    episode that reported it, and those per-episode numbers are averaged across
    episodes with a 95% t-interval.

    Args:
        episodes: One sequence of per-step info mappings per episode.
        specs: The metrics to compute.

    Returns:
        One :class:`~POMDPPlanners.core.simulation.metrics.MetricValue` per spec
        whose channel was reported by at least one episode, in spec order. A spec
        whose channel never appears is omitted rather than reported as zero, so a
        measurement that was never taken is never mistaken for a measurement of
        zero.

    Example:
        Task completion rate over three episodes::

            from POMDPPlanners.core.simulation.step_info_metrics import (
                EpisodeReduction,
                StepInfoMetric,
                aggregate_step_info_metrics,
            )

            episodes = [
                [{"success": 0.0}, {"success": 1.0}],
                [{"success": 0.0}],
                [{"success": 0.0}, {"success": 0.0}],
            ]
            spec = StepInfoMetric(
                name="task_completion_rate",
                channel="success",
                per_episode=EpisodeReduction.ANY,
            )
            metrics = aggregate_step_info_metrics(episodes, [spec])
            round(metrics[0].value, 4)  # 0.3333
    """
    metrics: List[MetricValue] = []
    for spec in specs:
        per_episode: List[float] = []
        for episode in episodes:
            values = _episode_values(episode, spec)
            if values is not None:
                per_episode.append(_reduce_episode(spec.per_episode, values) * spec.scale)
        if per_episode:
            metrics.append(_metric_from_episode_values(spec, per_episode))
    return metrics
