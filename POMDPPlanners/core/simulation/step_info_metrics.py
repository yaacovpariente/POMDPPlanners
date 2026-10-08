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
    order_and_fill_metrics: Impose a declared name order, filling any gap.
    require_measured_episodes: Reject episodes that ran but were never measured.
    require_non_empty_histories: Reject an empty episode batch.
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

    Note:
        An episode that reported the channel on no step at all contributes
        nothing: it is dropped from the average rather than filled with a
        stand-in. What an unmeasured episode "should" have measured is not a
        property of the metric, so there is no per-spec knob for it. An episode
        that ran but was never measured is a different matter, and is rejected
        upstream by :func:`require_measured_episodes` rather than scored as zero.
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


def unmeasured_episode_index(
    histories: Sequence["History"], specs: Sequence[StepInfoMetric]
) -> Optional[int]:
    """Find the first episode that recorded transitions but no declared channel.

    Such an episode was produced before the per-step channel existed, or by a
    runner that does not call ``step_info``. Callers use this either to refuse it
    (:func:`require_measured_episodes`) or to recompute it: the task manager
    treats a cached episode that fails this check as a cache miss, so a resumed
    run redoes the stale entries instead of scoring them as zero or aborting.

    An episode with no transition steps is exempt. It measured nothing
    legitimately, and an environment serving values cached during its own step
    (a live simulator) has nothing to report for the terminal bookkeeping step
    that such an episode consists of.

    Args:
        histories: The episode histories to check.
        specs: The metric specs whose channels are required. No specs means
            nothing is required, so every history passes.

    Returns:
        The index of the first unmeasured episode, or ``None`` if every episode
        either carries a declared channel or is exempt.
    """
    declared = {spec.channel for spec in specs}
    if not declared:
        return None
    for index, history in enumerate(histories):
        steps = history.history
        if not any(step.action is not None for step in steps):
            continue
        if any(declared & set(step.info or {}) for step in steps):
            continue
        return index
    return None


def require_non_empty_histories(histories: Sequence["History"], environment_name: str) -> None:
    """Reject an empty episode batch instead of scoring it.

    Every metric over no episodes is an average over nothing. Whatever an
    environment returns for it is invented: a zero-valued metric reads exactly
    like a genuine measurement of zero, an omitted metric silently shortens the
    declared name list, and an empty list claims the environment has no metrics
    at all. None of the three is a measurement, and the three disagreed across
    environments before this guard existed.

    Nothing in the simulation pipeline produces an empty batch — episode
    statistics are computed from the same histories and already fail earlier on
    an empty list — so this is a caller error, not a degenerate run.

    Args:
        histories: The episode histories about to be scored.
        environment_name: Name used in the error message.

    Raises:
        ValueError: If ``histories`` is empty.
    """
    if histories:
        return
    raise ValueError(
        f"{environment_name}.compute_metrics received no episode histories. Every "
        "metric would be an average over nothing, and any value reported for it "
        "would be indistinguishable from a genuine measurement. Check that the "
        "simulation produced episodes before computing metrics."
    )


def require_measured_episodes(
    histories: Sequence["History"],
    specs: Sequence[StepInfoMetric],
    environment_name: str,
) -> None:
    """Reject episodes that were never measured, instead of scoring them as zero.

    An environment whose metrics come from the per-step channel reads nothing at
    all out of a history recorded before ``step_info`` existed, or by a runner
    that does not call it. Left alone that is not a missing result but a wrong
    one: every channel is absent, so a rate reads 0.0 and a count reads 0 — "the
    planner never reached the goal" is indistinguishable from "nobody measured".
    Environments pinning a fixed declared name list through
    :func:`order_and_fill_metrics` turn the omission into that zero explicitly.

    The check is per episode, not over the batch: a partially warm cache yields
    some measured episodes beside unmeasured ones, and one measured episode must
    not vouch for the rest — they would silently drop out of every average.

    It also keys on the *declared* channels rather than on ``info`` being
    non-empty, so unrelated per-step bookkeeping cannot vouch for a measurement
    that was never taken. Any one declared channel is enough: an environment
    that declares a channel only when the corresponding sensor is configured
    legitimately reports a subset, and :func:`aggregate_step_info_metrics` omits
    the rest.

    See :func:`unmeasured_episode_index` for exactly which episodes qualify,
    including the exemption for one with no transition steps.

    Args:
        histories: The episode histories about to be scored.
        specs: The metric specs whose channels are required.
        environment_name: Name used in the error message.

    Raises:
        ValueError: If an episode recorded transition steps and not one of its
            steps carries any declared channel.
    """
    index = unmeasured_episode_index(histories, specs)
    if index is None:
        return
    raise ValueError(
        f"{environment_name} derives its metrics from the per-step measurement "
        f"channel, but episode {index} carries none of its channels "
        f"({sorted(spec.channel for spec in specs)}) on any of its "
        f"{len(histories[index].history)} recorded steps. That history was produced "
        "before the channel existed, or by a runner that does not call step_info; "
        "scoring it would report every metric as zero rather than as unmeasured. "
        "Re-run the affected configs, clearing the simulation cache if they were "
        "replayed from it."
    )


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
        zero. An individual episode that reported the channel on no step is
        dropped from that spec's average for the same reason.

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


def order_and_fill_metrics(
    names: Sequence[str],
    metrics: Sequence[MetricValue],
) -> List[MetricValue]:
    """Impose a declared name order on computed metrics, filling any gap.

    Environments that combine spec-driven metrics with hand-written ones need
    both halves emitted in their declared order, not concatenated: the position
    of a metric is observable through
    :func:`~POMDPPlanners.simulations.simulation_statistics.get_metric_names_from_environment_policy_pair`,
    which feeds hyperparameter-tuning objective selection.

    A declared name with no computed metric is filled with a zero-valued entry
    on an unbounded interval, so the produced name list always equals the
    declared one. That matters because
    :func:`aggregate_step_info_metrics` deliberately omits a metric whose channel
    no episode reported, which would otherwise shorten the list silently.

    Note:
        This is deliberately *not* applied by
        :meth:`~POMDPPlanners.core.environment.environment.Environment.compute_metrics`.
        For an environment that declares a channel only when the corresponding
        sensor is configured, omission is meaningful — it distinguishes "never
        measured" from "measured zero" — and filling would erase that. Use this
        only where the declared name list is a fixed contract.

    Args:
        names: The declared metric names, in the order they must be emitted.
        metrics: The computed metrics, in any order. Entries whose name is not in
            ``names`` are dropped; duplicates resolve to the last occurrence.

    Returns:
        One metric per declared name, in ``names`` order.
    """
    by_name = {metric.name: metric for metric in metrics}
    ordered: List[MetricValue] = []
    for name in names:
        metric = by_name.get(name)
        if metric is None:
            metric = MetricValue(
                name=name,
                value=0.0,
                lower_confidence_bound=float(_UNBOUNDED_INTERVAL[0]),
                upper_confidence_bound=float(_UNBOUNDED_INTERVAL[1]),
            )
        ordered.append(metric)
    return ordered
