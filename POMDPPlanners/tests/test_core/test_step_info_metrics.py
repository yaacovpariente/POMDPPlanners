# SPDX-License-Identifier: MIT

"""Tests for the shared per-step-channel metric aggregator.

Covers :mod:`POMDPPlanners.core.simulation.step_info_metrics` and the default
:meth:`~POMDPPlanners.core.environment.environment.Environment.compute_metrics`
implementation that is driven by it.
"""

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
    aggregate_step_info_metrics,
    extract_episode_step_infos,
    order_and_fill_metrics,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import sanity_pinned_kwargs
from POMDPPlanners.tests.test_utils.history_builders import build_test_history

# Plain literals: channel and metric names are environment-local by design.
_SUCCESS = "success"
_IMPACT = "contact_impulse_ns"


def _belief() -> WeightedParticleBelief:
    return WeightedParticleBelief(particles=["a", "b"], log_weights=np.array([0.0, -0.1]))


def _history(infos: List[Dict[str, float]]) -> History:
    steps = [
        StepData(
            state="s",
            action="a",
            next_state="s2",
            observation="o",
            reward=0.0,
            belief=_belief(),
            info=info,
        )
        for info in infos
    ]
    return build_test_history(steps=steps)


class _SpecSanity(SanityPOMDP):
    """Sanity-env variant driven entirely by declarative metric specs.

    SanityPOMDP overrides neither ``get_metric_names`` nor ``compute_metrics``,
    so the spec-driven defaults are exercised rather than shadowed.
    """

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        del state, next_state
        return {_SUCCESS: float(bool(action))}

    def get_metric_specs(self) -> List[StepInfoMetric]:
        return [
            StepInfoMetric(
                name="task_completion_rate",
                channel=_SUCCESS,
                per_episode=EpisodeReduction.ANY,
            )
        ]


class TestEpisodeReductions:
    """Reduction semantics within a single episode."""

    @pytest.mark.parametrize(
        "reduction,expected",
        [
            (EpisodeReduction.ANY, 1.0),
            (EpisodeReduction.ALL, 0.0),
            (EpisodeReduction.MAX, 4.0),
            (EpisodeReduction.SUM, 6.0),
            (EpisodeReduction.MEAN, 2.0),
            (EpisodeReduction.LAST, 4.0),
        ],
    )
    def test_reduction_collapses_episode_values(
        self, reduction: EpisodeReduction, expected: float
    ) -> None:
        """Test each within-episode reduction over a known value sequence.

        Purpose: Validates that every declared reduction collapses a channel's
            per-step values to the documented single number

        Given: One episode whose channel reports the values 0, 2, 4
        When: The channel is aggregated under each reduction
        Then: The metric value equals the reduction applied to those values

        Test type: unit
        """
        episodes = [[{_IMPACT: 0.0}, {_IMPACT: 2.0}, {_IMPACT: 4.0}]]
        spec = StepInfoMetric(name="m", channel=_IMPACT, per_episode=reduction)

        metrics = aggregate_step_info_metrics(episodes, [spec])

        assert len(metrics) == 1
        assert metrics[0].value == expected

    def test_any_reduction_across_episodes_yields_a_rate(self) -> None:
        """Test that ANY over episodes produces a completion rate.

        Purpose: Validates the shape used for task completion: "did it ever
            happen" per episode, averaged into a rate

        Given: Three episodes, exactly one of which reports a success
        When: The success channel is aggregated with the ANY reduction
        Then: The metric value is 1/3

        Test type: unit
        """
        episodes = [
            [{_SUCCESS: 0.0}, {_SUCCESS: 1.0}],
            [{_SUCCESS: 0.0}],
            [{_SUCCESS: 0.0}, {_SUCCESS: 0.0}],
        ]
        spec = StepInfoMetric(
            name="task_completion_rate",
            channel=_SUCCESS,
            per_episode=EpisodeReduction.ANY,
        )

        metrics = aggregate_step_info_metrics(episodes, [spec])

        assert math.isclose(metrics[0].value, 1.0 / 3.0)

    def test_all_reduction_requires_every_step(self) -> None:
        """Test that ALL distinguishes "never failed" from "succeeded once".

        Purpose: Validates the reduction a no-failure predicate needs. Under ANY,
            an episode where a legged robot falls on the final step would still
            score as a success because the earlier steps were upright

        Given: One episode upright throughout and one that fails on its last step
        When: The success channel is aggregated with ALL and with ANY
        Then: ALL scores the failing episode 0 (rate 0.5) while ANY scores it 1
            (rate 1.0)

        Test type: unit
        """
        episodes = [
            [{_SUCCESS: 1.0}, {_SUCCESS: 1.0}],
            [{_SUCCESS: 1.0}, {_SUCCESS: 0.0}],
        ]
        strict = StepInfoMetric(name="m", channel=_SUCCESS, per_episode=EpisodeReduction.ALL)
        lenient = StepInfoMetric(name="m", channel=_SUCCESS, per_episode=EpisodeReduction.ANY)

        assert aggregate_step_info_metrics(episodes, [strict])[0].value == 0.5
        assert aggregate_step_info_metrics(episodes, [lenient])[0].value == 1.0

    def test_scale_converts_units_before_averaging(self) -> None:
        """Test that the scale factor is applied to each per-episode value.

        Purpose: Validates unit conversion (e.g. force to impulse) without
            requiring the environment to pre-scale its raw channel

        Given: One episode whose peak impact is 4.0 and a spec with scale 0.5
        When: The channel is aggregated
        Then: The metric value is 2.0

        Test type: unit
        """
        spec = StepInfoMetric(
            name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MAX, scale=0.5
        )

        metrics = aggregate_step_info_metrics([[{_IMPACT: 4.0}]], [spec])

        assert metrics[0].value == 2.0


class TestEmptyEpisodes:
    """What an episode that reported nothing at all contributes."""

    def test_episode_reporting_nothing_is_excluded_by_default(self) -> None:
        """Test that an episode with no values does not enter the average.

        Purpose: Validates the default, which keeps an unmeasured episode from
            being silently counted as a zero

        Given: Two episodes, one reporting the channel and one reporting nothing
        When: The spec leaves empty_episode_value unset
        Then: The mean is taken over the reporting episode alone

        Test type: unit
        """
        spec = StepInfoMetric(name="rate", channel=_SUCCESS, per_episode=EpisodeReduction.ANY)
        metrics = aggregate_step_info_metrics([[{_SUCCESS: 1.0}], [{}]], [spec])

        assert metrics[0].value == 1.0

    def test_empty_episode_value_contributes_to_the_average(self) -> None:
        """Test that a declared empty-episode value is counted.

        Purpose: Validates the hook that lets an environment say what an episode
            measuring nothing means. A "never failed" predicate is vacuously
            true over no steps, so such an episode is a success, not an absence

        Given: Two episodes, one reporting 1.0 and one reporting nothing
        When: The spec declares empty_episode_value=0.0
        Then: The mean is taken over both, giving 0.5

        Test type: unit
        """
        spec = StepInfoMetric(
            name="rate",
            channel=_SUCCESS,
            per_episode=EpisodeReduction.ANY,
            empty_episode_value=0.0,
        )
        metrics = aggregate_step_info_metrics([[{_SUCCESS: 1.0}], [{}]], [spec])

        assert metrics[0].value == 0.5

    def test_empty_episode_value_alone_still_produces_the_metric(self) -> None:
        """Test that a metric survives when every episode measured nothing.

        Purpose: Validates that declaring an empty-episode value also prevents
            the metric being dropped, which is what distinguishes "every episode
            answered zero" from "nothing was ever measured"

        Given: Two episodes that report no channels at all
        When: The spec declares empty_episode_value=1.0
        Then: The metric is produced with value 1.0

        Test type: unit
        """
        spec = StepInfoMetric(
            name="alive",
            channel=_SUCCESS,
            per_episode=EpisodeReduction.ALL,
            empty_episode_value=1.0,
        )
        metrics = aggregate_step_info_metrics([[{}], [{}]], [spec])

        assert [(m.name, m.value) for m in metrics] == [("alive", 1.0)]


class TestOrderAndFill:
    """Imposing a declared name order on a partially computed metric list."""

    def test_missing_name_is_filled_and_order_follows_the_declaration(self) -> None:
        """Test that gaps are filled and ordering comes from the declared names.

        Purpose: Validates the helper that keeps a fixed declared metric list
            intact when the aggregator omitted one of its entries

        Given: Metrics computed out of order with one declared name missing
        When: order_and_fill_metrics is applied
        Then: Every declared name appears, in declaration order, the missing one
            carrying a zero on an unbounded interval

        Test type: unit
        """
        computed = [
            MetricValue(
                name="second", value=2.0, lower_confidence_bound=1.0, upper_confidence_bound=3.0
            )
        ]

        ordered = order_and_fill_metrics(["first", "second"], computed)

        assert [m.name for m in ordered] == ["first", "second"]
        assert ordered[0].value == 0.0
        assert ordered[0].lower_confidence_bound == -np.inf
        assert ordered[0].upper_confidence_bound == np.inf

    def test_optional_name_is_omitted_rather_than_filled(self) -> None:
        """Test that an optional declared name disappears when not computed.

        Purpose: Validates the escape hatch for metrics whose historical
            behaviour is to vanish when nothing contributed, rather than to
            report a zero that would read as a real measurement

        Given: A declared list whose second name was not computed
        When: That name is passed as optional
        Then: It is left out instead of filled

        Test type: unit
        """
        computed = [
            MetricValue(
                name="first", value=1.0, lower_confidence_bound=0.0, upper_confidence_bound=2.0
            )
        ]

        ordered = order_and_fill_metrics(["first", "second"], computed, optional=["second"])

        assert [m.name for m in ordered] == ["first"]


class TestMissingChannels:
    """Behaviour when a channel is absent, which is the main correctness trap."""

    def test_absent_channel_omits_the_metric_entirely(self) -> None:
        """Test that a never-reported channel produces no metric.

        Purpose: Validates that a measurement which was never taken is not
            reported as a measurement of zero, which would silently understate
            impact severity or overstate safety

        Given: Episodes that report only the success channel
        When: A spec over the impact channel is aggregated
        Then: No metric is produced for it

        Test type: unit
        """
        episodes = [[{_SUCCESS: 1.0}], [{_SUCCESS: 0.0}]]
        specs = [
            StepInfoMetric(name="completion", channel=_SUCCESS, per_episode=EpisodeReduction.ANY),
            StepInfoMetric(name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MAX),
        ]

        metrics = aggregate_step_info_metrics(episodes, specs)

        assert [metric.name for metric in metrics] == ["completion"]

    def test_steps_missing_the_channel_are_skipped_by_default(self) -> None:
        """Test that steps without the channel do not contribute a value.

        Purpose: Validates that a channel reported on only some steps is
            averaged over the reporting steps rather than diluted by zeros

        Given: An episode where only one of three steps reports the channel
        When: The channel is aggregated with the MEAN reduction
        Then: The mean is over the single reporting step

        Test type: unit
        """
        episodes = [[{}, {_IMPACT: 6.0}, {}]]
        spec = StepInfoMetric(name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MEAN)

        metrics = aggregate_step_info_metrics(episodes, [spec])

        assert metrics[0].value == 6.0

    def test_explicit_default_fills_missing_steps(self) -> None:
        """Test that an explicit default makes missing steps contribute a value.

        Purpose: Validates the opt-in path for channels where "not reported"
            genuinely means a known value

        Given: An episode where only one of three steps reports the channel and
            a spec with default 0.0
        When: The channel is aggregated with the MEAN reduction
        Then: The mean is over all three steps

        Test type: unit
        """
        episodes = [[{}, {_IMPACT: 6.0}, {}]]
        spec = StepInfoMetric(
            name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MEAN, default=0.0
        )

        metrics = aggregate_step_info_metrics(episodes, [spec])

        assert metrics[0].value == 2.0

    def test_episodes_without_the_channel_do_not_count_toward_the_rate(self) -> None:
        """Test that a non-reporting episode is excluded from the average.

        Purpose: Validates that episodes where a sensor was unavailable neither
            inflate nor deflate the metric

        Given: Two episodes reporting success and one reporting nothing
        When: The success channel is aggregated with the ANY reduction
        Then: The rate is computed over the two reporting episodes only

        Test type: unit
        """
        episodes = [[{_SUCCESS: 1.0}], [{_SUCCESS: 0.0}], [{}]]
        spec = StepInfoMetric(name="completion", channel=_SUCCESS, per_episode=EpisodeReduction.ANY)

        metrics = aggregate_step_info_metrics(episodes, [spec])

        assert metrics[0].value == 0.5


class TestConfidenceIntervals:
    """Confidence-interval behaviour, including the degenerate case."""

    def test_multiple_episodes_produce_a_finite_bracketing_interval(self) -> None:
        """Test that several episodes yield finite bounds around the mean.

        Purpose: Validates that reported statistics carry a usable 95% CI

        Given: Four episodes with differing peak impacts
        When: The channel is aggregated
        Then: Both bounds are finite and bracket the point estimate

        Test type: unit
        """
        episodes = [[{_IMPACT: value}] for value in (1.0, 2.0, 3.0, 4.0)]
        spec = StepInfoMetric(name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MAX)

        metric = aggregate_step_info_metrics(episodes, [spec])[0]

        assert math.isfinite(metric.lower_confidence_bound)
        assert math.isfinite(metric.upper_confidence_bound)
        assert metric.lower_confidence_bound <= metric.value <= metric.upper_confidence_bound

    def test_single_episode_reports_an_unbounded_interval(self) -> None:
        """Test that one episode yields an infinite interval rather than raising.

        Purpose: Validates the degenerate-sample fallback, matching what the
            existing environment metrics report for n < 2

        Given: A single episode reporting the channel
        When: The channel is aggregated
        Then: The bounds are -inf and +inf and the value is still reported

        Test type: unit
        """
        spec = StepInfoMetric(name="impact", channel=_IMPACT, per_episode=EpisodeReduction.MAX)

        metric = aggregate_step_info_metrics([[{_IMPACT: 7.0}]], [spec])[0]

        assert metric.value == 7.0
        assert metric.lower_confidence_bound == -math.inf
        assert metric.upper_confidence_bound == math.inf


class TestHistoryExtraction:
    """Bridging episode histories into the aggregator's input shape."""

    def test_extraction_maps_absent_info_to_an_empty_mapping(self) -> None:
        """Test that steps carrying no info become empty mappings.

        Purpose: Validates that the terminal bookkeeping step, which has no
            info, does not break aggregation

        Given: A history whose second step has info None
        When: extract_episode_step_infos() is called
        Then: That step yields an empty mapping rather than None

        Test type: unit
        """
        history = _history([{_SUCCESS: 1.0}, None])  # type: ignore[list-item]

        extracted = extract_episode_step_infos([history])

        assert extracted == [[{_SUCCESS: 1.0}, {}]]


class TestDefaultComputeMetrics:
    """The default compute_metrics driven by declared specs."""

    def test_declared_specs_drive_names_and_values(self) -> None:
        """Test that an environment gets metrics from specs alone.

        Purpose: Validates that declaring a spec is sufficient — no bespoke
            compute_metrics or get_metric_names body is needed

        Given: An environment declaring one completion-rate spec and reporting
            the success channel, with two of three episodes succeeding
        When: get_metric_names() and compute_metrics() are called
        Then: The declared name matches the produced metric and the value is 2/3

        Test type: integration
        """
        env = _SpecSanity(discount_factor=0.95, **sanity_pinned_kwargs())
        histories = [
            _history([{_SUCCESS: 0.0}, {_SUCCESS: 1.0}]),
            _history([{_SUCCESS: 1.0}]),
            _history([{_SUCCESS: 0.0}]),
        ]

        metrics = env.compute_metrics(histories)

        assert env.get_metric_names() == ["task_completion_rate"]
        assert [metric.name for metric in metrics] == env.get_metric_names()
        assert math.isclose(metrics[0].value, 2.0 / 3.0)

    def test_environment_without_specs_reports_nothing(self) -> None:
        """Test that the default hooks stay inert for environments not using them.

        Purpose: Validates that the new default is opt-in, so environments that
            never declare a spec behave exactly as before

        Given: A stock environment declaring no metric specs
        When: compute_metrics() is called on arbitrary histories
        Then: An empty list is returned

        Test type: unit
        """
        env = SanityPOMDP(discount_factor=0.95, **sanity_pinned_kwargs())

        assert not env.get_metric_specs()
        assert not Environment.compute_metrics(env, [_history([{_SUCCESS: 1.0}])])
