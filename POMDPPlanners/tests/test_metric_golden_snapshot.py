# SPDX-License-Identifier: MIT

"""Characterization tests pinning every environment's ``compute_metrics`` output.

These tests exist to protect a cross-cutting change: the per-step
``StepData.info`` channel touches a type that every environment flows through, so
"no existing environment moved" has to be demonstrated rather than assumed.

The baseline lives in :mod:`POMDPPlanners.tests.metric_golden_values` and is
replayed against the committed history fixture, so these tests contain no
randomness at all.
"""

import math
from typing import Dict, List

import pytest

from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.tests.metric_golden_values import (
    GOLDEN_METRIC_NAMES,
    GOLDEN_METRIC_ORDER,
    GOLDEN_METRIC_VALUES,
    GOLDEN_METRIC_VALUES_WITH_TERMINAL,
)
from POMDPPlanners.tests.test_utils.golden_metric_snapshot import (
    append_terminal_step,
    build_registry,
    compute_metric_order,
    compute_metric_snapshot,
    load_snapshot_histories,
)

_ENV_SLUGS = sorted(build_registry())


@pytest.fixture(name="frozen_histories", scope="module")
def _frozen_histories() -> Dict[str, List[History]]:
    """Load the committed history fixture once for the whole module."""
    return load_snapshot_histories()


class TestMetricGoldenSnapshot:
    """Characterization suite over every instantiable environment's metrics."""

    @pytest.mark.parametrize("slug", _ENV_SLUGS)
    def test_metric_values_match_frozen_baseline(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that each environment's metric values are unchanged.

        Purpose: Validates that shared metrics plumbing changes leave every
            existing environment's computed metric values bit-stable

        Given: The committed frozen histories and the pre-change golden values
        When: compute_metrics() is run for the environment against those histories
        Then: Every metric name is present and every value matches the baseline

        Test type: unit
        """
        environment = build_registry()[slug]()
        actual = compute_metric_snapshot(environment, frozen_histories[slug])
        expected = GOLDEN_METRIC_VALUES[slug]

        assert set(actual) == set(expected), (
            f"{slug}: produced metric names drifted from the frozen baseline. "
            f"Added={sorted(set(actual) - set(expected))}, "
            f"removed={sorted(set(expected) - set(actual))}"
        )
        for name, expected_value in expected.items():
            assert math.isclose(
                actual[name], expected_value, rel_tol=1e-9, abs_tol=1e-12
            ), f"{slug}.{name} changed: {expected_value} -> {actual[name]}"

    @pytest.mark.parametrize("slug", _ENV_SLUGS)
    def test_metric_values_match_frozen_baseline_with_terminal_step(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that metric values are unchanged on the terminated-episode shape.

        Purpose: Validates that the final state of a terminated episode is still
            counted exactly as before, which the raw fixture cannot show because
            it carries no terminal bookkeeping step

        Given: The frozen histories extended with a terminal bookkeeping step,
            and the pre-change golden values for that shape
        When: compute_metrics() is run for the environment against them
        Then: Every metric name is present and every value matches the baseline

        Test type: unit
        """
        environment = build_registry()[slug]()
        actual = compute_metric_snapshot(environment, append_terminal_step(frozen_histories[slug]))
        expected = GOLDEN_METRIC_VALUES_WITH_TERMINAL[slug]

        assert set(actual) == set(expected), (
            f"{slug}: produced metric names drifted from the terminal-shape baseline. "
            f"Added={sorted(set(actual) - set(expected))}, "
            f"removed={sorted(set(expected) - set(actual))}"
        )
        for name, expected_value in expected.items():
            assert math.isclose(
                actual[name], expected_value, rel_tol=1e-9, abs_tol=1e-12
            ), f"{slug}.{name} changed on the terminal shape: {expected_value} -> {actual[name]}"

    @pytest.mark.parametrize("slug", _ENV_SLUGS)
    def test_metric_order_matches_frozen_baseline(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that produced and declared metric orders are unchanged.

        Purpose: Validates that no environment reordered its metrics. Both other
            value/name assertions are order-blind -- one compares a dict, the
            other sorts -- so a reordering would otherwise pass unnoticed while
            silently changing positional consumers such as tuning objectives

        Given: An environment from the registry and its frozen order baseline
        When: compute_metrics() and get_metric_names() are called
        Then: Both emit exactly the baseline names in exactly the baseline order

        Test type: unit
        """
        environment = build_registry()[slug]()
        expected = GOLDEN_METRIC_ORDER[slug]

        assert (
            compute_metric_order(environment, frozen_histories[slug]) == expected
        ), f"{slug}: compute_metrics() changed the order it emits metrics in."
        assert (
            list(environment.get_metric_names()) == expected
        ), f"{slug}: get_metric_names() changed the order it declares metrics in."

    @pytest.mark.parametrize("slug", _ENV_SLUGS)
    def test_metric_names_match_frozen_baseline(self, slug: str) -> None:
        """Test that each environment's declared metric names are unchanged.

        Purpose: Validates that no metric was renamed, which would silently
            invalidate saved MLflow runs and Optuna objective configs

        Given: An environment from the registry and its frozen name list
        When: get_metric_names() is called
        Then: The sorted names match the baseline exactly

        Test type: unit
        """
        environment = build_registry()[slug]()
        assert sorted(environment.get_metric_names()) == GOLDEN_METRIC_NAMES[slug], (
            f"{slug}: declared metric names changed. Renaming a metric breaks "
            f"saved runs and tuning configs, so it needs an explicit migration."
        )

    def test_registry_covers_every_frozen_environment(self) -> None:
        """Test that the registry and the frozen baseline describe the same envs.

        Purpose: Validates that an environment cannot be silently dropped from
            the baseline, which would make these tests vacuously pass

        Given: The environment registry and both frozen baseline mappings
        When: Their key sets are compared
        Then: All three cover exactly the same environment slugs

        Test type: unit
        """
        assert (
            set(build_registry())
            == set(GOLDEN_METRIC_VALUES)
            == set(GOLDEN_METRIC_VALUES_WITH_TERMINAL)
            == set(GOLDEN_METRIC_NAMES)
            == set(GOLDEN_METRIC_ORDER)
        )

    def test_frozen_history_fixture_is_well_formed(
        self, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that the committed history fixture unpickles into usable histories.

        Purpose: Validates that the pickled baseline still loads, which is also
            what pins StepData wire compatibility across field additions

        Given: The committed history pickle, written before StepData gained info
        When: It is loaded and inspected
        Then: Every slug maps to non-empty histories of non-empty StepData lists,
            and every step's info defaults to None

        Test type: unit
        """
        assert set(frozen_histories) == set(build_registry())
        for slug, histories in frozen_histories.items():
            assert histories, f"{slug}: fixture has no episodes"
            for history in histories:
                assert isinstance(history, History)
                assert history.history, f"{slug}: fixture episode has no steps"
                assert all(isinstance(step, StepData) for step in history.history)
                # This fixture was pickled as a six-value StepData. It still loads
                # because the new field is trailing and defaulted -- which is what
                # keeps every cached History in diskcache/joblib readable. A
                # future field inserted anywhere but the end would fail here.
                assert all(step.info is None for step in history.history)
