# SPDX-License-Identifier: MIT

import random

import optuna
import pytest

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.utils.visualization.tuning_plots import (
    TrialRecord,
    extract_trial_records,
    load_trial_records,
    plot_pareto_front,
    plot_parameter_history,
    plot_parameter_slices,
    plot_secondary_metrics,
    plot_tuning_diagnostics,
    save_trial_records,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMETERS_TO_OPTIMIZE = [
    ("discounted_return", HyperParameterOptimizationDirection.MAXIMIZE),
    ("collision_rate", HyperParameterOptimizationDirection.MINIMIZE),
]


def _statistics(name, value):
    return {
        "name": name,
        "value": value,
        "lower_confidence_bound": value - 0.1,
        "upper_confidence_bound": value + 0.1,
    }


@pytest.fixture(name="study")
def study_fixture():
    """A small two-objective study shaped like a real tuning run."""
    rng = random.Random(0)
    study = optuna.create_study(directions=["maximize", "minimize"])

    def objective(trial):
        simulations = trial.suggest_int("num_simulations", 10, 500)
        constant = trial.suggest_float("exploration_constant", 0.1, 10.0)
        trial.suggest_categorical("rollout", ["random", "greedy"])

        returns = -((simulations - 300) / 300) ** 2 - constant / 20 + rng.gauss(0, 0.05)
        collisions = max(0.0, 0.5 - simulations / 1000)

        trial.set_user_attr("metric_discounted_return", returns)
        trial.set_user_attr("metric_collision_rate", collisions)
        trial.set_user_attr(
            "statistics",
            [
                _statistics("discounted_return", returns),
                _statistics("collision_rate", collisions),
                # Recorded but never optimized: the metric the search may pay with.
                _statistics("episode_length", 20 + simulations / 50),
            ],
        )
        return returns, collisions

    study.optimize(objective, n_trials=25)
    return study


@pytest.fixture(name="records")
def records_fixture(study):
    return extract_trial_records(study, PARAMETERS_TO_OPTIMIZE, confidence_interval_level=0.95)


class TestTrialRecords:
    def test_extracts_objectives_params_and_statistics(self, records):
        assert len(records) == 25
        record = records[0]
        assert record.state == "COMPLETE"
        assert set(record.objective_values) == {"discounted_return", "collision_rate"}
        assert set(record.params) == {"num_simulations", "exploration_constant", "rollout"}
        assert "episode_length" in record.metric_statistics
        assert len(record.metric_statistics["episode_length"]) == 3

    def test_marks_the_pareto_front(self, study, records):
        pareto_numbers = {trial.number for trial in study.best_trials}
        assert {record.number for record in records if record.is_pareto} == pareto_numbers

    def test_carries_the_confidence_level(self, records):
        # The plots state the level, so it has to travel with the bounds.
        assert all(record.confidence_interval_level == 0.95 for record in records)

    def test_json_round_trip(self, records, tmp_path):
        path = save_trial_records(records, tmp_path / "trial_records.json")
        reloaded = load_trial_records(path)

        assert [r.number for r in reloaded] == [r.number for r in records]
        assert reloaded[0].params == records[0].params
        assert reloaded[0].metric_statistics == records[0].metric_statistics
        assert reloaded[0].confidence_interval_level == 0.95

    def test_incomplete_trials_do_not_break_extraction(self):
        study = optuna.create_study(directions=["maximize", "minimize"])

        def failing(trial):
            raise ValueError("planner blew up")

        study.optimize(failing, n_trials=2, catch=(ValueError,))
        records = extract_trial_records(study, PARAMETERS_TO_OPTIMIZE)

        assert [record.state for record in records] == ["FAIL", "FAIL"]
        assert all(not record.objective_values for record in records)


class TestTuningPlots:
    def test_writes_every_diagnostic(self, records, study, tmp_path):
        written = plot_tuning_diagnostics(
            records=records,
            parameters_to_optimize=PARAMETERS_TO_OPTIMIZE,
            output_dir=tmp_path,
            front_quality_history=[(i, float(i) / 25) for i in range(5, 25)],
            stopped_at_trial=25,
            study=study,
        )
        names = {path.name for path in written}

        assert "front_quality_history.png" in names
        assert "objective_history.png" in names
        assert "objective_confidence_intervals.png" in names
        assert "pareto_front.png" in names
        assert "parameter_history.png" in names
        assert "parameter_slices.png" in names
        assert "secondary_metrics.png" in names
        assert all(path.stat().st_size > 0 for path in written)

    def test_tuned_parameters_are_plotted(self, records, tmp_path):
        history = plot_parameter_history(records, tmp_path / "parameter_history.png")
        slices = plot_parameter_slices(
            records, PARAMETERS_TO_OPTIMIZE, tmp_path / "parameter_slices.png"
        )

        assert history is not None and history.exists()
        assert slices is not None and slices.exists()

    def test_secondary_metrics_exclude_the_optimized_ones(self, records, tmp_path):
        # episode_length is the only non-optimized metric, so dropping it must
        # leave nothing to plot.
        stripped = [
            TrialRecord(
                number=record.number,
                state=record.state,
                params=record.params,
                objective_values=record.objective_values,
                metric_statistics={
                    name: bounds
                    for name, bounds in record.metric_statistics.items()
                    if name != "episode_length"
                },
            )
            for record in records
        ]

        assert (
            plot_secondary_metrics(records, PARAMETERS_TO_OPTIMIZE, tmp_path / "secondary.png")
            is not None
        )
        assert (
            plot_secondary_metrics(stripped, PARAMETERS_TO_OPTIMIZE, tmp_path / "none.png") is None
        )

    def test_single_objective_has_no_pareto_plot(self, records, tmp_path):
        assert (
            plot_pareto_front(records, PARAMETERS_TO_OPTIMIZE[:1], tmp_path / "front.png") is None
        )

    def test_no_completed_trials_writes_nothing(self, tmp_path):
        records = [TrialRecord(number=0, state="FAIL")]
        assert plot_tuning_diagnostics(records, PARAMETERS_TO_OPTIMIZE, tmp_path) == []

    def test_confidence_interval_title_says_all_when_nothing_is_trimmed(self, records, tmp_path):
        from POMDPPlanners.utils.visualization.tuning_plots import (
            plot_objective_confidence_intervals,
        )

        # top_k above the trial count must not claim the trials were filtered.
        path = plot_objective_confidence_intervals(
            records, PARAMETERS_TO_OPTIMIZE, tmp_path / "all.png", top_k=100
        )
        trimmed = plot_objective_confidence_intervals(
            records, PARAMETERS_TO_OPTIMIZE, tmp_path / "trimmed.png", top_k=5
        )

        assert path is not None and path.exists()
        assert trimmed is not None and trimmed.exists()
