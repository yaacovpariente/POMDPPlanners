# SPDX-License-Identifier: MIT

"""Unit tests for saving a fitted model and recording a run.

The property under test for persistence is that a reloaded model *is* the model:
same predictions, same density, same sampling stream. A round trip that keeps
the weights and drops the normalization statistics reloads without error and
predicts nonsense, so "it loaded" is not the assertion.

For the fingerprint the property is that it moves with the parameters and
reaches the environment's ``config_id``. That is what stops the simulation cache
from serving round n-1's episodes for round n -- a failure that looks like a flat
learning curve rather than like an error.
"""

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FactoredIsaacModelPOMDP,
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    LinearGaussianTransition,
)
from POMDPPlanners.training.model_learning import (
    ControlPoint,
    ProbabilisticEnsembleTransition,
    LearningCurve,
    LinearGaussianLearner,
    ProbabilisticEnsembleLearner,
    RoundResult,
    TransitionDataset,
    curve_summaries,
)

def _artifact_dir(run: Any) -> Path:
    """Local artifact directory of an MLflow run."""
    return Path(run.info.artifact_uri.replace("file://", ""))


SCHEMA = IsaacChannelSchema((("robot", 2), ("hazard_type", 2)))
PRESETS = [np.array([1.0]), np.array([-1.0])]


def _rollouts(num_rows: int, seed: int = 0):
    """Rollouts from a known linear-Gaussian system."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(num_rows, 2))
    actions = rng.normal(size=(num_rows, 1))
    next_states = (
        states @ np.array([[0.9, 0.1], [0.0, 0.95]]).T
        + actions @ np.array([[0.5], [0.2]]).T
        + rng.normal(scale=0.05, size=(num_rows, 2))
    )
    return states, actions, next_states


def _dataset(num_rows: int = 400, seed: int = 0) -> TransitionDataset:
    states, actions, next_states = _rollouts(num_rows, seed)
    dataset = TransitionDataset(holdout_fraction=0.25, seed=seed + 1)
    for start in range(0, num_rows, 50):
        dataset.add_episode(
            states[start : start + 50],
            actions[start : start + 50],
            next_states[start : start + 50],
            source="exploration",
        )
    return dataset


def _round(round_index: int, seed: int = 0) -> RoundResult:
    """One round's result, with a model of its own."""
    return RoundResult(
        round_index=round_index,
        model=_ensemble(seed=seed + round_index),
        dataset_size=60 * round_index,
        source_counts={"exploration": 60 * round_index},
        training_metrics={},
        diagnostics={"held_out_log_likelihood": 1.0},
        control=None,
    )


def _ensemble(seed: int = 0) -> ProbabilisticEnsembleTransition:
    """A fitted ensemble, as its own type -- the save and fingerprint hooks are its."""
    model = ProbabilisticEnsembleLearner(num_members=2, epochs=3, seed=seed).fit(_dataset(seed=seed))
    assert isinstance(model, ProbabilisticEnsembleTransition)
    return model


def _linear(seed: int = 0) -> LinearGaussianTransition:
    """A fitted linear-Gaussian transition, as its own type."""
    model = LinearGaussianLearner().fit(_dataset(seed=seed))
    assert isinstance(model, LinearGaussianTransition)
    return model


class TestEnsemblePersistence:
    """Round-tripping a fitted ensemble through a file."""

    def test_a_reloaded_ensemble_predicts_the_same_density(self, tmp_path) -> None:
        """Purpose: Validates that a saved model reloads as the same model

        Given: A fitted ensemble saved to disk
        When: It is loaded back and asked for the density of the same candidates
        Then: The log-densities match the original's exactly, so the reload kept
            the normalization statistics and not only the weights
        """
        model = _ensemble()
        state = np.array([0.3, -0.2])
        action = np.array([1.0])
        candidates = np.array([[0.25, -0.15], [1.0, 1.0]])
        expected = model.log_probability(state, action, candidates)

        reloaded = ProbabilisticEnsembleTransition.load(model.save(tmp_path / "round_1.pt"))

        assert np.allclose(reloaded.log_probability(state, action, candidates), expected)

    def test_a_reloaded_ensemble_continues_the_same_sampling_stream(self, tmp_path) -> None:
        """Purpose: Validates that the sampler's state survives the round trip

        Given: A fitted ensemble saved after some draws have been taken
        When: The original and the reloaded model each draw again
        Then: They draw the same successors, so a reloaded model reproduces the
            run it was scored in rather than restarting its stream
        """
        model = _ensemble()
        state = np.array([0.3, -0.2])
        action = np.array([1.0])
        model.sample_next_state(state, action, n_samples=4)

        reloaded = ProbabilisticEnsembleTransition.load(model.save(tmp_path / "round_1.pt"))

        assert np.allclose(
            reloaded.sample_next_state(state, action, n_samples=4),
            model.sample_next_state(state, action, n_samples=4),
        )

    def test_two_fits_do_not_share_a_fingerprint(self) -> None:
        """Purpose: Validates that the fingerprint tracks the fitted parameters

        Given: Two ensembles fitted from different seeds, and one fit read twice
        When: Their fingerprints are compared
        Then: The two fits differ and the repeated read is stable, which is what
            a cache key needs from it
        """
        first = _ensemble(seed=0)
        second = _ensemble(seed=1)

        assert first.fingerprint != second.fingerprint
        assert first.fingerprint == first.fingerprint


class TestLinearPersistence:
    """The floor baseline saves and loads too, or it cannot be compared to."""

    def test_a_reloaded_linear_transition_predicts_the_same_density(self, tmp_path) -> None:
        """Purpose: Validates the linear model's round trip

        Given: A fitted linear-Gaussian transition saved to disk
        When: It is loaded back
        Then: It reports the same log-density as the original
        """
        model = _linear()
        state = np.array([0.3, -0.2])
        action = np.array([1.0])
        candidates = np.array([[0.25, -0.15], [1.0, 1.0]])
        expected = model.log_probability(state, action, candidates)

        reloaded = LinearGaussianTransition.load(model.save(tmp_path / "round_1.npz"))

        assert np.allclose(reloaded.log_probability(state, action, candidates), expected)

    def test_a_refit_moves_the_fingerprint(self) -> None:
        """Purpose: Validates that the linear fit's fingerprint tracks its parameters

        Given: Two linear fits of different data
        When: Their fingerprints are compared
        Then: They differ
        """
        first = _linear(seed=0)
        second = _linear(seed=5)

        assert first.fingerprint != second.fingerprint


class TestFittedTransitionReachesTheCacheKey:
    """The reason the fingerprint exists: ``config_id`` must move with the fit."""

    @staticmethod
    def _model(transition: Any) -> FactoredIsaacModelPOMDP:
        return FactoredIsaacModelPOMDP(
            state_schema=SCHEMA,
            action_presets=PRESETS,
            discount_factor=0.99,
            transition=transition,
            transition_channels=("robot",),
        )

    def test_two_rounds_of_a_fit_are_not_the_same_experiment(self) -> None:
        """Purpose: Validates that a refitted transition changes the environment's config_id

        Given: Two environments identical but for the fitted transition they plan with
        When: Their config_ids are compared
        Then: They differ, so the simulation cache cannot serve one round's
            episodes as the next round's
        """
        first = self._model(_linear(seed=0))
        second = self._model(_linear(seed=5))

        assert first.config_id != second.config_id

    def test_an_analytic_transition_leaves_the_key_alone(self) -> None:
        """Purpose: Validates that envs without a fitted transition keep their key

        Given: An environment whose transition reports no fingerprint
        When: Its config_id is taken twice, from two equal instances
        Then: They agree, so adding the fingerprint did not invalidate the cache
            of every analytic environment
        """

        class _Analytic:
            def sample_next_state(self, state, action, n_samples=1):
                del action, n_samples
                return np.asarray(state, dtype=float)

        assert self._model(_Analytic()).config_id == self._model(_Analytic()).config_id


class _RecordingTracker:
    """A tracker that keeps what it was handed, so the trainer's calls are visible."""

    def __init__(self) -> None:
        self.rounds: List[Any] = []
        self.curves: List[LearningCurve] = []

    def log_round(self, result: Any) -> None:
        self.rounds.append(result)

    def log_curve(self, curve: LearningCurve) -> None:
        self.curves.append(curve)


class TestTrainerCallsTheTracker:
    """The loop must hand every round over as it goes, not at the end."""

    def test_every_round_is_recorded_with_its_own_model(self) -> None:
        """Purpose: Validates that the trainer reports each round to the tracker

        Given: A two-round loop with a recording tracker
        When: The loop runs
        Then: Two rounds were recorded, each carrying its own fitted model, so a
            tracker can save the whole sequence rather than the last round
        """
        from POMDPPlanners.training.model_learning import DAggerModelTrainer

        class _World:
            def initial_state_dist(self):
                class _Dist:
                    def sample(self, num_samples: int = 1):
                        del num_samples
                        return np.zeros(2)

                return _Dist()

            def sample_next_state(self, state, action, n_samples: int = 1):
                vector = np.asarray(state, dtype=float).reshape(-1)
                return vector * 0.9 + np.asarray(action, dtype=float).reshape(-1)[0] * 0.1

            def is_terminal(self, state) -> bool:
                del state
                return False

        tracker = _RecordingTracker()
        trainer = DAggerModelTrainer(
            world=_World(),
            learner=LinearGaussianLearner(),
            dataset=TransitionDataset(holdout_fraction=0.25, seed=1),
            action_presets=np.array([[1.0], [-1.0]]),
            planner_rollout_fn=None,
            num_rounds=2,
            episodes_per_round=6,
            steps_per_episode=20,
            horizon=5,
            tracker=tracker,
            seed=0,
        )
        rounds = trainer.run()

        assert [result.round_index for result in tracker.rounds] == [1, 2]
        assert [result.model for result in tracker.rounds] == [r.model for r in rounds]


class TestMLflowTracker:
    """What lands in MLflow, checked against a local file store."""

    def test_a_round_logs_its_metrics_and_its_model(self, tmp_path) -> None:
        """Purpose: Validates that a round's numbers and model both reach MLflow

        Given: A tracker pointed at a local MLflow store and one round's result
        When: The round and the curve are logged
        Then: The run holds the round's return and diagnostics as metrics, and
            the fitted model as an artifact that loads back
        """
        mlflow = pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import (
            MLflowModelLearningTracker,
            load_round_models,
        )

        model = _ensemble()
        result = RoundResult(
            round_index=1,
            model=model,
            dataset_size=120,
            source_counts={"exploration": 60, "planner": 60},
            training_metrics={"train_nll": [3.0, 1.5]},
            diagnostics={"held_out_log_likelihood": 2.5, "horizon_drift_ratio": 0.8},
            control=ControlPoint(round_index=1, cumulative_transitions=120, returns=(-1.0, -1.4)),
        )
        assert result.control is not None
        curve = LearningCurve(method="dagger", seed=0, points=(result.control,))

        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=0,
            params={"environment": "unit_test"},
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
        )
        tracker.log_round(result)
        run_id = tracker._run.info.run_id  # pylint: disable=protected-access
        tracker.log_curve(curve)

        run = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path / 'mlruns'}").get_run(
            run_id
        )
        assert run.data.metrics["mean_return"] == pytest.approx(-1.2)
        assert run.data.metrics["horizon_drift_ratio"] == pytest.approx(0.8)
        assert run.data.metrics["best_round_index"] == pytest.approx(1.0)
        assert run.data.params["method"] == "dagger"

        saved = load_round_models(_artifact_dir(run))
        assert set(saved) == {1}
        reloaded = ProbabilisticEnsembleTransition.load(saved[1])
        state, action = np.array([0.3, -0.2]), np.array([1.0])
        assert np.allclose(
            reloaded.log_probability(state, action, np.array([[0.25, -0.15]])),
            model.log_probability(state, action, np.array([[0.25, -0.15]])),
        )

    def test_the_per_round_record_names_the_model_of_each_round(self, tmp_path) -> None:
        """Purpose: Validates that the run keeps the fit-to-number link

        Given: Two rounds logged with different fitted models
        When: The run's rounds.json is read
        Then: Each round names its own model artifact and fingerprint, so a
            control number can be traced to the parameters that produced it
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import MLflowModelLearningTracker

        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=1,
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
        )
        for index, seed in enumerate((0, 1), start=1):
            tracker.log_round(
                RoundResult(
                    round_index=index,
                    model=_ensemble(seed=seed),
                    dataset_size=60 * index,
                    source_counts={"exploration": 60 * index},
                    training_metrics={},
                    diagnostics={"held_out_log_likelihood": 1.0},
                    control=None,
                )
            )
        run = tracker._run  # pylint: disable=protected-access
        tracker.finish()

        record = json.loads(
            (_artifact_dir(run) / "evaluation" / "rounds.json").read_text(encoding="utf-8")
        )
        assert [entry["round_index"] for entry in record] == [1, 2]
        assert record[0]["model_fingerprint"] != record[1]["model_fingerprint"]
        assert record[0]["model_artifact"] == "models/round_1.pt"


class TestTrackingSurvivesTheSimulator:
    """The rollouts being evaluated log to MLflow too, and they move the global state."""

    def test_a_hijacked_active_run_does_not_split_the_study(self, tmp_path) -> None:
        """Purpose: Validates that the tracker keeps its run when another component takes over

        Given: A round logged, then MLflow's tracking URI switched and its active
            run ended -- what LocalSimulationsAPI does on every rollout batch
        When: A second round is logged
        Then: Both rounds are in the same run, rather than round two landing in
            whichever store the rollouts happened to be writing to
        """
        mlflow = pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import MLflowModelLearningTracker

        tracking_uri = f"file://{tmp_path / 'mlruns'}"
        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=2,
            tracking_uri=tracking_uri,
        )
        tracker.log_round(_round(1))
        run_id = tracker._run.info.run_id  # pylint: disable=protected-access

        mlflow.set_tracking_uri(f"file://{tmp_path / 'other_mlruns'}")
        mlflow.set_experiment("someone_elses_experiment")
        mlflow.start_run()
        mlflow.end_run()

        tracker.log_round(_round(2))
        tracker.finish()

        run = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
        assert run.data.metrics["dataset_size"] == pytest.approx(120.0)
        record = json.loads(
            (_artifact_dir(run) / "evaluation" / "rounds.json").read_text(encoding="utf-8")
        )
        assert [entry["round_index"] for entry in record] == [1, 2]


class TestTrainingCurves:
    """The ensemble is a network, so the fit needs its own curves."""

    def test_the_fit_records_a_train_and_a_holdout_curve_per_epoch(self) -> None:
        """Purpose: Validates that the ensemble reports what its training did

        Given: An ensemble fitted for a known number of epochs on a dataset with
            a holdout split
        When: The training metrics are read
        Then: Both the training and the held-out loss have one value per epoch,
            so overfitting is visible rather than inferred
        """
        learner = ProbabilisticEnsembleLearner(num_members=2, epochs=4, seed=0)
        learner.fit(_dataset())

        metrics = learner.training_metrics()

        assert len(metrics["train_nll"]) == 4
        assert len(metrics["holdout_nll"]) == 4

    def test_every_member_keeps_its_own_curve(self) -> None:
        """Purpose: Validates that a diverged member is not hidden by the mean

        Given: A three-member ensemble
        When: The training metrics are read
        Then: There is one curve per member beside the mean, because the
            ensemble's spread is its uncertainty estimate and one failed member
            corrupts it invisibly
        """
        learner = ProbabilisticEnsembleLearner(num_members=3, epochs=2, seed=0)
        learner.fit(_dataset())

        metrics = learner.training_metrics()

        members = [key for key in metrics if key.startswith("train_nll_member_")]
        assert len(members) == 3

    def test_a_dataset_with_no_holdout_still_fits(self) -> None:
        """Purpose: Validates that the holdout curve is optional, not required

        Given: A dataset that holds nothing back
        When: An ensemble is fitted
        Then: The fit succeeds and reports a training curve with no holdout one
        """
        dataset = TransitionDataset(holdout_fraction=0.0, seed=0)
        states, actions, next_states = _rollouts(200)
        dataset.add_episode(states, actions, next_states, source="exploration")

        learner = ProbabilisticEnsembleLearner(num_members=2, epochs=2, seed=0)
        learner.fit(dataset)

        assert "holdout_nll" not in learner.training_metrics()


class TestTrainingPlots:
    """The plots must be drawable from a finished run, not only from live objects."""

    def test_the_plots_are_written_from_the_tracker_records(self, tmp_path) -> None:
        """Purpose: Validates that a run's JSON record is enough to redraw the fit

        Given: Per-round records in the shape the tracker writes to rounds.json
        When: The report is drawn
        Then: The training, per-member and diagnostic plots are all written, so
            a study can be re-read without re-fitting anything
        """
        from POMDPPlanners.utils.visualization.model_learning_plots import (
            plot_model_learning_report,
        )

        rounds = [
            {
                "round_index": index,
                "diagnostics": {"held_out_log_likelihood": 1.0 * index, "horizon_drift_ratio": 1.4},
                "training_metrics": {
                    "train_nll": [3.0, 2.0, 1.5],
                    "holdout_nll": [3.1, 2.4, 2.6],
                    "train_nll_member_0": [3.0, 2.1, 1.6],
                    "train_nll_member_1": [3.0, 1.9, 1.4],
                },
            }
            for index in (1, 2)
        ]

        written = plot_model_learning_report(rounds, tmp_path / "plots")

        assert set(written) == {"training_curves", "member_training_curves", "round_diagnostics"}
        assert all(path.exists() for path in written.values())

    def test_a_run_without_training_curves_draws_nothing(self, tmp_path) -> None:
        """Purpose: Validates that an unrecorded fit is reported, not faked

        Given: Rounds with no training metrics
        When: The training plot is drawn
        Then: Nothing is written and None comes back, rather than an empty axis
        """
        from POMDPPlanners.utils.visualization.model_learning_plots import plot_training_curves

        assert plot_training_curves([{"round_index": 1}], tmp_path / "none.png") is None

    def test_a_tracked_run_logs_its_plots(self, tmp_path) -> None:
        """Purpose: Validates that the plots reach MLflow with the numbers

        Given: A tracker with two logged rounds from a real ensemble fit
        When: The run is finished
        Then: The evaluation directory holds the training and diagnostic figures
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import MLflowModelLearningTracker

        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=3,
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
        )
        learner = ProbabilisticEnsembleLearner(num_members=2, epochs=3, seed=0)
        for index in (1, 2):
            model = learner.fit(_dataset(seed=index))
            tracker.log_round(
                RoundResult(
                    round_index=index,
                    model=model,
                    dataset_size=60 * index,
                    source_counts={"exploration": 60 * index},
                    training_metrics=learner.training_metrics(),
                    diagnostics={"held_out_log_likelihood": 1.0, "horizon_drift_ratio": 1.2},
                    control=None,
                )
            )
        run = tracker._run  # pylint: disable=protected-access
        tracker.finish()

        evaluation = _artifact_dir(run) / "evaluation"
        assert {path.name for path in evaluation.glob("*.png")} == {
            "training_curves.png",
            "member_training_curves.png",
            "round_diagnostics.png",
        }


class TestReadableReports:
    """A run has to be readable without opening JSON."""

    @staticmethod
    def _rounds():
        return [
            {
                "round_index": index,
                "dataset_size": 100 * index,
                "source_counts": {"exploration": 100, "planner": 50 * (index - 1)},
                "diagnostics": {
                    "held_out_log_likelihood": 40.0,
                    "horizon_drift_ratio": 0.5 if index == 1 else 1.8,
                },
                "training_metrics": {"holdout_nll": [5.0, 3.0, 4.0]},
                "model_artifact": f"models/round_{index}.pt",
                "control": {
                    "round_index": index,
                    "cumulative_transitions": 100 * index,
                    "returns": [-4.0 + index, -3.0 + index],
                },
            }
            for index in (1, 2)
        ]

    def test_the_table_marks_the_best_round_and_flags_overconfidence(self) -> None:
        """Purpose: Validates that the table carries the two judgements a reader needs

        Given: Two rounds, the second better but with a drift ratio above one
        When: The Markdown table is rendered
        Then: The second round is marked best and its drift ratio is flagged, so
            neither "report the last round" nor "a confident wrong model" passes
            unnoticed
        """
        from POMDPPlanners.training.model_learning import round_table_markdown

        table = round_table_markdown(self._rounds())

        best_line = [line for line in table.splitlines() if "**(best)**" in line]
        assert len(best_line) == 1
        assert best_line[0].startswith("| 2")
        assert "⚠" in best_line[0]

    def test_the_csv_has_one_row_per_round(self) -> None:
        """Purpose: Validates that the same rows are available for a spreadsheet

        Given: Two rounds
        When: The CSV is rendered
        Then: It has a header and one row per round, with the round's cost and return
        """
        from POMDPPlanners.training.model_learning import round_table_csv

        lines = round_table_csv(self._rounds()).strip().splitlines()

        assert lines[0].startswith("round,transitions")
        assert len(lines) == 3

    def test_write_reports_puts_both_files_beside_the_json(self, tmp_path) -> None:
        """Purpose: Validates the reports a run writes for a person

        Given: Rounds and a method's curve
        When: The reports are written
        Then: A summary Markdown and a rounds CSV exist, and the summary names
            the method
        """
        from POMDPPlanners.training.model_learning import write_reports

        curve = LearningCurve(
            method="dagger",
            seed=0,
            points=(ControlPoint(round_index=1, cumulative_transitions=100, returns=(-1.0, -2.0)),),
        )

        written = write_reports(self._rounds(), tmp_path, curves=[curve])

        assert set(written) == {"summary", "rounds_csv"}
        assert "dagger" in written["summary"].read_text(encoding="utf-8")

    def test_a_tracked_run_logs_the_tables(self, tmp_path) -> None:
        """Purpose: Validates that the tables reach MLflow with the JSON

        Given: A tracker with one logged round
        When: The run is finished
        Then: summary.md and rounds.csv sit beside rounds.json in the artifacts
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import MLflowModelLearningTracker

        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=4,
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
        )
        tracker.log_round(_round(1))
        run = tracker._run  # pylint: disable=protected-access
        tracker.finish()

        evaluation = _artifact_dir(run) / "evaluation"
        assert {path.name for path in evaluation.glob("*")} >= {
            "rounds.json",
            "summary.md",
            "rounds.csv",
        }


class TestArtifactLayout:
    """Two directories per run: the models, and what says which one to use."""

    def test_a_run_holds_only_models_and_evaluation(self, tmp_path) -> None:
        """Purpose: Validates that a finished run has one place for each question

        Given: A tracker with two logged rounds and a curve
        When: The run is finished
        Then: Its artifacts are exactly models/ and evaluation/, with the models
            named by round and the tables and figures together in evaluation
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import MLflowModelLearningTracker

        tracker = MLflowModelLearningTracker(
            experiment_name="model_learning_test",
            method="dagger",
            seed=6,
            tracking_uri=f"file://{tmp_path / 'mlruns'}",
        )
        learner = ProbabilisticEnsembleLearner(num_members=2, epochs=2, seed=0)
        for index in (1, 2):
            tracker.log_round(
                RoundResult(
                    round_index=index,
                    model=learner.fit(_dataset(seed=index)),
                    dataset_size=100 * index,
                    source_counts={"exploration": 100 * index},
                    training_metrics=learner.training_metrics(),
                    diagnostics={"held_out_log_likelihood": 2.0, "horizon_drift_ratio": 0.9},
                    control=ControlPoint(index, 100 * index, (-1.0, -2.0)),
                )
            )
        tracker.log_curve(LearningCurve("dagger", 6, (ControlPoint(1, 100, (-1.0,)),)))

        artifacts = tracker.artifact_dir
        assert artifacts is not None
        assert {path.name for path in artifacts.iterdir()} == {"models", "evaluation"}
        assert {path.name for path in (artifacts / "models").glob("*")} == {
            "round_1.pt",
            "round_2.pt",
        }
        assert {path.name for path in (artifacts / "evaluation").glob("*")} == {
            "rounds.json",
            "summary.md",
            "rounds.csv",
            "learning_curve.json",
            "training_curves.png",
            "member_training_curves.png",
            "round_diagnostics.png",
        }

    def test_the_comparison_lands_in_every_run(self, tmp_path) -> None:
        """Purpose: Validates that one run's evaluation dir is enough to decide with

        Given: Two logged runs and the curves comparing them
        When: The comparison is logged
        Then: Each run's evaluation/ gains the learning curve and the method
            table, so a reader never has to open a second run to judge a model
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import (
            MLflowModelLearningTracker,
            log_study_comparison,
        )

        tracking_uri = f"file://{tmp_path / 'mlruns'}"
        artifact_dirs = []
        curves = []
        for method in ("dagger", "batch"):
            tracker = MLflowModelLearningTracker(
                experiment_name="comparison_test",
                method=method,
                seed=0,
                tracking_uri=tracking_uri,
            )
            tracker.log_round(_round(1))
            artifact_dirs.append(tracker.artifact_dir)
            curve = LearningCurve(method, 0, (ControlPoint(1, 100, (-1.0, -2.0)),))
            curves.append(curve)
            tracker.log_curve(curve)

        updated = log_study_comparison(
            "comparison_test",
            curves,
            params={"environment": "FrankaReachFragile"},
            tracking_uri=tracking_uri,
        )

        assert len(updated) == 2
        for directory in artifact_dirs:
            assert directory is not None
            assert {path.name for path in directory.iterdir()} == {"models", "evaluation"}
            names = {path.name for path in (directory / "evaluation").glob("*")}
            assert {"learning_curves.png", "methods.md", "summary.md"} <= names

    def test_no_curves_means_nothing_is_written(self, tmp_path) -> None:
        """Purpose: Validates that an unevaluated study reports nothing

        Given: Curves with no evaluated rounds
        When: The comparison is logged
        Then: No run is updated
        """
        pytest.importorskip("mlflow")
        from POMDPPlanners.training.model_learning import log_study_comparison

        assert (
            log_study_comparison(
                "comparison_test",
                [LearningCurve("dagger", 0, ())],
                tracking_uri=f"file://{tmp_path / 'mlruns'}",
            )
            == []
        )


class TestCurveSummaries:
    """The one table to read first."""

    def test_the_summary_reports_the_best_round_not_the_last(self) -> None:
        """Purpose: Validates that the summary picks the round the guarantee covers

        Given: A curve whose second round is worse than its first
        When: The summary is taken
        Then: It reports round 1, because the sequence is not monotone and the
            bound is on the best model of it
        """
        curve = LearningCurve(
            method="dagger",
            seed=0,
            points=(
                ControlPoint(round_index=1, cumulative_transitions=60, returns=(-1.0, -1.0)),
                ControlPoint(round_index=2, cumulative_transitions=120, returns=(-4.0, -4.0)),
            ),
        )

        summary = curve_summaries([curve])["dagger_0"]

        assert summary["best_round_index"] == 1.0
        assert summary["mean_return"] == pytest.approx(-1.0)
