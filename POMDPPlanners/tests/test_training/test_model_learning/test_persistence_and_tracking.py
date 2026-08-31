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


def _ensemble(seed: int = 0):
    return ProbabilisticEnsembleLearner(num_members=2, epochs=3, seed=seed).fit(_dataset(seed=seed))


class TestEnsemblePersistence:
    """Round-tripping a fitted ensemble through a file."""

    def test_a_reloaded_ensemble_predicts_the_same_density(self, tmp_path) -> None:
        """Purpose: Validates that a saved model reloads as the same model

        Given: A fitted ensemble saved to disk
        When: It is loaded back and asked for the density of the same candidates
        Then: The log-densities match the original's exactly, so the reload kept
            the normalization statistics and not only the weights
        """
        from POMDPPlanners.training.model_learning import ProbabilisticEnsembleTransition

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
        from POMDPPlanners.training.model_learning import ProbabilisticEnsembleTransition

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
        model = LinearGaussianLearner().fit(_dataset())
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
        first = LinearGaussianLearner().fit(_dataset(seed=0))
        second = LinearGaussianLearner().fit(_dataset(seed=5))

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
        first = self._model(LinearGaussianLearner().fit(_dataset(seed=0)))
        second = self._model(LinearGaussianLearner().fit(_dataset(seed=5)))

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
            ProbabilisticEnsembleTransition,
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
