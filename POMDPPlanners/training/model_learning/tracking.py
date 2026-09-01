# SPDX-License-Identifier: MIT

"""Recording a model-learning run in MLflow: the numbers and the models themselves.

The loop already produces everything needed to decide whether a fitted model is
worth planning with. It produced it in memory, printed a log line, and dropped
it -- so the model a round was scored on could not be reloaded, and the curve
could not be redrawn without re-running the whole study. This module is where a
round is written down.

MLflow rather than a directory convention, because the rollouts inside this loop
already run through it: every planner episode goes through
``LocalSimulationsAPI``, which logs to MLflow, so a run of the loop with a
separate results tree would leave the control numbers and the models they came
from in two systems that agree by hand. One run per ``(method, seed)``, and the per-round
metrics are step-indexed so the learning curve is a chart rather than a file to
plot.

Each run holds exactly two artifact directories, because there are exactly two
things anyone comes for: ``models/`` -- one file per round, the objects to plan
with -- and ``evaluation/`` -- the tables and figures that say which of them to
use. Nothing is mirrored into a second tree: a copy has to be kept in step, and
the copy is what people then find stale.

Three things are logged that are easy to omit and hard to reconstruct later.

**The model of every round, not the best one.** Which round was best is known
only after the whole sequence has been scored, and the guarantee is about the
best model of the sequence. Saving as you go costs a file per round; saving at
the end costs a re-run. The one to load is copied to ``models/chosen`` at the
end, so nobody has to pick from a directory of candidates -- and nobody reaches
for the last round on the assumption that later is better.

**The fingerprint beside the model.** It is what ties a control number to the
exact parameters that produced it, and it is what the environment's cache key
moves on. A run whose fingerprints repeat across rounds is a run whose later
rounds were scored on an earlier round's episodes -- visible in the log rather
than as an unexplained flat curve.

**The fit's own curves, and the plots of them.** A learning curve that does not
rise has two causes -- the network never fitted, or it fitted and the model still
does not help the planner -- and only the per-epoch train and holdout losses
separate them, so they are logged in full and drawn at the end of the run.

**Both halves of the judgement.** The return says whether the model plans well;
the diagnostics say why it does not. A drift ratio above one means the model's
own error bars do not cover its error over the planning horizon, which is the
failure a risk-sensitive planner turns into confident bad decisions.

Classes:
    ModelLearningTracker: Protocol the trainer calls; implement it to record elsewhere.
    MLflowModelLearningTracker: Logs rounds, models and curves to MLflow.

Functions:
    log_study_comparison: Add the across-run comparison to every run's evaluation.
    load_round_models: Find a finished run's saved models.
    curve_summaries: Best round per curve, as a mapping.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from POMDPPlanners.training.model_learning.control_evaluation import (
    LearningCurve,
    best_point,
)
from POMDPPlanners.utils.logger import get_logger

#: Artifact sub-directory holding one file per round's fitted model.
MODELS_ARTIFACT_PATH = "models"

#: Artifact sub-directory holding the curve and the per-round record.
EVALUATION_ARTIFACT_PATH = "evaluation"


class ModelLearningTracker(Protocol):
    """What :class:`~...dagger_trainer.DAggerModelTrainer` needs of a tracker.

    A protocol rather than a base class so the trainer never imports MLflow: the
    loop is useful without tracking, and a test should be able to assert what was
    recorded without a tracking server.
    """

    def log_round(self, result: Any) -> None:
        """Record one round: its metrics and its fitted model."""

    def log_curve(self, curve: LearningCurve) -> None:
        """Record the finished curve, the round to report, and close the run."""


class MLflowModelLearningTracker:
    """Log a method-and-seed run of the loop to MLflow, models included.

    Args:
        experiment_name: MLflow experiment to log under.
        method: The method this run is, e.g. ``"dagger"`` or ``"batch"``. Logged
            as a parameter so curves of two methods can be pulled apart later.
        seed: The repetition's seed, logged for the same reason.
        params: Extra parameters worth pinning to the run -- environment name,
            learner settings, planner budget, commit. Anything the run cannot be
            reproduced without.
        tracking_uri: MLflow tracking URI. ``None`` uses the ambient setting.
        run_name: Name for the parent run. ``None`` builds one from method and seed.
        logger: Optional logger.

    Example:
        Wrap a trainer's run so every round is recorded::

            with MLflowModelLearningTracker("model_learning", "dagger", 0) as tracker:
                trainer = DAggerModelTrainer(..., tracker=tracker)
                trainer.run()
                tracker.log_curve(trainer.learning_curve("dagger"))
    """

    def __init__(
        self,
        experiment_name: str,
        method: str,
        seed: int,
        params: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.method = method
        self.seed = int(seed)
        self.params = dict(params or {})
        self.tracking_uri = tracking_uri
        self.run_name = run_name or f"{method}_seed{seed}"
        self._logger = logger or get_logger(__name__)
        self._client: Any = None
        self._run = None
        self._artifact_dir: Optional[Path] = None
        self._curves: list = []
        self._rounds: list = []

    def __enter__(self) -> "MLflowModelLearningTracker":
        """Start the parent run and pin the parameters to it."""
        self._ensure_run()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Write the per-round record and close the run."""
        self.finish()

    def _ensure_run(self) -> None:
        """Start the run on first use, so a tracker can be built before it is needed.

        A run started in a factory and finished by whoever collects the curve is
        the shape :func:`...curve_comparison.run_learning_curves` needs; requiring
        a ``with`` block there would put the tracker's lifetime in the wrong place.

        Everything afterwards goes through a client bound to this run's id rather
        than through MLflow's ambient "active run". The rollouts this loop
        evaluates with run through ``LocalSimulationsAPI``, which sets its own
        tracking URI and ends whatever run is active -- so a fluent-API tracker
        loses its run in the middle of round one and silently logs the rest of
        the study somewhere else.
        """
        # Deferred: importing MLflow costs seconds, and the loop runs without it.
        from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel

        if self._run is not None:
            return
        # MLflow >= 3.6 gates the local filesystem backend behind an opt-in, and
        # this project stores runs locally -- the same default the simulator sets.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self._client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = self._client.get_experiment_by_name(self.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else self._client.create_experiment(self.experiment_name)
        )
        self._run = self._client.create_run(experiment_id, run_name=self.run_name)
        uri = self._run.info.artifact_uri
        self._artifact_dir = Path(uri[len("file://") :]) if uri.startswith("file://") else None
        run_id = self._run.info.run_id
        self._client.log_param(run_id, "method", self.method)
        self._client.log_param(run_id, "seed", self.seed)
        for key, value in self.params.items():
            self._client.log_param(run_id, key, value)

    def finish(self) -> None:
        """Write the per-round record and close the run. Safe to call twice."""
        if self._run is None:
            return
        if self._rounds:
            self._log_json_artifact(self._rounds, "rounds.json")
            self._log_reports()
            self._log_plots()
        self._client.set_terminated(self._run.info.run_id)
        self._run = None

    @property
    def artifact_dir(self) -> Optional[Path]:
        """Local artifact directory of this run, when the store is on disk.

        It holds ``models/`` and ``evaluation/``, and outlives :meth:`finish` so
        a caller can read a finished run's files.

        Returns:
            The directory, or ``None`` before the run starts or when the store
            is remote.
        """
        return self._artifact_dir

    def log_round(self, result: Any) -> None:
        """Record one round's metrics and save its model as an artifact.

        Args:
            result: The round's
                :class:`~POMDPPlanners.training.model_learning.dagger_trainer.RoundResult`.
        """
        self._ensure_run()
        step = int(result.round_index)
        metrics: Dict[str, float] = {"dataset_size": float(result.dataset_size)}
        for source, count in result.source_counts.items():
            metrics[f"transitions_{source}"] = float(count)
        for name, value in result.diagnostics.items():
            metrics[name] = float(value)
        for name, values in result.training_metrics.items():
            if values:
                # The last epoch's value: the fit that was actually used.
                metrics[f"train_{name}"] = float(values[-1])
        if result.control is not None:
            metrics["mean_return"] = result.control.mean_return
            metrics["return_standard_error"] = result.control.standard_error
            metrics["cumulative_transitions"] = float(result.control.cumulative_transitions)
        run_id = self._run.info.run_id
        for name, value in metrics.items():
            # A NaN metric is rejected by some backends and silently plotted as a
            # gap by others; both hide that the quantity was never measured.
            if np.isfinite(value):
                self._client.log_metric(run_id, name, float(value), step=step)

        fingerprint = getattr(result.model, "fingerprint", None)
        model_artifact = self._log_model(result.model, step)
        self._rounds.append(
            {
                "round_index": step,
                "dataset_size": int(result.dataset_size),
                "source_counts": dict(result.source_counts),
                "diagnostics": {k: float(v) for k, v in result.diagnostics.items()},
                # The full per-epoch curves, not just their last value: they are
                # what the training plots are drawn from, and re-fitting to
                # recover them would not reproduce the fit that was scored.
                "training_metrics": {
                    name: [float(value) for value in values]
                    for name, values in result.training_metrics.items()
                },
                "metrics": {k: float(v) for k, v in metrics.items()},
                "model_fingerprint": fingerprint,
                "model_artifact": model_artifact,
                "control": None if result.control is None else result.control.to_dict(),
            }
        )

    def log_curve(self, curve: LearningCurve) -> None:
        """Record the finished curve and the round it should be reported at.

        The best round is logged rather than the last one, because that is the
        round the guarantee covers and the sequence is not monotone. This closes
        the run: a curve is the last thing a ``(method, seed)`` produces.

        Args:
            curve: The run's curve, from ``trainer.learning_curve(method)``.
        """
        self._ensure_run()
        run_id = self._run.info.run_id
        self._curves.append(curve)
        self._log_json_artifact(curve.to_dict(), "learning_curve.json")
        best = best_point(curve)
        if best is None:
            self._logger.warning(
                "no evaluated round for method %s seed %d: nothing to report",
                curve.method,
                curve.seed,
            )
        else:
            self._client.log_metric(run_id, "best_round_index", float(best.round_index))
            self._client.log_metric(run_id, "best_round_mean_return", best.mean_return)
            self._client.log_metric(
                run_id, "best_round_transitions", float(best.cumulative_transitions)
            )
            self._log_chosen_model(best.round_index)
        self.finish()

    def _log_model(self, model: Any, round_index: int) -> Optional[str]:
        """Save the round's model and log it under ``models/``.

        Returns:
            The artifact path, or ``None`` when the model cannot save itself --
            in which case the round's numbers still land, and the log says which
            model is missing rather than the run looking complete.
        """
        save = getattr(model, "save", None)
        if not callable(save):
            self._logger.warning(
                "round %d: %s has no save(); its metrics are logged but the model is not",
                round_index,
                type(model).__name__,
            )
            return None
        suffix = ".npz" if type(model).__name__ == "LinearGaussianTransition" else ".pt"
        with tempfile.TemporaryDirectory() as staging:
            path = str(save(Path(staging) / f"round_{round_index}{suffix}"))
            # save() may append its own suffix (np.savez does), so log what exists.
            written = Path(path) if Path(path).exists() else Path(f"{path}.npz")
            self._client.log_artifact(
                self._run.info.run_id, str(written), MODELS_ARTIFACT_PATH
            )
            return f"{MODELS_ARTIFACT_PATH}/{written.name}"

    def _log_chosen_model(self, round_index: int) -> None:
        """Copy the best round's model to ``models/chosen`` -- the one to load.

        Every round is kept because the sequence is not monotone and the best of
        it is known only at the end. That leaves a directory of equally-named
        candidates, and picking from it by eye is how the last round gets used
        for being last. The choice is made once, here, by the same rule the
        table reports.

        Args:
            round_index: The round :func:`best_point` selected.
        """
        source = next(
            (
                entry.get("model_artifact")
                for entry in self._rounds
                if entry["round_index"] == round_index and entry.get("model_artifact")
            ),
            None,
        )
        if source is None:
            self._logger.warning(
                "round %d has no saved model, so none was marked chosen", round_index
            )
            return
        run_id = self._run.info.run_id
        with tempfile.TemporaryDirectory() as staging:
            downloaded = Path(
                self._client.download_artifacts(run_id, source, staging)
            )
            chosen = Path(staging) / f"chosen{downloaded.suffix}"
            shutil.copyfile(downloaded, chosen)
            self._client.log_artifact(run_id, str(chosen), MODELS_ARTIFACT_PATH)
        self._client.log_param(run_id, "chosen_round", round_index)

    def _log_reports(self) -> None:
        """Write the readable tables beside the JSON and log them.

        JSON is what a plot is rebuilt from; a person deciding whether to plan
        with a model reads the table.
        """
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.training.model_learning.reporting import write_reports

        with tempfile.TemporaryDirectory() as staging:
            for path in write_reports(self._rounds, Path(staging), curves=self._curves).values():
                self._client.log_artifact(
                    self._run.info.run_id, str(path), EVALUATION_ARTIFACT_PATH
                )

    def _log_plots(self) -> None:
        """Draw the fit's diagnostic plots and log them under ``plots/``.

        Drawn from the same records that were just written, so a plot in the run
        and a plot redrawn later from ``rounds.json`` are the same figure. A
        failure here loses pictures, not data, so it is logged and swallowed
        rather than taking the run down at its last step.
        """
        # Deferred: matplotlib is a heavy import and the loop does not need it.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.utils.visualization.model_learning_plots import (
            plot_model_learning_report,
        )

        try:
            with tempfile.TemporaryDirectory() as staging:
                written = plot_model_learning_report(self._rounds, Path(staging))
                for path in written.values():
                    self._client.log_artifact(
                        self._run.info.run_id, str(path), EVALUATION_ARTIFACT_PATH
                    )
        except Exception as error:  # pylint: disable=broad-except
            self._logger.warning("could not draw the model-learning plots: %s", error)

    def _log_json_artifact(self, payload: Any, filename: str) -> None:
        """Write ``payload`` as JSON and log it under the evaluation artifacts."""
        with tempfile.TemporaryDirectory() as staging:
            path = Path(staging) / filename
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._client.log_artifact(
                self._run.info.run_id, str(path), EVALUATION_ARTIFACT_PATH
            )


def log_study_comparison(
    experiment_name: str,
    curves: Sequence[LearningCurve],
    params: Optional[Dict[str, Any]] = None,
    tracking_uri: Optional[str] = None,
    logger: Optional[Any] = None,
) -> List[str]:
    """Add the across-run comparison to every run's ``evaluation/``.

    The learning curve and the method table are what actually decide whether a
    model is worth using, and they are only known once every run has finished.
    They are copied into each run rather than parked in a run of their own: a
    reader opens one run's ``evaluation/`` and has everything the decision
    needs, instead of holding two directories in their head.

    Args:
        experiment_name: The experiment the runs were logged under.
        curves: Curves across methods and seeds.
        params: What was run -- environment, learner, planner budget, commit --
            pinned to every run, because a result whose settings live only in
            the script that produced it is not a result.
        tracking_uri: MLflow tracking URI. ``None`` uses the ambient setting.
        logger: Optional logger.

    Returns:
        The run ids that were updated.
    """
    # pylint: disable-next=import-outside-toplevel
    from mlflow.tracking import MlflowClient

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.training.model_learning.reporting import curve_table_markdown

    log = logger or get_logger(__name__)
    if not any(curve.points for curve in curves):
        log.warning("log_study_comparison: no evaluated rounds, nothing to compare")
        return []

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        log.warning("log_study_comparison: no experiment named %s", experiment_name)
        return []
    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        log.warning("log_study_comparison: experiment %s has no runs", experiment_name)
        return []

    with tempfile.TemporaryDirectory() as staging:
        table = Path(staging) / "methods.md"
        table.write_text(curve_table_markdown(curves), encoding="utf-8")
        files = [table]

        curves_json = Path(staging) / "learning_curves.json"
        curves_json.write_text(
            json.dumps([curve.to_dict() for curve in curves], indent=2), encoding="utf-8"
        )
        files.append(curves_json)

        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.utils.visualization.model_learning_plots import plot_learning_curves

        plot = plot_learning_curves(curves, Path(staging) / "learning_curves.png")
        if plot is not None:
            files.append(plot)

        updated: List[str] = []
        for run in runs:
            run_id = run.info.run_id
            for key, value in (params or {}).items():
                if key not in run.data.params:
                    client.log_param(run_id, key, value)
            for path in files:
                client.log_artifact(run_id, str(path), EVALUATION_ARTIFACT_PATH)
            updated.append(run_id)
    return updated


def load_round_models(run_artifacts_dir: Path) -> Dict[int, Path]:
    """Map round index to the saved model file, for a downloaded run's artifacts.

    Args:
        run_artifacts_dir: The run's artifact directory, containing ``models/``.

    Returns:
        ``{round_index: path}``, so the round a curve names can be loaded without
        guessing at file names.
    """
    models_dir = Path(run_artifacts_dir) / MODELS_ARTIFACT_PATH
    found: Dict[int, Path] = {}
    for path in sorted(models_dir.glob("round_*")):
        index = path.stem.split("_")[-1]
        if index.isdigit():
            found[int(index)] = path
    return found


def curve_summaries(curves: Sequence[LearningCurve]) -> Dict[str, Dict[str, float]]:
    """Best round per curve, keyed by ``method_seed`` -- the table to read first.

    Args:
        curves: Curves across methods and seeds.

    Returns:
        Per curve: the best round, its return and what it cost in transitions.
    """
    summaries: Dict[str, Dict[str, float]] = {}
    for curve in curves:
        best = best_point(curve)
        if best is None:
            continue
        summaries[f"{curve.method}_{curve.seed}"] = {
            "best_round_index": float(best.round_index),
            "mean_return": best.mean_return,
            "standard_error": best.standard_error,
            "cumulative_transitions": float(best.cumulative_transitions),
        }
    return summaries
