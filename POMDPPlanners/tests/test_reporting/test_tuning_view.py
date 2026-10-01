# SPDX-License-Identifier: MIT

"""Tests for the results site's view of a hyperparameter tuning study.

The fixture writes the three runs a study leaves -- the study, one tuned
planner, and the fresh evaluation nested under the study -- by hand, with the
names :mod:`POMDPPlanners.core.simulation.tuning_run_layout` fixes. Running a
real Optuna study here would take minutes and test the optimizer, which has
its own tests; these test that the site reads what the optimizer writes.
"""

import json
import os
from pathlib import Path
from typing import Dict

import pytest
from mlflow.tracking import MlflowClient

from POMDPPlanners.core.simulation import tuning_run_layout as layout
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.reporting import tuning
from POMDPPlanners.reporting.server import Router
from POMDPPlanners.reporting.store import RunIndex
from POMDPPlanners.tests.test_reporting.test_server import _episode
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_light_dark_pinned_kwargs,
)

ENV = "LightDark"
POLICY = "PFT_DPW"

SUMMARY = {
    "planner": "PFT_DPW",
    "policy_name": POLICY,
    "environment": "ContinuousLightDarkPOMDP",
    "environment_name": ENV,
    "objectives": [{"metric": "average_return", "direction": "maximize"}],
    "search_space": [
        {"name": "exploration_constant", "kind": "numerical", "low": 0.0, "high": 100.0},
        {"name": "depth", "kind": "numerical", "low": 2, "high": 10},
        {"name": "rollout", "kind": "categorical", "choices": ["random", "greedy"]},
    ],
    "best_parameters": {"exploration_constant": 75.0, "depth": 10, "rollout": "greedy"},
    "best_trial_number": 2,
    "best_pareto_score": 0.0,
    "best_trial_metrics": {"average_return": 9.5},
    "pareto_trial_numbers": [2],
    "pareto_scores": {"2": 0.0},
    "n_trials_budget": 300,
    "n_trials_completed": 3,
    "n_trials_started": 3,
    "early_stopping": {"patience": 1, "min_trials": 2, "min_relative_improvement": 0.001},
    "early_stopping_fired": True,
    "stopped_at_trial": 3,
    "front_quality_history": [[2, 1.0], [3, 1.0]],
    "episodes_per_trial": 5,
    "steps_per_episode": 20,
    "optimization_time_seconds": 42.0,
    "confidence_interval_level": 0.95,
}

TRIALS = [
    {
        "number": number,
        "state": "COMPLETE",
        "params": {"exploration_constant": c, "depth": d, "rollout": "greedy"},
        "objective_values": {"average_return": ret},
        "metric_statistics": {"average_return": [ret, ret - 1, ret + 1]},
        "duration_seconds": 3.0,
        "is_pareto": number == 2,
        "confidence_interval_level": 0.95,
    }
    for number, c, d, ret in ((0, 10.0, 3, 1.25), (1, 50.0, 5, 4.75), (2, 75.0, 10, 9.5))
]


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(name="study_dir")
def study_dir_fixture(tmp_path: Path) -> Dict[str, object]:
    """A store holding one study, laid out as ``optimize_and_evaluate`` leaves it."""
    # pylint: disable-next=import-outside-toplevel
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "study" / "mlruns"
    store.mkdir(parents=True)
    mlflow.set_tracking_uri(f"file://{store}")
    mlflow.set_experiment("tuning_view_test")
    staging = tmp_path / "staging"

    with mlflow.start_run(
        run_name="optimize_batch_1_configs", tags={layout.RUN_KIND_TAG: layout.RUN_KIND_STUDY}
    ) as study_run:
        mlflow.log_param("num_configurations", 1)
        with mlflow.start_run(
            run_name="config_1_ContinuousLightDarkPOMDP_PFT_DPW",
            nested=True,
            tags={layout.RUN_KIND_TAG: layout.RUN_KIND_CONFIG},
        ) as config_run:
            mlflow.log_params(
                {
                    "best_exploration_constant": 75.0,
                    "best_depth": 10,
                    "best_rollout": "greedy",
                    "param_range_exploration_constant": "0.0-100.0",
                    "param_range_depth": "2-10",
                    "param_range_rollout": "choices: ['random', 'greedy']",
                    "policy_type": "PFT_DPW",
                    "num_episodes": 5,
                }
            )
            mlflow.log_metric("best_trial_average_return", 9.5)
            mlflow.log_metric("best_trial_average_return_ci_lower", 8.5)
            mlflow.log_metric("best_trial_average_return_ci_upper", 10.5)
            tuning_files = staging / "tuning"
            _write(tuning_files / layout.STUDY_SUMMARY_FILE, SUMMARY)
            _write(tuning_files / layout.TRIAL_RECORDS_FILE, TRIALS)
            (tuning_files / "pareto_front.png").write_bytes(b"\x89PNG-stand-in")
            for item in sorted(tuning_files.iterdir()):
                mlflow.log_artifact(str(item), layout.TUNING_ARTIFACT_DIR)

        with mlflow.start_run(run_name="environment_policy_comparison", nested=True) as eval_run:
            mlflow.log_param("env_0_name", ENV)
            mlflow.log_param("env_0_policy_0_name", POLICY)
            mlflow.log_param("env_0_policy_0_type", "PFT_DPW")
            mlflow.log_metric(f"{ENV}_{POLICY}_average_return", 6.25)
            mlflow.log_metric(f"{ENV}_{POLICY}_average_return_ci_lower", 5.0)
            mlflow.log_metric(f"{ENV}_{POLICY}_average_return_ci_upper", 7.5)
            env = ContinuousLightDarkPOMDP(
                discount_factor=0.95, **continuous_light_dark_pinned_kwargs()
            )
            viz = staging / ENV / POLICY / "visualizations"
            viz.mkdir(parents=True)
            env.episode_visualizer().write(
                history=_episode(3), output_dir=viz, episode_index=0, policy_name=POLICY
            )
            mlflow.log_artifact(str(staging / ENV / POLICY), ENV)

    client = MlflowClient()
    client.set_tag(eval_run.info.run_id, layout.RUN_KIND_TAG, layout.RUN_KIND_EVALUATION)
    client.set_tag(config_run.info.run_id, layout.EVALUATION_RUN_ID_TAG, eval_run.info.run_id)

    return {
        "root": tmp_path,
        "study": study_run.info.run_id,
        "config": config_run.info.run_id,
        "evaluation": eval_run.info.run_id,
    }


@pytest.fixture(name="router")
def router_fixture(study_dir) -> Router:
    """A router over the study store."""
    return Router(RunIndex([study_dir["root"]]))


def _page(router: Router, run_id: str) -> str:
    experiment = router.index.experiments[0]
    status, _, body = router.resolve(f"/run/0/{experiment.experiment_id}/{run_id}")
    assert status == 200
    return body.decode("utf-8")


def test_load_study_reads_the_summary_the_records_and_the_link(router: Router, study_dir):
    """The study is read from the files the optimizer copies into the config run.

    Given: A config run with a study summary, trial records, a plot and a tag
        naming its evaluation run.
    When: The study is loaded.
    Then: Every field the view needs comes back, including early stopping and
        the evaluation run's id.
    """
    run = router.index.run(0, router.index.experiments[0].experiment_id, study_dir["config"])
    assert run is not None and tuning.is_tuning_config_run(run)

    study = tuning.load_study(run)

    assert study.has_summary
    assert study.planner == "PFT_DPW" and study.environment_name == ENV
    assert study.n_trials_completed == 3 and study.n_trials_budget == 300
    assert study.early_stopping_fired is True and study.stopped_at_trial == 3
    assert [t.number for t in study.trials] == [0, 1, 2]
    assert study.pareto_trial_numbers == [2]
    assert study.plots == ["tuning/pareto_front.png"]
    assert study.best_trial_scores["average_return"] == (9.5, 8.5, 10.5)
    assert study.evaluation_run_id == study_dir["evaluation"]
    depth = next(p for p in study.parameters if p.name == "depth")
    assert depth.best == 10 and depth.position == 1.0


def test_the_tuning_view_shows_the_study_and_its_fresh_evaluation(router: Router, study_dir):
    """A tuned planner's page is the study, not a table of params.

    Purpose: The study's figures used to show only as ``best_*`` and
    ``param_range_*`` rows in the parameter table, and the evaluation was not
    reachable from it at all.

    Given: The fixture study.
    When: The config run's page renders.
    Then: It shows the summary, the parameters against their ranges, the trial
        table with the best and Pareto trial marked, the plot, the best trial's
        score beside the fresh evaluation's, and a card for the evaluation
        episode that opens its replay.
    """
    page = _page(router, study_dir["config"])

    assert "3 of 300" in page
    assert "stopped after 3 trials" in page
    assert "#2" in page
    # Parameters against their ranges, the chosen depth pinned to the top end.
    assert "0 – 100" in page and "one of random, greedy" in page
    assert 'style="left:100.0%"' in page
    # The trial table marks the chosen trial and the Pareto front.
    assert "<tr class=best>" in page
    assert ">Best<" in page and ">Pareto<" in page
    assert "4.75" in page and "1.25" in page
    # The diagnostics are drawn on the page from the records, not shown as
    # the PNGs; those stay in the run and are only linked.
    assert 'class="chart tuning-chart"' in page
    assert "/static/chart-tips.js" in page
    assert "<img" not in page.split("Diagnostic charts", 1)[1].split("<h2>Evaluation", 1)[0]
    assert "tuning/pareto_front.png" in page
    assert "Pareto-front quality (early stopping)" in page
    # Tuning score beside the fresh evaluation, each with its interval.
    assert "Best trial (5 episodes)" in page and "Evaluation (1 episodes)" in page
    assert "9.5" in page and "6.25" in page and "5 – 7.5" in page
    # Its one episode shares its seed with every trial's episode 0, and says so.
    assert "Evaluation episodes 0–0 use the same seeds" in page
    # The evaluation's episode, with its replay one click away.
    assert f"/env/{ENV}/policy/{POLICY}/episode/0" in page
    assert 'class="thumb thumb-scene"' in page
    # Params already shown above are not repeated in the raw table.
    assert "param_range_depth" not in page


def test_a_study_run_lists_its_tuned_planners_and_evaluation(router: Router, study_dir):
    """The study run is where a reader lands, so it links both kinds of child.

    Given: The study run.
    When: Its page renders.
    Then: It lists the config run and the evaluation run, each labelled.
    """
    page = _page(router, study_dir["study"])

    assert "Runs in this study" in page
    assert "config_1_ContinuousLightDarkPOMDP_PFT_DPW" in page
    assert "Tuned planner" in page
    assert "Evaluation of the tuned planners" in page


def test_the_evaluation_run_links_back_to_its_study(router: Router, study_dir):
    """The evaluation is nested under the study, and its page says so."""
    page = _page(router, study_dir["evaluation"])

    assert "part of" in page and "optimize_batch_1_configs" in page


def test_a_config_run_without_a_summary_still_gets_a_tuning_view(tmp_path: Path):
    """Runs logged before the summary existed get a reduced view, not a crash.

    Given: A config run with only the ``best_*`` and ``param_range_*`` params
        and a ``best_trial_*`` metric, as older runs have.
    When: Its page renders.
    Then: The view shows the parameters and the best trial's score, and says
        that there are no trial records and no linked evaluation.
    """
    # pylint: disable-next=import-outside-toplevel
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "old" / "mlruns"
    store.mkdir(parents=True)
    mlflow.set_tracking_uri(f"file://{store}")
    mlflow.set_experiment("old_tuning")
    with mlflow.start_run(run_name="config_1_TigerPOMDP_POMCP") as run:
        mlflow.log_params(
            {
                "best_depth": 4,
                "param_range_depth": "1-5",
                "policy_type": "POMCP",
                "environment_type": "TigerPOMDP",
                "parameters_to_optimize": "[('average_return', 'maximize')]",
                "n_trials": 10,
            }
        )
        mlflow.log_metric("best_trial_average_return", 3.5)

    router = Router(RunIndex([tmp_path]))
    page = _page(router, run.info.run_id)

    assert "predates the study summary" in page
    assert "1-5" in page and "3.5" in page and "budget 10" in page
    assert "carries no trial records" in page
    assert "No evaluation run is linked" in page
