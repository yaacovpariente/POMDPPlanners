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


def test_the_experiment_lists_a_study_once_with_its_key_facts(router: Router, study_dir):
    """One study is one card, not one card per MLflow run.

    Purpose: Listed per run, a study read as three cards -- the parent, the
    tuned planner and the evaluation -- two of them saying "Episodes 0".

    Given: An experiment holding one study with one tuned planner.
    When: The experiment page renders.
    Then: One entry, titled for the planner and environment, linking straight
        to the tuning view, carrying the trials run, early stopping, the best
        trial's and the evaluation's score; no raw run name is a card title,
        and no "Episodes" field appears.
    """
    experiment = router.index.experiments[0]
    status, _, body = router.resolve(f"/experiment/0/{experiment.experiment_id}")
    page = body.decode("utf-8")
    cards = page.split('<div class="cards">', 1)[1].split('<div class="scroll table-view"', 1)[0]

    assert status == 200
    assert cards.count('<a class="card') == 1
    assert "PFT_DPW on LightDark — tuning study" in cards
    assert f'/run/0/{experiment.experiment_id}/{study_dir["config"]}"' in cards
    assert "3 of 300" in cards and "stopped after 3 trials" in cards
    assert "average_return 9.5" in cards
    assert "average_return 6.25 (1 episodes)" in cards
    assert "environment_policy_comparison" not in cards
    assert "Episodes" not in page.split("<main", 1)[1]
    # The table view groups the same way.
    table = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert table.count("<tr>") == 1


def test_the_index_counts_studies_not_their_runs(router: Router):
    """The landing page counts the study once."""
    status, _, body = router.resolve("/")
    page = body.decode("utf-8")

    assert status == 200
    assert "Tuning studies" in page and ">1<" in page
    assert "Other runs" not in page


def test_a_study_run_opens_the_study_page(router: Router, study_dir):
    """The study run's own page lists its tuned planners and links the evaluation.

    Given: The study run.
    When: Its page renders.
    Then: It lists the tuned planner with a summary of its chosen parameters
        and both scores, and links the raw evaluation run.
    """
    page = _page(router, study_dir["study"])

    assert "Tuned planners" in page
    assert "PFT_DPW on LightDark" in page
    assert "exploration_constant=75, depth=10, rollout=greedy" in page
    assert f'{study_dir["evaluation"]}"' in page and "raw evaluation run" in page


def test_the_tuning_view_states_each_fact_once(router: Router, study_dir):
    """Facts the view already shows elsewhere are not repeated in the summary.

    Given: The tuning view.
    When: It renders.
    Then: No summary stat repeats the best trial, the Pareto count or the
        episodes per trial, and the evaluation section has no planner card
        restating the scores the comparison already gives.
    """
    page = _page(router, study_dir["config"])
    stats = page.split('<div class="stats">', 1)[1].split("</div></div>", 1)[0]
    evaluation = page.split("<h2>Evaluation</h2>", 1)[1]

    assert "Pareto trials" not in stats and "Best trial" not in stats
    assert "Episodes per trial" not in stats
    # The title already names the planner and environment.
    assert "Planner" not in stats and "Environment" not in stats
    assert "planner class PFT_DPW" not in page
    assert "environment class ContinuousLightDarkPOMDP" in page
    assert "Avg. return" not in evaluation
    assert "<h1>PFT_DPW on LightDark" in page


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


def test_a_study_of_several_planners_gets_a_study_page_card(tmp_path: Path):
    """A study that tuned several planners opens its study page, which lists each.

    Given: A study run with two tuned planners and no evaluation.
    When: The experiment and study pages render.
    Then: One card titled for two planners, linking to the study page, which
        has a row per planner linking to its tuning view.
    """
    # pylint: disable-next=import-outside-toplevel
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "multi" / "mlruns"
    store.mkdir(parents=True)
    mlflow.set_tracking_uri(f"file://{store}")
    mlflow.set_experiment("multi_study")
    config_ids = []
    with mlflow.start_run(
        run_name="optimize_batch_2_configs", tags={layout.RUN_KIND_TAG: layout.RUN_KIND_STUDY}
    ) as study_run:
        for index, planner in enumerate(("POMCP", "PFT_DPW")):
            with mlflow.start_run(
                run_name=f"config_{index + 1}_Tiger_{planner}",
                nested=True,
                tags={layout.RUN_KIND_TAG: layout.RUN_KIND_CONFIG},
            ) as config_run:
                mlflow.log_params({"best_depth": 3, "param_range_depth": "1-5"})
                mlflow.log_metric("best_trial_average_return", 1.5 + index)
                summary = dict(
                    SUMMARY, planner=planner, policy_name=planner, environment_name="Tiger"
                )
                tuning_files = tmp_path / f"staging{index}" / "tuning"
                _write(tuning_files / layout.STUDY_SUMMARY_FILE, summary)
                mlflow.log_artifact(str(tuning_files / layout.STUDY_SUMMARY_FILE), "tuning")
                config_ids.append(config_run.info.run_id)

    router = Router(RunIndex([tmp_path]))
    experiment = router.index.experiments[0]
    _, _, body = router.resolve(f"/experiment/0/{experiment.experiment_id}")
    listing = body.decode("utf-8")
    study_page = _page(router, study_run.info.run_id)

    cards = listing.split('<div class="cards">', 1)[1].split('<div class="scroll table-view"', 1)[0]
    assert cards.count('<a class="card') == 1
    assert "Tuning study of 2 planners" in cards
    assert f'{study_run.info.run_id}"' in cards
    assert "POMCP on Tiger" in study_page and "PFT_DPW on Tiger" in study_page
    for config_id in config_ids:
        assert f'/{config_id}"' in study_page


def test_trials_and_episodes_are_collapsed_and_closed(router: Router, study_dir):
    """The trial table and the evaluation's episodes sit in closed sections.

    Purpose: A study can run thousands of trials and an evaluation hundreds
    of episodes; laid open, either buries the rest of the page.

    Given: The fixture study, with three trials and one evaluation episode.
    When: The tuning view renders.
    Then: Each is in a <details> that is not open, labelled with its count;
        both are small, so neither is deferred.
    """
    page = _page(router, study_dir["config"])

    assert "<details><summary>All trials (3)</summary>" in page
    assert "<details><summary>Episodes (1)</summary>" in page
    assert "data-lazy" not in page


def test_all_evaluation_metrics_sit_under_the_comparison(router: Router, study_dir):
    """The full evaluation metrics follow the table whose column they expand.

    Given: The tuning view.
    When: It renders.
    Then: "All evaluation metrics" comes after the comparison heading and
        before "Chosen parameters", and appears once.
    """
    page = _page(router, study_dir["config"])
    between = page.split("<h2>Best trial against the evaluation</h2>", 1)[1].split(
        "<h2>Chosen parameters</h2>", 1
    )[0]

    assert "<summary>All evaluation metrics</summary>" in between
    assert page.count("All evaluation metrics") == 1


def test_large_sections_are_built_only_when_opened(router: Router, study_dir, monkeypatch):
    """Past their limits, the trial rows and episode cards wait in a <template>.

    Given: Limits lowered below the fixture's counts.
    When: The tuning view renders.
    Then: Both sections are deferred, the trial rows and the scene scripts are
        inside their templates, and the loader is on the page.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.reporting import pages

    monkeypatch.setattr(pages, "TRIALS_INLINE_LIMIT", 1)
    monkeypatch.setattr(pages, "EPISODES_INLINE_LIMIT", 0)
    page = _page(router, study_dir["config"])

    trials = page.split("<summary>All trials (3)</summary>", 1)[1].split("</details>", 1)[0]
    episodes = page.split("<summary>Episodes (1)</summary>", 1)[1].split("</details>", 1)[0]
    assert page.count("<details data-lazy>") == 2
    assert trials.startswith("<template>") and "<tr class=best>" in trials
    assert episodes.startswith("<template>")
    assert "/static/viewer/scene-cards.js" in episodes
    assert "/static/lazy-details.js" in page


def test_five_thousand_trials_stay_out_of_the_layout():
    """A 5000-trial study's table is deferred and built quickly.

    Given: A synthetic study of 5000 trials.
    When: Its trial section is rendered.
    Then: It is deferred into a <template>, holds one row per trial, and takes
        well under a second to write.
    """
    # pylint: disable-next=import-outside-toplevel
    import time

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.reporting import pages

    trials = [
        tuning.Trial(n, "COMPLETE", {"depth": n % 9}, {"average_return": n / 7}, {}, 1.0, False)
        for n in range(5000)
    ]
    study = tuning.TuningStudy(
        planner="P",
        environment="E",
        environment_name="E",
        policy_name="P",
        objectives=[("average_return", "maximize")],
        parameters=[tuning.SearchParameter(name="depth", best=8, low=0, high=8)],
        best_trial_number=4999,
        best_trial_scores={},
        trials=trials,
    )
    start = time.perf_counter()
    # pylint: disable-next=protected-access
    section = pages._trials_section(study)
    elapsed = time.perf_counter() - start

    assert section.startswith("<details data-lazy><summary>All trials (5000)</summary><template>")
    assert section.count("<tr") == 5001
    assert elapsed < 1.0


def test_run_parameters_are_grouped_by_prefix():
    """A run's params read as one collapsed table per group, prefixes stripped.

    Given: Params from each group, a private ``env__`` one, and the tuning
        view's best and range params.
    When: They are grouped with the tuning view's omissions.
    Then: Environment, Policy, Fixed planner settings and Run setup each get a
        closed table with their own count, names lose their prefix, best and
        range params are left out, and a group with nothing is not shown.
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.reporting.pages import run_parameters

    params = {
        "env_discount_factor": "0.95",
        "env__hazard_terminal_enabled": "True",
        "policy_depth": "3",
        "constant_time_out_in_seconds": "1",
        "n_trials": "2",
        "num_steps": "30",
        "best_depth": "3",
        "param_range_depth": "2-10",
    }
    grouped = run_parameters(params, omit_prefixes=("best_", "param_range_"))

    assert '<details class="param-group"><summary>Environment parameters (2)</summary>' in grouped
    assert "<summary>Policy parameters (1)</summary>" in grouped
    assert "<summary>Fixed planner settings parameters (1)</summary>" in grouped
    assert "<summary>Run setup parameters (2)</summary>" in grouped
    assert "<td>hazard_terminal_enabled</td>" in grouped
    assert "<td>discount_factor</td>" in grouped and "env_discount_factor" not in grouped
    assert "best_depth" not in grouped and "param_range" not in grouped
    assert run_parameters({"policy_x": "1"}).count("<details") == 1


def test_a_plain_run_page_keeps_every_param(router: Router, study_dir):
    """Outside the tuning view nothing is dropped: unmatched params go to Run setup.

    Given: The evaluation run, whose params are env_0_* and run settings.
    When: Its page renders.
    Then: The env_0_* params are under Environment without the prefix, and
        the rest are under Run setup.
    """
    page = _page(router, study_dir["evaluation"])

    assert "<summary>Environment parameters (3)</summary>" in page
    assert "<td>0_policy_0_name</td>" in page
    assert "<summary>Run parameters</summary>" not in page


def test_comparison_params_are_all_kept():
    """A comparison run's params all appear, the unprefixed ones under Run setup."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.reporting.pages import run_parameters

    params = {
        "alpha": "0.05",
        "n_jobs": "5",
        "num_environments": "1",
        "env_0_name": "RockSample",
        "env_0_num_episodes": "4",
    }
    grouped = run_parameters(params)

    assert "<summary>Environment parameters (2)</summary>" in grouped
    assert "<summary>Run setup parameters (3)</summary>" in grouped
    assert grouped.count("<tr><td>") == len(params)
