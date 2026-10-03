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
    # The diagnostics are drawn on the page from the records; the PNGs stay
    # in the run and are neither shown nor listed.
    assert 'class="chart tuning-chart"' in page
    assert "/static/chart-tips.js" in page
    assert "<img" not in page.split("Diagnostic charts", 1)[1].split("<h2>Evaluation", 1)[0]
    assert "pareto_front.png" not in page
    assert "Pareto-front quality (early stopping)" in page
    # Tuning score beside the fresh evaluation, each with its interval.
    assert "Best trial (5 episodes)" in page and "Evaluation (1 episodes)" in page
    assert "9.5" in page and "6.25" in page and "5 – 7.5" in page
    # The comparison stands without explanatory notes under it.
    assert "noisy scores" not in page and "same seeds" not in page
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


def test_the_evaluation_episodes_offer_the_planner_pages_live_switch(router: Router, study_dir):
    """The tuning view's episode cards can all be played at once, as on a planner page.

    Purpose: The tuning view lists the evaluation's episodes with the same 3D
    cards as the planner page; the same Live switch must play them, and only
    them, so it belongs inside the Episodes section.

    Given: The fixture study, whose evaluation episode has a trace.
    When: The tuning view renders.
    Then: The Episodes section holds one Live switch, off, before the scene
        scripts; the page has no other.
    """
    page = _page(router, study_dir["config"])
    episodes = page.split("<summary>Episodes (1)</summary>", 1)[1].split("</details>", 1)[0]

    assert '<div class="episode-tools"><button type="button" class="tab" data-live' in episodes
    assert episodes.index("data-live") < episodes.index("/static/viewer/scene-cards.js")
    assert page.count("data-live") == 1


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
    # The Live switch travels with the cards it plays, ahead of the script
    # that binds it, so it works once the section is built.
    assert episodes.index('data-live aria-pressed="false"') < episodes.index("scene-cards.js")


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


def _two_by_two_study(tmp_path: Path):
    """A study of two planners on two environments, evaluated, as the optimizer leaves it."""
    # pylint: disable-next=import-outside-toplevel
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "two_by_two" / "mlruns"
    store.mkdir(parents=True)
    mlflow.set_tracking_uri(f"file://{store}")
    mlflow.set_experiment("two_by_two")
    scores = {
        ("Rocks", "POMCPOW"): (7.0, 1.0),
        ("Rocks", "PFT_DPW"): (12.0, 0.5),
        ("Push", "POMCPOW"): (-60.0, 0.5),
        ("Push", "PFT_DPW"): (-55.0, 0.75),
    }
    configs = {}
    with mlflow.start_run(
        run_name="optimize_batch_4_configs", tags={layout.RUN_KIND_TAG: layout.RUN_KIND_STUDY}
    ) as study_run:
        for index, (env, planner) in enumerate(scores):
            name = f"{planner}_{env}"
            with mlflow.start_run(
                run_name=f"config_{index + 1}_{env}_{planner}",
                nested=True,
                tags={layout.RUN_KIND_TAG: layout.RUN_KIND_CONFIG},
            ) as config_run:
                mlflow.log_params({"best_depth": 3, "param_range_depth": "1-5"})
                mlflow.log_metric("best_trial_average_return", scores[(env, planner)][0] + 1)
                summary = dict(
                    SUMMARY,
                    planner=planner,
                    policy_name=name,
                    environment=f"{env}POMDP",
                    environment_name=env,
                    objectives=[
                        {"metric": "average_return", "direction": "maximize"},
                        {"metric": "task_completion_rate", "direction": "maximize"},
                    ],
                )
                staged = tmp_path / f"staging{index}" / "tuning"
                _write(staged / layout.STUDY_SUMMARY_FILE, summary)
                mlflow.log_artifact(str(staged / layout.STUDY_SUMMARY_FILE), "tuning")
                configs[(env, planner)] = config_run.info.run_id
        with mlflow.start_run(run_name="environment_policy_comparison", nested=True) as eval_run:
            for env_index, env in enumerate(("Rocks", "Push")):
                mlflow.log_param(f"env_{env_index}_name", env)
                for policy_index, planner in enumerate(("POMCPOW", "PFT_DPW")):
                    name = f"{planner}_{env}"
                    mlflow.log_param(f"env_{env_index}_policy_{policy_index}_name", name)
                    ret, rate = scores[(env, planner)]
                    mlflow.log_metric(f"{env}_{name}_average_return", ret)
                    mlflow.log_metric(f"{env}_{name}_average_return_ci_lower", ret - 2)
                    mlflow.log_metric(f"{env}_{name}_average_return_ci_upper", ret + 2)
                    mlflow.log_metric(f"{env}_{name}_task_completion_rate", rate)
                    # A timing, where lower is better, and a planner counter
                    # whose better direction nobody can say.
                    mlflow.log_metric(f"{env}_{name}_average_action_time", 1.0 + policy_index)
                    mlflow.log_metric(
                        f"{env}_{name}_policy_info_tree_max_depth", 5.0 + policy_index
                    )
    client = MlflowClient()
    for run_id in configs.values():
        client.set_tag(run_id, layout.EVALUATION_RUN_ID_TAG, eval_run.info.run_id)
    client.set_tag(eval_run.info.run_id, layout.RUN_KIND_TAG, layout.RUN_KIND_EVALUATION)
    return Router(RunIndex([tmp_path])), study_run.info.run_id, configs


def test_the_study_page_compares_planners_per_environment(tmp_path: Path):
    """Each environment gets a block comparing its planners, first on the page.

    Purpose: Comparing planners meant finding the raw evaluation run.

    Given: Two planners evaluated on each of two environments.
    When: The study page renders.
    Then: Two comparison blocks with anchors come before the planner list;
        each has a row per planner linking to its tuning view, and the best
        evaluation return and task completion in each block are marked.
    """
    router, study_id, configs = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    rocks = page.split('id="compare-Rocks"', 1)[1].split("</section>", 1)[0]
    push = page.split('id="compare-Push"', 1)[1].split("</section>", 1)[0]

    assert page.index('id="compare-Rocks"') < page.index("<h2>Tuned planners</h2>")
    assert "Compare Tuned planners on Rocks" in rocks and "Compare Tuned planners on Push" in push
    for (env, _), run_id in configs.items():
        assert f'/{run_id}"' in (rocks if env == "Rocks" else push)
    # Rocks: PFT_DPW has the best evaluation return and best-trial score,
    # POMCPOW the best completion; each figure keeps its interval.
    assert 'class="num is-best" data-col="1" data-value="12.0"' in rocks
    assert 'class="num is-best" data-col="0" data-value="1.0"' in rocks
    assert 'class="num is-best" data-col="1" data-value="13.0"' in rocks
    # Return, completion, best-trial return and the lower action time.
    assert rocks.count("is-best") == 4
    assert 'is-best" data-col="1" data-value="-55.0"' in push
    assert 'is-best" data-col="1" data-value="0.75"' in push
    assert 'class="chart"' in rocks


def test_a_tuning_view_links_to_its_environment_comparison(tmp_path: Path):
    """A tuned planner's evaluation section links to the comparison on its environment."""
    router, study_id, configs = _two_by_two_study(tmp_path)
    page = _page(router, configs[("Push", "PFT_DPW")])

    assert "Compare with the other planners on Push" in page
    assert f'/{study_id}#compare-Push"' in page


def test_the_experiment_card_names_the_planners_and_environments(tmp_path: Path):
    """The study card says which planners on which environments, not just how many."""
    router, _, _ = _two_by_two_study(tmp_path)
    experiment = router.index.experiments[0]
    _, _, body = router.resolve(f"/experiment/0/{experiment.experiment_id}")
    page = body.decode("utf-8")
    cards = page.split('<div class="cards">', 1)[1].split('<div class="scroll table-view"', 1)[0]

    assert "Tuning study of 2 planners on 2 environments" in cards
    assert "POMCPOW, PFT_DPW" in cards
    assert ">Rocks<" in cards and ">Push<" in cards


def test_a_single_planner_view_has_no_comparison_link(router: Router, study_dir):
    """With no other planner on the environment there is nothing to compare."""
    page = _page(router, study_dir["config"])

    assert "Compare with the other planners" not in page


def test_the_comparison_lists_every_metric_with_directions_and_a_filter(tmp_path: Path):
    """Every logged metric is a row; "best" follows each metric's known direction.

    Given: Two planners per environment, with a timing and a planner counter
        logged beside return and task completion.
    When: The study page renders.
    Then: The Rocks table has a row per metric plus the best-trial and
        episodes rows, planners as columns; the lower action time is marked
        best, the counter is not marked at all, and the filter box and group
        toggles are there with each row tagged by group.
    """
    router, study_id, _ = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    rocks = page.split('id="compare-Rocks"', 1)[1].split("</section>", 1)[0]
    rows = {
        row.split('data-metric="', 1)[1].split('"', 1)[0]: row
        for row in rocks.split("<tr")[1:]
        if 'data-metric="' in row
    }

    assert set(rows) == {
        "average_return best trial",
        "episodes",
        "average_return",
        "task_completion_rate",
        "average_action_time",
        "policy_info_tree_max_depth",
    }
    assert 'is-best" data-col="0" data-value="1.0"' in rows["average_action_time"]
    assert 'data-direction="minimize"' in rows["average_action_time"]
    assert "lower is better" in rows["average_action_time"]
    assert "is-best" not in rows["policy_info_tree_max_depth"]
    assert 'data-direction=""' in rows["policy_info_tree_max_depth"]
    assert 'data-group="timing"' in rows["average_action_time"]
    assert 'data-group="policy"' in rows["policy_info_tree_max_depth"]
    assert 'type="search"' in rocks and 'data-group-toggle="timing"' in rocks
    assert page.count("/static/metric-filter.js") == 1


def test_metric_directions_are_known_only_where_the_name_says():
    """Directions come from the study's objectives, then from the metric's name."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.reporting.pages import metric_direction

    assert metric_direction("average_return") == "maximize"
    assert metric_direction("average_belief_update_time") == "minimize"
    assert metric_direction("average_dangerous_area_steps") == "minimize"
    assert metric_direction("policy_info_root_visit_count") is None
    assert metric_direction("average_rocks_sampled") is None
    assert metric_direction("average_rocks_sampled", [("average_rocks_sampled", "maximize")]) == (
        "maximize"
    )


def test_the_comparison_has_metric_and_planner_pickers(tmp_path: Path):
    """Dropdowns pick exactly which metric rows and planner columns show.

    Given: The two-by-two study.
    When: The study page renders.
    Then: Each block has a metric picker listing every row, grouped like the
        toggles, and a planner picker listing both planners; every planner
        cell carries its column so the page can hide it and re-mark "best".
    """
    router, study_id, _ = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    rocks = page.split('id="compare-Rocks"', 1)[1].split("</section>", 1)[0]
    metric_picker = rocks.split('data-picker="metric"', 1)[1].split("</details>", 1)[0]
    planner_picker = rocks.split('data-picker="planner"', 1)[1].split("</details>", 1)[0]

    assert "Metrics (6 of 6)" in metric_picker
    assert metric_picker.count("data-metric-choice=") == 6
    assert "<legend>Outcomes</legend>" in metric_picker
    assert "<legend>Timings</legend>" in metric_picker
    assert "<legend>Planner internals</legend>" in metric_picker
    assert "Planners (2 of 2)" in planner_picker
    assert 'data-choice-name="POMCPOW_Rocks"' in planner_picker
    assert 'data-choice-name="PFT_DPW_Rocks"' in planner_picker
    assert "data-picker-all" in metric_picker and "data-picker-none" in planner_picker
    assert '<th class="group" data-col="1" ' in rocks


def test_each_comparison_block_carries_its_own_chart_builder(tmp_path: Path):
    """The study page builds charts in place, one builder per environment.

    Given: The two-by-two study.
    When: The study page and a tuning view render.
    Then: Each comparison block holds a closed "Build a chart" section with a
        builder whose embedded data is that environment's planners; builders
        are found by role, not page-wide id, so nothing collides; the scripts
        load once, the shared kit first; the tuning view still links to the
        standalone builder.
    """
    # pylint: disable-next=import-outside-toplevel
    import json as json_module

    router, study_id, configs = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    view = _page(router, configs[("Push", "POMCPOW")])

    for env in ("Rocks", "Push"):
        block = page.split(f'id="compare-{env}"', 1)[1].split("</section>", 1)[0]
        builder = block.split("<summary>Build a chart</summary>", 1)[1]
        data = json_module.loads(
            builder.split('<script type="application/json" data-role="data">', 1)[1].split(
                "</script>", 1
            )[0]
        )
        assert data["environment"] == env
        assert {p["name"] for p in data["policies"]} == {f"POMCPOW_{env}", f"PFT_DPW_{env}"}
    assert page.count("data-chart-builder") == 2
    assert 'id="chart-' not in page
    assert page.count("/static/chart-builder.js") == 1
    assert page.index("/static/figure-kit.js") < page.index("/static/chart-builder.js")
    assert "Build a chart from this evaluation" in view and "/env/Push/chart" in view


def test_the_study_page_builds_tuning_charts_for_any_planner(tmp_path: Path):
    """A study-level tuning builder offers every tuned planner and fetches its trials.

    Given: The two-by-two study.
    When: The study page renders and one planner's builder data is fetched.
    Then: The closed "Build a tuning chart" section lists the four tuned
        planners, each with its environment and the URL of its data, and that
        URL answers with the planner's trial data as JSON.
    """
    # pylint: disable-next=import-outside-toplevel
    import json as json_module

    router, study_id, configs = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    section = page.split("<summary>Build a tuning chart</summary>", 1)[1].split("</details>", 1)[0]
    experiment = router.index.experiments[0]
    url = f"/run/0/{experiment.experiment_id}/{configs[('Push', 'PFT_DPW')]}/tuning-chart.json"

    assert section.count("<option value=") >= 4 + 4  # four planners, plus the chart kinds
    assert f'<option value="{url}" data-env="Push">PFT_DPW_Push on Push</option>' in section
    status, media_type, body = router.resolve(url)
    data = json_module.loads(body)
    assert status == 200 and media_type == "application/json"
    assert data["title"] == "PFT_DPW_Push on Push"


def test_a_study_of_several_environments_has_an_environment_filter(tmp_path: Path, study_dir):
    """Two or more environments get a filter; everything per environment is tagged.

    Given: The two-by-two study, and the single-environment fixture study.
    When: Their study pages render.
    Then: The first has an environment dropdown listing both, and its
        comparison blocks, planner cards and rows, and tuning-builder choices
        carry their environment; the second has no filter.
    """
    # Its own folder: the fixture study is under tmp_path too, and one index
    # over both would hold two experiments.
    router, study_id, _ = _two_by_two_study(tmp_path / "two_envs")
    page = _page(router, study_id)
    single = Router(RunIndex([study_dir["root"] / "study"]))

    picker = page.split('data-picker="environment"', 1)[1].split("</details>", 1)[0]
    assert "Environments (2 of 2)" in picker
    assert 'data-choice-name="Rocks"' in picker and 'data-choice-name="Push"' in picker
    assert page.index("data-env-filter") < page.index('id="compare-Rocks"')
    assert 'section class="env" id="compare-Push" data-env="Push"' in page
    assert page.count('<a class="card" href="') >= 4
    assert page.count(' data-env="Rocks"') >= 4  # block, card, row, builder choice
    assert "data-env-filter" not in _page(single, study_dir["study"])


def test_the_diagnostics_chart_builder_serves_the_study(router: Router, study_dir):
    """A tuned planner's diagnostics open in an editable chart builder.

    Given: The fixture study, with one objective and three trials.
    When: The tuning view and its builder page render.
    Then: The view links "Build a chart"; the builder carries every trial,
        the objectives, the parameters with their ranges, the chosen and
        Pareto trials and where early stopping fired, offers the chart kinds
        but no Pareto front for a single objective, and loads its scripts.
    """
    # pylint: disable-next=import-outside-toplevel
    import json as json_module

    experiment = router.index.experiments[0]
    view = _page(router, study_dir["config"])
    url = f"/run/0/{experiment.experiment_id}/{study_dir['config']}/tuning-chart"
    status, _, body = router.resolve(url)
    page = body.decode("utf-8")
    data = json_module.loads(
        page.split('<script type="application/json" data-role="data">', 1)[1].split("</script>", 1)[
            0
        ]
    )

    assert f'href="{url}"' in view
    assert status == 200
    assert [t["number"] for t in data["trials"]] == [0, 1, 2]
    assert data["objectives"] == [{"name": "average_return", "direction": "maximize"}]
    assert {p["name"]: (p["low"], p["high"]) for p in data["parameters"]}["depth"] == (2, 10)
    assert data["best"] == 2 and data["pareto"] == [2] and data["stopped_at"] == 3
    assert 'value="objective-history"' in page and 'value="parameter-slice"' in page
    assert 'value="pareto-front"' not in page
    assert page.index("/static/figure-kit.js") < page.index("/static/tuning-chart-builder.js")


def test_the_diagnostics_builder_offers_a_pareto_front_for_two_objectives(tmp_path: Path):
    """With two objectives, the builder offers the Pareto front."""
    router, _, configs = _two_by_two_study(tmp_path)
    experiment = router.index.experiments[0]
    run_id = configs[("Rocks", "PFT_DPW")]
    _, _, body = router.resolve(f"/run/0/{experiment.experiment_id}/{run_id}/tuning-chart")

    assert 'value="pareto-front"' in body.decode("utf-8")


def test_a_plain_run_has_no_diagnostics_builder(router: Router, study_dir):
    """The route answers 404 for a run that is not a tuning study."""
    experiment = router.index.experiments[0]
    status, _, _ = router.resolve(
        f"/run/0/{experiment.experiment_id}/{study_dir['evaluation']}/tuning-chart"
    )

    assert status == 404


def test_every_builder_defaults_to_the_colour_style(tmp_path: Path):
    """Each builder on the study page and its standalone pages opens in "Paper, colour"."""
    router, study_id, configs = _two_by_two_study(tmp_path)
    experiment = router.index.experiments[0]
    pages_to_check = [
        _page(router, study_id),
        router.resolve(
            f"/run/0/{experiment.experiment_id}/{configs[('Rocks', 'PFT_DPW')]}/tuning-chart"
        )[2].decode("utf-8"),
    ]
    evaluation = next(r for r in experiment.runs if r.run_name == "environment_policy_comparison")
    standalone = f"/run/0/{experiment.experiment_id}/{evaluation.run_id}/env/Rocks/chart"
    pages_to_check.append(router.resolve(standalone)[2].decode("utf-8"))

    for page in pages_to_check:
        styles = page.count('data-role="style"')
        assert styles >= 1
        assert page.count('<option value="colour" selected>Paper, colour</option>') == styles


def test_the_chart_builder_exports_the_plotted_metric_as_csv(tmp_path: Path):
    """Every evaluation builder offers the plotted metric's values as CSV."""
    router, study_id, _ = _two_by_two_study(tmp_path)
    page = _page(router, study_id)

    assert page.count('data-role="csv"') == 2
    assert "Download data (CSV)" in page


def test_a_tagged_evaluation_stays_reachable_when_its_links_failed(tmp_path: Path):
    """An evaluation tagged under the study, but named by no config, is still linked.

    Given: The two-by-two study with the configs' evaluation tags removed, as
        when writing them failed.
    When: The experiment and study pages render.
    Then: The evaluation run is still folded into the study card, and the
        study page links it as the raw evaluation run.
    """
    router, study_id, configs = _two_by_two_study(tmp_path)
    client = MlflowClient()
    for run_id in configs.values():
        client.delete_tag(run_id, layout.EVALUATION_RUN_ID_TAG)
    router = Router(RunIndex([tmp_path]))
    experiment = router.index.experiments[0]
    evaluation = next(r for r in experiment.runs if r.run_name == "environment_policy_comparison")
    _, _, body = router.resolve(f"/experiment/0/{experiment.experiment_id}")
    listing = body.decode("utf-8")
    page = _page(router, study_id)

    assert listing.count('<a class="card') == 1
    assert "raw evaluation run" in page and f'/{evaluation.run_id}"' in page


def test_comparison_tables_carry_their_numbers_for_the_csv_export(tmp_path: Path):
    """The compare table exports as shown or in full, at the run's precision.

    Given: The two-by-two study.
    When: The study page renders.
    Then: Each compare block has "as shown" and "all data" export buttons
        for its table; the table names its environment and planners with
        their episode counts; each value cell carries the logged value and
        interval in full and the best trial's value; the reference rows are
        left out of the export.
    """
    router, study_id, _ = _two_by_two_study(tmp_path)
    page = _page(router, study_id)
    rocks = page.split('id="compare-Rocks"', 1)[1].split("</section>", 1)[0]

    assert 'data-export-table="compare-table-compare-Rocks" data-export-mode="shown"' in rocks
    assert 'data-export-table="compare-table-compare-Rocks" data-export-mode="all"' in rocks
    assert 'data-env="Rocks" data-export-name="Rocks_compare"' in rocks
    assert 'data-planner="PFT_DPW_Rocks" data-episodes="0"' in rocks
    assert 'data-value="12.0" data-low="10.0" data-high="14.0"' in rocks
    assert 'data-best-trial="13.0"' in rocks
    assert rocks.count("data-export-skip") == 2
    assert page.count("/static/table-export.js") >= 1


def test_every_metric_table_has_a_csv_export(router: Router, study_dir):
    """The evaluation run's metric tables and the tuning view's export too.

    Given: The fixture evaluation run and tuned planner.
    When: The run page and the tuning view render.
    Then: Their metric tables carry an export button, the planner headers
        and the values the export reads.
    """
    run_page = _page(router, study_dir["evaluation"])
    view = _page(router, study_dir["config"])

    for page in (run_page, view):
        assert 'data-export-mode="shown">Export CSV</button>' in page
        assert f'data-planner="{POLICY}"' in page
        assert 'data-value="6.25" data-low="5.0" data-high="7.5"' in page
