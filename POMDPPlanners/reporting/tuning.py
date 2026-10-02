# SPDX-License-Identifier: MIT

"""Reading a tuning study back out of its MLflow run.

The optimizer leaves each tuned planner's study in one config run: the best
parameters and their search ranges as params, the best trial's scores as
``best_trial_*`` metrics, and under ``tuning/`` a summary, the per-trial
records and the diagnostic plots. :mod:`POMDPPlanners.core.simulation.tuning_run_layout`
names all of these; this module turns them into one :class:`TuningStudy`.

Runs written before the summary existed still carry the params and metrics,
so a study is built from those when the summary is missing. It is thinner --
no trial table, no plots -- but it is never wrong about what it does show.
"""

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from POMDPPlanners.core.simulation import tuning_run_layout as layout
from POMDPPlanners.reporting.store import RunView


@dataclass(frozen=True)
class SearchParameter:
    """One tuned parameter: its range and the value the study chose.

    Attributes:
        name: Parameter name.
        best: The chosen value as recorded, or ``None``.
        low: Lower bound, for a numerical parameter.
        high: Upper bound, for a numerical parameter.
        choices: The options, for a categorical parameter.
        range_text: The range as an older run recorded it, when nothing more
            structured is known.
    """

    name: str
    best: Any = None
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    range_text: str = ""

    @property
    def position(self) -> Optional[float]:
        """Where the chosen value sits in a numerical range, from 0 to 1.

        A value pinned to either end of its range is the usual sign the range
        was too narrow, which is why the page draws this.
        """
        if self.low is None or self.high is None or self.high <= self.low:
            return None
        try:
            value = float(self.best)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, (value - self.low) / (self.high - self.low)))


@dataclass(frozen=True)
class Trial:
    """One trial of the study, as its record describes it."""

    number: int
    state: str
    params: Dict[str, Any]
    objective_values: Dict[str, float]
    metric_statistics: Dict[str, List[float]]
    duration_seconds: Optional[float]
    is_pareto: bool


@dataclass(frozen=True)
class TuningStudy:
    """Everything the tuning view shows about one tuned planner.

    Attributes:
        planner: Planner class name.
        environment: Environment class name.
        environment_name: The environment's name, which is how the evaluation
            run files it.
        policy_name: The tuned planner's name, likewise.
        objectives: ``(metric, direction)`` pairs, in the study's order.
        parameters: The search space, with the chosen value of each.
        best_trial_number: The trial the study chose, or ``None``.
        best_trial_scores: The chosen trial's own value of each metric, with
            its interval when one was recorded: ``name -> (value, low, high)``.
        pareto_trial_numbers: Trials on the final Pareto front.
        n_trials_budget: The most trials the study was allowed.
        n_trials_completed: Trials that finished, or ``None`` when unknown.
        early_stopping: The early-stopping settings, or ``None`` when off.
        early_stopping_fired: Whether it ended the study, or ``None`` when unknown.
        stopped_at_trial: The completed-trial count it stopped at.
        front_quality_history: ``(completed trials, front quality)`` pairs,
            the curve early stopping watched; empty when it was off.
        episodes_per_trial: Episodes behind each trial's score.
        steps_per_episode: Step cap per episode.
        optimization_time_seconds: Wall-clock time of the study.
        trials: Every trial, in order; empty for runs that predate the records.
        plots: Relative artifact paths of the diagnostic plots.
        evaluation_run_id: The run that evaluated the chosen planner afresh.
        has_summary: False when the study was rebuilt from params alone.
    """

    planner: str
    environment: str
    environment_name: Optional[str]
    policy_name: Optional[str]
    objectives: List[tuple]
    parameters: List[SearchParameter]
    best_trial_number: Optional[int]
    best_trial_scores: Dict[str, tuple]
    pareto_trial_numbers: List[int] = field(default_factory=list)
    n_trials_budget: Optional[int] = None
    n_trials_completed: Optional[int] = None
    early_stopping: Optional[Dict[str, Any]] = None
    early_stopping_fired: Optional[bool] = None
    stopped_at_trial: Optional[int] = None
    front_quality_history: List[tuple] = field(default_factory=list)
    episodes_per_trial: Optional[int] = None
    steps_per_episode: Optional[int] = None
    optimization_time_seconds: Optional[float] = None
    trials: List[Trial] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    evaluation_run_id: Optional[str] = None
    has_summary: bool = True


def is_tuning_config_run(run: RunView) -> bool:
    """Whether a run holds one tuned planner's study.

    Args:
        run: Any run.

    Returns:
        True for a config run, whether tagged as one or, for older runs,
        recognised by the best-parameter and search-range params only a config
        run logs.
    """
    if run.run_kind == layout.RUN_KIND_CONFIG:
        return True
    keys = run.params.keys()
    return any(k.startswith(layout.PARAM_RANGE_PREFIX) for k in keys) and any(
        k.startswith(layout.BEST_PARAM_PREFIX) for k in keys
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _best_trial_scores(run: RunView) -> Dict[str, tuple]:
    prefix = layout.BEST_TRIAL_METRIC_PREFIX
    scores: Dict[str, tuple] = {}
    for key, value in run.metrics.items():
        if not key.startswith(prefix) or key.endswith(("_ci_lower", "_ci_upper")):
            continue
        name = key[len(prefix) :]
        if name == "number":
            continue
        scores[name] = (
            value,
            run.metrics.get(f"{key}_ci_lower"),
            run.metrics.get(f"{key}_ci_upper"),
        )
    return scores


def _trials(path: Path) -> List[Trial]:
    data = _read_json(path)
    if not isinstance(data, list):
        return []
    trials = []
    for item in data:
        if not isinstance(item, dict) or "number" not in item:
            continue
        trials.append(
            Trial(
                number=int(item["number"]),
                state=str(item.get("state", "")),
                params=dict(item.get("params") or {}),
                objective_values=dict(item.get("objective_values") or {}),
                metric_statistics=dict(item.get("metric_statistics") or {}),
                duration_seconds=item.get("duration_seconds"),
                is_pareto=bool(item.get("is_pareto")),
            )
        )
    return sorted(trials, key=lambda t: t.number)


def _parameters_from_summary(summary: Dict[str, Any]) -> List[SearchParameter]:
    best = summary.get("best_parameters") or {}
    parameters = []
    for entry in summary.get("search_space") or []:
        name = entry.get("name")
        if name is None:
            continue
        parameters.append(
            SearchParameter(
                name=name,
                best=best.get(name),
                low=entry.get("low"),
                high=entry.get("high"),
                choices=entry.get("choices"),
            )
        )
    return parameters


def _parameters_from_params(run: RunView) -> List[SearchParameter]:
    prefix = layout.PARAM_RANGE_PREFIX
    parameters = []
    for key in sorted(run.params):
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        parameters.append(
            SearchParameter(
                name=name,
                best=run.params.get(layout.BEST_PARAM_PREFIX + name),
                range_text=run.params[key],
            )
        )
    return parameters


def load_study(run: RunView) -> TuningStudy:
    """Read one config run's tuning study.

    Args:
        run: A run for which :func:`is_tuning_config_run` is true.

    Returns:
        The study. Fields the run did not record are ``None`` or empty.
    """
    tuning_dir = run.artifact_root / layout.TUNING_ARTIFACT_DIR
    summary = _read_json(tuning_dir / layout.STUDY_SUMMARY_FILE)
    plots = (
        sorted(
            f"{layout.TUNING_ARTIFACT_DIR}/{p.name}"
            for p in tuning_dir.iterdir()
            if p.is_file() and p.suffix == ".png"
        )
        if tuning_dir.is_dir()
        else []
    )
    trials = _trials(tuning_dir / layout.TRIAL_RECORDS_FILE)
    scores = _best_trial_scores(run)
    evaluation_run_id = run.tags.get(layout.EVALUATION_RUN_ID_TAG)

    if isinstance(summary, dict):
        return TuningStudy(
            planner=str(summary.get("planner") or run.params.get("policy_type", "")),
            environment=str(summary.get("environment") or run.params.get("environment_type", "")),
            environment_name=summary.get("environment_name"),
            policy_name=summary.get("policy_name"),
            objectives=[
                (o.get("metric"), o.get("direction")) for o in summary.get("objectives") or []
            ],
            parameters=_parameters_from_summary(summary),
            best_trial_number=_int_or_none(summary.get("best_trial_number")),
            best_trial_scores=scores
            or {k: (v, None, None) for k, v in (summary.get("best_trial_metrics") or {}).items()},
            pareto_trial_numbers=sorted(summary.get("pareto_trial_numbers") or []),
            n_trials_budget=_int_or_none(summary.get("n_trials_budget")),
            n_trials_completed=_int_or_none(summary.get("n_trials_completed")),
            early_stopping=summary.get("early_stopping"),
            early_stopping_fired=summary.get("early_stopping_fired"),
            stopped_at_trial=_int_or_none(summary.get("stopped_at_trial")),
            front_quality_history=[
                (int(n), float(q))
                for n, q in (summary.get("front_quality_history") or [])
                if n is not None and q is not None
            ],
            episodes_per_trial=_int_or_none(summary.get("episodes_per_trial")),
            steps_per_episode=_int_or_none(summary.get("steps_per_episode")),
            optimization_time_seconds=summary.get("optimization_time_seconds"),
            trials=trials,
            plots=plots,
            evaluation_run_id=evaluation_run_id,
        )

    # An older run: the params and metrics are all there is. The objectives
    # were logged as one stringified list, which is parsed only for names.
    objectives: List[tuple] = []
    raw = run.params.get("parameters_to_optimize", "")
    try:
        # Written by str() of a list of (name, direction) tuples.
        parsed = ast.literal_eval(raw) if raw else []
        objectives = [tuple(o) for o in parsed if isinstance(o, (list, tuple)) and len(o) == 2]
    except (ValueError, SyntaxError):
        objectives = []
    return TuningStudy(
        planner=run.params.get("policy_type", ""),
        environment=run.params.get("environment_type", ""),
        environment_name=run.params.get("env_name"),
        policy_name=run.params.get("policy_name"),
        objectives=objectives,
        parameters=_parameters_from_params(run),
        best_trial_number=_int_or_none(run.metrics.get("best_trial_number")),
        best_trial_scores=scores,
        n_trials_budget=_int_or_none(run.params.get("n_trials")),
        n_trials_completed=None,
        episodes_per_trial=_int_or_none(run.params.get("num_episodes")),
        steps_per_episode=_int_or_none(run.params.get("num_steps")),
        optimization_time_seconds=run.metrics.get("optimization_time"),
        trials=trials,
        plots=plots,
        evaluation_run_id=evaluation_run_id,
        has_summary=False,
    )


@dataclass(frozen=True)
class TunedConfig:
    """One tuned planner within a study: its run, its study, its evaluation run."""

    run: RunView
    study: TuningStudy
    evaluation_run: Optional[RunView] = None


@dataclass(frozen=True)
class StudyGroup:
    """One tuning study as a reader thinks of it, rather than as MLflow stores it.

    MLflow keeps a study as a parent run, one child per tuned planner, and an
    evaluation child. Listed flat, that is three cards for one piece of work,
    two of them saying the same thing; grouped, it is one.

    Attributes:
        study_run: The parent run, or ``None`` for a config run with no parent
            in the experiment (an older run, or one moved without its parent).
        configs: The tuned planners, oldest first.
        member_ids: Every run id the group stands for, the parent included.
    """

    study_run: Optional[RunView]
    configs: List[TunedConfig]
    member_ids: frozenset

    @property
    def anchor(self) -> RunView:
        """The run the group's card is dated and sorted by."""
        return self.study_run if self.study_run is not None else self.configs[0].run

    @property
    def evaluation_run(self) -> Optional[RunView]:
        """The study's evaluation run, shared by its configs."""
        return next((c.evaluation_run for c in self.configs if c.evaluation_run), None)


def group_studies(runs: Sequence[RunView]) -> List[StudyGroup]:
    """Gather an experiment's tuning runs into one group per study.

    A run is a study when it is tagged as one or when any of its children is
    a config run, which is how studies logged before the tags are found. Its
    config children, and every evaluation run they name or that is tagged
    under it, belong to the group. Runs that are no part of a study are left
    out, so an experiment without tuning yields no groups at all.

    Args:
        runs: Every run of one experiment.

    Returns:
        The groups, newest first.
    """
    by_id = {r.run_id: r for r in runs}
    children: Dict[str, List[RunView]] = {}
    for run in runs:
        if run.parent_run_id:
            children.setdefault(run.parent_run_id, []).append(run)

    groups: List[StudyGroup] = []
    claimed: set = set()
    for run in runs:
        configs = [c for c in children.get(run.run_id, []) if is_tuning_config_run(c)]
        if run.run_kind != layout.RUN_KIND_STUDY and not configs:
            continue
        groups.append(_group(run, configs, children.get(run.run_id, []), by_id))
        claimed |= groups[-1].member_ids
    # A config run whose parent is not here still stands as a study of one.
    for run in runs:
        if run.run_id in claimed or not is_tuning_config_run(run):
            continue
        if run.parent_run_id in by_id:
            continue
        groups.append(_group(None, [run], [], by_id))
        claimed |= groups[-1].member_ids
    groups.sort(key=lambda g: g.anchor.start_time or 0, reverse=True)
    return groups


def _group(
    study_run: Optional[RunView],
    config_runs: Sequence[RunView],
    siblings: Sequence[RunView],
    by_id: Dict[str, RunView],
) -> StudyGroup:
    configs = []
    members = {study_run.run_id} if study_run is not None else set()
    for run in sorted(config_runs, key=lambda r: r.start_time or 0):
        study = load_study(run)
        evaluation = by_id.get(study.evaluation_run_id or "")
        configs.append(TunedConfig(run=run, study=study, evaluation_run=evaluation))
        members.add(run.run_id)
        if evaluation is not None:
            members.add(evaluation.run_id)
    # The evaluation is a child of the study run, tagged as one, even when no
    # config names it -- a link that failed to write must not make it a stray.
    members |= {
        r.run_id
        for r in siblings
        if r.run_kind == layout.RUN_KIND_EVALUATION or is_tuning_config_run(r)
    }
    return StudyGroup(study_run=study_run, configs=configs, member_ids=frozenset(members))
