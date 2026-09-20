# SPDX-License-Identifier: MIT

"""Finding runs: the MLflow store, plus what is on disk beside it.

MLflow stays the tracking store; this module only reads it. Two facts about
how this project uses MLflow shape everything here.

First, **there is no single store**. Every simulation run gets its own
file-backed store at ``<run_dir>/mlruns``, because ``BaseSimulator`` points
``mlflow.set_tracking_uri`` at the run's own cache directory. So discovery is
a walk for ``*/mlruns/*/meta.yaml`` under the roots the user names, and the
site holds one ``MlflowClient`` per store it finds.

Second, **``artifact_uri`` in a run's ``meta.yaml`` is an absolute path baked
in when the run was written**. Move or copy a run directory — which is exactly
what happens when results are pulled off a cluster — and those URIs point at
nothing. So artifacts are resolved from the run directory on disk instead of
from the recorded URI.

One MLflow run can hold several environments and several policies, so the row
the site actually shows is ``(run, environment, policy)``, not ``run``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import os

from POMDPPlanners.reporting import artifacts as artifact_module
from POMDPPlanners.reporting.artifacts import EpisodeArtifact

# The params ``BaseSimulator`` writes. They are the authoritative list of which
# environments and policies a run covered: metric keys glue the two names
# together with an underscore and cannot be split back apart reliably.
ENV_NAME_PARAM = "env_{index}_name"
POLICY_NAME_PARAM = "env_{env}_policy_{policy}_name"
POLICY_TYPE_PARAM = "env_{env}_policy_{policy}_type"


@dataclass(frozen=True)
class PolicyView:
    """One policy's slice of one run, on one environment."""

    name: str
    policy_type: Optional[str]
    artifacts: List[EpisodeArtifact] = field(default_factory=list)

    @property
    def episodes(self) -> Dict[int, List[EpisodeArtifact]]:
        """This policy's artifacts grouped by episode index."""
        return artifact_module.group_by_episode(self.artifacts)

    @property
    def plots(self) -> List[EpisodeArtifact]:
        """Artifacts that belong to the policy rather than to one episode."""
        return [a for a in self.artifacts if a.episode_index is None]


@dataclass(frozen=True)
class EnvironmentView:
    """One environment's slice of one run."""

    name: str
    policies: List[PolicyView] = field(default_factory=list)
    artifacts: List[EpisodeArtifact] = field(default_factory=list)

    def policy(self, name: str) -> Optional[PolicyView]:
        """Look up one policy by name.

        Args:
            name: The policy's name as the run recorded it.

        Returns:
            The policy's view, or ``None`` when this run has no such policy.
        """
        return next((p for p in self.policies if p.name == name), None)


@dataclass(frozen=True)
class RunView:
    """One MLflow run, with its environments, policies and artifacts."""

    store_index: int
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: str
    status: str
    start_time: Optional[int]
    end_time: Optional[int]
    params: Dict[str, str]
    metrics: Dict[str, float]
    artifact_root: Path
    environments: List[EnvironmentView] = field(default_factory=list)

    def environment(self, name: str) -> Optional[EnvironmentView]:
        """Look up one environment by name.

        Args:
            name: The environment's ``name`` as the run recorded it.

        Returns:
            The environment's view, or ``None``.
        """
        return next((e for e in self.environments if e.name == name), None)

    def metrics_for(self, env_name: str, policy_name: str) -> Dict[str, float]:
        """Return the metrics this run logged for one environment and policy.

        ``BaseSimulator`` writes metric keys as ``f"{env}_{policy}_{metric}"``
        with no escaping, so the prefix is stripped by length rather than by
        splitting on the separator — splitting would break on any name that
        contains an underscore, which is all of them.

        Args:
            env_name: Environment name.
            policy_name: Policy name.

        Returns:
            Metric name to value, with the prefix removed.
        """
        prefix = f"{env_name}_{policy_name}_"
        return {
            key[len(prefix) :]: value
            for key, value in self.metrics.items()
            if key.startswith(prefix)
        }


@dataclass(frozen=True)
class ExperimentView:
    """One MLflow experiment within one store."""

    store_index: int
    experiment_id: str
    name: str
    store_path: Path
    runs: List[RunView] = field(default_factory=list)


def find_stores(roots: Sequence[Path]) -> List[Path]:
    """Find every MLflow file store under the given roots.

    Args:
        roots: Directories to search. A root that is itself an ``mlruns``
            directory, or that holds one, is handled the same way.

    Returns:
        Absolute paths to the ``mlruns`` directories found, deduplicated and
        sorted.
    """
    found: List[Path] = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            continue
        if root.name == "mlruns":
            found.append(root)
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if "mlruns" in dirnames:
                found.append(Path(dirpath) / "mlruns")
            # Do not descend into a store: everything below it is MLflow's own
            # layout, and walking it on a large results tree is slow for nothing.
            dirnames[:] = [d for d in dirnames if d != "mlruns" and not d.startswith(".")]
    return sorted(set(found))


def tracking_uri_for(store: Path) -> str:
    """Pick the tracking URI that reads one ``mlruns`` directory's metadata.

    A file store keeps its metadata in ``meta.yaml`` files inside ``mlruns``.
    MLflow 3.6 and later refuse that backend on some installations and fall
    back to a SQLite database beside it, leaving ``mlruns`` holding only the
    artifacts. Both layouts appear in this project's results trees, so the
    database wins when it exists and the run directories carry no metadata.

    Args:
        store: An ``mlruns`` directory.

    Returns:
        A tracking URI for :class:`MlflowClient`.
    """
    database = store.parent / "mlflow.db"
    has_file_metadata = any(store.glob("*/meta.yaml"))
    if database.is_file() and not has_file_metadata:
        return f"sqlite:///{database}"
    return f"file://{store}"


def _collect_artifacts(root: Path, recursive: bool = True) -> List[EpisodeArtifact]:
    """Classify the files under one artifact directory.

    Args:
        root: Directory to read.
        recursive: Walk the whole tree. An environment directory passes False,
            because its policy subdirectories are read separately and
            classifying a trace twice means parsing it twice.

    Returns:
        The artifacts found, in path order; files of no recognised kind are
        dropped.
    """
    if not root.is_dir():
        return []
    found: List[EpisodeArtifact] = []
    for path in sorted(root.rglob("*") if recursive else root.iterdir()):
        if not path.is_file():
            continue
        artifact = artifact_module.classify(path, path.relative_to(root).as_posix())
        if artifact is not None:
            found.append(artifact)
    return found


def _names_from_params(params: Dict[str, str]) -> Dict[str, List[str]]:
    """Read the environment and policy names a run recorded.

    Args:
        params: The run's MLflow params.

    Returns:
        A mapping from environment name to its policy names, in the order the
        run logged them.
    """
    layout: Dict[str, List[str]] = {}
    env_index = 0
    while True:
        env_name = params.get(ENV_NAME_PARAM.format(index=env_index))
        if env_name is None:
            break
        policies: List[str] = []
        policy_index = 0
        while True:
            key = POLICY_NAME_PARAM.format(env=env_index, policy=policy_index)
            policy_name = params.get(key)
            if policy_name is None:
                break
            policies.append(policy_name)
            policy_index += 1
        layout[env_name] = policies
        env_index += 1
    return layout


class RunIndex:
    """Every experiment and run the site can show, read once at startup.

    The index is built eagerly because a results tree is static while the
    server is up: a run directory is written by a finished simulation and not
    touched again. :meth:`refresh` rebuilds it when that is not true.
    """

    def __init__(self, roots: Sequence[Path]):
        """Build the index.

        Args:
            roots: Directories to search for MLflow stores.
        """
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        self.stores: List[Path] = []
        self.experiments: List[ExperimentView] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the roots and rebuild every experiment and run."""
        # Imported here so that importing this module — which the tests do to
        # check artifact classification — does not pay MLflow's import cost.
        # pylint: disable-next=import-outside-toplevel
        from mlflow.tracking import MlflowClient

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self.stores = find_stores(self.roots)
        experiments: List[ExperimentView] = []

        for store_index, store in enumerate(self.stores):
            client = MlflowClient(tracking_uri=tracking_uri_for(store))
            try:
                found = client.search_experiments()
            except Exception:  # pylint: disable=broad-exception-caught
                # A half-written or foreign store must not take the site down;
                # it is simply not listed.
                continue
            for experiment in found:
                runs = self._read_runs(client, store, store_index, experiment)
                if not runs:
                    continue
                experiments.append(
                    ExperimentView(
                        store_index=store_index,
                        experiment_id=experiment.experiment_id,
                        name=experiment.name,
                        store_path=store,
                        runs=runs,
                    )
                )

        experiments.sort(key=lambda e: (e.name, e.store_index))
        self.experiments = experiments

    def _read_runs(self, client, store: Path, store_index: int, experiment) -> List[RunView]:
        try:
            found = client.search_runs([experiment.experiment_id], max_results=500)
        except Exception:  # pylint: disable=broad-exception-caught
            return []

        runs: List[RunView] = []
        for run in found:
            params = dict(run.data.params)
            metrics = dict(run.data.metrics)
            # Resolved from the run directory, not from meta.yaml's
            # artifact_uri, which is an absolute path from write time.
            artifact_root = store / experiment.experiment_id / run.info.run_id / "artifacts"

            environments: List[EnvironmentView] = []
            layout = _names_from_params(params)
            for env_index, (env_name, policy_names) in enumerate(layout.items()):
                env_root = artifact_root / env_name
                # Only what sits directly in the environment directory: the
                # comparison plot. Anything deeper belongs to a policy.
                env_artifacts = _collect_artifacts(env_root, recursive=False)
                policies = [
                    PolicyView(
                        name=policy_name,
                        policy_type=params.get(
                            POLICY_TYPE_PARAM.format(env=env_index, policy=index)
                        ),
                        artifacts=_collect_artifacts(env_root / policy_name),
                    )
                    for index, policy_name in enumerate(policy_names)
                ]
                environments.append(
                    EnvironmentView(
                        name=env_name,
                        policies=policies,
                        artifacts=env_artifacts,
                    )
                )

            runs.append(
                RunView(
                    store_index=store_index,
                    experiment_id=experiment.experiment_id,
                    experiment_name=experiment.name,
                    run_id=run.info.run_id,
                    run_name=run.info.run_name or run.info.run_id,
                    status=run.info.status,
                    start_time=run.info.start_time,
                    end_time=run.info.end_time,
                    params=params,
                    metrics=metrics,
                    artifact_root=artifact_root,
                    environments=environments,
                )
            )

        runs.sort(key=lambda r: (r.start_time or 0), reverse=True)
        return runs

    def experiment(self, store_index: int, experiment_id: str) -> Optional[ExperimentView]:
        """Look up one experiment.

        Args:
            store_index: Index of the store within :attr:`stores`.
            experiment_id: MLflow experiment id.

        Returns:
            The experiment, or ``None``.
        """
        return next(
            (
                e
                for e in self.experiments
                if e.store_index == store_index and e.experiment_id == experiment_id
            ),
            None,
        )

    def run(self, store_index: int, experiment_id: str, run_id: str) -> Optional[RunView]:
        """Look up one run.

        Args:
            store_index: Index of the store within :attr:`stores`.
            experiment_id: MLflow experiment id.
            run_id: MLflow run id.

        Returns:
            The run, or ``None``.
        """
        experiment = self.experiment(store_index, experiment_id)
        if experiment is None:
            return None
        return next((r for r in experiment.runs if r.run_id == run_id), None)
