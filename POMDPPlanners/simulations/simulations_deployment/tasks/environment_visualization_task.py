"""Task that renders one environment's visualizations on a worker.

The task wraps an :class:`ExperimentVisualizer` along with the per-environment
inputs it needs. Dispatched via the simulator's task manager (Dask, Joblib,
PBS, Sequential), it runs on a worker process. The worker writes artifacts
into a local scratch directory it creates itself, then returns a mapping from
POSIX-style relative path to file bytes. The simulator (parent process)
materializes those bytes on its own filesystem and handles MLflow logging.

This contract is OS-agnostic: no filesystem path crosses the wire, so a
Linux client can dispatch tasks to Windows workers and vice versa.

Each task instance carries a unique config id so the cache-backed managers
(Joblib, PBS) never serve a stale return from a previous run.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from POMDPPlanners.core.simulation import ExperimentVisualizer, History, SimulationTask
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy


def _read_artifacts_to_bytes(root: Path) -> Dict[str, bytes]:
    """Read every file under ``root`` and return ``{posix_relative_path: bytes}``.

    Keys use ``/`` separators regardless of the worker OS so the parent process
    can recreate the same layout cross-platform via ``Path(key)``.
    """
    artifacts: Dict[str, bytes] = {}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_key = file_path.relative_to(root).as_posix()
        artifacts[rel_key] = file_path.read_bytes()
    return artifacts


class EnvironmentVisualizationTask(SimulationTask):
    """Per-environment visualization unit dispatched through the task manager.

    The task is intentionally a self-contained value object: it holds the
    visualizer, the environment, and the per-policy results. Workers pickle
    and execute it without needing a back-reference to the simulator (which
    would drag the live task-manager client into the pickle stream and break
    Dask dispatch) and without any client-supplied filesystem path (which
    would either fail to unpickle on a foreign OS or point at a directory the
    worker cannot reach).

    Attributes:
        visualizer: Strategy that renders the artifacts.
        env_name: Name of the environment, used as the task identifier.
        environment: Environment instance for which artifacts are rendered.
        policy_results: Mapping from policy name to histories.
        policies: Policy instances corresponding to ``policy_results`` keys.
        cache_visualizations: Whether to render per-episode env-specific caches.
    """

    def __init__(
        self,
        visualizer: ExperimentVisualizer,
        env_name: str,
        environment: "Environment",
        policy_results: Dict[str, List[History]],
        policies: Sequence["Policy"],
        cache_visualizations: bool,
    ):
        """Initialize a visualization task.

        Args:
            visualizer: Strategy used to render the artifacts.
            env_name: Name of the environment being visualized.
            environment: Environment instance to render.
            policy_results: Per-policy histories produced by the simulation phase.
            policies: Policy instances matching the keys of ``policy_results``.
            cache_visualizations: Whether to render per-episode env caches.
        """
        self.visualizer = visualizer
        self.env_name = env_name
        self.environment = environment
        self.policy_results = policy_results
        self.policies = list(policies)
        self.cache_visualizations = cache_visualizations
        self._config_id = self._generate_config_id()

    def _generate_config_id(self) -> str:
        components = {
            "kind": "env_viz",
            "env": self.env_name,
            "policies": sorted(self.policy_results.keys()),
            "cache_visualizations": self.cache_visualizations,
            "instance": uuid.uuid4().hex,
        }
        return config_to_id(components)

    def get_config_id(self) -> str:
        """Return a unique identifier for this visualization task.

        The id includes a per-instance UUID so the cached managers
        (Joblib, PBS) never serve a stale return from a previous run.
        """
        return self._config_id

    def run(self) -> Dict[str, bytes]:
        """Render the environment's visualization artifacts on the worker.

        The worker creates a private scratch directory via ``tempfile.mkdtemp``,
        invokes the visualizer there, reads every produced file into memory,
        cleans up the scratch directory, and returns the bytes dict.

        Returns:
            Mapping from POSIX-style relative file path to file bytes.
        """
        tmp = Path(tempfile.mkdtemp(prefix="env_viz_"))
        try:
            self.visualizer.render(
                env_name=self.env_name,
                environment=self.environment,
                policy_results=self.policy_results,
                policies=self.policies,
                output_dir=tmp,
                cache_visualizations=self.cache_visualizations,
            )
            return _read_artifacts_to_bytes(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "visualizer": self.visualizer,
            "env_name": self.env_name,
            "environment": self.environment,
            "policy_results": self.policy_results,
            "policies": self.policies,
            "cache_visualizations": self.cache_visualizations,
            "_config_id": self._config_id,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.visualizer = state["visualizer"]
        self.env_name = state["env_name"]
        self.environment = state["environment"]
        self.policy_results = state["policy_results"]
        self.policies = state["policies"]
        self.cache_visualizations = state["cache_visualizations"]
        self._config_id = state["_config_id"]
