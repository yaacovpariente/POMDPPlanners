"""Task that renders one environment's visualizations on a worker.

The task wraps an :class:`ExperimentVisualizer` along with the per-environment
inputs it needs. Dispatched via the simulator's task manager (Dask, Joblib,
PBS, Sequential), it runs on a worker process and returns the path of the
directory it wrote to. The simulator (parent process) handles the subsequent
MLflow artifact logging.

Each task instance carries a unique config id so it never matches the cache
of a previous run — the rendered output directories are deleted once their
contents are uploaded to MLflow, so cached return values would point to
non-existent paths on a re-run.
"""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from POMDPPlanners.core.simulation import ExperimentVisualizer, History, SimulationTask
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy


class EnvironmentVisualizationTask(SimulationTask):
    """Per-environment visualization unit dispatched through the task manager.

    The task is intentionally a self-contained value object: it holds the
    visualizer, the environment, the per-policy results, and the destination
    directory. Workers pickle and execute it without needing a back-reference
    to the simulator (which would drag the live task-manager client into the
    pickle stream and break Dask dispatch).

    Attributes:
        visualizer: Strategy that renders the artifacts.
        env_name: Name of the environment, used as the task identifier.
        environment: Environment instance for which artifacts are rendered.
        policy_results: Mapping from policy name to histories.
        policies: Policy instances corresponding to ``policy_results`` keys.
        output_dir: Directory where the visualizer writes its artifacts.
        cache_visualizations: Whether to render per-episode env-specific caches.
    """

    def __init__(
        self,
        visualizer: ExperimentVisualizer,
        env_name: str,
        environment: "Environment",
        policy_results: Dict[str, List[History]],
        policies: Sequence["Policy"],
        output_dir: Path,
        cache_visualizations: bool,
    ):
        """Initialize a visualization task.

        Args:
            visualizer: Strategy used to render the artifacts.
            env_name: Name of the environment being visualized.
            environment: Environment instance to render.
            policy_results: Per-policy histories produced by the simulation phase.
            policies: Policy instances matching the keys of ``policy_results``.
            output_dir: Existing directory under which artifacts are written.
            cache_visualizations: Whether to render per-episode env caches.
        """
        self.visualizer = visualizer
        self.env_name = env_name
        self.environment = environment
        self.policy_results = policy_results
        self.policies = list(policies)
        self.output_dir = output_dir
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
        (Joblib, PBS) never serve a stale ``Path`` from a previous run; the
        artifact directory itself is deleted after upload.
        """
        return self._config_id

    def run(self) -> Path:
        """Render the environment's visualization artifacts on the worker.

        Returns:
            Path to the directory containing the rendered artifacts.
        """
        return self.visualizer.render(
            env_name=self.env_name,
            environment=self.environment,
            policy_results=self.policy_results,
            policies=self.policies,
            output_dir=self.output_dir,
            cache_visualizations=self.cache_visualizations,
        )

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "visualizer": self.visualizer,
            "env_name": self.env_name,
            "environment": self.environment,
            "policy_results": self.policy_results,
            "policies": self.policies,
            "output_dir": self.output_dir,
            "cache_visualizations": self.cache_visualizations,
            "_config_id": self._config_id,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.visualizer = state["visualizer"]
        self.env_name = state["env_name"]
        self.environment = state["environment"]
        self.policy_results = state["policy_results"]
        self.policies = state["policies"]
        self.output_dir = state["output_dir"]
        self.cache_visualizations = state["cache_visualizations"]
        self._config_id = state["_config_id"]
