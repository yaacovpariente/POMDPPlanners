"""Tests for EnvironmentVisualizationTask."""

# pylint: disable=protected-access  # Tests need to access protected members

import pickle
from pathlib import Path
from typing import Dict, List, Sequence
from unittest.mock import MagicMock

import cloudpickle

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import ExperimentVisualizer, History
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.simulations.simulations_deployment.tasks import (
    EnvironmentVisualizationTask,
)
from POMDPPlanners.simulations.simulator import EpisodeReturnsVisualizer


class _RecordingVisualizer(ExperimentVisualizer):
    """Visualizer that records render calls and returns the output dir."""

    def __init__(self):
        self.calls: List[Dict] = []

    def render(
        self,
        env_name: str,
        environment: Environment,
        policy_results: Dict[str, List[History]],
        policies: Sequence[Policy],
        output_dir: Path,
        cache_visualizations: bool,
    ) -> Path:
        self.calls.append(
            {
                "env_name": env_name,
                "environment": environment,
                "policy_results": policy_results,
                "policies": list(policies),
                "output_dir": output_dir,
                "cache_visualizations": cache_visualizations,
            }
        )
        return output_dir


def _make_task(
    visualizer=None,
    output_dir=Path("/tmp/x"),
    env_name="env_a",
    use_real_env_and_policies: bool = False,
):
    if use_real_env_and_policies:
        env: Environment = TigerPOMDP(discount_factor=0.95)
        policy_results: Dict[str, List[History]] = {}
        policies: Sequence[Policy] = []
    else:
        env = MagicMock(spec=Environment)
        policy_results = {"policy_a": [], "policy_b": []}
        policies = [MagicMock(spec=Policy), MagicMock(spec=Policy)]
    return EnvironmentVisualizationTask(
        visualizer=visualizer if visualizer is not None else _RecordingVisualizer(),
        env_name=env_name,
        environment=env,
        policy_results=policy_results,
        policies=policies,
        output_dir=output_dir,
        cache_visualizations=False,
    )


def test_task_run_invokes_visualizer_render_with_args():
    """run() forwards every constructor arg to visualizer.render.

    Purpose: Validates the task delegates rendering to its visualizer with
        all per-environment inputs intact.

    Given: An EnvironmentVisualizationTask wrapping a recording visualizer.
    When: ``run()`` is called.
    Then: The recording visualizer captures one render call whose arguments
        match the task's attributes, and run returns the output directory.

    Test type: unit
    """
    visualizer = _RecordingVisualizer()
    output_dir = Path("/tmp/viz_artifacts/env_a")
    task = _make_task(visualizer=visualizer, output_dir=output_dir)

    result = task.run()

    assert result == output_dir
    assert len(visualizer.calls) == 1
    call = visualizer.calls[0]
    assert call["env_name"] == "env_a"
    assert call["output_dir"] == output_dir
    assert call["cache_visualizations"] is False
    assert set(call["policy_results"].keys()) == {"policy_a", "policy_b"}


def test_task_get_config_id_is_unique_per_instance():
    """get_config_id includes a per-instance UUID to avoid stale cache hits.

    Purpose: Validates that two tasks with identical inputs still get distinct
        config ids — required because the artifact directory is deleted after
        upload, so any cached return value would point at a non-existent path.

    Given: Two EnvironmentVisualizationTask instances with identical inputs.
    When: ``get_config_id`` is called on each.
    Then: The returned ids differ.

    Test type: unit
    """
    task_a = _make_task(env_name="env_a")
    task_b = _make_task(env_name="env_a")
    assert task_a.get_config_id() != task_b.get_config_id()


def test_task_pickles_cleanly_for_joblib():
    """Stdlib pickle round-trips a task without losing state.

    Purpose: Regression guard for the original Dask viz pickle bug — workers
        ship via stdlib pickle (joblib loky) or cloudpickle (Dask). Either
        must succeed.

    Given: An EnvironmentVisualizationTask with a default visualizer.
    When: pickle.dumps + pickle.loads round-trip the task.
    Then: The round-tripped task carries the same env_name and config id.

    Test type: unit
    """
    task = _make_task(visualizer=EpisodeReturnsVisualizer(), use_real_env_and_policies=True)
    restored = pickle.loads(pickle.dumps(task))
    assert restored.env_name == task.env_name
    assert restored.get_config_id() == task.get_config_id()


def test_task_pickles_cleanly_for_dask():
    """Cloudpickle round-trips a task without losing state.

    Purpose: Regression guard for the original Dask viz pickle bug.

    Given: An EnvironmentVisualizationTask with a default visualizer.
    When: cloudpickle.dumps + cloudpickle.loads round-trip the task.
    Then: The round-tripped task carries the same env_name and config id.

    Test type: unit
    """
    task = _make_task(visualizer=EpisodeReturnsVisualizer(), use_real_env_and_policies=True)
    restored = cloudpickle.loads(cloudpickle.dumps(task))
    assert restored.env_name == task.env_name
    assert restored.get_config_id() == task.get_config_id()
