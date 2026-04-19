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
    """Visualizer that records render calls and writes a fake artifact file.

    The fake artifact is written into the worker-provided ``output_dir`` so the
    task's bytes-bundling step has something to read back.
    """

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
        nested = output_dir / "policy_a" / "plots"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "discounted_returns_histogram.png").write_bytes(b"fake-png-bytes")
        (output_dir / "policy_comparison_histogram.png").write_bytes(b"fake-comparison")
        return output_dir


def _make_task(
    visualizer=None,
    env_name="env_a",
    use_real_env_and_policies: bool = False,
    # output_dir kept as an accepted kwarg so existing callers don't break;
    # the task no longer takes it (the worker creates its own scratch).
    output_dir=None,  # pylint: disable=unused-argument
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
        cache_visualizations=False,
    )


def test_task_run_invokes_visualizer_render_with_args():
    """run() forwards every constructor arg to visualizer.render.

    Purpose: Validates the task delegates rendering to its visualizer with
        all per-environment inputs intact.

    Given: An EnvironmentVisualizationTask wrapping a recording visualizer.
    When: ``run()`` is called.
    Then: The recording visualizer captures one render call whose arguments
        match the task's attributes (the ``output_dir`` is a worker-private
        scratch directory created inside ``run()``).

    Test type: unit
    """
    visualizer = _RecordingVisualizer()
    task = _make_task(visualizer=visualizer)

    task.run()

    assert len(visualizer.calls) == 1
    call = visualizer.calls[0]
    assert call["env_name"] == "env_a"
    assert isinstance(call["output_dir"], Path)
    assert call["cache_visualizations"] is False
    assert set(call["policy_results"].keys()) == {"policy_a", "policy_b"}


def test_task_run_returns_bytes_keyed_by_relative_path():
    """run() returns a ``Dict[str, bytes]`` keyed by POSIX-style relative paths.

    Purpose: Validates the OS-agnostic bytes-dict contract that lets the
        client materialize artifacts on its own filesystem regardless of the
        worker's OS.

    Given: A task whose visualizer writes two files into the worker-provided
        scratch dir (one nested, one at the root).
    When: ``run()`` is called.
    Then: The returned dict contains both files, keys use ``/`` separators,
        and the byte payloads match what the visualizer wrote.

    Test type: unit
    """
    task = _make_task(visualizer=_RecordingVisualizer())

    artifacts = task.run()

    assert isinstance(artifacts, dict)
    assert "policy_comparison_histogram.png" in artifacts
    assert "policy_a/plots/discounted_returns_histogram.png" in artifacts
    assert artifacts["policy_comparison_histogram.png"] == b"fake-comparison"
    assert artifacts["policy_a/plots/discounted_returns_histogram.png"] == b"fake-png-bytes"


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


def test_task_pickle_is_os_agnostic():
    """The pickled task must not embed OS-specific Path class names.

    Purpose: Regression for cross-OS worker crashes. Workers on a different
        OS than the client cannot unpickle ``pathlib.PosixPath`` /
        ``pathlib.WindowsPath`` shipped from the other side. The pickled
        task payload must therefore not reference either class.

    Given: An EnvironmentVisualizationTask constructed with a Path-typed
        ``output_dir`` (which is a PosixPath on Linux test runners and a
        WindowsPath on Windows test runners).
    When: ``pickle.dumps`` serializes the task.
    Then: The byte stream contains neither ``b"PosixPath"`` nor
        ``b"WindowsPath"`` anywhere — i.e. paths travel as plain strings.

    Test type: unit
    """
    task = _make_task(
        visualizer=EpisodeReturnsVisualizer(),
        use_real_env_and_policies=True,
        output_dir=Path("/tmp/some_path"),
    )
    blob = pickle.dumps(task)
    assert b"PosixPath" not in blob, (
        "EnvironmentVisualizationTask pickle stream embeds pathlib.PosixPath. "
        "Foreign-OS workers cannot unpickle this. Use str on the wire."
    )
    assert (
        b"WindowsPath" not in blob
    ), "EnvironmentVisualizationTask pickle stream embeds pathlib.WindowsPath."
