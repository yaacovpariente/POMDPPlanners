# SPDX-License-Identifier: MIT

"""Tests for EpisodeReturnsVisualizer."""

# pylint: disable=protected-access  # Tests need to access protected members

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import cloudpickle
import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import History
from POMDPPlanners.simulations.simulator import EpisodeReturnsVisualizer


def _history() -> History:
    """Build a History stand-in. The MagicMock with spec=History is treated
    as a History at runtime (attribute access is mocked) and we narrow the
    return type so call sites accept it without invariance complaints."""
    h = MagicMock(spec=History)
    h.history = ["step"]
    return h  # type: ignore[return-value]  # MagicMock(spec=History) acts as History


@pytest.fixture
def output_dir(tmp_path) -> Path:
    out = tmp_path / "viz_artifacts" / "env_a"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _policy(name: str) -> MagicMock:
    p = MagicMock(spec=Policy)
    p.name = name
    return p


def _environment(name: str = "env_a") -> MagicMock:
    e = MagicMock(spec=Environment)
    e.name = name
    return e


def test_render_returns_output_dir(output_dir):
    """render returns the output directory it was given.

    Purpose: Validates the visualizer's contract — it returns the path that
        the caller (and the simulator) uses to log MLflow artifacts.

    Given: A visualizer and an existing output directory.
    When: ``render`` is called with empty policy results.
    Then: The returned Path equals the input output_dir.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    with patch(
        "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
        "plot_discounted_returns_histogram_multiple_policies"
    ):
        result = visualizer.render(
            env_name="env_a",
            environment=_environment(),
            policy_results={},
            policies=[],
            output_dir=output_dir,
            cache_visualizations=False,
        )
    assert result == output_dir


def test_render_writes_per_policy_artifacts(output_dir):
    """render creates per-policy plot directories and invokes the histogram helper.

    Purpose: Validates per-policy artifact wiring (directory layout + plot calls).

    Given: Two policies with histories.
    When: ``render`` is called with cache_visualizations=False.
    Then: One ``<policy_name>/plots/`` dir is created per policy and the
        per-policy histogram helper is invoked once per policy.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    pol_a, pol_b = _policy("pol_a"), _policy("pol_b")
    policy_results = {"pol_a": [_history()], "pol_b": [_history()]}

    with (
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram"
        ) as mock_per_policy,
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram_multiple_policies"
        ),
    ):
        visualizer.render(
            env_name="env_a",
            environment=_environment(),
            policy_results=policy_results,
            policies=[pol_a, pol_b],
            output_dir=output_dir,
            cache_visualizations=False,
        )

    assert (output_dir / "pol_a" / "plots").is_dir()
    assert (output_dir / "pol_b" / "plots").is_dir()
    assert mock_per_policy.call_count == 2


def test_render_writes_comparison_histogram(output_dir):
    """render invokes the multi-policy comparison-histogram helper exactly once.

    Purpose: Validates the cross-policy aggregate plot is generated.

    Given: One policy with one history.
    When: ``render`` is called.
    Then: ``plot_discounted_returns_histogram_multiple_policies`` is called
        once with the comparison cache_path under ``output_dir``.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    pol = _policy("pol_a")

    with (
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram"
        ),
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram_multiple_policies"
        ) as mock_compare,
    ):
        visualizer.render(
            env_name="env_a",
            environment=_environment(),
            policy_results={"pol_a": [_history()]},
            policies=[pol],
            output_dir=output_dir,
            cache_visualizations=False,
        )

    mock_compare.assert_called_once()
    kwargs = mock_compare.call_args.kwargs
    assert kwargs["cache_path"] == output_dir / "policy_comparison_histogram.png"


def test_render_with_cache_visualizations_invokes_environment_cache(output_dir):
    """cache_visualizations=True writes each episode through the environment's visualizer.

    Purpose: Validates the per-episode visualizer is wired through.

    Given: One policy with two histories and an environment whose
        ``episode_visualizer`` is mocked.
    When: ``render`` is called with cache_visualizations=True.
    Then: The visualizer's ``write`` is invoked twice (once per episode), into
        the policy's ``visualizations`` directory, with the episode index and
        the policy name.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    pol = _policy("pol_a")
    env = _environment()
    history_a, history_b = _history(), _history()

    with (
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram"
        ),
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram_multiple_policies"
        ),
    ):
        visualizer.render(
            env_name="env_a",
            environment=env,
            policy_results={"pol_a": [history_a, history_b]},
            policies=[pol],
            output_dir=output_dir,
            cache_visualizations=True,
        )

    write = env.episode_visualizer.return_value.write
    assert write.call_count == 2
    assert [call.kwargs["episode_index"] for call in write.call_args_list] == [0, 1]
    assert write.call_args.kwargs["output_dir"] == output_dir / "pol_a" / "visualizations"
    assert write.call_args.kwargs["policy_name"] == "pol_a"


def test_render_skips_environments_without_a_visualizer(output_dir):
    """An environment with no visualizer gets no visualizations directory.

    Purpose: ``episode_visualizer()`` returning ``None`` is the default, and
        must not fail the render or leave an empty directory behind.

    Given: An environment whose ``episode_visualizer`` returns ``None``.
    When: ``render`` is called with cache_visualizations=True.
    Then: No ``visualizations`` directory is created.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    env = _environment()
    env.episode_visualizer.return_value = None

    with (
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram"
        ),
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram_multiple_policies"
        ),
    ):
        visualizer.render(
            env_name="env_a",
            environment=env,
            policy_results={"pol_a": [_history()]},
            policies=[_policy("pol_a")],
            output_dir=output_dir,
            cache_visualizations=True,
        )

    assert not (output_dir / "pol_a" / "visualizations").exists()


def test_render_swallows_per_episode_cache_errors_and_logs_warning(output_dir):
    """Per-episode visualizer failures are caught and logged.

    Purpose: Validates the visualizer's per-episode try/except so that one
        broken episode never aborts a whole render call.

    Given: One policy with one history and an environment whose
        visualizer's ``write`` raises.
    When: ``render`` is called with ``cache_visualizations=True``.
    Then: ``render`` returns the output directory without raising and the
        module-level logger emits a warning that names the failed episode.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    pol = _policy("pol_a")
    env = _environment()
    env.episode_visualizer.return_value.write.side_effect = RuntimeError("boom")

    from POMDPPlanners.simulations.simulator import (  # pylint: disable=import-outside-toplevel
        episode_returns_visualizer as _erv,
    )

    with (
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram"
        ),
        patch(
            "POMDPPlanners.simulations.simulator.episode_returns_visualizer."
            "plot_discounted_returns_histogram_multiple_policies"
        ),
        patch.object(_erv.logger, "warning") as mock_warning,
    ):
        result = visualizer.render(
            env_name="env_a",
            environment=env,
            policy_results={"pol_a": [_history()]},
            policies=[pol],
            output_dir=output_dir,
            cache_visualizations=True,
        )

    assert result == output_dir
    mock_warning.assert_called_once()
    assert "Visualization failed for episode" in mock_warning.call_args[0][0]


def test_episode_returns_visualizer_is_picklable():
    """The default visualizer round-trips through both pickle and cloudpickle.

    Purpose: Regression guard for the original Dask viz pickle bug — workers
        ship via stdlib pickle (joblib loky) or cloudpickle (Dask). Either
        must succeed.

    Given: A default EpisodeReturnsVisualizer instance.
    When: pickle.dumps and cloudpickle.dumps are called.
    Then: Round-trip succeeds and produces an EpisodeReturnsVisualizer.

    Test type: unit
    """
    visualizer = EpisodeReturnsVisualizer()
    assert isinstance(pickle.loads(pickle.dumps(visualizer)), EpisodeReturnsVisualizer)
    assert isinstance(cloudpickle.loads(cloudpickle.dumps(visualizer)), EpisodeReturnsVisualizer)
