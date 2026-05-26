# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Default ``ExperimentVisualizer`` rendering per-policy returns histograms.

Produces, per environment:
- A discounted-returns histogram for each policy.
- A multi-policy comparison histogram across all policies.
- Optional per-episode environment-specific caches (e.g. agent trajectory
  animations) when ``cache_visualizations=True``.

The visualizer is dispatched to worker processes via the simulator's task
manager, so it must remain picklable and stateless. Module-level helpers
are used in place of instance state; the only module-level reference held
across worker processes is the standard library logger.
"""

import gc
from pathlib import Path
from typing import Dict, List, Sequence

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import ExperimentVisualizer, History
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.utils.visualization import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies,
)


logger = get_logger(name=__name__)


class EpisodeReturnsVisualizer(ExperimentVisualizer):
    """Renders per-policy discounted-returns histograms and a comparison plot.

    For each policy in the environment's results, a histogram of discounted
    returns across episodes is written to
    ``output_dir/<policy_name>/plots/discounted_returns_histogram.png``.
    A multi-policy comparison histogram is written to
    ``output_dir/policy_comparison_histogram.png``. When
    ``cache_visualizations`` is True, the environment's own
    ``cache_visualization`` hook is invoked once per episode to render
    environment-specific artifacts (typically agent-path animations) under
    ``output_dir/<policy_name>/visualizations/``.

    The class holds no instance state and is therefore picklable for dispatch
    via any task manager (Dask, Joblib, PBS, Sequential).

    Example:
        Render artifacts for a single environment::

            from pathlib import Path
            from POMDPPlanners.simulations.simulator import EpisodeReturnsVisualizer

            visualizer = EpisodeReturnsVisualizer()
            output_dir = Path("./viz")
            output_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = visualizer.render(
                env_name=env.name,
                environment=env,
                policy_results={"my_policy": histories},
                policies=[my_policy],
                output_dir=output_dir,
                cache_visualizations=False,
            )
    """

    def render(
        self,
        env_name: str,
        environment: Environment,
        policy_results: Dict[str, List[History]],
        policies: Sequence[Policy],
        output_dir: Path,
        cache_visualizations: bool,
    ) -> Path:
        """Render aggregated artifacts for one environment.

        Args:
            env_name: Name of the environment being visualized (unused; kept
                to match the abstract signature and aid logging in subclasses).
            environment: Environment instance whose results are being rendered.
            policy_results: Mapping from policy name to a list of histories
                produced by that policy on this environment.
            policies: Sequence of policy instances corresponding to the keys
                of ``policy_results``.
            output_dir: Directory under which artifacts are written. Must
                already exist.
            cache_visualizations: When True, also produce per-episode
                environment-specific caches via ``environment.cache_visualization``.

        Returns:
            ``output_dir`` itself.
        """
        del env_name  # only used for subclass-side logging
        for policy_name, policy_histories in policy_results.items():
            policy = next(p for p in policies if p.name == policy_name)
            self._render_policy(
                policy=policy,
                environment=environment,
                policy_histories=policy_histories,
                env_dir=output_dir,
                cache_visualizations=cache_visualizations,
            )
            gc.collect()

        comparison_plot_path = output_dir / "policy_comparison_histogram.png"
        plot_discounted_returns_histogram_multiple_policies(
            histories=policy_results,
            policies=policies,
            environment=environment,
            cache_path=comparison_plot_path,
        )
        gc.collect()
        return output_dir

    def _render_policy(
        self,
        policy: Policy,
        environment: Environment,
        policy_histories: List[History],
        env_dir: Path,
        cache_visualizations: bool,
    ) -> None:
        policy_dir = env_dir / policy.name
        policy_dir.mkdir(exist_ok=True)

        plots_dir = policy_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "discounted_returns_histogram.png"
        plot_discounted_returns_histogram(
            histories=policy_histories,
            policy=policy,
            environment=environment,
            cache_path=plot_path,
        )

        if cache_visualizations:
            self._cache_episodes(
                environment=environment,
                policy_histories=policy_histories,
                policy_dir=policy_dir,
            )

    def _cache_episodes(
        self,
        environment: Environment,
        policy_histories: List[History],
        policy_dir: Path,
    ) -> None:
        viz_dir = policy_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        for episode_idx, history in enumerate(policy_histories):
            file_name = f"agent_path_{episode_idx}.gif"
            cache_path = viz_dir / file_name
            try:
                environment.cache_visualization(
                    history=history.history,
                    cache_path=cache_path,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Visualization failed for episode %s: %s", episode_idx, str(exc))
