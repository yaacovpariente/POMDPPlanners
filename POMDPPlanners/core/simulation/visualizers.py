"""Abstractions for rendering aggregated experiment visualizations.

Implementations are dispatched to worker processes via the simulator's task
manager (Dask, Joblib, PBS, Sequential), so they MUST be picklable and MUST
NOT capture live execution state (clients, sockets, async tasks, file handles).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    from POMDPPlanners.core.simulation.history import History


class ExperimentVisualizer(ABC):
    """Strategy for rendering aggregated per-environment experiment artifacts.

    An ``ExperimentVisualizer`` is invoked once per environment after the
    simulation phase completes. It receives the per-policy episode results and
    is responsible for writing visualization artifacts (plots, animations,
    summary files) into a caller-provided output directory.

    Implementations are dispatched to worker processes via the simulator's
    task manager, so they MUST be picklable and MUST NOT capture live
    execution state (live clients, open sockets, asyncio tasks, file handles,
    threading primitives).

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def render(
        self,
        env_name: str,
        environment: "Environment",
        policy_results: Dict[str, List["History"]],
        policies: Sequence["Policy"],
        output_dir: Path,
        cache_visualizations: bool,
    ) -> Path:
        """Render aggregated artifacts for one environment.

        Args:
            env_name: Name of the environment being visualized.
            environment: Environment instance whose results are being rendered.
            policy_results: Mapping from policy name to a list of histories
                produced by that policy on this environment.
            policies: Sequence of policy instances corresponding to
                ``policy_results`` keys.
            output_dir: Directory under which artifacts are written. The
                caller guarantees the directory exists.
            cache_visualizations: When True, implementations should also
                produce per-episode environment-specific caches (e.g. agent
                trajectory animations) under ``output_dir``.

        Returns:
            Path to the directory containing the rendered artifacts (typically
            ``output_dir`` itself).
        """
