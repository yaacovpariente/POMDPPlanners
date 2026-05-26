# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Concrete POMDP simulator implementation."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    ExperimentVisualizer,
    History,
    MetricValue,
    SimulationTask,
)
from POMDPPlanners.simulations.simulation_statistics import (
    compute_statistics_environment_policy_pair,
    get_metric_names_from_environment_policy_pair,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress import (
    NotificationConfig,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    TaskManagerConfig,
)
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.simulations.simulator.base_simulator import BaseSimulator


class POMDPSimulator(BaseSimulator):
    """Concrete implementation of BaseSimulator for POMDP planning algorithm comparisons.

    This class provides a complete simulation framework for comparing POMDP planning algorithms
    across multiple environments. It executes episodes in parallel, collects comprehensive
    statistics, and generates visualizations for analysis.

    The simulator supports:
    - Episode-based simulation with configurable number of steps and episodes
    - Parallel execution using Dask, Joblib, or PBS cluster task managers
    - Comprehensive timing and performance metrics collection
    - Statistical analysis with confidence intervals and risk measures
    - Pluggable visualization strategy via :class:`ExperimentVisualizer`
    - MLflow experiment tracking with structured logging

    Key Metrics Collected:
    - Average return and discounted return statistics
    - Conditional Value at Risk (CVaR) and Value at Risk (VaR)
    - Timing metrics (action selection, belief updates, state transitions)
    - Episode completion rates and terminal state statistics
    - Custom environment-specific metrics

    Example:
        Creating and configuring a POMDP simulator for algorithm comparison:

        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.simulator import POMDPSimulator
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import SparseSamplingDiscreteActionsPlanner
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import EnvironmentRunParams
        >>> from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import JoblibConfig

        >>> # Create environment
        >>> tiger_env = TigerPOMDP(discount_factor=0.95)
        >>> # Create initial belief for testing
        >>> initial_belief = get_initial_belief(tiger_env, n_particles=10)  # Reduced for testing

        >>> # Create policies to compare
        >>> pomcp = POMCP(
        ...     environment=tiger_env,
        ...     discount_factor=0.95,
        ...     depth=5,               # Reduced for testing
        ...     exploration_constant=1.0,
        ...     name="POMCP_Test",
        ...     n_simulations=10       # Reduced for testing
        ... )

        >>> sparse_sampling = SparseSamplingDiscreteActionsPlanner(
        ...     environment=tiger_env,
        ...     branching_factor=2,    # Reduced for testing
        ...     depth=3,               # Reduced for testing
        ...     name="SparseSampling_Test"
        ... )

        >>> # Configure simulation parameters
        >>> env_params = [
        ...     EnvironmentRunParams(
        ...         environment=tiger_env,
        ...         belief=initial_belief,
        ...         policies=[pomcp, sparse_sampling],
        ...         num_episodes=2,    # Reduced for testing
        ...         num_steps=5        # Reduced for testing
        ...     )
        ... ]

        >>> # Configure task manager
        >>> joblib_config = JoblibConfig(n_jobs=1)  # Single job for testing
        >>> joblib_config.n_jobs
        1

        >>> # Test simulator configuration
        >>> simulator = POMDPSimulator(
        ...     task_manager_config=joblib_config,
        ...     experiment_name="Tiger_POMDP_Test",
        ...     debug=False,
        ...     enable_profiling=False
        ... )
    """

    def __init__(
        self,
        task_manager_config: TaskManagerConfig,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
        task_console_output: bool = False,
        use_queue_logger: bool = False,
        console_output: bool = True,
        no_logs: bool = False,
        visualizer: Optional[ExperimentVisualizer] = None,
        notification_config: Optional[NotificationConfig] = None,
    ):
        """Initialize the POMDP simulator.

        Args:
            task_manager_config: Configuration object for task manager creation
            cache_dir_path: Path to store results and log files. Set to None to disable file logging.
            experiment_name: Name of the MLFlow experiment
            debug: Whether to enable debug logging
            enable_profiling: Whether to enable cProfile profiling
            profiling_output_limit: Maximum number of functions to show in profiling output (default: 50)
            task_console_output: Whether to enable console output for individual tasks (default: False).
                               Set to True to see console output from each task, but this can create
                               log mess when running in parallel.
            use_queue_logger: Whether to use queue-based logging
            console_output: Whether to print logs to console (default: True)
            no_logs: Whether to disable all logging (default: False)
            visualizer: Strategy for rendering aggregated experiment artifacts.
                Defaults to :class:`EpisodeReturnsVisualizer`.
            notification_config: Optional :class:`NotificationConfig` controlling
                Slack + progress-DB notifications. Forwarded to
                :class:`BaseSimulator`. When ``None``, the parent defaults to
                :meth:`NotificationConfig.disabled`.
        """
        super().__init__(
            task_manager_config=task_manager_config,
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
            use_queue_logger=use_queue_logger,
            console_output=console_output,
            no_logs=no_logs,
            visualizer=visualizer,
            notification_config=notification_config,
        )
        self.task_console_output = task_console_output

    def _create_simulation_tasks(
        self,
        environment_run_params: List[EnvironmentRunParams],
    ) -> Tuple[List[SimulationTask], List[Tuple[str, str]]]:
        """Create list of simulation tasks with deterministic ordering.

        Returns:
            Tuple containing:
            - List of simulation tasks
            - List of (env_name, policy_name) identifiers matching the tasks
        """
        simulation_tasks: List[SimulationTask] = []
        task_identifiers: List[Tuple[str, str]] = []
        total_tasks = 0

        for params in environment_run_params:
            env_name = params.environment.name
            for policy in params.policies:
                policy_name = policy.name
                for episode_id in range(params.num_episodes):
                    seed = int(
                        hashlib.md5(f"{env_name}_{policy_name}_{episode_id}".encode()).hexdigest(),
                        16,
                    ) % (2**32)
                    task = EpisodeSimulationTask(
                        environment=params.environment,
                        policy=policy,
                        initial_belief=params.belief,
                        num_steps=params.num_steps,
                        episode_id=episode_id,
                        seed=seed,
                        episode_number=episode_id,
                        cache_dir=self.cache_dir_path,
                        debug=self.debug,
                        console_output=self.task_console_output,
                    )
                    simulation_tasks.append(task)
                    task_identifiers.append((env_name, policy_name))
                    total_tasks += 1

        self.logger.info(
            "Created %s simulation tasks across %s environments and %s policies",
            total_tasks,
            len({params.environment.name for params in environment_run_params}),
            len({p.name for params in environment_run_params for p in params.policies}),
        )

        return simulation_tasks, task_identifiers

    def _compute_metrics(
        self,
        results: Dict[str, Dict[str, List[History]]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
    ) -> Dict[str, Dict[str, List[MetricValue]]]:
        """Compute metrics for all environment-policy pairs."""
        metrics_dict: Dict[str, Dict[str, List[MetricValue]]] = {}
        envs_dict = {
            params.environment.name: params.environment for params in environment_run_params
        }

        for env_name, policy_histories_dict in results.items():
            metrics_dict[env_name] = {}
            environment = envs_dict[env_name]

            for policy_name, histories in policy_histories_dict.items():
                metrics = compute_statistics_environment_policy_pair(
                    env=environment,
                    histories=histories,
                    alpha=alpha,
                    confidence_interval_level=confidence_interval_level,
                )
                metrics_dict[env_name][policy_name] = metrics

        return metrics_dict

    def get_output_metric_names(
        self, environment_policy_pairs: Sequence[Tuple[Environment, Type[Policy]]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get all metric names that will be output by the simulator for given environment-policy pairs.

        This method returns the complete list of metric names that will be computed and logged
        for each environment-policy combination during simulation. This is useful for:
        - Pre-allocating data structures for metric collection
        - Validating expected metric availability before running simulations
        - Understanding what metrics will be tracked in MLflow
        - Setting up metric-based optimization objectives

        Args:
            environment_policy_pairs: Sequence of (Environment, Policy class) tuples specifying
                the combinations to get metrics for. Note that Policy should be the class,
                not an instance.

        Returns:
            Dictionary with two-level nesting:
            - First level: environment name -> policy metrics dict
            - Second level: policy class name -> list of metric names
            Each list contains metric names in the order they will appear in simulation results:
            1. Environment-specific metrics
            2. Policy info variables (prefixed with "policy_info_")
            3. Standard metrics (return, CVaR, timing, etc.)

        Raises:
            TypeError: If environment_policy_pairs is not a sequence or contains invalid types
            ValueError: If environment_policy_pairs is empty

        Example:
            Get expected metrics for TigerPOMDP with POMCP and SparseSampling:

            >>> from pathlib import Path
            >>> from POMDPPlanners.simulations.simulator import POMDPSimulator
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import SparseSamplingDiscreteActionsPlanner
            >>> from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import JoblibConfig
            >>>
            >>> # Create environment
            >>> tiger_env = TigerPOMDP(discount_factor=0.95)
            >>>
            >>> # Create simulator
            >>> joblib_config = JoblibConfig(n_jobs=1)
            >>> simulator = POMDPSimulator(
            ...     task_manager_config=joblib_config,
            ...     experiment_name="Metric_Names_Test"
            ... )
            >>>
            >>> # Get metric names for environment-policy combinations
            >>> env_policy_pairs = [
            ...     (tiger_env, POMCP),
            ...     (tiger_env, SparseSamplingDiscreteActionsPlanner)
            ... ]
            >>> metric_names = simulator.get_output_metric_names(env_policy_pairs)
            >>>
            >>> # Check structure
            >>> 'TigerPOMDP' in metric_names
            True
            >>> 'POMCP' in metric_names['TigerPOMDP']
            True
            >>> 'SparseSamplingDiscreteActionsPlanner' in metric_names['TigerPOMDP']
            True
            >>>
            >>> # Check for standard metrics in POMCP results
            >>> pomcp_metrics = metric_names['TigerPOMDP']['POMCP']
            >>> 'average_return' in pomcp_metrics
            True
            >>> 'return_cvar' in pomcp_metrics
            True
            >>>
            >>> # Check for environment-specific metrics
            >>> 'success_rate' in pomcp_metrics
            True
            >>>
            >>> # POMCP has policy info metrics, SparseSampling does not
            >>> any(name.startswith('policy_info_') for name in pomcp_metrics)
            True
            >>> sparse_metrics = metric_names['TigerPOMDP']['SparseSamplingDiscreteActionsPlanner']
            >>> any(name.startswith('policy_info_') for name in sparse_metrics)
            False
            >>>
            >>> # Clean up simulator
            >>> simulator.cleanup_mlflow_runs()
        """
        if isinstance(environment_policy_pairs, str) or not hasattr(
            environment_policy_pairs, "__iter__"
        ):
            raise TypeError("environment_policy_pairs must be a sequence")
        if len(environment_policy_pairs) == 0:
            raise ValueError("environment_policy_pairs cannot be empty")

        for pair in environment_policy_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "Each element in environment_policy_pairs must be a tuple of (Environment, Type[Policy])"
                )
            env, policy_cls = pair
            if not isinstance(env, Environment):
                raise TypeError(f"Expected Environment instance, got {type(env)}")
            if not isinstance(policy_cls, type) or not issubclass(policy_cls, Policy):
                raise TypeError(f"Expected Policy class, got {type(policy_cls)}")

        metric_names_dict: Dict[str, Dict[str, List[str]]] = {}

        for env, policy_cls in environment_policy_pairs:
            env_name = env.name
            policy_cls_name = policy_cls.__name__

            if env_name not in metric_names_dict:
                metric_names_dict[env_name] = {}

            metric_names = get_metric_names_from_environment_policy_pair(env, policy_cls)
            metric_names_dict[env_name][policy_cls_name] = metric_names

        return metric_names_dict
