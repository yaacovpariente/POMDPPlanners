"""Abstract interface for POMDP simulation APIs.

This module defines the abstract base class interface that all simulation API
implementations must follow. It ensures consistent method signatures across
local, distributed, and cluster-based simulation execution modes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    OptimizedPolicyResult,
    ParallelizationLevel,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress import (
    NotificationConfig,
)


class SimulationsAPIInterface(ABC):
    """Abstract base class defining the interface for POMDP simulation APIs.

    This interface ensures that all simulation API implementations (local, Dask,
    PBS, etc.) provide a consistent set of methods with standardized signatures.
    Subclasses must implement all abstract methods to provide specific execution
    strategies while maintaining API compatibility.

    The interface defines methods for:
    - Running simulations with multiple environments and policies
    - Hyperparameter optimization
    - Comprehensive benchmarking with hyperparameter tuning
    - Optimize and evaluate workflows
    - All hyperparameter benchmarks

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Use concrete implementations like LocalSimulationsAPI or DaskSimulationsAPI.
    """

    @abstractmethod
    def __init__(
        self,
        cache_dir_path: Optional[Path] = None,
        debug: bool = False,
        notification_config: Optional[NotificationConfig] = None,
    ):
        """Initialize the SimulationsAPI.

        Args:
            cache_dir_path: Optional path for storing simulation results and logs
            debug: Whether to enable debug-level logging output
            notification_config: Optional :class:`NotificationConfig` controlling
                Slack + progress-DB notifications. Implementations default to
                :meth:`NotificationConfig.from_env` when ``None`` is passed,
                preserving the env-var-driven zero-config UX.
        """

    @abstractmethod
    def run_multiple_environments_and_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        scheduler_address: Optional[str] = None,
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run POMDP simulations with multiple environments and policies.

        This method executes POMDP simulations for the given environment and policy
        configurations. The specific execution strategy (local, distributed, cluster)
        is determined by the concrete implementation.

        Args:
            environment_run_params: List of environment configurations for simulation.
                Each configuration specifies an environment, belief state, policies,
                number of episodes, and number of steps per episode.
            alpha: Statistical significance level for confidence intervals (e.g., 0.05 for 95% CI).
                Used for computing risk metrics like Conditional Value at Risk (CVaR).
            confidence_interval_level: Confidence level for statistical analysis (e.g., 0.95).
                Determines the width of confidence intervals for performance metrics.
            experiment_name: Name for the experiment and MLflow tracking. Used to organize
                results and enable comparison across different experimental runs.
            debug: Whether to enable debug-level logging output. When True, provides
                detailed information about simulation progress and internal operations.
            scheduler_address: Address of the Dask scheduler for distributed execution.
                If None, uses local execution (LocalSimulationsAPI) or creates a local
                Dask cluster (DaskSimulationsAPI). Format: "tcp://scheduler-ip:port".
                This parameter is ignored by LocalSimulationsAPI.
            n_jobs: Number of parallel jobs for execution. Use -1 to use all available
                CPU cores/workers, or specify a positive integer for a specific number.
            cache_dir_path: Optional path for storing simulation results, logs, and artifacts.
                If None, results are stored in the current working directory.
            clear_cache_on_start: Whether to clear existing cache before starting simulation.
                Useful for ensuring clean runs when debugging or testing.
            enable_profiling: Whether to enable performance profiling using cProfile.
                Generates detailed timing information for optimization analysis.
            profiling_output_limit: Maximum number of profiling entries to display
                when profiling is enabled. Helps focus on the most time-consuming operations.

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  name, then policy name, containing lists of History objects for each episode.
                - pd.DataFrame: Statistical summary with confidence intervals, performance
                  metrics, and policy configuration details for analysis and comparison.
        """

    @abstractmethod
    def run_all_benchmark_environments_on_planner_generators(
        self,
        generators: Sequence[PlannerGenerator],
        n_particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        experiment_name: str = "All_Benchmark_Environments_On_Planner_Generators",
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
        cache_visualizations: bool = True,
        is_risk_averse: bool = False,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run all benchmark environments on planner generators.

        This method runs benchmark environments on planner generators.

        Args:
            generators: Sequence of PlannerGenerator objects.
            n_particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            alpha: Statistical significance level for confidence intervals.
            confidence_interval_level: Confidence level for statistical analysis.
            experiment_name: Name for the experiment.
            n_jobs: Number of parallel jobs for execution.
            cache_dir_path: Optional path for storing simulation results.
            clear_cache_on_start: Whether to clear existing cache before starting simulation.
            enable_profiling: Whether to enable performance profiling.
            profiling_output_limit: Maximum number of profiling entries to display.
            cache_visualizations: Whether to cache visualizations.
            is_risk_averse: Whether to run risk-averse benchmark.
        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  name, then policy name, containing lists of History objects for each episode.
                - pd.DataFrame: Statistical summary with confidence intervals, performance
                  metrics, and policy configuration details for analysis and comparison.
        """

    @abstractmethod
    def run_hyperparameter_optimization(
        self,
        environment_run_params: List[HyperParameterRunParams],
        experiment_name: str = "POMDP_Hyperparameter_Optimization",
        n_jobs: int = -1,
        cache_dir_path: Optional[Path] = None,
        clear_cache_on_start: bool = False,
        debug: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        use_queue_logger: bool = False,
        parallelization_level: ParallelizationLevel = ParallelizationLevel.OPTUNA_TRIALS,
    ) -> List[OptimizedPolicyResult]:
        """Run hyperparameter optimization for POMDP policies using Optuna.

        This method provides a high-level interface for hyperparameter optimization
        by wrapping the HyperParameterOptimizer class. It supports optimization
        of multiple environment-policy configurations with comprehensive MLflow
        tracking and statistical analysis.

        The optimization uses Optuna's advanced algorithms (TPE, CMA-ES, etc.) to
        efficiently search the hyperparameter space and find optimal configurations
        for POMDP policies.

        Args:
            environment_run_params: List of HyperParameterRunParams configurations,
                each specifying an environment, policy class, hyperparameter ranges,
                and optimization settings. Each configuration must include the
                required n_trials parameter.
            experiment_name: Name for the MLflow experiment tracking. Used to organize
                optimization runs and enable comparison across different experiments.
            n_jobs: Number of parallel jobs for episode execution. Use -1 to use all
                available CPU cores/workers, or specify a positive integer for a specific
                number of cores.
            cache_dir_path: Optional path for storing optimization results, logs,
                and MLflow artifacts. If None, results are stored in the current
                working directory.
            clear_cache_on_start: Whether to clear existing cache before starting
                optimization. Useful for ensuring clean runs when debugging or testing.
            debug: Whether to enable debug-level logging output. When True, provides
                detailed information about optimization progress and internal operations.
            confidence_interval_level: Confidence level for statistical analysis
                (between 0.0 and 1.0). Used for computing confidence intervals in
                performance statistics. Defaults to 0.95 for 95% confidence intervals.
            alpha: Significance level for statistical tests (between 0.0 and 1.0).
                Used for hypothesis testing and confidence interval calculations.
                Defaults to 0.05 for 5% significance level.
            use_queue_logger: Whether to use queue-based logging for distributed
                execution scenarios. Defaults to False for local execution.
            parallelization_level: Controls where parallelization is applied.
                OPTUNA_TRIALS (default) parallelizes across Optuna trials while
                running episodes sequentially. EPISODES parallelizes across
                episodes within each trial while running trials sequentially.

        Returns:
            List[OptimizedPolicyResult]: List of optimization results, each containing
                the optimized policy with its best hyperparameters, environment reference,
                and optimization metadata for each input configuration.

        Raises:
            ValueError: If any configuration contains invalid parameters or missing
                required fields like n_trials.
            TypeError: If policy classes are not Policy subclasses.
            RuntimeError: If optimization fails for any configuration.
        """

    @abstractmethod
    def run_hyperparameter_tuning_experiment_with_benchmarks(
        self,
        generators: Sequence[HyperParamPlannerConfigGenerator],
        particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        n_trials: int = 100,
        discount_factor: float = 0.95,
        time_out_in_seconds: float = 3.0,
        evaluation_episodes: int = 3,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        is_risk_averse: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "Comprehensive_Benchmark",
        debug: bool = False,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run comprehensive benchmark with hyperparameter optimization.

        This method runs hyperparameter optimization followed by policy evaluation
        for comprehensive benchmarking. It optimizes for average return across
        all configured environments and benchmark planners.

        Args:
            generators: Hyperparameter configuration generators list.
            particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            n_trials: Number of optimization trials.
            discount_factor: Discount factor for the MDP.
            time_out_in_seconds: Timeout for planner execution.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  and policy names.
                - pd.DataFrame: Statistical summary with performance metrics and comparisons.
        """

    @abstractmethod
    def run_optimize_and_evaluate(
        self,
        configs: List[HyperParameterRunParams],
        evaluation_episodes: int = 100,
        evaluation_steps: int = 100,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "Optimize_And_Evaluate",
        debug: bool = False,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run hyperparameter optimization and evaluation.

        This method runs hyperparameter optimization for the provided configurations,
        then evaluates the optimized policies.

        Args:
            configs: List of hyperparameter run configurations.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  and policy names.
                - pd.DataFrame: Statistical summary with performance metrics and comparisons.
        """

    @abstractmethod
    def run_all_hyperparameter_benchmarks(
        self,
        policy_space_info: PolicySpaceInfo,
        particles: int = 30,
        num_episodes: int = 10,
        num_steps: int = 20,
        n_trials: int = 100,
        discount_factor: float = 0.95,
        time_out_in_seconds: float = 3.0,
        evaluation_episodes: int = 3,
        evaluation_steps: int = 6,
        evaluation_n_jobs: int = 1,
        optimization_n_jobs: int = -1,
        is_risk_averse: bool = False,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        cache_dir_path: Optional[Path] = None,
        experiment_name: str = "All_Hyperparameter_Benchmarks",
        debug: bool = False,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Run all hyperparameter benchmarks with optimization.

        This method runs hyperparameter optimization for all compatible environments
        and planners for a given policy space, followed by evaluation.

        Args:
            policy_space_info: Policy space information specifying action and observation
                space types for compatibility matching.
            particles: Number of particles for belief representation.
            num_episodes: Number of episodes for optimization.
            num_steps: Maximum steps per episode for optimization.
            n_trials: Number of optimization trials.
            discount_factor: Discount factor for the MDP.
            time_out_in_seconds: Timeout for planner execution.
            evaluation_episodes: Number of episodes for evaluation.
            evaluation_steps: Maximum steps per episode for evaluation.
            evaluation_n_jobs: Number of parallel jobs for evaluation.
            optimization_n_jobs: Number of parallel jobs for optimization (-1 uses all cores).
            is_risk_averse: Whether to run risk-averse benchmark.
            confidence_interval_level: Confidence level for intervals.
            alpha: Significance level for statistical tests.
            cache_dir_path: Optional path for storing results.
            experiment_name: Name for the experiment.
            debug: Enable debug mode.
            cache_visualizations: Whether to cache visualizations.

        Returns:
            Tuple containing:
                - Dict[str, Dict[str, list]]: Raw simulation results organized by environment
                  and policy names.
                - pd.DataFrame: Statistical summary with performance metrics and comparisons.
        """
