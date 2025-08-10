from typing import List, Tuple, Dict, Optional
from pathlib import Path
import mlflow
import hashlib
import logging
import shutil
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import uuid
import tempfile
import cProfile
import pstats
import io

import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    MetricValue,
    SimulationTask,
    EnvironmentRunParams,
    DataBaseInterface,
    TaskManager,
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environments_policies_comparison, metrics_dict_to_dataframe, compute_statistics_environment_policy_pair
from POMDPPlanners.utils.visualization import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies,
    plot_policies_comparison_on_environment
)
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    TaskManagerType,
)
from POMDPPlanners.simulations.simulations_deployment import TaskManagerFactory


class BaseSimulator(ABC):
    """Abstract base class for POMDP simulation frameworks.
    
    This class provides a foundation for running large-scale POMDP simulations with
    support for parallel execution, experiment tracking, and result analysis. It handles
    the orchestration of multiple environment-policy combinations and provides infrastructure
    for caching, logging, and visualization.
    
    Key Features:
    - Parallel simulation execution using configurable task managers (Dask, Joblib)
    - MLflow integration for experiment tracking and artifact logging  
    - Statistical analysis with confidence intervals and risk metrics
    - Automatic visualization generation and caching
    - Flexible caching strategies and performance profiling
    
    Attributes:
        cache_dir_path: Path for storing simulation results and artifacts
        experiment_name: Name of the MLflow experiment for tracking
        debug: Whether debug logging is enabled
        n_jobs: Number of parallel jobs for simulation execution
        task_manager: Task manager instance for handling parallel execution
        
    Example:
        Creating a custom simulator by extending BaseSimulator::
        
            from pathlib import Path
            from POMDPPlanners.simulations.simulator import BaseSimulator
            from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
            
            class CustomSimulator(BaseSimulator):
                def _create_simulation_tasks(self, environment_run_params):
                    # Implement task creation logic
                    pass
                    
                def _compute_metrics(self, results, environment_run_params, alpha, confidence_interval_level):
                    # Implement metrics computation
                    pass
                    
            # Use the custom simulator
            with CustomSimulator(
                cache_dir_path=Path("./results"),
                experiment_name="Custom_Experiment",
                task_manager_type=TaskManagerType.DASK,
                n_jobs=4,
                enable_profiling=True
            ) as simulator:
                results = simulator.compare_multiple_environments_policies(
                    environment_run_params=env_params,
                    alpha=0.05,
                    confidence_interval_level=0.95
                )
    """
    
    def __init__(
        self,
        cache_dir_path: Path = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        task_manager_type: TaskManagerType = TaskManagerType.DASK,
        n_jobs: int = 1,
        scheduler_address: Optional[str] = None,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
    ):
        """Initialize the simulator.
        
        Args:
            cache_dir_path: Path to store results
            experiment_name: Name of the MLFlow experiment
            debug: Whether to enable debug logging
            task_manager_type: Type of task manager to use for simulations
            n_jobs: Number of parallel jobs (-1 for all cores)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_dir: Directory for joblib cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            enable_profiling: Whether to enable cProfile profiling
            profiling_output_limit: Maximum number of functions to show in profiling output (default: 50)
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.debug = debug
        self.n_jobs = n_jobs
        self.enable_profiling = enable_profiling
        self.profiling_output_limit = profiling_output_limit
        self.profiler = None
        
        self.logger = get_logger(
            name=f"simulator.{experiment_name}",
            debug=debug,
            output_dir=cache_dir_path
        )
        
        # Create task manager first (this may clear cache if clear_cache_on_start is True)
        self.task_manager = self._create_task_manager(
            task_manager_type=task_manager_type,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            cache_dir=cache_dir,
            clear_cache_on_start=clear_cache_on_start
        )
        
        # Setup MLFlow tracking AFTER task manager to avoid cache clearing issues
        if cache_dir_path is not None:
            self._setup_mlflow_tracking()
    
    def _setup_mlflow_tracking(self) -> None:
        """Configure MLFlow tracking."""
        if self.cache_dir_path is None:
            self.cache_dir_path = Path.cwd()
        
        # Ensure cache directory exists
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create mlruns directory and set tracking URI to it
        mlruns_path = self.cache_dir_path / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        
        # Create .trash directory to avoid MLflow errors
        trash_path = mlruns_path / ".trash"
        trash_path.mkdir(parents=True, exist_ok=True)
        
        # Set tracking URI to mlruns directory
        tracking_uri = f"file://{mlruns_path.absolute()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def _create_task_manager(
        self,
        task_manager_type: TaskManagerType,
        n_jobs: int = 1,
        scheduler_address: Optional[str] = None,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False
    ) -> TaskManager:
        """Create a task manager of the specified type.
        
        Args:
            task_manager_type: Type of task manager to create
            n_jobs: Number of parallel jobs (-1 for all cores)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_dir: Directory for joblib cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            
        Returns:
            TaskManager: The created task manager instance
            
        Raises:
            ValueError: If task_manager_type is invalid
        """
        # Determine cache directory
        if cache_dir is None and self.cache_dir_path is not None:
            cache_dir = str(self.cache_dir_path)
        elif cache_dir is None:
            cache_dir = "./cache"

        if task_manager_type == TaskManagerType.DASK:
            return TaskManagerFactory.create_dask(
                n_workers=n_jobs,
                scheduler_address=scheduler_address,
                clear_cache_on_start=clear_cache_on_start
            )
        elif task_manager_type == TaskManagerType.JOBLIB:
            return TaskManagerFactory.create_joblib(
                cache_dir=cache_dir,
                n_jobs=n_jobs,
                clear_cache_on_start=clear_cache_on_start
            )
        else:
            raise ValueError(f"Unknown task manager type: {task_manager_type}")

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self, 'task_manager'):
            self.task_manager.__exit__(exc_type, exc_val, exc_tb)
    
    def compare_multiple_environments_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Compare multiple policies on multiple environments with optional profiling.
        
        Args:
            environment_run_params: List of environment run parameters
            alpha: Alpha value for statistics computation
            confidence_interval_level: Confidence level for statistics
            n_jobs: Number of parallel jobs
            cache_visualizations: Whether to cache visualizations
            
        Returns:
            Tuple of results dictionary and statistics DataFrame
        """
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.logger.info("Profiling enabled - starting cProfile")
            
        try:
            result = self._compare_multiple_environments_policies(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                cache_visualizations=cache_visualizations,
            )
            return result
        finally:
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                self._log_profiling_results()
    
    def _log_profiling_results(self) -> None:
        """Log profiling results to logger and save to file."""
        if not self.profiler:
            return
            
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(self.profiling_output_limit)  # Show all functions, no restriction
        
        profiling_output = s.getvalue()
        self.logger.info(f"Profiling results (all functions):\n{profiling_output}")
        
        # Save profiling results to file if cache directory is available
        if self.cache_dir_path:
            profiling_file = self.cache_dir_path / "profiling_results.txt"
            with open(profiling_file, 'w') as f:
                ps = pstats.Stats(self.profiler, stream=f).sort_stats('cumulative')
                ps.print_stats(self.profiling_output_limit)  # Show all functions, no restriction
            self.logger.info(f"Detailed profiling results saved to: {profiling_file}")
    
    def _compare_multiple_environments_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        cache_visualizations: bool = True,
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """Compare multiple policies on multiple environments and cache results in MLFlow."""
        # Log environment and algorithm names
        env_algo_info = "\n".join([
            f"Environment: {params.environment.name} - Algorithms: {[p.name for p in params.policies]}"
            for params in environment_run_params
        ])
        self.logger.info(f"Running comparison with:\n{env_algo_info}")
        
        # Type checks for all parameters
        if not isinstance(environment_run_params, list):
            raise ValueError("environment_run_params must be a list")
        if not all(isinstance(param, EnvironmentRunParams) for param in environment_run_params):
            raise ValueError("All elements in environment_run_params must be EnvironmentRunParams")
        if not isinstance(alpha, float):
            raise ValueError("alpha must be a float")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if not isinstance(confidence_interval_level, float):
            raise ValueError("confidence_interval_level must be a float")
        if not (0 < confidence_interval_level < 1):
            raise ValueError("confidence_interval_level must be between 0 and 1")
        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be an integer")
        if not (n_jobs > 0 or n_jobs == -1):
            raise ValueError("n_jobs must be a positive integer or -1")
        if not isinstance(cache_visualizations, bool):
            raise ValueError("cache_visualizations must be a boolean")

        # Run main comparison
        with mlflow.start_run(run_name="environment_policy_comparison"):
            # Log input parameters
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("confidence_interval_level", confidence_interval_level)
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("cache_visualizations", cache_visualizations)
            mlflow.log_param("num_environments", len(environment_run_params))
            
            # Log environment and policy information
            for i, params in enumerate(environment_run_params):
                env_prefix = f"env_{i}"
                mlflow.log_param(f"{env_prefix}_name", params.environment.name)
                mlflow.log_param(f"{env_prefix}_num_episodes", params.num_episodes)
                mlflow.log_param(f"{env_prefix}_num_steps", params.num_steps)
                mlflow.log_param(f"{env_prefix}_num_policies", len(params.policies))
                
                for j, policy in enumerate(params.policies):
                    policy_prefix = f"{env_prefix}_policy_{j}"
                    mlflow.log_param(f"{policy_prefix}_name", policy.name)
                    mlflow.log_param(f"{policy_prefix}_type", policy.__class__.__name__)
            
            # Run simulations
            results = self.simulate_multiple_environments_and_policies_parallel(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )
            
            metrics = self._compute_metrics(
                results=results,
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
            )
            
            # Log metrics to MLflow
            self._log_metrics_to_mlflow(metrics)
            
            # Compute statistics
            statistics_df = metrics_dict_to_dataframe(metrics_dict=metrics)

            # Create and merge policy configurations
            policy_configs_df = self._create_policy_configurations_df([(params.environment, params.belief, params.policies) for params in environment_run_params])
            merged_df = pd.merge(
                statistics_df,
                policy_configs_df,
                on=['environment', 'policy']
            )

            # Log statistics and configurations
            mlflow.log_table(merged_df, "statistics/comparison_results.json")
            mlflow.log_table(policy_configs_df, "statistics/policy_configurations.json")

            # Create policy comparison plots using the metrics
            with tempfile.TemporaryDirectory() as temp_plots_dir_str:
                temp_plots_dir = Path(temp_plots_dir_str)
                plot_policies_comparison_on_environment(
                    metrics_dict=metrics,
                    cache_dir_path=temp_plots_dir
                )
                # Log the policy comparison plots - log contents, not the directory itself
                for item in temp_plots_dir.iterdir():
                    mlflow.log_artifact(str(item), "policy_comparison_plots")

            # Create visualizations for each environment and log them directly
            def create_and_log_env_viz(
                env_name: str, 
                policy_results_dict: Dict[str, list], 
                tracking_uri: str, 
                experiment_name: str, 
                run_id: str
            ) -> None:
                # Set up MLflow context in the worker process
                import mlflow
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                
                environment = next(params.environment for params in environment_run_params if params.environment.name == env_name)
                policies = [p for params in environment_run_params for p in params.policies if p.name in policy_results_dict]
                with tempfile.TemporaryDirectory() as temp_env_dir_str:
                    temp_env_dir = Path(temp_env_dir_str)
                    self._create_environment_visualizations(
                        env_name=env_name,
                        environment=environment,
                        policy_results_dict=policy_results_dict,
                        policies=policies,
                        results_dir=temp_env_dir,
                        cache_visualizations=cache_visualizations
                    )
                    # Log the contents of this environment's directory directly under env_name
                    # Use explicit run_id to avoid context issues
                    with mlflow.start_run(run_id=run_id):
                        for item in temp_env_dir.iterdir():
                            if item.is_dir():
                                mlflow.log_artifact(str(item), env_name)
                            else:
                                mlflow.log_artifact(str(item), env_name)
            
            # Get current MLflow context for parallel workers
            current_tracking_uri = mlflow.get_tracking_uri()
            current_experiment_name = self.experiment_name
            current_run_id = mlflow.active_run().info.run_id
            
            # Run visualizations in parallel with MLflow context
            Parallel(n_jobs=n_jobs)(
                delayed(create_and_log_env_viz)(
                    env_name, 
                    policy_results_dict,
                    current_tracking_uri,
                    current_experiment_name,
                    current_run_id
                )
                for env_name, policy_results_dict in results.items()
            )

        return results, merged_df
    
    def _create_policy_configurations_df(
        self,
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]]
    ) -> pd.DataFrame:
        """Create a DataFrame containing policy configurations for all environment-policy pairs."""
        policy_configs = []
        for env, belief, policies in environment_belief_policy_tuples:
            for policy in policies:
                # Get policy parameters
                policy_params = {
                    key: value 
                    for key, value in policy.__dict__.items()
                    if isinstance(value, (int, float, str, bool))
                }
                
                # Create row with environment and policy info
                row_data = {
                    'environment': env.name,
                    'policy': policy.name,
                    'policy_type': policy.__class__.__name__,
                    **policy_params  # Add all policy parameters
                }
                policy_configs.append(row_data)
        
        return pd.DataFrame(policy_configs)
    
    def _organize_simulation_results(
        self,
        results_list: list,
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
        num_episodes: int,
        task_identifiers: List[Tuple[str, str]]
    ) -> Dict[str, Dict[str, list]]:
        """Organize simulation results by environment and policy using task identifiers.
        
        Args:
            results_list: List of results from simulation tasks
            environment_belief_policy_tuples: List of (environment, belief, policies) tuples
            num_episodes: Number of episodes per policy
            task_identifiers: List of (env_name, policy_name) tuples matching the results
            
        Returns:
            Dict mapping environment names to dicts mapping policy names to lists of task results
        """
        self.logger.info("Organizing results by environment and policy")
        
        # Initialize results structure
        results = {}
        for env, _, policies in environment_belief_policy_tuples:
            results[env.name] = {policy.name: [] for policy in policies}
        
        # Group results by their (env, policy) identifier
        for result, (env_name, policy_name) in zip(results_list, task_identifiers):
            if env_name in results and policy_name in results[env_name]:
                results[env_name][policy_name].append(result)
                self.logger.debug(f"Added result for {env_name} with {policy_name}")
            else:
                self.logger.warning(f"Received result for unknown env-policy pair: {env_name}, {policy_name}")
        
        # Verify each policy has the expected number of results
        for env, _, policies in environment_belief_policy_tuples:
            for policy in policies:
                result = results[env.name][policy.name]
                if len(result) != num_episodes:
                    self.logger.warning(
                        f"Policy {policy.name} in environment {env.name} has {len(result)} results, "
                        f"expected {num_episodes}"
                    )
        
        return results
    
    def simulate_multiple_environments_and_policies_parallel(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
    ) -> Dict[str, Dict[str, list]]:
        """Simulate multiple policies on multiple environments in parallel."""
        self._validate_parallel_simulation_inputs(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs
        )

        # Create simulation tasks and their identifiers
        simulation_tasks, task_identifiers = self._create_simulation_tasks(
            environment_run_params=environment_run_params,
        )

        # Execute simulations using TaskManager
        results_list, task_identifiers = self.task_manager.run_tasks(simulation_tasks, task_identifiers)
        
        if len(results_list) != len(task_identifiers):
            raise ValueError("results_list and task_identifiers must have the same length")
        if len(results_list) == 0:
            raise ValueError("All tasks failed.")
        
        # Organize and return results
        return self._organize_simulation_results(
            results_list=results_list,
            environment_belief_policy_tuples=[(params.environment, params.belief, params.policies) for params in environment_run_params],
            num_episodes=environment_run_params[0].num_episodes,
            task_identifiers=task_identifiers
        )
        
    def _validate_parallel_simulation_inputs(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float,
        n_jobs: int
    ) -> None:
        """Validate all input parameters for parallel simulation."""
        if not isinstance(environment_run_params, list):
            raise ValueError("environment_run_params must be a list")
        if len(environment_run_params) == 0:
            raise ValueError("environment_run_params cannot be empty")
        for params in environment_run_params:
            if not isinstance(params, EnvironmentRunParams):
                raise ValueError(f"Expected EnvironmentRunParams, got {type(params)}")
            if not isinstance(params.environment, Environment):
                raise ValueError(f"Expected Environment, got {type(params.environment)}")
            if not isinstance(params.belief, Belief):
                raise ValueError(f"Expected Belief, got {type(params.belief)}")
            if not isinstance(params.policies, list):
                raise ValueError(f"Expected list of policies, got {type(params.policies)}")
            if len(params.policies) == 0:
                raise ValueError("Policy list cannot be empty")
            for policy in params.policies:
                if not isinstance(policy, Policy):
                    raise ValueError(f"Expected Policy, got {type(policy)}")
            if not (isinstance(params.num_episodes, int) and params.num_episodes > 0):
                raise ValueError("num_episodes must be a positive integer")
            if not (isinstance(params.num_steps, int) and params.num_steps > 0):
                raise ValueError("num_steps must be a positive integer")
        env_names = [params.environment.name for params in environment_run_params]
        if len(env_names) != len(set(env_names)):
            raise ValueError("All environments must have unique names")
        all_policies = [policy for params in environment_run_params for policy in params.policies]
        policy_names = [policy.name for policy in all_policies]
        if len(policy_names) != len(set(policy_names)):
            raise ValueError("All policies must have unique names")
        if not isinstance(alpha, float):
            raise ValueError("alpha must be a float")
        if not isinstance(confidence_interval_level, float):
            raise ValueError("confidence_interval_level must be a float")
        if not (0 <= confidence_interval_level <= 1):
            raise ValueError("confidence_interval_level must be between 0 and 1")
        if not (isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1)):
            raise ValueError("n_jobs must be a positive integer or -1")
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def _compute_metrics(
        self,
        results: Dict[str, Dict[str, list]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float
    ) -> Dict[str, Dict[str, List[MetricValue]]]:
        """Compute metrics for the simulation results."""
        pass

    def _log_metrics_to_mlflow(
        self,
        metrics: Dict[str, Dict[str, List[MetricValue]]]
    ) -> None:
        """Log all metrics to MLflow for tracking and comparison.
        
        Args:
            metrics: Dictionary mapping environment names to dictionaries mapping policy names to lists of MetricValue objects
        """
        self.logger.info("Logging metrics to MLflow")
        
        for env_name, policy_metrics_dict in metrics.items():
            for policy_name, metric_list in policy_metrics_dict.items():
                # Create metric prefix for this environment-policy pair
                metric_prefix = f"{env_name}_{policy_name}"
                
                for metric in metric_list:
                    # Log the main metric value
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}", metric.value)
                    
                    # Log confidence interval bounds
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}_ci_lower", metric.lower_confidence_bound)
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}_ci_upper", metric.upper_confidence_bound)
                    
                    # Log confidence interval width for easy comparison
                    ci_width = metric.upper_confidence_bound - metric.lower_confidence_bound
                    mlflow.log_metric(f"{metric_prefix}_{metric.name}_ci_width", ci_width)
        
        self.logger.info(f"Logged metrics for {len(metrics)} environments")

    def _create_environment_visualizations(
        self,
        env_name: str,
        environment: Environment,
        policy_results_dict: Dict[str, list],
        policies: List[Policy],
        results_dir: Path,
        cache_visualizations: bool
    ):
        pass
    
class POMDPSimulator(BaseSimulator):
    """Concrete implementation of BaseSimulator for POMDP planning algorithm comparisons.
    
    This class provides a complete simulation framework for comparing POMDP planning algorithms
    across multiple environments. It executes episodes in parallel, collects comprehensive
    statistics, and generates visualizations for analysis.
    
    The simulator supports:
    - Episode-based simulation with configurable number of steps and episodes
    - Parallel execution using Dask or Joblib task managers
    - Comprehensive timing and performance metrics collection
    - Statistical analysis with confidence intervals and risk measures
    - Automatic visualization generation for individual policies and comparisons
    - MLflow experiment tracking with structured logging
    
    Key Metrics Collected:
    - Average return and discounted return statistics
    - Conditional Value at Risk (CVaR) and Value at Risk (VaR)
    - Timing metrics (action selection, belief updates, state transitions)
    - Episode completion rates and terminal state statistics
    - Custom environment-specific metrics
    
    Example:
        Comparing POMCP and Sparse Sampling on Tiger POMDP::
        
            from pathlib import Path
            from POMDPPlanners.simulations.simulator import POMDPSimulator
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
            from POMDPPlanners.core.belief import get_initial_belief
            from POMDPPlanners.core.simulation import EnvironmentRunParams
            from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
            
            # Create environment
            tiger_env = TigerPOMDP(discount_factor=0.95)
            initial_belief = get_initial_belief(tiger_env, n_particles=1000)
            
            # Create policies to compare
            pomcp = POMCP(
                environment=tiger_env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=1.0,
                name="POMCP",
                n_simulations=1000
            )
            
            sparse_sampling = StandardSparseSamplingDiscreteActionsPlanner(
                environment=tiger_env,
                branching_factor=5,
                depth=5,
                name="SparseSampling"
            )
            
            # Configure simulation parameters
            env_params = [
                EnvironmentRunParams(
                    environment=tiger_env,
                    belief=initial_belief,
                    policies=[pomcp, sparse_sampling],
                    num_episodes=100,
                    num_steps=20
                )
            ]
            
            # Run simulation with profiling
            with POMDPSimulator(
                cache_dir_path=Path("./tiger_comparison"),
                experiment_name="Tiger_POMDP_Comparison",
                task_manager_type=TaskManagerType.JOBLIB,
                n_jobs=4,
                enable_profiling=True
            ) as simulator:
                results, statistics_df = simulator.compare_multiple_environments_policies(
                    environment_run_params=env_params,
                    alpha=0.05,  # 5% risk level for CVaR
                    confidence_interval_level=0.95,
                    n_jobs=4
                )
            
            # Analyze results
            print("Simulation completed!")
            print(f"Results structure: {results.keys()}")
            print(f"Statistics shape: {statistics_df.shape}")
            
            # Access policy-specific results
            tiger_results = results['TigerPOMDP']
            pomcp_histories = tiger_results['POMCP']
            sparse_histories = tiger_results['SparseSampling']
            
            # Compare average returns
            pomcp_stats = statistics_df[statistics_df['policy'] == 'POMCP']
            sparse_stats = statistics_df[statistics_df['policy'] == 'SparseSampling']
            
            print(f"POMCP average return: {pomcp_stats['average_return'].iloc[0]:.3f}")
            print(f"Sparse Sampling average return: {sparse_stats['average_return'].iloc[0]:.3f}")
    """
    
    def __init__(
        self,
        cache_dir_path: Path = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        task_manager_type: TaskManagerType = TaskManagerType.DASK,
        n_jobs: int = 1,
        scheduler_address: Optional[str] = None,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False,
        task_console_output: bool = False,
        enable_profiling: bool = False,
        profiling_output_limit: int = 50,
    ):
        """Initialize the simulator.
        
        Args:
            cache_dir_path: Path to store results
            experiment_name: Name of the MLFlow experiment
            debug: Whether to enable debug logging
            task_manager_type: Type of task manager to use for simulations
            n_jobs: Number of parallel jobs (-1 for all cores)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_dir: Directory for joblib cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            task_console_output: Whether to enable console output for individual tasks (default: False).
                               Set to True to see console output from each task, but this can create
                               log mess when running in parallel.
            enable_profiling: Whether to enable cProfile profiling
            profiling_output_limit: Maximum number of functions to show in profiling output (default: 50)
        """
        super().__init__(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=task_manager_type,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            cache_dir=cache_dir,
            clear_cache_on_start=clear_cache_on_start,
            enable_profiling=enable_profiling,
            profiling_output_limit=profiling_output_limit,
        )
        self.task_console_output = task_console_output
            
    def _create_simulation_tasks(
        self,
        environment_run_params: List[EnvironmentRunParams],
    ) -> Tuple[List[EpisodeSimulationTask], List[Tuple[str, str]]]:
        """Create list of simulation tasks with deterministic ordering.
        
        Returns:
            Tuple containing:
            - List of simulation tasks
            - List of (env_name, policy_name) identifiers matching the tasks
        """
        simulation_tasks = []
        task_identifiers = []  # Store (env_name, policy_name) for each task
        total_tasks = 0
        
        for params in environment_run_params:
            env_name = params.environment.name
            for policy in params.policies:
                policy_name = policy.name
                for episode_id in range(params.num_episodes):
                    seed = int(hashlib.md5(f"{env_name}_{policy_name}_{episode_id}".encode()).hexdigest(), 16) % (2**32)
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
                        console_output=self.task_console_output
                    )
                    simulation_tasks.append(task)
                    task_identifiers.append((env_name, policy_name))
                    total_tasks += 1

        self.logger.info(f"Created {total_tasks} simulation tasks across {len(set(params.environment.name for params in environment_run_params))} "
                    f"environments and {len(set(p.name for params in environment_run_params for p in params.policies))} policies")
        
        return simulation_tasks, task_identifiers
        
    def _compute_metrics(
        self,
        results: Dict[str, Dict[str, List[History]]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float
    ) -> Dict[str, Dict[str, List[MetricValue]]]:
        """Compute metrics for all environment-policy pairs.
        
        Args:
            results: Dictionary mapping environment names to dictionaries mapping policy names to lists of histories
            environment_run_params: List of environment run parameters containing environments and policies
            alpha: Alpha value for statistics computation
            confidence_interval_level: Confidence level for statistics
            
        Returns:
            Dictionary mapping environment names to dictionaries mapping policy names to lists of MetricValue objects
        """
        metrics_dict = {}
        envs_dict = {params.environment.name: params.environment for params in environment_run_params}
        
        for env_name, policy_histories_dict in results.items():
            metrics_dict[env_name] = {}
            environment = envs_dict[env_name]
            
            for policy_name, histories in policy_histories_dict.items():
                # Compute statistics for this environment-policy pair
                metrics = compute_statistics_environment_policy_pair(
                    env=environment,
                    histories=histories,
                    alpha=alpha,
                    confidence_interval_level=confidence_interval_level
                )
                metrics_dict[env_name][policy_name] = metrics
                
        return metrics_dict
    
    def _create_environment_visualizations(
        self,
        env_name: str,
        environment: Environment,
        policy_results_dict: Dict[str, list],
        policies: List[Policy],
        results_dir: Path,
        cache_visualizations: bool
    ) -> None:
        """Create and save visualizations for a specific environment."""
        self._create_environment_visualizations_custom(
            env_name=env_name,
            environment=environment,
            policy_histories_dict=policy_results_dict,
            policies=policies,
            results_dir=results_dir,
            cache_visualizations=cache_visualizations
        )
    
    def _create_environment_visualizations_custom(
        self,
        env_name: str,
        environment: Environment,
        policy_histories_dict: Dict[str, List[History]],
        policies: List[Policy],
        results_dir: Path,
        cache_visualizations: bool
    ) -> None:
        """Create and save visualizations for a specific environment."""
        # Don't create env_dir since the entire results_dir is already the environment directory
        # env_dir = results_dir / env_name
        # env_dir.mkdir(exist_ok=True)
        
        for policy_name, policy_histories in policy_histories_dict.items():
            policy = next(p for p in policies if p.name == policy_name)
            self._create_policy_visualizations(
                policy=policy,
                environment=environment,
                policy_histories=policy_histories,
                env_dir=results_dir,  # Use results_dir directly instead of env_dir
                cache_visualizations=cache_visualizations
            )
        
        # Create comparison plot
        comparison_plot_path = results_dir / "policy_comparison_histogram.png"  # Use results_dir directly
        plot_discounted_returns_histogram_multiple_policies(
            histories=policy_histories_dict,
            policies=policies,
            environment=environment,
            cache_path=comparison_plot_path
        )
    
    def _create_policy_visualizations(
        self,
        policy: Policy,
        environment: Environment,
        policy_histories: List[History],
        env_dir: Path,
        cache_visualizations: bool
    ) -> None:
        """Create and save visualizations for a specific policy."""
        policy_dir = env_dir / policy.name
        policy_dir.mkdir(exist_ok=True)
        
        # Create plots
        plots_dir = policy_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "discounted_returns_histogram.png"
        plot_discounted_returns_histogram(
            histories=policy_histories,
            policy=policy,
            environment=environment,
            cache_path=plot_path
        )
        
        if cache_visualizations:
            self._cache_episode_visualizations(
                environment=environment,
                policy_histories=policy_histories,
                policy_dir=policy_dir
            )
    
    def _cache_episode_visualizations(
        self,
        environment: Environment,
        policy_histories: List[History],
        policy_dir: Path
    ) -> None:
        """Cache visualizations for individual episodes."""
        viz_dir = policy_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for episode_idx, history in enumerate(policy_histories):
            file_name = f"agent_path_{episode_idx}.gif"
            cache_path = viz_dir / file_name
            try:
                environment.cache_visualization(
                    history=history,
                    cache_path=cache_path
                )
            except Exception as e:
                self.logger.warning(f"Visualization failed for episode {episode_idx}: {str(e)}") 