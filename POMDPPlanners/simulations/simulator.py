from typing import List, Tuple, Dict, Optional
from pathlib import Path
import mlflow
import hashlib
import logging
from abc import ABC, abstractmethod

import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    SimulationTask,
    EnvironmentRunParams,
    DataBaseInterface,
    TaskManager,
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environments_policies_comparison
from POMDPPlanners.utils.visualization import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies
)
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    TaskManagerType,
)
from POMDPPlanners.simulations.simulations_deployment import TaskManagerFactory

logger = get_logger(__name__)

class BaseSimulator(ABC):
    """A base class for POMDP simulators."""
    
    def __init__(
        self,
        cache_dir_path: Path = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        task_manager_type: TaskManagerType = TaskManagerType.DASK,
        n_jobs: int = 1,
        scheduler_address: Optional[str] = None,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False
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
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Setup MLFlow tracking if cache directory is provided
        if cache_dir_path is not None:
            self._setup_mlflow_tracking()
            
        # Create task manager
        self.task_manager = self._create_task_manager(
            task_manager_type=task_manager_type,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            cache_dir=cache_dir,
            clear_cache_on_start=clear_cache_on_start
        )
    
    def _setup_mlflow_tracking(self) -> None:
        """Configure MLFlow tracking."""
        if self.cache_dir_path is None:
            self.cache_dir_path = Path.cwd()
        mlruns_path = self.cache_dir_path / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlruns_path
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
            cache_dir = str(self.cache_dir_path / "cache")
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
        """Compare multiple policies on multiple environments and cache results in MLFlow."""
        # Type checks for all parameters
        assert isinstance(environment_run_params, list), "environment_run_params must be a list"
        assert all(isinstance(param, EnvironmentRunParams) for param in environment_run_params), "All elements in environment_run_params must be EnvironmentRunParams"
        
        assert isinstance(alpha, float), "alpha must be a float"
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
        
        assert isinstance(confidence_interval_level, float), "confidence_interval_level must be a float"
        assert 0 < confidence_interval_level < 1, "confidence_interval_level must be between 0 and 1"
        
        assert isinstance(n_jobs, int), "n_jobs must be an integer"
        assert n_jobs > 0 or n_jobs == -1, "n_jobs must be positive or -1 (for all available cores)"
        
        assert isinstance(cache_visualizations, bool), "cache_visualizations must be a boolean"

        # Run main comparison
        with mlflow.start_run(run_name="environment_policy_comparison"):
            # Run simulations
            results = self.simulate_multiple_environments_and_policies_parallel(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
            )
            
            # Compute statistics
            statistics_df = self._compute_statistics_df(
                results=results,
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
            )

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

            # Create results directory and visualizations
            results_dir = self.cache_dir_path / "results"
            results_dir.mkdir(exist_ok=True)
            
            for env_name, policy_results_dict in results.items():
                environment = next(params.environment for params in environment_run_params if params.environment.name == env_name)
                policies = [p for params in environment_run_params for p in params.policies if p.name in policy_results_dict]
                
                self._create_environment_visualizations(
                    env_name=env_name,
                    environment=environment,
                    policy_results_dict=policy_results_dict,
                    policies=policies,
                    results_dir=results_dir,
                    cache_visualizations=cache_visualizations
                )
            
            # Log all artifacts
            mlflow.log_artifact(str(results_dir), "results")

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
        histories_list: List[History],
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
        num_episodes: int,
        task_identifiers: List[Tuple[str, str]]
    ) -> Dict[str, Dict[str, list]]:
        """Organize simulation results by environment and policy using task identifiers.
        
        Args:
            histories_list: List of histories from simulation tasks
            environment_belief_policy_tuples: List of (environment, belief, policies) tuples
            num_episodes: Number of episodes per policy
            task_identifiers: List of (env_name, policy_name) tuples matching the histories
            
        Returns:
            Dict mapping environment names to dicts mapping policy names to lists of task results
        """
        self.logger.info("Organizing results by environment and policy")
        
        # Initialize results structure
        results = {}
        for env, _, policies in environment_belief_policy_tuples:
            results[env.name] = {policy.name: [] for policy in policies}
        
        # Group histories by their (env, policy) identifier
        for history, (env_name, policy_name) in zip(histories_list, task_identifiers):
            if env_name in results and policy_name in results[env_name]:
                results[env_name][policy_name].append(history)
                self.logger.debug(f"Added history for {env_name} with {policy_name}")
            else:
                self.logger.warning(f"Received history for unknown env-policy pair: {env_name}, {policy_name}")
        
        # Verify each policy has the expected number of histories
        for env, _, policies in environment_belief_policy_tuples:
            for policy in policies:
                histories = results[env.name][policy.name]
                if len(histories) != num_episodes:
                    self.logger.warning(
                        f"Policy {policy.name} in environment {env.name} has {len(histories)} histories, "
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
        histories_list = self.task_manager.run_tasks(simulation_tasks)
            
        # Organize and return results
        return self._organize_simulation_results(
            histories_list=histories_list,
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
        assert isinstance(environment_run_params, list), "environment_run_params must be a list"
        assert len(environment_run_params) > 0, "environment_run_params cannot be empty"
        
        for params in environment_run_params:
            assert isinstance(params, EnvironmentRunParams), f"Expected EnvironmentRunParams, got {type(params)}"
            assert isinstance(params.environment, Environment), f"Expected Environment, got {type(params.environment)}"
            assert isinstance(params.belief, Belief), f"Expected Belief, got {type(params.belief)}"
            assert isinstance(params.policies, list), f"Expected list of policies, got {type(params.policies)}"
            assert len(params.policies) > 0, "Policy list cannot be empty"
            for policy in params.policies:
                assert isinstance(policy, Policy), f"Expected Policy, got {type(policy)}"
            assert isinstance(params.num_episodes, int) and params.num_episodes > 0, "num_episodes must be a positive integer"
            assert isinstance(params.num_steps, int) and params.num_steps > 0, "num_steps must be a positive integer"

        # Verify unique environment names
        env_names = [params.environment.name for params in environment_run_params]
        assert len(env_names) == len(set(env_names)), "All environments must have unique names"

        # Verify unique policy names across all environments
        all_policies = [policy for params in environment_run_params for policy in params.policies]
        policy_names = [policy.name for policy in all_policies]
        assert len(policy_names) == len(set(policy_names)), "All policies must have unique names"

        assert isinstance(alpha, float), "alpha must be a float"
        assert isinstance(confidence_interval_level, float), "confidence_interval_level must be a float"
        assert 0 <= confidence_interval_level <= 1, "confidence_interval_level must be between 0 and 1"
        assert isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1), "n_jobs must be a positive integer or -1"
    
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
    def _compute_statistics_df(
        self,
        results: Dict[str, Dict[str, List[History]]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float
    ) -> pd.DataFrame:
        """Compute statistics for the simulation results.
        
        Args:
            results: Dictionary mapping environment names to dictionaries mapping policy names to lists of histories
            environment_run_params: List of environment run parameters containing environment, belief, and policy info
            alpha: Alpha value for statistics computation (e.g. for CVaR)
            confidence_interval_level: Confidence level for statistics (e.g. 0.95 for 95% CI)
            
        Returns:
            DataFrame with the following structure:
            - Required columns: 'environment' (str), 'policy' (str) - used for merging
            - Metric columns: Each metric from compute_statistics_environment_policy_pair
            - For each metric X, includes:
              - X: The metric value
              - X_ci_lower: Lower confidence bound
              - X_ci_upper: Upper confidence bound
            - One row per environment-policy pair
        """
        pass
    
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
    """A class to handle POMDP simulations and comparisons."""
    
    def __init__(
        self,
        cache_dir_path: Path = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False,
        task_manager_type: TaskManagerType = TaskManagerType.DASK,
        n_jobs: int = 1,
        scheduler_address: Optional[str] = None,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False
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
        """
        super().__init__(
            cache_dir_path=cache_dir_path,
            experiment_name=experiment_name,
            debug=debug,
            task_manager_type=task_manager_type,
            n_jobs=n_jobs,
            scheduler_address=scheduler_address,
            cache_dir=cache_dir,
            clear_cache_on_start=clear_cache_on_start
        )
            
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
                        episode_number=episode_id
                    )
                    simulation_tasks.append(task)
                    task_identifiers.append((env_name, policy_name))
                    total_tasks += 1

        self.logger.info(f"Created {total_tasks} simulation tasks across {len(set(params.environment.name for params in environment_run_params))} "
                    f"environments and {len(set(p.name for params in environment_run_params for p in params.policies))} policies")
        
        return simulation_tasks, task_identifiers
    
    def _organize_simulation_results(
        self,
        histories_list: List[History],
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
        num_episodes: int,
        task_identifiers: List[Tuple[str, str]]
    ) -> Dict[str, Dict[str, List[History]]]:
        """Organize simulation results by environment and policy using task identifiers.
        
        Args:
            histories_list: List of histories from simulation tasks
            environment_belief_policy_tuples: List of (environment, belief, policies) tuples
            num_episodes: Number of episodes per policy
            task_identifiers: List of (env_name, policy_name) tuples matching the histories
            
        Returns:
            Dict mapping environment names to dicts mapping policy names to lists of histories
        """
        return super()._organize_simulation_results(
            histories_list=histories_list,
            environment_belief_policy_tuples=environment_belief_policy_tuples,
            num_episodes=num_episodes,
            task_identifiers=task_identifiers
        )
    
    def simulate_multiple_environments_and_policies_parallel(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
    ) -> Dict[str, Dict[str, List[History]]]:
        """Simulate multiple policies on multiple environments in parallel."""
        return super().simulate_multiple_environments_and_policies_parallel(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs
        )
    
    def _compute_statistics_df(
        self,
        results: Dict[str, Dict[str, List[History]]],
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float
    ) -> pd.DataFrame:
        """Compute statistics for the simulation results.
        
        Args:
            results: Dictionary mapping environment names to dictionaries mapping policy names to lists of histories
            environment_run_params: List of environment run parameters containing environment, belief, and policy info
            alpha: Alpha value for statistics computation (e.g. for CVaR)
            confidence_interval_level: Confidence level for statistics (e.g. 0.95 for 95% CI)
            
        Returns:
            DataFrame with the following structure:
            - Required columns: 'environment' (str), 'policy' (str) - used for merging
            - Metric columns: Each metric from compute_statistics_environment_policy_pair
            - For each metric X, includes:
              - X: The metric value
              - X_ci_lower: Lower confidence bound
              - X_ci_upper: Upper confidence bound
            - One row per environment-policy pair
        """
        return compute_statistics_environments_policies_comparison(
            histories=results,
            environments=[params.environment for params in environment_run_params],
            alpha=alpha,
            confidence_interval_level=confidence_interval_level
        )
    
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
        env_dir = results_dir / env_name
        env_dir.mkdir(exist_ok=True)
        
        for policy_name, policy_histories in policy_histories_dict.items():
            policy = next(p for p in policies if p.name == policy_name)
            self._create_policy_visualizations(
                policy=policy,
                environment=environment,
                policy_histories=policy_histories,
                env_dir=env_dir,
                cache_visualizations=cache_visualizations
            )
        
        # Create comparison plot
        comparison_plot_path = env_dir / "policy_comparison_histogram.png"
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