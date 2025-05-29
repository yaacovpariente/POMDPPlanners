from typing import List, Type, Tuple, Dict
from typing import NamedTuple
import copy
from time import time
from pathlib import Path
import mlflow
import hashlib
import logging

from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    EnvironmentRunParams
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environments_policies_comparison
from POMDPPlanners.simulations.simulations_deployment import (
    LocalSimulationDeployment, 
    RemoteRaySimulationDeployment, 
    DeploymentType, 
    SimulationDeployment,
    DaskSimulationDeployment
)
from POMDPPlanners.utils.visualization import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies
)
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulations import run_episode as base_run_episode

logger = get_logger(__name__)

class POMDPSimulator:
    """A class to handle POMDP simulations and comparisons."""
    
    def __init__(
        self,
        cache_dir_path: Path = None,
        experiment_name: str = "POMDP_Planning_Comparison",
        debug: bool = False
    ):
        """Initialize the simulator.
        
        Args:
            cache_dir_path: Path to store results
            experiment_name: Name of the MLFlow experiment
            debug: Whether to enable debug logging
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        if cache_dir_path is not None:
            self._setup_mlflow_tracking()
    
    def _setup_mlflow_tracking(self) -> None:
        """Configure MLFlow tracking."""
        if self.cache_dir_path is None:
            self.cache_dir_path = Path.cwd()
        mlruns_path = self.cache_dir_path / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlruns_path
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def run_episode(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_steps: int,
    ) -> History:
        """Run a single episode without caching (delegates to simulations.py)."""
        self.logger.debug(f"Starting episode with {num_steps} steps")
        self._validate_episode_inputs(environment, policy, initial_belief, num_steps)
        return base_run_episode(environment=environment, policy=policy, initial_belief=initial_belief, num_steps=num_steps)
    
    def _validate_episode_inputs(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_steps: int,
    ) -> None:
        """Validate inputs for episode simulation."""
        assert isinstance(environment, Environment), "environment must be an Environment instance"
        assert isinstance(policy, Policy), "policy must be a Policy instance"
        assert isinstance(initial_belief, Belief), "initial_belief must be a Belief instance"
        assert isinstance(num_steps, int) and num_steps > 0, "num_steps must be a positive integer"
    
    def run_and_cache_episode(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_steps: int,
        episode_id: int,
        seed: int,
        simulation_deployment: SimulationDeployment,
    ) -> History:
        """Run an episode with optional caching support and deterministic seed."""
        self._validate_episode_inputs(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=num_steps,
        )

        if self.cache_dir_path is not None:
            general_config = {'episode_id': episode_id, 'num_steps': num_steps, 'seed': seed}

            # Try to load from cache
            try:
                cached_history = simulation_deployment.load_episode_simulation_results(
                    environment=environment,
                    policy=policy,
                    initial_belief=initial_belief,
                    cache_dir_path=self.cache_dir_path,
                    general_config=general_config
                )
                if len(cached_history) > 0:
                    self.logger.info(f"Loaded episode {episode_id} from cache for environment {environment.name} and policy {policy.name}")
                    return cached_history[0]
            except Exception as e:
                self.logger.warning(f"Failed to load episode {episode_id} from cache for environment {environment.name} and policy {policy.name}: {str(e)}")

        # Set random seed
        state = np.random.get_state()
        np.random.seed(seed)

        try:
            # Run the episode
            result = self.run_episode(
                environment=environment,
                policy=policy,
                initial_belief=initial_belief,
                num_steps=num_steps,
            )

            # Cache the result if caching is enabled
            if self.cache_dir_path is not None:
                try:
                    simulation_deployment.save_episode_simulation_results(
                        environment=environment,
                        policy=policy,
                        initial_belief=initial_belief,
                        results=[result],
                        cache_dir_path=self.cache_dir_path,
                        general_config=general_config
                    )
                    self.logger.debug(f"Cached episode {episode_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache episode {episode_id}: {str(e)}")

            return result
        finally:
            # Restore random state
            np.random.set_state(state)
    
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
    
    def _create_simulation_tasks(
        self,
        environment_run_params: List[EnvironmentRunParams],
        simulation_deployment: SimulationDeployment
    ) -> List[Dict]:
        """Create list of simulation tasks with deterministic ordering."""
        simulation_tasks = []
        total_tasks = 0
        
        for params in environment_run_params:
            for policy in params.policies:
                for episode_id in range(params.num_episodes):
                    seed = int(hashlib.md5(f"{params.environment.name}_{policy.name}_{episode_id}".encode()).hexdigest(), 16) % (2**32)
                    simulation_tasks.append({
                        'environment': params.environment,
                        'policy': policy,
                        'initial_belief': params.belief,
                        'num_steps': params.num_steps,
                        'episode_id': episode_id,
                        'seed': seed,
                        'simulation_deployment': simulation_deployment
                    })
                    total_tasks += 1

        simulation_tasks.sort(key=lambda x: x['seed'])
        self.logger.info(f"Created {total_tasks} simulation tasks across {len(set(params.environment.name for params in environment_run_params))} "
                    f"environments and {len(set(p.name for params in environment_run_params for p in params.policies))} policies")
        
        return simulation_tasks
    
    def _organize_simulation_results(
        self,
        histories_list: List[History],
        environment_belief_policy_tuples: List[Tuple[Environment, Belief, List[Policy]]],
        num_episodes: int
    ) -> Dict[str, Dict[str, List[History]]]:
        """Organize simulation results by environment and policy."""
        self.logger.info("Organizing results by environment and policy")
        results = {}
        current_idx = 0
        
        for environment, _, policies in environment_belief_policy_tuples:
            env_results = {}
            for policy in policies:
                policy_histories = histories_list[current_idx:current_idx + num_episodes]
                env_results[policy.name] = policy_histories
                current_idx += num_episodes
                self.logger.debug(f"Processed {len(policy_histories)} histories for {environment.name} with {policy.name}")
            results[environment.name] = env_results
        
        return results
    
    def simulate_multiple_environments_and_policies_parallel(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        deployment_type: DeploymentType = DeploymentType.LOCAL,
        scheduler_address: str = None  # For Dask distributed deployment
    ) -> Dict[str, Dict[str, List[History]]]:
        """Simulate multiple policies on multiple environments in parallel."""
        self._validate_parallel_simulation_inputs(
            environment_run_params=environment_run_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs
        )

        # Initialize deployment based on type
        if deployment_type == DeploymentType.LOCAL:
            simulation_deployment = LocalSimulationDeployment(n_jobs=n_jobs)
        elif deployment_type == DeploymentType.REMOTE_RAY:
            simulation_deployment = RemoteRaySimulationDeployment(num_cpus=n_jobs)
        elif deployment_type == DeploymentType.DASK_DISTRIBUTED:
            if not scheduler_address:
                raise ValueError("scheduler_address is required for Dask distributed deployment")
            simulation_deployment = DaskSimulationDeployment(
                n_jobs=n_jobs,
                scheduler_address=scheduler_address
            )
        else:
            raise ValueError(f"Unsupported deployment type: {deployment_type}")

        # Create simulation tasks
        simulation_tasks = self._create_simulation_tasks(
            environment_run_params=environment_run_params,
            simulation_deployment=simulation_deployment
        )

        # Execute simulations
        try:
            histories_list = simulation_deployment.run_multiple_episodes(
                func=self.run_and_cache_episode,
                episode_configs=simulation_tasks
            )
        except Exception as e:
            self.logger.error(f"Error running simulations: {str(e)}")
            raise e
        finally:
            # Cleanup deployment resources
            if deployment_type in [DeploymentType.REMOTE_RAY, DeploymentType.DASK_DISTRIBUTED]:
                del simulation_deployment
            
        # Organize and return results
        return self._organize_simulation_results(
            histories_list=histories_list,
            environment_belief_policy_tuples=[(params.environment, params.belief, params.policies) for params in environment_run_params],
            num_episodes=environment_run_params[0].num_episodes
        )
    
    def compare_multiple_environments_policies(
        self,
        environment_run_params: List[EnvironmentRunParams],
        alpha: float = 0.1,
        confidence_interval_level: float = 0.95,
        n_jobs: int = 1,
        cache_visualizations: bool = True,
        deployment_type: DeploymentType = DeploymentType.LOCAL,
    ) -> Tuple[Dict[str, Dict[str, List[History]]], pd.DataFrame]:
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
        assert isinstance(deployment_type, DeploymentType), "deployment_type must be a DeploymentType enum"

        # Run main comparison
        with mlflow.start_run(run_name="environment_policy_comparison"):
            # Run simulations
            histories = self.simulate_multiple_environments_and_policies_parallel(
                environment_run_params=environment_run_params,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                n_jobs=n_jobs,
                deployment_type=deployment_type
            )
            
            # Compute statistics
            statistics_df = compute_statistics_environments_policies_comparison(
                histories=histories,
                environments=[params.environment for params in environment_run_params],
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
            
            for env_name, policy_histories_dict in histories.items():
                environment = next(params.environment for params in environment_run_params if params.environment.name == env_name)
                policies = [p for params in environment_run_params for p in params.policies if p.name in policy_histories_dict]
                
                self._create_environment_visualizations(
                    env_name=env_name,
                    environment=environment,
                    policy_histories_dict=policy_histories_dict,
                    policies=policies,
                    results_dir=results_dir,
                    cache_visualizations=cache_visualizations
                )
            
            # Log all artifacts
            mlflow.log_artifact(str(results_dir), "results")

        return histories, merged_df
    
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
    
    def _create_environment_visualizations(
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