"""Hyperparameter optimization module for POMDP policies.

This module provides tools for optimizing hyperparameters of POMDP policies using
Optuna optimization framework with MLFlow experiment tracking. It supports both
single environment-policy optimization and multi-environment comparison studies.

The module integrates with the POMDPPlanners simulation framework to run parallel
episodes and compute performance statistics for hyperparameter evaluation.

Key Features:
    - Advanced optimization algorithms via Optuna (TPE, CMA-ES, etc.)
    - Parallel episode execution with caching support
    - Comprehensive MLFlow experiment tracking and visualization
    - Support for both categorical and numerical hyperparameters
    - Multi-environment comparison studies
    - Statistical analysis with confidence intervals
    - Automatic visualization generation and artifact logging

Classes:
    HyperParameterOptimizer: Main class for conducting hyperparameter optimization studies
    
Type Aliases:
    HyperParameterFeatures: Union type for categorical or numerical hyperparameter definitions

Example:
    Basic hyperparameter optimization workflow::
    
        from pathlib import Path
        from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterOptimizer
        from POMDPPlanners.core.simulation import NumericalHyperParameter
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        
        # Set up optimization
        optimizer = HyperParameterOptimizer(
            cache_dir_path=Path("./optimization_results"),
            experiment_name="POMCP_Tiger_Optimization",
            n_jobs=4,
            mlflow_tracking_uri=Path("/shared/mlflow_tracking")  # Optional: custom tracking directory
        )
        
        # Define parameter search space
        param_ranges = [
            NumericalHyperParameter("exploration_constant", 0.1, 10.0),
            NumericalHyperParameter("n_simulations", 100, 2000),
            NumericalHyperParameter("depth", 10, 100)
        ]
        
        # Run optimization
        best_params, best_value, histories = optimizer.optimize_policy_parameters(
            environment=TigerPOMDP(),
            policy_class=POMCP,
            param_ranges=param_ranges,
            num_episodes=50,          # Final evaluation episodes
            num_steps=100,
            n_particles=100,
            n_trials=100,
            num_episodes_tuning=10    # Faster tuning with fewer episodes
        )
        
        print(f"Best parameters: {best_params}")
        print(f"Best performance: {best_value}")

Note:
    This module requires Optuna and MLFlow to be installed. The optimization process
    can be computationally intensive for complex policies and large parameter spaces.
    Consider using distributed computing capabilities for large-scale studies.
"""

from typing import List, Type, Tuple, Dict, Optional, Union
from pathlib import Path
import mlflow
import optuna
import pandas as pd
import numpy as np
import tempfile
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import (
    History,
    StepData,
    CategoricalHyperParameter,
    NumericalHyperParameter,
    MetricValue,
)
from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair, compute_statistics_environments_policies_comparison
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
from POMDPPlanners.utils.visualization import plot_metrics_comparison, plot_discounted_returns_histogram, plot_environment_policy_pair_comparison, plot_discounted_returns_histogram_multiple_policies
from POMDPPlanners.utils.logger import get_logger

# TODO: change all of the assertions to raised exceptions

logger = get_logger(__name__)

HyperParameterFeatures = Union[CategoricalHyperParameter, NumericalHyperParameter]
"""Type alias for hyperparameter feature definitions.

Supports both categorical parameters (with discrete choice sets) and numerical 
parameters (with continuous or integer ranges) for optimization studies.
"""

class HyperParameterOptimizer:
    """Hyperparameter optimization for POMDP policies using Optuna and MLFlow.
    
    This class provides a complete framework for optimizing POMDP policy hyperparameters
    using Optuna's advanced optimization algorithms with integrated MLFlow experiment
    tracking. It supports parallel episode execution, statistical analysis, and
    visualization generation for comprehensive optimization studies.
    
    The optimizer integrates with the POMDPSimulator framework to leverage parallel
    computation and caching capabilities for efficient hyperparameter search across
    large parameter spaces.
    
    Attributes:
        cache_dir_path: Directory for storing optimization results and cached data
        experiment_name: Name of the MLFlow experiment for tracking
        n_jobs: Number of parallel jobs for episode execution
        confidence_interval_level: Statistical confidence level for metrics
        mlflow_tracking_uri: File URI for MLFlow tracking (always local file storage)
        mlruns_path: Path to MLFlow experiment tracking directory on local filesystem
        simulator: POMDPSimulator instance for parallel episode execution
        
    Example:
        Basic hyperparameter optimization::
        
            from pathlib import Path
            from POMDPPlanners.core.simulation import NumericalHyperParameter
            
            optimizer = HyperParameterOptimizer(
                cache_dir_path=Path("./optimization_cache"),
                experiment_name="POMCP_Tuning",
                n_jobs=4,
                mlflow_tracking_uri=Path("/shared/mlflow_tracking")  # Custom local directory
            )
            
            param_ranges = [
                NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                NumericalHyperParameter("n_simulations", 100, 1000)
            ]
            
            best_params, best_value, histories = optimizer.optimize_policy_parameters(
                environment=env,
                policy_class=POMCP,
                param_ranges=param_ranges,
                num_episodes=50,          # Final evaluation episodes
                num_steps=100,
                n_particles=20,
                n_trials=20,
                num_episodes_tuning=10    # Faster tuning with fewer episodes
            )
            
        Combined optimization and evaluation workflow::\
        
            # For comprehensive studies with detailed evaluation
            env_policy_pairs = [
                (tiger_env, (POMCP, [
                    NumericalHyperParameter(low=0.1, high=10.0, name="exploration_constant"),
                    NumericalHyperParameter(low=100, high=1000, name="n_simulations")
                ])),
                (lightdark_env, (POMCP, [
                    NumericalHyperParameter(low=0.1, high=10.0, name="exploration_constant"), 
                    NumericalHyperParameter(low=100, high=1000, name="n_simulations")
                ]))
            ]
            
            # Run optimization followed by comprehensive evaluation
            opt_results, opt_df, eval_results = optimizer.optimize_and_evaluate_multiple_environments(
                environment_policy_pairs=env_policy_pairs,
                num_episodes_tuning=10,      # Fast optimization
                num_episodes_evaluation=100, # Thorough evaluation
                num_steps=50,
                n_particles=100,
                n_trials=20,
                cache_visualizations=True    # Generate plots and animations
            )
            
            # Access optimization results
            for i, (best_params, best_value, histories) in enumerate(opt_results):
                env_name = env_policy_pairs[i][0].name
                print(f"{env_name} - Best params: {best_params}, Value: {best_value}")
            
            # Access comprehensive evaluation results with full simulator capabilities
            for env_name, policy_results in eval_results.items():
                for policy_name, policy_histories in policy_results.items():
                    print(f"{env_name}/{policy_name}: {len(policy_histories)} evaluation episodes")
    """
    
    def __init__(
        self,
        cache_dir_path: Path,
        experiment_name: str = "POMDP_Parameter_Optimization",
        n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        mlflow_tracking_uri: Optional[Path] = None,
    ):
        """Initialize the hyperparameter optimizer.
        
        Sets up MLFlow experiment tracking, creates the POMDPSimulator instance
        for parallel episode execution, and configures optimization parameters.
        
        Args:
            cache_dir_path: Directory path for storing optimization results,
                MLFlow experiments, and simulation cache. Will be created if
                it doesn't exist.
            experiment_name: Name for the MLFlow experiment. Used for organizing
                optimization runs and tracking results. Defaults to
                "POMDP_Parameter_Optimization".
            n_jobs: Number of parallel jobs for episode execution. Higher values
                can significantly speed up optimization but require more system
                resources. Defaults to 1 for single-threaded execution.
            confidence_interval_level: Statistical confidence level for metrics
                computation (between 0.0 and 1.0). Used for computing confidence
                intervals in performance statistics. Defaults to 0.95 for 95%
                confidence intervals.
            mlflow_tracking_uri: Path to custom MLFlow tracking directory on local machine.
                If None (default), uses cache_dir_path/mlruns for local file storage.
                If provided, must be a Path object pointing to the desired MLflow 
                tracking directory on the local filesystem.
                
        Raises:
            ValueError: If confidence_interval_level is not between 0.0 and 1.0
            TypeError: If cache_dir_path is not a Path object
        """
        self.cache_dir_path = cache_dir_path
        self.experiment_name = experiment_name
        self.n_jobs = n_jobs
        self.confidence_interval_level = confidence_interval_level
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Set up MLFlow tracking
        if mlflow_tracking_uri is None:
            # Use local file storage in cache_dir_path/mlruns
            self.mlruns_path = cache_dir_path / "mlruns"
        else:
            # Use user-provided tracking path
            self.mlruns_path = Path(mlflow_tracking_uri)
            
        # Create the mlruns directory and set up MLflow
        self.mlruns_path.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_uri = f"file://{self.mlruns_path.absolute()}"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Create simulator instance for running episodes
        self.simulator = POMDPSimulator(
            cache_dir_path=cache_dir_path,
            experiment_name=f"{experiment_name}_episodes", 
            task_manager_type=TaskManagerType.JOBLIB,
            n_jobs=n_jobs,
            debug=False  # Keep episode-level logging minimal for optimization
        )

    def run_multiple_episodes(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_episodes: int,
        num_steps: int,
        scheduler_address: Optional[str] = None,
    ) -> List[History]:
        """Run multiple episodes in parallel using the POMDPSimulator.
        
        Executes multiple POMDP episodes in parallel for the given environment-policy
        pair, leveraging the POMDPSimulator's parallel execution capabilities. This
        method is used internally during hyperparameter optimization to evaluate
        candidate parameter configurations efficiently.
        
        Args:
            environment: The POMDP environment to run episodes in
            policy: The policy to execute during episodes
            initial_belief: Initial belief state for all episodes
            num_episodes: Number of episodes to run in parallel
            num_steps: Maximum number of steps per episode
            scheduler_address: Optional Dask scheduler address for distributed
                computation. If None, uses local parallelization.
                
        Returns:
            List of History objects, one for each completed episode, containing
            the complete trajectory of states, actions, observations, rewards,
            and beliefs for statistical analysis.
            
        Raises:
            AssertionError: If any input parameter validation fails
            RuntimeError: If the simulator fails to execute episodes
            
        Note:
            This method uses the POMDPSimulator's direct parallel execution method
            to avoid creating additional MLflow experiments during optimization.
            Episodes are executed with caching and parallel processing support.
        """
        logger.info(f"Starting {num_episodes} episodes with {num_steps} steps each using {self.n_jobs} jobs")
        logger.info(f"Environment: {environment.name}, Policy: {policy.name}")
        
        assert isinstance(environment, Environment)
        assert isinstance(policy, Policy)
        assert isinstance(initial_belief, Belief)
        if scheduler_address is not None:
            assert isinstance(scheduler_address, str)

        assert num_episodes > 0
        assert num_steps > 0

        # Create EnvironmentRunParams for the simulator
        env_run_params = [
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ]

        # Use simulator's direct parallel execution method to avoid MLflow experiment creation
        results = self.simulator.simulate_multiple_environments_and_policies_parallel(
            environment_run_params=env_run_params,
            alpha=0.05,  # Default alpha for intermediate results
            confidence_interval_level=self.confidence_interval_level,
            n_jobs=self.n_jobs,
        )
        
        # Extract histories from results
        histories = results[environment.name][policy.name]
        logger.info(f"All episodes completed for {environment.name} with {policy.name}")

        return histories

    def simulation(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_episodes: int,
        num_steps: int,
        alpha: float,
        scheduler_address: Optional[str] = None,
    ) -> Tuple[List[History], List[MetricValue]]:
        """Run a complete simulation study and compute performance statistics.
        
        Executes multiple episodes for a given environment-policy configuration
        and computes comprehensive performance statistics including confidence
        intervals, returns, completion rates, and timing metrics.
        
        This method combines episode execution with statistical analysis to provide
        a complete evaluation of a policy's performance, making it suitable for
        hyperparameter optimization objective function evaluation.
        
        Args:
            environment: The POMDP environment for simulation
            policy: The policy to evaluate
            initial_belief: Initial belief state for episodes
            num_episodes: Number of episodes to run for statistical reliability
            num_steps: Maximum steps per episode before termination
            alpha: Significance level for confidence intervals (e.g., 0.05 for 95% CI)
            scheduler_address: Optional Dask scheduler for distributed execution
            
        Returns:
            Tuple containing:
                - List of History objects from all executed episodes
                - List of MetricValue objects with computed performance statistics
                  including average return, success rates, timing metrics, and
                  confidence intervals
                  
        Raises:
            AssertionError: If input validation fails
            ValueError: If alpha is not in valid range (0, 1)
            RuntimeError: If simulation execution fails
            
        Note:
            Statistics are computed using the configured confidence interval level
            from initialization. The alpha parameter controls the significance
            level for hypothesis testing within the statistics computation.
        """
        logger.info(f"Starting simulation with {num_episodes} episodes, {num_steps} steps, "
                    f"alpha={alpha}, confidence_interval={self.confidence_interval_level}")
        
        assert isinstance(environment, Environment)
        assert isinstance(policy, Policy)
        assert isinstance(initial_belief, Belief)
        assert isinstance(num_episodes, int)
        assert isinstance(num_steps, int)
        if scheduler_address is not None:
            assert isinstance(scheduler_address, str)

        assert 1 >= self.confidence_interval_level >= 0
        assert num_episodes > 0
        assert num_steps > 0

        histories = self.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=num_episodes,
            num_steps=num_steps,
            scheduler_address=scheduler_address,
        )

        logger.info("Computing statistics from simulation results")
        statistics = compute_statistics_environment_policy_pair(
            env=environment,
            histories=histories,
            alpha=alpha,
            confidence_interval_level=self.confidence_interval_level,
        )
        logger.info("Statistics computation completed")

        return histories, statistics

    def optimize_policy_parameters(
        self,
        environment: Environment,
        policy_class: Type[Policy],
        param_ranges: List[HyperParameterFeatures],
        num_episodes: int,
        num_steps: int,
        n_particles: int,
        parameter_to_optimize: str = "average_return",
        direction: str = "maximize",
        n_trials: int = 100,
        num_episodes_tuning: Optional[int] = None,
    ) -> Tuple[dict, float, List[History]]:
        """Optimize hyperparameters for a single environment-policy pair.
        
        Conducts hyperparameter optimization using Optuna's advanced algorithms
        to find the best parameter configuration for a given policy class in
        the specified environment. The optimization process evaluates candidate
        parameters by running multiple episodes and optimizing the specified
        performance metric.
        
        This method supports both continuous and discrete parameter spaces,
        automatically handling parameter type detection and appropriate Optuna
        suggestion methods. All trials are tracked in MLFlow for analysis.
        
        Args:
            environment: The POMDP environment for optimization
            policy_class: Policy class to optimize (must be Policy subclass)
            param_ranges: List of hyperparameter definitions specifying the
                search space. Each element should be either CategoricalHyperParameter
                or NumericalHyperParameter defining parameter name and valid range.
            num_episodes: Number of episodes for final evaluation of the best 
                parameters found. Used after optimization is complete to get 
                accurate performance estimates. Higher values provide more reliable 
                final evaluation but only affect post-optimization assessment.
            num_steps: Maximum steps per episode before forced termination
            n_particles: Number of belief particles for belief state representation
            parameter_to_optimize: Name of the performance metric to optimize.
                Must match a metric name returned by the statistics computation.
                Common options: "average_return", "success_rate", "average_steps".
                Defaults to "average_return".
            direction: Optimization direction, either "maximize" or "minimize".
                Defaults to "maximize".
            n_trials: Number of optimization trials to conduct. More trials
                generally find better solutions but require more computation.
                Defaults to 100.
            num_episodes_tuning: Number of episodes per trial during hyperparameter
                tuning. If None, uses num_episodes. Setting this lower than 
                num_episodes allows faster tuning with less accurate estimates,
                while final evaluation uses num_episodes for accuracy. Defaults to None.
                
        Returns:
            Tuple containing:
                - dict: Best parameter configuration found during optimization
                - float: Best objective value achieved
                - List[History]: Episode histories from the best trial for analysis
                
        Raises:
            AssertionError: If input validation fails
            ValueError: If parameter_to_optimize is not found in computed statistics
            TypeError: If policy_class is not a Policy subclass
            
        Example:
            Optimize POMCP exploration constant and simulation count::
            
                param_ranges = [
                    NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                    NumericalHyperParameter("n_simulations", 100, 2000)
                ]
                
                best_params, best_value, histories = optimizer.optimize_policy_parameters(
                    environment=tiger_env,
                    policy_class=POMCP,
                    param_ranges=param_ranges,
                    num_episodes=30,          # Final evaluation episodes
                    num_steps=50,
                    n_particles=100,
                    n_trials=50,
                    num_episodes_tuning=10    # Faster tuning with fewer episodes
                )
                
        Note:
            The optimization process creates policy instances dynamically by
            passing the environment and suggested parameters to the policy
            constructor. Ensure the policy class accepts all specified parameters.
        """
        assert isinstance(environment, Environment)
        assert isinstance(policy_class, type)
        assert issubclass(policy_class, Policy)
        assert isinstance(param_ranges, list)
        assert all(isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)) for param in param_ranges)
        assert isinstance(num_episodes, int)
        assert isinstance(num_steps, int)
        assert isinstance(n_particles, int)
        assert isinstance(parameter_to_optimize, str)
        assert isinstance(n_trials, int)
        if num_episodes_tuning is not None:
            assert isinstance(num_episodes_tuning, int)
            assert num_episodes_tuning > 0

        assert num_episodes > 0
        assert num_steps > 0
        assert n_particles > 0
        assert direction in ["maximize", "minimize"]
        
        # Set default tuning episodes if not provided
        episodes_for_tuning = num_episodes_tuning if num_episodes_tuning is not None else num_episodes
        
        logger.info(f"Starting hyperparameter optimization with {episodes_for_tuning} episodes per trial")
        if episodes_for_tuning < num_episodes:
            logger.info(f"Final evaluation will use {num_episodes} episodes for accuracy")
        logger.info(f"Optimizing {parameter_to_optimize} using {n_trials} trials")

        def evaluation_function(policy: Policy, trial: optuna.Trial) -> float:
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            histories, statistics = self.simulation(
                environment=environment,
                policy=policy,
                initial_belief=initial_belief,
                num_episodes=episodes_for_tuning,  # Use tuning episodes count
                num_steps=num_steps,
                alpha=0.05,  # Fixed alpha for optimization
                scheduler_address=None,
            )

            # Store histories as trial user attributes
            trial.set_user_attr("histories", histories)

            # Get the current value
            for metric in statistics:
                if metric.name == parameter_to_optimize:
                    return metric.value

            raise ValueError(f"Parameter {parameter_to_optimize} not found in statistics")

        def objective(trial: optuna.Trial) -> float:
            # Create parameters dictionary from hyperparameters
            policy_params = {"environment": environment}

            # Add optimization parameters
            for param in param_ranges:
                if isinstance(param, CategoricalHyperParameter):
                    policy_params[param.name] = trial.suggest_categorical(
                        param.name, param.choices
                    )
                elif isinstance(param, NumericalHyperParameter):
                    if isinstance(param.low, float):
                        policy_params[param.name] = trial.suggest_float(
                            param.name, param.low, param.high
                        )
                    else:
                        policy_params[param.name] = trial.suggest_int(
                            param.name, param.low, param.high
                        )

            # Create policy instance with suggested parameters
            policy = policy_class(**policy_params)

            # Evaluate policy and return objective value
            return evaluation_function(policy, trial)

        # Create and run the study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        # Get best parameters and value
        best_params = study.best_params
        best_value = study.best_value

        # If tuning used fewer episodes than requested for evaluation, 
        # run a final evaluation with the full episode count for accuracy
        if episodes_for_tuning < num_episodes:
            logger.info(f"Running final evaluation with {num_episodes} episodes (tuning used {episodes_for_tuning})")
            # Create policy with best parameters for final evaluation
            best_policy = policy_class(environment=environment, **best_params)
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            
            # Run final evaluation with full episode count
            histories, statistics = self.simulation(
                environment=environment,
                policy=best_policy,
                initial_belief=initial_belief,
                num_episodes=num_episodes,  # Use full episode count for final evaluation
                num_steps=num_steps,
                alpha=0.05,
                scheduler_address=None,
            )
            
            # Update best_value with the more accurate evaluation
            for metric in statistics:
                if metric.name == parameter_to_optimize:
                    best_value = metric.value
                    break
        else:
            # Get histories from the best trial (tuning and evaluation used same episode count)
            best_trial = study.best_trial
            histories = best_trial.user_attrs["histories"]

        return best_params, best_value, histories

    def optimize_multiple_environments(
        self,
        environment_policy_pairs: List[
            Tuple[Environment, Tuple[Type[Policy], List[HyperParameterFeatures]]]
        ],
        num_episodes: int,
        num_steps: int,
        n_particles: int,
        parameter_to_optimize: str = "average_return",
        direction: str = "maximize",
        n_trials: int = 100,
        num_episodes_tuning: Optional[int] = None,
    ) -> Tuple[List[Tuple[dict, float, List[History]]], pd.DataFrame]:
        """Conduct hyperparameter optimization across multiple environment-policy pairs.
        
        Performs comprehensive hyperparameter optimization studies across multiple
        environment-policy combinations, enabling comparison of optimal parameters
        across different domains or policy classes. Each environment-policy pair
        is optimized independently with full MLFlow tracking and visualization.
        
        This method is particularly useful for:
        - Comparing policy performance across different environments
        - Finding domain-specific optimal parameters
        - Conducting comprehensive benchmarking studies
        - Analysis of hyperparameter sensitivity across domains
        
        Results are automatically logged to MLFlow with detailed parameter tracking,
        performance metrics, and generated comparison visualizations.
        
        Args:
            environment_policy_pairs: List of tuples, each containing an environment
                and a tuple of (policy_class, param_ranges). The param_ranges define
                the hyperparameter search space for that specific policy class.
            num_episodes: Number of episodes for final evaluation of best parameters.
                Applied to all environment-policy pairs for consistent final assessment.
                Higher values provide more reliable final performance estimates.
            num_steps: Maximum steps per episode across all optimizations
            n_particles: Number of belief particles for all environment simulations
            parameter_to_optimize: Performance metric name to optimize across all
                pairs. Must be consistently available in all environments' statistics.
                Defaults to "average_return".
            direction: Optimization direction for all pairs ("maximize" or "minimize").
                Defaults to "maximize".
            n_trials: Number of optimization trials per environment-policy pair.
                Total computation scales linearly with number of pairs.
            num_episodes_tuning: Number of episodes per trial during hyperparameter
                tuning across all pairs. If None, uses num_episodes. Setting this
                lower allows faster tuning with final evaluation using num_episodes
                for accuracy. Defaults to None.
                
        Returns:
            Tuple containing:
                - List of optimization results, one per environment-policy pair.
                  Each result is a tuple of (best_params, best_value, histories).
                - pandas.DataFrame with aggregated results for analysis, containing
                  parameter ranges, best parameters, performance metrics, and
                  environment/policy metadata.
                  
        Raises:
            AssertionError: If input validation fails for any pair
            ValueError: If parameter_to_optimize is not found in any environment's statistics
            RuntimeError: If optimization fails for any environment-policy pair
            
        Example:
            Compare POMCP vs PFT-DPW across multiple environments::
            
                env_policy_pairs = [
                    (tiger_env, (POMCP, [
                        NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                        NumericalHyperParameter("n_simulations", 100, 1000)
                    ])),
                    (tiger_env, (PFT_DPW, [
                        NumericalHyperParameter("k_a", 0.5, 2.0),
                        NumericalHyperParameter("alpha_a", 0.3, 0.7)
                    ])),
                    (lightdark_env, (POMCP, [
                        NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                        NumericalHyperParameter("n_simulations", 100, 1000)
                    ]))
                ]
                
                results, df = optimizer.optimize_multiple_environments(
                    env_policy_pairs,
                    num_episodes=50,
                    num_steps=100,
                    n_particles=100,
                    n_trials=30
                )
                
        Note:
            Each environment-policy pair creates a separate MLFlow run for
            independent tracking. Generated visualizations compare performance
            across all pairs and are saved as MLFlow artifacts.
        """
        assert isinstance(environment_policy_pairs, List)
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in environment_policy_pairs
        )
        assert all(isinstance(env, Environment) for env, _ in environment_policy_pairs)
        assert all(
            isinstance(policy_config, tuple) and len(policy_config) == 2
            for _, policy_config in environment_policy_pairs
        )
        assert all(
            isinstance(policy_class, type) and issubclass(policy_class, Policy)
            for _, (policy_class, _) in environment_policy_pairs
        )
        assert all(
            isinstance(param_ranges, list)
            and all(isinstance(param, (CategoricalHyperParameter, NumericalHyperParameter)) for param in param_ranges)
            for _, (_, param_ranges) in environment_policy_pairs
        )

        results = []
        planner_statistics = []
        aggregated_data = []

        for pair_idx, (environment, (policy_class, param_ranges)) in enumerate(
            environment_policy_pairs
        ):
            # End any active run safely
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run()

            # Start a new MLFlow run for each environment-policy combination
            with mlflow.start_run(
                run_name=f"{environment.__class__.__name__}_{policy_class.__name__}_{pair_idx}"
            ):
                # Log common parameters
                common_params = {
                    "environment_type": environment.__class__.__name__,
                    "policy_type": policy_class.__name__,
                    "num_episodes": num_episodes,
                    "num_steps": num_steps,
                    "n_particles": n_particles,
                    "parameter_to_optimize": parameter_to_optimize,
                    "direction": direction,
                    "n_trials": n_trials,
                    "confidence_interval_level": self.confidence_interval_level,
                }

                # Log environment-specific parameters
                env_params = {
                    f"env_{key}": value
                    for key, value in environment.__dict__.items()
                    if isinstance(value, (int, float, str, bool))
                }

                # Log policy-specific parameter ranges
                policy_param_ranges = {}
                for param in param_ranges:
                    if isinstance(param, CategoricalHyperParameter):
                        policy_param_ranges[f"param_range_{param.name}"] = (
                            f"choices: {param.choices}"
                        )
                    elif isinstance(param, NumericalHyperParameter):
                        policy_param_ranges[f"param_range_{param.name}"] = (
                            f"{param.low}-{param.high}"
                        )

                # Combine all parameters
                all_params = {**common_params, **env_params, **policy_param_ranges}
                mlflow.log_params(all_params)

                # Run optimization
                best_params, best_value, histories = self.optimize_policy_parameters(
                    environment=environment,
                    policy_class=policy_class,
                    param_ranges=param_ranges,
                    num_episodes=num_episodes,
                    num_steps=num_steps,
                    n_particles=n_particles,
                    parameter_to_optimize=parameter_to_optimize,
                    direction=direction,
                    n_trials=n_trials,
                    num_episodes_tuning=num_episodes_tuning,
                )

                # Get statistics from the best trial
                initial_belief = get_initial_belief(
                    pomdp=environment, n_particles=n_particles
                )
                _, statistics = self.simulation(
                    environment=environment,
                    policy=policy_class(environment=environment, **best_params),
                    initial_belief=initial_belief,
                    num_episodes=num_episodes,
                    num_steps=num_steps,
                    alpha=0.05,  # Fixed alpha for evaluation
                )

                # Log best parameters and value
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("best_value", best_value)

                # Save the full results as a JSON artifact
                results_data = {
                    "best_parameters": best_params,
                    "best_value": best_value,
                    "parameters": all_params,
                    "param_ranges": [param._asdict() for param in param_ranges],
                    "statistics": [metric._asdict() for metric in statistics],
                }
                mlflow.log_dict(results_data, f"optimization_results_pair_{pair_idx}.json")

                results.append((best_params, best_value, histories))
                planner_statistics.append(statistics)

                # Aggregate data for DataFrame
                run_data = {
                    **all_params,
                    **{f"best_{k}": v for k, v in best_params.items()},
                    "best_value": best_value,
                    **{metric.name: metric.value for metric in statistics},
                }
                aggregated_data.append(run_data)

        # Convert aggregated data to DataFrame
        df = pd.DataFrame(aggregated_data)

        # Log DataFrame as table artifact through MLFlow
        mlflow.log_table(df, "optimization_results.json")

        # Plot statistics comparison
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            plot_metrics_comparison(
                statistics=planner_statistics,
                environments=[env for env, (policy_class, _) in environment_policy_pairs],
                policies=[policy_class for _, (policy_class, _) in environment_policy_pairs],
                cache_dir_path=temp_dir_path,
            )

            # Log all generated plots as artifacts
            for plot_file in temp_dir_path.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "statistics_plots")

        # End any active run safely before returning
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()

        return results, df

    def optimize_and_evaluate_multiple_environments(
        self,
        environment_policy_pairs: List[
            Tuple[Environment, Tuple[Type[Policy], List[HyperParameterFeatures]]]
        ],
        num_episodes_tuning: int,
        num_episodes_evaluation: int,
        num_steps: int,
        n_particles: int,
        parameter_to_optimize: str = "average_return",
        direction: str = "maximize",
        n_trials: int = 100,
        alpha: float = 0.05,
        confidence_interval_level: Optional[float] = None,
        cache_visualizations: bool = True,
    ) -> Tuple[List[Tuple[dict, float, List[History]]], pd.DataFrame, Dict[str, Dict[str, List[History]]]]:
        """Optimize hyperparameters and then evaluate optimized policies using the full simulator.
        
        This method combines hyperparameter optimization with comprehensive policy evaluation.
        First, it optimizes hyperparameters for each environment-policy pair using a smaller
        number of episodes for speed. Then, it evaluates the optimized policies using the 
        full simulator with more episodes for accuracy, generating complete statistics and
        visualizations.
        
        The evaluation phase uses POMDPSimulator to provide comprehensive analysis including:
        - Detailed statistical metrics with confidence intervals
        - Policy comparison visualizations and plots
        - MLFlow experiment tracking with artifacts
        - Parallel execution for efficient evaluation
        
        Args:
            environment_policy_pairs: List of tuples, each containing an environment
                and a tuple of (policy_class, param_ranges). The param_ranges define
                the hyperparameter search space for that specific policy class.
            num_episodes_tuning: Number of episodes per trial during hyperparameter
                optimization. Should be relatively small for faster optimization.
            num_episodes_evaluation: Number of episodes for final evaluation using
                the full simulator. Should be larger for accurate performance assessment.
            num_steps: Maximum steps per episode across all optimizations and evaluations
            n_particles: Number of belief particles for all environment simulations
            parameter_to_optimize: Performance metric name to optimize across all
                pairs. Must be consistently available in all environments' statistics.
                Defaults to "average_return".
            direction: Optimization direction for all pairs ("maximize" or "minimize").
                Defaults to "maximize".
            n_trials: Number of optimization trials per environment-policy pair.
                Total computation scales linearly with number of pairs.
            alpha: Significance level for confidence intervals in evaluation phase.
                Defaults to 0.05 for 5% significance level.
            confidence_interval_level: Confidence level for metrics computation during
                evaluation. If None, uses the optimizer's configured level.
            cache_visualizations: Whether to generate and cache detailed visualizations
                during evaluation phase. Defaults to True.
                
        Returns:
            Tuple containing:
                - List of optimization results, one per environment-policy pair.
                  Each result is a tuple of (best_params, best_value, tuning_histories).
                - pandas.DataFrame with aggregated optimization results for analysis.
                - Dictionary with evaluation results mapping environment names to
                  dictionaries mapping policy names to lists of evaluation histories.
                  
        Raises:
            AssertionError: If input validation fails for any pair
            ValueError: If parameter_to_optimize is not found in any environment's statistics
            RuntimeError: If optimization or evaluation fails for any environment-policy pair
            
        Example:
            Optimize and evaluate POMCP across multiple environments::\
            
                env_policy_pairs = [
                    (tiger_env, (POMCP, [
                        NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                        NumericalHyperParameter("n_simulations", 100, 1000)
                    ])),
                    (lightdark_env, (POMCP, [
                        NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                        NumericalHyperParameter("n_simulations", 100, 1000)
                    ]))
                ]
                
                opt_results, opt_df, eval_results = optimizer.optimize_and_evaluate_multiple_environments(
                    environment_policy_pairs=env_policy_pairs,
                    num_episodes_tuning=10,      # Fast optimization
                    num_episodes_evaluation=100, # Comprehensive evaluation
                    num_steps=50,
                    n_particles=100,
                    n_trials=20,
                    cache_visualizations=True
                )
                
        Note:
            This method creates two phases of MLFlow tracking:
            1. Optimization phase: Individual runs for each environment-policy optimization
            2. Evaluation phase: Comprehensive comparison using the full simulator
            The evaluation phase provides detailed visualizations and statistical analysis.
        """
        # Validate confidence interval level
        eval_confidence_level = confidence_interval_level if confidence_interval_level is not None else self.confidence_interval_level
        
        assert isinstance(num_episodes_tuning, int) and num_episodes_tuning > 0
        assert isinstance(num_episodes_evaluation, int) and num_episodes_evaluation > 0
        assert isinstance(alpha, float) and 0 < alpha < 1
        assert isinstance(eval_confidence_level, float) and 0 < eval_confidence_level < 1
        assert isinstance(cache_visualizations, bool)
        
        logger.info(f"Starting optimization and evaluation with {len(environment_policy_pairs)} environment-policy pairs")
        logger.info(f"Optimization: {num_episodes_tuning} episodes per trial, {n_trials} trials each")
        logger.info(f"Evaluation: {num_episodes_evaluation} episodes per optimized policy")
        
        # Phase 1: Hyperparameter Optimization
        logger.info("Phase 1: Running hyperparameter optimization")
        optimization_results, optimization_df = self.optimize_multiple_environments(
            environment_policy_pairs=environment_policy_pairs,
            num_episodes=num_episodes_evaluation,  # Use evaluation episodes for final accuracy
            num_steps=num_steps,
            n_particles=n_particles,
            parameter_to_optimize=parameter_to_optimize,
            direction=direction,
            n_trials=n_trials,
            num_episodes_tuning=num_episodes_tuning,  # Use tuning episodes for speed
        )
        
        # Phase 2: Comprehensive Evaluation using POMDPSimulator
        logger.info("Phase 2: Running comprehensive evaluation with optimized parameters")
        
        # Create optimized policies with best parameters for evaluation
        evaluation_env_params = []
        for pair_idx, (environment, (policy_class, param_ranges)) in enumerate(environment_policy_pairs):
            # Get the best parameters for this pair
            best_params, best_value, _ = optimization_results[pair_idx]
            
            # Create optimized policy instance
            optimized_policy = policy_class(environment=environment, **best_params)
            
            # Create initial belief
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            
            # Create evaluation parameters
            env_run_params = EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[optimized_policy],
                num_episodes=num_episodes_evaluation,
                num_steps=num_steps
            )
            evaluation_env_params.append(env_run_params)
        
        # Use the simulator for comprehensive evaluation with visualization and tracking
        evaluation_results, evaluation_df = self.simulator.compare_multiple_environments_policies(
            environment_run_params=evaluation_env_params,
            alpha=alpha,
            confidence_interval_level=eval_confidence_level,
            n_jobs=self.n_jobs,
            cache_visualizations=cache_visualizations,
        )
        
        logger.info("Optimization and evaluation completed successfully")
        logger.info(f"Optimization results: {len(optimization_results)} pairs optimized")
        logger.info(f"Evaluation results: {len(evaluation_results)} environments evaluated")
        
        return optimization_results, optimization_df, evaluation_results

