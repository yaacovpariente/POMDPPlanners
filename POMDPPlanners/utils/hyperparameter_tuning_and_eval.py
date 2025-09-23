"""Utility functions for hyperparameter tuning and evaluation of POMDP planners.

This module provides high-level functions for conducting hyperparameter optimization
followed by comprehensive evaluation using the POMDPSimulator. It encapsulates the
workflow from the hyper_param_runner.py example into reusable utility functions.

Key Features:
- Automated hyperparameter optimization using Optuna with MLflow tracking
- Policy evaluation using POMDPSimulator with comprehensive metrics
- Flexible configuration for different planners and environments
- Fast execution with optimized default parameters
- Comprehensive result reporting and visualization

Example:
    Basic usage for multiple planners on Tiger POMDP::

        from pathlib import Path
        from POMDPPlanners.utils.hyperparameter_tuning_utils import optimize_and_evaluate_planners, HyperParamPlannerConfig
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.sparse_sampling_planners.standard_sparse_sampling_discrete_actions import StandardSparseSamplingDiscreteActionsPlanner
        from POMDPPlanners.core.simulation import NumericalHyperParameter
        from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterOptimizationDirection
        from POMDPPlanners.core.belief import get_initial_belief

        # Set up environment
        env = TigerPOMDP(discount_factor=0.95, name="Tiger_095")
        initial_belief = get_initial_belief(env, n_particles=100)

        # Define planner configurations
        planner_configs = [
            # POMCP configuration
            HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=[
                    NumericalHyperParameter(0.1, 5.0, "exploration_constant"),
                    NumericalHyperParameter(50, 200, "n_simulations"),
                    NumericalHyperParameter(3, 8, "depth")
                ],
                constant_parameters={"discount_factor": env.discount_factor, "name": "OptimizedPOMCP"}
            ),
            # Sparse Sampling configuration
            HyperParamPlannerConfig(
                policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
                hyper_parameters=[
                    NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
                    NumericalHyperParameter(100, 500, "n_simulations")
                ],
                constant_parameters={"discount_factor": env.discount_factor, "name": "OptimizedSSDAP"}
            )
        ]

        # Run optimization and evaluation for all planners
        results = optimize_and_evaluate_planners(
            environment=env,
            initial_belief=initial_belief,
            planner_configs=planner_configs,
            cache_dir=Path("./optimization_results"),
            optimization_direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return"
        )

        # Access results
        optimization_results = results['optimization_results']  # List of OptimizedPolicyResult
        evaluation_results = results['evaluation_results']
        evaluation_statistics = results['evaluation_statistics']

        # Log results (logger is automatically available when importing this module)
        logger.info(f"Optimized {len(optimization_results)} planners")
        for i, result in enumerate(optimization_results):
            logger.info(f"Planner {i+1}: {result.policy.name} - Best hyperparameters: {result.chosen_hyper_parameters}")
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Type
from pathlib import Path
import pandas as pd
import logging

from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter,
    CategoricalHyperParameter,
    EnvironmentRunParams,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
    OptimizedPolicyResult,
)
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    TaskManagerType,
)


# Set up logger for this module
# The logger will output messages based on the logging level configured by the user
# Use configure_logging() function to set appropriate levels based on debug/verbose parameters
logger = logging.getLogger(__name__)


def configure_logging(debug: bool = False, verbose: bool = True) -> None:
    """Configure logging levels based on debug and verbose parameters.

    This function allows users to control the verbosity of logging output:
    - debug=True: Shows all messages (DEBUG, INFO, WARNING, ERROR)
    - verbose=True: Shows INFO, WARNING, and ERROR messages
    - verbose=False: Shows only WARNING and ERROR messages

    Args:
        debug: Whether to enable debug logging (most verbose)
        verbose: Whether to enable info logging (moderate verbosity)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif verbose:
        logger.setLevel(logging.INFO)
        logger.debug("Info logging enabled")
    else:
        logger.setLevel(logging.WARNING)
        logger.debug("Warning-only logging enabled")


# Type aliases for better code readability
HyperParameterFeature = Union[NumericalHyperParameter, CategoricalHyperParameter]


class HyperParamPlannerConfig:
    def __init__(
        self,
        policy_cls: type,
        hyper_parameters: List[HyperParameterFeature],
        constant_parameters: Dict[str, Any],
    ):
        self.policy_cls = policy_cls
        self.hyper_parameters = hyper_parameters
        self.constant_parameters = constant_parameters


def optimize_and_evaluate_planners(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    optimization_direction: HyperParameterOptimizationDirection = HyperParameterOptimizationDirection.MAXIMIZE,
    parameter_to_optimize: str = "average_return",
    experiment_name: str = "planner_optimization",
    # Optimization parameters
    optimization_episodes: int = 3,
    optimization_steps: int = 6,
    n_trials: int = 3,
    optimization_n_jobs: int = -1,
    # Evaluation parameters
    evaluation_episodes: int = 10,
    evaluation_steps: int = 8,
    evaluation_n_jobs: int = 1,
    # General parameters
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Perform hyperparameter optimization followed by comprehensive evaluation on multiple planners.

    This function provides a complete workflow for optimizing multiple POMDP planners'
    hyperparameters and then evaluating all optimized policies' performance together.

    Args:
        environment: The POMDP environment to optimize and evaluate on
        initial_belief: Initial belief state for the environment
        planner_configs: List of HyperParamPlannerConfig objects, each containing
                        a policy class, hyperparameters, and constant parameters
        cache_dir: Directory for storing optimization and evaluation results
        optimization_direction: Direction of optimization (MAXIMIZE or MINIMIZE)
        parameter_to_optimize: Name of the metric to optimize (e.g., "average_return")
        experiment_name: Name for the experiment (used in MLflow tracking)
        optimization_episodes: Number of episodes for optimization trials
        optimization_steps: Number of steps per optimization episode
        n_trials: Number of optimization trials to run per planner
        optimization_n_jobs: Number of parallel jobs for optimization (-1 for all cores)
        evaluation_episodes: Number of episodes for final evaluation
        evaluation_steps: Number of steps per evaluation episode
        evaluation_n_jobs: Number of parallel jobs for evaluation
        confidence_interval_level: Confidence level for statistical analysis
        alpha: Alpha value for risk metrics (CVaR, VaR)
        debug: Whether to enable debug logging
        verbose: Whether to print progress messages

    Returns:
        Dictionary containing:
        - 'optimization_results': List of OptimizedPolicyResult objects, one per planner
        - 'evaluation_results': Raw episode histories from evaluation for all planners
        - 'evaluation_statistics': DataFrame with evaluation statistics for all planners
        - 'cache_paths': Dictionary with paths to optimization and evaluation results
        - 'summary': Summary information including all optimized planners

    Raises:
        ValueError: If optimization fails for any planner or returns no results
        TypeError: If any policy_cls is not a valid Policy subclass

    Example:
        Optimize multiple planners on Tiger POMDP::

            planner_configs = [
                HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=[
                        NumericalHyperParameter(0.1, 5.0, "exploration_constant"),
                        NumericalHyperParameter(50, 200, "n_simulations")
                    ],
                    constant_parameters={"discount_factor": 0.95, "name": "POMCP"}
                ),
                HyperParamPlannerConfig(
                    policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
                    hyper_parameters=[
                        NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
                        NumericalHyperParameter(100, 500, "n_simulations")
                    ],
                    constant_parameters={"discount_factor": 0.95, "name": "SSSDAP"}
                )
            ]

            results = optimize_and_evaluate_planners(
                environment=TigerPOMDP(discount_factor=0.95),
                initial_belief=get_initial_belief(env, n_particles=100),
                planner_configs=planner_configs,
                cache_dir=Path("./results")
            )
    """
    # Configure logging based on parameters
    configure_logging(debug=debug, verbose=verbose)

    if verbose:
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            f"Starting hyperparameter optimization and evaluation for {len(planner_configs)} planners: {planner_names}"
        )
        logger.info(f"Environment: {environment.name}")
        logger.info(
            f"Optimization: {n_trials} trials per planner, {optimization_episodes} episodes, {optimization_steps} steps"
        )
        logger.info(f"Evaluation: {evaluation_episodes} episodes, {evaluation_steps} steps")

    # Step 1: Run hyperparameter optimization for all planners in a single call
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*60}")

    if debug:
        logger.debug(f"Starting optimization with {len(planner_configs)} planner configurations")
        for i, config in enumerate(planner_configs):
            logger.debug(
                f"Planner {i+1}: {config.policy_cls.__name__} with {len(config.hyper_parameters)} hyperparameters"
            )

    # Optimize all planners together in a single optimization run
    optimization_results = optimize_planner_hyperparameters(
        environment=environment,
        initial_belief=initial_belief,
        planner_configs=planner_configs,
        cache_dir=cache_dir,
        optimization_direction=optimization_direction,
        parameter_to_optimize=parameter_to_optimize,
        experiment_name=experiment_name,
        num_episodes=optimization_episodes,
        num_steps=optimization_steps,
        n_trials=n_trials,
        n_jobs=optimization_n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
    )

    if not optimization_results or len(optimization_results) != len(planner_configs):
        error_msg = f"Hyperparameter optimization failed - expected {len(planner_configs)} results but got {len(optimization_results) if optimization_results else 0}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract optimized policies
    optimized_policies = [result.policy for result in optimization_results]

    if verbose:
        logger.info(f"\n✓ Optimization completed for all {len(planner_configs)} planners!")
        for i, (result, config) in enumerate(zip(optimization_results, planner_configs)):
            logger.info(f"  {i+1}. {config.policy_cls.__name__}: {result.policy.name}")
            logger.info(f"     Best hyperparameters: {result.chosen_hyper_parameters}")

    # Step 2: Evaluate all optimized policies together
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: POLICY EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating {len(optimized_policies)} optimized policies together...")

    if debug:
        logger.debug(f"Starting evaluation with {len(optimized_policies)} optimized policies")
        for i, policy in enumerate(optimized_policies):
            logger.debug(f"Policy {i+1}: {policy.name} (type: {type(policy).__name__})")

    evaluation_results, evaluation_statistics = evaluate_multiple_optimized_planners(
        environment=environment,
        optimized_policies=optimized_policies,
        initial_belief=initial_belief,
        cache_dir=cache_dir,
        experiment_name=f"{experiment_name}_evaluation",
        num_episodes=evaluation_episodes,
        num_steps=evaluation_steps,
        n_jobs=evaluation_n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
    )

    if verbose:
        logger.info(f"\n✓ Evaluation completed successfully!")
        total_episodes = sum(
            len(evaluation_results[environment.name][policy.name]) for policy in optimized_policies
        )
        logger.info(
            f"Evaluated {total_episodes} total episodes across {len(optimized_policies)} policies"
        )

    # Prepare result summary
    cache_paths = {
        "optimization_cache": cache_dir,
        "evaluation_cache": cache_dir / "evaluation",
        "optimization_mlruns": cache_dir / "mlruns",
        "evaluation_mlruns": cache_dir / "evaluation" / "mlruns",
    }

    if debug:
        logger.debug("Preparing results summary")
        logger.debug(f"Cache paths: {cache_paths}")

    results = {
        "optimization_results": optimization_results,  # List of results, one per planner
        "evaluation_results": evaluation_results,
        "evaluation_statistics": evaluation_statistics,
        "cache_paths": cache_paths,
        "summary": {
            "planners": [
                {
                    "policy_name": result.policy.name,
                    "policy_type": planner_configs[i].policy_cls.__name__,
                    "best_hyperparameters": result.chosen_hyper_parameters,
                }
                for i, result in enumerate(optimization_results)
            ],
            "environment_name": environment.name,
            "num_planners": len(planner_configs),
            "optimization_trials_per_planner": n_trials,
            "evaluation_episodes": evaluation_episodes,
        },
    }

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION AND EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Environment: {environment.name}")
        logger.info(f"Optimized planners: {len(optimization_results)}")
        for i, result in enumerate(optimization_results):
            planner_type = planner_configs[i].policy_cls.__name__
            logger.info(f"  {i+1}. {result.policy.name} ({planner_type})")
        logger.info(f"Results saved to: {cache_dir}")
        logger.info(f"MLflow tracking: {cache_paths['optimization_mlruns']} (optimization)")
        logger.info(f"MLflow tracking: {cache_paths['evaluation_mlruns']} (evaluation)")

    if debug:
        logger.debug("Final results summary prepared and returning")

    return results


def optimize_planner_hyperparameters(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    optimization_direction: HyperParameterOptimizationDirection = HyperParameterOptimizationDirection.MAXIMIZE,
    parameter_to_optimize: str = "average_return",
    experiment_name: str = "planner_optimization",
    num_episodes: int = 3,
    num_steps: int = 6,
    n_trials: int = 3,
    n_jobs: int = -1,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.1,
    debug: bool = False,
    verbose: bool = True,
) -> List[OptimizedPolicyResult]:
    """Optimize hyperparameters for multiple POMDP planners using Optuna in a single optimization run.

    Args:
        environment: The POMDP environment to optimize on
        initial_belief: Initial belief state for the environment
        planner_configs: List of HyperParamPlannerConfig objects containing planner configurations
        cache_dir: Directory for storing optimization results
        optimization_direction: Direction of optimization (MAXIMIZE or MINIMIZE)
        parameter_to_optimize: Name of the metric to optimize
        experiment_name: Name for the optimization experiment
        num_episodes: Number of episodes per optimization trial
        num_steps: Number of steps per episode
        n_trials: Number of optimization trials per planner
        n_jobs: Number of parallel jobs (-1 for all cores)
        confidence_interval_level: Confidence level for statistics
        alpha: Alpha value for statistics computation
        debug: Whether to enable debug logging
        verbose: Whether to print progress messages

    Returns:
        List of OptimizedPolicyResult objects, one for each planner configuration
    """
    if verbose:
        total_hyperparams = sum(len(config.hyper_parameters) for config in planner_configs)
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            f"Optimizing {len(planner_configs)} planners ({planner_names}) with {total_hyperparams} total hyperparameters using {n_trials} trials each..."
        )

    if debug:
        logger.debug(f"Creating HyperParameterOptimizer with {n_jobs} jobs")
        logger.debug(f"Confidence interval level: {confidence_interval_level}, Alpha: {alpha}")

    # Create optimizer
    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        n_jobs=n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
    )

    # Configure optimization parameters for all planners
    if debug:
        logger.debug("Creating optimization configurations for all planners")

    optimization_configs = []
    for i, planner_config in enumerate(planner_configs):
        if debug:
            logger.debug(f"Creating config for planner {i+1}: {planner_config.policy_cls.__name__}")

        config = HyperParameterRunParams(
            environment=environment,
            belief=initial_belief,
            policy_cls=planner_config.policy_cls,
            hyper_parameters=planner_config.hyper_parameters,
            constant_parameters=planner_config.constant_parameters,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            direction=optimization_direction,
            parameter_to_optimize=parameter_to_optimize,
        )
        optimization_configs.append(config)

    if debug:
        logger.debug(f"Created {len(optimization_configs)} optimization configurations")

    # Run optimization for all planners in a single call
    if verbose:
        logger.info(f"Running optimization with:")
        for i, planner_config in enumerate(planner_configs):
            logger.info(f"  - Planner {i+1}: {planner_config.policy_cls.__name__}")
            logger.info(
                f"    Hyperparameters: {[hp.name for hp in planner_config.hyper_parameters]}"
            )
        logger.info(f"  - Environment: {environment.name}")
        logger.info(f"  - Trials per planner: {n_trials}")
        logger.info(f"  - Episodes per trial: {num_episodes}")
        logger.info(f"  - Steps per episode: {num_steps}")

    if debug:
        logger.debug("Starting optimization with HyperParameterOptimizer")

    optimization_results = optimizer.optimize(optimization_configs)

    if debug:
        logger.debug(
            f"Optimization completed, received {len(optimization_results) if optimization_results else 0} results"
        )

    # Return all results
    if optimization_results:
        if verbose:
            logger.info(f"Optimization completed for {len(optimization_results)} planners")
        if debug:
            logger.debug(f"Returning {len(optimization_results)} optimization results")
        return optimization_results
    else:
        if verbose:
            logger.warning("Warning: No optimization results returned")
        if debug:
            logger.debug("No optimization results to return")
        return []


def evaluate_optimized_planner(
    environment: Environment,
    optimized_policy: Policy,
    initial_belief: Belief,
    cache_dir: Path,
    experiment_name: str = "planner_evaluation",
    num_episodes: int = 10,
    num_steps: int = 8,
    n_jobs: int = 1,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Evaluate an optimized POMDP planner using comprehensive simulation.

    Args:
        environment: The POMDP environment to evaluate on
        optimized_policy: The optimized policy to evaluate
        initial_belief: Initial belief state for evaluation
        cache_dir: Directory for storing evaluation results
        experiment_name: Name for the evaluation experiment
        num_episodes: Number of episodes for evaluation
        num_steps: Number of steps per episode
        n_jobs: Number of parallel jobs for evaluation
        confidence_interval_level: Confidence level for statistics
        alpha: Alpha value for risk metrics
        debug: Whether to enable debug logging
        verbose: Whether to print progress messages

    Returns:
        Tuple containing:
        - Raw episode results organized by environment and policy
        - DataFrame with comprehensive evaluation statistics
    """
    # Create evaluation cache directory
    eval_cache_dir = cache_dir
    eval_cache_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        logger.debug(f"Evaluation cache directory: {eval_cache_dir}")
        logger.debug(
            f"Evaluation parameters: {num_episodes} episodes, {num_steps} steps, {n_jobs} jobs"
        )

    if verbose:
        logger.info(f"Evaluating policy '{optimized_policy.name}' on {num_episodes} episodes...")

    # Create environment run parameters for evaluation
    eval_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[optimized_policy],
            num_episodes=num_episodes,
            num_steps=num_steps,
        )
    ]

    # Run evaluation using POMDPSimulator
    if debug:
        logger.debug(f"Creating POMDPSimulator with cache_dir: {eval_cache_dir}")
        logger.debug(f"Experiment name: {experiment_name}, Task manager: {TaskManagerType.JOBLIB}")

    with POMDPSimulator(
        cache_dir_path=eval_cache_dir,
        experiment_name=experiment_name,
        task_manager_type=TaskManagerType.JOBLIB,
        n_jobs=n_jobs,
        debug=debug,
        task_console_output=False,  # Reduce output noise
        enable_profiling=False,
    ) as simulator:
        if debug:
            logger.debug("POMDPSimulator created successfully, starting evaluation...")

        # Run the evaluation
        if debug:
            logger.debug("Running evaluation with POMDPSimulator...")

        results, statistics_df = simulator.compare_multiple_environments_policies(
            environment_run_params=eval_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )

        if debug:
            logger.debug(
                f"Evaluation completed, received results for {len(results) if results else 0} environments"
            )
            if statistics_df is not None:
                logger.debug(f"Statistics DataFrame shape: {statistics_df.shape}")

        # Display evaluation summary if verbose
        if verbose:
            env_name = environment.name
            policy_name = optimized_policy.name

            if env_name in results and policy_name in results[env_name]:
                policy_histories = results[env_name][policy_name]

                logger.info(f"\n📊 EVALUATION SUMMARY")
                logger.info(f"-" * 40)
                logger.info(f"Environment: {env_name}")
                logger.info(f"Policy: {policy_name}")
                logger.info(f"Episodes completed: {len(policy_histories)}")
                logger.info(f"Steps per episode: {num_steps}")

                # Display key statistics
                if not statistics_df.empty:
                    if verbose:
                        logger.info(
                            f"\n📊 Statistics DataFrame columns: {list(statistics_df.columns)}"
                        )
                        logger.info(f"📊 Statistics DataFrame shape: {statistics_df.shape}")

                    if debug:
                        logger.debug(
                            f"Processing statistics DataFrame with {len(statistics_df)} rows"
                        )
                        logger.debug(f"DataFrame columns: {list(statistics_df.columns)}")

                    # Check if DataFrame has expected structure
                    if "policy" in statistics_df.columns:
                        policy_stats = statistics_df[statistics_df["policy"] == policy_name]
                        if not policy_stats.empty:
                            logger.info(f"\n📈 KEY METRICS:")
                            logger.info(f"-" * 40)

                            # Check if metric column exists
                            if "metric" in statistics_df.columns:
                                # Show important metrics
                                key_metrics = [
                                    "average_return",
                                    "return_cvar",
                                    "average_actual_num_steps",
                                ]
                                for _, row in policy_stats.iterrows():
                                    metric_name = row["metric"]
                                    if metric_name in key_metrics:
                                        value = row["value"]
                                        ci_lower = row["lower_confidence_bound"]
                                        ci_upper = row["upper_confidence_bound"]
                                        logger.info(
                                            f"{metric_name:25}: {value:8.3f} [{ci_lower:6.3f}, {ci_upper:6.3f}]"
                                        )
                            else:
                                # Alternative: display all available statistics
                                logger.info(f"Available statistics for {policy_name}:")
                                for col in policy_stats.columns:
                                    if col != "policy":
                                        values = policy_stats[col].values
                                        if len(values) > 0 and isinstance(values[0], (int, float)):
                                            logger.info(f"{col:25}: {values[0]:8.3f}")
                    else:
                        # DataFrame doesn't have expected structure, show what we have
                        logger.info(f"\n📈 AVAILABLE STATISTICS:")
                        logger.info(f"-" * 40)
                        logger.info(f"DataFrame columns: {list(statistics_df.columns)}")
                        logger.info(f"First few rows:")
                        logger.info(statistics_df.head())

                # Calculate basic episode statistics
                if debug:
                    logger.debug(
                        f"Calculating episode statistics for {len(policy_histories)} episodes"
                    )

                episode_returns = []
                episode_lengths = []

                for history in policy_histories:
                    # Filter out None rewards and handle edge cases
                    valid_rewards = [
                        step.reward for step in history.history if step.reward is not None
                    ]
                    if valid_rewards:
                        episode_returns.append(sum(valid_rewards))
                    else:
                        episode_returns.append(0.0)  # Default value if no valid rewards

                    episode_lengths.append(len(history.history))

                if debug:
                    logger.debug(
                        f"Episode returns: {len(episode_returns)} values, lengths: {len(episode_lengths)} values"
                    )

                logger.info(f"\n📋 EPISODE STATISTICS:")
                logger.info(f"-" * 40)
                logger.info(f"Average return: {sum(episode_returns)/len(episode_returns):8.3f}")
                logger.info(f"Best return: {max(episode_returns):8.3f}")
                logger.info(f"Worst return: {min(episode_returns):8.3f}")
                logger.info(f"Average length: {sum(episode_lengths)/len(episode_lengths):8.1f}")

                logger.info(f"\n💾 Results saved to: {eval_cache_dir}")
                logger.info(f"🔍 MLflow UI: cd {eval_cache_dir} && mlflow ui")

    if debug:
        logger.debug("Single evaluation completed, returning results and statistics")

    return results, statistics_df


def evaluate_multiple_optimized_planners(
    environment: Environment,
    optimized_policies: List[Policy],
    initial_belief: Belief,
    cache_dir: Path,
    experiment_name: str = "planner_evaluation",
    num_episodes: int = 10,
    num_steps: int = 8,
    n_jobs: int = 1,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Evaluate multiple optimized POMDP planners using comprehensive simulation.

    Args:
        environment: The POMDP environment to evaluate on
        optimized_policies: List of optimized policies to evaluate
        initial_belief: Initial belief state for evaluation
        cache_dir: Directory for storing evaluation results
        experiment_name: Name for the evaluation experiment
        num_episodes: Number of episodes for evaluation
        num_steps: Number of steps per episode
        n_jobs: Number of parallel jobs for evaluation
        confidence_interval_level: Confidence level for statistics
        alpha: Alpha value for risk metrics
        debug: Whether to enable debug logging
        verbose: Whether to print progress messages

    Returns:
        Tuple containing:
        - Raw episode results organized by environment and policy
        - DataFrame with comprehensive evaluation statistics
    """
    # Create evaluation cache directory
    eval_cache_dir = cache_dir
    eval_cache_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        logger.debug(f"Evaluation cache directory: {eval_cache_dir}")
        logger.debug(
            f"Evaluation parameters: {num_episodes} episodes, {num_steps} steps, {n_jobs} jobs"
        )

    if verbose:
        policy_names = [policy.name for policy in optimized_policies]
        logger.info(
            f"Evaluating {len(optimized_policies)} policies on {num_episodes} episodes each..."
        )
        logger.info(f"Policies: {policy_names}")

    if debug:
        logger.debug(f"Evaluation setup: {num_episodes} episodes, {num_steps} steps, {n_jobs} jobs")
        logger.debug(f"Environment: {environment.name}")
        for i, policy in enumerate(optimized_policies):
            logger.debug(f"Policy {i+1}: {policy.name} (type: {type(policy).__name__})")

    # Create environment run parameters for evaluation
    eval_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=optimized_policies,  # Pass all policies to be evaluated together
            num_episodes=num_episodes,
            num_steps=num_steps,
        )
    ]

    # Run evaluation using POMDPSimulator
    if debug:
        logger.debug(f"Creating POMDPSimulator with cache_dir: {eval_cache_dir}")
        logger.debug(f"Experiment name: {experiment_name}, Task manager: {TaskManagerType.JOBLIB}")

    with POMDPSimulator(
        cache_dir_path=eval_cache_dir,
        experiment_name=experiment_name,
        task_manager_type=TaskManagerType.JOBLIB,
        n_jobs=n_jobs,
        debug=debug,
        task_console_output=False,  # Reduce output noise
        enable_profiling=False,
    ) as simulator:
        if debug:
            logger.debug("POMDPSimulator created successfully, starting evaluation...")

        # Run the evaluation
        if debug:
            logger.debug("Running evaluation with POMDPSimulator...")

        results, statistics_df = simulator.compare_multiple_environments_policies(
            environment_run_params=eval_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )

        if debug:
            logger.debug(
                f"Evaluation completed, received results for {len(results) if results else 0} environments"
            )
            if statistics_df is not None:
                logger.debug(f"Statistics DataFrame shape: {statistics_df.shape}")

        # Display evaluation summary if verbose
        if verbose:
            env_name = environment.name

            logger.info(f"\n📊 EVALUATION SUMMARY")
            logger.info(f"-" * 40)
            logger.info(f"Environment: {env_name}")
            logger.info(f"Policies evaluated: {len(optimized_policies)}")
            logger.info(f"Episodes per policy: {num_episodes}")
            logger.info(f"Steps per episode: {num_steps}")

            # Display basic statistics for each policy
            for policy in optimized_policies:
                policy_name = policy.name

                if env_name in results and policy_name in results[env_name]:
                    policy_histories = results[env_name][policy_name]

                    # Calculate basic episode statistics
                    if debug:
                        logger.debug(
                            f"Calculating episode statistics for {policy_name}: {len(policy_histories)} episodes"
                        )

                    episode_returns = []
                    episode_lengths = []

                    for history in policy_histories:
                        # Filter out None rewards and handle edge cases
                        valid_rewards = [
                            step.reward for step in history.history if step.reward is not None
                        ]
                        if valid_rewards:
                            episode_returns.append(sum(valid_rewards))
                        else:
                            episode_returns.append(0.0)  # Default value if no valid rewards

                        episode_lengths.append(len(history.history))

                    if debug:
                        logger.debug(
                            f"Policy {policy_name}: {len(episode_returns)} returns, {len(episode_lengths)} lengths"
                        )

                    logger.info(f"\n📋 {policy_name} STATISTICS:")
                    logger.info(f"-" * 40)
                    logger.info(f"Episodes completed: {len(policy_histories)}")
                    logger.info(f"Average return: {sum(episode_returns)/len(episode_returns):8.3f}")
                    logger.info(f"Best return: {max(episode_returns):8.3f}")
                    logger.info(f"Worst return: {min(episode_returns):8.3f}")
                    logger.info(f"Average length: {sum(episode_lengths)/len(episode_lengths):8.1f}")

            logger.info(f"\n💾 Results saved to: {eval_cache_dir}")
            logger.info(f"🔍 MLflow UI: cd {eval_cache_dir} && mlflow ui")

    if debug:
        logger.debug("Multiple evaluation completed, returning results and statistics")

    return results, statistics_df


def create_numerical_hyperparameter_ranges(
    parameter_configs: Dict[str, Tuple[float, float]],
) -> List[NumericalHyperParameter]:
    """Helper function to create NumericalHyperParameter objects from a configuration dict.

    Args:
        parameter_configs: Dictionary mapping parameter names to (low, high) tuples

    Returns:
        List of NumericalHyperParameter objects

    Example:
        Create hyperparameters for POMCP::

            hyper_params = create_numerical_hyperparameter_ranges({
                "exploration_constant": (0.1, 5.0),
                "n_simulations": (50, 200),
                "depth": (3, 8)
            })
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Creating {len(parameter_configs)} numerical hyperparameters")
        for name, (low, high) in parameter_configs.items():
            logger.debug(f"  {name}: [{low}, {high}]")

    return [
        NumericalHyperParameter(low, high, name) for name, (low, high) in parameter_configs.items()
    ]


def create_categorical_hyperparameter_choices(
    parameter_configs: Dict[str, List[Any]],
) -> List[CategoricalHyperParameter]:
    """Helper function to create CategoricalHyperParameter objects from a configuration dict.

    Args:
        parameter_configs: Dictionary mapping parameter names to lists of choices

    Returns:
        List of CategoricalHyperParameter objects

    Example:
        Create hyperparameters::

            hyper_params = create_categorical_hyperparameter_choices({
                "algorithm": ["ucb", "thompson", "epsilon_greedy"],
                "heuristic": ["random", "informed"]
            })
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Creating {len(parameter_configs)} categorical hyperparameters")
        for name, choices in parameter_configs.items():
            logger.debug(f"  {name}: {choices}")

    return [CategoricalHyperParameter(name, choices) for name, choices in parameter_configs.items()]


def get_fast_optimization_defaults() -> Dict[str, Any]:
    """Get default parameters optimized for fast execution.

    Returns:
        Dictionary with recommended default parameters for fast optimization
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Getting fast optimization defaults")

    return {
        "optimization_episodes": 2,
        "optimization_steps": 1,
        "n_trials": 1,
        "evaluation_episodes": 1,
        "evaluation_steps": 1,
        "optimization_n_jobs": -1,
        "evaluation_n_jobs": 1,
        "confidence_interval_level": 0.95,
        "alpha": 0.05,
    }


def get_thorough_optimization_defaults() -> Dict[str, Any]:
    """Get default parameters for thorough but slower optimization.

    Returns:
        Dictionary with recommended default parameters for comprehensive optimization
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Getting thorough optimization defaults")

    return {
        "optimization_episodes": 10,
        "optimization_steps": 15,
        "n_trials": 10,
        "evaluation_episodes": 25,
        "evaluation_steps": 20,
        "optimization_n_jobs": -1,
        "evaluation_n_jobs": 4,
        "confidence_interval_level": 0.95,
        "alpha": 0.05,
    }


def get_benchmark_hyperparameter_planners(
    env: Environment, debug: bool = False, verbose: bool = False
) -> List[Type[Policy]]:
    """Get benchmark hyperparameter planners for the given environment.

    This function iterates over the POLICY_REGISTRY and returns a list of planner classes
    that are compatible with the given environment's space types.

    Compatibility rules:
    - A planner with MIXED or CONTINUOUS action space can solve DISCRETE action spaces
    - A planner with MIXED or CONTINUOUS observation space can solve DISCRETE observation spaces
    - A planner with DISCRETE action/observation space CANNOT solve CONTINUOUS action/observation spaces
    - MIXED planners can handle any space type

    Args:
        env: The environment to get benchmark planners for
        debug: Whether to enable debug logging
        verbose: Whether to enable verbose logging

    Returns:
        List of compatible policy classes from the POLICY_REGISTRY

    Example:
        Get compatible planners for Tiger POMDP (discrete actions, discrete observations)::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

            env = TigerPOMDP(discount_factor=0.95)
            compatible_planners = get_benchmark_hyperparameter_planners(env)
            print(f"Compatible planners: {[p.__name__ for p in compatible_planners]}")
    """
    from POMDPPlanners.planners import POLICY_REGISTRY

    if debug:
        logger.debug(f"Finding compatible planners for environment: {env.name}")
        logger.debug(
            f"Environment spaces - action: {env.space_info.action_space.value}, obs: {env.space_info.observation_space.value}"
        )

    compatible_planners = []

    # Get environment space types
    env_action_space = env.space_info.action_space
    env_obs_space = env.space_info.observation_space

    for planner_name, planner_class in POLICY_REGISTRY.items():
        try:
            # Skip abstract classes that don't have proper space info
            planner_space_info = planner_class.get_space_info()
            if planner_space_info is None:
                if debug:
                    logger.debug(
                        f"Skipping {planner_name}: no space info available (likely abstract class)"
                    )
                continue

            planner_action_space = planner_space_info.action_space
            planner_obs_space = planner_space_info.observation_space

            if debug:
                logger.debug(
                    f"Checking {planner_name}: action={planner_action_space.value}, obs={planner_obs_space.value}"
                )

            # Check action space compatibility
            action_compatible = _is_space_compatible(env_action_space, planner_action_space)

            # Check observation space compatibility
            obs_compatible = _is_space_compatible(env_obs_space, planner_obs_space)

            if action_compatible and obs_compatible:
                compatible_planners.append(planner_class)
                if debug:
                    logger.debug(f"✓ {planner_name} is compatible")
            else:
                if debug:
                    logger.debug(
                        f"✗ {planner_name} not compatible (action: {action_compatible}, obs: {obs_compatible})"
                    )

        except Exception as e:
            if debug:
                logger.debug(f"Skipping {planner_name} due to error: {e}")
            continue

    if verbose:
        compatible_names = [p.__name__ for p in compatible_planners]
        logger.info(
            f"Found {len(compatible_planners)} compatible planners for {env.name}: {compatible_names}"
        )

    return compatible_planners


def _is_space_compatible(env_space: SpaceType, planner_space: SpaceType) -> bool:
    """Check if a planner's space type is compatible with an environment's space type.

    Compatibility rules:
    - MIXED planners can handle any environment space type
    - CONTINUOUS planners can handle DISCRETE and CONTINUOUS environment spaces
    - DISCRETE planners can only handle DISCRETE environment spaces

    Args:
        env_space: The environment's space type
        planner_space: The planner's space type

    Returns:
        True if compatible, False otherwise
    """
    # MIXED planners can handle anything
    if planner_space == SpaceType.MIXED:
        return True

    # CONTINUOUS planners can handle DISCRETE and CONTINUOUS
    if planner_space == SpaceType.CONTINUOUS:
        return env_space in [SpaceType.DISCRETE, SpaceType.CONTINUOUS]

    # DISCRETE planners can only handle DISCRETE
    if planner_space == SpaceType.DISCRETE:
        return env_space == SpaceType.DISCRETE

    return False
