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
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import pandas as pd

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    OptimizedPolicyResult,
    HyperParamPlannerConfig,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    PBSConfig,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.planners import POLICY_REGISTRY

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


def _log_optimization_start(
    planner_configs: List[HyperParamPlannerConfig],
    environment: Environment,
    n_trials: int,
    optimization_episodes: int,
    optimization_steps: int,
    evaluation_episodes: int,
    evaluation_steps: int,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            "Starting hyperparameter optimization and evaluation for %d planners: %s",
            len(planner_configs),
            planner_names,
        )
        logger.info("Environment: %s", environment.name)
        logger.info(
            "Optimization: %d trials per planner, %d episodes, %d steps",
            n_trials,
            optimization_episodes,
            optimization_steps,
        )
        logger.info("Evaluation: %d episodes, %d steps", evaluation_episodes, evaluation_steps)

    if verbose:
        logger.info("\n%s", "=" * 60)
        logger.info("PHASE 1: HYPERPARAMETER OPTIMIZATION")
        logger.info("%s", "=" * 60)

    if debug:
        logger.debug("Starting optimization with %d planner configurations", len(planner_configs))
        for i, config in enumerate(planner_configs):
            logger.debug(
                "Planner %d: %s with %d hyperparameters",
                i + 1,
                config.policy_cls.__name__,
                len(config.hyper_parameters),
            )


def _log_optimization_completion(
    optimization_results: List[OptimizedPolicyResult],
    planner_configs: List[HyperParamPlannerConfig],
    verbose: bool,
) -> None:
    if verbose:
        logger.info("\n✓ Optimization completed for all %d planners!", len(planner_configs))
        for i, (result, config) in enumerate(zip(optimization_results, planner_configs)):
            logger.info("  %d. %s: %s", i + 1, config.policy_cls.__name__, result.policy.name)
            logger.info("     Best hyperparameters: %s", result.chosen_hyper_parameters)


def _log_evaluation_phase_start(
    optimized_policies: List[Policy], verbose: bool, debug: bool
) -> None:
    if verbose:
        logger.info("\n%s", "=" * 60)
        logger.info("PHASE 2: POLICY EVALUATION")
        logger.info("%s", "=" * 60)
        logger.info("Evaluating %d optimized policies together...", len(optimized_policies))

    if debug:
        logger.debug("Starting evaluation with %d optimized policies", len(optimized_policies))
        for i, policy in enumerate(optimized_policies):
            logger.debug("Policy %d: %s (type: %s)", i + 1, policy.name, type(policy).__name__)


def _log_evaluation_completion(
    evaluation_results: Dict,
    environment: Environment,
    optimized_policies: List[Policy],
    verbose: bool,
) -> None:
    if verbose:
        logger.info("\n✓ Evaluation completed successfully!")
        total_episodes = sum(
            len(evaluation_results[environment.name][policy.name]) for policy in optimized_policies
        )
        logger.info(
            "Evaluated %d total episodes across %d policies",
            total_episodes,
            len(optimized_policies),
        )


def _prepare_results_summary(
    optimization_results: List[OptimizedPolicyResult],
    planner_configs: List[HyperParamPlannerConfig],
    evaluation_results: Dict,
    evaluation_statistics: pd.DataFrame,
    cache_dir: Path,
    environment: Environment,
    n_trials: int,
    evaluation_episodes: int,
    debug: bool,
) -> Dict[str, Any]:
    cache_paths = {
        "optimization_cache": cache_dir,
        "evaluation_cache": cache_dir / "evaluation",
        "optimization_mlruns": cache_dir / "mlruns",
        "evaluation_mlruns": cache_dir / "evaluation" / "mlruns",
    }

    if debug:
        logger.debug("Preparing results summary")
        logger.debug("Cache paths: %s", cache_paths)

    return {
        "optimization_results": optimization_results,
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


def _log_final_summary(
    environment: Environment,
    optimization_results: List[OptimizedPolicyResult],
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    cache_paths: Dict[str, Path],
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        logger.info("\n%s", "=" * 60)
        logger.info("OPTIMIZATION AND EVALUATION COMPLETE")
        logger.info("%s", "=" * 60)
        logger.info("Environment: %s", environment.name)
        logger.info("Optimized planners: %d", len(optimization_results))
        for i, result in enumerate(optimization_results):
            planner_type = planner_configs[i].policy_cls.__name__
            logger.info("  %d. %s (%s)", i + 1, result.policy.name, planner_type)
        logger.info("Results saved to: %s", cache_dir)
        logger.info("MLflow tracking: %s (optimization)", cache_paths["optimization_mlruns"])
        logger.info("MLflow tracking: %s (evaluation)", cache_paths["evaluation_mlruns"])

    if debug:
        logger.debug("Final results summary prepared and returning")


def optimize_and_evaluate_planners(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    parameters_to_optimize: Optional[List[Tuple[str, HyperParameterOptimizationDirection]]] = None,
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
        parameters_to_optimize: List of tuples (metric_name, direction) for multi-objective optimization.
                               If None, defaults to [("average_return", MAXIMIZE)]
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

            >>> from pathlib import Path
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
            >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>>
            >>> # Create environment and belief
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> initial_belief = get_initial_belief(env, n_particles=100)
            >>>
            >>> # Configure planners (simplified for doctest)
            >>> planner_configs = [
            ...     HyperParamPlannerConfig(
            ...         policy_cls=POMCP,
            ...         hyper_parameters=[
            ...             NumericalHyperParameter(0.1, 5.0, "exploration_constant")
            ...         ],
            ...         constant_parameters={"discount_factor": 0.95, "name": "POMCP"}
            ...     )
            ... ]
            >>>
            >>> # This would run optimization (commented out for doctest)
            >>> # results = optimize_and_evaluate_planners(
            >>> #     environment=env,
            >>> #     initial_belief=initial_belief,
            >>> #     planner_configs=planner_configs,
            >>> #     cache_dir=Path("./results")
            >>> # )
            >>> len(planner_configs) == 1  # Verify example setup
            True
    """
    if parameters_to_optimize is None:
        parameters_to_optimize = [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]

    configure_logging(debug=debug, verbose=verbose)

    _log_optimization_start(
        planner_configs,
        environment,
        n_trials,
        optimization_episodes,
        optimization_steps,
        evaluation_episodes,
        evaluation_steps,
        verbose,
        debug,
    )

    optimization_results = optimize_planner_hyperparameters(
        environment=environment,
        initial_belief=initial_belief,
        planner_configs=planner_configs,
        cache_dir=cache_dir,
        parameters_to_optimize=parameters_to_optimize,
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

    optimized_policies = [result.policy for result in optimization_results]

    _log_optimization_completion(optimization_results, planner_configs, verbose)
    _log_evaluation_phase_start(optimized_policies, verbose, debug)

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

    _log_evaluation_completion(evaluation_results, environment, optimized_policies, verbose)

    results = _prepare_results_summary(
        optimization_results,
        planner_configs,
        evaluation_results,
        evaluation_statistics,
        cache_dir,
        environment,
        n_trials,
        evaluation_episodes,
        debug,
    )

    _log_final_summary(
        environment,
        optimization_results,
        planner_configs,
        cache_dir,
        results["cache_paths"],
        verbose,
        debug,
    )

    return results


def _log_pbs_optimization_start(
    planner_configs: List[HyperParamPlannerConfig],
    environment: Environment,
    queue: str,
    n_workers: int,
    cores: int,
    memory: str,
    n_trials: int,
    optimization_episodes: int,
    optimization_steps: int,
    evaluation_episodes: int,
    evaluation_steps: int,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            "Starting PBS hyperparameter optimization and evaluation for %d planners: %s",
            len(planner_configs),
            planner_names,
        )
        logger.info("Environment: %s", environment.name)
        logger.info(
            "PBS Configuration: queue=%s, workers=%d, cores=%d, memory=%s",
            queue,
            n_workers,
            cores,
            memory,
        )
        logger.info(
            "Optimization: %d trials per planner, %d episodes, %d steps",
            n_trials,
            optimization_episodes,
            optimization_steps,
        )
        logger.info("Evaluation: %d episodes, %d steps", evaluation_episodes, evaluation_steps)

    if verbose:
        logger.info("\n%s", "=" * 60)
        logger.info("PHASE 1: PBS HYPERPARAMETER OPTIMIZATION")
        logger.info("%s", "=" * 60)

    if debug:
        logger.debug(
            "Starting PBS optimization with %d planner configurations", len(planner_configs)
        )
        for i, config in enumerate(planner_configs):
            logger.debug(
                "Planner %d: %s with %d hyperparameters",
                i + 1,
                config.policy_cls.__name__,
                len(config.hyper_parameters),
            )


def _prepare_pbs_results_summary(
    optimization_results: List[OptimizedPolicyResult],
    planner_configs: List[HyperParamPlannerConfig],
    evaluation_results: Dict,
    evaluation_statistics: pd.DataFrame,
    cache_dir: Path,
    environment: Environment,
    n_trials: int,
    evaluation_episodes: int,
    queue: str,
    n_workers: int,
    cores: int,
    memory: str,
    walltime: str,
    debug: bool,
) -> Dict[str, Any]:
    cache_paths = {
        "optimization_cache": cache_dir,
        "evaluation_cache": cache_dir / "evaluation",
        "optimization_mlruns": cache_dir / "mlruns",
        "evaluation_mlruns": cache_dir / "evaluation" / "mlruns",
    }

    if debug:
        logger.debug("Preparing PBS results summary")
        logger.debug("Cache paths: %s", cache_paths)

    return {
        "optimization_results": optimization_results,
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
            "pbs_configuration": {
                "queue": queue,
                "n_workers": n_workers,
                "cores": cores,
                "memory": memory,
                "walltime": walltime,
            },
        },
    }


def _log_pbs_final_summary(
    environment: Environment,
    optimization_results: List[OptimizedPolicyResult],
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    cache_paths: Dict[str, Path],
    queue: str,
    n_workers: int,
    cores: int,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        logger.info("\n%s", "=" * 60)
        logger.info("PBS OPTIMIZATION AND EVALUATION COMPLETE")
        logger.info("%s", "=" * 60)
        logger.info("Environment: %s", environment.name)
        logger.info("Optimized planners: %d", len(optimization_results))
        for i, result in enumerate(optimization_results):
            planner_type = planner_configs[i].policy_cls.__name__
            logger.info("  %d. %s (%s)", i + 1, result.policy.name, planner_type)
        logger.info("PBS Queue: %s (%d workers, %d cores each)", queue, n_workers, cores)
        logger.info("Results saved to: %s", cache_dir)
        logger.info("MLflow tracking: %s (optimization)", cache_paths["optimization_mlruns"])
        logger.info("MLflow tracking: %s (evaluation)", cache_paths["evaluation_mlruns"])

    if debug:
        logger.debug("Final PBS results summary prepared and returning")


def optimize_and_evaluate_planners_pbs(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    queue: str,
    parameters_to_optimize: Optional[List[Tuple[str, HyperParameterOptimizationDirection]]] = None,
    experiment_name: str = "planner_optimization_pbs",
    # Optimization parameters
    optimization_episodes: int = 3,
    optimization_steps: int = 6,
    n_trials: int = 3,
    n_workers: int = 4,
    cores: int = 1,
    memory: str = "4GB",
    processes: int = 1,
    walltime: str = "01:00:00",
    job_extra: Optional[List[str]] = None,
    optimization_n_jobs: int = -1,
    enable_dashboard: bool = True,
    dashboard_address: str = "0.0.0.0",
    dashboard_port: int = 8787,
    dashboard_prefix: Optional[str] = None,
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
    """Perform hyperparameter optimization followed by comprehensive evaluation using PBS cluster computing.

    This function provides a complete workflow for optimizing multiple POMDP planners'
    hyperparameters using PBS cluster computing and then evaluating all optimized policies'
    performance together. It's similar to optimize_and_evaluate_planners but uses PBS
    for distributed hyperparameter optimization.

    Args:
        environment: The POMDP environment to optimize and evaluate on
        initial_belief: Initial belief state for the environment
        planner_configs: List of HyperParamPlannerConfig objects, each containing
                        a policy class, hyperparameters, and constant parameters
        cache_dir: Directory for storing optimization and evaluation results
        queue: PBS queue name to submit jobs to
        parameters_to_optimize: List of tuples (metric_name, direction) for multi-objective optimization.
                               If None, defaults to [("average_return", MAXIMIZE)]
        experiment_name: Name for the experiment (used in MLflow tracking)
        optimization_episodes: Number of episodes for optimization trials
        optimization_steps: Number of steps per optimization episode
        n_trials: Number of optimization trials to run per planner
        n_workers: Number of PBS worker nodes to request
        cores: Number of CPU cores per worker
        memory: Memory per worker (e.g., "4GB", "8GB")
        processes: Number of processes per worker
        walltime: Maximum job runtime in HH:MM:SS format
        job_extra: Additional PBS job directives
        optimization_n_jobs: Number of parallel jobs for optimization (-1 for all cores)
        enable_dashboard: Whether to enable Dask dashboard
        dashboard_address: Dashboard bind address
        dashboard_port: Dashboard port
        dashboard_prefix: Dashboard URL prefix
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
        Optimize multiple planners on Tiger POMDP using PBS cluster::

            >>> from pathlib import Path
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>>
            >>> # Create environment and belief
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> initial_belief = get_initial_belief(env, n_particles=100)
            >>>
            >>> # Configure planners
            >>> planner_configs = [
            ...     HyperParamPlannerConfig(
            ...         policy_cls=POMCP,
            ...         hyper_parameters=[
            ...             NumericalHyperParameter(0.1, 5.0, "exploration_constant")
            ...         ],
            ...         constant_parameters={"discount_factor": 0.95, "name": "POMCP"}
            ...     )
            ... ]
            >>>
            >>> # This would run PBS optimization (commented out for doctest)
            >>> # results = optimize_and_evaluate_planners_pbs(
            >>> #     environment=env,
            >>> #     initial_belief=initial_belief,
            >>> #     planner_configs=planner_configs,
            >>> #     cache_dir=Path("./results"),
            >>> #     queue="short",
            >>> #     n_workers=8,
            >>> #     cores=2,
            >>> #     memory="8GB",
            >>> #     walltime="02:00:00"
            >>> # )
            >>> len(planner_configs) == 1  # Verify example setup
            True
    """
    if parameters_to_optimize is None:
        parameters_to_optimize = [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]

    configure_logging(debug=debug, verbose=verbose)

    _log_pbs_optimization_start(
        planner_configs,
        environment,
        queue,
        n_workers,
        cores,
        memory,
        n_trials,
        optimization_episodes,
        optimization_steps,
        evaluation_episodes,
        evaluation_steps,
        verbose,
        debug,
    )

    optimization_results = optimize_planner_hyperparameters_pbs(
        environment=environment,
        initial_belief=initial_belief,
        planner_configs=planner_configs,
        cache_dir=cache_dir,
        queue=queue,
        parameters_to_optimize=parameters_to_optimize,
        experiment_name=experiment_name,
        num_episodes=optimization_episodes,
        num_steps=optimization_steps,
        n_trials=n_trials,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
        n_jobs=optimization_n_jobs,
        enable_dashboard=enable_dashboard,
        dashboard_address=dashboard_address,
        dashboard_port=dashboard_port,
        dashboard_prefix=dashboard_prefix,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
    )

    if not optimization_results or len(optimization_results) != len(planner_configs):
        error_msg = f"PBS hyperparameter optimization failed - expected {len(planner_configs)} results but got {len(optimization_results) if optimization_results else 0}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    optimized_policies = [result.policy for result in optimization_results]

    _log_optimization_completion(optimization_results, planner_configs, verbose)
    _log_evaluation_phase_start(optimized_policies, verbose, debug)

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

    _log_evaluation_completion(evaluation_results, environment, optimized_policies, verbose)

    results = _prepare_pbs_results_summary(
        optimization_results,
        planner_configs,
        evaluation_results,
        evaluation_statistics,
        cache_dir,
        environment,
        n_trials,
        evaluation_episodes,
        queue,
        n_workers,
        cores,
        memory,
        walltime,
        debug,
    )

    _log_pbs_final_summary(
        environment,
        optimization_results,
        planner_configs,
        cache_dir,
        results["cache_paths"],
        queue,
        n_workers,
        cores,
        verbose,
        debug,
    )

    return results


def optimize_and_evaluate_multiple_environments_and_policies(
    configs: List[HyperParameterRunParams],
    num_episodes_evaluation: int,
    num_steps_evaluation: int,
    cache_dir: Path,
    experiment_name: str,
    task_manager_config: JoblibConfig,
    n_jobs: int,
    confidence_interval_level: float,
    alpha: float,
    debug: bool = False,
    verbose: bool = True,
    cache_visualizations: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Optimize and evaluate multiple POMDP planners on multiple environments and policies."""
    if verbose:
        logger.info("Optimizing %d environments and policies", len(configs))

    if debug:
        logger.debug("Optimizing %d environments and policies", len(configs))

    if cache_dir is None:
        cache_dir = Path(f"./{experiment_name.lower().replace(' ', '_')}_results")

    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        task_manager_config=task_manager_config,
        n_jobs=n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
    )

    optimization_results: List[OptimizedPolicyResult] = optimizer.optimize(configs)
    optimization_results_organized: List[Tuple[Environment, Belief, List[Policy]]] = [
        (result.environment, configs[i].belief, [result.policy])
        for i, result in enumerate(optimization_results)
    ]

    chosen_planners_eval_configs = [
        EnvironmentRunParams(
            environment=env,
            belief=belief,
            policies=policies,
            num_episodes=num_episodes_evaluation,
            num_steps=num_steps_evaluation,
        )
        for env, belief, policies in optimization_results_organized
    ]

    with POMDPSimulator(
        task_manager_config=JoblibConfig(n_jobs=n_jobs),
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        debug=debug,
    ) as simulator:
        results = simulator.compare_multiple_environments_policies(
            environment_run_params=chosen_planners_eval_configs,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=cache_visualizations,
        )

    return results


def optimize_and_evaluate_multiple_environments_and_policies_pbs(
    configs: List[HyperParameterRunParams],
    num_episodes_evaluation: int,
    num_steps_evaluation: int,
    cache_dir: Path,
    experiment_name: str,
    queue: str,
    n_workers: int,
    cores: int,
    memory: str,
    processes: int,
    walltime: str,
    job_extra: Optional[List[str]],
    n_jobs: int,
    confidence_interval_level: float,
    alpha: float,
    debug: bool = False,
    verbose: bool = True,
    cache_visualizations: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Optimize and evaluate multiple POMDP planners on multiple environments and policies."""
    if verbose:
        logger.info("Optimizing %d environments and policies", len(configs))

    if debug:
        logger.debug("Optimizing %d environments and policies", len(configs))

    if cache_dir is None:
        cache_dir = Path(f"./{experiment_name.lower().replace(' ', '_')}_results")

    n_workers = min(n_workers, len(configs))

    task_manager_config = PBSConfig(
        queue=queue,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
    )

    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        task_manager_config=task_manager_config,
        n_jobs=n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
    )

    optimization_results: List[OptimizedPolicyResult] = optimizer.optimize(configs)
    optimization_results_organized: List[Tuple[Environment, Belief, List[Policy]]] = [
        (result.environment, configs[i].belief, [result.policy])
        for i, result in enumerate(optimization_results)
    ]

    chosen_planners_eval_configs = [
        EnvironmentRunParams(
            environment=env,
            belief=belief,
            policies=policies,
            num_episodes=num_episodes_evaluation,
            num_steps=num_steps_evaluation,
        )
        for env, belief, policies in optimization_results_organized
    ]

    with POMDPSimulator(
        task_manager_config=JoblibConfig(n_jobs=-1),
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        debug=debug,
    ) as simulator:
        results = simulator.compare_multiple_environments_policies(
            environment_run_params=chosen_planners_eval_configs,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=cache_visualizations,
        )

    return results


def optimize_planner_hyperparameters(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    parameters_to_optimize: Optional[List[Tuple[str, HyperParameterOptimizationDirection]]] = None,
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
        parameters_to_optimize: List of tuples (metric_name, direction) for multi-objective optimization.
                               If None, defaults to [("average_return", MAXIMIZE)]
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
    # Set default parameters if not provided
    if parameters_to_optimize is None:
        parameters_to_optimize = [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]

    if verbose:
        total_hyperparams = sum(len(config.hyper_parameters) for config in planner_configs)
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            "Optimizing %d planners (%s) with %d total hyperparameters using %d trials each...",
            len(planner_configs),
            planner_names,
            total_hyperparams,
            n_trials,
        )

    if debug:
        logger.debug("Creating HyperParameterOptimizer with %d jobs", n_jobs)
        logger.debug("Confidence interval level: %s, Alpha: %s", confidence_interval_level, alpha)

    task_manager_config = JoblibConfig(n_jobs=1)

    # Create optimizer
    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        task_manager_config=task_manager_config,
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
            logger.debug(
                "Creating config for planner %d: %s", i + 1, planner_config.policy_cls.__name__
            )

        config = HyperParameterRunParams(
            environment=environment,
            belief=initial_belief,
            hyper_param_planner_config=planner_config,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            parameters_to_optimize=parameters_to_optimize,
        )
        optimization_configs.append(config)

    if debug:
        logger.debug("Created %d optimization configurations", len(optimization_configs))

    # Run optimization for all planners in a single call
    if verbose:
        logger.info("Running optimization with:")
        for i, planner_config in enumerate(planner_configs):
            logger.info("  - Planner %d: %s", i + 1, planner_config.policy_cls.__name__)
            logger.info(
                "    Hyperparameters: %s", [hp.name for hp in planner_config.hyper_parameters]
            )
        logger.info("  - Environment: %s", environment.name)
        logger.info("  - Trials per planner: %d", n_trials)
        logger.info("  - Episodes per trial: %d", num_episodes)
        logger.info("  - Steps per episode: %d", num_steps)

    if debug:
        logger.debug("Starting optimization with HyperParameterOptimizer")

    optimization_results = optimizer.optimize(optimization_configs)

    if debug:
        logger.debug(
            "Optimization completed, received %d results",
            len(optimization_results) if optimization_results else 0,
        )

    # Return all results
    if optimization_results:
        if verbose:
            logger.info("Optimization completed for %d planners", len(optimization_results))
        if debug:
            logger.debug("Returning %d optimization results", len(optimization_results))
        return optimization_results
    else:
        if verbose:
            logger.warning("Warning: No optimization results returned")
        if debug:
            logger.debug("No optimization results to return")
        return []


def _log_pbs_hyperparameter_optimization_setup(
    planner_configs: List[HyperParamPlannerConfig],
    queue: str,
    n_workers: int,
    cores: int,
    memory: str,
    walltime: str,
    processes: int,
    n_trials: int,
    confidence_interval_level: float,
    alpha: float,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        total_hyperparams = sum(len(config.hyper_parameters) for config in planner_configs)
        planner_names = [config.policy_cls.__name__ for config in planner_configs]
        logger.info(
            "Optimizing %d planners (%s) with %d total hyperparameters using PBS cluster...",
            len(planner_configs),
            planner_names,
            total_hyperparams,
        )
        logger.info(
            "PBS Configuration: queue=%s, workers=%d, cores=%d, memory=%s",
            queue,
            n_workers,
            cores,
            memory,
        )
        logger.info("Walltime: %s, trials per planner: %d", walltime, n_trials)

    if debug:
        logger.debug("Creating HyperParameterOptimizer with PBS task manager")
        logger.debug("PBS queue: %s, workers: %d, cores: %d", queue, n_workers, cores)
        logger.debug("Memory: %s, processes: %d, walltime: %s", memory, processes, walltime)
        logger.debug("Confidence interval level: %s, Alpha: %s", confidence_interval_level, alpha)


def _create_optimization_configs(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    num_episodes: int,
    num_steps: int,
    n_trials: int,
    parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]],
    debug: bool,
) -> List[HyperParameterRunParams]:
    if debug:
        logger.debug("Creating optimization configurations for all planners")

    optimization_configs = []
    for i, planner_config in enumerate(planner_configs):
        if debug:
            logger.debug(
                "Creating config for planner %d: %s", i + 1, planner_config.policy_cls.__name__
            )

        config = HyperParameterRunParams(
            environment=environment,
            belief=initial_belief,
            hyper_param_planner_config=planner_config,
            num_episodes=num_episodes,
            num_steps=num_steps,
            n_trials=n_trials,
            parameters_to_optimize=parameters_to_optimize,
        )
        optimization_configs.append(config)

    if debug:
        logger.debug("Created %d optimization configurations", len(optimization_configs))

    return optimization_configs


def _log_pbs_optimization_details(
    planner_configs: List[HyperParamPlannerConfig],
    environment: Environment,
    n_trials: int,
    num_episodes: int,
    num_steps: int,
    n_workers: int,
    enable_dashboard: bool,
    dashboard_address: str,
    dashboard_port: int,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        logger.info("Running PBS cluster optimization with:")
        for i, planner_config in enumerate(planner_configs):
            logger.info("  - Planner %d: %s", i + 1, planner_config.policy_cls.__name__)
            logger.info(
                "    Hyperparameters: %s", [hp.name for hp in planner_config.hyper_parameters]
            )
        logger.info("  - Environment: %s", environment.name)
        logger.info("  - Trials per planner: %d", n_trials)
        logger.info("  - Episodes per trial: %d", num_episodes)
        logger.info("  - Steps per episode: %d", num_steps)
        logger.info("  - PBS workers: %d", n_workers)
        logger.info("  - Dashboard enabled: %s", enable_dashboard)
        if enable_dashboard:
            logger.info("  - Dashboard URL: http://%s:%d", dashboard_address, dashboard_port)

    if debug:
        logger.debug("Starting PBS optimization with HyperParameterOptimizer")


def _process_pbs_optimization_results(
    optimization_results: List[OptimizedPolicyResult],
    verbose: bool,
    debug: bool,
) -> List[OptimizedPolicyResult]:
    if debug:
        logger.debug(
            "PBS optimization completed, received %d results",
            len(optimization_results) if optimization_results else 0,
        )

    if optimization_results:
        if verbose:
            logger.info("PBS optimization completed for %d planners", len(optimization_results))
        if debug:
            logger.debug("Returning %d optimization results", len(optimization_results))
        return optimization_results
    else:
        if verbose:
            logger.warning("Warning: No optimization results returned from PBS cluster")
        if debug:
            logger.debug("No optimization results to return")
        return []


def optimize_planner_hyperparameters_pbs(
    environment: Environment,
    initial_belief: Belief,
    planner_configs: List[HyperParamPlannerConfig],
    cache_dir: Path,
    queue: str,
    parameters_to_optimize: Optional[List[Tuple[str, HyperParameterOptimizationDirection]]] = None,
    experiment_name: str = "planner_optimization_pbs",
    num_episodes: int = 3,
    num_steps: int = 6,
    n_trials: int = 3,
    n_workers: int = 4,
    cores: int = 1,
    memory: str = "4GB",
    processes: int = 1,
    walltime: str = "01:00:00",
    job_extra: Optional[List[str]] = None,
    n_jobs: int = -1,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.1,
    enable_dashboard: bool = True,
    dashboard_address: str = "0.0.0.0",
    dashboard_port: int = 8787,
    dashboard_prefix: Optional[str] = None,
    debug: bool = False,
    verbose: bool = True,
) -> List[OptimizedPolicyResult]:
    """Optimize hyperparameters for multiple POMDP planners using Optuna with PBS task manager.

    This function is similar to optimize_planner_hyperparameters but uses PBS cluster
    computing for distributed hyperparameter optimization, allowing for scalable
    computation across multiple cluster nodes.

    Args:
        environment: The POMDP environment to optimize on
        initial_belief: Initial belief state for the environment
        planner_configs: List of HyperParamPlannerConfig objects containing planner configurations
        cache_dir: Directory for storing optimization results
        queue: PBS queue name to submit jobs to
        parameters_to_optimize: List of tuples (metric_name, direction) for multi-objective optimization.
                               If None, defaults to [("average_return", MAXIMIZE)]
        experiment_name: Name for the optimization experiment
        num_episodes: Number of episodes per optimization trial
        num_steps: Number of steps per episode
        n_trials: Number of optimization trials per planner
        n_workers: Number of PBS worker nodes to request
        cores: Number of CPU cores per worker
        memory: Memory per worker (e.g., "4GB", "8GB")
        processes: Number of processes per worker
        walltime: Maximum job runtime in HH:MM:SS format
        job_extra: Additional PBS job directives
        n_jobs: Number of parallel jobs for optimization (-1 for all cores)
        confidence_interval_level: Confidence level for statistics
        alpha: Alpha value for statistics computation
        enable_dashboard: Whether to enable Dask dashboard
        dashboard_address: Dashboard bind address
        dashboard_port: Dashboard port
        dashboard_prefix: Dashboard URL prefix
        debug: Whether to enable debug logging
        verbose: Whether to print progress messages

    Returns:
        List of OptimizedPolicyResult objects, one for each planner configuration

    Example:
        Optimize planners using PBS cluster::

            >>> from pathlib import Path
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
            >>> from POMDPPlanners.core.belief import get_initial_belief
            >>>
            >>> # Create environment and belief
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> initial_belief = get_initial_belief(env, n_particles=100)
            >>>
            >>> # Configure planners
            >>> planner_configs = [
            ...     HyperParamPlannerConfig(
            ...         policy_cls=POMCP,
            ...         hyper_parameters=[
            ...             NumericalHyperParameter(0.1, 5.0, "exploration_constant")
            ...         ],
            ...         constant_parameters={"discount_factor": 0.95, "name": "POMCP"}
            ...     )
            ... ]
            >>>
            >>> # Run PBS optimization (commented out for doctest)
            >>> # results = optimize_planner_hyperparameters_pbs(
            >>> #     environment=env,
            >>> #     initial_belief=initial_belief,
            >>> #     planner_configs=planner_configs,
            >>> #     cache_dir=Path("./results"),
            >>> #     queue="short",
            >>> #     n_workers=8,
            >>> #     cores=2,
            >>> #     memory="8GB",
            >>> #     walltime="02:00:00"
            >>> # )
            >>> len(planner_configs) == 1  # Verify example setup
            True
    """
    if parameters_to_optimize is None:
        parameters_to_optimize = [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]

    _log_pbs_hyperparameter_optimization_setup(
        planner_configs,
        queue,
        n_workers,
        cores,
        memory,
        walltime,
        processes,
        n_trials,
        confidence_interval_level,
        alpha,
        verbose,
        debug,
    )

    task_manager_config = PBSConfig(
        queue=queue,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
        enable_dashboard=enable_dashboard,
        dashboard_address=dashboard_address,
        dashboard_port=dashboard_port,
        dashboard_prefix=dashboard_prefix,
    )

    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        task_manager_config=task_manager_config,
        n_jobs=n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
    )

    optimization_configs = _create_optimization_configs(
        environment,
        initial_belief,
        planner_configs,
        num_episodes,
        num_steps,
        n_trials,
        parameters_to_optimize,
        debug,
    )

    _log_pbs_optimization_details(
        planner_configs,
        environment,
        n_trials,
        num_episodes,
        num_steps,
        n_workers,
        enable_dashboard,
        dashboard_address,
        dashboard_port,
        verbose,
        debug,
    )

    optimization_results = optimizer.optimize(optimization_configs)

    return _process_pbs_optimization_results(optimization_results, verbose, debug)


def _log_evaluation_start(
    eval_cache_dir: Path,
    policy_name: str,
    num_episodes: int,
    num_steps: int,
    n_jobs: int,
    debug: bool,
    verbose: bool,
) -> None:
    """Log evaluation start information."""
    if debug:
        logger.debug("Evaluation cache directory: %s", eval_cache_dir)
        logger.debug(
            "Evaluation parameters: %d episodes, %d steps, %d jobs", num_episodes, num_steps, n_jobs
        )
    if verbose:
        logger.info("Evaluating policy '%s' on %d episodes...", policy_name, num_episodes)


def _log_simulator_creation(eval_cache_dir: Path, experiment_name: str, debug: bool) -> None:
    """Log simulator creation information."""
    if debug:
        logger.debug("Creating POMDPSimulator with cache_dir: %s", eval_cache_dir)
        logger.debug("Experiment name: %s, Task manager: JoblibConfig", experiment_name)


def _log_evaluation_results(results: Dict, statistics_df: pd.DataFrame, debug: bool) -> None:
    """Log evaluation results summary."""
    if debug:
        logger.debug(
            "Evaluation completed, received results for %d environments",
            len(results) if results else 0,
        )
        if statistics_df is not None:
            logger.debug("Statistics DataFrame shape: %s", statistics_df.shape)


def _display_key_metrics(policy_stats: pd.DataFrame, verbose: bool, debug: bool) -> None:
    """Display key metrics from policy statistics."""
    if "metric" in policy_stats.columns:
        key_metrics = ["average_return", "return_cvar", "average_actual_num_steps"]
        logger.info("\n📈 KEY METRICS:")
        logger.info("-" * 40)

        for _, row in policy_stats.iterrows():
            metric_name = row["metric"]
            if metric_name in key_metrics:
                logger.info(
                    "%s: %8.3f [%6.3f, %6.3f]",
                    metric_name,
                    row["value"],
                    row["lower_confidence_bound"],
                    row["upper_confidence_bound"],
                )
    else:
        logger.info("Available statistics for policy:")
        for col in policy_stats.columns:
            if col != "policy":
                values = policy_stats[col][0] if len(policy_stats) > 0 else None
                if values is not None and isinstance(values, (int, float)):
                    logger.info("%s: %8.3f", col, values)


def _display_statistics_summary(
    env_name: str,
    policy_name: str,
    policy_histories: List,
    statistics_df: pd.DataFrame,
    num_steps: int,
    eval_cache_dir: Path,
    verbose: bool,
    debug: bool,
) -> None:
    """Display comprehensive evaluation summary."""
    logger.info("\n📊 EVALUATION SUMMARY")
    logger.info("-" * 40)
    logger.info("Environment: %s", env_name)
    logger.info("Policy: %s", policy_name)
    logger.info("Episodes completed: %d", len(policy_histories))
    logger.info("Steps per episode: %d", num_steps)

    # Display key statistics
    if not statistics_df.empty:
        if verbose:
            logger.info("\n📊 Statistics DataFrame columns: %s", list(statistics_df.columns))
            logger.info("📊 Statistics DataFrame shape: %s", statistics_df.shape)

        if debug:
            logger.debug("Processing statistics DataFrame with %d rows", len(statistics_df))
            logger.debug("DataFrame columns: %s", list(statistics_df.columns))

        if "policy" in statistics_df.columns:
            policy_stats = statistics_df[statistics_df["policy"] == policy_name]
            if not policy_stats.empty:
                # Type assertion: boolean indexing always returns DataFrame
                _display_key_metrics(cast(pd.DataFrame, policy_stats), verbose, debug)
        else:
            logger.info("\n📈 AVAILABLE STATISTICS:")
            logger.info("-" * 40)
            logger.info("DataFrame columns: %s", list(statistics_df.columns))
            logger.info("First few rows:")
            logger.info(statistics_df.head())

    # Calculate basic episode statistics
    if debug:
        logger.debug("Calculating episode statistics for %d episodes", len(policy_histories))

    episode_returns = []
    episode_lengths = []

    for history in policy_histories:
        valid_rewards = [step.reward for step in history.history if step.reward is not None]
        episode_returns.append(sum(valid_rewards) if valid_rewards else 0.0)
        episode_lengths.append(len(history.history))

    if debug:
        logger.debug(
            "Episode returns: %d values, lengths: %d values",
            len(episode_returns),
            len(episode_lengths),
        )

    logger.info("\n📋 EPISODE STATISTICS:")
    logger.info("-" * 40)
    logger.info("Average return: %8.3f", sum(episode_returns) / len(episode_returns))
    logger.info("Best return: %8.3f", max(episode_returns))
    logger.info("Worst return: %8.3f", min(episode_returns))
    logger.info("Average length: %8.1f", sum(episode_lengths) / len(episode_lengths))

    logger.info("\n💾 Results saved to: %s", eval_cache_dir)
    logger.info("🔍 MLflow UI: cd %s && mlflow ui", eval_cache_dir)


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

    # Log evaluation start
    _log_evaluation_start(
        eval_cache_dir, optimized_policy.name, num_episodes, num_steps, n_jobs, debug, verbose
    )

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

    # Log simulator creation
    _log_simulator_creation(eval_cache_dir, experiment_name, debug)

    # Create task manager config for joblib
    task_manager_config = JoblibConfig(n_jobs=n_jobs)

    with POMDPSimulator(
        task_manager_config=task_manager_config,
        cache_dir_path=eval_cache_dir,
        experiment_name=experiment_name,
        debug=debug,
        task_console_output=False,  # Reduce output noise
        enable_profiling=False,
    ) as simulator:
        if debug:
            logger.debug("POMDPSimulator created successfully, starting evaluation...")
            logger.debug("Running evaluation with POMDPSimulator...")

        results, statistics_df = simulator.compare_multiple_environments_policies(
            environment_run_params=eval_params,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            n_jobs=n_jobs,
            cache_visualizations=True,
        )

        _log_evaluation_results(results, statistics_df, debug)

        # Display evaluation summary if verbose
        if verbose:
            env_name = environment.name
            policy_name = optimized_policy.name

            if env_name in results and policy_name in results[env_name]:
                policy_histories = results[env_name][policy_name]
                _display_statistics_summary(
                    env_name,
                    policy_name,
                    policy_histories,
                    statistics_df,
                    num_steps,
                    eval_cache_dir,
                    verbose,
                    debug,
                )

    if debug:
        logger.debug("Single evaluation completed, returning results and statistics")

    return results, statistics_df


def _log_multiple_evaluation_setup(
    optimized_policies: List[Policy],
    environment: Environment,
    num_episodes: int,
    num_steps: int,
    n_jobs: int,
    eval_cache_dir: Path,
    verbose: bool,
    debug: bool,
) -> None:
    if debug:
        logger.debug("Evaluation cache directory: %s", eval_cache_dir)
        logger.debug(
            "Evaluation parameters: %d episodes, %d steps, %d jobs", num_episodes, num_steps, n_jobs
        )

    if verbose:
        policy_names = [policy.name for policy in optimized_policies]
        logger.info(
            "Evaluating %d policies on %d episodes each...", len(optimized_policies), num_episodes
        )
        logger.info("Policies: %s", policy_names)

    if debug:
        logger.debug(
            "Evaluation setup: %d episodes, %d steps, %d jobs", num_episodes, num_steps, n_jobs
        )
        logger.debug("Environment: %s", environment.name)
        for i, policy in enumerate(optimized_policies):
            logger.debug("Policy %d: %s (type: %s)", i + 1, policy.name, type(policy).__name__)


def _calculate_policy_episode_statistics(
    policy_histories: List, policy_name: str, debug: bool
) -> Tuple[List[float], List[int]]:
    if debug:
        logger.debug(
            "Calculating episode statistics for %s: %d episodes",
            policy_name,
            len(policy_histories),
        )

    episode_returns = []
    episode_lengths = []

    for history in policy_histories:
        valid_rewards = [step.reward for step in history.history if step.reward is not None]
        if valid_rewards:
            episode_returns.append(sum(valid_rewards))
        else:
            episode_returns.append(0.0)

        episode_lengths.append(len(history.history))

    if debug:
        logger.debug(
            "Policy %s: %d returns, %d lengths",
            policy_name,
            len(episode_returns),
            len(episode_lengths),
        )

    return episode_returns, episode_lengths


def _display_policy_statistics(
    policy_name: str,
    policy_histories: List,
    episode_returns: List[float],
    episode_lengths: List[int],
) -> None:
    logger.info("\n📋 %s STATISTICS:", policy_name)
    logger.info("-" * 40)
    logger.info("Episodes completed: %d", len(policy_histories))
    logger.info("Average return: %8.3f", sum(episode_returns) / len(episode_returns))
    logger.info("Best return: %8.3f", max(episode_returns))
    logger.info("Worst return: %8.3f", min(episode_returns))
    logger.info("Average length: %8.1f", sum(episode_lengths) / len(episode_lengths))


def _display_multiple_evaluation_summary(
    environment: Environment,
    optimized_policies: List[Policy],
    num_episodes: int,
    num_steps: int,
    results: Dict,
    eval_cache_dir: Path,
    verbose: bool,
    debug: bool,
) -> None:
    if verbose:
        env_name = environment.name

        logger.info("\n📊 EVALUATION SUMMARY")
        logger.info("-" * 40)
        logger.info("Environment: %s", env_name)
        logger.info("Policies evaluated: %d", len(optimized_policies))
        logger.info("Episodes per policy: %d", num_episodes)
        logger.info("Steps per episode: %d", num_steps)

        for policy in optimized_policies:
            policy_name = policy.name

            if env_name in results and policy_name in results[env_name]:
                policy_histories = results[env_name][policy_name]
                episode_returns, episode_lengths = _calculate_policy_episode_statistics(
                    policy_histories, policy_name, debug
                )
                _display_policy_statistics(
                    policy_name, policy_histories, episode_returns, episode_lengths
                )

        logger.info("\n💾 Results saved to: %s", eval_cache_dir)
        logger.info("🔍 MLflow UI: cd %s && mlflow ui", eval_cache_dir)


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
    eval_cache_dir = cache_dir
    eval_cache_dir.mkdir(parents=True, exist_ok=True)

    _log_multiple_evaluation_setup(
        optimized_policies,
        environment,
        num_episodes,
        num_steps,
        n_jobs,
        eval_cache_dir,
        verbose,
        debug,
    )

    eval_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=optimized_policies,
            num_episodes=num_episodes,
            num_steps=num_steps,
        )
    ]

    if debug:
        logger.debug("Creating POMDPSimulator with cache_dir: %s", eval_cache_dir)
        logger.debug("Experiment name: %s, Task manager: JoblibConfig", experiment_name)

    task_manager_config = JoblibConfig(n_jobs=n_jobs)

    with POMDPSimulator(
        task_manager_config=task_manager_config,
        cache_dir_path=eval_cache_dir,
        experiment_name=experiment_name,
        debug=debug,
        task_console_output=False,
        enable_profiling=False,
    ) as simulator:
        if debug:
            logger.debug("POMDPSimulator created successfully, starting evaluation...")
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
                "Evaluation completed, received results for %d environments",
                len(results) if results else 0,
            )
            if statistics_df is not None:
                logger.debug("Statistics DataFrame shape: %s", statistics_df.shape)

        _display_multiple_evaluation_summary(
            environment,
            optimized_policies,
            num_episodes,
            num_steps,
            results,
            eval_cache_dir,
            verbose,
            debug,
        )

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

            >>> hyper_params = create_numerical_hyperparameter_ranges({
            ...     "exploration_constant": (0.1, 5.0),
            ...     "n_simulations": (50, 200),
            ...     "depth": (3, 8)
            ... })
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Creating %d numerical hyperparameters", len(parameter_configs))
        for name, (low, high) in parameter_configs.items():
            logger.debug("  %s: [%s, %s]", name, low, high)

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

            >>> hyper_params = create_categorical_hyperparameter_choices({
            ...     "algorithm": ["ucb", "thompson", "epsilon_greedy"],
            ...     "heuristic": ["random", "informed"]
            ... })
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Creating %d categorical hyperparameters", len(parameter_configs))
        for name, choices in parameter_configs.items():
            logger.debug("  %s: %s", name, choices)

    return [CategoricalHyperParameter(choices, name) for name, choices in parameter_configs.items()]


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

            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>>
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> compatible_planners = get_benchmark_hyperparameter_planners(env)
            >>> print(f"Compatible planners: {[p.__name__ for p in compatible_planners]}")
            Compatible planners: ['POMCP', 'StandardSparseSamplingDiscreteActionsPlanner', 'SparsePFT', 'POMCPOW', 'PFT_DPW', 'POMCP_DPW', 'DiscreteActionSequencesPlanner']
    """
    if debug:
        logger.debug("Finding compatible planners for environment: %s", env.name)
        logger.debug(
            "Environment spaces - action: %s, obs: %s",
            env.space_info.action_space.value,
            env.space_info.observation_space.value,
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
                        "Skipping %s: no space info available (likely abstract class)", planner_name
                    )
                continue

            planner_action_space = planner_space_info.action_space
            planner_obs_space = planner_space_info.observation_space

            if debug:
                logger.debug(
                    "Checking %s: action=%s, obs=%s",
                    planner_name,
                    planner_action_space.value,
                    planner_obs_space.value,
                )

            # Check action space compatibility
            action_compatible = _is_space_compatible(env_action_space, planner_action_space)

            # Check observation space compatibility
            obs_compatible = _is_space_compatible(env_obs_space, planner_obs_space)

            if action_compatible and obs_compatible:
                compatible_planners.append(planner_class)
                if debug:
                    logger.debug("✓ %s is compatible", planner_name)
            else:
                if debug:
                    logger.debug(
                        "✗ %s not compatible (action: %s, obs: %s)",
                        planner_name,
                        action_compatible,
                        obs_compatible,
                    )

        except Exception as e:
            if debug:
                logger.debug("Skipping %s due to error: %s", planner_name, e)
            continue

    if verbose:
        compatible_names = [p.__name__ for p in compatible_planners]
        logger.info(
            "Found %d compatible planners for %s: %s",
            len(compatible_planners),
            env.name,
            compatible_names,
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
