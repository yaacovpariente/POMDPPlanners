"""Integrated optimization and evaluation workflows.

This module provides both class-based and function-based APIs for running
optimization and evaluation workflows. The class-based API (LocalWorkflow,
PBSWorkflow) is recommended for new code, while the function-based API is
maintained for backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from POMDPPlanners.core.simulation.hyperparameter_tuning import ParameterToOptimizeMapper
from POMDPPlanners.configs.experiment_configs import AverageReturnParameterToOptimizeMapper
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfigGenerator,
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    PBSConfig,
)
from POMDPPlanners.configs.experiment_configs import (
    complete_environments_and_benchmarks_hyperparameter_optimization_configs,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.utils.logger import get_logger
import pandas as pd
from typing import Type
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import TaskManagerConfig
from POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationLocalWorkflow as LocalWorkflow,
    OptimizationEvaluationPBSWorkflow as PBSWorkflow,
)

logger = get_logger(__name__)


def optimize_and_evaluate_multiple_environments_and_policies(
    configs: List[HyperParameterRunParams],
    num_episodes_evaluation: int,
    num_steps_evaluation: int,
    cache_dir: Path,
    experiment_name: str,
    task_manager_config_hyperparameter: TaskManagerConfig,
    task_manager_config_evaluation: TaskManagerConfig,
    n_jobs: int,
    confidence_interval_level: float,
    alpha: float,
    debug: bool = False,
    verbose: bool = True,
    cache_visualizations: bool = True,
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Optimize and evaluate multiple POMDP planners on multiple environments and policies."""
    if verbose:
        logger.info(f"Optimizing {len(configs)} environments and policies")

    if debug:
        logger.debug(f"Optimizing {len(configs)} environments and policies")

    if cache_dir is None:
        cache_dir = Path(f"./{experiment_name.lower().replace(' ', '_')}_results")

    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir,
        experiment_name=experiment_name,
        task_manager_config=task_manager_config_hyperparameter,
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
        task_manager_config=task_manager_config_evaluation,
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


def run_comprehensive_benchmark_hyperparameter_optimization(
    generators: List["HyperParamPlannerConfigGenerator"],
    cache_dir: Path,
    task_manager_config_hyperparameter: TaskManagerConfig,
    task_manager_config_evaluation: TaskManagerConfig,
    particles: int = 30,
    num_episodes: int = 10,
    num_steps: int = 20,
    n_trials: int = 100,
    optimization_episodes: int = 3,
    optimization_steps: int = 6,
    evaluation_n_jobs: int = 1,
    cache_visualizations: bool = True,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
    experiment_name: str = "Comprehensive_Benchmark",
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    parameter_to_optimize_mapper = AverageReturnParameterToOptimizeMapper()
    optimization_configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
        generators=generators,
        parameter_to_optimize_mapper=parameter_to_optimize_mapper,
        particles=particles,
        num_episodes=num_episodes,
        num_steps=num_steps,
        n_trials=n_trials,
    )

    return optimize_and_evaluate_multiple_environments_and_policies(
        configs=optimization_configs,
        num_episodes_evaluation=optimization_episodes,
        num_steps_evaluation=optimization_steps,
        cache_dir=cache_dir,
        experiment_name=experiment_name,
        task_manager_config_hyperparameter=task_manager_config_hyperparameter,
        task_manager_config_evaluation=task_manager_config_evaluation,
        n_jobs=evaluation_n_jobs,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
        cache_visualizations=cache_visualizations,
    )


def run_comprehensive_benchmark_hyperparameter_optimization_local(
    generators: List["HyperParamPlannerConfigGenerator"],
    cache_dir: Path,
    task_manager_config_hyperparameter: TaskManagerConfig,
    task_manager_config_evaluation: TaskManagerConfig,
    particles: int = 30,
    num_episodes: int = 10,
    num_steps: int = 20,
    n_trials: int = 100,
    optimization_episodes: int = 3,
    optimization_steps: int = 6,
    evaluation_n_jobs: int = 1,
    optimization_n_jobs: int = -1,
    cache_visualizations: bool = True,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
    experiment_name: str = "Comprehensive_Benchmark",
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    task_manager_config_hyperparameter = JoblibConfig(n_jobs=optimization_n_jobs)
    task_manager_config_evaluation = JoblibConfig(n_jobs=evaluation_n_jobs)

    return run_comprehensive_benchmark_hyperparameter_optimization(
        generators=generators,
        cache_dir=cache_dir,
        task_manager_config_hyperparameter=task_manager_config_hyperparameter,
        task_manager_config_evaluation=task_manager_config_evaluation,
        particles=particles,
        num_episodes=num_episodes,
        num_steps=num_steps,
        n_trials=n_trials,
        optimization_episodes=optimization_episodes,
        optimization_steps=optimization_steps,
        evaluation_n_jobs=evaluation_n_jobs,
        cache_visualizations=cache_visualizations,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
        experiment_name=experiment_name,
    )


# def optimize_and_evaluate_multiple_environments_and_policies_pbs(
#     configs: List[HyperParameterRunParams],
#     num_episodes_evaluation: int,
#     num_steps_evaluation: int,
#     cache_dir: Path,
#     experiment_name: str,
#     queue: str,
#     n_workers: int,
#     cores: int,
#     memory: str,
#     processes: int,
#     walltime: str,
#     job_extra: Optional[List[str]],
#     n_jobs: int,
#     confidence_interval_level: float,
#     alpha: float,
#     debug: bool = False,
#     verbose: bool = True,
#     cache_visualizations: bool = True,
# ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
#     """Optimize and evaluate multiple POMDP planners on multiple environments and policies.
#     """
#     if verbose:
#         logger.info(f"Optimizing {len(configs)} environments and policies")

#     if debug:
#         logger.debug(f"Optimizing {len(configs)} environments and policies")

#     if cache_dir is None:
#         cache_dir = Path(f"./{experiment_name.lower().replace(' ', '_')}_results")

#     # Parallelization of hyperparameter optimization is using Optuna, and therefore
#     # each planner's optimization is limited by the number of cores on the machine.
#     n_workers_hyperparameter_optimization = min(n_workers, len(configs))

#     task_manager_config_hyperparameter = PBSConfig(
#         queue=queue,
#         n_workers=n_workers_hyperparameter_optimization,
#         cores=cores,
#         memory=memory,
#         processes=processes,
#         walltime=walltime,
#         job_extra=job_extra,
#     )

#     task_manager_config_evaluation = PBSConfig(
#         queue=queue,
#         n_workers=n_workers,
#         cores=cores,
#         memory=memory,
#         processes=processes,
#         walltime=walltime,
#         job_extra=job_extra,
#     )

#     return optimize_and_evaluate_multiple_environments_and_policies(
#         configs=configs,
#         num_episodes_evaluation=num_episodes_evaluation,
#         num_steps_evaluation=num_steps_evaluation,
#         cache_dir=cache_dir,
#         experiment_name=experiment_name,
#         task_manager_config_hyperparameter=task_manager_config_hyperparameter,
#         task_manager_config_evaluation=task_manager_config_evaluation,
#         n_jobs=n_jobs,
#         confidence_interval_level=confidence_interval_level,
#         alpha=alpha,
#         debug=debug,
#         verbose=verbose,
#         cache_visualizations=cache_visualizations,
#     )


def run_comprehensive_benchmark_hyperparameter_optimization_pbs(
    generators: List["HyperParamPlannerConfigGenerator"],
    cache_dir: Path,
    particles: int = 30,
    num_episodes: int = 10,
    num_steps: int = 20,
    n_trials: int = 100,
    optimization_episodes: int = 3,
    optimization_steps: int = 6,
    evaluation_n_jobs: int = 1,
    queue: str = "short",
    n_workers: int = 4,
    cores: int = 1,
    memory: str = "4GB",
    processes: int = 1,
    walltime: str = "03:00:00",
    job_extra: Optional[List[str]] = None,
    cache_visualizations: bool = True,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    debug: bool = False,
    verbose: bool = True,
    experiment_name: str = "Comprehensive_Benchmark",
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    task_manager_config = PBSConfig(
        queue=queue,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
    )

    return run_comprehensive_benchmark_hyperparameter_optimization(
        generators=generators,
        cache_dir=cache_dir,
        task_manager_config_hyperparameter=task_manager_config,
        task_manager_config_evaluation=task_manager_config,
        particles=particles,
        num_episodes=num_episodes,
        num_steps=num_steps,
        n_trials=n_trials,
        optimization_episodes=optimization_episodes,
        optimization_steps=optimization_steps,
        evaluation_n_jobs=evaluation_n_jobs,
        cache_visualizations=cache_visualizations,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        debug=debug,
        verbose=verbose,
        experiment_name=experiment_name,
    )
