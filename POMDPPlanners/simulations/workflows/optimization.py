"""Hyperparameter optimization workflows."""

from pathlib import Path
from typing import List, Optional

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
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
from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


def run_hyperparameter_optimization_local_run(
    environment_run_params: List[HyperParameterRunParams],
    experiment_name: str = "POMDP_Hyperparameter_Optimization",
    n_jobs: int = -1,
    cache_dir_path: Optional[Path] = None,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    use_queue_logger: bool = False,
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
            available CPU cores, or specify a positive integer for a specific
            number of cores.
        cache_dir_path: Optional path for storing optimization results, logs,
            and MLflow artifacts. If None, results are stored in the current
            working directory.
        confidence_interval_level: Confidence level for statistical analysis
            (between 0.0 and 1.0). Used for computing confidence intervals in
            performance statistics. Defaults to 0.95 for 95% confidence intervals.
        alpha: Significance level for statistical tests (between 0.0 and 1.0).
            Used for hypothesis testing and confidence interval calculations.
            Defaults to 0.05 for 5% significance level.
        use_queue_logger: Whether to use queue-based logging for distributed
            execution scenarios. Defaults to False for local execution.

    Returns:
        List[OptimizedPolicyResult]: List of optimization results, each containing
            the optimized policy with its best hyperparameters, environment reference,
            and optimization metadata for each input configuration.

    Raises:
        ValueError: If any configuration contains invalid parameters or missing
            required fields like n_trials.
        TypeError: If policy classes are not Policy subclasses.
        RuntimeError: If optimization fails for any configuration.

    Example:
        Running hyperparameter optimization for POMCP on Tiger POMDP:

        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import (
        ...     NumericalHyperParameter, CategoricalHyperParameter
        ... )
        >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
        ...     HyperParameterRunParams, HyperParameterOptimizationDirection, HyperParamPlannerConfig
        ... )
        >>> # Initialize the API
        >>> api = LocalSimulationsAPI(debug=True)
        >>> # Create environment and initial belief
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> # Define hyperparameter optimization configurations
        >>> optimization_configs = [
        ...     HyperParameterRunParams(
        ...         environment=tiger,
        ...         belief=initial_belief,
        ...         hyper_param_planner_config=HyperParamPlannerConfig(
        ...             policy_cls=POMCP,
        ...             hyper_parameters=[
        ...                 NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
        ...                 NumericalHyperParameter(10, 50, "n_simulations")
        ...             ],
        ...             constant_parameters={
        ...                 "discount_factor": 0.95,
        ...                 "name": "OptimizedPOMCP",
        ...                 "depth": 5
        ...             }
        ...         ),
        ...         num_episodes=2,       # Small for testing
        ...         num_steps=3,          # Small for testing
        ...         n_trials=3,          # Small number for testing
        ...         parameters_to_optimize=[
        ...             ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
        ...         ]
        ...     )
        ... ]
        >>> # Run hyperparameter optimization
        >>> results = api.run_hyperparameter_optimization(
        ...     environment_run_params=optimization_configs,
        ...     experiment_name="Tiger_POMCP_Optimization",
        ...     n_jobs=1,
        ...     debug=True
        ... ) # doctest: +SKIP
        >>> # Check optimization results
        >>> len(results) >= 1  # doctest: +SKIP
        True

    Note:
        This method requires Optuna and MLflow to be installed. The optimization
        process can be computationally intensive for complex policies and large
        parameter spaces. All optimization runs are automatically tracked in
        MLflow with comprehensive parameter logging, metrics recording, and
        artifact storage.

        Key Implementation Details:
        - Uses HyperParameterOptimizer internally for optimization logic
        - All optimization runs are automatically tracked in MLflow
        - Requires explicit belief initialization (not generated automatically)
        - n_trials parameter is mandatory in each HyperParameterRunParams
        - Results include optimized policies with their chosen hyperparameters
        - Supports both numerical and categorical hyperparameters
    """
    logger.info(
        "Starting hyperparameter optimization for %s configurations",
        len(environment_run_params),
    )
    logger.debug(
        "Parameters: experiment_name=%s, n_jobs=%s, "  # pylint: disable=implicit-str-concat
        "confidence_interval=%s, alpha=%s",
        experiment_name,
        n_jobs,
        confidence_interval_level,
        alpha,
    )

    # Set up cache directory
    if cache_dir_path is None:
        cache_dir_path = Path("./hyperparameter_optimization_results")

    # Create cache directory if it doesn't exist
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    task_manager_config = JoblibConfig(n_jobs=1)

    # Initialize the hyperparameter optimizer
    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir_path,
        experiment_name=experiment_name,
        n_jobs=n_jobs,
        task_manager_config=task_manager_config,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        use_queue_logger=use_queue_logger,
    )

    try:
        # Run optimization
        logger.info("Running hyperparameter optimization")
        results = optimizer.optimize(environment_run_params)

        logger.info(
            "Hyperparameter optimization completed successfully. "
            "Optimized %s out of %s configurations",
            len(results),
            len(environment_run_params),
        )

        # Log summary of results
        for i, result in enumerate(results):
            logger.info(
                "Configuration %s: %s "  # pylint: disable=implicit-str-concat
                "with %s - "
                "Best parameters: %s",
                i + 1,
                result.environment.__class__.__name__,
                result.policy.__class__.__name__,
                result.chosen_hyper_parameters,
            )

        return results

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Hyperparameter optimization failed: %s", e)
        raise RuntimeError(f"Hyperparameter optimization failed: {e}") from e

    finally:
        # Clean up optimizer resources
        try:
            optimizer.cleanup()
        except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
            logger.warning("Error during optimizer cleanup: %s", cleanup_error)


def run_hyperparameter_optimization_pbs(
    environment_run_params: List[HyperParameterRunParams],
    queue: str,
    experiment_name: str = "POMDP_Hyperparameter_Optimization_PBS",
    n_workers: int = 4,
    cores: int = 1,
    memory: str = "4GB",
    processes: int = 1,
    walltime: str = "01:00:00",
    job_extra: Optional[List[str]] = None,
    n_jobs: int = -1,
    cache_dir_path: Optional[Path] = None,
    clear_cache_on_start: bool = False,
    confidence_interval_level: float = 0.95,
    alpha: float = 0.05,
    use_queue_logger: bool = True,
    enable_dashboard: bool = True,
    dashboard_address: str = "0.0.0.0",
    dashboard_port: int = 8787,
    dashboard_prefix: Optional[str] = None,
) -> List[OptimizedPolicyResult]:
    """Run hyperparameter optimization for POMDP policies using Optuna with PBS cluster.

    This method provides a high-level interface for hyperparameter optimization using
    PBS cluster computing for distributed execution. It wraps the HyperParameterOptimizer
    class and supports optimization of multiple environment-policy configurations with
    comprehensive MLflow tracking and statistical analysis.

    The optimization uses Optuna's advanced algorithms (TPE, CMA-ES, etc.) to efficiently
    search the hyperparameter space, with PBS cluster resources for scalable computation.

    Args:
        environment_run_params: List of HyperParameterRunParams configurations,
            each specifying an environment, policy class, hyperparameter ranges,
            and optimization settings. Each configuration must include the
            required n_trials parameter.
        queue: PBS queue name to submit jobs to
        experiment_name: Name for the MLflow experiment tracking. Used to organize
            optimization runs and enable comparison across different experiments.
        n_workers: Number of PBS worker nodes to request
        cores: Number of CPU cores per worker
        memory: Memory per worker (e.g., "4GB", "8GB")
        processes: Number of processes per worker
        walltime: Maximum job runtime in HH:MM:SS format
        job_extra: Additional PBS job directives
        n_jobs: Number of parallel jobs for episode execution. Use -1 to use all
            available CPU cores, or specify a positive integer for a specific
            number of cores.
        cache_dir_path: Optional path for storing optimization results, logs,
            and MLflow artifacts. If None, results are stored in the current
            working directory.
        clear_cache_on_start: Whether to clear existing cache before starting
            optimization. Useful for ensuring clean runs when debugging or testing.
        confidence_interval_level: Confidence level for statistical analysis
            (between 0.0 and 1.0). Used for computing confidence intervals in
            performance statistics. Defaults to 0.95 for 95% confidence intervals.
        alpha: Significance level for statistical tests (between 0.0 and 1.0).
            Used for hypothesis testing and confidence interval calculations.
            Defaults to 0.05 for 5% significance level.
        use_queue_logger: Whether to use queue-based logging for distributed
            execution scenarios. Defaults to True for PBS execution.
        enable_dashboard: Whether to enable Dask dashboard
        dashboard_address: Dashboard bind address
        dashboard_port: Dashboard port
        dashboard_prefix: Dashboard URL prefix

    Returns:
        List[OptimizedPolicyResult]: List of optimization results, each containing
            the optimized policy with its best hyperparameters, environment reference,
            and optimization metadata for each input configuration.

    Raises:
        ValueError: If any configuration contains invalid parameters or missing
            required fields like n_trials.
        TypeError: If policy classes are not Policy subclasses.
        RuntimeError: If PBS optimization fails for any configuration.

    Example:
        Running PBS hyperparameter optimization for POMCP on Tiger POMDP:

        >>> from pathlib import Path
        >>> from POMDPPlanners.simulations.workflows.optimization import run_hyperparameter_optimization_pbs
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.core.simulation import NumericalHyperParameter
        >>> from POMDPPlanners.core.simulation.hyperparameter_tuning import (
        ...     HyperParameterRunParams, HyperParameterOptimizationDirection, HyperParamPlannerConfig
        ... )
        >>> # Create environment and initial belief
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> # Define hyperparameter optimization configurations
        >>> optimization_configs = [
        ...     HyperParameterRunParams(
        ...         environment=tiger,
        ...         belief=initial_belief,
        ...         hyper_param_planner_config=HyperParamPlannerConfig(
        ...             policy_cls=POMCP,
        ...             hyper_parameters=[
        ...                 NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
        ...                 NumericalHyperParameter(10, 50, "n_simulations")
        ...             ],
        ...             constant_parameters={
        ...                 "discount_factor": 0.95,
        ...                 "name": "OptimizedPOMCP",
        ...                 "depth": 5
        ...             }
        ...         ),
        ...         num_episodes=2,
        ...         num_steps=3,
        ...         n_trials=3,
        ...         parameters_to_optimize=[
        ...             ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
        ...         ]
        ...     )
        ... ]
        >>> # Run PBS hyperparameter optimization (commented out for doctest)
        >>> # results = run_hyperparameter_optimization_pbs(
        >>> #     environment_run_params=optimization_configs,
        >>> #     queue="short",
        >>> #     experiment_name="Tiger_POMCP_PBS_Optimization",
        >>> #     n_workers=4,
        >>> #     cores=2,
        >>> #     memory="8GB",
        >>> #     walltime="02:00:00",
        >>> #     n_jobs=1,
        >>> #     debug=True
        >>> # )
        >>> len(optimization_configs) >= 1  # doctest: +SKIP
        True

    Note:
        This method requires Optuna, MLflow, and PBS cluster access. The optimization
        process can be computationally intensive for complex policies and large
        parameter spaces. All optimization runs are automatically tracked in
        MLflow with comprehensive parameter logging, metrics recording, and
        artifact storage.

        Key Implementation Details:
        - Uses HyperParameterOptimizer with PBS task manager internally
        - All optimization runs are automatically tracked in MLflow
        - Requires explicit belief initialization (not generated automatically)
        - n_trials parameter is mandatory in each HyperParameterRunParams
        - Results include optimized policies with their chosen hyperparameters
        - Supports both numerical and categorical hyperparameters
        - PBS cluster resources enable large-scale distributed optimization
    """
    logger.info(
        "Starting PBS hyperparameter optimization for %s configurations",
        len(environment_run_params),
    )
    logger.debug(
        "PBS Parameters: queue=%s, n_workers=%s, cores=%s, "  # pylint: disable=implicit-str-concat
        "memory=%s, walltime=%s",
        queue,
        n_workers,
        cores,
        memory,
        walltime,
    )
    logger.debug(
        "Optimization Parameters: experiment_name=%s, n_jobs=%s, "
        "confidence_interval=%s, alpha=%s",
        experiment_name,
        n_jobs,
        confidence_interval_level,
        alpha,
    )

    # Set up cache directory
    if cache_dir_path is None:
        cache_dir_path = Path("./hyperparameter_optimization_pbs_results")

    # Create cache directory if it doesn't exist
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Create PBS task manager configuration
    task_manager_config = PBSConfig(
        queue=queue,
        n_workers=n_workers,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        job_extra=job_extra,
        clear_cache_on_start=clear_cache_on_start,
        enable_dashboard=enable_dashboard,
        dashboard_address=dashboard_address,
        dashboard_port=dashboard_port,
        dashboard_prefix=dashboard_prefix,
    )

    # Initialize the hyperparameter optimizer with PBS
    optimizer = HyperParameterOptimizer(
        cache_dir_path=cache_dir_path,
        experiment_name=experiment_name,
        n_jobs=n_jobs,
        task_manager_config=task_manager_config,
        confidence_interval_level=confidence_interval_level,
        alpha=alpha,
        use_queue_logger=use_queue_logger,
    )

    try:
        # Run optimization
        logger.info("Running PBS hyperparameter optimization")
        if enable_dashboard:
            logger.info("Dashboard available at: http://%s:%s", dashboard_address, dashboard_port)

        results = optimizer.optimize(environment_run_params)

        logger.info(
            "PBS hyperparameter optimization completed successfully. "
            "Optimized %s out of %s configurations",
            len(results),
            len(environment_run_params),
        )

        # Log summary of results
        for i, result in enumerate(results):
            logger.info(
                "Configuration %s: %s "  # pylint: disable=implicit-str-concat
                "with %s - "
                "Best parameters: %s",
                i + 1,
                result.environment.__class__.__name__,
                result.policy.__class__.__name__,
                result.chosen_hyper_parameters,
            )

        return results

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("PBS hyperparameter optimization failed: %s", e)
        raise RuntimeError(f"PBS hyperparameter optimization failed: {e}") from e

    finally:
        # Clean up optimizer resources
        try:
            optimizer.cleanup()
        except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
            logger.warning("Error during optimizer cleanup: %s", cleanup_error)
