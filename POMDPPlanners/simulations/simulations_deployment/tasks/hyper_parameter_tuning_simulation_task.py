import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import optuna
from optuna.trial import FrozenTrial

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    History,
    NumericalHyperParameter,
    SimulationTask,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    OptimizedPolicyResult,
)
from POMDPPlanners.core.simulation.simulation_configs import EnvironmentRunParams
from POMDPPlanners.simulations.simulation_statistics import (
    compute_statistics_environment_policy_pair,
)
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
)
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger

HyperParameterFeature = Union[CategoricalHyperParameter, NumericalHyperParameter]


class HyperParameterTuningSimulationTask(SimulationTask):
    def __init__(
        self,
        environment: Environment,
        belief: Belief,
        policy_cls: Type[Policy],
        hyper_parameters: Sequence[HyperParameterFeature],
        constant_parameters: Dict[str, Any],
        num_episodes: int,
        num_steps: int,
        parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]],
        experiment_name: str = "hyperparameter_optimization",
        n_trials: int = 50,
        cache_dir: Optional[Path] = None,
        debug: bool = False,
        console_output: bool = True,
        n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.1,
        seed: int = 42,
        use_queue_logger: bool = False,
    ):
        """Initialize a hyperparameter tuning simulation task.

        Args:
            environment: The environment to use for optimization
            belief: The initial belief state
            policy_cls: The policy class to optimize
            hyper_parameters: Sequence of hyperparameter definitions
            constant_parameters: Dictionary of constant parameters for policy creation
            num_episodes: Number of episodes to run per trial
            num_steps: Number of steps per episode
            parameters_to_optimize: List of (parameter_name, direction) tuples
            experiment_name: Name of the experiment
            n_trials: Number of optimization trials
            cache_dir: Directory for caching results
            debug: Whether to enable debug logging
            console_output: Whether to enable console output
            n_jobs: Number of parallel jobs
            confidence_interval_level: Confidence interval level (0-1)
            alpha: Alpha parameter for statistics (0-1)
            seed: Random seed for reproducibility
            use_queue_logger: Whether to use queue-based logging

        Raises:
            ValueError: If any input parameter is invalid
            TypeError: If any input parameter has incorrect type
        """
        self._validate_inputs(
            environment=environment,
            belief=belief,
            policy_cls=policy_cls,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
            num_episodes=num_episodes,
            num_steps=num_steps,
            parameters_to_optimize=parameters_to_optimize,
            experiment_name=experiment_name,
            n_trials=n_trials,
            cache_dir=cache_dir,
            debug=debug,
            console_output=console_output,
            n_jobs=n_jobs,
            confidence_interval_level=confidence_interval_level,
            alpha=alpha,
            seed=seed,
            use_queue_logger=use_queue_logger,
        )

        self.environment = environment
        self.belief = belief
        self.policy_cls = policy_cls
        self.hyper_parameters = hyper_parameters
        self.constant_parameters = constant_parameters
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.parameters_to_optimize = parameters_to_optimize
        self.n_trials = n_trials

        self.cache_dir = cache_dir
        self.debug = debug
        self.console_output = console_output
        self.n_jobs = n_jobs
        self.confidence_interval_level = confidence_interval_level
        self.alpha = alpha
        self.seed = seed
        self.use_queue_logger = use_queue_logger
        # Create task manager config for joblib
        task_manager_config = JoblibConfig(n_jobs=n_jobs, console_output=False, no_logs=True)

        self.simulator = POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=None,
            experiment_name=experiment_name,
            debug=debug,  # Keep episode-level logging minimal for optimization
            use_queue_logger=use_queue_logger,
            console_output=False,
            no_logs=True,
        )

        if cache_dir is not None:
            storage_dir = cache_dir / "study_storage" / self.get_config_id()
            self.study_storage = DiskCacheDB(cache_dir=str(storage_dir))
        else:
            self.study_storage = None

    @staticmethod
    def _validate_inputs(
        environment: Environment,
        belief: Belief,
        policy_cls: Type[Policy],
        hyper_parameters: Sequence[HyperParameterFeature],
        constant_parameters: Dict[str, Any],
        num_episodes: int,
        num_steps: int,
        parameters_to_optimize: List[Tuple[str, HyperParameterOptimizationDirection]],
        experiment_name: str,
        n_trials: int,
        cache_dir: Optional[Path],
        debug: bool,
        console_output: bool,
        n_jobs: int,
        confidence_interval_level: float,
        alpha: float,
        seed: int,
        use_queue_logger: bool,
    ) -> None:
        """Validate input parameters for HyperParameterTuningSimulationTask.

        Args:
            environment: The environment to use for optimization
            belief: The initial belief state
            policy_cls: The policy class to optimize
            hyper_parameters: Sequence of hyperparameter definitions
            constant_parameters: Dictionary of constant parameters for policy creation
            num_episodes: Number of episodes to run per trial
            num_steps: Number of steps per episode
            parameters_to_optimize: List of (parameter_name, direction) tuples
            experiment_name: Name of the experiment
            n_trials: Number of optimization trials
            cache_dir: Directory for caching results
            debug: Whether to enable debug logging
            console_output: Whether to enable console output
            n_jobs: Number of parallel jobs
            confidence_interval_level: Confidence interval level (0-1)
            alpha: Alpha parameter for statistics (0-1)
            seed: Random seed for reproducibility
            use_queue_logger: Whether to use queue-based logging

        Raises:
            ValueError: If any input parameter is invalid
            TypeError: If any input parameter has incorrect type
        """
        # Validate required objects
        if environment is None:
            raise ValueError("environment cannot be None")
        if not isinstance(environment, Environment):
            raise TypeError(f"environment must be an Environment instance, got {type(environment)}")
        if belief is None:
            raise ValueError("belief cannot be None")
        if not isinstance(belief, Belief):
            raise TypeError(f"belief must be a Belief instance, got {type(belief)}")
        if policy_cls is None:
            raise ValueError("policy_cls cannot be None")
        if not isinstance(policy_cls, type):
            raise TypeError(f"policy_cls must be a class (type), got {type(policy_cls)}")

        # Validate sequences and collections
        if not isinstance(hyper_parameters, Sequence):
            raise TypeError("hyper_parameters must be a sequence")
        if len(hyper_parameters) == 0:
            raise ValueError("hyper_parameters cannot be empty")
        if not isinstance(constant_parameters, dict):
            raise TypeError("constant_parameters must be a dictionary")

        # Validate parameters_to_optimize
        if not isinstance(parameters_to_optimize, list):
            raise TypeError("parameters_to_optimize must be a list")
        if len(parameters_to_optimize) == 0:
            raise ValueError("parameters_to_optimize cannot be empty")
        # Validate structure of each tuple before checking uniqueness
        for param_tuple in parameters_to_optimize:
            if not isinstance(param_tuple, tuple):
                raise TypeError("Each element in parameters_to_optimize must be a tuple")
            if len(param_tuple) != 2:
                raise ValueError(
                    "Each tuple in parameters_to_optimize must have exactly 2 elements"
                )
            param_name, direction = param_tuple
            if not isinstance(param_name, str):
                raise TypeError("First element of each tuple must be a string (parameter name)")
            if not isinstance(direction, HyperParameterOptimizationDirection):
                raise TypeError(
                    f"Second element must be HyperParameterOptimizationDirection, got {type(direction)}"
                )
        # Check uniqueness after structure validation
        if len(set(parameters_to_optimize)) != len(parameters_to_optimize):
            raise ValueError("parameters_to_optimize must be a list of unique tuples")

        # Validate integer parameters
        if not isinstance(num_episodes, int) or num_episodes <= 0:
            raise ValueError("num_episodes must be a positive integer")
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be a positive integer")
        if not isinstance(n_jobs, int) or (n_jobs <= 0 and n_jobs != -1):
            raise ValueError("n_jobs must be a positive integer or -1 (for all cores)")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        # Validate string parameters
        if not isinstance(experiment_name, str):
            raise TypeError("experiment_name must be a string")
        if len(experiment_name) == 0:
            raise ValueError("experiment_name cannot be empty")

        # Validate float parameters with range checks
        if not isinstance(confidence_interval_level, (int, float)):
            raise TypeError("confidence_interval_level must be a number")
        if confidence_interval_level <= 0 or confidence_interval_level > 1:
            raise ValueError("confidence_interval_level must be between 0 and 1")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        # Validate cache_dir
        if cache_dir is not None and not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a Path object or None")

        # Validate boolean parameters
        if not isinstance(debug, bool):
            raise TypeError("debug must be a boolean")
        if not isinstance(console_output, bool):
            raise TypeError("console_output must be a boolean")
        if not isinstance(use_queue_logger, bool):
            raise TypeError("use_queue_logger must be a boolean")

    @property
    def logger(self) -> logging.Logger:
        """All tasks should remain pickable and therefore the logger should be a property"""
        output_dir = None
        if self.cache_dir is not None:
            output_dir = self.cache_dir / "logs" / "hyper_parameter_tuning"

        return get_logger(
            name=f"task.{self.environment.name}.{self.policy_cls.__name__}",
            debug=self.debug,
            output_dir=output_dir,
            console_output=self.console_output,
            use_queue=self.use_queue_logger,
        )

    def run(self) -> Union[OptimizedPolicyResult, None]:
        """Run the hyperparameter optimization task using multi-objective optimization.

        This method performs hyperparameter optimization for the given policy class
        using Optuna's native multi-objective optimization framework. It evaluates
        multiple parameter configurations by running episodes and uses intelligent
        search algorithms to explore the Pareto frontier efficiently.

        The optimization process:
        1. Uses Optuna's multi-objective study to intelligently sample hyperparameters
        2. Each trial returns actual metric values (not dummy values)
        3. Optuna identifies the Pareto-optimal set of trials
        4. Selects the best trial from the Pareto set using normalized scoring

        Returns:
            OptimizedPolicyResult: NamedTuple containing optimization results with fields:
                - environment: The environment used for optimization
                - policy: The optimized policy instance
                - chosen_hyper_parameters: Best hyperparameters found
                - num_episodes: Number of episodes used for evaluation
                - num_steps: Number of steps per episode
                - parameters_to_optimize: List of (parameter_name, direction) tuples
                - optimized_metric_values: Dict of optimized metric values
            None: If optimization fails

        Raises:
            Exception: If optimization encounters critical errors
        """
        start_time = time.time()

        # Store current random state
        state = np.random.get_state()

        try:
            # Set random seed for reproducibility
            self.logger.debug("Setting random seed to %s", self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

            # Initialize and configure optimization session
            self._initialize_optimization_session()

            # Create the Optuna objective function
            objective_function = self._create_optuna_objective_function()

            # Execute the optimization study
            study = self._execute_optimization_study(
                objective_function, n_trials=self.n_trials, n_jobs=self.n_jobs
            )

            # Calculate optimization time
            optimization_time = time.time() - start_time

            # Build and return results
            return self._build_optimization_results(study, optimization_time)

        except Exception as e:
            self._handle_optimization_failure(e, start_time)
        finally:
            # Restore random state
            self.logger.debug("Restoring random state")
            np.random.set_state(state)
            self.logger.debug("Random state restored")

    def _initialize_optimization_session(self) -> None:
        """Initialize and log optimization session configuration."""
        self.logger.info("Starting hyperparameter optimization task with Pareto selection")

        # Log task configuration details
        self.logger.info(
            "Environment: %s (config_id: %s)", self.environment.name, self.environment.config_id
        )
        self.logger.info("Policy class: %s", self.policy_cls.__name__)

        # Log all parameters to optimize
        params_str = ", ".join(
            [
                f"{param_name} ({direction.value})"
                for param_name, direction in self.parameters_to_optimize
            ]
        )
        self.logger.info("Parameters to optimize: %s", params_str)

        self.logger.info("Hyperparameters: %s", [param.name for param in self.hyper_parameters])
        self.logger.info(
            "Configuration: num_episodes=%s, num_steps=%s", self.num_episodes, self.num_steps
        )
        self.logger.info(
            "Parallel jobs: %s, confidence_interval=%s", self.n_jobs, self.confidence_interval_level
        )
        self.logger.info("Seed: %s", self.seed)

        if self.cache_dir:
            self.logger.info("Cache directory: %s", self.cache_dir)
        self.logger.info("Task config ID: %s", self.get_config_id())

        # Log the configured number of trials
        self.logger.info("Running optimization with %d trials", self.n_trials)

    def _create_policy_parameter_suggestions(
        self, trial: FrozenTrial, hyperparameters: Sequence[HyperParameterFeature]
    ) -> Dict[str, Any]:
        """Create policy parameters from Optuna trial suggestions.

        Args:
            trial: Optuna trial object for parameter suggestion
            hyperparameters: List of hyperparameter definitions

        Returns:
            dict: Dictionary of suggested parameter values for policy construction
        """
        policy_params: Dict[str, Any] = {
            "environment": self.environment,
        }

        policy_params.update(self.constant_parameters)
        # Add optimization parameters based on their types
        for param in hyperparameters:
            if isinstance(param, CategoricalHyperParameter):
                policy_params[param.name] = trial.suggest_categorical(param.name, param.choices)
            elif isinstance(param, NumericalHyperParameter):
                if isinstance(param.low, float) and isinstance(param.high, float):
                    policy_params[param.name] = trial.suggest_float(
                        param.name, param.low, param.high
                    )
                elif isinstance(param.low, int) and isinstance(param.high, int):
                    policy_params[param.name] = trial.suggest_int(param.name, param.low, param.high)
                else:
                    raise ValueError(
                        f"Invalid parameter type: {type(param.low)} or {type(param.high)}"
                    )

        return policy_params

    def _evaluate_policy_configuration(
        self, policy: Policy, trial: FrozenTrial
    ) -> Dict[str, float]:
        """Evaluate a policy configuration and return multiple metrics.

        Args:
            policy: Policy instance to evaluate
            trial: Optuna trial object for storing metadata

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their values

        Raises:
            ValueError: If target optimization parameters not found in statistics
        """
        trial_id = str(trial._trial_id)
        if self.study_storage is not None and self.study_storage.is_key_in_cache(trial_id):
            stored_params = self.study_storage.get(trial_id)
            metric_values = stored_params["metric_values"]
            statistics = stored_params["statistics"]
            histories = stored_params["histories"]

            trial.set_user_attr("histories", histories)
            trial.set_user_attr("statistics", [stat._asdict() for stat in statistics])

            # Store individual metric values as trial attributes (needed for Pareto scoring)
            for metric_name, metric_value in metric_values.items():
                trial.set_user_attr(f"metric_{metric_name}", metric_value)

            return metric_values

        try:
            # Run multiple episodes to evaluate this parameter configuration
            histories = self.run_multiple_episodes(
                environment=self.environment,
                policy=policy,
                initial_belief=deepcopy(self.belief),
                num_episodes=self.num_episodes,
                num_steps=self.num_steps,
                scheduler_address=None,
            )

            # Compute statistics from the episodes
            statistics = compute_statistics_environment_policy_pair(
                env=self.environment,
                histories=histories,
                alpha=self.alpha,
                confidence_interval_level=self.confidence_interval_level,
            )

            # Store trial metadata for later analysis
            trial.set_user_attr("histories", histories)
            trial.set_user_attr("statistics", [stat._asdict() for stat in statistics])

            # Extract all target metric values
            metric_values = {}
            parameter_names = {param_name for param_name, _ in self.parameters_to_optimize}

            for metric in statistics:
                if metric.name in parameter_names:
                    metric_values[metric.name] = metric.value

            # Verify all required parameters were found
            missing_params = parameter_names - set(metric_values.keys())
            if missing_params:
                raise ValueError(f"Parameters {missing_params} not found in computed statistics")

            # Store individual metric values as trial attributes
            for metric_name, metric_value in metric_values.items():
                trial.set_user_attr(f"metric_{metric_name}", metric_value)

            params_to_store = {
                "metric_values": metric_values,
                "statistics": statistics,
                "histories": histories,
            }
            if self.study_storage is not None:
                self.study_storage.set(trial_id, params_to_store)

            return metric_values

        except Exception as e:
            self.logger.error("Error in evaluation function for trial %s: %s", trial_id, e)
            raise e

    def _compute_pareto_scores(self, study, pareto_trials=None) -> Dict[int, float]:
        """Compute normalized Pareto scores for specified trials.

        Args:
            study: Completed Optuna study with trial data
            pareto_trials: List of trials to score. If None, scores all completed trials.

        Returns:
            Dict mapping trial number to aggregated Pareto score
        """
        # Use provided pareto_trials if available, otherwise use all trials
        trials_to_score = pareto_trials if pareto_trials is not None else study.trials

        # Collect all metric values across trials to score
        trial_metrics = {}
        for trial in trials_to_score:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            metrics = {}
            for param_name, direction in self.parameters_to_optimize:
                metric_key = f"metric_{param_name}"
                if metric_key not in trial.user_attrs:
                    continue
                metrics[param_name] = trial.user_attrs[metric_key]

            if len(metrics) == len(self.parameters_to_optimize):
                trial_metrics[trial.number] = metrics

        if not trial_metrics:
            raise ValueError("No completed trials with all required metrics found")

        # Compute mean and std for each metric across trials to score
        metric_stats = {}
        for param_name, _ in self.parameters_to_optimize:
            values = [trial_metrics[trial_num][param_name] for trial_num in trial_metrics.keys()]
            metric_stats[param_name] = {
                "mean": np.mean(values),
                "std": np.std(values) if np.std(values) > 0 else 1.0,  # Avoid division by zero
            }

        # Compute normalized scores for each trial
        pareto_scores = {}
        for trial_num, metrics in trial_metrics.items():
            normalized_metrics = []

            for param_name, direction in self.parameters_to_optimize:
                value = metrics[param_name]
                mean = metric_stats[param_name]["mean"]
                std = metric_stats[param_name]["std"]

                # Normalize to z-score
                normalized = (value - mean) / std

                # Flip sign for minimization objectives
                if direction == HyperParameterOptimizationDirection.MINIMIZE:
                    normalized = -normalized

                normalized_metrics.append(normalized)

            # Average normalized metrics to get final score
            pareto_scores[trial_num] = np.mean(normalized_metrics)

        return pareto_scores

    def _create_optuna_objective_function(self):
        """Create the Optuna objective function for multi-objective optimization.

        Returns:
            Callable: Objective function that returns tuple of metric values for Optuna
        """

        def objective(trial: FrozenTrial) -> Tuple[float, ...]:
            """Optuna objective function for a single optimization trial.

            Returns:
                Tuple of metric values in the same order as self.parameters_to_optimize.
                Optuna uses these values to perform intelligent multi-objective optimization.
            """
            policy_params = None
            metric_values = None
            try:
                # Create parameters dictionary from hyperparameters
                policy_params = self._create_policy_parameter_suggestions(
                    trial, self.hyper_parameters
                )

                # Create policy instance with suggested parameters
                policy = self.policy_cls(**policy_params)

                # Evaluate and store metrics - actual values stored in trial.user_attrs
                metric_values = self._evaluate_policy_configuration(policy, trial)

                # Return metrics as tuple in the order specified by parameters_to_optimize
                # This allows Optuna to intelligently guide the hyperparameter search
                metric_tuple = tuple(
                    metric_values[param_name] for param_name, _ in self.parameters_to_optimize
                )

                return metric_tuple

            except Exception as e:
                if metric_values is not None:
                    self.logger.info(
                        "Trial %d completed with metrics: %s", trial.number, metric_values
                    )
                if policy_params is not None:
                    self.logger.debug(
                        "Trial %d: Testing parameters %s", trial.number, policy_params
                    )
                self.logger.error("Error in objective function for trial %d: %s", trial.number, e)
                self.logger.exception("Full exception details:")
                raise e

        return objective

    def _execute_optimization_study(self, objective_function, n_trials: int, n_jobs: int = 1):
        """Execute the Optuna optimization study with multi-objective optimization.

        Args:
            objective_function: Function to optimize (returns tuple of metric values)
            n_trials: Number of optimization trials to run
            n_jobs: Number of parallel jobs for optimization

        Returns:
            optuna.Study: Completed multi-objective optimization study
        """
        # Create and run the optimization study with multiple objectives
        self.logger.info("Creating Optuna study for multi-objective optimization...")

        # Create study with directions for each parameter to optimize
        directions = [direction.value for _, direction in self.parameters_to_optimize]
        self.logger.info("Optimization directions: %s", directions)

        study = optuna.create_study(directions=directions)

        self.logger.info("Starting optimization trials...")
        study.optimize(objective_function, n_trials=n_trials, n_jobs=n_jobs)

        # Log optimization completion
        self.logger.info("Optimization completed successfully!")
        self.logger.info("Found %d Pareto-optimal trials", len(study.best_trials))

        return study

    def _build_optimization_results(self, study, optimization_time: float) -> OptimizedPolicyResult:
        """Build OptimizedPolicyResult using Pareto-optimal policy selection.

        This method selects the best trial from Optuna's Pareto-optimal trials
        using normalized scoring across all optimization objectives.

        Args:
            study: Completed Optuna multi-objective study
            optimization_time: Total time spent on optimization

        Returns:
            OptimizedPolicyResult: NamedTuple containing optimization results
        """
        self.logger.info("Total optimization time: %.4f seconds", optimization_time)

        # Get Pareto-optimal trials from Optuna
        pareto_trials = study.best_trials
        self.logger.info("Optuna identified %d Pareto-optimal trials", len(pareto_trials))

        # Compute Pareto scores only for the Pareto-optimal trials
        pareto_scores = self._compute_pareto_scores(study, pareto_trials)

        # Find trial with highest Pareto score among Pareto-optimal trials
        best_trial_num = max(pareto_scores.keys(), key=lambda k: pareto_scores[k])
        best_trial = study.trials[best_trial_num]
        best_score = pareto_scores[best_trial_num]

        self.logger.info("Best trial: %d with Pareto score: %.4f", best_trial_num, best_score)

        # Log individual metrics for best trial
        for param_name, direction in self.parameters_to_optimize:
            metric_key = f"metric_{param_name}"
            metric_value = best_trial.user_attrs.get(metric_key)
            self.logger.info("  %s (%s): %s", param_name, direction.value, metric_value)

        # Create the optimized policy with best parameters
        best_policy_params = {
            "environment": self.environment,
            "name": f"{self.policy_cls.__name__}_{self.environment.name}_optimized",
        }
        best_policy_params.update(self.constant_parameters)

        for param_name, param_value in best_trial.params.items():
            best_policy_params[param_name] = param_value

        optimized_policy = self.policy_cls(**best_policy_params)

        # Store metadata
        self._last_optimization_metadata = {
            "best_pareto_score": best_score,
            "best_trial_metrics": {
                param_name: best_trial.user_attrs.get(f"metric_{param_name}")
                for param_name, _ in self.parameters_to_optimize
            },
            "n_trials": self.n_trials,
            "optimization_time": optimization_time,
            "config_id": self.get_config_id(),
            "best_trial_number": best_trial_num,
            "best_trial_statistics": best_trial.user_attrs.get("statistics"),
            "all_pareto_scores": pareto_scores,
            "num_pareto_optimal_trials": len(pareto_trials),
        }

        # Extract actual metric values from best trial
        optimized_metric_values = {
            param_name: best_trial.user_attrs.get(f"metric_{param_name}")
            for param_name, _ in self.parameters_to_optimize
        }

        # Create result
        result = OptimizedPolicyResult(
            environment=self.environment,
            policy=optimized_policy,
            chosen_hyper_parameters=best_trial.params,
            num_episodes=self.num_episodes,
            num_steps=self.num_steps,
            parameters_to_optimize=self.parameters_to_optimize,
            optimized_metric_values=optimized_metric_values,
        )

        self._last_optimization_result = result
        return result

    def _handle_optimization_failure(self, exception: Exception, start_time: float) -> None:
        """Handle optimization failure by logging error details and cleanup.

        Args:
            exception: Exception that caused the failure
            start_time: Optimization start time for duration calculation

        Returns:
            None: Indicates optimization failure
        """
        self.logger.error("Hyperparameter optimization failed: %s", exception)
        self.logger.exception("Full exception details:")

        # Log total execution time even for failures
        optimization_time = time.time() - start_time
        self.logger.error("Failed optimization time: %.4f seconds", optimization_time)

        raise exception

    def get_optimization_metadata(self) -> Optional[dict]:
        """Get additional optimization metadata for backward compatibility.

        This method provides access to additional optimization details that are not
        part of the OptimizedPolicyResult but may be needed by existing code.

        Returns:
            dict: Dictionary containing additional metadata with keys:
                - 'best_value': Best objective value achieved
                - 'n_trials': Number of optimization trials conducted
                - 'optimization_time': Total time spent on optimization
                - 'config_id': Configuration ID for this optimization task
                - 'best_trial_number': Number of the best trial
                - 'best_trial_statistics': Statistics from the best trial
            None: If no optimization has been run yet
        """
        return getattr(self, "_last_optimization_metadata", None)

    def get_optimization_result_dict(self) -> Optional[dict]:
        """Get the complete optimization result as a dictionary for backward compatibility.

        This method combines the OptimizedPolicyResult with additional metadata
        to provide the same structure as the previous dictionary return type.

        Returns:
            dict: Complete optimization result dictionary with all fields from
                the previous return type, or None if no optimization has been run
        """
        metadata = self.get_optimization_metadata()
        if metadata is None:
            return None

        # Get the last optimization result
        if not hasattr(self, "_last_optimization_result"):
            return None

        result = self._last_optimization_result
        return {
            "best_params": result.chosen_hyper_parameters,
            "best_pareto_score": metadata["best_pareto_score"],
            "best_trial_metrics": metadata["best_trial_metrics"],
            "n_trials": metadata["n_trials"],
            "optimization_time": metadata["optimization_time"],
            "config_id": metadata["config_id"],
            "environment_name": result.environment.name,
            "policy_class_name": result.policy.__class__.__name__,
            "parameters_to_optimize": [
                (param_name, direction.value)
                for param_name, direction in result.parameters_to_optimize
            ],
            "optimized_metric_values": result.optimized_metric_values,
            "num_episodes": result.num_episodes,
            "num_steps": result.num_steps,
            "best_trial_number": metadata["best_trial_number"],
            "best_trial_statistics": metadata["best_trial_statistics"],
            "all_pareto_scores": metadata.get("all_pareto_scores"),
            "seed": self.seed,
        }

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
        # Validate inputs first
        if not isinstance(environment, Environment):
            raise TypeError(f"environment must be an Environment instance, got {type(environment)}")
        if not isinstance(policy, Policy):
            raise TypeError(f"policy must be a Policy instance, got {type(policy)}")
        if not isinstance(initial_belief, Belief):
            raise TypeError(f"initial_belief must be a Belief instance, got {type(initial_belief)}")
        if scheduler_address is not None and not isinstance(scheduler_address, str):
            raise TypeError(
                f"scheduler_address must be a string or None, got {type(scheduler_address)}"
            )

        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        # Log after validation
        self.logger.info("Starting %d episodes with %d steps each", num_episodes, num_steps)
        self.logger.info("Environment: %s, Policy: %s", environment.name, policy.name)

        # Create EnvironmentRunParams for the simulator
        env_run_params = [
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps,
            )
        ]

        # Use simulator's direct parallel execution method to avoid MLflow experiment creation
        results = self.simulator.simulate_multiple_environments_and_policies_parallel(
            environment_run_params=env_run_params,
            alpha=self.alpha,  # Default alpha for intermediate results
            confidence_interval_level=self.confidence_interval_level,
            n_jobs=1,
        )

        # Extract histories from results
        histories = results[environment.name][policy.name]
        self.logger.info("All episodes completed for %s with %s", environment.name, policy.name)

        return histories

    def get_config_id(self) -> str:
        return config_to_id(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "environment": self.environment.config_id,
            "belief": self.belief.config_id,
            "policy_cls": str(self.policy_cls),
            "hyper_parameters": self.hyper_parameters,
            "constant_parameters": self.constant_parameters,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "parameters_to_optimize": [
                (param_name, direction.value)
                for param_name, direction in self.parameters_to_optimize
            ],
            "n_trials": self.n_trials,
            "seed": self.seed,
            "use_queue_logger": self.use_queue_logger,
        }

    def __eq__(self, other: object) -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, HyperParameterTuningSimulationTask):
            return False
        return self.get_config_id() == other.get_config_id()

    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self.get_config_id())
