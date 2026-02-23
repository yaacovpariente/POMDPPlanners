import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from POMDPPlanners.core.simulation import History, SimulationTask
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import (
    setup_task_logger_with_buffering,
    flush_buffered_task_logs,
    cleanup_task_logger,
)


class EpisodeSimulationTask(SimulationTask):
    """A class to represent a single simulation task with caching capabilities."""

    def __init__(
        self,
        environment: Any,  # Use Any to avoid circular import
        policy: Any,
        initial_belief: Any,
        num_steps: int,
        episode_id: int,
        seed: int,
        discount_factor: float = 1.0,
        episode_number: int = 0,
        cache_dir: Optional[Path] = None,
        debug: bool = False,
        console_output: bool = True,
        use_queue_logger: bool = False,
        log_only_on_failure: bool = True,
    ):
        """Initialize a simulation task.

        Args:
            environment: The environment to simulate
            policy: The policy to use
            initial_belief: The initial belief state
            num_steps: Number of steps to simulate
            episode_id: Unique identifier for this episode
            seed: Random seed for reproducibility
            discount_factor: Discount factor for reward calculation
            episode_number: The episode number for this simulation
            cache_dir: Directory for caching results
            debug: Whether to enable debug logging
            console_output: Whether to enable console output (default: True).
                          Set to False to disable console output while keeping file logging.
            use_queue_logger: Whether to use queue-based logging. Defaults to False.
                With the environment-policy pair logging design, multiple tasks
                using the same environment and policy share the same logger.
            log_only_on_failure: Whether to buffer logs and only write on failure (default: True).
                When enabled, all logs are buffered in memory and only written to disk/console
                if the episode fails or encounters an error. This dramatically reduces I/O
                overhead for successful episodes.
        Raises:
            ValueError: If any input parameter is invalid
            TypeError: If any input parameter has incorrect type
        """
        self._validate_inputs(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=num_steps,
            episode_id=episode_id,
            seed=seed,
            discount_factor=discount_factor,
            episode_number=episode_number,
            cache_dir=cache_dir,
            debug=debug,
            console_output=console_output,
            use_queue_logger=use_queue_logger,
            log_only_on_failure=log_only_on_failure,
        )

        self.environment = environment
        self.policy = policy
        self.initial_belief = initial_belief
        self.num_steps = num_steps
        self.episode_id = episode_id
        self.seed = seed
        self.discount_factor = discount_factor
        self.episode_number = episode_number
        self.debug = debug
        self.console_output = console_output
        self.cache_dir = cache_dir
        self.use_queue_logger = use_queue_logger
        self.log_only_on_failure = log_only_on_failure
        # Cache logger name to avoid issues after cleanup sets environment/policy to None
        self._env_policy_logger_name = f"env_policy.{self.environment.name}.{self.policy.name}"
        # Generate cache key after all attributes are set
        self._cache_key = self._generate_cache_key()

        # Log task creation using the logger property
        self.logger.debug(
            "Creating EpisodeSimulationTask with episode_id=%s, episode_number=%s",
            self.episode_id,
            self.episode_number,
        )
        self.logger.debug(
            "Task parameters: num_steps=%s, seed=%s, discount_factor=%s",
            self.num_steps,
            self.seed,
            self.discount_factor,
        )
        self.logger.debug("Cache directory: %s", self.cache_dir)
        self.logger.debug("Generated cache key: %s", self._cache_key)

    @property
    def logger(self) -> logging.Logger:
        """Get shared logger for this environment-policy combination.

        Uses one logger per environment-policy pair instead of per task for better
        performance and resource management. All episodes using the same environment
        and policy will share the same log file.

        When log_only_on_failure is enabled, wraps handlers with ConditionalMemoryHandler
        to buffer logs in memory and only flush to disk/console on failure.
        """
        logger_name = self._get_env_policy_logger_name()

        # Configure output directory
        output_dir = None
        if self.cache_dir is not None:
            output_dir = self.cache_dir / "env_policy"

        # Use helper function to set up logger with buffering
        return setup_task_logger_with_buffering(
            logger_name=logger_name,
            output_dir=output_dir,
            debug=self.debug,
            console_output=self.console_output,
            use_queue=self.use_queue_logger,
            log_only_on_failure=self.log_only_on_failure,
        )

    @staticmethod
    def _validate_inputs(  # pylint: disable=too-many-branches
        environment: Any,
        policy: Any,
        initial_belief: Any,
        num_steps: int,
        episode_id: int,
        seed: int,
        discount_factor: float,
        episode_number: int,
        cache_dir: Optional[Path],
        debug: bool,
        console_output: bool,
        use_queue_logger: bool,
        log_only_on_failure: bool,
    ) -> None:
        """Validate input parameters for EpisodeSimulationTask.

        Args:
            environment: The environment to simulate
            policy: The policy to use
            initial_belief: The initial belief state
            num_steps: Number of steps to simulate
            episode_id: Unique identifier for this episode
            seed: Random seed for reproducibility
            discount_factor: Discount factor for reward calculation
            episode_number: The episode number for this simulation
            cache_dir: Directory for caching results
            debug: Whether to enable debug logging
            console_output: Whether to enable console output
            use_queue_logger: Whether to use queue-based logging
            log_only_on_failure: Whether to buffer logs and only write on failure

        Raises:
            ValueError: If any input parameter is invalid
            TypeError: If any input parameter has incorrect type
        """
        # Validate required objects
        if environment is None:
            raise ValueError("environment cannot be None")
        if policy is None:
            raise ValueError("policy cannot be None")
        if initial_belief is None:
            raise ValueError("initial_belief cannot be None")

        # Validate integer parameters
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id must be a non-negative integer")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(episode_number, int) or episode_number < 0:
            raise ValueError("episode_number must be a non-negative integer")

        # Validate discount_factor
        if not isinstance(discount_factor, (int, float)):
            raise TypeError("discount_factor must be a number")
        if discount_factor < 0 or discount_factor > 1:
            raise ValueError("discount_factor must be between 0 and 1")

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
        if not isinstance(log_only_on_failure, bool):
            raise TypeError("log_only_on_failure must be a boolean")

    def _get_env_policy_logger_name(self) -> str:
        """Get the shared logger name for this environment-policy combination."""
        return self._env_policy_logger_name

    def _get_logger_name(self) -> str:
        """Get the name of the logger for this task (legacy method for compatibility)."""
        return self._get_env_policy_logger_name()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this task."""

        components = {
            "env": self.environment.config_id,
            "policy": self.policy.config_id,
            "belief": self.initial_belief.config_id,
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "num_steps": self.num_steps,
            "seed": self.seed,
            "discount_factor": self.discount_factor,
            "use_queue_logger": self.use_queue_logger,
        }
        return config_to_id(components)

    def get_config_id(self) -> str:
        """Get the configuration ID for this task.

        Returns:
            str: The cache key used to identify this task
        """
        return self._cache_key

    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization."""
        return {
            "environment": self.environment,
            "policy": self.policy,
            "initial_belief": self.initial_belief,
            "num_steps": self.num_steps,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "discount_factor": self.discount_factor,
            "episode_number": self.episode_number,
            "use_queue_logger": self.use_queue_logger,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSimulationTask":
        """Create a SimulationTask instance from a dictionary."""
        return cls(**data)

    def run(self) -> Union[History, None]:
        """Run the simulation task.

        Returns:
            History: The simulation history or None if execution fails.
        """
        start_time = time.time()
        episode_failed = False

        self._log_episode_start()
        self._log_debug_configuration()

        random_state = self._setup_random_seed()

        try:
            result = self._execute_episode()
            self._log_episode_completion(result)

            # Check if result indicates failure (None result)
            if result is None:
                episode_failed = True
                self._flush_buffered_logs()

        except Exception as e:  # pylint: disable=broad-exception-caught
            episode_failed = True
            self._log_episode_error(e)
            self._flush_buffered_logs()
            result = None
        finally:
            self._restore_random_state(random_state)
            self._log_execution_time(start_time)
            self.cleanup(is_final_task=True, episode_failed=episode_failed)
        return result

    def _flush_buffered_logs(self) -> None:
        """Flush buffered logs to file/console when failure occurs.

        This method triggers the flush of all buffered log records when
        log_only_on_failure is enabled and a failure is detected.
        """
        flush_buffered_task_logs(self._get_env_policy_logger_name())

    def cleanup_logger(self, episode_failed: bool = False) -> None:
        """Clean up logger resources to prevent file handle leaks.

        Note: With shared env-policy loggers, cleanup is less critical since
        multiple tasks share the same logger. This method is kept for compatibility
        but now only flushes handlers instead of closing them to avoid affecting
        other tasks using the same logger.

        Args:
            episode_failed: Whether the episode failed. If False and log_only_on_failure
                          is enabled, buffered logs will be discarded instead of flushed.
        """
        cleanup_task_logger(
            logger_name=self._get_env_policy_logger_name(),
            episode_failed=episode_failed,
            log_only_on_failure=self.log_only_on_failure,
        )

    def cleanup(self, is_final_task: bool = False, episode_failed: bool = False) -> None:
        """Comprehensive cleanup to prevent memory leaks.

        Args:
            is_final_task: If True and this is the last task using the shared logger,
                          close logger handlers instead of just flushing them.
            episode_failed: Whether the episode failed. Passed to cleanup_logger.
        """
        try:
            # Step 1: Clean up logger (flush or discard buffered logs based on episode result)
            self.cleanup_logger(episode_failed=episode_failed)

            # Step 2: Close handlers only if this is the final task
            if is_final_task:
                logger = logging.getLogger(self._get_env_policy_logger_name())
                for handler in logger.handlers[
                    :
                ]:  # Copy list to avoid modification during iteration
                    try:
                        handler.close()
                        logger.removeHandler(handler)
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass

            # Step 3: Clear references to large objects
            self.environment = None
            self.policy = None
            self.initial_belief = None
            # Note: Don't clear _cache_key as get_config_id() must return str

        except Exception:  # pylint: disable=broad-exception-caught
            # Don't let cleanup errors affect the main task
            pass

    def _log_episode_start(self) -> None:
        """Log episode start information."""
        self.logger.info(
            "[EPISODE_%03d] Starting episode simulation (num=%s, steps=%s, seed=%s)",
            self.episode_id,
            self.episode_number,
            self.num_steps,
            self.seed,
        )

    def _log_debug_configuration(self) -> None:
        """Log detailed configuration in debug mode."""
        if not self.debug:
            return

        self.logger.debug(
            "[EPISODE_%03d] Environment: %s (config_id: %s)",
            self.episode_id,
            self.environment.name,
            self.environment.config_id,
        )
        self.logger.debug(
            "[EPISODE_%03d] Policy: %s (config_id: %s)",
            self.episode_id,
            self.policy.name,
            self.policy.config_id,
        )
        self.logger.debug(
            "[EPISODE_%03d] Initial belief: %s",
            self.episode_id,
            self.initial_belief.config_id,
        )
        self.logger.debug(
            "[EPISODE_%03d] Configuration: discount_factor=%s",
            self.episode_id,
            self.discount_factor,
        )
        self.logger.debug(
            "[EPISODE_%03d] Debug mode: %s, Console output: %s",
            self.episode_id,
            self.debug,
            self.console_output,
        )
        if self.cache_dir:
            self.logger.debug(
                "[EPISODE_%03d] Cache directory: %s",
                self.episode_id,
                self.cache_dir,
            )
        self.logger.debug(
            "[EPISODE_%03d] Cache key: %s",
            self.episode_id,
            self._cache_key,
        )

    def _setup_random_seed(self) -> Any:
        """Set random seed and return previous state for restoration.

        Returns:
            Previous numpy random state for later restoration.
        """
        self.logger.debug(
            "[EPISODE_%03d] Setting random seed to %s",
            self.episode_id,
            self.seed,
        )
        state = np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed)
        return state

    def _restore_random_state(self, state: Any) -> None:
        """Restore numpy random state.

        Args:
            state: Previously saved numpy random state.
        """
        self.logger.debug(
            "[EPISODE_%03d] Restoring random state",
            self.episode_id,
        )
        np.random.set_state(state)
        self.logger.debug(
            "[EPISODE_%03d] Random state restored",
            self.episode_id,
        )

    def _execute_episode(self) -> Union[History, None]:
        """Execute the episode simulation.

        Returns:
            Simulation history or None if execution fails.
        """
        self.logger.debug(
            "[EPISODE_%03d] Starting episode simulation...",
            self.episode_id,
        )

        # Ensure attributes are not None before executing episode
        if self.environment is None or self.policy is None or self.initial_belief is None:
            raise RuntimeError("Cannot execute episode: task has been cleaned up")

        return run_episode(
            environment=self.environment,
            policy=self.policy,
            initial_belief=self.initial_belief,
            num_steps=self.num_steps,
            logger=self.logger,
        )

    def _extract_episode_metrics(self, result: History) -> tuple[float, int, bool]:
        """Extract metrics from episode result.

        Args:
            result: Episode simulation result.

        Returns:
            Tuple of (total_reward, actual_steps, reached_terminal).
        """
        total_reward = 0
        actual_steps = 0
        reached_terminal = False

        if hasattr(result, "history") and result.history:
            actual_steps = len(result.history)
            total_reward = sum(
                float(step.reward) for step in result.history if step.reward is not None
            )

        if hasattr(result, "reach_terminal_state"):
            reached_terminal = result.reach_terminal_state

        if hasattr(result, "actual_num_steps"):
            actual_steps = result.actual_num_steps

        return total_reward, actual_steps, reached_terminal

    def _log_episode_completion(self, result: Union[History, None]) -> None:
        """Log episode completion with metrics.

        Args:
            result: Episode simulation result or None.
        """
        if result is not None:
            total_reward, actual_steps, reached_terminal = self._extract_episode_metrics(result)

            self.logger.info(
                "[EPISODE_%03d] Completed successfully (reward=%.4f, steps=%s, terminal=%s)",
                self.episode_id,
                total_reward,
                actual_steps,
                reached_terminal,
            )

            if self.debug:
                self.logger.debug(
                    "[EPISODE_%03d] Result type: %s",
                    self.episode_id,
                    type(result),
                )
        else:
            self.logger.warning(
                "[EPISODE_%03d] Episode returned None result",
                self.episode_id,
            )

    def _log_episode_error(self, error: Exception) -> None:
        """Log episode execution error.

        Args:
            error: Exception that occurred during execution.
        """
        self.logger.error(
            "[EPISODE_%03d] Error running episode: %s",
            self.episode_id,
            error,
        )
        self.logger.exception(
            "[EPISODE_%03d] Full exception details:",
            self.episode_id,
        )

    def _log_execution_time(self, start_time: float) -> None:
        """Log total execution time.

        Args:
            start_time: Episode start timestamp.
        """
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(
            "[EPISODE_%03d] Execution completed in %.4f seconds",
            self.episode_id,
            execution_time,
        )

    def __eq__(self, other: object) -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, EpisodeSimulationTask):
            return False
        return self._cache_key == other._cache_key

    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self._cache_key)

    def __getstate__(self):
        """Prepare task state for pickling by excluding unpicklable attributes.

        Returns:
            dict: Serializable state dictionary with logger excluded
        """
        state = self.__dict__.copy()
        # Logger is recreated via @property, no need to pickle it
        # Remove it to avoid pickling issues with file handles and locks
        state.pop("logger", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore task state after unpickling.

        Args:
            state: State dictionary from pickle
        """
        vars(self).update(state)
        # Logger will be recreated via @property when first accessed
