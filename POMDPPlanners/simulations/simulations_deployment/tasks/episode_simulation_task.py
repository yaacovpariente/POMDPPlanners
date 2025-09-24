import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from POMDPPlanners.core.simulation import History, SimulationTask
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger


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
        Raises:
            ValueError: If num_steps is not positive
        """
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")

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
        # Generate cache key after all attributes are set
        self._cache_key = self._generate_cache_key()

        # Log task creation using the logger property
        self.logger.debug(
            f"Creating EpisodeSimulationTask with episode_id={self.episode_id}, episode_number={self.episode_number}"
        )
        self.logger.debug(
            f"Task parameters: num_steps={self.num_steps}, seed={self.seed}, discount_factor={self.discount_factor}"
        )
        self.logger.debug(f"Cache directory: {self.cache_dir}")
        self.logger.debug(f"Generated cache key: {self._cache_key}")

    @property
    def logger(self) -> logging.Logger:
        """Get shared logger for this environment-policy combination.

        Uses one logger per environment-policy pair instead of per task for better
        performance and resource management. All episodes using the same environment
        and policy will share the same log file.
        """
        output_dir = None
        if self.cache_dir is not None:
            # Note: get_logger automatically creates a 'logs' subdirectory,
            # so we only need to specify the parent directory
            output_dir = self.cache_dir / "env_policy"

        return get_logger(
            name=self._get_env_policy_logger_name(),
            debug=self.debug,
            output_dir=output_dir,
            console_output=self.console_output,
            use_queue=self.use_queue_logger,
        )

    def _get_env_policy_logger_name(self) -> str:
        """Get the shared logger name for this environment-policy combination."""
        return f"env_policy.{self.environment.name}.{self.policy.name}"

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

    def cleanup_logger(self) -> None:
        """Clean up logger resources to prevent file handle leaks.

        Note: With shared env-policy loggers, cleanup is less critical since
        multiple tasks share the same logger. This method is kept for compatibility
        but now only flushes handlers instead of closing them to avoid affecting
        other tasks using the same logger.
        """
        try:
            logger = logging.getLogger(self._get_env_policy_logger_name())
            # Flush handlers instead of closing them (other tasks may still be using them)
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()
        except Exception as e:
            # Don't let cleanup errors affect the main task
            pass

    def run(self) -> Union[History, None]:
        """Run the simulation task.

        Returns:
            History: The simulation history
        """
        # Start timing
        start_time = time.time()

        # Create episode context for structured logging
        episode_context = {
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "num_steps": self.num_steps,
            "seed": self.seed,
            "discount_factor": self.discount_factor,
            "cache_key": self._cache_key,
        }

        # Use structured logging with episode context
        self.logger.info(
            f"[EPISODE_{self.episode_id:03d}] Starting episode simulation "
            f"(num={self.episode_number}, steps={self.num_steps}, seed={self.seed})"
        )

        # Log detailed configuration in debug mode
        if self.debug:
            self.logger.debug(
                f"[EPISODE_{self.episode_id:03d}] Environment: {self.environment.name} (config_id: {self.environment.config_id})"
            )
            self.logger.debug(
                f"[EPISODE_{self.episode_id:03d}] Policy: {self.policy.name} (config_id: {self.policy.config_id})"
            )
            self.logger.debug(
                f"[EPISODE_{self.episode_id:03d}] Initial belief: {self.initial_belief.config_id}"
            )
            self.logger.debug(
                f"[EPISODE_{self.episode_id:03d}] Configuration: discount_factor={self.discount_factor}"
            )
            self.logger.debug(
                f"[EPISODE_{self.episode_id:03d}] Debug mode: {self.debug}, Console output: {self.console_output}"
            )
            if self.cache_dir:
                self.logger.debug(
                    f"[EPISODE_{self.episode_id:03d}] Cache directory: {self.cache_dir}"
                )
            self.logger.debug(f"[EPISODE_{self.episode_id:03d}] Cache key: {self._cache_key}")

        # Log random state setup
        self.logger.debug(f"[EPISODE_{self.episode_id:03d}] Setting random seed to {self.seed}")
        state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)

        try:
            # Run simulation with seed parameter
            self.logger.debug(f"[EPISODE_{self.episode_id:03d}] Starting episode simulation...")
            result = run_episode(
                environment=self.environment,
                policy=self.policy,
                initial_belief=self.initial_belief,
                num_steps=self.num_steps,
                logger=self.logger,
            )

            if result is not None:
                # Calculate total reward from history
                total_reward = 0
                actual_steps = 0
                reached_terminal = False

                if hasattr(result, "history") and result.history:
                    actual_steps = len(result.history)
                    total_reward = sum(
                        step.reward for step in result.history if step.reward is not None
                    )

                if hasattr(result, "reach_terminal_state"):
                    reached_terminal = result.reach_terminal_state

                if hasattr(result, "actual_num_steps"):
                    actual_steps = result.actual_num_steps

                # Structured success log with key metrics
                self.logger.info(
                    f"[EPISODE_{self.episode_id:03d}] Completed successfully "
                    f"(reward={total_reward:.4f}, steps={actual_steps}, terminal={reached_terminal})"
                )

                if self.debug:
                    self.logger.debug(
                        f"[EPISODE_{self.episode_id:03d}] Result type: {type(result)}"
                    )
            else:
                self.logger.warning(f"[EPISODE_{self.episode_id:03d}] Episode returned None result")

        except Exception as e:
            self.logger.error(f"[EPISODE_{self.episode_id:03d}] Error running episode: {e}")
            self.logger.exception(f"[EPISODE_{self.episode_id:03d}] Full exception details:")
            result = None
        finally:
            # Restore random state
            self.logger.debug(f"[EPISODE_{self.episode_id:03d}] Restoring random state")
            np.random.set_state(state)
            self.logger.debug(f"[EPISODE_{self.episode_id:03d}] Random state restored")

            # Log total execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.info(
                f"[EPISODE_{self.episode_id:03d}] Execution completed in {execution_time:.4f} seconds"
            )

            # Clean up logger resources (flush only, don't close shared logger)
            self.cleanup_logger()

        return result

    def __eq__(self, other: object) -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, EpisodeSimulationTask):
            return False
        return self._cache_key == other._cache_key

    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self._cache_key)
