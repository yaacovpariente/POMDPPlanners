# SPDX-License-Identifier: MIT

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from POMDPPlanners.utils.logger import get_logger


class SimulationTask(ABC):
    """Abstract base class for simulation tasks.

    This class defines the interface that all simulation tasks must implement.
    A simulation task represents a unit of work that can be executed and cached.

    Examples:
        >>> class MySimulationTask(SimulationTask):
        ...     def __init__(self, config_id):
        ...         self.config_id = config_id
        ...
        ...     def run(self):
        ...         return f"Result for {self.config_id}"
        ...
        ...     def get_config_id(self):
        ...         return self.config_id
        >>>
        >>> task = MySimulationTask("test_config")
        >>> task.get_config_id()
        'test_config'
        >>> task.run()
        'Result for test_config'
    """

    @abstractmethod
    def run(self) -> Any:
        """Execute the simulation task.

        Returns:
            Any: The result of the simulation task
        """

    @abstractmethod
    def get_config_id(self) -> str:
        """Get a unique identifier for this task's configuration.

        Returns:
            str: Unique configuration identifier for caching
        """


class DataBaseInterface(ABC):
    """Abstract interface for database operations used by task managers.

    This class defines the interface for caching simulation results,
    allowing different database implementations to be used interchangeably.

    Examples:
        >>> class MockDatabase(DataBaseInterface):
        ...     def __init__(self):
        ...         self.data = {}
        ...
        ...     def get(self, key):
        ...         return self.data.get(key)
        ...
        ...     def is_key_in_cache(self, key):
        ...         return key in self.data
        ...
        ...     def set(self, key, value):
        ...         self.data[key] = value
        ...
        ...     def clear(self):
        ...         self.data.clear()
        >>>
        >>> db = MockDatabase()
        >>> db.set("test_key", "test_value")
        >>> db.is_key_in_cache("test_key")
        True
        >>> db.get("test_key")
        'test_value'
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a value from the database.

        Args:
            key: The key to retrieve

        Returns:
            Any: The stored value
        """

    @abstractmethod
    def is_key_in_cache(self, key: str) -> bool:
        """Check if a key exists in the database.

        Args:
            key: The key to check

        Returns:
            bool: True if key exists, False otherwise
        """

    @abstractmethod
    def set(self, key: str, value: Any):
        """Store a value in the database.

        Args:
            key: The key to store under
            value: The value to store
        """

    @abstractmethod
    def clear(self):
        """Clear all data from the database."""


class TaskManager(ABC):
    """Abstract base class for task managers.

    Task managers coordinate the execution of simulation tasks,
    handling caching, parallelization, and result collection.

    Examples:
        >>> class SimpleTaskManager(TaskManager):
        ...     def run_tasks(self, tasks, task_identifiers):
        ...         results = []
        ...         identifiers = []
        ...         for task, identifier in zip(tasks, task_identifiers):
        ...             result = task.run()
        ...             results.append(result)
        ...             identifiers.append(identifier)
        ...         return results, identifiers
        >>>
        >>> class MyTask(SimulationTask):
        ...     def run(self): return "result"
        ...     def get_config_id(self): return "config"
        >>>
        >>> manager = SimpleTaskManager()
        >>> tasks = [MyTask()]
        >>> identifiers = ["task1"]
        >>> results, ids = manager.run_tasks(tasks, identifiers)
        >>> results[0]
        'result'
        >>> ids[0]
        'task1'
    """

    @abstractmethod
    def run_tasks(
        self, tasks: List[SimulationTask], task_identifiers: list
    ) -> Tuple[List[Any], list]:
        """Execute a list of simulation tasks.

        Args:
            tasks: List of simulation tasks to execute
            task_identifiers: List of identifiers for each task

        Returns:
            Tuple[List[Any], list]: Results and successful task identifiers
        """

    def set_progress_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a callback fired once per completed task.

        Subclasses that support per-task progress reporting (e.g.
        :class:`JoblibTaskManager`) override this to store the callback and
        invoke it from inside their parallel execution loop. The default
        implementation is a no-op so callers may invoke it unconditionally.

        Args:
            callback: A zero-argument callable invoked in the parent process
                after each task completes, or ``None`` to clear a previously
                set callback. Exceptions raised by the callable are caught
                and ignored by implementing task managers.
        """

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""


class TaskManagerExternalDB(TaskManager):
    """Task manager that uses an external database for caching.

    This task manager implements caching functionality using an external database
    interface, allowing simulation results to be cached and reused across runs.

    Attributes:
        cache_db: Database interface for caching results
        cache_dir: Optional directory for logging and cache files
        logger_debug: Whether to enable debug logging
        use_queue_logger: Whether to use queue-based logging

    Examples:
        >>> class MockDatabase(DataBaseInterface):
        ...     def __init__(self):
        ...         self.data = {}
        ...     def get(self, key): return self.data.get(key)
        ...     def is_key_in_cache(self, key): return key in self.data
        ...     def set(self, key, value): self.data[key] = value
        ...     def clear(self): self.data.clear()
        >>>
        >>> class MockTaskManager(TaskManagerExternalDB):
        ...     def _run_tasks(self, tasks):
        ...         return [task.run() for task in tasks]
        >>>
        >>> class MyTask(SimulationTask):
        ...     def run(self): return "cached_result"
        ...     def get_config_id(self): return "test_config"
        >>>
        >>> db = MockDatabase()
        >>> manager = MockTaskManager(db)
        >>> tasks = [MyTask()]
        >>> identifiers = ["task1"]
        >>> results, ids = manager.run_tasks(tasks, identifiers)
        >>> results[0]
        'cached_result'
        >>> db.is_key_in_cache("test_config")
        True
    """

    def __init__(
        self,
        cache_db: DataBaseInterface,
        cache_dir: Optional[Path] = None,
        logger_debug: bool = False,
        use_queue_logger: bool = False,
        console_output: bool = True,
        no_logs: bool = False,
    ):
        """Initialize the task manager with caching database.

        Args:
            cache_db: Database interface for caching results
            cache_dir: Optional directory for logging and cache files
            logger_debug: Whether to enable debug logging
            use_queue_logger: Whether to use queue-based logging
            console_output: Whether to print logs to console
            no_logs: Whether to disable all logging
        """
        self.cache_db = cache_db
        self.cache_dir = cache_dir
        self.logger_debug = logger_debug
        self.console_output = console_output
        self.use_queue_logger = use_queue_logger
        self.no_logs = no_logs

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance for this task manager.

        Returns:
            logging.Logger: Configured logger instance
        """
        return get_logger(
            name="task_manager",
            debug=self.logger_debug,
            output_dir=self.cache_dir if not self.no_logs else None,
            use_queue=self.use_queue_logger,
            console_output=self.console_output if not self.no_logs else False,
        )

    @abstractmethod
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        """Execute a list of tasks (to be implemented by subclasses).

        Args:
            tasks: List of simulation tasks to execute

        Returns:
            List[Any]: Results from executing the tasks
        """

    def _cached_result_is_usable(self, task: SimulationTask, result: Any, task_id: str) -> bool:
        """Decide whether a cached result can still be scored, or must be redone.

        The cache key is the task's configuration, which says nothing about the
        format of the episode it produced. An entry written before
        :attr:`~POMDPPlanners.core.simulation.history.StepData.info` existed
        therefore still hits, and unpickles cleanly with every measurement
        missing -- which, for an environment deriving its metrics from that
        channel, yields a full set of zero-valued metrics rather than an obvious
        failure.

        Such an entry is reported as unusable so the caller reruns that one task.
        Only the stale entries are redone; everything recorded in the current
        format is still reused, so a resumed run keeps its recovery.
        """
        environment = getattr(task, "environment", None)
        if environment is None or not hasattr(environment, "get_metric_specs"):
            return True
        if not hasattr(result, "history"):
            return True
        # Imported here rather than at module scope: core.simulation.__init__
        # imports this module, so a top-level import would close a cycle.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.core.simulation.step_info_metrics import unmeasured_episode_index

        if unmeasured_episode_index([result], environment.get_metric_specs()) is None:
            return True

        self.logger.warning(
            "Cached result for task %s predates the per-step measurement channel: it carries "
            "none of %s's channels, so its metrics would all read zero. Rerunning this task "
            "and replacing the cache entry.",
            task_id,
            type(environment).__name__,
        )
        return False

    def run_tasks(
        self, tasks: List[SimulationTask], task_identifiers: list
    ) -> Tuple[List[Any], list]:
        """Execute tasks with caching support.

        This method checks the cache for existing results before executing tasks,
        runs only uncached tasks, and stores new results in the cache.

        Args:
            tasks: List of simulation tasks to execute
            task_identifiers: List of identifiers for each task

        Returns:
            Tuple[List[Any], list]: Results and successful task identifiers
        """
        self.logger.info("Starting to process %s tasks", len(tasks))
        # Lists to store results and track which tasks need to be run
        results: List[Any] = [None] * len(tasks)
        tasks_to_run: List[SimulationTask] = []
        task_indices: List[int] = []  # Keep track of original indices for uncached tasks

        # First pass: check cache and collect tasks that need to be run
        cached_tasks = 0
        for i, task in enumerate(tasks):
            task_id = task.get_config_id()
            cached = self.cache_db.get(task_id) if self.cache_db.is_key_in_cache(task_id) else None
            if cached is not None and self._cached_result_is_usable(task, cached, task_id):
                results[i] = cached
                cached_tasks += 1
            else:
                tasks_to_run.append(task)
                task_indices.append(i)

        self.logger.info(
            "Cache status: %s tasks cached, %s tasks uncached out of %s total tasks",
            cached_tasks,
            len(tasks_to_run),
            len(tasks),
        )

        # Run only the tasks that weren't in cache
        if tasks_to_run:
            self.logger.info("Running %s uncached tasks", len(tasks_to_run))
            new_results = self._run_tasks(tasks_to_run)
            self.logger.info("Completed %s tasks", len(new_results))

            if len(new_results) != len(tasks_to_run):
                raise ValueError("new_results and tasks_to_run must have the same length")

            # Store new results in their original positions
            for idx, result in zip(task_indices, new_results):
                if result is None:  # prevents storing failed tasks
                    continue

                results[idx] = result
                # Cache the new result
                task_id = tasks[idx].get_config_id()
                self.logger.debug("Storing task %s in cache with config_id: %s", idx, task_id)
                self.cache_db.set(task_id, result)

        # Filter out failed tasks and their identifiers
        successful_results = []
        successful_identifiers = []
        for i, (result, identifier) in enumerate(zip(results, task_identifiers)):
            if result is not None:
                successful_results.append(result)
                successful_identifiers.append(identifier)
            else:
                task_id = tasks[i].get_config_id()
                self.logger.warning(
                    "Task %s (config_id: %s) failed - returned None result", i, task_id
                )

        n_failed_tasks = len(tasks) - len(successful_results)
        self.logger.info("%s tasks completed successfully", len(successful_results))

        if n_failed_tasks > 0:
            self.logger.warning("%s tasks failed.", n_failed_tasks)

        return successful_results, successful_identifiers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
