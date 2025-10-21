import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
        pass

    @abstractmethod
    def get_config_id(self) -> str:
        """Get a unique identifier for this task's configuration.

        Returns:
            str: Unique configuration identifier for caching
        """
        pass


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
        pass

    @abstractmethod
    def is_key_in_cache(self, key: str) -> bool:
        """Check if a key exists in the database.

        Args:
            key: The key to check

        Returns:
            bool: True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        """Store a value in the database.

        Args:
            key: The key to store under
            value: The value to store
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all data from the database."""
        pass


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
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


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
    ):
        """Initialize the task manager with caching database.

        Args:
            cache_db: Database interface for caching results
            cache_dir: Optional directory for logging and cache files
            logger_debug: Whether to enable debug logging
            use_queue_logger: Whether to use queue-based logging
        """
        self.cache_db = cache_db
        self.cache_dir = cache_dir
        self.logger_debug = logger_debug
        self.use_queue_logger = use_queue_logger

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance for this task manager.

        Returns:
            logging.Logger: Configured logger instance
        """
        return get_logger(
            name=f"task_manager",
            debug=self.logger_debug,
            output_dir=self.cache_dir,
            use_queue=self.use_queue_logger,
        )

    @abstractmethod
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        """Execute a list of tasks (to be implemented by subclasses).

        Args:
            tasks: List of simulation tasks to execute

        Returns:
            List[Any]: Results from executing the tasks
        """
        pass

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
            ...     def run(self): return "result"
            ...     def get_config_id(self): return "config1"
            >>>
            >>> db = MockDatabase()
            >>> manager = MockTaskManager(db)
            >>> tasks = [MyTask()]
            >>> identifiers = ["task1"]
            >>> results, ids = manager.run_tasks(tasks, identifiers)
            >>> len(results)
            1
            >>> results[0]
            'result'
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
            if self.cache_db.is_key_in_cache(task_id):
                results[i] = self.cache_db.get(task_id)
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
        pass
