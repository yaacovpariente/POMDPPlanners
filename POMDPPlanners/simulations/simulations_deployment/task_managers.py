import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dask.cache import Cache
from dask.distributed import Client, Future, LocalCluster
from dask_jobqueue.pbs import PBSCluster
from joblib import Memory, Parallel, delayed
from tqdm import tqdm

from POMDPPlanners.core.simulation import (
    DataBaseInterface,
    History,
    SimulationTask,
    TaskManager,
    TaskManagerExternalDB,
)


class TaskManagerType(Enum):
    """Enum representing different types of task managers for simulation execution."""

    DASK = "dask"  # Uses DaskTaskManager for distributed computing
    JOBLIB = "joblib"  # Uses JoblibTaskManager for parallel processing with caching
    PBS = "pbs"  # Uses PBSTaskManager for PBS cluster computing


class DaskTaskManager(TaskManager):
    """A class to manage simulation tasks and their execution using Dask."""

    def __init__(
        self,
        n_workers: Optional[int] = None,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),  # 2GB default cache size
        clear_cache_on_start: bool = False,
    ):
        """Initialize the task manager.

        Args:
            n_workers: Number of worker processes (None for auto)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_size: Size of cache in bytes
            clear_cache_on_start: If True, clears the cache at startup
        """
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers
        self.cache_size = cache_size
        self.clear_cache_on_start = clear_cache_on_start
        self.client: Optional[Client] = None
        self.cache = None
        self.cache_registered = False  # Track if cache was registered
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Dask client and cache with persistence."""
        if self.scheduler_address is not None:
            # Connect to existing cluster; if connection fails, raise a RuntimeError.
            try:
                self.client = Client(self.scheduler_address, timeout=0.1)
            except Exception as e:
                raise RuntimeError(
                    "Failed to connect to Dask scheduler at {}. Error: {}".format(
                        self.scheduler_address, e
                    )
                )
        else:
            # Create local cluster with persistent storage
            cluster = LocalCluster(n_workers=self.n_workers, local_directory="./dask-worker-space")
            self.client = Client(cluster)

        # Initialize cache but don't register it by default
        self.cache = Cache(self.cache_size)
        self.cache_registered = False  # Cache is not registered by default

    def submit_tasks(self, tasks: List[SimulationTask]) -> List[Future]:
        """Submit tasks for execution.

        Args:
            tasks: List of simulation tasks to execute

        Returns:
            List[Future]: List of Dask futures for the tasks
        """
        if not self.client:
            self._initialize_client()

        # Submit all tasks
        futures: List[Future] = []
        assert self.client is not None
        for task in tasks:
            future = self.client.submit(
                task.run, key=task.get_config_id()  # Use task's public config ID as key
            )
            futures.append(future)

        return futures

    def gather_results(self, futures: List[Future]) -> List[History]:
        """Gather results from submitted tasks.

        Args:
            futures: List of Dask futures to gather results from

        Returns:
            List[History]: List of simulation histories
        """
        if not self.client:
            raise RuntimeError("No Dask client available")

        return self.client.gather(futures)  # type: ignore

    def run_tasks(
        self, tasks: List[SimulationTask], task_identifiers: list
    ) -> Tuple[List[History], list]:
        """Run tasks using Dask (submit and gather results).

        Args:
            tasks: List of simulation tasks to execute
            task_identifiers: List of identifiers for the tasks

        Returns:
            Tuple[List[History], list]: A tuple containing:
                - List of successful simulation histories
                - List of identifiers for successful tasks
        """
        # Run tasks and get results
        results = self._run_tasks(tasks)

        # Filter successful results and their identifiers
        successful_results = []
        successful_identifiers = []
        for result, identifier in zip(results, task_identifiers):
            if result is not None:
                successful_results.append(result)
                successful_identifiers.append(identifier)

        return successful_results, successful_identifiers

    def _run_tasks(self, tasks: List[SimulationTask]) -> list:
        """Run tasks using Dask (submit and gather results)."""
        futures = self.submit_tasks(tasks)
        return self.gather_results(futures)

    def get_task_status(self, futures: List[Future]) -> Dict[str, str]:
        """Get status of submitted tasks.

        Args:
            futures: List of Dask futures to check

        Returns:
            Dict[str, str]: Dictionary mapping task keys to their status
        """
        if not self.client:
            raise RuntimeError("No Dask client available")

        return {future.key: future.status for future in futures}  # type: ignore

    def cancel_tasks(self, futures: List[Future]):
        """Cancel submitted tasks.

        Args:
            futures: List of Dask futures to cancel
        """
        if not self.client:
            raise RuntimeError("No Dask client available")

        for future in futures:
            self.client.cancel(future)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cache and self.cache_registered:
            self.cache.unregister()
            self.cache = None


class JoblibTaskManager(TaskManagerExternalDB):
    """A task manager that uses joblib for parallel execution and caching."""

    def __init__(
        self,
        cache_db: DataBaseInterface,
        n_jobs: int = -1,  # -1 means use all available cores
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False,
        verbose: int = 0,
        logger_debug: bool = False,
    ):
        """Initialize the joblib task manager.

        Args:
            cache_db: The cache database to use
            n_jobs: Number of parallel jobs (-1 for all cores)
            cache_dir: Directory for joblib cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            verbose: Verbosity level for joblib
            logger_debug: Whether to enable debug logging
        """
        super().__init__(
            cache_db=cache_db,
            cache_dir=Path(cache_dir) if cache_dir else None,
            logger_debug=logger_debug,
        )
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "joblib_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory cache in a subdirectory to avoid clearing other files
        # like mlruns when memory.clear() is called
        self.joblib_cache_dir = self.cache_dir / "joblib_cache"
        self.joblib_cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = Memory(self.joblib_cache_dir, verbose=verbose)
        if clear_cache_on_start:
            self.memory.clear()
            self.cache_db.clear()

        # Create a cached version of the task runner
        self._cached_run = self.memory.cache(self._run_single_task)

    def _run_single_task(self, task: SimulationTask) -> Any:
        """Run a single task and return its result.

        Args:
            task: The simulation task to run

        Returns:
            History: The simulation history
        """
        result = task.run()
        if result is not None:
            self.cache_db.set(task.get_config_id(), result)
        else:
            self.logger.warning(f"Task {task.get_config_id()} returned None result")

        return result

    def _run_tasks(self, tasks: List[SimulationTask]) -> list:
        """Run tasks in parallel using joblib."""
        import time

        # Log system information and setup
        self._log_system_info()
        self._log_parallel_processing_setup(tasks)

        start_time = time.time()

        # Run tasks with progress logging
        try:
            results = self._execute_tasks_parallel(tasks, start_time)
            self._log_completion_statistics(results, start_time)
            return results

        except Exception as e:
            self.logger.error(f"Error during parallel processing: {str(e)}")
            raise e

    def _log_parallel_processing_setup(self, tasks: List[SimulationTask]) -> None:
        """Log parallel processing setup information."""
        self.logger.info(f"Starting parallel processing with {self.n_jobs} jobs")
        self.logger.info(f"Processing {len(tasks)} tasks using joblib")

    def _create_progress_callback(self, tasks: List[SimulationTask], start_time: float):
        """Create a custom tqdm callback that logs progress to logger."""
        import time

        def tqdm_logger_callback(tqdm_obj):
            """Custom tqdm callback to log progress to logger."""
            last_logged = 0
            for i in tqdm_obj:
                current_progress = i + 1
                # Log every 10% or at least every 5 tasks
                if (
                    current_progress % max(1, min(5, len(tasks) // 10)) == 0
                    and current_progress != last_logged
                ):
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"Progress: {current_progress}/{len(tasks)} tasks completed ({current_progress/len(tasks)*100:.1f}%) - Elapsed: {elapsed:.1f}s"
                    )
                    last_logged = current_progress
                yield i

        return tqdm_logger_callback

    def _execute_tasks_parallel(self, tasks: List[SimulationTask], start_time: float) -> list:
        """Execute tasks in parallel using joblib with progress tracking."""
        import time

        # Use tqdm with custom callback for logging
        with tqdm(tasks, desc="Running tasks") as pbar:
            results = list(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._cached_run)(task) for task in pbar
                )
            )
        return results

    def _log_completion_statistics(self, results: list, start_time: float) -> None:
        """Log completion statistics and cache information."""
        import time

        end_time = time.time()
        total_time = end_time - start_time

        # Log completion statistics
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)

        self.logger.info(f"Parallel processing completed in {total_time:.2f}s")
        self.logger.info(f"Results: {len(successful_results)} successful, {failed_count} failed")

        if failed_count > 0:
            self.logger.warning(f"{failed_count} tasks failed during parallel processing")

        # Log cache statistics
        self._log_cache_statistics()

    def clear_cache(self):
        """Clear the joblib cache."""
        self.memory.clear()
        self.cache_db.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper resource cleanup."""
        import gc

        # Shutdown loky worker pool to prevent orphaned processes
        try:
            from joblib.externals.loky import get_reusable_executor

            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            self.logger.debug("Shut down loky worker pool")
        except Exception as e:
            self.logger.warning(f"Error shutting down loky executor: {e}")

        # Clear joblib Memory cache to free cached function results
        if hasattr(self, "memory") and self.memory is not None:
            try:
                self.memory.clear()
                self.logger.debug("Cleared joblib Memory cache")
            except Exception as e:
                self.logger.warning(f"Error clearing joblib Memory cache: {e}")

        # Force garbage collection to clean up joblib parallel backend resources
        gc.collect()

        # Call parent cleanup
        super().__exit__(exc_type, exc_val, exc_tb)

    def _log_system_info(self):
        """Log system information for debugging and monitoring."""
        import os

        import psutil

        try:
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())

            self.logger.info(
                f"System Info - CPU cores: {cpu_count}, "
                f"Memory: {memory.total / (1024**3):.1f}GB total, "
                f"{memory.available / (1024**3):.1f}GB available"
            )
            self.logger.info(
                f"Process Info - PID: {process.pid}, "
                f"Memory usage: {process.memory_info().rss / (1024**2):.1f}MB"
            )
        except ImportError:
            self.logger.warning("psutil not available - skipping system info logging")
        except Exception as e:
            self.logger.warning(f"Could not log system info: {str(e)}")

    def _log_cache_statistics(self):
        """Log cache statistics for performance monitoring."""
        try:
            # Get joblib cache statistics
            cache_stats = self.memory.get_stats()  # type: ignore
            self.logger.info(
                f"Joblib Cache Stats - "
                f"Cache hits: {cache_stats.get('hits', 0)}, "
                f"Cache misses: {cache_stats.get('misses', 0)}"
            )

            # Calculate hit rate
            total_requests = cache_stats.get("hits", 0) + cache_stats.get("misses", 0)
            if total_requests > 0:
                hit_rate = cache_stats.get("hits", 0) / total_requests * 100
                self.logger.info(f"Cache hit rate: {hit_rate:.1f}%")

        except Exception as e:
            self.logger.warning(f"Could not log cache statistics: {str(e)}")


class PBSTaskManager(DaskTaskManager):
    """A task manager that uses Dask with PBS cluster for distributed computing.

    This task manager submits jobs to a PBS (Portable Batch System) cluster and
    provides an optional Dask dashboard for monitoring job execution, resource
    utilization, and performance metrics.

    The dashboard is enabled by default and provides real-time visualization of:
    - Task progress and execution status
    - Resource utilization across PBS nodes
    - Task graphs and dependencies
    - Performance metrics and bottlenecks

    Example:
        Basic usage with default dashboard::

            with PBSTaskManager(queue="gpu_queue", n_workers=4) as manager:
                futures = manager.submit_tasks(tasks)
                print(f"Dashboard: {manager.get_dashboard_url()}")
                results = manager.gather_results(futures)

        Custom dashboard configuration::

            manager = PBSTaskManager(
                queue="compute_queue",
                n_workers=8,
                dashboard_port=8888,
                dashboard_address="0.0.0.0",
                enable_dashboard=True
            )

        Disable dashboard::

            manager = PBSTaskManager(
                queue="batch_queue",
                enable_dashboard=False
            )

    Note:
        The dashboard requires network access to the specified port. In PBS
        environments, ensure the dashboard port is accessible through firewalls
        and security policies. The dashboard will automatically shut down when
        the task manager context exits.
    """

    def __init__(
        self,
        queue: str,
        n_workers: int = 4,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        scheduler_address: Optional[str] = None,
        walltime: str = "01:00:00",
        job_extra: Optional[List[str]] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
        enable_dashboard: bool = True,
        dashboard_address: str = "0.0.0.0",
        dashboard_port: int = 8787,
        dashboard_prefix: Optional[str] = None,
    ):
        """Initialize the PBS task manager.

        Args:
            queue: PBS queue name to submit jobs to
            n_workers: Number of worker jobs to submit to PBS
            cores: Number of CPU cores per PBS job
            memory: Memory per PBS job (e.g., "4GB", "1000MB")
            processes: Number of processes per PBS job
            scheduler_address: Address of Dask scheduler (None for local)
            walltime: Maximum runtime per job in HH:MM:SS format
            job_extra: Additional PBS directives as list of strings
            cache_size: Size of cache in bytes
            clear_cache_on_start: If True, clears the cache at startup
            enable_dashboard: If True, enables the Dask dashboard
            dashboard_address: Address to bind the dashboard to
            dashboard_port: Port for the Dask dashboard
            dashboard_prefix: URL prefix for dashboard (useful with reverse proxies)
        """
        # Set PBS-specific attributes BEFORE calling super().__init__()
        # This prevents AttributeError when parent's _initialize_client() is called
        self.queue = queue
        self.cores = cores
        self.memory = memory
        self.processes = processes
        self.walltime = walltime
        self.job_extra = job_extra or []

        # Dashboard configuration
        self.enable_dashboard = enable_dashboard
        self.dashboard_address = dashboard_address
        self.dashboard_port = dashboard_port
        self.dashboard_prefix = dashboard_prefix

        # Initialize parent attributes that will be overridden
        self.client = None
        self.cache = None
        self.cache_registered = False
        self.cluster = None  # Store cluster reference for dashboard access

        # Call parent's __init__ which will call _initialize_client()
        # Note: We override _initialize_client() to use PBS cluster configuration
        super().__init__(
            n_workers=n_workers,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
        )

    def _initialize_client(self):
        """Initialize Dask client with PBS cluster."""
        try:
            # If scheduler_address is provided, connect to existing cluster
            if self.scheduler_address is not None:
                try:
                    self.client = Client(self.scheduler_address, timeout=0.1)
                    # Initialize cache
                    self.cache = Cache(self.cache_size)
                    self.cache_registered = False
                    return
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to connect to Dask scheduler at {self.scheduler_address}. Error: {e}"
                    )

            # Prepare scheduler options for dashboard configuration
            scheduler_options = {}
            if self.enable_dashboard:
                scheduler_options["dashboard_address"] = (
                    f"{self.dashboard_address}:{self.dashboard_port}"
                )
                if self.dashboard_prefix:
                    scheduler_options["dashboard_prefix"] = self.dashboard_prefix

            # Create PBS cluster configuration (without scheduler_address parameter)
            self.cluster = PBSCluster(
                queue=self.queue,
                cores=self.cores,
                memory=self.memory,
                processes=self.processes,
                walltime=self.walltime,
                job_extra=self.job_extra,
                local_directory="./dask-pbs-space",
                scheduler_options=scheduler_options if scheduler_options else None,
            )

            # Scale cluster to desired number of workers
            self.cluster.scale(jobs=self.n_workers)

            # Create client
            self.client = Client(self.cluster)

            # Log dashboard information if enabled
            if self.enable_dashboard:
                dashboard_url = self.get_dashboard_url()
                if dashboard_url:
                    print(f"Dask dashboard available at: {dashboard_url}")
                else:
                    print("Dashboard was enabled but URL could not be determined")

            # Initialize cache but don't register it by default
            self.cache = Cache(self.cache_size)
            self.cache_registered = False
        except ImportError as e:
            if "dask_jobqueue" in str(e):
                raise RuntimeError("dask-jobqueue is required for PBS support") from e
            raise

    def get_dashboard_url(self) -> Optional[str]:
        """Get the URL for the Dask dashboard.

        Returns:
            Dashboard URL if available, None otherwise.
        """
        if not self.enable_dashboard or not self.client:
            return None

        try:
            # Try to get dashboard URL from client
            if hasattr(self.client, "dashboard_link"):
                return self.client.dashboard_link

            # Fallback: construct URL from configuration
            if self.cluster and hasattr(self.cluster, "scheduler_address"):
                scheduler_address = self.cluster.scheduler_address
                if scheduler_address:
                    # Extract host from scheduler address
                    host = (
                        scheduler_address.split(":")[0]
                        if ":" in scheduler_address
                        else scheduler_address
                    )
                    base_url = f"http://{host}:{self.dashboard_port}"
                    if self.dashboard_prefix:
                        return f"{base_url}/{self.dashboard_prefix.lstrip('/')}"
                    return base_url

        except Exception:
            # Silently handle any exceptions in URL construction
            pass

        return None

    def is_dashboard_running(self) -> bool:
        """Check if the Dask dashboard is running.

        Returns:
            True if dashboard is running, False otherwise.
        """
        if not self.enable_dashboard or not self.client:
            return False

        try:
            # Simple check - if we can get the dashboard URL, it's likely running
            return self.get_dashboard_url() is not None
        except Exception:
            return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cluster cleanup."""
        # Close the client first
        if self.client:
            try:
                print("Closing Dask client...")
                self.client.close()
            except Exception as e:
                print(f"Warning: Error closing Dask client: {e}")
            finally:
                self.client = None

        # Close the cluster (this will stop the dashboard)
        if self.cluster:
            try:
                print("Closing PBS cluster and dashboard...")
                self.cluster.close()
            except Exception as e:
                print(f"Warning: Error closing PBS cluster: {e}")
            finally:
                self.cluster = None

        # Clean up cache
        if self.cache and self.cache_registered:
            try:
                self.cache.unregister()
            except Exception as e:
                print(f"Warning: Error unregistering cache: {e}")
            finally:
                self.cache = None

        print("PBS task manager cleanup completed.")


class SequentialTaskManager(JoblibTaskManager):
    """A task manager that uses sequential execution and caching."""

    def __init__(
        self,
        cache_db: DataBaseInterface,
        cache_dir: Optional[str] = None,
        clear_cache_on_start: bool = False,
        verbose: int = 0,
        logger_debug: bool = False,
    ):
        """Initialize the sequential task manager.

        Args:
            cache_db: The cache database to use
            cache_dir: Directory for sequential cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            verbose: Verbosity level for joblib
            logger_debug: Whether to enable debug logging
        """
        super().__init__(
            cache_db=cache_db,
            cache_dir=cache_dir,
            logger_debug=logger_debug,
            verbose=verbose,
            clear_cache_on_start=clear_cache_on_start,
        )

    def _run_tasks(self, tasks: List[SimulationTask]) -> list:
        """Run tasks in sequential."""
        import time

        start_time = time.time()

        # Log system information
        self._log_system_info()

        # Run tasks with progress logging
        try:
            # Use tqdm with custom callback for logging
            with tqdm(tasks, desc="Running tasks") as pbar:
                results = []
                for task in pbar:
                    results.append(self._cached_run(task))

            end_time = time.time()
            total_time = end_time - start_time

            # Log completion statistics
            successful_results = [r for r in results if r is not None]
            failed_count = len(results) - len(successful_results)

            self.logger.info(f"Sequential processing completed in {total_time:.2f}s")
            self.logger.info(
                f"Results: {len(successful_results)} successful, {failed_count} failed"
            )

            if failed_count > 0:
                self.logger.warning(f"{failed_count} tasks failed during sequential processing")

            # Log cache statistics
            self._log_cache_statistics()

            return results

        except Exception as e:
            self.logger.error(f"Error during sequential processing: {str(e)}")
            raise e
