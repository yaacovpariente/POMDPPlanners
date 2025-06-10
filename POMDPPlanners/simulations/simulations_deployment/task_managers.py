from typing import Optional, List, Dict, Tuple
import os
from pathlib import Path
from enum import Enum
from tqdm import tqdm

from dask.distributed import Future
from dask.distributed import Client, LocalCluster
from dask.cache import Cache
from joblib import Parallel, delayed, Memory

from POMDPPlanners.core.simulation import TaskManager, TaskManagerExternalDB, History, SimulationTask, DataBaseInterface


class TaskManagerType(Enum):
    """Enum representing different types of task managers for simulation execution."""
    DASK = "dask"  # Uses DaskTaskManager for distributed computing
    JOBLIB = "joblib"  # Uses JoblibTaskManager for parallel processing with caching


class DaskTaskManager(TaskManager):
    """A class to manage simulation tasks and their execution using Dask."""
    
    def __init__(
        self,
        n_workers: int = None,
        scheduler_address: Optional[str] = None,
        cache_size: int = 2e9,  # 2GB default cache size
        clear_cache_on_start: bool = False
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
        self.client = None
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
                 raise RuntimeError("Failed to connect to Dask scheduler at {}. Error: {}".format(self.scheduler_address, e))
        else:
             # Create local cluster with persistent storage
             cluster = LocalCluster(n_workers=self.n_workers, local_directory='./dask-worker-space')
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
        futures = []
        for task in tasks:
            future = self.client.submit(
                task.run,
                key=task._cache_key  # Use task's cache key for Dask caching
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
            
        return self.client.gather(futures)
    
    def run_tasks(self, tasks: List[SimulationTask], task_identifiers: list) -> Tuple[List[History], list]:
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
    
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[History]:
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
            
        return {
            future.key: future.status
            for future in futures
        }
    
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
        logger_debug: bool = False
    ):
        """Initialize the joblib task manager.
        
        Args:
            cache_db: The cache database to use
            n_jobs: Number of parallel jobs (-1 for all cores)
            cache_dir: Directory for joblib cache (None for default)
            clear_cache_on_start: If True, clears the cache at startup
            verbose: Verbosity level for joblib
        """
        super().__init__(cache_db=cache_db, cache_dir=cache_dir, logger_debug=logger_debug)
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'joblib_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory cache
        self.memory = Memory(self.cache_dir, verbose=verbose)
        if clear_cache_on_start:
            self.memory.clear()
            self.cache_db.clear()
            
        # Create a cached version of the task runner
        self._cached_run = self.memory.cache(self._run_single_task)
    
    def _run_single_task(self, task: SimulationTask) -> History:
        """Run a single task and return its result.
        
        Args:
            task: The simulation task to run
            
        Returns:
            History: The simulation history
        """
        return task.run()
    
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[History]:
        """Run tasks in parallel using joblib."""
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._cached_run)(task) for task in tqdm(tasks, desc="Running tasks")
        )
        return results
    
    def clear_cache(self):
        """Clear the joblib cache."""
        self.memory.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed for joblib
        pass
