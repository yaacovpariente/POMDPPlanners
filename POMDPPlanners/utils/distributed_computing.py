from typing import List, Any, Callable, Dict
from joblib import Parallel, delayed, Memory
from tqdm import tqdm

import ray

from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


def run_parallel_locally(
    func: Callable,
    kwargs_list: List[Dict[str, Any]],
    n_jobs: int = 1,
    description: str = "Running parallel tasks",
    unit: str = "task",
    cache_dir: str = None
) -> List[Any]:
    """Run a function in parallel with different keyword argument sets using joblib.
    
    Args:
        func: The function to run in parallel
        kwargs_list: List of keyword argument dictionaries, where each dict contains the kwargs for one function call
        n_jobs: Number of parallel jobs to use
        description: Description for the progress bar
        unit: Unit label for the progress bar
        cache_dir: Directory to store cached results. If None, caching is disabled.
        
    Returns:
        List of results from each function call
    """
    logger.info(f"Starting parallel execution with {len(kwargs_list)} tasks using {n_jobs} jobs")
    
    # Set up caching if cache_dir is provided
    if cache_dir is not None:
        logger.info(f"Using cache directory: {cache_dir}")
        memory = Memory(cache_dir, verbose=0)
        func = memory.cache(func)
    
    # Run tasks in parallel using joblib with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(**kwargs) for kwargs in tqdm(
            kwargs_list,
            total=len(kwargs_list),
            desc=description,
            unit=unit
        )
    )
    
    logger.info(f"All parallel tasks completed")
    return results


def run_distributed(
    func: Callable,
    kwargs_list: List[Dict[str, Any]],
    num_cpus: int = 1,
    num_gpus: int = 0,
    description: str = "Running distributed tasks",
    unit: str = "task",
    address: str = None,
    namespace: str = "POMDPPlanners",
    runtime_env: dict = None
) -> List[Any]:
    """Run a function in parallel across multiple machines using Ray.
    
    Args:
        func: The function to run in parallel
        kwargs_list: List of keyword argument dictionaries, where each dict contains the kwargs for one function call
        num_cpus: Number of CPUs to allocate per task
        num_gpus: Number of GPUs to allocate per task
        description: Description for the progress bar
        unit: Unit label for the progress bar
        address: Ray cluster address to connect to (if None, starts a local cluster)
        namespace: Ray namespace for the tasks
        runtime_env: Runtime environment configuration for Ray
        
    Returns:
        List of results from each function call
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        if address is None:
            logger.info("Starting local Ray cluster")
            ray.init(namespace=namespace, runtime_env=runtime_env)
        else:
            logger.info(f"Connecting to Ray cluster at {address}")
            ray.init(address=address, namespace=namespace, runtime_env=runtime_env)

    # Create a Ray remote function from the input function
    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    def remote_func(**kwargs):
        return func(**kwargs)

    logger.info(f"Starting distributed execution with {len(kwargs_list)} tasks")
    logger.info(f"Resources per task: {num_cpus} CPUs, {num_gpus} GPUs")

    # Submit all tasks
    futures = [remote_func.remote(**kwargs) for kwargs in kwargs_list]
    
    # Track progress and collect results
    results = []
    with tqdm(total=len(kwargs_list), desc=description, unit=unit) as pbar:
        while futures:
            # Wait for any task to complete
            done_id, futures = ray.wait(futures)
            # Get the result
            result = ray.get(done_id[0])
            results.append(result)
            pbar.update(1)

    logger.info("All distributed tasks completed")
    return results
