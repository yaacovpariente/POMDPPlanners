from enum import Enum

class DeploymentType(Enum):
    """Enum representing different types of deployment for simulations."""
    LOCAL = "local"
    REMOTE_RAY = "remote_ray"
    DASK_DISTRIBUTED = "dask_distributed"

from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    TaskManagerType,
    DaskTaskManager,
    JoblibTaskManager
)
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from typing import Optional, Union
from POMDPPlanners.core.simulation import TaskManager, DataBaseInterface

__all__ = ['DeploymentType', 'TaskManagerType', 'TaskManagerFactory']

class TaskManagerFactory:
    """Factory class for creating different types of task managers.
    
    This factory provides methods to create specific types of task managers:
    - create_dask: Creates a DaskTaskManager for distributed computing
    - create_joblib: Creates a JoblibTaskManager with disk caching for parallel processing
    """
    
    @staticmethod
    def create_dask(
        n_workers: Optional[int] = None,
        scheduler_address: Optional[str] = None,
        cache_size: int = 2e9,  # 2GB default
        clear_cache_on_start: bool = False,
    ) -> DaskTaskManager:
        """Create a DaskTaskManager for distributed computing.
        
        Args:
            n_workers: Number of worker processes (None for auto)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_size: Size of cache in bytes
            clear_cache_on_start: If True, clears the cache at startup
            
        Returns:
            A configured DaskTaskManager instance
        """
        return DaskTaskManager(
            n_workers=n_workers,
            scheduler_address=scheduler_address,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start
        )
    
    @staticmethod
    def create_joblib(
        cache_dir: str = "./cache",
        cache_size: int = 2e9,  # 2GB default
        n_jobs: int = -1,  # Use all available cores
        eviction_policy: str = "least-recently-used",
        clear_cache_on_start: bool = False,
        verbose: int = 0
    ) -> JoblibTaskManager:
        """Create a JoblibTaskManager with a configured DiskCacheDB.
        
        Args:
            cache_dir: Directory to store cache files
            cache_size: Maximum size of cache in bytes
            n_jobs: Number of parallel jobs (-1 for all cores)
            eviction_policy: Cache eviction policy ('least-recently-used' or 'least-frequently-used')
            clear_cache_on_start: If True, clears the cache at startup
            verbose: Verbosity level for joblib
            
        Returns:
            A configured JoblibTaskManager instance
        """
        cache_db = DiskCacheDB(
            cache_dir=cache_dir,
            size_limit=cache_size,
            eviction_policy=eviction_policy
        )
        
        return JoblibTaskManager(
            cache_db=cache_db,
            n_jobs=n_jobs,
            cache_dir=cache_dir,  # Use same directory for joblib cache
            clear_cache_on_start=clear_cache_on_start,
            verbose=verbose
        )
