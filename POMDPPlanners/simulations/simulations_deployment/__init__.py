from enum import Enum
from typing import Optional, Union

from POMDPPlanners.core.simulation import DataBaseInterface, TaskManager
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    DaskTaskManager,
    JoblibTaskManager,
    PBSTaskManager,
    TaskManagerType,
)


class DeploymentType(Enum):
    """Enum representing different types of deployment for simulations."""

    LOCAL = "local"
    REMOTE_RAY = "remote_ray"
    DASK_DISTRIBUTED = "dask_distributed"


__all__ = ["DeploymentType", "TaskManagerType", "TaskManagerFactory"]


class TaskManagerFactory:
    """Factory class for creating different types of task managers.

    This factory provides methods to create specific types of task managers:
    - create_dask: Creates a DaskTaskManager for distributed computing
    - create_joblib: Creates a JoblibTaskManager with disk caching for parallel processing
    - create_pbs: Creates a PBSTaskManager for PBS cluster computing
    """

    @staticmethod
    def create_dask(
        n_workers: Optional[int] = None,
        scheduler_address: Optional[str] = None,
        cache_size: int = int(2e9),  # 2GB default
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
            clear_cache_on_start=clear_cache_on_start,
        )

    @staticmethod
    def create_joblib(
        cache_dir: str = "./cache",
        cache_size: int = int(2e9),  # 2GB default
        n_jobs: int = -1,  # Use all available cores
        eviction_policy: str = "least-recently-used",
        clear_cache_on_start: bool = False,
        verbose: int = 0,
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
            cache_dir=cache_dir, size_limit=cache_size, eviction_policy=eviction_policy
        )

        return JoblibTaskManager(
            cache_db=cache_db,
            n_jobs=n_jobs,
            cache_dir=cache_dir,  # Use same directory for joblib cache
            clear_cache_on_start=clear_cache_on_start,
            verbose=verbose,
        )

    @staticmethod
    def create_pbs(
        queue: str,
        n_workers: int = 4,
        cores: int = 1,
        memory: str = "4GB",
        processes: int = 1,
        walltime: str = "01:00:00",
        job_extra: Optional[list] = None,
        cache_size: int = int(2e9),
        clear_cache_on_start: bool = False,
        enable_dashboard: bool = True,
        dashboard_address: str = "0.0.0.0",
        dashboard_port: int = 8787,
        dashboard_prefix: Optional[str] = None,
    ) -> PBSTaskManager:
        """Create a PBSTaskManager for PBS cluster computing.

        Args:
            queue: PBS queue name to submit jobs to
            n_workers: Number of worker jobs to submit to PBS
            cores: Number of CPU cores per PBS job
            memory: Memory per PBS job (e.g., "4GB", "1000MB")
            processes: Number of processes per PBS job
            walltime: Maximum runtime per job in HH:MM:SS format
            job_extra: Additional PBS directives as list of strings
            cache_size: Size of cache in bytes
            clear_cache_on_start: If True, clears the cache at startup
            enable_dashboard: If True, enables the Dask dashboard
            dashboard_address: Address to bind the dashboard to
            dashboard_port: Port for the Dask dashboard
            dashboard_prefix: URL prefix for dashboard (useful with reverse proxies)

        Returns:
            A configured PBSTaskManager instance

        Raises:
            RuntimeError: If dask-jobqueue is not installed
        """
        return PBSTaskManager(
            queue=queue,
            n_workers=n_workers,
            cores=cores,
            memory=memory,
            processes=processes,
            walltime=walltime,
            job_extra=job_extra,
            cache_size=cache_size,
            clear_cache_on_start=clear_cache_on_start,
            enable_dashboard=enable_dashboard,
            dashboard_address=dashboard_address,
            dashboard_port=dashboard_port,
            dashboard_prefix=dashboard_prefix,
        )
