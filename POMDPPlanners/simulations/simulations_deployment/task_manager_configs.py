# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from POMDPPlanners.core.simulation import TaskManager
from POMDPPlanners.simulations.simulations_deployment import TaskManagerFactory


class TaskManagerConfig(ABC):
    """Base configuration class for all task managers."""

    @abstractmethod
    def create_task_manager(self, cache_dir: Optional[str] = None) -> TaskManager:
        """Create and return the configured task manager."""


@dataclass
class DaskConfig(TaskManagerConfig):
    """Configuration for Dask task manager."""

    n_workers: int = 1
    scheduler_address: Optional[str] = None
    cache_size: int = int(2e9)
    clear_cache_on_start: bool = False

    def create_task_manager(self, cache_dir: Optional[str] = None) -> TaskManager:
        return TaskManagerFactory.create_dask(
            n_workers=self.n_workers,
            scheduler_address=self.scheduler_address,
            cache_size=self.cache_size,
            clear_cache_on_start=self.clear_cache_on_start,
        )


@dataclass
class PBSConfig(TaskManagerConfig):
    """Configuration for PBS cluster task manager."""

    queue: str
    n_workers: int = 4
    cores: int = 1
    memory: str = "4GB"
    processes: int = 1
    walltime: str = "01:00:00"
    job_extra: Optional[List[str]] = None
    cache_size: int = int(2e9)
    clear_cache_on_start: bool = False
    enable_dashboard: bool = True
    dashboard_address: str = "0.0.0.0"
    dashboard_port: int = 8787
    dashboard_prefix: Optional[str] = None

    def create_task_manager(self, cache_dir: Optional[str] = None) -> TaskManager:
        return TaskManagerFactory.create_pbs(
            queue=self.queue,
            n_workers=self.n_workers,
            cores=self.cores,
            memory=self.memory,
            processes=self.processes,
            walltime=self.walltime,
            job_extra=self.job_extra,
            cache_size=self.cache_size,
            clear_cache_on_start=self.clear_cache_on_start,
            enable_dashboard=self.enable_dashboard,
            dashboard_address=self.dashboard_address,
            dashboard_port=self.dashboard_port,
            dashboard_prefix=self.dashboard_prefix,
        )


@dataclass
class JoblibConfig(TaskManagerConfig):
    """Configuration for Joblib task manager."""

    n_jobs: int = -1
    cache_size: int = int(2e9)
    eviction_policy: str = "least-recently-used"
    verbose: int = 0
    clear_cache_on_start: bool = False
    console_output: bool = True
    no_logs: bool = False

    def create_task_manager(self, cache_dir: Optional[str] = None) -> TaskManager:
        return TaskManagerFactory.create_joblib(
            cache_dir=cache_dir or "./cache",
            cache_size=self.cache_size,
            n_jobs=self.n_jobs,
            eviction_policy=self.eviction_policy,
            clear_cache_on_start=self.clear_cache_on_start,
            verbose=self.verbose,
            console_output=self.console_output,
            no_logs=self.no_logs,
        )
