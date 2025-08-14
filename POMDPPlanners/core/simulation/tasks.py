from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from pathlib import Path
import logging
from POMDPPlanners.utils.logger import get_logger

class SimulationTask(ABC):
    @abstractmethod
    def run(self) -> Any:
        pass
    
    @abstractmethod
    def get_config_id(self) -> str:
        pass
    

class DataBaseInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        pass
    
    @abstractmethod
    def is_key_in_cache(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any):
        pass
    
    @abstractmethod
    def clear(self):
        pass
    
    
class TaskManager(ABC):
    @abstractmethod
    def run_tasks(self, tasks: List[SimulationTask], task_identifiers: list) -> Tuple[List[Any], list]:
        pass
    
class TaskManagerExternalDB(TaskManager):
    def __init__(self, cache_db: DataBaseInterface, cache_dir: Optional[Path] = None, logger_debug: bool = False):
        self.cache_db = cache_db
        self.cache_dir = cache_dir
        self.logger_debug = logger_debug
    
    @property
    def logger(self) -> logging.Logger:
        return get_logger(
            name=f"task_manager",
            debug=self.logger_debug,
            output_dir=self.cache_dir
        )
    
    @abstractmethod
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        pass
    
    def run_tasks(self, tasks: List[SimulationTask], task_identifiers: list) -> Tuple[List[Any], list]:
        self.logger.info(f"Starting to process {len(tasks)} tasks")
        # Lists to store results and track which tasks need to be run
        results = [None] * len(tasks)
        tasks_to_run = []
        task_indices = []  # Keep track of original indices for uncached tasks
        
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
        
        self.logger.info(f"Cache status: {cached_tasks} tasks cached, {len(tasks_to_run)} tasks uncached out of {len(tasks)} total tasks")
        
        # Run only the tasks that weren't in cache
        if tasks_to_run:
            self.logger.info(f"Running {len(tasks_to_run)} uncached tasks")
            new_results = self._run_tasks(tasks_to_run)
            self.logger.info(f"Completed {len(new_results)} tasks")
            
            if len(new_results) != len(tasks_to_run):
                raise ValueError("new_results and tasks_to_run must have the same length")
            
            # Store new results in their original positions
            for idx, result in zip(task_indices, new_results):
                if result is None:  # prevents storing failed tasks
                    continue

                results[idx] = result
                # Cache the new result
                task_id = tasks[idx].get_config_id()
                self.logger.debug(f"Storing task {idx} in cache with config_id: {task_id}")
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
                self.logger.warning(f"Task {i} (config_id: {task_id}) failed - returned None result")

        n_failed_tasks = len(tasks) - len(successful_results)
        self.logger.info(f"{len(successful_results)} tasks completed successfully")
        
        if n_failed_tasks > 0:
            self.logger.warning(f"{n_failed_tasks} tasks failed.")
            
        return successful_results, successful_identifiers