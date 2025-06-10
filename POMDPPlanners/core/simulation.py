from abc import ABC, abstractmethod
from typing import Any, NamedTuple, TYPE_CHECKING, List, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

import numpy as np
from POMDPPlanners.utils.logger import get_logger


if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.policy import PolicyRunData
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    
class StepData(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    observation: Any
    reward: float
    belief: 'Belief'


@dataclass(frozen=True)
class History:
    history: List[StepData]
    discount_factor: float
    average_state_sampling_time: float
    average_action_time: float
    average_observation_time: float
    average_belief_update_time: float
    average_reward_time: float
    actual_num_steps: int
    reach_terminal_state: bool
    policy_run_data: 'PolicyRunData'

    def __eq__(self, other: object) -> bool:
        """Compare two History objects for equality.
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if objects are equal, False otherwise
        """
        if not isinstance(other, History):
            return False
            
        # Compare all fields using dataclasses.fields()
        from dataclasses import fields
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in fields(self)
        )

    def to_dict(self) -> dict:
        """Convert History object to dictionary."""
        history_data = []
        for step in self.history:
            step_dict = step._asdict()
            if hasattr(step_dict['belief'], 'to_dict'):
                belief_dict = step_dict['belief'].to_dict()
                belief_dict['type'] = step_dict['belief'].__class__.__name__
                step_dict['belief'] = belief_dict
            history_data.append(step_dict)

        return {
            'history': history_data,
            'discount_factor': self.discount_factor,
            'average_state_sampling_time': self.average_state_sampling_time,
            'average_action_time': self.average_action_time,
            'average_observation_time': self.average_observation_time,
            'average_belief_update_time': self.average_belief_update_time,
            'average_reward_time': self.average_reward_time,
            'actual_num_steps': self.actual_num_steps,
            'reach_terminal_state': self.reach_terminal_state
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'History':
        """Create a History instance from a dictionary.
        
        Args:
            data: Dictionary containing History data
            
        Returns:
            History: New History instance
        """
        assert isinstance(data, dict)
        
        # Convert history list of dictionaries back to StepData objects
        history = []
        for step_data in data['history']:
            # Handle belief deserialization
            if isinstance(step_data['belief'], dict) and 'type' in step_data['belief']:
                belief_type = step_data['belief']['type']
                # Import the belief class dynamically
                if belief_type == 'WeightedParticleBelief':
                    from POMDPPlanners.core.belief import WeightedParticleBelief
                    step_data['belief'] = WeightedParticleBelief(
                        particles=step_data['belief']['particles'],
                        log_weights=np.array(step_data['belief']['log_weights']),
                        resampling=step_data['belief'].get('resampling', False)
                    )
            history.append(StepData(**step_data))

        # Handle policy_run_data deserialization
        policy_run_data = data.get('policy_run_data', None)
        if isinstance(policy_run_data, dict):
            from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable
            info_variables = [
                PolicyInfoVariable(name=iv['name'], value=iv['value'])
                for iv in policy_run_data.get('info_variables', [])
            ]
            policy_run_data = PolicyRunData(info_variables=info_variables)

        # Create and return History instance
        return History(
            history=history,
            discount_factor=data['discount_factor'],
            average_state_sampling_time=data['average_state_sampling_time'],
            average_action_time=data['average_action_time'],
            average_observation_time=data['average_observation_time'],
            average_belief_update_time=data['average_belief_update_time'],
            average_reward_time=data['average_reward_time'],
            actual_num_steps=data['actual_num_steps'],
            reach_terminal_state=data['reach_terminal_state'],
            policy_run_data=policy_run_data
        )


class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float

@dataclass(frozen=True)
class EnvironmentRunParams:
    environment: 'Environment'
    belief: 'Belief'
    policies: list['Policy']
    num_episodes: int
    num_steps: int


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
            self.logger.info(f"Successfully completed {len(new_results)} tasks")
            
            assert len(new_results) == len(tasks_to_run)
            
            # Store new results in their original positions
            for idx, result in zip(task_indices, new_results):
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

def history_to_discounted_return_value(history: History) -> float:
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)
