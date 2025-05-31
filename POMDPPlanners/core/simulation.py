from abc import ABC, abstractmethod
from typing import Any, NamedTuple, TYPE_CHECKING, List, Union
from dataclasses import dataclass

import numpy as np


if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.policy import PolicyRunData

    
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


class EnvironmentRunParams(NamedTuple):
    environment: Any  # Use Any to avoid circular import
    belief: Any
    policies: list[Any]
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
    def run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        pass
    
class TaskManagerExternalDB(TaskManager):
    def __init__(self, cache_db: DataBaseInterface):
        self.cache_db = cache_db
    
    @abstractmethod
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        pass
    
    def run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        # Lists to store results and track which tasks need to be run
        results = [None] * len(tasks)
        tasks_to_run = []
        task_indices = []  # Keep track of original indices for uncached tasks
        
        # First pass: check cache and collect tasks that need to be run
        for i, task in enumerate(tasks):
            if self.cache_db.is_key_in_cache(task.get_config_id()):
                results[i] = self.cache_db.get(task.get_config_id())
            else:
                tasks_to_run.append(task)
                task_indices.append(i)
        
        # Run only the tasks that weren't in cache
        if tasks_to_run:
            new_results = self._run_tasks(tasks_to_run)
            
            # Store new results in their original positions
            for idx, result in zip(task_indices, new_results):
                results[idx] = result
                # Cache the new result
                self.cache_db.set(tasks[idx].get_config_id(), result)
        
        return results

def history_to_discounted_return_value(history: History) -> float:
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)
