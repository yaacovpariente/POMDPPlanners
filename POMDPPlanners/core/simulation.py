from typing import Any, NamedTuple, Union
from dataclasses import dataclass
from typing import List


class StepData(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    observation: Any
    reward: float


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

    def to_dict(self) -> dict:
        """Convert History object to dictionary."""
        return {
            'history': [step._asdict() for step in self.history],
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
        history = [StepData(**step) for step in data['history']]
        
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
            reach_terminal_state=data['reach_terminal_state']
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

def history_to_discounted_return_value(history: History) -> float:
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)