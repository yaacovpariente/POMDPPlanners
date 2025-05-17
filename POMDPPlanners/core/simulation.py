from typing import Any, NamedTuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from typing import List
import numpy as np

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief


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


class EnvironmentRunParams(NamedTuple):
    environment: Any  # Use Any to avoid circular import
    belief: Any
    policies: list[Any]
    num_episodes: int
    num_steps: int

def history_to_discounted_return_value(history: History) -> float:
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)