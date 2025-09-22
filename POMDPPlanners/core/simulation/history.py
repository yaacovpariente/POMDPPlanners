from typing import Any, NamedTuple, TYPE_CHECKING, List
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
    """Complete history of a POMDP simulation episode.
    
    This class stores the complete history of a simulation episode, including
    all step data, timing information, and metadata about the episode.
    
    Attributes:
        history: List of StepData objects representing each step
        discount_factor: Discount factor used for reward calculation
        average_state_sampling_time: Average time spent sampling states
        average_action_time: Average time spent selecting actions
        average_observation_time: Average time spent processing observations
        average_belief_update_time: Average time spent updating beliefs
        average_reward_time: Average time spent calculating rewards
        actual_num_steps: Actual number of steps taken in the episode
        reach_terminal_state: Whether the episode reached a terminal state
        policy_run_data: Additional data from the policy execution
    
    Examples:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import WeightedParticleBelief
        >>> from POMDPPlanners.core.policy import PolicyRunData
        >>> 
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> import numpy as np
        >>> belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
        >>> step = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
        >>> policy_data = PolicyRunData(info_variables=[])
        >>> 
        >>> history = History(
        ...     history=[step],
        ...     discount_factor=0.95,
        ...     average_state_sampling_time=0.001,
        ...     average_action_time=0.01,
        ...     average_observation_time=0.002,
        ...     average_belief_update_time=0.005,
        ...     average_reward_time=0.001,
        ...     actual_num_steps=1,
        ...     reach_terminal_state=False,
        ...     policy_run_data=policy_data
        ... )
        >>> history.discount_factor
        0.95
        >>> len(history.history)
        1
        >>> history.reach_terminal_state
        False
    """
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
        
        Examples:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.core.belief import WeightedParticleBelief
            >>> from POMDPPlanners.core.policy import PolicyRunData
            >>> 
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> import numpy as np
        >>> belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
            >>> step = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
            >>> policy_data = PolicyRunData(info_variables=[])
            >>> 
            >>> history1 = History(
            ...     history=[step],
            ...     discount_factor=0.95,
            ...     average_state_sampling_time=0.001,
            ...     average_action_time=0.01,
            ...     average_observation_time=0.002,
            ...     average_belief_update_time=0.005,
            ...     average_reward_time=0.001,
            ...     actual_num_steps=1,
            ...     reach_terminal_state=False,
            ...     policy_run_data=policy_data
            ... )
            >>> history2 = History(
            ...     history=[step],
            ...     discount_factor=0.95,
            ...     average_state_sampling_time=0.001,
            ...     average_action_time=0.01,
            ...     average_observation_time=0.002,
            ...     average_belief_update_time=0.005,
            ...     average_reward_time=0.001,
            ...     actual_num_steps=1,
            ...     reach_terminal_state=False,
            ...     policy_run_data=policy_data
            ... )
            >>> history1 == history2
            True
            >>> history1 == "not_a_history"
            False
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
        """Convert History object to dictionary.
        
        Returns:
            dict: Dictionary representation of the History object
        
        Examples:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.core.belief import WeightedParticleBelief
            >>> from POMDPPlanners.core.policy import PolicyRunData
            >>> 
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> import numpy as np
        >>> belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
            >>> step = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
            >>> policy_data = PolicyRunData(info_variables=[])
            >>> 
            >>> history = History(
            ...     history=[step],
            ...     discount_factor=0.95,
            ...     average_state_sampling_time=0.001,
            ...     average_action_time=0.01,
            ...     average_observation_time=0.002,
            ...     average_belief_update_time=0.005,
            ...     average_reward_time=0.001,
            ...     actual_num_steps=1,
            ...     reach_terminal_state=False,
            ...     policy_run_data=policy_data
            ... )
            >>> history_dict = history.to_dict()
            >>> history_dict['discount_factor']
            0.95
            >>> history_dict['actual_num_steps']
            1
            >>> len(history_dict['history'])
            1
        """
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
        
        Examples:
            >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            >>> from POMDPPlanners.core.belief import WeightedParticleBelief
            >>> from POMDPPlanners.core.policy import PolicyRunData
            >>> 
            >>> env = TigerPOMDP(discount_factor=0.95)
            >>> import numpy as np
        >>> belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
            >>> step = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
            >>> policy_data = PolicyRunData(info_variables=[])
            >>> 
            >>> history = History(
            ...     history=[step],
            ...     discount_factor=0.95,
            ...     average_state_sampling_time=0.001,
            ...     average_action_time=0.01,
            ...     average_observation_time=0.002,
            ...     average_belief_update_time=0.005,
            ...     average_reward_time=0.001,
            ...     actual_num_steps=1,
            ...     reach_terminal_state=False,
            ...     policy_run_data=policy_data
            ... )
            >>> history_dict = history.to_dict()
            >>> restored_history = History.from_dict(history_dict)
            >>> restored_history.discount_factor
            0.95
            >>> restored_history.actual_num_steps
            1
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        
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


def history_to_discounted_return_value(history: History) -> float:
    """Calculate the discounted return value from a simulation history.
    
    This function computes the total discounted reward for an episode,
    where rewards are discounted by the discount factor raised to the power
    of the step index.
    
    Args:
        history: The simulation history containing step data and discount factor
        
    Returns:
        float: The total discounted return value
        
    Examples:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import WeightedParticleBelief
        >>> from POMDPPlanners.core.policy import PolicyRunData
        >>> 
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> import numpy as np
        >>> belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
        >>> step1 = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
        >>> step2 = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
        >>> policy_data = PolicyRunData(info_variables=[])
        >>> 
        >>> history = History(
        ...     history=[step1, step2],
        ...     discount_factor=0.9,
        ...     average_state_sampling_time=0.001,
        ...     average_action_time=0.01,
        ...     average_observation_time=0.002,
        ...     average_belief_update_time=0.005,
        ...     average_reward_time=0.001,
        ...     actual_num_steps=2,
        ...     reach_terminal_state=False,
        ...     policy_run_data=policy_data
        ... )
        >>> discounted_return = history_to_discounted_return_value(history)
        >>> # Should be -1.0 + (-1.0 * 0.9) = -1.9
        >>> abs(discounted_return - (-1.9)) < 1e-6
        True
    """
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)