from typing import Any, List, Tuple, Optional
from itertools import product
from pathlib import Path

import numpy as np

from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.belief import Belief

class DiscreteActionSequencesPlanner(Policy):
    def __init__(self, environment: DiscreteActionsEnvironment, discount_factor: float, name: str, depth: int, n_return_samples: int, log_path: Optional[Path] = None, debug: bool = False):
        super().__init__(environment=environment, discount_factor=discount_factor, name=name, log_path=log_path, debug=debug)

        assert depth > 0
        assert n_return_samples > 0
        assert 1 >= discount_factor >= 0
        
        self.depth = depth
        self.n_return_samples = n_return_samples
        self.discount_factor = discount_factor

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.MIXED
        )

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        return self.search(belief), PolicyRunData(info_variables=[])

    def search(self, belief: Belief) -> Any:
        actions = self.environment.get_actions()
        action_sequences = list(product(actions, repeat=self.depth))
        returns = []
        
        for action_sequence in action_sequences:
            returns.append(self.estimate_return(action_sequence=action_sequence, belief=belief))

        return list(action_sequences[np.argmax(returns)])

    def estimate_return(self, action_sequence: List[Any], belief: Belief) -> float:
        return_estimator = 0
        for _ in range(self.n_return_samples):
            state = belief.sample()
            return_sample = 0
        
            for i, action in enumerate(action_sequence):
                state, observation, reward = self.environment.sample_next_step(state, action)
                return_sample += reward * (self.discount_factor ** i)
            
            return_estimator += return_sample
            
        return_estimator /= self.n_return_samples
        
        return return_estimator
