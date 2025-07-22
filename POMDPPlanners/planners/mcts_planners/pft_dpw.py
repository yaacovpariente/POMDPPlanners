from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import time
from math import floor
import random
from pathlib import Path

import numpy as np

from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import PathSimulationPolicy


class ActionSampler(ABC):
    @abstractmethod
    def sample(self, belief_node: BeliefNode = None) -> Any:
        pass

class PFT_DPW(PathSimulationPolicy):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        depth: int, 
        name: str, 
        action_sampler: ActionSampler,
        k_a: float = 1.0,
        alpha_a: float = 0.5,
        k_o: float = 1.0,
        alpha_o: float = 0.5,
        exploration_constant: float = 1.0,
        time_out_in_seconds: int = None, 
        n_simulations: int = None, 
        min_samples_per_node: int = 10, 
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False
    ):
        super().__init__(
            environment=environment, 
            discount_factor=discount_factor, 
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            log_path=log_path,
            debug=debug
        )
        
        self.__type_check_inputs(
            min_samples_per_node=min_samples_per_node, 
            min_visit_count_per_action=min_visit_count_per_action
        )

        self.depth = depth
        self.action_sampler = action_sampler
        self.min_samples_per_node = min_samples_per_node
        self.min_visit_count_per_action = min_visit_count_per_action
        
        self.k_a = k_a
        self.alpha_a = alpha_a
        self.k_o = k_o
        self.alpha_o = alpha_o
        self.exploration_constant = exploration_constant
        
    def __type_check_inputs(
        self, 
        min_samples_per_node: int, 
        min_visit_count_per_action: int
    ):  
        if not isinstance(min_samples_per_node, int):
            raise TypeError("min_samples_per_node must be an int")
        if not isinstance(min_visit_count_per_action, int):
            raise TypeError("min_visit_count_per_action must be an int")
        # TODO: check if min_samples_per_node is greater than min_visit_count_per_action
        # TODO: check k and alpha params
        
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None
            return 0
        
        if self.environment.is_terminal(belief_node.belief.sample()):
            belief_node.visit_count += 1
            return 0
        
        action_node = self.action_progressive_widening(belief_node=belief_node)
        
        return_sample = self._simulate_return(
            belief_node=belief_node, 
            action_node=action_node,
            depth=depth
        )
        
        self._update_node_statistics(
            belief_node=belief_node, 
            action_node=action_node, 
            total=return_sample
        )
        
        return return_sample
    
    def action_progressive_widening(self, belief_node: BeliefNode) -> ActionNode:
        if belief_node.is_leaf or belief_node.visit_count == 0 or floor(belief_node.visit_count ** self.alpha_a) > floor((belief_node.visit_count - 1) ** self.alpha_a):
            action = self.action_sampler.sample()
            action_node = ActionNode(action=action, parent=belief_node)
            return action_node
        
        return self._explored_action_node(belief_node=belief_node)
        
    def _explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        q_vals = [child.q_value for child in belief_node.children]
        children_visit_counts = [child.visit_count for child in belief_node.children]
        
        ucb = q_vals + self.exploration_constant * np.sqrt(np.log(belief_node.visit_count) / children_visit_counts)
        
        return belief_node.children[np.argmax(ucb)]
        
    def _simulate_return(self, belief_node: BeliefNode, action_node: ActionNode, depth: int) -> float:
        if len(action_node.children) <= self.k_o * action_node.visit_count ** self.alpha_o:
            next_belief_node, immediate_reward = self._sample_new_belief_node(belief_node=belief_node, action_node=action_node)
            state = next_belief_node.belief.sample()
            total = immediate_reward + self.discount_factor * self._random_rollout(state=state, depth=depth + 1)
        else:
            next_belief_node, immediate_reward = self.sample_existing_belief_node(belief_node=belief_node, action_node=action_node)
            total = immediate_reward + self.discount_factor * self._simulate_path(belief_node=next_belief_node, depth=depth + 1)
        
        return total

    def _sample_new_belief_node(self, belief_node: BeliefNode, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        immediate_reward = belief_expectation_reward(belief=belief_node.belief, action=action_node.action, env=self.environment)
        belief_node.immediate_cost = -immediate_reward
        
        next_state, next_observation, reward = self.environment.sample_next_step(state=belief_node.belief.sample(), action=action_node.action)
        next_belief = belief_node.belief.update(observation=next_observation, action=action_node.action, pomdp=self.environment)
        next_belief_node = BeliefNode(belief=next_belief, parent=action_node)
        
        return next_belief_node, immediate_reward
    
    def _random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        
        action = self.action_sampler.sample()
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action)
        
        return reward + self.discount_factor * self._random_rollout(state=next_state, depth=depth + 1)
    
    def sample_existing_belief_node(self, belief_node: BeliefNode, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        immediate_reward = -belief_node.immediate_cost
        random_idx = np.random.randint(0, len(action_node.children))
        next_belief_node = action_node.children[random_idx]
        return next_belief_node, immediate_reward

    def _update_node_statistics(self, belief_node: BeliefNode, action_node: ActionNode, total: float):
        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (total - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max([child.q_value for child in belief_node.children])

    def get_space_info(self) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.MIXED
        )
