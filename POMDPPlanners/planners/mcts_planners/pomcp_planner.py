from typing import Any
from time import time
from abc import ABC, abstractmethod

import numpy as np

from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief, ParticleBeliefResampling
from POMDPPlanners.core.tree import BeliefNode, get_optimal_action, ActionNode

class POMCPPlanner(Policy, ABC):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        n_simulations: int, 
        depth: int,
        exploration_constant: float,
        time_limit: float
    ):
        super().__init__(environment, discount_factor)
        
        self.n_simulations = n_simulations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
    
    def action(self, belief: Belief) -> Any:
        start_time = time()
        tree = BeliefNode(belief=belief)
        
        while time() - start_time < self.time_limit:
            self.simulate(tree, belief.sample(), 0)
            
        return get_optimal_action(tree)
    
    def simulate(self, belief_node: BeliefNode, state: Any, current_depth: int) -> float:
        # similar to previous implementation
                
        action_node = self._get_explored_action_node(belief_node=belief_node)
        
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action_node.action)
        next_belief = belief_node.belief.update(action=action_node.action, observation=next_observation)
        
        return_ = reward + self.discount_factor * self.simulate(
            belief_node=belief_node,
            state=next_state,
            belief_node=next_belief,
            current_depth=current_depth + 1
        )
        
        # TODO: add state to belief
        self._update_node_statistics(belief_node=belief_node, action_node=action_node, return_=return_)
        
        return return_
    
    def _get_explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        q_values = np.array([child.q_value for child in belief_node.children])
        visit_counts = np.array([child.visit_count for child in belief_node.children])
        ucb_values = q_values + self.exploration_constant * np.sqrt(np.log(belief_node.visit_count) / visit_counts)
        action_node = belief_node.children[np.argmax(ucb_values)]

        return action_node
    
    def _update_node_statistics(self, belief_node: BeliefNode, action_node: BeliefNode, return_: float):
        action_node.visit_count += 1
        action_node.v_value += (return_ - action_node.v_value) / action_node.visit_count
        
        belief_node.visit_count += 1
        belief_node.v_value = max([child.q_value for child in belief_node.children])
        
    def rollout(self, state: Any, belief: Belief, current_depth: int) -> float:
        if self.discount_factor ** current_depth < self.epsilon:
            return 0
        
        action = self._rollout_policy(belief)
        next_state = self.environment.state_transition_model(state=state, action=action).sample()
        next_observation = self.environment.observation_model(next_state=next_state, action=action).sample()
        reward = self.environment.reward(state=state, action=action)
        next_belief = belief.update(action=action, observation=next_observation)
        
        return reward + self.discount_factor * self.rollout(
            state=next_state, 
            belief=next_belief, 
            current_depth=current_depth + 1
        )
        
    @abstractmethod
    def _rollout_policy(self, belief: Belief) -> Any:
        pass
