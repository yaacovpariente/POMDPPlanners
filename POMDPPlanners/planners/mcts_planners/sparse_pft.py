import random
from typing import Any, Tuple, Optional
from pathlib import Path

import numpy as np

from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.tree import BeliefNode, ActionNode, get_optimal_action_reward_setting, sample_belief_node_child
from POMDPPlanners.core.cost import belief_expectation_cost, belief_expectation_reward
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import PathSimulationPolicy


class SparsePFT(PathSimulationPolicy):
    def __init__(
        self, 
        environment: DiscreteActionsEnvironment, 
        discount_factor: float, 
        gamma: float, 
        depth: int, 
        c_ucb: float, 
        beta_ucb: float, 
        belief_child_num: int, 
        n_simulations: int, 
        name: str = "SparsePFT", 
        log_path: Optional[Path] = None, 
        debug: bool = False
    ):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=None,
            log_path=log_path,
            debug=debug
        )
        
        if not isinstance(environment, DiscreteActionsEnvironment):
            raise TypeError("environment must be a DiscreteActionsEnvironment instance")
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor must be a float")
        if not isinstance(gamma, float):
            raise TypeError("gamma must be a float")
        if not isinstance(depth, int):
            raise TypeError("depth must be an int")
        if not isinstance(c_ucb, float):
            raise TypeError("c_ucb must be a float")
        if not isinstance(beta_ucb, float):
            raise TypeError("beta_ucb must be a float")
        if not isinstance(belief_child_num, int):
            raise TypeError("belief_child_num must be an int")
        if not isinstance(n_simulations, int):
            raise TypeError("n_simulations must be an int")
        if not (1 >= discount_factor >= 0):
            raise ValueError("discount_factor must be between 0 and 1")
        
        self.gamma = gamma
        self.depth = depth
        self.c_ucb = c_ucb
        self.beta_ucb = beta_ucb
        self.belief_child_num = belief_child_num
        
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth: 
            belief_node.parent = None
            return 0
        
        if self.is_terminal_belief(belief=belief_node.belief):
            belief_node.visit_count += 1
            return 0
        
        if belief_node.is_leaf:
            for action in self.environment.get_actions():
                action_node = ActionNode(action=action, parent=belief_node, children=tuple())
            
            state = belief_node.belief.sample()
            belief_node.visit_count += 1

            return self.random_rollout(state=state, depth=depth)
        
        action_node = self.get_explored_action_node(belief_node=belief_node)
        
        if len(action_node.children) == self.belief_child_num:
            next_belief_node, immediate_reward = self._sample_next_existing_belief(action_node=action_node)
        else:
            next_belief_node, immediate_reward = self._generate_belief(action_node=action_node)
            
        return_sample = immediate_reward + self.gamma * self._simulate_path(belief_node=next_belief_node, depth=depth + 1)
        
        self.update_nodes(belief_node=belief_node, action_node=action_node, return_sample=return_sample)    
            
        return return_sample
            
    def is_terminal_belief(self, belief: Belief) -> bool:
        """Checks if all paricles are terminal states."""
        return sum(self.environment.is_terminal(state) for state in belief.particles) == len(belief.particles)     
        
    def get_explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        children_visit_counts = np.array([child.visit_count for child in belief_node.children])
        unvisited_action_indices = np.where(children_visit_counts == 0)[0]
        if len(unvisited_action_indices) > 0:
            return belief_node.children[np.random.choice(unvisited_action_indices)]

        q_vals = np.array([child.q_value for child in belief_node.children])
        children_visit_counts = np.array([child.visit_count for child in belief_node.children])
        
        sprase_pft_exploration_addtion = self.c_ucb * self.beta_ucb * belief_node.visit_count * 1 / np.sqrt(children_visit_counts)
        selected_action_index = np.argmax(q_vals + sprase_pft_exploration_addtion)
        
        return belief_node.children[selected_action_index]
    
    def _sample_next_existing_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        child_visit_counts = np.array([child.visit_count for child in action_node.children])
        if sum(child_visit_counts) == 0:
            # If no children have been visited, randomly select one and return with its immediate cost
            sampled_belief_node = np.random.choice(action_node.children)
            expected_reward = -sampled_belief_node.immediate_cost
            return sampled_belief_node, expected_reward
        
        weights = child_visit_counts / sum(child_visit_counts)
        sampled_belief_node = np.random.choice(action_node.children, p=weights)
        expected_reward = -sampled_belief_node.immediate_cost
        return sampled_belief_node, expected_reward 
    
    def _generate_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        belief = action_node.parent.belief
        state = belief.sample()
        next_state = self.environment.state_transition_model(state=state, action=action_node.action).sample()[0]
        next_observation = self.environment.observation_model(next_state=next_state, action=action_node.action).sample()[0]
        
        next_belief = belief.update(
            action=action_node.action,
            observation=next_observation,
            pomdp=self.environment
        )
        
        next_belief_node = BeliefNode(
            belief=next_belief,
            observation=next_observation,
            parent=action_node
        )
        next_belief_node.immediate_cost = belief_expectation_cost(belief=belief, action=action_node.action, env=self.environment)
        immediate_reward = -next_belief_node.immediate_cost
        
        return next_belief_node, immediate_reward
        
    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        
        action = random.choice(self.environment.get_actions())
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action)
        
        return reward + self.discount_factor * self.random_rollout(state=next_state, depth=depth + 1)
    
    def update_nodes(self, belief_node: BeliefNode, action_node: ActionNode, return_sample: float):
        belief_node.visit_count += 1
        action_node.visit_count += 1
        
        if action_node.immediate_cost is None:
            action_node.immediate_cost = belief_expectation_cost(belief=belief_node.belief, action=action_node.action, env=self.environment)
        
        action_node.q_value += (return_sample - action_node.q_value) / action_node.visit_count
        belief_node.v_value = max(child.q_value for child in belief_node.children)
        
    def get_space_info(self) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.MIXED
        )
