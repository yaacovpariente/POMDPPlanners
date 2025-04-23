from abc import ABC, abstractmethod
import numpy as np

from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief, sample_next_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode, get_optimal_action_cost_setting
from POMDPPlanners.core.cost import belief_expectation_cost
# TODO: update num visits and change update statistics to update q values
class BeliefBasedMCTS(Policy, ABC):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float,
        n_simulations: int, 
        depth: int
    ):
        super().__init__(
            environment=environment, 
            discount_factor=discount_factor
        )
        
        self.n_simulations = n_simulations
        self.depth = depth
        
    def action(self, belief: Belief):
        tree = self._learn_belief_tree(belief=belief)
        return get_optimal_action_cost_setting(belief_node=tree)
    
    def _learn_belief_tree(self, belief: Belief) -> BeliefNode:
        tree = BeliefNode(
            belief=belief,
            observation=None,  # Root node has no observation
            parent=None,
            children=tuple(),
            data=None
        )
        
        for _ in range(self.n_simulations):
            self._simulate_path(node=tree, current_depth=0)
        
        return tree
    
    def _simulate_path(
        self, 
        node: BeliefNode,
        current_depth: int
    ):
        assert isinstance(node, BeliefNode)
        node.visit_count += 1
        
        if current_depth > self.depth:
            return
        
        if node.is_leaf:
            self._expand_leaf_belief_node(node=node, current_depth=current_depth)
            return
        
        action_node = self._get_explored_action_node(node)
        action_node.visit_count += 1
        
        next_belief_node = self._sample_next_belief_node_and_expand_tree(
            belief_node=node,
            action_node=action_node
        )

        self._simulate_path(node=next_belief_node, current_depth=current_depth + 1)
        
        self._update_v_and_q_values(belief_node=node, action_node=action_node)

    def _expand_leaf_belief_node(self, node: BeliefNode, current_depth: int):
        for action in self.environment.get_actions():
            action_node = ActionNode(
                action=action,
                parent=node,
                children=tuple(),
                data=None
            )
            action_node.q_value = -float('inf')
            action_node.visit_count = 0
            
        node.v_value = self._random_rollout(
            belief=node.belief,
            current_depth=current_depth
        )
            
    @abstractmethod
    def _get_explored_action_node(self, node: BeliefNode) -> ActionNode:
        pass
    
    def _sample_next_belief_node_and_expand_tree(self, belief_node: BeliefNode, action_node: ActionNode) -> BeliefNode:
        # TODO: make this an abstract method because it assumes discrete observations
        state = belief_node.belief.sample()
        next_state = self.environment.state_transition_model(
            state=state,
            action=action_node.action   
        ).sample()
        next_observation = self.environment.observation_model(
            next_state=next_state,
            action=action_node.action
        ).sample()
        
        # Check if we already have a belief node for this observation
        for child in action_node.children:
            if self.environment.is_equal_observation(observation1=child.observation, observation2=next_observation):
                return child
            
        next_belief = belief_node.belief.update(
            action=action_node.action,
            observation=next_observation,
            pomdp=self.environment
        )
        
        child = BeliefNode(
            belief=next_belief,
            observation=next_observation,
            parent=action_node,
            children=tuple(),
            data=None
        )
        
        return child
    
    def _random_rollout(self, belief: Belief, current_depth: int) -> float:
        if current_depth > self.depth:
            return 0
        
        idx = np.random.randint(0, len(self.environment.get_actions()))
        random_action = self.environment.get_actions()[idx]

        next_belief, _ = sample_next_belief(
            belief=belief,
            action=random_action,
            pomdp=self.environment
        )
    
        cost_ = belief_expectation_cost(
            belief=next_belief,
            action=random_action,
            env=self.environment
        )
        
        return cost_ + self.discount_factor * self._random_rollout(
            belief=next_belief,
            current_depth=current_depth + 1
        )
    
    def _update_v_and_q_values(self, belief_node: BeliefNode, action_node: ActionNode):
        if action_node.is_leaf:
            self._update_leaf_node_q_value(node=action_node)
        else:
            self._update_non_leaf_action_node_q_value(node=action_node)

        if belief_node.is_leaf:
            pass
        else:
            self._update_non_leaf_belief_node_v_value(node=belief_node)

    @abstractmethod
    def _update_leaf_node_q_value(self, node: ActionNode):
        pass
    
    @abstractmethod
    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        pass
    
    @abstractmethod
    def _update_non_leaf_belief_node_v_value(self, node: BeliefNode):
        pass