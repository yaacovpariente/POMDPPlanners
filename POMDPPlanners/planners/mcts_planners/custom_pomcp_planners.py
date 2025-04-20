from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief, sample_next_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode, get_optimal_action
from POMDPPlanners.core.cost import belief_expectation_cost

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
        tree = self._learn_belief_tree(belief)
        return get_optimal_action(tree)
    
    def _learn_belief_tree(self, belief: Belief) -> BeliefNode:
        tree = BeliefNode(belief=belief)
        
        for _ in range(self.n_simulations):
            self._simulate_path(tree, current_depth=0)
        
        return tree
    
    def _simulate_path(
        self, 
        node: BeliefNode,
        current_depth: int
    ):
        assert isinstance(node, BeliefNode)
        
        if current_depth == self.depth:
            self._set_last_belief_node_statistics(node) # DONE
            return
        
        if node.is_leaf:
            self._set_leaf_belief_node_statistics_using_rollout(node, current_depth) # DONE
            return
        
        next_belief = self._sample_next_belief(
            action_node=self._get_explored_action(node)
        )

        self._simulate_path(next_belief, current_depth + 1)
        
        self._update_node_statistics(node)

    def _set_leaf_belief_node_statistics_using_rollout(self, node: BeliefNode, current_depth: int):
        for action in self.environment.get_actions():
            child = ActionNode(
                action=action,
                parent=node,
                children=tuple(),
                data=None
            )
            child.q_value = float('inf')
            
        chosen_child = np.random.choice(node.children)
        chosen_child.q_value = self._rollout(chosen_child, current_depth)
        chosen_child.num_visits = 1
        node.num_visits += 1
            
    @abstractmethod
    def _set_last_belief_node_statistics(self, node: BeliefNode):
        pass
    
    @abstractmethod
    def _get_explored_action(self, node: BeliefNode) -> ActionNode:
        pass
    
    def _sample_next_belief(self, belief_node: BeliefNode, action_node: ActionNode) -> Belief:
        state = belief_node.belief.sample()
        next_state = self.environment.state_transition_model(
            state=state,
            action=action_node.action   
        ).sample()
        next_observation = self.environment.observation_model(
            next_state=next_state,
            action=action_node.action
        ).sample()
        
        for child in belief_node.children:
            if child.observation == next_observation:
                return child.belief
            
        next_belief = belief_node.belief.update(
            action=action_node.action_node.action,
            observation=next_observation
        )
        
        child = BeliefNode(
            belief=next_belief,
            parent=action_node,
            children=tuple(),
            data=None
        )
        
        return next_belief
    
    @abstractmethod
    def _rollout(self, belief: Belief, action: Any, current_depth: int) -> float:
        pass
    
    def _update_node_statistics(self, tree: BeliefNode):
        for node in PostOrderIter(tree):
            if node.is_leaf:
                self._update_leaf_node_statistics(node)
            elif isinstance(node, ActionNode):
                self._update_non_leaf_action_node_statistics(node)
            elif isinstance(node, BeliefNode):
                self._update_belief_node_statistics(node)
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
    
    def _update_leaf_node_statistics(self, node: ActionNode):
        assert isinstance(node, ActionNode)
        assert node.height == 0
        
        node.visit_count += 1
        self._update_leaf_node_q_value(node)

    def _update_non_leaf_action_node_statistics(self, node: ActionNode):
        assert isinstance(node, ActionNode)
        
        node.visit_count += 1
        self._update_non_leaf_action_node_q_value(node)

    def _update_belief_node_statistics(self, node: BeliefNode):
        assert isinstance(node, BeliefNode)
        
        node.visit_count += 1
        self._update_belief_node_v_value(node)
    
    @abstractmethod
    def _update_leaf_node_q_value(self, node: ActionNode):
        pass
    
    @abstractmethod
    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        pass
    
    @abstractmethod
    def _update_belief_node_v_value(self, node: BeliefNode):
        pass

class StandardPOMCPPlanner(BeliefBasedMCTS):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float,
        n_simulations: int, 
        depth: int,
        exploration_constant: float
    ):
        super().__init__(
            environment=environment, 
            discount_factor=discount_factor,
            n_simulations=n_simulations,
            depth=depth
        )
        
        self.exploration_constant = exploration_constant
        
    def _set_last_belief_node_statistics(self, node: BeliefNode):
        self.num_visits += 1
        
        if node.is_leaf:
            for action in self.environment.get_actions():
                child = ActionNode(
                    action=action,
                    parent=node,
                    children=tuple(),
                    data=None
                )
                
                child.q_value = belief_expectation_cost(
                    belief=node.belief,
                    action=child.action,
                    env=self.environment
                )
                
                child.num_visits = 1
                
            node.v_value = np.min([child.q_value for child in node.children])
        
    def _get_explored_action(self, node: BeliefNode):
        assert isinstance(node, BeliefNode)
        
        action_nodes_q_values = [child.q_value for child in node.children]
        action_nodes_visits = np.array([child.num_visits for child in node.children])
        pomcp_exploration_addition = self.exploration_constant * np.sqrt(np.log(node.num_visits) / (1 + action_nodes_visits))

        pomcp_exploration_values = action_nodes_q_values + pomcp_exploration_addition
        return node.children[np.argmax(pomcp_exploration_values)]

    def _rollout(self, belief: Belief, action: Any, current_depth: int) -> float:
        if current_depth > self.depth:
            return 0
        
        next_belief, _ = sample_next_belief(
            belief=belief,
            action=action,
            pomdp=self.environment
        )
        
        next_action = np.random.choice(self.environment.get_actions())
        cost_ = belief_expectation_cost(
            belief=next_belief,
            action=next_action,
            env=self.environment
        )
        
        return cost_ + self.discount_factor * self._rollout(next_belief, next_action, current_depth + 1)
    
    def _update_leaf_node_q_value(self, node: ActionNode):
        if node.num_visits == 0:  # update only if the node has not been visited
            node.q_value = belief_expectation_cost(
                belief=node.parent.belief,
                action=node.action,
                env=self.environment
            )

    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        children_v_values = [child.v_value for child in node.children]
        node.q_value = node.immediate_cost + self.discount_factor * np.mean(children_v_values)
    
    def _update_belief_node_v_value(self, node: BeliefNode):
        children_q_values = [child.q_value for child in node.children]
        node.v_value = np.min(children_q_values)
