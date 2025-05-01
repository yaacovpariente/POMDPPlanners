import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.cost import belief_expectation_cost
from POMDPPlanners.planners.mcts_planners.custom_pomcp_planners import BeliefBasedMCTS

class POMCPPlanner(BeliefBasedMCTS):
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
            depth=depth,
            name="POMCPPlanner"
        )
        
        self.exploration_constant = exploration_constant
        
    def _get_explored_action_node(self, node: BeliefNode) -> ActionNode:
        assert isinstance(node, BeliefNode)
        
        action_nodes_q_values = np.array([child.q_value for child in node.children])
        action_nodes_visits = np.array([child.visit_count for child in node.children])
        pomcp_exploration_addition = self.exploration_constant * np.sqrt(np.log(node.visit_count) / (1 + action_nodes_visits))

        pomcp_exploration_values = action_nodes_q_values - pomcp_exploration_addition
        return node.children[np.argmin(pomcp_exploration_values)]

    def _update_leaf_node_q_value(self, node: ActionNode):
        if node.immediate_cost is None:
            node.immediate_cost = belief_expectation_cost(
                belief=node.parent.belief,
                action=node.action,
                env=self.environment
            )
            
        node.q_value = node.immediate_cost

    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        if node.immediate_cost is None:
            node.immediate_cost = belief_expectation_cost(
                belief=node.parent.belief,
                action=node.action,
                env=self.environment
            )
        
        children_visit_counts = np.array([child.visit_count for child in node.children if child.visit_count > 0])
        children_v_values = np.array([child.v_value for child in node.children if child.visit_count > 0])
        node.q_value = node.immediate_cost + self.discount_factor * np.sum(children_v_values * children_visit_counts) / np.sum(children_visit_counts)
        # TODO: change to updating mean
    def _update_non_leaf_belief_node_v_value(self, node: BeliefNode):
        children_q_values = np.array([child.q_value for child in node.children if child.q_value != -float('inf')])
        node.v_value = np.min(children_q_values)
