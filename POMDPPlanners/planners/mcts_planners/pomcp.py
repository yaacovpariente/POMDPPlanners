from typing import Any, List
import random
import time
import numpy as np

from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.tree import ActionNode, get_optimal_action_reward_setting, BeliefNode
from POMDPPlanners.utils.tree_statistics import compute_tree_metrics

class POMCP(Policy):
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        depth: int, 
        exploration_constant: float,
        name: str,
        time_out_in_seconds: int = None, 
        n_simulations: int = None, 
        min_samples_per_node: int = 10
    ):
        combination1 = time_out_in_seconds is not None and n_simulations is None
        combination2 = time_out_in_seconds is None and n_simulations is not None
        assert combination1 or combination2, "Only one of time_out_in_seconds and n_simulations must be provided."
        
        super().__init__(environment=environment, discount_factor=discount_factor, name=name)
        
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.timeout_in_seconds = time_out_in_seconds
        self.n_simulations = n_simulations
        self.min_samples_per_node = min_samples_per_node

    def action(self, belief: Belief) -> List[Any]:
        tree = self.search(belief=belief)
        tree_metrics = compute_tree_metrics(tree=tree)
        action = get_optimal_action_reward_setting(belief_node=tree)
        return [action], PolicyRunData(info_variables=tree_metrics)
    
    def search(self, belief: Belief) -> Any:
        belief_node = BeliefNode(belief=belief, observation=None)
        
        if self.timeout_in_seconds is not None:
            self._construct_tree_using_timeout(belief=belief, belief_node=belief_node)
        else:
            self._construct_tree_using_n_simulations(belief=belief, belief_node=belief_node)

        return belief_node
    
    def _construct_tree_using_timeout(self, belief: Belief, belief_node: BeliefNode) -> BeliefNode:
        assert self.timeout_in_seconds is not None
        
        start_time = time.time()
        while time.time() - start_time < self.timeout_in_seconds:
            state = belief.sample()
            self.simulate(state=state, belief_node=belief_node, depth=0)
            
    def _construct_tree_using_n_simulations(self, belief: Belief, belief_node: BeliefNode) -> BeliefNode:
        assert self.n_simulations is not None
        
        for _ in range(self.n_simulations):
            state = belief.sample()
            self.simulate(state=state, belief_node=belief_node, depth=0)
    
    def simulate(self, state: Any, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth: 
            belief_node.parent = None  # remove the node from the tree
            return 0
        
        if self.environment.is_terminal(state=state):
            belief_node.visit_count += 1
            return 0

        if belief_node.is_leaf:
            for action in self.environment.get_actions():
                action_node = ActionNode(action=action, parent=belief_node, children=tuple())
                
            belief_node.visit_count += 1
            return self.random_rollout(state=state, depth=depth)            
        
        action_node = self.get_explored_action_node(belief_node=belief_node)
        
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action_node.action)
        
        next_belief_node = None
        for belief_node_child in action_node.children:
            if self.environment.is_equal_observation(observation1=next_observation, observation2=belief_node_child.observation):
                next_belief_node = belief_node_child
                break

        if next_belief_node is None:
            next_belief_node = BeliefNode(belief=belief_node.belief, observation=next_observation, parent=action_node, children=tuple())
            next_belief_node.update_belief(action=action_node.action, observation=next_observation, pomdp=self.environment)
            
        return_sample = reward + self.discount_factor * self.simulate(
            state=next_state, 
            belief_node=next_belief_node, 
            depth=depth + 1
        )
        
        self.update_nodes(belief_node=belief_node, action_node=action_node, return_sample=return_sample)
            
        return return_sample
        
    def get_explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        assert isinstance(belief_node, BeliefNode)
        
        action_nodes_visits = np.array([child.visit_count for child in belief_node.children])
        unvisited_action_indices = np.where(action_nodes_visits == 0)[0]
        if len(unvisited_action_indices) > 0:
            return belief_node.children[np.random.choice(unvisited_action_indices)]

        action_nodes_q_values = np.array([child.q_value for child in belief_node.children])
        ucb = action_nodes_q_values + self.exploration_constant * np.sqrt(np.log(belief_node.visit_count) / action_nodes_visits)

        return belief_node.children[np.argmax(ucb)]
        
    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        
        action = random.choice(self.environment.get_actions())
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action)
        
        return reward + self.discount_factor * self.random_rollout(state=next_state, depth=depth + 1)
        
    def update_nodes(self, belief_node: BeliefNode, action_node: ActionNode, return_sample: float):
        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (return_sample - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max([child.q_value for child in belief_node.children if child.visit_count > 0])

    def get_space_info(self) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )
