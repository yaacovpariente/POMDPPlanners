from abc import abstractmethod
import time
from typing import Any, List, Tuple

from POMDPPlanners.core.policy import Policy, PolicyRunData
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.tree import BeliefNode, get_optimal_action_reward_setting
from POMDPPlanners.utils.tree_statistics import compute_tree_metrics

class PathSimulationPolicy(Policy):
    def __init__(
        self, 
        environment: "Environment", 
        discount_factor: float, 
        name: str, 
        n_simulations: int, 
        time_out_in_seconds: int
    ):
        super().__init__(
            environment=environment, 
            discount_factor=discount_factor, 
            name=name
        )

        self.n_simulations = n_simulations
        self.time_out_in_seconds = time_out_in_seconds
        
        assert not (n_simulations is not None and time_out_in_seconds is not None), "Cannot specify both n_simulations and time_out_in_seconds"

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        tree = self._learn_tree(belief=belief)
        tree_metrics = compute_tree_metrics(tree=tree)
        action = get_optimal_action_reward_setting(belief_node=tree)
        return [action], PolicyRunData(info_variables=tree_metrics)
    
    def _learn_tree(self, belief: Belief) -> BeliefNode:
        tree = BeliefNode(belief=belief)
        
        if self.n_simulations is not None:
            self._construct_tree_using_n_simulations(belief_node=tree)
        else:
            self._construct_tree_using_timeout(belief_node=tree)
        
        return tree
    
    def _construct_tree_using_n_simulations(self, belief_node: BeliefNode) -> BeliefNode:
        assert self.n_simulations is not None
        
        for _ in range(self.n_simulations):
            self._simulate_path(belief_node=belief_node, depth=0)

    def _construct_tree_using_timeout(self, belief_node: BeliefNode) -> BeliefNode:
        assert self.time_out_in_seconds is not None
        
        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:
            self._simulate_path(belief_node=belief_node, depth=0)

    @abstractmethod
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        pass
