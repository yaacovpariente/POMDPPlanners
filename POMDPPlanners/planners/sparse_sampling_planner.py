from typing import Any, List, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType

from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.tree import ActionNode, BeliefNode, get_optimal_action_cost_setting
from POMDPPlanners.core.cost import belief_expectation_cost

class SparseSamplingDiscreteActionsPlanner(Policy, ABC):
    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        branching_factor: int,
        depth: int,
        resampling: bool = False,
        name: str = "SparseSamplingDiscreteActionsPlanner",
        log_path: Optional[Path] = None,
        debug: bool = False
    ):
        if not isinstance(environment, DiscreteActionsEnvironment):
            raise TypeError("environment must be a DiscreteActionsEnvironment instance")
        if not isinstance(branching_factor, int):
            raise TypeError("branching_factor must be an int")
        if not isinstance(depth, int):
            raise TypeError("depth must be an int")
        if not isinstance(resampling, bool):
            raise TypeError("resampling must be a bool")
        if depth <= 0:
            raise ValueError("Depth must be greater than 0")
        if branching_factor <= 0:
            raise ValueError("Branching factor must be greater than 0")

        super().__init__(
            environment=environment, 
            discount_factor=environment.discount_factor,
            name=name,
            log_path=log_path,
            debug=debug
        )

        self.branching_factor = branching_factor
        self.depth = depth

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        if not isinstance(belief, Belief):
            raise TypeError("belief must be a Belief instance")

        tree = self._learn_belief_tree(belief)
        action = get_optimal_action_cost_setting(tree)
        return [action], PolicyRunData(info_variables=[])

    def _learn_belief_tree(self, belief: Belief) -> BeliefNode:
        tree = BeliefNode(belief=belief)

        self._build_tree(belief_node=tree, current_depth=0)
        self._update_node_statistics(tree)

        return tree

    def _build_tree(self, belief_node: BeliefNode, current_depth: int):
        if current_depth == self.depth:
            self._set_last_belief_node(belief_node)
            return

        for action in self.environment.get_actions():
            child = ActionNode(
                action=action, parent=belief_node, children=tuple(), data=None
            )

            state = belief_node.belief.sample()
            next_state = self.environment.state_transition_model(state, action).sample()[0]
            next_observation = self.environment.observation_model(
                next_state, action
            ).sample()[0]

            next_belief = belief_node.belief.update(
                action=action, observation=next_observation, pomdp=self.environment
            )
            next_belief_node = BeliefNode(
                belief=next_belief, parent=child, children=tuple(), data=None
            )

            self._build_tree(
                belief_node=next_belief_node, current_depth=current_depth + 1
            )

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
        if not isinstance(node, ActionNode):
            raise TypeError("node must be an ActionNode instance")
        if node.height != 0:
            raise ValueError("node.height must be 0")
        node.visit_count = 1

        self._update_leaf_node_q_value(node)

    @abstractmethod
    def _update_leaf_node_q_value(self, node: ActionNode):
        pass

    def _update_non_leaf_action_node_statistics(self, node: ActionNode):
        node.visit_count += 1
        self._update_non_leaf_action_node_q_value(node)

    @abstractmethod
    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        pass

    def _update_belief_node_statistics(self, node: BeliefNode):
        node.visit_count = sum([child.visit_count for child in node.children])
        self._update_belief_node_v_value(node)

    @abstractmethod
    def _update_belief_node_v_value(self, node: BeliefNode):
        pass

    def _set_last_belief_node(self, node: BeliefNode):
        for action in self.environment.get_actions():
            child = ActionNode(action=action, parent=node, children=tuple(), data=None)
            
    def get_space_info(self) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.MIXED
        )

    def random_rollout(self, belief_node: BeliefNode, depth: int) -> float:
        if depth == 0:
            return 0.0

        state = belief_node.belief.sample()
        action = np.random.choice(self.environment.get_actions())
        next_state = self.environment.state_transition_model(state, action).sample()[0]
        next_observation = self.environment.observation_model(
            next_state=next_state, action=action
        ).sample()[0]

        state = belief_node.belief.sample()
        next_state = self.environment.state_transition_model(state, action).sample()[0]
        next_observation = self.environment.observation_model(
            next_state=next_state, action=action
        ).sample()[0]

class StandardSparseSamplingDiscreteActionsPlanner(
    SparseSamplingDiscreteActionsPlanner
):
    def __init__(
        self, environment: DiscreteActionsEnvironment, branching_factor: int, depth: int, name: str = "StandardSparseSamplingDiscreteActionsPlanner"
    ):
        super().__init__(
            environment=environment, branching_factor=branching_factor, depth=depth, name=name
        )

    def _update_leaf_node_q_value(self, node: ActionNode):
        node.immediate_cost = belief_expectation_cost(
            belief=node.parent.belief, action=node.action, env=self.environment
        )
        node.q_value = node.immediate_cost

    def _update_non_leaf_action_node_q_value(self, node: ActionNode):
        if node.immediate_cost is None:
            node.immediate_cost = belief_expectation_cost(
                belief=node.parent.belief, action=node.action, env=self.environment
            )
        children_q_values = [child.v_value for child in node.children]
        node.q_value = node.immediate_cost + self.environment.discount_factor * np.mean(
            children_q_values
        )

    def _update_belief_node_v_value(self, node: BeliefNode):
        node.v_value = min([child.q_value for child in node.children])
