"""ICVaR PFT-DPW (Iterated CVaR Particle Filter Tree with Double Progressive Widening) Algorithm.

This module implements a risk-sensitive variant of PFT-DPW that uses the Iterated Conditional
Value at Risk (ICVaR) for value backups instead of the expected value. This makes the planner
focus on the worst-alpha fraction of outcomes, enabling risk-averse planning in POMDPs.

Reference:
    Pariente, Y., & Indelman, V. (2026). Online Risk-Averse Planning in POMDPs Using
    Iterated CVaR Value Function. arXiv preprint arXiv:2601.20554.
    https://arxiv.org/abs/2601.20554

Classes:
    ICVaR_PFT_DPW: Risk-sensitive PFT-DPW planner with CVaR-based value updates
"""

from typing import Tuple, Optional
import numpy as np

from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.cost import belief_expectation_cost_entropy_penalty
from POMDPPlanners.utils.statistics_utils import cvar_estimator_from_dist
from POMDPPlanners.utils.statistics_utils import get_min_and_max_cost
from POMDPPlanners.core.belief import Belief, is_terminal_belief
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.path_simulations_policy import (
    PathSimulationPolicyCostSetting,
)
from POMDPPlanners.planners.planners_utils.cvar_progressive_widening import (
    cvar_action_progressive_widening,
)


class ICVaR_PFT_DPW(PathSimulationPolicyCostSetting):
    def __init__(
        self,
        environment: Environment,
        name: str,
        depth: int,
        action_sampler: ActionSampler,
        discount_factor: float = 0.95,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        alpha: float = 0.1,
        delta: float = 0.1,
        belief_child_num: int = 5,
        min_immediate_cost: float = 0.0,
        max_immediate_cost: float = 1.0,
        min_visit_count_per_action: int = 1,
        exploration_constant: float = 1.0,
        k_a: float = 1.0,
        alpha_a: float = 0.5,
        k_o: float = 1.0,
        alpha_o: float = 0.5,
        entropy_weight: float = 0.0,
        visit_count_penalty: float = 0.0,
    ):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            action_sampler=action_sampler,
            time_out_in_seconds=time_out_in_seconds,
        )

        assert isinstance(alpha, float) and 0 <= alpha <= 1, "alpha must be a float between 0 and 1"
        assert isinstance(delta, float) and 0 <= delta <= 1, "delta must be a float between 0 and 1"
        assert isinstance(min_immediate_cost, (int, float)), "min_immediate_cost must be a number"
        assert isinstance(max_immediate_cost, (int, float)), "max_immediate_cost must be a number"
        assert (
            min_immediate_cost <= max_immediate_cost
        ), "min_immediate_cost must be less than or equal to max_immediate_cost"

        self.alpha = alpha
        self.delta = delta
        self.depth = depth
        self.max_depth = depth  # max_depth should be same as depth for planning horizon
        self.min_immediate_cost = min_immediate_cost
        self.max_immediate_cost = max_immediate_cost
        self.min_visit_count_per_action = min_visit_count_per_action
        self.belief_child_num = belief_child_num
        self.exploration_constant = exploration_constant
        self.action_sampler = action_sampler
        self.k_a = k_a
        self.alpha_a = alpha_a
        self.k_o = k_o
        self.alpha_o = alpha_o
        self.entropy_weight = entropy_weight
        self.discrete_actions = self.environment.space_info.action_space == SpaceType.DISCRETE
        self.visit_count_penalty = visit_count_penalty

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> None:
        if depth > self.depth:
            belief_node.parent = None
            return

        if self.is_terminal_belief(belief=belief_node.belief):
            belief_node.visit_count += 1
            return

        action_node = cvar_action_progressive_widening(
            belief_node=belief_node,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a,
            min_immediate_cost=self.min_immediate_cost,
            max_immediate_cost=self.max_immediate_cost,
            depth=self.depth,
            max_depth=self.max_depth,
            gamma=self.discount_factor,
            min_visit_count_per_action=self.min_visit_count_per_action,
            alpha=self.alpha,
            delta=self.delta,
            discrete_actions=self.discrete_actions,
            visit_count_penalty=self.visit_count_penalty,
        )

        if len(action_node.children) <= self.k_o * action_node.visit_count**self.alpha_o:
            next_belief_node, _ = self._generate_belief(action_node=action_node)
        else:
            next_belief_node, _ = self._sample_next_existing_belief(action_node=action_node)

        self._simulate_path(belief_node=next_belief_node, depth=depth + 1)

        self.update_nodes(belief_node=belief_node, action_node=action_node)

    def is_terminal_belief(self, belief: Belief) -> bool:
        """Checks if all paricles are terminal states."""
        return is_terminal_belief(belief=belief, env=self.environment)

    def _sample_next_existing_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        child_visit_counts = np.array([child.visit_count for child in action_node.children])
        min_visit_count_idx = np.argmin(child_visit_counts)
        if child_visit_counts[min_visit_count_idx] == 0:
            # If no children have been visited, randomly select one and return with its immediate cost
            sampled_belief_node = action_node.children[min_visit_count_idx]
            expected_reward = -sampled_belief_node.immediate_cost
            return sampled_belief_node, float(expected_reward)

        weights = child_visit_counts / sum(child_visit_counts)
        sampled_belief_node = np.random.choice(action_node.children, p=weights)
        expected_reward = -sampled_belief_node.immediate_cost
        return sampled_belief_node, float(expected_reward)

    def _generate_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        if action_node.parent is None:
            raise ValueError("Action node must have a parent belief node")
        belief = action_node.parent.belief
        state = belief.sample()
        next_state = self.environment.state_transition_model(
            state=state, action=action_node.action
        ).sample()[0]
        next_observation = self.environment.observation_model(
            next_state=next_state, action=action_node.action
        ).sample()[0]

        next_belief = belief.update(
            action=action_node.action, observation=next_observation, pomdp=self.environment
        )

        next_belief_node = BeliefNode(
            belief=next_belief, observation=next_observation, parent=action_node
        )
        min_cost, max_cost = get_min_and_max_cost(
            min_immediate_cost=self.min_immediate_cost,
            max_immediate_cost=self.max_immediate_cost,
            depth=self.depth,
            max_depth=self.max_depth,
            gamma=self.discount_factor,
        )
        next_belief_node.immediate_cost = belief_expectation_cost_entropy_penalty(
            belief=belief,
            action=action_node.action,
            env=self.environment,
            entropy_weight=self.entropy_weight,
            lower_clip=min_cost,
            upper_clip=max_cost,
        )
        immediate_reward = -next_belief_node.immediate_cost

        return next_belief_node, immediate_reward

    def update_nodes(self, belief_node: BeliefNode, action_node: ActionNode):
        belief_node.visit_count += 1  # Only increment once
        action_node.visit_count += 1

        if action_node.immediate_cost is None:
            min_cost, max_cost = get_min_and_max_cost(
                min_immediate_cost=self.min_immediate_cost,
                max_immediate_cost=self.max_immediate_cost,
                depth=self.depth,
                max_depth=self.max_depth,
                gamma=self.discount_factor,
            )
            action_node.immediate_cost = belief_expectation_cost_entropy_penalty(
                belief=belief_node.belief,
                action=action_node.action,
                env=self.environment,
                entropy_weight=self.entropy_weight,
                lower_clip=min_cost,
                upper_clip=max_cost,
            )

        if action_node.is_leaf:
            action_node.q_value = action_node.immediate_cost
        else:
            visit_counts = np.array([child.visit_count for child in action_node.children])
            v_values = np.array([child.v_value for child in action_node.children])
            action_node.q_value = (
                action_node.immediate_cost
                + self.discount_factor
                * cvar_estimator_from_dist(
                    values=v_values, weights=visit_counts / sum(visit_counts), alpha=self.alpha
                )
            )

        belief_node.v_value = np.min(
            [child.q_value for child in belief_node.children if child.visit_count > 0]
        )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get information about the policy's space."""
        return PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS,
        )
