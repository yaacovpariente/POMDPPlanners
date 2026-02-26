"""ICVaR POMCPOW (Iterated CVaR POMCPOW) Algorithm.

This module implements a risk-sensitive variant of POMCPOW that uses the Iterated Conditional
Value at Risk (ICVaR) for value backups instead of the expected value. This makes the planner
focus on the worst-alpha fraction of outcomes, enabling risk-averse planning in POMDPs with
continuous state, action, and observation spaces.

Reference:
    Pariente, Y., & Indelman, V. (2026). Online Risk-Averse Planning in POMDPs Using
    Iterated CVaR Value Function. arXiv preprint arXiv:2601.20554.
    https://arxiv.org/abs/2601.20554

Classes:
    ICVaR_POMCPOW: Risk-sensitive POMCPOW planner with CVaR-based value updates
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate, WeightedParticleBelief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.planners_utils.path_simulations_policy import (
    PathSimulationPolicyCostSetting,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.statistics_utils import cvar_estimator_from_dist
from POMDPPlanners.planners.planners_utils.cvar_progressive_widening import (
    cvar_action_progressive_widening,
)


class ICVaR_POMCPOW(PathSimulationPolicyCostSetting):
    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        depth: int,
        exploration_constant: float,
        k_o: float,
        k_a: float,
        alpha_o: float,
        alpha_a: float,
        min_immediate_cost: float,
        max_immediate_cost: float,
        min_visit_count_per_action: int,
        delta: float,
        name: str,
        action_sampler: ActionSampler,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        alpha: float = 0.05,
        min_samples_per_node: int = 10,
        log_path: Optional[Path] = None,
        debug: bool = False,
        visit_count_penalty: float = 0.0,
    ):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            action_sampler=action_sampler,
            time_out_in_seconds=time_out_in_seconds,
            log_path=log_path,
            debug=debug,
        )

        self.depth = depth
        self.exploration_constant = exploration_constant
        self.min_samples_per_node = min_samples_per_node
        self.action_sampler = action_sampler
        self.alpha = alpha

        self.k_o = k_o
        self.k_a = k_a
        self.alpha_o = alpha_o
        self.alpha_a = alpha_a

        self.min_immediate_cost = min_immediate_cost
        self.max_immediate_cost = max_immediate_cost
        self.min_visit_count_per_action = min_visit_count_per_action
        self.delta = delta
        self.max_depth = depth
        self.visit_count_penalty = visit_count_penalty

        self.discrete_actions = self.environment.space_info.action_space == SpaceType.DISCRETE

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> None:
        """Simulate a single MCTS path from belief node using sampled state.

        This method samples a state from the belief and delegates to _simulate_state_path
        for the actual simulation logic. This separation allows for different belief
        sampling strategies while maintaining the core simulation algorithm.

        Args:
            belief_node: Current belief node in the search tree
            depth: Current depth in the search tree
        """
        state = belief_node.belief.sample()
        self._simulate_state_path(state=state, belief_node=belief_node, depth=depth)

    def _simulate_state_path(self, state: Any, belief_node: BeliefNode, depth: int) -> None:
        """Simulate MCTS path from given state and belief node with progressive widening.

        This is the core simulation method that implements the POMCPOW algorithm:
        1. Check termination conditions (depth limit, terminal state)
        2. Select/add action using action progressive widening
        3. Sample next state and observation from environment
        4. Select/add observation node using observation progressive widening
        5. Update weighted particle belief in observation node
        6. Recursively continue simulation or perform rollout
        7. Backpropagate value updates

        Args:
            state: Current state to simulate from
            belief_node: Current belief node in search tree
            depth: Current depth in search tree

        Returns:
            Total discounted return from this simulation path
        """
        if self._check_termination_conditions(state=state, belief_node=belief_node, depth=depth):
            return

        action_node = self._select_action_with_progressive_widening(belief_node=belief_node)
        next_state, next_observation, reward = self._sample_environment_step(
            state=state, action_node=action_node
        )
        next_belief_node = self._select_or_create_observation_node(
            action_node=action_node, next_observation=next_observation
        )
        self._update_belief_with_state(
            belief_node=next_belief_node,
            action_node=action_node,
            observation=next_observation,
            state=next_state,
        )

        self._simulate_state_path(state=next_state, belief_node=next_belief_node, depth=depth + 1)

        self._backpropagate_values(
            state=state, belief_node=belief_node, action_node=action_node, reward=reward
        )

    def _check_termination_conditions(
        self, state: Any, belief_node: BeliefNode, depth: int
    ) -> bool:
        """Check if simulation should terminate due to depth limit or terminal state.

        Args:
            state: Current state to check
            belief_node: Current belief node in search tree
            depth: Current depth in search tree

        Returns:
            True if simulation should terminate, False otherwise
        """
        if depth > self.depth:
            belief_node.parent = None
            return True

        if self.environment.is_terminal(state=state):
            belief_node.visit_count += 1
            return True

        return False

    def _select_action_with_progressive_widening(self, belief_node: BeliefNode) -> Any:
        """Select or add action node using action progressive widening.

        Args:
            belief_node: Current belief node in search tree

        Returns:
            Selected action node
        """
        return cvar_action_progressive_widening(
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

    def _sample_environment_step(self, state: Any, action_node: Any) -> Tuple[Any, Any, float]:
        """Sample next state, observation, and reward from environment.

        Args:
            state: Current state
            action_node: Action node containing the action to take

        Returns:
            Tuple of (next_state, next_observation, reward)
        """
        return self.environment.sample_next_step(state=state, action=action_node.action)

    def _select_or_create_observation_node(
        self, action_node: Any, next_observation: Any
    ) -> BeliefNode:
        """Select or create observation node using observation progressive widening.

        Args:
            action_node: Action node to get children from
            next_observation: Observed observation

        Returns:
            Selected or newly created belief node for the observation
        """
        if len(action_node.children) <= self.k_o * action_node.visit_count**self.alpha_o:
            next_belief_node = action_node.get_belief_node_child(
                observation=next_observation, environment=self.environment
            )
            if next_belief_node is None:
                next_belief_node = BeliefNode(
                    belief=WeightedParticleBeliefStateUpdate(particles=[], weights=[]),
                    observation=next_observation,
                    parent=action_node,
                    weight=0,
                )

            next_belief_node.weight += 1
        else:
            next_belief_node = action_node.sample_child_node()

        return next_belief_node

    def _update_belief_with_state(
        self, belief_node: BeliefNode, action_node: Any, observation: Any, state: Any
    ) -> None:
        """Update weighted particle belief with new state information.

        Args:
            belief_node: Belief node to update
            action_node: Action node containing the action taken
            observation: Observation received
            state: New state to incorporate into belief
        """
        belief_node.belief.inplace_update(
            action=action_node.action, observation=observation, pomdp=self.environment, state=state
        )

    def _backpropagate_values(
        self, state: Any, belief_node: BeliefNode, action_node: Any, reward: float
    ) -> None:
        """Backpropagate value updates through the search tree.

        Updates visit counts, immediate costs, Q-values, and V-values based on
        the simulation results.

        Args:
            state: State that was used in the simulation
            belief_node: Belief node to update
            action_node: Action node to update
            reward: Reward received from the environment step
        """
        belief_node.visit_count += 1
        action_node.visit_count += 1

        self._update_immediate_cost(
            state=state, belief_node=belief_node, action_node=action_node, reward=reward
        )
        self._update_q_value(action_node=action_node)
        self._update_v_value(belief_node=belief_node)

    def _update_immediate_cost(
        self, state: Any, belief_node: BeliefNode, action_node: Any, reward: float
    ) -> None:  # pylint: disable=unused-argument
        """Update the immediate cost estimate for an action node.

        Args:
            state: State used in the simulation
            belief_node: Belief node containing belief information
            action_node: Action node to update
            reward: Reward received from the environment step
        """
        if action_node.immediate_cost is None:
            if isinstance(belief_node.belief, WeightedParticleBeliefStateUpdate):
                action_node.immediate_cost = -reward
            elif isinstance(belief_node.belief, WeightedParticleBelief):
                particle_weights = belief_node.belief.normalized_weights
                particle_costs = np.array(
                    [
                        -self.environment.reward(state=particle, action=action_node.action)
                        for particle in belief_node.belief.particles
                    ]
                )
                action_node.immediate_cost = sum(particle_costs * particle_weights)
            else:
                raise ValueError(f"Unsupported belief type: {type(belief_node.belief)}")
        elif isinstance(belief_node.belief, WeightedParticleBeliefStateUpdate):
            old_weights_sum = belief_node.belief.weights_sum - belief_node.belief.weights[-1]
            action_node.immediate_cost = (
                action_node.immediate_cost * old_weights_sum
                - reward * belief_node.belief.weights[-1]
            )
            action_node.immediate_cost /= belief_node.belief.weights_sum  # type: ignore

    def _update_q_value(self, action_node: Any) -> None:
        """Update the Q-value for an action node based on its children.

        Args:
            action_node: Action node to update
        """
        if not action_node.is_leaf:
            # Only use children that have been visited (weight > 0 or visit_count > 0)
            visited_children = [child for child in action_node.children if child.visit_count > 0]
            if len(visited_children) == 0:
                # If no children have been visited yet, use immediate cost only
                action_node.q_value = action_node.immediate_cost
            else:
                children_v_values = np.array([child.v_value for child in visited_children])
                children_weights = np.array([child.weight for child in visited_children])
                children_weights = children_weights / children_weights.sum()

                action_node.q_value = (
                    action_node.immediate_cost
                    + self.discount_factor
                    * cvar_estimator_from_dist(
                        values=children_v_values, weights=children_weights, alpha=self.alpha
                    )
                )
        else:
            action_node.q_value = action_node.immediate_cost

    def _update_v_value(self, belief_node: BeliefNode) -> None:
        """Update the V-value for a belief node based on its children.

        Args:
            belief_node: Belief node to update
        """
        # Only consider children that have been visited
        visited_children = [child for child in belief_node.children if child.visit_count > 0]
        if len(visited_children) > 0:
            belief_node.v_value = min(child.q_value for child in visited_children)
        # If no children have been visited, v_value remains at its current value (or 0.0 if uninitialized)

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get information about action and observation spaces.

        POMCPOW supports mixed-type spaces through its action sampler interface,
        allowing it to handle both discrete and continuous action spaces.

        Returns:
            PolicySpaceInfo with MIXED space types for both actions and observations
        """
        return PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)
