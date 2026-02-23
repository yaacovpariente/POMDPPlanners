"""PFT-DPW (Progressive Function Transfer with Double Progressive Widening) Algorithm.

This module implements PFT-DPW, a Monte Carlo Tree Search algorithm for continuous
action spaces in POMDPs. The algorithm uses progressive widening to gradually expand
the action and observation spaces during tree search, enabling effective planning
in problems with continuous or large discrete action spaces.

Key features:
- Progressive widening for both actions and observations
- Handles continuous action spaces through adaptive sampling
- Uses UCB1-style exploration with progressive expansion
- Supports custom action samplers for domain-specific action generation

The algorithm progressively expands the tree by:
1. Using action progressive widening to add new actions based on visit counts
2. Using observation progressive widening to add new observation branches
3. Balancing exploration of new actions with exploitation of promising ones
4. Performing random rollouts from leaf nodes for value estimation

Classes:
    ActionSampler: Abstract base class for action sampling strategies
    PFT_DPW: Main PFT-DPW planner with progressive widening for continuous actions
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    DoubleProgressiveWideningMCTSPolicy,
)
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening,
)


class PFT_DPW(DoubleProgressiveWideningMCTSPolicy):
    """PFT-DPW (Progressive Function Transfer with Double Progressive Widening) Algorithm.

    PFT-DPW is a Monte Carlo Tree Search algorithm designed for continuous action spaces
    in POMDPs. It uses progressive widening to gradually expand both the action and
    observation spaces during tree search, enabling effective planning in problems with
    continuous or very large discrete action spaces.

    Algorithm Overview:
    The algorithm operates through progressive expansion:
    1. **Action Progressive Widening**: Gradually adds new actions based on visit counts
    2. **Observation Progressive Widening**: Gradually adds new observation branches
    3. **UCB1 Exploration**: Balances exploration of new actions with exploitation
    4. **Random Rollouts**: Estimates values from leaf nodes using random simulations

    Key Features:
    - Handles continuous action spaces through adaptive sampling
    - Uses UCB1-style exploration with progressive expansion
    - Supports custom action samplers for domain-specific action generation
    - Balances exploration of new actions with exploitation of promising ones
    - Performs random rollouts from leaf nodes for value estimation

    Progressive Widening Parameters:
    - k_a, alpha_a: Control action space expansion (more actions added as visit_count^alpha_a)
    - k_o, alpha_o: Control observation space expansion
    - exploration_constant: UCB1 exploration parameter (higher = more exploration)

    Attributes:
        environment: The POMDP environment to plan for
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        depth: Maximum search depth for tree expansion
        action_sampler: Strategy for sampling new actions during progressive widening
        k_a, alpha_a: Action progressive widening parameters
        k_o, alpha_o: Observation progressive widening parameters
        exploration_constant: UCB1 exploration parameter
        n_simulations: Number of simulations to run (mutually exclusive with timeout)
        time_out_in_seconds: Time limit for planning (mutually exclusive with n_simulations)

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Create environment and planner
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> action_sampler = DiscreteActionSampler(tiger.get_actions())
        >>> planner = PFT_DPW(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     name="ExamplePlanner",
        ...     action_sampler=action_sampler,
        ...     k_a=2.0,
        ...     alpha_a=0.5,
        ...     n_simulations=10
        ... )
        >>>
        >>> # Basic planner interface usage
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> # Action selection from belief
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> # Planner space information
        >>> space_info = PFT_DPW.get_space_info()
        >>> space_info.action_space.name
        'MIXED'
    """

    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        k_a: float = 1.0,
        alpha_a: float = 0.5,
        k_o: float = 1.0,
        alpha_o: float = 0.5,
        exploration_constant: float = 1.0,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize PFT_DPW planner.

        Args:
            environment: The POMDP environment to plan for
            discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
            depth: Maximum search depth for tree expansion
            name: Identifier for the policy instance
            action_sampler: Action sampling strategy for progressive widening
            k_a: Action progressive widening coefficient (default: 1.0)
            alpha_a: Action progressive widening exponent (default: 0.5)
            k_o: Observation progressive widening coefficient (default: 1.0)
            alpha_o: Observation progressive widening exponent (default: 0.5)
            exploration_constant: UCB1 exploration parameter (default: 1.0)
            time_out_in_seconds: Time limit for planning
            n_simulations: Number of simulations to run
            min_visit_count_per_action: Minimum visits per action (PFT_DPW-specific, default: 1)
            log_path: Optional path for logging
            debug: Enable debug logging
            use_queue_logger: Use queue-based logging

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters have invalid values
        """
        # Validate PFT_DPW-specific parameters before calling super
        self._validate_pft_dpw_params(
            min_visit_count_per_action=min_visit_count_per_action,
        )

        # Base class handles common parameter validation
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            name=name,
            action_sampler=action_sampler,
            k_a=k_a,
            alpha_a=alpha_a,
            k_o=k_o,
            alpha_o=alpha_o,
            exploration_constant=exploration_constant,
            min_visit_count_per_action=min_visit_count_per_action,
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # PFT_DPW-specific attributes
        self.min_visit_count_per_action = min_visit_count_per_action

    @staticmethod
    def _validate_pft_dpw_params(
        min_visit_count_per_action: int,
    ) -> None:
        """Validate PFT_DPW-specific parameters.

        Args:
            min_visit_count_per_action: Minimum visits per action

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters have invalid values
        """
        if not isinstance(min_visit_count_per_action, int):
            raise TypeError(
                f"min_visit_count_per_action must be an int, "
                f"got {type(min_visit_count_per_action).__name__}"
            )
        if min_visit_count_per_action < 1:
            raise ValueError(
                f"min_visit_count_per_action must be >= 1, got {min_visit_count_per_action}"
            )

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None
            return 0

        if self.environment.is_terminal(belief_node.belief.sample()):
            belief_node.visit_count += 1
            return 0

        action_node = action_progressive_widening(
            belief_node=belief_node,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a,
            min_visit_count_per_action=self.min_visit_count_per_action,
        )

        return_sample = self._simulate_return(
            belief_node=belief_node, action_node=action_node, depth=depth
        )

        self._update_node_statistics(
            belief_node=belief_node, action_node=action_node, total=return_sample
        )

        return return_sample

    def _simulate_return(
        self, belief_node: BeliefNode, action_node: ActionNode, depth: int
    ) -> float:
        if len(action_node.children) <= self.k_o * action_node.visit_count**self.alpha_o:
            next_belief_node, immediate_reward = self._sample_new_belief_node(
                belief_node=belief_node, action_node=action_node
            )
            state = next_belief_node.belief.sample()
            total = immediate_reward + self.discount_factor * self._random_rollout(
                state=state, depth=depth + 1
            )
        else:
            next_belief_node, immediate_reward = self.sample_existing_belief_node(
                belief_node=belief_node, action_node=action_node
            )
            total = immediate_reward + self.discount_factor * self._simulate_path(
                belief_node=next_belief_node, depth=depth + 1
            )

        return total

    def _sample_new_belief_node(
        self, belief_node: BeliefNode, action_node: ActionNode
    ) -> Tuple[BeliefNode, float]:
        immediate_reward = belief_expectation_reward(
            belief=belief_node.belief, action=action_node.action, env=self.environment
        )
        belief_node.immediate_cost = -immediate_reward

        next_state, next_observation, reward = self.environment.sample_next_step(
            state=belief_node.belief.sample(), action=action_node.action
        )
        next_belief = belief_node.belief.update(
            observation=next_observation,
            action=action_node.action,
            pomdp=self.environment,
        )
        next_belief_node = BeliefNode(belief=next_belief, parent=action_node)

        return next_belief_node, immediate_reward

    def _random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0

        action = self.action_sampler.sample()
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=action
        )

        return reward + self.discount_factor * self._random_rollout(
            state=next_state, depth=depth + 1
        )

    def sample_existing_belief_node(
        self, belief_node: BeliefNode, action_node: ActionNode
    ) -> Tuple[BeliefNode, float]:
        immediate_reward = -belief_node.immediate_cost  # type: ignore
        next_belief_node = action_node.sample_child_node()
        return next_belief_node, immediate_reward

    def _update_node_statistics(
        self, belief_node: BeliefNode, action_node: ActionNode, total: float
    ):
        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (total - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max([child.q_value for child in belief_node.children])
