"""POMCP_DPW (Partially Observable Monte Carlo Planning with Double Progressive Widening) Algorithm.

This module implements POMCP_DPW, an advanced Monte Carlo Tree Search algorithm for POMDP
planning that extends POMCP with double progressive widening capabilities. POMCP_DPW combines
UCB1 action selection with progressive widening for both actions and observations, making
it particularly effective for problems with large or continuous action spaces.

Key features:
- Double progressive widening for actions and observations
- Unweighted particle-based belief representation (following POMCP tradition)
- UCB1-based exploration-exploitation balance
- Handles continuous and discrete action spaces
- Adaptive observation node expansion

The algorithm progressively expands the tree by:
1. Using action progressive widening to add new actions based on visit counts and α_a parameter
2. Using observation progressive widening to add new observation branches based on k_o and α_o
3. Maintaining unweighted particle beliefs in observation nodes (as per POMCP)
4. Balancing exploration of new actions with exploitation of promising ones
5. Performing random rollouts from leaf nodes for value estimation

Classes:
    POMCP_DPW: Monte Carlo Tree Search planner with double progressive widening extending POMCP
"""

from pathlib import Path
from typing import Any, Optional


# Python 3.9 compatibility - KW_ONLY was introduced in Python 3.10

from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    DoubleProgressiveWideningMCTSPolicy,
)
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening,
)
from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler


class POMCP_DPW(DoubleProgressiveWideningMCTSPolicy):
    """POMCP_DPW (Partially Observable Monte Carlo Planning with Double Progressive Widening) Algorithm.

    POMCP_DPW is an advanced Monte Carlo Tree Search algorithm for POMDP planning that extends
    POMCP with double progressive widening. It combines UCB1 action selection with progressive
    widening for both actions and observations, making it particularly effective for problems
    with large or continuous action spaces.

    Algorithm Overview:
    The algorithm operates through double progressive expansion:
    1. **Action Progressive Widening**: Gradually adds new actions based on visit counts and α_a
    2. **Observation Progressive Widening**: Gradually adds new observation branches based on k_o and α_o
    3. **Unweighted Particle Beliefs**: Maintains unweighted particle representations in observation nodes (POMCP tradition)
    4. **UCB1 Exploration**: Balances exploration of new actions with exploitation using UCB1
    5. **Random Rollouts**: Estimates values from leaf nodes using random simulations

    Key Features:
    - Handles continuous and discrete action spaces through ActionSampler interface
    - Uses double progressive widening to manage tree growth
    - Maintains unweighted particle beliefs for efficient belief approximation (following POMCP tradition)
    - Balances exploration of new actions with exploitation of promising ones
    - Supports configurable progressive widening parameters

    Progressive Widening Parameters:
    - **k_a, α_a**: Control action progressive widening (new actions added when ⌊n^α_a⌋ > ⌊(n-1)^α_a⌋)
    - **k_o, α_o**: Control observation progressive widening (max observations ≤ k_o * n^α_o)

    Attributes:
        environment: The POMDP environment to plan for
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        depth: Maximum search depth for tree expansion
        exploration_constant: UCB1 exploration parameter (higher = more exploration)
        k_o: Observation progressive widening coefficient
        k_a: Action progressive widening coefficient
        alpha_o: Observation progressive widening exponent
        alpha_a: Action progressive widening exponent
        action_sampler: Action sampling strategy for progressive widening
        time_out_in_seconds: Time limit for planning (mutually exclusive with n_simulations)
        n_simulations: Number of simulations to run (mutually exclusive with timeout)
        min_samples_per_node: Minimum samples before a node is considered reliable
        log_path: Optional path for logging policy execution
        debug: Enable debug logging if True

    Example:
        >>> import random
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Create action sampler
        >>> class DiscreteActionSampler(ActionSampler):
        ...     def __init__(self, actions):
        ...         self.actions = actions
        ...     def sample(self, belief_node=None):
        ...         return random.choice(self.actions)
        >>>
        >>> # Create environment and planner
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> action_sampler = DiscreteActionSampler(tiger.get_actions())
        >>> planner = POMCP_DPW(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     exploration_constant=1.0,
        ...     k_o=3.0,
        ...     k_a=3.0,
        ...     alpha_o=0.5,
        ...     alpha_a=0.5,
        ...     action_sampler=action_sampler,
        ...     n_simulations=10,
        ...     name="ExamplePlanner"
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
        >>> space_info = POMCP_DPW.get_space_info()
        >>> space_info.action_space.name
        'MIXED'
    """

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
        name: str,
        action_sampler: ActionSampler,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_samples_per_node: int = 10,
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize POMCP_DPW planner with double progressive widening parameters.

        Args:
            environment: The POMDP environment to plan for
            discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
            depth: Maximum search depth for tree expansion
            exploration_constant: UCB1 exploration parameter (higher = more exploration)
            k_o: Observation progressive widening coefficient (controls max observations)
            k_a: Action progressive widening coefficient (controls action expansion)
            alpha_o: Observation progressive widening exponent (0 < α_o ≤ 1)
            alpha_a: Action progressive widening exponent (0 < α_a ≤ 1)
            name: Identifier for the policy instance
            action_sampler: Action sampling strategy for progressive widening
            time_out_in_seconds: Time limit for planning in seconds (mutually exclusive with n_simulations)
            n_simulations: Number of MCTS simulations to run (mutually exclusive with time_out_in_seconds)
            min_samples_per_node: Minimum samples before a node is considered reliable
            log_path: Optional path for logging policy execution
            debug: Enable debug logging if True

        Raises:
            ValueError: If both time_out_in_seconds and n_simulations are provided or both are None
            TypeError: If parameters have incorrect types
            ValueError: If parameters have invalid values
        """
        # All validation and initialization is handled by the base class
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
            min_samples_per_node=min_samples_per_node,
            min_visit_count_per_action=min_visit_count_per_action,
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        # No POMCP_DPW-specific attributes needed

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        """Simulate a single MCTS path from belief node using sampled state.

        This method samples a state from the belief and delegates to _simulate_state_path
        for the actual simulation logic. This separation allows for different belief
        sampling strategies while maintaining the core simulation algorithm.

        Args:
            belief_node: Current belief node in the search tree
            depth: Current depth in the search tree

        Returns:
            Total discounted return from this simulation path
        """
        state = belief_node.belief.sample()
        return self._simulate_state_path(state=state, belief_node=belief_node, depth=depth)

    def _simulate_state_path(self, state: Any, belief_node: BeliefNode, depth: int) -> float:
        """Simulate MCTS path from given state and belief node with progressive widening.

        This is the core simulation method that implements the POMCP_DPW algorithm:
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
        if depth > self.depth:
            belief_node.parent = None
            return 0

        if self.environment.is_terminal(state=state):
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

        if len(action_node.children) <= self.k_o * action_node.visit_count**self.alpha_o:
            next_state = self.environment.state_transition_model(
                state=state, action=action_node.action
            ).sample()[0]
            reward = self.environment.reward(state=state, action=action_node.action)
            next_observation = self.environment.observation_model(
                next_state=next_state, action=action_node.action
            ).sample()[0]

            next_belief_node = action_node.get_belief_node_child(
                observation=next_observation, environment=self.environment
            )
            if next_belief_node is None:
                next_belief_node = BeliefNode(
                    belief=UnweightedParticleBeliefStateUpdate(),
                    observation=next_observation,
                    parent=action_node,
                    weight=0,
                )

            next_belief_node.belief.inplace_update(
                action=None, observation=None, pomdp=self.environment, state=next_state
            )
            next_belief_node.weight += 1

            if next_belief_node.visit_count == 0:
                next_belief_node.visit_count += 1
                total = reward + self.discount_factor * random_rollout_action_sampler(
                    state=next_state,
                    depth=depth + 1,
                    action_sampler=self.action_sampler,
                    environment=self.environment,
                    discount_factor=self.discount_factor,
                )
            else:
                total = reward + self.discount_factor * self._simulate_state_path(
                    state=next_state, belief_node=next_belief_node, depth=depth + 1
                )
        else:
            next_belief_node = action_node.sample_child_node()
            next_state = next_belief_node.belief.sample()
            reward = self.environment.reward(state=next_state, action=action_node.action)
            total = reward + self.discount_factor * self._simulate_state_path(
                state=next_state, belief_node=next_belief_node, depth=depth + 1
            )

        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (total - action_node.q_value) / action_node.visit_count
        belief_node.v_value = max([child.q_value for child in belief_node.children])

        return total
