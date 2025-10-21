"""POMCP (Partially Observable Monte Carlo Planning) Algorithm Implementation.

This module implements POMCP, a Monte Carlo Tree Search algorithm for POMDP planning.
POMCP builds a search tree by iteratively sampling trajectories and using UCB1
for action selection, providing an efficient approximation to optimal POMDP planning.

The algorithm works by:
1. Building a tree of belief-action nodes through Monte Carlo simulations
2. Using UCB1 (Upper Confidence Bounds) for action selection during tree traversal
3. Performing random rollouts from leaf nodes to estimate values
4. Updating node statistics (visit counts, Q-values) based on simulation returns

Key features:
- Handles large or continuous observation spaces through particle filtering
- Uses UCB1 for principled exploration-exploitation balance
- Can be configured with time limits or simulation count limits
- Provides theoretical convergence guarantees to optimal policy

Classes:
    POMCP: Monte Carlo Tree Search planner for POMDPs with UCB1 action selection
"""

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.tree import (
    ActionNode,
    BeliefNode,
)
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    PathSimulationPolicy,
)


class POMCP(PathSimulationPolicy):
    """POMCP (Partially Observable Monte Carlo Planning) algorithm.

    POMCP is a Monte Carlo Tree Search algorithm for POMDP planning that combines
    UCB1 action selection with particle filtering to handle continuous observation
    spaces. It builds a search tree through repeated simulations and provides
    theoretical convergence guarantees.

    The algorithm uses UCB1 (Upper Confidence Bounds) to balance exploration
    and exploitation when selecting actions during tree search. It maintains
    belief states using particle filters and performs random rollouts to
    estimate values at leaf nodes.

    Attributes:
        environment: The POMDP environment to plan for
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        depth: Maximum search depth for tree expansion
        exploration_constant: UCB1 exploration parameter (higher = more exploration)
        timeout_in_seconds: Time limit for planning (mutually exclusive with n_simulations)
        n_simulations: Number of simulations to run (mutually exclusive with timeout)
        min_samples_per_node: Minimum samples before a node is considered reliable

    Note:
        In the original POMCP paper, the belief structure used was an unweighted particle belief
        that can be found in :class:`POMDPPlanners.core.belief.UnweightedParticleBelief`. However,
        in this implementation, we keep the belief structure abstract to allow users to choose
        their preferred belief representation. In the usage example below, a weighted particle
        belief is used via the :func:`POMDPPlanners.core.belief.get_initial_belief` function.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Create environment and planner
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = POMCP(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     exploration_constant=1.0,
        ...     name="ExamplePlanner",
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
        >>> space_info = POMCP.get_space_info()
        >>> space_info.action_space.name
        'DISCRETE'
    """

    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        exploration_constant: float,
        name: str,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_samples_per_node: int = 10,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        combination1 = time_out_in_seconds is not None and n_simulations is None
        combination2 = time_out_in_seconds is None and n_simulations is not None
        if not (combination1 or combination2):
            raise ValueError("Only one of time_out_in_seconds and n_simulations must be provided.")

        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.depth = depth
        self.exploration_constant = exploration_constant
        self.min_samples_per_node = min_samples_per_node

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        state = belief_node.belief.sample()
        return self._simulate_state_path(state=state, belief_node=belief_node, depth=depth)

    def _simulate_state_path(self, state: Any, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None  # remove the node from the tree
            return 0

        if self.environment.is_terminal(state=state):
            belief_node.visit_count += 1
            return 0

        if belief_node.is_leaf:
            for action in self.environment.get_actions():  # type: ignore
                action_node = ActionNode(action=action, parent=belief_node, children=tuple())

            belief_node.visit_count += 1
            return self.random_rollout(state=state, depth=depth)

        action_node = self.get_explored_action_node(belief_node=belief_node)

        state = belief_node.belief.sample()
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=action_node.action
        )

        next_belief_node = None
        for belief_node_child in action_node.children:
            if self.environment.is_equal_observation(
                observation1=next_observation,
                observation2=belief_node_child.observation,
            ):
                next_belief_node = belief_node_child
                break

        if next_belief_node is None:
            belief = UnweightedParticleBeliefStateUpdate(particles=[next_state])
            next_belief_node = BeliefNode(
                belief=belief,
                observation=next_observation,
                parent=action_node,
                children=tuple(),
            )

        return_sample = reward + self.discount_factor * self._simulate_state_path(
            state=next_state, belief_node=next_belief_node, depth=depth + 1
        )

        self.update_nodes(
            belief_node=belief_node,
            action_node=action_node,
            return_sample=return_sample,
            state=state,
        )

        return return_sample

    def get_explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        if not isinstance(belief_node, BeliefNode):
            raise TypeError("belief_node must be a BeliefNode instance")

        action_nodes_visits = np.array([child.visit_count for child in belief_node.children])
        unvisited_action_indices = np.where(action_nodes_visits == 0)[0]
        if len(unvisited_action_indices) > 0:
            return belief_node.children[np.random.choice(unvisited_action_indices)]

        action_nodes_q_values = np.array([child.q_value for child in belief_node.children])
        ucb = action_nodes_q_values + self.exploration_constant * np.sqrt(
            np.log(belief_node.visit_count) / action_nodes_visits
        )

        return belief_node.children[np.argmax(ucb)]

    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0

        action = random.choice(self.environment.get_actions())  # type: ignore
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=action
        )

        return reward + self.discount_factor * self.random_rollout(
            state=next_state, depth=depth + 1
        )

    def update_nodes(
        self,
        belief_node: BeliefNode,
        action_node: ActionNode,
        return_sample: float,
        state: Any,
    ):
        if belief_node.parent is not None:  # prevents updating the initial belief.
            belief_node.belief.inplace_update(
                action=None, observation=None, pomdp=self.environment, state=state
            )

        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (return_sample - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max(
            [child.q_value for child in belief_node.children if child.visit_count > 0]
        )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
