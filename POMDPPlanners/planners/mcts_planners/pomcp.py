# SPDX-License-Identifier: MIT

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

Reference:
    Silver, D., & Veness, J. (2010). Monte-Carlo Planning in Large POMDPs.
    Advances in Neural Information Processing Systems, 23.
    https://papers.nips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html

Implementation note:
    This implementation operates on the column-store arena
    :class:`POMDPPlanners.core.tree.arena.Tree` (integer node IDs, parallel
    column lists) rather than the legacy anytree-based ``BeliefNode`` /
    ``ActionNode`` graph. Inherits from :class:`ArenaPathSimulationPolicy`.
    The external constructor signature, ``action()`` interface, and behavior
    are unchanged from earlier versions.

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
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaPathSimulationPolicy,
)


class POMCP(ArenaPathSimulationPolicy):
    """POMCP operating on the arena :class:`Tree` + integer node IDs.

    See module docstring for algorithm details and reference.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(42)
        >>>
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = POMCP(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     exploration_constant=1.0,
        ...     name="ExamplePlanner",
        ...     n_simulations=10
        ... )
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> space_info = POMCP.get_space_info()
        >>> space_info.action_space.name
        'DISCRETE'
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        exploration_constant: float,
        name: str,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        reserve_capacity: int = 0,
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
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.depth = depth
        self.exploration_constant = exploration_constant

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> float:
        state = tree.get_belief(belief_id).sample()
        return self._simulate_state_path(tree=tree, state=state, belief_id=belief_id, depth=depth)

    def _simulate_state_path(  # pylint: disable=too-many-locals
        self, tree: Tree, state: Any, belief_id: int, depth: int
    ) -> float:
        if depth > self.depth:
            return 0

        if self.environment.is_terminal(state=state):
            tree.increment_visit_count(belief_id)
            return 0

        # Leaf: expand action children for every available action, then rollout.
        if not tree.get_children_ids(belief_id):
            for action in self.environment.get_actions():  # type: ignore[attr-defined]
                tree.add_action_node(action=action, parent_id=belief_id)
            tree.increment_visit_count(belief_id)
            return self.random_rollout(state=state, depth=depth)

        action_id = self.get_explored_action_node(tree=tree, belief_id=belief_id)

        state = tree.get_belief(belief_id).sample()
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=tree.get_action(action_id)
        )

        # Find existing belief child for this observation; create one if absent.
        next_belief_id = tree.get_belief_child_indexed(action_id, next_observation)
        if next_belief_id is None:
            next_belief_id = tree.get_belief_child(action_id, next_observation, self.environment)
        if next_belief_id is None:
            next_belief_id = tree.add_belief_node(
                belief=UnweightedParticleBeliefStateUpdate(particles=[next_state]),
                observation=next_observation,
                parent_id=action_id,
            )

        return_sample = reward + self.discount_factor * self._simulate_state_path(
            tree=tree, state=next_state, belief_id=next_belief_id, depth=depth + 1
        )

        self.update_nodes(
            tree=tree,
            belief_id=belief_id,
            action_id=action_id,
            return_sample=return_sample,
            state=state,
        )
        return return_sample

    def get_explored_action_node(self, tree: Tree, belief_id: int) -> int:
        """Pick an action child via UCB1; if any child has zero visits, pick uniformly from those."""
        children = tree.get_children_ids(belief_id)
        action_visits = np.array([tree.get_visit_count(cid) for cid in children])
        unvisited_indices = np.where(action_visits == 0)[0]
        if len(unvisited_indices) > 0:
            return children[int(np.random.choice(unvisited_indices))]

        q_values = np.array([tree.get_q_value(cid) for cid in children])
        ucb = q_values + self.exploration_constant * np.sqrt(
            np.log(tree.get_visit_count(belief_id)) / action_visits
        )
        return children[int(np.argmax(ucb))]

    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0

        action = random.choice(self.environment.get_actions())  # type: ignore[attr-defined]
        next_state, _, reward = self.environment.sample_next_step(state=state, action=action)

        return reward + self.discount_factor * self.random_rollout(
            state=next_state, depth=depth + 1
        )

    def update_nodes(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        return_sample: float,
        state: Any,
    ) -> None:
        # Don't update the initial belief in place (matches legacy POMCP behaviour).
        if tree.get_parent_id(belief_id) is not None:
            tree.get_belief(belief_id).inplace_update(
                action=None, observation=None, pomdp=self.environment, state=state
            )

        tree.increment_visit_count(belief_id)
        tree.update_action_q_with_return(action_id, return_sample)
        # POMCP's v-backup is over visited children only (zero-visit children
        # have no real estimate yet); a generic max(q[child]) form would
        # always pick the unvisited q=0.0 entries.
        tree.v_value[belief_id] = float(
            np.max(
                [
                    tree.get_q_value(cid)
                    for cid in tree.get_children_ids(belief_id)
                    if tree.get_visit_count(cid) > 0
                ]
            )
        )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
