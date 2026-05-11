"""Sparse-PFT (Sparse Particle Filter Tree) Algorithm for POMDP Planning.

Implementation note:
    Operates on the column-store arena
    :class:`POMDPPlanners.core.tree.arena.Tree` (integer node IDs, parallel
    column lists) rather than the legacy anytree-based ``BeliefNode`` /
    ``ActionNode`` graph. Inherits from :class:`ArenaPathSimulationPolicy`.
    External constructor signature, ``action()`` interface, and behavior
    are unchanged.
"""

import random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import is_terminal_belief
from POMDPPlanners.core.cost import belief_expectation_cost
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaPathSimulationPolicy,
)


class SparsePFT(ArenaPathSimulationPolicy):
    """Sparse-PFT operating on the arena :class:`Tree` + integer node IDs.

    See module docstring for algorithm details.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(42)
        >>>
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = SparsePFT(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     c_ucb=1.0,
        ...     beta_ucb=2.0,
        ...     belief_child_num=3,
        ...     n_simulations=10,
        ...     name="ExamplePlanner"
        ... )
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> space_info = SparsePFT.get_space_info()
        >>> space_info.action_space.name
        'DISCRETE'
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        c_ucb: float,
        beta_ucb: float,
        belief_child_num: int,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        name: str = "SparsePFT",
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        if not isinstance(environment, DiscreteActionsEnvironment):
            raise TypeError("environment must be a DiscreteActionsEnvironment instance")
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor must be a float")
        if not isinstance(depth, int):
            raise TypeError("depth must be an int")
        if not isinstance(c_ucb, float):
            raise TypeError("c_ucb must be a float")
        if not isinstance(beta_ucb, float):
            raise TypeError("beta_ucb must be a float")
        if not isinstance(belief_child_num, int):
            raise TypeError("belief_child_num must be an int")
        if not 1 >= discount_factor >= 0:
            raise ValueError("discount_factor must be between 0 and 1")

        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            action_sampler=None,
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.depth = depth
        self.c_ucb = c_ucb
        self.beta_ucb = beta_ucb
        self.belief_child_num = belief_child_num

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> float:
        if depth > self.depth:
            return 0

        if is_terminal_belief(belief=tree.get_belief(belief_id), env=self.environment):
            tree.increment_visit_count(belief_id)
            return 0

        if not tree.get_children_ids(belief_id):
            for action in self.environment.get_actions():  # type: ignore[attr-defined]
                tree.add_action_node(action=action, parent_id=belief_id)
            state = tree.get_belief(belief_id).sample()
            tree.increment_visit_count(belief_id)
            return self.random_rollout(state=state, depth=depth)

        action_id = self.get_explored_action_node(tree=tree, belief_id=belief_id)

        if len(tree.get_children_ids(action_id)) == self.belief_child_num:
            next_belief_id, immediate_reward = self._sample_next_existing_belief(
                tree=tree, action_id=action_id
            )
        else:
            next_belief_id, immediate_reward = self._generate_belief(tree=tree, action_id=action_id)

        return_sample = immediate_reward + self.discount_factor * self._simulate_path(
            tree=tree, belief_id=next_belief_id, depth=depth + 1
        )

        self.update_nodes(
            tree=tree, belief_id=belief_id, action_id=action_id, return_sample=return_sample
        )
        return return_sample

    def get_explored_action_node(self, tree: Tree, belief_id: int) -> int:
        children = tree.get_children_ids(belief_id)
        children_visits = np.array([tree.get_visit_count(cid) for cid in children])
        unvisited_indices = np.where(children_visits == 0)[0]
        if len(unvisited_indices) > 0:
            return children[int(np.random.choice(unvisited_indices))]

        q_vals = np.array([tree.get_q_value(cid) for cid in children])
        sparse_addition = (
            self.c_ucb
            * self.beta_ucb
            * tree.get_visit_count(belief_id)
            * (1.0 / np.sqrt(children_visits))
        )
        return children[int(np.argmax(q_vals + sparse_addition))]

    def _sample_next_existing_belief(self, tree: Tree, action_id: int) -> Tuple[int, float]:
        # Sample a belief child via the arena's maintained CDF — O(log K).
        # ``add_belief_node`` defaults each child's ``weight`` to 1.0
        # (matching the +1 update_nodes increments at generation), and
        # ``increment_weight`` keeps weight ≡ visit_count thereafter, so
        # ``sample_belief_child`` is statistically equivalent to the previous
        # ``np.cumsum(visit_count) → searchsorted`` path.
        sampled_id = tree.sample_belief_child(action_id)
        tree.increment_weight(sampled_id, 1.0)
        immediate_cost = tree.get_immediate_cost(sampled_id)
        expected_reward = -immediate_cost if immediate_cost is not None else 0.0
        return sampled_id, expected_reward

    def _generate_belief(self, tree: Tree, action_id: int) -> Tuple[int, float]:
        # Parent of an action node is always a belief node (never None for non-root).
        parent_belief_id = tree.get_parent_id(action_id)
        assert parent_belief_id is not None, "action node must have a parent belief"
        belief = tree.get_belief(parent_belief_id)
        action = tree.get_action(action_id)
        state = belief.sample()
        next_state = self.environment.sample_next_state(state=state, action=action)
        next_observation = self.environment.sample_observation(next_state=next_state, action=action)

        next_belief = belief.update(
            action=action,
            observation=next_observation,
            pomdp=self.environment,
        )

        next_belief_id = tree.add_belief_node(
            belief=next_belief, observation=next_observation, parent_id=action_id
        )
        immediate_cost = belief_expectation_cost(belief=belief, action=action, env=self.environment)
        tree.set_immediate_cost(next_belief_id, immediate_cost)
        immediate_reward = -immediate_cost
        return next_belief_id, immediate_reward

    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        action = random.choice(self.environment.get_actions())  # type: ignore[attr-defined]
        next_state, _, reward = self.environment.sample_next_step(state=state, action=action)
        return reward + self.discount_factor * self.random_rollout(
            state=next_state, depth=depth + 1
        )

    def update_nodes(
        self, tree: Tree, belief_id: int, action_id: int, return_sample: float
    ) -> None:
        if tree.get_immediate_cost(action_id) is None:
            tree.set_immediate_cost(
                action_id,
                belief_expectation_cost(
                    belief=tree.get_belief(belief_id),
                    action=tree.get_action(action_id),
                    env=self.environment,
                ),
            )

        tree.increment_visit_count(belief_id)
        tree.update_action_q_with_return(action_id, return_sample)
        children = tree.get_children_ids(belief_id)
        if children:
            tree.v_value[belief_id] = float(max(tree.get_q_value(cid) for cid in children))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED)
