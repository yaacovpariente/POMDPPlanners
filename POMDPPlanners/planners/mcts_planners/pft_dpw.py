"""PFT-DPW (Particle Filter Tree with Double Progressive Widening) Algorithm.

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

Reference:
    Sunberg, Z. N., & Kochenderfer, M. J. (2018). Online Algorithms for POMDPs with
    Continuous State, Action, and Observation Spaces. Proceedings of the International
    Conference on Automated Planning and Scheduling, 28(1), 259-263.
    https://ojs.aaai.org/index.php/ICAPS/article/view/13882

Implementation note:
    This implementation operates on the column-store arena
    :class:`POMDPPlanners.core.tree.arena.Tree` (integer node IDs, parallel
    column lists) rather than the legacy anytree-based ``BeliefNode`` /
    ``ActionNode`` graph. Inherits from
    :class:`ArenaDoubleProgressiveWideningMCTSPolicy`. The external
    constructor signature, ``action()`` interface, and behavior are
    unchanged from earlier versions.

Classes:
    ActionSampler: Abstract base class for action sampling strategies (re-exported)
    PFT_DPW: Main PFT-DPW planner with progressive widening for continuous actions
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening_arena,
)
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaDoubleProgressiveWideningMCTSPolicy,
)


class PFT_DPW(ArenaDoubleProgressiveWideningMCTSPolicy):
    """PFT-DPW operating on the arena :class:`Tree` + integer node IDs.

    See module docstring for algorithm details and reference.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
        >>> np.random.seed(42)
        >>>
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
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> space_info = PFT_DPW.get_space_info()
        >>> space_info.action_space.name
        'MIXED'
    """

    def __init__(  # pylint: disable=too-many-arguments
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
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self._validate_pft_dpw_params(min_visit_count_per_action=min_visit_count_per_action)

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
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self.min_visit_count_per_action = min_visit_count_per_action

    @staticmethod
    def _validate_pft_dpw_params(min_visit_count_per_action: int) -> None:
        if not isinstance(min_visit_count_per_action, int):
            raise TypeError(
                f"min_visit_count_per_action must be an int, "
                f"got {type(min_visit_count_per_action).__name__}"
            )
        if min_visit_count_per_action < 1:
            raise ValueError(
                f"min_visit_count_per_action must be >= 1, got {min_visit_count_per_action}"
            )

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> float:
        if depth > self.depth:
            return 0

        if self.environment.is_terminal(tree.belief[belief_id].sample()):
            tree.visit_count[belief_id] += 1
            return 0

        action_id = action_progressive_widening_arena(
            tree=tree,
            belief_id=belief_id,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a,
            min_visit_count_per_action=self.min_visit_count_per_action,
        )

        return_sample = self._simulate_return(
            tree=tree, belief_id=belief_id, action_id=action_id, depth=depth
        )

        self._update_node_statistics(
            tree=tree, belief_id=belief_id, action_id=action_id, total=return_sample
        )
        return return_sample

    def _simulate_return(self, tree: Tree, belief_id: int, action_id: int, depth: int) -> float:
        action_visits = tree.visit_count[action_id]
        children_count = len(tree.children_ids[action_id])
        if children_count <= self.k_o * action_visits**self.alpha_o:
            next_belief_id, immediate_reward = self._sample_new_belief_node(
                tree=tree, belief_id=belief_id, action_id=action_id
            )
            state = tree.belief[next_belief_id].sample()
            total = immediate_reward + self.discount_factor * self._random_rollout(
                state=state, depth=depth + 1
            )
        else:
            next_belief_id, immediate_reward = self.sample_existing_belief_node(
                tree=tree, belief_id=belief_id, action_id=action_id
            )
            total = immediate_reward + self.discount_factor * self._simulate_path(
                tree=tree, belief_id=next_belief_id, depth=depth + 1
            )
        return total

    def _sample_new_belief_node(
        self, tree: Tree, belief_id: int, action_id: int
    ) -> Tuple[int, float]:
        action = tree.action[action_id]
        belief = tree.belief[belief_id]
        immediate_reward = belief_expectation_reward(
            belief=belief, action=action, env=self.environment
        )
        # Stash on parent belief so re-visits to this branch can recover the
        # immediate reward without recomputing it.
        tree.immediate_cost[belief_id] = -immediate_reward
        tree.immediate_reward[belief_id] = immediate_reward

        _, next_observation, _ = self.environment.sample_next_step(
            state=belief.sample(), action=action
        )
        next_belief = belief.update(
            observation=next_observation,
            action=action,
            pomdp=self.environment,
        )
        next_belief_id = tree.add_belief_node(belief=next_belief, parent_id=action_id)
        return next_belief_id, immediate_reward

    def _random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        action = self.action_sampler.sample()
        next_state, _, reward = self.environment.sample_next_step(state=state, action=action)
        return reward + self.discount_factor * self._random_rollout(
            state=next_state, depth=depth + 1
        )

    def sample_existing_belief_node(
        self, tree: Tree, belief_id: int, action_id: int
    ) -> Tuple[int, float]:
        # Recover the immediate reward stashed on the belief at allocation.
        cost = tree.immediate_cost[belief_id]
        immediate_reward = -cost if cost is not None else 0.0
        next_belief_id = tree.sample_belief_child(action_id)
        return next_belief_id, immediate_reward

    def _update_node_statistics(
        self, tree: Tree, belief_id: int, action_id: int, total: float
    ) -> None:
        tree.visit_count[belief_id] += 1
        tree.visit_count[action_id] += 1
        tree.q_value[action_id] += (total - tree.q_value[action_id]) / tree.visit_count[action_id]
        tree.v_value[belief_id] = float(
            np.max([tree.q_value[cid] for cid in tree.children_ids[belief_id]])
        )
