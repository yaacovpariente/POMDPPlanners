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
    POMCP_DPW: Monte Carlo Tree Search planner with double progressive widening extending POMCP
"""

from pathlib import Path
from typing import Any, Optional

from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening_arena,
)
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaDoubleProgressiveWideningMCTSPolicy,
)
from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler


class POMCP_DPW(ArenaDoubleProgressiveWideningMCTSPolicy):
    """POMCP_DPW operating on the arena :class:`Tree` + integer node IDs.

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
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> space_info = POMCP_DPW.get_space_info()
        >>> space_info.action_space.name
        'MIXED'
    """

    def __init__(  # pylint: disable=too-many-arguments
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
        min_visit_count_per_action: int = 1,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
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

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> float:
        state = tree.belief[belief_id].sample()
        return self._simulate_state_path(tree=tree, state=state, belief_id=belief_id, depth=depth)

    def _simulate_state_path(  # pylint: disable=too-many-locals
        self, tree: Tree, state: Any, belief_id: int, depth: int
    ) -> float:
        if depth > self.depth:
            return 0

        if self.environment.is_terminal(state=state):
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
        action = tree.action[action_id]

        children_count = len(tree.children_ids[action_id])
        action_visits = tree.visit_count[action_id]
        if children_count <= self.k_o * action_visits**self.alpha_o:
            next_state = self.environment.state_transition_model(
                state=state, action=action
            ).sample()[0]
            reward = self.environment.reward(state=state, action=action)
            next_observation = self.environment.observation_model(
                next_state=next_state, action=action
            ).sample()[0]

            next_belief_id = tree.get_belief_child_indexed(action_id, next_observation)
            if next_belief_id is None:
                next_belief_id = tree.get_belief_child(
                    action_id, next_observation, self.environment
                )
            if next_belief_id is None:
                next_belief_id = tree.add_belief_node(
                    belief=UnweightedParticleBeliefStateUpdate(),
                    observation=next_observation,
                    parent_id=action_id,
                    weight=1.0,
                )
            else:
                # Bump the weight of the existing belief child + patch parent CDF.
                tree.increment_weight(next_belief_id, delta=1.0)

            tree.belief[next_belief_id].inplace_update(
                action=None, observation=None, pomdp=self.environment, state=next_state
            )

            if tree.visit_count[next_belief_id] == 0:
                tree.visit_count[next_belief_id] += 1
                total = reward + self.discount_factor * random_rollout_action_sampler(
                    state=next_state,
                    depth=depth + 1,
                    action_sampler=self.action_sampler,
                    environment=self.environment,
                    discount_factor=self.discount_factor,
                )
            else:
                total = reward + self.discount_factor * self._simulate_state_path(
                    tree=tree, state=next_state, belief_id=next_belief_id, depth=depth + 1
                )
        else:
            next_belief_id = tree.sample_belief_child(action_id)
            next_state = tree.belief[next_belief_id].sample()
            reward = self.environment.reward(state=next_state, action=action)
            total = reward + self.discount_factor * self._simulate_state_path(
                tree=tree, state=next_state, belief_id=next_belief_id, depth=depth + 1
            )

        # Backup
        tree.visit_count[belief_id] += 1
        tree.visit_count[action_id] += 1
        tree.q_value[action_id] += (total - tree.q_value[action_id]) / tree.visit_count[action_id]
        tree.v_value[belief_id] = max(tree.q_value[cid] for cid in tree.children_ids[belief_id])
        return total
