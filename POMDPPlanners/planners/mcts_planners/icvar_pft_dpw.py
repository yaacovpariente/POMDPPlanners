"""ICVaR PFT-DPW (Iterated CVaR Particle Filter Tree with Double Progressive Widening) Algorithm.

This module implements a risk-sensitive variant of PFT-DPW that uses the Iterated Conditional
Value at Risk (ICVaR) for value backups instead of the expected value. This makes the planner
focus on the worst-alpha fraction of outcomes, enabling risk-averse planning in POMDPs.

Reference:
    Pariente, Y., & Indelman, V. (2026). Online Risk-Averse Planning in POMDPs Using
    Iterated CVaR Value Function. arXiv preprint arXiv:2601.20554.
    https://arxiv.org/abs/2601.20554

Implementation note:
    Operates on the column-store arena
    :class:`POMDPPlanners.core.tree.arena.Tree` (integer node IDs, parallel
    column lists) rather than the legacy anytree-based ``BeliefNode`` /
    ``ActionNode`` graph. Inherits from
    :class:`ArenaPathSimulationPolicyCostSetting`. External constructor
    signature, ``action()`` interface, and behavior are unchanged.

Classes:
    ICVaR_PFT_DPW: Risk-sensitive PFT-DPW planner with CVaR-based value updates
"""

from typing import Optional

import numpy as np

from POMDPPlanners.core.belief import Belief, is_terminal_belief
from POMDPPlanners.core.cost import belief_expectation_cost
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.cvar_progressive_widening import (
    cvar_action_progressive_widening_arena,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaPathSimulationPolicyCostSetting,
)
from POMDPPlanners.utils.statistics_utils import cvar_estimator_from_dist_fast


class ICVaR_PFT_DPW(ArenaPathSimulationPolicyCostSetting):
    """ICVaR PFT-DPW operating on the arena :class:`Tree` + integer node IDs.

    See module docstring for algorithm details and reference.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
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
        visit_count_penalty: float = 0.0,
        reserve_capacity: int = 0,
    ):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            action_sampler=action_sampler,
            time_out_in_seconds=time_out_in_seconds,
            reserve_capacity=reserve_capacity,
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
        self.max_depth = depth
        self.min_immediate_cost = min_immediate_cost
        self.max_immediate_cost = max_immediate_cost
        self.min_visit_count_per_action = min_visit_count_per_action
        self.belief_child_num = belief_child_num
        self.exploration_constant = exploration_constant
        self.action_sampler: ActionSampler = action_sampler
        self.k_a = k_a
        self.alpha_a = alpha_a
        self.k_o = k_o
        self.alpha_o = alpha_o
        self.discrete_actions = self.environment.space_info.action_space == SpaceType.DISCRETE
        self.visit_count_penalty = visit_count_penalty

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> None:
        if depth > self.depth:
            return

        if self.is_terminal_belief(belief=tree.belief[belief_id]):
            tree.visit_count[belief_id] += 1
            return

        action_id = cvar_action_progressive_widening_arena(
            tree=tree,
            belief_id=belief_id,
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
            environment=self.environment,
        )

        action_children_count = len(tree.children_ids[action_id])
        action_visits = tree.visit_count[action_id]
        if action_children_count <= self.k_o * action_visits**self.alpha_o:
            next_belief_id = self._generate_belief(tree=tree, action_id=action_id)
        else:
            next_belief_id = self._sample_next_existing_belief(tree=tree, action_id=action_id)

        self._simulate_path(tree=tree, belief_id=next_belief_id, depth=depth + 1)

        self.update_nodes(tree=tree, belief_id=belief_id, action_id=action_id)

    def is_terminal_belief(self, belief: Belief) -> bool:
        """Return True if all particles in ``belief`` are terminal states."""
        return is_terminal_belief(belief=belief, env=self.environment)

    def _sample_next_existing_belief(self, tree: Tree, action_id: int) -> int:
        # Belief children carry an arena-maintained CDF over their ``weight``
        # values. ``add_belief_node`` initialises each child with weight=1.0
        # (mirroring the +1 update_nodes increments at generation), so the
        # weighted sample below is statistically equivalent to the previous
        # ``np.cumsum(visit_count) → searchsorted`` path while running in
        # O(log K) instead of O(K). The matching ``increment_weight`` keeps
        # the CDF aligned with each child's running visit count.
        sampled_id = tree.sample_belief_child(action_id)
        tree.increment_weight(sampled_id, 1.0)
        return sampled_id

    def _generate_belief(self, tree: Tree, action_id: int) -> int:
        parent_belief_id = tree.parent_id[action_id]
        assert parent_belief_id is not None, "action node must have a parent belief"
        belief = tree.belief[parent_belief_id]
        action = tree.action[action_id]
        state = belief.sample()
        next_state = self.environment.sample_next_state(state=state, action=action)
        next_observation = self.environment.sample_observation(next_state=next_state, action=action)

        next_belief = belief.update(
            action=action, observation=next_observation, pomdp=self.environment
        )

        next_belief_id = tree.add_belief_node(
            belief=next_belief, observation=next_observation, parent_id=action_id
        )
        # Compute the (parent_belief, action) expected cost once and stash it on
        # the action node. ``update_nodes`` reads it back without recomputing.
        if tree.immediate_cost[action_id] is None:
            tree.set_immediate_cost(
                action_id,
                belief_expectation_cost(belief=belief, action=action, env=self.environment),
            )

        return next_belief_id

    def update_nodes(self, tree: Tree, belief_id: int, action_id: int) -> None:
        tree.visit_count[belief_id] += 1
        tree.visit_count[action_id] += 1

        action_immediate_cost = tree.immediate_cost[action_id]
        if action_immediate_cost is None:
            # Action selected via LCB exploration on a never-expanded path.
            action_immediate_cost = belief_expectation_cost(
                belief=tree.belief[belief_id],
                action=tree.action[action_id],
                env=self.environment,
            )
            tree.set_immediate_cost(action_id, action_immediate_cost)

        action_children = tree.children_ids[action_id]
        n_action_children = len(action_children)
        if n_action_children == 0:
            tree.q_value[action_id] = action_immediate_cost
        else:
            visit_counts = np.fromiter(
                (tree.visit_count[cid] for cid in action_children),
                dtype=np.float64,
                count=n_action_children,
            )
            total_visits = visit_counts.sum()
            if total_visits == 0:
                tree.q_value[action_id] = action_immediate_cost
            else:
                v_values = np.fromiter(
                    (tree.v_value[cid] for cid in action_children),
                    dtype=np.float64,
                    count=n_action_children,
                )
                tree.q_value[action_id] = action_immediate_cost + (
                    self.discount_factor
                    * cvar_estimator_from_dist_fast(
                        values=v_values,
                        weights=visit_counts / total_visits,
                        alpha=self.alpha,
                    )
                )

        belief_children = tree.children_ids[belief_id]
        best_q: Optional[float] = None
        for cid in belief_children:
            if tree.visit_count[cid] > 0:
                q = tree.q_value[cid]
                if best_q is None or q < best_q:
                    best_q = q
        if best_q is not None:
            tree.v_value[belief_id] = best_q

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS,
        )
