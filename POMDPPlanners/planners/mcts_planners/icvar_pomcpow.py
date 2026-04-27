"""ICVaR POMCPOW (Iterated CVaR POMCPOW) Algorithm.

This module implements a risk-sensitive variant of POMCPOW that uses the Iterated Conditional
Value at Risk (ICVaR) for value backups instead of the expected value. This makes the planner
focus on the worst-alpha fraction of outcomes, enabling risk-averse planning in POMDPs with
continuous state, action, and observation spaces.

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
    ICVaR_POMCPOW: Risk-sensitive POMCPOW planner with CVaR-based value updates
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief, WeightedParticleBeliefStateUpdate
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
from POMDPPlanners.utils.numba_kernels import cvar_estimator_from_dist_fast_kernel


class ICVaR_POMCPOW(ArenaPathSimulationPolicyCostSetting):
    """ICVaR POMCPOW operating on the arena :class:`Tree` + integer node IDs.

    See module docstring for algorithm details and reference.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
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
        reserve_capacity: int = 0,
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
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
        )

        self.depth = depth
        self.exploration_constant = exploration_constant
        self.min_samples_per_node = min_samples_per_node
        self.action_sampler: ActionSampler = action_sampler
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

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> None:
        state = tree.belief[belief_id].sample()
        self._simulate_state_path(tree=tree, state=state, belief_id=belief_id, depth=depth)

    def _simulate_state_path(self, tree: Tree, state: Any, belief_id: int, depth: int) -> None:
        """Core ICVaR-POMCPOW simulation step on the arena tree."""
        if self._check_termination_conditions(
            tree=tree, state=state, belief_id=belief_id, depth=depth
        ):
            return

        action_id = self._select_action_with_progressive_widening(tree=tree, belief_id=belief_id)
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=tree.action[action_id]
        )
        next_belief_id = self._select_or_create_observation_node(
            tree=tree, action_id=action_id, next_observation=next_observation
        )
        self._update_belief_with_state(
            tree=tree,
            belief_id=next_belief_id,
            action_id=action_id,
            observation=next_observation,
            state=next_state,
        )

        self._simulate_state_path(
            tree=tree, state=next_state, belief_id=next_belief_id, depth=depth + 1
        )

        self._backpropagate_values(
            tree=tree,
            belief_id=belief_id,
            action_id=action_id,
            reward=reward,
        )

    def _check_termination_conditions(
        self, tree: Tree, state: Any, belief_id: int, depth: int
    ) -> bool:
        if depth > self.depth:
            return True

        if self.environment.is_terminal(state=state):
            tree.visit_count[belief_id] += 1
            return True

        return False

    def _select_action_with_progressive_widening(self, tree: Tree, belief_id: int) -> int:
        return cvar_action_progressive_widening_arena(
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

    def _select_or_create_observation_node(
        self, tree: Tree, action_id: int, next_observation: Any
    ) -> int:
        children_count = len(tree.children_ids[action_id])
        action_visits = tree.visit_count[action_id]
        if children_count <= self.k_o * action_visits**self.alpha_o:
            try:
                obs_key = self.environment.hash_observation(next_observation)
                # Indexed lookup is authoritative when obs_key is available.
                # The contract on ``hash_observation`` guarantees:
                #   is_equal_observation(a, b) ==> hash_observation(a) ==
                #   hash_observation(b)
                # so a None result means no equal observation has been added.
                existing_id = tree.get_belief_child_indexed(action_id, obs_key=obs_key)
            except NotImplementedError:
                # Env not migrated; fall back to linear-scan path.
                obs_key = None
                existing_id = tree.get_belief_child(action_id, next_observation, self.environment)
            if existing_id is None:
                next_belief_id = tree.add_belief_node(
                    belief=WeightedParticleBeliefStateUpdate(particles=[], weights=[]),
                    observation=next_observation,
                    parent_id=action_id,
                    weight=1.0,
                    obs_key=obs_key,
                )
            else:
                tree.increment_weight(existing_id, delta=1.0)
                next_belief_id = existing_id
        else:
            next_belief_id = tree.sample_belief_child(action_id)
        return next_belief_id

    def _update_belief_with_state(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        observation: Any,
        state: Any,
    ) -> None:
        tree.belief[belief_id].inplace_update(
            action=tree.action[action_id],
            observation=observation,
            pomdp=self.environment,
            state=state,
        )

    def _backpropagate_values(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        reward: float,
    ) -> None:
        tree.visit_count[belief_id] += 1
        tree.visit_count[action_id] += 1

        self._update_immediate_cost(
            tree=tree, belief_id=belief_id, action_id=action_id, reward=reward
        )
        self._update_q_value(tree=tree, action_id=action_id)
        self._update_v_value(tree=tree, belief_id=belief_id)

    def _update_immediate_cost(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        reward: float,
    ) -> None:
        belief = tree.belief[belief_id]
        action = tree.action[action_id]
        immediate_cost = tree.immediate_cost[action_id]
        if immediate_cost is None:
            if isinstance(belief, WeightedParticleBeliefStateUpdate):
                tree.set_immediate_cost(action_id, -reward)
            elif isinstance(belief, WeightedParticleBelief):
                particle_weights = belief.normalized_weights
                particle_costs = np.array(
                    [
                        -self.environment.reward(state=particle, action=action)
                        for particle in belief.particles
                    ]
                )
                tree.set_immediate_cost(action_id, float(np.sum(particle_costs * particle_weights)))
            else:
                raise ValueError(f"Unsupported belief type: {type(belief)}")
        elif isinstance(belief, WeightedParticleBeliefStateUpdate):
            old_weights_sum = belief.weights_sum - belief.weights[-1]
            new_cost = (
                immediate_cost * old_weights_sum - reward * belief.weights[-1]
            ) / belief.weights_sum
            tree.set_immediate_cost(action_id, new_cost)

    def _update_q_value(self, tree: Tree, action_id: int) -> None:
        children = tree.children_ids[action_id]
        immediate_cost = tree.immediate_cost[action_id]
        if immediate_cost is None:
            raise ValueError("immediate_cost must be set before updating q_value")

        if not children:
            tree.q_value[action_id] = immediate_cost
            return

        visited_children = [cid for cid in children if tree.visit_count[cid] > 0]
        n_visited = len(visited_children)
        if n_visited == 0:
            tree.q_value[action_id] = immediate_cost
            return

        # Single Python loop fills both arrays — faster than two np.fromiter
        # calls over a Python-list source for small N (typical here).
        tree_v_value = tree.v_value
        tree_weight = tree.weight
        v_values = np.empty(n_visited, dtype=np.float64)
        weights = np.empty(n_visited, dtype=np.float64)
        weights_sum = 0.0
        for i, cid in enumerate(visited_children):
            v_values[i] = tree_v_value[cid]
            w = tree_weight[cid]
            weights[i] = w
            weights_sum += w
        if weights_sum > 0.0:
            weights = weights / weights_sum
        tree.q_value[action_id] = (
            immediate_cost
            + self.discount_factor
            * cvar_estimator_from_dist_fast_kernel(v_values, weights, self.alpha)
        )

    def _update_v_value(self, tree: Tree, belief_id: int) -> None:
        best_q: Optional[float] = None
        for cid in tree.children_ids[belief_id]:
            if tree.visit_count[cid] > 0:
                q = tree.q_value[cid]
                if best_q is None or q < best_q:
                    best_q = q
        if best_q is not None:
            tree.v_value[belief_id] = best_q

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)
