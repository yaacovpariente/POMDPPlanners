# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""C-PFT-DPW (Constrained Particle Filter Tree with Double Progressive Widening).

Online MCTS for cost-constrained POMDPs that uses particle-filter belief
children (PFT-style) under the dual-ascent Lagrangian scaffold from
:class:`POMDPPlanners.planners.planners_utils.constrained_mcts_mixin.ConstrainedMCTSMixin`.

The shared scaffold implements Listing 1 from Jamgochian et al. (ICAPS
2023); this module supplies only the variant-specific ``SIMULATE``
(Algorithm 2 in the paper) — PFT-DPW's belief-MDP recursion with
particle-filter belief updates, augmented with a per-belief-child cost
cache and the optional minimal-cost propagation trick.

This is the PFT-DPW counterpart of
:class:`POMDPPlanners.planners.mcts_planners.constrained_pomcpow.CPOMCPOW`.

References:
    Jamgochian, A., Corso, A., & Kochenderfer, M. J. (2023). Online Planning
    for Constrained POMDPs with Continuous Spaces through Dual Ascent. AAAI.
    arXiv:2212.12154.
    Lee, J., Kim, G.-H., Poupart, P., & Kim, K.-E. (2018). Monte-Carlo Tree
    Search for Constrained POMDPs. NeurIPS.

Classes:
    CPFT_DPW: Constrained PFT-DPW planner.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.environment import ConstrainedEnvironment, Environment
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.planners_utils.constrained_mcts_mixin import (
    ConstrainedMCTSMixin,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import cost_aware_random_rollout


class CPFT_DPW(ConstrainedMCTSMixin, PFT_DPW):
    """Constrained PFT-DPW with vector-valued dual ascent.

    The Lagrangian / dual-ascent layer lives on
    :class:`ConstrainedMCTSMixin`; this class supplies only the
    PFT-DPW-specific ``SIMULATE`` (Algorithm 2 in the paper) via
    :meth:`_simulate_path_with_cost` and its belief-cost helpers,
    alongside its constructor and a ``_reset_per_action_state`` override
    that also clears the per-belief-child cost cache.

    Args mirror :class:`PFT_DPW` plus:
        environment: A :class:`ConstrainedEnvironment` — constraint cost
            is read via ``environment.constraint_cost(s, a, s')``.
            Passing a plain :class:`Environment` raises ``TypeError``.
        cost_budget: Discounted-cost budget. Scalar or 1-D array of length
            ``K``. See :meth:`ConstrainedMCTSMixin._validate_and_pack_constraint_params`.
        lambda_init: Initial Lagrange multiplier per constraint dimension.
            Defaults to ``0.0``.
        lambda_step: Dual-ascent step size (> 0). Defaults to ``0.1``.
        return_minimal_cost: Enable the minimal-cost propagation trick
            from Jamgochian et al. (2023, Section 4 "Cost backpropagation").
            Defaults to ``True``.

    Raises:
        TypeError: If ``environment`` is not a :class:`ConstrainedEnvironment`.
        ValueError: See :class:`ConstrainedMCTSMixin` validation rules.

    Notes:
        - Per-belief-child cost is recorded at expansion time and reused
          on existing-child re-sampling (matches the per-belief-child
          ``(b', r, c)`` triple semantics of Algorithm 2 line 7 in the
          paper, generalised to vector ``c``).
        - Leaf expansion uses a cost-aware random rollout that
          accumulates ``Σ γ^t · constraint_cost(s_t, a_t, s_{t+1})``.
    """

    _belief_immediate_cost: Dict[int, np.ndarray]

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        environment: Environment,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        cost_budget: Union[float, np.ndarray],
        lambda_init: Union[float, np.ndarray] = 0.0,
        lambda_step: float = 0.1,
        return_minimal_cost: bool = True,
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
        if not isinstance(environment, ConstrainedEnvironment):
            raise TypeError(
                "CPFT_DPW requires environment to be a ConstrainedEnvironment; "
                f"got {type(environment).__name__}"
            )
        budget_arr, lambda_init_arr = self._validate_and_pack_constraint_params(
            cost_budget=cost_budget,
            lambda_init=lambda_init,
            lambda_step=lambda_step,
        )
        PFT_DPW.__init__(
            self,
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
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            min_visit_count_per_action=min_visit_count_per_action,
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self._init_constrained_state(
            environment=environment,
            cost_budget=budget_arr,
            lambda_init=lambda_init_arr,
            lambda_step=lambda_step,
            return_minimal_cost=return_minimal_cost,
        )
        self._belief_immediate_cost = {}

    def _reset_per_action_state(self) -> None:
        super()._reset_per_action_state()
        self._belief_immediate_cost = {}

    # ------------------------------------------------------------------
    # SIMULATE — paper Algorithm 2, PFT-DPW belief-based recursion with
    # cost track and optional minimal-cost propagation.
    # ------------------------------------------------------------------

    def _simulate_path_with_cost(
        self, tree: Tree, belief_id: int, depth: int
    ) -> Tuple[float, np.ndarray]:
        if depth > self.depth:
            return 0.0, np.zeros(self.n_constraints, dtype=np.float64)

        if self.environment.is_terminal(tree.get_belief(belief_id).sample()):
            tree.increment_visit_count(belief_id)
            return 0.0, np.zeros(self.n_constraints, dtype=np.float64)

        action_id = self._lagrangian_action_progressive_widening(tree=tree, belief_id=belief_id)
        total_v, total_c = self._simulate_return_with_cost(
            tree=tree, belief_id=belief_id, action_id=action_id, depth=depth
        )
        self._update_node_statistics_with_cost(
            tree=tree,
            belief_id=belief_id,
            action_id=action_id,
            total_v=total_v,
            total_c=total_c,
        )

        if self.return_minimal_cost:
            total_c = self._minimal_cost_propagation(
                tree=tree, belief_id=belief_id, fallback=total_c
            )

        return total_v, total_c

    def _simulate_return_with_cost(
        self, tree: Tree, belief_id: int, action_id: int, depth: int
    ) -> Tuple[float, np.ndarray]:
        action_visits = tree.get_visit_count(action_id)
        children_count = len(tree.get_children_ids(action_id))
        if children_count <= self.k_o * action_visits**self.alpha_o:
            next_belief_id, immediate_reward, immediate_cost = (
                self._sample_new_belief_node_with_cost(
                    tree=tree, belief_id=belief_id, action_id=action_id
                )
            )
            state = tree.get_belief(next_belief_id).sample()
            v_child, c_child = cost_aware_random_rollout(
                state=state,
                depth=depth + 1,
                action_sampler=self.action_sampler,
                environment=self.constrained_env,
                discount_factor=self.discount_factor,
                max_depth=self.depth + 1,
                n_constraints=self.n_constraints,
            )
            total_v = immediate_reward + self.discount_factor * v_child
            total_c = immediate_cost + self.discount_factor * c_child
        else:
            next_belief_id, immediate_reward, immediate_cost = (
                self._sample_existing_belief_node_with_cost(
                    tree=tree, belief_id=belief_id, action_id=action_id
                )
            )
            v_child, c_child = self._simulate_path_with_cost(
                tree=tree, belief_id=next_belief_id, depth=depth + 1
            )
            total_v = immediate_reward + self.discount_factor * v_child
            total_c = immediate_cost + self.discount_factor * c_child
        return total_v, total_c

    def _sample_new_belief_node_with_cost(
        self, tree: Tree, belief_id: int, action_id: int
    ) -> Tuple[int, float, np.ndarray]:
        action = tree.get_action(action_id)
        belief = tree.get_belief(belief_id)

        immediate_reward = belief_expectation_reward(
            belief=belief, action=action, env=self.environment
        )
        tree.set_immediate_reward(action_id, immediate_reward)

        state = belief.sample()
        next_state, next_observation, _ = self.environment.sample_next_step(
            state=state, action=action
        )
        immediate_cost = self._read_constraint_cost(
            state=state, action=action, next_state=next_state
        )

        next_belief = belief.update(
            observation=next_observation,
            action=action,
            pomdp=self.environment,
        )
        next_belief_id = tree.add_belief_node(belief=next_belief, parent_id=action_id)
        self._belief_immediate_cost[next_belief_id] = immediate_cost
        return next_belief_id, immediate_reward, immediate_cost

    def _sample_existing_belief_node_with_cost(
        self, tree: Tree, belief_id: int, action_id: int  # pylint: disable=unused-argument
    ) -> Tuple[int, float, np.ndarray]:
        immediate_reward = tree.get_immediate_reward(action_id) or 0.0
        next_belief_id = tree.sample_belief_child(action_id)
        cached = self._belief_immediate_cost.get(next_belief_id)
        immediate_cost = (
            cached if cached is not None else np.zeros(self.n_constraints, dtype=np.float64)
        )
        return next_belief_id, immediate_reward, immediate_cost

    def _update_node_statistics_with_cost(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        total_v: float,
        total_c: np.ndarray,
    ) -> None:
        tree.increment_visit_count(belief_id)
        tree.update_action_q_with_return(action_id, total_v)
        n_a = tree.get_visit_count(action_id)
        old_qc = self._cost_q(action_id)
        self._set_cost_q(action_id, old_qc + (total_c - old_qc) / n_a)
        children = tree.get_children_ids(belief_id)
        if children:
            tree.v_value[belief_id] = float(max(tree.get_q_value(cid) for cid in children))

    def _minimal_cost_propagation(
        self, tree: Tree, belief_id: int, fallback: np.ndarray
    ) -> np.ndarray:
        # Minimal-cost propagation (Jamgochian et al. 2023, Section 4):
        # replace the propagated cost with the QC of the current belief's
        # sibling action that minimises the Lagrangian score
        # ``(λ + ε)ᵀ · qc``. Returns a real sibling's qc verbatim (never
        # an elementwise min that could synthesise a vector no action
        # achieved). The action node's own QC update is unaffected.
        visited_siblings = [
            sib_id
            for sib_id in tree.get_children_ids(belief_id)
            if tree.get_visit_count(sib_id) > 0
        ]
        if not visited_siblings:
            return fallback
        sibling_qcs = np.stack([self._cost_q(sib_id) for sib_id in visited_siblings])
        lagrangian_norm = self._lambda + 1e-3
        scores = sibling_qcs @ lagrangian_norm
        best_idx = int(np.argmin(scores))
        return sibling_qcs[best_idx]
