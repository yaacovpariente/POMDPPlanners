# SPDX-License-Identifier: MIT

"""Shared abstract layer for CPOMDP-DPW MCTS variants.

Mirrors Listing 1 from Jamgochian, Corso, & Kochenderfer (ICAPS 2023),
*Online Planning for Constrained POMDPs with Continuous Spaces through
Dual Ascent* (arXiv:2212.12154). The paper factors all three variants
(CPOMCP-DPW, CPOMCPOW, CPFT-DPW) into:

* A shared layer of ``PLAN``, ``GREEDYPOLICY``, ``ACTIONPROGWIDEN``
  procedures with dual ascent on a Lagrange multiplier ``λ``.
* A variant-specific ``SIMULATE`` (Algorithm 1 for CPOMCPOW, Algorithm 2
  for CPFT-DPW).

:class:`ConstrainedMCTSMixin` implements the shared layer and declares
``_simulate_path_with_cost`` as abstract. Concrete subclasses
(``CPOMCPOW``, ``CPFT_DPW``, and any future variant) inherit the mixin
**first** so its overrides of ``action`` and ``_construct_tree_using_*``
win over the underlying unconstrained planner's implementations.

Classes:
    ConstrainedMCTSMixin: ABC providing the shared dual-ascent + Lagrangian
        action-selection scaffold, with one abstract method
        ``_simulate_path_with_cost``.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import ConstrainedEnvironment
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.dpw import _get_or_add_action_child
from POMDPPlanners.utils.tree_statistics import compute_arena_tree_metrics


class ConstrainedMCTSMixin(ABC):
    """Shared scaffold for constrained-POMDP MCTS planners.

    Implements paper Listing 1's ``PLAN``, ``GREEDYPOLICY`` (Lagrangian
    UCB + root greedy pick), ``ACTIONPROGWIDEN``, and the dual-ascent
    update on ``λ`` — together with constraint-parameter validation,
    cost-Q storage, and a boundary check on env-returned constraint
    cost vectors. Concrete planners implement only
    :meth:`_simulate_path_with_cost` (the paper's ``SIMULATE``).

    Inheritance pattern (mixin-first):

    .. code-block:: python

        class CPOMCPOW(ConstrainedMCTSMixin, POMCPOW): ...
        class CPFT_DPW(ConstrainedMCTSMixin, PFT_DPW): ...

    Required host-class surface (provided by ``POMCPOW`` / ``PFT_DPW`` via
    their parent chain ``ArenaDoubleProgressiveWideningMCTSPolicy →
    ArenaPathSimulationPolicy``): ``environment``, ``action_sampler``,
    ``depth``, ``k_a``, ``alpha_a``, ``exploration_constant``,
    ``min_visit_count_per_action``, ``discount_factor``, ``n_simulations``,
    ``time_out_in_seconds``, ``_learn_tree``, ``_is_terminal_belief``,
    ``_sample_random_action``.
    """

    # Attributes set by ``_init_constrained_state``. Declared here so type
    # checkers can resolve them on mixin methods; runtime values are
    # assigned by the host class's ``__init__``.
    constrained_env: ConstrainedEnvironment
    cost_budget: np.ndarray
    lambda_init: np.ndarray
    lambda_step: float
    return_minimal_cost: bool
    n_constraints: int
    _lambda: np.ndarray
    _lambda_max: np.ndarray
    _action_cost_q: Dict[int, np.ndarray]

    # ------------------------------------------------------------------
    # Constraint-parameter validation + init.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_pack_constraint_params(
        cost_budget: Union[float, np.ndarray],
        lambda_init: Union[float, np.ndarray],
        lambda_step: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        budget_arr = np.atleast_1d(np.asarray(cost_budget, dtype=np.float64))
        if budget_arr.ndim != 1:
            raise ValueError(f"cost_budget must be 1-D, got shape {budget_arr.shape}")
        if budget_arr.shape[0] == 0:
            raise ValueError(
                "cost_budget must have at least one constraint dimension; "
                "use the unconstrained sibling planner if no constraint is needed."
            )
        # Finite check before sign check: ``NaN < 0`` evaluates ``False``,
        # so a plain ``(arr < 0).any()`` test would silently admit NaN
        # (and ``+inf``) and corrupt λ-ascent downstream.
        if not np.isfinite(budget_arr).all():
            raise ValueError(f"cost_budget must be finite, got {budget_arr}")
        if (budget_arr < 0).any():
            raise ValueError(f"cost_budget must be element-wise non-negative, got {budget_arr}")
        k = budget_arr.shape[0]

        # ``np.atleast_1d`` lifts both Python scalars and 0-d ndarrays
        # (``np.array(0.0)``) to shape ``(1,)``. Length-1 inputs then
        # broadcast to ``(k,)`` so all "scalar-ish" input flavours work
        # uniformly.
        lambda_init_arr = np.atleast_1d(np.asarray(lambda_init, dtype=np.float64))
        if lambda_init_arr.ndim != 1:
            raise ValueError(f"lambda_init must be 1-D, got shape {lambda_init_arr.shape}")
        if lambda_init_arr.shape == (1,) and k > 1:
            lambda_init_arr = np.full(k, float(lambda_init_arr[0]), dtype=np.float64)
        if lambda_init_arr.shape != (k,):
            raise ValueError(
                f"lambda_init must broadcast to shape ({k},) to match cost_budget; "
                f"got shape {lambda_init_arr.shape}"
            )
        if not np.isfinite(lambda_init_arr).all():
            raise ValueError(f"lambda_init must be finite, got {lambda_init_arr}")
        if (lambda_init_arr < 0).any():
            raise ValueError(
                f"lambda_init must be element-wise non-negative, got {lambda_init_arr}"
            )

        if not np.isfinite(lambda_step):
            raise ValueError(f"lambda_step must be finite, got {lambda_step}")
        if lambda_step <= 0:
            raise ValueError(f"lambda_step must be positive, got {lambda_step}")
        return budget_arr, lambda_init_arr

    def _init_constrained_state(
        self,
        environment: ConstrainedEnvironment,
        cost_budget: np.ndarray,
        lambda_init: np.ndarray,
        lambda_step: float,
        return_minimal_cost: bool,
    ) -> None:
        self.constrained_env = environment
        self.cost_budget = cost_budget
        self.lambda_init = lambda_init
        self.lambda_step = lambda_step
        self.return_minimal_cost = return_minimal_cost
        self.n_constraints = cost_budget.shape[0]
        self._lambda = lambda_init.copy()
        self._lambda_max = self._compute_lambda_max(environment, cost_budget)
        self._action_cost_q = {}

    def _compute_lambda_max(
        self, environment: ConstrainedEnvironment, cost_budget: np.ndarray
    ) -> np.ndarray:
        """Compute the per-dimension upper bound on the Lagrange multiplier.

        Cap formula: ``λ_max[k] = (R_max − R_min) / ((1−γ) · ĉ_k)``.
        Without this cap, dual ascent on an infeasible problem grows ``λ``
        without bound; the cap keeps the Lagrangian-penalty term commensurate
        with the discounted reward range.

        Falls back to ``+∞`` per-dimension when:

        * The env has no ``reward_range`` (legacy or mock envs) — then the
          cap is effectively disabled and only the lower bound at ``0``
          applies, matching the planner's pre-cap behavior.
        * ``cost_budget[k] == 0`` — the constraint is "no cost ever";
          unbounded ``λ`` is the algorithm's natural enforcement mechanism.
        """
        reward_range = getattr(environment, "reward_range", None)
        if reward_range is None:
            return np.full(cost_budget.shape, np.inf, dtype=np.float64)
        r_min, r_max = reward_range
        denom = (1.0 - self.discount_factor) * cost_budget  # type: ignore[attr-defined]
        return np.divide(
            float(r_max - r_min),
            denom,
            out=np.full(cost_budget.shape, np.inf, dtype=np.float64),
            where=denom > 0,
        )

    def _reset_per_action_state(self) -> None:
        """Reset per-``action()`` mutable state.

        Default resets ``_lambda`` to ``lambda_init`` and clears the
        cost-Q dict. Subclasses with extra per-call caches (e.g. CPFT's
        ``_belief_immediate_cost``) override this and call
        ``super()._reset_per_action_state()``.
        """
        self._lambda = self.lambda_init.copy()
        self._action_cost_q = {}

    # ------------------------------------------------------------------
    # PLAN procedure (paper Listing 1, lines 1–8).
    # ------------------------------------------------------------------

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        """Plan one action by running ``n`` dual-ascent-coupled simulations.

        Mirrors ``PLAN(b)`` from the paper: for each iteration, run
        ``SIMULATE`` (variant-specific), pick the greedy Lagrangian
        action ``a*``, and update ``λ`` toward ``λ + α_i (QC(b, a*) − ĉ)``
        with non-negative projection. Returns the final greedy
        Lagrangian root action.
        """
        if self._is_terminal_belief(belief=belief):  # type: ignore[attr-defined]
            return (
                [self._sample_random_action(belief=belief)],  # type: ignore[attr-defined]
                PolicyRunData(info_variables=[]),
            )

        self._reset_per_action_state()
        tree, root_id = self._learn_tree(belief=belief)  # type: ignore[attr-defined]
        tree_metrics = compute_arena_tree_metrics(tree=tree, root_id=root_id)

        if not tree.get_children_ids(root_id):
            chosen_action = self._sample_random_action(belief=belief)  # type: ignore[attr-defined]
        else:
            chosen_action_id = self._lagrangian_best_action_id(tree=tree, belief_id=root_id)
            chosen_action = tree.get_action(chosen_action_id)

        return [chosen_action], PolicyRunData(info_variables=tree_metrics)

    # ------------------------------------------------------------------
    # Tree-construction loops threading dual ascent (Listing 1, lines 6–7).
    # ------------------------------------------------------------------

    def _construct_tree_using_n_simulations(self, tree: Tree, root_id: int) -> None:
        if self.n_simulations is None:  # type: ignore[attr-defined]
            raise ValueError("n_simulations must not be None")
        for _ in range(self.n_simulations):  # type: ignore[attr-defined]
            self._simulate_path_with_cost(tree=tree, belief_id=root_id, depth=0)
            self._dual_ascent_step(tree=tree, root_id=root_id)

    def _construct_tree_using_timeout(self, tree: Tree, root_id: int) -> None:
        if self.time_out_in_seconds is None:  # type: ignore[attr-defined]
            raise ValueError("time_out_in_seconds must not be None")
        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:  # type: ignore[attr-defined]
            self._simulate_path_with_cost(tree=tree, belief_id=root_id, depth=0)
            self._dual_ascent_step(tree=tree, root_id=root_id)

    def _dual_ascent_step(self, tree: Tree, root_id: int) -> None:
        if not tree.get_children_ids(root_id):
            return
        best_action_id = self._lagrangian_best_action_id(tree=tree, belief_id=root_id)
        qc_best = self._cost_q(best_action_id)
        # Clip to [0, λ_max] per dimension: lower bound = dual feasibility
        # (λ ≥ 0 for inequality constraints); upper bound = stability cap
        # ``(R_max − R_min) / ((1−γ) · ĉ_k)`` (see _compute_lambda_max).
        self._lambda = np.clip(
            self._lambda + self.lambda_step * (qc_best - self.cost_budget),
            0.0,
            self._lambda_max,
        )

    # ------------------------------------------------------------------
    # GREEDYPOLICY procedure (Listing 1, lines 9–11) — Lagrangian UCB +
    # greedy root pick.
    # ------------------------------------------------------------------

    def _lagrangian_ucb1(self, tree: Tree, belief_id: int) -> int:
        children = tree.get_children_ids(belief_id)
        belief_visits = tree.get_visit_count(belief_id)
        log_n = float(np.log(max(belief_visits, 1)))
        best_id = children[0]
        best_ucb = -float("inf")
        for cid in children:
            visits = max(tree.get_visit_count(cid), 1)
            q_lambda = tree.get_q_value(cid) - float(self._lambda @ self._cost_q(cid))
            ucb = q_lambda + self.exploration_constant * (log_n / visits) ** 0.5  # type: ignore[attr-defined]
            if ucb > best_ucb:
                best_ucb = ucb
                best_id = cid
        return best_id

    def _lagrangian_best_action_id(self, tree: Tree, belief_id: int) -> int:
        children = tree.get_children_ids(belief_id)
        best_id = children[0]
        best_score = -float("inf")
        for cid in children:
            score = tree.get_q_value(cid) - float(self._lambda @ self._cost_q(cid))
            if score > best_score:
                best_score = score
                best_id = cid
        return best_id

    # ------------------------------------------------------------------
    # ACTIONPROGWIDEN procedure (Listing 1, lines 12–17).
    # ------------------------------------------------------------------

    def _lagrangian_action_progressive_widening(self, tree: Tree, belief_id: int) -> int:
        # Root-only: ensure every existing action has been visited
        # ``min_visit_count_per_action`` times before any further widening.
        if tree.get_parent_id(belief_id) is None:
            for cid in tree.get_children_ids(belief_id):
                if tree.get_visit_count(cid) < self.min_visit_count_per_action:  # type: ignore[attr-defined]
                    return cid

        children = tree.get_children_ids(belief_id)
        belief_visits = tree.get_visit_count(belief_id)
        is_leaf = len(children) == 0

        if (
            is_leaf
            or belief_visits == 0
            or len(children) <= self.k_a * belief_visits**self.alpha_a  # type: ignore[attr-defined]
        ):
            action = self.action_sampler.sample()  # type: ignore[attr-defined]
            return _get_or_add_action_child(tree, belief_id, action, self.environment)  # type: ignore[attr-defined]

        return self._lagrangian_ucb1(tree=tree, belief_id=belief_id)

    # ------------------------------------------------------------------
    # Cost-Q storage (planner-local; tree.set_immediate_cost mirrors to
    # reward, which would clobber the reward channel).
    # ------------------------------------------------------------------

    def _cost_q(self, action_id: int) -> np.ndarray:
        existing = self._action_cost_q.get(action_id)
        if existing is None:
            return np.zeros(self.n_constraints, dtype=np.float64)
        return existing

    def _set_cost_q(self, action_id: int, value: np.ndarray) -> None:
        self._action_cost_q[action_id] = value

    # ------------------------------------------------------------------
    # Boundary read of env-returned constraint cost (shape + finite).
    # ------------------------------------------------------------------

    def _read_constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        cost = np.asarray(
            self.constrained_env.constraint_cost(state, action, next_state), dtype=np.float64
        )
        if cost.shape != (self.n_constraints,):
            raise ValueError(
                f"constraint_cost returned shape {cost.shape}, expected "
                f"({self.n_constraints},) — does the env's constraint dimensionality "
                f"match cost_budget?"
            )
        if not np.isfinite(cost).all():
            raise ValueError(
                f"constraint_cost returned non-finite values: {cost}; "
                f"a NaN/inf here would corrupt QC backups and λ-ascent."
            )
        return cost

    # ------------------------------------------------------------------
    # Abstract SIMULATE — variant-specific (Algorithm 1 for POMCPOW,
    # Algorithm 2 for PFT-DPW, etc.).
    # ------------------------------------------------------------------

    @abstractmethod
    def _simulate_path_with_cost(
        self, tree: Tree, belief_id: int, depth: int
    ) -> Tuple[float, np.ndarray]:
        """Run one tree-policy simulation from ``belief_id`` at ``depth``.

        Returns ``(discounted_reward_return, discounted_cost_vector)``.
        The cost vector has shape ``(self.n_constraints,)``.
        """
