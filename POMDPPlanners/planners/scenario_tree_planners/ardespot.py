# SPDX-License-Identifier: MIT

"""AR-DESPOT: Anytime Regularized DESPOT.

AR-DESPOT is the algorithm of Ye, Somani, Hsu & Lee (JAIR 2017, Sec. 5). It
searches the same determinized scenario tree as
:class:`~POMDPPlanners.planners.scenario_tree_planners.despot.DESPOT`, but it
carries the regularization *inside* the tree instead of applying it once at the
end, and it treats the search as anytime: the tree holds a complete, legal
answer after every trial, so an interruption at any point returns an action and
a longer budget returns a better one.

What that changes, relative to plain DESPOT
-------------------------------------------

Plain DESPOT keeps two numbers per belief node, ``l(b)`` and ``u(b)``, descends
on the upper bound, and (in this repository) charges ``lambda`` once, in a
dynamic program over the finished tree, purely to pick the final action. The
search itself never sees ``lambda``. AR-DESPOT charges it in the tree, and the
consequences reach every part of the algorithm:

1. **Three numbers per belief node.** ``l(b)``, ``U(b)`` and a third,
   ``mu(b)`` -- the *regularized* value. ``mu`` is what drives descent, the
   stopping test and pruning; ``l`` is only what the final action is read off.

2. **An unnormalized scale.** ``l``, ``mu``, ``l_0`` and ``rho`` all carry the
   factor ``gamma^Delta * |Phi_b| / K`` inside them. Plain DESPOT divides it
   back out at every node. Keeping it in is what lets a child's value be
   *added* to its parent's without renormalizing, and therefore what lets one
   ``lambda`` per action node be subtracted directly. ``U`` is the exception:
   it stays conditional (normalized by the node's own scenario count), exactly
   as in the reference, because it is only ever compared against ``l_0`` after
   being scaled back up in equation (12).

3. **Descent on ``ba_mu``**, not on the optimistic ``Q_u``.

4. **Excess uncertainty measured on ``mu - l``**::

       E(b) = (mu(b) - l(b)) - (|Phi_b| / K) * xi * (mu(root) - l(root))

   There is no ``gamma^(-d)`` factor here, unlike plain DESPOT's equation (5):
   the discount is already inside ``mu`` and ``l``.

5. **Blocking (paper eq. 12).** Before every descent step, the ancestors of the
   current node ``b`` are walked upward. An ancestor ``bp`` *blocks* when::

       (|Phi_bp| / K) * gamma^Delta_bp * U(bp) - l_0(bp)  <=  lambda * len

   where ``len`` is the number of belief levels from ``bp`` down to ``b``. The
   left side is the most value any policy could still gain below ``bp``; the
   right side is what the ``len`` extra policy nodes needed to collect it would
   cost. If the gain cannot pay for the size, the subtree is not worth building
   and ``b`` is collapsed onto its default policy
   (:meth:`ARDESPOT._make_default`, ``mu = l = l_0``) and backed up. This is
   the mechanism that makes the planner *regularized* rather than merely
   penalized at the end; plain DESPOT has nothing corresponding to it.

6. **The lower bound is no longer monotone.** Backup sets
   ``l(b) = max(l_0(b), max_a ba_l(b,a))`` -- the maximum over actions, *not*
   the running maximum with the node's previous ``l``. It has to be: blocking
   lowers a descendant's ``l`` back to its default value, and a running maximum
   would make that reduction invisible to every ancestor, leaving stale
   optimism in the tree that blocking had just retracted.

7. **Three simultaneous stopping conditions.** The trial loop runs while
   ``mu(root) - l(root) > epsilon_0`` *and* the wall clock is under
   ``time_out_in_seconds`` *and* the trial count is under ``n_simulations``.
   Plain DESPOT in this repository accepts exactly one budget; AR-DESPOT
   accepts either or both, because "anytime" means the wall clock is a first
   class stopping condition rather than an alternative to counting trials.

8. **Final action** ``argmax_a ba_l(b0, a)``, where
   ``ba_l = rho + sum_o l(b_o)`` already carries the ``-lambda``. So
   regularization is in the answer by construction, with no separate pass.

Equations, in the scale the implementation uses. For belief node ``b`` at
depth ``Delta`` with scenario set ``Phi_b``, and action child ``ba``:

* ``l_0(b) = (|Phi_b| / K) * gamma^Delta * L_0(b)``, with ``L_0`` the
  conditional default-policy value.
* ``mu_0(b) = max{ l_0(b), (|Phi_b| / K) * gamma^Delta * U_0(b) - lambda }``.
* ``rho(ba) = gamma^Delta * (sum_{phi in Phi_b} r(s_phi, a)) / K - lambda``.
* ``ba_l(ba) = rho(ba) + sum_o l(b_o)``;
  ``ba_mu(ba) = rho(ba) + sum_o mu(b_o)``.
* ``ba_U(ba) = (sum_phi r(s_phi,a) + gamma * sum_o |Phi_o| U(b_o)) / |Phi_b|``.
* ``l(b) = max{ l_0(b), max_a ba_l }``; ``mu(b) = max{ l_0(b), max_a ba_mu }``;
  ``U(b) = max_a ba_U``.

Two deliberate departures from the reference implementation, both because this
planner must stay comparable to plain DESPOT at the same ``depth``:

* The reference expands nodes at ``Delta <= D`` and therefore collects ``D+1``
  steps of reward. Here the trial stops at ``Delta >= depth``, so both planners
  optimise the same ``depth``-step discounted return and a QA run at one
  ``depth`` compares like with like.
* The reference drops a scenario that has already terminated, giving it no
  child and no weight. That is kept here, and it is correct in this scale
  precisely because the scale is unnormalized: a terminated scenario
  contributes ``0`` to ``sum_o l(b_o)``, and ``U`` divides by the *parent's*
  scenario count, so the missing weight contributes ``0`` there too. Plain
  DESPOT needs its explicit ``TERMINAL_OBSERVATION`` branch only because its
  conditional scale would otherwise renormalize the survivors.

References:
    Ye, N., Somani, A., Hsu, D., & Lee, W. S. (2017). DESPOT: Online POMDP
    Planning with Regularization. JAIR 58, 231-266. Algorithm 1 (RUNTRIAL,
    BACKUP, EXCESSUNCERTAINTY) and equation (12).

    Reference implementation read for structure (not transliterated):
    https://github.com/JuliaPOMDP/ARDESPOT.jl -- ``src/planner.jl``,
    ``src/tree.jl``, ``src/pomdps_glue.jl``.

Classes:
    ARDESPOT: The planner.
    ARDESPOTMetrics: Names of the search diagnostics it reports.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief, UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.tree.arena import BELIEF, Tree
from POMDPPlanners.planners.planners_utils.scenario_streams import ScenarioRandomStreams
from POMDPPlanners.planners.scenario_tree_planners.despot import (
    TINY,
    DESPOT,
)
from POMDPPlanners.utils.tree_statistics import TreeMetrics, compute_arena_tree_metrics


class ARDESPOTMetrics(Enum):
    """Search diagnostics reported through ``PolicyRunData.info_variables``.

    Every value is measured over the single decision that produced it and is
    reset when the next :meth:`ARDESPOT.action` call builds a new tree; nothing
    accumulates across decisions. The names are prefixed ``ardespot_`` rather
    than reusing DESPOT's ``despot_`` names on purpose: ``despot_root_gap`` is
    ``u - l`` on a conditional scale and ``ardespot_root_gap`` is ``mu - l`` on
    an unnormalized one, so a record that merged them under one name would
    average two different quantities.
    """

    #: Trials completed in this decision. Unit: count.
    N_TRIALS = "ardespot_n_trials"
    #: ``l(root)``. Unit: discounted reward over ``depth`` steps. At the root
    #: ``Delta = 0`` and the weight is 1, so this is directly comparable to
    #: ``despot_root_lower_bound`` -- except that it is charged ``lambda`` per
    #: policy node, so at ``lambda > 0`` it sits below DESPOT's by that cost.
    ROOT_LOWER_BOUND = "ardespot_root_lower_bound"
    #: ``mu(root)``, the regularized value the search is driven by. Same unit.
    ROOT_REGULARIZED_VALUE = "ardespot_root_regularized_value"
    #: ``U(root)``, the conditional upper bound. Same unit. Reported alongside
    #: ``mu`` because the two answer different questions: ``U`` bounds the
    #: unregularized value, ``mu`` the regularized one.
    ROOT_UPPER_BOUND = "ardespot_root_upper_bound"
    #: ``mu(root) - l(root)``: the quantity the stopping rule tests against
    #: ``epsilon_0``. Unit: same. Not ``U - l``.
    ROOT_GAP = "ardespot_root_gap"
    #: Scenarios ``K`` drawn for this decision. Unit: count.
    N_SCENARIOS = "ardespot_n_scenarios"
    #: Belief nodes allocated, including the unexpanded fringe. Unit: count.
    N_BELIEF_NODES = "ardespot_n_belief_nodes"
    #: Belief nodes a trial entered and backed up -- the paper's ``|D|``.
    #: Unit: count.
    N_TREE_BELIEF_NODES = "ardespot_n_tree_belief_nodes"
    #: Deepest belief depth a trial reached, in belief steps. Unit: count.
    #: Directly comparable to ``depth``, and *not* to the arena's
    #: ``tree_max_depth``, which counts arena edges and is about twice as large.
    MAX_TRIAL_DEPTH = "ardespot_max_trial_depth"
    #: Times a node was collapsed onto its default policy because an ancestor
    #: blocked it (equation 12). Unit: count. This is regularization's direct
    #: footprint on the search; it is 0 for every ``lambda`` small enough that
    #: no ancestor can ever block, which is the honest signal that the pruning
    #: constant is doing nothing.
    N_BLOCKED = "ardespot_n_blocked"
    #: Times a trial ran past the horizon and the node beyond it was collapsed
    #: onto its default policy. Unit: count. Distinct from ``N_BLOCKED``:
    #: this one is the horizon, not the regularizer.
    N_DEPTH_DEFAULTS = "ardespot_n_depth_defaults"
    #: 1 if the loop ended because ``mu(root) - l(root) <= epsilon_0``.
    #: Unit: flag.
    GAP_CLOSED = "ardespot_gap_closed"
    #: 1 if the loop ended because the wall-clock budget ran out. Unit: flag.
    #: The three stop flags are reported separately, not as one code, because a
    #: run in which the clock always won and one in which the gap always closed
    #: need different follow-up and a single code would not average.
    TIME_EXHAUSTED = "ardespot_time_exhausted"
    #: 1 if the loop ended because the trial budget ran out. Unit: flag.
    TRIALS_EXHAUSTED = "ardespot_trials_exhausted"
    #: 1 if a trial could neither expand nor descend, so further trials would
    #: repeat identically. Unit: flag.
    STALLED = "ardespot_stalled"
    #: Times a leaf's conditional bounds came back with ``L_0 > U_0`` by more
    #: than float noise. Unit: count. Inherited from DESPOT's leaf bound code.
    BOUND_CLAMPS = "ardespot_bound_clamps"


@dataclass
class _ARBeliefData:
    """Per-belief-node bookkeeping the arena ``Tree`` columns do not cover.

    The arena carries ``l`` in ``lower_confidence_bound``, ``U`` in
    ``upper_confidence_bound`` and ``mu`` in ``v_value``. Everything below is
    what has no column.
    """

    #: Belief depth ``Delta`` in belief steps, not arena edges.
    depth: int
    #: ``l_0(b)``, unnormalized. Equation (12) reads it, and backup floors both
    #: ``l`` and ``mu`` on it, so it must survive after ``l`` has moved.
    default_value: float
    #: ``U_0(b)``, the node's own conditional upper bound before any backup.
    #: Kept for the dump; the live value lives in the arena column.
    initial_upper: float
    #: Whether the node's action children have been created.
    expanded: bool = False
    #: Whether every scenario carried here is in a terminal state.
    terminal: bool = False
    #: Whether a trial entered and backed up this node -- membership in ``D``.
    in_tree: bool = False
    #: Whether the node has been collapsed onto its default policy, and why.
    #: ``None`` means it has not.
    defaulted_by: Optional[str] = None
    #: Scenario ids carried here, in scenario order.
    scenario_ids: List[int] = field(default_factory=list)


@dataclass
class _ARActionData:
    """Per-action-node bookkeeping. ``ba_l`` and ``ba_U`` live in arena columns.

    ``rho`` and ``mu`` do not: the arena has no column whose meaning matches a
    ``lambda``-charged unnormalized first-step reward, and writing them into
    ``immediate_reward`` would make the arena's generic reward metrics report a
    penalized quantity as if it were an environment reward.
    """

    #: ``rho(ba) = gamma^Delta * (sum_phi r) / K - lambda``, unnormalized.
    rho: float
    #: ``ba_mu = rho + sum_o mu(b_o)``, unnormalized. The descent quantity.
    mu: float = -np.inf


class ARDESPOT(DESPOT):
    """Anytime Regularized DESPOT.

    See the module docstring for the algorithm, its equations and the eight
    ways it differs from :class:`DESPOT`.

    Subclasses :class:`DESPOT` for what is genuinely shared and unchanged:
    systematic scenario resampling from the belief, the conditional leaf bounds
    ``(L_0, U_0)``, the one-sequence-per-node default policy, observation
    keying, the determinized ``(scenario, depth)`` random streams, and the
    search-state dump plumbing. Everything that touches the value scale, the
    descent, the backup, the stopping rule or the final action is overridden.

    Args:
        environment: Discrete-action POMDP to plan in.
        discount_factor: ``gamma``, in ``(0, 1]``.
        depth: Search horizon ``D`` in belief steps.
        name: Unique identifier for this planner instance.
        n_scenarios: ``K``, the number of determinized scenarios per decision.
        xi: Rate of target-gap reduction, in ``(0, 1)``. The reference's name
            for the parameter plain DESPOT calls ``eta``. Passed through to
            ``eta`` on the base class, and readable back as either name, so
            ``config_id`` carries one value rather than two names for it.
        pruning_constant: ``lambda``. Unlike in :class:`DESPOT`, ``0`` does not
            disable a separate pass -- it makes equation (12) unable to block
            anything, so the search degenerates to an unregularized anytime
            DESPOT on the unnormalized scale.
        epsilon_0: Absolute target gap on ``mu(root) - l(root)``. The search
            stops once the gap is at or below it. ``0`` means "only stop on a
            budget", which is the reference's default.
        max_reward: ``R_max`` for the trivial upper bound. Defaults to the
            environment's declared ``reward_range`` maximum.
        min_reward: ``R_min``, charged for the horizon a shortened rollout does
            not cover.
        rollout_depth: Steps of default policy used for the leaf lower bound.
            Defaults to ``depth``.
        scenario_seed: Base seed for the determinized random-number table and
            for tie-breaking.
        use_determinized_scenarios: Whether to pin transitions to
            ``(scenario, depth)``.
        time_out_in_seconds: Wall-clock budget per decision. May be combined
            with ``n_simulations``; at least one of the two is required.
        n_simulations: Maximum trials per decision.

    Raises:
        ValueError: If neither budget is given, if a numeric parameter is
            outside its stated range, or if no ``max_reward`` is available.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(0)
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = ARDESPOT(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     name="ExampleARDESPOT",
        ...     n_scenarios=8,
        ...     pruning_constant=0.01,
        ...     n_simulations=20,
        ... )
        >>> belief = get_initial_belief(tiger, n_particles=20)
        >>> actions, run_data = planner.action(belief)
        >>> actions[0] in tiger.get_actions()
        True
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        name: str,
        n_scenarios: int = 50,
        xi: float = 0.95,
        pruning_constant: float = 0.01,
        epsilon_0: float = 0.0,
        max_reward: Optional[float] = None,
        min_reward: Optional[float] = None,
        rollout_depth: Optional[int] = None,
        scenario_seed: int = 0,
        use_determinized_scenarios: bool = True,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        if time_out_in_seconds is None and n_simulations is None:
            raise ValueError(
                "AR-DESPOT needs at least one of time_out_in_seconds and n_simulations. "
                "Unlike DESPOT it accepts both at once: the wall clock is a first-class "
                "stopping condition for an anytime search, not an alternative to counting "
                "trials."
            )
        if not isinstance(epsilon_0, (int, float)) or isinstance(epsilon_0, bool):
            raise TypeError(f"epsilon_0 must be a number, got {type(epsilon_0).__name__}")
        if epsilon_0 < 0.0:
            raise ValueError(f"epsilon_0 must be non-negative, got {epsilon_0}")

        # The base class refuses two budgets at once, which is exactly the
        # combination this planner needs. Hand it the trial budget when both
        # are present and keep the clock here; ``_construct_tree`` below runs
        # its own loop and never reaches the base class's two loops.
        self._both_budgets = time_out_in_seconds is not None and n_simulations is not None
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            name=name,
            n_scenarios=n_scenarios,
            eta=xi,
            pruning_constant=pruning_constant,
            max_reward=max_reward,
            min_reward=min_reward,
            rollout_depth=rollout_depth,
            scenario_seed=scenario_seed,
            use_determinized_scenarios=use_determinized_scenarios,
            time_out_in_seconds=None if self._both_budgets else time_out_in_seconds,
            n_simulations=n_simulations,
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        # Restored after ``super().__init__`` so both budgets are visible to
        # the loop and to ``config_id``; the base class stored only one.
        self.time_out_in_seconds = time_out_in_seconds
        self.n_simulations = n_simulations
        self.epsilon_0 = float(epsilon_0)

        # --- transient per-call search state ---
        self._n_blocked: int = 0
        self._n_depth_defaults: int = 0
        self._time_exhausted: bool = False
        self._trials_exhausted: bool = False
        self._tiebreak_rng = np.random.default_rng(scenario_seed + 1)

    @property
    def xi(self) -> float:
        """The reference's name for the target-gap reduction rate.

        A property over :attr:`eta` rather than a second attribute: two
        attributes holding one number would appear twice in ``config_id`` and
        could be set apart by a caller, which would then mean nothing.
        """
        return self.eta

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return [metric.value for metric in TreeMetrics] + [
            metric.value for metric in ARDESPOTMetrics
        ]

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        """Plan one decision and report the search that produced it."""
        if self._is_terminal_belief(belief=belief):
            return [self._sample_random_action(belief=belief)], PolicyRunData(info_variables=[])

        saved_rng_state = (
            ScenarioRandomStreams.capture_global_state()
            if self.use_determinized_scenarios
            else None
        )
        try:
            tree, root_id = self._learn_tree(belief=belief)
        finally:
            if saved_rng_state is not None:
                ScenarioRandomStreams.restore_global_state(saved_rng_state)

        self._last_tree = tree
        self._last_root_id = root_id

        if not tree.children_ids[root_id]:
            chosen = self._sample_random_action(belief=belief)
        else:
            chosen = self._select_final_action(tree=tree, root_id=root_id)

        info_variables = compute_arena_tree_metrics(tree=tree, root_id=root_id)
        info_variables.extend(self._ardespot_metrics(tree=tree, root_id=root_id))

        if self._search_state_dump_dir is not None:
            self._write_search_state_dump(tree=tree, root_id=root_id)

        return [chosen], PolicyRunData(info_variables=info_variables)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def _learn_tree(self, belief: Belief) -> Tuple[Tree, int]:
        tree = Tree()
        capacity = self._effective_reserve_capacity()
        if capacity > 0:
            tree.reserve(capacity)

        self._call_index += 1
        self._streams = ScenarioRandomStreams(
            n_scenarios=self.n_scenarios,
            max_depth=max(self.depth, self.rollout_depth),
            base_seed=self.scenario_seed + self._call_index,
        )
        self._n_trials = 0
        self._max_trial_depth = 0
        self._gap_closed = False
        self._stalled = False
        self._bound_clamps = 0
        self._n_blocked = 0
        self._n_depth_defaults = 0
        self._time_exhausted = False
        self._trials_exhausted = False

        states = self._sample_scenario_states(belief=belief)
        root_id = self._add_belief_node(
            tree=tree,
            states=states,
            scenario_ids=list(range(self.n_scenarios)),
            depth=0,
            parent_id=None,
            observation=None,
            obs_key=None,
        )
        self._root_id = root_id

        self._construct_tree(tree=tree, root_id=root_id)

        self._last_tree_size = len(tree)
        return tree, root_id

    def _construct_tree(self, tree: Tree, root_id: int) -> None:
        """The anytime trial loop: three stopping conditions, checked together.

        Which one fired is recorded, because "the clock ran out every decision"
        and "the gap closed every decision" call for opposite adjustments and
        an unlabelled trial count cannot tell them apart.
        """
        start_time = time.time()
        while True:
            if self._root_gap(tree=tree, root_id=root_id) <= self.epsilon_0 + TINY:
                self._gap_closed = True
                return
            if self._stalled:
                return
            if self.n_simulations is not None and self._n_trials >= self.n_simulations:
                self._trials_exhausted = True
                return
            if (
                self.time_out_in_seconds is not None
                and time.time() - start_time >= self.time_out_in_seconds
            ):
                self._time_exhausted = True
                return
            self._simulate_path(tree=tree, belief_id=root_id, depth=0)

    def _root_gap(self, tree: Tree, root_id: int) -> float:
        """``mu(root) - l(root)``, the quantity ``epsilon_0`` is compared to."""
        return float(tree.v_value[root_id] - tree.lower_confidence_bound[root_id])

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> Optional[float]:
        """One AR-DESPOT trial: ``explore!`` then ``backup!``.

        Always returns ``None``; the base class's hook is typed
        ``Optional[float]`` because an MCTS simulation hands a sampled return
        back, whereas a trial here leaves its result in the nodes.
        """
        del depth  # a trial always starts at the root, at belief depth 0.
        leaf_id = self._explore(tree=tree, belief_id=belief_id)
        self._backup(tree=tree, belief_id=leaf_id)
        self._n_trials += 1

    def _explore(self, tree: Tree, belief_id: int) -> int:
        """Descend until the node is resolved, blocked, or past the horizon.

        Returns the node the trial stopped at, which is where the backup starts.
        """
        node_id = belief_id
        expanded_anything = False
        steps = 0

        while True:
            data: _ARBeliefData = tree.data[node_id]
            if data.depth >= self.depth:
                # Past the horizon: the node is worth its default policy and
                # nothing more, so say so rather than leaving optimistic
                # initial bounds for an ancestor to back up.
                self._make_default(tree=tree, belief_id=node_id, reason="depth")
                self._n_depth_defaults += 1
                break
            if data.terminal:
                break
            if self._excess_uncertainty(tree=tree, belief_id=node_id) <= 0.0:
                break
            if self._prune(tree=tree, belief_id=node_id):
                break

            if not data.expanded:
                self._expand(tree=tree, belief_id=node_id)
                expanded_anything = True

            tree.increment_visit_count(node_id)
            data.in_tree = True
            self._max_trial_depth = max(self._max_trial_depth, data.depth)

            next_id = self._next_best(tree=tree, belief_id=node_id)
            if next_id is None:
                break
            node_id = next_id
            steps += 1

        if not expanded_anything and steps == 0:
            # A trial that neither expanded nor moved will repeat identically
            # forever, and under a wall-clock budget would burn the whole
            # decision looking busy. Reported, not hidden.
            self._stalled = True
        return node_id

    def _excess_uncertainty(self, tree: Tree, belief_id: int) -> float:
        """``(mu(b) - l(b)) - (|Phi_b|/K) * xi * (mu(root) - l(root))``.

        No ``gamma^(-d)``: the discount is already inside ``mu`` and ``l`` on
        this scale, so re-applying it would target deep nodes twice.
        """
        width = tree.v_value[belief_id] - tree.lower_confidence_bound[belief_id]
        target = tree.weight[belief_id] * self.xi * self._root_gap(tree=tree, root_id=self._root_id)
        return float(width - target)

    def _next_best(self, tree: Tree, belief_id: int) -> Optional[int]:
        """Best action child by ``ba_mu``, then its best observation child by ``E``.

        The action step is the one that differs most visibly from plain DESPOT,
        which descends on ``Q_u``: here the regularizer is already inside
        ``ba_mu``, so an action whose subtree cannot pay for its own size is
        not followed in the first place.
        """
        action_children = tree.get_children_ids(belief_id)
        if not action_children:
            return None

        best_action_id = max(action_children, key=lambda action_id: tree.data[action_id].mu)
        tree.increment_visit_count(best_action_id)

        obs_children = tree.get_children_ids(best_action_id)
        if not obs_children:
            return None
        return max(
            obs_children,
            key=lambda child_id: self._excess_uncertainty(tree=tree, belief_id=child_id),
        )

    # ------------------------------------------------------------------
    # blocking (equation 12)
    # ------------------------------------------------------------------

    def _prune(self, tree: Tree, belief_id: int) -> bool:
        """Collapse ``belief_id`` and its ancestors while an ancestor blocks them.

        Returns whether anything was collapsed; the trial stops descending if
        so, because the node it was about to enter is now worth exactly its
        default policy and there is nothing left to resolve there.
        """
        node_id = belief_id
        blocked = False
        while node_id != self._root_id:
            if self._find_blocker(tree=tree, belief_id=node_id) is None:
                break
            self._make_default(tree=tree, belief_id=node_id, reason="blocked")
            self._n_blocked += 1
            self._backup(tree=tree, belief_id=node_id)
            blocked = True
            node_id = self._parent_belief_id(tree=tree, belief_id=node_id)
            if node_id is None:
                break
        return blocked

    def _find_blocker(self, tree: Tree, belief_id: int) -> Optional[int]:
        """Equation (12): the nearest ancestor whose remaining gain cannot pay.

        Walks up from ``belief_id``'s parent. For an ancestor ``bp`` at
        distance ``len`` belief levels, the most any policy could still gain
        below it is ``(|Phi_bp|/K) * gamma^Delta_bp * U(bp) - l_0(bp)``, and
        the ``len`` extra policy nodes needed to collect it cost
        ``lambda * len``. If the gain does not cover the cost, ``bp`` blocks.

        The root is never a blocker: blocking it would mean refusing to plan at
        all, and the final action still has to come from somewhere.
        """
        length = 1
        ancestor_id = self._parent_belief_id(tree=tree, belief_id=belief_id)
        while ancestor_id is not None and ancestor_id != self._root_id:
            data: _ARBeliefData = tree.data[ancestor_id]
            gain = (
                tree.weight[ancestor_id]
                * (self.discount_factor**data.depth)
                * tree.upper_confidence_bound[ancestor_id]
                - data.default_value
            )
            if gain <= self.pruning_constant * length:
                return ancestor_id
            ancestor_id = self._parent_belief_id(tree=tree, belief_id=ancestor_id)
            length += 1
        return None

    def _make_default(self, tree: Tree, belief_id: int, reason: str) -> None:
        """``mu(b) <- l_0(b)``, ``l(b) <- l_0(b)``: the node is worth its default policy.

        ``U`` is deliberately left alone. Equation (12) reads ``U`` on the way
        up the ancestor chain, and lowering it here would make a node's own
        collapse retroactively change whether its ancestors block, which the
        paper does not do.
        """
        data: _ARBeliefData = tree.data[belief_id]
        tree.v_value[belief_id] = data.default_value
        tree.lower_confidence_bound[belief_id] = data.default_value
        data.defaulted_by = reason

    def _parent_belief_id(self, tree: Tree, belief_id: int) -> Optional[int]:
        """The belief node two arena edges up (through the action node)."""
        del self
        action_id = tree.get_parent_id(belief_id)
        if action_id is None:
            return None
        return tree.get_parent_id(action_id)

    # ------------------------------------------------------------------
    # expansion and backup
    # ------------------------------------------------------------------

    def _expand(self, tree: Tree, belief_id: int) -> None:
        """Create one action child per action, and its observation children.

        Scenarios already in a terminal state are dropped: they earn nothing,
        contribute ``0`` to ``sum_o l(b_o)``, and ``U`` normalizes by the
        parent's count, so omitting them is exact on this scale. (Plain DESPOT
        cannot do that -- its conditional scale would renormalize over the
        survivors -- which is why it carries a ``TERMINAL_OBSERVATION``
        branch and this does not.)
        """
        data: _ARBeliefData = tree.data[belief_id]
        node_depth = data.depth
        states = self._node_states(tree=tree, belief_id=belief_id)
        scenario_ids = list(data.scenario_ids)
        n_at_node = len(scenario_ids)
        discount_at_node = self.discount_factor**node_depth

        for action in self.environment.get_actions():  # type: ignore[attr-defined]
            groups: Dict[Any, Tuple[Any, List[Any], List[int]]] = {}
            reward_sum = 0.0

            for scenario_id, state in zip(scenario_ids, states):
                if self.environment.is_terminal(state=state):
                    continue
                if self.use_determinized_scenarios and self._streams is not None:
                    self._streams.activate(scenario_id=scenario_id, depth=node_depth)
                next_state, observation, reward = self.environment.sample_next_step(
                    state=state, action=action
                )
                reward_sum += reward
                key = self._observation_key(observation)
                group = groups.get(key)
                if group is None:
                    groups[key] = (observation, [next_state], [scenario_id])
                else:
                    group[1].append(next_state)
                    group[2].append(scenario_id)

            action_id = tree.add_action_node(
                action=action,
                parent_id=belief_id,
                action_key=self.environment.hash_action(action),
            )
            # The arena's ``immediate_reward`` keeps the *raw* scenario sum, so
            # the generic tree metrics report an environment reward and not a
            # lambda-penalised one. ``rho`` lives in the node data.
            tree.set_immediate_reward(action_id, reward_sum)
            rho = discount_at_node * reward_sum / self.n_scenarios - self.pruning_constant
            tree.data[action_id] = _ARActionData(rho=rho)

            for key, (observation, next_states, child_scenarios) in groups.items():
                self._add_belief_node(
                    tree=tree,
                    states=next_states,
                    scenario_ids=child_scenarios,
                    depth=node_depth + 1,
                    parent_id=action_id,
                    observation=observation,
                    obs_key=key,
                )

            self._update_action_bounds(tree=tree, action_id=action_id, parent_count=n_at_node)

        data.expanded = True

    def _add_belief_node(  # pylint: disable=too-many-arguments
        self,
        tree: Tree,
        states: List[Any],
        scenario_ids: List[int],
        depth: int,
        parent_id: Optional[int],
        observation: Any,
        obs_key: Any,
    ) -> int:
        """Allocate a belief node with ``(l_0, mu_0, U_0)`` on the AR scale."""
        weight = len(scenario_ids) / self.n_scenarios
        node_id = tree.add_belief_node(
            belief=UnweightedParticleBeliefStateUpdate(particles=list(states)),
            observation=observation,
            weight=weight,
            parent_id=parent_id,
            obs_key=obs_key,
        )
        tree.sample[node_id] = list(scenario_ids)

        terminal = all(self.environment.is_terminal(state=state) for state in states)
        conditional_lower, conditional_upper = self._initial_bounds(
            states=states, scenario_ids=scenario_ids, depth=depth, terminal=terminal
        )
        scale = weight * (self.discount_factor**depth)
        lower = scale * conditional_lower
        # ``mu_0 = max(l_0, scale * U_0 - lambda)``: the regularized value of a
        # node nobody has expanded is either "run the default policy" or "keep
        # planning here", and the second option already owes one lambda.
        regularized = max(lower, scale * conditional_upper - self.pruning_constant)

        tree.lower_confidence_bound[node_id] = lower
        tree.upper_confidence_bound[node_id] = conditional_upper
        tree.v_value[node_id] = regularized
        tree.data[node_id] = _ARBeliefData(
            depth=depth,
            default_value=lower,
            initial_upper=conditional_upper,
            terminal=terminal,
            scenario_ids=list(scenario_ids),
        )
        return node_id

    def _update_action_bounds(  # pylint: disable=arguments-differ
        self, tree: Tree, action_id: int, parent_count: Optional[int] = None
    ) -> None:
        """Recompute ``ba_l``, ``ba_mu`` and ``ba_U`` for one action.

        ``ba_l`` and ``ba_mu`` are plain sums over the observation children --
        no weights, because the children's values already carry them. ``ba_U``
        is the one conditional quantity, so it *is* weighted, by scenario
        counts, and divided by the parent's own count.
        """
        action_data: _ARActionData = tree.data[action_id]
        if parent_count is None:
            parent_id = tree.get_parent_id(action_id)
            if parent_id is None:
                raise ValueError(f"action node {action_id} has no parent belief node")
            parent_count = len(tree.data[parent_id].scenario_ids)

        lower_sum = 0.0
        regularized_sum = 0.0
        weighted_upper = 0.0
        for child_id in tree.get_children_ids(action_id):
            lower_sum += tree.lower_confidence_bound[child_id]
            regularized_sum += tree.v_value[child_id]
            weighted_upper += (
                len(tree.data[child_id].scenario_ids) * tree.upper_confidence_bound[child_id]
            )

        reward_sum = tree.get_immediate_reward(action_id) or 0.0
        action_data.mu = action_data.rho + regularized_sum
        lower = action_data.rho + lower_sum
        upper = (
            (reward_sum + self.discount_factor * weighted_upper) / parent_count
            if parent_count > 0
            else 0.0
        )

        tree.lower_confidence_bound[action_id] = lower
        tree.upper_confidence_bound[action_id] = upper
        # ``q_value`` carries ``ba_l`` because that is what the final decision
        # is read off, so the arena's generic best-action helper and this
        # planner's own rule cannot disagree.
        tree.q_value[action_id] = lower

    def _backup(self, tree: Tree, belief_id: int) -> None:
        """Walk from ``belief_id`` to the root, recomputing every node on the way.

        ``l(b) = max(l_0(b), max_a ba_l)`` -- the maximum over actions, not the
        running maximum with the node's previous ``l``. Blocking lowers a
        descendant's ``l``, and a running maximum would hide that reduction
        from every ancestor, leaving retracted optimism in the tree. This is
        the one place where AR-DESPOT's backup is not merely a rescaling of
        plain DESPOT's.
        """
        node_id = self._parent_belief_id(tree=tree, belief_id=belief_id)
        while node_id is not None:
            data: _ARBeliefData = tree.data[node_id]
            n_at_node = len(data.scenario_ids)

            best_lower = -np.inf
            best_regularized = -np.inf
            best_upper = -np.inf
            for action_id in tree.get_children_ids(node_id):
                self._update_action_bounds(tree=tree, action_id=action_id, parent_count=n_at_node)
                best_lower = max(best_lower, tree.lower_confidence_bound[action_id])
                best_regularized = max(best_regularized, tree.data[action_id].mu)
                best_upper = max(best_upper, tree.upper_confidence_bound[action_id])

            if best_upper > -np.inf:
                tree.lower_confidence_bound[node_id] = max(data.default_value, best_lower)
                tree.v_value[node_id] = max(data.default_value, best_regularized)
                tree.upper_confidence_bound[node_id] = best_upper
                # A node whose value has been recomputed from live children is
                # no longer standing on its default policy.
                data.defaulted_by = None
            data.in_tree = True

            if node_id == self._root_id:
                return
            node_id = self._parent_belief_id(tree=tree, belief_id=node_id)

    # ------------------------------------------------------------------
    # final decision
    # ------------------------------------------------------------------

    def _select_final_action(self, tree: Tree, root_id: int) -> Any:
        """``argmax_a ba_l(b0, a)``, ties broken at random.

        ``ba_l`` already carries ``-lambda`` per node, so there is no separate
        regularization pass: the answer is the regularized one by construction.

        Ties are broken from a private generator rather than by taking the
        first action, following the reference. A deterministic first-index
        tie-break is not neutral -- when every action's lower bound is still 0,
        which happens whenever the horizon is too short for the default policy
        to reach any reward, it returns the same action every time and looks
        like a decision.
        """
        children = list(tree.get_children_ids(root_id))
        best_lower = max(tree.lower_confidence_bound[action_id] for action_id in children)
        tied = [
            action_id
            for action_id in children
            if tree.lower_confidence_bound[action_id] >= best_lower - TINY
        ]
        if len(tied) == 1:
            return tree.get_action(tied[0])
        return tree.get_action(tied[int(self._tiebreak_rng.integers(0, len(tied)))])

    # ------------------------------------------------------------------
    # metrics and search-state export
    # ------------------------------------------------------------------

    def _ardespot_metrics(self, tree: Tree, root_id: int) -> List[PolicyInfoVariable]:
        lower = float(tree.lower_confidence_bound[root_id])
        regularized = float(tree.v_value[root_id])
        upper = float(tree.upper_confidence_bound[root_id])
        n_belief_nodes = 0
        n_tree_belief_nodes = 0
        for node_id in range(len(tree)):
            if tree.kind[node_id] != BELIEF:
                continue
            n_belief_nodes += 1
            node_data = tree.data[node_id]
            if isinstance(node_data, _ARBeliefData) and node_data.in_tree:
                n_tree_belief_nodes += 1

        values: List[Tuple[ARDESPOTMetrics, float]] = [
            (ARDESPOTMetrics.N_TRIALS, self._n_trials),
            (ARDESPOTMetrics.ROOT_LOWER_BOUND, lower),
            (ARDESPOTMetrics.ROOT_REGULARIZED_VALUE, regularized),
            (ARDESPOTMetrics.ROOT_UPPER_BOUND, upper),
            (ARDESPOTMetrics.ROOT_GAP, regularized - lower),
            (ARDESPOTMetrics.N_SCENARIOS, self.n_scenarios),
            (ARDESPOTMetrics.N_BELIEF_NODES, n_belief_nodes),
            (ARDESPOTMetrics.N_TREE_BELIEF_NODES, n_tree_belief_nodes),
            (ARDESPOTMetrics.MAX_TRIAL_DEPTH, self._max_trial_depth),
            (ARDESPOTMetrics.N_BLOCKED, self._n_blocked),
            (ARDESPOTMetrics.N_DEPTH_DEFAULTS, self._n_depth_defaults),
            (ARDESPOTMetrics.GAP_CLOSED, int(self._gap_closed)),
            (ARDESPOTMetrics.TIME_EXHAUSTED, int(self._time_exhausted)),
            (ARDESPOTMetrics.TRIALS_EXHAUSTED, int(self._trials_exhausted)),
            (ARDESPOTMetrics.STALLED, int(self._stalled)),
            (ARDESPOTMetrics.BOUND_CLAMPS, self._bound_clamps),
        ]
        return [PolicyInfoVariable(name=metric.value, value=value) for metric, value in values]

    def _node_record(self, tree: Tree, node_id: int) -> Dict[str, Any]:
        """DESPOT's dump entry plus the quantities only AR-DESPOT has."""
        record = super()._node_record(tree=tree, node_id=node_id)
        node_data = tree.data[node_id]
        if isinstance(node_data, _ARBeliefData):
            record["depth"] = node_data.depth
            record["l_0"] = node_data.default_value
            record["mu"] = tree.v_value[node_id]
            record["initial_upper"] = node_data.initial_upper
            record["expanded"] = node_data.expanded
            record["terminal"] = node_data.terminal
            record["in_tree"] = node_data.in_tree
            record["defaulted_by"] = node_data.defaulted_by
        elif isinstance(node_data, _ARActionData):
            record["rho"] = node_data.rho
            record["ba_mu"] = node_data.mu
            record["ba_l"] = tree.lower_confidence_bound[node_id]
            record["ba_upper"] = tree.upper_confidence_bound[node_id]
            record["reward_sum"] = tree.immediate_reward[node_id]
        return record

    def _export_config(self) -> Dict[str, Any]:
        config = super()._export_config()
        config.pop("eta", None)
        config["xi"] = self.xi
        config["epsilon_0"] = self.epsilon_0
        return config

    def _export_search_summary(self, root_id: int) -> Dict[str, Any]:
        summary = super()._export_search_summary(root_id=root_id)
        summary["n_blocked"] = self._n_blocked
        summary["n_depth_defaults"] = self._n_depth_defaults
        summary["time_exhausted"] = self._time_exhausted
        summary["trials_exhausted"] = self._trials_exhausted
        return summary
