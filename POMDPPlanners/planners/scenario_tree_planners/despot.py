# SPDX-License-Identifier: MIT

"""DESPOT: Determinized Sparse Partially Observable Tree search.

DESPOT samples ``K`` *scenarios* from the current belief -- a start state plus
a predetermined random number for every depth -- and searches the belief tree
those scenarios induce. Every node is represented by the subset of scenarios
that reach it, and carries a lower and an upper bound on its value rather than
a running average of sampled returns. Search proceeds in *trials*: each trial
walks from the root down the branch that currently looks best under the upper
bound and least resolved under the gap between the bounds, expands the leaf it
reaches, and then backs the bounds up along the path it came down. The search
stops when the root's bounds have closed to within ``(1 - eta)`` of their
current width, or when the budget runs out.

Why that is not MCTS: there is no exploration bonus, no visit-count-driven
selection, and no averaging of returns. Branching is driven entirely by
branch-and-bound on an interval that is guaranteed to contain the node's value,
so a branch is abandoned when its interval can no longer matter, not when it
has been sampled enough times.

Equations, in the per-node conditional (weight-normalized) form the reference
Julia implementation uses. For a belief node ``b`` at depth ``d`` holding
scenario set ``Phi_b`` with weight ``w_b = |Phi_b| / K``:

1. First-step reward, ``rho(b,a) = sum_phi w_phi r(s_phi, a) / w_b``.
2. Action bounds, ``L(b,a) = sum_o (w_o / w_b) l(b_o)`` and ``U(b,a)``
   likewise over the upper bounds of the observation children.
3. Action value intervals, ``Q_l(b,a) = rho(b,a) + gamma L(b,a)`` and
   ``Q_u(b,a) = rho(b,a) + gamma U(b,a)``.
4. Backup, ``l(b) <- max(l(b), Q_l(b, a_u))`` where ``a_u = argmax_a Q_u(b,a)``,
   and ``u(b) <- max_a Q_u(b,a)``. The lower bound only ever rises, because a
   policy that achieved it remains available; the upper bound is recomputed
   from scratch over every action, because the previous maximiser's upper bound
   can fall below another action's.
5. Excess uncertainty, ``E(b) = (u(b) - l(b)) - eta (u(root) - l(root))
   gamma^(-d)``, and the weighted form ``WEU(b_o) = (w_o / w_b) E(b_o)`` that
   picks which observation branch a trial follows. A branch with
   ``WEU <= 0`` is already resolved to the precision the root needs, so the
   trial stops there. The ``gamma^(-d)`` factor is why deep nodes are allowed
   to stay looser: their contribution to the root is discounted anyway.
6. Leaf bounds. The lower bound is the value of a default policy run on the
   node's own scenarios; the upper bound is ``R_max`` sustained for the
   remaining horizon. Both respect the same finite horizon ``depth``, so
   ``l(b) <= V_depth(b) <= u(b)`` holds exactly rather than approximately.
7. Final action, ``argmax_a Q_l(b0, a)``: the action whose *guaranteed* value
   is highest, not the one whose optimistic value is highest. Under
   regularization (``pruning_constant > 0``) it is instead the maximiser of the
   regularized utility in (8).
8. Regularized utility (the NIPS 2013 contribution), with ``lambda`` the
   pruning constant and values in the unnormalized ``gamma^d w_b`` scale::

       nu(b) = max{ gamma^d w_b l_0(b) - lambda,
                    max_a [ gamma^d w_b rho(b,a) - lambda + sum_o nu(b_o) ] }

   One ``lambda`` is charged per policy node, so a policy that keeps branching
   must earn its size back. At ``lambda = 0`` this collapses to the plain
   lower-bound rule in (7).

Determinized scenarios. :meth:`Environment.sample_next_step` in this repository
takes no random generator, so the planner pins each transition by seeding the
global ``numpy.random`` and ``random`` generators from a fixed
``(scenario, depth)`` table (see
:class:`~POMDPPlanners.planners.planners_utils.scenario_streams.ScenarioRandomStreams`)
and restores the caller's generator state before returning. Environments whose
generative model runs in a native kernel with its own generator are not pinned
by this; for those, set ``use_determinized_scenarios=False`` so the search is
honestly a sparse-sampling tree rather than one that claims a determinization
it does not have.

References:
    Somani, A., Ye, N., Hsu, D., & Lee, W. S. (2013). DESPOT: Online POMDP
    Planning with Regularization. NeurIPS 26.
    Ye, N., Somani, A., Hsu, D., & Lee, W. S. (2017). DESPOT: Online POMDP
    Planning with Regularization. JAIR 58, 231-266.

    Reference implementation read for structure (not transliterated):
    https://github.com/JuliaPOMDP/DESPOT.jl -- ``src/solver.jl``,
    ``src/nodes.jl``, ``src/utils.jl``. That implementation leaves its
    regularization pass commented out; equation (8) here follows the paper.

Classes:
    DESPOT: The planner.
    DESPOTMetrics: Names of the search diagnostics it reports.
"""

# pylint: disable=too-many-lines
# The length is documentation, not code: the paper's eight equations, the units
# and reset semantics of every reported metric, and the two places this
# implementation knowingly departs from the reference are all recorded here
# rather than in a separate note that would drift away from the code.

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief, UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (
    ArenaPathSimulationPolicy,
)
from POMDPPlanners.planners.planners_utils.scenario_streams import ScenarioRandomStreams
from POMDPPlanners.utils.tree_statistics import TreeMetrics, compute_arena_tree_metrics


#: Absolute slack used when comparing two bounds. Bounds are sums of at most
#: ``depth`` discounted rewards, so anything below this is float noise rather
#: than a real ordering.
TINY: float = 1e-6


class _TerminalObservation:
    """Key for the branch holding scenarios that have already terminated.

    Terminated scenarios earn nothing further and must not be re-simulated, but
    they still carry weight that belongs in the parent's normalization. Giving
    them their own child -- with both bounds pinned to zero -- keeps
    ``sum_o w_o == w_b`` exactly, so the weighted averages in equation (2) stay
    correct instead of quietly renormalizing over the survivors.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<terminal>"


TERMINAL_OBSERVATION = _TerminalObservation()


class DESPOTMetrics(Enum):
    """Search diagnostics reported through ``PolicyRunData.info_variables``.

    Every value is measured over the single decision that produced it and is
    reset when the next :meth:`DESPOT.action` call builds a new tree; nothing
    accumulates across decisions.
    """

    #: Trials completed in this decision. Unit: count. DESPOT's tree grows by
    #: at most one path per trial, so this is the honest analogue of an MCTS
    #: simulation count.
    N_TRIALS = "despot_n_trials"
    #: ``l(root)`` after the search. Unit: discounted reward over ``depth``
    #: steps. This is the value DESPOT is willing to guarantee.
    ROOT_LOWER_BOUND = "despot_root_lower_bound"
    #: ``u(root)`` after the search, same unit.
    ROOT_UPPER_BOUND = "despot_root_upper_bound"
    #: ``u(root) - l(root)``. Unit: same. How much the search did not resolve.
    ROOT_GAP = "despot_root_gap"
    #: Scenarios ``K`` drawn for this decision. Unit: count.
    N_SCENARIOS = "despot_n_scenarios"
    #: Belief nodes allocated, including unexpanded frontier nodes. Unit: count.
    N_BELIEF_NODES = "despot_n_belief_nodes"
    #: Belief nodes a trial actually entered and backed up -- the paper's
    #: ``|D|``. Unit: count. Strictly smaller than ``N_BELIEF_NODES``, which
    #: also counts the fringe created by the last expansion.
    N_TREE_BELIEF_NODES = "despot_n_tree_belief_nodes"
    #: Deepest belief depth a trial reached. Unit: edges of one belief-action
    #: pair, so it is directly comparable to ``depth`` and *not* to the arena
    #: ``tree_max_depth`` metric, which counts arena edges and is therefore
    #: about twice as large.
    MAX_TRIAL_DEPTH = "despot_max_trial_depth"
    #: 1 if the loop ended because ``(1 - eta)(u - l) <= TINY`` at the root,
    #: 0 if it ended on the budget or on a stalled trial. Unit: flag.
    GAP_CLOSED = "despot_gap_closed"
    #: 1 if the loop ended because a trial could neither expand nor descend.
    #: Unit: flag. See :meth:`DESPOT._simulate_path`.
    STALLED = "despot_stalled"
    #: Times a backup found ``u < l`` and raised ``u`` to ``l``. Unit: count.
    #: Nonzero means the supplied ``max_reward`` or the default policy is
    #: inconsistent; it is reported rather than silently repaired.
    BOUND_CLAMPS = "despot_bound_clamps"
    #: 1 if the regularized pass preferred the default policy at the root over
    #: every action subtree, so the returned action is the lower-bound fallback
    #: rather than a regularized maximiser. Unit: flag. Emitted only when
    #: ``pruning_constant > 0``; its absence means regularization was off, not
    #: that the value was zero.
    REGULARIZED_FELL_BACK = "despot_regularized_fell_back"


@dataclass
class _BeliefNodeData:
    """Per-belief-node bookkeeping the arena ``Tree`` columns do not cover."""

    #: Belief depth in the planner's own units (one belief-action pair each),
    #: not arena edge depth.
    depth: int
    #: ``l_0(b)``: the default policy's value on this node's scenarios. Kept
    #: after ``l(b)`` starts rising because equation (8) needs the original.
    default_value: float
    #: Whether the node's action children have been created.
    expanded: bool = False
    #: Whether every scenario at this node is in a terminal state.
    terminal: bool = False
    #: Whether a trial entered and backed up this node -- membership in ``D``.
    in_tree: bool = False
    #: Action child currently maximising ``Q_u``; the branch a trial follows.
    best_upper_action_id: Optional[int] = None
    #: Scenario ids carried here, aligned with the belief's particle list.
    scenario_ids: List[int] = field(default_factory=list)


class DESPOT(ArenaPathSimulationPolicy):
    """Determinized Sparse Partially Observable Tree search.

    See the module docstring for the algorithm and its equations.

    Args:
        environment: Discrete-action POMDP to plan in.
        discount_factor: ``gamma``, in ``(0, 1]``.
        depth: Search horizon ``D`` in belief steps. Both bounds respect it, so
            the planner optimises the ``D``-step discounted return.
        name: Unique identifier for this planner instance.
        n_scenarios: ``K``, the number of determinized scenarios per decision.
        eta: Target-precision parameter in ``(0, 1)``. The search stops once the
            root gap has shrunk to ``eta`` of its own width. ``eta = 1`` is
            rejected: it makes the stopping test true before the first trial,
            so the planner would return without searching.
        pruning_constant: ``lambda`` in equation (8). ``0`` disables
            regularization and reproduces the reference implementation.
        max_reward: ``R_max`` for the trivial upper bound. Defaults to the
            environment's declared ``reward_range`` maximum.
        min_reward: ``R_min``, charged for the horizon a shortened rollout does
            not cover. Only consulted when ``rollout_depth < depth``; defaults
            to the environment's declared ``reward_range`` minimum.
        rollout_depth: Steps of default policy used for the leaf lower bound.
            Defaults to ``depth``. Lower it to buy trials at the cost of a
            looser lower bound; it does not change what the planner optimises,
            only how well it is bounded.
        scenario_seed: Base seed for the determinized random-number table.
        use_determinized_scenarios: Whether to pin transitions to
            ``(scenario, depth)``. Set ``False`` for environments whose
            generative model has its own generator the planner cannot reach.
        time_out_in_seconds: Wall-clock budget per decision. Mutually exclusive
            with ``n_simulations``.
        n_simulations: Maximum trials per decision. Mutually exclusive with
            ``time_out_in_seconds``.

    Raises:
        ValueError: If the budget arguments are not exactly one of the two, if
            a numeric parameter is outside its stated range, or if no
            ``max_reward`` is available.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(0)
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = DESPOT(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     depth=5,
        ...     name="ExampleDESPOT",
        ...     n_scenarios=8,
        ...     n_simulations=20,
        ... )
        >>> belief = get_initial_belief(tiger, n_particles=20)
        >>> actions, run_data = planner.action(belief)
        >>> actions[0] in tiger.get_actions()
        True
        >>> DESPOT.get_space_info().action_space.name
        'DISCRETE'
    """

    # pylint: disable=too-many-instance-attributes
    # The paper's configuration is genuinely this wide; collapsing it into a
    # dict would take the parameters out of ``config_id`` and out of tuning.

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        name: str,
        n_scenarios: int = 50,
        eta: float = 0.95,
        pruning_constant: float = 0.0,
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
        exactly_one_budget = (time_out_in_seconds is None) != (n_simulations is None)
        if not exactly_one_budget:
            raise ValueError("Only one of time_out_in_seconds and n_simulations must be provided.")
        self._validate_search_params(
            depth=depth,
            n_scenarios=n_scenarios,
            eta=eta,
            pruning_constant=pruning_constant,
            rollout_depth=rollout_depth,
        )

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
        self.n_scenarios = n_scenarios
        self.eta = eta
        self.pruning_constant = pruning_constant
        self.rollout_depth = depth if rollout_depth is None else rollout_depth
        self.scenario_seed = scenario_seed
        self.use_determinized_scenarios = use_determinized_scenarios
        self.max_reward = self._resolve_reward_bound(
            supplied=max_reward, index=1, label="max_reward"
        )
        # Only needed when the default-policy rollout stops short of the search
        # horizon; resolved lazily so an environment without a declared
        # reward_range stays usable at the default ``rollout_depth == depth``.
        self.min_reward = (
            self._resolve_reward_bound(supplied=min_reward, index=0, label="min_reward")
            if self.rollout_depth < depth or min_reward is not None
            else 0.0
        )

        # --- transient per-call search state ---
        # Everything below is underscore-prefixed so it stays out of
        # ``config_id``: these are one search's numbers, and an integer node id
        # means nothing outside the tree that produced it.
        self._scenario_rng = np.random.default_rng(scenario_seed)
        self._streams: Optional[ScenarioRandomStreams] = None
        self._call_index: int = 0
        self._root_id: int = 0
        self._n_trials: int = 0
        self._max_trial_depth: int = 0
        self._gap_closed: bool = False
        self._stalled: bool = False
        self._bound_clamps: int = 0
        self._regularized_fell_back: bool = False
        self._last_tree: Optional[Tree] = None
        self._last_root_id: Optional[int] = None
        self._search_state_dump_dir: Optional[Path] = None
        self._search_state_dump_index: int = 0

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_search_params(
        depth: int,
        n_scenarios: int,
        eta: float,
        pruning_constant: float,
        rollout_depth: Optional[int],
    ) -> None:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise TypeError(f"depth must be an int, got {type(depth).__name__}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if not isinstance(n_scenarios, int) or isinstance(n_scenarios, bool):
            raise TypeError(f"n_scenarios must be an int, got {type(n_scenarios).__name__}")
        if n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be positive, got {n_scenarios}")
        if not isinstance(eta, (int, float)):
            raise TypeError(f"eta must be a number, got {type(eta).__name__}")
        if not 0.0 < eta < 1.0:
            # eta == 1 makes (1 - eta) * gap == 0 <= TINY before the first
            # trial, so the planner would answer from the initial bounds alone.
            raise ValueError(f"eta must be strictly between 0 and 1, got {eta}")
        if not isinstance(pruning_constant, (int, float)):
            raise TypeError(
                f"pruning_constant must be a number, got {type(pruning_constant).__name__}"
            )
        if pruning_constant < 0:
            raise ValueError(f"pruning_constant must be non-negative, got {pruning_constant}")
        if rollout_depth is not None:
            if not isinstance(rollout_depth, int) or isinstance(rollout_depth, bool):
                raise TypeError(f"rollout_depth must be an int, got {type(rollout_depth).__name__}")
            if rollout_depth < 0:
                raise ValueError(f"rollout_depth must be non-negative, got {rollout_depth}")

    def _resolve_reward_bound(self, supplied: Optional[float], index: int, label: str) -> float:
        """Settle a per-step reward bound, or refuse to guess.

        A bound that is not actually a bound turns the branch-and-bound into an
        arbitrary heuristic and silently voids the paper's guarantee, so an
        unusable ``reward_range`` is an error rather than a default.
        """
        if supplied is not None:
            value = float(supplied)
        else:
            reward_range = getattr(self.environment, "reward_range", None)
            if reward_range is None:
                raise ValueError(
                    f"DESPOT needs {label}, a per-step reward bound. Environment "
                    f"{self.environment.name} declares no reward_range, so pass "
                    f"{label} explicitly."
                )
            value = float(reward_range[index])
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value}")
        return value

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """DESPOT branches on observation equality, so both spaces are discrete.

        A continuous observation space would give almost every scenario its own
        branch, which is not an error but makes the tree degenerate; declaring
        DISCRETE lets the compatibility check say so up front.
        """
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return [metric.value for metric in TreeMetrics] + [metric.value for metric in DESPOTMetrics]

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
        info_variables.extend(self._despot_metrics(tree=tree, root_id=root_id))

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
        self._regularized_fell_back = False

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

        if self.n_simulations is not None:
            self._construct_tree_using_n_simulations(tree=tree, root_id=root_id)
        else:
            self._construct_tree_using_timeout(tree=tree, root_id=root_id)

        self._last_tree_size = len(tree)
        return tree, root_id

    def _sample_scenario_states(self, belief: Belief) -> List[Any]:
        """Draw ``K`` start states, one per scenario.

        Systematic (low-variance) resampling, matching the reference: it hits
        every region of the belief whose weight exceeds ``1/K`` at least once,
        whereas independent draws can miss a well-supported state entirely and
        make a whole decision blind to it.

        The draw uses the planner's private generator rather than the global
        one, because :meth:`action` restores the global state on the way out --
        drawing from it would make two consecutive decisions from an unchanged
        belief see identical scenarios.
        """
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            # Gaussian and mixture beliefs have no particle list; fall back to
            # their own sampler.
            return self._sample_from_belief_sampler(belief=belief)

        weights = self._particle_weights(belief=belief, n_particles=len(particles))
        cumulative = np.cumsum(weights)
        offset = float(self._scenario_rng.random()) / self.n_scenarios
        positions = offset + np.arange(self.n_scenarios, dtype=np.float64) / self.n_scenarios
        indices = np.clip(np.searchsorted(cumulative, positions), 0, len(particles) - 1)
        return [particles[int(index)] for index in indices]

    @staticmethod
    def _particle_weights(belief: Belief, n_particles: int) -> np.ndarray:
        """Normalized weight per particle, uniform when the belief carries none.

        The belief classes disagree on the attribute: ``WeightedParticleBelief``
        precomputes ``normalized_weights``, ``WeightedParticleBeliefStateUpdate``
        keeps raw ``weights`` and normalizes only inside its own ``sample``, and
        the unweighted particle beliefs expose neither. Reading only
        ``normalized_weights`` pushed the latter two onto the ``belief.sample()``
        fallback, which draws from the stdlib ``random`` stream that
        :meth:`action` then rewinds -- so every decision from an unchanged
        belief saw the identical scenario set. Resolving the attribute here puts
        every particle belief back on the systematic-resampling path.

        A belief whose weights are missing, mis-shaped, non-finite or sum to
        zero is treated as uniform rather than refused: systematic resampling of
        a degenerate weight vector is undefined, and a uniform draw over the
        particles the belief does hold is still a usable scenario set.
        """
        uniform = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
        raw = getattr(belief, "normalized_weights", None)
        if raw is None:
            raw = getattr(belief, "weights", None)
        if raw is None:
            return uniform

        array = np.asarray(raw, dtype=np.float64).reshape(-1)
        if array.shape[0] != n_particles or not np.all(np.isfinite(array)):
            return uniform
        total = float(array.sum())
        if total <= 0.0 or np.any(array < 0.0):
            return uniform
        return array / total

    def _sample_from_belief_sampler(self, belief: Belief) -> List[Any]:
        """``K`` draws from a belief that has no particle list of its own.

        The draw is seeded from the planner's private generator, so it is fresh
        on every call yet reproducible from ``scenario_seed``. Both global
        streams are seeded because the planner cannot tell which one an
        arbitrary belief's ``sample`` reaches for, and both are put back
        afterwards: seeding was previously unconditional while :meth:`action`
        only restored global state when determinization was on, so a planner
        with ``use_determinized_scenarios=False`` silently overwrote the
        simulator's numpy stream on every decision.
        """
        saved_state = ScenarioRandomStreams.capture_global_state()
        try:
            seed = int(self._scenario_rng.integers(0, 2**32))
            np.random.seed(seed)
            random.seed(seed)
            return [belief.sample() for _ in range(self.n_scenarios)]
        finally:
            ScenarioRandomStreams.restore_global_state(saved_state)

    def _construct_tree_using_n_simulations(self, tree: Tree, root_id: int) -> None:
        if self.n_simulations is None:
            raise ValueError("n_simulations must not be None")
        for _ in range(self.n_simulations):
            if self._should_stop(tree=tree, root_id=root_id):
                break
            self._simulate_path(tree=tree, belief_id=root_id, depth=0)

    def _construct_tree_using_timeout(self, tree: Tree, root_id: int) -> None:
        if self.time_out_in_seconds is None:
            raise ValueError("time_out_in_seconds must not be None")
        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:
            if self._should_stop(tree=tree, root_id=root_id):
                break
            self._simulate_path(tree=tree, belief_id=root_id, depth=0)

    def _should_stop(self, tree: Tree, root_id: int) -> bool:
        """Whether another trial can still change anything.

        Two conditions. The first is the paper's: the root's excess uncertainty
        ``(1 - eta)(u - l)`` has fallen to numerical noise, so the answer is as
        resolved as ``eta`` asks for. The second is a guard the reference does
        not have -- a trial that can neither expand a node nor descend past the
        root repeats identically forever, and under a wall-clock budget that
        burns the whole decision doing nothing while looking busy. It is
        reported through ``despot_stalled`` rather than hidden.
        """
        if self._stalled:
            return True
        gap = tree.upper_confidence_bound[root_id] - tree.lower_confidence_bound[root_id]
        if (1.0 - self.eta) * gap <= TINY:
            self._gap_closed = True
            return True
        return False

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> Optional[float]:
        """Run one DESPOT trial: descend, expand the leaf, back the bounds up.

        Iterative rather than recursive so a long horizon cannot exhaust the
        Python stack, and so the backup order (deepest first) is explicit.

        Always returns ``None``. The base class's hook is typed
        ``Optional[float]`` because an MCTS simulation hands a sampled return
        back to its caller; a DESPOT trial leaves its result in the nodes'
        bounds instead, so there is nothing return-shaped to give.
        """
        del depth  # a trial always starts at the root, at belief depth 0.
        path: List[int] = []
        node_id = belief_id
        expanded_anything = False

        while True:
            data: _BeliefNodeData = tree.data[node_id]
            if data.terminal or data.depth >= self.depth:
                break
            if not data.expanded:
                self._expand(tree=tree, belief_id=node_id)
                expanded_anything = True

            path.append(node_id)
            tree.increment_visit_count(node_id)
            self._max_trial_depth = max(self._max_trial_depth, data.depth)

            action_id = data.best_upper_action_id
            if action_id is None:
                break
            tree.increment_visit_count(action_id)

            child_id, weighted_excess = self._best_excess_uncertainty_child(
                tree=tree, action_id=action_id
            )
            if child_id is None or weighted_excess <= 0.0:
                break
            node_id = child_id

        for visited_id in reversed(path):
            self._backup(tree=tree, belief_id=visited_id)
            tree.data[visited_id].in_tree = True

        self._n_trials += 1
        if not expanded_anything and len(path) <= 1:
            self._stalled = True

    def _expand(self, tree: Tree, belief_id: int) -> None:
        """Create one action child per action, and its observation children.

        Every scenario at this node is pushed through the generative model once
        per action, under the random number fixed for its ``(scenario, depth)``
        pair, and the resulting successors are grouped by observation. Grouping
        is what makes the tree sparse: ``K`` scenarios produce at most ``K``
        branches and usually far fewer.
        """
        data: _BeliefNodeData = tree.data[belief_id]
        node_depth = data.depth
        node_weight = tree.weight[belief_id]
        states = self._node_states(tree=tree, belief_id=belief_id)
        scenario_ids = list(data.scenario_ids)
        scenario_weight = 1.0 / self.n_scenarios

        best_upper = -np.inf
        best_upper_action_id: Optional[int] = None

        for action in self.environment.get_actions():  # type: ignore[attr-defined]
            groups: Dict[Any, Tuple[Any, List[Any], List[int]]] = {}
            weighted_reward = 0.0

            for scenario_id, state in zip(scenario_ids, states):
                if self.environment.is_terminal(state=state):
                    # Already absorbed: no further reward, and no re-simulation.
                    # The terminal reward was charged on the transition that
                    # produced this state, so charging anything here would
                    # count it twice.
                    key: Any = TERMINAL_OBSERVATION
                    observation: Any = TERMINAL_OBSERVATION
                    next_state = state
                    reward = 0.0
                else:
                    if self.use_determinized_scenarios and self._streams is not None:
                        self._streams.activate(scenario_id=scenario_id, depth=node_depth)
                    next_state, observation, reward = self.environment.sample_next_step(
                        state=state, action=action
                    )
                    key = self._observation_key(observation)

                weighted_reward += scenario_weight * reward
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
            tree.set_immediate_reward(action_id, weighted_reward / node_weight)

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

            self._update_action_bounds(tree=tree, action_id=action_id)
            upper = tree.upper_confidence_bound[action_id]
            if upper > best_upper + TINY:
                best_upper = upper
                best_upper_action_id = action_id

        data.expanded = True
        data.best_upper_action_id = best_upper_action_id

    @staticmethod
    def _node_states(tree: Tree, belief_id: int) -> List[Any]:
        """The scenario states at a belief node, aligned with ``tree.sample``.

        Every belief node this planner allocates holds an
        ``UnweightedParticleBeliefStateUpdate``, whose ``particles`` list is the
        scenario states in scenario order. The base ``Belief`` interface does not
        promise a particle list, so the access is localized here rather than
        repeated with a type suppression at each use.
        """
        return list(tree.get_belief(belief_id).particles)  # type: ignore[attr-defined]

    def _observation_key(self, observation: Any) -> Any:
        """Hashable key an observation branch is indexed by.

        DESPOT merges scenarios that produced the same observation; without a
        hashable key there is no merge and the tree stops being sparse, so an
        unhashable observation is refused with a message that says which
        environment method to fix rather than silently degrading.
        """
        key = self.environment.hash_observation(observation)
        try:
            hash(key)
        except TypeError as error:
            raise TypeError(
                f"DESPOT groups scenarios by observation, which needs a hashable key. "
                f"{self.environment.name}.hash_observation returned {type(key).__name__}. "
                f"Override hash_observation to return a hashable key."
            ) from error
        return key

    # ------------------------------------------------------------------
    # nodes and bounds
    # ------------------------------------------------------------------

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
        """Allocate a belief node and give it its initial bounds."""
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
        lower, upper = self._initial_bounds(
            states=states, scenario_ids=scenario_ids, depth=depth, terminal=terminal
        )
        tree.lower_confidence_bound[node_id] = lower
        tree.upper_confidence_bound[node_id] = upper
        tree.v_value[node_id] = lower
        tree.data[node_id] = _BeliefNodeData(
            depth=depth,
            default_value=lower,
            terminal=terminal,
            scenario_ids=list(scenario_ids),
        )
        return node_id

    def _initial_bounds(
        self, states: List[Any], scenario_ids: List[int], depth: int, terminal: bool
    ) -> Tuple[float, float]:
        """``(l_0(b), u_0(b))`` for a fresh node.

        Both respect the same horizon ``depth .. self.depth``, so the interval
        brackets the ``D``-step value the planner is actually optimising. A
        node at or past the horizon, and a node whose scenarios have all
        terminated, are worth exactly zero and get a zero-width interval, which
        is what stops a trial from descending into them.
        """
        if terminal or depth >= self.depth:
            return 0.0, 0.0

        upper_total = 0.0
        remaining = self.depth - depth
        sustained = self._discount_sum(remaining)
        for state in states:
            if self.environment.is_terminal(state=state):
                continue
            upper_total += self.max_reward * sustained
        upper = upper_total / len(states)

        policy_actions = self._default_policy_actions(
            reference_scenario_id=scenario_ids[0], depth=depth
        )
        lower_total = 0.0
        for scenario_id, state in zip(scenario_ids, states):
            lower_total += self._default_policy_value(
                scenario_id=scenario_id,
                state=state,
                depth=depth,
                policy_actions=policy_actions,
            )
        lower = lower_total / len(states)

        return lower, self._reconcile_bounds(lower=lower, upper=upper)

    def _reconcile_bounds(self, lower: float, upper: float) -> float:
        """Return an upper bound that is not below ``lower``.

        Branch-and-bound needs ``l <= u`` at every node, and the two are
        computed by different arithmetic -- a closed-form geometric sum against
        an accumulated rollout -- so they can disagree in the last bit even
        when they are mathematically equal. Those crossings are repaired
        silently. A crossing wider than ``TINY`` is a real inconsistency in the
        supplied ``max_reward`` or the default policy and is counted, so
        ``despot_bound_clamps`` says something rather than tracking float noise.
        """
        if lower <= upper:
            return upper
        if lower > upper + TINY:
            self._bound_clamps += 1
        return lower

    def _discount_sum(self, n_terms: int) -> float:
        """``sum_{t<n} gamma^t``, written so ``gamma == 1`` is not a special case."""
        if n_terms <= 0:
            return 0.0
        gamma = self.discount_factor
        if gamma == 1.0:
            return float(n_terms)
        return float((1.0 - gamma**n_terms) / (1.0 - gamma))

    def _default_policy_actions(self, reference_scenario_id: int, depth: int) -> List[Any]:
        """The default policy's action at each depth, for one belief node.

        One sequence for the whole node, not one per scenario. That is not a
        detail: a "policy" that reads the scenario is clairvoyant, its value is
        not achievable by anything the planner could execute, and using it as
        ``l_0`` would put a number above the node's true value into the lower
        bound -- which is exactly the ``u < l`` inconsistency the clamp counter
        was catching before this was fixed. Keying the sequence on the node's
        first scenario id keeps it varied across nodes while staying constant
        across the scenarios inside one node.

        The sequence is read out of the seed table rather than drawn, so it
        consumes no randomness and leaves each scenario's stream to be spent
        only on its transition. A scenario's transition at a given depth is
        then the same number in this rollout as it is in the tree.
        """
        actions = self.environment.get_actions()  # type: ignore[attr-defined]
        streams = self._streams
        horizon = min(self.depth, depth + self.rollout_depth)
        if not self.use_determinized_scenarios or streams is None:
            return [random.choice(actions) for _ in range(depth, horizon)]
        return [
            actions[streams.seed_for(scenario_id=reference_scenario_id, depth=step) % len(actions)]
            for step in range(depth, horizon)
        ]

    def _default_policy_value(
        self, scenario_id: int, state: Any, depth: int, policy_actions: List[Any]
    ) -> float:
        """Value of ``policy_actions`` on one scenario, from ``depth`` onward.

        When the rollout stops before the search horizon (``rollout_depth`` is
        shorter than ``depth``), the unrolled remainder is charged at
        ``min_reward`` rather than at zero. Charging zero would be an
        over-estimate whenever future rewards can be negative, and an
        over-estimating "lower bound" silently voids the branch-and-bound.
        """
        total = 0.0
        discount = 1.0
        current_state = state
        current_depth = depth
        streams = self._streams
        terminated = False

        for action in policy_actions:
            if self.environment.is_terminal(state=current_state):
                terminated = True
                break
            if self.use_determinized_scenarios and streams is not None:
                streams.activate(scenario_id=scenario_id, depth=current_depth)
            next_state, _, reward = self.environment.sample_next_step(
                state=current_state, action=action
            )
            total += discount * reward
            discount *= self.discount_factor
            current_state = next_state
            current_depth += 1

        if not terminated and not self.environment.is_terminal(state=current_state):
            remaining = self.depth - current_depth
            if remaining > 0:
                total += discount * self.min_reward * self._discount_sum(remaining)
        return total

    def _update_action_bounds(self, tree: Tree, action_id: int) -> None:
        """Recompute ``Q_l`` and ``Q_u`` for one action from its children.

        The weights normalize by the *parent belief's* weight, not by a sum
        recomputed over the children: every scenario at the parent lands in
        exactly one child (terminated ones included, in their own branch), so
        the two are equal by construction and using the parent's value keeps a
        dropped scenario from silently renormalizing the average.
        """
        parent_id = tree.get_parent_id(action_id)
        if parent_id is None:
            raise ValueError(f"action node {action_id} has no parent belief node")
        parent_weight = tree.weight[parent_id]

        lower_sum = 0.0
        upper_sum = 0.0
        for child_id in tree.get_children_ids(action_id):
            child_weight = tree.weight[child_id]
            lower_sum += child_weight * tree.lower_confidence_bound[child_id]
            upper_sum += child_weight * tree.upper_confidence_bound[child_id]

        immediate = tree.get_immediate_reward(action_id) or 0.0
        gamma = self.discount_factor
        lower = immediate + gamma * lower_sum / parent_weight
        upper = immediate + gamma * upper_sum / parent_weight

        tree.lower_confidence_bound[action_id] = lower
        tree.upper_confidence_bound[action_id] = upper
        # ``q_value`` carries the lower bound because that is the quantity the
        # final decision is made on; keeping the two in step means the arena's
        # generic ``best_action_by_reward`` and DESPOT's own rule agree.
        tree.q_value[action_id] = lower

    def _backup(self, tree: Tree, belief_id: int) -> None:
        """Equation (4): raise the lower bound, recompute the upper bound."""
        children = tree.get_children_ids(belief_id)
        if not children:
            return

        best_upper = -np.inf
        best_upper_action_id: Optional[int] = None
        for action_id in children:
            self._update_action_bounds(tree=tree, action_id=action_id)
            upper = tree.upper_confidence_bound[action_id]
            if upper > best_upper + TINY:
                best_upper = upper
                best_upper_action_id = action_id

        if best_upper_action_id is None:
            return

        candidate_lower = tree.lower_confidence_bound[best_upper_action_id]
        lower = max(tree.lower_confidence_bound[belief_id], candidate_lower)
        upper = self._reconcile_bounds(lower=lower, upper=float(best_upper))

        tree.lower_confidence_bound[belief_id] = lower
        tree.upper_confidence_bound[belief_id] = upper
        tree.v_value[belief_id] = lower
        tree.data[belief_id].best_upper_action_id = best_upper_action_id

    def _best_excess_uncertainty_child(
        self, tree: Tree, action_id: int
    ) -> Tuple[Optional[int], float]:
        """Equation (5): pick the observation branch worth resolving next."""
        children = tree.get_children_ids(action_id)
        if not children:
            return None, -np.inf

        parent_id = tree.get_parent_id(action_id)
        parent_weight = tree.weight[parent_id] if parent_id is not None else 1.0
        root_gap = (
            tree.upper_confidence_bound[self._root_id] - tree.lower_confidence_bound[self._root_id]
        )
        gamma = self.discount_factor

        best_id: Optional[int] = None
        best_value = -np.inf
        for child_id in children:
            child_depth = tree.data[child_id].depth
            width = tree.upper_confidence_bound[child_id] - tree.lower_confidence_bound[child_id]
            target = self.eta * root_gap * gamma ** (-child_depth)
            weighted = (tree.weight[child_id] / parent_weight) * (width - target)
            if weighted > best_value:
                best_value = weighted
                best_id = child_id
        return best_id, float(best_value)

    # ------------------------------------------------------------------
    # final decision
    # ------------------------------------------------------------------

    def _select_final_action(self, tree: Tree, root_id: int) -> Any:
        """Equation (7), or (8) when regularization is on.

        Deliberately not the search rule. The trial descends on the *upper*
        bound because that is where an improvement might still be found; the
        answer is taken on the *lower* bound because that is the value DESPOT
        can actually stand behind. Returning the optimistic action would return
        whichever branch the search understood least.
        """
        if self.pruning_constant > 0.0:
            regularized_action = self._regularized_action(tree=tree, root_id=root_id)
            if regularized_action is not None:
                return regularized_action
            # Every action subtree lost to "just run the default policy". The
            # paper's answer is the default policy's action; this repository
            # has no default-policy action to return, so the lower-bound action
            # is used and the substitution is reported through
            # ``despot_regularized_fell_back`` instead of passing unnoticed.
            self._regularized_fell_back = True

        return self._action_of_best_lower_bound(tree=tree, root_id=root_id)

    def _action_of_best_lower_bound(self, tree: Tree, root_id: int) -> Any:
        best_id = max(
            tree.get_children_ids(root_id),
            key=lambda action_id: tree.lower_confidence_bound[action_id],
        )
        return tree.get_action(best_id)

    def _regularized_action(self, tree: Tree, root_id: int) -> Optional[Any]:
        """Equation (8): the action maximising the regularized utility.

        Returns ``None`` when the default policy beats every action subtree at
        the root -- the regularized optimum is then to stop planning, which is
        a real answer and not an error.
        """
        _, best_action_id = self._regularized_value(tree=tree, belief_id=root_id)
        if best_action_id is None:
            return None
        return tree.get_action(best_action_id)

    def _regularized_value(self, tree: Tree, belief_id: int) -> Tuple[float, Optional[int]]:
        """``nu(b)`` and the action child attaining it, in the unnormalized scale.

        Values here are multiplied by ``gamma^depth * weight`` so that a child's
        contribution can be added to its parent's without renormalizing, which
        is what lets one ``lambda`` per policy node be subtracted directly.
        """
        data: _BeliefNodeData = tree.data[belief_id]
        scale = (self.discount_factor**data.depth) * tree.weight[belief_id]
        lam = self.pruning_constant

        best_value = scale * data.default_value - lam
        best_action_id: Optional[int] = None

        for action_id in tree.get_children_ids(belief_id):
            immediate = tree.get_immediate_reward(action_id) or 0.0
            value = scale * immediate - lam
            for child_id in tree.get_children_ids(action_id):
                child_value, _ = self._regularized_value(tree=tree, belief_id=child_id)
                value += child_value
            if value > best_value + TINY:
                best_value = value
                best_action_id = action_id

        return best_value, best_action_id

    # ------------------------------------------------------------------
    # metrics and search-state export
    # ------------------------------------------------------------------

    def _despot_metrics(self, tree: Tree, root_id: int) -> List[PolicyInfoVariable]:
        lower = float(tree.lower_confidence_bound[root_id])
        upper = float(tree.upper_confidence_bound[root_id])
        n_belief_nodes = 0
        n_tree_belief_nodes = 0
        for node_id in range(len(tree)):
            if tree.kind[node_id] != BELIEF:
                continue
            n_belief_nodes += 1
            node_data = tree.data[node_id]
            if isinstance(node_data, _BeliefNodeData) and node_data.in_tree:
                n_tree_belief_nodes += 1

        metrics = [
            PolicyInfoVariable(name=DESPOTMetrics.N_TRIALS.value, value=self._n_trials),
            PolicyInfoVariable(name=DESPOTMetrics.ROOT_LOWER_BOUND.value, value=lower),
            PolicyInfoVariable(name=DESPOTMetrics.ROOT_UPPER_BOUND.value, value=upper),
            PolicyInfoVariable(name=DESPOTMetrics.ROOT_GAP.value, value=upper - lower),
            PolicyInfoVariable(name=DESPOTMetrics.N_SCENARIOS.value, value=self.n_scenarios),
            PolicyInfoVariable(name=DESPOTMetrics.N_BELIEF_NODES.value, value=n_belief_nodes),
            PolicyInfoVariable(
                name=DESPOTMetrics.N_TREE_BELIEF_NODES.value, value=n_tree_belief_nodes
            ),
            PolicyInfoVariable(
                name=DESPOTMetrics.MAX_TRIAL_DEPTH.value, value=self._max_trial_depth
            ),
            PolicyInfoVariable(name=DESPOTMetrics.GAP_CLOSED.value, value=int(self._gap_closed)),
            PolicyInfoVariable(name=DESPOTMetrics.STALLED.value, value=int(self._stalled)),
            PolicyInfoVariable(name=DESPOTMetrics.BOUND_CLAMPS.value, value=self._bound_clamps),
        ]
        if self.pruning_constant > 0.0:
            # Absent when regularization is off: a missing metric says "not
            # applicable", a zero would claim the fallback did not happen.
            metrics.append(
                PolicyInfoVariable(
                    name=DESPOTMetrics.REGULARIZED_FELL_BACK.value,
                    value=int(self._regularized_fell_back),
                )
            )
        return metrics

    def enable_search_state_dump(self, directory: Path) -> None:
        """Write a full node dump for every subsequent decision into ``directory``.

        Off by default and deliberately not a constructor argument: a full tree
        per decision is a QA artifact, not something a performance run should
        pay for, and keeping it off ``__init__`` keeps it out of ``config_id``
        so a dumping run and a normal run stay cache-compatible.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._search_state_dump_dir = directory
        self._search_state_dump_index = 0

    def disable_search_state_dump(self) -> None:
        self._search_state_dump_dir = None

    def _write_search_state_dump(self, tree: Tree, root_id: int) -> Path:
        path = (
            self._search_state_dump_dir
            / f"{self.name}_decision_{self._search_state_dump_index:04d}.json"
        )
        self._search_state_dump_index += 1
        return self.export_search_state(path=path, tree=tree, root_id=root_id)

    def export_search_state(
        self,
        path: Path,
        tree: Optional[Tree] = None,
        root_id: Optional[int] = None,
    ) -> Path:
        """Write the search tree of a decision to ``path`` as JSON.

        Defaults to the most recent decision. Each :meth:`action` call builds a
        fresh ``Tree``, so the last one survives until the next call and this
        can be invoked after the fact; call it before the next decision if the
        tree must be kept.

        Raises:
            ValueError: If no search has run yet.
        """
        if tree is None or root_id is None:
            tree, root_id = self._last_tree, self._last_root_id
        if tree is None or root_id is None:
            raise ValueError("no search state to export; call action() first")

        nodes = [self._node_record(tree=tree, node_id=node_id) for node_id in range(len(tree))]

        payload = {
            "planner": self.name,
            "planner_class": type(self).__name__,
            "config_id": self.config_id,
            "environment": self.environment.name,
            "config": self._export_config(),
            "search": self._export_search_summary(root_id=root_id),
            "nodes": nodes,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def _node_record(self, tree: Tree, node_id: int) -> Dict[str, Any]:
        """One node's dump entry.

        A hook rather than an inline loop so a subclass with extra per-node
        quantities -- AR-DESPOT carries ``mu`` and a blocked flag that plain
        DESPOT has no notion of -- can add them without copying the whole
        exporter, which would then drift from this one.
        """
        kind = tree.kind[node_id]
        record: Dict[str, Any] = {
            "id": node_id,
            "kind": "belief" if kind == BELIEF else "action",
            "parent_id": tree.parent_id[node_id],
            "children_ids": list(tree.children_ids[node_id]),
            "visit_count": tree.visit_count[node_id],
            "weight": tree.weight[node_id],
            "lower_bound": tree.lower_confidence_bound[node_id],
            "upper_bound": tree.upper_confidence_bound[node_id],
        }
        if kind == ACTION:
            record["action"] = repr(tree.action[node_id])
            record["immediate_reward"] = tree.immediate_reward[node_id]
            record["q_lower"] = tree.q_value[node_id]
        else:
            node_data = tree.data[node_id]
            record["observation"] = repr(tree.observation[node_id])
            record["v_value"] = tree.v_value[node_id]
            record["scenario_ids"] = list(tree.sample[node_id] or [])
            record["particles"] = [
                repr(state) for state in self._node_states(tree=tree, belief_id=node_id)
            ]
            if isinstance(node_data, _BeliefNodeData):
                record["depth"] = node_data.depth
                record["default_value"] = node_data.default_value
                record["expanded"] = node_data.expanded
                record["terminal"] = node_data.terminal
                record["in_tree"] = node_data.in_tree
                record["best_upper_action_id"] = node_data.best_upper_action_id
        return record

    def _export_config(self) -> Dict[str, Any]:
        """The configuration recorded in a dump. Hook, for the same reason."""
        return {
            "depth": self.depth,
            "discount_factor": self.discount_factor,
            "n_scenarios": self.n_scenarios,
            "eta": self.eta,
            "pruning_constant": self.pruning_constant,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "rollout_depth": self.rollout_depth,
            "scenario_seed": self.scenario_seed,
            "use_determinized_scenarios": self.use_determinized_scenarios,
            "n_simulations": self.n_simulations,
            "time_out_in_seconds": self.time_out_in_seconds,
        }

    def _export_search_summary(self, root_id: int) -> Dict[str, Any]:
        """What the search did, for a dump. Hook, for the same reason."""
        return {
            "root_id": root_id,
            "n_trials": self._n_trials,
            "gap_closed": self._gap_closed,
            "stalled": self._stalled,
            "bound_clamps": self._bound_clamps,
            "max_trial_depth": self._max_trial_depth,
        }
