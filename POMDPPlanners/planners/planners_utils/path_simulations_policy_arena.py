"""Arena-tree variants of the MCTS path-simulation base classes.

Mirrors :class:`POMDPPlanners.planners.planners_utils.path_simulations_policy.PathSimulationPolicy`
and :class:`DoubleProgressiveWideningMCTSPolicy` but operates on the
column-store :class:`POMDPPlanners.core.tree.arena.Tree` (integer node
IDs) instead of the anytree-based :class:`BeliefNode`/:class:`ActionNode`
objects.

Subclasses implement ``_simulate_path(tree, belief_id, depth)`` against
the arena Tree API. The base class handles the simulation loop, root
construction, action selection, and tree-metrics collection.
"""

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import (
    Belief,
    GaussianBelief,
    GaussianMixtureBelief,
    VectorizedWeightedParticleBelief,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
    is_terminal_belief,
)
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.tree import (
    BeliefNode,
)  # only used for the pre-tree action_sampler fallback wrapper
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.tree_statistics import TreeMetrics, compute_arena_tree_metrics


class ArenaPathSimulationPolicy(Policy):
    """MCTS base class operating on the arena :class:`Tree` + integer node IDs.

    Mirrors :class:`PathSimulationPolicy`'s shape and validation but uses
    ``(tree, belief_id)`` instead of node objects. Subclasses implement
    :meth:`_simulate_path` against the arena Tree API.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        environment: Environment,
        discount_factor: float,
        name: str,
        n_simulations: Optional[int],
        time_out_in_seconds: Optional[int],
        action_sampler: Optional[ActionSampler] = None,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.n_simulations = n_simulations
        self.time_out_in_seconds = time_out_in_seconds
        self.action_sampler = action_sampler
        # Optional sizehint for Tree column buffers (POMCPOW.jl-style).
        self.reserve_capacity = reserve_capacity
        # Adaptive sizehint: remember the last tree's final node count so
        # the next ``_learn_tree`` call can reserve a buffer that already
        # covers the typical workload. Cuts the realloc bursts that
        # ``_allocate`` would otherwise pay on every call.
        self._last_tree_size: int = 0

        if n_simulations is not None and time_out_in_seconds is not None:
            raise ValueError("Cannot specify both n_simulations and time_out_in_seconds")

        if (
            self.action_sampler is None
            and self.environment.space_info.action_space == SpaceType.CONTINUOUS
        ):
            raise ValueError("Action sampler must be provided for continuous action spaces")

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        if self._is_terminal_belief(belief=belief):
            return [self._sample_random_action(belief=belief)], PolicyRunData(info_variables=[])

        tree, root_id = self._learn_tree(belief=belief)
        tree_metrics = compute_arena_tree_metrics(tree=tree, root_id=root_id)

        if not tree.children_ids[root_id]:
            action = self._sample_random_action(belief=belief)
        else:
            action = tree.best_action_by_reward(root_id)

        return [action], PolicyRunData(info_variables=tree_metrics)

    def _sample_random_action(self, belief: Belief) -> Any:
        if self.environment.space_info.action_space == SpaceType.DISCRETE:
            return np.random.choice(self.environment.get_actions())  # type: ignore[attr-defined]
        if self.environment.space_info.action_space == SpaceType.CONTINUOUS:
            if self.action_sampler is None:
                raise ValueError("action_sampler must not be None for continuous action spaces")
            # For backward-compat with custom samplers that inspect the
            # belief node's belief, wrap in a one-shot anytree BeliefNode
            # for this pre-tree fallback (cheap; called at most once per action()).
            return self.action_sampler.sample(belief_node=BeliefNode(belief=belief))
        raise ValueError(
            f"Unsupported action space type: {self.environment.space_info.action_space}"
        )

    def _learn_tree(self, belief: Belief) -> Tuple[Tree, int]:
        tree = Tree()
        capacity = self._effective_reserve_capacity()
        if capacity > 0:
            tree.reserve(capacity)
        root_id = tree.add_belief_node(belief)

        if self.n_simulations is not None:
            self._construct_tree_using_n_simulations(tree=tree, root_id=root_id)
        elif self.time_out_in_seconds is not None:
            self._construct_tree_using_timeout(tree=tree, root_id=root_id)
        else:
            raise ValueError("Either n_simulations or time_out_in_seconds must be provided")

        # Remember the final size so the next call can pre-reserve a
        # buffer that already covers the typical workload.
        self._last_tree_size = len(tree)
        return tree, root_id

    def _effective_reserve_capacity(self) -> int:
        """Capacity passed to :meth:`Tree.reserve` for each new tree.

        If the caller supplied an explicit ``reserve_capacity`` we honour it
        verbatim. Otherwise, prefer a sizehint derived from the previous
        call's final tree (adaptive, dialled in after one warmup) and fall
        back to ``2 * n_simulations`` when no history is available — each
        simulation adds at most one belief node and one action node per
        level it descends, so this is a safe-but-not-bloated baseline that
        covers shallow trees and lets deeper trees fall back to
        ``list.append`` past the reserved zone.
        """
        if self.reserve_capacity > 0:
            return self.reserve_capacity
        if self._last_tree_size > 0:
            # 1.25x headroom so a slightly larger tree still hits the
            # reserved zone instead of falling back to append.
            return (self._last_tree_size * 5) // 4
        if self.n_simulations is not None:
            return 2 * self.n_simulations
        return 0

    def _construct_tree_using_n_simulations(self, tree: Tree, root_id: int) -> None:
        if self.n_simulations is None:
            raise ValueError("n_simulations must not be None")
        for _ in range(self.n_simulations):
            self._simulate_path(tree=tree, belief_id=root_id, depth=0)

    def _construct_tree_using_timeout(self, tree: Tree, root_id: int) -> None:
        if self.time_out_in_seconds is None:
            raise ValueError("time_out_in_seconds must not be None")
        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:
            self._simulate_path(tree=tree, belief_id=root_id, depth=0)

    def _is_terminal_belief(self, belief: Belief) -> bool:
        if isinstance(
            belief,
            (
                WeightedParticleBelief,
                WeightedParticleBeliefStateUpdate,
                VectorizedWeightedParticleBelief,
                GaussianBelief,
                GaussianMixtureBelief,
            ),
        ):
            return is_terminal_belief(belief=belief, env=self.environment)
        raise ValueError("Unsupported belief type")

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return [metric.value for metric in TreeMetrics]

    @abstractmethod
    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> Optional[float]: ...


class ArenaDoubleProgressiveWideningMCTSPolicy(ArenaPathSimulationPolicy):
    """Arena-tree variant of :class:`DoubleProgressiveWideningMCTSPolicy`.

    Adds the standard double-progressive-widening configuration (k_a,
    alpha_a, k_o, alpha_o, exploration_constant, min_visit_count_per_action)
    on top of :class:`ArenaPathSimulationPolicy`. Same validation as the
    legacy class; subclasses (POMCPOW, PFT-DPW, etc.) implement
    :meth:`_simulate_path` using the arena Tree API.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        environment: Environment,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        k_a: float,
        alpha_a: float,
        k_o: float,
        alpha_o: float,
        exploration_constant: float,
        min_visit_count_per_action: int = 1,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self._validate_progressive_widening_params(
            depth=depth,
            k_a=k_a,
            alpha_a=alpha_a,
            k_o=k_o,
            alpha_o=alpha_o,
            exploration_constant=exploration_constant,
        )

        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            action_sampler=action_sampler,
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.depth = depth
        self.exploration_constant = exploration_constant
        self.min_visit_count_per_action = min_visit_count_per_action
        self.action_sampler: ActionSampler = action_sampler
        self.k_o = k_o
        self.k_a = k_a
        self.alpha_o = alpha_o
        self.alpha_a = alpha_a

    @staticmethod
    def _validate_progressive_widening_params(
        depth: int,
        k_a: float,
        alpha_a: float,
        k_o: float,
        alpha_o: float,
        exploration_constant: float,
    ) -> None:
        if not isinstance(depth, int):
            raise TypeError(f"depth must be an int, got {type(depth).__name__}")
        if not isinstance(k_a, (int, float)):
            raise TypeError(f"k_a must be a number, got {type(k_a).__name__}")
        if not isinstance(alpha_a, (int, float)):
            raise TypeError(f"alpha_a must be a number, got {type(alpha_a).__name__}")
        if not isinstance(k_o, (int, float)):
            raise TypeError(f"k_o must be a number, got {type(k_o).__name__}")
        if not isinstance(alpha_o, (int, float)):
            raise TypeError(f"alpha_o must be a number, got {type(alpha_o).__name__}")
        if not isinstance(exploration_constant, (int, float)):
            raise TypeError(
                f"exploration_constant must be a number, got {type(exploration_constant).__name__}"
            )
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if k_a <= 0:
            raise ValueError(f"k_a must be positive, got {k_a}")
        if alpha_a < 0 or alpha_a > 1:
            raise ValueError(f"alpha_a must be in (0, 1], got {alpha_a}")
        if k_o <= 0:
            raise ValueError(f"k_o must be positive, got {k_o}")
        if alpha_o < 0 or alpha_o > 1:
            raise ValueError(f"alpha_o must be in (0, 1], got {alpha_o}")
        if exploration_constant < 0:
            raise ValueError(
                f"exploration_constant must be non-negative, got {exploration_constant}"
            )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)


class ArenaPathSimulationPolicyCostSetting(ArenaPathSimulationPolicy):
    """Arena-tree variant of :class:`PathSimulationPolicyCostSetting`.

    Same shape as :class:`ArenaPathSimulationPolicy` but selects the
    optimal action by *minimum* q_value (cost setting) instead of maximum
    (reward setting). Used by the ICVaR planners.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        environment: Environment,
        discount_factor: float,
        name: str,
        action_sampler: Optional[ActionSampler] = None,
        n_simulations: Optional[int] = None,
        time_out_in_seconds: Optional[int] = None,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
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
            use_queue_logger=use_queue_logger,
        )

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        if is_terminal_belief(belief=belief, env=self.environment):
            return [self._sample_random_action(belief=belief)], PolicyRunData(info_variables=[])

        tree, root_id = self._learn_tree(belief=belief)
        tree_metrics = compute_arena_tree_metrics(tree=tree, root_id=root_id)

        if not tree.children_ids[root_id]:
            action = self._sample_random_action(belief=belief)
        else:
            # Cost setting: pick the action with the LOWEST q_value.
            action = tree.best_action_by_cost(root_id)

        return [action], PolicyRunData(info_variables=tree_metrics)
