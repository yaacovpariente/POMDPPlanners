import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import (
    Belief,
    is_terminal_belief,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.tree import BeliefNode, get_optimal_action_reward_setting
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.tree_statistics import TreeMetrics, compute_tree_metrics


class PathSimulationPolicy(Policy):
    """Abstract base class for Monte Carlo Tree Search algorithms in POMDP planning.

    This class provides a common framework for MCTS-based POMDP planners that build
    search trees through path simulations. It handles the core tree construction loop
    and provides hooks for algorithm-specific simulation strategies.

    The class supports two termination criteria:
    1. **Simulation count**: Run a fixed number of MCTS simulations
    2. **Time limit**: Run simulations for a specified time duration

    Key Components:
    - Tree construction with configurable termination criteria
    - Automatic tree metrics collection for analysis
    - Action selection from the constructed search tree
    - Abstract simulation interface for algorithm specialization

    Subclass Responsibilities:
    Concrete implementations must provide the `_simulate_path` method that defines
    how individual MCTS simulations are performed, including:
    - Node expansion strategies
    - Action selection during tree traversal
    - Value estimation and backpropagation

    Attributes:
        environment: The POMDP environment for planning
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        n_simulations: Number of MCTS simulations to run (mutually exclusive with timeout)
        time_out_in_seconds: Time limit for planning in seconds (mutually exclusive with n_simulations)
        name: Identifier for the policy instance

    Algorithm Integration:
    This base class is used by several MCTS algorithms in the framework:
    - **POMCP**: Uses UCB1 for action selection with particle filtering
    - **PFT-DPW**: Implements progressive widening for continuous action spaces
    - **Sparse-PFT**: Combines sparse sampling with progressive widening

    The common interface allows easy comparison and benchmarking of different
    MCTS variants while sharing the core tree construction infrastructure.
    """

    def __init__(
        self,
        environment: "Environment",
        discount_factor: float,
        name: str,
        n_simulations: Optional[int],
        time_out_in_seconds: Optional[int],
        action_sampler: Optional[ActionSampler] = None,
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

        tree = self._learn_tree(belief=belief)
        tree_metrics = compute_tree_metrics(tree=tree)

        if tree.is_leaf:
            action = self._sample_random_action(belief=belief)
        else:
            action = get_optimal_action_reward_setting(belief_node=tree)

        return [action], PolicyRunData(info_variables=tree_metrics)

    def _sample_random_action(self, belief: Belief) -> Any:
        if self.environment.space_info.action_space == SpaceType.DISCRETE:
            return np.random.choice(self.environment.get_actions())  # type: ignore[attr-defined]
        elif self.environment.space_info.action_space == SpaceType.CONTINUOUS:
            if self.action_sampler is None:
                raise ValueError("action_sampler must not be None for continuous action spaces")
            return self.action_sampler.sample(belief_node=BeliefNode(belief=belief))

    def _learn_tree(self, belief: Belief) -> BeliefNode:
        tree = BeliefNode(belief=belief)

        if self.n_simulations is not None:
            self._construct_tree_using_n_simulations(belief_node=tree)
        elif self.time_out_in_seconds is not None:
            self._construct_tree_using_timeout(belief_node=tree)
        else:
            raise ValueError("Either n_simulations or time_out_in_seconds must be provided")

        return tree

    def _construct_tree_using_n_simulations(self, belief_node: BeliefNode):
        if self.n_simulations is None:
            raise ValueError("n_simulations must not be None")

        for _ in range(self.n_simulations):
            self._simulate_path(belief_node=belief_node, depth=0)

    def _construct_tree_using_timeout(self, belief_node: BeliefNode):
        if self.time_out_in_seconds is None:
            raise ValueError("time_out_in_seconds must not be None")

        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:
            self._simulate_path(belief_node=belief_node, depth=0)

    def _is_terminal_belief(self, belief: Belief) -> bool:
        if isinstance(belief, (WeightedParticleBelief, WeightedParticleBeliefStateUpdate)):
            return is_terminal_belief(belief=belief, env=self.environment)

        raise ValueError("Unsupported belief type")

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """Get names of tree metric info variables produced by path simulation policies.

        Returns:
            List of metric names from tree statistics
        """
        return [metric.value for metric in TreeMetrics]

    @abstractmethod
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> Optional[float]:
        pass


class DoubleProgressiveWideningMCTSPolicy(PathSimulationPolicy):
    """Abstract base class for MCTS planners using double progressive widening.

    This base class provides common initialization, parameter validation, and attributes
    for MCTS planners that use double progressive widening (both action and observation
    progressive widening). Subclasses implement their own simulation strategies while
    sharing common parameters and validation logic.

    Progressive Widening Overview:
    Double progressive widening controls tree growth by limiting how many actions
    and observations are added to the tree based on visit counts:
    - Action widening: New actions added when ⌊k_a * n^α_a⌋ increases
    - Observation widening: Max observations limited by ⌊k_o * n^α_o⌋

    Common Progressive Widening Parameters:
    - k_a, alpha_a: Control action progressive widening
    - k_o, alpha_o: Control observation progressive widening
    - exploration_constant: UCB1 exploration parameter

    Attributes:
        depth: Maximum search depth for tree expansion
        exploration_constant: UCB1 exploration parameter (c in UCB1 formula)
        min_samples_per_node: Minimum samples before a node is considered reliable
        action_sampler: Action sampling strategy for progressive widening
        k_o: Observation progressive widening coefficient (k_o > 0)
        k_a: Action progressive widening coefficient (k_a > 0)
        alpha_o: Observation progressive widening exponent (0 < α_o ≤ 1)
        alpha_a: Action progressive widening exponent (0 < α_a ≤ 1)

    Subclasses:
        - POMCP_DPW: Uses unweighted particle beliefs with double progressive widening
        - POMCPOW: Uses weighted particle beliefs with double progressive widening
        - PFT_DPW: Uses progressive function transfer with custom simulation strategy

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the `_simulate_path` method.
    """

    def __init__(
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
        min_samples_per_node: int = 10,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize progressive widening MCTS planner with parameter validation.

        Args:
            environment: The POMDP environment to plan for
            discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
            depth: Maximum search depth for tree expansion (depth > 0)
            name: Identifier for the policy instance
            action_sampler: Action sampling strategy for progressive widening
            k_a: Action progressive widening coefficient (k_a > 0)
            alpha_a: Action progressive widening exponent (0 < α_a ≤ 1)
            k_o: Observation progressive widening coefficient (k_o > 0)
            alpha_o: Observation progressive widening exponent (0 < α_o ≤ 1)
            exploration_constant: UCB1 exploration parameter (exploration_constant ≥ 0)
            min_samples_per_node: Minimum samples before node is reliable (min_samples_per_node ≥ 1)
            time_out_in_seconds: Time limit for planning (mutually exclusive with n_simulations)
            n_simulations: Number of simulations to run (mutually exclusive with timeout)
            log_path: Optional path for logging policy execution
            debug: Enable debug logging if True
            use_queue_logger: Use queue-based logging for multiprocessing

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters have invalid values
        """
        # Validate inputs before calling super().__init__
        self._validate_progressive_widening_params(
            depth=depth,
            k_a=k_a,
            alpha_a=alpha_a,
            k_o=k_o,
            alpha_o=alpha_o,
            exploration_constant=exploration_constant,
            min_samples_per_node=min_samples_per_node,
        )

        # Call parent class constructor
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            action_sampler=action_sampler,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Initialize progressive widening attributes
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.min_samples_per_node = min_samples_per_node
        self.action_sampler: ActionSampler = action_sampler

        # Progressive widening parameters
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
        min_samples_per_node: int,
    ) -> None:
        """Validate progressive widening parameters.

        Args:
            depth: Maximum search depth
            k_a: Action progressive widening coefficient
            alpha_a: Action progressive widening exponent
            k_o: Observation progressive widening coefficient
            alpha_o: Observation progressive widening exponent
            exploration_constant: UCB1 exploration parameter
            min_samples_per_node: Minimum samples per node

        Raises:
            TypeError: If parameters have incorrect types
            ValueError: If parameters have invalid values
        """
        # Type checks
        if not isinstance(depth, int):
            raise TypeError(f"depth must be an int, got {type(depth).__name__}")
        if not isinstance(min_samples_per_node, int):
            raise TypeError(
                f"min_samples_per_node must be an int, got {type(min_samples_per_node).__name__}"
            )
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

        # Value checks
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if min_samples_per_node < 1:
            raise ValueError(f"min_samples_per_node must be >= 1, got {min_samples_per_node}")
        if k_a <= 0:
            raise ValueError(f"k_a must be positive, got {k_a}")
        if alpha_a <= 0 or alpha_a > 1:
            raise ValueError(f"alpha_a must be in (0, 1], got {alpha_a}")
        if k_o <= 0:
            raise ValueError(f"k_o must be positive, got {k_o}")
        if alpha_o <= 0 or alpha_o > 1:
            raise ValueError(f"alpha_o must be in (0, 1], got {alpha_o}")
        if exploration_constant < 0:
            raise ValueError(
                f"exploration_constant must be non-negative, got {exploration_constant}"
            )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get information about action and observation spaces.

        Default implementation returns MIXED space types, which is appropriate
        for most progressive widening MCTS planners that support both discrete
        and continuous action spaces through the action sampler interface.

        Subclasses can override this method to specify different space requirements
        (e.g., PFT_DPW specifies CONTINUOUS action space).

        Returns:
            PolicySpaceInfo with MIXED space types for both actions and observations
        """
        return PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)
