from abc import abstractmethod
import time
from typing import Any, List, Tuple, Optional
from pathlib import Path
import numpy as np

from POMDPPlanners.core.policy import Policy, PolicyRunData
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.belief import Belief, is_terminal_belief
from POMDPPlanners.core.tree import BeliefNode, get_optimal_action_reward_setting
from POMDPPlanners.utils.tree_statistics import compute_tree_metrics
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


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
        n_simulations: int,
        time_out_in_seconds: int,
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
        if is_terminal_belief(belief=belief, env=self.environment):
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
            return np.random.choice(self.environment.get_actions())
        elif self.environment.space_info.action_space == SpaceType.CONTINUOUS:
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

    def _construct_tree_using_n_simulations(self, belief_node: BeliefNode) -> BeliefNode:
        if self.n_simulations is None:
            raise ValueError("n_simulations must not be None")

        for _ in range(self.n_simulations):
            self._simulate_path(belief_node=belief_node, depth=0)

    def _construct_tree_using_timeout(self, belief_node: BeliefNode) -> BeliefNode:
        if self.time_out_in_seconds is None:
            raise ValueError("time_out_in_seconds must not be None")

        start_time = time.time()
        while time.time() - start_time < self.time_out_in_seconds:
            self._simulate_path(belief_node=belief_node, depth=0)

    @abstractmethod
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        pass
