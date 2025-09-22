from typing import Any, List, Tuple, Optional
from itertools import product
from pathlib import Path

import numpy as np

from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.belief import Belief


class DiscreteActionSequencesPlanner(Policy):
    """Open-loop planner for discrete action spaces using exhaustive sequence search.

    This planner uses an open-loop strategy to find optimal action sequences by
    enumerating all possible action sequences up to a specified depth and selecting
    the sequence with the highest expected return. It's particularly useful for
    problems with small action spaces and short planning horizons.

    The algorithm works by:
    1. Generating all possible action sequences of the specified depth
    2. For each sequence, estimating the expected return through Monte Carlo sampling
    3. Selecting the sequence with the maximum expected return
    4. Returning the first action in the optimal sequence

    **Open-Loop vs Closed-Loop Planning:**
    - **Open-loop**: Plans a complete action sequence without considering future observations
    - **Closed-loop**: Re-plans at each step based on new observations (like MCTS algorithms)

    This approach is computationally intensive (O(|A|^depth)) but provides optimal
    solutions for the open-loop setting when the action space is manageable.

    Args:
        environment: The discrete actions POMDP environment
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        name: Identifier for the planner instance
        depth: Planning horizon (number of actions in sequence)
        n_return_samples: Number of Monte Carlo samples for return estimation
        log_path: Optional path for logging planner execution details
        debug: Enable debug mode for detailed execution traces

    Example:
        Basic usage with Tiger POMDP::

            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
                DiscreteActionSequencesPlanner
            )
            from POMDPPlanners.core.belief import get_initial_belief

            # Create Tiger environment
            tiger = TigerPOMDP(discount_factor=0.95)

            # Create open-loop planner
            planner = DiscreteActionSequencesPlanner(
                environment=tiger,
                discount_factor=0.95,
                name="OpenLoop_Tiger",
                depth=3,                 # Plan 3 steps ahead
                n_return_samples=100     # Monte Carlo samples per sequence
            )

            # Plan from initial belief
            initial_belief = get_initial_belief(tiger, n_particles=200)
            action_sequence, run_data = planner.action(initial_belief)

            print(f"Optimal sequence starts with: {action_sequence[0]}")
            # Output: Optimal sequence starts with: listen

    Example:
        Comparing planning depths::

            # Short-term planning (fast, myopic)
            short_planner = DiscreteActionSequencesPlanner(
                environment=tiger,
                discount_factor=0.95,
                name="ShortTerm_OpenLoop",
                depth=2,
                n_return_samples=50
            )

            # Long-term planning (slower, more thorough)
            long_planner = DiscreteActionSequencesPlanner(
                environment=tiger,
                discount_factor=0.95,
                name="LongTerm_OpenLoop",
                depth=4,
                n_return_samples=200
            )

            # Compare strategies
            short_action, _ = short_planner.action(initial_belief)
            long_action, _ = long_planner.action(initial_belief)

            print(f"Short-term strategy: {short_action[0]}")
            print(f"Long-term strategy: {long_action[0]}")

    Performance Considerations:
        **Computational Complexity**: O(|A|^depth × n_return_samples)

        - Tiger POMDP (3 actions): depth=3 → 27 sequences
        - CartPole (2 actions): depth=5 → 32 sequences
        - Large action spaces quickly become intractable

        **Memory Usage**: Stores all action sequences simultaneously

        **Scalability Guidelines**:
        - depth ≤ 4 for |A| ≥ 5 actions
        - depth ≤ 6 for |A| ≤ 3 actions
        - n_return_samples: 50-500 depending on environment stochasticity

    Algorithm Comparison:
        **vs MCTS algorithms (POMCP, PFT-DPW):**
        - **Advantages**: Globally optimal for open-loop setting, simpler implementation
        - **Disadvantages**: Exponential complexity, no adaptation to observations

        **vs Sparse Sampling:**
        - **Advantages**: Examines all sequences exhaustively
        - **Disadvantages**: No selective exploration, poor scalability

        **Best Use Cases:**
        - Small discrete action spaces (|A| ≤ 5)
        - Short planning horizons (depth ≤ 4)
        - Environments where open-loop planning is reasonable
        - Baseline for comparing more sophisticated algorithms
    """

    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        name: str,
        depth: int,
        n_return_samples: int,
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

        if depth <= 0:
            raise ValueError("depth must be greater than 0")
        if n_return_samples <= 0:
            raise ValueError("n_return_samples must be greater than 0")
        if not (1 >= discount_factor >= 0):
            raise ValueError("discount_factor must be between 0 and 1")

        self.depth = depth
        self.n_return_samples = n_return_samples
        self.discount_factor = discount_factor

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED
        )

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        return self.search(belief), PolicyRunData(info_variables=[])

    def search(self, belief: Belief) -> Any:
        actions = self.environment.get_actions()
        action_sequences = list(product(actions, repeat=self.depth))
        returns = []

        for action_sequence in action_sequences:
            returns.append(
                self.estimate_return(action_sequence=action_sequence, belief=belief)
            )

        return list(action_sequences[np.argmax(returns)])

    def estimate_return(self, action_sequence: List[Any], belief: Belief) -> float:
        return_estimator = 0
        for _ in range(self.n_return_samples):
            state = belief.sample()
            return_sample = 0

            for i, action in enumerate(action_sequence):
                state, observation, reward = self.environment.sample_next_step(
                    state, action
                )
                return_sample += reward * (self.discount_factor**i)

            return_estimator += return_sample

        return_estimator /= self.n_return_samples

        return return_estimator
