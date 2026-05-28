# SPDX-License-Identifier: MIT

from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo


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
        >>> import numpy as np
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Create environment and planner
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>> planner = DiscreteActionSequencesPlanner(
        ...     environment=tiger,
        ...     discount_factor=0.95,
        ...     name="ExamplePlanner",
        ...     depth=2,
        ...     n_return_samples=10
        ... )
        >>>
        >>> # Basic planner interface usage
        >>> planner.name
        'ExamplePlanner'
        >>>
        >>> # Action selection from belief
        >>> initial_belief = get_initial_belief(tiger, n_particles=10)
        >>> actions, run_data = planner.action(initial_belief)
        >>>
        >>> # Planner space information
        >>> space_info = DiscreteActionSequencesPlanner.get_space_info()
        >>> space_info.action_space.name
        'DISCRETE'
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
        if not 1 >= discount_factor >= 0:
            raise ValueError("discount_factor must be between 0 and 1")

        self.depth = depth
        self.n_return_samples = n_return_samples
        self.discount_factor = discount_factor
        # Override type annotation for mypy
        self.environment: DiscreteActionsEnvironment = environment

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED)

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """Get names of policy info variables.

        Discrete action sequences planner does not produce any info variables.

        Returns:
            Empty list as this planner produces no info variables
        """
        return []

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        return self.search(belief), PolicyRunData(info_variables=[])

    def search(self, belief: Belief) -> Any:
        actions = self.environment.get_actions()
        action_sequences = list(product(actions, repeat=self.depth))
        returns = []

        for action_sequence in action_sequences:
            returns.append(
                self.estimate_return(action_sequence=list(action_sequence), belief=belief)
            )

        return list(action_sequences[np.argmax(returns)])

    def estimate_return(self, action_sequence: List[Any], belief: Belief) -> float:
        return_estimator: float = 0.0
        for _ in range(self.n_return_samples):
            state = belief.sample()
            return_sample = 0.0

            for i, action in enumerate(action_sequence):
                state, _, reward = self.environment.sample_next_step(state, action)
                return_sample += reward * (self.discount_factor**i)

            return_estimator += return_sample

        return_estimator /= self.n_return_samples

        return return_estimator
