"""Sanity Check POMDP Environment Implementation.

This module implements a simple test environment used for debugging and sanity
checking POMDP algorithms. The environment has deterministic dynamics and
perfect observability, making it ideal for verifying algorithm correctness.

The Sanity POMDP features:
- Two discrete states: 0 (good) and 1 (bad)
- Two discrete actions: 0 (go to good state) and 1 (go to bad state)
- Perfect observations: observation always equals the state
- Simple reward structure: 1.0 for good state, 0.0 for bad state
- No terminal states (infinite horizon)

This environment is primarily used for:
- Testing POMDP algorithm implementations
- Debugging belief updates and planning algorithms
- Verifying that algorithms can solve trivial cases
- Performance benchmarking baseline

Classes:
    SanityInitialStateDist: Always starts in good state
    SanityInitialObservationDist: Initial observation distribution
    SanityPOMDP: Main environment class for sanity testing
"""

from pathlib import Path
from collections.abc import Hashable
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)


class SanityInitialStateDist(Distribution):
    """Initial state distribution for Sanity POMDP.

    This distribution always returns state 0 (good state) as the initial state,
    providing a deterministic and predictable starting condition for testing.

    Example:
        Using the initial state distribution::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> # Create initial state distribution
            >>> initial_dist = SanityInitialStateDist()
            >>>
            >>> # Sample initial state (always returns good state)
            >>> initial_state = initial_dist.sample()[0]  # Returns 0
            >>> initial_state == 0
            True
            >>>
            >>> # Sample multiple initial states
            >>> states = initial_dist.sample(n_samples=5)  # Returns [0, 0, 0, 0, 0]
            >>> len(states) == 5
            True
            >>> all(state == 0 for state in states)
            True
            >>>
            >>> # Check probability of initial states
            >>> prob_good = initial_dist.probability([0])  # Returns [1.0]
            >>> bool(prob_good[0] == 1.0)
            True
            >>> prob_bad = initial_dist.probability([1])   # Returns [0.0]
            >>> bool(prob_bad[0] == 0.0)
            True
    """

    def sample(self, n_samples: int = 1) -> List[int]:
        """Sample initial states.

        Args:
            n_samples: Number of samples to return

        Returns:
            List of initial states (always [0, 0, ...])
        """
        # Always start in good state (0)
        return [0] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == 0:
                result[i] = 1.0
        return result


class SanityInitialObservationDist(Distribution):
    """Initial observation distribution for Sanity POMDP.

    This distribution always returns observation 0 (corresponding to the good state)
    as the initial observation, maintaining consistency with the initial state
    distribution and perfect observability property.

    Example:
        Using the initial observation distribution::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> # Create initial observation distribution
            >>> initial_obs_dist = SanityInitialObservationDist()
            >>>
            >>> # Sample initial observation
            >>> initial_obs = initial_obs_dist.sample()[0]  # Returns 0
            >>> initial_obs == 0
            True
            >>>
            >>> # Sample multiple observations
            >>> observations = initial_obs_dist.sample(n_samples=3)  # Returns [0, 0, 0]
            >>> len(observations) == 3
            True
            >>> all(obs == 0 for obs in observations)
            True
            >>>
            >>> # Check observation probabilities
            >>> prob = initial_obs_dist.probability([0])  # Returns [1.0]
            >>> bool(prob[0] == 1.0)
            True
    """

    def sample(self, n_samples: int = 1) -> List[int]:
        """Sample initial observations.

        Args:
            n_samples: Number of samples to return

        Returns:
            List of initial observations (always [0, 0, ...])
        """
        # Initial observation always matches initial state (0)
        return [0] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == 0:
                result[i] = 1.0
        return result


class SanityPOMDP(DiscreteActionsEnvironment):
    """Simple sanity check POMDP environment for testing and debugging.

    This environment provides the simplest possible POMDP formulation with
    deterministic dynamics and perfect observability. It serves as a baseline
    for testing POMDP algorithms and ensuring correctness.

    Problem Structure:
    - States: 0 (good), 1 (bad)
    - Actions: 0 (choose good), 1 (choose bad)
    - Observations: Same as states (perfect observability)
    - Rewards: 1.0 for good state, 0.0 for bad state
    - Dynamics: Deterministic state transitions based on action

    The optimal policy is trivial: always choose action 0 to stay in the good state.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = SanityPOMDP(discount_factor=0.95)
        >>>
        >>> # Get initial state and actions
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>> actions = env.get_actions()
        >>>
        >>> # Sample complete step using convenience method
        >>> action = actions[0]
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the Sanity POMDP environment.

        Args:
            discount_factor: Discount factor for future rewards. Defaults to 0.95.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Binary action space
            observation_space=SpaceType.DISCRETE,  # Binary observation space
        )
        super().__init__(
            discount_factor=discount_factor,
            name="SanityPOMDP",
            space_info=space_info,
            reward_range=(0.0, 1.0),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    # Both transitions and observations are deterministic, so no RNG
    # draws occur in any of the env-API methods below.

    def sample_next_state(
        self, state: int, action: int, n_samples: int = 1
    ) -> Union[int, np.ndarray]:
        next_state = 0 if action == 0 else 1
        if n_samples == 1:
            return next_state
        return np.full(n_samples, next_state, dtype=np.int64)

    def sample_observation(
        self, next_state: int, action: int, n_samples: int = 1
    ) -> Union[int, np.ndarray]:
        if n_samples == 1:
            return next_state
        return np.full(n_samples, next_state, dtype=np.int64)

    def transition_log_probability(
        self,
        state: int,
        action: int,
        next_states: Union[Sequence[int], np.ndarray],
    ) -> np.ndarray:
        next_states_arr = np.asarray(next_states)
        expected = 0 if action == 0 else 1
        probs = np.where(next_states_arr == expected, 1.0, 0.0)
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self,
        next_state: int,
        action: int,
        observations: Union[Sequence[int], np.ndarray],
    ) -> np.ndarray:
        observations_arr = np.asarray(observations)
        probs = np.where(observations_arr == next_state, 1.0, 0.0)
        return np.log(probs + 1e-300)

    def reward(self, state: int, action: int) -> float:
        # Higher reward for being in good state (0)
        next_state = self.sample_next_state(state, action)
        return 1.0 if next_state == 0 else 0.0

    def is_terminal(self, state: int) -> bool:
        return False  # No terminal states

    def initial_state_dist(self) -> SanityInitialStateDist:
        return SanityInitialStateDist()

    def initial_observation_dist(self) -> SanityInitialObservationDist:
        return SanityInitialObservationDist()

    def get_actions(self) -> List[int]:
        return [0, 1]  # Two actions: 0 (good) and 1 (bad)

    def is_equal_observation(self, observation1: int, observation2: int) -> bool:
        return observation1 == observation2

    def hash_action(self, action: Any) -> Hashable:
        # Discrete int actions (0, 1); already hashable.
        return action
