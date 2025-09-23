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
    SanityStateTransitionModel: Deterministic state transitions
    SanityObservationModel: Perfect state observation
    SanityInitialStateDist: Always starts in good state
    SanityInitialObservationDist: Initial observation distribution
    SanityPOMDP: Main environment class for sanity testing
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)


class SanityStateTransitionModel(Distribution):
    """Deterministic state transition model for Sanity POMDP.

    This model implements completely deterministic state transitions where:
    - Action 0 always leads to state 0 (good state)
    - Action 1 always leads to state 1 (bad state)

    The deterministic nature makes this ideal for testing and debugging
    POMDP algorithms since the outcomes are predictable.

    Attributes:
        state: Current state (0 or 1)
        action: Action to be executed (0 or 1)

    Example:
        Using the state transition model::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> # Create transition model from bad state with good action
            >>> transition_model = SanityStateTransitionModel(state=1, action=0)
            >>>
            >>> # Sample next state (always deterministic)
            >>> next_state = transition_model.sample()[0]  # Returns 0 (good state)
            >>> next_state == 0
            True
            >>>
            >>> # Check probability of specific outcomes
            >>> prob = transition_model.probability([0])  # Returns [1.0]
            >>> bool(prob[0] == 1.0)
            True
            >>> prob_wrong = transition_model.probability([1])  # Returns [0.0]
            >>> bool(prob_wrong[0] == 0.0)
            True
    """

    def __init__(self, state: int, action: int):
        """Initialize the state transition model.

        Args:
            state: Current state (0 for good, 1 for bad)
            action: Action to execute (0 for go to good, 1 for go to bad)
        """
        super().__init__()
        self.state = state
        self.action = action

    def sample(self, n_samples: int = 1) -> List[int]:
        # Action 0 always leads to state 0 (good state)
        # Action 1 always leads to state 1 (bad state)
        next_state = 0 if self.action == 0 else 1
        return [next_state] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        expected_next_state = 0 if self.action == 0 else 1
        for i, value in enumerate(values):
            if value == expected_next_state:
                result[i] = 1.0
        return result


class SanityObservationModel(Distribution):
    """Perfect observation model for Sanity POMDP.

    This model provides perfect observability where the observation
    always exactly matches the state. This eliminates partial observability
    and makes the problem fully observable, which is ideal for testing
    algorithms in the simplest possible setting.

    Attributes:
        next_state: The state after action execution
        action: The action that was taken (not used in observation generation)

    Example:
        Using the observation model::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> # Create observation model for good state
            >>> obs_model = SanityObservationModel(next_state=0, action=0)
            >>>
            >>> # Sample observation (always matches state)
            >>> observation = obs_model.sample()[0]  # Returns 0
            >>> observation == 0
            True
            >>>
            >>> # Check observation probabilities
            >>> prob_correct = obs_model.probability([0])  # Returns [1.0]
            >>> bool(prob_correct[0] == 1.0)
            True
            >>> prob_wrong = obs_model.probability([1])  # Returns [0.0]
            >>> bool(prob_wrong[0] == 0.0)
            True
    """

    def __init__(self, next_state: int, action: int):
        """Initialize the observation model.

        Args:
            next_state: State after taking the action
            action: Action that was executed (not used for observation)
        """
        super().__init__()
        self.next_state = next_state
        self.action = action

    def sample(self, n_samples: int = 1) -> List[int]:
        # Observation always matches the state
        return [self.next_state] * n_samples

    def probability(self, values: List[int]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            if value == self.next_state:
                result[i] = 1.0
        return result


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
        Creating and using a Sanity POMDP::

            >>> import numpy as np
            >>> np.random.seed(42)  # For reproducible results
            >>> # Create sanity test environment
            >>> sanity = SanityPOMDP(discount_factor=0.95)
            >>>
            >>> # Get actions and verify simple dynamics
            >>> actions = sanity.get_actions()  # [0, 1]
            >>> len(actions) == 2
            True
            >>> 0 in actions and 1 in actions
            True
            >>>
            >>> # Test state transitions
            >>> reward_good = sanity.reward(state=0, action=0)  # Should be 1.0
            >>> reward_good == 1.0
            True
            >>> reward_bad = sanity.reward(state=1, action=0)   # Should be 1.0 (goes to good state)
            >>> reward_bad == 1.0
            True
            >>>
            >>> # Verify perfect observability
            >>> obs_model = sanity.observation_model(next_state=0, action=0)
            >>> observation = obs_model.sample()[0]  # Should be 0
            >>> observation == 0
            True
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

    def state_transition_model(self, state: int, action: int) -> SanityStateTransitionModel:
        return SanityStateTransitionModel(state, action)

    def observation_model(self, next_state: int, action: int) -> SanityObservationModel:
        return SanityObservationModel(next_state, action)

    def reward(self, state: int, action: int) -> float:
        # Higher reward for being in good state (0)
        next_state = self.state_transition_model(state, action).sample()[0]
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
