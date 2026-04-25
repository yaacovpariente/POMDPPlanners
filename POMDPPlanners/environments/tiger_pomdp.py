"""Tiger POMDP Environment Implementation.

This module implements the classic Tiger problem, a benchmark POMDP environment
where an agent must determine which of two doors conceals a treasure and which
conceals a tiger, using only noisy acoustic observations.

The Tiger problem features:
- Two doors (left and right) with a tiger behind one and treasure behind the other
- Three actions: listen (to get information), open_left, open_right
- Three observations: hear_left, hear_right, hear_nothing
- Listening provides 85% accurate information about the tiger's location
- Opening the correct door yields +10 reward, opening wrong door yields -100
- Listening costs -1 per action

Classes:
    TigerStateTransition: State transition model for the Tiger problem
    TigerObservation: Observation model with noisy acoustic feedback
    TigerPOMDP: Main environment class implementing the Tiger problem
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue

STATES = ["tiger_left", "tiger_right"]
ACTIONS = ["listen", "open_left", "open_right"]
OBSERVATIONS = ["hear_left", "hear_right", "hear_nothing"]


class TigerPOMDPMetrics(Enum):
    """Metric names for Tiger POMDP environment."""

    SUCCESS_RATE = "success_rate"
    AVERAGE_LISTENS = "average_listens"


class TigerStateTransition(StateTransitionModel):
    """State transition model for the Tiger POMDP.

    The state only changes when a door is opened, after which the tiger
    is randomly placed behind one of the two doors for the next episode.

    Attributes:
        state: Current state (tiger_left or tiger_right)
        action: Action to be taken (listen, open_left, or open_right)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Create transition model for listening action
        >>> transition_listen = TigerStateTransition(state="tiger_left", action="listen")
        >>> next_state_listen = transition_listen.sample()[0]
        >>> next_state_listen == "tiger_left"  # No state change when listening
        True

        >>> # Create transition model for opening door
        >>> transition_open = TigerStateTransition(state="tiger_left", action="open_left")
        >>> next_state_open = transition_open.sample()[0]
        >>> next_state_open in ["tiger_left", "tiger_right"]  # Random outcome
        True

        >>> # Check probabilities for different outcomes
        >>> prob_same = transition_listen.probability(["tiger_left"])
        >>> bool(prob_same[0] == 1.0)  # Probability remains same when listening
        True
        >>> prob_random = transition_open.probability(["tiger_left"])
        >>> bool(prob_random[0] == 0.5)  # Equal probability when opening
        True
    """

    def __init__(self, state: str, action: str):
        """Initialize the state transition model.

        Args:
            state: Current state indicating tiger location
            action: Action being executed
        """
        super().__init__(state=state, action=action)

    def sample(self, n_samples: int = 1) -> List[str]:
        samples = []
        for _ in range(n_samples):
            # State only changes when opening a door
            if self.action in ("open_left", "open_right"):
                # After opening a door, tiger is randomly placed behind either door
                chosen_state = np.random.choice(STATES)
                # Ensure we return a Python string, not numpy string
                samples.append(str(chosen_state))
            else:
                samples.append(self.state)
        return samples

    def probability(self, values: List[Any]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, next_state in enumerate(values):
            if self.action in ("open_left", "open_right"):
                result[i] = 0.5
            else:
                result[i] = 1.0 if next_state == self.state else 0.0
        return result


class TigerObservation(ObservationModel):
    """Observation model for the Tiger POMDP.

    Provides noisy acoustic feedback when listening, with 85% accuracy.
    When doors are opened, no meaningful observation is provided.

    Attributes:
        next_state: The state after action execution
        action: The action that was taken

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Create observation model for listening when tiger is left
        >>> obs_listen = TigerObservation(next_state="tiger_left", action="listen")
        >>> observation = obs_listen.sample()[0]
        >>> observation in ["hear_left", "hear_right"]  # Listen gives acoustic feedback
        True

        >>> # Create observation model for opening door
        >>> obs_open = TigerObservation(next_state="tiger_left", action="open_left")
        >>> observation_open = obs_open.sample()[0]
        >>> observation_open == "hear_nothing"  # Opening always gives no sound
        True

        >>> # Check observation probabilities
        >>> prob_correct = obs_listen.probability(["hear_left"])
        >>> bool(prob_correct[0] == 0.85)  # Correct observation probability
        True
        >>> prob_wrong = obs_listen.probability(["hear_right"])
        >>> bool(prob_wrong[0] == 0.15)  # Wrong observation probability
        True
        >>> prob_nothing = obs_open.probability(["hear_nothing"])
        >>> bool(prob_nothing[0] == 1.0)  # Opening door always gives no sound
        True
    """

    def __init__(self, next_state: str, action: str):
        """Initialize the observation model.

        Args:
            next_state: State after taking the action
            action: Action that was executed
        """
        super().__init__(next_state=next_state, action=action)

    def sample(self, n_samples: int = 1) -> List[str]:
        samples = []
        for _ in range(n_samples):
            if self.action == "listen":
                # Listen action is 85% accurate
                if self.next_state == "tiger_left":
                    samples.append("hear_left" if np.random.random() < 0.85 else "hear_right")
                else:
                    samples.append("hear_right" if np.random.random() < 0.85 else "hear_left")
            else:
                # When opening a door, observation is random (hearing nothing)
                samples.append("hear_nothing")
        return samples

    def probability(self, values: List[Any]) -> np.ndarray:
        result = np.zeros(len(values))
        for i, next_observation in enumerate(values):
            if next_observation not in OBSERVATIONS:
                raise ValueError(f"Invalid observation: {next_observation}")

            if self.action == "listen":
                if self.next_state == "tiger_left":
                    result[i] = 0.85 if next_observation == "hear_left" else 0.15
                else:
                    result[i] = 0.85 if next_observation == "hear_right" else 0.15
            else:
                # For non-listen actions, only hear_nothing has probability 1.0, others have probability 0.0
                result[i] = 1.0 if next_observation == "hear_nothing" else 0.0
        return result


class TigerPOMDP(DiscreteActionsEnvironment):
    """Tiger POMDP environment implementation.

    This is the classic Tiger problem where an agent must decide which door to open
    to find treasure while avoiding the tiger. The agent can listen for acoustic cues
    but receives noisy observations.

    Problem Structure:
    - States: tiger_left, tiger_right (tiger's location)
    - Actions: listen, open_left, open_right
    - Observations: hear_left, hear_right, hear_nothing
    - Rewards: listen(-1), correct_door(+10), wrong_door(-100)

    Attributes:
        states: List of possible states
        actions: List of possible actions
        observations: List of possible observations

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> tiger = TigerPOMDP(discount_factor=0.95)
        >>>
        >>> # Get initial state and actions
        >>> initial_state = tiger.initial_state_dist().sample()[0]
        >>> actions = tiger.get_actions()
        >>>
        >>> # Sample complete step using convenience method
        >>> action = actions[0]
        >>> next_state, observation, reward = tiger.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> tiger.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        name: str = "TigerPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the Tiger POMDP environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Name identifier for this environment instance. Defaults to "TigerPOMDP".
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to True.

        Raises:
            ValueError: If discount_factor is not in valid range [0, 1]
        """
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Actions are discrete: listen, open_left, open_right
            observation_space=SpaceType.DISCRETE,  # Observations: hear_left, hear_right, hear_nothing
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(-100.0, 10.0),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self.states = STATES
        self.actions = ACTIONS
        self.observations = OBSERVATIONS

    def state_transition_model(self, state: str, action: str) -> StateTransitionModel:
        return TigerStateTransition(state=state, action=action)

    def observation_model(self, next_state: str, action: str) -> ObservationModel:
        return TigerObservation(next_state=next_state, action=action)

    # ── Hot-path sampling overrides ─────────────────────────────────
    # Inline the wrapper's sample() body to skip per-call wrapper
    # allocation. The RNG draw sequence is preserved bit-for-bit.

    def sample_next_state(self, state: str, action: str) -> str:
        if action in ("open_left", "open_right"):
            return str(np.random.choice(STATES))
        return state

    def sample_observation(self, next_state: str, action: str) -> str:
        if action == "listen":
            if next_state == "tiger_left":
                return "hear_left" if np.random.random() < 0.85 else "hear_right"
            return "hear_right" if np.random.random() < 0.85 else "hear_left"
        return "hear_nothing"

    def reward(self, state: str, action: str) -> float:
        if action == "listen":
            return -1.0  # Cost of listening
        if action == "open_left":
            if state == "tiger_left":
                return -100.0  # Opening door with tiger
            return 10.0  # Opening door with treasure
        # open_right
        if state == "tiger_right":
            return -100.0  # Opening door with tiger
        return 10.0  # Opening door with treasure

    def is_terminal(self, state: str) -> bool:
        # Game ends when a door is opened
        return False  # Since we handle terminal states in the transition model

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(values=STATES, probs=np.ones(len(STATES)) / len(STATES))

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(values=["hear_nothing"], probs=np.array([1.0]))

    def get_actions(self) -> List[Any]:
        return self.actions

    def cache_history_artifacts(self, history: History, cache_path: Path) -> None:
        pass

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def get_metric_names(self) -> List[str]:
        """Get names of Tiger POMDP specific metrics.

        Returns:
            List containing metric names: success_rate and average_listens
        """
        return [metric.value for metric in TigerPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute Tiger POMDP specific metrics from simulation histories.

        Args:
            histories: List of simulation histories

        Returns:
            List of MetricValue objects containing the computed metrics
        """
        # Calculate success rate (opening correct door)
        success_count = 0
        total_episodes = len(histories)

        for history in histories:
            # Check if the last action was opening a door
            last_step = history.history[-1]
            if last_step.action in ["open_left", "open_right"]:
                # Check if the door opened was correct
                if (last_step.action == "open_left" and last_step.state == "tiger_right") or (
                    last_step.action == "open_right" and last_step.state == "tiger_left"
                ):
                    success_count += 1

        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0

        # Calculate average number of listens before opening a door
        listen_counts = []
        for history in histories:
            listen_count = sum(1 for step in history.history if step.action == "listen")
            listen_counts.append(listen_count)

        avg_listens = sum(listen_counts) / len(listen_counts) if listen_counts else 0.0

        # Create MetricValue objects
        return [
            MetricValue(
                name=TigerPOMDPMetrics.SUCCESS_RATE.value,
                value=success_rate,
                lower_confidence_bound=success_rate,  # Simple implementation, could be improved with proper confidence intervals
                upper_confidence_bound=success_rate,
            ),
            MetricValue(
                name=TigerPOMDPMetrics.AVERAGE_LISTENS.value,
                value=avg_listens,
                lower_confidence_bound=avg_listens,
                upper_confidence_bound=avg_listens,
            ),
        ]
