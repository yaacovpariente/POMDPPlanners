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
from collections.abc import Hashable
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue

STATES = ["tiger_left", "tiger_right"]
ACTIONS = ["listen", "open_left", "open_right"]
OBSERVATIONS = ["hear_left", "hear_right", "hear_nothing"]


class TigerPOMDPMetrics(Enum):
    """Metric names for Tiger POMDP environment."""

    SUCCESS_RATE = "success_rate"
    AVERAGE_LISTENS = "average_listens"


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

    # ── Hot-path sampling overrides ─────────────────────────────────
    # Inline the wrapper's sample() body to skip per-call wrapper
    # allocation. The RNG draw sequence is preserved bit-for-bit.

    def sample_next_state(self, state: str, action: str, n_samples: int = 1):
        if action in ("open_left", "open_right"):
            # randint(0, 2) is ~3.5x faster than np.random.choice over a list
            if n_samples == 1:
                return STATES[np.random.randint(0, 2)]
            idxs = np.random.randint(0, 2, size=n_samples)
            return [STATES[i] for i in idxs]
        if n_samples == 1:
            return state
        return [state] * n_samples

    def sample_observation(self, next_state: str, action: str, n_samples: int = 1):
        if action != "listen":
            if n_samples == 1:
                return "hear_nothing"
            return ["hear_nothing"] * n_samples
        # Listen: 0.85 prob hear matching ear, 0.15 hear opposite.
        # Single draw of np.random.random per call (n=1) or one batched draw (n>1).
        correct = "hear_left" if next_state == "tiger_left" else "hear_right"
        wrong = "hear_right" if next_state == "tiger_left" else "hear_left"
        if n_samples == 1:
            return correct if np.random.random() < 0.85 else wrong
        draws = np.random.random(size=n_samples)
        return [correct if d < 0.85 else wrong for d in draws]

    def transition_log_probability(self, state: str, action: str, next_states) -> np.ndarray:
        result = np.zeros(len(next_states))
        if action in ("open_left", "open_right"):
            for i, ns in enumerate(next_states):
                result[i] = np.log(0.5) if ns in STATES else -np.inf
        else:
            for i, ns in enumerate(next_states):
                result[i] = 0.0 if ns == state else -np.inf
        return result

    def observation_log_probability(self, next_state: str, action: str, observations) -> np.ndarray:
        result = np.zeros(len(observations))
        if action == "listen":
            # Listen emits only directional observations; ``hear_nothing`` is
            # impossible under listen and reserved for the open_* actions.
            # Using ``OBSERVATIONS`` here would let ``hear_nothing`` fall through
            # and incorrectly receive log(0.15), making the kernel sum to 1.15.
            for i, obs in enumerate(observations):
                if obs not in ("hear_left", "hear_right"):
                    result[i] = -np.inf
                    continue
                correct = "hear_left" if next_state == "tiger_left" else "hear_right"
                result[i] = np.log(0.85) if obs == correct else np.log(0.15)
        else:
            for i, obs in enumerate(observations):
                result[i] = 0.0 if obs == "hear_nothing" else -np.inf
        return result

    def reward(self, state: str, action: str, next_state: Any = None) -> float:
        del next_state
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

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: str,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        del next_states
        if action == "listen":
            return np.full(len(states), -1.0)
        # open_left: -100 if tiger_left, else +10
        # open_right: -100 if tiger_right, else +10
        if action == "open_left":
            return np.array([-100.0 if s == "tiger_left" else 10.0 for s in states])
        # open_right
        return np.array([-100.0 if s == "tiger_right" else 10.0 for s in states])

    def sample_next_state_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: str
    ) -> List[str]:
        n = len(states)
        if action in ("open_left", "open_right"):
            idxs = np.random.randint(0, 2, size=n)
            return [STATES[i] for i in idxs]
        return list(states)

    def observation_log_probability_per_state(
        self,
        next_states: Union[np.ndarray, Sequence[Any]],
        action: str,
        observation: str,
    ) -> np.ndarray:
        n = len(next_states)
        if action != "listen":
            fill = 0.0 if observation == "hear_nothing" else -np.inf
            return np.full(n, fill)
        # listen: correct ear =0.85, other =0.15; vectorised string compare
        states_arr = np.asarray(next_states)
        is_left = states_arr == "tiger_left"
        # correct observation for tiger_left is hear_left; for tiger_right is hear_right
        obs_is_left = observation == "hear_left"
        obs_is_right = observation == "hear_right"
        log_correct = np.log(0.85)
        log_wrong = np.log(0.15)
        result = np.full(n, -np.inf)
        if obs_is_left:
            result = np.where(is_left, log_correct, log_wrong)
        elif obs_is_right:
            result = np.where(is_left, log_wrong, log_correct)
        # else hear_nothing while listening -> -inf (stays -inf)
        return result

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

    def hash_action(self, action: Any) -> Hashable:
        # Discrete-action env: actions are str labels (LISTEN/OPEN_LEFT/...).
        return action

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
