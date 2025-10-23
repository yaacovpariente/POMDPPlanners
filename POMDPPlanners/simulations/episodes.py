import copy
from logging import Logger
from time import time
from typing import Any, List, Optional, Tuple

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy, PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.utils.logger import get_logger


def _validate_episode_inputs(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
) -> None:
    if environment is None:
        raise ValueError("environment cannot be None")
    if policy is None:
        raise ValueError("policy cannot be None")
    if initial_belief is None:
        raise ValueError("initial_belief cannot be None")
    if num_steps is None:
        raise ValueError("num_steps cannot be None")

    if not isinstance(environment, Environment):
        raise TypeError(f"environment must be an instance of Environment, got {type(environment)}")
    if not isinstance(policy, Policy):
        raise TypeError(f"policy must be an instance of Policy, got {type(policy)}")
    if not isinstance(initial_belief, Belief):
        raise TypeError(f"initial_belief must be an instance of Belief, got {type(initial_belief)}")
    if not isinstance(num_steps, int):
        raise TypeError(f"num_steps must be an integer, got {type(num_steps)}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if not hasattr(environment, "state_transition_model") or not hasattr(
        environment, "observation_model"
    ):
        raise ValueError("environment must implement state_transition_model and observation_model")
    if not hasattr(policy, "action"):
        raise ValueError("policy must implement action method")
    if not hasattr(initial_belief, "sample") or not hasattr(initial_belief, "update"):
        raise ValueError("initial_belief must implement sample and update methods")


class EpisodeRunner:
    """Executes a single POMDP episode and collects performance metrics."""

    def __init__(
        self,
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        num_steps: int,
        logger: Optional[Logger] = None,
    ):
        _validate_episode_inputs(environment, policy, initial_belief, num_steps)

        self.environment = environment
        self.policy = policy
        self.num_steps = num_steps
        self.logger = logger or get_logger(f"episode.{environment.name}.{policy.name}")

        # Episode state
        self.belief = copy.deepcopy(initial_belief)
        self.state = initial_belief.sample()
        self.history = []
        self.policy_run_data = []
        self.current_step = 0
        self.reach_terminal_state = False

        # Timing metrics (updated incrementally)
        self.average_action_time = 0.0
        self.average_reward_time = 0.0
        self.average_state_sampling_time = 0.0
        self.average_observation_time = 0.0
        self.average_belief_update_time = 0.0

    def run(self) -> History:
        """Execute the episode and return history with metrics."""
        self.logger.debug("Starting episode with %d steps", self.num_steps)

        while self._should_continue():
            self._execute_policy_step()

        self._log_completion()
        return self._build_history()

    def _should_continue(self) -> bool:
        """Check if episode should continue."""
        if self.current_step >= self.num_steps:
            return False

        if self.environment.is_terminal(self.state):
            self.reach_terminal_state = True
            self._add_terminal_step()
            return False

        return True

    def _execute_policy_step(self) -> None:
        """Execute one policy action selection and all resulting actions."""
        actions, policy_run_data = self._select_actions()
        self.policy_run_data.append(policy_run_data)

        for action in actions:
            if self.current_step >= self.num_steps:
                break
            self._execute_single_action(action)
            self.current_step += 1

    def _select_actions(self) -> Tuple[List, PolicyRunData]:
        """Select actions using policy and update timing."""
        start_time = time()
        actions, policy_run_data = self.policy.action(self.belief)
        elapsed = time() - start_time

        # Update average action time accounting for multiple actions
        step_count = self.current_step if self.current_step > 0 else 1
        total_actions = step_count - 1 + len(actions)
        self.average_action_time = (
            self.average_action_time * (step_count - 1) + elapsed
        ) / total_actions

        return actions, policy_run_data

    def _execute_single_action(self, action: Any) -> None:
        """Execute one action: compute reward, sample transition, update belief."""
        reward = self._compute_reward(action)
        next_state = self._sample_next_state(action)
        observation = self._sample_observation(next_state=next_state, action=action)

        self._record_step(action, next_state, observation, reward)
        self._update_belief(action, observation, next_state)

        self.state = next_state

    def _compute_reward(self, action: Any) -> float:
        """Compute reward and update timing metric."""
        start_time = time()
        reward = self.environment.reward(self.state, action)
        elapsed = time() - start_time

        step = self.current_step + 1
        self.average_reward_time = (self.average_reward_time * self.current_step + elapsed) / step

        return reward

    def _sample_next_state(self, action: Any) -> Any:
        """Sample next state and update timing metric."""
        start_time = time()
        next_state = self.environment.state_transition_model(self.state, action).sample()[0]
        elapsed = time() - start_time

        step = self.current_step + 1
        self.average_state_sampling_time = (
            self.average_state_sampling_time * self.current_step + elapsed
        ) / step

        return next_state

    def _sample_observation(self, next_state: Any, action: Any) -> Any:
        """Sample observation and update timing metric."""
        start_time = time()
        observation = self.environment.observation_model(next_state, action).sample()[0]
        elapsed = time() - start_time

        step = self.current_step + 1
        self.average_observation_time = (
            self.average_observation_time * self.current_step + elapsed
        ) / step

        return observation

    def _update_belief(self, action: Any, observation: Any, next_state: Any) -> None:
        """Update belief and update timing metric."""
        start_time = time()
        self.belief = self.belief.update(
            action=action,
            observation=observation,
            pomdp=self.environment,
            state=next_state,
        )
        elapsed = time() - start_time

        step = self.current_step + 1
        self.average_belief_update_time = (
            self.average_belief_update_time * self.current_step + elapsed
        ) / step

    def _record_step(self, action: Any, next_state: Any, observation: Any, reward: float) -> None:
        """Record step data in history."""
        step_data = StepData(
            state=self.state,
            action=action,
            next_state=next_state,
            observation=observation,
            reward=reward,
            belief=self.belief,
        )
        self.history.append(step_data)

    def _add_terminal_step(self) -> None:
        """Add terminal step to history."""
        terminal_step = StepData(
            state=self.state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=self.belief,
        )
        self.history.append(terminal_step)

    def _log_completion(self) -> None:
        """Log episode completion with timing metrics."""
        self.logger.debug(
            "Episode completed with average times: action=%.4fs, "
            "reward=%.4fs, state_sampling=%.4fs, "
            "observation=%.4fs, belief_update=%.4fs",
            self.average_action_time,
            self.average_reward_time,
            self.average_state_sampling_time,
            self.average_observation_time,
            self.average_belief_update_time,
        )

    def _build_history(self) -> History:
        """Construct final History object."""
        return History(
            history=self.history,
            discount_factor=self.environment.discount_factor,
            average_state_sampling_time=self.average_state_sampling_time,
            average_action_time=self.average_action_time,
            average_observation_time=self.average_observation_time,
            average_belief_update_time=self.average_belief_update_time,
            average_reward_time=self.average_reward_time,
            actual_num_steps=self.current_step,
            reach_terminal_state=self.reach_terminal_state,
            policy_run_data=self.policy_run_data,
        )


def run_episode(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
    logger: Optional[Logger],
) -> History:
    """Run a single episode without caching and collect detailed performance metrics.

    This function executes a single episode of interaction between a policy and environment,
    collecting detailed timing information and performance metrics at each step. The episode
    runs until either the maximum number of steps is reached or a terminal state is encountered.

    The function tracks several timing metrics:
    - Action selection time: Time spent by the policy choosing actions
    - State sampling time: Time spent sampling next states from transition model
    - Observation time: Time spent generating observations
    - Belief update time: Time spent updating belief states
    - Reward computation time: Time spent computing rewards

    Args:
        environment: The POMDP environment to run the episode in
        policy: The policy to use for action selection
        initial_belief: Initial belief state for the episode
        num_steps: Maximum number of steps to run
        logger: Logger instance for debugging and information output

    Returns:
        History object containing:
        - Step-by-step episode data (states, actions, observations, rewards)
        - Average timing metrics for each component
        - Episode summary statistics (actual steps taken, terminal state reached)
        - Policy run data from action selection

    Raises:
        ValueError: If any input parameters are invalid (None, non-positive num_steps)
        TypeError: If any input parameters are of incorrect type

    Example:
        Running a single episode with POMCP on Tiger POMDP:

        >>> from POMDPPlanners.simulations.episodes import run_episode
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.logger import get_logger

        >>> # Create environment and policy
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> policy = POMCP(
        ...     environment=env,
        ...     discount_factor=0.95,
        ...     depth=5,                 # Reduced for testing
        ...     exploration_constant=1.0,
        ...     name="POMCP_Tiger",
        ...     n_simulations=10         # Reduced for testing
        ... )

        >>> # Set up initial belief and logger
        >>> initial_belief = get_initial_belief(env, n_particles=10)  # Reduced for testing
        >>> logger = get_logger("episode_runner", debug=False)  # Disable debug for testing

        >>> # Run episode
        >>> history = run_episode(
        ...     environment=env,
        ...     policy=policy,
        ...     initial_belief=initial_belief,
        ...     num_steps=3,             # Reduced for testing
        ...     logger=logger
        ... )

        >>> # Access results
        >>> isinstance(history.history, list)
        True
        >>> # Print history steps
        >>> for i, step in enumerate(history.history):
        ...     print(f"Step {i}: state={step.state}, action={step.action}, reward={step.reward}")  # doctest: +ELLIPSIS
        Step 0: state=..., action=..., reward=...
    """
    runner = EpisodeRunner(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
        logger=logger,
    )
    return runner.run()
