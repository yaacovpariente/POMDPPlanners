import copy
from logging import Logger
from time import time
from typing import Optional

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.utils.logger import get_logger


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
    # Input validation
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

    # Validate that environment and policy are compatible
    if not hasattr(environment, "state_transition_model") or not hasattr(
        environment, "observation_model"
    ):
        raise ValueError("environment must implement state_transition_model and observation_model")
    if not hasattr(policy, "action"):
        raise ValueError("policy must implement action method")
    if not hasattr(initial_belief, "sample") or not hasattr(initial_belief, "update"):
        raise ValueError("initial_belief must implement sample and update methods")

    if logger is None:
        logger = get_logger(
            name=f"episode.{environment.name}.{policy.name}",
        )

    logger.debug(f"Starting episode with {num_steps} steps")

    # Initialize timing metrics with deterministic values
    average_state_sampling_time = 0.0
    average_action_time = 0.0
    average_observation_time = 0.0
    average_belief_update_time = 0.0
    average_reward_time = 0.0
    actual_num_steps = 0
    reach_terminal_state = False

    belief = copy.deepcopy(initial_belief)
    state = belief.sample()

    history = []
    policy_run_data_list = []
    i = 1
    while i <= num_steps:
        if environment.is_terminal(state=state):
            reach_terminal_state = True
            history.append(
                StepData(
                    state=state,
                    action=None,
                    next_state=None,
                    observation=None,
                    reward=None,
                    belief=belief,
                )
            )
            break

        actions_start_time = time()
        actions, policy_run_data = policy.action(belief)  # TODO: bug here
        actions_time = time() - actions_start_time
        policy_run_data_list.append(policy_run_data)

        average_action_time = (average_action_time * (i - 1) + actions_time) / (
            i - 1 + len(actions)
        )

        for action in actions:
            reward_start_time = time()
            reward = environment.reward(state, action)
            reward_time = time() - reward_start_time
            average_reward_time = (average_reward_time * (i - 1) + reward_time) / i

            state_sampling_start_time = time()
            next_state = environment.state_transition_model(state, action).sample()[0]
            state_sampling_time = time() - state_sampling_start_time
            average_state_sampling_time = (
                average_state_sampling_time * (i - 1) + state_sampling_time
            ) / i

            observation_start_time = time()
            observation = environment.observation_model(next_state, action).sample()[0]
            observation_time = time() - observation_start_time
            average_observation_time = (average_observation_time * (i - 1) + observation_time) / i

            history.append(
                StepData(
                    state=state,
                    action=action,
                    next_state=next_state,
                    observation=observation,
                    reward=reward,
                    belief=belief,
                )
            )

            belief_update_start_time = time()
            belief = belief.update(
                action=action,
                observation=observation,
                pomdp=environment,
                state=next_state,
            )
            belief_update_time = time() - belief_update_start_time
            average_belief_update_time = (
                average_belief_update_time * (i - 1) + belief_update_time
            ) / i

            actual_num_steps += 1
            state = next_state
            i += 1

            if i > num_steps:
                break

    logger.debug(
        f"Episode completed with average times: action={average_action_time:.4f}s, "
        f"reward={average_reward_time:.4f}s, state_sampling={average_state_sampling_time:.4f}s, "
        f"observation={average_observation_time:.4f}s, belief_update={average_belief_update_time:.4f}s"
    )

    return History(
        history=history,
        discount_factor=environment.discount_factor,
        average_state_sampling_time=average_state_sampling_time,
        average_action_time=average_action_time,
        average_observation_time=average_observation_time,
        average_belief_update_time=average_belief_update_time,
        average_reward_time=average_reward_time,
        actual_num_steps=actual_num_steps,
        reach_terminal_state=reach_terminal_state,
        policy_run_data=policy_run_data_list,
    )
