import copy
from time import time
from typing import Optional
from logging import Logger

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    StepData,
)
from POMDPPlanners.utils.logger import get_logger


def run_episode(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
    logger: Logger,
) -> History:
    """Run a single episode without caching.
    
    Args:
        environment: The environment to run the episode in
        policy: The policy to use
        initial_belief: Initial belief state
        num_steps: Number of steps to run
        
    Returns:
        History object containing episode data
        
    Raises:
        ValueError: If any input parameters are invalid
        TypeError: If any input parameters are of incorrect type
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
    if not hasattr(environment, 'state_transition_model') or not hasattr(environment, 'observation_model'):
        raise ValueError("environment must implement state_transition_model and observation_model")
    if not hasattr(policy, 'action'):
        raise ValueError("policy must implement action method")
    if not hasattr(initial_belief, 'sample') or not hasattr(initial_belief, 'update'):
        raise ValueError("initial_belief must implement sample and update methods")

    if logger is None:
        logger = get_logger(
            name=f"episode.{environment.name}.{policy.name}",
            debug=logger.debug,
            output_dir=logger.output_dir
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
        actions, policy_run_data = policy.action(belief)
        actions_time = time() - actions_start_time
        
        average_action_time = (average_action_time * (i - 1) + actions_time) / (i - 1 + len(actions))

        for action in actions:
            reward_start_time = time()
            reward = environment.reward(state, action)
            reward_time = time() - reward_start_time
            average_reward_time = (average_reward_time * (i - 1) + reward_time) / i

            state_sampling_start_time = time()
            next_state = environment.state_transition_model(state, action).sample()[0]
            state_sampling_time = time() - state_sampling_start_time
            average_state_sampling_time = (
                (average_state_sampling_time * (i - 1) + state_sampling_time) / i
            )

            observation_start_time = time()
            observation = environment.observation_model(next_state, action).sample()[0]
            observation_time = time() - observation_start_time
            average_observation_time = (
                (average_observation_time * (i - 1) + observation_time) / i
            )

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
                action=action, observation=observation, pomdp=environment
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

    logger.debug(f"Episode completed with average times: action={average_action_time:.4f}s, "
                f"reward={average_reward_time:.4f}s, state_sampling={average_state_sampling_time:.4f}s, "
                f"observation={average_observation_time:.4f}s, belief_update={average_belief_update_time:.4f}s")

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
        policy_run_data=policy_run_data,
    )
