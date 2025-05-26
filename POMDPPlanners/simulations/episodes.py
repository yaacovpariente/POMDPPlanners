import copy
from time import time

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import (
    History,
    StepData,
)
from POMDPPlanners.simulations.simulations import validate_episode_inputs
from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


def run_episode(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    num_steps: int,
) -> History:
    """Run a single episode without caching.
    
    Args:
        environment: The environment to run the episode in
        policy: The policy to use
        initial_belief: Initial belief state
        num_steps: Number of steps to run
        
    Returns:
        History object containing episode data
    """
    logger.debug(f"Starting episode with {num_steps} steps")

    validate_episode_inputs(environment, policy, initial_belief, num_steps)

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
            next_state = environment.state_transition_model(state, action).sample()
            state_sampling_time = time() - state_sampling_start_time
            average_state_sampling_time = (
                average_state_sampling_time * (i - 1) + state_sampling_time
            ) / i

            observation_start_time = time()
            observation = environment.observation_model(next_state, action).sample()
            observation_time = time() - observation_start_time
            average_observation_time = (
                average_observation_time * (i - 1) + observation_time
            ) / i

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
