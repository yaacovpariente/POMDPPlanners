from typing import List
import copy
from time import time
from pathlib import Path
import mlflow
import json
import os

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.simulations.simulation_statistics import compute_statistics

def run_multiple_episodes(
    environment: Environment, 
    policy: Policy, 
    initial_belief: Belief, 
    discount_factor: float, 
    num_episodes: int, 
    num_steps: int
) -> List[History]:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    
    assert 1 >= discount_factor >= 0
    assert num_episodes > 0
    assert num_steps > 0
    
    histories = []
    for _ in range(num_episodes):
        history = run_episode(environment, policy, initial_belief, discount_factor, num_steps)
        histories.append(history)
    
    return histories

def run_episode(
    environment: Environment, 
    policy: Policy, 
    initial_belief: Belief, 
    discount_factor: float, 
    num_steps: int
) -> History:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    
    assert 1 >= discount_factor >= 0
    assert num_steps > 0
    
    average_state_sampling_time = 0.
    average_action_time = 0.
    average_observation_time = 0.
    average_belief_update_time = 0.
    average_reward_time = 0.
    
    belief = copy.deepcopy(initial_belief)
    state = belief.sample()
    
    history = []
    for i in range(1, num_steps + 1):
        action_start_time = time()
        action = policy.action(belief)
        action_time = time() - action_start_time
        average_action_time = (average_action_time * (i - 1) + action_time) / i

        reward_start_time = time()
        reward = environment.reward(state, action)
        reward_time = time() - reward_start_time
        average_reward_time = (average_reward_time * (i - 1) + reward_time) / i
        
        state_sampling_start_time = time()
        next_state = environment.state_transition_model(state, action).sample()
        state_sampling_time = time() - state_sampling_start_time
        average_state_sampling_time = (average_state_sampling_time * (i - 1) + state_sampling_time) / i

        observation_start_time = time()
        observation = environment.observation_model(next_state, action).sample()
        observation_time = time() - observation_start_time
        average_observation_time = (average_observation_time * (i - 1) + observation_time) / i
        
        history.append(
            StepData(
                state=state, 
                action=action, 
                next_state=next_state, 
                observation=observation, 
                reward=reward
            )
        )
        
        belief_update_start_time = time()
        belief = belief.update(action=action, observation=observation, pomdp=environment)
        belief_update_time = time() - belief_update_start_time
        average_belief_update_time = (average_belief_update_time * (i - 1) + belief_update_time) / i

        state = next_state
        
    return History(
        history=history, 
        discount_factor=discount_factor, 
        average_state_sampling_time=average_state_sampling_time, 
        average_action_time=average_action_time, 
        average_observation_time=average_observation_time, 
        average_belief_update_time=average_belief_update_time,
        average_reward_time=average_reward_time
    )

def simulation(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    cache_dir_path: Path,
    confidence_interval_level: float = 0.95
) -> dict:
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(confidence_interval_level, float)
    
    assert 1 >= confidence_interval_level >= 0
    assert num_episodes > 0
    assert num_steps > 0
    
    histories = run_multiple_episodes(
        environment=environment, 
        policy=policy, 
        initial_belief=initial_belief, 
        discount_factor=discount_factor, 
        num_episodes=num_episodes, 
        num_steps=num_steps
    )
    
    statistics = compute_statistics(
        histories=histories, 
        alpha=alpha, 
        confidence_interval_level=confidence_interval_level
    )
    
    return histories, statistics

def simulation_with_mlflow(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    discount_factor: float,
    num_episodes: int,
    num_steps: int,
    alpha: float,
    cache_dir_path: Path,
    confidence_interval_level: float = 0.95
):
    assert isinstance(environment, Environment)
    assert isinstance(policy, Policy)
    assert isinstance(initial_belief, Belief)
    assert isinstance(discount_factor, float)
    assert isinstance(num_episodes, int)
    assert isinstance(num_steps, int)
    assert isinstance(cache_dir_path, Path)
    assert isinstance(confidence_interval_level, float)
    
    assert 1 >= confidence_interval_level >= 0
    assert num_episodes > 0
    assert num_steps > 0

    histories, statistics = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        discount_factor=discount_factor,
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=alpha,
        cache_dir_path=cache_dir_path,
        confidence_interval_level=confidence_interval_level
    )
    
    
    
    
