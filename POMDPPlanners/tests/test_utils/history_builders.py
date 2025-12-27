"""Helper functions for building test History objects.

This module provides utility functions to reduce code duplication when
constructing History objects in tests.
"""

from typing import List, Optional

from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData


def build_test_history(
    steps: List[StepData],
    reach_terminal: bool = False,
    discount_factor: float = 0.95,
    average_state_sampling_time: float = 0.0,
    average_action_time: float = 0.0,
    average_observation_time: float = 0.0,
    average_belief_update_time: float = 0.0,
    average_reward_time: float = 0.0,
    actual_num_steps: Optional[int] = None,
    policy_run_data: Optional[List[PolicyRunData]] = None,
) -> History:
    """Build a History object with standard test parameters.

    This helper function reduces code duplication when constructing History
    objects in tests by providing sensible defaults for timing parameters.

    Args:
        steps: List of StepData objects representing the episode history.
        reach_terminal: Whether the terminal state was reached. Defaults to False.
        discount_factor: Discount factor for the history. Defaults to 0.95.
        average_state_sampling_time: Average time for state sampling. Defaults to 0.0.
        average_action_time: Average time for action selection. Defaults to 0.0.
        average_observation_time: Average time for observation generation. Defaults to 0.0.
        average_belief_update_time: Average time for belief updates. Defaults to 0.0.
        average_reward_time: Average time for reward calculation. Defaults to 0.0.
        actual_num_steps: Actual number of steps in the history. If None, uses len(steps).
        policy_run_data: Policy run data for the history. If None, uses empty list.

    Returns:
        History object with standard timing values and provided steps.

    Example:
        >>> from POMDPPlanners.core.simulation import StepData
        >>> from POMDPPlanners.core.belief import WeightedParticleBelief
        >>> import numpy as np
        >>> belief = WeightedParticleBelief(particles=[0, 1], log_weights=np.array([0.0, -0.1]))
        >>> steps = [StepData(state=0, action=1, next_state=0, observation=2, reward=10.0, belief=belief)]
        >>> history = build_test_history(steps, reach_terminal=True)
        >>> history.reach_terminal_state
        True
        >>> history.actual_num_steps
        1
    """
    if actual_num_steps is None:
        actual_num_steps = len(steps)
    if policy_run_data is None:
        policy_run_data = []

    return History(
        history=steps,
        discount_factor=discount_factor,
        average_state_sampling_time=average_state_sampling_time,
        average_action_time=average_action_time,
        average_observation_time=average_observation_time,
        average_belief_update_time=average_belief_update_time,
        average_reward_time=average_reward_time,
        actual_num_steps=actual_num_steps,
        reach_terminal_state=reach_terminal,
        policy_run_data=policy_run_data,
    )
