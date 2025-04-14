from typing import List

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.utils.statistics import cvar_estimator, confidence_interval

def compute_statistics(histories: List[History], alpha: float, confidence_interval_level: float = 0.95) -> dict:
    assert isinstance(histories, list)
    assert len(histories) > 0
    assert all(isinstance(h, History) for h in histories)
    
    return_samples = []
    average_state_sampling_time = []
    average_action_time = []
    average_observation_time = []
    average_belief_update_time = []
    average_reward_time = []
    
    for i, h in enumerate(histories):
        return_ = sum([h.history[j].reward * h.discount_factor ** j for j in range(len(h.history))])
        return_samples.append(return_)
        
        average_state_sampling_time.append(h.average_state_sampling_time)
        average_action_time.append(h.average_action_time)
        average_observation_time.append(h.average_observation_time)
        average_belief_update_time.append(h.average_belief_update_time)
        average_reward_time.append(h.average_reward_time)
        
    average_return = sum(return_samples) / len(return_samples)
    
    return_cvar = cvar_estimator(return_samples, alpha)
    return_value_at_risk = np.percentile(return_samples, (1 - alpha) * 100)
    
    return_confidence_interval = confidence_interval(data=return_samples, confidence=confidence_interval_level)
    average_return_confidence_interval = confidence_interval(data=return_samples, confidence=confidence_interval_level)
    average_state_sampling_time_confidence_interval = confidence_interval(data=average_state_sampling_time, confidence=confidence_interval_level)
    average_action_time_confidence_interval = confidence_interval(data=average_action_time, confidence=confidence_interval_level)
    average_observation_time_confidence_interval = confidence_interval(data=average_observation_time, confidence=confidence_interval_level)
    average_belief_update_time_confidence_interval = confidence_interval(data=average_belief_update_time, confidence=confidence_interval_level)
    average_reward_time_confidence_interval = confidence_interval(data=average_reward_time, confidence=confidence_interval_level)
    
    return {
        "average_return": (average_return, average_return_confidence_interval),
        "return_cvar": (return_cvar, return_confidence_interval),
        "return_value_at_risk": (return_value_at_risk, return_confidence_interval),
        "average_state_sampling_time": (np.mean(average_state_sampling_time), average_state_sampling_time_confidence_interval),
        "average_action_time": (np.mean(average_action_time), average_action_time_confidence_interval),
        "average_observation_time": (np.mean(average_observation_time), average_observation_time_confidence_interval),
        "average_belief_update_time": (np.mean(average_belief_update_time), average_belief_update_time_confidence_interval),
        "average_reward_time": (np.mean(average_reward_time), average_reward_time_confidence_interval),
    }
