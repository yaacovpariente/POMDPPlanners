from typing import List, Dict

import numpy as np
import pandas as pd

from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.utils.statistics import cvar_estimator, confidence_interval, cvar_confidence_interval, quantile_confidence_interval
from POMDPPlanners.core.environment import Environment

def compute_statistics_environment_policy_pair(
    env: Environment, histories: List[History], alpha: float, confidence_interval_level: float = 0.95
) -> List[MetricValue]:
    if not isinstance(histories, list):
        raise TypeError("histories must be a list")
    if len(histories) == 0:
        raise ValueError("histories must not be empty")
    if not all(isinstance(h, History) for h in histories):
        raise TypeError("All elements of histories must be History instances")

    return_samples = []
    average_state_sampling_time = []
    average_action_time = []
    average_observation_time = []
    average_belief_update_time = []
    average_reward_time = []
    average_actual_num_steps = []
    average_reach_terminal_state = []

    # Dictionary to store policy info variables by name
    policy_info_variables: Dict[str, List[float]] = {}

    for i, h in enumerate(histories):
        return_ = sum(h.history[j].reward * h.discount_factor**j for j in range(len(h.history)) if h.history[j].reward is not None)
        return_samples.append(return_)

        average_state_sampling_time.append(h.average_state_sampling_time)
        average_action_time.append(h.average_action_time)
        average_observation_time.append(h.average_observation_time)
        average_belief_update_time.append(h.average_belief_update_time)
        average_reward_time.append(h.average_reward_time)
        average_actual_num_steps.append(h.actual_num_steps)
        average_reach_terminal_state.append(1 if h.reach_terminal_state else 0)

        # Collect policy info variables
        for info_var in h.policy_run_data.info_variables:
            if info_var.name not in policy_info_variables:
                policy_info_variables[info_var.name] = []
            policy_info_variables[info_var.name].append(float(info_var.value))

    average_return = sum(return_samples) / len(return_samples)

    return_cvar = cvar_estimator(return_samples, alpha)
    return_value_at_risk = np.percentile(return_samples, (1 - alpha) * 100)

    cvar_return_confidence_interval = cvar_confidence_interval(
        data=return_samples, alpha=alpha, delta=1-confidence_interval_level
    )
    return_value_at_risk_confidence_interval = quantile_confidence_interval(
        data=return_samples, alpha=1-alpha, conf_level=1-confidence_interval_level
    )
    average_return_confidence_interval = confidence_interval(
        data=return_samples, confidence=confidence_interval_level
    )
    average_state_sampling_time_confidence_interval = confidence_interval(
        data=average_state_sampling_time, confidence=confidence_interval_level
    )
    average_action_time_confidence_interval = confidence_interval(
        data=average_action_time, confidence=confidence_interval_level
    )
    average_observation_time_confidence_interval = confidence_interval(
        data=average_observation_time, confidence=confidence_interval_level
    )
    average_belief_update_time_confidence_interval = confidence_interval(
        data=average_belief_update_time, confidence=confidence_interval_level
    )
    average_reward_time_confidence_interval = confidence_interval(
        data=average_reward_time, confidence=confidence_interval_level
    )
    average_actual_num_steps_confidence_interval = confidence_interval(
        data=average_actual_num_steps, confidence=confidence_interval_level
    )
    average_reach_terminal_state_confidence_interval = confidence_interval(
        data=average_reach_terminal_state, confidence=confidence_interval_level
    )

    # Create metrics for policy info variables
    policy_info_metrics = []
    for name, values in policy_info_variables.items():
        mean_value = np.mean(values)
        ci = confidence_interval(data=values, confidence=confidence_interval_level)
        policy_info_metrics.append(
            MetricValue(
                name=f"policy_info_{name}",
                value=mean_value,
                lower_confidence_bound=ci[0],
                upper_confidence_bound=ci[1],
            )
        )

    custom_environment_metrics = env.compute_metrics(histories=histories)
    return custom_environment_metrics + policy_info_metrics + [
        MetricValue(
            name="average_return",
            value=average_return,
            lower_confidence_bound=average_return_confidence_interval[0],
            upper_confidence_bound=average_return_confidence_interval[1],
        ),
        MetricValue(
            name="return_cvar",
            value=return_cvar,
            lower_confidence_bound=cvar_return_confidence_interval[0],
            upper_confidence_bound=cvar_return_confidence_interval[1],
        ),
        MetricValue(
            name="return_value_at_risk",
            value=return_value_at_risk,
            lower_confidence_bound=return_value_at_risk_confidence_interval[0],
            upper_confidence_bound=return_value_at_risk_confidence_interval[1],
        ),
        MetricValue(
            name="average_state_sampling_time",
            value=np.mean(average_state_sampling_time),
            lower_confidence_bound=average_state_sampling_time_confidence_interval[0],
            upper_confidence_bound=average_state_sampling_time_confidence_interval[1],
        ),
        MetricValue(
            name="average_action_time",
            value=np.mean(average_action_time),
            lower_confidence_bound=average_action_time_confidence_interval[0],
            upper_confidence_bound=average_action_time_confidence_interval[1],
        ),
        MetricValue(
            name="average_observation_time",
            value=np.mean(average_observation_time),
            lower_confidence_bound=average_observation_time_confidence_interval[0],
            upper_confidence_bound=average_observation_time_confidence_interval[1],
        ),
        MetricValue(
            name="average_belief_update_time",
            value=np.mean(average_belief_update_time),
            lower_confidence_bound=average_belief_update_time_confidence_interval[0],
            upper_confidence_bound=average_belief_update_time_confidence_interval[1],
        ),
        MetricValue(
            name="average_reward_time",
            value=np.mean(average_reward_time),
            lower_confidence_bound=average_reward_time_confidence_interval[0],
            upper_confidence_bound=average_reward_time_confidence_interval[1],
        ),
        MetricValue(
            name="average_actual_num_steps",
            value=np.mean(average_actual_num_steps),
            lower_confidence_bound=average_actual_num_steps_confidence_interval[0],
            upper_confidence_bound=average_actual_num_steps_confidence_interval[1],
        ),
        MetricValue(
            name="average_reach_terminal_state",
            value=np.mean(average_reach_terminal_state),
            lower_confidence_bound=average_reach_terminal_state_confidence_interval[0],
            upper_confidence_bound=average_reach_terminal_state_confidence_interval[1],
        ),
    ]

def compute_statistics_environments_policies_comparison(
    histories: Dict[str, Dict[str, List[History]]],
    environments: List[Environment],
    alpha: float,
    confidence_interval_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute statistics for multiple environments and policies, returning results in a DataFrame.
    
    Args:
        histories: Dictionary mapping environment names to dictionaries mapping policy names to lists of histories
        alpha: Alpha value for statistics computation
        confidence_interval_level: Confidence level for statistics
        
    Returns:
        DataFrame where each row represents an environment-policy pair and columns are metrics
    """
    if not isinstance(histories, dict):
        raise TypeError("histories must be a dict")
    if len(histories) == 0:
        raise ValueError("histories must not be empty")
    if not all(isinstance(env_histories, dict) for env_histories in histories.values()):
        raise TypeError("All values in histories must be dicts")
    if not all(
        isinstance(policy_histories, list) and all(isinstance(h, History) for h in policy_histories)
        for env_histories in histories.values()
        for policy_histories in env_histories.values()
    ):
        raise TypeError("All policy histories must be lists of History instances")

    # List to store all statistics
    all_statistics = []
    envs_dict = {env.name: env for env in environments}
    # Process each environment-policy pair
    for env_name, policy_histories_dict in histories.items():
        for policy_name, policy_histories in policy_histories_dict.items():
            # Compute statistics for this environment-policy pair
            statistics = compute_statistics_environment_policy_pair(
                env=envs_dict[env_name],
                histories=policy_histories,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level
            )
            
            # Create row data with environment and policy names
            row_data = {
                'environment': env_name,
                'policy': policy_name
            }
            
            # Add all metrics to row data
            for metric in statistics:
                row_data[metric.name] = metric.value
                row_data[f'{metric.name}_ci_lower'] = metric.lower_confidence_bound
                row_data[f'{metric.name}_ci_upper'] = metric.upper_confidence_bound
            
            all_statistics.append(row_data)
    
    # Create DataFrame from all statistics
    df = pd.DataFrame(all_statistics)
    
    return df

def metrics_dict_to_dataframe(
    metrics_dict: Dict[str, Dict[str, List[MetricValue]]]
) -> pd.DataFrame:
    """
    Convert a dictionary of metrics into a DataFrame with environment and policy columns.
    
    Args:
        metrics_dict: Dictionary mapping environment names to dictionaries mapping policy names to lists of MetricValue objects
        
    Returns:
        DataFrame where each row represents an environment-policy pair and columns are metrics with confidence intervals
    """
    if not isinstance(metrics_dict, dict):
        raise TypeError("metrics_dict must be a dict")
    if len(metrics_dict) == 0:
        raise ValueError("metrics_dict must not be empty")
    if not all(isinstance(policy_metrics, dict) for policy_metrics in metrics_dict.values()):
        raise TypeError("All values in metrics_dict must be dicts")
    if not all(
        isinstance(metric, MetricValue)
        for policy_metrics in metrics_dict.values()
        for metric in policy_metrics.values()
    ):
        raise TypeError("All metrics must be MetricValue instances")

    # List to store all statistics
    all_statistics = []

    # Process each environment-policy pair
    for env_name, policy_metrics_dict in metrics_dict.items():
        for policy_name, metrics in policy_metrics_dict.items():
            # Create row data with environment and policy names
            row_data = {
                'environment': env_name,
                'policy': policy_name
            }
            
            # Add all metrics to row data
            for metric in metrics:
                row_data[metric.name] = metric.value
                row_data[f'{metric.name}_ci_lower'] = metric.lower_confidence_bound
                row_data[f'{metric.name}_ci_upper'] = metric.upper_confidence_bound
            
            all_statistics.append(row_data)
    
    # Create DataFrame from all statistics
    df = pd.DataFrame(all_statistics)
    
    return df

