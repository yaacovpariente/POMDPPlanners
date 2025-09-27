from typing import Dict, List, Union

import numpy as np
import pandas as pd

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.utils.statistics import (
    confidence_interval,
    cvar_confidence_interval,
    cvar_estimator,
    quantile_confidence_interval,
)


def compute_statistics_environment_policy_pair(
    env: Environment,
    histories: List[History],
    alpha: float,
    confidence_interval_level: float = 0.95,
) -> List[MetricValue]:
    """Compute comprehensive statistics and metrics for a single environment-policy pair.

    This function analyzes simulation histories to compute a wide range of performance
    metrics including returns, risk measures, timing statistics, and custom environment
    metrics. All metrics include confidence intervals for statistical significance testing.

    Computed Metrics:
    - Average return: Mean discounted return across episodes
    - CVaR (Conditional Value at Risk): Expected return in worst α fraction of cases
    - VaR (Value at Risk): α-quantile of return distribution
    - Timing metrics: Average times for each simulation component
    - Episode statistics: Step counts and terminal state rates
    - Custom environment metrics: Domain-specific performance measures

    Args:
        env: The POMDP environment used in the simulation
        histories: List of episode histories from simulation runs
        alpha: Risk level for CVaR and VaR computation (0 < α < 1)
        confidence_interval_level: Confidence level for statistical intervals (0 < level < 1)

    Returns:
        List of MetricValue objects, each containing:
        - name: Metric identifier
        - value: Point estimate of the metric
        - lower_confidence_bound: Lower bound of confidence interval
        - upper_confidence_bound: Upper bound of confidence interval

    Raises:
        TypeError: If histories is not a list or contains non-History objects
        ValueError: If histories is empty

    Example:
        Computing statistics for POMCP on Tiger POMDP::

            from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            from POMDPPlanners.core.belief import get_initial_belief
            from POMDPPlanners.simulations.episodes import run_episode
            from POMDPPlanners.utils.logger import get_logger

            # Set up environment and policy
            env = TigerPOMDP(discount_factor=0.95)
            policy = POMCP(
                environment=env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=1.0,
                name="POMCP",
                n_simulations=1000
            )

            # Run multiple episodes
            histories = []
            initial_belief = get_initial_belief(env, n_particles=1000)
            logger = get_logger("stats_example")

            for episode in range(50):
                history = run_episode(
                    environment=env,
                    policy=policy,
                    initial_belief=initial_belief,
                    num_steps=20,
                    logger=logger
                )
                histories.append(history)

            # Compute comprehensive statistics
            metrics = compute_statistics_environment_policy_pair(
                env=env,
                histories=histories,
                alpha=0.05,  # 5% risk level
                confidence_interval_level=0.95  # 95% confidence intervals
            )

            # Analyze results
            for metric in metrics:
                print(f"{metric.name}: {metric.value:.3f} "
                      f"[{metric.lower_confidence_bound:.3f}, {metric.upper_confidence_bound:.3f}]")

            # Find specific metrics
            avg_return = next(m for m in metrics if m.name == "average_return")
            cvar = next(m for m in metrics if m.name == "return_cvar")
            action_time = next(m for m in metrics if m.name == "average_action_time")

            print(f"\\nKey Performance Indicators:")
            print(f"Average Return: {avg_return.value:.3f} ± {(avg_return.upper_confidence_bound - avg_return.lower_confidence_bound)/2:.3f}")
            print(f"CVaR (5%): {cvar.value:.3f}")
            print(f"Planning Time: {action_time.value:.4f} seconds/step")
    """
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
        return_ = sum(
            h.history[j].reward * h.discount_factor**j  # type: ignore
            for j in range(len(h.history))
            if h.history[j].reward is not None
        )
        return_samples.append(return_)

        average_state_sampling_time.append(h.average_state_sampling_time)
        average_action_time.append(h.average_action_time)
        average_observation_time.append(h.average_observation_time)
        average_belief_update_time.append(h.average_belief_update_time)
        average_reward_time.append(h.average_reward_time)
        average_actual_num_steps.append(h.actual_num_steps)
        average_reach_terminal_state.append(1 if h.reach_terminal_state else 0)

        # Collect policy info variables
        for info_var in h.policy_run_data.info_variables:  # type: ignore
            if info_var.name not in policy_info_variables:
                policy_info_variables[info_var.name] = []
            policy_info_variables[info_var.name].append(float(info_var.value))

    average_return = sum(return_samples) / len(return_samples)

    return_cvar = cvar_estimator(np.array(return_samples), alpha)
    return_value_at_risk = np.percentile(return_samples, (1 - alpha) * 100)

    cvar_return_confidence_interval = cvar_confidence_interval(
        data=return_samples, alpha=alpha, delta=1 - confidence_interval_level
    )
    return_value_at_risk_confidence_interval = quantile_confidence_interval(
        data=return_samples, alpha=1 - alpha, conf_level=1 - confidence_interval_level
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
                value=float(mean_value),
                lower_confidence_bound=float(ci[0]),
                upper_confidence_bound=float(ci[1]),
            )
        )

    custom_environment_metrics = env.compute_metrics(histories=histories)
    return (
        custom_environment_metrics
        + policy_info_metrics
        + [
            MetricValue(
                name="average_return",
                value=average_return,
                lower_confidence_bound=average_return_confidence_interval[0],
                upper_confidence_bound=average_return_confidence_interval[1],
            ),
            MetricValue(
                name="return_cvar",
                value=float(return_cvar),
                lower_confidence_bound=cvar_return_confidence_interval[0],
                upper_confidence_bound=cvar_return_confidence_interval[1],
            ),
            MetricValue(
                name="return_value_at_risk",
                value=float(return_value_at_risk),
                lower_confidence_bound=return_value_at_risk_confidence_interval[0],
                upper_confidence_bound=return_value_at_risk_confidence_interval[1],
            ),
            MetricValue(
                name="average_state_sampling_time",
                value=float(np.mean(average_state_sampling_time)),
                lower_confidence_bound=average_state_sampling_time_confidence_interval[0],
                upper_confidence_bound=average_state_sampling_time_confidence_interval[1],
            ),
            MetricValue(
                name="average_action_time",
                value=float(np.mean(average_action_time)),
                lower_confidence_bound=average_action_time_confidence_interval[0],
                upper_confidence_bound=average_action_time_confidence_interval[1],
            ),
            MetricValue(
                name="average_observation_time",
                value=float(np.mean(average_observation_time)),
                lower_confidence_bound=average_observation_time_confidence_interval[0],
                upper_confidence_bound=average_observation_time_confidence_interval[1],
            ),
            MetricValue(
                name="average_belief_update_time",
                value=float(np.mean(average_belief_update_time)),
                lower_confidence_bound=average_belief_update_time_confidence_interval[0],
                upper_confidence_bound=average_belief_update_time_confidence_interval[1],
            ),
            MetricValue(
                name="average_reward_time",
                value=float(np.mean(average_reward_time)),
                lower_confidence_bound=average_reward_time_confidence_interval[0],
                upper_confidence_bound=average_reward_time_confidence_interval[1],
            ),
            MetricValue(
                name="average_actual_num_steps",
                value=float(np.mean(average_actual_num_steps)),
                lower_confidence_bound=average_actual_num_steps_confidence_interval[0],
                upper_confidence_bound=average_actual_num_steps_confidence_interval[1],
            ),
            MetricValue(
                name="average_reach_terminal_state",
                value=float(np.mean(average_reach_terminal_state)),
                lower_confidence_bound=average_reach_terminal_state_confidence_interval[0],
                upper_confidence_bound=average_reach_terminal_state_confidence_interval[1],
            ),
        ]
    )


def compute_statistics_environments_policies_comparison(
    histories: Dict[str, Dict[str, List[History]]],
    environments: List[Environment],
    alpha: float,
    confidence_interval_level: float = 0.95,
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
                confidence_interval_level=confidence_interval_level,
            )

            # Create row data with environment and policy names
            row_data: Dict[str, Union[str, float]] = {
                "environment": env_name,
                "policy": policy_name,
            }

            # Add all metrics to row data
            for metric in statistics:
                row_data[metric.name] = metric.value
                row_data[f"{metric.name}_ci_lower"] = metric.lower_confidence_bound
                row_data[f"{metric.name}_ci_upper"] = metric.upper_confidence_bound

            all_statistics.append(row_data)

    # Create DataFrame from all statistics
    df = pd.DataFrame(all_statistics)

    return df


def metrics_dict_to_dataframe(
    metrics_dict: Dict[str, Dict[str, List[MetricValue]]],
) -> pd.DataFrame:
    """Convert a nested dictionary of metrics into a structured DataFrame for analysis.

    This function transforms the hierarchical metrics structure returned by simulation
    runs into a flat DataFrame format suitable for statistical analysis, visualization,
    and reporting. Each row represents one environment-policy combination.

    The resulting DataFrame includes:
    - Environment and policy identification columns
    - Point estimates for all computed metrics
    - Confidence interval bounds for each metric (lower and upper)
    - Automatically handles missing metrics with appropriate defaults

    Args:
        metrics_dict: Nested dictionary with structure:
            {environment_name: {policy_name: [MetricValue, ...], ...}, ...}
            where MetricValue objects contain name, value, and confidence bounds

    Returns:
        DataFrame with columns:
        - 'environment': Environment name identifier
        - 'policy': Policy name identifier
        - For each metric 'X': 'X', 'X_ci_lower', 'X_ci_upper' columns

    Raises:
        TypeError: If metrics_dict structure is invalid
        ValueError: If metrics_dict is empty

    Example:
        Converting simulation results to DataFrame for analysis::

            from POMDPPlanners.simulations.simulation_statistics import (
                compute_statistics_environment_policy_pair, metrics_dict_to_dataframe
            )
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            # ... other imports and setup ...

            # Assume we have simulation results for multiple environment-policy pairs
            metrics_dict = {
                'TigerPOMDP': {
                    'POMCP': pomcp_metrics,  # List[MetricValue] from compute_statistics_...
                    'SparseSampling': sparse_metrics
                },
                'CartPolePOMDP': {
                    'POMCP': cartpole_pomcp_metrics
                }
            }

            # Convert to DataFrame
            results_df = metrics_dict_to_dataframe(metrics_dict)

            # Analyze results
            print("DataFrame shape:", results_df.shape)
            print("Columns:", results_df.columns.tolist())
            print("\\nEnvironments:", results_df['environment'].unique())
            print("Policies:", results_df['policy'].unique())

            # Compare average returns across policies
            avg_return_comparison = results_df[['environment', 'policy', 'average_return',
                                             'average_return_ci_lower', 'average_return_ci_upper']]
            print("\\nAverage Return Comparison:")
            print(avg_return_comparison)

            # Find best performing policy per environment
            for env in results_df['environment'].unique():
                env_data = results_df[results_df['environment'] == env]
                best_policy = env_data.loc[env_data['average_return'].idxmax(), 'policy']
                best_return = env_data['average_return'].max()
                print(f"{env}: Best policy is {best_policy} with return {best_return:.3f}")

            # Statistical significance testing
            import scipy.stats as stats

            tiger_data = results_df[results_df['environment'] == 'TigerPOMDP']
            if len(tiger_data) >= 2:
                policy1_stats = tiger_data.iloc[0]
                policy2_stats = tiger_data.iloc[1]

                # Check if confidence intervals overlap
                p1_ci = (policy1_stats['average_return_ci_lower'], policy1_stats['average_return_ci_upper'])
                p2_ci = (policy2_stats['average_return_ci_lower'], policy2_stats['average_return_ci_upper'])

                overlap = not (p1_ci[1] < p2_ci[0] or p2_ci[1] < p1_ci[0])
                print(f"\\nConfidence intervals overlap: {overlap}")
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
        for metrics_list in policy_metrics.values()
        for metric in metrics_list
    ):
        raise TypeError("All metrics must be MetricValue instances")

    # List to store all statistics
    all_statistics = []

    # Process each environment-policy pair
    for env_name, policy_metrics_dict in metrics_dict.items():
        for policy_name, metrics in policy_metrics_dict.items():
            # Create row data with environment and policy names
            row_data: Dict[str, Union[str, float]] = {
                "environment": env_name,
                "policy": policy_name,
            }

            # Add all metrics to row data
            for metric in metrics:
                row_data[metric.name] = metric.value
                row_data[f"{metric.name}_ci_lower"] = metric.lower_confidence_bound
                row_data[f"{metric.name}_ci_upper"] = metric.upper_confidence_bound

            all_statistics.append(row_data)

    # Create DataFrame from all statistics
    df = pd.DataFrame(all_statistics)

    return df
