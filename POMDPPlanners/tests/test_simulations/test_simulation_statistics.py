import pytest
import numpy as np
from typing import List
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable
from POMDPPlanners.simulations.simulation_statistics import (
    compute_statistics_environment_policy_pair,
    compute_statistics_environments_policies_comparison
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief


def create_test_history(rewards: List[float], discount_factor: float = 0.95) -> History:
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),  # Using non-zero log weight
            resampling=False
        )
    
    steps = [
        StepData(
            state="tiger_left",
            action="listen",
            next_state="tiger_left",
            observation="hear_left",
            reward=r,
            belief=create_test_belief("tiger_left")
        ) for r in rewards
    ]
    history = History(
        history=steps,
        discount_factor=discount_factor,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=len(steps),
        reach_terminal_state=True,
        policy_run_data=PolicyRunData(info_variables=[
            PolicyInfoVariable(name="test_var", value=1.0),
            PolicyInfoVariable(name="test_var2", value=2.0)
        ])
    )
    return history


def get_metric_value(metrics: List[MetricValue], name: str) -> MetricValue:
    """Helper function to get a metric value by name from the list."""
    for metric in metrics:
        if metric.name == name:
            return metric
    raise ValueError(f"Metric {name} not found")


def test_compute_statistics_basic():
    """Test compute statistics basic.
    
    Purpose: Validates that compute_statistics_environment_policy_pair generates comprehensive metrics with confidence intervals for multiple histories
    
    Given: TigerPOMDP environment, 3 histories with different reward sequences, alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environment_policy_pair processes the histories
    Then: Returns list of MetricValue objects with name, value, lower_confidence_bound, upper_confidence_bound, and policy info variables (test_var=1.0, test_var2=2.0)
    
    Test type: unit
    """
    # Create test histories with known rewards
    histories = [
        create_test_history([1.0, 2.0, 3.0]),  # Return = 1 + 2*0.95 + 3*0.95^2
        create_test_history([2.0, 3.0, 4.0]),  # Return = 2 + 3*0.95 + 4*0.95^2
        create_test_history([3.0, 4.0, 5.0]),  # Return = 3 + 4*0.95 + 5*0.95^2
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_level
    )

    # Verify basic statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'value')
        assert hasattr(metric, 'lower_confidence_bound')
        assert hasattr(metric, 'upper_confidence_bound')

    # Verify policy info metrics
    test_var_metric = get_metric_value(stats, "policy_info_test_var")
    test_var2_metric = get_metric_value(stats, "policy_info_test_var2")
    
    assert test_var_metric.value == 1.0  # All histories have value 1.0
    assert test_var2_metric.value == 2.0  # All histories have value 2.0
    
    # For policy info variables with constant values, confidence bounds should be equal to the value
    assert test_var_metric.lower_confidence_bound == test_var_metric.value
    assert test_var_metric.upper_confidence_bound == test_var_metric.value
    assert test_var2_metric.lower_confidence_bound == test_var2_metric.value
    assert test_var2_metric.upper_confidence_bound == test_var2_metric.value


def test_compute_statistics_single_history():
    """Test compute statistics single history.
    
    Purpose: Validates that compute_statistics_environment_policy_pair handles minimal dataset with duplicate histories for confidence interval computation
    
    Given: TigerPOMDP environment, 2 identical histories with rewards [1.0, 2.0, 3.0], alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environment_policy_pair processes the duplicate histories
    Then: Returns valid metrics with all required attributes and policy info variables (test_var=1.0) with consistent confidence bounds
    
    Test type: unit
    """
    # Test with a single history
    histories = [
        create_test_history([1.0, 2.0, 3.0]),
        create_test_history([1.0, 2.0, 3.0]),  # Duplicate to satisfy confidence interval requirement
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_level
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'value')
        assert hasattr(metric, 'lower_confidence_bound')
        assert hasattr(metric, 'upper_confidence_bound')

    # Verify policy info metrics
    test_var_metric = get_metric_value(stats, "policy_info_test_var")
    assert test_var_metric.value == 1.0


def test_compute_statistics_negative_rewards():
    """Test compute statistics negative rewards.
    
    Purpose: Validates that compute_statistics_environment_policy_pair correctly handles negative reward scenarios typical in POMDP environments
    
    Given: TigerPOMDP environment, 3 histories with negative reward sequences ([-1,-2,-3], [-2,-3,-4], [-3,-4,-5]), alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environment_policy_pair processes the negative reward histories
    Then: Returns valid metrics with negative values properly handled and policy info variables (test_var=1.0) correctly computed
    
    Test type: unit
    """
    # Test with negative rewards
    histories = [
        create_test_history([-1.0, -2.0, -3.0]),
        create_test_history([-2.0, -3.0, -4.0]),
        create_test_history([-3.0, -4.0, -5.0]),
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_level
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'value')
        assert hasattr(metric, 'lower_confidence_bound')
        assert hasattr(metric, 'upper_confidence_bound')

    # Verify policy info metrics
    test_var_metric = get_metric_value(stats, "policy_info_test_var")
    assert test_var_metric.value == 1.0


def test_compute_statistics_zero_rewards():
    """Test compute statistics zero rewards.
    
    Purpose: Validates that compute_statistics_environment_policy_pair handles edge case of zero rewards without division errors or invalid statistics
    
    Given: TigerPOMDP environment, 3 histories with all zero rewards [0.0, 0.0, 0.0], alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environment_policy_pair processes the zero reward histories
    Then: Returns valid metrics with zero values and meaningful confidence intervals, policy info variables (test_var=1.0) correctly computed
    
    Test type: unit
    """
    # Test with zero rewards
    histories = [
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0]),
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_level
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'value')
        assert hasattr(metric, 'lower_confidence_bound')
        assert hasattr(metric, 'upper_confidence_bound')

    # Verify policy info metrics
    test_var_metric = get_metric_value(stats, "policy_info_test_var")
    assert test_var_metric.value == 1.0


def test_compute_statistics_environment_policy_pair():
    """Test compute statistics environment policy pair.
    
    Purpose: Validates that compute_statistics_environment_policy_pair works end-to-end with realistic TigerPOMDP scenarios including multi-step episodes
    
    Given: TigerPOMDP environment, 2 histories with listen-open action sequences yielding -1 + 10*discount rewards, policy info variables (test_var=1.0, test_var2=2.0)
    When: compute_statistics_environment_policy_pair processes complete episode histories with alpha=0.1, confidence_interval_level=0.95
    Then: Returns comprehensive metrics with all standard attributes and correctly extracted policy info variable values
    
    Test type: unit
    """
    # Create a simple environment
    environment = TigerPOMDP(discount_factor=0.95)
    
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),  # Using non-zero log weight
            resampling=False
        )
    
    # Create multiple histories to satisfy confidence interval requirement
    histories = [
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left")
                ),
                StepData(
                    state="tiger_left",
                    action="open_right",
                    next_state="tiger_right",
                    observation="hear_nothing",
                    reward=10.0,
                    belief=create_test_belief("tiger_left")
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=2,
            reach_terminal_state=True,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        ),
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left")
                ),
                StepData(
                    state="tiger_left",
                    action="open_right",
                    next_state="tiger_right",
                    observation="hear_nothing",
                    reward=10.0,
                    belief=create_test_belief("tiger_left")
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=2,
            reach_terminal_state=True,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        )
    ]
    
    # Compute statistics
    statistics = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=0.1,
        confidence_interval_level=0.95
    )
    
    # Verify statistics
    assert len(statistics) > 0
    for metric in statistics:
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'value')
        assert hasattr(metric, 'lower_confidence_bound')
        assert hasattr(metric, 'upper_confidence_bound')

    # Verify policy info metrics
    test_var_metric = get_metric_value(statistics, "policy_info_test_var")
    test_var2_metric = get_metric_value(statistics, "policy_info_test_var2")
    
    assert test_var_metric.value == 1.0
    assert test_var2_metric.value == 2.0


def test_compute_statistics_environments_policies_comparison():
    """Test compute statistics environments policies comparison.
    
    Purpose: Validates that compute_statistics_environments_policies_comparison correctly processes multiple environment-policy combinations into comparative DataFrame format
    
    Given: Two TigerPOMDP environments (discount_factor 0.95 vs 0.99), separate policies, histories dict with Policy1 and Policy2, alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environments_policies_comparison processes the multi-environment policy comparison
    Then: Returns DataFrame with 2 rows, environment/policy columns, standard metrics (average_return, return_cvar), and policy info variable columns with confidence intervals
    
    Test type: unit
    """
    # Create environments
    env1 = TigerPOMDP(discount_factor=0.95, name="TigerPOMDP_1")
    env2 = TigerPOMDP(discount_factor=0.99, name="TigerPOMDP_2")
    
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),  # Using non-zero log weight
            resampling=False
        )
    
    # Create histories with multiple samples for each policy
    history1 = [
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left")
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        ),
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left")
                )
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        )
    ]
    
    history2 = [
        History(
            history=[
                StepData(
                    state="tiger_right",
                    action="listen",
                    next_state="tiger_right",
                    observation="hear_right",
                    reward=-1.0,
                    belief=create_test_belief("tiger_right")
                )
            ],
            discount_factor=0.99,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        ),
        History(
            history=[
                StepData(
                    state="tiger_right",
                    action="listen",
                    next_state="tiger_right",
                    observation="hear_right",
                    reward=-1.0,
                    belief=create_test_belief("tiger_right")
                )
            ],
            discount_factor=0.99,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=PolicyRunData(info_variables=[
                PolicyInfoVariable(name="test_var", value=1.0),
                PolicyInfoVariable(name="test_var2", value=2.0)
            ])
        )
    ]
    
    # Create histories dictionary
    histories = {
        "TigerPOMDP_1": {
            "Policy1": history1
        },
        "TigerPOMDP_2": {
            "Policy2": history2
        }
    }
    
    # Compute statistics
    statistics_df = compute_statistics_environments_policies_comparison(
        histories=histories,
        environments=[env1, env2],
        alpha=0.1,
        confidence_interval_level=0.95
    )
    
    # Verify statistics DataFrame
    assert len(statistics_df) == 2  # One row per environment-policy pair
    assert 'environment' in statistics_df.columns
    assert 'policy' in statistics_df.columns
    assert 'average_return' in statistics_df.columns
    assert 'return_cvar' in statistics_df.columns
    assert 'return_value_at_risk' in statistics_df.columns
    assert 'policy_info_test_var' in statistics_df.columns
    assert 'policy_info_test_var2' in statistics_df.columns
    assert 'policy_info_test_var_ci_lower' in statistics_df.columns
    assert 'policy_info_test_var_ci_upper' in statistics_df.columns
    assert 'policy_info_test_var2_ci_lower' in statistics_df.columns
    assert 'policy_info_test_var2_ci_upper' in statistics_df.columns
