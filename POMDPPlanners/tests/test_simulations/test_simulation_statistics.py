import random
from typing import List

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.simulations.simulation_statistics import (
    StandardMetrics,
    compute_statistics_environment_policy_pair,
    compute_statistics_environments_policies_comparison,
    get_metric_names_from_environment_policy_pair,
)

np.random.seed(42)
random.seed(42)


def create_test_history(rewards: List[float], discount_factor: float = 0.95) -> History:
    # Create a simple belief for testing
    def create_test_belief(state):
        return WeightedParticleBelief(
            particles=[state],
            log_weights=np.array([1.0]),  # Using non-zero log weight
            resampling=False,
        )

    steps = [
        StepData(
            state="tiger_left",
            action="listen",
            next_state="tiger_left",
            observation="hear_left",
            reward=r,
            belief=create_test_belief("tiger_left"),
        )
        for r in rewards
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
        policy_run_data=[
            PolicyRunData(
                info_variables=[
                    PolicyInfoVariable(name="test_var", value=1.0),
                    PolicyInfoVariable(name="test_var2", value=2.0),
                ]
            )
        ],
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
        confidence_interval_level=confidence_level,
    )

    # Verify basic statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, "name")
        assert hasattr(metric, "value")
        assert hasattr(metric, "lower_confidence_bound")
        assert hasattr(metric, "upper_confidence_bound")

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
        create_test_history(
            [1.0, 2.0, 3.0]
        ),  # Duplicate to satisfy confidence interval requirement
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    alpha = 0.1
    confidence_level = 0.95

    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=alpha,
        confidence_interval_level=confidence_level,
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, "name")
        assert hasattr(metric, "value")
        assert hasattr(metric, "lower_confidence_bound")
        assert hasattr(metric, "upper_confidence_bound")

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
        confidence_interval_level=confidence_level,
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, "name")
        assert hasattr(metric, "value")
        assert hasattr(metric, "lower_confidence_bound")
        assert hasattr(metric, "upper_confidence_bound")

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
        confidence_interval_level=confidence_level,
    )

    # Verify statistics
    assert len(stats) > 0
    for metric in stats:
        assert hasattr(metric, "name")
        assert hasattr(metric, "value")
        assert hasattr(metric, "lower_confidence_bound")
        assert hasattr(metric, "upper_confidence_bound")

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
            resampling=False,
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
                    belief=create_test_belief("tiger_left"),
                ),
                StepData(
                    state="tiger_left",
                    action="open_right",
                    next_state="tiger_right",
                    observation="hear_nothing",
                    reward=10.0,
                    belief=create_test_belief("tiger_left"),
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=2,
            reach_terminal_state=True,
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left"),
                ),
                StepData(
                    state="tiger_left",
                    action="open_right",
                    next_state="tiger_right",
                    observation="hear_nothing",
                    reward=10.0,
                    belief=create_test_belief("tiger_left"),
                ),
            ],
            discount_factor=0.95,
            average_state_sampling_time=0.001,
            average_action_time=0.002,
            average_observation_time=0.003,
            average_belief_update_time=0.004,
            average_reward_time=0.005,
            actual_num_steps=2,
            reach_terminal_state=True,
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
    ]

    # Compute statistics
    statistics = compute_statistics_environment_policy_pair(
        env=environment, histories=histories, alpha=0.1, confidence_interval_level=0.95
    )

    # Verify statistics
    assert len(statistics) > 0
    for metric in statistics:
        assert hasattr(metric, "name")
        assert hasattr(metric, "value")
        assert hasattr(metric, "lower_confidence_bound")
        assert hasattr(metric, "upper_confidence_bound")

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
            resampling=False,
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
                    belief=create_test_belief("tiger_left"),
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
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
        History(
            history=[
                StepData(
                    state="tiger_left",
                    action="listen",
                    next_state="tiger_left",
                    observation="hear_left",
                    reward=-1.0,
                    belief=create_test_belief("tiger_left"),
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
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
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
                    belief=create_test_belief("tiger_right"),
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
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
        History(
            history=[
                StepData(
                    state="tiger_right",
                    action="listen",
                    next_state="tiger_right",
                    observation="hear_right",
                    reward=-1.0,
                    belief=create_test_belief("tiger_right"),
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
            policy_run_data=[
                PolicyRunData(
                    info_variables=[
                        PolicyInfoVariable(name="test_var", value=1.0),
                        PolicyInfoVariable(name="test_var2", value=2.0),
                    ]
                )
            ],
        ),
    ]

    # Create histories dictionary
    histories = {
        "TigerPOMDP_1": {"Policy1": history1},
        "TigerPOMDP_2": {"Policy2": history2},
    }

    # Compute statistics
    statistics_df = compute_statistics_environments_policies_comparison(
        histories=histories,
        environments=[env1, env2],
        alpha=0.1,
        confidence_interval_level=0.95,
    )

    # Verify statistics DataFrame
    assert len(statistics_df) == 2  # One row per environment-policy pair
    assert "environment" in statistics_df.columns
    assert "policy" in statistics_df.columns
    assert "average_return" in statistics_df.columns
    assert "return_cvar" in statistics_df.columns
    assert "return_value_at_risk" in statistics_df.columns
    assert "policy_info_test_var" in statistics_df.columns
    assert "policy_info_test_var2" in statistics_df.columns
    assert "policy_info_test_var_ci_lower" in statistics_df.columns
    assert "policy_info_test_var_ci_upper" in statistics_df.columns
    assert "policy_info_test_var2_ci_lower" in statistics_df.columns
    assert "policy_info_test_var2_ci_upper" in statistics_df.columns


def test_get_metric_names_from_environment_policy_pair_basic():
    """Test get_metric_names_from_environment_policy_pair returns correct metric names.

    Purpose: Validates that get_metric_names_from_environment_policy_pair returns all expected metric names in correct order

    Given: TigerPOMDP environment and POMCP policy class
    When: get_metric_names_from_environment_policy_pair is called
    Then: Returns list containing environment metrics (success_rate, average_listens), policy info metrics (prefixed with policy_info_), and all standard metrics in correct order

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)
    metric_names = get_metric_names_from_environment_policy_pair(env, POMCP)

    # Verify all standard metrics are present
    for standard_metric in StandardMetrics:
        assert (
            standard_metric.value in metric_names
        ), f"Missing standard metric: {standard_metric.value}"

    # Verify environment-specific metrics are present
    assert "success_rate" in metric_names
    assert "average_listens" in metric_names

    # Verify policy info metrics are present with proper prefix
    pomcp_info_vars = POMCP.get_info_variable_names()
    for var_name in pomcp_info_vars:
        prefixed_name = f"policy_info_{var_name}"
        assert prefixed_name in metric_names, f"Missing policy info metric: {prefixed_name}"


def test_get_metric_names_from_environment_policy_pair_order():
    """Test get_metric_names_from_environment_policy_pair returns metrics in correct order.

    Purpose: Validates that metrics are returned in the exact order as compute_statistics_environment_policy_pair

    Given: TigerPOMDP environment and POMCP policy class
    When: get_metric_names_from_environment_policy_pair is called
    Then: Returns metrics in order: environment-specific, policy info, standard metrics

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)
    metric_names = get_metric_names_from_environment_policy_pair(env, POMCP)

    # Get expected counts for each section
    env_metrics = env.get_metric_names()
    policy_metrics = [f"policy_info_{name}" for name in POMCP.get_info_variable_names()]
    standard_metrics = [metric.value for metric in StandardMetrics]

    # Environment metrics should come first
    for i, env_metric in enumerate(env_metrics):
        assert (
            metric_names[i] == env_metric
        ), f"Environment metric {env_metric} not in correct position"

    # Policy info metrics should come next
    offset = len(env_metrics)
    for i, policy_metric in enumerate(policy_metrics):
        assert (
            metric_names[offset + i] == policy_metric
        ), f"Policy info metric {policy_metric} not in correct position"

    # Standard metrics should come last
    offset = len(env_metrics) + len(policy_metrics)
    for i, standard_metric in enumerate(standard_metrics):
        assert (
            metric_names[offset + i] == standard_metric
        ), f"Standard metric {standard_metric} not in correct position"


def test_get_metric_names_from_environment_policy_pair_policy_without_info_vars():
    """Test get_metric_names_from_environment_policy_pair with policy that has no info variables.

    Purpose: Validates correct handling of policies that don't produce info variables

    Given: TigerPOMDP environment and SparseSamplingDiscreteActionsPlanner policy class (no info variables)
    When: get_metric_names_from_environment_policy_pair is called
    Then: Returns environment metrics and standard metrics without any policy info metrics

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)
    metric_names = get_metric_names_from_environment_policy_pair(
        env, SparseSamplingDiscreteActionsPlanner
    )

    # Verify no policy_info_ prefixed metrics
    policy_info_metrics = [name for name in metric_names if name.startswith("policy_info_")]
    assert (
        len(policy_info_metrics) == 0
    ), "Should have no policy info metrics for SparseSamplingDiscreteActionsPlanner"

    # Verify standard metrics are present
    for standard_metric in StandardMetrics:
        assert standard_metric.value in metric_names

    # Verify environment metrics are present
    assert "success_rate" in metric_names
    assert "average_listens" in metric_names


def test_get_metric_names_from_environment_policy_pair_matches_actual_computation():
    """Test that get_metric_names structure matches actual compute_statistics_environment_policy_pair output.

    Purpose: Validates that get_metric_names_from_environment_policy_pair returns correct metric structure

    Given: TigerPOMDP environment and test histories with custom policy info variables (test_var, test_var2)
    When: Comparing metric structure from compute_statistics with expected structure
    Then: Environment metrics and standard metrics match, policy info metrics follow correct naming pattern

    Test type: integration
    """
    env = TigerPOMDP(discount_factor=0.95)

    # Create test histories with custom policy info variables (not POMCP's actual ones)
    histories = [
        create_test_history([1.0, 2.0, 3.0]),
        create_test_history([2.0, 3.0, 4.0]),
    ]

    # Compute actual statistics
    actual_metrics = compute_statistics_environment_policy_pair(
        env=env, histories=histories, alpha=0.1, confidence_interval_level=0.95
    )

    # Extract actual metric names
    actual_metric_names = [metric.name for metric in actual_metrics]

    # Verify structure matches expected pattern:
    # 1. Environment-specific metrics come first
    env_metric_names = env.get_metric_names()
    for i, env_metric in enumerate(env_metric_names):
        assert (
            actual_metric_names[i] == env_metric
        ), f"Environment metric {env_metric} not at expected position {i}"

    # 2. Policy info metrics come next (should have policy_info_ prefix)
    offset = len(env_metric_names)
    policy_info_found = []
    for name in actual_metric_names[offset:]:
        if name.startswith("policy_info_"):
            policy_info_found.append(name)
        else:
            break  # Stop when we hit non-policy-info metrics

    # In our test histories, we have test_var and test_var2
    assert (
        "policy_info_test_var" in policy_info_found or "policy_info_test_var2" in policy_info_found
    )

    # 3. Standard metrics come last
    standard_metric_values = [metric.value for metric in StandardMetrics]
    # Find where standard metrics start
    standard_start_idx = offset + len(policy_info_found)
    for i, standard_metric in enumerate(standard_metric_values):
        assert (
            actual_metric_names[standard_start_idx + i] == standard_metric
        ), f"Standard metric {standard_metric} not at expected position"


def test_get_metric_names_from_environment_policy_pair_multiple_environments():
    """Test get_metric_names_from_environment_policy_pair with different environments.

    Purpose: Validates that function correctly adapts to different environment metric sets

    Given: Multiple different environment types (TigerPOMDP with different configurations)
    When: get_metric_names_from_environment_policy_pair is called for each
    Then: Returns environment-specific metrics for each, but same policy info and standard metrics

    Test type: unit
    """
    env1 = TigerPOMDP(discount_factor=0.95, name="Tiger1")
    env2 = TigerPOMDP(discount_factor=0.99, name="Tiger2")

    metric_names1 = get_metric_names_from_environment_policy_pair(env1, POMCP)
    metric_names2 = get_metric_names_from_environment_policy_pair(env2, POMCP)

    # Both should have the same metrics since they're the same environment type
    assert metric_names1 == metric_names2

    # Verify they contain the expected TigerPOMDP metrics
    assert "success_rate" in metric_names1
    assert "average_listens" in metric_names1


def test_get_metric_names_from_environment_policy_pair_different_policies():
    """Test get_metric_names_from_environment_policy_pair with different policy classes.

    Purpose: Validates that function correctly adapts to different policy info variable sets

    Given: Same TigerPOMDP environment with POMCP (has info vars) vs SparseSamplingDiscreteActionsPlanner (no info vars)
    When: get_metric_names_from_environment_policy_pair is called for each policy
    Then: POMCP results include policy_info_ metrics, SparseSamplingDiscreteActionsPlanner results do not

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)

    pomcp_metrics = get_metric_names_from_environment_policy_pair(env, POMCP)
    sparse_metrics = get_metric_names_from_environment_policy_pair(
        env, SparseSamplingDiscreteActionsPlanner
    )

    # POMCP should have policy_info_ metrics
    pomcp_policy_info = [name for name in pomcp_metrics if name.startswith("policy_info_")]
    assert len(pomcp_policy_info) > 0, "POMCP should have policy info metrics"

    # SparseSamplingDiscreteActionsPlanner should not have policy_info_ metrics
    sparse_policy_info = [name for name in sparse_metrics if name.startswith("policy_info_")]
    assert (
        len(sparse_policy_info) == 0
    ), "SparseSamplingDiscreteActionsPlanner should have no policy info metrics"

    # Both should have the same environment and standard metrics
    pomcp_env_standard = [name for name in pomcp_metrics if not name.startswith("policy_info_")]
    sparse_env_standard = [name for name in sparse_metrics if not name.startswith("policy_info_")]
    assert pomcp_env_standard == sparse_env_standard


def test_get_metric_names_from_environment_policy_pair_consistency():
    """Test get_metric_names_from_environment_policy_pair returns consistent results.

    Purpose: Validates that function is deterministic and returns same results on repeated calls

    Given: TigerPOMDP environment and POMCP policy class
    When: get_metric_names_from_environment_policy_pair is called multiple times
    Then: Returns identical lists on each call

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)

    # Call multiple times
    result1 = get_metric_names_from_environment_policy_pair(env, POMCP)
    result2 = get_metric_names_from_environment_policy_pair(env, POMCP)
    result3 = get_metric_names_from_environment_policy_pair(env, POMCP)

    # All results should be identical
    assert result1 == result2 == result3


def test_metric_estimates_within_confidence_intervals():
    """Test that point estimates fall within their confidence intervals for all metrics.

    Purpose: Validates that every computed metric satisfies lower_confidence_bound <= value <= upper_confidence_bound

    Given: TigerPOMDP environment, 10 histories with varying reward sequences to produce
           non-degenerate confidence intervals, alpha=0.1, confidence_interval_level=0.95
    When: compute_statistics_environment_policy_pair processes the histories
    Then: For every returned MetricValue, value is between lower_confidence_bound and
          upper_confidence_bound (inclusive), covering average_return, return_cvar,
          return_value_at_risk, timing metrics, and policy info metrics

    Test type: unit
    """
    histories = [
        create_test_history([1.0, 2.0, 3.0]),
        create_test_history([4.0, 5.0, 6.0]),
        create_test_history([-1.0, -2.0, -3.0]),
        create_test_history([0.5, 1.5, 2.5]),
        create_test_history([10.0, -5.0, 3.0]),
        create_test_history([-4.0, 0.0, 4.0]),
        create_test_history([2.0, 2.0, 2.0]),
        create_test_history([7.0, -1.0, 1.0]),
        create_test_history([-2.0, -2.0, -2.0]),
        create_test_history([3.0, 3.0, 3.0]),
    ]

    environment = TigerPOMDP(discount_factor=0.95)
    stats = compute_statistics_environment_policy_pair(
        env=environment,
        histories=histories,
        alpha=0.1,
        confidence_interval_level=0.95,
    )

    tol = 1e-9
    for metric in stats:
        assert (
            metric.lower_confidence_bound - tol
            <= metric.value
            <= metric.upper_confidence_bound + tol
        ), (
            f"Metric '{metric.name}': value {metric.value} not within CI "
            f"[{metric.lower_confidence_bound}, {metric.upper_confidence_bound}]"
        )
