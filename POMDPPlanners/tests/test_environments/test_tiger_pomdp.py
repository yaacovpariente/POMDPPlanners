# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.tiger_pomdp import STATES, TigerPOMDP
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.tests.test_utils.history_builders import build_test_history
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_history_returns_bounded,
    verify_metric_sanity,
    verify_return_shift_linearity,
)

np.random.seed(42)
random.seed(42)


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


def test_initialization(tiger_pomdp):
    """Test TigerPOMDP environment initialization and basic attributes.

    Purpose: Validates that TigerPOMDP initializes correctly with expected states, actions, and observations

    Given: A TigerPOMDP environment with discount_factor=0.95
    When: Environment instance is created
    Then: Environment has correct attributes: discount_factor=0.95, states (tiger_left, tiger_right), actions (listen, open_left, open_right), observations (hear_left, hear_right, hear_nothing)

    Test type: unit
    """
    assert tiger_pomdp.discount_factor == 0.95
    assert set(tiger_pomdp.states) == {"tiger_left", "tiger_right"}
    assert set(tiger_pomdp.actions) == {"listen", "open_left", "open_right"}
    assert set(tiger_pomdp.observations) == {"hear_left", "hear_right", "hear_nothing"}


def test_get_actions(tiger_pomdp):
    """Test action space retrieval.

    Purpose: Validates that TigerPOMDP get_actions method returns the correct action space

    Given: A TigerPOMDP environment with default configuration
    When: get_actions method is called
    Then: Returns list of 3 actions: listen, open_left, open_right, representing all available actions in the Tiger environment

    Test type: unit
    """
    actions = tiger_pomdp.get_actions()
    assert len(actions) == 3
    assert set(actions) == {"listen", "open_left", "open_right"}


def test_initial_state_distribution(tiger_pomdp):
    """Test initial state distribution sampling.

    Purpose: Validates that TigerPOMDP initial state distribution works correctly

    Given: A TigerPOMDP environment
    When: Initial state distribution is sampled 100 times
    Then: All samples are valid states (tiger_left or tiger_right) and both states appear, demonstrating proper distribution

    Test type: unit
    """
    dist = tiger_pomdp.initial_state_dist()
    # Sample multiple times to ensure we get both states
    samples = dist.sample(n_samples=100)
    assert all(s in tiger_pomdp.states for s in samples)
    assert len(set(samples)) == 2  # Should get both states


def test_initial_observation_distribution(tiger_pomdp):
    """Test initial observation distribution sampling.

    Purpose: Validates that TigerPOMDP initial observation distribution works correctly

    Given: A TigerPOMDP environment
    When: Initial observation distribution is sampled 10 times
    Then: All samples return "hear_nothing", confirming that initial observations are always "hear_nothing"

    Test type: unit
    """
    dist = tiger_pomdp.initial_observation_dist()
    samples = dist.sample(n_samples=10)
    assert all(s == "hear_nothing" for s in samples)


def test_state_transition_listen(tiger_pomdp):
    """Test state transition for listen action.

    Purpose: Validates that TigerPOMDP listen action doesn't change the state

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right)
    When: env.sample_next_state is called for the listen action repeatedly
    Then: All samples return the same state as the input state, confirming that listening doesn't change the tiger's position

    Test type: unit
    """
    # Listening shouldn't change the state
    for state in tiger_pomdp.states:
        samples = [tiger_pomdp.sample_next_state(state=state, action="listen") for _ in range(10)]
        assert all(s == state for s in samples)


def test_state_transition_open_door(tiger_pomdp):
    """Test state transition for open door actions.

    Purpose: Validates that TigerPOMDP open door actions randomly place tiger behind either door

    Given: A TigerPOMDP environment starting from tiger_left state and both open actions (open_left, open_right)
    When: env.sample_next_state is called for the open actions 100 times
    Then: All samples are valid states and both states appear, demonstrating random placement after opening doors

    Test type: unit
    """
    # Opening a door should randomly place tiger behind either door
    for action in ["open_left", "open_right"]:
        samples = [
            tiger_pomdp.sample_next_state(state="tiger_left", action=action) for _ in range(100)
        ]
        assert all(s in tiger_pomdp.states for s in samples)
        assert len(set(samples)) == 2  # Should get both states


def test_observation_model_listen(tiger_pomdp):
    """Test observation model for listen action.

    Purpose: Validates that TigerPOMDP listen action provides mostly correct observations with some noise

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right)
    When: env.sample_observation is called for the listen action 100 times
    Then: Most observations are correct (hear_left for tiger_left, hear_right for tiger_right) with approximately 85% accuracy

    Test type: unit
    """
    # Test listen action with both states
    for state in tiger_pomdp.states:
        samples = [
            tiger_pomdp.sample_observation(next_state=state, action="listen") for _ in range(100)
        ]
        assert all(s in tiger_pomdp.observations for s in samples)

        # Should mostly get correct observation
        expected_obs = "hear_left" if state == "tiger_left" else "hear_right"
        correct_count = sum(1 for s in samples if s == expected_obs)
        assert correct_count > 70  # Should be around 85% correct


def test_observation_model_open_door(tiger_pomdp):
    """Test observation model for open door actions.

    Purpose: Validates that TigerPOMDP open door actions always provide "hear_nothing" observations

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right) with both open actions (open_left, open_right)
    When: env.sample_observation is called for the open actions 10 times
    Then: All samples return "hear_nothing", confirming that opening doors provides no auditory information

    Test type: unit
    """
    # Opening a door should always give 'hear_nothing'
    for state in tiger_pomdp.states:
        for action in ["open_left", "open_right"]:
            samples = [
                tiger_pomdp.sample_observation(next_state=state, action=action) for _ in range(10)
            ]
            assert all(s == "hear_nothing" for s in samples)


def test_reward_func_listen(tiger_pomdp):
    """Test reward function for listen action.

    Purpose: Validates that TigerPOMDP listen action always provides -1.0 reward

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right)
    When: Reward function is called with listen action for each state
    Then: All calls return -1.0, confirming that listening always incurs a small penalty

    Test type: unit
    """
    # Listening should always give -1 reward
    for state in tiger_pomdp.states:
        assert tiger_pomdp.reward(state, "listen") == -1.0


def test_reward_func_open_door(tiger_pomdp):
    """Test reward function for open door actions.

    Purpose: Validates that TigerPOMDP open door actions provide correct rewards based on tiger location

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right)
    When: Reward function is called with open actions for each state
    Then: Returns -100.0 for opening door with tiger, 10.0 for opening door with treasure, demonstrating correct reward structure

    Test type: unit
    """
    # Test opening doors with tiger
    assert tiger_pomdp.reward("tiger_left", "open_left") == -100.0
    assert tiger_pomdp.reward("tiger_right", "open_right") == -100.0

    # Test opening doors with treasure
    assert tiger_pomdp.reward("tiger_right", "open_left") == 10.0
    assert tiger_pomdp.reward("tiger_left", "open_right") == 10.0


def test_is_terminal(tiger_pomdp):
    """Test terminal state detection.

    Purpose: Validates that TigerPOMDP correctly identifies terminal states

    Given: A TigerPOMDP environment and all possible states
    When: is_terminal method is called for each state
    Then: All states return False, confirming that TigerPOMDP currently has no terminal states

    Test type: unit
    """
    # Currently always returns False
    for state in tiger_pomdp.states:
        assert not tiger_pomdp.is_terminal(state)


def test_reward_range(tiger_pomdp):
    """Test that reward range is correctly set.

    Purpose: Validates that TigerPOMDP has the correct reward range parameters

    Given: A TigerPOMDP environment with default configuration
    When: Environment reward_range attribute is checked
    Then: Returns (-100.0, 10.0) representing the minimum (opening door with tiger) and maximum (opening door with treasure) rewards

    Test type: unit
    """
    assert tiger_pomdp.reward_range == (-100.0, 10.0)

    # Verify the actual rewards match the range
    min_reward = min(
        tiger_pomdp.reward("tiger_left", "open_left"),  # Opening door with tiger: -100.0
        tiger_pomdp.reward("tiger_right", "open_right"),  # Opening door with tiger: -100.0
        tiger_pomdp.reward("tiger_left", "listen"),  # Listening: -1.0
        tiger_pomdp.reward("tiger_right", "listen"),  # Listening: -1.0
    )
    max_reward = max(
        tiger_pomdp.reward("tiger_left", "open_right"),  # Opening door with treasure: 10.0
        tiger_pomdp.reward("tiger_right", "open_left"),  # Opening door with treasure: 10.0
        tiger_pomdp.reward("tiger_left", "listen"),  # Listening: -1.0
        tiger_pomdp.reward("tiger_right", "listen"),  # Listening: -1.0
    )

    assert min_reward == tiger_pomdp.reward_range[0]
    assert max_reward == tiger_pomdp.reward_range[1]


class TestTigerPOMDPConfigId:
    """Test suite for TigerPOMDP config_id functionality.

    Purpose: Validates that TigerPOMDP config_id generates proper identifiers and changes with configuration differences

    Given: Various TigerPOMDP configurations with different parameters
    When: Config IDs are generated and compared
    Then: Config IDs behave correctly: consistent for same config, different for different configs, proper format, and deterministic

    Test type: configuration
    """

    def test_config_id_consistency(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is consistent for identical configurations.

        Purpose: Validates that TigerPOMDP config_id generates consistent identifiers for identical configurations

        Given: Two TigerPOMDP environments with identical discount_factor=0.95
        When: Config IDs are generated for both environments
        Then: Both environments have identical config_ids, demonstrating consistency for same configuration

        Test type: configuration
        """
        other_env = TigerPOMDP(discount_factor=0.95)
        assert tiger_pomdp.config_id == other_env.config_id

    def test_config_id_different_discount_factor(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different discount factors.

        Purpose: Validates that TigerPOMDP config_id generates different identifiers for different discount factors

        Given: Two TigerPOMDP environments with different discount factors (0.95 vs 0.8)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different configurations

        Test type: configuration
        """
        other_env = TigerPOMDP(discount_factor=0.8)
        assert tiger_pomdp.config_id != other_env.config_id

    def test_config_id_different_states(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different states.

        Purpose: Validates that TigerPOMDP config_id generates different identifiers for different state configurations

        Given: Two TigerPOMDP environments with different states (2 states vs 3 states including tiger_middle)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different state configurations

        Test type: configuration
        """
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.states = [
            "tiger_left",
            "tiger_right",
            "tiger_middle",
        ]  # Different states
        assert tiger_pomdp.config_id != other_env.config_id

    def test_config_id_different_actions(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different actions.

        Purpose: Validates that TigerPOMDP config_id generates different identifiers for different action configurations

        Given: Two TigerPOMDP environments with different actions (3 actions vs 4 actions including wait)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different action configurations

        Test type: configuration
        """
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.actions = [
            "listen",
            "open_left",
            "open_right",
            "wait",
        ]  # Different actions
        assert tiger_pomdp.config_id != other_env.config_id

    def test_config_id_different_observations(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different observations.

        Purpose: Validates that TigerPOMDP config_id generates different identifiers for different observation configurations

        Given: Two TigerPOMDP environments with different observations (3 observations vs 4 observations including hear_both)
        When: Config IDs are generated for both environments
        Then: Environments have different config_ids, demonstrating uniqueness for different observation configurations

        Test type: configuration
        """
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.observations = [
            "hear_left",
            "hear_right",
            "hear_nothing",
            "hear_both",
        ]  # Different observations
        assert tiger_pomdp.config_id != other_env.config_id

    def test_config_id_format(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is a valid SHA-256 hash.

        Purpose: Validates that TigerPOMDP config_id generates properly formatted SHA-256 hash identifiers

        Given: A TigerPOMDP environment with specific configuration
        When: Config ID is generated for the environment
        Then: Returns a 64-character string containing only valid hexadecimal characters (0-9, a-f)

        Test type: configuration
        """
        config_id = tiger_pomdp.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in "0123456789abcdef" for c in config_id)  # Valid hex characters

    def test_config_id_deterministic(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is deterministic (same input always produces same output).

        Purpose: Validates that TigerPOMDP config_id generates deterministic identifiers for identical configurations

        Given: A TigerPOMDP environment with specific configuration
        When: Config ID is generated multiple times for the same environment
        Then: All generated config_ids are identical, demonstrating deterministic behavior

        Test type: unit
        """
        config_id1 = tiger_pomdp.config_id
        config_id2 = tiger_pomdp.config_id
        assert config_id1 == config_id2


class TestTigerPOMDPMetrics:
    """Test suite for TigerPOMDP compute_metrics functionality."""

    def test_compute_metrics_perfect_agent(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for a perfect agent that always opens the correct door.

        Purpose: Validates that TigerPOMDP compute_metrics correctly calculates success rate and average listens for perfect performance

        Given: A TigerPOMDP environment and 10 perfect agent histories (listen 3 times then open correct door)
        When: compute_metrics is called with the perfect agent histories
        Then: Returns 100% success rate and average of 3.0 listens, confirming perfect agent metrics

        Test type: unit
        """

        # Create histories where agent always opens correct door
        histories = []
        for _ in range(10):
            # Randomly choose initial state
            state = np.random.choice(["tiger_left", "tiger_right"])
            # Agent listens a few times then opens correct door
            steps = []
            for _ in range(3):  # Listen 3 times
                steps.append(
                    StepData(
                        state=state,
                        action="listen",
                        next_state=state,
                        observation="hear_nothing",
                        reward=-1.0,
                        belief=Mock(spec=Belief),  # Mock belief for testing
                    )
                )
            # Open correct door
            correct_action = "open_right" if state == "tiger_left" else "open_left"
            steps.append(
                StepData(
                    state=state,
                    action=correct_action,
                    next_state=state,
                    observation="hear_nothing",
                    reward=10.0 if correct_action == "open_right" else -100.0,
                    belief=Mock(spec=Belief),  # Mock belief for testing
                )
            )
            histories.append(
                build_test_history(
                    steps=steps,
                    actual_num_steps=4,
                    reach_terminal=True,
                    policy_run_data=[PolicyRunData(info_variables=[])],
                )
            )

        # Compute metrics for the perfect agent
        metrics = tiger_pomdp.compute_metrics(histories)

        # Should have 100% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 1.0

        # Should have average of 3 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 3.0

    def test_compute_metrics_failing_agent(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for an agent that always opens the wrong door.

        Purpose: Validates that TigerPOMDP compute_metrics correctly calculates success rate and average listens for failing performance

        Given: A TigerPOMDP environment and 10 failing agent histories (listen 2 times then open wrong door)
        When: compute_metrics is called with the failing agent histories
        Then: Returns 0% success rate and average of 2.0 listens, confirming failing agent metrics

        Test type: unit
        """

        # Create histories where agent always opens wrong door
        histories = []
        for _ in range(10):
            state = np.random.choice(["tiger_left", "tiger_right"])
            steps = []
            for _ in range(2):  # Listen 2 times
                steps.append(
                    StepData(
                        state=state,
                        action="listen",
                        next_state=state,
                        observation="hear_nothing",
                        reward=-1.0,
                        belief=Mock(spec=Belief),  # Mock belief for testing
                    )
                )
            # Open wrong door
            wrong_action = "open_left" if state == "tiger_left" else "open_right"
            steps.append(
                StepData(
                    state=state,
                    action=wrong_action,
                    next_state=state,
                    observation="hear_nothing",
                    reward=-100.0 if wrong_action == "open_left" else 10.0,
                    belief=Mock(spec=Belief),  # Mock belief for testing
                )
            )
            histories.append(
                build_test_history(
                    steps=steps,
                    actual_num_steps=3,
                    reach_terminal=True,
                    policy_run_data=[PolicyRunData(info_variables=[])],
                )
            )

        # Compute metrics for the failing agent
        metrics = tiger_pomdp.compute_metrics(histories)

        # Should have 0% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.0

        # Should have average of 2 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 2.0

    def test_compute_metrics_mixed_performance(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for an agent with mixed performance.

        Purpose: Validates that TigerPOMDP compute_metrics correctly calculates success rate and average listens for mixed performance

        Given: A TigerPOMDP environment and 10 mixed performance histories (varying listens, alternating correct/incorrect actions)
        When: compute_metrics is called with the mixed performance histories
        Then: Returns 50% success rate and average of 1.9 listens, confirming mixed performance metrics

        Test type: unit
        """

        # Create histories with mixed success/failure
        histories = []
        for i in range(10):
            state = np.random.choice(["tiger_left", "tiger_right"])
            steps = []
            for _ in range(i % 3 + 1):  # Varying number of listens
                steps.append(
                    StepData(
                        state=state,
                        action="listen",
                        next_state=state,
                        observation="hear_nothing",
                        reward=-1.0,
                        belief=Mock(spec=Belief),  # Mock belief for testing
                    )
                )
            # Alternate between correct and incorrect actions
            action = "open_right" if (i % 2 == 0) == (state == "tiger_left") else "open_left"
            steps.append(
                StepData(
                    state=state,
                    action=action,
                    next_state=state,
                    observation="hear_nothing",
                    reward=10.0 if action == "open_right" else -100.0,
                    belief=Mock(spec=Belief),  # Mock belief for testing
                )
            )
            histories.append(
                build_test_history(
                    steps=steps,
                    reach_terminal=True,
                    policy_run_data=[PolicyRunData(info_variables=[])],
                )
            )

        # Compute metrics for the mixed performance agent
        metrics = tiger_pomdp.compute_metrics(histories)

        # Should have 50% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.5

        # Should have average of 1.9 listens (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 1.9

    def test_compute_metrics_empty_histories(self, tiger_pomdp: TigerPOMDP):
        """Test metrics with empty history list.

        Purpose: Validates that TigerPOMDP compute_metrics handles empty history lists correctly

        Given: A TigerPOMDP environment and empty history list []
        When: compute_metrics is called with empty histories
        Then: Returns 0% success rate and 0 listens, confirming proper handling of empty input

        Test type: unit
        """
        metrics = tiger_pomdp.compute_metrics([])

        # Should have 0% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.0

        # Should have 0 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 0.0

    def test_compute_metrics_history_with_zero_steps(self, tiger_pomdp: TigerPOMDP):
        """Regression: compute_metrics must not IndexError when an episode has zero steps.

        Purpose: Validates that TigerPOMDP compute_metrics tolerates History objects whose
        ``history`` step-list is empty (e.g., episode terminated before any action was taken)

        Given: A TigerPOMDP environment and a list containing one History with history=[]
        When: compute_metrics is called
        Then: The call returns metrics without raising IndexError; the zero-step episode is
            treated as a non-success contributing 0 listens

        Test type: unit
        """
        empty_history = History(
            history=[],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=0,
            reach_terminal_state=False,
            policy_run_data=[],
        )
        metrics = tiger_pomdp.compute_metrics([empty_history])

        success_metric = next(m for m in metrics if m.name == "success_rate")
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert success_metric.value == 0.0
        assert listens_metric.value == 0.0


def test_metrics_confidence_intervals(tiger_pomdp):
    """Test that metric values fall within their confidence intervals.

    Purpose: Validates that TigerPOMDP compute_metrics returns metric values within their confidence bounds

    Given: A TigerPOMDP environment and diverse performance histories
    When: compute_metrics is called with the histories
    Then: Each metric value falls within its lower_confidence_bound and upper_confidence_bound

    Test type: unit
    """

    # Create diverse performance histories with different outcomes
    histories = []
    np.random.seed(42)  # For reproducible test

    for i in range(20):  # Use enough episodes for meaningful statistics
        state = np.random.choice(["tiger_left", "tiger_right"])
        steps = []

        # Vary number of listens (0-5)
        num_listens = i % 6
        for _ in range(num_listens):
            steps.append(
                StepData(
                    state=state,
                    action="listen",
                    next_state=state,
                    observation="hear_nothing",
                    reward=-1.0,
                    belief=Mock(spec=Belief),
                )
            )

        # Vary success rate: 70% success rate
        if i % 10 < 7:  # 70% success
            correct_action = "open_right" if state == "tiger_left" else "open_left"
            reward = 10.0
        else:  # 30% failure
            correct_action = "open_left" if state == "tiger_left" else "open_right"
            reward = -100.0

        steps.append(
            StepData(
                state=state,
                action=correct_action,
                next_state=state,
                observation="hear_nothing",
                reward=reward,
                belief=Mock(spec=Belief),
            )
        )

        histories.append(
            build_test_history(
                steps=steps,
                reach_terminal=True,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        )

    # Compute metrics
    metrics = tiger_pomdp.compute_metrics(histories)

    # Use generic confidence interval verification
    verify_metrics_within_confidence_intervals(metrics)
    verify_metric_sanity(metrics, histories, tiger_pomdp)
    verify_history_returns_bounded(histories, tiger_pomdp)
    verify_return_shift_linearity(histories, tiger_pomdp, shift=1.5)


def test_metric_name_consistency(tiger_pomdp):
    """Test that declared metric names match actual produced metrics.

    Purpose: Validates that TigerPOMDP get_metric_names() returns exactly the metric names produced by compute_metrics()

    Given: A TigerPOMDP environment and sample episode histories
    When: get_metric_names() is called and compute_metrics() is executed
    Then: The metric names declared match exactly the metric names produced (no missing or extra metrics)

    Test type: unit
    """
    from POMDPPlanners.tests.test_metric_consistency_utils import (
        verify_environment_metric_consistency,
    )

    # Create sample histories
    steps = [
        StepData(
            state="tiger_left",
            action="listen",
            next_state="tiger_left",
            observation="hear_left",
            reward=-1,
            belief=Mock(spec=Belief),
        )
        for _ in range(3)
    ]

    histories = [
        build_test_history(
            steps=steps,
            reach_terminal=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
    ]

    # Verify consistency using reusable utility function
    verify_environment_metric_consistency(tiger_pomdp, histories)


# ---------------------------------------------------------------------------
# Batch-method equivalence tests
# ---------------------------------------------------------------------------


class TestTigerBatchMethods:
    """Equivalence tests for TigerPOMDP vectorised batch API."""

    N = 16

    def test_reward_batch_listen_all_minus_one(self, tiger_pomdp: TigerPOMDP):
        """reward_batch with listen returns -1.0 for every state.

        Purpose: Validates reward_batch for listen action returns correct scalar reward vectorised

        Given: 16 particle states (mixed tiger_left / tiger_right) and action=listen
        When: reward_batch is called
        Then: All rewards equal -1.0 (same as scalar reward)

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.reward_batch(states, "listen")
        expected = np.array([tiger_pomdp.reward(s, "listen") for s in states])
        np.testing.assert_array_equal(result, expected)

    def test_reward_batch_open_left_matches_scalar(self, tiger_pomdp: TigerPOMDP):
        """reward_batch(open_left) matches scalar reward for all particles.

        Purpose: Validates reward_batch for open_left matches per-state scalar reward

        Given: 16 states alternating tiger_left/tiger_right and action=open_left
        When: reward_batch is called
        Then: Output equals [reward(s, open_left) for s in states] exactly

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.reward_batch(states, "open_left")
        expected = np.array([tiger_pomdp.reward(s, "open_left") for s in states])
        np.testing.assert_array_equal(result, expected)

    def test_reward_batch_open_right_matches_scalar(self, tiger_pomdp: TigerPOMDP):
        """reward_batch(open_right) matches scalar reward for all particles.

        Purpose: Validates reward_batch for open_right matches per-state scalar reward

        Given: 16 states alternating tiger_left/tiger_right and action=open_right
        When: reward_batch is called
        Then: Output equals [reward(s, open_right) for s in states] exactly

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.reward_batch(states, "open_right")
        expected = np.array([tiger_pomdp.reward(s, "open_right") for s in states])
        np.testing.assert_array_equal(result, expected)

    def test_sample_next_state_batch_listen_preserves_states(self, tiger_pomdp: TigerPOMDP):
        """sample_next_state_batch with listen returns each input state unchanged.

        Purpose: Validates that listen action leaves states unchanged in batch mode

        Given: 16 states alternating tiger_left/tiger_right and action=listen
        When: sample_next_state_batch is called
        Then: Each output state equals the corresponding input state

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.sample_next_state_batch(states, "listen")
        assert len(result) == self.N
        for inp, out in zip(states, result):
            assert inp == out

    def test_sample_next_state_batch_open_door_valid_states(self, tiger_pomdp: TigerPOMDP):
        """sample_next_state_batch with open actions produces valid STATES.

        Purpose: Validates open actions produce valid random states in batch mode

        Given: 16 states all tiger_left and action=open_left
        When: sample_next_state_batch is called with fixed seed
        Then: All outputs are in STATES and both states appear across 32 draws

        Test type: unit
        """
        np.random.seed(0)
        states = ["tiger_left"] * self.N
        result = tiger_pomdp.sample_next_state_batch(states, "open_left")
        assert len(result) == self.N
        assert all(s in STATES for s in result)
        # With 16 draws we expect both states to appear (probability of only one ≈ 2^-15)
        assert len(set(result)) == 2

    def test_observation_log_probability_per_state_listen_hear_left(self, tiger_pomdp: TigerPOMDP):
        """observation_log_probability_per_state for listen+hear_left matches scalar.

        Purpose: Validates vectorised per-state log-prob matches scalar reference for listen action

        Given: 16 states (alternating tiger_left/tiger_right), action=listen, observation=hear_left
        When: observation_log_probability_per_state is called
        Then: Output matches [observation_log_probability(s, listen, [hear_left])[0] for s in states]

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        obs = "hear_left"
        result = tiger_pomdp.observation_log_probability_per_state(states, "listen", obs)
        expected = np.array(
            [tiger_pomdp.observation_log_probability(s, "listen", [obs])[0] for s in states]
        )
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_observation_log_probability_per_state_open_door(self, tiger_pomdp: TigerPOMDP):
        """observation_log_probability_per_state for open action matches scalar.

        Purpose: Validates per-state log-prob for non-listen action (hear_nothing) is 0.0

        Given: 16 states, action=open_left, observation=hear_nothing
        When: observation_log_probability_per_state is called
        Then: All outputs are 0.0 (log of probability 1.0)

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.observation_log_probability_per_state(
            states, "open_left", "hear_nothing"
        )
        np.testing.assert_array_equal(result, np.zeros(self.N))

    def test_observation_log_probability_per_state_open_door_wrong_obs(
        self, tiger_pomdp: TigerPOMDP
    ):
        """observation_log_probability_per_state returns -inf for invalid obs with open action.

        Purpose: Validates invalid observation yields -inf log-prob for open actions

        Given: 16 states, action=open_right, observation=hear_left (invalid for open action)
        When: observation_log_probability_per_state is called
        Then: All outputs are -inf

        Test type: unit
        """
        states = ["tiger_left", "tiger_right"] * (self.N // 2)
        result = tiger_pomdp.observation_log_probability_per_state(
            states, "open_right", "hear_left"
        )
        assert np.all(result == -np.inf)


class TestTigerListenObservationKernelValidDistribution:
    """Tests pinning the listen-action observation kernel as a valid distribution.

    Regressions for C1: scalar ``observation_log_probability`` previously returned
    ``log(0.15)`` for the impossible event ``(action=listen, obs=hear_nothing, *)``,
    making the kernel sum to 1.15 across declared observations and disagreeing
    with both the batch path and the sampler.
    """

    def test_listen_kernel_sums_to_one_tiger_left(self, tiger_pomdp: TigerPOMDP):
        """Listen-action observation kernel sums to 1.0 over all declared observations.

        Purpose: Validates the listen scalar kernel is a valid probability distribution
            over OBSERVATIONS for state=tiger_left.

        Given: TigerPOMDP env, state=tiger_left, action=listen.
        When: scalar observation_log_probability is queried over the full
            observation vocabulary [hear_left, hear_right, hear_nothing].
        Then: sum(exp(log_probs)) == 1.0 within atol=1e-12.

        Test type: unit
        """
        log_probs = tiger_pomdp.observation_log_probability(
            "tiger_left", "listen", ["hear_left", "hear_right", "hear_nothing"]
        )
        assert float(np.sum(np.exp(log_probs))) == pytest.approx(1.0, abs=1e-12)

    def test_listen_kernel_sums_to_one_tiger_right(self, tiger_pomdp: TigerPOMDP):
        """Listen-action observation kernel sums to 1.0 — symmetric case.

        Purpose: Same as the tiger_left case but with state=tiger_right to confirm
            the bug was symmetric across states.

        Given: TigerPOMDP env, state=tiger_right, action=listen.
        When: scalar observation_log_probability is queried over the full
            observation vocabulary.
        Then: sum(exp(log_probs)) == 1.0 within atol=1e-12.

        Test type: unit
        """
        log_probs = tiger_pomdp.observation_log_probability(
            "tiger_right", "listen", ["hear_left", "hear_right", "hear_nothing"]
        )
        assert float(np.sum(np.exp(log_probs))) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("state", ["tiger_left", "tiger_right"])
    def test_listen_with_hear_nothing_returns_neg_inf(self, tiger_pomdp: TigerPOMDP, state: str):
        """observation_log_probability(listen, hear_nothing, *) returns -inf.

        Purpose: Validates that ``hear_nothing`` is treated as impossible under
            ``listen`` — only ``hear_left`` and ``hear_right`` are emittable.

        Given: TigerPOMDP env, action=listen, observation=hear_nothing, parametrized
            over both tiger states.
        When: scalar observation_log_probability is queried.
        Then: log-prob is -inf, agreeing with the batch path's documented contract.

        Test type: unit
        """
        scalar = tiger_pomdp.observation_log_probability(state, "listen", ["hear_nothing"])[0]
        batch = tiger_pomdp.observation_log_probability_per_state(
            [state], "listen", "hear_nothing"
        )[0]
        assert scalar == -np.inf
        assert batch == -np.inf
        assert scalar == batch  # parity
