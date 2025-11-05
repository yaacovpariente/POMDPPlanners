import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
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
    When: State transition model is created for listen action and next state is sampled
    Then: All samples return the same state as the input state, confirming that listening doesn't change the tiger's position

    Test type: unit
    """
    # Listening shouldn't change the state
    for state in tiger_pomdp.states:
        dist = tiger_pomdp.state_transition_model(state, "listen")
        samples = dist.sample(n_samples=10)
        assert all(s == state for s in samples)


def test_state_transition_open_door(tiger_pomdp):
    """Test state transition for open door actions.

    Purpose: Validates that TigerPOMDP open door actions randomly place tiger behind either door

    Given: A TigerPOMDP environment starting from tiger_left state and both open actions (open_left, open_right)
    When: State transition model is created for open actions and next states are sampled 100 times
    Then: All samples are valid states and both states appear, demonstrating random placement after opening doors

    Test type: unit
    """
    # Opening a door should randomly place tiger behind either door
    for action in ["open_left", "open_right"]:
        dist = tiger_pomdp.state_transition_model("tiger_left", action)
        samples = dist.sample(n_samples=100)
        assert all(s in tiger_pomdp.states for s in samples)
        assert len(set(samples)) == 2  # Should get both states


def test_observation_model_listen(tiger_pomdp):
    """Test observation model for listen action.

    Purpose: Validates that TigerPOMDP listen action provides mostly correct observations with some noise

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right)
    When: Observation model is created for listen action and observations are sampled 100 times
    Then: Most observations are correct (hear_left for tiger_left, hear_right for tiger_right) with approximately 85% accuracy

    Test type: unit
    """
    # Test listen action with both states
    for state in tiger_pomdp.states:
        dist = tiger_pomdp.observation_model(state, "listen")
        samples = dist.sample(n_samples=100)
        assert all(s in tiger_pomdp.observations for s in samples)

        # Should mostly get correct observation
        expected_obs = "hear_left" if state == "tiger_left" else "hear_right"
        correct_count = sum(1 for s in samples if s == expected_obs)
        assert correct_count > 70  # Should be around 85% correct


def test_observation_model_open_door(tiger_pomdp):
    """Test observation model for open door actions.

    Purpose: Validates that TigerPOMDP open door actions always provide "hear_nothing" observations

    Given: A TigerPOMDP environment and both states (tiger_left, tiger_right) with both open actions (open_left, open_right)
    When: Observation model is created for open actions and observations are sampled 10 times
    Then: All samples return "hear_nothing", confirming that opening doors provides no auditory information

    Test type: unit
    """
    # Opening a door should always give 'hear_nothing'
    for state in tiger_pomdp.states:
        for action in ["open_left", "open_right"]:
            dist = tiger_pomdp.observation_model(state, action)
            samples = dist.sample(n_samples=10)
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
                History(
                    history=steps,
                    discount_factor=0.95,
                    average_state_sampling_time=0.0,
                    average_action_time=0.0,
                    average_observation_time=0.0,
                    average_belief_update_time=0.0,
                    average_reward_time=0.0,
                    actual_num_steps=4,
                    reach_terminal_state=True,
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
                History(
                    history=steps,
                    discount_factor=0.95,
                    average_state_sampling_time=0.0,
                    average_action_time=0.0,
                    average_observation_time=0.0,
                    average_belief_update_time=0.0,
                    average_reward_time=0.0,
                    actual_num_steps=3,
                    reach_terminal_state=True,
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
                History(
                    history=steps,
                    discount_factor=0.95,
                    average_state_sampling_time=0.0,
                    average_action_time=0.0,
                    average_observation_time=0.0,
                    average_belief_update_time=0.0,
                    average_reward_time=0.0,
                    actual_num_steps=len(steps),
                    reach_terminal_state=True,
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
            History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=True,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        )

    # Compute metrics
    metrics = tiger_pomdp.compute_metrics(histories)

    # Use generic confidence interval verification
    verify_metrics_within_confidence_intervals(metrics)


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
        History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(steps),
            reach_terminal_state=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
    ]

    # Verify consistency using reusable utility function
    verify_environment_metric_consistency(tiger_pomdp, histories)
