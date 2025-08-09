import pytest
import numpy as np
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


def test_initialization(tiger_pomdp):
    """Test initialization.
    
    Purpose: Validates proper initialization of 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    assert tiger_pomdp.discount_factor == 0.95
    assert set(tiger_pomdp.states) == {"tiger_left", "tiger_right"}
    assert set(tiger_pomdp.actions) == {"listen", "open_left", "open_right"}
    assert set(tiger_pomdp.observations) == {"hear_left", "hear_right", "hear_nothing"}


def test_get_actions(tiger_pomdp):
    """Test get actions.
    
    Purpose: Validates get actions
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    actions = tiger_pomdp.get_actions()
    assert len(actions) == 3
    assert set(actions) == {"listen", "open_left", "open_right"}


def test_initial_state_distribution(tiger_pomdp):
    """Test initial state distribution.
    
    Purpose: Validates initial state distribution
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    dist = tiger_pomdp.initial_state_dist()
    # Sample multiple times to ensure we get both states
    samples = dist.sample(n_samples=100)
    assert all(s in tiger_pomdp.states for s in samples)
    assert len(set(samples)) == 2  # Should get both states


def test_initial_observation_distribution(tiger_pomdp):
    """Test initial observation distribution.
    
    Purpose: Validates initial observation distribution
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    dist = tiger_pomdp.initial_observation_dist()
    samples = dist.sample(n_samples=10)
    assert all(s == "hear_nothing" for s in samples)


def test_state_transition_listen(tiger_pomdp):
    """Test state transition listen.
    
    Purpose: Validates state transition listen
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Listening shouldn't change the state
    for state in tiger_pomdp.states:
        dist = tiger_pomdp.state_transition_model(state, "listen")
        samples = dist.sample(n_samples=10)
        assert all(s == state for s in samples)


def test_state_transition_open_door(tiger_pomdp):
    """Test state transition open door.
    
    Purpose: Validates state transition open door
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Opening a door should randomly place tiger behind either door
    for action in ["open_left", "open_right"]:
        dist = tiger_pomdp.state_transition_model("tiger_left", action)
        samples = dist.sample(n_samples=100)
        assert all(s in tiger_pomdp.states for s in samples)
        assert len(set(samples)) == 2  # Should get both states


def test_observation_model_listen(tiger_pomdp):
    """Test observation model listen.
    
    Purpose: Validates observation model listen
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
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
    """Test observation model open door.
    
    Purpose: Validates observation model open door
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Opening a door should always give 'hear_nothing'
    for state in tiger_pomdp.states:
        for action in ["open_left", "open_right"]:
            dist = tiger_pomdp.observation_model(state, action)
            samples = dist.sample(n_samples=10)
            assert all(s == "hear_nothing" for s in samples)


def test_reward_func_listen(tiger_pomdp):
    """Test reward func open door.
    
    Purpose: Validates reward func open door
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    """Test reward func listen.
    
    Purpose: Validates reward func listen
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Listening should always give -1 reward
    for state in tiger_pomdp.states:
        assert tiger_pomdp.reward(state, "listen") == -1.0


def test_reward_func_open_door(tiger_pomdp):
    """Test is terminal.
    
    Purpose: Validates is terminal
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test opening doors with tiger
    assert tiger_pomdp.reward("tiger_left", "open_left") == -100.0
    assert tiger_pomdp.reward("tiger_right", "open_right") == -100.0

    # Test opening doors with treasure
    assert tiger_pomdp.reward("tiger_right", "open_left") == 10.0
    assert tiger_pomdp.reward("tiger_left", "open_right") == 10.0


def test_is_terminal(tiger_pomdp):
    # Currently always returns False
    for state in tiger_pomdp.states:
        assert not tiger_pomdp.is_terminal(state)


class TestTigerPOMDPConfigId:
    """Test that config_id changes with different discount factor.
    
    Purpose: Validates config_id behavior for  different discount factor
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
    
    def test_config_id_consistency(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different states.
    
    Purpose: Validates config_id behavior for  different states
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = TigerPOMDP(discount_factor=0.95)
        assert tiger_pomdp.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different actions.
    
    Purpose: Validates config_id behavior for  different actions
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = TigerPOMDP(discount_factor=0.8)
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_states(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different observations.
    
    Purpose: Validates config_id behavior for  different observations
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.states = ["tiger_left", "tiger_right", "tiger_middle"]  # Different states
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_actions(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is a valid SHA-256 hash.
    
    Purpose: Validates config_id behavior for  format
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.actions = ["listen", "open_left", "open_right", "wait"]  # Different actions
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_different_observations(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id changes with different observations."""
        other_env = TigerPOMDP(discount_factor=0.95)
        other_env.observations = ["hear_left", "hear_right", "hear_nothing", "hear_both"]  # Different observations
        assert tiger_pomdp.config_id != other_env.config_id
    
    def test_config_id_format(self, tiger_pomdp: TigerPOMDP):
        """Test that config_id is deterministic (same input always produces same output).
    
    Purpose: Validates config_id behavior for  deterministic
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
        config_id = tiger_pomdp.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for a perfect agent that always opens the correct door.
    
    Purpose: Validates compute metrics perfect agent
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        config_id1 = tiger_pomdp.config_id
        config_id2 = tiger_pomdp.config_id
        assert config_id1 == config_id2


class TestTigerPOMDPMetrics:
    """Test suite for TigerPOMDP compute_metrics functionality."""
    
    def test_compute_metrics_perfect_agent(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for a perfect agent that always opens the correct door."""
        from POMDPPlanners.core.simulation import History, StepData
        
        # Create histories where agent always opens correct door
        histories = []
        for _ in range(10):
            # Randomly choose initial state
            state = np.random.choice(["tiger_left", "tiger_right"])
            # Agent listens a few times then opens correct door
            steps = []
            for _ in range(3):  # Listen 3 times
                steps.append(StepData(
                    state=state,
                    action="listen",
                    next_state=state,
                    observation="hear_nothing",
                    reward=-1.0,
                    belief=None  # Belief not needed for metrics
                ))
            # Open correct door
            correct_action = "open_right" if state == "tiger_left" else "open_left"
            steps.append(StepData(
                state=state,
                action=correct_action,
                next_state=state,
                observation="hear_nothing",
                reward=10.0 if correct_action == "open_right" else -100.0,
                belief=None  # Belief not needed for metrics
            ))
            histories.append(History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=4,
                reach_terminal_state=True,
                policy_run_data=PolicyRunData(info_variables=[])
            ))
        
        """Test metrics for an agent that always opens the wrong door.
    
    Purpose: Validates compute metrics failing agent
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        
        # Should have 100% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 1.0
        
        # Should have average of 3 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 3.0
    
    def test_compute_metrics_failing_agent(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for an agent that always opens the wrong door."""
        from POMDPPlanners.core.simulation import History, StepData
        
        # Create histories where agent always opens wrong door
        histories = []
        for _ in range(10):
            state = np.random.choice(["tiger_left", "tiger_right"])
            steps = []
            for _ in range(2):  # Listen 2 times
                steps.append(StepData(
                    state=state,
                    action="listen",
                    next_state=state,
                    observation="hear_nothing",
                    reward=-1.0,
                    belief=None  # Belief not needed for metrics
                ))
            # Open wrong door
            wrong_action = "open_left" if state == "tiger_left" else "open_right"
            steps.append(StepData(
                state=state,
                action=wrong_action,
                next_state=state,
                observation="hear_nothing",
                reward=-100.0 if wrong_action == "open_left" else 10.0,
                belief=None  # Belief not needed for metrics
            ))
            histories.append(History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=3,
                reach_terminal_state=True,
                policy_run_data=PolicyRunData(info_variables=[])
            ))
        
        """Test metrics for an agent with mixed performance.
    
    Purpose: Validates compute metrics mixed performance
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: performance
    """
        
        # Should have 0% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.0
        
        # Should have average of 2 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 2.0
    
    def test_compute_metrics_mixed_performance(self, tiger_pomdp: TigerPOMDP):
        """Test metrics for an agent with mixed performance."""
        from POMDPPlanners.core.simulation import History, StepData
        
        # Create histories with mixed success/failure
        histories = []
        for i in range(10):
            state = np.random.choice(["tiger_left", "tiger_right"])
            steps = []
            for _ in range(i % 3 + 1):  # Varying number of listens
                steps.append(StepData(
                    state=state,
                    action="listen",
                    next_state=state,
                    observation="hear_nothing",
                    reward=-1.0,
                    belief=None  # Belief not needed for metrics
                ))
            # Alternate between correct and incorrect actions
            action = "open_right" if (i % 2 == 0) == (state == "tiger_left") else "open_left"
            steps.append(StepData(
                state=state,
                action=action,
                next_state=state,
                observation="hear_nothing",
                reward=10.0 if action == "open_right" else -100.0,
                belief=None  # Belief not needed for metrics
            ))
            histories.append(History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=True,
                policy_run_data=PolicyRunData(info_variables=[])
            ))
        
        """Test metrics with empty history list.
    
    Purpose: Validates compute metrics empty histories
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        
        # Should have 50% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.5
        
        # Should have average of 1.9 listens (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 1.9
    
    def test_compute_metrics_empty_histories(self, tiger_pomdp: TigerPOMDP):
        """Test metrics with empty history list."""
        metrics = tiger_pomdp.compute_metrics([])
        
        # Should have 0% success rate
        success_metric = next(m for m in metrics if m.name == "success_rate")
        assert success_metric.value == 0.0
        
        # Should have 0 listens
        listens_metric = next(m for m in metrics if m.name == "average_listens")
        assert listens_metric.value == 0.0
