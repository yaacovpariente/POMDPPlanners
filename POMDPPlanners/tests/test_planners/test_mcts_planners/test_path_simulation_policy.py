"""Tests for PathSimulationPolicy implementations.

This module tests the PathSimulationPolicy abstract base class, focusing on:
- Class initialization and parameter validation
- Terminal belief handling in the action method
- Error conditions and edge cases
- Abstract method contract enforcement
"""

import random
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief, is_terminal_belief
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    PathSimulationPolicy,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class MockEnvironment(Environment):
    """Mock environment for testing PathSimulationPolicy."""

    def __init__(self, discount_factor: float, action_space: SpaceType = SpaceType.DISCRETE):
        space_info = SpaceInfo(action_space=action_space, observation_space=SpaceType.DISCRETE)
        super().__init__(
            discount_factor=discount_factor,
            name="MockEnvironment",
            space_info=space_info,
        )

    def state_transition_model(self, state, action):
        from unittest.mock import Mock
        from POMDPPlanners.core.environment import StateTransitionModel

        mock_model = Mock(spec=StateTransitionModel)
        mock_model.probability.return_value = 1.0
        return mock_model

    def observation_model(self, next_state, action):
        from unittest.mock import Mock
        from POMDPPlanners.core.environment import ObservationModel

        mock_model = Mock(spec=ObservationModel)
        mock_model.probability.return_value = 1.0
        return mock_model

    def reward(self, state, action):
        return 1.0

    def is_terminal(self, state):
        return state == "terminal"

    def initial_state_dist(self):
        mock_dist = Mock()
        mock_dist.sample = Mock(return_value=["state1", "state2"])
        return mock_dist

    def initial_observation_dist(self):
        mock_dist = Mock()
        mock_dist.sample = Mock(return_value=[0])
        return mock_dist

    def is_equal_observation(self, observation1, observation2):
        return observation1 == observation2

    def is_equal_state(self, state1, state2):
        return state1 == state2

    def get_actions(self):
        return [0, 1, 2]


class MockActionSampler(ActionSampler):
    """Mock action sampler for continuous action spaces."""

    def __init__(self, actions=None):
        self.actions = actions or [0.0, 1.0, 2.0]

    def sample(self, belief_node=None):
        return np.random.choice(self.actions)

    def get_space(self):
        return self.actions


class ConcretePathSimulationPolicy(PathSimulationPolicy):
    """Concrete implementation of PathSimulationPolicy for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulate_path_calls = []

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        """Mock implementation that records calls and creates action nodes."""
        self.simulate_path_calls.append((belief_node, depth))

        # Create action nodes if they don't exist
        if belief_node.is_leaf:
            for action in self.environment.get_actions():  # type: ignore[attr-defined]
                action_node = ActionNode(action=action, parent=belief_node, children=tuple())
                # Set Q-values so that action 1 has the highest value
                if action == 1:
                    action_node.q_value = 2.0
                else:
                    action_node.q_value = 1.0

        return 1.0  # Mock reward

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get space type requirements for this test policy."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )


@pytest.fixture
def discrete_environment():
    """Environment with discrete action space."""
    return MockEnvironment(discount_factor=0.9, action_space=SpaceType.DISCRETE)


@pytest.fixture
def continuous_environment():
    """Environment with continuous action space."""
    return MockEnvironment(discount_factor=0.9, action_space=SpaceType.CONTINUOUS)


@pytest.fixture
def action_sampler():
    """Mock action sampler for continuous spaces."""
    return MockActionSampler()


@pytest.fixture
def belief():
    """Mock belief for testing."""
    particles = ["state1", "state2", "state3"]
    log_weights = np.log([0.3, 0.4, 0.3])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def terminal_belief():
    """Mock terminal belief for testing."""
    particles = ["terminal", "terminal"]
    log_weights = np.log([0.5, 0.5])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


class TestPathSimulationPolicyInitialization:
    """Test PathSimulationPolicy initialization and parameter validation."""

    def test_discrete_environment_initialization_with_n_simulations(self, discrete_environment):
        """Test initialization with discrete environment and simulation count.

        Purpose: Validates PathSimulationPolicy initializes correctly for discrete action spaces with simulation count

        Given: A discrete action environment and simulation count parameter
        When: PathSimulationPolicy is initialized with n_simulations
        Then: Policy is created with correct parameters and no action sampler required

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
        )

        assert policy.environment == discrete_environment
        assert policy.discount_factor == 0.95
        assert policy.name == "test_policy"
        assert policy.n_simulations == 100
        assert policy.time_out_in_seconds is None
        assert policy.action_sampler is None

    def test_discrete_environment_initialization_with_timeout(self, discrete_environment):
        """Test initialization with discrete environment and timeout.

        Purpose: Validates PathSimulationPolicy initializes correctly for discrete action spaces with timeout

        Given: A discrete action environment and timeout parameter
        When: PathSimulationPolicy is initialized with time_out_in_seconds
        Then: Policy is created with correct parameters and no action sampler required

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=None,
            time_out_in_seconds=30,
        )

        assert policy.environment == discrete_environment
        assert policy.discount_factor == 0.95
        assert policy.name == "test_policy"
        assert policy.n_simulations is None
        assert policy.time_out_in_seconds == 30
        assert policy.action_sampler is None

    def test_continuous_environment_initialization_with_action_sampler(
        self, continuous_environment, action_sampler
    ):
        """Test initialization with continuous environment and action sampler.

        Purpose: Validates PathSimulationPolicy initializes correctly for continuous action spaces with action sampler

        Given: A continuous action environment and action sampler
        When: PathSimulationPolicy is initialized with action_sampler
        Then: Policy is created with correct parameters including the action sampler

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=continuous_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
            action_sampler=action_sampler,
        )

        assert policy.environment == continuous_environment
        assert policy.discount_factor == 0.95
        assert policy.name == "test_policy"
        assert policy.n_simulations == 100
        assert policy.time_out_in_seconds is None
        assert policy.action_sampler == action_sampler

    def test_initialization_with_optional_parameters(self, discrete_environment):
        """Test initialization with optional debug and log_path parameters.

        Purpose: Validates PathSimulationPolicy initializes correctly with optional parameters

        Given: Environment and optional debug and log_path parameters
        When: PathSimulationPolicy is initialized with debug=True and log_path
        Then: Policy is created with all parameters set correctly

        Test type: unit
        """
        log_path = Path("/tmp/test_log")

        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
            debug=True,
            log_path=log_path,
        )

        assert policy.debug is True
        assert policy.log_path == log_path

    def test_both_n_simulations_and_timeout_raises_error(self, discrete_environment):
        """Test that providing both n_simulations and timeout raises ValueError.

        Purpose: Validates proper error handling when both termination criteria are provided

        Given: Environment and both n_simulations and time_out_in_seconds parameters
        When: PathSimulationPolicy is initialized with both parameters
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(
            ValueError,
            match="Cannot specify both n_simulations and time_out_in_seconds",
        ):
            ConcretePathSimulationPolicy(
                environment=discrete_environment,
                discount_factor=0.95,
                name="test_policy",
                n_simulations=100,
                time_out_in_seconds=30,
            )

    def test_continuous_environment_without_action_sampler_raises_error(
        self, continuous_environment
    ):
        """Test that continuous environment without action sampler raises ValueError.

        Purpose: Validates proper error handling when action sampler is required but not provided

        Given: A continuous action environment without action sampler
        When: PathSimulationPolicy is initialized without action_sampler
        Then: ValueError is raised with appropriate message about action sampler requirement

        Test type: unit
        """
        with pytest.raises(
            ValueError,
            match="Action sampler must be provided for continuous action spaces",
        ):
            ConcretePathSimulationPolicy(
                environment=continuous_environment,
                discount_factor=0.95,
                name="test_policy",
                n_simulations=100,
                time_out_in_seconds=None,
            )

    def test_neither_n_simulations_nor_timeout_in_learn_tree_raises_error(
        self, discrete_environment
    ):
        """Test that providing neither n_simulations nor timeout raises error during tree learning.

        Purpose: Validates proper error handling when no termination criteria are provided

        Given: A policy initialized with both n_simulations=None and time_out_in_seconds=None
        When: _learn_tree method is called
        Then: ValueError is raised about missing termination criteria

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=None,
            time_out_in_seconds=None,
        )

        particles = ["state1", "state2"]
        log_weights = np.log([0.5, 0.5])
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

        with pytest.raises(
            ValueError,
            match="Either n_simulations or time_out_in_seconds must be provided",
        ):
            policy._learn_tree(belief)


class TestPathSimulationPolicyTerminalBeliefHandling:
    """Test terminal belief handling in PathSimulationPolicy action method."""

    @patch("POMDPPlanners.planners.mcts_planners.path_simulations_policy.is_terminal_belief")
    def test_terminal_belief_returns_random_action_discrete(
        self, mock_is_terminal, discrete_environment
    ):
        """Test terminal belief handling returns random action for discrete environment.

        Purpose: Validates that terminal beliefs trigger random action selection in discrete environments

        Given: A discrete environment and a terminal belief state
        When: action method is called with terminal belief
        Then: Random action from environment actions is returned with empty PolicyRunData

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
        )

        mock_is_terminal.return_value = True

        particles = ["terminal"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        terminal_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

        actions, policy_data = policy.action(terminal_belief)

        assert len(actions) == 1
        assert actions[0] in discrete_environment.get_actions()
        assert isinstance(policy_data, PolicyRunData)
        assert policy_data.info_variables == []
        mock_is_terminal.assert_called_once_with(belief=terminal_belief, env=discrete_environment)

    @patch("POMDPPlanners.planners.mcts_planners.path_simulations_policy.is_terminal_belief")
    def test_terminal_belief_returns_random_action_continuous(
        self, mock_is_terminal, continuous_environment, action_sampler
    ):
        """Test terminal belief handling returns random action for continuous environment.

        Purpose: Validates that terminal beliefs trigger action sampler usage in continuous environments

        Given: A continuous environment with action sampler and a terminal belief state
        When: action method is called with terminal belief
        Then: Action from action sampler is returned with empty PolicyRunData

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=continuous_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
            action_sampler=action_sampler,
        )

        mock_is_terminal.return_value = True

        particles = ["terminal"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        terminal_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

        # Mock the action sampler sample method
        with patch.object(action_sampler, "sample", return_value=1.5) as mock_sample:
            actions, policy_data = policy.action(terminal_belief)

            assert len(actions) == 1
            assert actions[0] == 1.5
            assert isinstance(policy_data, PolicyRunData)
            assert policy_data.info_variables == []
            mock_is_terminal.assert_called_once_with(
                belief=terminal_belief, env=continuous_environment
            )
            mock_sample.assert_called_once()

    @patch("POMDPPlanners.planners.mcts_planners.path_simulations_policy.is_terminal_belief")
    @patch("POMDPPlanners.planners.mcts_planners.path_simulations_policy.compute_tree_metrics")
    @patch(
        "POMDPPlanners.planners.mcts_planners.path_simulations_policy.get_optimal_action_reward_setting"
    )
    def test_non_terminal_belief_builds_tree(
        self,
        mock_get_optimal,
        mock_compute_metrics,
        mock_is_terminal,
        discrete_environment,
    ):
        """Test non-terminal belief triggers tree construction and action selection.

        Purpose: Validates that non-terminal beliefs trigger proper tree construction and action selection

        Given: A discrete environment and a non-terminal belief state
        When: action method is called with non-terminal belief
        Then: Tree is built, metrics are computed, and optimal action is selected

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=5,  # Small number for testing
            time_out_in_seconds=None,
        )

        mock_is_terminal.return_value = False
        mock_compute_metrics.return_value = [
            PolicyInfoVariable("nodes", 10),
            PolicyInfoVariable("depth", 3),
        ]
        mock_get_optimal.return_value = 1

        particles = ["state1", "state2"]
        log_weights = np.array([-0.3, -0.7])  # Non-zero log weights for two particles
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

        actions, policy_data = policy.action(belief)

        assert len(actions) == 1
        assert actions[0] == 1
        assert isinstance(policy_data, PolicyRunData)
        assert policy_data.info_variables == [
            PolicyInfoVariable("nodes", 10),
            PolicyInfoVariable("depth", 3),
        ]

        # Verify tree construction happened by checking simulate_path was called
        assert len(policy.simulate_path_calls) == 5

        mock_is_terminal.assert_called_once_with(belief=belief, env=discrete_environment)
        mock_compute_metrics.assert_called_once()
        mock_get_optimal.assert_called_once()


class TestPathSimulationPolicyTreeConstruction:
    """Test tree construction methods in PathSimulationPolicy."""

    def test_construct_tree_using_n_simulations(self, discrete_environment):
        """Test tree construction using simulation count.

        Purpose: Validates that tree construction runs the correct number of simulations

        Given: A policy configured with n_simulations=10
        When: _construct_tree_using_n_simulations is called
        Then: _simulate_path is called exactly 10 times

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=10,
            time_out_in_seconds=None,
        )

        particles = ["state1"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        belief_node = BeliefNode(belief=belief)

        policy._construct_tree_using_n_simulations(belief_node)

        assert len(policy.simulate_path_calls) == 10
        for call in policy.simulate_path_calls:
            assert call[0] == belief_node
            assert call[1] == 0  # depth

    def test_construct_tree_using_timeout(self, discrete_environment):
        """Test tree construction using timeout.

        Purpose: Validates that tree construction respects time limits

        Given: A policy configured with time_out_in_seconds=1
        When: _construct_tree_using_timeout is called
        Then: _simulate_path is called multiple times within the time limit

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=None,
            time_out_in_seconds=1,  # 1 second timeout
        )

        particles = ["state1"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        belief_node = BeliefNode(belief=belief)

        policy._construct_tree_using_timeout(belief_node)

        # Should have called simulate_path multiple times within 1 second
        assert len(policy.simulate_path_calls) > 0
        for call in policy.simulate_path_calls:
            assert call[0] == belief_node
            assert call[1] == 0  # depth

    def test_construct_tree_with_none_n_simulations_raises_error(self, discrete_environment):
        """Test error when n_simulations is None during simulation-based construction.

        Purpose: Validates proper error handling when n_simulations is None but method expects it

        Given: A policy with n_simulations set to None
        When: _construct_tree_using_n_simulations is called
        Then: ValueError is raised about n_simulations being None

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=None,
            time_out_in_seconds=30,
        )

        particles = ["state1"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        belief_node = BeliefNode(belief=belief)

        with pytest.raises(ValueError, match="n_simulations must not be None"):
            policy._construct_tree_using_n_simulations(belief_node)

    def test_construct_tree_with_none_timeout_raises_error(self, discrete_environment):
        """Test error when time_out_in_seconds is None during timeout-based construction.

        Purpose: Validates proper error handling when time_out_in_seconds is None but method expects it

        Given: A policy with time_out_in_seconds set to None
        When: _construct_tree_using_timeout is called
        Then: ValueError is raised about time_out_in_seconds being None

        Test type: unit
        """
        policy = ConcretePathSimulationPolicy(
            environment=discrete_environment,
            discount_factor=0.95,
            name="test_policy",
            n_simulations=100,
            time_out_in_seconds=None,
        )

        particles = ["state1"]
        log_weights = np.array([-0.5])  # Non-zero log weight for single particle
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        belief_node = BeliefNode(belief=belief)

        with pytest.raises(ValueError, match="time_out_in_seconds must not be None"):
            policy._construct_tree_using_timeout(belief_node)


class TestPathSimulationPolicyAbstractContract:
    """Test abstract method contract enforcement."""

    def test_cannot_instantiate_abstract_base_class(self, discrete_environment):
        """Test that PathSimulationPolicy cannot be instantiated directly.

        Purpose: Validates that abstract base class cannot be instantiated without implementing abstract methods

        Given: PathSimulationPolicy abstract base class
        When: Attempting to instantiate PathSimulationPolicy directly
        Then: TypeError is raised about abstract methods

        Test type: unit
        """
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class PathSimulationPolicy"
        ):
            PathSimulationPolicy(  # type: ignore[abstract]
                environment=discrete_environment,
                discount_factor=0.95,
                name="test_policy",
                n_simulations=100,
                time_out_in_seconds=None,
            )
