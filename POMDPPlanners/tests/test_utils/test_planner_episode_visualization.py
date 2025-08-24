import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, call
import random

from POMDPPlanners.utils.planner_episode_visualization import visualize_planner_episode
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData

np.random.seed(42)
random.seed(42)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def tiger_environment():
    """Create TigerPOMDP environment for testing."""
    return TigerPOMDP(discount_factor=0.95, name="TestTiger")


@pytest.fixture
def mock_planner():
    """Create mock planner for testing."""
    planner = Mock()
    planner.name = "TestPlanner"
    return planner


@pytest.fixture
def mock_belief():
    """Create mock belief for testing."""
    belief = Mock()
    belief.name = "TestBelief"
    return belief


@pytest.fixture
def sample_episode_history():
    """Create a sample episode history for testing."""
    step1 = StepData(
        state="tiger_left",
        action="listen",
        next_state="tiger_left", 
        observation="growl_left",
        reward=-1.0,
        belief=None
    )
    step2 = StepData(
        state="tiger_left",
        action="open_left",
        next_state="tiger_right",
        observation="tiger_left",
        reward=10.0,
        belief=None
    )
    
    history = History(
        history=[step1, step2],
        actual_num_steps=2,
        reach_terminal_state=False,
        average_action_time=0.1,
        average_belief_update_time=0.05,
        average_observation_time=0.02,
        average_reward_time=0.01,
        average_state_sampling_time=0.03,
        discount_factor=0.95,
        policy_run_data={}
    )
    
    return history


class TestVisualizePlannerEpisode:
    """Test cases for visualize_planner_episode function."""

    def test_visualize_planner_episode_basic_functionality(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test basic functionality of visualize_planner_episode.
        
        Purpose: Validates that the function runs episodes and calls environment visualization
        
        Given: Mock planner, environment, belief, and cache directory
        When: visualize_planner_episode is called with n_episodes=2
        Then: Episodes are run and environment visualization is cached for each episode
        
        Test type: unit
        """
        # Mock run_episode to return sample history directly
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result) as mock_run_episode:
            # Mock environment.cache_visualization
            tiger_environment.cache_visualization = Mock()
            
            # Execute function
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=2,
                cache_dir=temp_cache_dir,
                num_steps=5  # Reduced for testing
            )
            
            # Verify run_episode was called twice
            assert mock_run_episode.call_count == 2
            
            # Verify run_episode was called with correct parameters (episode-specific calls)
            for i in range(2):
                call_args = mock_run_episode.call_args_list[i]
                kwargs = call_args[1]
                assert kwargs['environment'] == tiger_environment
                assert kwargs['policy'] == mock_planner
                assert kwargs['initial_belief'] == mock_belief
                assert kwargs['num_steps'] == 5
                assert 'logger' in kwargs
            
            # Verify environment visualization was called twice
            assert tiger_environment.cache_visualization.call_count == 2
            
            # Verify cache paths are correct
            expected_cache_paths = [
                temp_cache_dir / "TestPlanner_0.gif",
                temp_cache_dir / "TestPlanner_1.gif"
            ]
            
            for i, call_args in enumerate(tiger_environment.cache_visualization.call_args_list):
                history_arg = call_args[1]['history']
                cache_path_arg = call_args[1]['cache_path']
                
                assert history_arg == sample_episode_history
                assert cache_path_arg == expected_cache_paths[i]

    def test_visualize_planner_episode_single_episode(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test visualization with single episode.
        
        Purpose: Validates function works correctly with n_episodes=1
        
        Given: Mock components, belief, and n_episodes=1
        When: visualize_planner_episode is called
        Then: Single episode is run and visualized
        
        Test type: unit
        """
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result) as mock_run_episode:
            tiger_environment.cache_visualization = Mock()
            
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=1,
                cache_dir=temp_cache_dir,
                num_steps=5
            )
            
            # Verify single call with correct parameters
            mock_run_episode.assert_called_once()
            call_args = mock_run_episode.call_args[1]
            assert call_args['environment'] == tiger_environment
            assert call_args['policy'] == mock_planner
            assert call_args['initial_belief'] == mock_belief
            assert call_args['num_steps'] == 5
            assert 'logger' in call_args
            
            tiger_environment.cache_visualization.assert_called_once_with(
                history=sample_episode_history,
                cache_path=temp_cache_dir / "TestPlanner_0.gif"
            )

    def test_visualize_planner_episode_zero_episodes(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir):
        """Test behavior with zero episodes.
        
        Purpose: Validates function handles n_episodes=0 correctly
        
        Given: Mock components, belief, and n_episodes=0
        When: visualize_planner_episode is called
        Then: No episodes are run and no visualization calls are made
        
        Test type: edge case
        """
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode') as mock_run_episode:
            tiger_environment.cache_visualization = Mock()
            
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=0,
                cache_dir=temp_cache_dir,
                num_steps=5
            )
            
            # Verify no calls were made
            mock_run_episode.assert_not_called()
            tiger_environment.cache_visualization.assert_not_called()

    def test_visualize_planner_episode_cache_path_formatting(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test cache path formatting for different planner names and episode IDs.
        
        Purpose: Validates that cache paths are formatted correctly with planner name and episode ID
        
        Given: Planner with specific name, belief, and multiple episodes
        When: visualize_planner_episode generates cache paths
        Then: Cache paths follow expected format: {planner_name}_{episode_id}.
        
        Test type: unit
        """
        # Set specific planner name for testing
        mock_planner.name = "POMCP_TestPlanner_123"
        
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result):
            tiger_environment.cache_visualization = Mock()
            
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=3,
                cache_dir=temp_cache_dir,
                num_steps=5
            )
            
            # Extract cache paths from calls
            cache_paths = []
            for call_args in tiger_environment.cache_visualization.call_args_list:
                cache_paths.append(call_args[1]['cache_path'])
            
            # Verify cache path formatting
            expected_paths = [
                temp_cache_dir / "POMCP_TestPlanner_123_0.gif",
                temp_cache_dir / "POMCP_TestPlanner_123_1.gif",
                temp_cache_dir / "POMCP_TestPlanner_123_2.gif"
            ]
            
            assert cache_paths == expected_paths

    def test_visualize_planner_episode_real_policy_integration(self, temp_cache_dir):
        """Test integration with real POMCP policy and TigerPOMDP.
        
        Purpose: Validates function works with actual policy and environment instances
        
        Given: Real TigerPOMDP environment, POMCP policy, and belief
        When: visualize_planner_episode is called with real components
        Then: Episodes run successfully and visualization is attempted
        
        Test type: integration
        """
        # Create real environment and policy
        env = TigerPOMDP(discount_factor=0.95, name="RealTiger")
        
        # Create real POMCP policy
        pomcp_policy = POMCP(
            environment=env,
            discount_factor=0.95,
            name="RealPOMCP",
            exploration_constant=1.0,
            n_simulations=10,  # Reduced for test speed
            depth=3
        )
        
        # Create a proper belief for testing using the environment's initial state distribution
        from POMDPPlanners.core.belief import get_initial_belief
        belief = get_initial_belief(env, n_particles=20)  # Reduced for testing
        
        # Mock the environment's cache_visualization to avoid file I/O
        env.cache_visualization = Mock()
        
        # Use the policy as both planner and policy (common pattern)
        visualize_planner_episode(
            planner=pomcp_policy,
            environment=env,
            belief=belief,
            n_episodes=2,
            cache_dir=temp_cache_dir,
            num_steps=5  # Reduced for testing
        )
        
        # Verify visualization was called
        assert env.cache_visualization.call_count == 2
        
        # Verify cache paths
        expected_paths = [
            temp_cache_dir / "RealPOMCP_0.gif",
            temp_cache_dir / "RealPOMCP_1.gif"
        ]
        
        for i, call_args in enumerate(env.cache_visualization.call_args_list):
            cache_path_arg = call_args[1]['cache_path']
            assert cache_path_arg == expected_paths[i]
            
            # Verify history is a valid History object
            history_arg = call_args[1]['history']
            assert isinstance(history_arg, History)  # Should be a History object

    def test_visualize_planner_episode_exception_handling(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir):
        """Test exception handling when run_episode fails.
        
        Purpose: Validates that exceptions in run_episode are properly propagated
        
        Given: run_episode that raises an exception
        When: visualize_planner_episode is called
        Then: Exception is propagated to caller
        
        Test type: error handling
        """
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', side_effect=RuntimeError("Episode execution failed")):
            tiger_environment.cache_visualization = Mock()
            
            with pytest.raises(RuntimeError, match="Episode execution failed"):
                visualize_planner_episode(
                    planner=mock_planner,
                    environment=tiger_environment,
                    belief=mock_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5
                )

    def test_visualize_planner_episode_visualization_exception(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test exception handling when environment visualization fails.
        
        Purpose: Validates that exceptions in cache_visualization are properly propagated
        
        Given: environment.cache_visualization that raises an exception
        When: visualize_planner_episode is called
        Then: Exception is propagated to caller
        
        Test type: error handling
        """
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result):
            tiger_environment.cache_visualization = Mock(side_effect=IOError("Visualization cache failed"))
            
            with pytest.raises(IOError, match="Visualization cache failed"):
                visualize_planner_episode(
                    planner=mock_planner,
                    environment=tiger_environment,
                    belief=mock_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5
                )

    def test_visualize_planner_episode_parameter_validation(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test parameter validation and type checking.
        
        Purpose: Validates that function accepts correct parameter types
        
        Given: Valid parameters of correct types
        When: visualize_planner_episode is called
        Then: Function executes without type-related errors
        
        Test type: unit
        """
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result):
            tiger_environment.cache_visualization = Mock()
            
            # Test with various valid parameter types
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=5,  # int
                cache_dir=temp_cache_dir,  # Path
                num_steps=5
            )
            
            # Should execute without errors
            assert tiger_environment.cache_visualization.call_count == 5

    def test_visualize_planner_episode_large_number_episodes(self, mock_planner, tiger_environment, mock_belief, temp_cache_dir, sample_episode_history):
        """Test performance with larger number of episodes.
        
        Purpose: Validates function can handle larger episode counts efficiently
        
        Given: Large number of episodes (10)
        When: visualize_planner_episode is called
        Then: All episodes are processed and visualized correctly
        
        Test type: performance
        """
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result) as mock_run_episode:
            tiger_environment.cache_visualization = Mock()
            
            n_episodes = 10
            visualize_planner_episode(
                planner=mock_planner,
                environment=tiger_environment,
                belief=mock_belief,
                n_episodes=n_episodes,
                cache_dir=temp_cache_dir,
                num_steps=3  # Reduced for performance testing
            )
            
            # Verify all episodes were processed
            assert mock_run_episode.call_count == n_episodes
            assert tiger_environment.cache_visualization.call_count == n_episodes
            
            # Verify all calls have correct parameters
            for call_args in mock_run_episode.call_args_list:
                kwargs = call_args[1]
                assert kwargs['environment'] == tiger_environment
                assert kwargs['policy'] == mock_planner
                assert kwargs['initial_belief'] == mock_belief
                assert kwargs['num_steps'] == 3
                assert 'logger' in kwargs

    def test_visualize_planner_episode_different_environments(self, mock_planner, mock_belief, temp_cache_dir, sample_episode_history):
        """Test function works with different environment types.
        
        Purpose: Validates function is environment-agnostic
        
        Given: Different mock environments with cache_visualization method
        When: visualize_planner_episode is called with each environment
        Then: Function works with all environment types
        
        Test type: compatibility
        """
        mock_episode_result = sample_episode_history
        
        # Test with different mock environments
        environments = []
        for i, env_name in enumerate(["TigerPOMDP", "LightDarkPOMDP", "CartPolePOMDP"]):
            env = Mock()
            env.name = env_name
            env.cache_visualization = Mock()
            environments.append(env)
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result):
            
            for env in environments:
                visualize_planner_episode(
                    planner=mock_planner,
                    environment=env,
                    belief=mock_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5
                )
                
                # Verify visualization was called for each environment
                env.cache_visualization.assert_called_once()

    def test_visualize_planner_episode_cache_dir_types(self, mock_planner, tiger_environment, mock_belief, sample_episode_history):
        """Test function accepts different cache directory path types.
        
        Purpose: Validates function works with both string and Path objects for cache_dir
        
        Given: Cache directory as both string and Path object
        When: visualize_planner_episode is called with different cache_dir types
        Then: Function works with both types and generates correct paths
        
        Test type: compatibility
        """
        mock_episode_result = sample_episode_history
        
        with patch('POMDPPlanners.utils.planner_episode_visualization.run_episode', return_value=mock_episode_result):
            tiger_environment.cache_visualization = Mock()
            
            # Test with Path object
            with tempfile.TemporaryDirectory() as temp_str:
                temp_path = Path(temp_str)
                
                visualize_planner_episode(
                    planner=mock_planner,
                    environment=tiger_environment,
                    belief=mock_belief,
                    n_episodes=1,
                    cache_dir=temp_path,  # Path object
                    num_steps=5
                )
                
                # Verify cache path is constructed correctly
                expected_cache_path = temp_path / "TestPlanner_0.gif"
                tiger_environment.cache_visualization.assert_called_with(
                    history=sample_episode_history,
                    cache_path=expected_cache_path
                )

    def test_visualize_planner_episode_docstring_example(self, temp_cache_dir):
        """Test usage example that could be in the docstring.
        
        Purpose: Validates typical usage pattern works correctly
        
        Given: Real environment and policy setup typical of user code
        When: Function is called in typical usage pattern
        Then: Function executes successfully with expected behavior
        
        Test type: example
        """
        # Create real components as user would
        env = TigerPOMDP(discount_factor=0.95, name="Tiger")
        policy = POMCP(
            environment=env,
            discount_factor=0.95,
            name="POMCP",
            exploration_constant=1.4,
            n_simulations=5,  # Minimal for testing
            depth=2
        )
        
        # Create a proper belief for testing using the environment's initial state distribution
        from POMDPPlanners.core.belief import get_initial_belief
        belief = get_initial_belief(env, n_particles=20)  # Reduced for testing
        
        # Mock visualization to avoid file I/O
        env.cache_visualization = Mock()
        
        # Execute typical usage
        visualize_planner_episode(
            planner=policy,  # Often same as policy
            environment=env,
            belief=belief,
            n_episodes=3,
            cache_dir=temp_cache_dir,
            num_steps=5  # Reduced for testing
        )
        
        # Verify expected behavior
        assert env.cache_visualization.call_count == 3
        
        # Verify cache paths follow expected pattern
        for i in range(3):
            call_args = env.cache_visualization.call_args_list[i]
            cache_path = call_args[1]['cache_path']
            expected_path = temp_cache_dir / f"POMCP_{i}.gif"
            assert cache_path == expected_path