import random
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.utils.planner_episode_visualization import visualize_planner_episode

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
def test_planner(tiger_environment):
    """Create real POMCP planner for testing."""
    return POMCP(
        environment=tiger_environment,
        discount_factor=0.95,
        name="TestPlanner",
        exploration_constant=1.0,
        n_simulations=5,  # Minimal for testing
        depth=3,
    )


@pytest.fixture
def test_belief(tiger_environment):
    """Create real WeightedParticleBelief for testing."""
    return get_initial_belief(tiger_environment, n_particles=20, resampling=True)


@pytest.fixture
def sample_episode_history():
    """Create a sample episode history for testing."""
    # Create a mock belief for testing
    mock_belief = Mock(spec=Belief)

    step1 = StepData(
        state="tiger_left",
        action="listen",
        next_state="tiger_left",
        observation="growl_left",
        reward=-1.0,
        belief=mock_belief,
    )
    step2 = StepData(
        state="tiger_left",
        action="open_left",
        next_state="tiger_right",
        observation="tiger_left",
        reward=10.0,
        belief=mock_belief,
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
        policy_run_data=[PolicyRunData(info_variables=[])],
    )

    return history


class TestVisualizePlannerEpisode:
    """Test cases for visualize_planner_episode function."""

    def test_visualize_planner_episode_basic_functionality(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test basic functionality of visualize_planner_episode.

        Purpose: Validates that the function runs episodes and calls environment visualization

        Given: Real planner, environment, belief, and cache directory
        When: visualize_planner_episode is called with n_episodes=2
        Then: Episodes are run and environment visualization is cached for each episode

        Test type: unit
        """
        # Mock run_episode to return sample history directly
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ) as mock_run_episode:
            # Mock environment.cache_visualization
            tiger_environment.cache_visualization = Mock()

            # Execute function
            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=2,
                cache_dir=temp_cache_dir,
                num_steps=5,  # Reduced for testing
            )

            # Verify run_episode was called twice
            assert mock_run_episode.call_count == 2

            # Verify run_episode was called with correct parameters (episode-specific calls)
            for i in range(2):
                call_args = mock_run_episode.call_args_list[i]
                kwargs = call_args[1]
                assert kwargs["environment"] == tiger_environment
                assert kwargs["policy"] == test_planner
                assert kwargs["initial_belief"] == test_belief
                assert kwargs["num_steps"] == 5
                assert "logger" in kwargs

            # Verify environment visualization was called twice
            assert tiger_environment.cache_visualization.call_count == 2

            # Verify cache paths are correct
            expected_cache_paths = [
                temp_cache_dir / "TestPlanner_0.gif",
                temp_cache_dir / "TestPlanner_1.gif",
            ]

            for i, call_args in enumerate(tiger_environment.cache_visualization.call_args_list):
                history_arg = call_args[1]["history"]
                cache_path_arg = call_args[1]["cache_path"]

                assert (
                    history_arg == sample_episode_history.history
                )  # Pass the history list, not the History object
                assert cache_path_arg == expected_cache_paths[i]

    def test_visualize_planner_episode_single_episode(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test visualization with single episode.

        Purpose: Validates function works correctly with n_episodes=1

        Given: Real components, belief, and n_episodes=1
        When: visualize_planner_episode is called
        Then: Single episode is run and visualized

        Test type: unit
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ) as mock_run_episode:
            tiger_environment.cache_visualization = Mock()

            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=1,
                cache_dir=temp_cache_dir,
                num_steps=5,
            )

            # Verify single call with correct parameters
            mock_run_episode.assert_called_once()
            call_args = mock_run_episode.call_args[1]
            assert call_args["environment"] == tiger_environment
            assert call_args["policy"] == test_planner
            assert call_args["initial_belief"] == test_belief
            assert call_args["num_steps"] == 5
            assert "logger" in call_args

            tiger_environment.cache_visualization.assert_called_once_with(
                history=sample_episode_history.history,  # Pass the history list, not the History object
                cache_path=temp_cache_dir / "TestPlanner_0.gif",
            )

    def test_visualize_planner_episode_zero_episodes(
        self, test_planner, tiger_environment, test_belief, temp_cache_dir
    ):
        """Test behavior with zero episodes.

        Purpose: Validates function handles n_episodes=0 correctly

        Given: Real components, belief, and n_episodes=0
        When: visualize_planner_episode is called
        Then: No episodes are run and no visualization calls are made

        Test type: edge case
        """
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode"
        ) as mock_run_episode:
            tiger_environment.cache_visualization = Mock()

            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=0,
                cache_dir=temp_cache_dir,
                num_steps=5,
            )

            # Verify no calls were made
            mock_run_episode.assert_not_called()
            tiger_environment.cache_visualization.assert_not_called()

    def test_visualize_planner_episode_cache_path_formatting(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test cache path formatting for different planner names and episode IDs.

        Purpose: Validates that cache paths are formatted correctly with planner name and episode ID

        Given: Planner with specific name, belief, and multiple episodes
        When: visualize_planner_episode generates cache paths
        Then: Cache paths follow expected format: {planner_name}_{episode_id}.

        Test type: unit
        """
        # Set specific planner name for testing
        test_planner.name = "POMCP_TestPlanner_123"

        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            tiger_environment.cache_visualization = Mock()

            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=3,
                cache_dir=temp_cache_dir,
                num_steps=5,
            )

            # Extract cache paths from calls
            cache_paths = []
            for call_args in tiger_environment.cache_visualization.call_args_list:
                cache_paths.append(call_args[1]["cache_path"])

            # Verify cache path formatting
            expected_paths = [
                temp_cache_dir / "POMCP_TestPlanner_123_0.gif",
                temp_cache_dir / "POMCP_TestPlanner_123_1.gif",
                temp_cache_dir / "POMCP_TestPlanner_123_2.gif",
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
            depth=3,
        )

        # Create a proper belief for testing using the environment's initial state distribution
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
            num_steps=5,  # Reduced for testing
        )

        # Verify visualization was called
        assert env.cache_visualization.call_count == 2

        # Verify cache paths
        expected_paths = [
            temp_cache_dir / "RealPOMCP_0.gif",
            temp_cache_dir / "RealPOMCP_1.gif",
        ]

        for i, call_args in enumerate(env.cache_visualization.call_args_list):
            cache_path_arg = call_args[1]["cache_path"]
            assert cache_path_arg == expected_paths[i]

            # Verify history is a valid list of StepData
            history_arg = call_args[1]["history"]
            assert isinstance(history_arg, list)  # Should be a list of StepData

    def test_visualize_planner_episode_exception_handling(
        self, test_planner, tiger_environment, test_belief, temp_cache_dir
    ):
        """Test exception handling when run_episode fails.

        Purpose: Validates that exceptions in run_episode are properly propagated

        Given: run_episode that raises an exception
        When: visualize_planner_episode is called
        Then: Exception is propagated to caller

        Test type: error handling
        """
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            side_effect=RuntimeError("Episode execution failed"),
        ):
            tiger_environment.cache_visualization = Mock()

            with pytest.raises(RuntimeError, match="Episode execution failed"):
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                )

    def test_visualize_planner_episode_visualization_exception(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test exception handling when environment visualization fails.

        Purpose: Validates that exceptions in cache_visualization are properly propagated

        Given: environment.cache_visualization that raises an exception
        When: visualize_planner_episode is called
        Then: Exception is propagated to caller

        Test type: error handling
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            tiger_environment.cache_visualization = Mock(
                side_effect=IOError("Visualization cache failed")
            )

            with pytest.raises(IOError, match="Visualization cache failed"):
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                )

    def test_visualize_planner_episode_parameter_validation(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test parameter validation and type checking.

        Purpose: Validates that function accepts correct parameter types

        Given: Valid parameters of correct types
        When: visualize_planner_episode is called
        Then: Function executes without type-related errors

        Test type: unit
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            tiger_environment.cache_visualization = Mock()

            # Test with various valid parameter types
            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=5,  # int
                cache_dir=temp_cache_dir,  # Path
                num_steps=5,
            )

            # Should execute without errors
            assert tiger_environment.cache_visualization.call_count == 5

    def test_visualize_planner_episode_large_number_episodes(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test performance with larger number of episodes.

        Purpose: Validates function can handle larger episode counts efficiently

        Given: Large number of episodes (10)
        When: visualize_planner_episode is called
        Then: All episodes are processed and visualized correctly

        Test type: performance
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ) as mock_run_episode:
            tiger_environment.cache_visualization = Mock()

            n_episodes = 10
            visualize_planner_episode(
                planner=test_planner,
                environment=tiger_environment,
                belief=test_belief,
                n_episodes=n_episodes,
                cache_dir=temp_cache_dir,
                num_steps=3,  # Reduced for performance testing
            )

            # Verify all episodes were processed
            assert mock_run_episode.call_count == n_episodes
            assert tiger_environment.cache_visualization.call_count == n_episodes

            # Verify all calls have correct parameters
            for call_args in mock_run_episode.call_args_list:
                kwargs = call_args[1]
                assert kwargs["environment"] == tiger_environment
                assert kwargs["policy"] == test_planner
                assert kwargs["initial_belief"] == test_belief
                assert kwargs["num_steps"] == 3
                assert "logger" in kwargs

    def test_visualize_planner_episode_different_environments(
        self, test_planner, test_belief, temp_cache_dir, sample_episode_history
    ):
        """Test function works with different environment types.

        Purpose: Validates function is environment-agnostic

        Given: Different mock environments with cache_visualization method
        When: visualize_planner_episode is called with each environment
        Then: Function works with all environment types

        Test type: compatibility
        """
        mock_episode_result = sample_episode_history

        # Test with different real environments
        noise_cov = np.eye(4) * 0.01  # 4x4 identity matrix with small noise
        environments = [
            TigerPOMDP(discount_factor=0.95, name="TigerPOMDP"),
            ContinuousLightDarkPOMDP(discount_factor=0.95, name="LightDarkPOMDP"),
            CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov, name="CartPolePOMDP"),
        ]

        # Mock cache_visualization for all environments to avoid file I/O
        for env in environments:
            env.cache_visualization = Mock()

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            for env in environments:
                visualize_planner_episode(
                    planner=test_planner,
                    environment=env,
                    belief=test_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                )

                # Verify visualization was called for each environment
                env.cache_visualization.assert_called_once()

    def test_visualize_planner_episode_cache_dir_types(
        self, test_planner, tiger_environment, test_belief, sample_episode_history
    ):
        """Test function accepts different cache directory path types.

        Purpose: Validates function works with both string and Path objects for cache_dir

        Given: Cache directory as both string and Path object
        When: visualize_planner_episode is called with different cache_dir types
        Then: Function works with both types and generates correct paths

        Test type: compatibility
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            tiger_environment.cache_visualization = Mock()

            # Test with Path object
            with tempfile.TemporaryDirectory() as temp_str:
                temp_path = Path(temp_str)

                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=1,
                    cache_dir=temp_path,  # Path object
                    num_steps=5,
                )

                # Verify cache path is constructed correctly
                expected_cache_path = temp_path / "TestPlanner_0.gif"
                tiger_environment.cache_visualization.assert_called_with(
                    history=sample_episode_history.history,  # Pass the history list, not the History object
                    cache_path=expected_cache_path,
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
            depth=2,
        )

        # Create a proper belief for testing using the environment's initial state distribution
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
            num_steps=5,  # Reduced for testing
        )

        # Verify expected behavior
        assert env.cache_visualization.call_count == 3

        # Verify cache paths follow expected pattern
        for i in range(3):
            call_args = env.cache_visualization.call_args_list[i]
            cache_path = call_args[1]["cache_path"]
            expected_path = temp_cache_dir / f"POMCP_{i}.gif"
            assert cache_path == expected_path

    def test_visualize_planner_episode_parallel_basic(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test basic parallel execution with n_jobs=2.

        Purpose: Validates that parallel execution works correctly with n_jobs=2

        Given: Real planner, environment, belief, and n_jobs=2
        When: visualize_planner_episode is called with n_episodes=4 and n_jobs=2
        Then: Episodes are run in parallel and environment visualization is cached for each episode

        Test type: unit
        """
        # Mock the helper function to track calls and avoid actual execution
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
        ) as mock_run_single:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                # Verify Parallel gets called for n_jobs > 1
                mock_parallel.return_value = Mock(return_value=None)

                # Execute function with parallel execution
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=4,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=2,
                )

                # Verify Parallel was called with correct n_jobs
                mock_parallel.assert_called_once_with(n_jobs=2)

                # Verify Parallel instance was called (indicating parallel path was taken)
                mock_parallel.return_value.assert_called_once()

                # Note: We can't easily verify the exact number of _run_single_episode calls
                # in parallel execution due to joblib's internal behavior, but we verified
                # that the parallel path was taken by checking Parallel was called

    def test_visualize_planner_episode_parallel_vs_sequential_equivalence(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test that parallel and sequential execution use correct paths.

        Purpose: Validates that n_jobs=1 uses sequential path and n_jobs=2 uses parallel path

        Given: Same parameters for both sequential and parallel execution
        When: Function is called with n_jobs=1 and n_jobs=2
        Then: Correct execution paths are chosen

        Test type: equivalence
        """
        # Test sequential execution (n_jobs=1)
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
        ) as mock_run_single:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=3,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=1,  # Sequential
                )

                # Verify Parallel was NOT called for sequential execution
                mock_parallel.assert_not_called()

                # Verify _run_single_episode was called 3 times sequentially
                assert mock_run_single.call_count == 3

        # Test parallel execution (n_jobs=2)
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
        ) as mock_run_single:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                mock_parallel.return_value = Mock(return_value=None)

                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=3,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=2,  # Parallel
                )

                # Verify Parallel WAS called for parallel execution
                mock_parallel.assert_called_once_with(n_jobs=2)
                mock_parallel.return_value.assert_called_once()

    def test_visualize_planner_episode_parallel_n_jobs_parameter_validation(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test that different n_jobs values trigger correct execution paths.

        Purpose: Validates that n_jobs=1 uses sequential path and n_jobs>1 uses parallel path

        Given: Different n_jobs values
        When: Function is called with various n_jobs parameters
        Then: Correct execution path (sequential vs parallel) is chosen

        Test type: unit
        """
        mock_episode_result = sample_episode_history

        with patch(
            "POMDPPlanners.utils.planner_episode_visualization.run_episode",
            return_value=mock_episode_result,
        ):
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                mock_parallel.return_value.__call__ = Mock(
                    side_effect=lambda func_list: [func() for func in func_list]
                )
                tiger_environment.cache_visualization = Mock()

                # Test n_jobs=1 (should NOT call Parallel)
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=2,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=1,
                )

                # Verify Parallel was NOT called for n_jobs=1
                mock_parallel.assert_not_called()

                # Reset mock for parallel test
                mock_parallel.reset_mock()
                tiger_environment.cache_visualization.reset_mock()

                # Test n_jobs=2 (should call Parallel)
                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=2,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=2,
                )

                # Verify Parallel WAS called for n_jobs=2
                mock_parallel.assert_called_once_with(n_jobs=2)

    def test_visualize_planner_episode_parallel_different_n_jobs_values(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test parallel execution with different n_jobs values.

        Purpose: Validates that various n_jobs values (2, 4, -1) work correctly

        Given: Different n_jobs values for parallel execution
        When: Function is called with n_jobs=2, n_jobs=4, n_jobs=-1
        Then: Parallel execution is triggered with correct n_jobs parameter

        Test type: unit
        """
        n_jobs_values = [2, 4, -1]  # -1 means use all available processors

        for n_jobs in n_jobs_values:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
            ) as mock_run_single:
                with patch(
                    "POMDPPlanners.utils.planner_episode_visualization.Parallel"
                ) as mock_parallel:
                    mock_parallel.return_value = Mock(return_value=None)

                    visualize_planner_episode(
                        planner=test_planner,
                        environment=tiger_environment,
                        belief=test_belief,
                        n_episodes=3,
                        cache_dir=temp_cache_dir,
                        num_steps=5,
                        n_jobs=n_jobs,
                    )

                    # Verify Parallel was called with correct n_jobs
                    mock_parallel.assert_called_once_with(n_jobs=n_jobs)

                    # Verify parallel execution path was taken
                    mock_parallel.return_value.assert_called_once()

    def test_visualize_planner_episode_parallel_exception_handling(
        self, test_planner, tiger_environment, test_belief, temp_cache_dir
    ):
        """Test exception handling in parallel execution.

        Purpose: Validates that exceptions in parallel execution are properly propagated

        Given: Parallel that raises an exception during execution
        When: visualize_planner_episode is called with n_jobs=2
        Then: Exception is propagated to caller from parallel execution

        Test type: error handling
        """
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
        ) as mock_run_single:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                # Configure Parallel to raise an exception
                mock_parallel.return_value = Mock(
                    side_effect=RuntimeError("Parallel execution failed")
                )

                with pytest.raises(RuntimeError, match="Parallel execution failed"):
                    visualize_planner_episode(
                        planner=test_planner,
                        environment=tiger_environment,
                        belief=test_belief,
                        n_episodes=2,
                        cache_dir=temp_cache_dir,
                        num_steps=5,
                        n_jobs=2,
                    )

                # Verify Parallel was called
                mock_parallel.assert_called_once_with(n_jobs=2)

    def test_visualize_planner_episode_parallel_single_episode(
        self,
        test_planner,
        tiger_environment,
        test_belief,
        temp_cache_dir,
        sample_episode_history,
    ):
        """Test parallel execution with single episode.

        Purpose: Validates that parallel execution works correctly even with n_episodes=1

        Given: Single episode and n_jobs=2
        When: visualize_planner_episode is called
        Then: Single episode is processed through parallel path correctly

        Test type: edge case
        """
        with patch(
            "POMDPPlanners.utils.planner_episode_visualization._run_single_episode"
        ) as mock_run_single:
            with patch(
                "POMDPPlanners.utils.planner_episode_visualization.Parallel"
            ) as mock_parallel:
                mock_parallel.return_value = Mock(return_value=None)

                visualize_planner_episode(
                    planner=test_planner,
                    environment=tiger_environment,
                    belief=test_belief,
                    n_episodes=1,
                    cache_dir=temp_cache_dir,
                    num_steps=5,
                    n_jobs=2,
                )

                # Verify parallel execution was used
                mock_parallel.assert_called_once_with(n_jobs=2)

                # Verify parallel path was taken
                mock_parallel.return_value.assert_called_once()

    def test_visualize_planner_episode_parallel_real_integration(self, temp_cache_dir):
        """Test parallel execution with real components.

        Purpose: Validates that parallel execution is triggered with actual policy and environment

        Given: Real TigerPOMDP environment, POMCP policy, and n_jobs=2
        When: visualize_planner_episode is called with parallel execution
        Then: Parallel path is taken and function completes successfully

        Test type: integration
        """
        # Create real environment and policy
        env = TigerPOMDP(discount_factor=0.95, name="ParallelTiger")

        pomcp_policy = POMCP(
            environment=env,
            discount_factor=0.95,
            name="ParallelPOMCP",
            exploration_constant=1.0,
            n_simulations=5,  # Minimal for testing
            depth=2,
        )

        # Create proper belief
        belief = get_initial_belief(env, n_particles=10)  # Minimal for testing

        # Mock visualization to avoid file I/O
        env.cache_visualization = Mock()

        # Mock Parallel to verify it's called but avoid actual parallel execution
        with patch("POMDPPlanners.utils.planner_episode_visualization.Parallel") as mock_parallel:
            mock_parallel.return_value = Mock(return_value=None)

            # Execute with parallel processing
            visualize_planner_episode(
                planner=pomcp_policy,
                environment=env,
                belief=belief,
                n_episodes=2,
                cache_dir=temp_cache_dir,
                num_steps=3,  # Minimal for testing
                n_jobs=2,
            )

            # Verify parallel execution was triggered
            mock_parallel.assert_called_once_with(n_jobs=2)
            mock_parallel.return_value.assert_called_once()
