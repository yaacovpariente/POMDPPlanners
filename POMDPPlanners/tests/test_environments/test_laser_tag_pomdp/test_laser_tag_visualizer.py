"""Tests for LaserTag POMDP visualization functionality.

This module tests the LaserTag POMDP visualization features, focusing on:
- Visualization generation and caching
- Error handling for invalid visualization parameters
"""

import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestLaserTagVisualization:
    """Test LaserTag POMDP visualization functionality.

    Purpose: Validates visualization generation, caching, and error handling

    Given: LaserTag POMDP environments and episode histories
    When: Visualization methods are called with various parameters
    Then: Visualizations are generated correctly or appropriate errors are raised

    Test type: integration
    """

    def test_cache_visualization_functionality(self):
        """Test cache_visualization method basic functionality.

        Purpose: Validates cache_visualization runs without errors and creates output

        Given: LaserTag environment and mock episode history
        When: cache_visualization is called with valid parameters
        Then: Method executes without error and creates visualization file

        Test type: integration
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        # Create simple test history

        # Create a belief for testing
        dummy_particles = [np.array([0.0, 0.0, 6.0, 10.0, 0.0])]
        dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        steps = []
        for i in range(3):
            state = np.array([float(i), 0.0, float(6 - i), 10.0, 0.0])
            step = StepData(
                state=state,
                action=1,  # South
                next_state=state,
                observation=(
                    6 - i + 0.1,
                    10.1,
                    1.0,
                    1.5,
                    2.0,
                    1.2,
                    0.8,
                    1.8,
                ),  # 8D laser observation
                reward=-1.0,
                belief=test_belief,
            )
            steps.append(step)

        history = History(
            history=steps,
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=3,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        cache_path = Path("test_laser_tag_visualization.gif")

        try:
            # This should not raise an exception
            env.cache_visualization(history.history, cache_path)

            # Clean up if file was created
            if cache_path.exists():
                cache_path.unlink()

        except Exception as e:
            pytest.fail(f"cache_visualization raised an exception: {e}")

    def test_cache_visualization_error_handling(self):
        """Test cache_visualization error handling with invalid inputs.

        Purpose: Validates cache_visualization properly handles invalid inputs

        Given: Invalid cache_path types and empty histories
        When: cache_visualization is called with invalid parameters
        Then: Appropriate exceptions are raised with descriptive messages

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        # Create dummy history for testing
        dummy_history = History(
            history=[],
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=0,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        # Create non-empty history for cache_path tests
        dummy_particles = [np.array([0.0, 0.0, 6.0, 10.0, 0.0])]
        dummy_log_weights = np.array([-0.1])
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        state = np.array([0.0, 0.0, 6.0, 10.0, 0.0])
        step = StepData(
            state=state,
            action=1,  # South
            next_state=state,
            observation=(6 + 0.1, 10.1, 1.0, 1.5, 2.0, 1.2, 0.8, 1.8),
            reward=-1.0,
            belief=test_belief,
        )

        non_empty_history = History(
            history=[step],
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=1,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        # Test with non-Path cache_path
        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            env.cache_visualization(non_empty_history.history, cast(Path, "invalid_path"))

        # Test with non-gif extension
        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            env.cache_visualization(non_empty_history.history, Path("test.png"))

        # Test with empty history
        with pytest.raises(ValueError, match="Cannot visualize empty history"):
            env.cache_visualization(dummy_history.history, Path("test.gif"))


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
