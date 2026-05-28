# SPDX-License-Identifier: MIT

"""Tests for RockSample POMDP visualization functionality.

This module tests the RockSample POMDP visualization features, focusing on:
- Visualization generation and caching
- Error handling for invalid visualization parameters
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    RockSampleState,
    create_rock_sample_state,
)


class TestVisualization:
    """Test visualization functionality."""

    def test_visualize_path_parameter_validation(self):
        """Test visualization parameter validation.

        Purpose: Validates proper parameter validation for visualization

        Given: Invalid cache_path parameters
        When: visualize_path() is called
        Then: Appropriate errors are raised

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
        state = create_rock_sample_state((0, 0), (True, False))

        with pytest.raises(TypeError, match="cache_path must be a Path object"):
            pomdp.visualize_path([state], [1], "invalid_path")  # type: ignore

        with pytest.raises(ValueError, match="cache_path must end with .gif"):
            pomdp.visualize_path([state], [1], Path("test.png"))

    def test_cache_visualization_empty_history(self):
        """Test visualization caching with empty history.

        Purpose: Validates proper error handling for empty history

        Given: Empty history
        When: cache_visualization() is called
        Then: ValueError is raised

        Test type: unit
        """
        pomdp = RockSamplePOMDP()
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

        with pytest.raises(ValueError, match="Cannot visualize empty history"):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_path = Path(temp_dir) / "test.gif"
                pomdp.cache_visualization(empty_history.history, cache_path)

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.animation.FuncAnimation.save")
    def test_cache_visualization_success(self, mock_save, mock_close):
        """Test successful visualization caching.

        Purpose: Validates successful generation and caching of visualization

        Given: Valid history with steps
        When: cache_visualization() is called
        Then: Visualization is generated and saved without errors

        Test type: unit
        """
        pomdp = RockSamplePOMDP()

        # Create history with steps
        state1 = create_rock_sample_state((0, 0), (True, False))
        state2 = create_rock_sample_state((0, 1), (True, False))

        step1 = StepData(
            state=state1,
            action=2,  # East
            next_state=state2,
            observation="none",
            reward=-1.0,
            belief=Mock(spec=Belief),
        )
        step2 = StepData(
            state=state2,
            action=None,
            next_state=state2,
            observation="none",
            reward=0.0,
            belief=Mock(spec=Belief),
        )

        history = History(
            history=[step1, step2],
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test.gif"

            # Should not raise any exceptions
            pomdp.cache_visualization(history.history, cache_path)

            # Verify mocks were called
            mock_save.assert_called_once()
            mock_close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
