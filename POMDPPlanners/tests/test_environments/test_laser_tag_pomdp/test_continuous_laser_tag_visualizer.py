"""Tests for the Continuous LaserTag visualizer module.

Tests cover input validation, history data extraction, and basic
visualization creation (without actually saving GIFs in most tests).
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
)


@pytest.fixture
def visualizer():
    """Create a basic visualizer instance."""
    return ContinuousLaserTagVisualizer(
        grid_size=np.array([11.0, 7.0]),
        walls=np.array([[5.0, 3.0, 0.5, 0.5]]),
        robot_radius=0.3,
        opponent_radius=0.3,
        dangerous_areas=[(3.0, 5.0)],
        dangerous_area_radius=1.0,
    )


@pytest.fixture
def simple_history():
    """Create a simple 3-step history for testing."""
    steps = []
    for i in range(3):
        state = np.array([1.0 + i, 3.0, 8.0, 5.0, 0.0])
        next_state = np.array([2.0 + i, 3.0, 7.5, 4.5, 0.0])
        steps.append(
            StepData(
                state=state,
                action=np.array([1.0, 0.0, 0.0]),
                next_state=next_state,
                observation=np.random.rand(8),
                reward=-1.0,
                belief=None,  # type: ignore[arg-type]
            )
        )
    return steps


class TestVisualizerValidation:
    """Tests for visualizer input validation."""

    def test_empty_history_raises(self, visualizer):
        """Test that empty history raises ValueError.

        Purpose: Validates error on empty history.

        Given: An empty history list.
        When: create_visualization is called.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="empty"):
            visualizer.create_visualization([], Path("/tmp/test.gif"))

    def test_non_gif_path_raises(self, visualizer, simple_history):
        """Test that non-gif path raises ValueError.

        Purpose: Validates file extension check.

        Given: A history and a path not ending in .gif.
        When: create_visualization is called.
        Then: ValueError is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match=".gif"):
            visualizer.create_visualization(simple_history, Path("/tmp/test.png"))

    def test_non_path_raises(self, visualizer, simple_history):
        """Test that non-Path cache_path raises TypeError.

        Purpose: Validates type check on cache_path.

        Given: A string instead of Path.
        When: create_visualization is called.
        Then: TypeError is raised.

        Test type: unit
        """
        with pytest.raises(TypeError, match="Path"):
            visualizer.create_visualization(simple_history, "/tmp/test.gif")


class TestVisualizerDataExtraction:
    """Tests for history data extraction."""

    def test_extract_history_positions(self, visualizer, simple_history):
        """Test that history positions are correctly extracted.

        Purpose: Validates history data extraction.

        Given: A simple 3-step history.
        When: _extract_history is called.
        Then: Robot and opponent paths have correct lengths and values.

        Test type: unit
        """
        robot_path, opponent_path, actions, beliefs = visualizer._extract_history(simple_history)
        assert len(robot_path) == 3
        assert len(opponent_path) == 3
        assert len(actions) == 3
        np.testing.assert_array_almost_equal(robot_path[0], [1.0, 3.0])
        np.testing.assert_array_almost_equal(opponent_path[0], [8.0, 5.0])


class TestVisualizerCreation:
    """Tests for visualization GIF creation."""

    def test_create_visualization_saves_gif(self, visualizer, simple_history, tmp_path):
        """Test that visualization creates a GIF file.

        Purpose: Validates end-to-end visualization creation.

        Given: A simple history and a temporary path.
        When: create_visualization is called.
        Then: A .gif file is created at the specified path.

        Test type: integration
        """
        gif_path = tmp_path / "test_episode.gif"
        visualizer.create_visualization(simple_history, gif_path)
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0
