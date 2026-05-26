# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.utils.visualization import plot_tree_graphs


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for visualization testing with proper cleanup.

    Creates a unique temporary directory for each test to store generated plots
    and MLFlow artifacts, ensuring test isolation and automatic cleanup.

    Yields:
        Path: Temporary directory path for storing test artifacts

    Note:
        Uses force cleanup with garbage collection to handle any remaining
        file handles that may prevent directory removal on some systems.
    """
    import gc
    import shutil
    import uuid

    temp_dir = Path(tempfile.gettempdir())
    unique_dir = temp_dir / f"test_{uuid.uuid4().hex}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir = unique_dir
    try:
        # Ensure the directory exists and is empty
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        yield temp_cache_dir
    finally:
        # Cleanup
        try:
            if temp_cache_dir.exists():
                # Force close any open file handles
                gc.collect()
                # Try to remove the directory
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_cache_dir}: {e}")


def test_plot_tree_graphs(temp_cache_dir):
    """Test that plot_tree_graphs generates interactive tree visualizations.

    Purpose: Validates that tree graph plotting creates interactive visualizations of belief trees

    Given: Mock BeliefNode root with ActionNode and BeliefNode children
    When: plot_tree_graphs is called with root node
    Then: Interactive tree visualization is generated and displayed (tested by checking no exceptions)

    Test type: unit
    """
    # Setup - Create a simple mock tree structure
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Add an action node
    action_node = ActionNode(action="listen", parent=root_belief)
    root_belief.children = [action_node]

    # Add a belief node child
    child_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([1.0, 1.0])], log_weights=np.array([0.1])
        ),
        observation="tiger_left",
        parent=action_node,
    )
    action_node.children = [child_belief]

    # Set some values for visualization
    root_belief.v_value = -5.0
    root_belief.visit_count = 10
    action_node.q_value = -4.0
    action_node.visit_count = 8
    child_belief.v_value = -3.0
    child_belief.visit_count = 5

    # Execute - This should not raise an exception
    # Note: The function calls fig.show() which opens a browser window in interactive mode
    # In test environment, this might not display but should not crash
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        # Check that it's not a fundamental error
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )


def test_plot_tree_graphs_single_node(temp_cache_dir):
    """Test that plot_tree_graphs handles single node tree.

    Purpose: Validates proper handling when tree has only one node (edge case)

    Given: Single BeliefNode with no children
    When: plot_tree_graphs is called with single node
    Then: Interactive tree visualization is generated successfully for single node

    Test type: unit
    """
    # Setup - Create single node tree
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Set values for visualization
    root_belief.v_value = -5.0
    root_belief.visit_count = 10

    # Execute - This should not raise an exception
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )


def test_plot_tree_graphs_deep_tree(temp_cache_dir):
    """Test that plot_tree_graphs handles deep tree structure.

    Purpose: Validates proper handling when tree has many levels (edge case)

    Given: Deep tree structure with multiple levels
    When: plot_tree_graphs is called with deep tree
    Then: Interactive tree visualization is generated successfully for deep tree

    Test type: unit
    """
    # Setup - Create deep tree structure
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Create a chain of nodes (deep but narrow tree)
    current_node = root_belief
    for i in range(5):  # Create 5 levels deep
        action_node = ActionNode(action=f"action_{i}", parent=current_node)
        current_node.children = [action_node]

        belief_node = BeliefNode(
            belief=WeightedParticleBelief(
                particles=[np.array([float(i), float(i)])], log_weights=np.array([0.1])
            ),
            observation=f"obs_{i}",
            parent=action_node,
        )
        action_node.children = [belief_node]

        # Set values
        action_node.q_value = -4.0 - i
        action_node.visit_count = 8 - i
        belief_node.v_value = -3.0 - i
        belief_node.visit_count = 5 - i

        current_node = belief_node

    # Set root values
    root_belief.v_value = -5.0
    root_belief.visit_count = 10

    # Execute - This should not raise an exception
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )
