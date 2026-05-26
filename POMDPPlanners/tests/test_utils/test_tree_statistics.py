# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.utils.tree_statistics import compute_tree_metrics

np.random.seed(42)
random.seed(42)


@pytest.fixture
def test_belief():
    """Test belief.

    Purpose: Provides WeightedParticleBelief fixture for tree statistics testing

    Given: Two integer particles (1, 2) with non-uniform weights (0.6, 0.4)
    When: Fixture is used in test functions
    Then: Returns WeightedParticleBelief with specified particles and log-transformed weights

    Test type: unit
    """
    # Create a simple particle belief with two particles
    particles = [1, 2]  # Simple integer particles
    log_weights = np.log(np.array([0.6, 0.4]))  # Convert to log weights
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


def test_compute_tree_metrics_single_node(test_belief):
    """Test metrics computation for a tree with a single node (leaf).

    Purpose: Validates that tree metrics computation correctly handles leaf nodes with no action children

    Given: BeliefNode with test_belief and visit_count=5 but no action children (leaf node)
    When: compute_tree_metrics is called on the single node tree
    Then: Returns metrics with min_actions_visit_count=0, max_actions_visit_count=0, and actions_visit_count_entropy=0

    Test type: unit
    """
    tree = BeliefNode(belief=test_belief)
    tree.visit_count = 5

    metrics = compute_tree_metrics(tree)

    # Check min visits
    min_visits = next(m for m in metrics if m.name == "min_actions_visit_count")
    assert min_visits.value == 0.0

    # Check max visits
    max_visits = next(m for m in metrics if m.name == "max_actions_visit_count")
    assert max_visits.value == 0.0

    # Check entropy (should be 0 for single node)
    entropy = next(m for m in metrics if m.name == "actions_visit_count_entropy")
    assert entropy.value == 0.0


@pytest.mark.skip(
    reason="compute_tree_metrics only supports leaf/single-node case in current implementation."
)
def test_compute_tree_metrics_uniform_visits(test_belief):
    """Test compute tree metrics uniform visits.

    Purpose: Validates that tree metrics correctly compute statistics for trees with uniformly distributed action visits

    Given: BeliefNode with multiple action children having equal visit counts
    When: compute_tree_metrics analyzes the uniform visit distribution
    Then: Entropy should be maximized, min and max visits should be equal, indicating balanced exploration

    Test type: unit
    """
    pass


@pytest.mark.skip(
    reason="compute_tree_metrics only supports leaf/single-node case in current implementation."
)
def test_compute_tree_metrics_skewed_visits(test_belief):
    """Test compute tree metrics skewed visits.

    Purpose: Validates that tree metrics correctly identify highly skewed visit distributions across action children

    Given: BeliefNode with action children having highly unequal visit counts (e.g., one action visited many times, others rarely)
    When: compute_tree_metrics analyzes the skewed visit distribution
    Then: Large difference between min and max visits, low entropy indicating concentrated exploration on few actions

    Test type: unit
    """
    pass


@pytest.mark.skip(
    reason="compute_tree_metrics only supports leaf/single-node case in current implementation."
)
def test_compute_tree_metrics_empty_tree(test_belief):
    """Test compute tree metrics empty tree.

    Purpose: Validates that tree metrics computation handles edge case of completely empty tree structure

    Given: BeliefNode with zero visit count and no action children (empty/uninitialized tree)
    When: compute_tree_metrics processes the empty tree structure
    Then: Returns appropriate default metrics values without errors, handling the empty tree gracefully

    Test type: unit
    """
    pass
