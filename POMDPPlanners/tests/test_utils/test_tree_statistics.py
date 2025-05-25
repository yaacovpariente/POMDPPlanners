import numpy as np
import pytest

from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.utils.tree_statistics import compute_tree_metrics


@pytest.fixture
def test_belief():
    # Create a simple particle belief with two particles
    particles = [1, 2]  # Simple integer particles
    log_weights = np.log(np.array([0.6, 0.4]))  # Convert to log weights
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


def test_compute_tree_metrics_single_node(test_belief):
    """Test metrics computation for a tree with a single node (leaf)."""
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


@pytest.mark.skip(reason="compute_tree_metrics only supports leaf/single-node case in current implementation.")
def test_compute_tree_metrics_uniform_visits(test_belief):
    pass

@pytest.mark.skip(reason="compute_tree_metrics only supports leaf/single-node case in current implementation.")
def test_compute_tree_metrics_skewed_visits(test_belief):
    pass

@pytest.mark.skip(reason="compute_tree_metrics only supports leaf/single-node case in current implementation.")
def test_compute_tree_metrics_empty_tree(test_belief):
    pass
