"""Utilities for testing state transition probability methods.

This module provides general utility functions for validating that probability
methods correctly match empirical sampling distributions across different POMDP
environments.
"""

from typing import Any, Callable, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from POMDPPlanners.core.environment import StateTransitionModel


def validate_probability_matches_empirical_distribution(
    transition_model: StateTransitionModel,
    num_samples: int = 1000,
    max_js_divergence: float = 0.05,
    max_wasserstein_distance: float = 0.1,
    normalization_tolerance: float = 0.01,
    seed: int = 42,
    distance_metric: str = "js",
) -> dict:
    """Test that computed probabilities match empirical sampling distribution.

    This function validates that a state transition model's probability() method
    correctly computes transition probabilities by comparing the computed probability
    distribution against an empirical distribution obtained through repeated sampling.

    The method works by:
    1. Sampling many states (default 1000) from the transition model
    2. Computing empirical PDF by counting state occurrences
    3. Computing theoretical probabilities for all sampled states using probability()
    4. Normalizing the computed probabilities
    5. Measuring distance between the two distributions (JS divergence or Wasserstein)

    Args:
        transition_model: State transition model to test
        num_samples: Number of samples for building distributions. Defaults to 1000.
        max_js_divergence: Maximum allowed Jensen-Shannon divergence. Defaults to 0.05.
        max_wasserstein_distance: Maximum allowed Wasserstein distance. Defaults to 0.1.
        normalization_tolerance: Tolerance for probability normalization check. Defaults to 0.01.
        seed: Random seed for reproducibility. Defaults to 42.
        distance_metric: Distance metric to use ("js" for Jensen-Shannon or "wasserstein").
            Defaults to "js".

    Returns:
        Dictionary containing test results with keys:
            - 'unique_states': List of unique sampled states
            - 'empirical_probs': Empirical probability distribution (normalized counts)
            - 'computed_probs': Computed probabilities from probability() method (normalized)
            - 'distance': Distance between distributions (JS divergence or Wasserstein)
            - 'distance_metric': Which distance metric was used
            - 'probabilities_normalized': Whether computed probabilities sum to ~1.0
            - 'num_unique_states': Number of unique states found

    Raises:
        AssertionError: If computed probabilities don't match empirical distribution
            within the specified tolerance, or if probabilities don't normalize properly

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerStateTransition
        >>> transition = TigerStateTransition(state="tiger_left", action="listen")
        >>> results = validate_probability_matches_empirical_distribution(transition)
        >>> print(f"JS Divergence: {results['distance']:.6f}")
        JS Divergence: 0.000123
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    def states_equal(s1, s2):
        """Check if two states are equal (handles numpy arrays and other types)."""
        if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
            return np.array_equal(s1, s2)
        return s1 == s2

    def find_state_index(state, state_list):
        """Find index of state in list, handling numpy arrays."""
        for i, s in enumerate(state_list):
            if states_equal(state, s):
                return i
        return -1

    # Step 1: Sample num_samples states from the transition model
    all_samples = []
    for _ in range(num_samples):
        sampled_state = transition_model.sample()[0]
        all_samples.append(sampled_state)

    # Step 2: Build empirical PDF by counting unique states
    unique_states = []
    state_counts = []

    for sample in all_samples:
        idx = find_state_index(sample, unique_states)
        if idx == -1:
            # New unique state
            unique_states.append(sample)
            state_counts.append(1)
        else:
            # Existing state
            state_counts[idx] += 1

    # Handle case where we got no samples
    if len(unique_states) == 0:
        raise ValueError("Could not sample any states from the transition model")

    # Convert counts to empirical probabilities
    state_counts = np.array(state_counts, dtype=float)
    empirical_probs = state_counts / num_samples

    # Step 3: Compute theoretical probabilities for all unique states
    computed_probs_raw = transition_model.probability(unique_states)

    # Step 4: Normalize computed probabilities
    prob_sum = np.sum(computed_probs_raw)
    probabilities_normalized = abs(prob_sum - 1.0) < normalization_tolerance

    # Normalize the computed probabilities
    if prob_sum > 0:
        computed_probs = computed_probs_raw / prob_sum
    else:
        computed_probs = computed_probs_raw

    # Step 5: Compute distance between distributions
    if distance_metric == "js":
        # Jensen-Shannon divergence (symmetric, bounded between 0 and 1)
        distance = jensenshannon(empirical_probs, computed_probs)
        max_distance = max_js_divergence
        distance_name = "Jensen-Shannon divergence"
    elif distance_metric == "wasserstein":
        # Wasserstein distance (requires positions for discrete distributions)
        # Use state indices as positions
        positions = np.arange(len(unique_states))
        distance = wasserstein_distance(positions, positions, empirical_probs, computed_probs)
        max_distance = max_wasserstein_distance
        distance_name = "Wasserstein distance"
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Verify that the distance is within tolerance
    assert distance <= max_distance, (
        f"{distance_name} = {distance:.6f} exceeds maximum {max_distance:.6f}\n"
        f"Empirical probs: {empirical_probs}\n"
        f"Computed probs: {computed_probs}"
    )

    # Verify normalization
    assert (
        probabilities_normalized
    ), f"Computed probabilities sum to {prob_sum:.6f}, expected 1.0 (±{normalization_tolerance})"

    return {
        "unique_states": unique_states,
        "empirical_probs": empirical_probs,
        "computed_probs": computed_probs,
        "distance": float(distance),
        "distance_metric": distance_metric,
        "probabilities_normalized": probabilities_normalized,
        "num_unique_states": len(unique_states),
        "state_counts": state_counts.tolist(),
    }
