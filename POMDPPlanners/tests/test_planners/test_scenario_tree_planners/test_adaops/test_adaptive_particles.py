# SPDX-License-Identifier: MIT

"""Hand-calculated tests for AdaOPS particle operations."""

import numpy as np
import pytest

from POMDPPlanners.planners.scenario_tree_planners.adaptive_particles import (
    adaptive_resample,
    bounded_kld_sample_size,
    design_effect,
    effective_sample_size,
    kld_sample_size,
    l1_weight_distance,
    systematic_resample,
)


def test_effective_sample_size_and_design_effect_are_independent_calculations():
    weights = [0.9, 0.1]
    assert effective_sample_size(weights) == pytest.approx(1.0 / 0.82, abs=1e-12)
    assert design_effect(weights) == pytest.approx(2.0 * 0.82, abs=1e-12)


def test_kld_formula_and_particle_caps_cover_both_boundaries():
    expected = int(
        np.ceil((1.0 / 0.1) * (1.0 - 2.0 / 9.0 + 1.644853626951 * np.sqrt(2.0 / 9.0)) ** 3)
    )
    assert kld_sample_size(2, 0.05) == expected
    assert bounded_kld_sample_size(1, 0.05, 5, 20) == 5
    assert bounded_kld_sample_size(100, 0.05, 5, 20) == 20


def test_systematic_resampling_preserves_count_and_resets_weights():
    particles, weights = systematic_resample(
        ["rare", "common"], [0.1, 0.9], 10, np.random.default_rng(3)
    )
    assert len(particles) == 10
    assert particles.count("common") == 9
    assert weights.tolist() == pytest.approx([0.1] * 10)


def test_adaptive_resampling_requires_explicit_bins_or_uses_the_documented_cap():
    particles, weights, occupied = adaptive_resample(
        [0, 1], [0.5, 0.5], np.random.default_rng(1), 2, 6, 0.05, None
    )
    assert len(particles) == 6
    assert occupied == 0
    assert weights.sum() == pytest.approx(1.0)

    particles, _, occupied = adaptive_resample(
        [0, 1], [0.5, 0.5], np.random.default_rng(1), 2, 6, 0.05, lambda s: s
    )
    assert occupied == 2
    assert 2 <= len(particles) <= 6


@pytest.mark.parametrize(
    ("right", "expected"),
    [([0.55, 0.45], 0.1), ([0.5, 0.5], 0.0), ([1.0, 0.0], 1.0)],
)
def test_l1_distance_has_exact_packing_boundary(right, expected):
    assert l1_weight_distance([0.5, 0.5], right) == pytest.approx(expected)
