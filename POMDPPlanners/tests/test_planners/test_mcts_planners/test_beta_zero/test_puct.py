"""Tests for the BetaZero PUCT action selection module.

This module tests the PUCT selection rule and progressive widening with PUCT
used in BetaZero, covering formula correctness, Q-value normalisation, prior
weighting, and widening threshold behaviour.
"""

from typing import Optional

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.planners.mcts_planners.beta_zero.puct import (
    puct_action_progressive_widening,
    puct_selection,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


class SimpleActionSampler(ActionSampler):
    def __init__(self):
        self._counter = 0

    def sample(self, belief_node=None):
        self._counter += 1
        return f"new_action_{self._counter}"


def _make_belief_node_with_children(q_values, visit_counts, parent_visits):
    particles = [[0.0], [1.0]]
    log_weights = np.log([0.5, 0.5])
    belief = WeightedParticleBelief(particles, log_weights)
    node = BeliefNode(belief=belief)
    node.visit_count = parent_visits
    for q, n in zip(q_values, visit_counts):
        child = ActionNode(action=f"a_{q}", parent=node)
        child.q_value = q
        child.visit_count = n
    return node


class TestPUCTSelection:
    """Tests for the puct_selection function."""

    def test_puct_selects_highest_prior_when_all_unvisited(self):
        """Test that PUCT selects the action with the highest prior when all visit counts are zero.

        Purpose: Validates that with zero visit counts and equal Q-values the
        exploration bonus is dominated entirely by the prior probabilities.

        Given: A belief node with 3 action children, all with Q=0 and N=0,
               and action priors [0.1, 0.7, 0.2].
        When: puct_selection is called with exploration_constant=1.0.
        Then: The action child corresponding to the highest prior (0.7) is selected.

        Test type: unit
        """
        node = _make_belief_node_with_children(
            q_values=[0.0, 0.0, 0.0],
            visit_counts=[0, 0, 0],
            parent_visits=1,
        )
        priors = np.array([0.1, 0.7, 0.2])

        selected = puct_selection(node, exploration_constant=1.0, action_priors=priors)

        assert selected is node.children[1]

    def test_puct_converges_to_highest_q_with_many_visits(self):
        """Test that PUCT selects the highest-Q action when visit counts are large.

        Purpose: Validates that the exploration bonus vanishes relative to the
        Q-value term when actions have been visited many times, causing PUCT
        to behave greedily with respect to Q.

        Given: A belief node with 3 actions having Q=[10.0, 5.0, 8.0] and very
               high visit counts [10000, 10000, 10000], parent_visits=30000,
               and uniform priors.
        When: puct_selection is called with a moderate exploration constant.
        Then: The action with the highest Q-value (10.0) is selected.

        Test type: unit
        """
        node = _make_belief_node_with_children(
            q_values=[10.0, 5.0, 8.0],
            visit_counts=[10000, 10000, 10000],
            parent_visits=30000,
        )

        selected = puct_selection(node, exploration_constant=1.0)

        assert selected is node.children[0]

    def test_puct_manual_formula_computation(self):
        """Test PUCT scores against a manual computation with known values.

        Purpose: Validates the numerical correctness of the PUCT formula by
        comparing the selected action to a hand-computed expected result.

        Given: Q=[1.0, 0.5], N=[10, 5], parent_N=15, priors=[0.6, 0.4], c=1.0.
               Normalised Q: [1.0, 0.0].
               Exploration for a0: 1.0 * 0.6 * sqrt(15) / (1+10) = 0.6*3.8730/11 ~ 0.2113
               Exploration for a1: 1.0 * 0.4 * sqrt(15) / (1+5)  = 0.4*3.8730/6  ~ 0.2582
               PUCT(a0) = 1.0 + 0.2113 ~ 1.2113
               PUCT(a1) = 0.0 + 0.2582 ~ 0.2582
        When: puct_selection is called.
        Then: Action a0 (index 0) is selected because its PUCT score is higher.

        Test type: unit
        """
        node = _make_belief_node_with_children(
            q_values=[1.0, 0.5],
            visit_counts=[10, 5],
            parent_visits=15,
        )
        priors = np.array([0.6, 0.4])

        selected = puct_selection(node, exploration_constant=1.0, action_priors=priors)

        # Verify by manual calculation
        sqrt_parent = np.sqrt(15)
        q_norm = np.array([1.0, 0.0])  # normalised
        exploration = 1.0 * priors * sqrt_parent / (1.0 + np.array([10, 5]))
        expected_scores = q_norm + exploration
        expected_idx = int(np.argmax(expected_scores))

        assert selected is node.children[expected_idx]
        assert expected_idx == 0

    def test_q_values_normalized_to_01(self):
        """Test that Q-values are normalised to [0, 1] before computing PUCT scores.

        Purpose: Validates that arbitrarily scaled Q-values are mapped to the
        unit interval so that the exploration constant is problem-independent.

        Given: Q=[100, 200] with high visit counts and uniform priors.
               After normalisation: Q_norm=[0.0, 1.0].
        When: puct_selection is called with a small exploration constant so
              that the Q-term dominates.
        Then: The action with Q=200 (normalised to 1.0) is selected.

        Test type: unit
        """
        node = _make_belief_node_with_children(
            q_values=[100, 200],
            visit_counts=[1000, 1000],
            parent_visits=2000,
        )

        selected = puct_selection(node, exploration_constant=0.01)

        assert selected is node.children[1]

    def test_low_visit_high_prior_preferred_over_high_visit_low_prior(self):
        """Test that an under-visited action with a high prior beats a well-visited action with a low prior.

        Purpose: Validates the interplay between visit counts and priors in the
        exploration bonus: low N(b,a) and high P(a|b) should produce a large
        exploration term that can outweigh a higher visit count.

        Given: Two actions with equal Q-values. Action 0 has N=100 and
               prior=0.1; Action 1 has N=1 and prior=0.9. Parent visits=101.
        When: puct_selection is called with a sufficiently large exploration constant.
        Then: Action 1 (low visits, high prior) is selected.

        Test type: unit
        """
        node = _make_belief_node_with_children(
            q_values=[5.0, 5.0],
            visit_counts=[100, 1],
            parent_visits=101,
        )
        priors = np.array([0.1, 0.9])

        selected = puct_selection(node, exploration_constant=2.0, action_priors=priors)

        assert selected is node.children[1]


class TestPUCTActionProgressiveWidening:
    """Tests for the puct_action_progressive_widening function."""

    def test_widening_adds_action_at_threshold(self):
        """Test that progressive widening adds new actions when the threshold is met and stops when it is not.

        Purpose: Validates the widening gate condition
        len(children) <= k_a * N^alpha_a. When the condition holds, a new
        action should be sampled; once it no longer holds, existing actions
        should be selected via PUCT.

        Given: A belief node at depth 1 (non-root) with k_a=1.0 and alpha_a=0.5,
               and a SimpleActionSampler that produces unique action names.
        When: The node starts with 0 children and visit_count=0 (widening
              condition met) so the first call adds a child. Then visit_count
              and children are manipulated so that a second call still meets
              the threshold (adds another child), and finally a state where
              the threshold is exceeded so PUCT selection is used instead.
        Then: New actions are created while the widening condition is met, and
              an existing action is returned (via PUCT) once the condition
              is no longer met.

        Test type: unit
        """
        particles = [[0.0], [1.0]]
        log_weights = np.log([0.5, 0.5])
        belief = WeightedParticleBelief(particles, log_weights)
        node = BeliefNode(belief=belief)
        # Place the node at depth > 0 by giving it a parent so the
        # root min-visit-count logic does not interfere.
        parent_belief = WeightedParticleBelief(particles, log_weights)
        parent_node = BeliefNode(belief=parent_belief)
        action_bridge = ActionNode(action="bridge", parent=parent_node)
        node.parent = action_bridge

        sampler = SimpleActionSampler()
        k_a = 1.0
        alpha_a = 0.5

        # visit_count=0 -> is_leaf or visit_count==0 triggers widening
        node.visit_count = 0
        result1 = puct_action_progressive_widening(
            belief_node=node,
            alpha_a=alpha_a,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=k_a,
        )
        assert result1.action == "new_action_1"
        assert len(node.children) == 1

        # visit_count=1 -> len(children)=1 <= 1.0 * 1^0.5 = 1.0 -> widen
        node.visit_count = 1
        result2 = puct_action_progressive_widening(
            belief_node=node,
            alpha_a=alpha_a,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=k_a,
        )
        assert result2.action == "new_action_2"
        assert len(node.children) == 2

        # Give the children some visits so PUCT can select among them.
        node.children[0].visit_count = 5
        node.children[0].q_value = 10.0
        node.children[1].visit_count = 5
        node.children[1].q_value = 1.0

        # visit_count=100 -> len(children)=2 <= 1.0 * 100^0.5 = 10.0 -> still widen
        # To stop widening we need len(children) > k_a * N^alpha_a.
        # With k_a=1.0 and alpha_a=0.5 we need 2 > 1.0 * N^0.5, i.e. N < 4.
        # Set visit_count=3 -> 1.0 * 3^0.5 ~ 1.732 < 2 -> no widening.
        node.visit_count = 3
        result3 = puct_action_progressive_widening(
            belief_node=node,
            alpha_a=alpha_a,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=k_a,
        )
        # Should return one of the existing children via PUCT, not a new action.
        assert result3 in node.children
        assert len(node.children) == 2
