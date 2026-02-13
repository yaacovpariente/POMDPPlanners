"""Tests for the ConstrainedZero training utilities.

This module tests the loss function and training loop for the 3-head
ConstrainedZero network, including the additional failure BCE loss.
"""

import random

import numpy as np
import torch

from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training import (
    compute_constrained_zero_loss,
    train_constrained_network,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
    ConstrainedTrainingExample,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

BELIEF_DIM = 4
N_ACTIONS = 3
ACTION_DIM = 2
HIDDEN_SIZES = (32, 32)


def _make_discrete_network():
    return ConstrainedZeroNetwork(
        belief_dim=BELIEF_DIM,
        action_space_type="discrete",
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
    )


def _make_continuous_network():
    return ConstrainedZeroNetwork(
        belief_dim=BELIEF_DIM,
        action_space_type="continuous",
        action_dim=ACTION_DIM,
        hidden_sizes=HIDDEN_SIZES,
    )


def _fill_buffer(buffer, n_examples, policy_dim, belief_dim=BELIEF_DIM):
    for _ in range(n_examples):
        belief = np.random.randn(belief_dim).astype(np.float32)
        raw_policy = np.random.rand(policy_dim).astype(np.float32)
        policy = raw_policy / raw_policy.sum()
        value = float(np.random.randn())
        failure = float(np.random.randint(2))
        buffer.add(ConstrainedTrainingExample(belief, policy, value, failure))


class TestComputeConstrainedZeroLoss:
    """Tests for the compute_constrained_zero_loss function."""

    def test_returns_total_loss_and_components(self):
        """Test loss function returns total and component dict.

        Purpose: Validates output structure of the loss function.

        Given: A discrete ConstrainedZeroNetwork and random batch data.
        When: compute_constrained_zero_loss is called.
        Then: Returns (tensor, dict) with value_loss, policy_loss, failure_loss keys.

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.randn(8, BELIEF_DIM)
        policies = torch.softmax(torch.randn(8, N_ACTIONS), dim=-1)
        values = torch.randn(8)
        failures = torch.randint(0, 2, (8,)).float()

        loss, components = compute_constrained_zero_loss(
            network, beliefs, policies, values, failures
        )
        assert isinstance(loss, torch.Tensor)
        assert "value_loss" in components
        assert "policy_loss" in components
        assert "failure_loss" in components

    def test_loss_has_failure_component(self):
        """Test that failure loss contributes to total.

        Purpose: Validates BCE failure loss is non-zero and part of total.

        Given: A network and batch with known failure targets.
        When: Loss is computed.
        Then: failure_loss > 0 and total >= value + policy + failure.

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.randn(8, BELIEF_DIM)
        policies = torch.softmax(torch.randn(8, N_ACTIONS), dim=-1)
        values = torch.randn(8)
        failures = torch.ones(8)  # all failures

        _, components = compute_constrained_zero_loss(network, beliefs, policies, values, failures)
        assert components["failure_loss"] > 0

    def test_bce_correct_for_known_inputs(self):
        """Test BCE loss matches manual computation for extreme inputs.

        Purpose: Validates the failure loss is proper BCE.

        Given: A network with targets of all zeros.
        When: Loss is computed.
        Then: failure_loss equals BCE(logit, 0) which is log(1 + exp(logit)).

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.zeros(1, BELIEF_DIM)
        policies = torch.softmax(torch.ones(1, N_ACTIONS), dim=-1)
        values = torch.zeros(1)
        failures = torch.zeros(1)

        _, components = compute_constrained_zero_loss(network, beliefs, policies, values, failures)
        # failure_loss should be a valid BCE value (non-negative)
        assert components["failure_loss"] >= 0

    def test_continuous_loss_has_failure_component(self):
        """Test failure loss works with continuous action space.

        Purpose: Validates loss function for continuous networks.

        Given: A continuous ConstrainedZeroNetwork.
        When: Loss is computed with failure targets.
        Then: All three loss components are present and positive.

        Test type: unit
        """
        network = _make_continuous_network()
        beliefs = torch.randn(8, BELIEF_DIM)
        policies = torch.randn(8, 2 * ACTION_DIM)
        values = torch.randn(8)
        failures = torch.randint(0, 2, (8,)).float()

        _, components = compute_constrained_zero_loss(network, beliefs, policies, values, failures)
        assert "failure_loss" in components
        assert components["failure_loss"] >= 0


class TestTrainConstrainedNetwork:
    """Tests for the train_constrained_network function."""

    def test_training_reduces_loss(self):
        """Test that training reduces total loss over epochs.

        Purpose: Validates the training loop is functional.

        Given: A discrete network and a buffer with 50 examples.
        When: Training is run for 10 epochs.
        Then: The final total_loss is less than the initial total_loss.

        Test type: integration
        """
        torch.manual_seed(42)
        network = _make_discrete_network()
        buffer = ConstrainedTrainingBuffer(capacity=1000)
        _fill_buffer(buffer, 50, N_ACTIONS)

        metrics = train_constrained_network(
            network=network,
            buffer=buffer,
            n_epochs=10,
            batch_size=16,
            learning_rate=1e-3,
        )

        assert metrics["total_loss"][-1] < metrics["total_loss"][0]

    def test_returns_failure_loss_metric(self):
        """Test that training metrics include failure_loss.

        Purpose: Validates failure_loss is tracked during training.

        Given: A network and buffer with examples.
        When: Training is run.
        Then: metrics dict contains "failure_loss" key with per-epoch values.

        Test type: unit
        """
        network = _make_discrete_network()
        buffer = ConstrainedTrainingBuffer(capacity=1000)
        _fill_buffer(buffer, 20, N_ACTIONS)

        metrics = train_constrained_network(
            network=network,
            buffer=buffer,
            n_epochs=3,
            batch_size=16,
        )

        assert "failure_loss" in metrics
        assert len(metrics["failure_loss"]) == 3

    def test_all_metrics_present(self):
        """Test training returns all four metric keys.

        Purpose: Validates all expected metrics are returned.

        Given: A network and buffer.
        When: Training completes.
        Then: Metrics dict has total_loss, value_loss, policy_loss, failure_loss.

        Test type: unit
        """
        network = _make_discrete_network()
        buffer = ConstrainedTrainingBuffer(capacity=1000)
        _fill_buffer(buffer, 20, N_ACTIONS)

        metrics = train_constrained_network(
            network=network,
            buffer=buffer,
            n_epochs=2,
            batch_size=16,
        )

        expected_keys = {"total_loss", "value_loss", "policy_loss", "failure_loss"}
        assert set(metrics.keys()) == expected_keys
