"""Tests for BetaZero training utilities.

This module tests the ``compute_beta_zero_loss`` and ``train_network``
functions that drive BetaZero policy-iteration updates.
"""

import numpy as np
import pytest
import torch

from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    BetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.training import (
    compute_beta_zero_loss,
    train_network,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
    TrainingExample,
)

torch.manual_seed(42)
np.random.seed(42)

BELIEF_DIM = 4
N_ACTIONS = 3
ACTION_DIM = 2
HIDDEN_SIZES = (32, 32)


def _make_discrete_network():
    return BetaZeroNetwork(
        belief_dim=BELIEF_DIM,
        action_space_type="discrete",
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
    )


def _make_continuous_network():
    return BetaZeroNetwork(
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
        buffer.add(TrainingExample(belief, policy, value))


class TestComputeBetaZeroLoss:
    """Tests for the compute_beta_zero_loss function."""

    def test_loss_returns_component_breakdown(self):
        """Verify loss_dict contains value_loss and policy_loss as floats.

        Purpose: Validates that compute_beta_zero_loss returns a dictionary
            with the expected keys and that each value is a Python float.

        Given: A discrete BetaZero network and a small random batch.
        When: compute_beta_zero_loss is called.
        Then: The returned loss_dict has exactly "value_loss" and "policy_loss"
            keys, both mapping to float values, and total_loss is a scalar tensor.

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.randn(8, BELIEF_DIM)
        raw = torch.rand(8, N_ACTIONS)
        policies = raw / raw.sum(dim=-1, keepdim=True)
        values = torch.randn(8)

        total_loss, loss_dict = compute_beta_zero_loss(network, beliefs, policies, values)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0, "total_loss should be a scalar tensor"

        assert "value_loss" in loss_dict
        assert "policy_loss" in loss_dict
        assert isinstance(loss_dict["value_loss"], float)
        assert isinstance(loss_dict["policy_loss"], float)

    def test_discrete_policy_uses_cross_entropy(self):
        """Verify loss is computable and valid for a discrete network.

        Purpose: Validates that the discrete policy branch computes a finite,
            positive loss (cross-entropy between log-softmax output and target
            distribution).

        Given: A discrete BetaZero network and a batch with valid probability
            distribution targets.
        When: compute_beta_zero_loss is called.
        Then: The policy_loss component is a finite positive number (not NaN
            and not negative).

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.randn(16, BELIEF_DIM)
        raw = torch.rand(16, N_ACTIONS)
        policies = raw / raw.sum(dim=-1, keepdim=True)
        values = torch.randn(16)

        total_loss, loss_dict = compute_beta_zero_loss(network, beliefs, policies, values)

        policy_loss = loss_dict["policy_loss"]
        assert not np.isnan(policy_loss), "policy_loss should not be NaN"
        assert np.isfinite(policy_loss), "policy_loss should be finite"
        assert policy_loss > 0, "cross-entropy against non-degenerate targets should be positive"

    def test_continuous_policy_uses_gaussian_nll(self):
        """Verify loss is computable for a continuous network.

        Purpose: Validates that the continuous policy branch (Gaussian NLL)
            computes a finite loss.

        Given: A continuous BetaZero network and a batch with continuous action
            targets (mean values).
        When: compute_beta_zero_loss is called.
        Then: The policy_loss component is a finite number (not NaN) and the
            total loss has a valid gradient.

        Test type: unit
        """
        network = _make_continuous_network()
        beliefs = torch.randn(16, BELIEF_DIM)
        # Continuous policy targets: only the mean part is used from targets
        policies = torch.randn(16, 2 * ACTION_DIM)
        values = torch.randn(16)

        total_loss, loss_dict = compute_beta_zero_loss(network, beliefs, policies, values)

        policy_loss = loss_dict["policy_loss"]
        assert not np.isnan(policy_loss), "policy_loss should not be NaN"
        assert np.isfinite(policy_loss), "policy_loss should be finite"

        # Verify gradient flows
        total_loss.backward()
        for param in network.parameters():
            assert param.grad is not None, "all parameters should receive gradients"

    def test_value_loss_is_mse(self):
        """Verify the value loss component is the MSE between predictions and targets.

        Purpose: Validates that the value_loss key in the returned dictionary
            matches the MSE between the network's value predictions and the
            supplied value targets.

        Given: A discrete BetaZero network and a batch of known inputs.
        When: compute_beta_zero_loss is called, and the MSE is independently
            computed from the network's forward pass output.
        Then: The reported value_loss matches the independently computed MSE.

        Test type: unit
        """
        network = _make_discrete_network()
        beliefs = torch.randn(8, BELIEF_DIM)
        raw = torch.rand(8, N_ACTIONS)
        policies = raw / raw.sum(dim=-1, keepdim=True)
        values = torch.randn(8)

        _, loss_dict = compute_beta_zero_loss(network, beliefs, policies, values)

        # Independently compute MSE from the network's value predictions
        with torch.no_grad():
            _, value_pred = network(beliefs)
            value_pred = value_pred.squeeze(-1)
            expected_mse = torch.nn.functional.mse_loss(value_pred, values).item()

        assert loss_dict["value_loss"] == pytest.approx(
            expected_mse, abs=1e-5
        ), "value_loss should match independently computed MSE"


class TestTrainNetwork:
    """Tests for the train_network function."""

    @pytest.mark.slow
    def test_training_reduces_loss(self):
        """Verify that training on consistent data reduces the total loss.

        Purpose: Validates that the train_network function drives the loss
            downward when given a buffer of consistent training examples.

        Given: A discrete BetaZero network and a buffer filled with 500
            consistent (non-random) training examples mapping similar beliefs
            to a fixed policy and value target.
        When: train_network is called for 20 epochs with a small batch size.
        Then: The average total loss in the last epoch is strictly less than
            the average total loss in the first epoch.

        Test type: integration
        """
        torch.manual_seed(42)
        np.random.seed(42)

        network = _make_discrete_network()

        # Fill buffer with consistent data: constant policy + value for similar beliefs
        buffer = TrainingBuffer(n_buffer=1)
        buffer.begin_iteration()
        fixed_policy = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        fixed_value = 5.0

        for _ in range(500):
            belief = np.random.randn(BELIEF_DIM).astype(np.float32) * 0.1
            buffer.add(TrainingExample(belief, fixed_policy, fixed_value))

        metrics = train_network(
            network,
            buffer,
            n_epochs=20,
            batch_size=64,
            learning_rate=1e-3,
        )

        first_loss = metrics["total_loss"][0]
        last_loss = metrics["total_loss"][-1]

        assert (
            last_loss < first_loss
        ), f"Training should reduce loss: first={first_loss:.4f}, last={last_loss:.4f}"
