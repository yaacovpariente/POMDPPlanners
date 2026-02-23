"""Tests for the ConstrainedZero 3-head neural network.

This module tests the ConstrainedZeroNetwork, which extends BetaZeroNetwork
with an additional failure probability head.
"""

import random

import numpy as np
import pytest
import torch

from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    BetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


@pytest.fixture
def discrete_net():
    return ConstrainedZeroNetwork(
        belief_dim=4,
        action_space_type="discrete",
        n_actions=3,
        hidden_sizes=(64, 64),
    )


@pytest.fixture
def continuous_net():
    return ConstrainedZeroNetwork(
        belief_dim=4,
        action_space_type="continuous",
        action_dim=2,
        hidden_sizes=(64, 64),
    )


class TestConstrainedZeroNetwork:
    """Tests for the ConstrainedZeroNetwork class."""

    def test_inherits_from_beta_zero_network(self, discrete_net):
        """Test that ConstrainedZeroNetwork inherits from BetaZeroNetwork.

        Purpose: Validates that the 3-head network is a proper subclass.

        Given: A ConstrainedZeroNetwork instance.
        When: Checking isinstance.
        Then: It is an instance of BetaZeroNetwork.

        Test type: unit
        """
        assert isinstance(discrete_net, BetaZeroNetwork)

    def test_has_failure_head(self, discrete_net):
        """Test that the network has a failure head attribute.

        Purpose: Validates the failure head is built during construction.

        Given: A ConstrainedZeroNetwork instance.
        When: Accessing the failure_head attribute.
        Then: It exists and is an nn.Sequential module.

        Test type: unit
        """
        assert hasattr(discrete_net, "failure_head")
        assert isinstance(discrete_net.failure_head, torch.nn.Sequential)

    def test_forward_returns_three_tensors_discrete(self, discrete_net):
        """Test forward pass returns 3 tensors for discrete action space.

        Purpose: Validates forward() output shape and count.

        Given: A discrete ConstrainedZeroNetwork with belief_dim=4, n_actions=3.
        When: forward() is called with a batch of 5 inputs.
        Then: Returns 3 tensors: policy (5,3), value (5,1), failure (5,1).

        Test type: unit
        """
        x = torch.randn(5, 4)
        result = discrete_net(x)
        assert len(result) == 3
        log_policy, value, failure_logit = result
        assert log_policy.shape == (5, 3)
        assert value.shape == (5, 1)
        assert failure_logit.shape == (5, 1)

    def test_forward_returns_three_tensors_continuous(self, continuous_net):
        """Test forward pass returns 3 tensors for continuous action space.

        Purpose: Validates forward() output shape for continuous actions.

        Given: A continuous ConstrainedZeroNetwork with belief_dim=4, action_dim=2.
        When: forward() is called with a batch of 5 inputs.
        Then: Returns 3 tensors: policy (5,4), value (5,1), failure (5,1).

        Test type: unit
        """
        x = torch.randn(5, 4)
        result = continuous_net(x)
        assert len(result) == 3
        log_policy, value, failure_logit = result
        assert log_policy.shape == (5, 4)  # 2 * action_dim
        assert value.shape == (5, 1)
        assert failure_logit.shape == (5, 1)

    def test_predict_returns_three_values_discrete(self, discrete_net):
        """Test predict returns 3-tuple for discrete action space.

        Purpose: Validates predict() output types and shapes.

        Given: A discrete ConstrainedZeroNetwork with belief_dim=4, n_actions=3.
        When: predict() is called with a single input vector.
        Then: Returns (policy_ndarray shape (3,), float value, float failure_prob).

        Test type: unit
        """
        features = np.zeros(4, dtype=np.float32)
        result = discrete_net.predict(features)
        assert len(result) == 3
        policy, value, failure_prob = result
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (3,)
        assert isinstance(value, float)
        assert isinstance(failure_prob, float)

    def test_predict_returns_three_values_continuous(self, continuous_net):
        """Test predict returns 3-tuple for continuous action space.

        Purpose: Validates predict() output for continuous actions.

        Given: A continuous ConstrainedZeroNetwork with belief_dim=4, action_dim=2.
        When: predict() is called with a single input vector.
        Then: Returns (policy_ndarray shape (4,), float value, float failure_prob).

        Test type: unit
        """
        features = np.zeros(4, dtype=np.float32)
        result = continuous_net.predict(features)
        assert len(result) == 3
        policy, value, failure_prob = result
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (4,)
        assert isinstance(value, float)
        assert isinstance(failure_prob, float)

    def test_failure_prob_in_zero_one(self, discrete_net):
        """Test failure probability is in [0, 1] after sigmoid.

        Purpose: Validates sigmoid activation is applied to failure output.

        Given: A ConstrainedZeroNetwork.
        When: predict() is called with various inputs.
        Then: failure_prob is always in [0, 1].

        Test type: unit
        """
        for _ in range(20):
            features = np.random.randn(4).astype(np.float32)
            _, _, failure_prob = discrete_net.predict(features)
            assert 0.0 <= failure_prob <= 1.0

    def test_discrete_policy_sums_to_one(self, discrete_net):
        """Test discrete policy output sums to 1.

        Purpose: Validates policy probabilities are properly normalized.

        Given: A discrete ConstrainedZeroNetwork.
        When: predict() is called.
        Then: Policy probabilities sum to 1.

        Test type: unit
        """
        features = np.random.randn(4).astype(np.float32)
        policy, _, _ = discrete_net.predict(features)
        np.testing.assert_allclose(policy.sum(), 1.0, atol=1e-5)

    def test_save_load_weights(self, discrete_net, tmp_path):
        """Test save and load preserves network weights.

        Purpose: Validates serialization roundtrip.

        Given: A ConstrainedZeroNetwork with random weights.
        When: Weights are saved and loaded into a new network.
        Then: The loaded network produces identical outputs.

        Test type: unit
        """
        features = np.zeros(4, dtype=np.float32)
        original_output = discrete_net.predict(features)

        filepath = tmp_path / "weights.pt"
        discrete_net.save_weights(filepath)

        loaded = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
        )
        loaded.load_weights(filepath)
        loaded_output = loaded.predict(features)

        np.testing.assert_allclose(original_output[0], loaded_output[0], atol=1e-6)
        assert abs(original_output[1] - loaded_output[1]) < 1e-6
        assert abs(original_output[2] - loaded_output[2]) < 1e-6

    def test_forward_single_sample(self, discrete_net):
        """Test forward works with unbatched input.

        Purpose: Validates forward() handles single (belief_dim,) input.

        Given: A ConstrainedZeroNetwork.
        When: forward() is called with a 1D tensor.
        Then: Returns valid outputs without error.

        Test type: unit
        """
        x = torch.randn(4)
        log_policy, value, failure_logit = discrete_net(x)
        assert log_policy.shape == (3,)
        assert value.shape == (1,)
        assert failure_logit.shape == (1,)

    def test_dropout_applied_in_train_mode(self):
        """Test that dropout produces stochastic outputs in training mode.

        Purpose: Validates that use_dropout=True causes non-deterministic trunk
            outputs during training, confirming dropout is active.

        Given: A ConstrainedZeroNetwork with use_dropout=True, p_dropout=0.5 in train mode.
        When: Twenty forward passes are run with the same input tensor.
        Then: At least two outputs differ (dropout is stochastic in train mode).

        Test type: unit
        """
        net = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
            use_dropout=True,
            p_dropout=0.5,
        )
        net.train()
        x = torch.ones(1, 4)
        outputs = []
        for _ in range(20):
            log_policy, _, _ = net(x)
            outputs.append(log_policy.detach().clone())
        any_differ = any(not torch.allclose(outputs[0], outputs[i]) for i in range(1, len(outputs)))
        assert any_differ, "Expected stochastic outputs with dropout in train mode"

    def test_no_dropout_in_eval_mode(self):
        """Test that predict() produces deterministic outputs (eval mode disables dropout).

        Purpose: Validates that predict() switches to eval mode so dropout is
            inactive, yielding identical results for repeated calls.

        Given: A ConstrainedZeroNetwork with use_dropout=True.
        When: predict() is called twice with the same input.
        Then: Both outputs are identical.

        Test type: unit
        """
        net = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
            use_dropout=True,
            p_dropout=0.5,
        )
        features = np.ones(4, dtype=np.float32)
        policy1, value1, fp1 = net.predict(features)
        policy2, value2, fp2 = net.predict(features)
        np.testing.assert_array_equal(policy1, policy2)
        assert value1 == value2
        assert fp1 == fp2

    def test_use_dropout_false_disables_dropout(self):
        """Test that use_dropout=False yields deterministic outputs in train mode.

        Purpose: Validates that setting use_dropout=False removes all dropout
            layers from the trunk, making forward passes deterministic.

        Given: A ConstrainedZeroNetwork with use_dropout=False in train mode.
        When: Two forward passes are run with the same input.
        Then: Outputs are identical (no dropout regardless of training mode).

        Test type: unit
        """
        net = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
            use_dropout=False,
        )
        net.train()
        x = torch.ones(1, 4)
        log_policy1, _, _ = net(x)
        log_policy2, _, _ = net(x)
        assert torch.allclose(
            log_policy1, log_policy2
        ), "Expected identical outputs with use_dropout=False in train mode"

    def test_predict_restores_training_mode(self):
        """Test that predict() restores training mode after inference.

        Purpose: Validates that predict() restores the network to train mode
            after temporarily switching to eval mode for dropout suppression.

        Given: A ConstrainedZeroNetwork put into train mode.
        When: predict() is called.
        Then: network.training is True after the call.

        Test type: unit
        """
        net = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
            use_dropout=True,
        )
        net.train()
        features = np.zeros(4, dtype=np.float32)
        net.predict(features)
        assert net.training, "Expected network to be in train mode after predict()"

    def test_predict_batch_restores_training_mode(self):
        """Test that predict_batch() restores training mode after inference.

        Purpose: Validates that predict_batch() restores the network to train
            mode after temporarily switching to eval mode for dropout suppression.

        Given: A ConstrainedZeroNetwork put into train mode.
        When: predict_batch() is called.
        Then: network.training is True after the call.

        Test type: unit
        """
        net = ConstrainedZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
            n_actions=3,
            hidden_sizes=(64, 64),
            use_dropout=True,
        )
        net.train()
        features_batch = np.zeros((3, 4), dtype=np.float32)
        net.predict_batch(features_batch)
        assert net.training, "Expected network to be in train mode after predict_batch()"
