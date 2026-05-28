# SPDX-License-Identifier: MIT

"""Tests for BetaZeroNetwork dual-head neural network.

This module tests the BetaZeroNetwork class including construction, forward pass,
prediction, serialisation, training, and input validation for both discrete and
continuous action space types.
"""

import numpy as np
import pytest
import torch

from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    BetaZeroNetwork,
)

np.random.seed(42)
torch.manual_seed(42)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def discrete_net():
    return BetaZeroNetwork(
        belief_dim=4,
        action_space_type="discrete",
        n_actions=3,
        hidden_sizes=(64, 64),
    )


@pytest.fixture
def continuous_net():
    return BetaZeroNetwork(
        belief_dim=4,
        action_space_type="continuous",
        action_dim=2,
        hidden_sizes=(64, 64),
    )


# ── Tests ─────────────────────────────────────────────────────────────────


def test_discrete_policy_outputs_valid_probability_distribution(discrete_net):
    """Test that discrete network prediction produces a valid probability distribution.

    Purpose: Validates that the discrete policy head outputs non-negative
        probabilities that sum to approximately 1.

    Given: A BetaZeroNetwork configured for discrete action space with 3 actions.
    When: predict is called with a random belief feature vector.
    Then: The policy output contains only non-negative values and their sum is
        approximately 1.0.

    Test type: unit
    """
    belief = np.random.randn(4).astype(np.float32)
    policy, _ = discrete_net.predict(belief)

    assert policy.shape == (3,)
    assert np.all(policy >= 0.0), "All probabilities must be non-negative"
    assert np.isclose(
        policy.sum(), 1.0, atol=1e-5
    ), f"Probabilities must sum to 1, got {policy.sum()}"


def test_continuous_policy_outputs_mean_and_log_std(continuous_net):
    """Test that continuous network prediction outputs mean and log_std.

    Purpose: Validates that the continuous policy head outputs a vector of
        size 2 * action_dim containing concatenated mean and log_std.

    Given: A BetaZeroNetwork configured for continuous action space with action_dim=2.
    When: predict is called with a random belief feature vector.
    Then: The policy output has shape (4,) which equals 2 * action_dim.

    Test type: unit
    """
    belief = np.random.randn(4).astype(np.float32)
    policy, _ = continuous_net.predict(belief)

    expected_dim = 2 * 2  # 2 * action_dim
    assert policy.shape == (
        expected_dim,
    ), f"Continuous policy shape should be ({expected_dim},), got {policy.shape}"


def test_value_head_outputs_scalar(discrete_net):
    """Test that predict returns a scalar float value.

    Purpose: Validates that the value head produces a single Python float,
        not an array or tensor.

    Given: A BetaZeroNetwork configured for discrete action space.
    When: predict is called with a random belief feature vector.
    Then: The returned value is a Python float.

    Test type: unit
    """
    belief = np.random.randn(4).astype(np.float32)
    policy, value = discrete_net.predict(belief)

    assert isinstance(value, float), f"Value should be a Python float, got {type(value)}"


def test_forward_batch_shapes(discrete_net, continuous_net):
    """Test forward pass output shapes with a batch of inputs.

    Purpose: Validates that the forward method produces correctly shaped
        tensors for batched input.

    Given: Discrete and continuous BetaZeroNetwork instances.
    When: forward is called with a batch of 5 belief feature vectors of dim 4.
    Then: Discrete policy has shape (5, 3), continuous policy has shape (5, 4),
        and both value outputs have shape (5, 1).

    Test type: unit
    """
    batch = torch.randn(5, 4)

    d_policy, d_value = discrete_net.forward(batch)
    assert d_policy.shape == (
        5,
        3,
    ), f"Discrete policy batch shape should be (5, 3), got {d_policy.shape}"
    assert d_value.shape == (5, 1), f"Value batch shape should be (5, 1), got {d_value.shape}"

    c_policy, c_value = continuous_net.forward(batch)
    assert c_policy.shape == (
        5,
        4,
    ), f"Continuous policy batch shape should be (5, 4), got {c_policy.shape}"
    assert c_value.shape == (5, 1), f"Value batch shape should be (5, 1), got {c_value.shape}"


def test_predict_returns_numpy(discrete_net):
    """Test that predict returns numpy array and float.

    Purpose: Validates the return types of the predict method, ensuring
        policy is a numpy ndarray and value is a Python float.

    Given: A BetaZeroNetwork configured for discrete action space.
    When: predict is called with a numpy belief feature vector.
    Then: The policy output is an np.ndarray and the value is a float.

    Test type: unit
    """
    belief = np.random.randn(4).astype(np.float32)
    policy, value = discrete_net.predict(belief)

    assert isinstance(policy, np.ndarray), f"Policy should be np.ndarray, got {type(policy)}"
    assert isinstance(value, float), f"Value should be float, got {type(value)}"


def test_save_load_weights_roundtrip(tmp_path, discrete_net):
    """Test that saving and loading weights preserves network predictions.

    Purpose: Validates that the save_weights / load_weights roundtrip
        produces a network that gives identical predictions.

    Given: A BetaZeroNetwork with random initial weights.
    When: Weights are saved to a file, then loaded into a fresh network
        with the same architecture.
    Then: Both networks produce identical policy and value outputs for the
        same input.

    Test type: unit
    """
    belief = np.zeros(4, dtype=np.float32)
    policy_before, value_before = discrete_net.predict(belief)

    filepath = tmp_path / "weights.pt"
    discrete_net.save_weights(filepath)

    loaded_net = BetaZeroNetwork(
        belief_dim=4,
        action_space_type="discrete",
        n_actions=3,
        hidden_sizes=(64, 64),
    )
    loaded_net.load_weights(filepath)

    policy_after, value_after = loaded_net.predict(belief)

    np.testing.assert_array_almost_equal(
        policy_before,
        policy_after,
        err_msg="Policy outputs differ after save/load roundtrip",
    )
    assert np.isclose(
        value_before, value_after, atol=1e-7
    ), f"Value outputs differ after save/load: {value_before} vs {value_after}"


def test_loss_decreases_over_gradient_steps(discrete_net):
    """Test that training on fixed data reduces the loss.

    Purpose: Validates that the network can learn by verifying the loss
        decreases over multiple gradient steps on a fixed mini-batch.

    Given: A discrete BetaZeroNetwork, a fixed batch of belief features,
        target action probabilities, and target values.
    When: The network is trained for 10 gradient steps using MSE loss for
        value and KL-divergence loss for policy.
    Then: The final loss is strictly lower than the initial loss.

    Test type: unit
    """
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(discrete_net.parameters(), lr=1e-3)

    beliefs = torch.randn(8, 4)
    target_actions = torch.zeros(8, dtype=torch.long)
    target_values = torch.ones(8, 1)

    losses = []
    for _ in range(10):
        log_policy, values = discrete_net.forward(beliefs)
        policy_loss = torch.nn.functional.nll_loss(log_policy, target_actions)
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert (
        losses[-1] < losses[0]
    ), f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


def test_invalid_action_space_type_raises():
    """Test that an invalid action_space_type raises ValueError.

    Purpose: Validates input validation for the action_space_type parameter.

    Given: An action_space_type string that is neither "discrete" nor "continuous".
    When: BetaZeroNetwork is instantiated with action_space_type="invalid".
    Then: A ValueError is raised with a descriptive message.

    Test type: unit
    """
    with pytest.raises(ValueError, match="action_space_type must be"):
        BetaZeroNetwork(
            belief_dim=4,
            action_space_type="invalid",
            n_actions=3,
        )


def test_discrete_missing_n_actions_raises():
    """Test that discrete action space without n_actions raises ValueError.

    Purpose: Validates that the required n_actions parameter is enforced
        for discrete action space configuration.

    Given: action_space_type is "discrete" and n_actions is not provided.
    When: BetaZeroNetwork is instantiated without n_actions.
    Then: A ValueError is raised indicating n_actions is required.

    Test type: unit
    """
    with pytest.raises(ValueError, match="n_actions is required"):
        BetaZeroNetwork(
            belief_dim=4,
            action_space_type="discrete",
        )
