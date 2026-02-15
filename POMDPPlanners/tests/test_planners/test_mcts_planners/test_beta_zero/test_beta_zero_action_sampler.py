"""Tests for BetaZeroActionSampler network-guided action sampling.

This module tests the BetaZeroActionSampler class including fallback behaviour,
discrete and continuous network-guided sampling, and pickle serialisation.
"""

# pylint: disable=protected-access  # Tests need to verify internal state

import pickle

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
    BetaZeroActionSampler,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    BetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    ParticleMeanStdRepresentation,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

np.random.seed(42)


# ── Helpers ──────────────────────────────────────────────────────────────


class SimpleFallbackSampler(ActionSampler):
    def sample(self, belief_node=None):  # noqa: ARG002
        return "fallback_action"


def _make_belief_node(state_dim=2, n_particles=10):
    particles = [np.random.randn(state_dim).tolist() for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    belief = WeightedParticleBelief(particles, log_weights)
    return BeliefNode(belief=belief)


# ── Tests ────────────────────────────────────────────────────────────────


def test_fallback_without_belief_node():
    """Test that the fallback sampler is used when belief_node is None.

    Purpose: Validates that BetaZeroActionSampler delegates to the fallback
        sampler when no belief node context is provided.

    Given: A BetaZeroActionSampler with a SimpleFallbackSampler and no
        network/representation attached.
    When: sample is called with belief_node=None.
    Then: The returned action equals "fallback_action" from the fallback sampler.

    Test type: unit
    """
    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=["a", "b", "c"])

    result = sampler.sample(belief_node=None)

    assert (
        result == "fallback_action"
    ), f"Expected 'fallback_action' when belief_node is None, got {result}"


def test_fallback_without_network():
    """Test that the fallback sampler is used when the network is not set.

    Purpose: Validates that BetaZeroActionSampler delegates to the fallback
        sampler when set_network_and_representation has not been called, even
        if a valid belief node is provided.

    Given: A BetaZeroActionSampler without network/representation and a valid
        belief node.
    When: sample is called with the belief node.
    Then: The returned action equals "fallback_action" from the fallback sampler.

    Test type: unit
    """
    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=["a", "b", "c"])
    belief_node = _make_belief_node(state_dim=2)

    result = sampler.sample(belief_node=belief_node)

    assert (
        result == "fallback_action"
    ), f"Expected 'fallback_action' when network is not set, got {result}"


def test_discrete_sampling_follows_policy():
    """Test that discrete sampling produces a non-uniform distribution guided by the network.

    Purpose: Validates that when a network and belief representation are attached,
        the sampler draws actions according to the network's softmax policy rather
        than uniformly at random.

    Given: A BetaZeroActionSampler with a discrete BetaZeroNetwork, a
        ParticleMeanStdRepresentation, and a list of three actions.
    When: sample is called 100 times with a valid belief node.
    Then: The distribution of sampled actions is non-uniform. Specifically, the
        most frequently sampled action is selected more often than 1/3 of the time
        (the uniform expectation), indicating the network policy influences sampling.

    Test type: unit
    """
    np.random.seed(42)
    state_dim = 2
    actions = ["left", "right", "stay"]
    belief_dim = 2 * state_dim  # ParticleMeanStdRepresentation outputs 2*state_dim

    network = BetaZeroNetwork(
        belief_dim=belief_dim,
        action_space_type="discrete",
        n_actions=len(actions),
        hidden_sizes=(32, 32),
    )
    representation = ParticleMeanStdRepresentation(state_dim=state_dim)

    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=actions, noise_scale=0.1)
    sampler.set_network_and_representation(network, representation)

    belief_node = _make_belief_node(state_dim=state_dim)

    counts = {a: 0 for a in actions}
    n_samples = 100
    for _ in range(n_samples):
        action = sampler.sample(belief_node=belief_node)
        assert action in actions, f"Sampled action '{action}' not in action list"
        counts[action] += 1

    max_count = max(counts.values())
    min_count = min(counts.values())
    assert max_count > min_count, (
        f"Expected non-uniform distribution from network policy, "
        f"but all actions were sampled equally: {counts}"
    )


def test_continuous_sampling_centered_on_predicted_mean():
    """Test that continuous samples are in a reasonable range around the network mean.

    Purpose: Validates that for a continuous action space, the sampler produces
        action vectors whose components are within a plausible range of the
        network's predicted mean (not wildly divergent).

    Given: A BetaZeroActionSampler with a continuous BetaZeroNetwork (action_dim=2),
        a ParticleMeanStdRepresentation, and no discrete actions list.
    When: sample is called 50 times with a valid belief node.
    Then: All samples are finite numpy arrays of the correct shape (action_dim,),
        and the empirical mean of samples is within 5 standard deviations of
        the network's predicted mean.

    Test type: unit
    """
    np.random.seed(42)
    state_dim = 2
    action_dim = 2
    belief_dim = 2 * state_dim

    network = BetaZeroNetwork(
        belief_dim=belief_dim,
        action_space_type="continuous",
        action_dim=action_dim,
        hidden_sizes=(32, 32),
    )
    representation = ParticleMeanStdRepresentation(state_dim=state_dim)

    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=None, noise_scale=0.1)
    sampler.set_network_and_representation(network, representation)

    belief_node = _make_belief_node(state_dim=state_dim)

    # Get the network's predicted mean for reference
    features = representation(belief_node.belief)
    policy_output, _ = network.predict(features)
    predicted_mean = policy_output[:action_dim]

    samples = []
    n_samples = 50
    for _ in range(n_samples):
        action = sampler.sample(belief_node=belief_node)
        assert isinstance(
            action, np.ndarray
        ), f"Expected np.ndarray for continuous action, got {type(action)}"
        assert action.shape == (
            action_dim,
        ), f"Expected action shape ({action_dim},), got {action.shape}"
        assert np.all(np.isfinite(action)), f"Action contains non-finite values: {action}"
        samples.append(action)

    samples_array = np.array(samples)
    empirical_mean = samples_array.mean(axis=0)
    empirical_std = samples_array.std(axis=0)

    # The empirical mean should be reasonably close to the predicted mean
    deviation = np.abs(empirical_mean - predicted_mean)
    tolerance = 5.0 * (empirical_std + 0.1)
    assert np.all(deviation < tolerance), (
        f"Empirical mean {empirical_mean} deviates too much from predicted mean "
        f"{predicted_mean}. Deviation: {deviation}, tolerance: {tolerance}"
    )


def test_pickle_serialization():
    """Test that BetaZeroActionSampler serialisation strips the network.

    Purpose: Validates that __getstate__ removes the network and belief
        representation, that pickle.dumps succeeds, and that a manual
        reconstruction via __setstate__ restores a working sampler that
        falls back correctly.

    Given: A BetaZeroActionSampler with network and representation attached.
    When: __getstate__ is called (as pickle.dumps does internally), the state
        is inspected, and a fresh instance is reconstructed via __setstate__.
    Then: The serialised state has _network and _belief_representation set to
        None, pickle.dumps succeeds, and the reconstructed sampler retains
        the fallback sampler, actions list, and noise_scale, and delegates
        to the fallback when sampled without re-attaching a network.

    Test type: unit
    """
    state_dim = 2
    actions = ["a", "b", "c"]
    belief_dim = 2 * state_dim

    network = BetaZeroNetwork(
        belief_dim=belief_dim,
        action_space_type="discrete",
        n_actions=len(actions),
        hidden_sizes=(32, 32),
    )
    representation = ParticleMeanStdRepresentation(state_dim=state_dim)

    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=actions, noise_scale=0.25)
    sampler.set_network_and_representation(network, representation)

    # Verify __getstate__ strips network and representation
    state = sampler.__getstate__()
    assert state["_network"] is None, "Network should be None in serialised state"
    assert (
        state["_belief_representation"] is None
    ), "Belief representation should be None in serialised state"

    # Verify pickle.dumps succeeds (serialisation direction works)
    data = pickle.dumps(sampler)
    assert len(data) > 0, "pickle.dumps should produce non-empty bytes"

    # Reconstruct via object.__new__ + __setstate__ (mirrors pickle protocol)
    restored = object.__new__(BetaZeroActionSampler)
    restored.__setstate__(state)

    assert restored._network is None, "Network should be None after deserialization"
    assert (
        restored._belief_representation is None
    ), "Belief representation should be None after deserialization"
    assert restored.actions == actions, f"Actions should be preserved, got {restored.actions}"
    assert (
        restored.noise_scale == 0.25
    ), f"noise_scale should be preserved, got {restored.noise_scale}"

    # Without network, restored sampler should use fallback
    belief_node = _make_belief_node(state_dim=state_dim)
    result = restored.sample(belief_node=belief_node)
    assert (
        result == "fallback_action"
    ), f"Restored sampler without network should use fallback, got {result}"


def test_pickle_round_trip():
    """Test full pickle.dumps → pickle.loads round trip for BetaZeroActionSampler.

    Purpose: Validates that BetaZeroActionSampler can be successfully pickled
        and unpickled using the full pickle protocol, which is critical for
        joblib/multiprocessing compatibility.

    Given: A BetaZeroActionSampler with network and representation attached.
    When: The sampler is pickled with pickle.dumps and then unpickled with
        pickle.loads (simulating joblib/multiprocessing serialization).
    Then: The unpickled sampler retains all non-network attributes (fallback,
        actions, noise_scale), has network and representation set to None,
        and correctly delegates to fallback when used.

    Test type: unit
    """
    state_dim = 2
    actions = ["a", "b", "c"]
    belief_dim = 2 * state_dim

    network = BetaZeroNetwork(
        belief_dim=belief_dim,
        action_space_type="discrete",
        n_actions=len(actions),
        hidden_sizes=(32, 32),
    )
    representation = ParticleMeanStdRepresentation(state_dim=state_dim)

    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=actions, noise_scale=0.25)
    sampler.set_network_and_representation(network, representation)

    # Full pickle round trip (what joblib/multiprocessing actually does)
    pickled_data = pickle.dumps(sampler)
    restored = pickle.loads(pickled_data)

    # Verify attributes are preserved
    assert restored._network is None, "Network should be None after pickle.loads"
    assert (
        restored._belief_representation is None
    ), "Belief representation should be None after pickle.loads"
    assert restored.actions == actions, f"Actions should be preserved, got {restored.actions}"
    assert (
        restored.noise_scale == 0.25
    ), f"noise_scale should be preserved, got {restored.noise_scale}"
    assert isinstance(
        restored.fallback_sampler, SimpleFallbackSampler
    ), "Fallback sampler should be preserved and of correct type"

    # Verify functionality: without network, should use fallback
    belief_node = _make_belief_node(state_dim=state_dim)
    result = restored.sample(belief_node=belief_node)
    assert (
        result == "fallback_action"
    ), f"Restored sampler should use fallback after pickle.loads, got {result}"


def test_pickle_round_trip_continuous():
    """Test full pickle round trip for BetaZeroActionSampler with continuous actions.

    Purpose: Validates that BetaZeroActionSampler for continuous action spaces
        can be successfully pickled and unpickled.

    Given: A BetaZeroActionSampler configured for continuous actions with
        network and representation attached.
    When: The sampler is pickled and unpickled via pickle.dumps/loads.
    Then: The unpickled sampler preserves all attributes and functions correctly.

    Test type: unit
    """
    state_dim = 2
    action_dim = 2
    belief_dim = 2 * state_dim

    network = BetaZeroNetwork(
        belief_dim=belief_dim,
        action_space_type="continuous",
        action_dim=action_dim,
        hidden_sizes=(32, 32),
    )
    representation = ParticleMeanStdRepresentation(state_dim=state_dim)

    fallback = SimpleFallbackSampler()
    sampler = BetaZeroActionSampler(fallback_sampler=fallback, actions=None, noise_scale=0.15)
    sampler.set_network_and_representation(network, representation)

    # Full pickle round trip
    pickled_data = pickle.dumps(sampler)
    restored = pickle.loads(pickled_data)

    # Verify attributes
    assert restored._network is None, "Network should be None after pickle.loads"
    assert (
        restored._belief_representation is None
    ), "Belief representation should be None after pickle.loads"
    assert restored.actions is None, "Actions should be None for continuous space"
    assert (
        restored.noise_scale == 0.15
    ), f"noise_scale should be preserved, got {restored.noise_scale}"

    # Verify functionality
    belief_node = _make_belief_node(state_dim=state_dim)
    result = restored.sample(belief_node=belief_node)
    assert result == "fallback_action", f"Restored sampler should use fallback, got {result}"
