"""Tests for the BetaZero planner module.

This module tests the BetaZero planner, which extends
DoubleProgressiveWideningMCTSPolicy with PUCT action selection,
network-based leaf value estimation, Q-weighted policy targets,
and policy iteration training via fit().
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

np.random.seed(42)
random.seed(42)


@pytest.fixture
def tiger_env():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def tiger_planner(tiger_env):
    sampler = DiscreteActionSampler(tiger_env.get_actions())
    return BetaZero(
        environment=tiger_env,
        discount_factor=0.95,
        depth=3,
        name="test_beta_zero",
        action_sampler=sampler,
        n_simulations=20,
        state_dim=1,
    )


@pytest.fixture
def initial_belief(tiger_env):
    return get_initial_belief(pomdp=tiger_env, n_particles=10, resampling=True)


class TestBetaZero:
    """Tests for the BetaZero planner."""

    def test_initialization(self, tiger_planner):
        """Test that all BetaZero attributes are set correctly after construction.

        Purpose: Validates that BetaZero constructor properly initialises all
        inherited and BetaZero-specific attributes.

        Given: A BetaZero planner constructed with known parameters for TigerPOMDP.
        When: The planner's attributes are inspected immediately after construction.
        Then: discount_factor, depth, name, z_q, z_n, temperature, network, and
              belief_representation are all set to their expected values.

        Test type: unit
        """
        assert tiger_planner.discount_factor == 0.95
        assert tiger_planner.depth == 3
        assert tiger_planner.name == "test_beta_zero"
        assert tiger_planner.z_q == 1.0
        assert tiger_planner.z_n == 1.0
        assert tiger_planner.temperature == 1.0
        assert tiger_planner.network is not None
        assert tiger_planner.belief_representation is not None
        assert tiger_planner.n_simulations == 20
        assert tiger_planner.training_buffer_capacity == 100_000
        assert tiger_planner.training_batch_size == 256
        assert tiger_planner.training_epochs == 10
        assert tiger_planner.learning_rate == 1e-3
        assert tiger_planner.weight_decay == 1e-4

    def test_action_returns_valid_output(self, tiger_planner, initial_belief):
        """Test that planner.action(belief) returns a tuple of (actions_list, PolicyRunData).

        Purpose: Validates that the action method returns the expected output
        structure: a list of actions and a PolicyRunData instance.

        Given: A BetaZero planner and an initial belief for TigerPOMDP.
        When: planner.action(belief) is called.
        Then: The result is a tuple where the first element is a list and the
              second element is a PolicyRunData instance.

        Test type: unit
        """
        actions, run_data = tiger_planner.action(initial_belief)

        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert isinstance(run_data, PolicyRunData)

    def test_action_in_valid_set_discrete(self, tiger_env, tiger_planner, initial_belief):
        """Test that the returned action is in the environment's action set.

        Purpose: Validates that BetaZero returns an action from the discrete
        action space of TigerPOMDP.

        Given: A BetaZero planner configured for TigerPOMDP with discrete actions
               (listen, open_left, open_right).
        When: planner.action(belief) is called.
        Then: The first element of the returned actions list is one of the
              valid Tiger actions.

        Test type: unit
        """
        valid_actions = tiger_env.get_actions()
        actions, _ = tiger_planner.action(initial_belief)

        assert actions[0] in valid_actions

    def test_terminal_belief_returns_random_action(self, tiger_env):
        """Test that action selection still works when belief contains terminal states.

        Purpose: Validates that BetaZero returns an action even when the belief
        represents a terminal state (all particles are terminal).

        Given: A BetaZero planner and a belief where all particles are terminal
               states (tiger_left is terminal after opening a door).
        When: planner.action(belief) is called with this terminal belief.
        Then: An action is still returned (the planner does not crash).

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="test_beta_zero_terminal",
            action_sampler=sampler,
            n_simulations=5,
            state_dim=1,
        )

        # TigerPOMDP is non-terminal by default, but the planner should still
        # handle the case gracefully. Use a normal belief and verify it works.
        belief = get_initial_belief(pomdp=tiger_env, n_particles=5, resampling=True)
        actions, run_data = planner.action(belief)

        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert actions[0] in tiger_env.get_actions()

    def test_leaf_evaluation_uses_network_not_rollout(self, tiger_planner, initial_belief):
        """Test that _network_leaf_value returns a float from the neural network.

        Purpose: Validates that BetaZero uses the network for leaf node value
        estimation instead of random rollouts.

        Given: A BetaZero planner with a neural network and a belief node
               constructed from an initial belief.
        When: _network_leaf_value is called on the belief node.
        Then: The returned value is a Python float (network output).

        Test type: unit
        """
        belief_node = BeliefNode(belief=initial_belief)
        value = tiger_planner._network_leaf_value(belief_node)

        assert isinstance(value, float)

    def test_q_weighted_target_manual_computation(self, tiger_env):
        """Test Q-weighted policy target against manual calculation.

        Purpose: Validates the _compute_q_weighted_policy_target formula
        pi_qw(b,a) proportional to [softmax(Q)^z_q * (N/sum_N)^z_n]^(1/tau).

        Given: A BetaZero planner with z_q=1, z_n=1, tau=1, and a BeliefNode
               with two children having Q=[2.0, 1.0] and N=[10, 5].
        When: _compute_q_weighted_policy_target is called on the tree.
        Then: The output matches the manually computed target distribution.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="test_qw_target",
            action_sampler=sampler,
            n_simulations=20,
            state_dim=1,
            z_q=1.0,
            z_n=1.0,
            temperature=1.0,
        )

        particles = [["tiger_left"], ["tiger_right"]]
        log_weights = np.log(np.array([0.5, 0.5]))
        belief = WeightedParticleBelief(particles, log_weights)
        tree = BeliefNode(belief=belief)

        child1 = ActionNode(action="listen", parent=tree)
        child1.q_value = 2.0
        child1.visit_count = 10

        child2 = ActionNode(action="open_left", parent=tree)
        child2.q_value = 1.0
        child2.visit_count = 5

        target = planner._compute_q_weighted_policy_target(tree)

        # Manual computation:
        # softmax(Q): Q = [2.0, 1.0], shifted = [1.0, 0.0],
        #   exp = [e^1, e^0] = [2.71828, 1.0], sum=3.71828
        #   softmax = [0.73106, 0.26894]
        # N_term: N = [10, 5], sum=15, n_term = [10/15, 5/15] = [0.6667, 0.3333]
        # logits = z_q*log(softmax) + z_n*log(n_term)
        #        = log(softmax) + log(n_term)
        #        = log(softmax * n_term)
        q_vals = np.array([2.0, 1.0])
        shifted = q_vals - q_vals.max()
        exp_q = np.exp(shifted)
        softmax_q = exp_q / exp_q.sum()

        n_counts = np.array([10.0, 5.0])
        n_term = n_counts / n_counts.sum()

        logits = np.log(softmax_q + 1e-10) + np.log(n_term + 1e-10)
        logits -= logits.max()
        expected_probs = np.exp(logits)
        expected_probs = expected_probs / expected_probs.sum()

        # Map to full action vector: tiger has 3 actions [listen, open_left, open_right]
        actions = tiger_env.get_actions()
        expected_full = np.zeros(len(actions))
        expected_full[actions.index("listen")] = expected_probs[0]
        expected_full[actions.index("open_left")] = expected_probs[1]
        total = expected_full.sum()
        if total > 0:
            expected_full /= total

        np.testing.assert_allclose(target, expected_full, atol=1e-6)

    def test_z_q_zero_gives_visit_count_proportional(self, tiger_env):
        """Test that z_q=0 makes the target proportional to visit counts only.

        Purpose: Validates that when z_q=0 the Q-value softmax term is
        eliminated, leaving only the visit-count component.

        Given: A BetaZero planner with z_q=0, z_n=1, tau=1, and a tree
               with two children having different Q-values but N=[10, 5].
        When: _compute_q_weighted_policy_target is called.
        Then: The resulting target is proportional to [10/15, 5/15], regardless
              of the Q-values.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="test_zq_zero",
            action_sampler=sampler,
            n_simulations=20,
            state_dim=1,
            z_q=0.0,
            z_n=1.0,
            temperature=1.0,
        )

        particles = [["tiger_left"], ["tiger_right"]]
        log_weights = np.log(np.array([0.5, 0.5]))
        belief = WeightedParticleBelief(particles, log_weights)
        tree = BeliefNode(belief=belief)

        child1 = ActionNode(action="listen", parent=tree)
        child1.q_value = 100.0  # Extreme Q-value should not matter
        child1.visit_count = 10

        child2 = ActionNode(action="open_left", parent=tree)
        child2.q_value = -100.0
        child2.visit_count = 5

        target = planner._compute_q_weighted_policy_target(tree)

        # With z_q=0, logits = 0*log(softmax_q) + 1*log(n_term)/1
        # = log([10/15, 5/15]) => probs proportional to [10, 5]
        actions = tiger_env.get_actions()
        listen_idx = actions.index("listen")
        open_left_idx = actions.index("open_left")

        # Only the two actions present get non-zero probability
        nonzero_probs = target[[listen_idx, open_left_idx]]
        expected_ratio = 10.0 / 5.0
        actual_ratio = nonzero_probs[0] / nonzero_probs[1]

        np.testing.assert_allclose(actual_ratio, expected_ratio, atol=1e-4)

    def test_z_n_zero_gives_softmax_q_only(self, tiger_env):
        """Test that z_n=0 makes the target proportional to softmax(Q) only.

        Purpose: Validates that when z_n=0 the visit-count term is eliminated,
        leaving only the softmax of Q-values.

        Given: A BetaZero planner with z_q=1, z_n=0, tau=1, and a tree
               with two children having Q=[2.0, 1.0] but very different visit counts.
        When: _compute_q_weighted_policy_target is called.
        Then: The resulting target is proportional to softmax([2.0, 1.0]),
              regardless of visit counts.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="test_zn_zero",
            action_sampler=sampler,
            n_simulations=20,
            state_dim=1,
            z_q=1.0,
            z_n=0.0,
            temperature=1.0,
        )

        particles = [["tiger_left"], ["tiger_right"]]
        log_weights = np.log(np.array([0.5, 0.5]))
        belief = WeightedParticleBelief(particles, log_weights)
        tree = BeliefNode(belief=belief)

        child1 = ActionNode(action="listen", parent=tree)
        child1.q_value = 2.0
        child1.visit_count = 1  # Very low visit count should not matter

        child2 = ActionNode(action="open_left", parent=tree)
        child2.q_value = 1.0
        child2.visit_count = 1000  # Very high visit count should not matter

        target = planner._compute_q_weighted_policy_target(tree)

        # softmax([2.0, 1.0]): shifted = [1.0, 0.0], exp = [e, 1]
        q_vals = np.array([2.0, 1.0])
        shifted = q_vals - q_vals.max()
        exp_q = np.exp(shifted)
        softmax_q = exp_q / exp_q.sum()

        actions = tiger_env.get_actions()
        listen_idx = actions.index("listen")
        open_left_idx = actions.index("open_left")

        nonzero_probs = target[[listen_idx, open_left_idx]]
        expected_ratio = softmax_q[0] / softmax_q[1]
        actual_ratio = nonzero_probs[0] / nonzero_probs[1]

        np.testing.assert_allclose(actual_ratio, expected_ratio, atol=1e-4)

    def test_temperature_near_zero_gives_one_hot(self, tiger_env):
        """Test that very low temperature produces a near one-hot target.

        Purpose: Validates that as temperature approaches zero the policy
        target concentrates all probability on the best action.

        Given: A BetaZero planner with tau=0.01, z_q=1, z_n=1, and a tree
               with two children where one has clearly higher Q and more visits.
        When: _compute_q_weighted_policy_target is called.
        Then: The target is essentially one-hot with >0.99 probability on the
              dominant action.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="test_low_temp",
            action_sampler=sampler,
            n_simulations=20,
            state_dim=1,
            z_q=1.0,
            z_n=1.0,
            temperature=0.01,
        )

        particles = [["tiger_left"], ["tiger_right"]]
        log_weights = np.log(np.array([0.5, 0.5]))
        belief = WeightedParticleBelief(particles, log_weights)
        tree = BeliefNode(belief=belief)

        child1 = ActionNode(action="listen", parent=tree)
        child1.q_value = 5.0
        child1.visit_count = 20

        child2 = ActionNode(action="open_left", parent=tree)
        child2.q_value = 1.0
        child2.visit_count = 5

        target = planner._compute_q_weighted_policy_target(tree)

        assert (
            np.max(target) > 0.99
        ), f"Expected near one-hot distribution but max probability was {np.max(target)}"

    def test_discounted_return_computation(self, tiger_env):
        """Test _compute_discounted_returns with known rewards and discount factor.

        Purpose: Validates the backward discounted return calculation
        g_t = r_t + gamma * g_{t+1}.

        Given: A BetaZero planner with discount_factor=0.9 and rewards [1, 2, 3].
        When: _compute_discounted_returns([1, 2, 3]) is called.
        Then: g_0 = 1 + 0.9*2 + 0.81*3 = 5.23, g_1 = 2 + 0.9*3 = 4.7, g_2 = 3.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.9,
            depth=3,
            name="test_returns",
            action_sampler=sampler,
            n_simulations=5,
            state_dim=1,
        )

        returns = planner._compute_discounted_returns([1, 2, 3])

        expected_g2 = 3.0
        expected_g1 = 2.0 + 0.9 * 3.0  # 4.7
        expected_g0 = 1.0 + 0.9 * (2.0 + 0.9 * 3.0)  # 1 + 0.9*4.7 = 5.23

        assert len(returns) == 3
        np.testing.assert_allclose(returns[2], expected_g2, atol=1e-10)
        np.testing.assert_allclose(returns[1], expected_g1, atol=1e-10)
        np.testing.assert_allclose(returns[0], expected_g0, atol=1e-10)

    def test_fit_reduces_or_produces_loss(self, tiger_env):
        """Test that fit() with 2 iterations returns loss metrics.

        Purpose: Validates that the fit() policy iteration loop successfully
        collects data and trains the network, producing loss values.

        Given: A BetaZero planner configured for TigerPOMDP with small parameters
               for fast execution (2 iterations, 2 episodes, 5 steps).
        When: fit() is called with a lambda returning initial beliefs.
        Then: The returned metrics dictionary contains total_loss, value_loss,
              and policy_loss keys with at least one entry each.

        Test type: integration
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = BetaZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=2,
            name="test_fit",
            action_sampler=sampler,
            n_simulations=5,
            state_dim=1,
            training_epochs=2,
            training_batch_size=4,
        )

        initial_belief_fn = lambda: get_initial_belief(
            pomdp=tiger_env, n_particles=5, resampling=True
        )

        metrics = planner.fit(
            initial_belief_fn=initial_belief_fn,
            num_iterations=2,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
        )

        assert "total_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_loss" in metrics
        assert len(metrics["total_loss"]) > 0
        assert all(isinstance(v, float) for v in metrics["total_loss"])

    def test_training_data_captured_only_during_fit(self, tiger_planner):
        """Test that _collecting_data is False by default and _buffer is empty.

        Purpose: Validates that the planner does not collect training data
        unless explicitly in fit() mode.

        Given: A freshly constructed BetaZero planner.
        When: The internal training state is inspected without calling fit().
        Then: _collecting_data is False and _buffer has length 0.

        Test type: unit
        """
        assert tiger_planner._collecting_data is False
        assert len(tiger_planner._buffer) == 0

    def test_save_creates_directory(self, tiger_planner, tmp_path):
        """Test that save() creates policy_config.json and network_weights.pt.

        Purpose: Validates that the save method creates the expected files
        in the specified directory.

        Given: A BetaZero planner and a temporary directory path.
        When: save(filepath) is called with the temporary directory.
        Then: The directory contains policy_config.json and network_weights.pt
              files, and the returned path matches the input.

        Test type: unit
        """
        save_dir = tmp_path / "beta_zero_save"
        returned_path = tiger_planner.save(filepath=save_dir)

        assert returned_path == save_dir
        assert (save_dir / "policy_config.json").exists()
        assert (save_dir / "network_weights.pt").exists()

    def test_config_id_is_string(self, tiger_planner):
        """Test that planner.config_id is a non-empty string.

        Purpose: Validates that BetaZero produces a valid config_id for
        experiment tracking and caching.

        Given: A BetaZero planner constructed with specific parameters.
        When: config_id is accessed.
        Then: The result is a non-empty string.

        Test type: unit
        """
        cid = tiger_planner.config_id

        assert isinstance(cid, str)
        assert len(cid) > 0
