"""Tests for the ConstrainedZero planner module.

This module tests the ConstrainedZero planner, which extends BetaZero with
safety-constrained PUCT, 3-head network integration, adaptive failure
thresholds, and constrained policy targets.
"""

# pylint: disable=protected-access

import random

import numpy as np
import pytest
import torch

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.simulation.history import History, StepData
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import (
    BetaZero,
    _PendingExample,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
    ConstrainedTrainingExample,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
    ConstrainedZero,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)
from POMDPPlanners.training import PolicyTrainer
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def _no_failure(_):
    return False


def _always_failure(_):
    return True


def _tiger_left_failure(state):
    return state == "tiger_left"


def _make_dummy_belief():
    particles = [[0], [1]]
    log_weights = np.log([0.5, 0.5])
    return WeightedParticleBelief(particles, log_weights)


def _make_step_data(state="tiger_left", next_state="tiger_left"):
    return StepData(
        state=state,
        action="listen",
        next_state=next_state,
        observation="tiger_left",
        reward=-1.0,
        belief=_make_dummy_belief(),
    )


def _make_history(steps):
    return History(
        history=steps,
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=len(steps),
        reach_terminal_state=False,
        policy_run_data=[],
    )


def _make_initial_belief(tiger_env):
    return get_initial_belief(pomdp=tiger_env, n_particles=10, resampling=True)


@pytest.fixture
def tiger_env():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def tiger_planner(tiger_env):
    sampler = DiscreteActionSampler(tiger_env.get_actions())
    return ConstrainedZero(
        environment=tiger_env,
        discount_factor=0.95,
        depth=3,
        name="test_constrained_zero",
        action_sampler=sampler,
        n_simulations=20,
        state_dim=1,
        failure_fn=_no_failure,
        delta_0=0.1,
        eta=0.1,
    )


@pytest.fixture
def initial_belief(tiger_env):
    return _make_initial_belief(tiger_env)


class TestConstrainedZeroInitialization:
    """Tests for ConstrainedZero construction and attributes."""

    def test_inherits_from_beta_zero(self, tiger_planner):
        """Test that ConstrainedZero is a BetaZero subclass.

        Purpose: Validates the inheritance chain.

        Given: A ConstrainedZero planner.
        When: Checking isinstance.
        Then: It is an instance of BetaZero.

        Test type: unit
        """
        assert isinstance(tiger_planner, BetaZero)

    def test_has_failure_fn(self, tiger_planner):
        """Test planner stores the failure function.

        Purpose: Validates failure_fn is accessible.

        Given: A ConstrainedZero planner constructed with a failure_fn.
        When: Accessing the failure_fn attribute.
        Then: It matches the provided function.

        Test type: unit
        """
        assert tiger_planner.failure_fn is _no_failure

    def test_has_constrained_params(self, tiger_planner):
        """Test planner stores ConstrainedZero-specific parameters.

        Purpose: Validates delta_0, eta, delta_compounding are stored.

        Given: A ConstrainedZero planner with delta_0=0.1, eta=0.1.
        When: Accessing the parameters.
        Then: They match the provided values.

        Test type: unit
        """
        assert tiger_planner.delta_0 == 0.1
        assert tiger_planner.eta == 0.1
        assert tiger_planner.delta_compounding == 1.0

    def test_creates_constrained_network(self, tiger_planner):
        """Test planner creates a ConstrainedZeroNetwork by default.

        Purpose: Validates _create_default_network produces a 3-head network.

        Given: A ConstrainedZero planner with no explicit network.
        When: Checking the network type.
        Then: It is a ConstrainedZeroNetwork instance.

        Test type: unit
        """
        assert isinstance(tiger_planner.network, ConstrainedZeroNetwork)

    def test_creates_constrained_buffer(self, tiger_planner):
        """Test planner uses a ConstrainedTrainingBuffer.

        Purpose: Validates the buffer is the constrained variant.

        Given: A ConstrainedZero planner.
        When: Checking the buffer type.
        Then: It is a ConstrainedTrainingBuffer instance.

        Test type: unit
        """
        assert isinstance(tiger_planner._buffer, ConstrainedTrainingBuffer)

    def test_failure_and_delta_dicts_initialized(self, tiger_planner):
        """Test failure_dict and delta_dict are initialized as empty.

        Purpose: Validates tracking dicts start empty.

        Given: A ConstrainedZero planner.
        When: Checking the dicts.
        Then: Both are empty dictionaries.

        Test type: unit
        """
        assert tiger_planner._failure_dict == {}
        assert tiger_planner._delta_dict == {}


class TestConstrainedZeroAction:
    """Tests for ConstrainedZero online planning via action()."""

    def test_action_returns_valid_output(self, tiger_planner, initial_belief):
        """Test action() returns valid action from the environment.

        Purpose: Validates the planner produces a legal action.

        Given: A ConstrainedZero planner and an initial belief.
        When: action() is called.
        Then: The returned action is in env.get_actions().

        Test type: integration
        """
        actions, _ = tiger_planner.action(initial_belief)
        assert actions[0] in tiger_planner.environment.get_actions()

    def test_failure_dict_populated_after_action(self, tiger_planner, initial_belief):
        """Test _failure_dict is populated after planning.

        Purpose: Validates failure tracking occurs during MCTS.

        Given: A ConstrainedZero planner.
        When: action() is called.
        Then: _failure_dict is non-empty (has entries for visited action nodes).

        Test type: integration
        """
        tiger_planner.action(initial_belief)
        # After planning, failure_dict should have entries for action nodes in the tree
        assert len(tiger_planner._failure_dict) > 0


class TestConstrainedZeroHelpers:
    """Tests for ConstrainedZero helper methods."""

    def test_estimate_belief_failure_prob_no_failure(self, tiger_planner):
        """Test _estimate_belief_failure_prob returns 0 for no-failure fn.

        Purpose: Validates belief failure estimation with trivial failure_fn.

        Given: A planner whose failure_fn always returns False.
        When: _estimate_belief_failure_prob is called.
        Then: Returns 0.0.

        Test type: unit
        """
        belief = _make_dummy_belief()
        prob = tiger_planner._estimate_belief_failure_prob(belief)
        assert prob == 0.0

    def test_estimate_belief_failure_prob_all_fail(self, tiger_env):
        """Test _estimate_belief_failure_prob returns 1 when all states fail.

        Purpose: Validates belief failure estimation with always-fail fn.

        Given: A planner whose failure_fn always returns True.
        When: _estimate_belief_failure_prob is called.
        Then: Returns 1.0.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="all_fail",
            action_sampler=sampler,
            n_simulations=10,
            state_dim=1,
            failure_fn=_always_failure,
        )
        belief = _make_dummy_belief()
        prob = planner._estimate_belief_failure_prob(belief)
        assert prob == 1.0

    def test_compound_failure_formula(self, tiger_planner):
        """Test _compound_failure implements p + delta*(1-p)*p_next.

        Purpose: Validates the compounding formula.

        Given: p_immediate=0.2, p_next=0.3, delta_compounding=1.0.
        When: _compound_failure is called.
        Then: Result is 0.2 + 1.0 * 0.8 * 0.3 = 0.44.

        Test type: unit
        """
        result = tiger_planner._compound_failure(0.2, 0.3)
        expected = 0.2 + 1.0 * (1.0 - 0.2) * 0.3
        assert abs(result - expected) < 1e-10

    def test_compound_failure_zero_immediate(self, tiger_planner):
        """Test compound failure when p_immediate is 0.

        Purpose: Validates compound formula reduces to delta * p_next.

        Given: p_immediate=0.0, p_next=0.5.
        When: _compound_failure is called.
        Then: Result is 0.5.

        Test type: unit
        """
        result = tiger_planner._compound_failure(0.0, 0.5)
        assert abs(result - 0.5) < 1e-10

    def test_compound_failure_certain_immediate(self, tiger_planner):
        """Test compound failure when p_immediate is 1.

        Purpose: Validates compound formula returns 1 if immediate failure is certain.

        Given: p_immediate=1.0, p_next=0.5.
        When: _compound_failure is called.
        Then: Result is 1.0.

        Test type: unit
        """
        result = tiger_planner._compound_failure(1.0, 0.5)
        assert abs(result - 1.0) < 1e-10

    def test_get_delta_prime_default(self, tiger_planner):
        """Test _get_delta_prime returns delta_0 when no stored value.

        Purpose: Validates default delta_prime.

        Given: A belief node ID with no entry in delta_dict.
        When: _get_delta_prime is called.
        Then: Returns delta_0.

        Test type: unit
        """
        belief = _make_dummy_belief()
        tree = Tree()
        belief_id = tree.add_belief_node(belief)
        delta = tiger_planner._get_delta_prime(belief_id)
        assert delta == tiger_planner.delta_0

    def test_update_action_failure_running_average(self, tiger_planner):
        """Test _update_action_failure computes running average.

        Purpose: Validates running average update formula.

        Given: An action node with visit_count=1 then 2.
        When: _update_action_failure is called with failure=0.6 then 0.2.
        Then: The stored value follows the running-average formula.

        Test type: unit
        """
        belief = _make_dummy_belief()
        tree = Tree()
        parent_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action="test", parent_id=parent_id)
        tree.visit_count[action_id] = 1

        tiger_planner._update_action_failure(tree=tree, action_id=action_id, failure=0.6)
        assert abs(tiger_planner._failure_dict[action_id] - 0.6) < 1e-10

        tree.visit_count[action_id] = 2
        tiger_planner._update_action_failure(tree=tree, action_id=action_id, failure=0.2)
        expected = 0.6 + (0.2 - 0.6) / 2
        assert abs(tiger_planner._failure_dict[action_id] - expected) < 1e-10

    def test_compute_per_timestep_failures_no_failure(self, tiger_planner):
        """Test _compute_per_timestep_failures returns all zeros when no failure states.

        Purpose: Validates per-timestep failure detection with trivial failure_fn.

        Given: A history where failure_fn returns False for all states.
        When: _compute_per_timestep_failures is called.
        Then: Returns [0.0] for each timestep.

        Test type: unit
        """
        history = _make_history([_make_step_data()])
        result = tiger_planner._compute_per_timestep_failures(history)
        assert result == [0.0]

    def test_compute_per_timestep_failures_with_failure(self, tiger_env):
        """Test _compute_per_timestep_failures returns 1.0 for timesteps at or before a failure.

        Purpose: Validates per-timestep failure detection with a failing state.

        Given: A planner with failure_fn that fails on "tiger_left".
        When: _compute_per_timestep_failures is called on history containing "tiger_left".
        Then: Returns [1.0] for the single timestep where failure occurs.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="fail_test",
            action_sampler=sampler,
            n_simulations=10,
            state_dim=1,
            failure_fn=_tiger_left_failure,
        )

        history = _make_history(
            [
                _make_step_data(state="tiger_left", next_state="tiger_right"),
            ]
        )
        result = planner._compute_per_timestep_failures(history)
        assert result == [1.0]

    def test_finalize_episode_data_uses_episode_level_failure(self, tiger_env):
        """All training examples from a failing episode get failure_target == 1.0.

        Purpose: Validates that ``_finalize_episode_data`` records the
        episode-level failure flag for every timestep — matching the
        ``ConstrainedTrainingExample.failure_target`` docstring which
        already reads "Binary episode-level failure indicator (1.0 if
        failure occurred)". The previous per-step backward-accumulation
        rule produced 0.0 for examples added after the failure step,
        contradicting the documented intent.

        Given: A 3-step history where only step 1 has a failure
            (``failure_fn(next_state)`` is True), with three pending
            examples seeded into the planner so each step contributes
            one training datum.
        When: ``_finalize_episode_data(history)`` runs.
        Then: All three examples in the buffer carry ``failure_target ==
            1.0`` (the episode-level "any failure occurred?" answer).

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="finalize_test",
            action_sampler=sampler,
            n_simulations=10,
            state_dim=1,
            failure_fn=_tiger_left_failure,
        )

        # Three steps; only step 1's next_state triggers failure.
        history = _make_history(
            [
                _make_step_data(state="tiger_right", next_state="tiger_right"),
                _make_step_data(state="tiger_right", next_state="tiger_left"),
                _make_step_data(state="tiger_right", next_state="tiger_right"),
            ]
        )

        for _ in range(3):
            planner._pending_examples.append(
                _PendingExample(
                    belief_features=np.zeros(1, dtype=np.float32),
                    policy_target=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
                )
            )
        planner._buffer.begin_iteration()

        planner._finalize_episode_data(history)

        examples = planner._buffer._current
        assert len(examples) == 3
        assert all(isinstance(ex, ConstrainedTrainingExample) for ex in examples)
        targets = [
            ex.failure_target for ex in examples if isinstance(ex, ConstrainedTrainingExample)
        ]
        assert all(
            t == 1.0 for t in targets
        ), f"all examples must carry the episode-level failure flag; got {targets}"


class TestConstrainedZeroPolicyTarget:
    """Tests for constrained Q-weighted policy target computation."""

    def test_constrained_target_filters_unsafe(self, tiger_env):
        """Test policy target zeros out unsafe actions.

        Purpose: Validates that the constrained target applies safety mask.

        Given: A ConstrainedZero planner and a tree with known Q-values and
               failure probabilities where one action is unsafe.
        When: _compute_q_weighted_policy_target is called.
        Then: The unsafe action gets zero probability.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="target_test",
            action_sampler=sampler,
            n_simulations=10,
            state_dim=1,
            failure_fn=_no_failure,
            delta_0=0.1,
        )

        # Create an arena tree with children
        belief = _make_dummy_belief()
        tree = Tree()
        root_id = tree.add_belief_node(belief)
        tree.visit_count[root_id] = 20

        actions = tiger_env.get_actions()
        child_ids = []
        for i, a in enumerate(actions):
            cid = tree.add_action_node(action=a, parent_id=root_id)
            tree.q_value[cid] = float(i + 1)
            tree.visit_count[cid] = 5 + i
            child_ids.append(cid)

        # Make last action unsafe
        for cid in child_ids[:-1]:
            planner._failure_dict[cid] = 0.01  # safe
        planner._failure_dict[child_ids[-1]] = 0.5  # unsafe

        target = planner._compute_q_weighted_policy_target(tree, root_id)

        # Find the index of the unsafe action
        unsafe_idx = actions.index(tree.action[child_ids[-1]])
        assert target[unsafe_idx] == 0.0

        # Safe actions should have positive probability
        for cid in child_ids[:-1]:
            safe_idx = actions.index(tree.action[cid])
            assert target[safe_idx] > 0.0

    def test_constrained_target_sums_to_one(self, tiger_env):
        """Test constrained policy target sums to 1.

        Purpose: Validates normalization of constrained target.

        Given: A tree with mixed safe and unsafe actions.
        When: _compute_q_weighted_policy_target is called.
        Then: Target probabilities sum to 1.

        Test type: unit
        """
        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            name="sum_test",
            action_sampler=sampler,
            n_simulations=10,
            state_dim=1,
            failure_fn=_no_failure,
        )

        belief = _make_dummy_belief()
        tree = Tree()
        root_id = tree.add_belief_node(belief)
        tree.visit_count[root_id] = 20

        actions = tiger_env.get_actions()
        for i, a in enumerate(actions):
            cid = tree.add_action_node(action=a, parent_id=root_id)
            tree.q_value[cid] = float(i + 1)
            tree.visit_count[cid] = 5 + i
            planner._failure_dict[cid] = 0.01  # all safe

        target = planner._compute_q_weighted_policy_target(tree, root_id)
        np.testing.assert_allclose(target.sum(), 1.0, atol=1e-6)


class TestConstrainedZeroFit:
    """Tests for ConstrainedZero training via PolicyTrainer."""

    def test_fit_returns_failure_loss(self, tiger_env):
        """Test PolicyTrainer.train() returns metrics including failure_loss.

        Purpose: Validates that policy iteration tracks failure loss.

        Given: A ConstrainedZero planner with minimal parameters.
        When: PolicyTrainer.train() is called for 1 iteration with 2 episodes.
        Then: Returned metrics dict contains "failure_loss".

        Test type: integration
        """
        torch.manual_seed(42)
        np.random.seed(42)

        sampler = DiscreteActionSampler(tiger_env.get_actions())
        planner = ConstrainedZero(
            environment=tiger_env,
            discount_factor=0.95,
            depth=2,
            name="fit_test",
            action_sampler=sampler,
            n_simulations=5,
            state_dim=1,
            failure_fn=_no_failure,
            training_epochs=2,
            training_batch_size=8,
        )

        def initial_belief_fn():
            return _make_initial_belief(tiger_env)

        trainer = PolicyTrainer(
            policy=planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=1,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
        )
        metrics = trainer.train()

        assert "failure_loss" in metrics


class TestConstrainedZeroTreeStructure:
    """Tests for tree structure and statistics after a real plan."""

    def test_tree_structure_after_plan(self, tiger_planner, initial_belief):
        """Test arena tree has well-formed structure after _learn_tree.

        Purpose: Validates root-node invariants, BELIEF/ACTION alternation,
        per-node payload presence, depth bound, and visit-count monotonicity.

        Given: A ConstrainedZero planner and an initial belief.
        When: _learn_tree is called to build the search tree.
        Then: The root is a BELIEF with no parent and no observation; every
              non-root node has a parent; visit counts are non-negative;
              BELIEF and ACTION nodes alternate; BELIEF nodes carry a belief
              and ACTION nodes carry an action; BFS-depth from root is at
              most 2 * planner.depth + 2; and every node's visit_count is
              at least the sum of its children's visit_counts.

        Test type: unit
        """
        tree, root_id = tiger_planner._learn_tree(initial_belief)

        assert tree.kind[root_id] == BELIEF
        assert tree.parent_id[root_id] is None
        assert tree.observation[root_id] is None
        assert len(tree.children_ids[root_id]) > 0

        max_observed_depth = 0
        frontier = [(root_id, 0)]
        visited_ids = []
        while frontier:
            node_id, d = frontier.pop()
            visited_ids.append(node_id)
            max_observed_depth = max(max_observed_depth, d)
            for cid in tree.children_ids[node_id]:
                frontier.append((cid, d + 1))

        for node_id in visited_ids:
            assert tree.visit_count[node_id] >= 0
            if node_id != root_id:
                assert tree.parent_id[node_id] is not None
            if tree.kind[node_id] == BELIEF:
                assert tree.belief[node_id] is not None
                for cid in tree.children_ids[node_id]:
                    assert tree.kind[cid] == ACTION
            else:
                assert tree.kind[node_id] == ACTION
                assert tree.action[node_id] is not None
                for cid in tree.children_ids[node_id]:
                    assert tree.kind[cid] == BELIEF

            children = tree.children_ids[node_id]
            if children:
                children_visit_sum = sum(tree.visit_count[c] for c in children)
                assert tree.visit_count[node_id] >= children_visit_sum

        assert max_observed_depth <= 2 * tiger_planner.depth + 2

    def test_failure_dict_populated_after_plan(self, tiger_planner, initial_belief):
        """Test _failure_dict has entries for every visited action node.

        Purpose: Validates the failure-tracking dict is populated for all
        action nodes that were actually visited during MCTS, with values in
        the valid probability range [0.0, 1.0].

        Given: A ConstrainedZero planner and an initial belief.
        When: _learn_tree is called to build the search tree.
        Then: _failure_dict is non-empty, and every ACTION node with
              visit_count > 0 in the tree has an entry in _failure_dict
              whose value lies in [0.0, 1.0].

        Test type: unit
        """
        tree, _ = tiger_planner._learn_tree(initial_belief)

        assert len(tiger_planner._failure_dict) > 0

        for node_id in range(len(tree)):
            if tree.kind[node_id] != ACTION:
                continue
            if tree.visit_count[node_id] <= 0:
                continue
            assert node_id in tiger_planner._failure_dict
            failure_value = tiger_planner._failure_dict[node_id]
            assert 0.0 <= failure_value <= 1.0

    def test_v_value_matches_max_q_of_children(self, tiger_planner, initial_belief):
        """Test v_value of each visited belief equals max q over its children.

        Purpose: Validates the ConstrainedZero v-value update rule
        (_update_node_statistics_constrained) which sets
        v_value[belief] = max(q_value[c] for c in children).

        Given: A ConstrainedZero planner and an initial belief.
        When: _learn_tree is called to build the search tree.
        Then: For every BELIEF node with at least one action child where
              some child has visit_count > 0, the stored v_value matches
              the max of q_values over ALL children (unvisited children
              contribute q_value=0.0 by default).

        Test type: unit
        """
        tree, _ = tiger_planner._learn_tree(initial_belief)

        checked_any = False
        for node_id in range(len(tree)):
            if tree.kind[node_id] != BELIEF:
                continue
            children = tree.children_ids[node_id]
            if not children:
                continue
            if not any(tree.visit_count[c] > 0 for c in children):
                continue
            expected = max(tree.q_value[c] for c in children)
            assert np.isclose(tree.v_value[node_id], expected)
            checked_any = True

        assert checked_any
