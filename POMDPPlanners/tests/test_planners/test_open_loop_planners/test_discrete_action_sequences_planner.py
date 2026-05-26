# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)

np.random.seed(42)
random.seed(42)


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def planner(tiger_pomdp):
    return DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.95,
        name="test_planner",
        depth=2,
        n_return_samples=10,
    )


def test_initialization(tiger_pomdp):
    """Test initialization.

    Purpose: Validates proper initialization of DiscreteActionSequencesPlanner with parameter validation

    Given: TigerPOMDP environment and various parameter combinations (valid and invalid depth, n_return_samples, discount_factor)
    When: DiscreteActionSequencesPlanner is instantiated with these parameters
    Then: Valid parameters create planner with correct attributes, invalid parameters raise ValueError

    Test type: unit
    """
    # Test valid initialization
    planner = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.95,
        name="test_planner",
        depth=2,
        n_return_samples=10,
    )
    assert planner.depth == 2
    assert planner.n_return_samples == 10
    assert planner.discount_factor == 0.95
    assert planner.name == "test_planner"

    # Test invalid depth
    with pytest.raises(ValueError):
        DiscreteActionSequencesPlanner(
            environment=tiger_pomdp,
            discount_factor=0.95,
            name="test_planner",
            depth=0,
            n_return_samples=10,
        )

    # Test invalid n_return_samples
    with pytest.raises(ValueError):
        DiscreteActionSequencesPlanner(
            environment=tiger_pomdp,
            discount_factor=0.95,
            name="test_planner",
            depth=2,
            n_return_samples=0,
        )

    # Test invalid discount_factor
    with pytest.raises(ValueError):
        DiscreteActionSequencesPlanner(
            environment=tiger_pomdp,
            discount_factor=1.5,
            name="test_planner",
            depth=2,
            n_return_samples=10,
        )


def test_action_selection(planner, tiger_pomdp):
    """Test action selection.

    Purpose: Validates that action selection returns valid TigerPOMDP actions and proper PolicyRunData for different beliefs

    Given: DiscreteActionSequencesPlanner and WeightedParticleBelief instances with different probability distributions (equal vs tiger_right-biased)
    When: action() method is called with each belief
    Then: Returns valid tiger actions (listen/open_left/open_right) and PolicyRunData with empty info_variables

    Test type: unit
    """
    # Create a belief with equal probability for both states
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # Get action from planner
    actions, run_data = planner.action(belief)
    action = actions[0]  # action() returns a list, we take first element

    # Verify action is one of the valid actions
    assert action in tiger_pomdp.get_actions()
    assert isinstance(run_data, PolicyRunData)
    assert len(run_data.info_variables) == 0  # Currently no metrics are returned

    # Test with different belief
    particles_right = ["tiger_right"] * 9 + ["tiger_left"]  # 9 right, 1 left
    log_weights_right = np.log(np.ones(10) / 10)
    belief_right = WeightedParticleBelief(particles=particles_right, log_weights=log_weights_right)
    actions_right, run_data_right = planner.action(belief_right)
    action_right = actions_right[0]
    assert action_right in tiger_pomdp.get_actions()
    assert isinstance(run_data_right, PolicyRunData)
    assert len(run_data_right.info_variables) == 0  # Currently no metrics are returned


def test_compute_return(planner, tiger_pomdp):
    """Test compute return.

    Purpose: Validates that return estimation produces finite numerical values for various action sequences

    Given: DiscreteActionSequencesPlanner, belief with equal tiger probabilities, and action sequences (listen-listen, listen-open_left, open_left-open_right)
    When: estimate_return() is called for each action sequence
    Then: Returns finite float values representing expected discounted rewards

    Test type: unit
    """
    # Create a belief
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # Test return computation for different action sequences
    action_sequences = [
        ["listen", "listen"],
        ["listen", "open_left"],
        ["open_left", "open_right"],
    ]

    for action_sequence in action_sequences:
        return_value = planner.estimate_return(action_sequence, belief)
        assert isinstance(return_value, float)
        # Return should be finite
        assert np.isfinite(return_value)


def test_search_behavior(planner, tiger_pomdp):
    """Test search behavior.

    Purpose: Validates that search returns action sequences of correct length with valid tiger actions

    Given: DiscreteActionSequencesPlanner with depth=2 and belief with equal tiger probabilities
    When: search() method is called to find optimal action sequence
    Then: Returns action sequence with length equal to planner depth containing only valid tiger actions

    Test type: unit
    """
    # Create a belief
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # Test search returns a valid action sequence
    action_sequence = planner.search(belief)
    assert len(action_sequence) == planner.depth
    for action in action_sequence:
        assert action in tiger_pomdp.get_actions()


def test_integration_with_tiger_pomdp(planner, tiger_pomdp):
    """Test integration with tiger pomdp.

    Purpose: Validates that DiscreteActionSequencesPlanner integrates properly with TigerPOMDP environment across different belief states

    Given: DiscreteActionSequencesPlanner and TigerPOMDP with various belief distributions (equal, left-biased, right-biased)
    When: Actions are selected and executed in the tiger environment
    Then: All actions are valid, PolicyRunData is correct, and environment state transitions work properly

    Test type: integration
    """
    # Test with different initial beliefs
    beliefs = [
        WeightedParticleBelief(
            particles=["tiger_left", "tiger_right"] * 5,
            log_weights=np.log(np.ones(10) / 10),
        ),
        WeightedParticleBelief(
            particles=["tiger_left"] * 9 + ["tiger_right"],
            log_weights=np.log(np.ones(10) / 10),
        ),
        WeightedParticleBelief(
            particles=["tiger_right"] * 9 + ["tiger_left"],
            log_weights=np.log(np.ones(10) / 10),
        ),
    ]

    for belief in beliefs:
        actions, run_data = planner.action(belief)
        action = actions[0]
        assert action in tiger_pomdp.get_actions()
        assert isinstance(run_data, PolicyRunData)
        assert len(run_data.info_variables) == 0  # Currently no metrics are returned

        # Verify the action leads to valid next state
        state = belief.sample()
        next_state, observation, reward = tiger_pomdp.sample_next_step(state, action)
        assert next_state in tiger_pomdp.states
        assert observation in tiger_pomdp.observations
        assert isinstance(reward, float)


# Config ID Tests


def test_discrete_action_sequences_config_id_consistency_identical_parameters(
    tiger_pomdp,
):
    """Test that config_id is consistent for identical DiscreteActionSequencesPlanner parameters.

    Purpose: Validates that DiscreteActionSequencesPlanner with identical parameters produces identical config_id

    Given: Two DiscreteActionSequencesPlanner instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two DiscreteActionSequencesPlanner instances with identical parameters
    planner1 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test1",
        depth=8,
        n_return_samples=25,
    )

    planner2 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test1",  # Same name
        depth=8,
        n_return_samples=25,
    )

    # Config IDs should be identical
    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_discrete_action_sequences_config_id_different_depth(tiger_pomdp):
    """Test that config_id changes when depth parameter differs.

    Purpose: Validates that config_id changes when depth parameter differs

    Given: Two DiscreteActionSequencesPlanner instances with different depth values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    planner1 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test",
        depth=5,
        n_return_samples=25,
    )

    planner2 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test",
        depth=12,  # Different depth
        n_return_samples=25,
    )

    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 != config_id2


def test_discrete_action_sequences_config_id_different_n_return_samples(tiger_pomdp):
    """Test that config_id changes when n_return_samples parameter differs.

    Purpose: Validates that config_id changes when n_return_samples parameter differs

    Given: Two DiscreteActionSequencesPlanner instances with different n_return_samples values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    planner1 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test",
        depth=8,
        n_return_samples=15,
    )

    planner2 = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Test",
        depth=8,
        n_return_samples=30,  # Different n_return_samples
    )

    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 != config_id2


def test_discrete_action_sequences_config_id_consistency_across_evaluations(
    tiger_pomdp,
):
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single DiscreteActionSequencesPlanner instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    planner = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Consistency_Test",
        depth=3,  # Reduced for testing
        n_return_samples=10,  # Reduced for testing
    )

    # Get initial config_id
    initial_config_id = planner.config_id

    # Create initial belief
    particles = ["tiger_left", "tiger_right"] * 5
    log_weights = np.log(np.ones(10) / 10)
    initial_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

    # Perform multiple policy evaluations
    for i in range(3):
        actions, run_data = planner.action(initial_belief)

        # Check config_id remains the same
        current_config_id = planner.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(actions, list)
        assert len(actions) == planner.depth  # Should return full action sequence
        for action in actions:
            assert action in tiger_pomdp.get_actions()
        assert run_data is not None

    # Final check
    final_config_id = planner.config_id
    assert final_config_id == initial_config_id


def test_discrete_action_sequences_config_id_hash_properties(tiger_pomdp):
    """Test that config_id has proper hash properties.

    Purpose: Validates that config_id produces valid hash strings

    Given: DiscreteActionSequencesPlanner instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties

    Test type: unit
    """
    planner = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.98,
        name="DiscreteActionSeq_Hash_Test",
        depth=8,
        n_return_samples=25,
    )

    config_id = planner.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in "0123456789abcdef" for c in config_id.lower())
