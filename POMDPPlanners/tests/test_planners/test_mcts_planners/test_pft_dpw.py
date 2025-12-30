# pylint: disable=protected-access  # Tests need to access protected members
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.planners_utils.dpw import (
    action_progressive_widening,
)
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler

np.random.seed(42)
random.seed(42)


@pytest.fixture
def environment():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture
def action_sampler():
    return UnitCircleActionSampler(max_action_magnitude=1.0)


@pytest.fixture
def planner(environment, action_sampler):
    return PFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=5,
        name="test_pft_dpw",
        action_sampler=action_sampler,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=100,
        min_samples_per_node=10,
        min_visit_count_per_action=1,
    )


@pytest.fixture
def initial_belief(environment):
    return get_initial_belief(
        pomdp=environment,
        n_particles=20,  # Small number of particles for testing
        resampling=True,
    )


def test_initialization(planner):
    """Test that the planner initializes correctly with valid parameters.

    Purpose: Validates proper initialization of PFT_DPW planner with all configuration parameters

    Given: PFT_DPW constructor with depth=5, progressive widening parameters (k_a=1.0, alpha_a=0.5, k_o=1.0, alpha_o=0.5), and sampling settings
    When: Planner object is initialized with these parameters
    Then: All attributes are correctly set and accessible (depth, k_a, alpha_a, k_o, alpha_o, exploration_constant)

    Test type: unit
    """
    assert planner.depth == 5
    assert planner.min_samples_per_node == 10
    assert planner.min_visit_count_per_action == 1
    assert planner.k_a == 1.0
    assert planner.alpha_a == 0.5
    assert planner.k_o == 1.0
    assert planner.alpha_o == 0.5
    assert planner.exploration_constant == 1.0


def test_action_sampler(action_sampler):
    """Test that the action sampler returns valid actions.

    Purpose: Validates sampling behavior for action r

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    action = action_sampler.sample()
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert np.linalg.norm(action) <= 1.0  # Action should be within unit circle


def test_action_progressive_widening(planner, initial_belief):
    """Test that action progressive widening creates new action nodes when needed.

    Purpose: Validates that progressive widening correctly expands action nodes in belief tree according to DPW algorithm

    Given: BeliefNode with initial belief and PFT_DPW planner with progressive widening parameters (alpha_a=0.5, k_a=1.0)
    When: action_progressive_widening is called multiple times on the same belief node
    Then: New action nodes are progressively added to belief node children, or existing nodes are reused if duplicate actions are sampled

    Test type: unit
    """
    belief_node = BeliefNode(belief=initial_belief)

    # First call should create a new action node
    action_node1 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a,
    )
    assert len(belief_node.children) == 1
    assert action_node1 in belief_node.children

    # Second call may create another action node or reuse existing one if duplicate action is sampled
    # Keep trying until we get a different action or reach max attempts
    initial_children_count = len(belief_node.children)
    for _ in range(10):  # Try up to 10 times to get a different action
        action_node2 = action_progressive_widening(
            belief_node=belief_node,
            alpha_a=planner.alpha_a,
            action_sampler=planner.action_sampler,
            exploration_constant=planner.exploration_constant,
            k_a=planner.k_a,
        )
        if len(belief_node.children) > initial_children_count:
            # New action was created
            assert len(belief_node.children) == 2
            assert action_node2 in belief_node.children
            assert action_node1 != action_node2
            break
        elif action_node2 != action_node1:
            # Different action node was returned (shouldn't happen with same action)
            assert action_node2 in belief_node.children
            break
    else:
        # Same action was sampled multiple times, which is valid
        # Verify that the same node is reused
        assert len(belief_node.children) == 1
        assert action_node2 == action_node1


def test_simulate_path(planner, initial_belief, environment):
    """Test that path simulation updates node statistics correctly.

    Purpose: Validates that simulate path method correctly updates tree node statistics during MCTS simulation

    Given: BeliefNode with initial belief, PFT_DPW planner, and ContinuousLightDarkPOMDP environment with grid_size and reward bounds
    When: _simulate_path is called with depth=0 to run a single simulation step
    Then: Node visit count increases, node is no longer leaf, and return value is within expected reward range (-grid_size*sqrt(2)-10 to 10)

    Test type: unit
    """
    belief_node = BeliefNode(belief=initial_belief)

    # Run a simulation
    return_value = planner._simulate_path(belief_node=belief_node, depth=0)

    # Verify node statistics were updated
    assert belief_node.visit_count > 0
    assert not belief_node.is_leaf

    # Verify return value is within expected range
    # For LightDarkPOMDP, rewards are typically between -10 (obstacle hit) and 10 (goal reached)
    assert return_value >= (-10 - environment.grid_size * np.sqrt(2)) * 5  # Minimum possible reward
    assert return_value <= 10  # Maximum possible reward


# Config ID Tests


def test_pft_dpw_config_id_consistency_identical_parameters(environment, action_sampler):
    """Test that config_id is consistent for identical PFT_DPW parameters.

    Purpose: Validates that PFT_DPW with identical parameters produces identical config_id

    Given: Two PFT_DPW instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two PFT_DPW instances with identical parameters
    pft_dpw1 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test1",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=100,
    )

    pft_dpw2 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test1",  # Same name
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=100,
    )

    # Config IDs should be identical
    config_id1 = pft_dpw1.config_id
    config_id2 = pft_dpw2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_pft_dpw_config_id_different_action_sampler_values(environment):
    """Test that config_id changes when action sampler is initialized with different values.

    Purpose: Validates that config_id changes when action sampler parameters differ

    Given: Two PFT_DPW instances with action samplers having different max_action_magnitude
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    # Create action samplers with different max_action_magnitude
    sampler1 = UnitCircleActionSampler(max_action_magnitude=1.0)
    sampler2 = UnitCircleActionSampler(max_action_magnitude=2.0)

    pft_dpw1 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test",
        action_sampler=sampler1,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=100,
    )

    pft_dpw2 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test",
        action_sampler=sampler2,  # Different sampler parameters
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=100,
    )

    config_id1 = pft_dpw1.config_id
    config_id2 = pft_dpw2.config_id

    assert config_id1 != config_id2
    assert isinstance(config_id1, str)
    assert isinstance(config_id2, str)
    assert len(config_id1) > 0
    assert len(config_id2) > 0


def test_pft_dpw_config_id_different_planner_parameters(environment, action_sampler):
    """Test that config_id changes when PFT_DPW planner parameters differ.

    Purpose: Validates that config_id changes when core PFT_DPW parameters differ

    Given: Two PFT_DPW instances with different exploration_constant values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    pft_dpw1 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,  # Different exploration constant
        n_simulations=100,
    )

    pft_dpw2 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Test",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=2.0,  # Different exploration constant
        n_simulations=100,
    )

    config_id1 = pft_dpw1.config_id
    config_id2 = pft_dpw2.config_id

    assert config_id1 != config_id2


def test_pft_dpw_config_id_consistency_across_evaluations(environment, action_sampler):
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single PFT_DPW instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    pft_dpw = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=5,  # Reduced for testing
        name="PFT_DPW_Consistency_Test",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=10,  # Reduced for testing
    )

    # Get initial config_id
    initial_config_id = pft_dpw.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(environment, n_particles=50)

    # Perform multiple policy evaluations
    for i in range(3):
        action, run_data = pft_dpw.action(initial_belief)

        # Check config_id remains the same
        current_config_id = pft_dpw.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert isinstance(action[0], np.ndarray)
        assert action[0].shape == (2,)
        assert run_data is not None

    # Final check
    final_config_id = pft_dpw.config_id
    assert final_config_id == initial_config_id


def test_pft_dpw_config_id_action_sampler_attribute_changes(environment):
    """Test config_id changes when action sampler attributes are modified.

    Purpose: Validates that modifying action sampler attributes affects config_id

    Given: PFT_DPW instance with modifiable action sampler
    When: Action sampler attributes are changed
    Then: config_id reflects the change

    Test type: unit
    """
    # Create initial action sampler and PFT_DPW
    action_sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

    pft_dpw = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Attribute_Test",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        n_simulations=100,
    )

    # Get initial config_id
    initial_config_id = pft_dpw.config_id

    # Modify the action sampler's attribute
    action_sampler.max_action_magnitude = 2.0

    # Config ID should be different after modification
    modified_config_id = pft_dpw.config_id
    assert modified_config_id != initial_config_id

    # Restore original value
    action_sampler.max_action_magnitude = 1.0

    # Config ID should return to original
    restored_config_id = pft_dpw.config_id
    assert restored_config_id == initial_config_id


def test_pft_dpw_timeout_configuration(environment, action_sampler):
    """Test that PFT_DPW works correctly with time_out_in_seconds instead of n_simulations.

    Purpose: Validates that PFT_DPW can be configured with time-based termination instead of simulation count

    Given: PFT_DPW constructor with time_out_in_seconds=2 and no n_simulations
    When: PFT_DPW instance is created with timeout configuration
    Then: Instance is created successfully with correct timeout value and n_simulations is None

    Test type: unit
    """
    pft_dpw = PFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=5,
        name="PFT_DPW_Timeout_Test",
        action_sampler=action_sampler,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        time_out_in_seconds=2,  # Use timeout instead of n_simulations
        min_samples_per_node=10,
        min_visit_count_per_action=1,
    )

    # Verify timeout configuration is set correctly
    assert pft_dpw.time_out_in_seconds == 2
    assert pft_dpw.n_simulations is None

    # Verify other parameters are set correctly
    assert pft_dpw.depth == 5
    assert pft_dpw.k_a == 1.0
    assert pft_dpw.alpha_a == 0.5
    assert pft_dpw.k_o == 1.0
    assert pft_dpw.alpha_o == 0.5
    assert pft_dpw.exploration_constant == 1.0


def test_pft_dpw_mutual_exclusivity_constraint(environment, action_sampler):
    """Test that PFT_DPW enforces mutual exclusivity between time_out_in_seconds and n_simulations.

    Purpose: Validates that PFT_DPW raises ValueError when both time_out_in_seconds and n_simulations are provided

    Given: PFT_DPW constructor with both time_out_in_seconds=2 and n_simulations=100
    When: PFT_DPW instance creation is attempted with both parameters
    Then: ValueError is raised with appropriate error message

    Test type: unit
    """
    with pytest.raises(
        ValueError, match="Cannot specify both n_simulations and time_out_in_seconds"
    ):
        PFT_DPW(
            environment=environment,
            discount_factor=0.95,
            depth=5,
            name="PFT_DPW_Mutual_Exclusivity_Test",
            action_sampler=action_sampler,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
            time_out_in_seconds=2,  # Both parameters provided
            n_simulations=100,  # Both parameters provided
            min_samples_per_node=10,
            min_visit_count_per_action=1,
        )


def test_pft_dpw_timeout_policy_execution(environment, action_sampler):
    """Test that PFT_DPW with timeout configuration can execute policy actions.

    Purpose: Validates that PFT_DPW configured with timeout can successfully execute policy actions within time limit

    Given: PFT_DPW instance with time_out_in_seconds=1 and initial belief
    When: Policy action is requested
    Then: Action is returned successfully within timeout period

    Test type: integration
    """
    pft_dpw = PFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=3,  # Reduced depth for faster execution
        name="PFT_DPW_Timeout_Execution_Test",
        action_sampler=action_sampler,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        time_out_in_seconds=1,  # Short timeout for testing
        min_samples_per_node=5,  # Reduced for faster execution
        min_visit_count_per_action=1,
    )

    # Create initial belief
    initial_belief = get_initial_belief(
        pomdp=environment,
        n_particles=20,  # Small number of particles for testing
        resampling=True,
    )

    # Execute policy action
    action, run_data = pft_dpw.action(initial_belief)

    # Verify action is returned successfully
    assert isinstance(action, list)
    assert len(action) == 1
    assert isinstance(action[0], np.ndarray)
    assert action[0].shape == (2,)
    assert run_data is not None

    # Verify timeout configuration is preserved
    assert pft_dpw.time_out_in_seconds == 1
    assert pft_dpw.n_simulations is None


def test_pft_dpw_config_id_timeout_consistency(environment, action_sampler):
    """Test that config_id is consistent for PFT_DPW with timeout configuration.

    Purpose: Validates that PFT_DPW with identical timeout parameters produces identical config_id

    Given: Two PFT_DPW instances with identical parameters including time_out_in_seconds=3
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two PFT_DPW instances with identical timeout parameters
    pft_dpw1 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Timeout_Config_Test",
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        time_out_in_seconds=3,  # Use timeout instead of n_simulations
    )

    pft_dpw2 = PFT_DPW(
        environment=environment,
        discount_factor=0.99,
        depth=10,
        name="PFT_DPW_Timeout_Config_Test",  # Same name
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.41,
        time_out_in_seconds=3,  # Same timeout value
    )

    # Config IDs should be identical
    config_id1 = pft_dpw1.config_id
    config_id2 = pft_dpw2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0

    # Verify timeout configuration is set correctly
    assert pft_dpw1.time_out_in_seconds == 3
    assert pft_dpw2.time_out_in_seconds == 3
    assert pft_dpw1.n_simulations is None
    assert pft_dpw2.n_simulations is None
