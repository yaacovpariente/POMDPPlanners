# pylint: disable=protected-access  # Tests need to access protected members
import random
from typing import Any, List
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
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
        if action_node2 != action_node1:
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
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)

    return_value = planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    assert tree.visit_count[root_id] > 0
    assert len(tree.children_ids[root_id]) > 0  # arena's "not is_leaf"
    assert return_value >= (-10 - environment.grid_size * np.sqrt(2)) * 5
    assert return_value <= 10


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
    for _ in range(3):
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


def test_min_visit_count_per_action_enforcement(environment, action_sampler):
    """Test that min_visit_count_per_action ensures minimum visits for each action node.

    Purpose: Validates that min_visit_count_per_action parameter correctly enforces minimum visit counts
    for each action node at the root when using k_a=2.0 and alpha_a=0.0

    Given: PFT_DPW planner with k_a=2.0, alpha_a=0.0, and min_visit_count_per_action=5
    When: Planning is performed with sufficient simulations
    Then: All action nodes at the root have at least min_visit_count_per_action visits

    Test type: unit
    """
    # ARRANGE: Setup PFT_DPW with k_a=2.0, alpha_a=0.0 to limit to 2 actions
    # and min_visit_count_per_action=5 to ensure each action gets at least 5 visits
    min_visit_count = 25
    planner = PFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        name="TestPFT_DPW_MinVisit",
        action_sampler=action_sampler,
        k_a=2.0,  # With alpha_a=0.0, this allows max 2 actions
        alpha_a=0.0,  # alpha_a=0.0 means k_a * n^0 = k_a = 2.0 (constant)
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=50,  # Enough simulations to reach min_visit_count
        min_visit_count_per_action=min_visit_count,
    )

    n_particles = 10
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)

    action_ids = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert len(action_ids) > 0, "At least one action node should be created"

    for action_id in action_ids:
        visits = tree.visit_count[action_id]
        assert visits == min_visit_count, (
            f"Action node {tree.action[action_id]} has {visits} visits, "
            f"expected at least {min_visit_count}"
        )


def test_max_depth_reached_with_timeout(environment, action_sampler):
    """Test tree structure construction with timeout.

    Purpose: Validates that PFT_DPW builds proper tree structure with BeliefNode and ActionNode hierarchy during MCTS
    when using a time-based termination criterion

    Given: PFT_DPW planner with timeout=0.5 seconds, ContinuousLightDarkPOMDP environment, initial belief
    When: MCTS tree construction creates belief-action tree structure with timeout
    Then: Tree has root BeliefNode, action children, belief grandchildren, proper parent-child relationships,
    and reaches the configured maximum depth

    Test type: unit
    """
    depth = 3
    planner = PFT_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=depth,
        name="TestPFT_DPW_MaxDepth",
        action_sampler=action_sampler,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        time_out_in_seconds=0.5,  # type: ignore[arg-type]  # Plan for 0.5 seconds
        min_visit_count_per_action=1,
    )

    n_particles = 10
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._construct_tree_using_timeout(tree=tree, root_id=root_id)

    # Walk the arena tree to compute max depth (longest path from root).
    max_observed_depth = 0
    frontier = [(root_id, 0)]
    while frontier:
        node_id, d = frontier.pop()
        max_observed_depth = max(max_observed_depth, d)
        for cid in tree.children_ids[node_id]:
            frontier.append((cid, d + 1))
    assert depth + 1 <= max_observed_depth <= 2 * depth + 2

    # Per-node invariants on the arena tree.
    for node_id in range(len(tree)):
        assert tree.visit_count[node_id] >= 0
        if tree.kind[node_id] == BELIEF:
            assert tree.belief[node_id] is not None
            assert tree.v_value[node_id] is not None
        elif tree.kind[node_id] == ACTION:
            assert tree.action[node_id] is not None
            assert tree.q_value[node_id] is not None

    # Root belief node invariants in arena form.
    assert tree.observation[root_id] is None
    assert tree.parent_id[root_id] is None
    assert len(tree.children_ids[root_id]) > 0
    assert tree.visit_count[root_id] > 0
    assert tree.v_value[root_id] is not None


# ---------------------------------------------------------------------------
# Regression test for the native-rollout-dispatcher migration.
#
# Background: PFT-DPW used to roll out via a hand-rolled Python recursion
# that allowed one final action at ``depth == self.depth`` (it returned 0
# only when ``depth > self.depth``). The migration routes the rollout
# through ``random_rollout_action_sampler`` whose Python fallback
# returns 0 when ``depth >= max_depth``. To preserve the old boundary
# we pass ``max_depth=self.depth + 1``.
#
# This test pins the step count down so a future change cannot silently
# shift the boundary by one.
# ---------------------------------------------------------------------------


class _StepCountingEnv(Environment):
    """Minimal env that counts ``sample_next_state`` calls. No native rollout kernel."""

    def __init__(self) -> None:
        super().__init__(
            discount_factor=0.95,
            name="StepCountingEnv",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
        )
        self.sample_next_state_calls = 0

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        self.sample_next_state_calls += 1
        return state if n_samples == 1 else [state] * n_samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        return 0 if n_samples == 1 else [0] * n_samples

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        return np.zeros(len(next_states))

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        return np.zeros(len(observations))

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del state, action, next_state
        return 1.0

    def is_terminal(self, state: Any) -> bool:
        return False

    def initial_state_dist(self) -> Distribution:
        mock_dist = Mock(spec=Distribution)
        mock_dist.sample = Mock(return_value=[0])
        return mock_dist

    def initial_observation_dist(self) -> Distribution:
        mock_dist = Mock(spec=Distribution)
        mock_dist.sample = Mock(return_value=[0])
        return mock_dist

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def hash_action(self, action: Any) -> Any:
        return action

    def get_actions(self) -> List[Any]:
        return [0, 1, 2]


class _SingleActionSampler(ActionSampler):
    """Always returns the same action; no randomness so step counts are deterministic."""

    def sample(self, belief_node: Any = None) -> Any:
        return 0

    def get_space(self) -> List[Any]:
        return [0]


def _make_pft_dpw_with_step_counter(self_depth: int) -> PFT_DPW:
    env = _StepCountingEnv()
    return PFT_DPW(
        environment=env,
        discount_factor=0.95,
        depth=self_depth,
        name="StepCountPFT",
        action_sampler=_SingleActionSampler(),
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=1,
        min_visit_count_per_action=1,
    )


@pytest.mark.parametrize(
    "self_depth, entry_depth, expected_steps",
    [
        # Old: while depth <= self_depth: action; recurse(depth+1). At entry=1,
        # actions taken at depths 1..self_depth (= self_depth steps).
        (5, 1, 5),
        # Entry mid-rollout: actions at depths 3..5 (= 3 steps).
        (5, 3, 3),
        # Boundary case: entry == self_depth. Old code allowed exactly one
        # final action, returning 0 on the recursive call at depth+1.
        (5, 5, 1),
        # Out-of-budget: entry == self_depth + 1. Old code returned 0
        # immediately with no actions taken.
        (5, 6, 0),
        # depth == 0 at entry shouldn't happen in practice (rollouts enter
        # at depth+1 from _simulate_return), but the bound still applies.
        (3, 0, 4),
    ],
)
def test_random_rollout_depth_semantics_preserved(
    self_depth: int, entry_depth: int, expected_steps: int
) -> None:
    """Verify that routing through ``random_rollout_action_sampler`` preserves step count.

    Purpose: Validates that the native-rollout dispatcher migration in PFT-DPW
        preserves the original Python recursion's step count exactly. The old
        code returned 0 when ``depth > self.depth`` (allowing one final action
        at ``depth == self.depth``); the new code passes ``max_depth=self.depth + 1``
        to ``random_rollout_action_sampler`` so the ``depth >= max_depth`` boundary
        stops at the same point.

    Given: A minimal counting environment with no ``simulate_random_rollout``
        attribute (so the dispatcher falls through to ``python_random_rollout``),
        a deterministic single-action sampler, a PFT-DPW planner constructed
        with ``self.depth = self_depth``, and an entry depth ``entry_depth``.
    When: ``planner._random_rollout(state=0, depth=entry_depth)`` is called once.
    Then: The number of ``sample_next_state`` calls equals ``expected_steps``,
        which matches what the original Python recursion would have produced.

    Test type: unit
    """
    planner = _make_pft_dpw_with_step_counter(self_depth=self_depth)
    env = planner.environment
    assert isinstance(env, _StepCountingEnv)
    # Sanity: the dispatcher should fall through to python_random_rollout
    # (i.e. no native rollout on this minimal env).
    assert not hasattr(env, "simulate_random_rollout")

    env.sample_next_state_calls = 0
    planner._random_rollout(state=0, depth=entry_depth)

    assert env.sample_next_state_calls == expected_steps, (
        f"Depth semantics changed: self.depth={self_depth}, entry_depth={entry_depth}, "
        f"expected {expected_steps} steps, got {env.sample_next_state_calls}."
    )


def test_tree_structure_comprehensive(planner, initial_belief):
    """Comprehensive structural invariants on a freshly-built PFT-DPW arena tree.

    Purpose: Validates that ``_learn_tree`` produces an arena ``Tree`` whose
        kind/parent/children/visit/value fields satisfy the documented invariants
        for a PFT-DPW search: belief/action level alternation, populated payloads
        per kind, non-negative visit counts, parent visits dominating the sum of
        child visits, BFS depth bounded by ``2 * planner.depth + 2``, and
        progressive widening bounds on per-node child counts.

    Given: The ``planner`` fixture (PFT_DPW with depth=5, n_simulations=100,
        k_a=k_o=1.0, alpha_a=alpha_o=0.5) and the ``initial_belief`` fixture
        (continuous light-dark POMDP belief with 20 particles).
    When: ``planner._learn_tree(initial_belief)`` builds the tree, then a BFS
        from ``root_id`` collects ``(node_id, depth)`` tuples and per-node
        invariants are checked across the full arena.
    Then: Root is a BELIEF node with no parent/observation and at least one
        child; every node has non-negative ``visit_count``; BELIEF nodes carry
        a populated ``belief``; ACTION nodes carry a populated ``action`` and
        ``q_value``; non-root nodes have a non-None parent; kinds alternate
        along every parent-child edge; visit-count is conserved (parent >=
        sum of children's visits); BFS depth <= ``2 * planner.depth + 2``;
        and progressive-widening bounds hold at every visited node.

    Test type: unit
    """
    tree, root_id = planner._learn_tree(initial_belief)

    # Root invariants.
    assert tree.kind[root_id] == BELIEF
    assert tree.parent_id[root_id] is None
    assert tree.observation[root_id] is None
    assert len(tree.children_ids[root_id]) > 0

    # BFS to collect (node_id, depth) for the bound check; meanwhile
    # record per-node depth so kind alternation can be checked against it.
    max_observed_depth = 0
    frontier = [(root_id, 0)]
    while frontier:
        node_id, d = frontier.pop()
        max_observed_depth = max(max_observed_depth, d)
        for cid in tree.children_ids[node_id]:
            frontier.append((cid, d + 1))

    assert max_observed_depth <= 2 * planner.depth + 2

    # Per-node invariants. Iterate over the full arena (every allocated id).
    n_nodes = len(tree)
    for node_id in range(n_nodes):
        # Visit count non-negative everywhere.
        assert tree.visit_count[node_id] >= 0

        # Non-root nodes have a parent; root parent is None and was checked.
        if node_id != root_id:
            assert tree.parent_id[node_id] is not None

        # Kind-specific payload presence.
        if tree.kind[node_id] == BELIEF:
            assert tree.belief[node_id] is not None
        else:
            assert tree.kind[node_id] == ACTION
            assert tree.action[node_id] is not None
            assert tree.q_value[node_id] is not None

        # Kind alternation along every parent-child edge.
        parent = tree.parent_id[node_id]
        if parent is not None:
            assert tree.kind[node_id] != tree.kind[parent]

        # Visit-count consistency: parent visits >= sum of children's visits.
        children = tree.children_ids[node_id]
        if children:
            child_visits_sum = sum(tree.visit_count[c] for c in children)
            assert tree.visit_count[node_id] >= child_visits_sum

        # Progressive-widening bounds at visited nodes only.
        if tree.visit_count[node_id] > 0:
            n_children = len(children)
            if tree.kind[node_id] == BELIEF:
                pw_bound = int(planner.k_a * (tree.visit_count[node_id] ** planner.alpha_a)) + 1
                assert n_children <= pw_bound
            else:
                pw_bound = int(planner.k_o * (tree.visit_count[node_id] ** planner.alpha_o)) + 1
                assert n_children <= pw_bound


def test_q_value_v_value_consistency(planner, initial_belief):
    """Verify ``v_value`` of every BELIEF node equals max ``q_value`` over its children.

    Purpose: Pins down the contract enforced by ``_update_node_statistics``:
        after every backup, the V-value at a belief node equals the max
        Q-value across all of its action children (visited or not). This
        guards against silent drift if a future change reorders updates,
        filters by visit count, or computes a different aggregate.

    Given: The ``planner`` fixture (PFT_DPW) and the ``initial_belief``
        fixture; ``_learn_tree`` builds the arena tree.
    When: Every BELIEF node with non-empty children is enumerated, and the
        observed ``tree.v_value[belief_id]`` is compared against
        ``max(tree.q_value[c] for c in tree.children_ids[belief_id])`` taken
        over ALL children (no visit-count filter).
    Then: The values agree within ``atol=1e-9`` (float-tolerant equality via
        ``pytest.approx``).

    Test type: unit
    """
    tree, root_id = planner._learn_tree(initial_belief)
    assert tree.kind[root_id] == BELIEF  # sanity: root is a belief node

    n_nodes = len(tree)
    n_belief_nodes_checked = 0
    for node_id in range(n_nodes):
        if tree.kind[node_id] != BELIEF:
            continue
        children = tree.children_ids[node_id]
        if not children:
            continue
        # PFT-DPW takes max over ALL children (including unvisited q=0.0).
        expected_v = max(tree.q_value[c] for c in children)
        observed_v = tree.v_value[node_id]
        assert observed_v == pytest.approx(expected_v, abs=1e-9)
        n_belief_nodes_checked += 1

    # Sanity: at least the root should have been checked.
    assert n_belief_nodes_checked >= 1
