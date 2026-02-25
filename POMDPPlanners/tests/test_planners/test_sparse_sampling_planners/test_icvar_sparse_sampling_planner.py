# pylint: disable=protected-access  # Tests need to access protected members
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling_planner import (
    ICVaRSparseSampling,
)
from POMDPPlanners.utils.statistics_utils import cvar_estimator_from_dist

np.random.seed(42)
random.seed(42)


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def initial_belief(tiger_pomdp):
    states = list(tiger_pomdp.states)
    particles = states * 10
    log_weights = np.log(np.ones(len(particles)) / len(particles))
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def planner(tiger_pomdp):
    return ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=0.3,
    )


def test_initialization(planner, tiger_pomdp):
    """Test that the ICVaR planner initializes correctly with alpha parameter.

    Purpose: Validates proper initialization of ICVaRSparseSampling with all parameters

    Given: TigerPOMDP environment, branching_factor=2, depth=2, alpha=0.3
    When: ICVaRSparseSampling is initialized
    Then: All attributes are set correctly including inherited ones

    Test type: unit
    """
    assert planner.environment == tiger_pomdp
    assert planner.branching_factor == 2
    assert planner.depth == 2
    assert planner.alpha == 0.3


def test_initialization_alpha_one(tiger_pomdp):
    """Test that alpha=1.0 is accepted as a valid boundary value.

    Purpose: Validates that alpha=1.0 (risk-neutral) is accepted

    Given: TigerPOMDP environment and alpha=1.0
    When: ICVaRSparseSampling is initialized
    Then: Planner is created successfully with alpha=1.0

    Test type: unit
    """
    planner = ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=1.0,
    )
    assert planner.alpha == 1.0


def test_invalid_alpha_zero(tiger_pomdp):
    """Test that alpha=0.0 raises ValueError.

    Purpose: Validates that alpha=0 (exclusive lower bound) is rejected

    Given: TigerPOMDP environment and alpha=0.0
    When: ICVaRSparseSampling constructor is called
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="alpha must be in"):
        ICVaRSparseSampling(
            environment=tiger_pomdp,
            branching_factor=2,
            depth=2,
            alpha=0.0,
        )


def test_invalid_alpha_negative(tiger_pomdp):
    """Test that negative alpha raises ValueError.

    Purpose: Validates that negative alpha values are rejected

    Given: TigerPOMDP environment and alpha=-0.5
    When: ICVaRSparseSampling constructor is called
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="alpha must be in"):
        ICVaRSparseSampling(
            environment=tiger_pomdp,
            branching_factor=2,
            depth=2,
            alpha=-0.5,
        )


def test_invalid_alpha_greater_than_one(tiger_pomdp):
    """Test that alpha > 1 raises ValueError.

    Purpose: Validates that alpha values exceeding 1 are rejected

    Given: TigerPOMDP environment and alpha=1.5
    When: ICVaRSparseSampling constructor is called
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="alpha must be in"):
        ICVaRSparseSampling(
            environment=tiger_pomdp,
            branching_factor=2,
            depth=2,
            alpha=1.5,
        )


def test_invalid_alpha_type(tiger_pomdp):
    """Test that non-float alpha raises TypeError.

    Purpose: Validates that non-float alpha types are rejected

    Given: TigerPOMDP environment and alpha=1 (int instead of float)
    When: ICVaRSparseSampling constructor is called
    Then: TypeError is raised

    Test type: unit
    """
    with pytest.raises(TypeError, match="alpha must be a float"):
        ICVaRSparseSampling(
            environment=tiger_pomdp,
            branching_factor=2,
            depth=2,
            alpha=1,  # type: ignore
        )


def test_action_selection(planner, initial_belief):
    """Test that action selection returns a valid action from TigerPOMDP.

    Purpose: Validates that ICVaRSparseSampling returns valid actions through sparse sampling

    Given: ICVaRSparseSampling with alpha=0.3, TigerPOMDP environment, uniform initial belief
    When: action method performs sparse sampling tree construction and action selection
    Then: Returns list with single valid tiger action and PolicyRunData with no info_variables

    Test type: unit
    """
    action, run_data = planner.action(initial_belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.environment.get_actions()
    assert isinstance(run_data, PolicyRunData)
    assert len(run_data.info_variables) == 0


def test_get_space_info():
    """Test that get_space_info returns discrete actions space.

    Purpose: Validates that ICVaRSparseSampling reports correct space information

    Given: ICVaRSparseSampling class
    When: get_space_info is called
    Then: Returns PolicySpaceInfo with DISCRETE action space and MIXED observation space

    Test type: unit
    """
    space_info = ICVaRSparseSampling.get_space_info()
    assert space_info.action_space == SpaceType.DISCRETE
    assert space_info.observation_space == SpaceType.MIXED


def test_non_leaf_action_node_uses_cvar(planner, initial_belief):
    """Test that non-leaf action node Q-value uses CVaR instead of mean.

    Purpose: Validates that _update_non_leaf_action_node_q_value computes Q-value using CVaR

    Given: ActionNode with immediate_cost=1.0, two BeliefNode children with v_values [2.0, 6.0],
           discount_factor=0.95, alpha=0.3
    When: _update_non_leaf_action_node_statistics computes q_value from children
    Then: q_value uses CVaR (not mean) of children v_values

    Test type: unit
    """
    tree = BeliefNode(belief=initial_belief)

    action_node = ActionNode(action="listen", parent=tree, children=tuple(), data=None)

    belief_node1 = BeliefNode(
        belief=initial_belief, parent=action_node, children=tuple(), data=None
    )
    belief_node1.v_value = 2.0

    belief_node2 = BeliefNode(
        belief=initial_belief, parent=action_node, children=tuple(), data=None
    )
    belief_node2.v_value = 6.0

    action_node.immediate_cost = 1.0

    planner._update_non_leaf_action_node_statistics(action_node)

    # Compute expected CVaR-based Q-value
    children_v_values = np.array([2.0, 6.0])
    uniform_weights = np.array([0.5, 0.5])
    expected_cvar = cvar_estimator_from_dist(
        values=children_v_values, weights=uniform_weights, alpha=planner.alpha
    )
    expected_q_value = 1.0 + planner.discount_factor * expected_cvar

    # Verify CVaR is used (not mean)
    mean_based_q_value = 1.0 + planner.discount_factor * float(np.mean(children_v_values))
    assert np.isclose(action_node.q_value, expected_q_value)
    assert not np.isclose(action_node.q_value, mean_based_q_value)


def test_cvar_produces_higher_q_than_mean_for_cost(planner, initial_belief):
    """Test that CVaR produces higher Q-values than mean for asymmetric cost distributions.

    Purpose: Validates that CVaR focuses on the worst-alpha outcomes in cost setting

    Given: ActionNode with children having asymmetric v_values [1.0, 1.0, 1.0, 10.0],
           alpha=0.3 (risk-sensitive)
    When: _update_non_leaf_action_node_statistics computes q_value
    Then: CVaR-based Q-value is higher than mean-based Q-value since CVaR captures tail risk

    Test type: unit
    """
    tree = BeliefNode(belief=initial_belief)
    action_node = ActionNode(action="listen", parent=tree, children=tuple(), data=None)

    v_values = [1.0, 1.0, 1.0, 10.0]
    for v_val in v_values:
        child = BeliefNode(belief=initial_belief, parent=action_node, children=tuple(), data=None)
        child.v_value = v_val

    action_node.immediate_cost = 0.0

    planner._update_non_leaf_action_node_statistics(action_node)

    mean_based = planner.discount_factor * float(np.mean(v_values))
    assert action_node.q_value >= mean_based


def test_config_id_differs_with_alpha(tiger_pomdp):
    """Test that config_id changes when alpha parameter differs.

    Purpose: Validates that different alpha values produce different config_ids

    Given: Two ICVaRSparseSampling instances with different alpha values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    planner1 = ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=0.1,
        name="ICVaRTest",
    )
    planner2 = ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=0.5,
        name="ICVaRTest",
    )

    assert planner1.config_id != planner2.config_id


def test_config_id_consistent_for_identical_parameters(tiger_pomdp):
    """Test that config_id is consistent for identical ICVaRSparseSampling parameters.

    Purpose: Validates that identical parameters produce identical config_ids

    Given: Two ICVaRSparseSampling instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both return the same config_id

    Test type: unit
    """
    planner1 = ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=0.3,
        name="ICVaRTest",
    )
    planner2 = ICVaRSparseSampling(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2,
        alpha=0.3,
        name="ICVaRTest",
    )

    assert planner1.config_id == planner2.config_id
