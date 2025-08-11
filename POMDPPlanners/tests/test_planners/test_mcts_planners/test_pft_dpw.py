import pytest
import numpy as np
import random
from typing import Any
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.dpw import action_progressive_widening
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler


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
        min_visit_count_per_action=1
    )

@pytest.fixture
def initial_belief(environment):
    return get_initial_belief(
        pomdp=environment,
        n_particles=20,  # Small number of particles for testing
        resampling=True
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
    Then: New action nodes are progressively added to belief node children (1 after first call, 2 after second call)
    
    Test type: unit
    """
    belief_node = BeliefNode(belief=initial_belief)
    
    # First call should create a new action node
    action_node1 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a
    )
    assert len(belief_node.children) == 1
    
    # Second call should create another action node
    action_node2 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a
    )
    assert len(belief_node.children) == 2

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
