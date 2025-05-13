import pytest
import numpy as np
import random
from typing import Any
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.core.environment import Environment


class RandomActionSampler(ActionSampler):
    def __init__(self, environment: Environment):
        self.environment = environment
        
    def sample(self) -> Any:
        return random.choice(self.environment.get_actions())


@pytest.fixture
def environment():
    return TigerPOMDP(discount_factor=0.95)

@pytest.fixture
def action_sampler(environment):
    return RandomActionSampler(environment=environment)

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
def initial_belief():
    return WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )

def test_initialization(planner):
    """Test that the planner initializes correctly with valid parameters."""
    assert planner.depth == 5
    assert planner.min_samples_per_node == 10
    assert planner.min_visit_count_per_action == 1
    assert planner.k_a == 1.0
    assert planner.alpha_a == 0.5
    assert planner.k_o == 1.0
    assert planner.alpha_o == 0.5
    assert planner.exploration_constant == 1.0

def test_action_sampler(action_sampler, environment):
    """Test that the action sampler returns valid actions."""
    action = action_sampler.sample()
    assert action in environment.get_actions()

def test_action_progressive_widening(planner, initial_belief):
    """Test that action progressive widening creates new action nodes when needed."""
    belief_node = BeliefNode(belief=initial_belief)
    
    # First call should create a new action node
    action_node1 = planner.action_progressive_widening(belief_node=belief_node)
    assert len(belief_node.children) == 1
    
    # Second call should create another action node
    action_node2 = planner.action_progressive_widening(belief_node=belief_node)
    assert len(belief_node.children) == 2

def test_simulate_path(planner, initial_belief):
    """Test that path simulation updates node statistics correctly."""
    belief_node = BeliefNode(belief=initial_belief)
    
    # Run a simulation
    return_value = planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Verify node statistics were updated
    assert belief_node.visit_count > 0
    assert len(belief_node.children) > 0
    
    # Verify return value is within expected range
    assert return_value >= -100  * 5  # Minimum possible reward
    assert return_value <= 10 * 5  # Maximum possible reward

def test_planning_behavior(planner, initial_belief, environment):
    """Test that the planner makes reasonable decisions in the Tiger POMDP."""
    # Get the optimal action
    optimal_action = planner.action(initial_belief)
    
    # Verify the action is valid
    assert optimal_action in environment.get_actions()
    
    # In the Tiger POMDP, the optimal first action should be "listen"
    # since it provides information about the tiger's location
    assert optimal_action == "listen"
