import pytest

from anytree import PostOrderIter

from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode

import time

@pytest.fixture
def discount_factor():
    return 0.95


@pytest.fixture
def depth():
    return 3


@pytest.fixture
def exploration_constant():
    return 1.0


@pytest.fixture
def n_simulations():
    return 100


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def planner(environment, discount_factor, depth, exploration_constant, n_simulations):
    return POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_simulations
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(
        pomdp=environment,
        n_particles=n_particles,
        resampling=True
    )


def test_initialization_with_n_simulations(environment, discount_factor, depth, exploration_constant):
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=100
    )
    assert planner.n_simulations == 100
    assert planner.timeout_in_seconds is None


def test_initialization_with_timeout(environment, discount_factor, depth, exploration_constant):
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        time_out_in_seconds=5
    )
    assert planner.n_simulations is None
    assert planner.timeout_in_seconds == 5


def test_invalid_initialization(environment, discount_factor, depth, exploration_constant):
    with pytest.raises(AssertionError):
        POMCP(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            n_simulations=100,
            time_out_in_seconds=5
        )


def test_action_selection(planner, belief, environment):
    action = planner.action(belief)
    assert action in environment.actions


def test_search_behavior_with_initial_belief(planner, belief, environment):
    action = planner.search(belief)
    assert action in environment.actions
    

def test_random_rollout(planner):
    state = "tiger_left"
    return_value = planner.random_rollout(state=state, depth=0)
    assert isinstance(return_value, float)


def test_integration_with_tiger_pomdp(planner, belief, environment, n_particles):
    current_belief = belief
    for _ in range(5):
        action = planner.action(current_belief)
        assert action in environment.actions
        
        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.state_transition_model(state, action).sample()
        observation = environment.observation_model(next_state, action).sample()
        
        # Update belief
        current_belief = current_belief.update(action, observation, environment)
        
        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_construct_tree_using_timeout(environment, discount_factor, depth, exploration_constant):
    timeout_seconds = 1
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        time_out_in_seconds=timeout_seconds
    )
    
    belief = get_initial_belief(environment, n_particles=100, resampling=True)
    belief_node = BeliefNode(belief=belief, observation=None)
    
    start_time = time.time()
    planner._construct_tree_using_timeout(belief=belief, belief_node=belief_node)
    end_time = time.time()
    
    # Verify the function ran for approximately the timeout duration
    assert abs(end_time - start_time - timeout_seconds) < 0.5  # Allow 0.5s margin
    
    # Verify tree structure was created
    assert len(belief_node.children) > 0  # Should have at least one action node
    assert all(isinstance(child, ActionNode) for child in belief_node.children)


def test_construct_tree_using_n_simulations(environment, discount_factor, depth, exploration_constant):
    n_sims = 50
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_sims
    )
    
    belief = get_initial_belief(environment, n_particles=100, resampling=True)
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Count total visits to verify number of simulations
    initial_visit_count = belief_node.visit_count
    
    planner._construct_tree_using_n_simulations(belief=belief, belief_node=belief_node)
    
    # Verify tree structure was created
    assert len(belief_node.children) > 0  # Should have at least one action node
    assert all(isinstance(child, ActionNode) for child in belief_node.children)
    
    # Verify total visits increased by approximately n_sims
    # Note: The actual number might be slightly different due to tree structure
    assert belief_node.visit_count >= initial_visit_count + n_sims * 0.5  # Allow for some variance


def test_tree_structure_construction(environment, discount_factor, depth, exploration_constant):
    n_sims = 100
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_sims
    )
    
    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    root_belief_node = BeliefNode(belief=belief, observation=None)
    
    planner._construct_tree_using_n_simulations(belief=belief, belief_node=root_belief_node)
    
    assert root_belief_node.height == 2 * depth + 1
    for node in PostOrderIter(root_belief_node):
        assert node.visit_count >= 0
        if isinstance(node, BeliefNode):
            assert node.belief is not None
            assert node.v_value is not None
            
            if node.height > 1 and node.depth > 0:
                assert node.v_value != 0
                assert node.observation is not None
                n_children_visits = sum(child.visit_count for child in node.children)
                assert node.visit_count == n_children_visits + 1  # +1 for the rollout

        elif isinstance(node, ActionNode):
            assert node.action is not None
            assert node.q_value is not None
            if not node.is_leaf:
                assert node.q_value != 0
                assert node.visit_count == sum(child.visit_count for child in node.children)
    
    # Verify root belief node
    assert root_belief_node.observation is None
    assert root_belief_node.parent is None
    assert len(root_belief_node.children) == len(environment.get_actions())
    assert root_belief_node.visit_count == n_sims  # 
    assert root_belief_node.v_value is not None

