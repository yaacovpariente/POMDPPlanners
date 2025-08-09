import pytest
import numpy as np

from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import DiscreteActionSequencesPlanner
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.policy import PolicyRunData


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
        n_return_samples=10
    )


def test_initialization(tiger_pomdp):
    """Test initialization.
    
    Purpose: Validates proper initialization of 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    # Test valid initialization
    planner = DiscreteActionSequencesPlanner(
        environment=tiger_pomdp,
        discount_factor=0.95,
        name="test_planner",
        depth=2,
        n_return_samples=10
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
            n_return_samples=10
        )

    # Test invalid n_return_samples
    with pytest.raises(ValueError):
        DiscreteActionSequencesPlanner(
            environment=tiger_pomdp,
            discount_factor=0.95,
            name="test_planner",
            depth=2,
            n_return_samples=0
        )

    # Test invalid discount_factor
    with pytest.raises(ValueError):
        DiscreteActionSequencesPlanner(
            environment=tiger_pomdp,
            discount_factor=1.5,
            name="test_planner",
            depth=2,
            n_return_samples=10
        )


def test_action_selection(planner, tiger_pomdp):
    """Test action selection.
    
    Purpose: Validates action selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Create a belief with equal probability for both states
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights
    )

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
    belief_right = WeightedParticleBelief(
        particles=particles_right,
        log_weights=log_weights_right
    )
    actions_right, run_data_right = planner.action(belief_right)
    action_right = actions_right[0]
    assert action_right in tiger_pomdp.get_actions()
    assert isinstance(run_data_right, PolicyRunData)
    assert len(run_data_right.info_variables) == 0  # Currently no metrics are returned


def test_compute_return(planner, tiger_pomdp):
    """Test compute return.
    
    Purpose: Validates compute return
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Create a belief
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights
    )

    # Test return computation for different action sequences
    action_sequences = [
        ["listen", "listen"],
        ["listen", "open_left"],
        ["open_left", "open_right"]
    ]

    for action_sequence in action_sequences:
        return_value = planner.estimate_return(action_sequence, belief)
        assert isinstance(return_value, float)
        # Return should be finite
        assert np.isfinite(return_value)


def test_search_behavior(planner, tiger_pomdp):
    """Test search behavior.
    
    Purpose: Validates search behavior
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Create a belief
    particles = ["tiger_left", "tiger_right"] * 5  # 10 particles total
    log_weights = np.log(np.ones(10) / 10)  # Equal weights
    belief = WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights
    )

    # Test search returns a valid action sequence
    action_sequence = planner.search(belief)
    assert len(action_sequence) == planner.depth
    for action in action_sequence:
        assert action in tiger_pomdp.get_actions()


def test_integration_with_tiger_pomdp(planner, tiger_pomdp):
    """Test integration with tiger pomdp.
    
    Purpose: Validates integration with tiger pomdp
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
    # Test with different initial beliefs
    beliefs = [
        WeightedParticleBelief(
            particles=["tiger_left", "tiger_right"] * 5,
            log_weights=np.log(np.ones(10) / 10)
        ),
        WeightedParticleBelief(
            particles=["tiger_left"] * 9 + ["tiger_right"],
            log_weights=np.log(np.ones(10) / 10)
        ),
        WeightedParticleBelief(
            particles=["tiger_right"] * 9 + ["tiger_left"],
            log_weights=np.log(np.ones(10) / 10)
        )
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
