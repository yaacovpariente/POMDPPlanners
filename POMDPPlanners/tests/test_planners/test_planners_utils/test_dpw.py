import pytest
import numpy as np
from math import floor
from typing import Any

from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler, 
    action_progressive_widening, 
    ucb1_exploration
)
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.belief import WeightedParticleBelief


class TestActionSampler(ActionSampler):
    """Concrete implementation of ActionSampler for testing."""
    
    def __init__(self, actions=None):
        if actions is None:
            self.actions = [0, 1, 2]  # Default discrete actions
        else:
            self.actions = actions
    
    def sample(self, belief_node=None) -> Any:
        """Sample a random action from the available actions."""
        return np.random.choice(self.actions)


class TestContinuousActionSampler(ActionSampler):
    """Concrete implementation for continuous action sampling."""
    
    def __init__(self, action_bounds=(-1.0, 1.0), action_dim=2):
        self.action_bounds = action_bounds
        self.action_dim = action_dim
    
    def sample(self, belief_node=None) -> Any:
        """Sample a continuous action vector."""
        low, high = self.action_bounds
        return np.random.uniform(low, high, size=self.action_dim)


@pytest.fixture
def discrete_action_sampler():
    """Fixture for discrete action sampler."""
    return TestActionSampler(actions=[0, 1, 2])


@pytest.fixture
def continuous_action_sampler():
    """Fixture for continuous action sampler."""
    return TestContinuousActionSampler(action_bounds=(-1.0, 1.0), action_dim=2)


@pytest.fixture
def belief_node():
    """Fixture for a belief node."""
    # Create a simple belief for testing
    particles = [[0.0, 0.0], [1.0, 1.0]]  # List of particles
    log_weights = np.log(np.array([0.5, 0.5]))  # Log weights as numpy array
    belief = WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights
    )
    return BeliefNode(belief=belief)


@pytest.fixture
def belief_node_with_children(belief_node):
    """Fixture for a belief node with existing action children."""
    # Add some action nodes as children
    action1 = ActionNode(action=0, parent=belief_node)
    action1.visit_count = 5
    action1.q_value = 0.8
    
    action2 = ActionNode(action=1, parent=belief_node)
    action2.visit_count = 3
    action2.q_value = 0.6
    
    action3 = ActionNode(action=2, parent=belief_node)
    action3.visit_count = 7
    action3.q_value = 0.9
    
    belief_node.visit_count = 15
    return belief_node


def test_action_sampler_abstract_class():
    """Test that ActionSampler is an abstract base class.
    
    Purpose: Validates sampling behavior for action r abstract class
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    # Should not be able to instantiate ActionSampler directly
    with pytest.raises(TypeError):
        ActionSampler()


def test_concrete_action_sampler_implementation(discrete_action_sampler):
    """Test that concrete ActionSampler implementation works correctly.
    
    Purpose: Validates sampling behavior for concrete action r implementation
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    # Test sampling
    action = discrete_action_sampler.sample()
    assert action in discrete_action_sampler.actions
    
    # Test multiple samples
    actions = [discrete_action_sampler.sample() for _ in range(10)]
    assert all(action in discrete_action_sampler.actions for action in actions)


def test_continuous_action_sampler(continuous_action_sampler):
    """Test continuous action sampler.
    
    Purpose: Validates sampling behavior for continuous action r
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    action = continuous_action_sampler.sample()
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert all(-1.0 <= val <= 1.0 for val in action)


def test_action_progressive_widening_new_action(belief_node, discrete_action_sampler):
    """Test action progressive widening when a new action should be created.
    
    Purpose: Validates action progressive widening new action
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Set up belief node as leaf (no children)
    belief_node.visit_count = 0
    
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    
    # Should create a new action node
    assert isinstance(action_node, ActionNode)
    assert action_node.parent == belief_node
    assert action_node.action in discrete_action_sampler.actions
    assert len(belief_node.children) == 1


def test_action_progressive_widening_existing_action(belief_node_with_children, discrete_action_sampler):
    """Test action progressive widening when an existing action should be selected.
    
    Purpose: Validates action progressive widening existing action
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = belief_node_with_children
    
    # Set visit count so that floor(n^alpha) == floor((n-1)^alpha)
    # For alpha=0.5, this happens when n=1, 4, 9, 16, etc.
    belief_node.visit_count = 4  # floor(4^0.5) = 2, floor(3^0.5) = 1
    
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    
    # Should select an existing action node
    assert isinstance(action_node, ActionNode)
    assert action_node in belief_node.children


def test_action_progressive_widening_progressive_expansion(belief_node, discrete_action_sampler):
    """Test that action progressive widening expands the action space progressively.
    
    Purpose: Validates action progressive widening progressive expansion
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node.visit_count = 0
    
    # First call should create a new action
    action_node1 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert len(belief_node.children) == 1
    
    # Second call should create another action (since visit_count is still 0)
    action_node2 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert len(belief_node.children) == 2
    assert action_node1 != action_node2


def test_action_progressive_widening_alpha_parameter(belief_node, discrete_action_sampler):
    """Test that alpha parameter affects progressive widening behavior.
    
    Purpose: Validates action progressive widening alpha parameter
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node.visit_count = 1
    
    # With alpha=0.5, floor(1^0.5) = 1, floor(0^0.5) = 0, so should create new action
    action_node1 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert len(belief_node.children) == 1
    
    # Reset and test with alpha=0.1 (more conservative)
    belief_node.children = ()
    belief_node.visit_count = 1
    
    action_node2 = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.1,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert len(belief_node.children) == 1


def test_ucb1_exploration(belief_node_with_children):
    """Test UCB1 exploration for selecting existing actions.
    
    Purpose: Validates ucb1 exploration
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = belief_node_with_children
    
    action_node = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=1.0
    )
    
    # Should return one of the existing action nodes
    assert isinstance(action_node, ActionNode)
    assert action_node in belief_node.children


def test_ucb1_exploration_exploration_vs_exploitation(belief_node_with_children):
    """Test that UCB1 balances exploration and exploitation.
    
    Purpose: Validates ucb1 exploration exploration vs exploitation
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = belief_node_with_children
    
    # Test with different exploration constants
    action_node_low_exploration = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=0.1
    )
    
    action_node_high_exploration = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=2.0
    )
    
    # Both should return valid action nodes
    assert isinstance(action_node_low_exploration, ActionNode)
    assert isinstance(action_node_high_exploration, ActionNode)
    assert action_node_low_exploration in belief_node.children
    assert action_node_high_exploration in belief_node.children


def test_ucb1_exploration_mathematical_correctness(belief_node_with_children):
    """Test that UCB1 calculation is mathematically correct.
    
    Purpose: Validates ucb1 exploration mathematical correctness
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = belief_node_with_children
    
    # Calculate UCB1 values manually
    q_vals = [child.q_value for child in belief_node.children]
    children_visit_counts = [child.visit_count for child in belief_node.children]
    
    exploration_constant = 1.0
    ucb_values = [
        q_val + exploration_constant * np.sqrt(np.log(belief_node.visit_count) / visit_count)
        for q_val, visit_count in zip(q_vals, children_visit_counts)
    ]
    
    expected_action_node = belief_node.children[np.argmax(ucb_values)]
    
    # Compare with function result
    actual_action_node = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=exploration_constant
    )
    
    assert actual_action_node == expected_action_node


def test_action_progressive_widening_edge_cases(belief_node, discrete_action_sampler):
    """Test edge cases for action progressive widening.
    
    Purpose: Validates action progressive widening edge cases
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test with alpha_a = 0
    belief_node.visit_count = 1
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.0,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert isinstance(action_node, ActionNode)
    
    # Test with alpha_a = 1
    belief_node.children = ()
    belief_node.visit_count = 1
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=1.0,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert isinstance(action_node, ActionNode)


def test_ucb1_exploration_edge_cases(belief_node_with_children):
    """Test edge cases for UCB1 exploration.
    
    Purpose: Validates ucb1 exploration edge cases
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = belief_node_with_children
    
    # Test with exploration_constant = 0 (pure exploitation)
    action_node = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=0.0
    )
    assert isinstance(action_node, ActionNode)
    
    # Test with very high exploration_constant
    action_node = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=100.0
    )
    assert isinstance(action_node, ActionNode)


def test_action_progressive_widening_integration(belief_node, discrete_action_sampler):
    """Test integration of action progressive widening with UCB1 exploration.
    
    Purpose: Validates action progressive widening integration
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
    belief_node.visit_count = 0
    
    # First few calls should create new actions
    for i in range(3):
        action_node = action_progressive_widening(
            belief_node=belief_node,
            alpha_a=0.5,
            action_sampler=discrete_action_sampler,
            exploration_constant=1.0
        )
        assert isinstance(action_node, ActionNode)
        assert action_node.parent == belief_node
    
    # After creating some actions, should start using UCB1
    belief_node.visit_count = 4  # floor(4^0.5) = 2, floor(3^0.5) = 1
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,
        action_sampler=discrete_action_sampler,
        exploration_constant=1.0
    )
    assert action_node in belief_node.children


def test_action_sampler_with_belief_node_context(discrete_action_sampler, belief_node):
    """Test that action sampler can optionally use belief node context.
    
    Purpose: Validates sampling behavior for action r with belief node context
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    # This test verifies that the action sampler interface supports belief node context
    action = discrete_action_sampler.sample(belief_node=belief_node)
    assert action in discrete_action_sampler.actions
    
    # Test without context
    action = discrete_action_sampler.sample()
    assert action in discrete_action_sampler.actions


# Usage Example Tests
# Following the project standard of testing all usage examples from docstrings

def test_continuous_control_sampler_usage_example():
    """Test the ContinuousControlSampler usage example from ActionSampler docstring.
    
    Purpose: Validates sampling behavior for continuous control r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    import numpy as np
    
    class ContinuousControlSampler(ActionSampler):
        def __init__(self, action_bounds=(-1.0, 1.0), action_dim=2):
            self.action_bounds = action_bounds
            self.action_dim = action_dim
        
        def sample(self, belief_node=None):
            # Sample uniformly from action space
            low, high = self.action_bounds
            return np.random.uniform(low, high, size=self.action_dim)
    
    # Usage with PFT-DPW (from docstring)
    sampler = ContinuousControlSampler(action_bounds=(-2.0, 2.0), action_dim=4)
    action = sampler.sample()  # Returns 4D action vector
    
    # Verify action properties
    assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
    assert len(action) == 4, f"Expected 4D action, got {len(action)} dimensions"
    assert all(-2.0 <= a <= 2.0 for a in action), f"Action values outside bounds: {action}"


def test_weighted_discrete_action_sampler_usage_example():
    """Test the WeightedDiscreteActionSampler usage example from ActionSampler docstring.
    
    Purpose: Validates sampling behavior for weighted discrete action r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    import numpy as np
    
    class WeightedDiscreteActionSampler(ActionSampler):
        def __init__(self, actions, probabilities=None):
            self.actions = actions
            # Use uniform probabilities if none provided
            if probabilities is None:
                self.probabilities = np.ones(len(actions)) / len(actions)
            else:
                self.probabilities = np.array(probabilities)
                self.probabilities /= np.sum(self.probabilities)  # Normalize
        
        def sample(self, belief_node=None):
            return np.random.choice(self.actions, p=self.probabilities)
    
    # Prefer certain actions over others (from docstring)
    actions = ["up", "down", "left", "right", "stay"]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform
    sampler = WeightedDiscreteActionSampler(actions, probs)
    
    # Test multiple samples
    samples = [sampler.sample() for _ in range(100)]
    
    # Verify all samples are valid actions
    assert all(action in actions for action in samples), "Some sampled actions are invalid"
    
    # Verify probabilities are normalized
    assert np.isclose(np.sum(sampler.probabilities), 1.0), "Probabilities don't sum to 1"


def test_adaptive_action_sampler_usage_example():
    """Test the AdaptiveActionSampler usage example from ActionSampler docstring.
    
    Purpose: Validates sampling behavior for adaptive action r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    from POMDPPlanners.core.tree import BeliefNode, ActionNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    import numpy as np
    
    class AdaptiveActionSampler(ActionSampler):
        def __init__(self, base_actions, exploration_noise=0.1):
            self.base_actions = base_actions
            self.exploration_noise = exploration_noise
        
        def sample(self, belief_node=None):
            if belief_node is not None and belief_node.visit_count > 10:
                # Use belief state to inform sampling
                best_action = self._get_best_action_from_belief(belief_node)
                # Add exploration noise
                noise = np.random.normal(0, self.exploration_noise, len(best_action))
                return best_action + noise
            else:
                # Random exploration for new nodes
                return np.random.choice(self.base_actions)
        
        def _get_best_action_from_belief(self, belief_node):
            # Simplified: return action from best child
            if belief_node.children:
                best_child = max(belief_node.children, key=lambda x: x.q_value)
                return best_child.action
            return np.random.choice(self.base_actions)
    
    sampler = AdaptiveActionSampler([0, 1, 2, 3], exploration_noise=0.05)
    
    # Test with no belief node (should use random exploration)
    action_no_belief = sampler.sample()
    assert action_no_belief in sampler.base_actions, "Action not from base actions"
    
    # Test with low visit count belief node (should use random exploration)
    particles = [[0.0], [1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    belief_node.visit_count = 5  # Low visit count
    
    action_low_visits = sampler.sample(belief_node=belief_node)
    assert action_low_visits in sampler.base_actions, "Action not from base actions"
    
    # Test with high visit count and children (should use informed sampling)
    belief_node.visit_count = 15
    action_child = ActionNode(action=np.array([1, 0]), parent=belief_node)
    action_child.q_value = 0.8
    
    action_informed = sampler.sample(belief_node=belief_node)
    # Should be the best action plus noise, so roughly [1, 0] + noise
    assert isinstance(action_informed, np.ndarray), "Expected array for informed sampling"
    assert len(action_informed) == 2, "Expected 2D action for informed sampling"


def test_multi_modal_action_sampler_usage_example():
    """Test the MultiModalActionSampler usage example from ActionSampler docstring.
    
    Purpose: Validates sampling behavior for multi modal action r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    import numpy as np
    
    class MultiModalActionSampler(ActionSampler):
        def __init__(self, discrete_actions, continuous_bounds, mode_prob=0.5):
            self.discrete_actions = discrete_actions
            self.continuous_bounds = continuous_bounds
            self.mode_prob = mode_prob  # Probability of discrete vs continuous
        
        def sample(self, belief_node=None):
            if np.random.random() < self.mode_prob:
                # Sample discrete action
                return {"type": "discrete", "action": np.random.choice(self.discrete_actions)}
            else:
                # Sample continuous action
                low, high = self.continuous_bounds
                continuous_action = np.random.uniform(low, high, size=2)
                return {"type": "continuous", "action": continuous_action}
    
    # For environments with both discrete and continuous actions (from docstring)
    discrete_acts = ["stop", "emergency_brake", "lane_change"]
    continuous_bounds = (-5.0, 5.0)  # Steering/acceleration range
    sampler = MultiModalActionSampler(discrete_acts, continuous_bounds)
    
    # Test multiple samples to verify both modes work
    samples = [sampler.sample() for _ in range(50)]
    discrete_samples = [s for s in samples if s["type"] == "discrete"]
    continuous_samples = [s for s in samples if s["type"] == "continuous"]
    
    # Should have both types
    assert len(discrete_samples) > 0, "No discrete actions sampled"
    assert len(continuous_samples) > 0, "No continuous actions sampled"
    
    # Verify discrete samples
    for sample in discrete_samples:
        assert sample["action"] in discrete_acts, f"Invalid discrete action: {sample['action']}"
    
    # Verify continuous samples
    for sample in continuous_samples:
        action = sample["action"]
        assert isinstance(action, np.ndarray), "Continuous action should be ndarray"
        assert len(action) == 2, "Expected 2D continuous action"
        assert all(-5.0 <= a <= 5.0 for a in action), f"Continuous action out of bounds: {action}"


def test_goal_directed_action_sampler_usage_example():
    """Test the GoalDirectedActionSampler usage example from ActionSampler docstring.
    
    Purpose: Validates sampling behavior for goal directed action r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    from POMDPPlanners.core.tree import BeliefNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    import numpy as np
    
    class GoalDirectedActionSampler(ActionSampler):
        def __init__(self, goal_position, action_magnitude=1.0, goal_bias=0.7):
            self.goal_position = np.array(goal_position)
            self.action_magnitude = action_magnitude
            self.goal_bias = goal_bias
        
        def sample(self, belief_node=None):
            if np.random.random() < self.goal_bias and belief_node is not None:
                # Sample action towards goal based on current belief
                current_position = self._estimate_position(belief_node)
                direction = self.goal_position - current_position
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    return direction * self.action_magnitude
            
            # Random exploration
            angle = np.random.uniform(0, 2 * np.pi)
            return self.action_magnitude * np.array([np.cos(angle), np.sin(angle)])
        
        def _estimate_position(self, belief_node):
            # Simplified: use mean of particles in belief
            if hasattr(belief_node.belief, 'particles'):
                positions = [p[:2] for p in belief_node.belief.particles]  # First 2D as position
                return np.mean(positions, axis=0)
            return np.array([0.0, 0.0])
    
    # Navigation towards specific goal (from docstring)
    goal = [10.0, 5.0]
    sampler = GoalDirectedActionSampler(goal, action_magnitude=2.0, goal_bias=0.8)
    
    # Test random exploration (no belief node)
    action_random = sampler.sample()
    assert isinstance(action_random, np.ndarray), "Expected ndarray action"
    assert len(action_random) == 2, "Expected 2D action"
    assert np.isclose(np.linalg.norm(action_random), 2.0), "Action magnitude should be 2.0"
    
    # Test goal-directed behavior (with belief node containing position particles)
    particles = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]  # Positions near origin
    log_weights = np.log(np.array([1/3, 1/3, 1/3]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    
    # Multiple samples to test goal-directed behavior
    goal_directed_actions = [sampler.sample(belief_node=belief_node) for _ in range(10)]
    
    for action in goal_directed_actions:
        assert isinstance(action, np.ndarray), "Expected ndarray action"
        assert len(action) == 2, "Expected 2D action"
        assert np.linalg.norm(action) <= 2.1, "Action magnitude should be approximately 2.0"


def test_action_progressive_widening_basic_usage_example():
    """Test the basic action_progressive_widening usage example from function docstring.
    
    Purpose: Validates action progressive widening basic usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
    from POMDPPlanners.core.tree import BeliefNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Create action sampler (from docstring)
    class SimpleActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.uniform(-1, 1, size=2)
    
    # Create belief node (from docstring)
    particles = [[0.0, 0.0], [1.0, 1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    
    # Progressive widening (from docstring)
    action_sampler = SimpleActionSampler()
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=0.5,  # Moderate exploration
        action_sampler=action_sampler,
        exploration_constant=1.0
    )
    
    # Verify results
    assert action_node.parent == belief_node, "Action node should have belief node as parent"
    assert isinstance(action_node.action, np.ndarray), "Action should be ndarray"
    assert len(action_node.action) == 2, "Action should be 2D"
    assert all(-1 <= a <= 1 for a in action_node.action), "Action should be in [-1, 1] range"
    assert len(belief_node.children) == 1, "Should have created one action node"


def test_action_progressive_widening_alpha_comparison_example():
    """Test the alpha_a comparison example from action_progressive_widening docstring.
    
    Purpose: Validates action progressive widening alpha comparison example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
    from POMDPPlanners.core.tree import BeliefNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    class SimpleActionSampler(ActionSampler):
        def sample(self, belief_node=None):
            return np.random.uniform(-1, 1, size=2)
    
    particles = [[0.0, 0.0], [1.0, 1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    action_sampler = SimpleActionSampler()
    
    # Conservative exploration (fewer new actions) - from docstring
    belief_node_conservative = BeliefNode(belief=belief)
    conservative_action = action_progressive_widening(
        belief_node=belief_node_conservative,
        alpha_a=0.25,  # Low alpha = fewer actions
        action_sampler=action_sampler,
        exploration_constant=1.0
    )
    
    # Aggressive exploration (more new actions) - from docstring
    belief_node_aggressive = BeliefNode(belief=belief)
    aggressive_action = action_progressive_widening(
        belief_node=belief_node_aggressive,
        alpha_a=0.75,  # High alpha = more actions
        action_sampler=action_sampler,
        exploration_constant=1.0
    )
    
    # Both should create action nodes for initial calls
    assert conservative_action.parent == belief_node_conservative, "Conservative action should be child of belief node"
    assert aggressive_action.parent == belief_node_aggressive, "Aggressive action should be child of belief node"
    assert len(belief_node_conservative.children) == 1, "Conservative should create one action"
    assert len(belief_node_aggressive.children) == 1, "Aggressive should create one action"


def test_action_progressive_widening_loop_simulation_example():
    """Test the progressive widening loop simulation example from docstring.
    
    Purpose: Validates action progressive widening loop simulation example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
    from POMDPPlanners.core.tree import BeliefNode, ActionNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Setup (from docstring)
    class DiscreteActionSampler(ActionSampler):
        def __init__(self, actions):
            self.actions = actions
            
        def sample(self, belief_node=None):
            return np.random.choice(self.actions)
    
    particles = [[0], [1], [2]]
    log_weights = np.log(np.array([1/3, 1/3, 1/3]))
    belief = WeightedParticleBelief(particles, log_weights)
    root_node = BeliefNode(belief=belief)
    
    sampler = DiscreteActionSampler(['up', 'down', 'left', 'right'])
    
    # Simulate multiple selections (from docstring)
    action_counts = []
    selected_actions = []
    
    for i in range(10):
        root_node.visit_count = i  # Simulate increasing visits
        action_node = action_progressive_widening(
            belief_node=root_node,
            alpha_a=0.5,
            action_sampler=sampler,
            exploration_constant=1.41  # sqrt(2)
        )
        action_counts.append(len(root_node.children))
        selected_actions.append(action_node.action)
    
    # Verify progressive behavior
    assert action_counts[0] >= 1, "Should create at least one action initially"
    assert action_counts[-1] <= 4, "Should not exceed number of possible actions"
    assert all(action in sampler.actions for action in selected_actions), "All actions should be valid"
    
    # Action count should generally increase or stay the same (non-decreasing)
    for i in range(1, len(action_counts)):
        assert action_counts[i] >= action_counts[i-1], f"Action count decreased at step {i}"


def test_ucb1_exploration_basic_usage_example():
    """Test the basic UCB1 exploration usage example from function docstring.
    
    Purpose: Validates ucb1 exploration basic usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration
    from POMDPPlanners.core.tree import BeliefNode, ActionNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Create belief node with action children (from docstring)
    particles = [[0.0], [1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    belief_node.visit_count = 100
    
    # Add action nodes with different Q-values and visit counts (from docstring)
    actions_data = [
        {"action": "up", "q_value": 0.8, "visits": 30},
        {"action": "down", "q_value": 0.6, "visits": 20}, 
        {"action": "left", "q_value": 0.9, "visits": 40},
        {"action": "right", "q_value": 0.4, "visits": 10}
    ]
    
    for data in actions_data:
        action_node = ActionNode(action=data["action"], parent=belief_node)
        action_node.q_value = data["q_value"]
        action_node.visit_count = data["visits"]
    
    # Select action using UCB1 (from docstring)
    selected_action = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=1.41  # sqrt(2)
    )
    
    # Verify results
    assert selected_action in belief_node.children, "Selected action should be one of the children"
    assert selected_action.action in ["up", "down", "left", "right"], "Selected action should be valid"


def test_ucb1_exploration_constants_comparison_example():
    """Test the exploration constants comparison example from UCB1 docstring.
    
    Purpose: Validates ucb1 exploration constants comparison example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration
    from POMDPPlanners.core.tree import BeliefNode, ActionNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Setup belief node with actions
    particles = [[0.0], [1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    belief_node.visit_count = 100
    
    actions_data = [
        {"action": "up", "q_value": 0.8, "visits": 30},
        {"action": "down", "q_value": 0.6, "visits": 20},
    ]
    
    for data in actions_data:
        action_node = ActionNode(action=data["action"], parent=belief_node)
        action_node.q_value = data["q_value"]
        action_node.visit_count = data["visits"]
    
    # Low exploration (favor exploitation) - from docstring
    conservative_action = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=0.1
    )
    
    # High exploration (favor exploration) - from docstring  
    exploratory_action = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=3.0
    )
    
    # Balanced approach (theoretical optimum) - from docstring
    balanced_action = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=1.41  # sqrt(2)
    )
    
    # All should return valid actions
    assert conservative_action in belief_node.children, "Conservative selection should be valid child"
    assert exploratory_action in belief_node.children, "Exploratory selection should be valid child"
    assert balanced_action in belief_node.children, "Balanced selection should be valid child"


def test_ucb1_exploration_manual_calculation_example():
    """Test the manual UCB1 calculation example from function docstring.
    
    Purpose: Validates ucb1 exploration manual calculation example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    import numpy as np
    from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration
    from POMDPPlanners.core.tree import BeliefNode, ActionNode
    from POMDPPlanners.core.belief import WeightedParticleBelief
    
    # Setup belief node
    particles = [[0.0], [1.0]]
    log_weights = np.log(np.array([0.5, 0.5]))
    belief = WeightedParticleBelief(particles, log_weights)
    belief_node = BeliefNode(belief=belief)
    belief_node.visit_count = 50
    
    actions_data = [
        {"action": "action1", "q_value": 0.7, "visits": 15},
        {"action": "action2", "q_value": 0.5, "visits": 10},
    ]
    
    for data in actions_data:
        action_node = ActionNode(action=data["action"], parent=belief_node)
        action_node.q_value = data["q_value"]
        action_node.visit_count = data["visits"]
    
    # Calculate UCB1 values manually (from docstring)
    exploration_constant = 1.0
    ucb1_values = []
    
    for child in belief_node.children:
        exploration_term = exploration_constant * np.sqrt(
            np.log(belief_node.visit_count) / child.visit_count
        )
        ucb1 = child.q_value + exploration_term
        ucb1_values.append(ucb1)
    
    # Verify our function selects the highest UCB1 (from docstring)
    expected_best_idx = np.argmax(ucb1_values)
    selected_action = ucb1_exploration(belief_node, exploration_constant)
    actual_best_idx = list(belief_node.children).index(selected_action)
    
    assert expected_best_idx == actual_best_idx, "UCB1 selection should match manual calculation"
    
    # Verify UCB1 values are calculated correctly
    for i, child in enumerate(belief_node.children):
        expected_exploration = exploration_constant * np.sqrt(np.log(belief_node.visit_count) / child.visit_count)
        expected_ucb1 = child.q_value + expected_exploration
        assert np.isclose(ucb1_values[i], expected_ucb1), f"UCB1 calculation mismatch for child {i}"


def test_progressive_widening_parameter_tuning_example():
    """Test the progressive widening parameter tuning example from docstring.
    
    Purpose: Validates progressive widening parameter tuning example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    from math import floor
    
    # Effect of alpha_a on action creation (from docstring)
    visit_counts = range(1, 21)
    alpha_values = [0.25, 0.5, 0.75, 1.0]
    
    results = {}
    for alpha in alpha_values:
        action_counts = []
        for n in visit_counts:
            # Calculate when new actions would be created (from docstring)
            should_create = floor(n ** alpha) > floor((n-1) ** alpha) if n > 0 else True
            action_counts.append(1 if should_create else 0)
        
        total_new_actions = sum(action_counts)
        results[alpha] = total_new_actions
    
    # Verify expected behavior: higher alpha should create more actions
    assert results[0.25] <= results[0.5], "Lower alpha should create fewer or equal actions"
    assert results[0.5] <= results[0.75], "Lower alpha should create fewer or equal actions"
    assert results[0.75] <= results[1.0], "Lower alpha should create fewer or equal actions"
    
    # All should create at least one action (at visit count 1)
    assert all(count >= 1 for count in results.values()), "All alpha values should create at least one action"
    
    # Alpha = 1.0 should create an action at every visit (linear growth)
    assert results[1.0] == 20, "Alpha = 1.0 should create action at every visit"
