#!/usr/bin/env python3
"""Test script to verify all usage examples from MCTS planners classes work correctly."""

import sys
import traceback
import numpy as np
import os
from pathlib import Path

# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pomcp_comprehensive_usage_example():
    """Test the comprehensive POMCP usage example from the class docstring.
    
    Purpose: Validates pomcp comprehensive usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing POMCP comprehensive usage example...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        
        # Create environment and planner (using reduced parameters for testing)
        env = TigerPOMDP(discount_factor=0.95)
        planner = POMCP(
            environment=env,
            discount_factor=env.discount_factor,
            depth=5,                   # Reduced for testing
            exploration_constant=1.0,
            name="POMCP_Tiger",
            n_simulations=10          # Reduced for testing
        )
        
        # Plan action from initial belief
        initial_belief = get_initial_belief(env, n_particles=50)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Verify results match documented example
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree stats available: {len(run_data.info_variables) > 0}")
        print("  ✓ POMCP comprehensive usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ POMCP comprehensive usage example failed: {e}")
        traceback.print_exc()
        return False

def test_pft_dpw_cartpole_usage_example():
    """Test the CartPole usage example from PFT_DPW class docstring.
    
    Purpose: Validates pft dpw cartpole usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing PFT-DPW CartPole usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
        from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Custom action sampler for CartPole (from docstring)
        class CartPoleActionSampler(ActionSampler):
            def sample(self, belief_node=None):
                return np.random.choice([0, 1])  # Left or right force
        
        # Create CartPole environment
        cartpole = CartPolePOMDP(
            discount_factor=0.99,
            noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
        )
        
        # Create PFT-DPW planner (reduced parameters for testing)
        pft_planner = PFT_DPW(
            environment=cartpole,
            discount_factor=0.99,
            depth=5,                    # Reduced for testing
            name="PFT_DPW_CartPole",
            action_sampler=CartPoleActionSampler(),
            k_a=2.0,                    # Action widening parameter
            alpha_a=0.5,                # Action widening exponent
            k_o=1.0,                    # Observation widening parameter  
            alpha_o=0.5,                # Observation widening exponent
            exploration_constant=1.41,   # UCB1 exploration (√2)
            n_simulations=5             # Reduced for testing
        )
        
        # Plan from initial belief
        initial_belief = get_initial_belief(cartpole, n_particles=25)  # Reduced for testing
        action, run_data = pft_planner.action(initial_belief)
        
        # Verify results match documented example
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in [0, 1], f"Expected action 0 or 1, got {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Planning completed with {len(run_data.info_variables)} metrics")
        print("  ✓ PFT-DPW CartPole usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PFT-DPW CartPole usage example failed: {e}")
        traceback.print_exc()
        return False

def test_pft_dpw_navigation_usage_example():
    """Test the 2D navigation usage example from PFT_DPW class docstring.
    
    Purpose: Validates pft dpw navigation usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing PFT-DPW 2D navigation usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        
        # Test the NavigationActionSampler from the docstring
        class NavigationActionSampler(ActionSampler):
            def __init__(self, max_velocity=1.0):
                self.max_velocity = max_velocity
            
            def sample(self, belief_node=None):
                # Sample 2D velocity vector
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0, self.max_velocity)
                return np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        # Test the action sampler functionality
        navigation_sampler = NavigationActionSampler(max_velocity=2.0)
        
        # Sample multiple actions to verify functionality
        actions = []
        for _ in range(5):
            action = navigation_sampler.sample()
            actions.append(action)
            
            # Verify action properties
            assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
            assert len(action) == 2, f"Expected 2D action, got {len(action)} dimensions"
            
            speed = np.linalg.norm(action)
            assert speed <= 2.0, f"Speed {speed} exceeds max_velocity {2.0}"
        
        # Verify actions are different (randomness check)
        action_differences = [np.linalg.norm(actions[i] - actions[0]) for i in range(1, 5)]
        assert any(diff > 0.1 for diff in action_differences), "Actions appear to be too similar (randomness issue)"
        
        print(f"  ✓ Action sampler generates 2D velocity vectors")
        print(f"  ✓ Speed constraints respected: max observed = {max(np.linalg.norm(a) for a in actions):.3f}")
        print(f"  ✓ Actions show appropriate randomness")
        print("  ✓ PFT-DPW 2D navigation usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PFT-DPW 2D navigation usage example failed: {e}")
        traceback.print_exc()
        return False

def test_action_sampler_usage_example():
    """Test the ActionSampler usage example from the class docstring.
    
    Purpose: Validates sampling behavior for action r usage example
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: example
    """
    print("Testing ActionSampler usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        
        # ContinuousControlSampler from docstring
        class ContinuousControlSampler(ActionSampler):
            def __init__(self, action_bounds=(-1.0, 1.0), action_dim=2):
                self.action_bounds = action_bounds
                self.action_dim = action_dim
            
            def sample(self, belief_node=None):
                # Sample uniformly from action space
                low, high = self.action_bounds
                return np.random.uniform(low, high, size=self.action_dim)
        
        # Test the usage example from docstring
        sampler = ContinuousControlSampler(action_bounds=(-2.0, 2.0), action_dim=4)
        
        # Test sampling functionality
        action = sampler.sample()
        
        # Verify action properties
        assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
        assert len(action) == 4, f"Expected 4D action, got {len(action)} dimensions"
        assert all(-2.0 <= a <= 2.0 for a in action), f"Action values outside bounds: {action}"
        
        # Test multiple samples for consistency
        actions = [sampler.sample() for _ in range(10)]
        assert all(len(a) == 4 for a in actions), "Inconsistent action dimensions"
        assert all(all(-2.0 <= val <= 2.0 for val in a) for a in actions), "Some actions outside bounds"
        
        print(f"  ✓ Sampler generates {len(action)}D actions within bounds [-2.0, 2.0]")
        print(f"  ✓ Sample action: [{', '.join(f'{x:.3f}' for x in action)}]")
        print("  ✓ ActionSampler usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ ActionSampler usage example failed: {e}")
        traceback.print_exc()
        return False

def test_path_simulation_policy_custom_mcts_example():
    """Test the custom MCTS implementation from PathSimulationPolicy docstring.
    
    Purpose: Validates path simulation policy custom mcts example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing PathSimulationPolicy custom MCTS usage example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.path_simulations_policy import PathSimulationPolicy
        from POMDPPlanners.core.tree import BeliefNode, ActionNode
        from POMDPPlanners.core.environment import Environment
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        import numpy as np
        
        # CustomMCTS implementation from docstring (simplified for testing)
        class CustomMCTS(PathSimulationPolicy):
            def __init__(self, environment: Environment, discount_factor: float, 
                       name: str, n_simulations: int, exploration_constant: float = 1.0):
                super().__init__(
                    environment=environment,
                    discount_factor=discount_factor,
                    name=name,
                    n_simulations=n_simulations,
                    time_out_in_seconds=None
                )
                self.exploration_constant = exploration_constant
            
            def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
                # Custom implementation of MCTS simulation (simplified for testing)
                if depth > 5:  # Reduced max depth for testing
                    return 0.0
                
                if belief_node.is_leaf:
                    # Expand node with all available actions
                    for action in self.environment.get_actions():
                        ActionNode(action=action, parent=belief_node)
                    return self._random_rollout(belief_node.belief.sample(), depth)
                
                # UCB1 action selection
                action_node = self._select_best_action(belief_node)
                
                # Simulate step and continue recursively
                state = belief_node.belief.sample()
                next_state, observation, reward = self.environment.sample_next_step(
                    state=state, action=action_node.action
                )
                next_belief = belief_node.belief.update(
                    action=action_node.action, 
                    observation=observation, 
                    pomdp=self.environment
                )
                next_node = BeliefNode(belief=next_belief, parent=action_node)
                
                future_value = self._simulate_path(next_node, depth + 1)
                total_return = reward + self.discount_factor * future_value
                
                # Update statistics
                action_node.visit_count += 1
                action_node.q_value += (total_return - action_node.q_value) / action_node.visit_count
                belief_node.visit_count += 1
                
                return total_return
            
            def _select_best_action(self, belief_node: BeliefNode) -> ActionNode:
                # UCB1 action selection
                best_ucb = float('-inf')
                best_action = None
                
                for child in belief_node.children:
                    if child.visit_count == 0:
                        return child  # Prioritize unvisited actions
                    
                    ucb = child.q_value + self.exploration_constant * np.sqrt(
                        np.log(belief_node.visit_count) / child.visit_count
                    )
                    
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_action = child
                
                return best_action
            
            def _random_rollout(self, state, depth: int) -> float:
                # Simple random rollout for value estimation
                if depth > 8 or self.environment.is_terminal(state):  # Reduced depth for testing
                    return 0.0
                
                action = np.random.choice(self.environment.get_actions())
                next_state, _, reward = self.environment.sample_next_step(state, action)
                
                return reward + self.discount_factor * self._random_rollout(next_state, depth + 1)
        
        # Usage example from docstring
        env = TigerPOMDP(discount_factor=0.95)
        planner = CustomMCTS(
            environment=env,
            discount_factor=0.95,
            name="CustomMCTS_Tiger",
            n_simulations=5,            # Reduced for testing
            exploration_constant=1.41   # √2
        )
        
        initial_belief = get_initial_belief(env, n_particles=50)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Verify results match documented example
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree metrics: {[m.name for m in run_data.info_variables]}")
        print("  ✓ PathSimulationPolicy custom MCTS usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PathSimulationPolicy custom MCTS usage example failed: {e}")
        traceback.print_exc()
        return False

def test_sparse_pft_tiger_usage_example():
    """Test the Tiger POMDP usage example from SparsePFT class docstring.
    
    Purpose: Validates sparse pft tiger usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing SparsePFT Tiger usage example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create Tiger POMDP environment
        tiger = TigerPOMDP(discount_factor=0.95)
        
        # Create Sparse-PFT planner with controlled branching (reduced parameters for testing)
        sparse_pft = SparsePFT(
            environment=tiger,
            discount_factor=0.95,
            gamma=0.95,                  # Discount for recursive calls
            depth=5,                     # Reduced for testing
            c_ucb=1.0,                   # Base exploration constant
            beta_ucb=2.0,                # Enhanced exploration parameter
            belief_child_num=3,          # Reduced for testing
            n_simulations=8,             # Reduced for testing
            name="SparsePFT_Tiger"
        )
        
        # Plan action from initial belief
        initial_belief = get_initial_belief(tiger, n_particles=50)  # Reduced for testing
        action, run_data = sparse_pft.action(initial_belief)
        
        # Verify results match documented example
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree metrics collected: {len(run_data.info_variables)}")
        
        # Access tree statistics for analysis (from docstring)
        tree_metrics = run_data.info_variables
        metric_names = [metric.name for metric in tree_metrics]
        print(f"  ✓ Available metrics: {metric_names}")
        
        print("  ✓ SparsePFT Tiger usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ SparsePFT Tiger usage example failed: {e}")
        traceback.print_exc()
        return False

def test_sparse_pft_parameter_comparison_example():
    """Test the parameter comparison example from SparsePFT class docstring.
    
    Purpose: Validates sparse pft parameter comparison example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing SparsePFT parameter comparison usage example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        tiger = TigerPOMDP(discount_factor=0.95)
        initial_belief = get_initial_belief(tiger, n_particles=30)  # Reduced for testing
        
        # Conservative branching (small tree, faster planning) - from docstring
        conservative_pft = SparsePFT(
            environment=tiger,
            discount_factor=0.95,
            gamma=0.95,
            depth=8,                     # Reduced for testing
            c_ucb=1.0,
            beta_ucb=1.0,                # Lower exploration
            belief_child_num=2,          # Reduced for testing
            n_simulations=3,             # Reduced for testing
            name="Conservative_SparsePFT"
        )
        
        # Aggressive branching (larger tree, more thorough search) - from docstring
        aggressive_pft = SparsePFT(
            environment=tiger,
            discount_factor=0.95,
            gamma=0.95,
            depth=5,                     # Reduced for testing
            c_ucb=2.0,                   # Higher base exploration
            beta_ucb=3.0,                # Higher enhanced exploration
            belief_child_num=4,          # Reduced for testing
            n_simulations=6,             # Reduced for testing
            name="Aggressive_SparsePFT"
        )
        
        # Compare performance (from docstring)
        conservative_action, conservative_data = conservative_pft.action(initial_belief)
        aggressive_action, aggressive_data = aggressive_pft.action(initial_belief)
        
        # Verify both return valid results
        assert len(conservative_action) == 1, "Conservative planner should return single action"
        assert len(aggressive_action) == 1, "Aggressive planner should return single action"
        assert conservative_action[0] in ["listen", "open_left", "open_right"], "Invalid conservative action"
        assert aggressive_action[0] in ["listen", "open_left", "open_right"], "Invalid aggressive action"
        
        print("Conservative approach:")
        print(f"  ✓ Action: {conservative_action[0]}")
        print(f"  ✓ Metrics collected: {len(conservative_data.info_variables)}")
        
        print("Aggressive approach:")  
        print(f"  ✓ Action: {aggressive_action[0]}")
        print(f"  ✓ Metrics collected: {len(aggressive_data.info_variables)}")
        
        print("  ✓ SparsePFT parameter comparison usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ SparsePFT parameter comparison usage example failed: {e}")
        traceback.print_exc()
        return False

def test_pomcpow_comprehensive_usage_example():
    """Test the comprehensive POMCPOW usage example from the class docstring.
    
    Purpose: Validates pomcpow comprehensive usage example
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: example
    """
    print("Testing POMCPOW comprehensive usage example...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
        from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        import random
        
        # Create a simple action sampler (from docstring)
        class DiscreteActionSampler(ActionSampler):
            def __init__(self, actions):
                self.actions = actions
                
            def sample(self, belief_node=None):
                return random.choice(self.actions)
        
        # Initialize environment and belief (from docstring)
        environment = TigerPOMDP(discount_factor=0.95)
        action_sampler = DiscreteActionSampler(environment.get_actions())
        
        # Create POMCPOW planner (from docstring, reduced parameters for testing)
        planner = POMCPOW(
            environment=environment,
            discount_factor=0.95,
            depth=5,                    # Reduced for testing
            exploration_constant=1.0,
            k_o=3.0,                    # Observation progressive widening coefficient
            k_a=3.0,                    # Action progressive widening coefficient  
            alpha_o=0.5,                # Observation progressive widening exponent
            alpha_a=0.5,                # Action progressive widening exponent
            action_sampler=action_sampler,
            n_simulations=10,           # Reduced for testing
            name="POMCPOW_Planner"
        )
        
        # Get initial belief and plan action (from docstring)
        belief = get_initial_belief(
            pomdp=environment,
            n_particles=50,             # Reduced for testing
            resampling=True
        )
        
        action, run_data = planner.action(belief)
        
        # Verify results match documented example
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        assert len(run_data.info_variables) > 0, "No tree metrics collected"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree metrics: {[m.name for m in run_data.info_variables]}")
        print("  ✓ POMCPOW comprehensive usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ POMCPOW comprehensive usage example failed: {e}")
        traceback.print_exc()
        return False

def test_mcts_algorithms_integration():
    """Test that all MCTS algorithms work together and can be compared.
    
    Purpose: Validates mcts algorithms integration
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
    print("Testing MCTS algorithms integration...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
        from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.core.environment import SpaceType
        import numpy as np
        
        # Simple action sampler for PFT-DPW
        class DiscreteActionSampler(ActionSampler):
            def __init__(self, actions):
                self.actions = actions
            
            def sample(self, belief_node=None):
                return np.random.choice(self.actions)
        
        # Create common environment
        env = TigerPOMDP(discount_factor=0.95)
        initial_belief = get_initial_belief(env, n_particles=30)  # Reduced for testing
        
        # Create all MCTS planners (small parameters for testing)
        pomcp = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            name="POMCP_Integration",
            n_simulations=3
        )
        
        sparse_pft = SparsePFT(
            environment=env,
            discount_factor=0.95,
            gamma=0.95,
            depth=3,
            c_ucb=1.0,
            beta_ucb=1.0,
            belief_child_num=2,
            n_simulations=3,
            name="SparsePFT_Integration"
        )
        
        pft_dpw = PFT_DPW(
            environment=env,
            discount_factor=0.95,
            depth=3,
            name="PFT_DPW_Integration",
            action_sampler=DiscreteActionSampler(env.get_actions()),
            n_simulations=3
        )
        
        pomcpow = POMCPOW(
            environment=env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            k_o=2.0,
            k_a=2.0,
            alpha_o=0.5,
            alpha_a=0.5,
            action_sampler=DiscreteActionSampler(env.get_actions()),
            n_simulations=3,
            name="POMCPOW_Integration"
        )
        
        # Test all planners
        planners = [pomcp, sparse_pft, pft_dpw, pomcpow]
        results = []
        
        for planner in planners:
            action, run_data = planner.action(initial_belief)
            results.append((planner.name, action[0], len(run_data.info_variables)))
            
            # Verify space info consistency
            space_info = planner.get_space_info()
            if planner.name.startswith("PFT_DPW"):
                expected_action_space = SpaceType.CONTINUOUS
            elif planner.name.startswith("POMCPOW"):
                expected_action_space = SpaceType.MIXED
            else:
                expected_action_space = SpaceType.DISCRETE
            
            assert space_info.action_space == expected_action_space, f"{planner.name} reports incorrect action space"
        
        # Verify all planners returned valid actions
        for name, action, metrics_count in results:
            assert action in ["listen", "open_left", "open_right"], f"{name} returned invalid action: {action}"
            assert metrics_count > 0, f"{name} returned no metrics"
        
        print("Algorithm comparison results:")
        for name, action, metrics_count in results:
            print(f"  ✓ {name}: action={action}, metrics={metrics_count}")
        
        print("  ✓ All MCTS algorithms work together and are comparable")
        print("  ✓ MCTS algorithms integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ MCTS algorithms integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all MCTS planner usage examples."""
    print("Running ALL usage examples from MCTS planners directory classes...\n")
    
    tests = [
        test_pomcp_comprehensive_usage_example,
        test_pft_dpw_cartpole_usage_example,
        test_pft_dpw_navigation_usage_example,
        test_action_sampler_usage_example,
        test_path_simulation_policy_custom_mcts_example,
        test_sparse_pft_tiger_usage_example,
        test_sparse_pft_parameter_comparison_example,
        test_pomcpow_comprehensive_usage_example,
        test_mcts_algorithms_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} MCTS planner test groups passed")
    
    if passed == total:
        print("🎉 ALL MCTS planner usage examples work correctly!")
        return 0
    else:
        print("❌ Some MCTS planner usage examples have issues.")
        return 1

if __name__ == "__main__":
    exit(main())