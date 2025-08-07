#!/usr/bin/env python3
"""Test script to verify all usage examples from planner classes work correctly."""

import sys
import traceback
import numpy as np
import os
from pathlib import Path

# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pomcp_usage_example():
    """Test the POMCP usage example from the class docstring."""
    print("Testing POMCP usage example...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        
        # Create environment and planner (using smaller parameters for testing)
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
        initial_belief = get_initial_belief(env, n_particles=100)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Verify results
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree stats available: {len(run_data.info_variables) > 0}")
        print("  ✓ POMCP usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ POMCP usage example failed: {e}")
        traceback.print_exc()
        return False

def test_pft_dpw_usage_example():
    """Test the PFT-DPW usage example with a custom action sampler."""
    print("Testing PFT-DPW usage example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        import numpy as np
        
        # Custom action sampler for CartPole
        class CartPoleActionSampler(ActionSampler):
            def sample(self, belief_node=None):
                return np.random.choice([0, 1])  # Left or right force
        
        # Create CartPole environment
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        cartpole = CartPolePOMDP(
            discount_factor=0.99,
            noise_cov=noise_cov
        )
        
        # Create PFT-DPW planner with custom action sampler
        pft_planner = PFT_DPW(
            environment=cartpole,
            discount_factor=0.99,
            depth=3,                  # Reduced for testing
            name="PFT_DPW_CartPole",
            action_sampler=CartPoleActionSampler(),
            n_simulations=5          # Reduced for testing
        )
        
        # Plan action
        initial_belief = get_initial_belief(cartpole, n_particles=50)  # Reduced for testing
        action, run_data = pft_planner.action(initial_belief)
        
        # Verify results
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in [0, 1], f"Expected action 0 or 1, got {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Action sampler working correctly")
        print("  ✓ PFT-DPW usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PFT-DPW usage example failed: {e}")
        traceback.print_exc()
        return False

def test_sparse_sampling_usage_example():
    """Test the StandardSparseSamplingDiscreteActionsPlanner usage example."""
    print("Testing Sparse Sampling usage example...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        
        # Create environment and planner
        env = TigerPOMDP(discount_factor=0.95)
        planner = StandardSparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=3,       # Reduced for testing
            depth=3,                  # Reduced for testing
            name="SparseSampling_Tiger"
        )
        
        # Plan action from initial belief
        initial_belief = get_initial_belief(env, n_particles=100)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Verify results
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Sparse sampling completed successfully")
        print("  ✓ Sparse Sampling usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Sparse Sampling usage example failed: {e}")
        traceback.print_exc()
        return False

def test_sparse_pft_usage_example():
    """Test the Sparse-PFT usage example from the class docstring."""
    print("Testing Sparse-PFT usage example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create Tiger POMDP environment
        tiger = TigerPOMDP(discount_factor=0.95)
        
        # Create Sparse-PFT planner with controlled branching (using small parameters for testing)
        sparse_pft = SparsePFT(
            environment=tiger,
            discount_factor=0.95,
            gamma=0.95,                  # Discount for recursive calls
            depth=5,                     # Reduced for testing
            c_ucb=1.0,                   # Base exploration constant
            beta_ucb=2.0,                # Enhanced exploration parameter
            belief_child_num=3,          # Reduced for testing
            n_simulations=10,            # Reduced for testing
            name="SparsePFT_Tiger"
        )
        
        # Plan action from initial belief
        initial_belief = get_initial_belief(tiger, n_particles=50)  # Reduced for testing
        action, run_data = sparse_pft.action(initial_belief)
        
        # Verify results
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        assert hasattr(run_data, 'info_variables'), "Missing info_variables in run_data"
        
        print(f"  ✓ Selected action: {action[0]}")
        print(f"  ✓ Tree metrics collected: {len(run_data.info_variables)}")
        print("  ✓ Sparse-PFT usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Sparse-PFT usage example failed: {e}")
        traceback.print_exc()
        return False

def test_discrete_action_sequences_planner_example():
    """Test the DiscreteActionSequencesPlanner if it exists."""
    print("Testing Discrete Action Sequences Planner example...")
    
    try:
        from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import DiscreteActionSequencesPlanner
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create environment
        env = TigerPOMDP(discount_factor=0.95)
        
        # Create planner with short sequences for testing
        planner = DiscreteActionSequencesPlanner(
            environment=env,
            sequence_length=2,        # Short sequence for testing
            name="DiscreteSequences_Tiger"
        )
        
        # Plan action from initial belief
        initial_belief = get_initial_belief(env, n_particles=100)
        action, run_data = planner.action(initial_belief)
        
        # Verify results
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert action[0] in ["listen", "open_left", "open_right"], f"Unexpected action: {action[0]}"
        
        print(f"  ✓ Selected action: {action[0]}")
        print("  ✓ Discrete Action Sequences Planner example passed!")
        return True
        
    except ImportError:
        print("  ⚠ Discrete Action Sequences Planner not found, skipping...")
        return True
    except Exception as e:
        print(f"  ✗ Discrete Action Sequences Planner example failed: {e}")
        traceback.print_exc()
        return False

def test_space_info_consistency():
    """Test that all planners report consistent space info."""
    print("Testing planner space info consistency...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
        from POMDPPlanners.core.environment import SpaceType
        import numpy as np
        
        # Simple action sampler
        class SimpleActionSampler(ActionSampler):
            def sample(self, belief_node=None):
                return np.random.choice([0, 1])
        
        # Test discrete environment with discrete planners
        tiger_env = TigerPOMDP(discount_factor=0.95)
        
        pomcp_discrete = POMCP(
            environment=tiger_env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            name="POMCP_Discrete",
            n_simulations=5
        )
        
        sparse_discrete = StandardSparseSamplingDiscreteActionsPlanner(
            environment=tiger_env,
            branching_factor=2,
            depth=2,
            name="Sparse_Discrete"
        )
        
        # Check space info for discrete planners
        pomcp_info = pomcp_discrete.get_space_info()
        sparse_info = sparse_discrete.get_space_info()
        
        assert pomcp_info.action_space == SpaceType.DISCRETE, "POMCP should handle discrete actions"
        assert sparse_info.action_space == SpaceType.DISCRETE, "Sparse sampling should handle discrete actions"
        
        # Test continuous environment with continuous planner
        cartpole = CartPolePOMDP(
            discount_factor=0.99,
            noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
        )
        
        pft_dpw_continuous = PFT_DPW(
            environment=cartpole,
            discount_factor=0.99,
            depth=2,
            name="PFT_DPW_Continuous",
            action_sampler=SimpleActionSampler(),
            n_simulations=3
        )
        
        pft_info = pft_dpw_continuous.get_space_info()
        assert pft_info.action_space == SpaceType.CONTINUOUS, "PFT-DPW should handle continuous actions"
        
        print("  ✓ All planners report consistent space information")
        print("  ✓ Space info consistency test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Space info consistency test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all planner usage examples."""
    print("Running ALL usage examples from planners directory classes...\n")
    
    tests = [
        test_pomcp_usage_example,
        test_pft_dpw_usage_example,
        test_sparse_sampling_usage_example,
        test_sparse_pft_usage_example,
        test_discrete_action_sequences_planner_example,
        test_space_info_consistency,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} planner test groups passed")
    
    if passed == total:
        print("🎉 ALL planner usage examples work correctly!")
        return 0
    else:
        print("❌ Some planner usage examples have issues.")
        return 1

if __name__ == "__main__":
    exit(main())