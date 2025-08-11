#!/usr/bin/env python3
"""Test script to verify all usage examples from simulation classes work correctly."""

import sys
import traceback
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil

# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_run_episode_usage_example():
    """Test the run_episode usage example from the function docstring.
    
    Purpose: Validates that run_episode function works correctly with TigerPOMDP and POMCP policy configuration
    
    Given: TigerPOMDP environment, POMCP policy with reduced parameters, initial belief with 100 particles, and 5 episode steps
    When: run_episode is executed with these parameters
    Then: Returns History object with expected attributes (history, discount_factor, actual_num_steps, timing data, terminal state flag)
    
    Test type: example
    """
    print("Testing run_episode usage example...")
    
    try:
        from POMDPPlanners.simulations.episodes import run_episode
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.utils.logger import get_logger
        
        # Create environment and policy (using smaller parameters for testing)
        env = TigerPOMDP(discount_factor=0.95)
        policy = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=5,                  # Reduced for testing
            exploration_constant=1.0,
            name="POMCP_Tiger",
            n_simulations=10         # Reduced for testing
        )
        
        # Set up initial belief and logger
        initial_belief = get_initial_belief(env, n_particles=100)  # Reduced for testing
        logger = get_logger("episode_runner_test", debug=False)
        
        # Run episode
        history = run_episode(
            environment=env,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=5,             # Reduced for testing
            logger=logger
        )
        
        # Access and verify results
        total_reward = sum(step.reward for step in history.history if step.reward is not None)
        
        assert hasattr(history, 'history'), "History should have history attribute"
        assert hasattr(history, 'discount_factor'), "History should have discount_factor"
        assert hasattr(history, 'actual_num_steps'), "History should have actual_num_steps"
        assert hasattr(history, 'average_action_time'), "History should have average_action_time"
        assert hasattr(history, 'reach_terminal_state'), "History should have reach_terminal_state"
        
        assert isinstance(total_reward, (int, float)), f"Total reward should be numeric, got {type(total_reward)}"
        assert isinstance(history.actual_num_steps, int), f"Actual steps should be int, got {type(history.actual_num_steps)}"
        assert isinstance(history.average_action_time, float), f"Action time should be float, got {type(history.average_action_time)}"
        assert isinstance(history.reach_terminal_state, bool), f"Terminal state should be bool, got {type(history.reach_terminal_state)}"
        
        print(f"  ✓ Total reward: {total_reward:.3f}")
        print(f"  ✓ Episode length: {history.actual_num_steps}")
        print(f"  ✓ Average action time: {history.average_action_time:.6f}s")
        print(f"  ✓ Reached terminal: {history.reach_terminal_state}")
        print("  ✓ run_episode usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ run_episode usage example failed: {e}")
        traceback.print_exc()
        return False

def test_pomdp_simulator_usage_example():
    """Test the POMDPSimulator usage example (simplified version).
    
    Purpose: Validates that POMDPSimulator correctly compares multiple policies on TigerPOMDP environment
    
    Given: TigerPOMDP environment, POMCP and StandardSparseSampling policies, 3 episodes with 5 steps each, JOBLIB task manager
    When: POMDPSimulator.compare_multiple_environments_policies is called
    Then: Returns structured results dict with policy histories and statistics DataFrame with environment, policy, and average_return columns
    
    Test type: example
    """
    print("Testing POMDPSimulator usage example...")
    
    try:
        from POMDPPlanners.simulations.simulator import POMDPSimulator
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.core.simulation import EnvironmentRunParams
        from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir)
            
            # Create environment
            tiger_env = TigerPOMDP(discount_factor=0.95)
            initial_belief = get_initial_belief(tiger_env, n_particles=50)  # Reduced for testing
            
            # Create policies to compare (using small parameters for testing)
            pomcp = POMCP(
                environment=tiger_env,
                discount_factor=0.95,
                depth=3,                  # Reduced for testing
                exploration_constant=1.0,
                name="POMCP",
                n_simulations=5          # Reduced for testing
            )
            
            sparse_sampling = StandardSparseSamplingDiscreteActionsPlanner(
                environment=tiger_env,
                branching_factor=2,       # Reduced for testing
                depth=2,                  # Reduced for testing
                name="SparseSampling"
            )
            
            # Configure simulation parameters (very small for testing)
            env_params = [
                EnvironmentRunParams(
                    environment=tiger_env,
                    belief=initial_belief,
                    policies=[pomcp, sparse_sampling],
                    num_episodes=3,          # Reduced for testing
                    num_steps=5              # Reduced for testing
                )
            ]
            
            # Run simulation (without profiling for testing)
            with POMDPSimulator(
                cache_dir_path=cache_path,
                experiment_name="Tiger_POMDP_Test",
                task_manager_type=TaskManagerType.JOBLIB,
                n_jobs=1,                    # Single job for testing
                enable_profiling=False       # Disabled for testing
            ) as simulator:
                results, statistics_df = simulator.compare_multiple_environments_policies(
                    environment_run_params=env_params,
                    alpha=0.05,
                    confidence_interval_level=0.95,
                    n_jobs=1                 # Single job for testing
                )
            
            # Verify results structure
            assert isinstance(results, dict), f"Results should be dict, got {type(results)}"
            assert 'TigerPOMDP' in results, f"Expected TigerPOMDP in results: {results.keys()}"
            assert 'POMCP' in results['TigerPOMDP'], f"Expected POMCP in results: {results['TigerPOMDP'].keys()}"
            assert 'SparseSampling' in results['TigerPOMDP'], f"Expected SparseSampling in results: {results['TigerPOMDP'].keys()}"
            
            # Verify statistics DataFrame
            assert hasattr(statistics_df, 'shape'), f"Statistics should be DataFrame, got {type(statistics_df)}"
            assert statistics_df.shape[0] == 2, f"Expected 2 rows (2 policies), got {statistics_df.shape[0]}"
            assert 'environment' in statistics_df.columns, "Missing 'environment' column"
            assert 'policy' in statistics_df.columns, "Missing 'policy' column"
            assert 'average_return' in statistics_df.columns, "Missing 'average_return' column"
            
            # Access policy-specific results
            tiger_results = results['TigerPOMDP']
            pomcp_histories = tiger_results['POMCP']
            sparse_histories = tiger_results['SparseSampling']
            
            assert len(pomcp_histories) == 3, f"Expected 3 POMCP histories, got {len(pomcp_histories)}"
            assert len(sparse_histories) == 3, f"Expected 3 Sparse histories, got {len(sparse_histories)}"
            
            # Compare average returns
            pomcp_stats = statistics_df[statistics_df['policy'] == 'POMCP']
            sparse_stats = statistics_df[statistics_df['policy'] == 'SparseSampling']
            
            assert len(pomcp_stats) == 1, f"Expected 1 POMCP stats row, got {len(pomcp_stats)}"
            assert len(sparse_stats) == 1, f"Expected 1 Sparse stats row, got {len(sparse_stats)}"
            
            pomcp_return = pomcp_stats['average_return'].iloc[0]
            sparse_return = sparse_stats['average_return'].iloc[0]
            
            print(f"  ✓ Simulation completed successfully")
            print(f"  ✓ Results structure: {list(results.keys())}")
            print(f"  ✓ Statistics shape: {statistics_df.shape}")
            print(f"  ✓ POMCP average return: {pomcp_return:.3f}")
            print(f"  ✓ Sparse Sampling average return: {sparse_return:.3f}")
            print("  ✓ POMDPSimulator usage example passed!")
            return True
        
    except Exception as e:
        print(f"  ✗ POMDPSimulator usage example failed: {e}")
        traceback.print_exc()
        return False

def test_simulations_api_usage_example():
    """Test the SimulationsAPI usage example (simplified version).
    
    Purpose: Validates that SimulationsAPI correctly executes multi-policy comparison with confidence intervals
    
    Given: SimulationsAPI instance, TigerPOMDP with POMCP and SparseSampling policies, 3 episodes with 5 steps, alpha=0.05 and 95% confidence
    When: run_multiple_environments_and_policies_local_run is executed
    Then: Returns tuple of (results_dict, stats_df) with policy-specific histories and confidence interval statistics
    
    Test type: example
    """
    print("Testing SimulationsAPI usage example...")
    
    try:
        from POMDPPlanners.simulations.simulations_api import SimulationsAPI
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.core.simulation import EnvironmentRunParams
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir)
            
            # Initialize the API
            api = SimulationsAPI(
                cache_dir_path=cache_path,
                debug=False  # Disabled for testing
            )
            
            # Create environment (simplified)
            tiger = TigerPOMDP(discount_factor=0.95)
            
            # Create policies for the environment (small parameters)
            tiger_policies = [
                POMCP(
                    environment=tiger,
                    discount_factor=0.95,
                    depth=3,                  # Reduced for testing
                    exploration_constant=1.0,
                    name="POMCP_Tiger",
                    n_simulations=5          # Reduced for testing
                ),
                StandardSparseSamplingDiscreteActionsPlanner(
                    environment=tiger,
                    branching_factor=2,       # Reduced for testing
                    depth=2,                  # Reduced for testing
                    name="SparseSampling_Tiger"
                )
            ]
            
            # Configure simulation parameters (very small for testing)
            environment_run_params = [
                EnvironmentRunParams(
                    environment=tiger,
                    belief=get_initial_belief(tiger, n_particles=50),  # Reduced for testing
                    policies=tiger_policies,
                    num_episodes=3,           # Reduced for testing
                    num_steps=5               # Reduced for testing
                )
            ]
            
            # Run simplified simulation (no debug run for testing)
            results, statistics_df = api.run_multiple_environments_and_policies_local_run(
                environment_run_params=environment_run_params,
                alpha=0.05,
                confidence_interval_level=0.95,
                experiment_name="Test_Comparison",
                n_jobs=1,                    # Single job for testing
                enable_profiling=False       # Disabled for testing
            )
            
            # Verify results
            assert isinstance(results, tuple), f"Expected tuple result, got {type(results)}"
            results_dict, stats_df = results
            
            assert isinstance(results_dict, dict), f"Results should be dict, got {type(results_dict)}"
            assert isinstance(stats_df, type(statistics_df)), f"Stats should be DataFrame, got {type(stats_df)}"
            
            # Analyze results
            environments = stats_df['environment'].unique()
            policies = stats_df['policy'].unique()
            
            print(f"  ✓ Simulation Results Summary:")
            print(f"    Environments tested: {list(environments)}")
            print(f"    Policies tested: {list(policies)}")
            print(f"    Total configurations: {len(stats_df)}")
            
            # Compare policies within each environment
            for env_name in environments:
                env_stats = stats_df[stats_df['environment'] == env_name]
                print(f"  ✓ {env_name} Results:")
                for policy_name in env_stats['policy'].unique():
                    policy_stats = env_stats[env_stats['policy'] == policy_name]
                    avg_return = policy_stats['average_return'].iloc[0]
                    ci_lower = policy_stats['average_return_ci_lower'].iloc[0]
                    ci_upper = policy_stats['average_return_ci_upper'].iloc[0]
                    print(f"      {policy_name}: {avg_return:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            print("  ✓ SimulationsAPI usage example passed!")
            return True
        
    except Exception as e:
        print(f"  ✗ SimulationsAPI usage example failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_statistics_usage_example():
    """Test the compute_statistics_environment_policy_pair usage example.
    
    Purpose: Validates that compute_statistics_environment_policy_pair generates comprehensive metrics with confidence intervals
    
    Given: TigerPOMDP environment, POMCP policy, 3 episode histories with 5 steps each, alpha=0.05 and 95% confidence level
    When: compute_statistics_environment_policy_pair processes the histories
    Then: Returns list of MetricValue objects including average_return, return_cvar, and average_action_time with confidence bounds
    
    Test type: example
    """
    print("Testing simulation statistics usage example...")
    
    try:
        from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.simulations.episodes import run_episode
        from POMDPPlanners.utils.logger import get_logger
        
        # Set up environment and policy (small parameters)
        env = TigerPOMDP(discount_factor=0.95)
        policy = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=3,                  # Reduced for testing
            exploration_constant=1.0,
            name="POMCP",
            n_simulations=5          # Reduced for testing
        )
        
        # Run multiple episodes (small number for testing)
        histories = []
        initial_belief = get_initial_belief(env, n_particles=50)  # Reduced for testing
        logger = get_logger("stats_example_test", debug=False)
        
        for episode in range(3):      # Reduced for testing
            history = run_episode(
                environment=env,
                policy=policy,
                initial_belief=initial_belief,
                num_steps=5,          # Reduced for testing
                logger=logger
            )
            histories.append(history)
        
        # Compute comprehensive statistics
        metrics = compute_statistics_environment_policy_pair(
            env=env,
            histories=histories,
            alpha=0.05,               # 5% risk level
            confidence_interval_level=0.95  # 95% confidence intervals
        )
        
        # Verify results
        assert isinstance(metrics, list), f"Metrics should be list, got {type(metrics)}"
        assert len(metrics) > 0, "Should have some metrics"
        
        # Check that we have expected metrics
        metric_names = [m.name for m in metrics]
        expected_metrics = ["average_return", "return_cvar", "average_action_time"]
        
        for expected in expected_metrics:
            assert expected in metric_names, f"Missing expected metric: {expected}"
        
        # Find specific metrics
        avg_return = next(m for m in metrics if m.name == "average_return")
        cvar = next(m for m in metrics if m.name == "return_cvar")
        action_time = next(m for m in metrics if m.name == "average_action_time")
        
        # Verify metric structure
        assert hasattr(avg_return, 'value'), "Metric should have value"
        assert hasattr(avg_return, 'lower_confidence_bound'), "Metric should have lower bound"
        assert hasattr(avg_return, 'upper_confidence_bound'), "Metric should have upper bound"
        
        # Analyze results
        print(f"  ✓ Computed {len(metrics)} metrics")
        for metric in metrics[:5]:  # Show first 5 metrics
            print(f"    {metric.name}: {metric.value:.3f} "
                  f"[{metric.lower_confidence_bound:.3f}, {metric.upper_confidence_bound:.3f}]")
        
        print(f"  ✓ Key Performance Indicators:")
        ci_width = (avg_return.upper_confidence_bound - avg_return.lower_confidence_bound) / 2
        print(f"    Average Return: {avg_return.value:.3f} ± {ci_width:.3f}")
        print(f"    CVaR (5%): {cvar.value:.3f}")
        print(f"    Planning Time: {action_time.value:.6f} seconds/step")
        print("  ✓ Simulation statistics usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation statistics usage example failed: {e}")
        traceback.print_exc()
        return False

def test_metrics_dict_to_dataframe_usage_example():
    """Test the metrics_dict_to_dataframe usage example.
    
    Purpose: Validates that metrics_dict_to_dataframe correctly converts nested policy metrics into comparative DataFrame format
    
    Given: TigerPOMDP with POMCP and SparseSampling policies, 2 episodes each with 3 steps, computed metrics for both policies
    When: metrics_dict_to_dataframe processes the nested metrics dictionary
    Then: Returns DataFrame with environment, policy columns and confidence interval columns for policy comparison analysis
    
    Test type: example
    """
    print("Testing metrics_dict_to_dataframe usage example...")
    
    try:
        from POMDPPlanners.simulations.simulation_statistics import (
            compute_statistics_environment_policy_pair, metrics_dict_to_dataframe
        )
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.simulations.episodes import run_episode
        from POMDPPlanners.utils.logger import get_logger
        
        # Create environment and policies (small parameters)
        env = TigerPOMDP(discount_factor=0.95)
        
        pomcp = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=2,                  # Reduced for testing
            exploration_constant=1.0,
            name="POMCP",
            n_simulations=3          # Reduced for testing
        )
        
        sparse = StandardSparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=2,       # Reduced for testing
            depth=2,                  # Reduced for testing
            name="SparseSampling"
        )
        
        # Run episodes for both policies (very small for testing)
        initial_belief = get_initial_belief(env, n_particles=30)
        logger = get_logger("dataframe_test", debug=False)
        
        # Generate histories for both policies
        pomcp_histories = []
        sparse_histories = []
        
        for episode in range(2):      # Very small for testing
            # POMCP episode
            pomcp_history = run_episode(
                environment=env,
                policy=pomcp,
                initial_belief=initial_belief,
                num_steps=3,          # Reduced for testing
                logger=logger
            )
            pomcp_histories.append(pomcp_history)
            
            # Sparse sampling episode
            sparse_history = run_episode(
                environment=env,
                policy=sparse,
                initial_belief=initial_belief,
                num_steps=3,          # Reduced for testing
                logger=logger
            )
            sparse_histories.append(sparse_history)
        
        # Compute metrics for both policies
        pomcp_metrics = compute_statistics_environment_policy_pair(
            env=env,
            histories=pomcp_histories,
            alpha=0.05,
            confidence_interval_level=0.95
        )
        
        sparse_metrics = compute_statistics_environment_policy_pair(
            env=env,
            histories=sparse_histories,
            alpha=0.05,
            confidence_interval_level=0.95
        )
        
        # Create metrics dictionary
        metrics_dict = {
            'TigerPOMDP': {
                'POMCP': pomcp_metrics,
                'SparseSampling': sparse_metrics
            }
        }
        
        # Convert to DataFrame
        results_df = metrics_dict_to_dataframe(metrics_dict)
        
        # Verify DataFrame structure
        assert hasattr(results_df, 'shape'), f"Should be DataFrame, got {type(results_df)}"
        assert results_df.shape[0] == 2, f"Expected 2 rows, got {results_df.shape[0]}"
        assert 'environment' in results_df.columns, "Missing 'environment' column"
        assert 'policy' in results_df.columns, "Missing 'policy' column"
        assert 'average_return' in results_df.columns, "Missing 'average_return' column"
        assert 'average_return_ci_lower' in results_df.columns, "Missing CI lower bound column"
        assert 'average_return_ci_upper' in results_df.columns, "Missing CI upper bound column"
        
        # Analyze results
        environments = results_df['environment'].unique()
        policies = results_df['policy'].unique()
        
        print(f"  ✓ DataFrame shape: {results_df.shape}")
        print(f"  ✓ Environments: {list(environments)}")
        print(f"  ✓ Policies: {list(policies)}")
        
        # Compare average returns across policies
        avg_return_comparison = results_df[['environment', 'policy', 'average_return', 
                                          'average_return_ci_lower', 'average_return_ci_upper']]
        print(f"  ✓ Average Return Comparison:")
        for _, row in avg_return_comparison.iterrows():
            print(f"    {row['policy']}: {row['average_return']:.3f} "
                  f"[{row['average_return_ci_lower']:.3f}, {row['average_return_ci_upper']:.3f}]")
        
        # Find best performing policy
        best_idx = results_df['average_return'].idxmax()
        best_policy = results_df.loc[best_idx, 'policy']
        best_return = results_df.loc[best_idx, 'average_return']
        
        print(f"  ✓ Best policy: {best_policy} with return {best_return:.3f}")
        
        # Check confidence interval overlap (statistical significance test)
        if len(results_df) >= 2:
            policy1_stats = results_df.iloc[0]
            policy2_stats = results_df.iloc[1]
            
            p1_ci = (policy1_stats['average_return_ci_lower'], policy1_stats['average_return_ci_upper'])
            p2_ci = (policy2_stats['average_return_ci_lower'], policy2_stats['average_return_ci_upper'])
            
            overlap = not (p1_ci[1] < p2_ci[0] or p2_ci[1] < p1_ci[0])
            print(f"  ✓ Confidence intervals overlap: {overlap}")
        
        print("  ✓ metrics_dict_to_dataframe usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ metrics_dict_to_dataframe usage example failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all simulation usage examples."""
    print("Running ALL usage examples from simulations directory classes...\n")
    
    tests = [
        test_run_episode_usage_example,
        test_pomdp_simulator_usage_example,
        test_simulations_api_usage_example,
        test_simulation_statistics_usage_example,
        test_metrics_dict_to_dataframe_usage_example,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} simulation test groups passed")
    
    if passed == total:
        print("🎉 ALL simulation usage examples work correctly!")
        return 0
    else:
        print("❌ Some simulation usage examples have issues.")
        return 1

if __name__ == "__main__":
    exit(main())