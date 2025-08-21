#!/usr/bin/env python3
"""Test script to verify all usage examples from utils directory classes work correctly."""

import sys
import traceback
import numpy as np
import os
from pathlib import Path
import tempfile
import logging
import random

np.random.seed(42)
random.seed(42)


# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unit_circle_action_sampler_basic_usage():
    """Test the basic usage example from UnitCircleActionSampler docstring.
    
    Purpose: Validates that UnitCircleActionSampler generates 2D unit circle actions within magnitude limits for continuous navigation
    
    Given: ContinuousLightDarkPOMDP environment, UnitCircleActionSampler with max_action_magnitude=2.0, PFT_DPW planner integration
    When: ActionSampler generates multiple 2D actions and integrates with PFT_DPW for planning
    Then: All actions are 2D numpy arrays with magnitude ≤2.0, planner integration successful with valid action output
    
    Test type: example
    """
    print("Testing UnitCircleActionSampler basic usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
        from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
            ContinuousLightDarkPOMDP
        )
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create 2D navigation environment
        nav_env = ContinuousLightDarkPOMDP(
            discount_factor=0.99,
            goal_state=np.array([10, 0]),
            start_state=np.array([0, 0])
        )
        
        # Create unit circle action sampler
        action_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)
        
        # Test action sampling
        actions = [action_sampler.sample() for _ in range(5)]
        for i, action in enumerate(actions):
            magnitude = np.linalg.norm(action)
            assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
            assert len(action) == 2, f"Expected 2D action, got {len(action)}"
            assert magnitude <= 2.0, f"Magnitude {magnitude} exceeds max_action_magnitude 2.0"
        
        # Use with PFT-DPW planner (reduced parameters for testing)
        planner = PFT_DPW(
            environment=nav_env,
            discount_factor=0.99,
            depth=3,                    # Reduced for testing
            name="PFT_DPW_Navigation",
            action_sampler=action_sampler,
            n_simulations=5             # Reduced for testing
        )
        
        initial_belief = get_initial_belief(nav_env, n_particles=25)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Verify planner integration
        assert len(action) == 1, f"Expected single action, got {len(action)}"
        assert isinstance(action[0], np.ndarray), f"Expected ndarray action, got {type(action[0])}"
        assert len(action[0]) == 2, f"Expected 2D action, got {len(action[0])}"
        
        print(f"  ✓ Action sampler generates 2D actions with max magnitude 2.0")
        print(f"  ✓ Planner integration successful")
        print("  ✓ UnitCircleActionSampler basic usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ UnitCircleActionSampler basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_unit_circle_action_sampler_magnitude_comparison():
    """Test the magnitude comparison example from UnitCircleActionSampler docstring.
    
    Purpose: Validates that UnitCircleActionSampler supports different movement strategies through configurable magnitude limits
    
    Given: Conservative sampler (max_magnitude=0.5) vs aggressive sampler (max_magnitude=2.0), 20 samples each
    When: Both samplers generate action distributions with different magnitude constraints
    Then: Conservative actions ≤0.5, aggressive actions ≤2.0, aggressive max > conservative max, demonstrating different movement strategies
    
    Test type: unit
    """
    print("Testing UnitCircleActionSampler magnitude comparison example...")
    
    try:
        from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
        import numpy as np
        
        # Conservative movement sampler
        conservative_sampler = UnitCircleActionSampler(max_action_magnitude=0.5)
        
        # Aggressive movement sampler  
        aggressive_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)
        
        # Compare action distributions
        conservative_actions = [conservative_sampler.sample() for _ in range(20)]  # Reduced sample size
        aggressive_actions = [aggressive_sampler.sample() for _ in range(20)]     # Reduced sample size
        
        conservative_magnitudes = [np.linalg.norm(a) for a in conservative_actions]
        aggressive_magnitudes = [np.linalg.norm(a) for a in aggressive_actions]
        
        # Verify magnitude constraints
        assert all(mag <= 0.5 for mag in conservative_magnitudes), "Conservative sampler exceeded magnitude limit"
        assert all(mag <= 2.0 for mag in aggressive_magnitudes), "Aggressive sampler exceeded magnitude limit"
        
        # Verify that aggressive sampler can generate larger magnitudes
        max_conservative = max(conservative_magnitudes)
        max_aggressive = max(aggressive_magnitudes)
        assert max_aggressive > max_conservative, f"Aggressive max {max_aggressive} not > conservative max {max_conservative}"
        
        avg_conservative = np.mean(conservative_magnitudes)
        avg_aggressive = np.mean(aggressive_magnitudes)
        
        print(f"  ✓ Conservative: max={max_conservative:.3f}, avg={avg_conservative:.3f}")
        print(f"  ✓ Aggressive: max={max_aggressive:.3f}, avg={avg_aggressive:.3f}")
        print("  ✓ UnitCircleActionSampler magnitude comparison example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ UnitCircleActionSampler magnitude comparison example failed: {e}")
        traceback.print_exc()
        return False

def test_config_loader_basic_usage():
    """Test the basic usage example from load_config docstring.
    
    Purpose: Validates that load_config correctly parses YAML configuration files with environment, planners, and simulation sections
    
    Given: Temporary YAML config with TigerPOMDP environment, two planners (POMCP, PFT_DPW), and simulation parameters
    When: load_config parses the YAML file structure
    Then: Returns dict with correct sections, environment name=TigerPOMDP, 2 planners, episodes_per_run=100
    
    Test type: example
    """
    print("Testing config_loader basic usage example...")
    
    try:
        from POMDPPlanners.utils.config_loader import load_config
        import tempfile
        import yaml
        import os
        
        # Create temporary config file for testing
        config_data = {
            'environment': {
                'name': 'TigerPOMDP',
                'discount_factor': 0.95,
                'observation_accuracy': 0.85
            },
            'planners': [
                {
                    'name': 'POMCP',
                    'n_simulations': 1000,
                    'depth': 20,
                    'exploration_constant': 1.0
                },
                {
                    'name': 'PFT_DPW',
                    'n_simulations': 500,
                    'depth': 15,
                    'k_a': 2.0
                }
            ],
            'simulation': {
                'episodes_per_run': 100,
                'num_runs': 10
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Test loading configuration
            config = load_config(temp_config_path)
            
            # Verify structure matches expected format
            assert 'environment' in config, "Missing environment section"
            assert 'planners' in config, "Missing planners section"
            assert 'simulation' in config, "Missing simulation section"
            
            # Access configuration sections (from docstring example)
            env_config = config['environment']
            planner_configs = config['planners']
            simulation_config = config['simulation']
            
            assert env_config['name'] == 'TigerPOMDP', f"Expected TigerPOMDP, got {env_config['name']}"
            assert len(planner_configs) == 2, f"Expected 2 planners, got {len(planner_configs)}"
            assert simulation_config['episodes_per_run'] == 100, f"Expected 100, got {simulation_config['episodes_per_run']}"
            
            print(f"  ✓ Environment: {env_config['name']}")
            print(f"  ✓ Planners: {[p['name'] for p in planner_configs]}")
            print(f"  ✓ Episodes per run: {simulation_config['episodes_per_run']}")
            print("  ✓ config_loader basic usage example passed!")
            return True
            
        finally:
            os.unlink(temp_config_path)
        
    except Exception as e:
        print(f"  ✗ config_loader basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_config_loader_environment_integration():
    """Test environment integration example from load_config docstring.
    
    Purpose: Validates that load_config enables seamless environment and planner creation from configuration files
    
    Given: YAML config with TigerPOMDP environment (discount=0.95) and POMCP planner settings
    When: Config is loaded and used to create actual TigerPOMDP and POMCP instances
    Then: Environment created with correct discount factor, planner created with correct name, integration successful
    
    Test type: integration
    """
    print("Testing config_loader environment integration example...")
    
    try:
        from POMDPPlanners.utils.config_loader import load_config
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        import tempfile
        import yaml
        import os
        
        # Create config for environment integration test
        config_data = {
            'environment': {
                'name': 'TigerPOMDP',
                'discount_factor': 0.95
            },
            'planners': [
                {
                    'name': 'POMCP',
                    'n_simulations': 10,   # Reduced for testing
                    'depth': 5,            # Reduced for testing
                    'exploration_constant': 1.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            config = load_config(temp_config_path)
            
            # Create environment from config
            env = TigerPOMDP(discount_factor=config['environment']['discount_factor'])
            
            # Create planners from config
            planners = []
            for planner_config in config['planners']:
                if planner_config['name'] == 'POMCP':
                    planner = POMCP(
                        environment=env,
                        discount_factor=env.discount_factor,
                        n_simulations=planner_config['n_simulations'],
                        depth=planner_config['depth'],
                        exploration_constant=planner_config['exploration_constant'],
                        name=planner_config['name']
                    )
                    planners.append(planner)
            
            # Verify integration works
            assert len(planners) == 1, f"Expected 1 planner, got {len(planners)}"
            assert planners[0].name == 'POMCP', f"Expected POMCP, got {planners[0].name}"
            assert env.discount_factor == 0.95, f"Expected 0.95, got {env.discount_factor}"
            
            print(f"  ✓ Environment created: TigerPOMDP with discount {env.discount_factor}")
            print(f"  ✓ Planner created: {planners[0].name}")
            print("  ✓ config_loader environment integration example passed!")
            return True
            
        finally:
            os.unlink(temp_config_path)
        
    except Exception as e:
        print(f"  ✗ config_loader environment integration example failed: {e}")
        traceback.print_exc()
        return False

def test_logger_basic_usage():
    """Test the basic logger usage example from get_logger docstring.
    
    Purpose: Validates that get_logger creates functional loggers with correct names, levels, and handler setup
    
    Given: Logger name="POMCP_Tiger", level=logging.INFO
    When: get_logger creates logger instance with specified parameters
    Then: Logger has correct name, INFO level or lower, at least one handler, callable methods (info/warning/error)
    
    Test type: example
    """
    print("Testing logger basic usage example...")
    
    try:
        from POMDPPlanners.utils.logger import get_logger
        import logging
        
        # Create logger for POMCP algorithm
        logger = get_logger("POMCP_Tiger", level=logging.INFO)
        
        # Test logger functionality without actually logging to avoid output clutter
        assert logger.name == "POMCP_Tiger", f"Expected POMCP_Tiger, got {logger.name}"
        assert logger.level <= logging.INFO, f"Expected INFO level or lower, got {logger.level}"
        
        # Verify logger has handlers
        assert len(logger.handlers) > 0, "Logger should have at least one handler"
        
        # Test that logger methods exist and are callable
        assert hasattr(logger, 'info'), "Logger missing info method"
        assert hasattr(logger, 'warning'), "Logger missing warning method"
        assert hasattr(logger, 'error'), "Logger missing error method"
        assert callable(logger.info), "Logger info method not callable"
        
        print(f"  ✓ Logger created with name: {logger.name}")
        print(f"  ✓ Logger level: {logger.level}")
        print(f"  ✓ Logger handlers: {len(logger.handlers)}")
        print("  ✓ logger basic usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ logger basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_logger_file_logging():
    """Test file logging example from get_logger docstring.
    
    Purpose: Validates that get_logger supports file-based logging for experiment tracking with proper directory structure
    
    Given: Logger name="TigerExperiment", output_dir=temporary experiment directory, console_output=False
    When: Logger setup creates logs directory and writes experiment tracking messages
    Then: Logs directory created, log file created with non-zero size, experiment messages successfully written
    
    Test type: unit
    """
    print("Testing logger file logging example...")
    
    try:
        from POMDPPlanners.utils.logger import get_logger
        import tempfile
        import logging
        from pathlib import Path
        
        # Create temporary directory for log testing
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "tiger_study_test"
            
            # Setup file logging for experiment tracking
            logger = get_logger(
                name="TigerExperiment",
                level=logging.INFO,
                output_dir=experiment_dir,
                console_output=False  # Disable console to avoid test output clutter
            )
            
            # Test that log directory was created
            logs_dir = experiment_dir / "logs"
            assert logs_dir.exists(), f"Logs directory not created at {logs_dir}"
            
            # Test logging functionality
            logger.info("Starting Tiger POMDP comparative study")
            logger.info("Environments: TigerPOMDP")
            logger.info("Planners: POMCP, PFT-DPW, SparsePFT")
            
            # Check that log file was created
            log_files = list(logs_dir.glob("TigerExperiment_*.log"))
            assert len(log_files) >= 1, f"Expected at least 1 log file, found {len(log_files)}"
            
            # Verify log file contains content
            log_file = log_files[0]
            assert log_file.stat().st_size > 0, "Log file should not be empty"
            
            print(f"  ✓ Log directory created: {logs_dir}")
            print(f"  ✓ Log file created: {log_file.name}")
            print(f"  ✓ Log file size: {log_file.stat().st_size} bytes")
            print("  ✓ logger file logging example passed!")
            return True
        
    except Exception as e:
        print(f"  ✗ logger file logging example failed: {e}")
        traceback.print_exc()
        return False

def test_cvar_estimator_basic_usage():
    """Test the basic CVaR usage example from cvar_estimator docstring.
    
    Purpose: Validates that cvar_estimator calculates risk metrics for algorithm performance analysis using tail risk measures
    
    Given: Algorithm returns array [12.5, 8.3, 15.7, -2.1, 9.8, 13.2, 6.4, 11.0, -1.5, 14.3], alpha values 0.9 and 0.95
    When: cvar_estimator computes mean return, CVaR(90%), and CVaR(95%) for risk analysis
    Then: All values are numeric, CVaR ≥ corresponding quantile, risk metrics calculated correctly for worst-case scenarios
    
    Test type: example
    """
    print("Testing cvar_estimator basic usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.utils.statistics import cvar_estimator
        
        # Simulate algorithm returns from multiple episodes  
        returns = np.array([12.5, 8.3, 15.7, -2.1, 9.8, 13.2, 6.4, 11.0, -1.5, 14.3])
        
        # Calculate risk metrics
        mean_return = np.mean(returns)
        cvar_90 = cvar_estimator(returns, alpha=0.9)  # Worst 10% outcomes
        cvar_95 = cvar_estimator(returns, alpha=0.95) # Worst 5% outcomes
        
        # Verify results are reasonable
        assert isinstance(mean_return, (float, np.floating)), f"Expected float, got {type(mean_return)}"
        assert isinstance(cvar_90, (float, np.floating)), f"Expected float, got {type(cvar_90)}"
        assert isinstance(cvar_95, (float, np.floating)), f"Expected float, got {type(cvar_95)}"
        
        # CVaR should be at least as large as the corresponding quantile
        sorted_returns = np.sort(returns)
        assert cvar_90 >= sorted_returns[int(0.9 * len(returns))], "CVaR should be >= quantile"
        assert cvar_95 >= sorted_returns[int(0.95 * len(returns))], "CVaR should be >= quantile"
        
        print(f"  ✓ Mean return: {mean_return:.2f}")
        print(f"  ✓ CVaR (90%): {cvar_90:.2f}")
        print(f"  ✓ CVaR (95%): {cvar_95:.2f}")
        print("  ✓ cvar_estimator basic usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ cvar_estimator basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_cvar_estimator_algorithm_comparison():
    """Test algorithm comparison example from cvar_estimator docstring.
    
    Purpose: Validates that cvar_estimator enables risk-adjusted performance comparison between POMDP planning algorithms
    
    Given: POMCP returns and PFT_DPW returns from experimental data, alpha=0.9 for tail risk analysis
    When: cvar_estimator computes mean and CVaR(90%) for both algorithms to compare risk-adjusted performance
    Then: All metrics are numeric, enabling comparative analysis of algorithm performance considering worst-case scenarios
    
    Test type: unit
    """
    print("Testing cvar_estimator algorithm comparison example...")
    
    try:
        from POMDPPlanners.utils.statistics import cvar_estimator
        import numpy as np
        
        # Algorithm performance data from experiments
        pomcp_returns = np.array([10.2, 12.8, 9.5, 11.3, 8.7, 12.1, 10.9, 9.8, 11.5, 10.4])
        pft_returns = np.array([15.1, 7.2, 14.8, 13.3, 6.9, 15.5, 8.1, 14.2, 12.7, 9.3])
        
        # Compare risk-adjusted performance
        pomcp_mean = np.mean(pomcp_returns)
        pft_mean = np.mean(pft_returns) 
        
        pomcp_cvar = cvar_estimator(pomcp_returns, alpha=0.9)
        pft_cvar = cvar_estimator(pft_returns, alpha=0.9)
        
        # Verify calculations are reasonable
        assert isinstance(pomcp_mean, (float, np.floating)), "POMCP mean should be numeric"
        assert isinstance(pft_mean, (float, np.floating)), "PFT mean should be numeric"
        assert isinstance(pomcp_cvar, (float, np.floating)), "POMCP CVaR should be numeric"
        assert isinstance(pft_cvar, (float, np.floating)), "PFT CVaR should be numeric"
        
        print("  ✓ Algorithm Risk Comparison:")
        print(f"    POMCP - Mean: {pomcp_mean:.2f}, CVaR(90%): {pomcp_cvar:.2f}")
        print(f"    PFT-DPW - Mean: {pft_mean:.2f}, CVaR(90%): {pft_cvar:.2f}")
        print("  ✓ cvar_estimator algorithm comparison example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ cvar_estimator algorithm comparison example failed: {e}")
        traceback.print_exc()
        return False

def test_confidence_interval_basic_usage():
    """Test the basic confidence interval usage example.
    
    Purpose: Validates that confidence_interval calculates statistical confidence bounds for algorithm performance comparison
    
    Given: POMCP and PFT_DPW reward data from multiple runs, confidence=0.95 for statistical analysis
    When: confidence_interval computes 95% confidence bounds for both algorithm reward distributions
    Then: CI tuples with lower≤upper bounds, CI contains mean, enables statistical significance testing between algorithms
    
    Test type: example
    """
    print("Testing confidence_interval basic usage example...")
    
    try:
        import numpy as np
        from POMDPPlanners.utils.statistics import confidence_interval
        
        # Algorithm performance from multiple runs
        pomcp_rewards = [12.3, 11.8, 13.1, 12.7, 11.9, 12.5, 13.0, 12.1, 12.8, 12.4]
        pft_rewards = [11.5, 13.2, 12.8, 11.9, 12.3, 13.5, 12.1, 12.9, 11.7, 12.6]
        
        # Calculate 95% confidence intervals
        pomcp_ci = confidence_interval(pomcp_rewards, confidence=0.95)
        pft_ci = confidence_interval(pft_rewards, confidence=0.95)
        
        # Verify results
        assert len(pomcp_ci) == 2, f"Expected tuple of length 2, got {len(pomcp_ci)}"
        assert len(pft_ci) == 2, f"Expected tuple of length 2, got {len(pft_ci)}"
        assert pomcp_ci[0] <= pomcp_ci[1], "Lower bound should be <= upper bound"
        assert pft_ci[0] <= pft_ci[1], "Lower bound should be <= upper bound"
        
        # Verify confidence intervals contain the mean
        pomcp_mean = np.mean(pomcp_rewards)
        pft_mean = np.mean(pft_rewards)
        assert pomcp_ci[0] <= pomcp_mean <= pomcp_ci[1], "CI should contain mean"
        assert pft_ci[0] <= pft_mean <= pft_ci[1], "CI should contain mean"
        
        print(f"  ✓ POMCP mean: {pomcp_mean:.2f}")
        print(f"  ✓ POMCP 95% CI: [{pomcp_ci[0]:.2f}, {pomcp_ci[1]:.2f}]")
        print(f"  ✓ PFT-DPW mean: {pft_mean:.2f}")
        print(f"  ✓ PFT-DPW 95% CI: [{pft_ci[0]:.2f}, {pft_ci[1]:.2f}]")
        
        # Check for statistical significance (simplified test)
        if pomcp_ci[1] < pft_ci[0]:
            significance = "PFT-DPW significantly outperforms POMCP"
        elif pft_ci[1] < pomcp_ci[0]:
            significance = "POMCP significantly outperforms PFT-DPW"
        else:
            significance = "No clear statistical difference"
        
        print(f"  ✓ Statistical comparison: {significance}")
        print("  ✓ confidence_interval basic usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ confidence_interval basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_confidence_interval_multi_algorithm():
    """Test multi-algorithm confidence interval example.
    
    Purpose: Validates that confidence_interval supports comprehensive statistical analysis across multiple POMDP planning algorithms
    
    Given: Performance data for 4 algorithms (POMCP, PFT_DPW, SparsePFT, OpenLoop) with confidence=0.95
    When: confidence_interval computes statistical bounds for comparative study analysis
    Then: All algorithms have valid CIs containing means, positive CI widths, enabling systematic performance comparison
    
    Test type: unit
    """
    print("Testing confidence_interval multi-algorithm example...")
    
    try:
        from POMDPPlanners.utils.statistics import confidence_interval
        import numpy as np
        
        # Results from comparative study
        algorithms = {
            'POMCP': [15.2, 14.8, 15.5, 14.9, 15.1, 15.3, 14.7, 15.0, 15.2, 14.8],
            'PFT_DPW': [16.1, 15.9, 16.3, 15.7, 16.0, 16.2, 15.8, 16.1, 15.9, 16.0],
            'SparsePFT': [14.5, 14.9, 14.7, 14.3, 14.8, 14.6, 14.4, 14.7, 14.5, 14.6],
            'OpenLoop': [12.3, 12.7, 12.1, 12.5, 12.4, 12.6, 12.2, 12.8, 12.3, 12.5]
        }
        
        # Statistical analysis with confidence intervals
        results = {}
        for name, rewards in algorithms.items():
            mean_reward = np.mean(rewards)
            ci_lower, ci_upper = confidence_interval(rewards, confidence=0.95)
            ci_width = ci_upper - ci_lower
            
            results[name] = {
                'mean': mean_reward,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_width
            }
            
            # Verify results are reasonable
            assert ci_lower <= mean_reward <= ci_upper, f"{name}: CI should contain mean"
            assert ci_width > 0, f"{name}: CI width should be positive"
            
        # Verify we have results for all algorithms
        assert len(results) == 4, f"Expected 4 algorithms, got {len(results)}"
        
        print("  ✓ Algorithm Performance Analysis (95% CI):")
        for name, stats in results.items():
            print(f"    {name:12}: {stats['mean']:.2f} [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}] (±{stats['ci_width']/2:.2f})")
            
        print("  ✓ confidence_interval multi-algorithm example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ confidence_interval multi-algorithm example failed: {e}")
        traceback.print_exc()
        return False

def test_tree_statistics_basic_usage():
    """Test the basic tree statistics usage example.
    
    Purpose: Validates that tree statistics extraction provides MCTS exploration analysis metrics from planner run data
    
    Given: POMCP planner with TigerPOMDP environment, 20 simulations, depth=5, initial belief with 50 particles
    When: POMCP executes planning and tree statistics are extracted from PolicyRunData
    Then: Metrics include min/max actions visit counts and entropy, all values are numeric and non-NaN
    
    Test type: example
    """
    print("Testing tree statistics basic usage example...")
    
    try:
        from POMDPPlanners.utils.tree_statistics import compute_tree_metrics
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create POMCP planner and run planning (reduced parameters for testing)
        env = TigerPOMDP(discount_factor=0.95)
        planner = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=5,                    # Reduced for testing
            exploration_constant=1.0,
            name="POMCP_Analysis",
            n_simulations=20            # Reduced for testing
        )
        
        initial_belief = get_initial_belief(env, n_particles=50)  # Reduced for testing
        action, run_data = planner.action(initial_belief)
        
        # Extract tree metrics from run data
        metrics = run_data.info_variables
        assert len(metrics) > 0, "Should have at least some metrics"
        
        # Verify expected metric types are present
        metric_names = [m.name for m in metrics]
        expected_metrics = ['min_actions_visit_count', 'max_actions_visit_count', 'actions_visit_count_entropy']
        
        for expected_metric in expected_metrics:
            assert expected_metric in metric_names, f"Missing expected metric: {expected_metric}"
        
        # Verify metric values are reasonable
        for metric in metrics:
            assert isinstance(metric.value, (int, float, np.integer, np.floating)), f"Metric {metric.name} value should be numeric"
            assert not np.isnan(metric.value), f"Metric {metric.name} should not be NaN"
        
        print("  ✓ Tree metrics extracted successfully:")
        for metric in metrics:
            print(f"    {metric.name}: {metric.value:.3f}")
            
        print("  ✓ tree_statistics basic usage example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ tree_statistics basic usage example failed: {e}")
        traceback.print_exc()
        return False

def test_tree_statistics_algorithm_comparison():
    """Test algorithm comparison example from tree statistics.
    
    Purpose: Validates that tree statistics enable comparative analysis of MCTS exploration patterns between different algorithms
    
    Given: POMCP and SparsePFT planners with TigerPOMDP environment, 15 simulations each, initial belief with 30 particles
    When: Both algorithms execute planning and tree exploration statistics are compared
    Then: Both provide min/max visit counts and entropy metrics, enabling analysis of exploration strategies and visit distribution patterns
    
    Test type: unit
    """
    print("Testing tree statistics algorithm comparison example...")
    
    try:
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        env = TigerPOMDP(discount_factor=0.95)
        initial_belief = get_initial_belief(env, n_particles=30)  # Reduced for testing
        
        # Compare tree statistics between different MCTS algorithms (reduced parameters)
        algorithms = {
            'POMCP': POMCP(env, 0.95, depth=5, exploration_constant=1.0, 
                         name="POMCP", n_simulations=15),
            'SparsePFT': SparsePFT(env, 0.95, depth=5, c_ucb=1.0, beta_ucb=2.0,
                                 belief_child_num=3, name="SparsePFT", n_simulations=15)
        }
        
        results = {}
        for name, planner in algorithms.items():
            action, run_data = planner.action(initial_belief)
            
            # Extract metrics
            metrics_dict = {m.name: m.value for m in run_data.info_variables}
            
            # Verify expected metrics are present
            assert 'min_actions_visit_count' in metrics_dict, f"{name} missing min_actions_visit_count"
            assert 'max_actions_visit_count' in metrics_dict, f"{name} missing max_actions_visit_count"
            assert 'actions_visit_count_entropy' in metrics_dict, f"{name} missing actions_visit_count_entropy"
            
            min_visits = metrics_dict['min_actions_visit_count']
            max_visits = metrics_dict['max_actions_visit_count']
            entropy = metrics_dict['actions_visit_count_entropy']
            
            # Verify metrics are reasonable
            assert min_visits >= 0, f"{name} min visits should be non-negative"
            assert max_visits >= min_visits, f"{name} max visits should be >= min visits"
            assert entropy >= 0, f"{name} entropy should be non-negative"
            
            results[name] = {
                'min_visits': min_visits,
                'max_visits': max_visits,
                'entropy': entropy,
                'ratio': max_visits / min_visits if min_visits > 0 else float('inf')
            }
        
        print("  ✓ Algorithm Exploration Analysis:")
        for name, stats in results.items():
            print(f"    {name}:")
            print(f"      Visit range: {stats['min_visits']:.0f} - {stats['max_visits']:.0f}")
            print(f"      Exploration entropy: {stats['entropy']:.3f}")
            print(f"      Visit ratio: {stats['ratio']:.2f}")
            
        print("  ✓ tree_statistics algorithm comparison example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ tree_statistics algorithm comparison example failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all utils usage examples."""
    print("Running ALL usage examples from utils directory classes...\n")
    
    tests = [
        test_unit_circle_action_sampler_basic_usage,
        test_unit_circle_action_sampler_magnitude_comparison,
        test_config_loader_basic_usage,
        test_config_loader_environment_integration,
        test_logger_basic_usage,
        test_logger_file_logging,
        test_cvar_estimator_basic_usage,
        test_cvar_estimator_algorithm_comparison,
        test_confidence_interval_basic_usage,
        test_confidence_interval_multi_algorithm,
        test_tree_statistics_basic_usage,
        test_tree_statistics_algorithm_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} utils test groups passed")
    
    if passed == total:
        print("🎉 ALL utils usage examples work correctly!")
        return 0
    else:
        print("❌ Some utils usage examples have issues.")
        return 1

if __name__ == "__main__":
    exit(main())