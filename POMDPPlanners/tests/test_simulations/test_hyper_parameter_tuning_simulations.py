import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import time
import inspect

from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterOptimizer


@pytest.fixture
def temp_cache_dir():
    # Create a temporary directory for MLFlow cache
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    # Ensure the directory exists and is empty
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        # Wait a bit to ensure all Redis connections are closed
        time.sleep(0.1)
        # Try multiple times to delete the directory
        for _ in range(5):
            try:
                shutil.rmtree(temp_path)
                break
            except PermissionError:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
                break


@pytest.fixture
def optimizer(temp_cache_dir):
    """Create a HyperParameterOptimizer instance for testing."""
    return HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="test_optimization",
        n_jobs=1,
        confidence_interval_level=0.95,
    )


def test_optimizer_initialization(temp_cache_dir):
    """Test that the optimizer initializes correctly."""
    optimizer = HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="test_init",
        n_jobs=2,
        confidence_interval_level=0.99,
    )
    
    assert optimizer.cache_dir_path == temp_cache_dir
    assert optimizer.experiment_name == "test_init"
    assert optimizer.n_jobs == 2
    assert optimizer.confidence_interval_level == 0.99
    assert optimizer.mlruns_path == temp_cache_dir / "mlruns"
    assert optimizer.mlruns_path.exists()


def test_optimize_policy_parameters(optimizer):
    """Test optimizing parameters for a single environment-policy pair."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Execute
    best_params, best_value, histories = optimizer.optimize_policy_parameters(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        num_episodes=2,
        num_steps=2,
        n_particles=10,
        n_trials=2,
    )

    # Assert
    assert isinstance(best_params, dict)
    assert "branching_factor" in best_params
    assert "depth" in best_params
    assert isinstance(best_value, float)
    assert isinstance(histories, list)
    assert len(histories) > 0


def test_optimize_policy_parameters_invalid_params(optimizer):
    """Test that invalid parameters raise appropriate errors."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Test invalid parameters
    with pytest.raises(AssertionError):
        optimizer.optimize_policy_parameters(
            environment=environment,
            policy_class=StandardSparseSamplingDiscreteActionsPlanner,
            param_ranges=param_ranges,
            num_episodes=0,  # Invalid num_episodes
            num_steps=2,
            n_particles=10,
            n_trials=2,
        )


def test_optimize_multiple_environments(optimizer):
    """Test optimizing parameters for multiple environment-policy pairs."""
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    environment_policy_pairs = [
        (environment1, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges)),
    ]

    # Execute
    results, df = optimizer.optimize_multiple_environments(
        environment_policy_pairs=environment_policy_pairs,
        num_episodes=2,
        num_steps=2,
        n_particles=5,  # Reduced for faster testing
        n_trials=1,     # Reduced for faster testing
    )

    # Assert
    assert isinstance(results, list)
    assert len(results) == 1  # One result per environment
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 3  # (best_params, best_value, histories)
        best_params, best_value, histories = result
        assert isinstance(best_params, dict)
        assert isinstance(best_value, float)
        assert isinstance(histories, list)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "param_range_branching_factor" in df.columns
    assert "param_range_depth" in df.columns


def test_simulation_method(optimizer):
    """Test the simulation method directly."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
    )
    initial_belief = get_initial_belief(pomdp=environment, n_particles=10)

    # Execute
    histories, statistics = optimizer.simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=2,
        num_steps=2,
        alpha=0.05,
    )

    # Assert
    assert isinstance(histories, list)
    assert len(histories) == 2
    assert isinstance(statistics, list)
    assert len(statistics) > 0


def test_run_multiple_episodes(optimizer):
    """Test running multiple episodes in parallel."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
    )
    initial_belief = get_initial_belief(pomdp=environment, n_particles=10)

    # Execute
    histories = optimizer.run_multiple_episodes(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=2,
        num_steps=2,
    )

    # Assert
    assert isinstance(histories, list)
    assert len(histories) == 2
    for history in histories:
        from POMDPPlanners.core.simulation import History
        assert isinstance(history, History)
        assert len(history.history) == 2  # num_steps


def test_module_level_usage_example(temp_cache_dir):
    """Test the module-level usage example API pattern from hyper_parameter_tuning_simulations.py.
    
    Purpose: Validates that the documented module-level usage example API can be instantiated correctly
    
    Given: The exact configuration from the module docstring example
    When: Creating optimizer and parameter objects following the documented pattern
    Then: All objects are created successfully with correct types and attributes
    
    Test type: example
    
    Note: Tests the API pattern without running expensive optimization, focusing on 
    validating the documented usage syntax and parameter configuration.
    """
    from pathlib import Path
    from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterOptimizer
    from POMDPPlanners.core.simulation import NumericalHyperParameter
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
    
    # Test optimizer creation (exactly as in module docstring)
    optimizer = HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="POMCP_Tiger_Optimization",
        n_jobs=4
    )
    
    # Validate optimizer attributes match documentation
    assert optimizer.cache_dir_path == temp_cache_dir
    assert optimizer.experiment_name == "POMCP_Tiger_Optimization"
    assert optimizer.n_jobs == 4
    assert optimizer.mlruns_path == temp_cache_dir / "mlruns"
    assert optimizer.mlruns_path.exists()
    
    # Test parameter range definition (exactly as in module docstring)
    param_ranges = [
        NumericalHyperParameter(0.1, 10.0, "exploration_constant"),
        NumericalHyperParameter(100, 2000, "n_simulations"),
        NumericalHyperParameter(10, 100, "depth")
    ]
    
    # Validate parameter ranges structure
    assert len(param_ranges) == 3
    assert param_ranges[0].name == "exploration_constant"
    assert param_ranges[0].low == 0.1
    assert param_ranges[0].high == 10.0
    assert param_ranges[1].name == "n_simulations"
    assert param_ranges[2].name == "depth"
    
    # Test environment creation (exactly as in module docstring)
    environment = TigerPOMDP(discount_factor=0.95)
    assert environment.discount_factor == 0.95
    
    # Validate the optimization method signature matches documentation
    # (without actually running the expensive optimization)
    sig = inspect.signature(optimizer.optimize_policy_parameters)
    expected_params = ['environment', 'policy_class', 'param_ranges', 'num_episodes', 
                      'num_steps', 'n_particles', 'parameter_to_optimize', 'direction', 'n_trials']
    actual_params = list(sig.parameters.keys())
    for param in expected_params:
        assert param in actual_params, f"Expected parameter {param} not found in method signature"


def test_class_level_usage_example(temp_cache_dir):
    """Test the class-level usage example API pattern from HyperParameterOptimizer docstring.
    
    Purpose: Validates that the documented class-level usage example API can be used correctly
    
    Given: The exact configuration from the HyperParameterOptimizer class docstring example
    When: Creating optimizer and parameter objects following the documented pattern
    Then: All objects are created successfully with correct types matching the documented interface
    
    Test type: example
    
    Note: Tests the API pattern without running expensive optimization, focusing on
    validating the documented usage syntax and object instantiation.
    """
    from pathlib import Path
    from POMDPPlanners.core.simulation import NumericalHyperParameter
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
    
    # Create optimizer (exactly as in class docstring)
    optimizer = HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="POMCP_Tuning",
        n_jobs=4
    )
    
    # Validate optimizer creation matches documentation
    assert isinstance(optimizer, HyperParameterOptimizer)
    assert optimizer.experiment_name == "POMCP_Tuning"
    assert optimizer.n_jobs == 4
    
    # Define parameter ranges (exactly as in class docstring)
    param_ranges = [
        NumericalHyperParameter(0.1, 10.0, "exploration_constant"),
        NumericalHyperParameter(100, 1000, "n_simulations")
    ]
    
    # Validate parameter range structure matches documentation
    assert len(param_ranges) == 2
    assert all(isinstance(p, NumericalHyperParameter) for p in param_ranges)
    assert param_ranges[0].name == "exploration_constant"
    assert param_ranges[1].name == "n_simulations"
    
    # Create environment for testing (following docstring pattern)
    env = TigerPOMDP(discount_factor=0.95)
    assert isinstance(env, TigerPOMDP)
    assert env.discount_factor == 0.95
    
    # Validate that the documented method exists and has correct signature
    assert hasattr(optimizer, 'optimize_policy_parameters')
    method = getattr(optimizer, 'optimize_policy_parameters')
    assert callable(method)
    
    # Test that we can create policy class reference (without instantiation)
    assert POMCP is not None
    assert inspect.isclass(POMCP)
    from POMDPPlanners.core.policy import Policy
    assert issubclass(POMCP, Policy)
    
    # Test the simulation method also exists as documented
    assert hasattr(optimizer, 'simulation')
    assert callable(getattr(optimizer, 'simulation'))
    
    # Test run_multiple_episodes method exists as documented  
    assert hasattr(optimizer, 'run_multiple_episodes')
    assert callable(getattr(optimizer, 'run_multiple_episodes'))


def test_mlflow_nested_run_fix(optimizer):
    """Test that MLflow nested runs work correctly when simulator is called within an active run.
    
    Purpose: Validates that the MLflow nested run fix allows simulator to run within optimization context
    
    Given: An active MLflow run context
    When: The simulator is called from within that context
    Then: No MLflow exception is raised and the operation completes successfully
    
    Test type: unit
    """
    import mlflow
    from POMDPPlanners.core.belief import get_initial_belief
    from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
    
    # Setup minimal test data
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
    )
    initial_belief = get_initial_belief(pomdp=environment, n_particles=5)
    
    # Test that we can run simulation within an MLflow run context
    with mlflow.start_run(run_name="test_parent_run"):
        # This should not raise an MLflow exception due to nested run fix
        histories = optimizer.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=2,
            num_steps=2,
        )
        
        # Verify that the operation completed successfully
        assert isinstance(histories, list)
        assert len(histories) == 2
        
        # Verify the nested run functionality worked
        # by checking that we can still log to the parent run
        mlflow.log_param("test_param", "nested_run_fix_success")


def test_optimize_and_evaluate_multiple_environments(optimizer):
    """Test the combined optimization and evaluation method.
    
    Purpose: Validates that the optimize_and_evaluate_multiple_environments method works correctly
    
    Given: A single environment-policy pair with minimal parameters for fast testing
    When: Running the combined optimization and evaluation method
    Then: Both optimization and evaluation phases complete successfully and return correct data structures
    
    Test type: unit
    """
    # Setup minimal test configuration
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]
    
    environment_policy_pairs = [
        (environment, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges)),
    ]
    
    # Execute the combined method with minimal episodes for speed
    opt_results, opt_df, eval_results = optimizer.optimize_and_evaluate_multiple_environments(
        environment_policy_pairs=environment_policy_pairs,
        num_episodes_tuning=2,      # Minimum for statistics
        num_episodes_evaluation=3,  # Small but larger than tuning
        num_steps=2,
        n_particles=5,
        n_trials=1,                 # Single trial for speed
        cache_visualizations=False  # Disable to speed up test
    )
    
    # Validate optimization results
    assert isinstance(opt_results, list)
    assert len(opt_results) == 1  # One result per environment-policy pair
    best_params, best_value, opt_histories = opt_results[0]
    assert isinstance(best_params, dict)
    assert "branching_factor" in best_params
    assert "depth" in best_params
    assert isinstance(best_value, float)
    assert isinstance(opt_histories, list)
    
    # Validate optimization DataFrame
    assert isinstance(opt_df, pd.DataFrame)
    assert len(opt_df) == 1  # One row per environment-policy pair
    assert "param_range_branching_factor" in opt_df.columns
    assert "param_range_depth" in opt_df.columns
    
    # Validate evaluation results
    assert isinstance(eval_results, dict)
    assert environment.name in eval_results
    env_results = eval_results[environment.name]
    assert isinstance(env_results, dict)
    
    # Should contain one policy with optimized parameters
    assert len(env_results) == 1
    policy_name = list(env_results.keys())[0]
    policy_histories = env_results[policy_name]
    assert isinstance(policy_histories, list)
    assert len(policy_histories) == 3  # num_episodes_evaluation


def test_optimize_and_evaluate_multiple_environments_invalid_params(optimizer):
    """Test that invalid parameters raise appropriate errors in optimize_and_evaluate_multiple_environments.
    
    Purpose: Validates proper error handling for invalid input parameters
    
    Given: Invalid parameter combinations
    When: Calling the optimize_and_evaluate_multiple_environments method
    Then: Appropriate assertion errors are raised
    
    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
    ]
    environment_policy_pairs = [
        (environment, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges)),
    ]
    
    # Test invalid num_episodes_tuning
    with pytest.raises(AssertionError):
        optimizer.optimize_and_evaluate_multiple_environments(
            environment_policy_pairs=environment_policy_pairs,
            num_episodes_tuning=0,      # Invalid: must be > 0
            num_episodes_evaluation=2,
            num_steps=2,
            n_particles=5,
        )
    
    # Test invalid num_episodes_evaluation
    with pytest.raises(AssertionError):
        optimizer.optimize_and_evaluate_multiple_environments(
            environment_policy_pairs=environment_policy_pairs,
            num_episodes_tuning=2,
            num_episodes_evaluation=0,  # Invalid: must be > 0
            num_steps=2,
            n_particles=5,
        )
    
    # Test invalid alpha
    with pytest.raises(AssertionError):
        optimizer.optimize_and_evaluate_multiple_environments(
            environment_policy_pairs=environment_policy_pairs,
            num_episodes_tuning=2,
            num_episodes_evaluation=3,
            num_steps=2,
            n_particles=5,
            alpha=1.5,  # Invalid: must be between 0 and 1
        )


def test_optimize_and_evaluate_usage_example_api(temp_cache_dir):
    """Test the optimize_and_evaluate_multiple_environments usage example API pattern.
    
    Purpose: Validates that the documented usage example API can be instantiated correctly
    
    Given: Configuration following the usage example pattern from the method docstring
    When: Creating optimizer and calling optimize_and_evaluate_multiple_environments with documented parameters
    Then: All objects are created successfully and method signature matches documentation
    
    Test type: example
    
    Note: Tests the API pattern without running expensive optimization, focusing on
    validating the documented usage syntax and method signature.
    """
    from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
    from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
    
    # Test optimizer creation as shown in usage example
    optimizer = HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="Test_Optimization_Evaluation",
        n_jobs=1
    )
    
    # Test environment-policy pairs creation (following documented pattern)
    tiger_env = TigerPOMDP(discount_factor=0.95)
    env_policy_pairs = [
        (tiger_env, (POMCP, [
            NumericalHyperParameter(low=0.1, high=10.0, name="exploration_constant"),
            NumericalHyperParameter(low=100, high=1000, name="n_simulations")
        ])),
    ]
    
    # Validate that the method exists and has correct signature
    assert hasattr(optimizer, 'optimize_and_evaluate_multiple_environments')
    method = getattr(optimizer, 'optimize_and_evaluate_multiple_environments')
    assert callable(method)
    
    # Check method signature matches documentation
    sig = inspect.signature(method)
    expected_params = [
        'environment_policy_pairs', 'num_episodes_tuning', 'num_episodes_evaluation',
        'num_steps', 'n_particles', 'parameter_to_optimize', 'direction', 'n_trials',
        'alpha', 'confidence_interval_level', 'cache_visualizations'
    ]
    actual_params = list(sig.parameters.keys())
    for param in expected_params:
        assert param in actual_params, f"Expected parameter {param} not found in method signature"
    
    # Test that environment-policy pairs structure matches documentation
    assert len(env_policy_pairs) == 1
    env, (policy_class, param_ranges) = env_policy_pairs[0]
    assert isinstance(env, TigerPOMDP)
    assert policy_class == POMCP
    assert len(param_ranges) == 2
    assert param_ranges[0].name == "exploration_constant"
    assert param_ranges[1].name == "n_simulations"
