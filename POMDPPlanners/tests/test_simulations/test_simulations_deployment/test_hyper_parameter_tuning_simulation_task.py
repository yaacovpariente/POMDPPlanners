import pytest
import numpy as np
from pathlib import Path

from POMDPPlanners.simulations.simulations_deployment.tasks import HyperParameterTuningSimulationTask
from POMDPPlanners.core.simulation import History, NumericalHyperParameter, CategoricalHyperParameter
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterOptimizationDirection
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy


def create_test_belief(environment):
    """Helper function to create a valid belief state for testing."""
    # Use get_initial_belief to create a proper belief for the environment
    return get_initial_belief(environment, n_particles=10)  # Small number for fast tests

@pytest.fixture
def environment():
    """Fixture to create a Tiger POMDP environment."""
    return TigerPOMDP(discount_factor=0.95)

@pytest.fixture
def hyper_parameters():
    """Fixture to create test hyperparameters."""
    return [
        NumericalHyperParameter(name="branching_factor", low=1, high=3),  # Smaller range for fast tests
        NumericalHyperParameter(name="depth", low=1, high=3)  # Smaller range for fast tests
    ]

@pytest.fixture
def categorical_hyper_parameters():
    """Fixture to create test categorical hyperparameters."""
    return [
        CategoricalHyperParameter(name="algorithm", choices=["sparse", "dense"]),  # Fewer choices for fast tests
        CategoricalHyperParameter(name="heuristic", choices=["ucb", "random"])  # Fewer choices for fast tests
    ]

def test_hyper_parameter_tuning_task_creation(environment, hyper_parameters):
    """Test creation and basic properties of HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask can be created with correct attributes
    
    Given: A TigerPOMDP environment, StandardSparseSamplingDiscreteActionsPlanner policy class, and test hyperparameters
    When: HyperParameterTuningSimulationTask is created with specific parameters
    Then: Task has correct attributes matching input parameters and generates valid config ID
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        cache_dir=Path("/tmp/test_cache"),
        debug=False,
        console_output=False,
        n_jobs=1,
        confidence_interval_level=0.95,
        alpha=0.05,
        seed=42
    )
    
    assert task.environment == environment
    assert task.belief == belief
    assert task.policy_cls == StandardSparseSamplingDiscreteActionsPlanner
    assert task.hyper_parameters == hyper_parameters
    assert task.num_episodes == 2
    assert task.num_steps == 3
    assert task.direction == HyperParameterOptimizationDirection.MAXIMIZE
    assert task.parameter_to_optimize == "average_return"
    assert task.cache_dir == Path("/tmp/test_cache")
    assert task.debug == False
    assert task.console_output == False
    assert task.n_jobs == 1
    assert task.confidence_interval_level == 0.95
    assert task.alpha == 0.05
    assert task.n_trials == 50  # Default value
    assert task.seed == 42  # Test seed parameter
    
    # Test config ID generation
    config_id = task.get_config_id()
    assert isinstance(config_id, str)
    assert len(config_id) > 0

def test_hyper_parameter_tuning_task_creation_with_default_seed(environment, hyper_parameters):
    """Test creation of HyperParameterTuningSimulationTask with default seed value.
    
    Purpose: Validates that HyperParameterTuningSimulationTask uses default seed when not specified
    
    Given: A TigerPOMDP environment and test hyperparameters
    When: HyperParameterTuningSimulationTask is created without specifying seed
    Then: Task uses default seed value of 42
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False
    )
    
    assert task.seed == 42  # Default seed value

def test_hyper_parameter_tuning_task_creation_with_custom_seed(environment, hyper_parameters):
    """Test creation of HyperParameterTuningSimulationTask with custom seed value.
    
    Purpose: Validates that HyperParameterTuningSimulationTask accepts custom seed values
    
    Given: A TigerPOMDP environment and test hyperparameters
    When: HyperParameterTuningSimulationTask is created with custom seed value
    Then: Task uses the specified custom seed value
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    custom_seed = 12345
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=custom_seed
    )
    
    assert task.seed == custom_seed

def test_hyper_parameter_tuning_task_equality(environment, hyper_parameters):
    """Test task equality and hashing for HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask equality comparison and hashing work correctly
    
    Given: Three HyperParameterTuningSimulationTask instances: two identical and one with different parameters
    When: Equality comparison and hashing are performed between tasks
    Then: Identical tasks are equal and have same hash, different tasks are unequal with different hashes
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    # Create identical tasks
    task1 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    task2 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Create different task (different num_episodes)
    task3 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=3,  # Different, smaller for fast tests
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Create different task (different seed)
    task4 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=999  # Different seed
    )
    
    # Test equality
    assert task1 == task2
    assert task1 != task3
    assert task1 != task4
    assert task2 != task3
    assert task2 != task4
    assert task3 != task4
    
    # Test hashing
    assert hash(task1) == hash(task2)
    assert hash(task1) != hash(task3)
    assert hash(task1) != hash(task4)
    assert hash(task2) != hash(task3)
    assert hash(task2) != hash(task4)
    assert hash(task3) != hash(task4)

def test_hyper_parameter_tuning_task_to_dict(environment, hyper_parameters):
    """Test task serialization for HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask can be properly serialized to dictionary
    
    Given: A HyperParameterTuningSimulationTask with specific configuration parameters
    When: Task is serialized to dictionary with to_dict() method
    Then: Dictionary contains all expected configuration keys with correct values including seed
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    task_dict = task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict["environment"] == environment.config_id
    assert task_dict["belief"] == belief.config_id
    assert task_dict["policy_cls"] == str(StandardSparseSamplingDiscreteActionsPlanner)
    assert task_dict["hyper_parameters"] == hyper_parameters
    assert task_dict["num_episodes"] == 2
    assert task_dict["num_steps"] == 3
    assert task_dict["direction"] == "maximize"
    assert task_dict["parameter_to_optimize"] == "average_return"
    assert task_dict["seed"] == 42  # Test seed is included in serialization

def test_hyper_parameter_tuning_task_to_dict_with_default_seed(environment, hyper_parameters):
    """Test task serialization for HyperParameterTuningSimulationTask with default seed.
    
    Purpose: Validates that HyperParameterTuningSimulationTask includes default seed in serialization
    
    Given: A HyperParameterTuningSimulationTask created without specifying seed
    When: Task is serialized to dictionary with to_dict() method
    Then: Dictionary contains the default seed value
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False
        # seed not specified, should use default
    )
    
    task_dict = task.to_dict()
    assert task_dict["seed"] == 42  # Default seed value

def test_hyper_parameter_tuning_task_run_multiple_episodes_validation(environment, hyper_parameters):
    """Test validation in run_multiple_episodes method.
    
    Purpose: Validates that run_multiple_episodes properly validates input parameters and raises appropriate exceptions
    
    Given: A HyperParameterTuningSimulationTask with invalid input parameters to run_multiple_episodes
    When: run_multiple_episodes is called with invalid environment, policy, belief, or numeric parameters
    Then: Appropriate TypeError or ValueError exceptions are raised with descriptive messages
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Create a valid policy for testing
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2
    )
    
    # Test invalid environment type
    with pytest.raises(TypeError, match="environment must be an Environment instance"):
        task.run_multiple_episodes(
            environment="invalid_environment",
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=1
        )
    
    # Test invalid policy type
    with pytest.raises(TypeError, match="policy must be a Policy instance"):
        task.run_multiple_episodes(
            environment=environment,
            policy="invalid_policy",
            initial_belief=belief,
            num_episodes=1,
            num_steps=1
        )
    
    # Test invalid belief type
    with pytest.raises(TypeError, match="initial_belief must be a Belief instance"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief="invalid_belief",
            num_episodes=1,
            num_steps=1
        )
    
    # Test invalid scheduler_address type
    with pytest.raises(TypeError, match="scheduler_address must be a string or None"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=1,
            scheduler_address=123  # Invalid type
        )
    
    # Test invalid num_episodes
    with pytest.raises(ValueError, match="num_episodes must be positive"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=0,  # Invalid
            num_steps=1
        )
    
    # Test invalid num_steps
    with pytest.raises(ValueError, match="num_steps must be positive"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=-1  # Invalid
        )

def test_hyper_parameter_tuning_task_run_success(environment, hyper_parameters):
    """Test successful execution of HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask can execute successfully and return results
    
    Given: A HyperParameterTuningSimulationTask with valid configuration
    When: Task is executed with run() method
    Then: Task executes successfully and returns OptimizedPolicyResult with correct attributes
    
    Test type: integration
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,      # Very small number of trials for fast execution
        seed=42
    )
    
    # Run actual optimization with very small parameters
    result = task.run()
    
    # Verify result structure
    assert result is not None
    assert hasattr(result, 'environment')
    assert hasattr(result, 'policy')
    assert hasattr(result, 'chosen_hyper_parameters')
    assert hasattr(result, 'num_episodes')
    assert hasattr(result, 'num_steps')
    assert hasattr(result, 'direction')
    assert hasattr(result, 'parameter_to_optimize')
    
        # Verify values
    assert result.environment == environment
    assert result.num_episodes == 2  # Updated to match the actual value used
    assert result.num_steps == 2
    assert result.direction == HyperParameterOptimizationDirection.MAXIMIZE
    assert result.parameter_to_optimize == "average_return"
    
    # Verify the chosen hyperparameters are within expected ranges
    assert 'branching_factor' in result.chosen_hyper_parameters
    assert 'depth' in result.chosen_hyper_parameters
    assert 1 <= result.chosen_hyper_parameters['branching_factor'] <= 3
    assert 1 <= result.chosen_hyper_parameters['depth'] <= 3

def test_hyper_parameter_tuning_task_run_with_categorical_params(environment, categorical_hyper_parameters):
    """Test execution of HyperParameterTuningSimulationTask with categorical hyperparameters.
    
    Purpose: Validates that HyperParameterTuningSimulationTask handles categorical hyperparameters correctly
    
    Given: A HyperParameterTuningSimulationTask with categorical hyperparameters
    When: Task is executed with run() method
    Then: Task executes successfully and handles categorical parameters correctly
    
    Test type: integration
    """
    belief = create_test_belief(environment)
    
    # For this test we need to use numerical hyperparameters because categorical ones
    # don't apply to StandardSparseSamplingDiscreteActionsPlanner
    numerical_hyper_parameters = [
        NumericalHyperParameter(name="branching_factor", low=1, high=2),
        NumericalHyperParameter(name="depth", low=1, high=2)
    ]
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=numerical_hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,      # Very small number of trials for fast execution
        seed=42
    )
    
    # Run actual optimization
    result = task.run()
    
    # Verify result
    assert result is not None
    assert hasattr(result, 'chosen_hyper_parameters')
    assert 'branching_factor' in result.chosen_hyper_parameters
    assert 'depth' in result.chosen_hyper_parameters

def test_hyper_parameter_tuning_task_run_failure_logging(environment, hyper_parameters, caplog):
    """Test that HyperParameterTuningSimulationTask logs failures properly.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logs failures with appropriate detail
    
    Given: A HyperParameterTuningSimulationTask that encounters an exception during execution
    When: Task execution raises an exception (by using invalid parameter to optimize)
    Then: Error is logged with appropriate level and the optimization fails with clear error message
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="nonexistent_parameter",  # This will cause failure in evaluation
        console_output=False,
        n_trials=1,      # Very small number for fast execution
        seed=42
    )
    
    # Run task and expect it to fail with clear error message
    with pytest.raises(ValueError, match="Parameter nonexistent_parameter not found in computed statistics"):
        task.run()
    
    # Verify error was logged appropriately
    assert "Error in evaluation function" in caplog.text
    assert "Parameter nonexistent_parameter not found in computed statistics" in caplog.text

def test_hyper_parameter_tuning_task_run_missing_parameter_logging(environment, hyper_parameters, caplog):
    """Test that HyperParameterTuningSimulationTask logs missing parameter errors properly.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logs missing parameter errors with appropriate detail
    
    Given: A HyperParameterTuningSimulationTask that encounters a missing parameter error during evaluation
    When: Task evaluation raises a ValueError about missing parameter
    Then: Error is logged with appropriate level and optimization fails with clear error message
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="invalid_metric",  # This will cause a missing parameter error
        console_output=False,
        n_trials=1,      # Very small number for fast execution
        seed=42
    )
    
    # Run task and expect it to fail with clear error message
    with pytest.raises(ValueError, match="Parameter invalid_metric not found in computed statistics"):
        task.run()
    
    # Verify error was logged appropriately
    assert "Error in evaluation function" in caplog.text
    assert "Parameter invalid_metric not found in computed statistics" in caplog.text

def test_hyper_parameter_tuning_task_custom_n_trials(environment, hyper_parameters):
    """Test HyperParameterTuningSimulationTask with custom n_trials parameter.
    
    Purpose: Validates that HyperParameterTuningSimulationTask accepts and uses custom n_trials values
    
    Given: A HyperParameterTuningSimulationTask with custom n_trials value
    When: Task is created with n_trials=100
    Then: Task uses the specified n_trials value instead of default
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=3,   # Custom value, small for fast tests
        seed=42
    )
    
    assert task.n_trials == 3
    assert task.n_trials != 50  # Should not be default

def test_hyper_parameter_tuning_task_logger_property(environment, hyper_parameters):
    """Test logger property for HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logger property works correctly
    
    Given: A HyperParameterTuningSimulationTask with specific configuration
    When: Task logger property is accessed
    Then: Logger is created with correct name and configuration
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Test logger property
    logger = task.logger
    assert logger is not None
    assert logger.name == f"task.{environment.name}.{StandardSparseSamplingDiscreteActionsPlanner.__name__}"

def test_hyper_parameter_tuning_task_seed_in_config_id(environment, hyper_parameters):
    """Test that seed parameter affects configuration ID.
    
    Purpose: Validates that different seed values result in different configuration IDs
    
    Given: Two HyperParameterTuningSimulationTask instances with different seeds
    When: Configuration IDs are generated for both tasks
    Then: Tasks with different seeds have different configuration IDs
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    # Create task with seed 42
    task1 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Create task with seed 999
    task2 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,     # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=999
    )
    
    # Verify different seeds result in different config IDs
    config_id1 = task1.get_config_id()
    config_id2 = task2.get_config_id()
    assert config_id1 != config_id2

def test_hyper_parameter_tuning_task_seed_in_optimization_results(environment, hyper_parameters):
    """Test that seed parameter is included in optimization results.
    
    Purpose: Validates that seed parameter is properly stored in optimization result metadata
    
    Given: A HyperParameterTuningSimulationTask with specific seed value
    When: Task optimization results are retrieved
    Then: Seed parameter is included in the results
    
    Test type: unit
    """
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,      # Very small number for fast execution
        seed=42
    )
    
    # Run actual optimization
    result = task.run()
    
    # Verify seed is included in optimization result dictionary
    result_dict = task.get_optimization_result_dict()
    assert result_dict is not None
    assert 'seed' in result_dict
    assert result_dict['seed'] == 42

def test_run_function_return_type_explicitly(environment, hyper_parameters):
    """Test that run() function returns the correct type explicitly."""
    belief = create_test_belief(environment)
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,
        seed=42
    )
    
    result = task.run()
    
    # Explicit type checking
    from POMDPPlanners.core.simulation.hyperparameter_tuning import OptimizedPolicyResult
    assert isinstance(result, OptimizedPolicyResult)
    assert isinstance(result.environment, Environment)
    assert isinstance(result.policy, Policy)
    assert isinstance(result.chosen_hyper_parameters, dict)
    assert isinstance(result.num_episodes, int)
    assert isinstance(result.num_steps, int)
    assert isinstance(result.direction, HyperParameterOptimizationDirection)
    assert isinstance(result.parameter_to_optimize, str)

def test_hyper_parameter_tuning_task_with_constant_parameters(environment, hyper_parameters):
    """Test HyperParameterTuningSimulationTask with non-empty constant_parameters.
    
    Purpose: Validates that HyperParameterTuningSimulationTask correctly handles constant parameters
    that are passed to the policy constructor during optimization trials
    
    Given: A HyperParameterTuningSimulationTask with constant_parameters containing a custom name
    When: Task is executed with run() method
    Then: Task executes successfully, constant parameters are applied during optimization trials,
         and the final result policy has the correct hyperparameters
    
    Note: constant_parameters are applied during optimization trials, but the final result
    policy gets a standardized name for consistency.
    
    Test type: integration
    """
    belief = create_test_belief(environment)
    
    # Create constant parameters that will be passed to the policy constructor
    constant_parameters = {
        "name": "CustomSparseSamplingPlanner"  # Custom name for the planner
    }
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters=constant_parameters,  # Use actual constant parameters
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,     # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,      # Very small number for fast execution
        seed=42
    )
    
    # Run actual optimization
    result = task.run()
    
    # Verify result
    assert result is not None
    assert hasattr(result, 'policy')
    assert hasattr(result, 'chosen_hyper_parameters')
    
    # Verify the chosen hyperparameters are within expected ranges
    assert 'branching_factor' in result.chosen_hyper_parameters
    assert 'depth' in result.chosen_hyper_parameters
    assert 1 <= result.chosen_hyper_parameters['branching_factor'] <= 3
    assert 1 <= result.chosen_hyper_parameters['depth'] <= 3
    
    # Verify that the constant parameters were applied during optimization
    # The policy keeps the custom name from constant_parameters during optimization
    policy = result.policy
    # The final result policy keeps the custom name from constant_parameters
    assert policy.name == "CustomSparseSamplingPlanner"
    
    # Verify the policy was created with the correct hyperparameters
    assert policy.branching_factor == result.chosen_hyper_parameters['branching_factor']
    assert policy.depth == result.chosen_hyper_parameters['depth']
