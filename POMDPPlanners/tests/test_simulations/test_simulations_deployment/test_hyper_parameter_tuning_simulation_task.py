import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from POMDPPlanners.simulations.simulations_deployment.tasks import HyperParameterTuningSimulationTask
from POMDPPlanners.core.simulation import History, NumericalHyperParameter, CategoricalHyperParameter
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterOptimizationDirection


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        resampling=False
    )

@pytest.fixture
def environment():
    """Fixture to create a Tiger POMDP environment."""
    return TigerPOMDP(discount_factor=0.95)

@pytest.fixture
def hyper_parameters():
    """Fixture to create test hyperparameters."""
    return [
        NumericalHyperParameter(name="branching_factor", low=1, high=5),
        NumericalHyperParameter(name="depth", low=1, high=10)
    ]

@pytest.fixture
def categorical_hyper_parameters():
    """Fixture to create test categorical hyperparameters."""
    return [
        CategoricalHyperParameter(name="algorithm", choices=["sparse", "dense", "hybrid"]),
        CategoricalHyperParameter(name="heuristic", choices=["ucb", "epsilon_greedy", "random"])
    ]

def test_hyper_parameter_tuning_task_creation(environment, hyper_parameters):
    """Test creation and basic properties of HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask can be created with correct attributes
    
    Given: A TigerPOMDP environment, StandardSparseSamplingDiscreteActionsPlanner policy class, and test hyperparameters
    When: HyperParameterTuningSimulationTask is created with specific parameters
    Then: Task has correct attributes matching input parameters and generates valid config ID
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    assert task.num_episodes == 5
    assert task.num_steps == 10
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
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    belief = create_test_belief()
    custom_seed = 12345
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    belief = create_test_belief()
    
    # Create identical tasks
    task1 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
        num_episodes=5,
        num_steps=10,
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
        num_episodes=10,  # Different
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
        num_episodes=5,
        num_steps=10,
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
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    assert task_dict["num_episodes"] == 5
    assert task_dict["num_steps"] == 10
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
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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

def test_hyper_parameter_tuning_task_run_success(mock_create_study, environment, hyper_parameters):
    """Test successful execution of HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask can execute successfully and return results
    
    Given: A HyperParameterTuningSimulationTask with valid configuration and mocked Optuna study
    When: Task is executed with run() method
    Then: Task executes successfully and returns OptimizedPolicyResult with correct attributes
    
    Test type: integration
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Mock the run method to avoid actual execution
    with patch.object(task, '_evaluate_policy_configuration') as mock_evaluate:
        mock_evaluate.return_value = 0.8
        
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
        assert result.num_episodes == 5
        assert result.num_steps == 10
        assert result.direction == HyperParameterOptimizationDirection.MAXIMIZE
        assert result.parameter_to_optimize == "average_return"

def test_hyper_parameter_tuning_task_run_with_categorical_params(mock_create_study, environment, categorical_hyper_parameters):
    """Test execution of HyperParameterTuningSimulationTask with categorical hyperparameters.
    
    Purpose: Validates that HyperParameterTuningSimulationTask handles categorical hyperparameters correctly
    
    Given: A HyperParameterTuningSimulationTask with categorical hyperparameters
    When: Task is executed with run() method
    Then: Task executes successfully and handles categorical parameters correctly
    
    Test type: integration
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=categorical_hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Mock the run method to avoid actual execution
    with patch.object(task, '_evaluate_policy_configuration') as mock_evaluate:
        mock_evaluate.return_value = 0.8
        
        result = task.run()
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'chosen_hyper_parameters')

def test_hyper_parameter_tuning_task_run_failure_logging(mock_create_study, environment, hyper_parameters, caplog):
    """Test that HyperParameterTuningSimulationTask logs failures properly.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logs failures with appropriate detail
    
    Given: A HyperParameterTuningSimulationTask that encounters an exception during execution
    When: Task execution raises an exception
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Mock the Optuna study to fail during optimization
    with patch('optuna.create_study') as mock_create_study:
        # Create a mock study that will fail
        mock_study = Mock()
        mock_study.optimize.side_effect = Exception("Test execution error")
        mock_create_study.return_value = mock_study
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Hyperparameter optimization failed: Test execution error" in caplog.text
        assert result is None

def test_hyper_parameter_tuning_task_run_missing_parameter_logging(mock_create_study, environment, hyper_parameters, caplog):
    """Test that HyperParameterTuningSimulationTask logs missing parameter errors properly.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logs missing parameter errors with appropriate detail
    
    Given: A HyperParameterTuningSimulationTask that encounters a missing parameter error during evaluation
    When: Task evaluation raises a ValueError about missing parameter
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Mock the Optuna study to fail during optimization with a specific ValueError
    with patch('optuna.create_study') as mock_create_study:
        # Create a mock study that will fail
        mock_study = Mock()
        mock_study.optimize.side_effect = ValueError("Parameter average_return not found in computed statistics")
        mock_create_study.return_value = mock_study
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Hyperparameter optimization failed: Parameter average_return not found in computed statistics" in caplog.text
        assert result is None

def test_hyper_parameter_tuning_task_custom_n_trials(environment, hyper_parameters):
    """Test HyperParameterTuningSimulationTask with custom n_trials parameter.
    
    Purpose: Validates that HyperParameterTuningSimulationTask accepts and uses custom n_trials values
    
    Given: A HyperParameterTuningSimulationTask with custom n_trials value
    When: Task is created with n_trials=100
    Then: Task uses the specified n_trials value instead of default
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=100,  # Custom value
        seed=42
    )
    
    assert task.n_trials == 100
    assert task.n_trials != 50  # Should not be default

def test_hyper_parameter_tuning_task_logger_property(environment, hyper_parameters):
    """Test logger property for HyperParameterTuningSimulationTask.
    
    Purpose: Validates that HyperParameterTuningSimulationTask logger property works correctly
    
    Given: A HyperParameterTuningSimulationTask with specific configuration
    When: Task logger property is accessed
    Then: Logger is created with correct name and configuration
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
    belief = create_test_belief()
    
    # Create task with seed 42
    task1 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
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
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=999
    )
    
    # Verify different seeds result in different config IDs
    config_id1 = task1.get_config_id()
    config_id2 = task2.get_config_id()
    assert config_id1 != config_id2

def test_hyper_parameter_tuning_task_seed_in_optimization_results(mock_create_study, environment, hyper_parameters):
    """Test that seed parameter is included in optimization results.
    
    Purpose: Validates that seed parameter is properly stored in optimization result metadata
    
    Given: A HyperParameterTuningSimulationTask with specific seed value
    When: Task optimization results are retrieved
    Then: Seed parameter is included in the results
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        num_episodes=5,
        num_steps=10,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42
    )
    
    # Mock the run method to avoid actual execution
    with patch.object(task, '_evaluate_policy_configuration') as mock_evaluate:
        mock_evaluate.return_value = 0.8
        
        result = task.run()
        
        # Verify seed is included in optimization result dictionary
        result_dict = task.get_optimization_result_dict()
        assert result_dict is not None
        assert 'seed' in result_dict
        assert result_dict['seed'] == 42

# Mock fixture for Optuna study
@pytest.fixture
def mock_create_study():
    """Mock fixture for Optuna study creation."""
    with patch('optuna.create_study') as mock:
        study = Mock()
        study.best_params = {'branching_factor': 3, 'depth': 5}
        study.best_value = 0.8
        study.best_trial = Mock()
        study.best_trial.number = 1
        study.best_trial.user_attrs = {'statistics': [{'name': 'average_return', 'value': 0.8}]}
        mock.return_value = study
        yield mock
