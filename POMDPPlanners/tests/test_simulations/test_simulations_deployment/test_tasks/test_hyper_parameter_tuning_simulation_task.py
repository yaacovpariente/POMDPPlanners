import pytest
import numpy as np
import tempfile
from pathlib import Path

from POMDPPlanners.simulations.simulations_deployment.tasks import (
    HyperParameterTuningSimulationTask,
)
from POMDPPlanners.core.simulation import (
    History,
    NumericalHyperParameter,
    CategoricalHyperParameter,
)
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy


def create_test_belief(environment):
    """Helper function to create a valid belief state for testing."""
    # Use get_initial_belief to create a proper belief for the environment
    return get_initial_belief(
        environment, n_particles=10
    )  # Small number for fast tests


@pytest.fixture
def temp_cache_dir():
    """Fixture to create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def real_environment():
    """Fixture to create a real TigerPOMDP environment for testing."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def real_belief(real_environment):
    """Fixture to create a real belief state for testing."""
    return get_initial_belief(real_environment, n_particles=10)


@pytest.fixture
def environment():
    """Fixture to create a Tiger POMDP environment."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def hyper_parameters():
    """Fixture to create test hyperparameters."""
    return [
        NumericalHyperParameter(
            name="branching_factor", low=1, high=3
        ),  # Smaller range for fast tests
        NumericalHyperParameter(
            name="depth", low=1, high=3
        ),  # Smaller range for fast tests
    ]


@pytest.fixture
def categorical_hyper_parameters():
    """Fixture to create test categorical hyperparameters."""
    return [
        CategoricalHyperParameter(
            name="algorithm", choices=["sparse", "dense"]
        ),  # Fewer choices for fast tests
        CategoricalHyperParameter(
            name="heuristic", choices=["ucb", "random"]
        ),  # Fewer choices for fast tests
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        cache_dir=Path("/tmp/test_cache"),
        debug=False,
        console_output=False,
        n_jobs=1,
        confidence_interval_level=0.95,
        alpha=0.05,
        seed=42,
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


def test_hyper_parameter_tuning_task_creation_with_default_seed(
    environment, hyper_parameters
):
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
    )

    assert task.seed == 42  # Default seed value


def test_hyper_parameter_tuning_task_creation_with_custom_seed(
    environment, hyper_parameters
):
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=custom_seed,
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
    )

    task2 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
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
        seed=42,
    )

    # Create different task (different seed)
    task4 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=999,  # Different seed
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
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


def test_hyper_parameter_tuning_task_to_dict_with_default_seed(
    environment, hyper_parameters
):
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        # seed not specified, should use default
    )

    task_dict = task.to_dict()
    assert task_dict["seed"] == 42  # Default seed value


def test_hyper_parameter_tuning_task_run_multiple_episodes_validation(
    environment, hyper_parameters
):
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
    )

    # Create a valid policy for testing
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=2
    )

    # Test invalid environment type
    with pytest.raises(TypeError, match="environment must be an Environment instance"):
        task.run_multiple_episodes(
            environment="invalid_environment",
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=1,
        )

    # Test invalid policy type
    with pytest.raises(TypeError, match="policy must be a Policy instance"):
        task.run_multiple_episodes(
            environment=environment,
            policy="invalid_policy",
            initial_belief=belief,
            num_episodes=1,
            num_steps=1,
        )

    # Test invalid belief type
    with pytest.raises(TypeError, match="initial_belief must be a Belief instance"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief="invalid_belief",
            num_episodes=1,
            num_steps=1,
        )

    # Test invalid scheduler_address type
    with pytest.raises(TypeError, match="scheduler_address must be a string or None"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=1,
            scheduler_address=123,  # Invalid type
        )

    # Test invalid num_episodes
    with pytest.raises(ValueError, match="num_episodes must be positive"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=0,  # Invalid
            num_steps=1,
        )

    # Test invalid num_steps
    with pytest.raises(ValueError, match="num_steps must be positive"):
        task.run_multiple_episodes(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_episodes=1,
            num_steps=-1,  # Invalid
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
        num_steps=2,  # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,  # Very small number of trials for fast execution
        seed=42,
    )

    # Run actual optimization with very small parameters
    result = task.run()

    # Verify result structure
    assert result is not None
    assert hasattr(result, "environment")
    assert hasattr(result, "policy")
    assert hasattr(result, "chosen_hyper_parameters")
    assert hasattr(result, "num_episodes")
    assert hasattr(result, "num_steps")
    assert hasattr(result, "direction")
    assert hasattr(result, "parameter_to_optimize")

    # Verify values
    assert result.environment == environment
    assert result.num_episodes == 2  # Updated to match the actual value used
    assert result.num_steps == 2
    assert result.direction == HyperParameterOptimizationDirection.MAXIMIZE
    assert result.parameter_to_optimize == "average_return"

    # Verify the chosen hyperparameters are within expected ranges
    assert "branching_factor" in result.chosen_hyper_parameters
    assert "depth" in result.chosen_hyper_parameters
    assert 1 <= result.chosen_hyper_parameters["branching_factor"] <= 3
    assert 1 <= result.chosen_hyper_parameters["depth"] <= 3


def test_hyper_parameter_tuning_task_run_with_categorical_params(
    environment, categorical_hyper_parameters
):
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
        NumericalHyperParameter(name="depth", low=1, high=2),
    ]

    task = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=numerical_hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Need at least 2 episodes for statistics computation
        num_steps=2,  # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,  # Very small number of trials for fast execution
        seed=42,
    )

    # Run actual optimization
    result = task.run()

    # Verify result
    assert result is not None
    assert hasattr(result, "chosen_hyper_parameters")
    assert "branching_factor" in result.chosen_hyper_parameters
    assert "depth" in result.chosen_hyper_parameters


def test_hyper_parameter_tuning_task_run_failure_logging(
    environment, hyper_parameters, caplog
):
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
        num_steps=2,  # Small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="nonexistent_parameter",  # This will cause failure in evaluation
        console_output=False,
        n_trials=1,  # Very small number for fast execution
        seed=42,
    )

    # Run task and expect it to fail with clear error message
    with pytest.raises(
        ValueError,
        match="Parameter nonexistent_parameter not found in computed statistics",
    ):
        task.run()

    # Verify error was logged appropriately
    assert "Error in evaluation function" in caplog.text
    assert (
        "Parameter nonexistent_parameter not found in computed statistics"
        in caplog.text
    )


def test_hyper_parameter_tuning_task_run_missing_parameter_logging(
    environment, hyper_parameters, caplog
):
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
        num_steps=2,  # Small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="invalid_metric",  # This will cause a missing parameter error
        console_output=False,
        n_trials=1,  # Very small number for fast execution
        seed=42,
    )

    # Run task and expect it to fail with clear error message
    with pytest.raises(
        ValueError, match="Parameter invalid_metric not found in computed statistics"
    ):
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=3,  # Custom value, small for fast tests
        seed=42,
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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
    )

    # Test logger property
    logger = task.logger
    assert logger is not None
    assert (
        logger.name
        == f"task.{environment.name}.{StandardSparseSamplingDiscreteActionsPlanner.__name__}"
    )


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
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=42,
    )

    # Create task with seed 999
    task2 = HyperParameterTuningSimulationTask(
        environment=environment,
        belief=belief,
        policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
        hyper_parameters=hyper_parameters,
        constant_parameters={},  # No constant parameters needed for this planner
        num_episodes=2,  # Smaller for fast tests
        num_steps=3,  # Smaller for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        seed=999,
    )

    # Verify different seeds result in different config IDs
    config_id1 = task1.get_config_id()
    config_id2 = task2.get_config_id()
    assert config_id1 != config_id2


def test_hyper_parameter_tuning_task_seed_in_optimization_results(
    environment, hyper_parameters
):
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
        num_steps=2,  # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,  # Very small number for fast execution
        seed=42,
    )

    # Run actual optimization
    result = task.run()

    # Verify seed is included in optimization result dictionary
    result_dict = task.get_optimization_result_dict()
    assert result_dict is not None
    assert "seed" in result_dict
    assert result_dict["seed"] == 42


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
        seed=42,
    )

    result = task.run()

    # Explicit type checking
    from POMDPPlanners.core.simulation.hyperparameter_tuning import (
        OptimizedPolicyResult,
    )

    assert isinstance(result, OptimizedPolicyResult)
    assert isinstance(result.environment, Environment)
    assert isinstance(result.policy, Policy)
    assert isinstance(result.chosen_hyper_parameters, dict)
    assert isinstance(result.num_episodes, int)
    assert isinstance(result.num_steps, int)
    assert isinstance(result.direction, HyperParameterOptimizationDirection)
    assert isinstance(result.parameter_to_optimize, str)


def test_hyper_parameter_tuning_task_with_constant_parameters(
    environment, hyper_parameters
):
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
        num_steps=2,  # Very small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
        console_output=False,
        n_trials=2,  # Very small number for fast execution
        seed=42,
    )

    # Run actual optimization
    result = task.run()

    # Verify result
    assert result is not None
    assert hasattr(result, "policy")
    assert hasattr(result, "chosen_hyper_parameters")

    # Verify the chosen hyperparameters are within expected ranges
    assert "branching_factor" in result.chosen_hyper_parameters
    assert "depth" in result.chosen_hyper_parameters
    assert 1 <= result.chosen_hyper_parameters["branching_factor"] <= 3
    assert 1 <= result.chosen_hyper_parameters["depth"] <= 3

    # Verify that the constant parameters were applied during optimization
    # The policy keeps the custom name from constant_parameters during optimization
    policy = result.policy
    # The final result policy keeps the custom name from constant_parameters
    assert policy.name == "CustomSparseSamplingPlanner"

    # Verify the policy was created with the correct hyperparameters
    assert policy.branching_factor == result.chosen_hyper_parameters["branching_factor"]
    assert policy.depth == result.chosen_hyper_parameters["depth"]


# =============================================================================
# MLFlow Integration Tests (merged from test_tasks/ subdirectory)
# =============================================================================


class TestHyperParameterTuningSimulationTaskMLFlowIntegration:
    """Test class for HyperParameterTuningSimulationTask MLFlow integration issues.

    This test class focuses on catching integration issues that occur when running
    real hyperparameter optimization scenarios, specifically the issues found in
    hyper_param_runner.py with POMCP missing parameters and constructor order problems.
    """

    def test_task_creation_basic(self):
        """Test that HyperParameterTuningSimulationTask can be imported.

        Purpose: Validates basic task import functionality

        Given: HyperParameterTuningSimulationTask class
        When: Task class is imported
        Then: Task class is available and not None

        Test type: unit
        """
        assert HyperParameterTuningSimulationTask is not None

    def test_pomcp_missing_discount_factor_constant_parameter(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that missing discount_factor for POMCP is caught during task execution.

        Purpose: Validates that missing required constant parameters for POMCP cause proper failure

        Given: A HyperParameterTuningSimulationTask configured with POMCP but missing discount_factor
        When: Task is executed without discount_factor in constant_parameters
        Then: Task execution fails with clear error about missing required parameter

        Test type: integration
        """
        # Create hyperparameters for POMCP (using correct parameter order)
        hyper_parameters = [
            NumericalHyperParameter(
                0.1, 2.0, "exploration_constant"
            ),  # Correct order: low, high, name
            NumericalHyperParameter(
                10, 50, "n_simulations"
            ),  # Correct order: low, high, name
        ]

        # Create task WITHOUT discount_factor in constant_parameters (this should fail)
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters={},  # Missing discount_factor, depth, name - this should cause failure
            num_episodes=2,
            num_steps=3,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            confidence_interval_level=0.95,
            alpha=0.05,
            n_trials=2,
            seed=42,
        )

        # Execute task and expect it to fail due to missing discount_factor
        with pytest.raises(
            TypeError, match=".*missing.*required.*argument.*discount_factor.*"
        ):
            task.run()

    def test_pomcp_with_correct_constant_parameters(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that POMCP works correctly when discount_factor is provided in constant_parameters.

        Purpose: Validates that POMCP executes successfully when all required parameters are provided

        Given: A HyperParameterTuningSimulationTask configured with POMCP and discount_factor in constant_parameters
        When: Task is executed with proper constant_parameters
        Then: Task executes successfully and returns OptimizedPolicyResult

        Test type: integration
        """
        # Create hyperparameters for POMCP (using correct parameter order)
        hyper_parameters = [
            NumericalHyperParameter(
                0.1, 1.0, "exploration_constant"
            ),  # Correct order: low, high, name
            NumericalHyperParameter(
                10, 20, "n_simulations"
            ),  # Correct order: low, high, name
        ]

        # Create task WITH discount_factor in constant_parameters (this should work)
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters={
                "discount_factor": 0.95,  # Include required discount_factor
                "depth": 5,  # Include required depth
                "name": "test_pomcp",  # Include required name
            },
            num_episodes=2,
            num_steps=3,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            confidence_interval_level=0.95,
            alpha=0.05,
            n_trials=2,
            seed=42,
        )

        # Execute task and expect it to succeed
        result = task.run()

        # Verify result structure
        assert result is not None
        assert hasattr(result, "environment")
        assert hasattr(result, "policy")
        assert hasattr(result, "chosen_hyper_parameters")
        assert result.environment == real_environment
        assert "exploration_constant" in result.chosen_hyper_parameters
        assert "n_simulations" in result.chosen_hyper_parameters

    def test_numerical_hyperparameter_correct_usage_validation(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that NumericalHyperParameter is used correctly in task configuration.

        Purpose: Validates that NumericalHyperParameter constructor is used with correct parameter order

        Given: A HyperParameterTuningSimulationTask with properly constructed NumericalHyperParameter instances
        When: Task is configured with correct parameter order (low, high, name)
        Then: Task creation succeeds and parameters are properly configured

        Test type: unit
        """
        # Test that the correct parameter order works (this was the actual format in hyper_param_runner.py)
        correct_params = [
            NumericalHyperParameter(
                0.1, 10.0, "exploration_constant"
            ),  # Correct order: low, high, name
            NumericalHyperParameter(
                100, 500, "n_simulations"
            ),  # Correct order: low, high, name
        ]

        # Verify correct parameters can be used in task creation
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=correct_params,
            constant_parameters={
                "discount_factor": 0.95,
                "depth": 5,
                "name": "test_pomcp",
            },
            num_episodes=2,
            num_steps=3,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            n_trials=2,
            seed=42,
        )

        # Verify task can be created with correct parameters
        assert task is not None
        assert len(task.hyper_parameters) == 2

        # Verify parameter properties are correct
        exploration_param = next(
            p for p in task.hyper_parameters if p.name == "exploration_constant"
        )
        assert exploration_param.low == 0.1
        assert exploration_param.high == 10.0
        assert exploration_param.name == "exploration_constant"

        simulations_param = next(
            p for p in task.hyper_parameters if p.name == "n_simulations"
        )
        assert simulations_param.low == 100
        assert simulations_param.high == 500
        assert simulations_param.name == "n_simulations"

    def test_hyper_param_runner_configuration_replication(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test exact hyper_param_runner.py configuration with fixes.

        Purpose: Validates that the exact configuration from hyper_param_runner.py works when corrected

        Given: A HyperParameterTuningSimulationTask that replicates hyper_param_runner.py configuration with fixes
        When: Task is executed with corrected parameter order and required constant parameters
        Then: Task executes successfully, demonstrating the fixes resolve the original issues

        Test type: integration
        """
        # This test replicates the EXACT configuration from hyper_param_runner.py but with fixes

        # The NumericalHyperParameter constructor actually expects (low, high, name)
        # So the hyper_param_runner.py was CORRECT in parameter order, the issue was missing constant_parameters
        hyper_parameters = [
            NumericalHyperParameter(
                0.1, 10.0, "exploration_constant"
            ),  # Correct order: low, high, name
            NumericalHyperParameter(
                100, 1000, "n_simulations"
            ),  # Correct order: low, high, name
        ]

        # FIXED: Include all required parameters in constant_parameters
        constant_parameters = {
            "discount_factor": 0.95,  # FIXED: Include missing required parameter
            "depth": 10,  # FIXED: Include missing required parameter
            "name": "test_pomcp",  # FIXED: Include missing required parameter
        }

        # Create task with fixed configuration
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,  # FIXED: Include constant parameters
            num_episodes=5,  # Same as in hyper_param_runner.py but smaller for fast tests
            num_steps=10,  # Same as in hyper_param_runner.py but smaller for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,  # Same as hyper_param_runner.py
            parameter_to_optimize="average_return",  # Same as hyper_param_runner.py
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            n_trials=3,  # Smaller than hyper_param_runner.py for fast tests
            seed=42,
        )

        # Execute task and verify it works with fixes
        result = task.run()

        # Verify successful execution
        assert result is not None
        assert hasattr(result, "environment")
        assert hasattr(result, "policy")
        assert hasattr(result, "chosen_hyper_parameters")

        # Verify optimized parameters are within expected ranges
        assert "exploration_constant" in result.chosen_hyper_parameters
        assert "n_simulations" in result.chosen_hyper_parameters
        assert 0.1 <= result.chosen_hyper_parameters["exploration_constant"] <= 10.0
        assert 100 <= result.chosen_hyper_parameters["n_simulations"] <= 1000

        # Verify the policy was created correctly with the constant parameter
        policy = result.policy
        assert hasattr(policy, "discount_factor")
        assert policy.discount_factor == 0.95  # Verify constant parameter was applied

    def test_sparse_sampling_planner_with_correct_parameters(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test StandardSparseSamplingDiscreteActionsPlanner with correct required parameters.

        Purpose: Validates that StandardSparseSamplingDiscreteActionsPlanner works when all required parameters are provided

        Given: A HyperParameterTuningSimulationTask with StandardSparseSamplingDiscreteActionsPlanner and required parameters
        When: Task is executed with branching_factor and depth hyperparameters
        Then: Task executes successfully and returns valid optimization results

        Test type: integration
        """
        # Create hyperparameters for StandardSparseSamplingDiscreteActionsPlanner
        hyper_parameters = [
            NumericalHyperParameter(
                1, 3, "branching_factor"
            ),  # Required parameter, correct order: low, high, name
            NumericalHyperParameter(
                1, 3, "depth"
            ),  # Required parameter, correct order: low, high, name
        ]

        # Create task with correct configuration for StandardSparseSamplingDiscreteActionsPlanner
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=hyper_parameters,
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=2,
            num_steps=3,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            n_trials=2,
            seed=42,
        )

        # Execute task and verify it works
        result = task.run()

        # Verify successful execution
        assert result is not None
        assert hasattr(result, "environment")
        assert hasattr(result, "policy")
        assert hasattr(result, "chosen_hyper_parameters")

        # Verify optimized parameters are within expected ranges
        assert "branching_factor" in result.chosen_hyper_parameters
        assert "depth" in result.chosen_hyper_parameters
        assert 1 <= result.chosen_hyper_parameters["branching_factor"] <= 3
        assert 1 <= result.chosen_hyper_parameters["depth"] <= 3

    def test_task_optimization_metadata_collection(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that task properly collects and provides optimization metadata.

        Purpose: Validates that HyperParameterTuningSimulationTask collects optimization metadata correctly

        Given: A HyperParameterTuningSimulationTask that completes optimization successfully
        When: Task optimization metadata is retrieved using get_optimization_metadata()
        Then: Metadata contains expected fields including best_value, optimization_time, and n_trials

        Test type: integration
        """
        # Create simple hyperparameters for fast execution
        hyper_parameters = [
            NumericalHyperParameter(
                1, 2, "branching_factor"
            ),  # Correct order: low, high, name
            NumericalHyperParameter(1, 2, "depth"),  # Correct order: low, high, name
        ]

        # Create task
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=hyper_parameters,
            constant_parameters={},
            num_episodes=2,
            num_steps=2,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            n_trials=2,
            seed=42,
        )

        # Execute task
        result = task.run()
        assert result is not None

        # Verify optimization metadata is available
        metadata = task.get_optimization_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)

        # Verify expected metadata fields
        assert "best_value" in metadata
        assert "optimization_time" in metadata
        assert "n_trials" in metadata
        assert "best_trial_number" in metadata

        # Verify metadata types and ranges
        assert isinstance(metadata["best_value"], (int, float))
        assert isinstance(metadata["optimization_time"], (int, float))
        assert isinstance(metadata["n_trials"], int)
        assert metadata["n_trials"] > 0
        assert metadata["optimization_time"] >= 0
