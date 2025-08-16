import pytest
import tempfile
from pathlib import Path
from POMDPPlanners.simulations.simulations_deployment.tasks import HyperParameterTuningSimulationTask
from POMDPPlanners.core.simulation import NumericalHyperParameter, CategoricalHyperParameter
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParameterOptimizationDirection


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

    def test_pomcp_missing_discount_factor_constant_parameter(self, temp_cache_dir, real_environment, real_belief):
        """Test that missing discount_factor for POMCP is caught during task execution.
        
        Purpose: Validates that missing required constant parameters for POMCP cause proper failure
        
        Given: A HyperParameterTuningSimulationTask configured with POMCP but missing discount_factor
        When: Task is executed without discount_factor in constant_parameters
        Then: Task execution fails with clear error about missing required parameter
        
        Test type: integration
        """
        # Create hyperparameters for POMCP (using correct parameter order)
        hyper_parameters = [
            NumericalHyperParameter(0.1, 2.0, "exploration_constant"),  # Correct order: low, high, name
            NumericalHyperParameter(10, 50, "n_simulations")            # Correct order: low, high, name
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
            seed=42
        )
        
        # Execute task and expect it to fail due to missing discount_factor
        with pytest.raises(TypeError, match=".*missing.*required.*argument.*discount_factor.*"):
            task.run()

    def test_pomcp_with_correct_constant_parameters(self, temp_cache_dir, real_environment, real_belief):
        """Test that POMCP works correctly when discount_factor is provided in constant_parameters.
        
        Purpose: Validates that POMCP executes successfully when all required parameters are provided
        
        Given: A HyperParameterTuningSimulationTask configured with POMCP and discount_factor in constant_parameters
        When: Task is executed with proper constant_parameters
        Then: Task executes successfully and returns OptimizedPolicyResult
        
        Test type: integration
        """
        # Create hyperparameters for POMCP (using correct parameter order)
        hyper_parameters = [
            NumericalHyperParameter(0.1, 1.0, "exploration_constant"),  # Correct order: low, high, name
            NumericalHyperParameter(10, 20, "n_simulations")            # Correct order: low, high, name
        ]
        
        # Create task WITH discount_factor in constant_parameters (this should work)
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters={
                "discount_factor": 0.95,  # Include required discount_factor
                "depth": 5,               # Include required depth
                "name": "test_pomcp"      # Include required name
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
            seed=42
        )
        
        # Execute task and expect it to succeed
        result = task.run()
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, 'environment')
        assert hasattr(result, 'policy')
        assert hasattr(result, 'chosen_hyper_parameters')
        assert result.environment == real_environment
        assert 'exploration_constant' in result.chosen_hyper_parameters
        assert 'n_simulations' in result.chosen_hyper_parameters

    def test_numerical_hyperparameter_correct_usage_validation(self, temp_cache_dir, real_environment, real_belief):
        """Test that NumericalHyperParameter is used correctly in task configuration.
        
        Purpose: Validates that NumericalHyperParameter constructor is used with correct parameter order
        
        Given: A HyperParameterTuningSimulationTask with properly constructed NumericalHyperParameter instances
        When: Task is configured with correct parameter order (low, high, name)
        Then: Task creation succeeds and parameters are properly configured
        
        Test type: unit
        """
        # Test that the correct parameter order works (this was the actual format in hyper_param_runner.py)
        correct_params = [
            NumericalHyperParameter(0.1, 10.0, "exploration_constant"),  # Correct order: low, high, name
            NumericalHyperParameter(100, 500, "n_simulations")           # Correct order: low, high, name
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
                "name": "test_pomcp"
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
            seed=42
        )
        
        # Verify task can be created with correct parameters
        assert task is not None
        assert len(task.hyper_parameters) == 2
        
        # Verify parameter properties are correct
        exploration_param = next(p for p in task.hyper_parameters if p.name == "exploration_constant")
        assert exploration_param.low == 0.1
        assert exploration_param.high == 10.0
        assert exploration_param.name == "exploration_constant"
        
        simulations_param = next(p for p in task.hyper_parameters if p.name == "n_simulations")
        assert simulations_param.low == 100
        assert simulations_param.high == 500
        assert simulations_param.name == "n_simulations"

    def test_hyper_param_runner_configuration_replication(self, temp_cache_dir, real_environment, real_belief):
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
            NumericalHyperParameter(0.1, 10.0, "exploration_constant"),  # Correct order: low, high, name
            NumericalHyperParameter(100, 1000, "n_simulations")          # Correct order: low, high, name
        ]
        
        # FIXED: Include all required parameters in constant_parameters
        constant_parameters = {
            "discount_factor": 0.95,  # FIXED: Include missing required parameter
            "depth": 10,              # FIXED: Include missing required parameter
            "name": "test_pomcp"      # FIXED: Include missing required parameter
        }
        
        # Create task with fixed configuration
        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=real_belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,  # FIXED: Include constant parameters
            num_episodes=5,     # Same as in hyper_param_runner.py but smaller for fast tests
            num_steps=10,       # Same as in hyper_param_runner.py but smaller for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,  # Same as hyper_param_runner.py
            parameter_to_optimize="average_return",                   # Same as hyper_param_runner.py
            cache_dir=temp_cache_dir,
            debug=False,
            console_output=False,
            n_jobs=1,
            n_trials=3,  # Smaller than hyper_param_runner.py for fast tests
            seed=42
        )
        
        # Execute task and verify it works with fixes
        result = task.run()
        
        # Verify successful execution
        assert result is not None
        assert hasattr(result, 'environment')
        assert hasattr(result, 'policy')
        assert hasattr(result, 'chosen_hyper_parameters')
        
        # Verify optimized parameters are within expected ranges
        assert 'exploration_constant' in result.chosen_hyper_parameters
        assert 'n_simulations' in result.chosen_hyper_parameters
        assert 0.1 <= result.chosen_hyper_parameters['exploration_constant'] <= 10.0
        assert 100 <= result.chosen_hyper_parameters['n_simulations'] <= 1000
        
        # Verify the policy was created correctly with the constant parameter
        policy = result.policy
        assert hasattr(policy, 'discount_factor')
        assert policy.discount_factor == 0.95  # Verify constant parameter was applied

    def test_sparse_sampling_planner_with_correct_parameters(self, temp_cache_dir, real_environment, real_belief):
        """Test StandardSparseSamplingDiscreteActionsPlanner with correct required parameters.
        
        Purpose: Validates that StandardSparseSamplingDiscreteActionsPlanner works when all required parameters are provided
        
        Given: A HyperParameterTuningSimulationTask with StandardSparseSamplingDiscreteActionsPlanner and required parameters
        When: Task is executed with branching_factor and depth hyperparameters
        Then: Task executes successfully and returns valid optimization results
        
        Test type: integration
        """
        # Create hyperparameters for StandardSparseSamplingDiscreteActionsPlanner
        hyper_parameters = [
            NumericalHyperParameter(1, 3, "branching_factor"),  # Required parameter, correct order: low, high, name
            NumericalHyperParameter(1, 3, "depth")              # Required parameter, correct order: low, high, name
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
            seed=42
        )
        
        # Execute task and verify it works
        result = task.run()
        
        # Verify successful execution
        assert result is not None
        assert hasattr(result, 'environment')
        assert hasattr(result, 'policy')
        assert hasattr(result, 'chosen_hyper_parameters')
        
        # Verify optimized parameters are within expected ranges
        assert 'branching_factor' in result.chosen_hyper_parameters
        assert 'depth' in result.chosen_hyper_parameters
        assert 1 <= result.chosen_hyper_parameters['branching_factor'] <= 3
        assert 1 <= result.chosen_hyper_parameters['depth'] <= 3

    def test_task_optimization_metadata_collection(self, temp_cache_dir, real_environment, real_belief):
        """Test that task properly collects and provides optimization metadata.
        
        Purpose: Validates that HyperParameterTuningSimulationTask collects optimization metadata correctly
        
        Given: A HyperParameterTuningSimulationTask that completes optimization successfully
        When: Task optimization metadata is retrieved using get_optimization_metadata()
        Then: Metadata contains expected fields including best_value, optimization_time, and n_trials
        
        Test type: integration
        """
        # Create simple hyperparameters for fast execution
        hyper_parameters = [
            NumericalHyperParameter(1, 2, "branching_factor"),  # Correct order: low, high, name
            NumericalHyperParameter(1, 2, "depth")              # Correct order: low, high, name
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
            seed=42
        )
        
        # Execute task
        result = task.run()
        assert result is not None
        
        # Verify optimization metadata is available
        metadata = task.get_optimization_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)
        
        # Verify expected metadata fields
        assert 'best_value' in metadata
        assert 'optimization_time' in metadata
        assert 'n_trials' in metadata
        assert 'best_trial_number' in metadata
        
        # Verify metadata types and ranges
        assert isinstance(metadata['best_value'], (int, float))
        assert isinstance(metadata['optimization_time'], (int, float))
        assert isinstance(metadata['n_trials'], int)
        assert metadata['n_trials'] > 0
        assert metadata['optimization_time'] >= 0
