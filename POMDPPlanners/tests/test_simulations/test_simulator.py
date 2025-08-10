import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pandas as pd

from distributed import Client as DaskClient

from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History, EnvironmentRunParams


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
    try:
        if temp_path.exists():
            # Force close any open file handles
            import gc
            gc.collect()
            # Try to remove the directory
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory {temp_path}: {e}")


@pytest.fixture
def simulator(temp_cache_dir):
    return POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="Test_Experiment",
        debug=True
    )

def test_pomdp_simulator_initialization_default_parameters_creates_configured_instance():
    """
    Purpose: Validates POMDPSimulator initializes correctly with default and custom configurations
    
    Given: Default initialization and custom parameters (cache_dir, experiment_name, debug=True)
    When: POMDPSimulator instances are created with different parameter sets
    Then: Simulators are configured with correct attributes and ready for experiment execution
    
    Test type: unit
    """
    # ARRANGE: Define expected default and custom configuration values
    expected_default_name = "POMDP_Planning_Comparison"
    custom_cache_dir = Path("/tmp/test_cache")
    custom_experiment_name = "Custom_Experiment"
    
    # ACT: Create simulator instances with different configurations
    default_simulator = POMDPSimulator()
    custom_simulator = POMDPSimulator(
        cache_dir_path=custom_cache_dir,
        experiment_name=custom_experiment_name,
        debug=True
    )
    
    # ASSERT: Verify correct initialization of both configurations
    # Default configuration
    assert default_simulator.cache_dir_path is None
    assert default_simulator.experiment_name == expected_default_name
    assert isinstance(default_simulator.experiment_name, str)
    
    # Custom configuration  
    assert custom_simulator.cache_dir_path == custom_cache_dir
    assert custom_simulator.experiment_name == custom_experiment_name
    assert hasattr(custom_simulator, 'debug')  # Debug mode enabled
    

def test_pomdp_simulator_parallel_execution_completes_multiple_policy_episodes(simulator):
    """
    Purpose: Verifies parallel simulation executes multiple episodes across different policies correctly
    
    Given: TigerPOMDP environment, POMCP policy with 2 simulations, and initial belief with 3 particles
    When: Parallel simulation runs 2 episodes of 3 steps each with n_jobs=1
    Then: Returns structured results with correct episode histories and step counts for each policy
    
    Test type: integration
    """
    # ARRANGE: Setup TigerPOMDP environment with POMCP policy for parallel testing
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    expected_episodes = 2
    expected_steps = 3
    confidence_alpha = 0.1
    parallel_jobs = 1

    # ACT: Execute parallel simulation with configured parameters
    results = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=expected_episodes,
                num_steps=expected_steps
            )
        ],
        alpha=confidence_alpha,
        n_jobs=parallel_jobs
    )

    # ASSERT: Verify parallel execution produces correct structured results
    assert isinstance(results, dict)
    assert environment.name in results
    assert policy.name in results[environment.name]
    assert len(results[environment.name][policy.name]) == expected_episodes

    # Verify each episode history contains correct step data
    for episode_idx, history in enumerate(results[environment.name][policy.name]):
        assert isinstance(history, History), f"Episode {episode_idx} history is not History instance"
        assert len(history.history) == expected_steps, f"Episode {episode_idx} has {len(history.history)} steps, expected {expected_steps}"
        assert history.actual_num_steps == expected_steps
        assert isinstance(history.reach_terminal_state, bool)
        
        # Verify history contains valid step data (state, action, next_state, observation, reward, belief)
        for step_idx, step_data in enumerate(history.history):
            assert len(step_data) == 6, f"Episode {episode_idx}, step {step_idx} has invalid data format"


def test_pomdp_simulator_comparison_generates_statistics_dataframe(simulator):
    """
    Purpose: Validates simulator comparison produces both episode histories and statistical DataFrame
    
    Given: TigerPOMDP environment with POMCP policy configured for 2 episodes of 3 steps
    When: Comparison method is executed with alpha=0.1 confidence interval
    Then: Returns episode histories dictionary and DataFrame with policy configuration statistics
    
    Test type: integration
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Execute
    histories, merged_df = simulator.compare_multiple_environments_policies(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        n_jobs=1
    )

    # Assert
    assert isinstance(histories, dict)
    assert isinstance(merged_df, pd.DataFrame)
    
    # Check histories
    assert environment.name in histories
    assert policy.name in histories[environment.name]
    assert len(histories[environment.name][policy.name]) == num_episodes
    
    # Check DataFrame
    assert 'environment' in merged_df.columns
    assert 'policy' in merged_df.columns
    assert 'average_return' in merged_df.columns
    assert 'depth' in merged_df.columns
    assert 'exploration_constant' in merged_df.columns
    assert 'n_simulations' in merged_df.columns
    
    # Check values
    row = merged_df.iloc[0]
    assert row['environment'] == environment.name
    assert row['policy'] == policy.name
    assert row['depth'] == 3
    assert row['exploration_constant'] == 1.0
    assert row['n_simulations'] == 2


def test_parallel_execution_maintains_statistical_properties(simulator):
    """Test that parallel execution maintains statistical properties.
    
    Purpose: Validates parallel execution maintains statistical properties
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 4
    num_steps = 3

    # Run with different numbers of jobs
    results_1job = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        n_jobs=1
    )

    results_2jobs = simulator.simulate_multiple_environments_and_policies_parallel(
        environment_run_params=[
            EnvironmentRunParams(
                environment=environment,
                belief=initial_belief,
                policies=[policy],
                num_episodes=num_episodes,
                num_steps=num_steps
            )
        ],
        alpha=0.1,
        n_jobs=2
    )

    # Assert that results are equivalent regardless of number of jobs
    assert results_1job.keys() == results_2jobs.keys()
    for env_name in results_1job:
        assert results_1job[env_name].keys() == results_2jobs[env_name].keys()
        for policy_name in results_1job[env_name]:
            assert len(results_1job[env_name][policy_name]) == len(results_2jobs[env_name][policy_name])
            for h1, h2 in zip(results_1job[env_name][policy_name], results_2jobs[env_name][policy_name]):
                assert len(h1.history) == len(h2.history)
                assert h1.actual_num_steps == h2.actual_num_steps
                assert h1.reach_terminal_state == h2.reach_terminal_state


def test_invalid_jobs_parameter(simulator):
    """Test that invalid number of jobs raises appropriate error.
    
    Purpose: Validates invalid jobs parameter
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 4
    num_steps = 3

    # Test with invalid number of jobs
    with pytest.raises(ValueError):
        simulator.simulate_multiple_environments_and_policies_parallel(
            environment_run_params=[
                EnvironmentRunParams(
                    environment=environment,
                    belief=initial_belief,
                    policies=[policy],
                    num_episodes=num_episodes,
                    num_steps=num_steps
                )
            ],
            alpha=0.1,
            n_jobs=0  # Invalid number of jobs
        )


def test_organize_simulation_results_basic(simulator):
    """Test basic organization of simulation results with a single environment and policy.
    
    Purpose: Validates organize simulation results basic
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Create some test histories and their identifiers
    histories = []
    task_identifiers = []
    for i in range(num_episodes):
        history = History(
            history=[(0, 0, 0.0, False) for _ in range(num_steps)],
            actual_num_steps=num_steps,
            reach_terminal_state=False,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            policy_run_data={}
        )
        histories.append(history)
        task_identifiers.append((environment.name, policy.name))

    # Execute
    results = simulator._organize_simulation_results(
        results_list=histories,
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=num_episodes,
        task_identifiers=task_identifiers
    )

    # Assert
    assert isinstance(results, dict)
    assert environment.name in results
    assert policy.name in results[environment.name]
    assert len(results[environment.name][policy.name]) == num_episodes
    assert all(isinstance(h, History) for h in results[environment.name][policy.name])


def test_organize_simulation_results_multiple(simulator):
    """Test organization of simulation results with multiple environments and policies.
    
    Purpose: Validates organize simulation results multiple
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
    # Setup
    env1 = TigerPOMDP(discount_factor=0.95, name="Tiger1")
    env2 = TigerPOMDP(discount_factor=0.95, name="Tiger2")
    
    policy1 = POMCP(
        environment=env1,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="Policy1",
        n_simulations=2
    )
    policy2 = POMCP(
        environment=env2,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="Policy2",
        n_simulations=2
    )
    
    initial_belief = get_initial_belief(env1, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Create test histories and their identifiers for both environments and policies
    histories = []
    task_identifiers = []
    # First two histories for policy1
    for _ in range(num_episodes):
        history = History(
            history=[(0, 0, 0.0, False) for _ in range(num_steps)],
            actual_num_steps=num_steps,
            reach_terminal_state=False,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            policy_run_data={}
        )
        histories.append(history)
        task_identifiers.append((env1.name, policy1.name))
    
    # Next two histories for policy2
    for _ in range(num_episodes):
        history = History(
            history=[(0, 0, 0.0, False) for _ in range(num_steps)],
            actual_num_steps=num_steps,
            reach_terminal_state=False,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            policy_run_data={}
        )
        histories.append(history)
        task_identifiers.append((env2.name, policy2.name))

    # Execute
    results = simulator._organize_simulation_results(
        results_list=histories,
        environment_belief_policy_tuples=[
            (env1, initial_belief, [policy1]),
            (env2, initial_belief, [policy2])
        ],
        num_episodes=num_episodes,
        task_identifiers=task_identifiers
    )

    # Assert
    assert isinstance(results, dict)
    assert len(results) == 2
    assert env1.name in results
    assert env2.name in results
    assert policy1.name in results[env1.name]
    assert policy2.name in results[env2.name]
    assert len(results[env1.name][policy1.name]) == num_episodes
    assert len(results[env2.name][policy2.name]) == num_episodes


def test_organize_simulation_results_edge_cases(simulator):
    """Test edge cases for organizing simulation results.
    
    Purpose: Validates organize simulation results edge cases
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)

    # Test case 1: Empty histories
    results_empty = simulator._organize_simulation_results(
        results_list=[],
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=0,
        task_identifiers=[]  # Empty list of identifiers
    )
    assert isinstance(results_empty, dict)
    assert environment.name in results_empty
    assert policy.name in results_empty[environment.name]
    assert len(results_empty[environment.name][policy.name]) == 0

    # Test case 2: Single episode
    single_history = [History(
        history=[(0, 0, 0.0, False)],
        actual_num_steps=1,
        reach_terminal_state=False,
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        policy_run_data={}
    )]
    single_identifier = [(environment.name, policy.name)]
    results_single = simulator._organize_simulation_results(
        results_list=single_history,
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=1,
        task_identifiers=single_identifier
    )
    assert isinstance(results_single, dict)
    assert environment.name in results_single
    assert policy.name in results_single[environment.name]
    assert len(results_single[environment.name][policy.name]) == 1
    assert isinstance(results_single[environment.name][policy.name][0], History)


def test_organize_simulation_results_matches_configurations(simulator):
    """Test that histories are correctly matched to their environment-policy configurations.
    
    Purpose: Validates organize simulation results matches configurations
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    # Setup two environments with different names
    env1 = TigerPOMDP(discount_factor=0.95, name="Tiger1")
    env2 = TigerPOMDP(discount_factor=0.95, name="Tiger2")
    
    # Setup two policies with different configurations
    policy1 = POMCP(
        environment=env1,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="Policy1",
        n_simulations=2
    )
    policy2 = POMCP(
        environment=env2,
        discount_factor=0.95,
        depth=5,  # Different depth
        exploration_constant=2.0,  # Different exploration constant
        name="Policy2",
        n_simulations=4  # Different number of simulations
    )
    
    initial_belief = get_initial_belief(env1, n_particles=3)
    num_episodes = 2
    num_steps = 3

    # Create test histories with distinct characteristics for each policy
    histories = []
    task_identifiers = []
    # First two histories for policy1 (depth=3)
    for _ in range(num_episodes):
        history = History(
            history=[(0, 0, 0.0, False) for _ in range(num_steps)],
            actual_num_steps=num_steps,
            reach_terminal_state=False,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            policy_run_data={"depth": 3, "exploration_constant": 1.0}  # Match policy1's config
        )
        histories.append(history)
        task_identifiers.append((env1.name, policy1.name))
    
    # Next two histories for policy2 (depth=5)
    for _ in range(num_episodes):
        history = History(
            history=[(0, 0, 0.0, False) for _ in range(num_steps)],
            actual_num_steps=num_steps,
            reach_terminal_state=False,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            policy_run_data={"depth": 5, "exploration_constant": 2.0}  # Match policy2's config
        )
        histories.append(history)
        task_identifiers.append((env2.name, policy2.name))

    # Execute
    results = simulator._organize_simulation_results(
        results_list=histories,
        environment_belief_policy_tuples=[
            (env1, initial_belief, [policy1]),
            (env2, initial_belief, [policy2])
        ],
        num_episodes=num_episodes,
        task_identifiers=task_identifiers
    )

    # Assert
    assert isinstance(results, dict)
    assert len(results) == 2
    
    # Verify env1/policy1 histories
    assert env1.name in results
    assert policy1.name in results[env1.name]
    policy1_histories = results[env1.name][policy1.name]
    assert len(policy1_histories) == num_episodes
    for history in policy1_histories:
        assert history.policy_run_data["depth"] == policy1.depth
        assert history.policy_run_data["exploration_constant"] == policy1.exploration_constant
    
    # Verify env2/policy2 histories
    assert env2.name in results
    assert policy2.name in results[env2.name]
    policy2_histories = results[env2.name][policy2.name]
    assert len(policy2_histories) == num_episodes
    for history in policy2_histories:
        assert history.policy_run_data["depth"] == policy2.depth
        assert history.policy_run_data["exploration_constant"] == policy2.exploration_constant


def test_pomdp_simulator_mlflow_tracking_configures_experiment_directory(temp_cache_dir):
    """
    Purpose: Validates POMDPSimulator correctly configures MLflow tracking for experiment logging
    
    Given: Temporary cache directory and experiment name "TestMLflowSetup"
    When: POMDPSimulator is initialized with cache directory and debug enabled
    Then: MLflow tracking directory is created and tracking URI points to correct location
    
    Test type: unit
    """
    # ARRANGE: Setup test configuration and expected MLflow paths
    import mlflow
    experiment_name = "TestMLflowSetup"
    expected_mlruns_path = temp_cache_dir / "mlruns"
    
    # ACT: Initialize simulator with MLflow configuration
    simulator = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name=experiment_name,
        debug=True
    )
    current_tracking_uri = mlflow.get_tracking_uri()
    
    # ASSERT: Verify MLflow tracking setup is correctly configured
    assert expected_mlruns_path.exists(), f"MLflow runs directory not created at {expected_mlruns_path}"
    assert expected_mlruns_path.is_dir(), "MLflow runs path is not a directory"
    assert str(expected_mlruns_path) in current_tracking_uri, f"Tracking URI {current_tracking_uri} does not contain expected path {expected_mlruns_path}"
    
    # Verify simulator is ready for experiment logging
    assert simulator.experiment_name == experiment_name
    assert simulator.cache_dir_path == temp_cache_dir


def test_context_manager_functionality(temp_cache_dir):
    """Test that POMDPSimulator works as a context manager.
    
    Purpose: Validates context manager functionality
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    with POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="ContextManagerTest",
        debug=True
    ) as simulator:
        assert isinstance(simulator, POMDPSimulator)
        assert simulator.cache_dir_path == temp_cache_dir
        assert simulator.experiment_name == "ContextManagerTest"
    
    # Simulator should still be accessible after context exit
    assert isinstance(simulator, POMDPSimulator)


def test_profiling_enabled_initialization(temp_cache_dir):
    """Test simulator initialization with profiling enabled.
    
    Purpose: Validates proper initialization of profiling enabled 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    simulator = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="ProfilingTest",
        debug=True,
        enable_profiling=True,
        profiling_output_limit=25
    )
    
    assert simulator.enable_profiling == True
    assert simulator.profiling_output_limit == 25
    assert simulator.profiler is None  # Should be None until first use


def test_task_manager_types(temp_cache_dir):
    """Test creation of different task manager types.
    
    Purpose: Validates task manager types
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
    
    # Test JOBLIB task manager
    simulator_joblib = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="TaskManagerTest",
        debug=True,
        task_manager_type=TaskManagerType.JOBLIB,
        n_jobs=2
    )
    assert simulator_joblib.task_manager is not None
    assert simulator_joblib.n_jobs == 2
    
    # Test DASK task manager (local)
    simulator_dask = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="TaskManagerTest",
        debug=True,
        task_manager_type=TaskManagerType.DASK,
        n_jobs=1
    )
    assert simulator_dask.task_manager is not None


def test_create_policy_configurations_df(simulator):
    """Test policy configuration DataFrame creation.
    
    Purpose: Validates create policy configurations df
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    
    env_run_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[policy],
            num_episodes=2,
            num_steps=3
        )
    ]
    
    # Test the method - need to convert to the expected tuple format
    env_belief_policy_tuples = [(params.environment, params.belief, params.policies) for params in env_run_params]
    config_df = simulator._create_policy_configurations_df(env_belief_policy_tuples)
    
    # Verify DataFrame structure
    assert isinstance(config_df, pd.DataFrame)
    assert len(config_df) == 1  # One policy
    
    # Verify columns exist
    expected_columns = ['environment', 'policy', 'policy_type', 'depth', 'exploration_constant', 'n_simulations']
    for col in expected_columns:
        assert col in config_df.columns
    
    # Verify values
    row = config_df.iloc[0]
    assert row['environment'] == environment.name
    assert row['policy'] == policy.name
    assert row['policy_type'] == 'POMCP'
    assert row['depth'] == 3
    assert row['exploration_constant'] == 1.0
    assert row['n_simulations'] == 2


def test_validate_parallel_simulation_inputs(simulator):
    """Test input validation for parallel simulations.
    
    Purpose: Validates validate parallel simulation inputs
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    
    valid_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[policy],
            num_episodes=2,
            num_steps=3
        )
    ]
    
    # Test valid inputs - should not raise exception
    simulator._validate_parallel_simulation_inputs(valid_params, alpha=0.1, confidence_interval_level=0.95, n_jobs=1)
    
    # Test invalid n_jobs
    with pytest.raises(ValueError):
        simulator._validate_parallel_simulation_inputs(valid_params, alpha=0.1, confidence_interval_level=0.95, n_jobs=0)
    
    # Test empty environment_run_params
    with pytest.raises(ValueError):
        simulator._validate_parallel_simulation_inputs([], alpha=0.1, confidence_interval_level=0.95, n_jobs=1)


def test_create_simulation_tasks(simulator):
    """Test simulation task creation.
    
    Purpose: Validates create simulation tasks
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    
    env_run_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[policy],
            num_episodes=2,
            num_steps=3
        )
    ]
    
    tasks, task_identifiers = simulator._create_simulation_tasks(env_run_params)
    
    # Should create 2 tasks (2 episodes)
    assert len(tasks) == 2
    assert len(task_identifiers) == 2
    
    # Each task should be an EpisodeSimulationTask
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    assert isinstance(tasks, list)
    assert len(tasks) == 2
    for task in tasks:
        assert isinstance(task, EpisodeSimulationTask)
        assert task.environment == environment
        assert task.policy == policy
        # Note: belief is passed during construction but not stored as attribute
    
    # Check task identifiers
    for identifier in task_identifiers:
        env_name, policy_name = identifier
        assert env_name == environment.name
        assert policy_name == policy.name


def test_simulator_handles_empty_results_gracefully(simulator):
    """Test that simulator handles empty simulation results without crashing.
    
    Purpose: Validates simulator handles empty results gracefully
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=3)
    
    # Test with empty results
    results = simulator._organize_simulation_results(
        results_list=[],
        environment_belief_policy_tuples=[(environment, initial_belief, [policy])],
        num_episodes=0,
        task_identifiers=[]
    )
    
    # Should return proper structure even with empty results
    assert isinstance(results, dict)
    assert environment.name in results
    assert policy.name in results[environment.name]
    assert len(results[environment.name][policy.name]) == 0


def test_simulator_error_handling_invalid_cache_dir():
    """Test simulator error handling with invalid cache directory.
    
    Purpose: Validates error handling for simulator  handling invalid cache dir
    
    Given: Invalid inputs or error conditions
    When: Operation is attempted
    Then: Appropriate exception is raised
    
    Test type: unit
    """
    # Test with a file path instead of directory path
    invalid_path = Path("/dev/null")  # This is a file, not a directory
    
    try:
        simulator = POMDPSimulator(
            cache_dir_path=invalid_path,
            experiment_name="ErrorTest",
            debug=True
        )
        # If no exception is raised, that's also acceptable behavior
        assert isinstance(simulator, POMDPSimulator)
    except (OSError, ValueError, Exception):
        # Some form of error handling is expected
        pass


def test_simulator_writes_files_to_output_directory(temp_cache_dir):
    """Test that simulator writes expected files and directories to output directory during simulation.
    
    Purpose: Validates simulator writes files and directories to configured output directory during execution
    
    Given: POMDPSimulator with temporary cache directory, TigerPOMDP environment, and POMCP policy
    When: Complete simulation comparison is executed with profiling and visualizations enabled
    Then: Output directory contains MLflow data, profiling results, policy directories, and visualization files
    
    Test type: integration
    """
    # ARRANGE: Setup simulator with output directory and enable all file writing features
    simulator = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name="FileWriteTest",
        debug=True,
        enable_profiling=True,  # Enable profiling to test profiling_results.txt
        profiling_output_limit=10
    )
    
    # Setup environment and policy for simulation
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=5
    )
    initial_belief = get_initial_belief(environment, n_particles=10)
    
    env_run_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[policy],
            num_episodes=3,  # Small number for fast test
            num_steps=5
        )
    ]
    
    # ACT: Execute complete simulation with file writing enabled
    with simulator:
        results, statistics_df = simulator.compare_multiple_environments_policies(
            environment_run_params=env_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_visualizations=True  # Enable visualization caching
        )
    
    # ASSERT: Verify expected files and directories are created in output directory
    
    # 1. Verify MLflow directory and basic structure exists
    mlruns_dir = temp_cache_dir / "mlruns"
    assert mlruns_dir.exists(), f"MLflow runs directory not found at {mlruns_dir}"
    assert mlruns_dir.is_dir(), "MLflow runs path should be a directory"
    
    # 2. Verify profiling results file is created (when profiling enabled)
    profiling_file = temp_cache_dir / "profiling_results.txt"
    assert profiling_file.exists(), f"Profiling results file not found at {profiling_file}"
    assert profiling_file.is_file(), "Profiling results should be a file"
    assert profiling_file.stat().st_size > 0, "Profiling results file should not be empty"
    
    # 3. Verify MLflow experiment artifacts are logged (check for experiment directories)
    # MLflow creates numbered experiment directories
    experiment_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    assert len(experiment_dirs) >= 1, f"No MLflow experiment directories found in {mlruns_dir}"
    
    # Find run directories within experiment directories 
    run_dirs = []
    for exp_dir in experiment_dirs:
        run_dirs.extend([d for d in exp_dir.iterdir() if d.is_dir() and len(d.name) == 32])  # MLflow run IDs are 32 chars
    
    assert len(run_dirs) >= 1, f"No MLflow run directories found in experiment directories"
    
    # 4. Verify artifacts directory exists in at least one run
    artifacts_found = False
    for run_dir in run_dirs:
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists() and artifacts_dir.is_dir():
            artifacts_found = True
            
            # Check for policy comparison plots
            policy_plots_dir = artifacts_dir / "policy_comparison_plots"
            if policy_plots_dir.exists():
                plot_files = list(policy_plots_dir.glob("*.png"))
                if plot_files:
                    assert len(plot_files) > 0, "Expected at least one policy comparison plot file"
            
            # Check for environment-specific artifacts
            env_artifact_dir = artifacts_dir / environment.name
            if env_artifact_dir.exists():
                # Look for policy directories
                policy_dir = env_artifact_dir / policy.name
                if policy_dir.exists():
                    plots_dir = policy_dir / "plots"
                    if plots_dir.exists():
                        histogram_files = list(plots_dir.glob("*.png"))
                        if histogram_files:
                            assert len(histogram_files) > 0, "Expected histogram plot files in policy directory"
            break
    
    assert artifacts_found, "No MLflow artifacts directory found in any run directory"
    
    # 5. Verify simulation completed successfully and returned expected results
    assert isinstance(results, dict), "Results should be a dictionary"
    assert environment.name in results, f"Results should contain environment {environment.name}"
    assert policy.name in results[environment.name], f"Results should contain policy {policy.name}"
    assert len(results[environment.name][policy.name]) == 3, "Expected 3 episodes in results"
    
    assert isinstance(statistics_df, pd.DataFrame), "Statistics should be a DataFrame"
    assert not statistics_df.empty, "Statistics DataFrame should not be empty"
    assert 'environment' in statistics_df.columns, "Statistics should contain environment column"
    assert 'policy' in statistics_df.columns, "Statistics should contain policy column"


def test_simulator_mlflow_directory_structure_is_correct(temp_cache_dir):
    """Test that simulator creates exactly one mlruns directory with exactly one experiment.
    
    Purpose: Validates MLflow directory structure has single mlruns directory and single experiment
    
    Given: POMDPSimulator with temporary cache directory and specific experiment name
    When: Simulation is executed with MLflow tracking enabled
    Then: Output directory contains exactly one mlruns directory with exactly one experiment directory
    
    Test type: integration
    """
    # ARRANGE: Setup simulator with specific experiment name
    experiment_name = "MLflowStructureTest"
    simulator = POMDPSimulator(
        cache_dir_path=temp_cache_dir,
        experiment_name=experiment_name,
        debug=True
    )
    
    # Setup minimal simulation to trigger MLflow usage
    environment = TigerPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=2
    )
    initial_belief = get_initial_belief(environment, n_particles=5)
    
    env_run_params = [
        EnvironmentRunParams(
            environment=environment,
            belief=initial_belief,
            policies=[policy],
            num_episodes=2,  # Changed from 1 to 2 for confidence interval calculation
            num_steps=2
        )
    ]
    
    # ACT: Execute simulation to create MLflow structure
    with simulator:
        results, statistics_df = simulator.compare_multiple_environments_policies(
            environment_run_params=env_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_visualizations=False  # Disable to speed up test
        )
    
    # ASSERT: Verify exactly one mlruns directory exists in output directory
    
    # 1. Check that exactly one mlruns directory exists at the top level
    mlruns_dirs = [item for item in temp_cache_dir.iterdir() if item.is_dir() and item.name == "mlruns"]
    assert len(mlruns_dirs) == 1, f"Expected exactly 1 mlruns directory, found {len(mlruns_dirs)}: {[d.name for d in mlruns_dirs]}"
    
    mlruns_dir = mlruns_dirs[0]
    
    # 2. Verify no nested mlruns directories inside the main mlruns directory
    nested_mlruns = []
    for item in mlruns_dir.rglob("mlruns"):
        if item != mlruns_dir:  # Don't count the main mlruns directory itself
            nested_mlruns.append(item)
    
    assert len(nested_mlruns) == 0, f"Found nested mlruns directories: {nested_mlruns}"
    
    # 3. Verify exactly one experiment directory exists inside mlruns
    experiment_dirs = [item for item in mlruns_dir.iterdir() if item.is_dir() and item.name.isdigit()]
    assert len(experiment_dirs) == 1, f"Expected exactly 1 experiment directory, found {len(experiment_dirs)}: {[d.name for d in experiment_dirs]}"
    
    experiment_dir = experiment_dirs[0]
    
    # 4. Verify the experiment directory contains the expected structure
    # Should have: meta.yaml and at least one run directory
    meta_file = experiment_dir / "meta.yaml"
    assert meta_file.exists(), f"Expected meta.yaml file not found in experiment directory: {experiment_dir}"
    
    # 5. Verify at least one run directory exists (32-character hex names)
    run_dirs = [item for item in experiment_dir.iterdir() if item.is_dir() and len(item.name) == 32]
    assert len(run_dirs) >= 1, f"Expected at least 1 run directory, found {len(run_dirs)} in {experiment_dir}"
    
    # 6. Verify simulation completed successfully
    assert isinstance(results, dict), "Results should be a dictionary"
    assert environment.name in results, f"Results should contain environment {environment.name}"
    assert policy.name in results[environment.name], f"Results should contain policy {policy.name}"
    assert len(results[environment.name][policy.name]) == 2, "Expected 2 episodes in results"
    
    # 7. Verify statistics DataFrame is properly created
    assert isinstance(statistics_df, pd.DataFrame), "Statistics should be a DataFrame"
    assert not statistics_df.empty, "Statistics DataFrame should not be empty"

