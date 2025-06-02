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


def test_simulator_initialization():
    """Test that the simulator initializes correctly."""
    # Test with default parameters
    simulator = POMDPSimulator()
    assert simulator.cache_dir_path is None
    assert simulator.experiment_name == "POMDP_Planning_Comparison"
    
    # Test with custom parameters
    cache_dir = Path("/tmp/test_cache")
    simulator = POMDPSimulator(
        cache_dir_path=cache_dir,
        experiment_name="Custom_Experiment",
        debug=True
    )
    assert simulator.cache_dir_path == cache_dir
    assert simulator.experiment_name == "Custom_Experiment"
    

def test_simulate_multiple_environments_and_policies_parallel(simulator):
    """Test parallel simulation of multiple environments and policies."""
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
    results = simulator.simulate_multiple_environments_and_policies_parallel(
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
    assert isinstance(results, dict)
    assert environment.name in results
    assert policy.name in results[environment.name]
    assert len(results[environment.name][policy.name]) == num_episodes

    # Check each history
    for history in results[environment.name][policy.name]:
        assert isinstance(history, History)
        assert len(history.history) == num_steps
        assert history.actual_num_steps == num_steps
        assert isinstance(history.reach_terminal_state, bool)


def test_compare_multiple_environments_policies(simulator):
    """Test comparison of multiple environments and policies."""
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
    """Test that parallel execution maintains statistical properties."""
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
    """Test that invalid number of jobs raises appropriate error."""
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


# def test_visualization_creation(simulator):
#     """Test that visualizations are created correctly."""
#     # Setup
#     environment = TigerPOMDP(discount_factor=0.95)
#     policy = POMCP(
#         environment=environment,
#         discount_factor=0.95,
#         depth=3,
#         exploration_constant=1.0,
#         name="TestPolicy",
#         n_simulations=2
#     )
#     initial_belief = get_initial_belief(environment, n_particles=3)
#     num_episodes = 2
#     num_steps = 3

#     # Execute comparison with visualizations
#     histories, _ = simulator.compare_multiple_environments_policies(
#         environment_run_params=[
#             EnvironmentRunParams(
#                 environment=environment,
#                 belief=initial_belief,
#                 policies=[policy],
#                 num_episodes=num_episodes,
#                 num_steps=num_steps
#             )
#         ],
#         alpha=0.1,
#         cache_visualizations=True,
#         n_jobs=1
#     )

#     # Check that visualization directories were created
#     results_dir = simulator.cache_dir_path / "results"
#     env_dir = results_dir / environment.name
#     policy_dir = env_dir / policy.name
    
#     assert results_dir.exists()
#     assert env_dir.exists()
#     assert policy_dir.exists()
    
#     # Check for plot files
#     plots_dir = policy_dir / "plots"
#     assert plots_dir.exists()
#     assert (plots_dir / "discounted_returns_histogram.png").exists()
    
#     # Check for comparison plot
#     assert (env_dir / "policy_comparison_histogram.png").exists()
    
#     # Check for visualization directory
#     viz_dir = policy_dir / "visualizations"
#     assert viz_dir.exists()


def test_organize_simulation_results_basic(simulator):
    """Test basic organization of simulation results with a single environment and policy."""
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
    """Test organization of simulation results with multiple environments and policies."""
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
    """Test edge cases for organizing simulation results."""
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
    """Test that histories are correctly matched to their environment-policy configurations."""
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
