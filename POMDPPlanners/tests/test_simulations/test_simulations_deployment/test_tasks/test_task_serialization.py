"""Tests for POMDP simulation task serialization.

This module tests that all simulation task classes can be properly serialized
and deserialized using pickle. Serialization is crucial for:
- Distributed computing with Dask/Ray
- Task queuing and scheduling
- Checkpointing long-running optimizations
- Parallel execution across multiple processes
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.environments import TigerPOMDP
from POMDPPlanners.planners import POMCP, SparsePFT
from POMDPPlanners.simulations.simulations_deployment.tasks import (
    EpisodeSimulationTask,
    HyperParameterTuningSimulationTask,
)

# Set seeds for reproducible tests
np.random.seed(42)


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)


class TestEpisodeSimulationTaskSerialization:
    """Test cases for EpisodeSimulationTask serialization using pickle."""

    def setup_method(self):
        """Set up test environment and policy for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.policy = SparsePFT(
            environment=self.env,
            discount_factor=0.95,
            gamma=0.95,
            depth=3,
            c_ucb=1.0,
            beta_ucb=0.5,
            belief_child_num=4,
            n_simulations=2,
        )
        self.belief = create_test_belief()

    def test_episode_simulation_task_serialization(self):
        """Test EpisodeSimulationTask serialization.

        Purpose: Validates that EpisodeSimulationTask can be pickled and unpickled

        Given: EpisodeSimulationTask instance with environment, policy, and belief
        When: Task is pickled and unpickled
        Then: Unpickled task maintains all properties and functionality

        Test type: unit
        """
        task = EpisodeSimulationTask(
            environment=self.env,
            policy=self.policy,
            initial_belief=self.belief,
            num_steps=5,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Pickle the task
        pickled = pickle.dumps(task)

        # Unpickle the task
        unpickled_task = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_task.num_steps == task.num_steps
        assert unpickled_task.episode_id == task.episode_id
        assert unpickled_task.seed == task.seed
        assert unpickled_task.discount_factor == task.discount_factor
        assert unpickled_task.episode_number == task.episode_number
        assert unpickled_task.console_output == task.console_output

        # Verify environment and policy are preserved
        assert unpickled_task.environment.name == task.environment.name
        assert unpickled_task.policy.name == task.policy.name

        # Verify cache key is preserved
        assert unpickled_task._cache_key == task._cache_key

    def test_episode_simulation_task_serialization_with_cache_dir(self):
        """Test EpisodeSimulationTask serialization with cache directory.

        Purpose: Validates that EpisodeSimulationTask with cache_dir can be pickled

        Given: EpisodeSimulationTask instance with cache_dir specified
        When: Task is pickled and unpickled
        Then: Cache directory path is preserved correctly

        Test type: unit
        """
        cache_dir = Path("/tmp/test_cache")

        task = EpisodeSimulationTask(
            environment=self.env,
            policy=self.policy,
            initial_belief=self.belief,
            num_steps=5,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert unpickled_task.cache_dir == cache_dir

    def test_episode_simulation_task_serialization_with_debug(self):
        """Test EpisodeSimulationTask serialization with debug enabled.

        Purpose: Validates that EpisodeSimulationTask with debug=True can be pickled

        Given: EpisodeSimulationTask instance with debug=True
        When: Task is pickled and unpickled
        Then: Debug flag is preserved correctly

        Test type: unit
        """
        task = EpisodeSimulationTask(
            environment=self.env,
            policy=self.policy,
            initial_belief=self.belief,
            num_steps=5,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            debug=True,
            console_output=False,
        )

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert unpickled_task.debug == task.debug

    def test_episode_simulation_task_multiple_serialization(self):
        """Test serialization of multiple EpisodeSimulationTask instances.

        Purpose: Validates that multiple tasks can be pickled together

        Given: List of multiple EpisodeSimulationTask instances with different parameters
        When: List is pickled and unpickled
        Then: All tasks are correctly restored with their respective properties

        Test type: integration
        """
        tasks = [
            EpisodeSimulationTask(
                environment=self.env,
                policy=self.policy,
                initial_belief=self.belief,
                num_steps=5,
                episode_id=i,
                seed=42 + i,
                discount_factor=0.95,
                episode_number=i,
                console_output=False,
            )
            for i in range(3)
        ]

        pickled = pickle.dumps(tasks)
        unpickled_tasks = pickle.loads(pickled)

        assert len(unpickled_tasks) == 3
        for i, task in enumerate(unpickled_tasks):
            assert task.episode_id == i
            assert task.seed == 42 + i
            assert task.episode_number == i

    def test_episode_simulation_task_serialization_with_different_policies(self):
        """Test EpisodeSimulationTask serialization with different policy types.

        Purpose: Validates that tasks with different planner types serialize correctly

        Given: EpisodeSimulationTask instances with POMCP and SparsePFT policies
        When: Tasks are pickled and unpickled
        Then: Both policy types are correctly preserved

        Test type: unit
        """
        pomcp_policy = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            name="POMCP_Test",
            n_simulations=2,
        )

        tasks = {
            "sparse_pft": EpisodeSimulationTask(
                environment=self.env,
                policy=self.policy,
                initial_belief=self.belief,
                num_steps=5,
                episode_id=1,
                seed=42,
                console_output=False,
            ),
            "pomcp": EpisodeSimulationTask(
                environment=self.env,
                policy=pomcp_policy,
                initial_belief=self.belief,
                num_steps=5,
                episode_id=2,
                seed=43,
                console_output=False,
            ),
        }

        pickled = pickle.dumps(tasks)
        unpickled_tasks = pickle.loads(pickled)

        assert len(unpickled_tasks) == 2
        assert "sparse_pft" in unpickled_tasks
        assert "pomcp" in unpickled_tasks
        assert unpickled_tasks["sparse_pft"].policy.name == self.policy.name
        assert unpickled_tasks["pomcp"].policy.name == pomcp_policy.name


class TestHyperParameterTuningSimulationTaskSerialization:
    """Test cases for HyperParameterTuningSimulationTask serialization."""

    def setup_method(self):
        """Set up test environment and hyperparameters for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.belief = create_test_belief()

        # Define hyperparameters for POMCP
        self.hyper_parameters = [
            NumericalHyperParameter(low=3, high=10, name="depth"),
            NumericalHyperParameter(low=0.1, high=100.0, name="exploration_constant"),
            NumericalHyperParameter(low=10, high=1000, name="n_simulations"),
        ]

        self.constant_parameters = {
            "environment": self.env,
            "discount_factor": 0.95,
            "name": "POMCP_Tuned",
        }

    def test_hyperparameter_tuning_task_serialization(self):
        """Test HyperParameterTuningSimulationTask serialization.

        Purpose: Validates that HyperParameterTuningSimulationTask can be pickled and unpickled

        Given: HyperParameterTuningSimulationTask instance with hyperparameters
        When: Task is pickled and unpickled
        Then: Unpickled task maintains all properties and configurations

        Test type: unit
        """
        task = HyperParameterTuningSimulationTask(
            environment=self.env,
            belief=self.belief,
            policy_cls=POMCP,
            hyper_parameters=self.hyper_parameters,
            constant_parameters=self.constant_parameters,
            num_episodes=5,
            num_steps=10,
            parameters_to_optimize=[
                ("avg_discounted_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            experiment_name="test_optimization",
            n_trials=10,
            console_output=False,
            n_jobs=1,
            seed=42,
        )

        # Pickle the task
        pickled = pickle.dumps(task)

        # Unpickle the task
        unpickled_task = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_task.num_episodes == task.num_episodes
        assert unpickled_task.num_steps == task.num_steps
        assert unpickled_task.n_trials == task.n_trials
        assert unpickled_task.n_jobs == task.n_jobs

        # Verify environment is preserved
        assert unpickled_task.environment.name == task.environment.name

        # Verify policy class is preserved
        assert unpickled_task.policy_cls == task.policy_cls

        # Verify hyperparameters are preserved
        assert len(unpickled_task.hyper_parameters) == len(task.hyper_parameters)

    def test_hyperparameter_tuning_task_serialization_with_categorical(self):
        """Test HyperParameterTuningSimulationTask with categorical hyperparameters.

        Purpose: Validates that tasks with categorical hyperparameters serialize correctly

        Given: HyperParameterTuningSimulationTask with both numerical and categorical hyperparameters
        When: Task is pickled and unpickled
        Then: All hyperparameter types are preserved correctly

        Test type: unit
        """
        hyper_params_with_categorical = self.hyper_parameters + [
            CategoricalHyperParameter(
                name="strategy", choices=["aggressive", "defensive", "balanced"]
            )
        ]

        task = HyperParameterTuningSimulationTask(
            environment=self.env,
            belief=self.belief,
            policy_cls=POMCP,
            hyper_parameters=hyper_params_with_categorical,
            constant_parameters=self.constant_parameters,
            num_episodes=5,
            num_steps=10,
            parameters_to_optimize=[
                ("avg_discounted_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            experiment_name="test_categorical",
            n_trials=10,
            console_output=False,
            n_jobs=1,
            seed=42,
        )

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert len(unpickled_task.hyper_parameters) == 4
        # Check that categorical hyperparameter is preserved
        categorical_params = [
            hp for hp in unpickled_task.hyper_parameters if hasattr(hp, "choices")
        ]
        assert len(categorical_params) == 1

    def test_hyperparameter_tuning_task_serialization_with_cache_dir(self):
        """Test HyperParameterTuningSimulationTask serialization with cache directory.

        Purpose: Validates that tasks with cache_dir can be pickled

        Given: HyperParameterTuningSimulationTask with cache_dir specified
        When: Task is pickled and unpickled
        Then: Cache directory path is preserved correctly

        Test type: unit
        """
        cache_dir = Path("/tmp/test_hyperparam_cache")

        task = HyperParameterTuningSimulationTask(
            environment=self.env,
            belief=self.belief,
            policy_cls=POMCP,
            hyper_parameters=self.hyper_parameters,
            constant_parameters=self.constant_parameters,
            num_episodes=5,
            num_steps=10,
            parameters_to_optimize=[
                ("avg_discounted_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            experiment_name="test_with_cache",
            n_trials=10,
            cache_dir=cache_dir,
            console_output=False,
            n_jobs=1,
            seed=42,
        )

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert unpickled_task.cache_dir == cache_dir

    def test_hyperparameter_tuning_task_serialization_with_multiple_optimization_params(
        self,
    ):
        """Test HyperParameterTuningSimulationTask with multiple optimization parameters.

        Purpose: Validates that tasks optimizing multiple parameters serialize correctly

        Given: HyperParameterTuningSimulationTask optimizing multiple parameters
        When: Task is pickled and unpickled
        Then: All optimization parameters and directions are preserved

        Test type: unit
        """
        task = HyperParameterTuningSimulationTask(
            environment=self.env,
            belief=self.belief,
            policy_cls=POMCP,
            hyper_parameters=self.hyper_parameters,
            constant_parameters=self.constant_parameters,
            num_episodes=5,
            num_steps=10,
            parameters_to_optimize=[
                ("avg_discounted_return", HyperParameterOptimizationDirection.MAXIMIZE),
                ("avg_episode_length", HyperParameterOptimizationDirection.MINIMIZE),
            ],
            experiment_name="test_multi_objective",
            n_trials=10,
            console_output=False,
            n_jobs=1,
            seed=42,
        )

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert len(unpickled_task.parameters_to_optimize) == 2
        assert unpickled_task.parameters_to_optimize[0][0] == "avg_discounted_return"
        assert unpickled_task.parameters_to_optimize[1][0] == "avg_episode_length"


class TestTaskSerializationRoundTrip:
    """Test cases for complete task serialization round trips."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.policy = SparsePFT(
            environment=self.env,
            discount_factor=0.95,
            gamma=0.95,
            depth=3,
            c_ucb=1.0,
            beta_ucb=0.5,
            belief_child_num=4,
            n_simulations=2,
        )
        self.belief = create_test_belief()

    def test_task_serialization_preserves_execution_capability(self):
        """Test that serialized tasks maintain execution capability.

        Purpose: Validates that tasks work correctly after serialization round trip

        Given: EpisodeSimulationTask instance
        When: Task is pickled, unpickled, and executed
        Then: Unpickled task can be executed successfully

        Test type: integration
        """
        task = EpisodeSimulationTask(
            environment=self.env,
            policy=self.policy,
            initial_belief=self.belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Pickle and unpickle
        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        # Execute the unpickled task
        result = unpickled_task.run()

        # Verify result is valid History object
        assert result is not None
        # History object has attributes like discounted_return
        assert hasattr(result, "discounted_return") or hasattr(
            result, "history"
        )  # Check for History attributes

    def test_multiple_tasks_serialization_together(self):
        """Test serialization of different task types together.

        Purpose: Validates that different task types can be pickled in same structure

        Given: Dictionary containing EpisodeSimulationTask and HyperParameterTuningSimulationTask
        When: Dictionary is pickled and unpickled
        Then: All tasks are correctly restored

        Test type: integration
        """
        hyper_parameters = [
            NumericalHyperParameter(low=3, high=10, name="depth"),
        ]

        constant_parameters = {
            "environment": self.env,
            "discount_factor": 0.95,
            "name": "POMCP_Test",
        }

        tasks_dict = {
            "episode": EpisodeSimulationTask(
                environment=self.env,
                policy=self.policy,
                initial_belief=self.belief,
                num_steps=5,
                episode_id=1,
                seed=42,
                console_output=False,
            ),
            "hyperparam": HyperParameterTuningSimulationTask(
                environment=self.env,
                belief=self.belief,
                policy_cls=POMCP,
                hyper_parameters=hyper_parameters,
                constant_parameters=constant_parameters,
                num_episodes=2,
                num_steps=5,
                parameters_to_optimize=[
                    ("avg_discounted_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                experiment_name="test_mixed",
                n_trials=5,
                console_output=False,
                n_jobs=1,
                seed=42,
            ),
        }

        pickled = pickle.dumps(tasks_dict)
        unpickled_dict = pickle.loads(pickled)

        assert len(unpickled_dict) == 2
        assert "episode" in unpickled_dict
        assert "hyperparam" in unpickled_dict
        assert isinstance(unpickled_dict["episode"], EpisodeSimulationTask)
        assert isinstance(unpickled_dict["hyperparam"], HyperParameterTuningSimulationTask)


class TestTaskSerializationEdgeCases:
    """Test cases for edge cases in task serialization."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.policy = SparsePFT(
            environment=self.env,
            discount_factor=0.95,
            gamma=0.95,
            depth=3,
            c_ucb=1.0,
            beta_ucb=0.5,
            belief_child_num=4,
            n_simulations=2,
        )
        self.belief = create_test_belief()

    def test_task_serialization_with_different_seeds(self):
        """Test task serialization preserves different seeds.

        Purpose: Validates that tasks with different seeds serialize correctly

        Given: Multiple tasks with different seed values
        When: Tasks are pickled and unpickled
        Then: Each task preserves its unique seed value

        Test type: unit
        """
        tasks = [
            EpisodeSimulationTask(
                environment=self.env,
                policy=self.policy,
                initial_belief=self.belief,
                num_steps=5,
                episode_id=i,
                seed=seed,
                console_output=False,
            )
            for i, seed in enumerate([42, 123, 456, 789])
        ]

        pickled = pickle.dumps(tasks)
        unpickled_tasks = pickle.loads(pickled)

        expected_seeds = [42, 123, 456, 789]
        for task, expected_seed in zip(unpickled_tasks, expected_seeds):
            assert task.seed == expected_seed

    def test_task_cache_key_consistency_after_serialization(self):
        """Test that cache keys remain consistent after serialization.

        Purpose: Validates that task cache keys are stable across serialization

        Given: EpisodeSimulationTask with generated cache key
        When: Task is pickled and unpickled
        Then: Cache key remains identical

        Test type: unit
        """
        task = EpisodeSimulationTask(
            environment=self.env,
            policy=self.policy,
            initial_belief=self.belief,
            num_steps=5,
            episode_id=1,
            seed=42,
            console_output=False,
        )

        original_cache_key = task._cache_key

        pickled = pickle.dumps(task)
        unpickled_task = pickle.loads(pickled)

        assert unpickled_task._cache_key == original_cache_key
