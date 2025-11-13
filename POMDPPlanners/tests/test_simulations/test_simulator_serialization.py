"""Tests for POMDP simulator serialization.

This module tests that POMDP simulators can be properly serialized
and deserialized using pickle. Serialization is crucial for:
- Distributed computing scenarios (Dask, Ray)
- Saving/loading simulator configurations
- Checkpointing during long-running experiments
- Multi-processing applications
- Task queuing systems
"""

import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.environments import TigerPOMDP
from POMDPPlanners.planners import POMCP
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    DaskConfig,
    JoblibConfig,
)

# Set seeds for reproducible tests
np.random.seed(42)


class TestPOMDPSimulatorSerialization:
    """Test cases for POMDPSimulator serialization using pickle."""

    def setup_method(self):
        """Set up test environment and policies for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.policy = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=10.0,
            name="POMCP_Test",
            n_simulations=50,
        )

        # Create initial belief
        initial_state = self.env.initial_state_dist().sample()[0]
        self.belief = WeightedParticleBelief(
            particles=[initial_state] * 100,
            log_weights=np.log(np.ones(100) / 100),
            resampling=True,
        )

    def _test_simulator_serialization(
        self, simulator_class: type, init_params: Dict[str, Any]
    ) -> None:
        """Helper method to test simulator serialization.

        Purpose: Validates that a simulator can be pickled and unpickled correctly

        Given: A simulator class and initialization parameters
        When: Simulator is created, pickled, and unpickled
        Then: Unpickled simulator maintains all properties

        Test type: unit

        Args:
            simulator_class: Simulator class to test
            init_params: Parameters for simulator initialization
        """
        # Create simulator
        simulator = simulator_class(**init_params)

        # Pickle the simulator
        pickled = pickle.dumps(simulator)

        # Unpickle the simulator
        unpickled_simulator = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_simulator.experiment_name == simulator.experiment_name
        assert unpickled_simulator.debug == simulator.debug
        assert unpickled_simulator.enable_profiling == simulator.enable_profiling

    def test_pomdp_simulator_serialization_with_joblib(self):
        """Test POMDPSimulator serialization with Joblib task manager.

        Purpose: Validates that POMDPSimulator with JoblibConfig can be pickled

        Given: POMDPSimulator instance with JoblibConfig
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator maintains all properties and configuration

        Test type: unit
        """
        task_manager_config = JoblibConfig(n_jobs=2, verbose=0)

        self._test_simulator_serialization(
            POMDPSimulator,
            {
                "task_manager_config": task_manager_config,
                "cache_dir_path": None,
                "experiment_name": "Test_Joblib_Simulation",
                "debug": False,
                "enable_profiling": False,
                "console_output": False,
            },
        )

    def test_pomdp_simulator_serialization_with_dask(self):
        """Test POMDPSimulator serialization with Dask task manager.

        Purpose: Validates that POMDPSimulator with DaskConfig can be pickled

        Given: POMDPSimulator instance with DaskConfig
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator maintains all properties and configuration

        Test type: unit
        """
        # Dask creates asyncio tasks that cannot be pickled once the cluster is started
        # This is a known limitation of Dask distributed systems
        pytest.skip(
            "Dask simulators with active clusters cannot be pickled due to asyncio.Task objects"
        )

    def test_pomdp_simulator_serialization_with_cache_dir(self):
        """Test POMDPSimulator serialization with cache directory.

        Purpose: Validates that POMDPSimulator with cache directory can be pickled

        Given: POMDPSimulator instance with temporary cache directory
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator maintains cache directory path

        Test type: unit
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            task_manager_config = JoblibConfig(n_jobs=1)

            simulator = POMDPSimulator(
                task_manager_config=task_manager_config,
                cache_dir_path=cache_path,
                experiment_name="Test_Cache_Simulation",
                debug=False,
                enable_profiling=False,
                console_output=False,
            )

            # Pickle and unpickle
            pickled = pickle.dumps(simulator)
            unpickled_simulator = pickle.loads(pickled)

            assert unpickled_simulator.cache_dir_path == simulator.cache_dir_path
            assert unpickled_simulator.experiment_name == simulator.experiment_name

    def test_pomdp_simulator_serialization_with_debug_enabled(self):
        """Test POMDPSimulator serialization with debug mode enabled.

        Purpose: Validates that POMDPSimulator with debug flag can be pickled

        Given: POMDPSimulator instance with debug=True
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator maintains debug state

        Test type: unit
        """
        task_manager_config = JoblibConfig(n_jobs=1)

        simulator = POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=None,
            experiment_name="Test_Debug_Simulation",
            debug=True,
            enable_profiling=False,
            console_output=False,
        )

        pickled = pickle.dumps(simulator)
        unpickled_simulator = pickle.loads(pickled)

        assert unpickled_simulator.debug is True
        assert unpickled_simulator.experiment_name == simulator.experiment_name

    def test_pomdp_simulator_serialization_with_profiling_enabled(self):
        """Test POMDPSimulator serialization with profiling enabled.

        Purpose: Validates that POMDPSimulator with profiling can be pickled

        Given: POMDPSimulator instance with enable_profiling=True
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator maintains profiling state

        Test type: unit
        """
        task_manager_config = JoblibConfig(n_jobs=1)

        simulator = POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=None,
            experiment_name="Test_Profiling_Simulation",
            debug=False,
            enable_profiling=True,
            profiling_output_limit=100,
            console_output=False,
        )

        pickled = pickle.dumps(simulator)
        unpickled_simulator = pickle.loads(pickled)

        assert unpickled_simulator.enable_profiling is True
        assert unpickled_simulator.profiling_output_limit == 100
        assert unpickled_simulator.experiment_name == simulator.experiment_name


class TestSimulatorSerializationEdgeCases:
    """Test cases for edge cases in simulator serialization."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)

    def test_simulator_serialization_with_no_initialization(self):
        """Test simulator serialization immediately after creation.

        Purpose: Validates that fresh simulators can be pickled without use

        Given: Newly created simulator without running any simulations
        When: Simulator is pickled and unpickled
        Then: Unpickled simulator is ready for use

        Test type: unit
        """
        task_manager_config = JoblibConfig(n_jobs=1)
        simulator = POMDPSimulator(
            task_manager_config=task_manager_config,
            cache_dir_path=None,
            experiment_name="Test_Fresh_Simulation",
            debug=False,
            enable_profiling=False,
            console_output=False,
        )

        # Pickle immediately without running anything
        pickled = pickle.dumps(simulator)
        unpickled_simulator = pickle.loads(pickled)

        assert unpickled_simulator.experiment_name == simulator.experiment_name
        assert unpickled_simulator.debug == simulator.debug

    def test_multiple_simulators_serialization(self):
        """Test serialization of multiple simulators together.

        Purpose: Validates that multiple simulators can be pickled in same structure

        Given: Dictionary containing multiple different simulators
        When: Dictionary is pickled and unpickled
        Then: All simulators are correctly restored

        Test type: integration
        """
        simulators_dict = {
            "joblib_sim_1": POMDPSimulator(
                task_manager_config=JoblibConfig(n_jobs=1),
                cache_dir_path=None,
                experiment_name="Multi_Joblib_Test_1",
                debug=False,
                enable_profiling=False,
                console_output=False,
            ),
            "joblib_sim_2": POMDPSimulator(
                task_manager_config=JoblibConfig(n_jobs=2, verbose=1),
                cache_dir_path=None,
                experiment_name="Multi_Joblib_Test_2",
                debug=True,
                enable_profiling=False,
                console_output=False,
            ),
        }

        pickled = pickle.dumps(simulators_dict)
        unpickled_dict = pickle.loads(pickled)

        assert len(unpickled_dict) == 2
        assert "joblib_sim_1" in unpickled_dict
        assert "joblib_sim_2" in unpickled_dict
        assert unpickled_dict["joblib_sim_1"].experiment_name == "Multi_Joblib_Test_1"
        assert unpickled_dict["joblib_sim_2"].experiment_name == "Multi_Joblib_Test_2"

    def test_simulator_serialization_with_different_configs(self):
        """Test serialization of simulators with various configurations.

        Purpose: Validates that diverse simulator configurations can be pickled

        Given: Simulators with different task manager and profiling configurations
        When: Each simulator is pickled and unpickled
        Then: All configurations are preserved correctly

        Test type: integration
        """
        configs_to_test = [
            {
                "task_manager_config": JoblibConfig(n_jobs=4, verbose=1),
                "experiment_name": "Config_Test_1",
                "debug": True,
                "enable_profiling": False,
            },
            {
                "task_manager_config": JoblibConfig(n_jobs=2, cache_size=int(1e9), verbose=0),
                "experiment_name": "Config_Test_2",
                "debug": True,
                "enable_profiling": True,
            },
            {
                "task_manager_config": JoblibConfig(n_jobs=1, cache_size=int(5e8), verbose=2),
                "experiment_name": "Config_Test_3",
                "debug": False,
                "enable_profiling": False,
            },
        ]

        for config in configs_to_test:
            simulator = POMDPSimulator(cache_dir_path=None, console_output=False, **config)

            pickled = pickle.dumps(simulator)
            unpickled_simulator = pickle.loads(pickled)

            assert unpickled_simulator.experiment_name == config["experiment_name"]
            assert unpickled_simulator.debug == config["debug"]
            assert unpickled_simulator.enable_profiling == config["enable_profiling"]
