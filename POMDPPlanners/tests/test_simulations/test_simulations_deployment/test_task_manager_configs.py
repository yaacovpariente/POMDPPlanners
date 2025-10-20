"""Tests for task manager configuration classes.

This module contains comprehensive tests for the task manager configuration
classes and their factory methods.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from POMDPPlanners.core.simulation import TaskManager
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    DaskConfig,
    JoblibConfig,
    PBSConfig,
    TaskManagerConfig,
)


class TestTaskManagerConfig:
    """Test the abstract TaskManagerConfig base class."""

    def test_task_manager_config_is_abstract(self):
        """Test that TaskManagerConfig cannot be instantiated directly.

        Purpose: Validates that TaskManagerConfig is properly abstract

        Given: The TaskManagerConfig class
        When: Attempting to instantiate it directly
        Then: TypeError is raised due to abstract method

        Test type: unit
        """
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TaskManagerConfig()  # type: ignore[abstract]

    def test_create_task_manager_is_abstract(self):
        """Test that create_task_manager is abstract.

        Purpose: Validates that create_task_manager method must be implemented

        Given: A concrete subclass that doesn't implement create_task_manager
        When: Attempting to instantiate the subclass
        Then: TypeError is raised due to unimplemented abstract method

        Test type: unit
        """

        class IncompleteConfig(TaskManagerConfig):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteConfig()  # type: ignore[abstract]


class TestDaskConfig:
    """Test the DaskConfig configuration class."""

    def test_dask_config_default_initialization(self):
        """Test DaskConfig initialization with default values.

        Purpose: Validates that DaskConfig initializes with correct defaults

        Given: No initialization parameters
        When: Creating a DaskConfig instance
        Then: All attributes have expected default values

        Test type: unit
        """
        config = DaskConfig()

        assert config.n_workers == 1
        assert config.scheduler_address is None
        assert config.cache_size == int(2e9)
        assert config.clear_cache_on_start is False

    def test_dask_config_custom_initialization(self):
        """Test DaskConfig initialization with custom values.

        Purpose: Validates that DaskConfig accepts and stores custom parameters

        Given: Custom initialization parameters
        When: Creating a DaskConfig instance with custom values
        Then: All attributes match the provided values

        Test type: unit
        """
        config = DaskConfig(
            n_workers=4,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(1e10),
            clear_cache_on_start=True,
        )

        assert config.n_workers == 4
        assert config.scheduler_address == "tcp://localhost:8786"
        assert config.cache_size == int(1e10)
        assert config.clear_cache_on_start is True

    @patch(
        "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
    )
    def test_dask_config_create_task_manager(self, mock_factory):
        """Test DaskConfig creates task manager via factory.

        Purpose: Validates that DaskConfig properly delegates to TaskManagerFactory

        Given: A DaskConfig instance and mocked TaskManagerFactory
        When: Calling create_task_manager method
        Then: Factory create_dask method is called with correct parameters

        Test type: unit
        """
        mock_task_manager = Mock(spec=TaskManager)
        mock_factory.create_dask.return_value = mock_task_manager

        config = DaskConfig(
            n_workers=2,
            scheduler_address="tcp://localhost:8787",
            cache_size=int(3e9),
            clear_cache_on_start=True,
        )

        result = config.create_task_manager(cache_dir="/test/cache")

        mock_factory.create_dask.assert_called_once_with(
            n_workers=2,
            scheduler_address="tcp://localhost:8787",
            cache_size=int(3e9),
            clear_cache_on_start=True,
        )
        assert result == mock_task_manager

    def test_dask_config_is_task_manager_config(self):
        """Test that DaskConfig is instance of TaskManagerConfig.

        Purpose: Validates proper inheritance relationship

        Given: A DaskConfig instance
        When: Checking isinstance with TaskManagerConfig
        Then: Returns True

        Test type: unit
        """
        config = DaskConfig()
        assert isinstance(config, TaskManagerConfig)


class TestPBSConfig:
    """Test the PBSConfig configuration class."""

    def test_pbs_config_required_queue_parameter(self):
        """Test that PBSConfig requires queue parameter.

        Purpose: Validates that queue parameter is mandatory for PBS configuration

        Given: No queue parameter provided
        When: Attempting to create PBSConfig instance
        Then: TypeError is raised for missing required argument

        Test type: unit
        """
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'queue'"):
            PBSConfig()  # type: ignore[call-arg]  # pylint: disable=no-value-for-parameter

    def test_pbs_config_default_initialization(self):
        """Test PBSConfig initialization with required parameter and defaults.

        Purpose: Validates that PBSConfig initializes with correct defaults when queue provided

        Given: Only the required queue parameter
        When: Creating a PBSConfig instance
        Then: All other attributes have expected default values

        Test type: unit
        """
        config = PBSConfig(queue="normal")

        assert config.queue == "normal"
        assert config.n_workers == 4
        assert config.cores == 1
        assert config.memory == "4GB"
        assert config.processes == 1
        assert config.walltime == "01:00:00"
        assert config.job_extra is None
        assert config.cache_size == int(2e9)
        assert config.clear_cache_on_start is False
        # Dashboard default values
        assert config.enable_dashboard is True
        assert config.dashboard_address == "0.0.0.0"
        assert config.dashboard_port == 8787
        assert config.dashboard_prefix is None

    def test_pbs_config_custom_initialization(self):
        """Test PBSConfig initialization with custom values.

        Purpose: Validates that PBSConfig accepts and stores custom parameters

        Given: Custom initialization parameters including required queue
        When: Creating a PBSConfig instance with custom values
        Then: All attributes match the provided values

        Test type: unit
        """
        job_extra = ["#PBS -l feature=gpu", "#PBS -m ae"]
        config = PBSConfig(
            queue="gpu",
            n_workers=8,
            cores=4,
            memory="16GB",
            processes=2,
            walltime="02:30:00",
            job_extra=job_extra,
            cache_size=int(5e9),
            clear_cache_on_start=True,
            enable_dashboard=True,
            dashboard_address="127.0.0.1",
            dashboard_port=9999,
            dashboard_prefix="/my-cluster",
        )

        assert config.queue == "gpu"
        assert config.n_workers == 8
        assert config.cores == 4
        assert config.memory == "16GB"
        assert config.processes == 2
        assert config.walltime == "02:30:00"
        assert config.job_extra == job_extra
        assert config.cache_size == int(5e9)
        assert config.clear_cache_on_start is True
        # Dashboard custom values
        assert config.enable_dashboard is True
        assert config.dashboard_address == "127.0.0.1"
        assert config.dashboard_port == 9999
        assert config.dashboard_prefix == "/my-cluster"

    @patch(
        "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
    )
    def test_pbs_config_create_task_manager(self, mock_factory):
        """Test PBSConfig creates task manager via factory.

        Purpose: Validates that PBSConfig properly delegates to TaskManagerFactory

        Given: A PBSConfig instance and mocked TaskManagerFactory
        When: Calling create_task_manager method
        Then: Factory create_pbs method is called with correct parameters

        Test type: unit
        """
        mock_task_manager = Mock(spec=TaskManager)
        mock_factory.create_pbs.return_value = mock_task_manager

        job_extra = ["#PBS -l gpu=1"]
        config = PBSConfig(
            queue="gpu",
            n_workers=4,
            cores=2,
            memory="8GB",
            processes=1,
            walltime="01:30:00",
            job_extra=job_extra,
            cache_size=int(4e9),
            clear_cache_on_start=False,
            enable_dashboard=True,
            dashboard_address="192.168.1.100",
            dashboard_port=8888,
            dashboard_prefix="/cluster-dashboard",
        )

        result = config.create_task_manager(cache_dir="/test/cache")

        mock_factory.create_pbs.assert_called_once_with(
            queue="gpu",
            n_workers=4,
            cores=2,
            memory="8GB",
            processes=1,
            walltime="01:30:00",
            job_extra=job_extra,
            cache_size=int(4e9),
            clear_cache_on_start=False,
            enable_dashboard=True,
            dashboard_address="192.168.1.100",
            dashboard_port=8888,
            dashboard_prefix="/cluster-dashboard",
        )
        assert result == mock_task_manager

    def test_pbs_config_dashboard_disabled_initialization(self):
        """Test PBSConfig initialization with dashboard disabled.

        Purpose: Validates that PBSConfig can be configured with dashboard disabled

        Given: PBSConfig with enable_dashboard=False
        When: Creating a PBSConfig instance with dashboard disabled
        Then: Dashboard is disabled but other dashboard parameters are still stored

        Test type: unit
        """
        config = PBSConfig(
            queue="batch",
            enable_dashboard=False,
            dashboard_port=9999,
            dashboard_address="10.0.0.1",
        )

        assert config.enable_dashboard is False
        assert config.dashboard_port == 9999  # Should still store the parameter
        assert config.dashboard_address == "10.0.0.1"  # Should still store the parameter

    def test_pbs_config_dashboard_parameter_validation(self):
        """Test PBSConfig accepts various dashboard parameter values.

        Purpose: Validates that PBSConfig accepts different valid dashboard configurations

        Given: PBSConfig with various dashboard parameter combinations
        When: Creating PBSConfig instances with different dashboard settings
        Then: All valid parameter combinations are accepted and stored correctly

        Test type: unit
        """
        # Test various port values
        test_ports = [8787, 8888, 9999, 8080, 3000]
        for port in test_ports:
            config = PBSConfig(queue="test", dashboard_port=port)
            assert config.dashboard_port == port

        # Test various address values
        test_addresses = ["0.0.0.0", "127.0.0.1", "192.168.1.100", "localhost"]
        for address in test_addresses:
            config = PBSConfig(queue="test", dashboard_address=address)
            assert config.dashboard_address == address

        # Test various prefix values
        test_prefixes = [None, "/dashboard", "/my-app", "/cluster-monitor", "api/v1"]
        for prefix in test_prefixes:
            config = PBSConfig(queue="test", dashboard_prefix=prefix)
            assert config.dashboard_prefix == prefix

    def test_pbs_config_is_task_manager_config(self):
        """Test that PBSConfig is instance of TaskManagerConfig.

        Purpose: Validates proper inheritance relationship

        Given: A PBSConfig instance
        When: Checking isinstance with TaskManagerConfig
        Then: Returns True

        Test type: unit
        """
        config = PBSConfig(queue="normal")
        assert isinstance(config, TaskManagerConfig)


class TestJoblibConfig:
    """Test the JoblibConfig configuration class."""

    def test_joblib_config_default_initialization(self):
        """Test JoblibConfig initialization with default values.

        Purpose: Validates that JoblibConfig initializes with correct defaults

        Given: No initialization parameters
        When: Creating a JoblibConfig instance
        Then: All attributes have expected default values

        Test type: unit
        """
        config = JoblibConfig()

        assert config.n_jobs == -1
        assert config.cache_size == int(2e9)
        assert config.eviction_policy == "least-recently-used"
        assert config.verbose == 0
        assert config.clear_cache_on_start is False

    def test_joblib_config_custom_initialization(self):
        """Test JoblibConfig initialization with custom values.

        Purpose: Validates that JoblibConfig accepts and stores custom parameters

        Given: Custom initialization parameters
        When: Creating a JoblibConfig instance with custom values
        Then: All attributes match the provided values

        Test type: unit
        """
        config = JoblibConfig(
            n_jobs=4,
            cache_size=int(1e10),
            eviction_policy="least-frequently-used",
            verbose=2,
            clear_cache_on_start=True,
        )

        assert config.n_jobs == 4
        assert config.cache_size == int(1e10)
        assert config.eviction_policy == "least-frequently-used"
        assert config.verbose == 2
        assert config.clear_cache_on_start is True

    @patch(
        "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
    )
    def test_joblib_config_create_task_manager(self, mock_factory):
        """Test JoblibConfig creates task manager via factory.

        Purpose: Validates that JoblibConfig properly delegates to TaskManagerFactory

        Given: A JoblibConfig instance and mocked TaskManagerFactory
        When: Calling create_task_manager method
        Then: Factory create_joblib method is called with correct parameters

        Test type: unit
        """
        mock_task_manager = Mock(spec=TaskManager)
        mock_factory.create_joblib.return_value = mock_task_manager

        config = JoblibConfig(
            n_jobs=8,
            cache_size=int(3e9),
            eviction_policy="random",
            verbose=1,
            clear_cache_on_start=True,
        )

        result = config.create_task_manager(cache_dir="/test/cache")

        mock_factory.create_joblib.assert_called_once_with(
            cache_dir="/test/cache",
            cache_size=int(3e9),
            n_jobs=8,
            eviction_policy="random",
            clear_cache_on_start=True,
            verbose=1,
        )
        assert result == mock_task_manager

    @patch(
        "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
    )
    def test_joblib_config_create_task_manager_default_cache_dir(self, mock_factory):
        """Test JoblibConfig uses default cache directory when none provided.

        Purpose: Validates that JoblibConfig handles missing cache_dir parameter

        Given: A JoblibConfig instance and no cache_dir parameter
        When: Calling create_task_manager without cache_dir
        Then: Factory is called with "./cache" as default cache_dir

        Test type: unit
        """
        mock_task_manager = Mock(spec=TaskManager)
        mock_factory.create_joblib.return_value = mock_task_manager

        config = JoblibConfig()
        result = config.create_task_manager()

        mock_factory.create_joblib.assert_called_once_with(
            cache_dir="./cache",
            cache_size=int(2e9),
            n_jobs=-1,
            eviction_policy="least-recently-used",
            clear_cache_on_start=False,
            verbose=0,
        )
        assert result == mock_task_manager

    def test_joblib_config_is_task_manager_config(self):
        """Test that JoblibConfig is instance of TaskManagerConfig.

        Purpose: Validates proper inheritance relationship

        Given: A JoblibConfig instance
        When: Checking isinstance with TaskManagerConfig
        Then: Returns True

        Test type: unit
        """
        config = JoblibConfig()
        assert isinstance(config, TaskManagerConfig)


class TestTaskManagerConfigIntegration:
    """Integration tests for task manager configurations."""

    def test_all_configs_inherit_from_base(self):
        """Test that all concrete configs inherit from TaskManagerConfig.

        Purpose: Validates inheritance hierarchy for all config classes

        Given: All concrete config classes
        When: Checking inheritance from TaskManagerConfig
        Then: All concrete classes are subclasses of TaskManagerConfig

        Test type: integration
        """
        concrete_configs = [DaskConfig(), PBSConfig(queue="test"), JoblibConfig()]

        for config in concrete_configs:
            assert isinstance(config, TaskManagerConfig)
            assert hasattr(config, "create_task_manager")
            assert callable(config.create_task_manager)

    def test_config_polymorphism(self):
        """Test polymorphic behavior of config objects.

        Purpose: Validates that all configs can be used polymorphically

        Given: A list of different config instances
        When: Calling create_task_manager on each via base class interface
        Then: All calls succeed without errors

        Test type: integration
        """
        configs = [
            DaskConfig(n_workers=2),
            PBSConfig(queue="normal", n_workers=2),
            JoblibConfig(n_jobs=2),
        ]

        # Test that all configs can be used polymorphically
        with patch(
            "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
        ) as mock_factory:
            mock_factory.create_dask.return_value = Mock(spec=TaskManager)
            mock_factory.create_pbs.return_value = Mock(spec=TaskManager)
            mock_factory.create_joblib.return_value = Mock(spec=TaskManager)

            for config in configs:
                # This should work for all config types
                task_manager = config.create_task_manager()
                assert task_manager is not None

    def test_config_cache_dir_parameter_handling(self):
        """Test cache_dir parameter handling across all configs.

        Purpose: Validates consistent cache_dir parameter handling

        Given: Different config types with cache_dir parameter
        When: Calling create_task_manager with cache_dir
        Then: All configs handle the parameter appropriately

        Test type: integration
        """
        cache_dir = "/custom/cache/path"

        with patch(
            "POMDPPlanners.simulations.simulations_deployment.task_manager_configs.TaskManagerFactory"
        ) as mock_factory:
            mock_factory.create_dask.return_value = Mock(spec=TaskManager)
            mock_factory.create_pbs.return_value = Mock(spec=TaskManager)
            mock_factory.create_joblib.return_value = Mock(spec=TaskManager)

            # Test DaskConfig (cache_dir not used by Dask)
            dask_config = DaskConfig()
            dask_config.create_task_manager(cache_dir=cache_dir)
            mock_factory.create_dask.assert_called_once()

            # Test PBSConfig (cache_dir not used by PBS)
            pbs_config = PBSConfig(queue="normal")
            pbs_config.create_task_manager(cache_dir=cache_dir)
            mock_factory.create_pbs.assert_called_once()

            # Test JoblibConfig (cache_dir used by Joblib)
            joblib_config = JoblibConfig()
            joblib_config.create_task_manager(cache_dir=cache_dir)
            mock_factory.create_joblib.assert_called_once_with(
                cache_dir=cache_dir,
                cache_size=int(2e9),
                n_jobs=-1,
                eviction_policy="least-recently-used",
                clear_cache_on_start=False,
                verbose=0,
            )


class TestPBSConfigIntegration:
    """Test PBS configuration integration tests."""

    @pytest.mark.pbs_cluster
    def test_pbs_config_creation_integration(self):
        """Test PBS configuration creation with integration test.

        Purpose: Validates that PBS configuration can be created with proper parameters
        and default values are correctly set.

        Given: PBS configuration parameters
        When: Creating a PBSConfig instance
        Then: All parameters are correctly set and defaults are applied

        Test type: integration
        """
        config = PBSConfig(
            queue="test_queue", n_workers=2, cores=1, memory="2GB", walltime="00:30:00"
        )

        assert config.queue == "test_queue"
        assert config.n_workers == 2
        assert config.cores == 1
        assert config.memory == "2GB"
        assert config.walltime == "00:30:00"
        assert config.enable_dashboard is True  # Default value
        assert config.dashboard_address == "0.0.0.0"  # Default value
        assert config.dashboard_port == 8787  # Default value
        assert config.dashboard_prefix is None  # Default value
