import numpy as np

from POMDPPlanners.utils.config_to_id import config_to_id, NumpyEncoder


class TestNumpyEncoder:
    """Test cases for the NumpyEncoder JSON serialization class."""

    def test_numpy_array_encoding(self):
        """Test encoding of NumPy arrays to lists.

        Purpose: Validates that NumPy arrays are properly converted to Python lists

        Given: A NumpyEncoder instance and various NumPy arrays
        When: Arrays are serialized using the encoder
        Then: Arrays are converted to their list representations

        Test type: unit
        """
        encoder = NumpyEncoder()

        # Test 1D array
        arr_1d = np.array([1, 2, 3])
        result_1d = encoder.default(arr_1d)
        assert result_1d == [1, 2, 3]

        # Test 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result_2d = encoder.default(arr_2d)
        assert result_2d == [[1, 2], [3, 4]]

        # Test empty array
        arr_empty = np.array([])
        result_empty = encoder.default(arr_empty)
        assert result_empty == []

    def test_numpy_scalar_encoding(self):
        """Test encoding of NumPy scalar types to Python primitives.

        Purpose: Validates that NumPy scalars are converted to native Python types

        Given: A NumpyEncoder instance and various NumPy scalar types
        When: Scalars are serialized using the encoder
        Then: Scalars are converted to equivalent Python primitives

        Test type: unit
        """
        encoder = NumpyEncoder()

        # Test integer types
        np_int32 = np.int32(42)
        assert encoder.default(np_int32) == 42
        assert isinstance(encoder.default(np_int32), int)

        np_int64 = np.int64(100)
        assert encoder.default(np_int64) == 100
        assert isinstance(encoder.default(np_int64), int)

        # Test floating types
        np_float32 = np.float32(3.14)
        result_float32 = encoder.default(np_float32)
        assert isinstance(result_float32, float)
        assert abs(result_float32 - 3.14) < 1e-6  # Account for float32 precision

        np_float64 = np.float64(2.71)
        assert encoder.default(np_float64) == 2.71
        assert isinstance(encoder.default(np_float64), float)

    def test_config_id_object_encoding(self):
        """Test encoding of objects with config_id attribute.

        Purpose: Validates that objects with config_id are serialized with class metadata

        Given: A mock object with config_id attribute and class information
        When: Object is serialized using the encoder
        Then: Object is converted to dict with class and config_id information

        Test type: unit
        """
        encoder = NumpyEncoder()

        # Create mock object with config_id
        class MockObject:
            def __init__(self):
                self.config_id = "test_config_123"

        mock_obj = MockObject()
        result = encoder.default(mock_obj)

        expected = {
            "__class__": "MockObject",
            "__module__": "POMDPPlanners.tests.test_utils.test_config_to_id",
            "__config_id__": "test_config_123",
        }
        assert result == expected

    def test_getstate_object_encoding(self):
        """Test encoding of objects with __getstate__ method.

        Purpose: Validates that objects with __getstate__ are serialized with their state

        Given: A mock object with __getstate__ method
        When: Object is serialized using the encoder
        Then: Object is converted to dict with class and state information

        Test type: unit
        """
        encoder = NumpyEncoder()

        # Create mock object with __getstate__
        class MockStatefulObject:
            def __init__(self, value):
                self.value = value

            def __getstate__(self):
                return {"value": self.value, "extra": "state_data"}

        mock_obj = MockStatefulObject("test_value")
        result = encoder.default(mock_obj)

        expected = {
            "__class__": "MockStatefulObject",
            "__module__": "POMDPPlanners.tests.test_utils.test_config_to_id",
            "__state__": {"value": "test_value", "extra": "state_data"},
        }
        assert result == expected

    def test_object_without_config_id_uses_getstate(self):
        """Test that objects without config_id fall back to __getstate__ mechanism.

        Purpose: Validates that objects without config_id attribute use __getstate__ fallback

        Given: A mock object without config_id attribute but with __getstate__ method
        When: Object is serialized using the encoder
        Then: Encoder uses __getstate__ mechanism for serialization

        Test type: unit
        """
        encoder = NumpyEncoder()

        class MockObjectWithoutConfigId:
            def __init__(self, value):
                self.value = value

            def __getstate__(self):
                return {"value": self.value, "type": "no_config_id"}

        mock_obj = MockObjectWithoutConfigId("test_value")
        result = encoder.default(mock_obj)

        # Should use __getstate__ since no config_id exists
        expected = {
            "__class__": "MockObjectWithoutConfigId",
            "__module__": "POMDPPlanners.tests.test_utils.test_config_to_id",
            "__state__": {"value": "test_value", "type": "no_config_id"},
        }
        assert result == expected

    def test_getstate_exception_fallback(self):
        """Test fallback to string when __getstate__ raises exception.

        Purpose: Validates proper fallback when __getstate__ method fails

        Given: A mock object with __getstate__ that raises exception
        When: Object is serialized using the encoder
        Then: Object is converted to string representation

        Test type: unit
        """
        encoder = NumpyEncoder()

        class MockObjectWithFailingGetState:
            def __getstate__(self):
                raise RuntimeError("GetState failed")

        mock_obj = MockObjectWithFailingGetState()
        result = encoder.default(mock_obj)

        # Should fall back to string representation
        assert isinstance(result, str)
        assert "MockObjectWithFailingGetState" in result

    def test_unsupported_type_fallback(self):
        """Test behavior with object types that fall back to __getstate__.

        Purpose: Validates that objects without custom handling use __getstate__ fallback

        Given: An object type that doesn't match NumPy or config_id patterns
        When: Object is serialized using the encoder
        Then: Object is handled by __getstate__ fallback mechanism

        Test type: unit
        """
        encoder = NumpyEncoder()

        # Complex number should fall back to __getstate__ mechanism
        complex_num = complex(1, 2)
        result = encoder.default(complex_num)

        # Should get dict with class info and state
        assert isinstance(result, dict)
        assert result["__class__"] == "complex"
        assert result["__module__"] == "builtins"
        assert "__state__" in result


class TestConfigToId:
    """Test cases for the config_to_id function."""

    def test_simple_dict_hashing(self):
        """Test hashing of simple dictionary configurations.

        Purpose: Validates that simple dictionaries produce consistent hash IDs

        Given: Simple dictionaries with basic Python types
        When: config_to_id is called on the dictionaries
        Then: Consistent hexadecimal hash strings are returned

        Test type: unit
        """
        config1 = {"param1": "value1", "param2": 42}
        config2 = {"param1": "value1", "param2": 42}
        config3 = {"param1": "different", "param2": 42}

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)
        id3 = config_to_id(config3)

        # Same configs should produce same IDs
        assert id1 == id2

        # Different configs should produce different IDs
        assert id1 != id3

        # IDs should be valid SHA-256 hex strings
        assert len(id1) == 64  # SHA-256 produces 64 character hex strings
        assert all(c in "0123456789abcdef" for c in id1)

    def test_key_order_independence(self):
        """Test that dictionary key order does not affect hash ID.

        Purpose: Validates that dictionaries with same content but different key order produce same hash

        Given: Two dictionaries with identical content but different key insertion order
        When: config_to_id is called on both dictionaries
        Then: Both produce identical hash IDs

        Test type: unit
        """
        config1 = {"b": 2, "a": 1, "c": 3}
        config2 = {"a": 1, "b": 2, "c": 3}
        config3 = {"c": 3, "a": 1, "b": 2}

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)
        id3 = config_to_id(config3)

        assert id1 == id2 == id3

    def test_numpy_array_handling(self):
        """Test config_to_id with NumPy arrays in configuration.

        Purpose: Validates that NumPy arrays are properly handled in configuration hashing

        Given: Configurations containing various NumPy array types
        When: config_to_id is called on the configurations
        Then: Arrays are properly serialized and consistent hashes are produced

        Test type: unit
        """
        config1 = {
            "array_1d": np.array([1, 2, 3]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "param": "value",
        }

        config2 = {
            "array_1d": np.array([1, 2, 3]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "param": "value",
        }

        config3 = {
            "array_1d": np.array([1, 2, 4]),  # Different array
            "array_2d": np.array([[1, 2], [3, 4]]),
            "param": "value",
        }

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)
        id3 = config_to_id(config3)

        # Same arrays should produce same ID
        assert id1 == id2

        # Different arrays should produce different ID
        assert id1 != id3

    def test_numpy_scalar_handling(self):
        """Test config_to_id with NumPy scalar types in configuration.

        Purpose: Validates that NumPy scalars are properly handled in configuration hashing

        Given: Configurations containing various NumPy scalar types
        When: config_to_id is called on the configurations
        Then: Scalars are properly serialized and consistent hashes are produced

        Test type: unit
        """
        config1 = {
            "int_param": np.int32(42),
            "float_param": np.float64(3.14),
            "regular_param": "value",
        }

        config2 = {
            "int_param": np.int32(42),
            "float_param": np.float64(3.14),
            "regular_param": "value",
        }

        # Convert NumPy scalars to regular Python types for comparison
        config3 = {
            "int_param": 42,  # Regular Python int
            "float_param": 3.14,  # Regular Python float
            "regular_param": "value",
        }

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)
        id3 = config_to_id(config3)

        # Same NumPy scalars should produce same ID
        assert id1 == id2

        # NumPy scalars should produce same ID as equivalent Python types
        assert id1 == id3

    def test_nested_dict_handling(self):
        """Test config_to_id with nested dictionary structures.

        Purpose: Validates that nested dictionaries are properly handled in configuration hashing

        Given: Configurations with nested dictionary structures
        When: config_to_id is called on the configurations
        Then: Nested structures are properly serialized and consistent hashes are produced

        Test type: unit
        """
        config1 = {
            "level1": {
                "level2": {"param": "value", "array": np.array([1, 2, 3])},
                "simple_param": 42,
            },
            "top_level": "test",
        }

        config2 = {
            "level1": {
                "level2": {"param": "value", "array": np.array([1, 2, 3])},
                "simple_param": 42,
            },
            "top_level": "test",
        }

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)

        assert id1 == id2

    def test_empty_dict_handling(self):
        """Test config_to_id with empty dictionary.

        Purpose: Validates that empty dictionaries produce consistent hash IDs

        Given: Empty dictionary configuration
        When: config_to_id is called on the empty dictionary
        Then: Consistent hash ID is produced

        Test type: unit
        """
        config1 = {}
        config2 = {}

        id1 = config_to_id(config1)
        id2 = config_to_id(config2)

        assert id1 == id2
        assert len(id1) == 64  # Valid SHA-256 hex string

    def test_mixed_data_types(self):
        """Test config_to_id with mixed data types including edge cases.

        Purpose: Validates proper handling of configurations with diverse data types

        Given: Configuration containing strings, numbers, lists, None, and boolean values
        When: config_to_id is called on the configuration
        Then: All types are properly serialized and consistent hash is produced

        Test type: unit
        """
        config = {
            "string": "test_string",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "none_value": None,
            "list": [1, 2, "three"],
            "numpy_array": np.array([10, 20, 30]),
            "nested": {"inner": "value", "inner_array": np.array([[1, 2], [3, 4]])},
        }

        # Should not raise any exceptions
        config_id = config_to_id(config)

        # Should be valid SHA-256 hex string
        assert len(config_id) == 64
        assert all(c in "0123456789abcdef" for c in config_id)

        # Should be reproducible
        assert config_id == config_to_id(config)

    def test_deterministic_hashing(self):
        """Test that config_to_id produces deterministic results across multiple calls.

        Purpose: Validates that identical configurations always produce identical hash IDs

        Given: The same configuration dictionary used multiple times
        When: config_to_id is called repeatedly on the same configuration
        Then: All calls produce identical hash IDs

        Test type: unit
        """
        config = {
            "param1": "value1",
            "param2": np.array([1, 2, 3, 4, 5]),
            "param3": {"nested": "value", "number": 42},
        }

        # Generate multiple IDs
        ids = [config_to_id(config) for _ in range(10)]

        # All should be identical
        assert len(set(ids)) == 1  # Only one unique ID

    def test_hash_collision_resistance(self):
        """Test that similar configurations produce different hash IDs.

        Purpose: Validates that small changes in configuration produce different hash IDs

        Given: Multiple configurations with small differences
        When: config_to_id is called on each configuration
        Then: Each configuration produces a unique hash ID

        Test type: unit
        """
        base_config = {"param": "value", "number": 42}

        configs = [
            base_config,
            {"param": "value", "number": 43},  # Different number
            {"param": "VALUE", "number": 42},  # Different case
            {"param": "value", "number": 42, "extra": None},  # Extra parameter
            {"param": "value", "number": 42.0},  # Different type (int vs float)
        ]

        ids = [config_to_id(config) for config in configs]

        # All IDs should be different
        assert len(set(ids)) == len(ids)


class TestConfigToIdIntegration:
    """Integration tests for config_to_id with real POMDP components."""

    def test_cartpole_environment_config_id_consistency(self):
        """Test that CartPole POMDP environments produce consistent config IDs.

        Purpose: Validates that identical CartPole POMDP configurations produce identical config IDs

        Given: Multiple CartPole POMDP instances with identical parameters
        When: config_to_id is called on their configuration dictionaries
        Then: All instances produce identical config IDs

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

        # Create noise covariance matrix
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])

        # Create multiple identical environments
        env1 = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)
        env2 = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov.copy())

        # Extract configuration dictionaries
        config1 = {
            "discount_factor": env1.discount_factor,
            "noise_cov": env1.noise_cov,
            "gravity": env1.gravity,
            "masscart": env1.masscart,
            "masspole": env1.masspole,
            "force_mag": env1.force_mag,
        }

        config2 = {
            "discount_factor": env2.discount_factor,
            "noise_cov": env2.noise_cov,
            "gravity": env2.gravity,
            "masscart": env2.masscart,
            "masspole": env2.masspole,
            "force_mag": env2.force_mag,
        }

        # Generate config IDs
        id1 = config_to_id(config1)
        id2 = config_to_id(config2)

        # Should be identical
        assert id1 == id2
        assert len(id1) == 64  # Valid SHA-256 hex string

    def test_cartpole_environment_different_configs_produce_different_ids(self):
        """Test that different CartPole configurations produce different config IDs.

        Purpose: Validates that CartPole environments with different parameters produce different config IDs

        Given: CartPole POMDP instances with different parameters
        When: config_to_id is called on their configuration dictionaries
        Then: Each configuration produces a unique config ID

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

        # Create different configurations
        configs = []

        # Configuration 1: Base configuration
        noise_cov1 = np.diag([0.1, 0.1, 0.1, 0.1])
        env1 = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov1)
        configs.append(
            {
                "discount_factor": env1.discount_factor,
                "noise_cov": env1.noise_cov,
                "force_mag": env1.force_mag,
            }
        )

        # Configuration 2: Different discount factor
        noise_cov2 = np.diag([0.1, 0.1, 0.1, 0.1])
        env2 = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov2)
        configs.append(
            {
                "discount_factor": env2.discount_factor,
                "noise_cov": env2.noise_cov,
                "force_mag": env2.force_mag,
            }
        )

        # Configuration 3: Different noise covariance
        noise_cov3 = np.diag([0.2, 0.2, 0.2, 0.2])
        env3 = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov3)
        configs.append(
            {
                "discount_factor": env3.discount_factor,
                "noise_cov": env3.noise_cov,
                "force_mag": env3.force_mag,
            }
        )

        # Generate config IDs
        ids = [config_to_id(config) for config in configs]

        # All IDs should be different
        assert len(set(ids)) == len(ids)
        assert all(len(config_id) == 64 for config_id in ids)

    def test_pomcp_planner_config_id_consistency(self):
        """Test that POMCP planners produce consistent config IDs.

        Purpose: Validates that identical POMCP configurations produce identical config IDs

        Given: Multiple POMCP planner instances with identical parameters
        When: config_to_id is called on their configuration dictionaries
        Then: All instances produce identical config IDs

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

        # Create environment
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

        # Create multiple identical planners
        planner1 = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=1.0,
            name="TestPlanner",
            n_simulations=100,
        )

        planner2 = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=1.0,
            name="TestPlanner",
            n_simulations=100,
        )

        # Extract configuration dictionaries
        config1 = {
            "discount_factor": planner1.discount_factor,
            "depth": planner1.depth,
            "exploration_constant": planner1.exploration_constant,
            "n_simulations": planner1.n_simulations,
            "min_samples_per_node": planner1.min_samples_per_node,
        }

        config2 = {
            "discount_factor": planner2.discount_factor,
            "depth": planner2.depth,
            "exploration_constant": planner2.exploration_constant,
            "n_simulations": planner2.n_simulations,
            "min_samples_per_node": planner2.min_samples_per_node,
        }

        # Generate config IDs
        id1 = config_to_id(config1)
        id2 = config_to_id(config2)

        # Should be identical
        assert id1 == id2
        assert len(id1) == 64

    def test_weighted_particle_belief_config_id_consistency(self):
        """Test that WeightedParticleBelief instances produce consistent config IDs.

        Purpose: Validates that identical weighted particle beliefs produce identical config IDs

        Given: Multiple WeightedParticleBelief instances with identical particles and weights
        When: config_id property is accessed
        Then: All instances produce identical config IDs

        Test type: integration
        """
        from POMDPPlanners.core.belief import WeightedParticleBelief

        # Create particles and weights
        particles1 = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        particles2 = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        log_weights1 = np.log(np.array([0.5, 0.3, 0.2]))
        log_weights2 = np.log(np.array([0.5, 0.3, 0.2]))

        # Create beliefs
        belief1 = WeightedParticleBelief(
            particles=particles1, log_weights=log_weights1, resampling=True, ess_factor=0.5
        )

        belief2 = WeightedParticleBelief(
            particles=particles2, log_weights=log_weights2, resampling=True, ess_factor=0.5
        )

        # Test config_id consistency
        id1 = belief1.config_id
        id2 = belief2.config_id

        assert id1 == id2
        assert len(id1) == 64

    def test_weighted_particle_belief_different_configs_produce_different_ids(self):
        """Test that different WeightedParticleBelief configurations produce different config IDs.

        Purpose: Validates that weighted particle beliefs with different configurations produce different config IDs

        Given: WeightedParticleBelief instances with different particles, weights, or parameters
        When: config_id property is accessed
        Then: Each configuration produces a unique config ID

        Test type: integration
        """
        from POMDPPlanners.core.belief import WeightedParticleBelief

        beliefs = []

        # Belief 1: Base configuration
        particles1 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        log_weights1 = np.log(np.array([0.6, 0.4]))
        belief1 = WeightedParticleBelief(
            particles=particles1, log_weights=log_weights1, resampling=True, ess_factor=0.5
        )
        beliefs.append(belief1)

        # Belief 2: Different particles
        particles2 = [np.array([1.5, 2.5]), np.array([3.0, 4.0])]
        log_weights2 = np.log(np.array([0.6, 0.4]))
        belief2 = WeightedParticleBelief(
            particles=particles2, log_weights=log_weights2, resampling=True, ess_factor=0.5
        )
        beliefs.append(belief2)

        # Belief 3: Different weights
        particles3 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        log_weights3 = np.log(np.array([0.7, 0.3]))
        belief3 = WeightedParticleBelief(
            particles=particles3, log_weights=log_weights3, resampling=True, ess_factor=0.5
        )
        beliefs.append(belief3)

        # Belief 4: Different resampling parameter
        particles4 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        log_weights4 = np.log(np.array([0.6, 0.4]))
        belief4 = WeightedParticleBelief(
            particles=particles4, log_weights=log_weights4, resampling=False, ess_factor=0.5
        )
        beliefs.append(belief4)

        # Generate config IDs
        ids = [belief.config_id for belief in beliefs]

        # All IDs should be different
        assert len(set(ids)) == len(ids)
        assert all(len(config_id) == 64 for config_id in ids)

    def test_complex_pomdp_configuration_with_numpy_arrays(self):
        """Test config_to_id with complex POMDP configuration containing NumPy arrays.

        Purpose: Validates that complex POMDP configurations with nested NumPy arrays produce consistent hashes

        Given: A complex configuration dictionary with CartPole environment, POMCP planner, and belief parameters
        When: config_to_id is called on the configuration
        Then: A consistent hash ID is produced that handles all NumPy arrays correctly

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

        # Create comprehensive POMDP configuration
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])

        complex_config = {
            "environment": {
                "type": "CartPolePOMDP",
                "discount_factor": 0.99,
                "noise_covariance": noise_cov,
                "physics_params": {
                    "gravity": 9.8,
                    "masscart": 1.0,
                    "masspole": 0.1,
                    "force_magnitude": 10.0,
                    "time_step": 0.02,
                },
            },
            "planner": {
                "type": "POMCP",
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 1.41,
                "n_simulations": 1000,
                "min_samples_per_node": 5,
            },
            "belief": {
                "type": "WeightedParticleBelief",
                "n_particles": 100,
                "initial_particles": np.random.randn(5, 4),  # 5 particles, 4D state
                "log_weights": np.log(np.ones(5) / 5),
                "resampling_params": {
                    "enable": True,
                    "ess_factor": 0.5,
                    "reinvigoration_fraction": 0.1,
                },
            },
            "simulation": {
                "episodes": 50,
                "max_steps": 200,
                "random_seed": 42,
                "initial_states": np.array([[0.0, 0.0, 0.1, 0.0], [0.1, 0.0, 0.0, 0.0]]),
            },
        }

        # Generate config ID
        config_id = config_to_id(complex_config)

        # Verify it's a valid hash
        assert len(config_id) == 64
        assert all(c in "0123456789abcdef" for c in config_id)

        # Test reproducibility
        config_id_repeat = config_to_id(complex_config)
        assert config_id == config_id_repeat

    def test_config_to_id_with_objects_having_config_id_attribute(self):
        """Test config_to_id with objects that have config_id attributes (environments, beliefs, etc.).

        Purpose: Validates that objects with config_id attributes are properly serialized using their config_id

        Given: A configuration containing POMDP objects with config_id attributes
        When: config_to_id is called on the configuration
        Then: Objects are serialized using their config_id rather than their full state

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        from POMDPPlanners.core.belief import WeightedParticleBelief

        # Create POMDP components with config_id attributes
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

        particles = [np.array([0.0, 0.0, 0.1, 0.0]), np.array([0.1, 0.0, 0.0, 0.0])]
        log_weights = np.log(np.array([0.5, 0.5]))
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

        # Create configuration with objects
        config_with_objects = {
            "environment": env,
            "belief": belief,
            "params": {"discount": 0.95, "noise_matrix": noise_cov, "time_horizon": 100},
        }

        # Generate config ID (should work despite complex objects)
        config_id = config_to_id(config_with_objects)

        # Verify it's a valid hash
        assert len(config_id) == 64
        assert all(c in "0123456789abcdef" for c in config_id)

        # Test that it's deterministic
        config_id_repeat = config_to_id(config_with_objects)
        assert config_id == config_id_repeat

    def test_config_to_id_order_invariance_with_pomdp_objects(self):
        """Test that config_to_id is invariant to dictionary key order with POMDP objects.

        Purpose: Validates that dictionary key order does not affect config_to_id with complex POMDP objects

        Given: Two dictionaries with identical POMDP content but different key ordering
        When: config_to_id is called on both dictionaries
        Then: Both produce identical config IDs

        Test type: integration
        """
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        env = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)

        # Configuration 1: Keys in one order
        config1 = {
            "environment": env,
            "planner_params": {"depth": 10, "exploration": 1.0},
            "noise_covariance": noise_cov,
            "discount_factor": 0.95,
            "simulation_steps": 1000,
        }

        # Configuration 2: Same content, different key order
        config2 = {
            "discount_factor": 0.95,
            "simulation_steps": 1000,
            "environment": env,
            "noise_covariance": noise_cov,
            "planner_params": {"depth": 10, "exploration": 1.0},
        }

        # Generate config IDs
        id1 = config_to_id(config1)
        id2 = config_to_id(config2)

        # Should be identical despite different ordering
        assert id1 == id2
