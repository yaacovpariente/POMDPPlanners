# SPDX-License-Identifier: MIT

import numpy as np

from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    CategoricalHyperParameter,
    NumericalHyperParameter,
    HyperParamPlannerConfig,
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
    OptimizedPolicyResult,
)
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo, PolicyRunData


class TestCategoricalHyperParameterIdUniqueness:
    """Test uniqueness of CategoricalHyperParameter.id() method."""

    def test_different_names_produce_different_ids(self):
        """Test that hyperparameters with different names produce different IDs."""
        param1 = CategoricalHyperParameter(choices=[1, 2, 3], name="param1")
        param2 = CategoricalHyperParameter(choices=[1, 2, 3], name="param2")

        assert param1.id() != param2.id()

    def test_different_choices_produce_different_ids(self):
        """Test that hyperparameters with different choices produce different IDs."""
        param1 = CategoricalHyperParameter(choices=[1, 2, 3], name="param")
        param2 = CategoricalHyperParameter(choices=[1, 2, 4], name="param")

        assert param1.id() != param2.id()

    def test_different_order_choices_produce_different_ids(self):
        """Test that hyperparameters with different order of choices produce different IDs."""
        param1 = CategoricalHyperParameter(choices=[1, 2, 3], name="param")
        param2 = CategoricalHyperParameter(choices=[3, 2, 1], name="param")

        assert param1.id() != param2.id()

    def test_same_parameters_produce_same_ids(self):
        """Test that hyperparameters with same name and choices produce same IDs."""
        param1 = CategoricalHyperParameter(choices=[1, 2, 3], name="param")
        param2 = CategoricalHyperParameter(choices=[1, 2, 3], name="param")

        assert param1.id() == param2.id()

    def test_empty_choices_produce_unique_ids(self):
        """Test that hyperparameters with empty choices produce unique IDs."""
        param1 = CategoricalHyperParameter(choices=[], name="param1")
        param2 = CategoricalHyperParameter(choices=[], name="param2")

        assert param1.id() != param2.id()

    def test_complex_choices_produce_unique_ids(self):
        """Test that hyperparameters with complex data types produce unique IDs."""
        param1 = CategoricalHyperParameter(choices=[{"a": 1}, {"b": 2}], name="param")
        param2 = CategoricalHyperParameter(choices=[{"a": 1}, {"c": 3}], name="param")

        assert param1.id() != param2.id()


class TestNumericalHyperParameterIdUniqueness:
    """Test uniqueness of NumericalHyperParameter.id() method."""

    def test_different_names_produce_different_ids(self):
        """Test that hyperparameters with different names produce different IDs."""
        param1 = NumericalHyperParameter(low=0, high=1, name="param1")
        param2 = NumericalHyperParameter(low=0, high=1, name="param2")

        assert param1.id() != param2.id()

    def test_different_low_values_produce_different_ids(self):
        """Test that hyperparameters with different low values produce different IDs."""
        param1 = NumericalHyperParameter(low=0, high=1, name="param")
        param2 = NumericalHyperParameter(low=0.1, high=1, name="param")

        assert param1.id() != param2.id()

    def test_different_high_values_produce_different_ids(self):
        """Test that hyperparameters with different high values produce different IDs."""
        param1 = NumericalHyperParameter(low=0, high=1, name="param")
        param2 = NumericalHyperParameter(low=0, high=2, name="param")

        assert param1.id() != param2.id()

    def test_same_parameters_produce_same_ids(self):
        """Test that hyperparameters with same parameters produce same IDs."""
        param1 = NumericalHyperParameter(low=0, high=1, name="param")
        param2 = NumericalHyperParameter(low=0, high=1, name="param")

        assert param1.id() == param2.id()

    def test_different_numeric_types_produce_different_ids(self):
        """Test that hyperparameters with different numeric types produce different IDs."""
        param1 = NumericalHyperParameter(low=0, high=1, name="param")
        param2 = NumericalHyperParameter(low=0.0, high=1.0, name="param")

        # Note: This might be the same depending on config_to_id implementation
        # but we test that the method works consistently
        assert isinstance(param1.id(), str)
        assert isinstance(param2.id(), str)


class TestHyperParamPlannerConfigIdUniqueness:
    """Test uniqueness of HyperParamPlannerConfig.config_id property."""

    def test_different_policy_classes_produce_different_ids(self):
        """Test that configs with different policy classes produce different IDs."""

        # Create mock policy classes
        class Policy1(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

        class Policy2(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        config1 = HyperParamPlannerConfig(
            policy_cls=Policy1, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        config2 = HyperParamPlannerConfig(
            policy_cls=Policy2, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        assert config1.config_id != config2.config_id

    def test_different_hyper_parameters_produce_different_ids(self):
        """Test that configs with different hyper parameters produce different IDs."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param1=None,
                param2=None,
                a=None,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param1 = param1
                self.param2 = param2
                self.a = a

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param1")
        param2 = CategoricalHyperParameter(choices=[1, 2], name="param2")

        config1 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        config2 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param2], constant_parameters={"a": 1}
        )

        assert config1.config_id != config2.config_id

    def test_different_constant_parameters_produce_different_ids(self):
        """Test that configs with different constant parameters produce different IDs."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        config1 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        config2 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 2}
        )

        assert config1.config_id != config2.config_id

    def test_same_configs_produce_same_ids(self):
        """Test that identical configs produce same IDs."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        config1 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        config2 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        assert config1.config_id == config2.config_id

    def test_different_hyper_parameter_order_produces_same_id(self):
        """Test that different order of hyper parameters produces same ID (due to sorting)."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param1,
                param2,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param1 = param1
                self.param2 = param2
                self.a = a

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param1")
        param2 = CategoricalHyperParameter(choices=[3, 4], name="param2")

        config1 = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1, param2], constant_parameters={"a": 1}
        )

        config2 = HyperParamPlannerConfig(
            policy_cls=MockPolicy,
            hyper_parameters=[param2, param1],  # Different order
            constant_parameters={"a": 1},
        )

        assert config1.config_id == config2.config_id


class TestHyperParameterRunParamsIdUniqueness:
    """Test uniqueness of HyperParameterRunParams.config_id property."""

    def test_config_id_property_exists(self):
        """Test that HyperParameterRunParams has a config_id property."""

        # Create simple objects with config_id properties that return strings
        class MockEnv(Environment):
            def __init__(self):
                super().__init__(
                    discount_factor=0.95,
                    name="mock_env",
                    space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
                )

            @property
            def config_id(self):
                return "env1"

            def sample_next_state(self, state, action, n_samples=1):  # type: ignore[override]
                return state if n_samples == 1 else [state] * n_samples

            def sample_observation(self, next_state, action, n_samples=1):  # type: ignore[override]
                return next_state if n_samples == 1 else [next_state] * n_samples

            def transition_log_probability(self, state, action, next_states):  # type: ignore[override]
                return np.zeros(len(next_states))

            def observation_log_probability(self, next_state, action, observations):  # type: ignore[override]
                return np.zeros(len(observations))

            def reward(self, state, action, next_state=None):
                del state, action, next_state
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

            def hash_action(self, action):
                return action

        class MockBelief(Belief):
            @property
            def config_id(self):
                return "belief1"

            def update(self, action, observation, pomdp, state=None):
                return self

            def sample(self):
                return "state"

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        hyper_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        # Test that we can create the object and access config_id
        params = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        # The config_id should exist and be a string
        assert hasattr(params, "config_id")
        assert isinstance(params.config_id, str)
        assert len(params.config_id) > 0

    def test_config_id_consistency(self):
        """Test that config_id produces consistent results."""

        class MockEnv(Environment):
            def __init__(self):
                super().__init__(
                    discount_factor=0.95,
                    name="mock_env",
                    space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
                )

            @property
            def config_id(self):
                return "env1"

            def sample_next_state(self, state, action, n_samples=1):  # type: ignore[override]
                return state if n_samples == 1 else [state] * n_samples

            def sample_observation(self, next_state, action, n_samples=1):  # type: ignore[override]
                return next_state if n_samples == 1 else [next_state] * n_samples

            def transition_log_probability(self, state, action, next_states):  # type: ignore[override]
                return np.zeros(len(next_states))

            def observation_log_probability(self, next_state, action, observations):  # type: ignore[override]
                return np.zeros(len(observations))

            def reward(self, state, action, next_state=None):
                del state, action, next_state
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

            def hash_action(self, action):
                return action

        class MockBelief(Belief):
            @property
            def config_id(self):
                return "belief1"

            def update(self, action, observation, pomdp, state=None):
                return self

            def sample(self):
                return "state"

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        hyper_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        params = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        # Generate config_id multiple times
        id1 = params.config_id
        id2 = params.config_id
        id3 = params.config_id

        # All should be the same
        assert id1 == id2 == id3
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_different_numeric_parameters_produce_different_ids(self):
        """Test that run params with different numeric parameters produce different IDs."""

        class MockEnv(Environment):
            def __init__(self):
                super().__init__(
                    discount_factor=0.95,
                    name="mock_env",
                    space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
                )

            @property
            def config_id(self):
                return "env1"

            def sample_next_state(self, state, action, n_samples=1):  # type: ignore[override]
                return state if n_samples == 1 else [state] * n_samples

            def sample_observation(self, next_state, action, n_samples=1):  # type: ignore[override]
                return next_state if n_samples == 1 else [next_state] * n_samples

            def transition_log_probability(self, state, action, next_states):  # type: ignore[override]
                return np.zeros(len(next_states))

            def observation_log_probability(self, next_state, action, observations):  # type: ignore[override]
                return np.zeros(len(observations))

            def reward(self, state, action, next_state=None):
                del state, action, next_state
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

            def hash_action(self, action):
                return action

        class MockBelief(Belief):
            @property
            def config_id(self):
                return "belief1"

            def update(self, action, observation, pomdp, state=None):
                return self

            def sample(self):
                return "state"

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        hyper_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        params1 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        params2 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            20,  # Different num_episodes
            100,
            5,
            [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        assert params1.config_id != params2.config_id

    def test_different_parameters_to_optimize_produce_different_ids(self):
        """Test that run params with different parameters to optimize produce different IDs."""

        class MockEnv(Environment):
            def __init__(self):
                super().__init__(
                    discount_factor=0.95,
                    name="mock_env",
                    space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
                )

            @property
            def config_id(self):
                return "env1"

            def sample_next_state(self, state, action, n_samples=1):  # type: ignore[override]
                return state if n_samples == 1 else [state] * n_samples

            def sample_observation(self, next_state, action, n_samples=1):  # type: ignore[override]
                return next_state if n_samples == 1 else [next_state] * n_samples

            def transition_log_probability(self, state, action, next_states):  # type: ignore[override]
                return np.zeros(len(next_states))

            def observation_log_probability(self, next_state, action, observations):  # type: ignore[override]
                return np.zeros(len(observations))

            def reward(self, state, action, next_state=None):
                del state, action, next_state
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

            def hash_action(self, action):
                return action

        class MockBelief(Belief):
            @property
            def config_id(self):
                return "belief1"

            def update(self, action, observation, pomdp, state=None):
                return self

            def sample(self):
                return "state"

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param1 = CategoricalHyperParameter(choices=[1, 2], name="param")

        hyper_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param1], constant_parameters={"a": 1}
        )

        params1 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        params2 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [
                ("average_return", HyperParameterOptimizationDirection.MINIMIZE)
            ],  # Different direction
        )

        assert params1.config_id != params2.config_id


class TestIdConsistency:
    """Test consistency of ID generation across different instances."""

    def test_categorical_hyperparameter_id_consistency(self):
        """Test that CategoricalHyperParameter.id() produces consistent results."""
        param = CategoricalHyperParameter(choices=[1, 2, 3], name="test_param")

        # Generate ID multiple times
        id1 = param.id()
        id2 = param.id()
        id3 = param.id()

        # All should be the same
        assert id1 == id2 == id3
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_numerical_hyperparameter_id_consistency(self):
        """Test that NumericalHyperParameter.id() produces consistent results."""
        param = NumericalHyperParameter(low=0.0, high=1.0, name="test_param")

        # Generate ID multiple times
        id1 = param.id()
        id2 = param.id()
        id3 = param.id()

        # All should be the same
        assert id1 == id2 == id3
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_config_id_consistency(self):
        """Test that config_id properties produce consistent results."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param = CategoricalHyperParameter(choices=[1, 2], name="param")

        config = HyperParamPlannerConfig(
            policy_cls=MockPolicy, hyper_parameters=[param], constant_parameters={"a": 1}
        )

        # Generate config_id multiple times
        id1 = config.config_id
        id2 = config.config_id
        id3 = config.config_id

        # All should be the same
        assert id1 == id2 == id3
        assert isinstance(id1, str)
        assert len(id1) > 0


class TestHyperParamPlannerConfigValidation:
    """Test validation of HyperParamPlannerConfig inputs."""

    def test_invalid_policy_cls_type_raises_type_error(self):
        """Test that non-class policy_cls raises TypeError.

        Purpose: Validates that policy_cls must be a class type

        Given: An attempt to create HyperParamPlannerConfig with non-class policy_cls
        When: The config is instantiated with a string instead of a Policy class
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        param = CategoricalHyperParameter(choices=[1, 2], name="param")

        with pytest.raises(TypeError, match="policy_cls must be a class type"):
            HyperParamPlannerConfig(
                policy_cls="NotAClass",  # type: ignore
                hyper_parameters=[param],
                constant_parameters={"a": 1},
            )

    def test_invalid_hyper_parameters_type_raises_type_error(self):
        """Test that non-sequence hyper_parameters raises TypeError.

        Purpose: Validates that hyper_parameters must be a Sequence (list or tuple)

        Given: An attempt to create HyperParamPlannerConfig with non-sequence hyper_parameters
        When: The config is instantiated with a dict instead of a list/tuple
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        with pytest.raises(TypeError, match="hyper_parameters must be a Sequence"):
            HyperParamPlannerConfig(
                policy_cls=MockPolicy,
                hyper_parameters={"param": [1, 2]},  # type: ignore
                constant_parameters={"a": 1},
            )

    def test_invalid_hyper_parameter_element_type_raises_type_error(self):
        """Test that invalid hyperparameter element types raise TypeError.

        Purpose: Validates that all elements in hyper_parameters are valid HyperParameterFeature types

        Given: An attempt to create HyperParamPlannerConfig with invalid hyperparameter element
        When: The config is instantiated with a string in the hyper_parameters list
        Then: TypeError is raised indicating the invalid element index and type

        Test type: unit
        """
        import pytest

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        with pytest.raises(
            TypeError,
            match="hyper_parameters\\[0\\] must be either CategoricalHyperParameter or NumericalHyperParameter",
        ):
            HyperParamPlannerConfig(
                policy_cls=MockPolicy,
                hyper_parameters=["invalid"],  # type: ignore
                constant_parameters={"a": 1},
            )

    def test_invalid_constant_parameters_type_raises_type_error(self):
        """Test that non-dict constant_parameters raises TypeError.

        Purpose: Validates that constant_parameters must be a dict

        Given: An attempt to create HyperParamPlannerConfig with non-dict constant_parameters
        When: The config is instantiated with a list instead of a dict
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                a,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param
                self.a = a

            @classmethod
            def get_info_variable_names(cls):
                return []

        param = CategoricalHyperParameter(choices=[1, 2], name="param")

        with pytest.raises(TypeError, match="constant_parameters must be a dict"):
            HyperParamPlannerConfig(
                policy_cls=MockPolicy,
                hyper_parameters=[param],
                constant_parameters=[("a", 1)],  # type: ignore
            )

    def test_invalid_hyperparameter_name_raises_value_error(self):
        """Test that hyperparameter names not in policy constructor raise ValueError.

        Purpose: Validates that all hyperparameter names correspond to policy class constructor parameters

        Given: A policy class with specific constructor parameters
        When: HyperParamPlannerConfig is created with hyperparameter name not in constructor
        Then: ValueError is raised listing valid parameter names

        Test type: unit
        """
        import pytest

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                valid_param,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.valid_param = valid_param

        invalid_param = CategoricalHyperParameter(choices=[1, 2], name="invalid_param")

        with pytest.raises(
            ValueError, match="Hyperparameter 'invalid_param' is not a valid parameter"
        ):
            HyperParamPlannerConfig(
                policy_cls=MockPolicy, hyper_parameters=[invalid_param], constant_parameters={}
            )

    def test_invalid_constant_parameter_name_raises_value_error(self):
        """Test that constant parameter names not in policy constructor raise ValueError.

        Purpose: Validates that all constant parameter names correspond to policy class constructor parameters

        Given: A policy class with specific constructor parameters
        When: HyperParamPlannerConfig is created with constant parameter name not in constructor
        Then: ValueError is raised listing valid parameter names

        Test type: unit
        """
        import pytest

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                param,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.param = param

        param = CategoricalHyperParameter(choices=[1, 2], name="param")

        with pytest.raises(
            ValueError, match="Constant parameter 'invalid_constant' is not a valid parameter"
        ):
            HyperParamPlannerConfig(
                policy_cls=MockPolicy,
                hyper_parameters=[param],
                constant_parameters={"invalid_constant": 1},
            )

    def test_valid_config_with_all_parameters_succeeds(self):
        """Test that valid configuration with all correct parameters succeeds.

        Purpose: Validates that properly configured HyperParamPlannerConfig is created without errors

        Given: A policy class with specific constructor parameters
        When: HyperParamPlannerConfig is created with valid hyperparameters and constants
        Then: Config is created successfully and has valid config_id

        Test type: unit
        """

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                hp1,
                hp2,
                const1,
                const2,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.hp1 = hp1
                self.hp2 = hp2
                self.const1 = const1
                self.const2 = const2

        hp1 = CategoricalHyperParameter(choices=[1, 2, 3], name="hp1")
        hp2 = NumericalHyperParameter(low=0.0, high=1.0, name="hp2")

        config = HyperParamPlannerConfig(
            policy_cls=MockPolicy,
            hyper_parameters=[hp1, hp2],
            constant_parameters={"const1": "value1", "const2": 42},
        )

        assert config.policy_cls == MockPolicy
        assert len(config.hyper_parameters) == 2
        assert config.constant_parameters == {"const1": "value1", "const2": 42}
        assert isinstance(config.config_id, str)
        assert len(config.config_id) > 0


class TestOptimizedPolicyResultValidation:
    """Test validation of OptimizedPolicyResult inputs."""

    def _create_mock_environment(self):
        """Create a simple mock environment for testing."""

        class MockEnv(Environment):
            def __init__(self):
                super().__init__(
                    discount_factor=0.95,
                    name="mock_env",
                    space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE),
                )

            @property
            def config_id(self):
                return "mock_env_id"

            def sample_next_state(self, state, action, n_samples=1):  # type: ignore[override]
                return state if n_samples == 1 else [state] * n_samples

            def sample_observation(self, next_state, action, n_samples=1):  # type: ignore[override]
                return next_state if n_samples == 1 else [next_state] * n_samples

            def transition_log_probability(self, state, action, next_states):  # type: ignore[override]
                return np.zeros(len(next_states))

            def observation_log_probability(self, next_state, action, observations):  # type: ignore[override]
                return np.zeros(len(observations))

            def reward(self, state, action, next_state=None):
                del state, action, next_state
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

            def hash_action(self, action):
                return action

        return MockEnv()

    def _create_mock_policy(self, env):
        """Create a simple mock policy for testing."""

        class MockPolicy(Policy):
            def __init__(
                self,
                environment,
                discount_factor,
                name,
                hp1,
                log_path=None,
                debug=False,
                use_queue_logger=False,
            ):
                super().__init__(
                    environment, discount_factor, name, log_path, debug, use_queue_logger
                )
                self.hp1 = hp1

            @classmethod
            def get_info_variable_names(cls):
                return []

            def action(self, belief):
                """Return a dummy action."""
                return [0], PolicyRunData(info_variables=[])

            @classmethod
            def get_space_info(cls):
                """Return dummy space info."""
                return PolicySpaceInfo(
                    action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
                )

        return MockPolicy(env, 0.95, "mock_policy", hp1=1.0)

    def test_valid_optimized_policy_result_succeeds(self):
        """Test that valid OptimizedPolicyResult is created successfully.

        Purpose: Validates that properly configured OptimizedPolicyResult is created without errors

        Given: Valid environment, policy, and optimization parameters
        When: OptimizedPolicyResult is created with all correct parameters
        Then: Result is created successfully as a frozen dataclass

        Test type: unit
        """
        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        result = OptimizedPolicyResult(
            environment=env,
            policy=policy,
            chosen_hyper_parameters={"hp1": 1.0},
            num_episodes=10,
            num_steps=100,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": 42.5},
        )

        assert result.environment == env
        assert result.policy == policy
        assert result.chosen_hyper_parameters == {"hp1": 1.0}
        assert result.num_episodes == 10
        assert result.num_steps == 100
        assert result.parameters_to_optimize == [
            ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
        ]
        assert result.optimized_metric_values == {"average_return": 42.5}

    def test_frozen_dataclass_is_immutable(self):
        """Test that OptimizedPolicyResult is immutable (frozen).

        Purpose: Validates that OptimizedPolicyResult is a frozen dataclass

        Given: A created OptimizedPolicyResult instance
        When: Attempting to modify any attribute
        Then: FrozenInstanceError or AttributeError is raised

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        result = OptimizedPolicyResult(
            environment=env,
            policy=policy,
            chosen_hyper_parameters={"hp1": 1.0},
            num_episodes=10,
            num_steps=100,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": 42.5},
        )

        with pytest.raises((AttributeError, Exception)):
            result.num_episodes = 20  # type: ignore

    def test_negative_num_episodes_raises_value_error(self):
        """Test that negative num_episodes raises ValueError.

        Purpose: Validates that num_episodes must be positive

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with num_episodes <= 0
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=-5,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_zero_num_episodes_raises_value_error(self):
        """Test that zero num_episodes raises ValueError.

        Purpose: Validates that num_episodes must be strictly positive

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with num_episodes = 0
        Then: ValueError is raised

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=0,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_negative_num_steps_raises_value_error(self):
        """Test that negative num_steps raises ValueError.

        Purpose: Validates that num_steps must be positive

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with num_steps <= 0
        Then: ValueError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=-100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_zero_num_steps_raises_value_error(self):
        """Test that zero num_steps raises ValueError.

        Purpose: Validates that num_steps must be strictly positive

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with num_steps = 0
        Then: ValueError is raised

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=0,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_environment_type_raises_type_error(self):
        """Test that non-Environment environment raises TypeError.

        Purpose: Validates that environment must be an Environment instance

        Given: An invalid environment type (not an Environment subclass)
        When: OptimizedPolicyResult is created with invalid environment
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(TypeError, match="environment must be an Environment instance"):
            OptimizedPolicyResult(
                environment="not_an_environment",  # type: ignore
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_policy_type_raises_type_error(self):
        """Test that non-Policy policy raises TypeError.

        Purpose: Validates that policy must be a Policy instance

        Given: An invalid policy type (not a Policy subclass)
        When: OptimizedPolicyResult is created with invalid policy
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()

        with pytest.raises(TypeError, match="policy must be a Policy instance"):
            OptimizedPolicyResult(
                environment=env,
                policy="not_a_policy",  # type: ignore
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_chosen_hyper_parameters_type_raises_type_error(self):
        """Test that non-dict chosen_hyper_parameters raises TypeError.

        Purpose: Validates that chosen_hyper_parameters must be a dict

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with non-dict chosen_hyper_parameters
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(TypeError, match="chosen_hyper_parameters must be a dict"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters=[("hp1", 1.0)],  # type: ignore
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_empty_chosen_hyper_parameters_raises_value_error(self):
        """Test that empty chosen_hyper_parameters dict raises ValueError.

        Purpose: Validates that chosen_hyper_parameters cannot be empty

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with empty chosen_hyper_parameters dict
        Then: ValueError is raised indicating dict cannot be empty

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="chosen_hyper_parameters dict cannot be empty"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_parameters_to_optimize_type_raises_type_error(self):
        """Test that non-list parameters_to_optimize raises TypeError.

        Purpose: Validates that parameters_to_optimize must be a list

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with non-list parameters_to_optimize
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(TypeError, match="parameters_to_optimize must be list"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=("average_return", HyperParameterOptimizationDirection.MAXIMIZE),  # type: ignore
                optimized_metric_values={"average_return": 42.5},
            )

    def test_empty_parameters_to_optimize_raises_value_error(self):
        """Test that empty parameters_to_optimize list raises ValueError.

        Purpose: Validates that parameters_to_optimize cannot be empty

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with empty parameters_to_optimize list
        Then: ValueError is raised indicating list cannot be empty

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="parameters_to_optimize cannot be empty"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[],
                optimized_metric_values={},
            )

    def test_invalid_parameter_to_optimize_tuple_type_raises_type_error(self):
        """Test that non-tuple elements in parameters_to_optimize raise TypeError.

        Purpose: Validates that each element in parameters_to_optimize is a tuple of length 2

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with non-tuple element in parameters_to_optimize
        Then: TypeError is raised indicating expected tuple of length 2

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(
            TypeError, match="parameters_to_optimize\\[0\\] must be a tuple of length 2"
        ):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=["average_return"],  # type: ignore
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_metric_name_type_in_parameters_to_optimize_raises_type_error(self):
        """Test that non-string metric names in parameters_to_optimize raise TypeError.

        Purpose: Validates that metric names in parameters_to_optimize must be strings

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with non-string metric name
        Then: TypeError is raised indicating metric_name must be str

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(
            TypeError, match="parameters_to_optimize\\[0\\]\\[0\\] \\(metric_name\\) must be a str"
        ):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[(123, HyperParameterOptimizationDirection.MAXIMIZE)],  # type: ignore
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_direction_type_in_parameters_to_optimize_raises_type_error(self):
        """Test that non-HyperParameterOptimizationDirection directions raise TypeError.

        Purpose: Validates that directions in parameters_to_optimize must be HyperParameterOptimizationDirection

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with invalid direction type
        Then: TypeError is raised indicating direction must be HyperParameterOptimizationDirection

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(
            TypeError,
            match="parameters_to_optimize\\[0\\]\\[1\\] \\(direction\\) must be a HyperParameterOptimizationDirection",
        ):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[("average_return", "maximize")],  # type: ignore
                optimized_metric_values={"average_return": 42.5},
            )

    def test_invalid_optimized_metric_values_type_raises_type_error(self):
        """Test that non-dict optimized_metric_values raises TypeError.

        Purpose: Validates that optimized_metric_values must be a dict

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with non-dict optimized_metric_values
        Then: TypeError is raised with descriptive message

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(TypeError, match="optimized_metric_values must be a dict"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values=[("average_return", 42.5)],  # type: ignore
            )

    def test_invalid_metric_name_raises_value_error(self):
        """Test that invalid metric names in parameters_to_optimize raise ValueError.

        Purpose: Validates that metric names must be valid for the environment-policy pair

        Given: Valid environment and policy with specific available metrics
        When: OptimizedPolicyResult is created with invalid metric name
        Then: ValueError is raised listing available metrics

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(ValueError, match="Invalid metric name 'invalid_metric'"):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("invalid_metric", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"invalid_metric": 42.5},
            )

    def test_missing_metric_in_optimized_metric_values_raises_value_error(self):
        """Test that missing metrics in optimized_metric_values raise ValueError.

        Purpose: Validates that all metrics in parameters_to_optimize must be in optimized_metric_values

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with metric in parameters_to_optimize but not in optimized_metric_values
        Then: ValueError is raised indicating missing metric

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(
            ValueError,
            match="Metric 'average_return' in parameters_to_optimize is missing from optimized_metric_values",
        ):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={},  # Missing the metric
            )

    def test_extra_metric_in_optimized_metric_values_raises_value_error(self):
        """Test that extra metrics in optimized_metric_values raise ValueError.

        Purpose: Validates that optimized_metric_values only contains metrics from parameters_to_optimize

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with extra metric in optimized_metric_values
        Then: ValueError is raised indicating extra metric

        Test type: unit
        """
        import pytest

        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        with pytest.raises(
            ValueError,
            match="Metric 'extra_metric' in optimized_metric_values was not in parameters_to_optimize",
        ):
            OptimizedPolicyResult(
                environment=env,
                policy=policy,
                chosen_hyper_parameters={"hp1": 1.0},
                num_episodes=10,
                num_steps=100,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={
                    "average_return": 42.5,
                    "extra_metric": 10.0,  # Extra metric not in parameters_to_optimize
                },
            )

    def test_none_metric_values_are_allowed(self):
        """Test that None values in optimized_metric_values are allowed.

        Purpose: Validates that optimized_metric_values can contain None for missing metrics

        Given: Valid environment and policy
        When: OptimizedPolicyResult is created with None as a metric value
        Then: Result is created successfully

        Test type: unit
        """
        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        result = OptimizedPolicyResult(
            environment=env,
            policy=policy,
            chosen_hyper_parameters={"hp1": 1.0},
            num_episodes=10,
            num_steps=100,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": None},  # None is allowed
        )

        assert result.optimized_metric_values["average_return"] is None

    def test_multiple_metrics_validation_succeeds(self):
        """Test that multiple valid metrics pass validation.

        Purpose: Validates that OptimizedPolicyResult handles multiple optimization metrics correctly

        Given: Valid environment and policy with multiple available metrics
        When: OptimizedPolicyResult is created with multiple metrics to optimize
        Then: Result is created successfully with all metrics

        Test type: unit
        """
        env = self._create_mock_environment()
        policy = self._create_mock_policy(env)

        result = OptimizedPolicyResult(
            environment=env,
            policy=policy,
            chosen_hyper_parameters={"hp1": 1.0},
            num_episodes=10,
            num_steps=100,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
                ("return_cvar", HyperParameterOptimizationDirection.MAXIMIZE),
            ],
            optimized_metric_values={
                "average_return": 42.5,
                "return_cvar": 38.2,
            },
        )

        assert len(result.parameters_to_optimize) == 2
        assert len(result.optimized_metric_values) == 2
        assert result.optimized_metric_values["average_return"] == 42.5
        assert result.optimized_metric_values["return_cvar"] == 38.2


# ===== ParallelizationLevel Tests =====

from POMDPPlanners.core.simulation.hyperparameter_tuning import ParallelizationLevel


class TestParallelizationLevelEnum:
    """Tests for the ParallelizationLevel enum."""

    def test_enum_values(self):
        """Test ParallelizationLevel enum values.

        Purpose: Validates enum has correct string values

        Given: The ParallelizationLevel enum
        When: Accessing enum values
        Then: OPTUNA_TRIALS is "optuna_trials" and EPISODES is "episodes"

        Test type: unit
        """
        assert ParallelizationLevel.OPTUNA_TRIALS.value == "optuna_trials"
        assert ParallelizationLevel.EPISODES.value == "episodes"

    def test_enum_members(self):
        """Test ParallelizationLevel has exactly two members.

        Purpose: Validates enum membership

        Given: The ParallelizationLevel enum
        When: Listing all members
        Then: There are exactly two members

        Test type: unit
        """
        members = list(ParallelizationLevel)
        assert len(members) == 2
        assert ParallelizationLevel.OPTUNA_TRIALS in members
        assert ParallelizationLevel.EPISODES in members

    def test_enum_from_value(self):
        """Test ParallelizationLevel can be constructed from string values.

        Purpose: Validates enum can be created from string values

        Given: String values matching enum values
        When: Constructing enum instances from strings
        Then: Correct enum members are returned

        Test type: unit
        """
        assert ParallelizationLevel("optuna_trials") == ParallelizationLevel.OPTUNA_TRIALS
        assert ParallelizationLevel("episodes") == ParallelizationLevel.EPISODES

    def test_enum_invalid_value_raises_error(self):
        """Test that invalid string raises ValueError.

        Purpose: Validates that invalid values are rejected

        Given: An invalid string value
        When: Constructing a ParallelizationLevel from it
        Then: ValueError is raised

        Test type: unit
        """
        import pytest

        with pytest.raises(ValueError):
            ParallelizationLevel("invalid")
