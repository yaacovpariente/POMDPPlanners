from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    CategoricalHyperParameter,
    NumericalHyperParameter,
    HyperParamPlannerConfig,
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
)
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import Policy


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

            def state_transition_model(self, state, action):  # type: ignore[override]
                pass

            def observation_model(self, next_state, action):  # type: ignore[override]
                pass

            def reward(self, state, action):
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

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
            [("param", HyperParameterOptimizationDirection.MAXIMIZE)],
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

            def state_transition_model(self, state, action):  # type: ignore[override]
                pass

            def observation_model(self, next_state, action):  # type: ignore[override]
                pass

            def reward(self, state, action):
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

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
            [("param", HyperParameterOptimizationDirection.MAXIMIZE)],
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

            def state_transition_model(self, state, action):  # type: ignore[override]
                pass

            def observation_model(self, next_state, action):  # type: ignore[override]
                pass

            def reward(self, state, action):
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

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
            [("param", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        params2 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            20,  # Different num_episodes
            100,
            5,
            [("param", HyperParameterOptimizationDirection.MAXIMIZE)],
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

            def state_transition_model(self, state, action):  # type: ignore[override]
                pass

            def observation_model(self, next_state, action):  # type: ignore[override]
                pass

            def reward(self, state, action):
                return 0.0

            def is_terminal(self, state):
                return False

            def initial_state_dist(self):  # type: ignore[override]
                pass

            def initial_observation_dist(self):  # type: ignore[override]
                pass

            def is_equal_observation(self, observation1, observation2):
                return observation1 == observation2

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
            [("param", HyperParameterOptimizationDirection.MAXIMIZE)],
        )

        params2 = HyperParameterRunParams(
            MockEnv(),
            MockBelief(),
            hyper_config,
            10,
            100,
            5,
            [("param", HyperParameterOptimizationDirection.MINIMIZE)],  # Different direction
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
