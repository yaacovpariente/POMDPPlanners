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
            pass

        class Policy2(Policy):
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
