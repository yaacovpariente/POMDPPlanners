"""Tests for POMDP environment serialization.

This module tests that all POMDP environments can be properly serialized
and deserialized using pickle. Serialization is crucial for:
- Distributed computing scenarios
- Saving/loading environment configurations
- Caching environment instances
- Multi-processing applications
"""

import pickle
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments import (
    CartPolePOMDP,
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    DiscreteLightDarkPOMDP,
    LaserTagPOMDP,
    MountainCarPOMDP,
    PacManPOMDP,
    PushPOMDP,
    RockSamplePOMDP,
    SafeAntVelocityPOMDP,
    SanityPOMDP,
    TigerPOMDP,
)

# Set seeds for reproducible tests
np.random.seed(42)


class TestEnvironmentSerialization:
    """Test cases for environment serialization using pickle."""

    def _test_environment_serialization(self, env_class: type, init_params: Dict[str, Any]) -> None:
        """Helper method to test environment serialization.

        Purpose: Validates that an environment can be pickled and unpickled correctly

        Given: An environment class and initialization parameters
        When: Environment is created, pickled, and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit

        Args:
            env_class: Environment class to test
            init_params: Parameters for environment initialization
        """
        # Create environment
        env = env_class(**init_params)

        # Pickle the environment
        pickled = pickle.dumps(env)

        # Unpickle the environment
        unpickled_env = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_env.name == env.name
        assert unpickled_env.discount_factor == env.discount_factor
        assert unpickled_env.space_info.action_space == env.space_info.action_space
        assert unpickled_env.space_info.observation_space == env.space_info.observation_space

        # Test that unpickled environment can sample initial state
        initial_state = unpickled_env.initial_state_dist().sample()[0]
        assert initial_state is not None

        # Test that unpickled environment can perform state transitions
        # For discrete action environments
        if hasattr(unpickled_env, "get_actions"):
            actions = unpickled_env.get_actions()
            if actions:
                action = actions[0]
                next_state, obs, reward = unpickled_env.sample_next_step(initial_state, action)
                assert next_state is not None
                assert obs is not None
                assert isinstance(reward, (int, float))

    def test_tiger_pomdp_serialization(self):
        """Test TigerPOMDP serialization.

        Purpose: Validates that TigerPOMDP can be pickled and unpickled

        Given: TigerPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(TigerPOMDP, {"discount_factor": 0.95})

    def test_sanity_pomdp_serialization(self):
        """Test SanityPOMDP serialization.

        Purpose: Validates that SanityPOMDP can be pickled and unpickled

        Given: SanityPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(SanityPOMDP, {"discount_factor": 0.95})

    def test_cartpole_pomdp_serialization(self):
        """Test CartPolePOMDP serialization.

        Purpose: Validates that CartPolePOMDP can be pickled and unpickled

        Given: CartPolePOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        # CartPolePOMDP requires noise_cov parameter
        noise_cov = np.eye(4) * 0.1
        self._test_environment_serialization(
            CartPolePOMDP, {"discount_factor": 0.95, "noise_cov": noise_cov}
        )

    def test_mountain_car_pomdp_serialization(self):
        """Test MountainCarPOMDP serialization.

        Purpose: Validates that MountainCarPOMDP can be pickled and unpickled

        Given: MountainCarPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(MountainCarPOMDP, {"discount_factor": 0.95})

    def test_discrete_light_dark_pomdp_serialization(self):
        """Test DiscreteLightDarkPOMDP serialization.

        Purpose: Validates that DiscreteLightDarkPOMDP can be pickled and unpickled

        Given: DiscreteLightDarkPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(DiscreteLightDarkPOMDP, {"discount_factor": 0.95})

    def test_continuous_light_dark_pomdp_serialization(self):
        """Test ContinuousLightDarkPOMDP serialization.

        Purpose: Validates that ContinuousLightDarkPOMDP can be pickled and unpickled

        Given: ContinuousLightDarkPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(ContinuousLightDarkPOMDP, {"discount_factor": 0.95})

    def test_continuous_light_dark_pomdp_discrete_actions_serialization(self):
        """Test ContinuousLightDarkPOMDPDiscreteActions serialization.

        Purpose: Validates that ContinuousLightDarkPOMDPDiscreteActions can be pickled and unpickled

        Given: ContinuousLightDarkPOMDPDiscreteActions instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(
            ContinuousLightDarkPOMDPDiscreteActions, {"discount_factor": 0.95}
        )

    def test_push_pomdp_serialization(self):
        """Test PushPOMDP serialization.

        Purpose: Validates that PushPOMDP can be pickled and unpickled

        Given: PushPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(PushPOMDP, {"discount_factor": 0.95})

    def test_laser_tag_pomdp_serialization(self):
        """Test LaserTagPOMDP serialization.

        Purpose: Validates that LaserTagPOMDP can be pickled and unpickled

        Given: LaserTagPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(LaserTagPOMDP, {"discount_factor": 0.95})

    def test_rock_sample_pomdp_serialization(self):
        """Test RockSamplePOMDP serialization.

        Purpose: Validates that RockSamplePOMDP can be pickled and unpickled

        Given: RockSamplePOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        # RockSamplePOMDP uses map_size instead of grid_size
        self._test_environment_serialization(
            RockSamplePOMDP, {"discount_factor": 0.95, "map_size": (5, 5)}
        )

    def test_pacman_pomdp_serialization(self):
        """Test PacManPOMDP serialization.

        Purpose: Validates that PacManPOMDP can be pickled and unpickled

        Given: PacManPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(PacManPOMDP, {"discount_factor": 0.95})

    def test_safety_ant_velocity_pomdp_serialization(self):
        """Test SafeAntVelocityPOMDP serialization.

        Purpose: Validates that SafeAntVelocityPOMDP can be pickled and unpickled

        Given: SafeAntVelocityPOMDP instance with default parameters
        When: Environment is pickled and unpickled
        Then: Unpickled environment maintains all properties and functionality

        Test type: unit
        """
        self._test_environment_serialization(SafeAntVelocityPOMDP, {"discount_factor": 0.95})


class TestEnvironmentStateSerialization:
    """Test cases for environment state serialization."""

    def test_tiger_state_serialization(self):
        """Test TigerPOMDP state serialization.

        Purpose: Validates that TigerPOMDP states can be pickled and unpickled

        Given: TigerPOMDP environment and sampled state
        When: State is pickled and unpickled
        Then: Unpickled state equals original state

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]

        pickled_state = pickle.dumps(state)
        unpickled_state = pickle.loads(pickled_state)

        assert unpickled_state == state

    def test_pacman_state_serialization(self):
        """Test PacManPOMDP state serialization.

        Purpose: Validates that PacManPOMDP states can be pickled and unpickled

        Given: PacManPOMDP environment and sampled state
        When: State is pickled and unpickled
        Then: Unpickled state equals original state

        Test type: unit
        """
        env = PacManPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]

        pickled_state = pickle.dumps(state)
        unpickled_state = pickle.loads(pickled_state)

        # State is a numpy array, not an object with position attribute
        np.testing.assert_array_equal(unpickled_state, state)

    def test_discrete_light_dark_state_serialization(self):
        """Test DiscreteLightDarkPOMDP state serialization.

        Purpose: Validates that DiscreteLightDarkPOMDP states can be pickled and unpickled

        Given: DiscreteLightDarkPOMDP environment and sampled state
        When: State is pickled and unpickled
        Then: Unpickled state equals original state

        Test type: unit
        """
        env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]

        pickled_state = pickle.dumps(state)
        unpickled_state = pickle.loads(pickled_state)

        # State is a numpy array, not an object with position attribute
        assert np.array_equal(unpickled_state, state)

    def test_rock_sample_state_serialization(self):
        """Test RockSamplePOMDP state serialization.

        Purpose: Validates that RockSamplePOMDP states can be pickled and unpickled

        Given: RockSamplePOMDP environment and sampled state
        When: State is pickled and unpickled
        Then: Unpickled state equals original state

        Test type: unit
        """
        env = RockSamplePOMDP(discount_factor=0.95, map_size=(5, 5))
        state = env.initial_state_dist().sample()[0]

        pickled_state = pickle.dumps(state)
        unpickled_state = pickle.loads(pickled_state)

        # State is a numpy array, not an object with attributes
        assert np.array_equal(unpickled_state, state)


class TestEnvironmentTransitionModelSerialization:
    """Test cases for environment transition model serialization."""

    def test_tiger_transition_model_serialization(self):
        """Test TigerPOMDP transition model serialization.

        Purpose: Validates that TigerPOMDP transition models can be pickled and unpickled

        Given: TigerPOMDP environment, state, and action
        When: Transition model is pickled and unpickled
        Then: Unpickled model can sample valid next states

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]
        actions = env.get_actions()
        action = actions[0]

        transition_model = env.state_transition_model(state, action)
        pickled = pickle.dumps(transition_model)
        unpickled_model = pickle.loads(pickled)

        # Test that unpickled model can sample states
        next_state = unpickled_model.sample()[0]
        assert next_state is not None

    @pytest.mark.skip(
        reason=(
            "PacManStateTransitionModel inherits from a pybind11 C++ class "
            "(_native.PacManTransitionCpp), which is not pickleable by default. "
            "Serializing the env-level state array is still covered by "
            "test_pacman_state_serialization; there's no planner code path "
            "that pickles the transition-model wrapper itself."
        )
    )
    def test_pacman_transition_model_serialization(self):
        """Test PacManPOMDP transition model serialization.

        Purpose: Validates that PacManPOMDP transition models can be pickled and unpickled

        Given: PacManPOMDP environment, state, and action
        When: Transition model is pickled and unpickled
        Then: Unpickled model can sample valid next states

        Test type: unit
        """
        env = PacManPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]
        actions = env.get_actions()
        action = actions[0]

        transition_model = env.state_transition_model(state, action)
        pickled = pickle.dumps(transition_model)
        unpickled_model = pickle.loads(pickled)

        # Test that unpickled model can sample states
        next_state = unpickled_model.sample()[0]
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (env._state_dim,)  # pylint: disable=protected-access
        env.get_pacman_pos(next_state)
        env.get_ghost_positions(next_state)


class TestEnvironmentObservationModelSerialization:
    """Test cases for environment observation model serialization."""

    def test_tiger_observation_model_serialization(self):
        """Test TigerPOMDP observation model serialization.

        Purpose: Validates that TigerPOMDP observation models can be pickled and unpickled

        Given: TigerPOMDP environment, next state, and action
        When: Observation model is pickled and unpickled
        Then: Unpickled model can sample valid observations

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]
        actions = env.get_actions()
        action = actions[0]

        # Sample next state
        next_state = env.state_transition_model(state, action).sample()[0]

        # Create observation model
        obs_model = env.observation_model(next_state, action)
        pickled = pickle.dumps(obs_model)
        unpickled_model = pickle.loads(pickled)

        # Test that unpickled model can sample observations
        obs = unpickled_model.sample()[0]
        assert obs is not None

    @pytest.mark.skip(
        reason=(
            "PacManObservationModel inherits from a pybind11 C++ class "
            "(_native.PacManObservationCpp), which is not pickleable by default. "
            "No planner code path pickles the observation-model wrapper itself."
        )
    )
    def test_pacman_observation_model_serialization(self):
        """Test PacManPOMDP observation model serialization.

        Purpose: Validates that PacManPOMDP observation models can be pickled and unpickled

        Given: PacManPOMDP environment, next state, and action
        When: Observation model is pickled and unpickled
        Then: Unpickled model can sample valid observations

        Test type: unit
        """
        env = PacManPOMDP(discount_factor=0.95)
        state = env.initial_state_dist().sample()[0]
        actions = env.get_actions()
        action = actions[0]

        # Sample next state
        next_state = env.state_transition_model(state, action).sample()[0]

        # Create observation model
        obs_model = env.observation_model(next_state, action)
        pickled = pickle.dumps(obs_model)
        unpickled_model = pickle.loads(pickled)

        # Test that unpickled model can sample observations
        obs = unpickled_model.sample()[0]
        assert obs is not None
        assert isinstance(obs, tuple)  # Multi-ghost observations


class TestEnvironmentSerializationRoundTrip:
    """Test cases for complete environment serialization round trips."""

    def test_environment_serialization_preserves_functionality(self):
        """Test that serialized environments maintain full functionality.

        Purpose: Validates that environments work correctly after serialization round trip

        Given: Multiple environment instances
        When: Environments are pickled, unpickled, and used for simulations
        Then: Unpickled environments produce valid results

        Test type: integration
        """
        environments = [
            TigerPOMDP(discount_factor=0.95),
            SanityPOMDP(discount_factor=0.95),
            PacManPOMDP(discount_factor=0.95),
            DiscreteLightDarkPOMDP(discount_factor=0.95),
        ]

        for env in environments:
            # Pickle and unpickle
            pickled = pickle.dumps(env)
            unpickled_env = pickle.loads(pickled)

            # Run a simple episode
            state = unpickled_env.initial_state_dist().sample()[0]

            for _ in range(3):  # Run 3 steps
                if unpickled_env.is_terminal(state):
                    break

                # Sample action
                if hasattr(unpickled_env, "get_actions"):
                    actions = unpickled_env.get_actions()
                    action = actions[0]

                    # Sample next step
                    next_state, obs, reward = unpickled_env.sample_next_step(state, action)

                    assert next_state is not None
                    assert obs is not None
                    assert isinstance(reward, (int, float))

                    state = next_state

    def test_environment_serialization_with_numpy_arrays(self):
        """Test serialization of environments with numpy array states.

        Purpose: Validates that environments with numpy arrays serialize correctly

        Given: Environments that use numpy arrays in states
        When: Environments are pickled and unpickled
        Then: Numpy arrays are preserved correctly

        Test type: unit
        """
        # Test continuous environments that use numpy arrays
        env = ContinuousLightDarkPOMDP(discount_factor=0.95)

        # Sample state
        state = env.initial_state_dist().sample()[0]

        # Pickle environment and state
        pickled_env = pickle.dumps(env)
        pickled_state = pickle.dumps(state)

        # Unpickle
        unpickled_env = pickle.loads(pickled_env)
        unpickled_state = pickle.loads(pickled_state)

        # Verify numpy arrays preserved (state is directly a numpy array)
        assert np.array_equal(unpickled_state, state)

        # Verify environment works with unpickled state
        action = np.array([0.5, 0.5])  # Sample action
        next_state, obs, reward = unpickled_env.sample_next_step(unpickled_state, action)
        assert next_state is not None


class TestEnvironmentDictSerialization:
    """Test environment to_dict/from_dict serialization for all environments."""

    def _test_environment_dict_serialization(
        self, env_class: type, init_params: Dict[str, Any]
    ) -> None:
        """Helper to test environment dict serialization.

        Purpose: Validates that environment can be serialized to dict and reconstructed

        Given: An environment class and initialization parameters
        When: Environment is converted to dict and reconstructed with from_dict
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface

        Args:
            env_class: Environment class to test
            init_params: Parameters for environment initialization
        """
        # Create environment
        env = env_class(**init_params)

        # Serialize to dict
        env_dict = env.to_dict()

        # Verify dict structure
        assert "class" in env_dict
        assert "module" in env_dict
        assert "params" in env_dict
        assert "config_id" in env_dict

        # Reconstruct from dict
        from POMDPPlanners.core.environment import Environment

        reconstructed_env = Environment.from_dict(env_dict)

        # Verify properties preserved
        assert reconstructed_env.name == env.name
        assert reconstructed_env.discount_factor == env.discount_factor
        assert reconstructed_env.config_id == env.config_id
        assert reconstructed_env.space_info.action_space == env.space_info.action_space
        assert reconstructed_env.space_info.observation_space == env.space_info.observation_space

        # Test functionality
        initial_state = reconstructed_env.initial_state_dist().sample()[0]
        assert initial_state is not None

        # Test state transitions for environments with get_actions
        get_actions = getattr(reconstructed_env, "get_actions", None)
        if get_actions is not None:
            actions = get_actions()
            if actions:
                action = actions[0]
                next_state, obs, reward = reconstructed_env.sample_next_step(initial_state, action)
                assert next_state is not None
                assert obs is not None
                assert isinstance(reward, (int, float))

    def test_tiger_pomdp_dict_serialization(self):
        """Test TigerPOMDP dict serialization.

        Purpose: Validates TigerPOMDP can be serialized to dict and reconstructed

        Given: TigerPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(TigerPOMDP, {"discount_factor": 0.95})

    def test_sanity_pomdp_dict_serialization(self):
        """Test SanityPOMDP dict serialization.

        Purpose: Validates SanityPOMDP can be serialized to dict and reconstructed

        Given: SanityPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(SanityPOMDP, {"discount_factor": 0.95})

    def test_cartpole_pomdp_dict_serialization(self):
        """Test CartPolePOMDP dict serialization.

        Purpose: Validates CartPolePOMDP can be serialized to dict and reconstructed

        Given: CartPolePOMDP instance with noise covariance parameter
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        noise_cov = np.eye(4) * 0.1
        self._test_environment_dict_serialization(
            CartPolePOMDP, {"discount_factor": 0.95, "noise_cov": noise_cov}
        )

    def test_mountain_car_pomdp_dict_serialization(self):
        """Test MountainCarPOMDP dict serialization.

        Purpose: Validates MountainCarPOMDP can be serialized to dict and reconstructed

        Given: MountainCarPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(MountainCarPOMDP, {"discount_factor": 0.95})

    def test_discrete_light_dark_pomdp_dict_serialization(self):
        """Test DiscreteLightDarkPOMDP dict serialization.

        Purpose: Validates DiscreteLightDarkPOMDP can be serialized to dict and reconstructed

        Given: DiscreteLightDarkPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(DiscreteLightDarkPOMDP, {"discount_factor": 0.95})

    def test_continuous_light_dark_pomdp_dict_serialization(self):
        """Test ContinuousLightDarkPOMDP dict serialization.

        Purpose: Validates ContinuousLightDarkPOMDP can be serialized to dict and reconstructed

        Given: ContinuousLightDarkPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(
            ContinuousLightDarkPOMDP, {"discount_factor": 0.95}
        )

    def test_continuous_light_dark_pomdp_discrete_actions_dict_serialization(self):
        """Test ContinuousLightDarkPOMDPDiscreteActions dict serialization.

        Purpose: Validates ContinuousLightDarkPOMDPDiscreteActions can be serialized and reconstructed

        Given: ContinuousLightDarkPOMDPDiscreteActions instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(
            ContinuousLightDarkPOMDPDiscreteActions, {"discount_factor": 0.95}
        )

    def test_push_pomdp_dict_serialization(self):
        """Test PushPOMDP dict serialization.

        Purpose: Validates PushPOMDP can be serialized to dict and reconstructed

        Given: PushPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(PushPOMDP, {"discount_factor": 0.95})

    def test_laser_tag_pomdp_dict_serialization(self):
        """Test LaserTagPOMDP dict serialization.

        Purpose: Validates LaserTagPOMDP can be serialized to dict and reconstructed

        Given: LaserTagPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(LaserTagPOMDP, {"discount_factor": 0.95})

    def test_rock_sample_pomdp_dict_serialization(self):
        """Test RockSamplePOMDP dict serialization.

        Purpose: Validates RockSamplePOMDP can be serialized to dict and reconstructed

        Given: RockSamplePOMDP instance with map_size parameter
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(
            RockSamplePOMDP, {"discount_factor": 0.95, "map_size": (5, 5)}
        )

    def test_pacman_pomdp_dict_serialization(self):
        """Test PacManPOMDP dict serialization.

        Purpose: Validates PacManPOMDP can be serialized to dict and reconstructed

        Given: PacManPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(PacManPOMDP, {"discount_factor": 0.95})

    def test_safety_ant_velocity_pomdp_dict_serialization(self):
        """Test SafeAntVelocityPOMDP dict serialization.

        Purpose: Validates SafeAntVelocityPOMDP can be serialized to dict and reconstructed

        Given: SafeAntVelocityPOMDP instance with default parameters
        When: Environment is serialized to dict and reconstructed
        Then: Reconstructed environment maintains all properties and functionality

        Test type: interface
        """
        self._test_environment_dict_serialization(SafeAntVelocityPOMDP, {"discount_factor": 0.95})
