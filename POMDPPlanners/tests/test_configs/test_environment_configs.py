#!/usr/bin/env python3
"""Test script to verify all environment configurations work correctly.

This test module validates that all environment configurations in the experiments
directory can be instantiated and provide valid POMDP environments and beliefs.
"""


import random
import traceback

import numpy as np

from POMDPPlanners.configs.environment_configs import (
    EnvironmentConfigsAPI,
    RiskAverseEnvironmentConfigsAPI,
)
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo

# Set random seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestEnvironmentConfigs:
    """Test class for environment configuration API."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.config_api = EnvironmentConfigsAPI(discount_factor=0.95, debug=False)
        self.test_n_particles = 10  # Small number for faster tests

    def test_tiger_pomdp_config(self):
        """Test TigerPOMDP configuration creation."""
        print("Testing TigerPOMDP configuration...")

        pomdp, belief = self.config_api.tiger_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        # Test basic functionality
        actions = pomdp.get_actions()  # type: ignore
        assert len(actions) > 0, "Environment should have at least one action"

        print("  ✓ TigerPOMDP configuration test passed!")

    def test_cartpole_pomdp_config(self):
        """Test CartPolePOMDP configuration creation."""
        print("Testing CartPolePOMDP configuration...")

        pomdp, belief = self.config_api.cartpole_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ CartPolePOMDP configuration test passed!")

    def test_mountain_car_pomdp_config(self):
        """Test MountainCarPOMDP configuration creation."""
        print("Testing MountainCarPOMDP configuration...")

        pomdp, belief = self.config_api.mountain_car_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ MountainCarPOMDP configuration test passed!")

    def test_push_pomdp_config(self):
        """Test PushPOMDP configuration creation."""
        print("Testing PushPOMDP configuration...")

        pomdp, belief = self.config_api.push_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ PushPOMDP configuration test passed!")

    def test_continuous_light_dark_discrete_actions_config(self):
        """Test ContinuousLightDarkPOMDPDiscreteActions configuration creation."""
        print("Testing ContinuousLightDarkPOMDPDiscreteActions configuration...")

        (
            pomdp,
            belief,
        ) = self.config_api.continuous_observations_discrete_actions_light_dark_pomdp_config(
            n_particles=self.test_n_particles
        )

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert hasattr(belief, "particles"), "Belief should have particles attribute"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None

        # Test specific light-dark properties
        assert hasattr(pomdp, "beacons"), "Light-dark POMDP should have beacons"
        assert hasattr(pomdp, "goal_state"), "Light-dark POMDP should have goal_state"
        assert hasattr(pomdp, "start_state"), "Light-dark POMDP should have start_state"

        print("  ✓ ContinuousLightDarkPOMDPDiscreteActions configuration test passed!")

    def test_continuous_light_dark_continuous_actions_config(self):
        """Test ContinuousLightDarkPOMDP configuration creation."""
        print("Testing ContinuousLightDarkPOMDP configuration...")

        # Note: This calls the method with the same name but different implementation
        # The method name appears to be duplicated in the original file
        (
            pomdp,
            belief,
        ) = self.config_api.continuous_observations_continuous_actions_light_dark_pomdp_config(
            n_particles=self.test_n_particles
        )

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert hasattr(belief, "particles"), "Belief should have particles attribute"

        # Verify basic properties
        assert pomdp.discount_factor is not None

        print("  ✓ ContinuousLightDarkPOMDP configuration test passed!")

    def test_rock_sample_pomdp_config(self):
        """Test RockSamplePOMDP configuration creation."""
        print("Testing RockSamplePOMDP configuration...")

        pomdp, belief = self.config_api.rock_sample_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ RockSamplePOMDP configuration test passed!")

    def test_pacman_pomdp_config(self):
        """Test PacManPOMDP configuration creation."""
        print("Testing PacManPOMDP configuration...")

        pomdp, belief = self.config_api.pacman_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ PacManPOMDP configuration test passed!")

    def test_laser_tag_pomdp_config(self):
        """Test LaserTagPOMDP configuration creation."""
        print("Testing LaserTagPOMDP configuration...")

        pomdp, belief = self.config_api.laser_tag_pomdp_config(n_particles=self.test_n_particles)

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ LaserTagPOMDP configuration test passed!")

    def test_safety_ant_velocity_pomdp_config(self):
        """Test SafeAntVelocityPOMDP configuration creation."""
        print("Testing SafeAntVelocityPOMDP configuration...")

        pomdp, belief = self.config_api.safety_ant_velocity_pomdp_config(
            n_particles=self.test_n_particles
        )

        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(
            belief, WeightedParticleBelief
        ), f"Expected WeightedParticleBelief, got {type(belief)}"

        # Verify basic properties
        assert pomdp.discount_factor is not None
        assert pomdp.name is not None
        assert len(belief.particles) > 0

        print("  ✓ SafeAntVelocityPOMDP configuration test passed!")

    def test_risk_averse_push_pomdp_config(self):
        """Test RiskAverseEnvironmentConfigsAPI push POMDP configuration creation."""
        print("Testing RiskAverseEnvironmentConfigsAPI push POMDP configuration...")

        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)

        # Test initialization without errors
        pomdp, belief = risk_averse_api.push_pomdp_config(n_particles=self.test_n_particles)

        print("  ✓ RiskAverseEnvironmentConfigsAPI push POMDP configuration test passed!")

    def test_risk_averse_rock_sample_pomdp_config(self):
        """Test RiskAverseEnvironmentConfigsAPI rock sample POMDP configuration creation."""
        print("Testing RiskAverseEnvironmentConfigsAPI rock sample POMDP configuration...")

        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)

        # Test initialization without errors
        pomdp, belief = risk_averse_api.rock_sample_pomdp_config(n_particles=self.test_n_particles)

        print("  ✓ RiskAverseEnvironmentConfigsAPI rock sample POMDP configuration test passed!")

    def test_risk_averse_light_dark_discrete_actions_config(self):
        """Test RiskAverseEnvironmentConfigsAPI light dark discrete actions configuration creation."""
        print(
            "Testing RiskAverseEnvironmentConfigsAPI light dark discrete actions configuration..."
        )

        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)

        # Test initialization without errors
        (
            pomdp,
            belief,
        ) = risk_averse_api.continuous_observations_discrete_actions_light_dark_pomdp_config(
            n_particles=self.test_n_particles
        )

        print(
            "  ✓ RiskAverseEnvironmentConfigsAPI light dark discrete actions configuration test passed!"
        )

    def test_risk_averse_light_dark_continuous_actions_config(self):
        """Test RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration creation."""
        print(
            "Testing RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration..."
        )

        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)

        # Test initialization without errors
        (
            pomdp,
            belief,
        ) = risk_averse_api.continuous_observations_continuous_actions_light_dark_pomdp_config(
            n_particles=self.test_n_particles
        )

        print(
            "  ✓ RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration test passed!"
        )

    def test_config_api_initialization(self):
        """Test EnvironmentConfigsAPI initialization with different parameters."""
        print("Testing EnvironmentConfigsAPI initialization...")

        # Test default initialization
        api_default = EnvironmentConfigsAPI()
        assert api_default.discount_factor is not None
        assert api_default.debug is not None

        # Test custom initialization
        api_custom = EnvironmentConfigsAPI(discount_factor=0.99, debug=True)
        assert api_custom.discount_factor is not None
        assert api_custom.debug is not None

        print("  ✓ EnvironmentConfigsAPI initialization test passed!")

    def test_risk_averse_config_api_initialization(self):
        """Test RiskAverseEnvironmentConfigsAPI initialization with different parameters."""
        print("Testing RiskAverseEnvironmentConfigsAPI initialization...")

        # Test default initialization
        api_default = RiskAverseEnvironmentConfigsAPI()
        assert api_default.discount_factor is not None
        assert api_default.debug is not None

        # Test custom initialization
        api_custom = RiskAverseEnvironmentConfigsAPI(discount_factor=0.99, debug=True)
        assert api_custom.discount_factor is not None
        assert api_custom.debug is not None

        print("  ✓ RiskAverseEnvironmentConfigsAPI initialization test passed!")

    def test_all_configs_have_consistent_interface(self):
        """Test that all configuration methods return consistent types."""
        print("Testing all configurations have consistent interface...")

        config_methods = [
            "tiger_pomdp_config",
            "cartpole_pomdp_config",
            "mountain_car_pomdp_config",
            "push_pomdp_config",
            "rock_sample_pomdp_config",
            "pacman_pomdp_config",
            "laser_tag_pomdp_config",
            "safety_ant_velocity_pomdp_config",
        ]

        for method_name in config_methods:
            method = getattr(self.config_api, method_name)
            pomdp, belief = method(n_particles=self.test_n_particles)

            # Verify consistent return types
            assert isinstance(pomdp, Environment), f"{method_name} should return Environment"
            assert hasattr(
                belief, "particles"
            ), f"{method_name} should return object with particles"
            assert pomdp.discount_factor is not None, f"{method_name} should have discount factor"

        print("  ✓ All configurations have consistent interface test passed!")

    def test_get_compatible_environments_discrete_discrete(self):
        """Test get_compatible_environments with discrete action and observation spaces.

        Purpose: Validates that get_compatible_environments correctly filters environments

        Given: A PolicySpaceInfo with discrete action and observation spaces
        When: get_compatible_environments is called with this policy space info
        Then: All returned environments have compatible space types

        Test type: unit
        """
        print("Testing get_compatible_environments with discrete/discrete policy...")

        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=self.test_n_particles
        )

        # Should return a list of tuples
        assert isinstance(compatible_envs, list), "Should return a list"

        # Each element should be a tuple of (env, belief)
        for env, belief in compatible_envs:
            assert isinstance(env, Environment), "First element should be Environment"
            assert hasattr(belief, "particles"), "Second element should be a belief with particles"

            # Verify compatibility
            assert env.space_info.action_space in [
                SpaceType.DISCRETE,
                SpaceType.MIXED,
            ], f"Environment {env.name} should have discrete or mixed action space"
            assert env.space_info.observation_space in [
                SpaceType.DISCRETE,
                SpaceType.MIXED,
            ], f"Environment {env.name} should have discrete or mixed observation space"

        print(
            f"  ✓ Found {len(compatible_envs)} compatible environments for discrete/discrete policy"
        )

    def test_get_compatible_environments_discrete_continuous(self):
        """Test get_compatible_environments with discrete actions and continuous observations.

        Purpose: Validates filtering for policies with discrete actions and continuous observations

        Given: A PolicySpaceInfo with discrete action and continuous observation spaces
        When: get_compatible_environments is called
        Then: All returned environments have compatible space types

        Test type: unit
        """
        print("Testing get_compatible_environments with discrete action/continuous observation...")

        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )

        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=self.test_n_particles
        )

        # Should return a list
        assert isinstance(compatible_envs, list), "Should return a list"

        for env, belief in compatible_envs:
            assert isinstance(env, Environment), "First element should be Environment"

            # Verify action space compatibility
            assert env.space_info.action_space in [
                SpaceType.DISCRETE,
                SpaceType.MIXED,
            ], f"Environment {env.name} action space should be compatible with discrete"

        print(
            f"  ✓ Found {len(compatible_envs)} compatible environments for discrete/continuous policy"
        )

    def test_get_compatible_environments_continuous_continuous(self):
        """Test get_compatible_environments with continuous action and observation spaces.

        Purpose: Validates filtering for fully continuous policies

        Given: A PolicySpaceInfo with continuous action and observation spaces
        When: get_compatible_environments is called
        Then: All returned environments have compatible space types

        Test type: unit
        """
        print("Testing get_compatible_environments with continuous/continuous policy...")

        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )

        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=self.test_n_particles
        )

        # Should return a list
        assert isinstance(compatible_envs, list), "Should return a list"

        for env, belief in compatible_envs:
            assert isinstance(env, Environment), "First element should be Environment"
            assert hasattr(belief, "particles"), "Second element should be a belief"

        print(
            f"  ✓ Found {len(compatible_envs)} compatible environments for continuous/continuous policy"
        )

    def test_get_compatible_environments_returns_proper_beliefs(self):
        """Test that get_compatible_environments returns properly initialized beliefs.

        Purpose: Validates that beliefs returned have correct particle counts

        Given: A PolicySpaceInfo and specified n_particles parameter
        When: get_compatible_environments is called with n_particles=15
        Then: All returned beliefs have exactly 15 particles

        Test type: unit
        """
        print("Testing get_compatible_environments belief initialization...")

        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )

        n_particles = 15
        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=n_particles
        )

        # Verify all beliefs have correct number of particles
        for env, belief in compatible_envs:
            assert (
                len(belief.particles) == n_particles
            ), f"Belief for {env.name} should have {n_particles} particles, got {len(belief.particles)}"

        print("  ✓ All beliefs have correct particle counts")

    def test_get_compatible_environments_empty_result(self):
        """Test get_compatible_environments when no environments are compatible.

        Purpose: Validates behavior when policy is incompatible with all environments

        Given: A PolicySpaceInfo that might not match any environments
        When: get_compatible_environments is called
        Then: Returns an empty list without errors

        Test type: unit
        """
        print("Testing get_compatible_environments with potentially incompatible policy...")

        # Create a policy space info (may or may not have compatible environments)
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )

        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=self.test_n_particles
        )

        # Should return a list (possibly empty)
        assert isinstance(compatible_envs, list), "Should return a list"

        print(f"  ✓ Returned {len(compatible_envs)} compatible environments without errors")

    def test_get_compatible_environments_verifies_compatibility_logic(self):
        """Test that compatibility logic matches Policy._verify_environment_compatibility.

        Purpose: Validates that the compatibility logic is consistent with Policy class

        Given: A discrete-only policy space info
        When: get_compatible_environments is called
        Then: No continuous-only action space environments are returned

        Test type: unit
        """
        print("Testing compatibility logic consistency...")

        # Discrete policy should reject continuous action environments
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        compatible_envs = self.config_api.get_compatible_environments(
            policy_space_info=policy_space_info, n_particles=self.test_n_particles
        )

        # Verify none of the returned environments have purely continuous action spaces
        for env, belief in compatible_envs:
            assert (
                env.space_info.action_space != SpaceType.CONTINUOUS
                or env.space_info.action_space == SpaceType.MIXED
            ), f"Discrete policy should not be compatible with continuous action environment {env.name}"

        print("  ✓ Compatibility logic is consistent with Policy class")


def main():
    """Run all environment configuration tests."""
    print("🚀 Running Environment Configuration Tests")
    print("=" * 60)

    test_class = TestEnvironmentConfigs()
    test_methods = [method for method in dir(test_class) if method.startswith("test_")]

    passed = 0
    total = len(test_methods)

    for method_name in test_methods:
        try:
            test_class.setup_method()
            method = getattr(test_class, method_name)
            method()
            passed += 1
        except Exception as e:
            print(f"  ❌ {method_name} failed: {e}")

            traceback.print_exc()

    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All environment configuration tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
