#!/usr/bin/env python3
"""Test script to verify all environment configurations work correctly.

This test module validates that all environment configurations in the experiments
directory can be instantiated and provide valid POMDP environments and beliefs.
"""

import sys
import os
import pytest
import numpy as np
import random
from pathlib import Path

from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI, RiskAverseEnvironmentConfigsAPI
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import WeightedParticleBelief

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
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "TigerPOMDP", f"Expected 'TigerPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        # Test basic functionality
        actions = pomdp.get_actions()
        assert len(actions) > 0, "Environment should have at least one action"
        
        print("  ✓ TigerPOMDP configuration test passed!")
    
    def test_cartpole_pomdp_config(self):
        """Test CartPolePOMDP configuration creation."""
        print("Testing CartPolePOMDP configuration...")
        
        pomdp, belief = self.config_api.cartpole_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "CartPolePOMDP", f"Expected 'CartPolePOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ CartPolePOMDP configuration test passed!")
    
    def test_mountain_car_pomdp_config(self):
        """Test MountainCarPOMDP configuration creation."""
        print("Testing MountainCarPOMDP configuration...")
        
        pomdp, belief = self.config_api.mountain_car_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "MountainCarPOMDP", f"Expected 'MountainCarPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ MountainCarPOMDP configuration test passed!")
    
    def test_push_pomdp_config(self):
        """Test PushPOMDP configuration creation."""
        print("Testing PushPOMDP configuration...")
        
        pomdp, belief = self.config_api.push_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "PushPOMDP", f"Expected 'PushPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ PushPOMDP configuration test passed!")
    
    def test_continuous_light_dark_discrete_actions_config(self):
        """Test ContinuousLightDarkPOMDPDiscreteActions configuration creation."""
        print("Testing ContinuousLightDarkPOMDPDiscreteActions configuration...")
        
        pomdp, belief = self.config_api.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert hasattr(belief, 'particles'), f"Belief should have particles attribute"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "ContinuousLightDarkPOMDPDiscreteActions", f"Expected 'ContinuousLightDarkPOMDPDiscreteActions', got {pomdp.name}"
        
        # Test specific light-dark properties
        assert hasattr(pomdp, 'beacons'), "Light-dark POMDP should have beacons"
        assert hasattr(pomdp, 'goal_state'), "Light-dark POMDP should have goal_state"
        assert hasattr(pomdp, 'start_state'), "Light-dark POMDP should have start_state"
        
        print("  ✓ ContinuousLightDarkPOMDPDiscreteActions configuration test passed!")
    
    def test_continuous_light_dark_continuous_actions_config(self):
        """Test ContinuousLightDarkPOMDP configuration creation."""
        print("Testing ContinuousLightDarkPOMDP configuration...")
        
        # Note: This calls the method with the same name but different implementation
        # The method name appears to be duplicated in the original file
        pomdp, belief = self.config_api.continuous_observations_continuous_actions_light_dark_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert hasattr(belief, 'particles'), f"Belief should have particles attribute"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        
        print("  ✓ ContinuousLightDarkPOMDP configuration test passed!")
    
    def test_rock_sample_pomdp_config(self):
        """Test RockSamplePOMDP configuration creation."""
        print("Testing RockSamplePOMDP configuration...")
        
        pomdp, belief = self.config_api.rock_sample_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "RockSamplePOMDP", f"Expected 'RockSamplePOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ RockSamplePOMDP configuration test passed!")
    
    def test_pacman_pomdp_config(self):
        """Test PacManPOMDP configuration creation."""
        print("Testing PacManPOMDP configuration...")
        
        pomdp, belief = self.config_api.pacman_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "PacManPOMDP", f"Expected 'PacManPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ PacManPOMDP configuration test passed!")
    
    def test_laser_tag_pomdp_config(self):
        """Test LaserTagPOMDP configuration creation."""
        print("Testing LaserTagPOMDP configuration...")
        
        pomdp, belief = self.config_api.laser_tag_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "LaserTagPOMDP", f"Expected 'LaserTagPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
        print("  ✓ LaserTagPOMDP configuration test passed!")
    
    def test_safety_ant_velocity_pomdp_config(self):
        """Test SafeAntVelocityPOMDP configuration creation."""
        print("Testing SafeAntVelocityPOMDP configuration...")
        
        pomdp, belief = self.config_api.safety_ant_velocity_pomdp_config(n_particles=self.test_n_particles)
        
        # Verify types
        assert isinstance(pomdp, Environment), f"Expected Environment, got {type(pomdp)}"
        assert isinstance(belief, WeightedParticleBelief), f"Expected WeightedParticleBelief, got {type(belief)}"
        
        # Verify basic properties
        assert pomdp.discount_factor == 0.95, f"Expected 0.95, got {pomdp.discount_factor}"
        assert pomdp.name == "SafeAntVelocityPOMDP", f"Expected 'SafeAntVelocityPOMDP', got {pomdp.name}"
        assert len(belief.particles) == self.test_n_particles, f"Expected {self.test_n_particles} particles, got {len(belief.particles)}"
        
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
        print("Testing RiskAverseEnvironmentConfigsAPI light dark discrete actions configuration...")
        
        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)
        
        # Test initialization without errors
        pomdp, belief = risk_averse_api.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=self.test_n_particles)
        
        print("  ✓ RiskAverseEnvironmentConfigsAPI light dark discrete actions configuration test passed!")
    
    def test_risk_averse_light_dark_continuous_actions_config(self):
        """Test RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration creation."""
        print("Testing RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration...")
        
        # Create risk averse config API
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95, debug=False)
        
        # Test initialization without errors
        pomdp, belief = risk_averse_api.continuous_observations_continuous_actions_light_dark_pomdp_config(n_particles=self.test_n_particles)
        
        print("  ✓ RiskAverseEnvironmentConfigsAPI light dark continuous actions configuration test passed!")
    
    def test_config_api_initialization(self):
        """Test EnvironmentConfigsAPI initialization with different parameters."""
        print("Testing EnvironmentConfigsAPI initialization...")
        
        # Test default initialization
        api_default = EnvironmentConfigsAPI()
        assert api_default.discount_factor == 0.95, f"Expected 0.95, got {api_default.discount_factor}"
        assert api_default.debug == False, f"Expected False, got {api_default.debug}"
        
        # Test custom initialization
        api_custom = EnvironmentConfigsAPI(discount_factor=0.99, debug=True)
        assert api_custom.discount_factor == 0.99, f"Expected 0.99, got {api_custom.discount_factor}"
        assert api_custom.debug == True, f"Expected True, got {api_custom.debug}"
        
        print("  ✓ EnvironmentConfigsAPI initialization test passed!")
    
    def test_risk_averse_config_api_initialization(self):
        """Test RiskAverseEnvironmentConfigsAPI initialization with different parameters."""
        print("Testing RiskAverseEnvironmentConfigsAPI initialization...")
        
        # Test default initialization
        api_default = RiskAverseEnvironmentConfigsAPI()
        assert api_default.discount_factor == 0.95, f"Expected 0.95, got {api_default.discount_factor}"
        assert api_default.debug == False, f"Expected False, got {api_default.debug}"
        
        # Test custom initialization
        api_custom = RiskAverseEnvironmentConfigsAPI(discount_factor=0.99, debug=True)
        assert api_custom.discount_factor == 0.99, f"Expected 0.99, got {api_custom.discount_factor}"
        assert api_custom.debug == True, f"Expected True, got {api_custom.debug}"
        
        print("  ✓ RiskAverseEnvironmentConfigsAPI initialization test passed!")
    
    def test_all_configs_have_consistent_interface(self):
        """Test that all configuration methods return consistent types."""
        print("Testing all configurations have consistent interface...")
        
        config_methods = [
            'tiger_pomdp_config',
            'cartpole_pomdp_config', 
            'mountain_car_pomdp_config',
            'push_pomdp_config',
            'rock_sample_pomdp_config',
            'pacman_pomdp_config',
            'laser_tag_pomdp_config',
            'safety_ant_velocity_pomdp_config'
        ]
        
        for method_name in config_methods:
            method = getattr(self.config_api, method_name)
            pomdp, belief = method(n_particles=self.test_n_particles)
            
            # Verify consistent return types
            assert isinstance(pomdp, Environment), f"{method_name} should return Environment"
            assert hasattr(belief, 'particles'), f"{method_name} should return object with particles"
            assert pomdp.discount_factor == 0.95, f"{method_name} should use correct discount factor"
            
        print("  ✓ All configurations have consistent interface test passed!")


def main():
    """Run all environment configuration tests."""
    print("🚀 Running Environment Configuration Tests")
    print("=" * 60)
    
    test_class = TestEnvironmentConfigs()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
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
            import traceback
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
