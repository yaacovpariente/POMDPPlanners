"""Tests to verify package installation and basic functionality."""
import importlib
import pkg_resources
import sys
from pathlib import Path

def test_package_installed():
    """Test that the package is installed and importable.
    
    Purpose: Validates package installed
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    try:
        import POMDPPlanners
        assert POMDPPlanners.__version__ == "0.1.0"
    except ImportError as e:
        raise AssertionError(f"Failed to import POMDPPlanners: {e}")

def test_required_packages():
    """Test that all required packages are installed with correct versions.
    
    Purpose: Validates required packages
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Get the requirements from requirements.txt
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Convert requirements to pkg_resources.Requirement objects
    required_packages = [pkg_resources.Requirement.parse(req) for req in requirements]
    
    # Check each requirement
    for requirement in required_packages:
        try:
            pkg_resources.require(str(requirement))
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
            raise AssertionError(f"Package requirement not met: {requirement} - {e}")

def test_imports():
    """Test that key package modules can be imported.
    
    Purpose: Validates imports
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Add your main package modules here
    modules_to_test = [
        "POMDPPlanners",
        "POMDPPlanners.core",
        "POMDPPlanners.planners",
        "POMDPPlanners.environments",
        "POMDPPlanners.utils",
        "POMDPPlanners.simulations",
    ]
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            raise AssertionError(f"Failed to import {module_name}: {e}") 