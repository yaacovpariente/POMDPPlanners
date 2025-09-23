"""Tests to verify package installation and basic functionality."""

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pkg_resources

np.random.seed(42)
random.seed(42)


def test_package_installed():
    """Test that the package is installed and importable.

    Purpose: Validates that POMDPPlanners package is properly installed and accessible with correct version

    Given: POMDPPlanners package should be installed in the Python environment with version 0.1.0
    When: Package is imported and version is checked
    Then: Import succeeds without error and version matches expected 0.1.0

    Test type: unit
    """
    try:
        import POMDPPlanners

        assert POMDPPlanners.__version__ == "0.1.0"
    except ImportError as e:
        raise AssertionError(f"Failed to import POMDPPlanners: {e}")


def test_required_packages():
    """Test that all required packages are installed with correct versions.

    Purpose: Validates that all dependencies listed in requirements.txt are properly installed with compatible versions

    Given: requirements.txt file containing package dependencies with version constraints
    When: Each required package is checked against installed packages using pkg_resources
    Then: All requirements are satisfied without DistributionNotFound or VersionConflict errors

    Test type: unit
    """
    # Get the requirements from requirements.txt
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

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

    Purpose: Validates that all main POMDPPlanners modules can be successfully imported without errors

    Given: POMDPPlanners package with core, planners, environments, utils, and simulations submodules
    When: Each main module is imported using importlib.import_module
    Then: All imports succeed without ImportError exceptions

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
