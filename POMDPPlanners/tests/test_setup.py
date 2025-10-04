"""Tests to verify package installation and basic functionality."""

import importlib
import importlib.metadata
import random
import sys
from pathlib import Path

import numpy as np

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
    When: Each required package is checked against installed packages using importlib.metadata
    Then: All requirements are satisfied without PackageNotFoundError or version conflicts

    Test type: unit
    """
    # Get the requirements from requirements.txt
    requirements_file = Path(__file__).parent.parent.parent / "requirements.txt"
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Check each requirement
    for req in requirements:
        try:
            # Parse package name and version constraint
            if "==" in req:
                package_name, version = req.split("==")
                installed_version = importlib.metadata.version(package_name.strip())
                if installed_version != version.strip():
                    raise AssertionError(
                        f"Package {package_name} version mismatch: expected {version}, got {installed_version}"
                    )
            elif ">=" in req:
                package_name, min_version = req.split(">=")
                installed_version = importlib.metadata.version(package_name.strip())
                # Simple version comparison (could be improved with packaging.version)
                if installed_version < min_version.strip():
                    raise AssertionError(
                        f"Package {package_name} version too old: expected >= {min_version}, got {installed_version}"
                    )
            else:
                # Just check if package is installed
                importlib.metadata.version(req.strip())
        except importlib.metadata.PackageNotFoundError as e:
            raise AssertionError(f"Package requirement not met: {req} - {e}")


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
