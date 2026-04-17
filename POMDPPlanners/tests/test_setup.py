"""Tests to verify package installation and basic functionality."""

import importlib
import importlib.metadata
import random

import numpy as np
import POMDPPlanners
from packaging import version

np.random.seed(42)
random.seed(42)


def test_package_installed():
    """Test that the package is installed and importable.

    Purpose: Validates that POMDPPlanners package is properly installed and accessible with correct version

    Given: POMDPPlanners package should be installed in the Python environment with version 0.2.0
    When: Package is imported and version is checked
    Then: Import succeeds without error and version matches expected 0.2.0

    Test type: unit
    """
    try:
        assert POMDPPlanners.__version__ == "0.2.0"
    except ImportError as e:
        raise AssertionError(f"Failed to import POMDPPlanners: {e}")


def test_required_packages():
    """Test that all required packages are installed with correct versions.

    Purpose: Validates that all dependencies declared in pyproject.toml are properly installed with compatible versions

    Given: POMDPPlanners package installed with dependencies declared in pyproject.toml
    When: Each required package is checked against installed packages using importlib.metadata
    Then: All requirements are satisfied without PackageNotFoundError or version conflicts

    Test type: unit
    """
    requirements = importlib.metadata.requires("POMDPPlanners") or []
    # Filter to direct dependencies only (exclude extras like dev/docs)
    requirements = [r for r in requirements if "; extra ==" not in r]

    # Check each requirement
    for req in requirements:
        try:
            # Parse package name and version constraint
            if "==" in req:
                package_name, expected_version = req.split("==")
                installed_version = importlib.metadata.version(package_name.strip())
                if installed_version != expected_version.strip():
                    raise AssertionError(
                        f"Package {package_name} version mismatch: expected {expected_version}, got {installed_version}"
                    )
            elif ">=" in req:
                package_name, min_version = req.split(">=")
                installed_version = importlib.metadata.version(package_name.strip())
                # Proper semantic version comparison
                if version.parse(installed_version) < version.parse(min_version.strip()):
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
