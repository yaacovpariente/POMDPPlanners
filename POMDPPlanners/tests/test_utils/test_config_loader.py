# SPDX-License-Identifier: MIT

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from POMDPPlanners.utils.config_loader import load_config

np.random.seed(42)
random.seed(42)


def test_load_config_valid_yaml():
    """Test loading a valid YAML configuration.

    Purpose: Validates that load_config correctly parses and returns valid YAML configuration data

    Given: Temporary YAML file with valid nested structure containing environment configuration with name, type, and parameters
    When: load_config is called with the YAML file path
    Then: Returns dictionary with correct structure and all values preserved (strings, integers, nested dicts)

    Test type: configuration
    """
    # Create a temporary YAML file with valid content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_content = """
        environment:
          name: "test_env"
          type: "TestEnv"
          params:
            param1: 1
            param2: "test"
        """
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Load the configuration
        config = load_config(temp_path)

        # Verify the loaded configuration
        assert isinstance(config, dict)
        assert "environment" in config
        assert config["environment"]["name"] == "test_env"
        assert config["environment"]["type"] == "TestEnv"
        assert config["environment"]["params"]["param1"] == 1
        assert config["environment"]["params"]["param2"] == "test"
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()


def test_load_config_invalid_yaml():
    """Test loading an invalid YAML configuration.

    Purpose: Validates that load_config raises YAMLError when given malformed YAML syntax

    Given: Temporary YAML file with invalid syntax (mixing dict and list elements incorrectly)
    When: load_config attempts to parse the malformed YAML file
    Then: YAMLError exception is raised indicating parsing failure

    Test type: configuration
    """
    # Create a temporary YAML file with invalid content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        invalid_yaml = """
        environment:
          name: "test_env"
          type: "TestEnv"
          params:
            param1: 1
            param2: "test"
            - invalid: yaml
        """
        f.write(invalid_yaml)
        temp_path = f.name

    try:
        # Verify that loading invalid YAML raises an exception
        with pytest.raises(yaml.YAMLError):
            load_config(temp_path)
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()


def test_load_config_nonexistent_file():
    """Test loading a configuration from a nonexistent file.

    Purpose: Validates that load_config raises FileNotFoundError when file does not exist

    Given: Non-existent YAML file path ("nonexistent_file.yaml")
    When: load_config is called with the invalid file path
    Then: FileNotFoundError exception is raised

    Test type: configuration
    """
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")


def test_load_config_empty_file():
    """Test loading an empty YAML file.

    Purpose: Validates that load_config returns None when processing empty YAML files

    Given: Empty temporary YAML file with no content
    When: load_config processes the empty file
    Then: Returns None indicating no configuration data present

    Test type: configuration
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        # Load the empty configuration
        config = load_config(temp_path)

        # Verify that an empty YAML file returns None
        assert config is None
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()


def test_load_config_complex_yaml():
    """Test loading a complex YAML configuration with nested structures.

    Purpose: Validates that load_config correctly handles complex YAML with deeply nested dictionaries, lists, and mixed data types

    Given: Complex YAML file with environment config (nested dict with lists), policies array, and various data types (strings, booleans, integers, floats)
    When: load_config processes the complex nested structure
    Then: All nested structures are preserved with correct types including lists [1,2,3], nested dicts, boolean true, and policy arrays

    Test type: configuration
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        complex_yaml = """
        environment:
          name: "complex_env"
          type: "ComplexEnv"
          params:
            nested:
              list: [1, 2, 3]
              dict:
                key1: value1
                key2: value2
            settings:
              enabled: true
              timeout: 30
        policies:
          - name: "policy1"
            type: "Type1"
            params:
              learning_rate: 0.01
          - name: "policy2"
            type: "Type2"
            params:
              batch_size: 32
        """
        f.write(complex_yaml)
        temp_path = f.name

    try:
        # Load the configuration
        config = load_config(temp_path)

        # Verify the loaded configuration
        assert isinstance(config, dict)
        assert "environment" in config
        assert "policies" in config

        # Check environment configuration
        env_config = config["environment"]
        assert env_config["name"] == "complex_env"
        assert env_config["type"] == "ComplexEnv"
        assert env_config["params"]["nested"]["list"] == [1, 2, 3]
        assert env_config["params"]["nested"]["dict"]["key1"] == "value1"
        assert env_config["params"]["settings"]["enabled"] is True
        assert env_config["params"]["settings"]["timeout"] == 30

        # Check policies configuration
        policies = config["policies"]
        assert len(policies) == 2
        assert policies[0]["name"] == "policy1"
        assert policies[0]["type"] == "Type1"
        assert policies[0]["params"]["learning_rate"] == 0.01
        assert policies[1]["name"] == "policy2"
        assert policies[1]["type"] == "Type2"
        assert policies[1]["params"]["batch_size"] == 32
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()
