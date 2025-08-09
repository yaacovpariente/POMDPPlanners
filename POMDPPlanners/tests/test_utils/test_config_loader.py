import pytest
import yaml
from pathlib import Path
import tempfile
from POMDPPlanners.utils.config_loader import load_config

def test_load_config_valid_yaml():
    """Test loading a valid YAML configuration.
    
    Purpose: Validates load config valid yaml
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    # Create a temporary YAML file with valid content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        assert 'environment' in config
        assert config['environment']['name'] == "test_env"
        assert config['environment']['type'] == "TestEnv"
        assert config['environment']['params']['param1'] == 1
        assert config['environment']['params']['param2'] == "test"
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()

def test_load_config_invalid_yaml():
    """Test loading an invalid YAML configuration.
    
    Purpose: Validates load config invalid yaml
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    # Create a temporary YAML file with invalid content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
    
    Purpose: Validates load config nonexistent file
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")

def test_load_config_empty_file():
    """Test loading an empty YAML file.
    
    Purpose: Validates load config empty file
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
    
    Purpose: Validates load config complex yaml
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: configuration
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        assert 'environment' in config
        assert 'policies' in config
        
        # Check environment configuration
        env_config = config['environment']
        assert env_config['name'] == "complex_env"
        assert env_config['type'] == "ComplexEnv"
        assert env_config['params']['nested']['list'] == [1, 2, 3]
        assert env_config['params']['nested']['dict']['key1'] == "value1"
        assert env_config['params']['settings']['enabled'] is True
        assert env_config['params']['settings']['timeout'] == 30
        
        # Check policies configuration
        policies = config['policies']
        assert len(policies) == 2
        assert policies[0]['name'] == "policy1"
        assert policies[0]['type'] == "Type1"
        assert policies[0]['params']['learning_rate'] == 0.01
        assert policies[1]['name'] == "policy2"
        assert policies[1]['type'] == "Type2"
        assert policies[1]['params']['batch_size'] == 32
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink() 