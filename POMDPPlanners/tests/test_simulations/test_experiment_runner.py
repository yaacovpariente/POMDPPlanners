"""Tests for the experiment runner module."""

import pytest
import tempfile
import yaml
from pathlib import Path
import pandas as pd
from POMDPPlanners.simulations.experiment_runner import ExperimentRunner
from POMDPPlanners.core.simulation import History
from POMDPPlanners.utils.weighted_particle_beliefs import WeightedParticleBeliefDiscreteLightDark

@pytest.fixture
def minimal_config():
    """Create a minimal valid configuration for testing."""
    config = {
        'environment': {
            'type': 'SanityPOMDP',  # Using the simplest environment
            'params': {}  # SanityPOMDP doesn't require any parameters
        },
        'belief': {
            'type': 'WeightedParticleBeliefSanityPOMDP',
            'params': {
                'n_particles': 10,
                'resampling': True,
                'ess_threshold': 0.5,
                'reinvigoration_fraction': 0.2
            }
        },
        'policies': [
            {
                'type': 'POMCP',  # Using the simplest policy
                'params': {
                    'depth': 10,  # Maximum depth for MCTS
                    'exploration_constant': 1.0,
                    'n_simulations': 10,  # Number of simulations per action
                    'min_samples_per_node': 10,
                    'discount_factor': 0.95,  # Match environment's discount factor
                    'name': 'TestPOMCP'  # Add a name for the policy
                }
            }
        ],
        'num_steps': 10,
        'num_episodes': 2
    }
    
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()

def test_experiment_runner_initialization(minimal_config):
    """Test that the experiment runner initializes correctly."""
    runner = ExperimentRunner(minimal_config)
    assert runner.config is not None
    assert runner.environment is None
    assert runner.policies == []

def test_setup_environment(minimal_config):
    """Test environment setup."""
    runner = ExperimentRunner(minimal_config)
    runner.setup_environment()
    assert runner.environment is not None
    assert runner.environment.__class__.__name__ == 'SanityPOMDP'

def test_setup_policies(minimal_config):
    """Test policy setup."""
    runner = ExperimentRunner(minimal_config)
    runner.setup_environment()
    runner.setup_policies()
    assert len(runner.policies) == 1
    assert runner.policies[0].__class__.__name__ == 'POMCP'

def test_run_experiment(minimal_config):
    """Test running a complete experiment."""
    runner = ExperimentRunner(minimal_config)
    with tempfile.TemporaryDirectory() as temp_dir:
        histories, statistics_df = runner.run_experiment(cache_dir_path=Path(temp_dir))
    
    # Verify results
    assert isinstance(histories, dict)
    assert isinstance(statistics_df, pd.DataFrame)
    assert len(statistics_df) > 0  # Should have some statistics

def test_invalid_config():
    """Test handling of invalid configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content")
        temp_path = f.name
    
    try:
        with pytest.raises(yaml.YAMLError):
            ExperimentRunner(temp_path)
    finally:
        Path(temp_path).unlink()

def test_missing_environment_config():
    """Test handling of missing environment configuration."""
    config = {
        'belief': {
            'type': 'WeightedParticleBeliefDiscreteLightDark',
            'params': {
                'n_particles': 10,
                'resampling': True
            }
        },
        'policies': [
            {
                'type': 'POMCP',
                'params': {
                    'num_simulations': 10,
                    'exploration_constant': 1.0,
                    'discount_factor': 0.95,
                    'name': 'TestPOMCP'
                }
            }
        ],
        'num_steps': 10,
        'num_episodes': 2
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(KeyError, match="Missing required 'environment' or 'type' in configuration"):
            ExperimentRunner(temp_path)
    finally:
        Path(temp_path).unlink()

def test_missing_policies_config():
    """Test handling of missing policies configuration."""
    config = {
        'environment': {
            'type': 'SanityPOMDP',
            'params': {}
        },
        'belief': {
            'type': 'WeightedParticleBeliefDiscreteLightDark',
            'params': {
                'n_particles': 10,
                'resampling': True
            }
        },
        'num_steps': 10,
        'num_episodes': 2
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(KeyError, match="Missing required 'policies' list in configuration"):
            ExperimentRunner(temp_path)
    finally:
        Path(temp_path).unlink()

def test_missing_belief_config():
    """Test handling of missing belief configuration."""
    config = {
        'environment': {
            'type': 'SanityPOMDP',
            'params': {}
        },
        'policies': [
            {
                'type': 'POMCP',
                'params': {
                    'depth': 10,
                    'exploration_constant': 1.0,
                    'n_simulations': 10,
                    'discount_factor': 0.95,
                    'name': 'TestPOMCP'
                }
            }
        ],
        'num_steps': 10,
        'num_episodes': 2
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(KeyError, match="Missing required 'belief' or 'type' in configuration"):
            ExperimentRunner(temp_path)
    finally:
        Path(temp_path).unlink()

def test_invalid_belief_type():
    """Test handling of invalid belief type."""
    config = {
        'environment': {
            'type': 'SanityPOMDP',
            'params': {}
        },
        'belief': {
            'type': 'InvalidBeliefType',
            'params': {
                'n_particles': 10,
                'resampling': True
            }
        },
        'policies': [
            {
                'type': 'POMCP',
                'params': {
                    'depth': 10,
                    'exploration_constant': 1.0,
                    'n_simulations': 10,
                    'discount_factor': 0.95,
                    'name': 'TestPOMCP'
                }
            }
        ],
        'num_steps': 10,
        'num_episodes': 2
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        runner = ExperimentRunner(temp_path)
        with pytest.raises(ValueError, match="Belief class 'InvalidBeliefType' not found"):
            runner.run_experiment()
    finally:
        Path(temp_path).unlink() 