"""Interface tests for POMDP planner save/load functionality.

This module tests that all POMDP planners can be properly saved to JSON
and loaded back, preserving all configuration parameters. This is crucial for:
- Saving trained/configured planners for later use
- Sharing planner configurations
- Version control of planner setups
- Reproducible experiments
"""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments import DiscreteLightDarkPOMDP, TigerPOMDP
from POMDPPlanners.planners import (
    POMCP,
    POMCP_DPW,
    POMCPOW,
    PFT_DPW,
    DiscreteActionSequencesPlanner,
    SparsePFT,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

# Set seeds for reproducible tests
np.random.seed(42)


class TestPlannerSaveLoadInterface:
    """Interface tests for planner save/load across all planner types."""

    def setup_method(self):
        """Set up test environments for each test."""
        self.tiger_env = TigerPOMDP(discount_factor=0.95)
        self.light_dark_env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    def _test_planner_save_load(self, planner_class: type, init_params: Dict[str, Any]):
        """Helper to test planner save/load.

        Purpose: Validates that planner can be saved to JSON and loaded

        Given: A planner class and initialization parameters
        When: Planner is saved to file and loaded
        Then: Loaded planner has identical parameters and config_id

        Test type: interface

        Args:
            planner_class: Planner class to test
            init_params: Parameters for planner initialization
        """
        # Create planner
        planner = planner_class(**init_params)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            filepath = planner.save(temp_path)

            # Load planner
            loaded_planner = planner_class.load(filepath)

            # Verify properties
            assert loaded_planner.name == planner.name
            assert loaded_planner.discount_factor == planner.discount_factor
            assert loaded_planner.config_id == planner.config_id
            assert loaded_planner.environment.config_id == planner.environment.config_id

            # Verify specific parameters based on planner type
            if hasattr(planner, "depth"):
                assert loaded_planner.depth == planner.depth
            if hasattr(planner, "exploration_constant"):
                assert loaded_planner.exploration_constant == planner.exploration_constant
            if hasattr(planner, "n_simulations"):
                assert loaded_planner.n_simulations == planner.n_simulations

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pomcp_save_load(self):
        """Test POMCP save/load interface.

        Purpose: Validates POMCP can be saved and loaded with all parameters preserved

        Given: POMCP planner instance with specific parameters
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original configuration

        Test type: interface
        """
        self._test_planner_save_load(
            POMCP,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 10.0,
                "name": "POMCP_Test",
                "n_simulations": 100,
            },
        )

    def test_pomcp_dpw_save_load(self):
        """Test POMCP_DPW save/load interface.

        Purpose: Validates POMCP_DPW can be saved and loaded with DPW parameters

        Given: POMCP_DPW planner with progressive widening parameters
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original configuration including DPW params

        Test type: interface
        """
        # Create action sampler
        action_sampler = DiscreteActionSampler(self.light_dark_env.get_actions())

        self._test_planner_save_load(
            POMCP_DPW,
            {
                "environment": self.light_dark_env,
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 10.0,
                "k_a": 5.0,
                "alpha_a": 0.5,
                "k_o": 5.0,
                "alpha_o": 0.5,
                "action_sampler": action_sampler,
                "name": "POMCP_DPW_Test",
                "n_simulations": 100,
            },
        )

    def test_pomcpow_save_load(self):
        """Test POMCPOW save/load interface.

        Purpose: Validates POMCPOW can be saved and loaded

        Given: POMCPOW planner with observation widening
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original configuration

        Test type: interface
        """
        # Create action sampler
        action_sampler = DiscreteActionSampler(self.light_dark_env.get_actions())

        self._test_planner_save_load(
            POMCPOW,
            {
                "environment": self.light_dark_env,
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 10.0,
                "k_a": 5.0,
                "alpha_a": 0.5,
                "k_o": 5.0,
                "alpha_o": 0.5,
                "action_sampler": action_sampler,
                "name": "POMCPOW_Test",
                "n_simulations": 100,
            },
        )

    def test_pft_dpw_save_load(self):
        """Test PFT_DPW save/load interface.

        Purpose: Validates PFT_DPW can be saved and loaded with ActionSampler

        Given: PFT_DPW planner with action sampler
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original including action sampler

        Test type: interface
        """
        # Create action sampler
        action_sampler = DiscreteActionSampler(self.light_dark_env.get_actions())

        self._test_planner_save_load(
            PFT_DPW,
            {
                "environment": self.light_dark_env,
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 10.0,
                "k_a": 5.0,
                "alpha_a": 0.5,
                "k_o": 5.0,
                "alpha_o": 0.5,
                "action_sampler": action_sampler,
                "min_visit_count_per_action": 1,
                "name": "PFT_DPW_Test",
                "n_simulations": 100,
            },
        )

    def test_sparse_pft_save_load(self):
        """Test SparsePFT save/load interface.

        Purpose: Validates SparsePFT can be saved and loaded

        Given: SparsePFT planner with specific parameters
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original configuration

        Test type: interface
        """
        self._test_planner_save_load(
            SparsePFT,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "gamma": 0.95,
                "depth": 10,
                "c_ucb": 1.0,
                "beta_ucb": 0.5,
                "belief_child_num": 5,
                "name": "SparsePFT_Test",
                "n_simulations": 100,
            },
        )

    def test_discrete_action_sequences_planner_save_load(self):
        """Test DiscreteActionSequencesPlanner save/load interface.

        Purpose: Validates DiscreteActionSequencesPlanner can be saved and loaded

        Given: DiscreteActionSequencesPlanner (non-MCTS planner)
        When: Planner is saved to JSON and loaded
        Then: Loaded planner matches original configuration

        Test type: interface
        """
        self._test_planner_save_load(
            DiscreteActionSequencesPlanner,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "depth": 3,
                "n_return_samples": 10,
                "name": "DiscreteActionSeq_Test",
            },
        )

    def test_save_with_default_filepath(self):
        """Test default filepath generation.

        Purpose: Validates automatic filepath generation works correctly

        Given: A planner without specifying save path
        When: save() is called without filepath argument
        Then: File is saved to default location with correct structure

        Test type: interface
        """
        planner = POMCP(
            environment=self.tiger_env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=10.0,
            name="TestPlanner",
            n_simulations=100,
        )

        try:
            # Save with default path
            filepath = planner.save()

            # Verify file exists
            assert filepath.exists()

            # Verify path structure
            assert "TigerPOMDP" in str(filepath)
            assert "POMCP" in str(filepath)
            assert "TestPlanner" in str(filepath)

            # Verify can load from default path
            loaded_planner = POMCP.load(filepath)
            assert loaded_planner.config_id == planner.config_id

        finally:
            # Cleanup - remove the created file and directories
            if filepath.exists():
                filepath.unlink()
                # Try to remove parent directories if empty
                try:
                    filepath.parent.rmdir()  # Remove policy_class dir
                    filepath.parent.parent.rmdir()  # Remove env_name dir
                    filepath.parent.parent.parent.rmdir()  # Remove saved_policies dir
                except OSError:
                    pass  # Directories not empty, that's fine

    def test_load_nonexistent_file_raises_error(self):
        """Test FileNotFoundError on missing file.

        Purpose: Validates proper error handling for missing files

        Given: A non-existent filepath
        When: load() is called with that filepath
        Then: FileNotFoundError is raised

        Test type: interface
        """
        with pytest.raises(FileNotFoundError):
            POMCP.load("nonexistent_policy.json")

    def test_load_invalid_json_raises_error(self):
        """Test ValueError on corrupted JSON.

        Purpose: Validates proper error handling for invalid JSON

        Given: A file with invalid JSON content
        When: load() is called with that filepath
        Then: ValueError is raised

        Test type: interface
        """
        # Create temp file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                POMCP.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
