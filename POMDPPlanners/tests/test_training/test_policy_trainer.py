# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the PolicyTrainer class.

This module tests the PolicyTrainer collect-then-train loop, including
integration with BetaZero and ConstrainedZero, callback dispatch,
and batched collection mode.
"""

# pylint: disable=protected-access

import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.policy import TrainablePolicy
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
    ConstrainedZero,
)
from POMDPPlanners.training import PolicyTrainer
from POMDPPlanners.training.callbacks import EarlyStopping

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def _no_failure(_):
    return False


@pytest.fixture
def tiger_env():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def beta_zero_planner(tiger_env):
    from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

    sampler = DiscreteActionSampler(tiger_env.get_actions())
    return BetaZero(
        environment=tiger_env,
        discount_factor=0.95,
        depth=2,
        name="test_trainer_bz",
        action_sampler=sampler,
        n_simulations=5,
        state_dim=1,
        training_epochs=2,
        training_batch_size=4,
    )


@pytest.fixture
def constrained_zero_planner(tiger_env):
    from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

    sampler = DiscreteActionSampler(tiger_env.get_actions())
    return ConstrainedZero(
        environment=tiger_env,
        discount_factor=0.95,
        depth=2,
        name="test_trainer_cz",
        action_sampler=sampler,
        n_simulations=5,
        state_dim=1,
        failure_fn=_no_failure,
        training_epochs=2,
        training_batch_size=8,
    )


@pytest.fixture
def initial_belief_fn(tiger_env):
    return lambda: get_initial_belief(pomdp=tiger_env, n_particles=5, resampling=True)


class TestPolicyTrainerBasic:
    """Basic tests for PolicyTrainer construction and validation."""

    def test_requires_trainable_policy(self):
        """Test that PolicyTrainer rejects a non-TrainablePolicy.

        Purpose: Validates constructor type checking.

        Given: A plain MagicMock that does not implement TrainablePolicy.
        When: PolicyTrainer is constructed with it.
        Then: TypeError is raised.

        Test type: unit
        """
        with pytest.raises(TypeError, match="TrainablePolicy"):
            PolicyTrainer(
                policy=MagicMock(),
                initial_belief_fn=lambda: None,
            )

    def test_beta_zero_is_trainable(self, beta_zero_planner):
        """Test that BetaZero implements TrainablePolicy.

        Purpose: Validates that BetaZero is recognised as a TrainablePolicy.

        Given: A BetaZero planner instance.
        When: isinstance check against TrainablePolicy.
        Then: Returns True.

        Test type: unit
        """
        assert isinstance(beta_zero_planner, TrainablePolicy)

    def test_constrained_zero_is_trainable(self, constrained_zero_planner):
        """Test that ConstrainedZero inherits TrainablePolicy through BetaZero.

        Purpose: Validates TrainablePolicy inheritance chain.

        Given: A ConstrainedZero planner instance.
        When: isinstance check against TrainablePolicy.
        Then: Returns True.

        Test type: unit
        """
        assert isinstance(constrained_zero_planner, TrainablePolicy)


class TestPolicyTrainerTrain:
    """Integration tests for PolicyTrainer.train()."""

    def test_train_returns_correct_metric_keys_beta_zero(
        self, beta_zero_planner, initial_belief_fn
    ):
        """Test train() returns dict with BetaZero metric keys.

        Purpose: Validates that training returns the expected metric structure.

        Given: A BetaZero planner and a PolicyTrainer with 1 iteration.
        When: trainer.train() is called.
        Then: Returned dict has keys total_loss, value_loss, policy_loss.

        Test type: integration
        """
        trainer = PolicyTrainer(
            policy=beta_zero_planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=1,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
        )
        metrics = trainer.train()

        assert set(metrics.keys()) == {"total_loss", "value_loss", "policy_loss"}
        assert len(metrics["total_loss"]) > 0

    def test_train_returns_failure_loss_for_constrained_zero(
        self, constrained_zero_planner, initial_belief_fn
    ):
        """Test train() returns failure_loss for ConstrainedZero.

        Purpose: Validates that ConstrainedZero's extra metric key appears.

        Given: A ConstrainedZero planner and a PolicyTrainer with 1 iteration.
        When: trainer.train() is called.
        Then: Returned dict includes failure_loss.

        Test type: integration
        """
        trainer = PolicyTrainer(
            policy=constrained_zero_planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=1,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
        )
        metrics = trainer.train()

        assert "failure_loss" in metrics

    def test_collecting_data_flag_toggled(self, beta_zero_planner, initial_belief_fn):
        """Test that _collecting_data is toggled during training.

        Purpose: Validates that the trainer sets _collecting_data=True during
        collection and False afterwards.

        Given: A BetaZero planner with _collecting_data=False.
        When: PolicyTrainer.train() completes.
        Then: _collecting_data is False after training finishes.

        Test type: unit
        """
        assert beta_zero_planner._collecting_data is False

        trainer = PolicyTrainer(
            policy=beta_zero_planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=1,
            episodes_per_iteration=1,
            episode_length=3,
            verbose=False,
        )
        trainer.train()

        assert beta_zero_planner._collecting_data is False

    def test_batched_collection_mode(self, beta_zero_planner, initial_belief_fn):
        """Test that batched_collection=True uses network-only rollouts.

        Purpose: Validates that the batched collection path works end-to-end.

        Given: A BetaZero planner and PolicyTrainer with batched_collection=True.
        When: trainer.train() is called.
        Then: Training completes without error and returns metric values.

        Test type: integration
        """
        trainer = PolicyTrainer(
            policy=beta_zero_planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=1,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
            batched_collection=True,
        )
        metrics = trainer.train()

        assert "total_loss" in metrics
        assert len(metrics["total_loss"]) > 0


class TestPolicyTrainerEarlyStopping:
    """Tests for early stopping via callbacks."""

    def test_early_stopping_limits_iterations(self, beta_zero_planner, initial_belief_fn):
        """Test that EarlyStopping can stop training before num_iterations.

        Purpose: Validates callback-driven early stopping.

        Given: A PolicyTrainer with num_iterations=100 and EarlyStopping
               with patience=0 (stop on first non-improving iteration).
        When: trainer.train() is called.
        Then: Training ends before completing all 100 iterations.

        Test type: integration
        """
        cb = EarlyStopping(monitor="total_loss", patience=0, mode="min")

        trainer = PolicyTrainer(
            policy=beta_zero_planner,
            initial_belief_fn=initial_belief_fn,
            num_iterations=100,
            episodes_per_iteration=2,
            episode_length=5,
            verbose=False,
            callbacks=[cb],
        )
        metrics = trainer.train()

        # With patience=0, training stops as soon as a non-improving iteration is seen,
        # which must happen well before all 100 iterations complete.
        assert len(metrics["total_loss"]) < 100
