# SPDX-License-Identifier: MIT

"""Tests for simulation configuration classes."""

import pytest

from POMDPPlanners.core.simulation.simulation_configs import EnvironmentRunParams
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import tiger_pinned_kwargs
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)


class TestEnvironmentRunParamsConfigId:
    """Test suite for EnvironmentRunParams config_id uniqueness and equality."""

    def test_identical_params_produce_same_config_id(self):
        """Test that identical parameters produce the same config_id.

        Purpose: Validates that config_id is deterministic for identical configurations

        Given: Two EnvironmentRunParams instances with identical parameters
        When: config_id is computed for both instances
        Then: Both instances have the same config_id

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params1.config_id == params2.config_id

    def test_different_environments_produce_different_config_ids(self):
        """Test that different environments produce different config_ids.

        Purpose: Validates that config_id distinguishes between different environments

        Given: Two EnvironmentRunParams with different TigerPOMDP configurations
        When: config_id is computed for both instances
        Then: config_ids are different

        Test type: unit
        """
        env1 = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        env2 = TigerPOMDP(discount_factor=0.90, **tiger_pinned_kwargs())

        belief1 = get_initial_belief(env1, n_particles=100)
        belief2 = get_initial_belief(env2, n_particles=100)

        policy1 = DiscreteActionSequencesPlanner(
            environment=env1,
            discount_factor=0.95,
            name="test_planner1",
            depth=5,
            n_return_samples=10,
        )
        policy2 = DiscreteActionSequencesPlanner(
            environment=env2,
            discount_factor=0.90,
            name="test_planner2",
            depth=5,
            n_return_samples=10,
        )

        params1 = EnvironmentRunParams(
            environment=env1, belief=belief1, policies=[policy1], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env2, belief=belief2, policies=[policy2], num_episodes=10, num_steps=20
        )

        assert params1.config_id != params2.config_id

    def test_different_num_particles_produce_different_config_ids(self):
        """Test that different belief configurations produce different config_ids.

        Purpose: Validates that config_id distinguishes between different belief states

        Given: Two EnvironmentRunParams with different particle counts in belief
        When: config_id is computed for both instances
        Then: config_ids are different

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())

        belief1 = get_initial_belief(env, n_particles=100)
        belief2 = get_initial_belief(env, n_particles=200)

        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief1, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief2, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params1.config_id != params2.config_id

    def test_different_policies_produce_different_config_ids(self):
        """Test that different policy configurations produce different config_ids.

        Purpose: Validates that config_id distinguishes between different policy configurations

        Given: Two EnvironmentRunParams with different planner hyperparameters
        When: config_id is computed for both instances
        Then: config_ids are different

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)

        policy1 = DiscreteActionSequencesPlanner(
            environment=env,
            discount_factor=0.95,
            name="test_planner1",
            depth=5,
            n_return_samples=10,
        )
        policy2 = DiscreteActionSequencesPlanner(
            environment=env,
            discount_factor=0.95,
            name="test_planner2",
            depth=5,
            n_return_samples=20,
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy1], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy2], num_episodes=10, num_steps=20
        )

        assert params1.config_id != params2.config_id

    def test_different_num_episodes_produce_different_config_ids(self):
        """Test that different num_episodes values produce different config_ids.

        Purpose: Validates that config_id distinguishes between different episode counts

        Given: Two EnvironmentRunParams with different num_episodes values
        When: config_id is computed for both instances
        Then: config_ids are different

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=20, num_steps=20
        )

        assert params1.config_id != params2.config_id

    def test_different_num_steps_produce_different_config_ids(self):
        """Test that different num_steps values produce different config_ids.

        Purpose: Validates that config_id distinguishes between different step counts

        Given: Two EnvironmentRunParams with different num_steps values
        When: config_id is computed for both instances
        Then: config_ids are different

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=30
        )

        assert params1.config_id != params2.config_id

    def test_policy_order_does_not_affect_config_id(self):
        """Test that policy order does not affect config_id due to sorting.

        Purpose: Validates that config_id is independent of policy ordering

        Given: Two EnvironmentRunParams with same policies in different orders
        When: config_id is computed for both instances
        Then: Both instances have the same config_id

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)

        policy1 = DiscreteActionSequencesPlanner(
            environment=env,
            discount_factor=0.95,
            name="test_planner1",
            depth=5,
            n_return_samples=10,
        )
        policy2 = DiscreteActionSequencesPlanner(
            environment=env,
            discount_factor=0.95,
            name="test_planner2",
            depth=5,
            n_return_samples=20,
        )

        params1 = EnvironmentRunParams(
            environment=env,
            belief=belief,
            policies=[policy1, policy2],
            num_episodes=10,
            num_steps=20,
        )

        params2 = EnvironmentRunParams(
            environment=env,
            belief=belief,
            policies=[policy2, policy1],
            num_episodes=10,
            num_steps=20,
        )

        assert params1.config_id == params2.config_id


class TestEnvironmentRunParamsHashAndEquality:
    """Test suite for EnvironmentRunParams hash and equality implementation."""

    def test_hash_consistency_with_config_id(self):
        """Test that hash is consistent with config_id.

        Purpose: Validates that hash is based on config_id for proper set/dict behavior

        Given: Two EnvironmentRunParams instances with identical parameters
        When: hash() is called on both instances
        Then: Both instances have the same hash value

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert hash(params1) == hash(params2)

    def test_equality_with_identical_params(self):
        """Test that equality works correctly for identical parameters.

        Purpose: Validates that __eq__ correctly identifies identical configurations

        Given: Two EnvironmentRunParams instances with identical parameters
        When: Equality comparison is performed
        Then: Instances are equal

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params1 == params2

    def test_inequality_with_different_params(self):
        """Test that inequality works correctly for different parameters.

        Purpose: Validates that __eq__ correctly identifies different configurations

        Given: Two EnvironmentRunParams instances with different num_episodes
        When: Equality comparison is performed
        Then: Instances are not equal

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=20, num_steps=20
        )

        assert params1 != params2

    def test_equality_with_non_environment_run_params(self):
        """Test that equality returns False for non-EnvironmentRunParams objects.

        Purpose: Validates that __eq__ handles type checking correctly

        Given: An EnvironmentRunParams instance and a non-EnvironmentRunParams object
        When: Equality comparison is performed
        Then: Result is False

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params != "not_an_environment_run_params"
        assert params != 42
        assert params is not None

    def test_usable_in_set(self):
        """Test that EnvironmentRunParams can be used in sets correctly.

        Purpose: Validates that hash and equality enable proper set behavior

        Given: Multiple EnvironmentRunParams instances including duplicates
        When: Instances are added to a set
        Then: Set contains only unique configurations based on config_id

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params3 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=20, num_steps=20
        )

        params_set = {params1, params2, params3}
        assert len(params_set) == 2  # params1 and params2 are duplicates

    def test_usable_as_dict_key(self):
        """Test that EnvironmentRunParams can be used as dictionary keys.

        Purpose: Validates that hash and equality enable proper dict key behavior

        Given: EnvironmentRunParams instances used as dictionary keys
        When: Dictionary operations are performed
        Then: Duplicate configs map to the same key

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params1 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params2 = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        params_dict = {params1: "value1"}
        params_dict[params2] = "value2"

        assert len(params_dict) == 1  # params1 and params2 are the same key
        assert params_dict[params1] == "value2"


class TestEnvironmentRunParamsInputValidation:
    """Test suite for EnvironmentRunParams input validation."""

    def test_num_episodes_must_be_integer(self):
        """Test that num_episodes must be an integer.

        Purpose: Validates that TypeError is raised when num_episodes is not an integer

        Given: EnvironmentRunParams with non-integer num_episodes
        When: EnvironmentRunParams is instantiated
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(TypeError, match="num_episodes must be an integer"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=[policy],
                num_episodes=10.5,  # type: ignore[arg-type]
                num_steps=20,
            )

        with pytest.raises(TypeError, match="num_episodes must be an integer"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=[policy],
                num_episodes="10",  # type: ignore[arg-type]
                num_steps=20,
            )

    def test_num_steps_must_be_integer(self):
        """Test that num_steps must be an integer.

        Purpose: Validates that TypeError is raised when num_steps is not an integer

        Given: EnvironmentRunParams with non-integer num_steps
        When: EnvironmentRunParams is instantiated
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(TypeError, match="num_steps must be an integer"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=[policy],
                num_episodes=10,
                num_steps=20.5,  # type: ignore[arg-type]
            )

        with pytest.raises(TypeError, match="num_steps must be an integer"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=[policy],
                num_episodes=10,
                num_steps="20",  # type: ignore[arg-type]
            )

    def test_num_episodes_must_be_positive(self):
        """Test that num_episodes must be positive.

        Purpose: Validates that ValueError is raised when num_episodes is non-positive

        Given: EnvironmentRunParams with non-positive num_episodes
        When: EnvironmentRunParams is instantiated
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            EnvironmentRunParams(
                environment=env, belief=belief, policies=[policy], num_episodes=0, num_steps=20
            )

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            EnvironmentRunParams(
                environment=env, belief=belief, policies=[policy], num_episodes=-1, num_steps=20
            )

    def test_num_steps_must_be_positive(self):
        """Test that num_steps must be positive.

        Purpose: Validates that ValueError is raised when num_steps is non-positive

        Given: EnvironmentRunParams with non-positive num_steps
        When: EnvironmentRunParams is instantiated
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(ValueError, match="num_steps must be positive"):
            EnvironmentRunParams(
                environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=0
            )

        with pytest.raises(ValueError, match="num_steps must be positive"):
            EnvironmentRunParams(
                environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=-1
            )

    def test_environment_must_be_environment_instance(self):
        """Test that environment must be an Environment instance.

        Purpose: Validates that TypeError is raised when environment is not an Environment instance

        Given: EnvironmentRunParams with non-Environment environment
        When: EnvironmentRunParams is instantiated
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(TypeError, match="environment must be an Environment instance"):
            EnvironmentRunParams(
                environment="not_an_environment",  # type: ignore[arg-type]
                belief=belief,
                policies=[policy],
                num_episodes=10,
                num_steps=20,
            )

        with pytest.raises(TypeError, match="environment must be an Environment instance"):
            EnvironmentRunParams(
                environment=42,  # type: ignore[arg-type]
                belief=belief,
                policies=[policy],
                num_episodes=10,
                num_steps=20,
            )

    def test_belief_must_be_belief_instance(self):
        """Test that belief must be a Belief instance.

        Purpose: Validates that TypeError is raised when belief is not a Belief instance

        Given: EnvironmentRunParams with non-Belief belief
        When: EnvironmentRunParams is instantiated
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(TypeError, match="belief must be a Belief instance"):
            EnvironmentRunParams(
                environment=env,
                belief="not_a_belief",  # type: ignore[arg-type]
                policies=[policy],
                num_episodes=10,
                num_steps=20,
            )

        with pytest.raises(TypeError, match="belief must be a Belief instance"):
            EnvironmentRunParams(
                environment=env,
                belief=42,  # type: ignore[arg-type]
                policies=[policy],
                num_episodes=10,
                num_steps=20,
            )

    def test_policies_must_not_be_empty(self):
        """Test that policies list must not be empty.

        Purpose: Validates that ValueError is raised when policies list is empty

        Given: EnvironmentRunParams with empty policies list
        When: EnvironmentRunParams is instantiated
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)

        with pytest.raises(ValueError, match="policies list cannot be empty"):
            EnvironmentRunParams(
                environment=env, belief=belief, policies=[], num_episodes=10, num_steps=20
            )

    def test_policies_must_be_policy_instances(self):
        """Test that all policies must be Policy instances.

        Purpose: Validates that TypeError is raised when any policy is not a Policy instance

        Given: EnvironmentRunParams with non-Policy in policies list
        When: EnvironmentRunParams is instantiated
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        with pytest.raises(TypeError, match="policies\\[0\\] must be a Policy instance"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=["not_a_policy"],  # type: ignore[list-item]
                num_episodes=10,
                num_steps=20,
            )

        with pytest.raises(TypeError, match="policies\\[1\\] must be a Policy instance"):
            EnvironmentRunParams(
                environment=env,
                belief=belief,
                policies=[policy, "not_a_policy"],  # type: ignore[list-item]
                num_episodes=10,
                num_steps=20,
            )

    def test_valid_params_do_not_raise_errors(self):
        """Test that valid parameters do not raise any errors.

        Purpose: Validates that valid parameters successfully create EnvironmentRunParams

        Given: Valid EnvironmentRunParams parameters
        When: EnvironmentRunParams is instantiated
        Then: No exceptions are raised

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95, **tiger_pinned_kwargs())
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        # Should not raise any exceptions
        params = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params.environment == env
        assert params.belief == belief
        assert params.policies == [policy]
        assert params.num_episodes == 10
        assert params.num_steps == 20
