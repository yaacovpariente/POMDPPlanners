"""Tests for simulation configuration classes."""

from POMDPPlanners.core.simulation.simulation_configs import EnvironmentRunParams
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env1 = TigerPOMDP(discount_factor=0.95)
        env2 = TigerPOMDP(discount_factor=0.90)

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
        env = TigerPOMDP(discount_factor=0.95)

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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=100)
        policy = DiscreteActionSequencesPlanner(
            environment=env, discount_factor=0.95, name="test_planner", depth=5, n_return_samples=10
        )

        params = EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=10, num_steps=20
        )

        assert params != "not_an_environment_run_params"
        assert params != 42
        assert params != None

    def test_usable_in_set(self):
        """Test that EnvironmentRunParams can be used in sets correctly.

        Purpose: Validates that hash and equality enable proper set behavior

        Given: Multiple EnvironmentRunParams instances including duplicates
        When: Instances are added to a set
        Then: Set contains only unique configurations based on config_id

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
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
        env = TigerPOMDP(discount_factor=0.95)
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
