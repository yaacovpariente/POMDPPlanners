"""Tests for LaserTag POMDP environment.

This module tests the LaserTag POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random
from collections import Counter

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.laser_tag_pomdp import (
    LaserTagObservation,
    LaserTagPOMDP,
    LaserTagState,
    LaserTagStateTransition,
)
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.utils.logger import get_logger

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class RandomPolicy(Policy):
    """Simple random policy for testing purposes."""

    def __init__(self, environment, name="RandomPolicy"):
        """Initialize random policy."""
        super().__init__(
            environment=environment,
            discount_factor=environment.discount_factor,
            name=name,
        )
        self.actions = environment.get_actions()

    def action(self, belief):
        """Return random action."""
        action = np.random.choice(self.actions)
        return [action], PolicyRunData(info_variables=[])

    @classmethod
    def get_space_info(cls):
        """Return space info from environment."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )


class TestLaserTagState:
    """Test LaserTagState representation and methods.

    Purpose: Validates LaserTag state representation and equality operations

    Given: LaserTagState instances with robot/opponent positions and terminal flags
    When: States are created, compared, and hashed
    Then: State operations work correctly with proper equality and hashing

    Test type: unit
    """

    def test_state_creation_and_equality(self):
        """Test LaserTagState creation and equality comparison.

        Purpose: Validates state creation and equality comparison functionality

        Given: Two identical LaserTag states and one different state
        When: States are created and compared using == operator
        Then: Identical states are equal and different states are not equal

        Test type: unit
        """
        state1 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state2 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state3 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=True)

        assert state1 == state2
        assert state1 != state3
        assert state1 != "not a state"

    def test_state_hashing(self):
        """Test LaserTagState hashing for use in sets and dictionaries.

        Purpose: Validates that identical states have same hash for container usage

        Given: Two identical LaserTag states
        When: Hash values are computed
        Then: States have same hash and can be used as dictionary keys

        Test type: unit
        """
        state1 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        state2 = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)

        assert hash(state1) == hash(state2)

        # Test usage in set
        state_set = {state1, state2}
        assert len(state_set) == 1


class TestLaserTagStateTransition:
    """Test LaserTagStateTransition model functionality.

    Purpose: Validates state transition dynamics including robot movement and opponent behavior

    Given: LaserTag state transition models with various states and actions
    When: Transitions are sampled and probabilities calculated
    Then: Robot and opponent movements follow expected dynamics

    Test type: unit
    """

    def test_robot_movement_basic(self):
        """Test basic robot movement without walls.

        Purpose: Validates robot movement follows action directions correctly

        Given: Robot at center position with clear movement directions
        When: Movement actions (North, South, East, West) are executed
        Then: Robot moves to expected adjacent positions

        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(1, 1), terminal=False)
        floor_shape = (7, 11)
        walls = set()

        # Test North movement (action 0)
        transition = LaserTagStateTransition(state, 0, floor_shape, walls)
        next_states = transition.sample(n_samples=10)
        for next_state in next_states:
            assert next_state.robot == (2, 5)  # One row up
            assert not next_state.terminal

    def test_robot_boundary_collision(self):
        """Test robot cannot move outside grid boundaries.

        Purpose: Validates robot respects grid boundaries and stays in place when blocked

        Given: Robot at grid boundary positions
        When: Actions try to move robot outside boundaries
        Then: Robot stays at current position instead of moving

        Test type: unit
        """
        floor_shape = (7, 11)
        walls = set()

        # Test robot at top boundary trying to go North
        state = LaserTagState(robot=(0, 5), opponent=(3, 3), terminal=False)
        transition = LaserTagStateTransition(state, 0, floor_shape, walls)
        next_states = transition.sample(n_samples=5)
        for next_state in next_states:
            assert next_state.robot == (0, 5)  # Should stay in place

    def test_robot_wall_collision(self):
        """Test robot cannot move into walls.

        Purpose: Validates robot respects wall positions and cannot move through them

        Given: Robot adjacent to wall positions
        When: Actions try to move robot into walls
        Then: Robot stays at current position instead of moving into wall

        Test type: unit
        """
        state = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)
        floor_shape = (7, 11)
        walls = {(3, 3)}  # Wall to the East

        # Test robot trying to move East into wall
        transition = LaserTagStateTransition(state, 2, floor_shape, walls)
        next_states = transition.sample(n_samples=5)
        for next_state in next_states:
            assert next_state.robot == (3, 2)  # Should stay in place

    def test_opponent_movement_toward_robot(self):
        """Test opponent tends to move toward robot position.

        Purpose: Validates opponent movement probabilities favor moves toward robot

        Given: Opponent in position where it can move toward robot
        When: Many state transitions are sampled
        Then: Opponent moves toward robot more frequently than random

        Test type: unit
        """
        state = LaserTagState(robot=(2, 5), opponent=(5, 5), terminal=False)
        floor_shape = (7, 11)
        walls = set()

        transition = LaserTagStateTransition(state, 1, floor_shape, walls)  # Robot moves South
        samples = transition.sample(n_samples=1000)

        # Count opponent positions
        opponent_positions = [s.opponent for s in samples]
        pos_counts = Counter(opponent_positions)

        # Opponent should prefer moving toward robot (should prefer North: (4,5))
        toward_robot_count = pos_counts.get((4, 5), 0)
        total_samples = len(samples)
        toward_robot_prob = toward_robot_count / total_samples

        # Should be around 0.4 based on implementation
        assert (
            toward_robot_prob > 0.3
        ), f"Expected >0.3 probability toward robot, got {toward_robot_prob}"

    def test_successful_tagging(self):
        """Test successful tagging creates terminal state.

        Purpose: Validates that tag action at same position creates terminal state

        Given: Robot and opponent at same position
        When: Tag action (action 4) is executed
        Then: Next state is terminal with same positions

        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=False)
        floor_shape = (7, 11)
        walls = set()

        transition = LaserTagStateTransition(state, 4, floor_shape, walls)
        next_states = transition.sample(n_samples=10)

        for next_state in next_states:
            assert next_state.terminal
            assert next_state.robot == (3, 5)
            assert next_state.opponent == (3, 5)

    def test_failed_tagging(self):
        """Test failed tagging does not create terminal state.

        Purpose: Validates that tag action at different positions remains non-terminal

        Given: Robot and opponent at different positions
        When: Tag action (action 4) is executed
        Then: Next state remains non-terminal with normal opponent movement

        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        floor_shape = (7, 11)
        walls = set()

        transition = LaserTagStateTransition(state, 4, floor_shape, walls)
        next_states = transition.sample(n_samples=10)

        for next_state in next_states:
            assert not next_state.terminal
            assert next_state.robot == (3, 5)  # Robot doesn't move on tag


class TestLaserTagObservation:
    """Test LaserTagObservation model functionality.

    Purpose: Validates observation generation with Gaussian noise around opponent position

    Given: LaserTag observation models with various states
    When: Observations are sampled and probabilities calculated
    Then: Observations follow expected noise characteristics

    Test type: unit
    """

    def test_observation_noise_characteristics(self):
        """Test observation model adds appropriate Gaussian noise.

        Purpose: Validates observation noise follows Gaussian distribution around true position

        Given: Observation model with known opponent position and noise level
        When: Many observations are sampled
        Then: Observations are distributed around true position with correct noise characteristics

        Test type: unit
        """
        state = LaserTagState(robot=(2, 2), opponent=(2, 4), terminal=False)
        obs_model = LaserTagObservation(
            state,
            0,
            measurement_noise=0.1,  # Low noise for testing
            floor_shape=(5, 5),
            walls=set(),
        )

        observations = obs_model.sample(n_samples=100)
        obs_array = np.array(observations)

        # Check all observations are 8-dimensional
        assert obs_array.shape[1] == 8, "Observations should be 8-dimensional laser measurements"

        # Check observations are reasonable laser ranges (non-negative)
        assert np.all(obs_array >= 0), "Laser measurements should be non-negative"

        # Check standard deviation is around measurement_noise for each direction
        std_obs = np.std(obs_array, axis=0)
        assert np.all(
            std_obs <= 1.0
        ), "Standard deviation should be reasonable for laser measurements"

    def test_terminal_state_observation(self):
        """Test terminal state produces special observation.

        Purpose: Validates terminal states generate designated terminal observation

        Given: Terminal LaserTag state
        When: Observations are sampled
        Then: All observations are the special terminal observation (-1, -1)

        Test type: unit
        """
        state = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=True)
        obs_model = LaserTagObservation(
            state, 4, measurement_noise=1.0, floor_shape=(7, 11), walls=set()
        )

        observations = obs_model.sample(n_samples=10)
        terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        for obs in observations:
            assert obs == terminal_obs, f"Terminal observation should be {terminal_obs}"


class TestLaserTagPOMDP:
    """Test main LaserTag POMDP environment functionality.

    Purpose: Validates complete environment behavior including rewards, terminals, and metrics

    Given: LaserTag POMDP environments with various configurations
    When: Environment methods are called with different states and actions
    Then: Environment behaves according to LaserTag problem specification

    Test type: integration
    """

    def test_environment_initialization(self):
        """Test LaserTag POMDP environment initialization.

        Purpose: Validates environment initializes with correct default parameters

        Given: LaserTag environment constructor with discount factor
        When: Environment is created
        Then: Environment has expected attributes and space configuration

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        assert env.discount_factor == 0.95
        assert env.floor_shape == (11, 7)
        # Default walls - 8 walls within valid grid bounds
        expected_walls = {
            (1, 2),
            (3, 0),
            (3, 4),
            (5, 0),
            (6, 4),
            (9, 1),
            (9, 4),
            (10, 6),
        }
        assert env.walls == expected_walls
        assert len(env.walls) == 8  # Should have 8 walls by default
        assert env.tag_reward == 10.0
        assert env.tag_penalty == 10.0
        assert env.step_cost == 1.0
        assert len(env.get_actions()) == 5

    def test_environment_initialization_with_walls(self):
        """Test LaserTag POMDP environment initialization with walls.

        Purpose: Validates environment initializes correctly with wall configuration

        Given: LaserTag environment constructor with custom wall set
        When: Environment is created with walls parameter
        Then: Environment stores walls correctly

        Test type: unit
        """
        walls = {(3, 3), (4, 4)}
        env = LaserTagPOMDP(discount_factor=0.95, walls=walls)

        assert env.walls == walls

    def test_reward_structure(self):
        """Test reward function returns correct values.

        Purpose: Validates reward function implements LaserTag reward structure correctly

        Given: Various state-action combinations
        When: Reward function is called
        Then: Returns correct rewards for tagging success/failure and movement

        Test type: unit
        """
        # Create environment with no dangerous areas to get deterministic rewards
        env = LaserTagPOMDP(discount_factor=0.95, dangerous_areas=set())

        # Test successful tag - using position (1, 1) which is not in default dangerous areas
        same_pos_state = LaserTagState(robot=(1, 1), opponent=(1, 1), terminal=False)
        tag_reward = env.reward(same_pos_state, 4)
        assert tag_reward == env.tag_reward

        # Test failed tag
        diff_pos_state = LaserTagState(robot=(1, 1), opponent=(2, 4), terminal=False)
        failed_tag_reward = env.reward(diff_pos_state, 4)
        assert failed_tag_reward == -env.tag_penalty

        # Test movement cost
        move_reward = env.reward(diff_pos_state, 0)  # North
        assert move_reward == -env.step_cost

        # Test terminal state
        terminal_state = LaserTagState(robot=(1, 1), opponent=(1, 1), terminal=True)
        terminal_reward = env.reward(terminal_state, 0)
        assert terminal_reward == 0.0

    def test_terminal_state_detection(self):
        """Test terminal state detection.

        Purpose: Validates environment correctly identifies terminal states

        Given: States with different terminal flags
        When: is_terminal method is called
        Then: Returns correct boolean based on state terminal flag

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        non_terminal = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        terminal = LaserTagState(robot=(3, 5), opponent=(3, 5), terminal=True)

        assert not env.is_terminal(non_terminal)
        assert env.is_terminal(terminal)

    def test_reward_range(self):
        """Test that reward range is correctly calculated.

        Purpose: Validates that LaserTagPOMDP reward range is properly calculated based on tag rewards and penalties

        Given: A LaserTagPOMDP environment with specific tag reward and penalty parameters
        When: Environment reward_range attribute is checked
        Then: Returns range based on worst case (-tag_penalty) and best case (tag_reward)

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95, tag_reward=15.0, tag_penalty=20.0, step_cost=2.0)

        # Expected calculation from LaserTagPOMDP constructor:
        # reward_range=(-tag_penalty, tag_reward) = (-20.0, 15.0)
        expected_min = -20.0  # Failed tag penalty
        expected_max = 15.0  # Successful tag reward

        assert env.reward_range == (expected_min, expected_max)

        # Test with different parameters
        env2 = LaserTagPOMDP(
            discount_factor=0.95, tag_reward=50.0, tag_penalty=100.0, step_cost=1.0
        )

        expected_min2 = -100.0  # Failed tag penalty
        expected_max2 = 50.0  # Successful tag reward

        assert env2.reward_range == (expected_min2, expected_max2)

    def test_initial_state_distribution(self):
        """Test initial state distribution properties.

        Purpose: Validates initial state distribution covers valid robot-opponent combinations

        Given: LaserTag environment
        When: Initial state distribution is sampled
        Then: States have robot and opponent at different valid positions

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        initial_dist = env.initial_state_dist()

        # Sample several initial states
        initial_states = initial_dist.sample(n_samples=10)

        for state in initial_states:
            assert isinstance(state, LaserTagState)
            assert state.robot != state.opponent  # Should start at different positions
            assert not state.terminal

    def test_observation_equality(self):
        """Test observation equality comparison.

        Purpose: Validates observation equality handles 8-dimensional laser measurements correctly

        Given: Pairs of similar and different 8D laser observations
        When: is_equal_observation method is called
        Then: Returns correct equality based on small tolerance for floating point

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        obs1 = (2.0, 4.0, 1.5, 3.2, 2.8, 1.1, 0.9, 2.5)
        obs2 = (2.0, 4.0, 1.5, 3.2, 2.8, 1.1, 0.9, 2.5)
        obs3 = (2.1, 4.1, 1.6, 3.3, 2.9, 1.2, 1.0, 2.6)

        assert env.is_equal_observation(obs1, obs2)
        assert not env.is_equal_observation(obs1, obs3)

    def test_compute_metrics_with_wall_collisions(self):
        """Test compute_metrics includes wall collision counting.

        Purpose: Validates metrics computation includes wall collision tracking

        Given: Mock history with wall collision scenario
        When: compute_metrics is called
        Then: Returns metrics including wall collision count

        Test type: unit
        """
        walls = {(3, 3)}
        env = LaserTagPOMDP(discount_factor=0.95, walls=walls)

        # Create mock history with wall collision
        # Create a simple belief for testing
        dummy_particles = [LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)]
        dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        # Step 1: Robot tries to move East into wall
        state1 = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)
        step1 = StepData(
            state=state1,
            action=2,  # East
            next_state=LaserTagState(
                robot=(3, 2), opponent=(1, 0), terminal=False
            ),  # Robot stayed due to wall
            observation=(
                1.1,
                0.9,
                1.2,
                0.8,
                1.0,
                1.1,
                0.9,
                1.3,
            ),  # 8D laser observation
            reward=-1.0,
            belief=test_belief,
        )

        # Step 2: Normal movement
        state2 = LaserTagState(robot=(3, 2), opponent=(1, 0), terminal=False)
        step2 = StepData(
            state=state2,
            action=0,  # North
            next_state=LaserTagState(robot=(2, 2), opponent=(1, 1), terminal=False),
            observation=(
                1.0,
                1.1,
                1.05,
                1.15,
                1.02,
                1.08,
                1.12,
                1.04,
            ),  # 8D laser observation
            reward=-1.0,
            belief=test_belief,
        )

        history = History(
            history=[step1, step2],
            discount_factor=0.95,
            average_state_sampling_time=0.1,
            average_action_time=0.1,
            average_observation_time=0.1,
            average_belief_update_time=0.1,
            average_reward_time=0.1,
            actual_num_steps=2,
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        histories = [history]
        metrics = env.compute_metrics(histories)

        # Find wall collision metric
        collision_metric = None
        for metric in metrics:
            if metric.name == "average_obstacle_collisions":
                collision_metric = metric
                break

        assert collision_metric is not None, "Wall collision metric not found"
        assert (
            collision_metric.value == 1.0
        ), f"Expected 1.0 collision, got {collision_metric.value}"

        # Find dangerous area steps metric
        dangerous_area_metric = None
        for metric in metrics:
            if metric.name == "average_dangerous_area_steps":
                dangerous_area_metric = metric
                break

        assert dangerous_area_metric is not None, "Dangerous area steps metric not found"
        # Should have 0 dangerous area steps since robot positions are not in default dangerous areas
        assert (
            dangerous_area_metric.value == 0.0
        ), f"Expected 0.0 dangerous area steps, got {dangerous_area_metric.value}"

    def test_wall_collision_penalty(self):
        """Test wall collision applies dangerous area penalty.

        Purpose: Validates wall collisions apply additional dangerous area penalty

        Given: LaserTag environment with walls and robot attempting wall collision
        When: Robot tries to move into a wall
        Then: Reward includes both step cost and dangerous area penalty

        Test type: unit
        """
        walls = {(3, 3)}
        env = LaserTagPOMDP(
            discount_factor=0.95, walls=walls, dangerous_area_penalty=5.0, step_cost=1.0
        )

        # Test wall collision penalty
        state = LaserTagState(robot=(3, 2), opponent=(1, 1), terminal=False)

        # Try to move East into wall at (3, 3)
        reward = env.reward(state, 2)  # East action

        # Expected: -1.0 (step cost) - 5.0 (wall collision penalty) = -6.0
        expected_reward = -env.step_cost - env.dangerous_area_penalty
        assert (
            reward == expected_reward
        ), f"Expected {expected_reward} for wall collision, got {reward}"

    def test_compute_metrics_with_simulator_generated_history(self):
        """Test compute_metrics with realistic history generated using environment simulation.

        Purpose: Validates compute_metrics function with realistic episode data generated by environment simulation

        Given: LaserTag environment and manually generated episode using environment step sampling
        When: Episode histories are generated and compute_metrics is called
        Then: All expected metrics are returned with valid values and proper types

        Test type: integration
        """
        # Create LaserTag environment with known configuration
        env = LaserTagPOMDP(
            discount_factor=0.95,
            tag_reward=10.0,
            tag_penalty=10.0,
            step_cost=1.0,
            measurement_noise=0.1,
            dangerous_areas={(3, 3), (4, 4)},  # Add some dangerous areas
        )

        # Generate realistic episode histories by running environment simulation
        histories = []
        np.random.seed(42)  # For reproducibility

        for episode_idx in range(3):  # Generate 3 episodes
            # Initialize episode
            initial_state = env.initial_state_dist().sample()[0]
            current_state = initial_state
            steps = []

            # Run episode for up to 15 steps
            for step_idx in range(15):
                # Choose random action (simulate random policy)
                action = np.random.choice(env.get_actions())

                # Sample next step using environment
                next_state, observation, reward = env.sample_next_step(current_state, action)

                # Create a belief for testing
                dummy_particles = [current_state]
                dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
                test_belief = WeightedParticleBelief(
                    particles=dummy_particles, log_weights=dummy_log_weights
                )

                # Create step data
                step = StepData(
                    state=current_state,
                    action=action,
                    next_state=next_state,
                    observation=observation,
                    reward=reward,
                    belief=test_belief,
                )
                steps.append(step)

                # Update current state
                current_state = next_state

                # End episode if terminal or successful tag
                if env.is_terminal(next_state) or (action == 4 and reward > 0):
                    break

            # Create history object
            history = History(
                history=steps,
                discount_factor=env.discount_factor,
                average_state_sampling_time=0.01,
                average_action_time=0.01,
                average_observation_time=0.01,
                average_belief_update_time=0.01,
                average_reward_time=0.01,
                actual_num_steps=len(steps),
                reach_terminal_state=env.is_terminal(current_state),
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
            histories.append(history)

        # Test compute_metrics with generated histories
        metrics = env.compute_metrics(histories)

        # Validate that we get the expected LaserTag-specific metrics
        metric_names = {metric.name for metric in metrics}
        expected_metrics = {
            "tag_success_rate",
            "average_episode_length",
            "average_failed_tag_attempts",
            "average_obstacle_collisions",
            "average_dangerous_area_steps",
        }

        assert expected_metrics.issubset(
            metric_names
        ), f"Missing metrics: {expected_metrics - metric_names}"

        # Validate metric value types and ranges
        for metric in metrics:
            assert isinstance(
                metric.value, (int, float)
            ), f"Metric {metric.name} value must be numeric"
            assert (
                metric.value >= 0
            ), f"Metric {metric.name} value must be non-negative: {metric.value}"
            assert hasattr(
                metric, "lower_confidence_bound"
            ), f"Metric {metric.name} missing lower confidence bound"
            assert hasattr(
                metric, "upper_confidence_bound"
            ), f"Metric {metric.name} missing upper confidence bound"

            # Validate specific metric ranges
            if metric.name == "tag_success_rate":
                assert (
                    0.0 <= metric.value <= 1.0
                ), f"Tag success rate must be between 0 and 1: {metric.value}"
            elif metric.name == "average_episode_length":
                assert (
                    metric.value <= 15
                ), f"Average episode length should not exceed max: {metric.value}"

        # Test that histories contain realistic LaserTag data
        for history in histories:
            assert len(history.history) > 0, "History should contain steps"

            for step in history.history:
                assert isinstance(step.state, LaserTagState), "State should be LaserTagState"
                assert (
                    step.action in env.get_actions()
                ), f"Action {step.action} not in valid actions"
                assert isinstance(step.reward, (int, float)), "Reward should be numeric"
                assert len(step.observation) == 8, "LaserTag observation should be 8-dimensional"

                # Validate robot and opponent positions are within grid bounds
                robot_pos = step.state.robot
                opponent_pos = step.state.opponent
                assert (
                    0 <= robot_pos[0] < env.floor_shape[0]
                ), f"Robot row {robot_pos[0]} out of bounds"
                assert (
                    0 <= robot_pos[1] < env.floor_shape[1]
                ), f"Robot col {robot_pos[1]} out of bounds"
                assert (
                    0 <= opponent_pos[0] < env.floor_shape[0]
                ), f"Opponent row {opponent_pos[0]} out of bounds"
                assert (
                    0 <= opponent_pos[1] < env.floor_shape[1]
                ), f"Opponent col {opponent_pos[1]} out of bounds"

    def test_compute_metrics_handles_none_rewards(self):
        """Test compute_metrics properly handles None reward values without crashing.

        Purpose: Validates that compute_metrics handles None reward values gracefully without TypeError

        Given: Episode history with some steps containing None rewards
        When: compute_metrics is called on the history
        Then: Metrics are computed without error and None rewards are ignored

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)

        # Create test belief
        dummy_particles = [LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)]
        dummy_log_weights = np.array([-0.1])  # Small non-zero log weight
        test_belief = WeightedParticleBelief(
            particles=dummy_particles, log_weights=dummy_log_weights
        )

        # Step 1: Normal reward
        step1 = StepData(
            state=LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False),
            action=0,  # North
            next_state=LaserTagState(robot=(2, 5), opponent=(1, 4), terminal=False),
            observation=(1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 0.8, 2.1),
            reward=-1.0,
            belief=test_belief,
        )

        # Step 2: None reward (this should not crash)
        step2 = StepData(
            state=LaserTagState(robot=(2, 5), opponent=(1, 4), terminal=False),
            action=4,  # Tag action
            next_state=LaserTagState(robot=(2, 5), opponent=(0, 4), terminal=False),
            observation=(1.1, 2.1, 3.1, 1.6, 2.6, 1.3, 0.9, 2.2),
            reward=None,  # This is the problematic case
            belief=test_belief,
        )

        # Step 3: Successful tag with positive reward
        step3 = StepData(
            state=LaserTagState(robot=(2, 5), opponent=(0, 4), terminal=False),
            action=4,  # Tag action
            next_state=LaserTagState(robot=(2, 5), opponent=(2, 5), terminal=True),
            observation=(
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ),  # Terminal obs
            reward=10.0,  # Successful tag
            belief=test_belief,
        )

        history = History(
            history=[step1, step2, step3],
            discount_factor=env.discount_factor,
            average_state_sampling_time=0.01,
            average_action_time=0.01,
            average_observation_time=0.01,
            average_belief_update_time=0.01,
            average_reward_time=0.01,
            actual_num_steps=3,
            reach_terminal_state=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )

        # This should not raise any exceptions
        try:
            metrics = env.compute_metrics([history])

            # Validate metrics were computed
            metric_names = {metric.name for metric in metrics}
            expected_metrics = {
                "tag_success_rate",
                "average_episode_length",
                "average_failed_tag_attempts",
                "average_obstacle_collisions",
                "average_dangerous_area_steps",
            }

            assert expected_metrics.issubset(
                metric_names
            ), f"Missing metrics: {expected_metrics - metric_names}"

            # Find success rate metric
            success_rate_metric = None
            for metric in metrics:
                if metric.name == "tag_success_rate":
                    success_rate_metric = metric
                    break

            assert success_rate_metric is not None, "tag_success_rate metric not found"
            # Should be 1.0 since last step has positive reward (successful tag)
            assert (
                success_rate_metric.value == 1.0
            ), f"Expected success rate 1.0, got {success_rate_metric.value}"

            # Find failed tag attempts metric
            failed_tags_metric = None
            for metric in metrics:
                if metric.name == "average_failed_tag_attempts":
                    failed_tags_metric = metric
                    break

            assert failed_tags_metric is not None, "average_failed_tag_attempts metric not found"
            # Should be 0.0 since None reward is ignored (not counted as failed tag)
            assert (
                failed_tags_metric.value == 0.0
            ), f"Expected 0.0 failed tags, got {failed_tags_metric.value}"

        except Exception as e:
            pytest.fail(f"compute_metrics should handle None rewards gracefully, but raised: {e}")


def test_metrics_confidence_intervals():
    """Test that metric values fall within their confidence intervals using realistic episodes.

    Purpose: Validates that LaserTagPOMDP compute_metrics returns metric values within their confidence bounds

    Given: A LaserTagPOMDP environment and realistic episode histories generated using run_episode
    When: compute_metrics is called with the histories
    Then: Each metric value falls within its lower_confidence_bound and upper_confidence_bound

    Test type: integration
    """
    # Create a LaserTagPOMDP environment with smaller grid for faster testing
    env = LaserTagPOMDP(
        discount_factor=0.95,
        floor_shape=(7, 7),  # Smaller grid for faster episodes
        walls={(2, 2), (3, 4), (5, 1)},  # Some walls for variety
        dangerous_areas={(1, 3), (4, 4)},  # Some dangerous areas
    )

    # Create random policy and initial belief
    policy = RandomPolicy(env, name="TestRandomPolicy")

    # Create initial belief using particles from initial state distribution
    initial_state = env.initial_state_dist().sample()[0]
    particles = [initial_state] * 50  # Simple particle belief
    log_weights = np.full(50, -np.log(50))  # Uniform weights
    initial_belief = WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=False
    )

    # Create logger for episode runs
    logger = get_logger(name="confidence_test", debug=False)

    # Generate realistic episode histories using run_episode
    histories = []
    np.random.seed(42)  # For reproducible test

    for episode_idx in range(20):  # Use enough episodes for meaningful statistics
        # Run episode with varying episode lengths
        max_steps = 5 + (episode_idx % 8)  # Vary episode length 5-12 steps

        history = run_episode(
            environment=env,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=max_steps,
            logger=logger,
        )

        histories.append(history)

    # Ensure we have at least some histories
    assert len(histories) > 0, "No histories were generated"

    # Compute metrics using the realistic histories
    metrics = env.compute_metrics(histories)

    # Verify that metrics were computed
    assert len(metrics) > 0, "No metrics were computed"

    # Use generic confidence interval verification
    verify_metrics_within_confidence_intervals(metrics)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
