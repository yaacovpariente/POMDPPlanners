#!/usr/bin/env python3
"""Test script to verify all usage examples from ALL environment classes work correctly."""

import sys
import traceback
import numpy as np
import os

# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sanity_pomdp_main_example():
    """Test the main SanityPOMDP class example."""
    print("Testing SanityPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
        
        # Create sanity test environment
        sanity = SanityPOMDP(discount_factor=0.95)
        
        # Get actions and verify simple dynamics
        actions = sanity.get_actions()  # [0, 1]
        assert actions == [0, 1], f"Expected [0, 1], got {actions}"
        
        # Test state transitions
        reward_good = sanity.reward(state=0, action=0)  # Should be 1.0
        reward_bad = sanity.reward(state=1, action=0)   # Should be 1.0 (goes to good state)
        assert reward_good == 1.0, f"Expected 1.0, got {reward_good}"
        assert reward_bad == 1.0, f"Expected 1.0, got {reward_bad}"
        
        # Verify perfect observability
        obs_model = sanity.observation_model(next_state=0, action=0)
        observation = obs_model.sample()[0]  # Should be 0
        assert observation == 0, f"Expected 0, got {observation}"
        
        print("  ✓ SanityPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ SanityPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_tiger_pomdp_main_example():
    """Test the main TigerPOMDP class example."""
    print("Testing TigerPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        # Create Tiger environment
        tiger = TigerPOMDP(discount_factor=0.95)
        
        # Get initial belief and sample episode
        # Note: get_initial_belief might not be available, so let's test basic functionality
        
        # Sample state and take action
        initial_state_dist = tiger.initial_state_dist()
        state = initial_state_dist.sample()[0]
        actions = tiger.get_actions()
        reward = tiger.reward(state, "listen")
        
        assert state in ["tiger_left", "tiger_right"], f"Unexpected state: {state}"
        assert "listen" in actions, f"listen not in actions: {actions}"
        assert "open_left" in actions, f"open_left not in actions: {actions}"
        assert "open_right" in actions, f"open_right not in actions: {actions}"
        assert reward == -1.0, f"Expected -1.0 for listen, got {reward}"
        
        # Check for terminal condition
        is_done = tiger.is_terminal(state)
        assert is_done == False, f"Expected False, got {is_done}"
        
        print("  ✓ TigerPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ TigerPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_cartpole_pomdp_main_example():
    """Test the main CartPolePOMDP class example."""
    print("Testing CartPolePOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        
        # Create CartPole environment with observation noise
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])  # Noise covariance matrix
        cartpole = CartPolePOMDP(
            discount_factor=0.99,
            noise_cov=noise_cov
        )
        
        # Get initial state and take action
        initial_state_dist = cartpole.initial_state_dist()
        state = initial_state_dist.sample()[0]
        
        # Apply force action (0=left, 1=right)
        action = 1  # Apply right force
        reward = cartpole.reward(state, action)
        
        # Check if episode should terminate
        is_done = cartpole.is_terminal(state)
        
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 4, f"Expected length 4, got {len(state)}"
        assert action in [0, 1], f"Action should be 0 or 1, got {action}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        assert isinstance(is_done, (bool, np.bool_)), f"Expected bool, got {type(is_done)}"
        
        print("  ✓ CartPolePOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ CartPolePOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_mountain_car_pomdp_main_example():
    """Test the main MountainCarPOMDP class example."""
    print("Testing MountainCarPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
        
        # Create Mountain Car environment
        mountain_car = MountainCarPOMDP(discount_factor=0.99)
        
        # Get initial state and available actions
        initial_state_dist = mountain_car.initial_state_dist()
        state = initial_state_dist.sample()[0]  # [position, velocity]
        actions = mountain_car.get_actions()  # [-1, 0, 1]
        
        # Take action and get reward
        action = 1  # Accelerate forward
        reward = mountain_car.reward(state, action)
        
        # Check if goal reached
        is_done = mountain_car.is_terminal(state)
        
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 2, f"Expected length 2, got {len(state)}"
        assert actions == [-1, 0, 1], f"Expected [-1, 0, 1], got {actions}"
        assert action in actions, f"Action {action} not in {actions}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        assert isinstance(is_done, (bool, np.bool_)), f"Expected bool, got {type(is_done)}"
        
        print("  ✓ MountainCarPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ MountainCarPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_push_pomdp_main_example():
    """Test the main PushPOMDP class example."""
    print("Testing PushPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.push_pomdp import PushPOMDP
        
        # Create push environment
        push_env = PushPOMDP(
            discount_factor=0.99,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1
        )
        
        # Get initial state
        initial_state_dist = push_env.initial_state_dist()
        state = initial_state_dist.sample()[0]
        
        # Move robot and potentially push object
        actions = push_env.get_actions()  # ["up", "down", "left", "right"]
        action = "right"
        reward = push_env.reward(state, action)
        
        # Check if object reached target
        is_done = push_env.is_terminal(state)
        
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 6, f"Expected length 6, got {len(state)}"
        assert actions == ["up", "down", "right", "left"], f"Expected directional actions, got {actions}"
        assert action in actions, f"Action {action} not in {actions}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        assert isinstance(is_done, (bool, np.bool_)), f"Expected bool, got {type(is_done)}"
        
        print("  ✓ PushPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ PushPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_safety_ant_velocity_pomdp_main_example():
    """Test the main SafeAntVelocityPOMDP class example."""
    print("Testing SafeAntVelocityPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
        
        # Create safety-critical environment
        safe_env = SafeAntVelocityPOMDP(
            discount_factor=0.99,
            safe_velocity_threshold=2.0,
            safety_violation_penalty=-100.0,
            movement_reward_scale=1.0
        )
        
        # Get initial state
        initial_state_dist = safe_env.initial_state_dist()
        state = initial_state_dist.sample()[0]  # [x, y, vx, vy]
        
        # Choose force magnitude action
        actions = safe_env.get_actions()  # [0, 1, 2, 3]
        action = 1  # Apply small force
        reward = safe_env.reward(state, action)
        
        # Check safety constraint
        velocity = state[2:4]
        speed = np.linalg.norm(velocity)
        is_safe = speed <= safe_env.safe_velocity_threshold
        
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 4, f"Expected length 4, got {len(state)}"
        assert actions == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {actions}"
        assert action in actions, f"Action {action} not in {actions}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        assert isinstance(is_safe, (bool, np.bool_)), f"Expected bool, got {type(is_safe)}"
        assert len(velocity) == 2, f"Expected velocity length 2, got {len(velocity)}"
        assert isinstance(speed, (float, np.float64)), f"Expected float, got {type(speed)}"
        
        print("  ✓ SafeAntVelocityPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ SafeAntVelocityPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_continuous_light_dark_pomdp_main_example():
    """Test the main ContinuousLightDarkPOMDP class example."""
    print("Testing ContinuousLightDarkPOMDP main class example...")
    
    try:
        from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
            ContinuousLightDarkPOMDP, RewardModelType
        )
        
        # Create environment with custom parameters
        env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            goal_state=np.array([10, 5]),
            start_state=np.array([0, 5]),
            reward_model_type=RewardModelType.STANDARD
        )
        
        # Sample initial state and take continuous action
        state_dist = env.initial_state_dist()
        state = state_dist.sample()[0]
        
        # Move toward goal with continuous action
        action = np.array([1.0, 0.0])  # Move right
        reward = env.reward(state, action)
        
        # Check termination
        is_done = env.is_terminal(state)
        
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 2, f"Expected length 2, got {len(state)}"
        assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
        assert len(action) == 2, f"Expected length 2, got {len(action)}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        assert isinstance(is_done, (bool, np.bool_)), f"Expected bool, got {type(is_done)}"
        
        print("  ✓ ContinuousLightDarkPOMDP main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ ContinuousLightDarkPOMDP main example failed: {e}")
        traceback.print_exc()
        return False

def test_continuous_light_dark_pomdp_discrete_actions_example():
    """Test the ContinuousLightDarkPOMDPDiscreteActions class example."""
    print("Testing ContinuousLightDarkPOMDPDiscreteActions main class example...")
    
    try:
        from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
            ContinuousLightDarkPOMDPDiscreteActions
        )
        
        # Create environment with discrete actions
        env = ContinuousLightDarkPOMDPDiscreteActions(
            discount_factor=0.95,
            goal_state=np.array([10, 5]),
            start_state=np.array([0, 5])
        )
        
        # Get available actions and take one
        actions = env.get_actions()  # ["up", "down", "right", "left"]
        action = "right"  # Move right
        
        # Simulate step
        state = env.start_state
        reward = env.reward(state, action)
        
        assert actions == ["up", "down", "right", "left"], f"Expected directional actions, got {actions}"
        assert action in actions, f"Action {action} not in {actions}"
        assert isinstance(state, np.ndarray), f"Expected ndarray, got {type(state)}"
        assert len(state) == 2, f"Expected length 2, got {len(state)}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        
        print("  ✓ ContinuousLightDarkPOMDPDiscreteActions main example passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ ContinuousLightDarkPOMDPDiscreteActions main example failed: {e}")
        traceback.print_exc()
        return False

def test_all_planners_usage_examples():
    """Test usage examples from all planner classes."""
    print("Testing all planners usage examples...")
    
    try:
        # Test POMCP usage example
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        
        env = TigerPOMDP(discount_factor=0.95)
        planner = POMCP(
            environment=env,
            discount_factor=env.discount_factor,
            depth=3,
            exploration_constant=1.0,
            name="POMCP_Tiger",
            n_simulations=10
        )
        
        initial_belief = get_initial_belief(env, n_particles=100)
        action, run_data = planner.action(initial_belief)
        
        assert action[0] in ["listen", "open_left", "open_right"]
        assert hasattr(run_data, 'info_variables')
        
        # Test Sparse Sampling usage example
        from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
        
        sparse_planner = StandardSparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=3,
            depth=2,
            name="SparseSampling_Tiger"
        )
        
        action, run_data = sparse_planner.action(initial_belief)
        assert action[0] in ["listen", "open_left", "open_right"]
        
        # Test PFT-DPW usage example
        from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
        from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
        import numpy as np
        
        class SimpleActionSampler(ActionSampler):
            def sample(self, belief_node=None):
                return np.random.choice([0, 1])
        
        cartpole = CartPolePOMDP(
            discount_factor=0.99,
            noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
        )
        
        pft_planner = PFT_DPW(
            environment=cartpole,
            discount_factor=0.99,
            depth=3,
            name="PFT_DPW_CartPole",
            action_sampler=SimpleActionSampler(),
            n_simulations=5
        )
        
        cartpole_belief = get_initial_belief(cartpole, n_particles=50)
        action, run_data = pft_planner.action(cartpole_belief)
        
        assert action[0] in [0, 1]
        
        print("  ✓ All planners usage examples work!")
        return True
        
    except Exception as e:
        print(f"  ✗ Planners usage examples failed: {e}")
        traceback.print_exc()
        return False

def test_all_supporting_class_examples():
    """Test all supporting class examples with comprehensive validation."""
    print("Testing all supporting class examples...")
    
    try:
        # Test Sanity POMDP supporting classes
        print("  Testing Sanity POMDP supporting classes...")
        from POMDPPlanners.environments.sanity_pomdp import (
            SanityStateTransitionModel, SanityObservationModel, 
            SanityInitialStateDist, SanityInitialObservationDist
        )
        
        # Test SanityStateTransitionModel
        transition_model = SanityStateTransitionModel(state=1, action=0)
        next_state = transition_model.sample()[0]
        prob = transition_model.probability([0])
        prob_wrong = transition_model.probability([1])
        assert next_state == 0, f"Expected 0, got {next_state}"
        assert prob[0] == 1.0, f"Expected [1.0], got {prob}"
        assert prob_wrong[0] == 0.0, f"Expected [0.0], got {prob_wrong}"
        
        # Test SanityObservationModel
        obs_model = SanityObservationModel(next_state=0, action=0)
        observation = obs_model.sample()[0]
        prob_correct = obs_model.probability([0])
        prob_wrong = obs_model.probability([1])
        assert observation == 0, f"Expected 0, got {observation}"
        assert prob_correct[0] == 1.0, f"Expected [1.0], got {prob_correct}"
        assert prob_wrong[0] == 0.0, f"Expected [0.0], got {prob_wrong}"
        
        # Test SanityInitialStateDist
        initial_dist = SanityInitialStateDist()
        initial_state = initial_dist.sample()[0]
        states = initial_dist.sample(n_samples=5)
        prob_good = initial_dist.probability([0])
        prob_bad = initial_dist.probability([1])
        assert initial_state == 0, f"Expected 0, got {initial_state}"
        assert all(s == 0 for s in states), f"Expected all 0s, got {states}"
        assert prob_good[0] == 1.0, f"Expected [1.0], got {prob_good}"
        assert prob_bad[0] == 0.0, f"Expected [0.0], got {prob_bad}"
        
        # Test SanityInitialObservationDist
        initial_obs_dist = SanityInitialObservationDist()
        initial_obs = initial_obs_dist.sample()[0]
        observations = initial_obs_dist.sample(n_samples=3)
        prob = initial_obs_dist.probability([0])
        assert initial_obs == 0, f"Expected 0, got {initial_obs}"
        assert all(o == 0 for o in observations), f"Expected all 0s, got {observations}"
        assert prob[0] == 1.0, f"Expected [1.0], got {prob}"
        
        # Test Tiger POMDP supporting classes
        print("  Testing Tiger POMDP supporting classes...")
        from POMDPPlanners.environments.tiger_pomdp import (
            TigerStateTransition, TigerObservation
        )
        
        # Test TigerStateTransition for listening
        transition_listen = TigerStateTransition(state="tiger_left", action="listen")
        next_state_listen = transition_listen.sample()[0]
        prob_same = transition_listen.probability(["tiger_left"])
        assert next_state_listen == "tiger_left", f"Expected tiger_left, got {next_state_listen}"
        assert prob_same[0] == 1.0, f"Expected [1.0], got {prob_same}"
        
        # Test TigerStateTransition for opening door
        transition_open = TigerStateTransition(state="tiger_left", action="open_left")
        next_state_open = transition_open.sample()[0]
        prob_random = transition_open.probability(["tiger_left"])
        assert next_state_open in ["tiger_left", "tiger_right"], f"Unexpected state: {next_state_open}"
        assert prob_random[0] == 0.5, f"Expected [0.5], got {prob_random}"
        
        # Test TigerObservation for listening
        obs_listen = TigerObservation(next_state="tiger_left", action="listen")
        observation = obs_listen.sample()[0]
        prob_correct = obs_listen.probability(["hear_left"])
        prob_wrong = obs_listen.probability(["hear_right"])
        assert observation in ["hear_left", "hear_right"], f"Unexpected observation: {observation}"
        assert prob_correct[0] == 0.85, f"Expected [0.85], got {prob_correct}"
        assert prob_wrong[0] == 0.15, f"Expected [0.15], got {prob_wrong}"
        
        # Test CartPole supporting classes
        print("  Testing CartPole POMDP supporting classes...")
        from POMDPPlanners.environments.cartpole_pomdp import (
            CartPoleStateTransition, CartPoleObservation, CartPoleInitialStateDistribution
        )
        
        # Test CartPoleStateTransition
        state = np.array([0.0, 0.0, 0.1, 0.0])
        cartpole_transition = CartPoleStateTransition(
            state=state, action=1, force_mag=10.0, total_mass=1.1,
            polemass_length=0.05, gravity=9.8, length=0.5,
            kinematics_integrator="euler", tau=0.02, masspole=0.1
        )
        cartpole_next_state = cartpole_transition.sample()[0]
        assert isinstance(cartpole_next_state, np.ndarray), f"Expected ndarray, got {type(cartpole_next_state)}"
        assert len(cartpole_next_state) == 4, f"Expected length 4, got {len(cartpole_next_state)}"
        
        # Test CartPoleObservation
        true_state = np.array([0.1, 0.05, 0.02, -0.1])
        noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
        obs_model = CartPoleObservation(next_state=true_state, action=1, noise_cov=noise_cov)
        observation = obs_model.sample()[0]
        prob = obs_model.probability([observation])
        assert isinstance(observation, np.ndarray), f"Expected ndarray, got {type(observation)}"
        assert len(observation) == 4, f"Expected length 4, got {len(observation)}"
        assert isinstance(prob, (np.ndarray, float, np.float64)), f"Expected array or float, got {type(prob)}"
        
        # Test CartPoleInitialStateDistribution
        initial_dist = CartPoleInitialStateDistribution()
        initial_state = initial_dist.sample()[0]
        states = initial_dist.sample(n_samples=3)
        assert isinstance(initial_state, np.ndarray), f"Expected ndarray, got {type(initial_state)}"
        assert len(initial_state) == 4, f"Expected length 4, got {len(initial_state)}"
        assert len(states) == 3, f"Expected 3 states, got {len(states)}"
        assert all(len(s) == 4 for s in states), "All states should have length 4"
        
        # Test additional supporting classes from other environments
        print("  Testing additional supporting classes...")
        from POMDPPlanners.environments.mountain_car_pomdp import (
            MountainCarTransition, MountainCarObservation
        )
        from POMDPPlanners.environments.push_pomdp import (
            PushStateTransition, PushObservation
        )
        from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
            SafeAntVelocityStateTransition, SafeAntVelocityObservation
        )
        from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
            StateTransitionModel
        )
        
        # Quick tests to ensure imports and basic functionality work
        # MountainCar
        mc_state = (-0.5, 0.0)
        mc_transition = MountainCarTransition(
            state=mc_state, action=1, power=0.001, gravity=0.0025, 
            max_speed=0.07, min_position=-1.2, max_position=0.6
        )
        mc_next_state = mc_transition.sample()[0]
        assert isinstance(mc_next_state, np.ndarray), "MountainCar transition should return ndarray"
        
        # Light-Dark
        ld_state = np.array([3.0, 4.0])
        ld_action = np.array([1.0, 0.5])
        ld_transition = StateTransitionModel(
            state=ld_state, action=ld_action, state_transition_cov_matrix=np.eye(2) * 0.1
        )
        ld_next_state = ld_transition.sample()[0]
        assert isinstance(ld_next_state, np.ndarray), "Light-Dark transition should return ndarray"
        
        print("  ✓ All supporting class examples work!")
        return True
        
    except Exception as e:
        print(f"  ✗ Supporting class examples failed: {e}")
        traceback.print_exc()
        return False

def test_all_simulations_class_examples():
    """Test usage examples from simulations classes."""
    print("Testing all simulations class examples...")
    
    try:
        # Test episode running
        from POMDPPlanners.simulations.episodes import run_episode
        from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.utils.logger import get_logger
        
        env = SanityPOMDP(discount_factor=0.95)
        planner = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=2,
            exploration_constant=1.0,
            name="POMCP_Sanity",
            n_simulations=5
        )
        
        initial_belief = get_initial_belief(env, n_particles=10)
        logger = get_logger("test")
        
        history = run_episode(
            environment=env,
            policy=planner,
            initial_belief=initial_belief,
            num_steps=3,
            logger=logger
        )
        
        assert hasattr(history, 'history')
        assert hasattr(history, 'discount_factor')
        
        # Test SimulationsAPI
        from POMDPPlanners.simulations.simulations_api import SimulationsAPI
        from POMDPPlanners.core.simulation import EnvironmentRunParams
        
        api = SimulationsAPI()
        assert hasattr(api, 'run_multiple_environments_and_policies_local_run')
        
        # Test simulation statistics
        from POMDPPlanners.simulations.simulation_statistics import (
            compute_statistics_environment_policy_pair,
            metrics_dict_to_dataframe
        )
        
        stats = compute_statistics_environment_policy_pair(
            env=env,
            histories=[history],
            alpha=0.1,
            confidence_interval_level=0.95
        )
        
        assert isinstance(stats, list)
        assert len(stats) > 0
        
        print("  ✓ All simulations class examples work!")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulations class examples failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all environment class examples."""
    print("Running ALL usage examples from environments directory classes...\n")
    
    tests = [
        test_sanity_pomdp_main_example,
        test_tiger_pomdp_main_example,
        test_cartpole_pomdp_main_example,
        test_mountain_car_pomdp_main_example,
        test_push_pomdp_main_example,
        test_safety_ant_velocity_pomdp_main_example,
        test_continuous_light_dark_pomdp_main_example,
        test_continuous_light_dark_pomdp_discrete_actions_example,
        test_all_supporting_class_examples,
        test_all_planners_usage_examples,
        test_all_simulations_class_examples,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} test groups passed")
    
    if passed == total:
        print("🎉 ALL environment class examples work correctly!")
        return 0
    else:
        print("❌ Some environment class examples have issues.")
        return 1

if __name__ == "__main__":
    exit(main())