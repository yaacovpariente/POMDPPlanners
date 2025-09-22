import numpy as np
from typing import Tuple

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.utils.weighted_particle_beliefs import (
    WeightedParticleBeliefContinuousLightDarkFullCoverage,
    WeightedParticleBeliefDiscreteLightDarkFullCoverage,
)

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    RewardModelType,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP


class EnvironmentConfigsAPI:
    def __init__(self, discount_factor: float = 0.95, debug: bool = False):
        self.debug = debug
        self.discount_factor = discount_factor

    def tiger_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = TigerPOMDP(discount_factor=self.discount_factor, name="TigerPOMDP")
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )

        return pomdp, belief

    def cartpole_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        # Create noise covariance matrix for CartPole observations
        noise_cov = np.diag(
            [0.1, 0.1, 0.1, 0.1]
        )  # Noise for [cart_pos, cart_vel, pole_angle, pole_vel]
        pomdp = CartPolePOMDP(
            discount_factor=self.discount_factor,
            noise_cov=noise_cov,
            name="CartPolePOMDP",
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def mountain_car_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = MountainCarPOMDP(
            discount_factor=self.discount_factor, name="MountainCarPOMDP"
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def push_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = PushPOMDP(discount_factor=self.discount_factor, name="PushPOMDP")
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def continuous_observations_discrete_actions_light_dark_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        DISCOUNT_FACTOR = self.discount_factor
        STATE_TRANSITION_COV_MATRIX = (
            np.eye(2) * 0.075
        )  # Identity matrix for state transitions
        OBSERVATION_COV_MATRIX = np.array(
            [[0.075, 0.01], [0.01, 0.075]]
        )  # Anisotropic observation noise
        BEACONS = [(1, 1), (1, 4), (4, 4), (4, 1)]  # Grid pattern as list of tuples
        GOAL_STATE = np.array([4, 4])  # Goal at (4,4)
        START_STATE = np.array([1, 1])  # Start at (1,1)
        OBSTACLES = [(3, 1), (3, 2), (4, 1)]  # Two obstacles as list of tuples
        OBSTACLE_HIT_PROBABILITY = 0.2  # 20% chance of hitting obstacle
        OBSTACLE_REWARD = -10.0  # Penalty for hitting obstacle
        GOAL_REWARD = 10.0  # Reward for reaching goal
        FUEL_COST = 2.0  # Cost per action
        GRID_SIZE = 5  # Size of the grid
        GOAL_STATE_RADIUS = 1.5  # Radius around goal to consider as reached
        BEACON_RADIUS = 1.0  # Radius around beacons for observations
        OBSTACLE_RADIUS = 1.5  # Radius around obstacles for collision
        REWARD_MODEL_TYPE = RewardModelType.STANDARD  # Standard reward model
        PENALTY_DECAY = 1.0  # No decay in penalty

        pomdp = ContinuousLightDarkPOMDPDiscreteActions(
            discount_factor=DISCOUNT_FACTOR,
            state_transition_cov_matrix=STATE_TRANSITION_COV_MATRIX,
            observation_cov_matrix=OBSERVATION_COV_MATRIX,
            beacons=BEACONS,
            goal_state=GOAL_STATE,
            start_state=START_STATE,
            obstacles=OBSTACLES,
            obstacle_hit_probability=OBSTACLE_HIT_PROBABILITY,
            obstacle_reward=OBSTACLE_REWARD,
            goal_reward=GOAL_REWARD,
            fuel_cost=FUEL_COST,
            grid_size=GRID_SIZE,
            goal_state_radius=GOAL_STATE_RADIUS,
            beacon_radius=BEACON_RADIUS,
            obstacle_radius=OBSTACLE_RADIUS,
            reward_model_type=REWARD_MODEL_TYPE,
            penalty_decay=PENALTY_DECAY,
            name="ContinuousLightDarkPOMDPDiscreteActions",
        )

        # Get initial belief and extract particles
        initial_belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=initial_belief.particles,
            log_weights=np.log(np.ones(n_particles) / n_particles),
            ess_factor=0.5,
            reinvigoration_fraction=0.1,
        )
        return pomdp, belief

    def continuous_observations_continuous_actions_light_dark_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        DISCOUNT_FACTOR = self.discount_factor
        STATE_TRANSITION_COV_MATRIX = (
            np.eye(2) * 0.075
        )  # Identity matrix for state transitions
        OBSERVATION_COV_MATRIX = np.array(
            [[0.075, 0.01], [0.01, 0.075]]
        )  # Anisotropic observation noise
        BEACONS = [(1, 1), (1, 4), (4, 4), (4, 1)]  # Grid pattern as list of tuples
        GOAL_STATE = np.array([4, 4])  # Goal at (4,4)
        START_STATE = np.array([1, 1])  # Start at (1,1)
        OBSTACLES = [(3, 1), (3, 2), (4, 1)]  # Two obstacles as list of tuples
        OBSTACLE_HIT_PROBABILITY = 0.2  # 20% chance of hitting obstacle
        OBSTACLE_REWARD = -10.0  # Penalty for hitting obstacle
        GOAL_REWARD = 10.0  # Reward for reaching goal
        FUEL_COST = 2.0  # Cost per action
        GRID_SIZE = 5  # Size of the grid
        GOAL_STATE_RADIUS = 1.5  # Radius around goal to consider as reached
        BEACON_RADIUS = 1.0  # Radius around beacons for observations
        OBSTACLE_RADIUS = 1.5  # Radius around obstacles for collision
        REWARD_MODEL_TYPE = RewardModelType.STANDARD  # Standard reward model
        PENALTY_DECAY = 1.0  # No decay in penalty

        pomdp = ContinuousLightDarkPOMDP(
            discount_factor=DISCOUNT_FACTOR,
            name="ContinuousLightDarkPOMDP",
            state_transition_cov_matrix=STATE_TRANSITION_COV_MATRIX,
            observation_cov_matrix=OBSERVATION_COV_MATRIX,
            beacons=BEACONS,
            goal_state=GOAL_STATE,
            start_state=START_STATE,
            obstacles=OBSTACLES,
            obstacle_hit_probability=OBSTACLE_HIT_PROBABILITY,
            obstacle_reward=OBSTACLE_REWARD,
            goal_reward=GOAL_REWARD,
            fuel_cost=FUEL_COST,
            grid_size=GRID_SIZE,
            goal_state_radius=GOAL_STATE_RADIUS,
            beacon_radius=BEACON_RADIUS,
            obstacle_radius=OBSTACLE_RADIUS,
            reward_model_type=REWARD_MODEL_TYPE,
            penalty_decay=PENALTY_DECAY,
            is_obstacle_hit_terminal=True,
        )

        # Get initial belief and extract particles
        initial_belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=initial_belief.particles,
            log_weights=np.log(np.ones(n_particles) / n_particles),
            ess_factor=0.5,
            reinvigoration_fraction=0.1,
        )

        return pomdp, belief

    def rock_sample_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = RockSamplePOMDP(
            discount_factor=self.discount_factor,
            name="RockSamplePOMDP",
            dangerous_areas=None,
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def pacman_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = PacManPOMDP(discount_factor=self.discount_factor, name="PacManPOMDP")
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def laser_tag_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = LaserTagPOMDP(
            discount_factor=self.discount_factor, name="LaserTagPOMDP"
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def safety_ant_velocity_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = SafeAntVelocityPOMDP(
            discount_factor=self.discount_factor, name="SafeAntVelocityPOMDP"
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief


class RiskAverseEnvironmentConfigsAPI(EnvironmentConfigsAPI):
    def __init__(self, discount_factor: float = 0.95, debug: bool = False):
        super().__init__(discount_factor=discount_factor, debug=debug)

    def continuous_observations_discrete_actions_light_dark_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        DISCOUNT_FACTOR = self.discount_factor
        STATE_TRANSITION_COV_MATRIX = (
            np.eye(2) * 0.075
        )  # Identity matrix for state transitions
        OBSERVATION_COV_MATRIX = np.array(
            [[0.075, 0.01], [0.01, 0.075]]
        )  # Anisotropic observation noise
        BEACONS = [(1, 1), (4, 4), (4, 1), (1, 4)]  # Grid pattern as list of tuples
        GOAL_STATE = np.array([4, 4])  # Goal at (4,4)
        START_STATE = np.array([1, 1])  # Start at (1,1)
        OBSTACLES = [
            (3, 1),
            (3, 2),
            (4, 1),
        ]  # Two obstacles as list of tuples - moved away from start
        OBSTACLE_HIT_PROBABILITY = 0.2  # 20% chance of hitting obstacle
        OBSTACLE_REWARD = -10.0  # Penalty for hitting obstacle
        GOAL_REWARD = 10.0  # Reward for reaching goal
        FUEL_COST = 2.0  # Cost per action
        GRID_SIZE = 5  # Size of the grid
        GOAL_STATE_RADIUS = 1.5  # Radius around goal to consider as reached
        BEACON_RADIUS = 1.0  # Radius around beacons for observations
        OBSTACLE_RADIUS = 1.2  # Radius around obstacles for collision
        REWARD_MODEL_TYPE = RewardModelType.DANGEROUS_STATES  # Standard reward model
        PENALTY_DECAY = 1.0  # No decay in penalty

        pomdp = ContinuousLightDarkPOMDPDiscreteActions(
            discount_factor=DISCOUNT_FACTOR,
            state_transition_cov_matrix=STATE_TRANSITION_COV_MATRIX,
            observation_cov_matrix=OBSERVATION_COV_MATRIX,
            beacons=BEACONS,
            goal_state=GOAL_STATE,
            start_state=START_STATE,
            obstacles=OBSTACLES,
            obstacle_hit_probability=OBSTACLE_HIT_PROBABILITY,
            obstacle_reward=OBSTACLE_REWARD,
            goal_reward=GOAL_REWARD,
            fuel_cost=FUEL_COST,
            grid_size=GRID_SIZE,
            goal_state_radius=GOAL_STATE_RADIUS,
            beacon_radius=BEACON_RADIUS,
            obstacle_radius=OBSTACLE_RADIUS,
            reward_model_type=REWARD_MODEL_TYPE,
            penalty_decay=PENALTY_DECAY,
            name="ContinuousLightDarkPOMDPDiscreteActions",
        )

        # Get initial belief and extract particles
        initial_belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=initial_belief.particles,
            log_weights=np.log(np.ones(n_particles) / n_particles),
            ess_factor=0.5,
            reinvigoration_fraction=0.1,
        )
        return pomdp, belief

    def continuous_observations_continuous_actions_light_dark_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        DISCOUNT_FACTOR = self.discount_factor
        STATE_TRANSITION_COV_MATRIX = (
            np.eye(2) * 0.075
        )  # Identity matrix for state transitions
        OBSERVATION_COV_MATRIX = np.array(
            [[0.075, 0.01], [0.01, 0.075]]
        )  # Anisotropic observation noise
        BEACONS = [(1, 1), (4, 4), (4, 1), (1, 4)]  # Grid pattern as list of tuples
        GOAL_STATE = np.array([4, 4])  # Goal at (4,4)
        START_STATE = np.array([1, 1])  # Start at (1,1)
        OBSTACLES = [
            (3, 1),
            (3, 2),
            (4, 1),
        ]  # Two obstacles as list of tuples - moved away from start
        OBSTACLE_HIT_PROBABILITY = 0.2  # 20% chance of hitting obstacle
        OBSTACLE_REWARD = -10.0  # Penalty for hitting obstacle
        GOAL_REWARD = 10.0  # Reward for reaching goal
        FUEL_COST = 2.0  # Cost per action
        GRID_SIZE = 5  # Size of the grid
        GOAL_STATE_RADIUS = 1.5  # Radius around goal to consider as reached
        BEACON_RADIUS = 1.0  # Radius around beacons for observations
        OBSTACLE_RADIUS = 1.2  # Radius around obstacles for collision
        REWARD_MODEL_TYPE = RewardModelType.DANGEROUS_STATES  # Standard reward model
        PENALTY_DECAY = 1.0  # No decay in penalty

        pomdp = ContinuousLightDarkPOMDP(
            discount_factor=DISCOUNT_FACTOR,
            name="ContinuousLightDarkPOMDP",
            state_transition_cov_matrix=STATE_TRANSITION_COV_MATRIX,
            observation_cov_matrix=OBSERVATION_COV_MATRIX,
            beacons=BEACONS,
            goal_state=GOAL_STATE,
            start_state=START_STATE,
            obstacles=OBSTACLES,
            obstacle_hit_probability=OBSTACLE_HIT_PROBABILITY,
            obstacle_reward=OBSTACLE_REWARD,
            goal_reward=GOAL_REWARD,
            fuel_cost=FUEL_COST,
            grid_size=GRID_SIZE,
            goal_state_radius=GOAL_STATE_RADIUS,
            beacon_radius=BEACON_RADIUS,
            obstacle_radius=OBSTACLE_RADIUS,
            reward_model_type=REWARD_MODEL_TYPE,
            penalty_decay=PENALTY_DECAY,
            is_obstacle_hit_terminal=True,
        )

        # Get initial belief and extract particles
        initial_belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
            particles=initial_belief.particles,
            log_weights=np.log(np.ones(n_particles) / n_particles),
            ess_factor=0.5,
            reinvigoration_fraction=0.1,
        )

        return pomdp, belief

    def rock_sample_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = RockSamplePOMDP(
            discount_factor=self.discount_factor,
            map_size=(5, 5),
            rock_positions=[(1, 1), (3, 2), (2, 4)],
            dangerous_areas=[(2, 2), (4, 1)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=5.0,
        )

        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def push_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = PushPOMDP(
            discount_factor=self.discount_factor,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.3,
            observation_noise=0.1,
            obstacles=[(3.0, 4.0), (6.0, 7.0), (2.0, 8.0)],
            obstacle_radius=1,
            obstacle_penalty=-10.0,
            name="PushPOMDP",
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def pacman_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = PacManPOMDP(
            discount_factor=self.discount_factor,
            name="PacManPOMDP",
            max_observation_noise=0.5,
            ghost_collision_penalty=-50.0,
            pellet_reward=50.0,
            observation_noise_factor=0.1,
            win_reward=100.0,
            num_ghosts=2,
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def laser_tag_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = LaserTagPOMDP(
            discount_factor=self.discount_factor, name="LaserTagPOMDP"
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief

    def safety_ant_velocity_pomdp_config(
        self, n_particles: int = 20
    ) -> Tuple[Environment, WeightedParticleBelief]:
        pomdp = SafeAntVelocityPOMDP(
            discount_factor=self.discount_factor, name="SafeAntVelocityPOMDP"
        )
        belief = get_initial_belief(
            pomdp=pomdp, n_particles=n_particles, resampling=True
        )
        return pomdp, belief
