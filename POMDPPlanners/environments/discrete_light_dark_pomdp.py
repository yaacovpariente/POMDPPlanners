from typing import List, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, ObservationModel
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.statistics import confidence_interval

class DiscreteLDObservationModel(ObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: Any,
        beacons: np.ndarray,
        obstacles: np.ndarray,
        beacon_radius: float,
        observation_error_prob: float,
    ):
        self.next_state = next_state
        self.action = action
        self.beacons = beacons
        self.obstacles = obstacles
        self.beacon_radius = beacon_radius
        self.observation_error_prob = observation_error_prob
        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }
        
        is_obstacle_hit = np.any(
            np.all(next_state.reshape(-1, 1) == self.obstacles, axis=0)
        )
        if is_obstacle_hit:
            distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
            min_distance = np.min(distances)
            if min_distance < self.beacon_radius:
                beacon_error_factor = 0.2
            else:
                beacon_error_factor = 1.0

            values = [next_state + self.action_to_vector[action] for action in self.actions]
            values.append(next_state)
            
            observation_error_prob = self.observation_error_prob * beacon_error_factor
            probs = np.ones(len(values)) * (observation_error_prob / (len(values) - 1))
            probs[-1] = 1 - observation_error_prob
        else:
            values = [next_state]
            probs = np.array([1.0])

        self.distribution = DiscreteDistribution(values=values, probs=probs)

    def sample(self):
        return self.distribution.sample()

    def probability(self, next_observation: Any) -> float:
        return self.distribution.probability(next_observation)


class DiscreteLightDarkPOMDP(DiscreteActionsEnvironment):
    def __init__(
        self,
        discount_factor: float,
        name: str = "DiscreteLightDarkPOMDP",
        transition_error_prob: float = 0.05,
        observation_error_prob: float = 0.05,
        beacons: np.ndarray = np.array(
            [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
        ),
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: np.ndarray = np.array([[3, 7], [5, 5]]),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        is_stochastic_reward: bool = True,
    ):
        self.__type_check(
            discount_factor=discount_factor,
            transition_error_prob=transition_error_prob,
            observation_error_prob=observation_error_prob,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
            is_stochastic_reward=is_stochastic_reward,
        )

        super().__init__(discount_factor=discount_factor, name=name)

        self.transition_error_prob = transition_error_prob
        self.observation_error_prob = observation_error_prob
        self.beacons = beacons
        self.goal_state = goal_state
        self.start_state = start_state
        self.obstacles = obstacles
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward
        self.fuel_cost = fuel_cost
        self.is_stochastic_reward = is_stochastic_reward
        self.grid_size = grid_size
        self.beacon_radius = beacon_radius

        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def __type_check(
        self,
        discount_factor: float,
        transition_error_prob: float,
        observation_error_prob: float,
        beacons: np.ndarray,
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: np.ndarray,
        obstacle_hit_probability: float,
        obstacle_reward: float,
        goal_reward: float,
        beacon_radius: float,
        fuel_cost: float,
        grid_size: int,
        is_stochastic_reward: bool,
    ):
        assert isinstance(
            observation_error_prob, float
        ), "observation_error_prob must be a float"
        assert isinstance(beacons, np.ndarray), "beacons must be a numpy array"
        assert isinstance(goal_state, np.ndarray), "goal_state must be a numpy array"
        assert isinstance(start_state, np.ndarray), "start_state must be a numpy array"
        assert isinstance(obstacles, np.ndarray), "obstacles must be a numpy array"
        assert isinstance(
            obstacle_hit_probability, float
        ), "obstacle_hit_probability must be a float"
        assert isinstance(obstacle_reward, float), "obstacle_reward must be a float"
        assert isinstance(goal_reward, float), "goal_reward must be a float"
        assert isinstance(beacon_radius, float), "beacon_radius must be a float"
        assert isinstance(fuel_cost, float), "fuel_cost must be a float"
        assert isinstance(grid_size, int), "grid_size must be an integer"
        assert isinstance(
            is_stochastic_reward, bool
        ), "is_stochastic_reward must be a boolean"

        # Value range checks
        assert 0 <= discount_factor <= 1, "discount_factor must be between 0 and 1"
        assert (
            0 <= transition_error_prob <= 1
        ), "transition_error_prob must be between 0 and 1"
        assert (
            0 <= observation_error_prob <= 1
        ), "observation_error_prob must be between 0 and 1"
        assert (
            0 <= obstacle_hit_probability <= 1
        ), "obstacle_hit_probability must be between 0 and 1"
        assert grid_size > 0, "grid_size must be positive"
        assert beacon_radius > 0, "beacon_radius must be positive"

        # Shape checks
        assert beacons.shape[0] == 2, "beacons must be a 2xN array"
        assert goal_state.shape == (2,), "goal_state must be a 2D vector"
        assert start_state.shape == (2,), "start_state must be a 2D vector"
        assert obstacles.shape[0] == 2, "obstacles must be a 2xN array"

        # Range checks for states
        assert np.all(beacons >= 0) and np.all(
            beacons <= grid_size
        ), "beacons coordinates must be within grid"
        assert np.all(goal_state >= 0) and np.all(
            goal_state <= grid_size
        ), "goal_state must be within grid"
        assert np.all(start_state >= 0) and np.all(
            start_state <= grid_size
        ), "start_state must be within grid"
        assert np.all(obstacles >= 0) and np.all(
            obstacles <= grid_size
        ), "obstacles must be within grid"

    def state_transition_model(self, state: np.ndarray, action: Any) -> Distribution:
        action_index = self.actions.index(action)
        values = [state + self.action_to_vector[action] for action in self.actions]

        # Distribute error probability equally among other actions
        probs = np.ones(len(values)) * (
            self.transition_error_prob / (len(self.actions) - 1)
        )
        probs[action_index] = 1 - self.transition_error_prob
        s = sum(probs)
        probs[0] += 1 - s

        return DiscreteDistribution(values, probs)

    def observation_model(self, next_state: np.ndarray, action: Any) -> Distribution:
        return DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=self.beacons,
            obstacles=self.obstacles,
            beacon_radius=self.beacon_radius,
            observation_error_prob=self.observation_error_prob,
        )

    def reward(self, state: np.ndarray, action: Any) -> float:
        assert state.shape == (2,)

        next_state = state + self.action_to_vector[action]

        is_goal_state = np.all(next_state == self.goal_state)
        is_obstacle_hit = np.any(
            np.all(next_state.reshape(-1, 1) == self.obstacles, axis=0)
        )
        is_out_of_grid = np.any(next_state < 0) or np.any(next_state > self.grid_size)

        # Start with base reward (fuel cost)
        reward = -self.fuel_cost - np.linalg.norm(next_state - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_obstacle_hit or is_out_of_grid:
            reward += self.obstacle_reward

        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        assert state.shape == (2,)

        is_goal_state = np.all(state == self.goal_state)
        is_obstacle_hit = np.any(np.all(state.reshape(-1, 1) == self.obstacles, axis=0))
        is_out_of_grid = np.any(state < 0) or np.any(state > self.grid_size)

        is_terminal = is_goal_state or is_obstacle_hit or is_out_of_grid

        return is_terminal

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(values=[self.start_state], probs=np.array([1.0]))

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(values=[1.0], probs=np.array([1.0]))

    def get_actions(self) -> List[Any]:
        return self.actions

    def visualize_path(self, path: List[np.ndarray], cache_path: Path):
        assert isinstance(cache_path, Path)
        assert str(cache_path).endswith(".gif")

        fig, ax = plt.subplots()
        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_xticks(np.arange(-1, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(-1, self.grid_size + 1, 1))
        ax.grid()

        # Plot the beacons
        ax.scatter(self.beacons[0], self.beacons[1], color="blue", label="Beacons")
        # Plot the goal state
        ax.scatter(
            self.goal_state[0], self.goal_state[1], color="green", label="Goal State"
        )
        # Plot the start state
        ax.scatter(
            self.start_state[0], self.start_state[1], color="red", label="Start State"
        )
        # Plot the obstacles
        ax.scatter(
            self.obstacles[0], self.obstacles[1], color="black", label="Obstacles"
        )

        # Initialize the agent's position and path line
        (agent,) = ax.plot([], [], "ro", markersize=10)
        (path_line,) = ax.plot([], [], "r-", alpha=0.5, linewidth=2)

        def init():
            agent.set_data([], [])
            path_line.set_data([], [])
            return agent, path_line

        def update(frame):
            # Update current position
            x = float(path[frame][0])
            y = float(path[frame][1])
            agent.set_data([x], [y])

            # Update path line up to current position
            path_x = [float(p[0]) for p in path[: frame + 1]]
            path_y = [float(p[1]) for p in path[: frame + 1]]
            path_line.set_data(path_x, path_y)

            return agent, path_line

        ani = animation.FuncAnimation(
            fig, update, frames=len(path), init_func=init, blit=True, repeat=False
        )
        plt.legend()
        plt.title("Agent Path Visualization")

        # Save the animation
        if cache_path is not None:
            ani.save(cache_path, writer="pillow", fps=2)  # Use pillow instead of ffmpeg

        plt.close()

    def cache_visualization(self, history: History, cache_path: Path) -> None:
        agent_path = [step.state for step in history.history]
        self.visualize_path(path=agent_path, cache_path=cache_path)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        # Calculate time to reach goal for each history
        goal_reached = []
        obstacle_hits = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False
            
            for i, step in enumerate(history.history):
                if np.array_equal(step.state, self.goal_state):
                    goal_reached_in_history = True
                    break
                
                if np.any(np.all(step.state.reshape(-1, 1) == self.obstacles, axis=0)):
                    obstacle_hit_in_history = True
            
            goal_reached.append(1 if goal_reached_in_history else 0)
            obstacle_hits.append(1 if obstacle_hit_in_history else 0)

        avg_goal_reached = np.mean(goal_reached)
        avg_obstacle_hits = np.mean(obstacle_hits)

        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)

        avg_obstacle_hits = np.mean(obstacle_hits)
        obstacle_hits_ci = confidence_interval(data=obstacle_hits, confidence=0.95)

        return [
            MetricValue(
                name="goal_reaching_rate",
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name="obstacle_hit_rate",
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            )
        ]
