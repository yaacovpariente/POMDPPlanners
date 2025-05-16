from typing import List, Any
from pathlib import Path

import numpy as np

from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    StateTransitionModel,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History, MetricValue

STATES = ["tiger_left", "tiger_right"]
ACTIONS = ["listen", "open_left", "open_right"]
OBSERVATIONS = ["hear_left", "hear_right", "hear_nothing"]


class TigerStateTransition(StateTransitionModel):
    def __init__(self, state: str, action: str):
        self.state = state
        self.action = action

    def sample(self) -> str:
        # State only changes when opening a door
        if self.action == "open_left" or self.action == "open_right":
            # After opening a door, tiger is randomly placed behind either door
            return np.random.choice(STATES)
        return self.state

    def probability(self, next_state: Any) -> float:
        if self.action == "open_left" or self.action == "open_right":
            return 0.5
        return 1.0


class TigerObservation(ObservationModel):
    def __init__(self, next_state: str, action: str):
        self.next_state = next_state
        self.action = action

    def sample(self) -> str:
        if self.action == "listen":
            # Listen action is 85% accurate
            if self.next_state == "tiger_left":
                return "hear_left" if np.random.random() < 0.85 else "hear_right"
            else:
                return "hear_right" if np.random.random() < 0.85 else "hear_left"
        else:
            # When opening a door, observation is random (hearing nothing)
            return "hear_nothing"

    def probability(self, next_observation: Any) -> float:
        if next_observation not in OBSERVATIONS:
            raise ValueError(f"Invalid observation: {next_observation}")
            
        if self.action == "listen":
            if self.next_state == "tiger_left":
                return 0.85 if next_observation == "hear_left" else 0.15
            else:
                return 0.85 if next_observation == "hear_right" else 0.15

        # For non-listen actions, only hear_nothing has probability 1.0, others have probability 0.0
        return 1.0 if next_observation == "hear_nothing" else 0.0


class TigerPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float, name: str = "TigerPOMDP"):
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")
        
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Actions are discrete: listen, open_left, open_right
            observation_space=SpaceType.DISCRETE  # Observations are discrete: hear_left, hear_right, hear_nothing
        )
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info)
        self.states = STATES
        self.actions = ACTIONS
        self.observations = OBSERVATIONS

    def state_transition_model(self, state: str, action: str) -> Distribution:
        return TigerStateTransition(state=state, action=action)

    def observation_model(self, next_state: str, action: str) -> Distribution:
        return TigerObservation(next_state=next_state, action=action)

    def reward(self, state: str, action: str) -> float:
        if action == "listen":
            return -1.0  # Cost of listening
        elif action == "open_left":
            if state == "tiger_left":
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure
        else:  # open_right
            if state == "tiger_right":
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure

    def is_terminal(self, state: str) -> bool:
        # Game ends when a door is opened
        return False  # Since we handle terminal states in the transition model

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(
            values=STATES, probs=np.ones(len(STATES)) / len(STATES)
        )

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(values=["hear_nothing"], probs=np.array([1.0]))

    def get_actions(self) -> List[Any]:
        return self.actions

    def cache_history_artifacts(self, history: History, cache_path: Path) -> None:
        """Create a visualization of the agent's path through the Tiger problem.

        Args:
            history: The history of states, actions, and observations
            cache_path: Path where to save the visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        assert isinstance(cache_path, Path)
        assert str(cache_path).endswith(".gif")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw doors
        left_door = plt.Rectangle((0, 0), 0.2, 1, facecolor="brown")
        right_door = plt.Rectangle((1, 0), 0.2, 1, facecolor="brown")
        ax.add_patch(left_door)
        ax.add_patch(right_door)

        # Initialize agent position
        (agent,) = ax.plot([], [], "ro", markersize=10)
        state_text = ax.text(0.5, 1.2, "", ha="center")
        action_text = ax.text(0.5, -0.2, "", ha="center")

        def init():
            agent.set_data([], [])
            state_text.set_text("")
            action_text.set_text("")
            return agent, state_text, action_text

        def update(frame):
            step = history.history[frame]
            # Update agent position based on state
            if step.state == "tiger_left":
                agent.set_data([0.1], [0.5])  # Left door
            else:
                agent.set_data([1.1], [0.5])  # Right door

            # Update text
            state_text.set_text(f"State: {step.state}")
            action_text.set_text(f"Action: {step.action}")

            return agent, state_text, action_text

        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(history.history),
            init_func=init,
            blit=True,
            repeat=False,
        )

        # Save animation
        ani.save(cache_path, writer="pillow", fps=2)
        plt.close()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute Tiger POMDP specific metrics from simulation histories.
        
        Args:
            histories: List of simulation histories
            
        Returns:
            List of MetricValue objects containing the computed metrics
        """
        # Calculate success rate (opening correct door)
        success_count = 0
        total_episodes = len(histories)
        
        for history in histories:
            # Check if the last action was opening a door
            last_step = history.history[-1]
            if last_step.action in ["open_left", "open_right"]:
                # Check if the door opened was correct
                if (last_step.action == "open_left" and last_step.state == "tiger_right") or \
                   (last_step.action == "open_right" and last_step.state == "tiger_left"):
                    success_count += 1
        
        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
        
        # Calculate average number of listens before opening a door
        listen_counts = []
        for history in histories:
            listen_count = sum(1 for step in history.history if step.action == "listen")
            listen_counts.append(listen_count)
        
        avg_listens = sum(listen_counts) / len(listen_counts) if listen_counts else 0.0
        
        # Create MetricValue objects
        return [
            MetricValue(
                name="success_rate",
                value=success_rate,
                lower_confidence_bound=success_rate,  # Simple implementation, could be improved with proper confidence intervals
                upper_confidence_bound=success_rate
            ),
            MetricValue(
                name="average_listens",
                value=avg_listens,
                lower_confidence_bound=avg_listens,
                upper_confidence_bound=avg_listens
            )
        ]
