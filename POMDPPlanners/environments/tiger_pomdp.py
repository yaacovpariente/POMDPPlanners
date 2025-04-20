from typing import List, Any
import numpy as np
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, ObservationModel, StateTransitionModel
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History
from pathlib import Path

STATES = ['tiger_left', 'tiger_right']
ACTIONS = ['listen', 'open_left', 'open_right']
OBSERVATIONS = ['hear_left', 'hear_right', 'hear_nothing']


class TigerTransition(StateTransitionModel):
    def __init__(self, state: str, action: str):
        self.state = state
        self.action = action
        
    def sample(self) -> str:
        # State only changes when opening a door
        if self.action == 'open_left' or self.action == 'open_right':
            # After opening a door, tiger is randomly placed behind either door
            return np.random.choice(STATES)
        return self.state
    
    def probability(self, next_state: Any) -> float:
        if self.action == 'open_left' or self.action == 'open_right':
            return 0.5
        return 1.0
                
class TigerObservation(ObservationModel):
    def __init__(self, next_state: str, action: str):
        self.next_state = next_state
        self.action = action
        
    def sample(self) -> str:
        if self.action == 'listen':
            # Listen action is 85% accurate
            if self.next_state == 'tiger_left':
                return 'hear_left' if np.random.random() < 0.85 else 'hear_right'
            else:
                return 'hear_right' if np.random.random() < 0.85 else 'hear_left'
        else:
            # When opening a door, observation is random (hearing nothing)
            return 'hear_nothing'
        
    def probability(self, next_observation: Any) -> float:
        if self.action == 'listen':
            if self.next_state == 'tiger_left':
                return 0.85 if next_observation == 'hear_left' else 0.15
            else:
                return 0.85 if next_observation == 'hear_right' else 0.15
            
        if next_observation == 'hear_nothing':
            return 1.0
        else:
            raise ValueError(f"Invalid observation: {next_observation}")


class TigerPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float):
        super().__init__(discount_factor)
        self.states = STATES
        self.actions = ACTIONS
        self.observations = OBSERVATIONS
        
    def state_transition_model(self, state: str, action: str) -> Distribution:
        return TigerTransition(state=state, action=action)

    def observation_model(self, next_state: str, action: str) -> Distribution:                    
        return TigerObservation(next_state=next_state, action=action)

    def reward(self, state: str, action: str) -> float:
        if action == 'listen':
            return -1.0  # Cost of listening
        elif action == 'open_left':
            if state == 'tiger_left':
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure
        else:  # open_right
            if state == 'tiger_right':
                return -100.0  # Opening door with tiger
            else:
                return 10.0  # Opening door with treasure

    def is_terminal(self, state: str) -> bool:
        # Game ends when a door is opened
        return False  # Since we handle terminal states in the transition model

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(
            values=STATES,
            probs=np.ones(len(STATES)) / len(STATES)
        )

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(
            values=['hear_nothing'],
            probs=np.array([1.0])
        )

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
        left_door = plt.Rectangle((0, 0), 0.2, 1, facecolor='brown')
        right_door = plt.Rectangle((1, 0), 0.2, 1, facecolor='brown')
        ax.add_patch(left_door)
        ax.add_patch(right_door)
        
        # Initialize agent position
        agent, = ax.plot([], [], 'ro', markersize=10)
        state_text = ax.text(0.5, 1.2, '', ha='center')
        action_text = ax.text(0.5, -0.2, '', ha='center')
        
        def init():
            agent.set_data([], [])
            state_text.set_text('')
            action_text.set_text('')
            return agent, state_text, action_text
        
        def update(frame):
            step = history.history[frame]
            # Update agent position based on state
            if step.state == 'tiger_left':
                agent.set_data([0.1], [0.5])  # Left door
            else:
                agent.set_data([1.1], [0.5])  # Right door
                
            # Update text
            state_text.set_text(f'State: {step.state}')
            action_text.set_text(f'Action: {step.action}')
            
            return agent, state_text, action_text
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(history.history),
            init_func=init, blit=True, repeat=False
        )
        
        # Save animation
        ani.save(cache_path, writer='pillow', fps=2)
        plt.close()
