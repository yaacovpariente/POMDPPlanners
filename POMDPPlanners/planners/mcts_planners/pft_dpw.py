"""PFT-DPW (Progressive Function Transfer with Double Progressive Widening) Algorithm.

This module implements PFT-DPW, a Monte Carlo Tree Search algorithm for continuous
action spaces in POMDPs. The algorithm uses progressive widening to gradually expand
the action and observation spaces during tree search, enabling effective planning
in problems with continuous or large discrete action spaces.

Key features:
- Progressive widening for both actions and observations
- Handles continuous action spaces through adaptive sampling
- Uses UCB1-style exploration with progressive expansion
- Supports custom action samplers for domain-specific action generation

The algorithm progressively expands the tree by:
1. Using action progressive widening to add new actions based on visit counts
2. Using observation progressive widening to add new observation branches
3. Balancing exploration of new actions with exploitation of promising ones
4. Performing random rollouts from leaf nodes for value estimation

Classes:
    ActionSampler: Abstract base class for action sampling strategies
    PFT_DPW: Main PFT-DPW planner with progressive widening for continuous actions
"""

from typing import Any, Tuple, Optional
from pathlib import Path

import numpy as np

from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import PathSimulationPolicy
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening


class PFT_DPW(PathSimulationPolicy):
    """PFT-DPW (Progressive Function Transfer with Double Progressive Widening) Algorithm.
    
    PFT-DPW is a Monte Carlo Tree Search algorithm designed for continuous action spaces
    in POMDPs. It uses progressive widening to gradually expand both the action and 
    observation spaces during tree search, enabling effective planning in problems with
    continuous or very large discrete action spaces.
    
    Algorithm Overview:
    The algorithm operates through progressive expansion:
    1. **Action Progressive Widening**: Gradually adds new actions based on visit counts
    2. **Observation Progressive Widening**: Gradually adds new observation branches  
    3. **UCB1 Exploration**: Balances exploration of new actions with exploitation
    4. **Random Rollouts**: Estimates values from leaf nodes using random simulations
    
    Key Features:
    - Handles continuous action spaces through adaptive sampling
    - Uses UCB1-style exploration with progressive expansion
    - Supports custom action samplers for domain-specific action generation
    - Balances exploration of new actions with exploitation of promising ones
    - Performs random rollouts from leaf nodes for value estimation
    
    Progressive Widening Parameters:
    - k_a, alpha_a: Control action space expansion (more actions added as visit_count^alpha_a)
    - k_o, alpha_o: Control observation space expansion
    - exploration_constant: UCB1 exploration parameter (higher = more exploration)
    
    Attributes:
        environment: The POMDP environment to plan for
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        depth: Maximum search depth for tree expansion
        action_sampler: Strategy for sampling new actions during progressive widening
        k_a, alpha_a: Action progressive widening parameters
        k_o, alpha_o: Observation progressive widening parameters  
        exploration_constant: UCB1 exploration parameter
        n_simulations: Number of simulations to run (mutually exclusive with timeout)
        time_out_in_seconds: Time limit for planning (mutually exclusive with n_simulations)
        
    Example:
        Using PFT-DPW for continuous control in CartPole::
        
            import numpy as np
            from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
            from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
            from POMDPPlanners.core.belief import get_initial_belief
            
            # Custom action sampler for CartPole
            class CartPoleActionSampler(ActionSampler):
                def sample(self, belief_node=None):
                    return np.random.choice([0, 1])  # Left or right force
            
            # Create CartPole environment
            cartpole = CartPolePOMDP(
                discount_factor=0.99,
                noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
            )
            
            # Create PFT-DPW planner
            pft_planner = PFT_DPW(
                environment=cartpole,
                discount_factor=0.99,
                depth=10,
                name="PFT_DPW_CartPole",
                action_sampler=CartPoleActionSampler(),
                k_a=2.0,                    # Action widening parameter
                alpha_a=0.5,                # Action widening exponent
                k_o=1.0,                    # Observation widening parameter  
                alpha_o=0.5,                # Observation widening exponent
                exploration_constant=1.41,   # UCB1 exploration (√2)
                n_simulations=1000          # Number of MCTS simulations
            )
            
            # Plan from initial belief
            initial_belief = get_initial_belief(cartpole, n_particles=500)
            action, run_data = pft_planner.action(initial_belief)
            
            print(f"Selected action: {action[0]}")
            print(f"Planning completed with {len(run_data.info_variables)} metrics")
            
    Example:
        Using PFT-DPW for 2D continuous navigation::
        
            import numpy as np
            from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
            
            class NavigationActionSampler(ActionSampler):
                def __init__(self, max_velocity=1.0):
                    self.max_velocity = max_velocity
                
                def sample(self, belief_node=None):
                    # Sample 2D velocity vector
                    angle = np.random.uniform(0, 2 * np.pi)
                    speed = np.random.uniform(0, self.max_velocity)
                    return np.array([speed * np.cos(angle), speed * np.sin(angle)])
            
            # Assume we have a 2D navigation environment
            # nav_env = NavigationPOMDP(...)
            
            navigation_planner = PFT_DPW(
                environment=nav_env,
                discount_factor=0.95,
                depth=15,
                name="PFT_DPW_Navigation",
                action_sampler=NavigationActionSampler(max_velocity=2.0),
                k_a=3.0,                    # More aggressive action widening
                alpha_a=0.6,                # Faster action space expansion
                exploration_constant=2.0,    # Higher exploration
                n_simulations=2000          # More simulations for complex space
            )
            
    Mathematical Background:
    Action progressive widening condition:
        Add new action when: ⌊N(s)^α_a⌋ > ⌊(N(s)-1)^α_a⌋
        where N(s) is the visit count of belief node s
        
    Observation progressive widening condition:
        Add new observation when: |C(s,a)| ≤ k_o * N(s,a)^α_o
        where |C(s,a)| is the number of observation children
        
    UCB1 action selection:
        UCB(s,a) = Q(s,a) + c * √(ln(N(s)) / N(s,a))
        where c is the exploration constant
        
    References:
    - Couëtoux, A., et al. "Continuous Upper Confidence Trees." LION 2011.
    - Browne, C., et al. "A Survey of Monte Carlo Tree Search Methods." IEEE TCIAIG 2012.
    """
    def __init__(
        self, 
        environment: Environment, 
        discount_factor: float, 
        depth: int, 
        name: str, 
        action_sampler: ActionSampler,
        k_a: float = 1.0,
        alpha_a: float = 0.5,
        k_o: float = 1.0,
        alpha_o: float = 0.5,
        exploration_constant: float = 1.0,
        time_out_in_seconds: int = None, 
        n_simulations: int = None, 
        min_samples_per_node: int = 10, 
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False
    ):
        super().__init__(
            environment=environment, 
            discount_factor=discount_factor, 
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            log_path=log_path,
            debug=debug
        )
        
        self.__type_check_inputs(
            min_samples_per_node=min_samples_per_node, 
            min_visit_count_per_action=min_visit_count_per_action
        )

        self.depth = depth
        self.action_sampler = action_sampler
        self.min_samples_per_node = min_samples_per_node
        self.min_visit_count_per_action = min_visit_count_per_action
        
        self.k_a = k_a
        self.alpha_a = alpha_a
        self.k_o = k_o
        self.alpha_o = alpha_o
        self.exploration_constant = exploration_constant
        
    def __type_check_inputs(
        self, 
        min_samples_per_node: int, 
        min_visit_count_per_action: int
    ):  
        if not isinstance(min_samples_per_node, int):
            raise TypeError("min_samples_per_node must be an int")
        if not isinstance(min_visit_count_per_action, int):
            raise TypeError("min_visit_count_per_action must be an int")
        # TODO: check if min_samples_per_node is greater than min_visit_count_per_action
        # TODO: check k and alpha params
        
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None
            return 0
        
        if self.environment.is_terminal(belief_node.belief.sample()):
            belief_node.visit_count += 1
            return 0
        
        action_node = action_progressive_widening(
            belief_node=belief_node,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a
        )
        
        return_sample = self._simulate_return(
            belief_node=belief_node, 
            action_node=action_node,
            depth=depth
        )
        
        self._update_node_statistics(
            belief_node=belief_node, 
            action_node=action_node, 
            total=return_sample
        )
        
        return return_sample
    
    def _simulate_return(self, belief_node: BeliefNode, action_node: ActionNode, depth: int) -> float:
        if len(action_node.children) <= self.k_o * action_node.visit_count ** self.alpha_o:
            next_belief_node, immediate_reward = self._sample_new_belief_node(belief_node=belief_node, action_node=action_node)
            state = next_belief_node.belief.sample()
            total = immediate_reward + self.discount_factor * self._random_rollout(state=state, depth=depth + 1)
        else:
            next_belief_node, immediate_reward = self.sample_existing_belief_node(belief_node=belief_node, action_node=action_node)
            total = immediate_reward + self.discount_factor * self._simulate_path(belief_node=next_belief_node, depth=depth + 1)
        
        return total

    def _sample_new_belief_node(self, belief_node: BeliefNode, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        immediate_reward = belief_expectation_reward(belief=belief_node.belief, action=action_node.action, env=self.environment)
        belief_node.immediate_cost = -immediate_reward
        
        next_state, next_observation, reward = self.environment.sample_next_step(state=belief_node.belief.sample(), action=action_node.action)
        next_belief = belief_node.belief.update(observation=next_observation, action=action_node.action, pomdp=self.environment)
        next_belief_node = BeliefNode(belief=next_belief, parent=action_node)
        
        return next_belief_node, immediate_reward
    
    def _random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0
        
        action = self.action_sampler.sample()
        next_state, next_observation, reward = self.environment.sample_next_step(state=state, action=action)
        
        return reward + self.discount_factor * self._random_rollout(state=next_state, depth=depth + 1)
    
    def sample_existing_belief_node(self, belief_node: BeliefNode, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        immediate_reward = -belief_node.immediate_cost
        random_idx = np.random.randint(0, len(action_node.children))
        next_belief_node = action_node.children[random_idx]
        return next_belief_node, immediate_reward

    def _update_node_statistics(self, belief_node: BeliefNode, action_node: ActionNode, total: float):
        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (total - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max([child.q_value for child in belief_node.children])

    def get_space_info(self) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.MIXED
        )
