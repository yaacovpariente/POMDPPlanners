from typing import Any

from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.core.environment import Environment

def random_rollout_action_sampler(
    state: Any, 
    depth: int, 
    action_sampler: ActionSampler, 
    environment: Environment, 
    discount_factor: float, 
    max_depth: int = 10
) -> float:
    """Perform random rollout to estimate value from leaf node.
    
    Rollout policy samples random actions using the action_sampler until
    reaching maximum depth or terminal state. This provides value estimates
    for leaf nodes in the search tree during Monte Carlo Tree Search.
    
    The rollout uses a random policy (via action_sampler) to quickly estimate
    the value of a state without expensive planning. This is a key component
    of MCTS algorithms where accurate value estimation is traded off against
    computational efficiency.
    
    Args:
        state: Current state to rollout from
        depth: Current depth in rollout (starts at 0)
        action_sampler: Action sampler for selecting rollout actions
        environment: POMDP environment to simulate in
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        max_depth: Maximum rollout depth to prevent infinite loops
        
    Returns:
        Total discounted return from rollout simulation
        
    Examples:
        Basic rollout with discrete actions::
        
            import numpy as np
            from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            
            # Simple action sampler for Tiger POMDP
            class TigerActionSampler(ActionSampler):
                def sample(self, belief_node=None):
                    return np.random.choice(["listen", "open_left", "open_right"])
            
            # Create environment and sampler
            tiger = TigerPOMDP(discount_factor=0.95)
            action_sampler = TigerActionSampler()
            
            # Perform rollout from initial state
            initial_state = "tiger_left"
            rollout_value = random_rollout_action_sampler(
                state=initial_state,
                depth=0,
                action_sampler=action_sampler,
                environment=tiger,
                discount_factor=0.95,
                max_depth=10
            )
            print(f"Rollout value estimate: {rollout_value}")
            
        Rollout with continuous actions::
        
            import numpy as np
            from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
            from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
            
            class CartPoleActionSampler(ActionSampler):
                def sample(self, belief_node=None):
                    return np.random.choice([0, 1])  # Left or right force
            
            # Create CartPole environment
            noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
            cartpole = CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)
            action_sampler = CartPoleActionSampler()
            
            # Sample initial state and perform rollout
            initial_state_dist = cartpole.initial_state_dist()
            state = initial_state_dist.sample()[0]
            
            rollout_value = random_rollout_action_sampler(
                state=state,
                depth=0,
                action_sampler=action_sampler,
                environment=cartpole,
                discount_factor=0.99,
                max_depth=20  # Longer horizon for control task
            )
            
        Multiple rollouts for value estimation::
        
            import numpy as np
            from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
            from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
            
            class SanityActionSampler(ActionSampler):
                def sample(self, belief_node=None):
                    return np.random.choice([0, 1])
            
            sanity = SanityPOMDP(discount_factor=0.95)
            action_sampler = SanityActionSampler()
            
            # Perform multiple rollouts to reduce variance
            state = 0  # Good state
            n_rollouts = 100
            rollout_values = []
            
            for _ in range(n_rollouts):
                value = random_rollout_action_sampler(
                    state=state,
                    depth=0,
                    action_sampler=action_sampler,
                    environment=sanity,
                    discount_factor=0.95,
                    max_depth=15
                )
                rollout_values.append(value)
            
            # Estimate value with confidence intervals
            mean_value = np.mean(rollout_values)
            std_value = np.std(rollout_values)
            print(f"Value estimate: {mean_value:.3f} ± {1.96 * std_value / np.sqrt(n_rollouts):.3f}")
            
        Comparing different rollout depths::
        
            import numpy as np
            from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
            
            # Test effect of max_depth on value estimates
            depths = [5, 10, 20, 50]
            depth_results = {}
            
            for max_depth in depths:
                values = []
                for _ in range(50):  # Multiple rollouts per depth
                    value = random_rollout_action_sampler(
                        state=state,
                        depth=0,
                        action_sampler=action_sampler,
                        environment=sanity,
                        discount_factor=0.95,
                        max_depth=max_depth
                    )
                    values.append(value)
                
                depth_results[max_depth] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                
            for depth, stats in depth_results.items():
                print(f"Depth {depth}: {stats['mean']:.3f} ± {stats['std']:.3f}")
                
        Custom rollout policy with domain knowledge::
        
            import numpy as np
            from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
            
            class InformedActionSampler(ActionSampler):
                '''Action sampler that uses domain knowledge for better rollouts'''
                def __init__(self, environment):
                    self.environment = environment
                
                def sample(self, belief_node=None):
                    # For Tiger POMDP: listen more often early, then choose door
                    if np.random.random() < 0.7:
                        return "listen"  # Gather information first
                    else:
                        return np.random.choice(["open_left", "open_right"])
            
            # Compare random vs informed rollouts
            tiger = TigerPOMDP(discount_factor=0.95)
            random_sampler = TigerActionSampler()  # From earlier example
            informed_sampler = InformedActionSampler(tiger)
            
            initial_state = "tiger_left"
            n_trials = 100
            
            random_values = [
                random_rollout_action_sampler(
                    initial_state, 0, random_sampler, tiger, 0.95, 15
                ) for _ in range(n_trials)
            ]
            
            informed_values = [
                random_rollout_action_sampler(
                    initial_state, 0, informed_sampler, tiger, 0.95, 15  
                ) for _ in range(n_trials)
            ]
            
            print(f"Random policy: {np.mean(random_values):.3f}")
            print(f"Informed policy: {np.mean(informed_values):.3f}")
    """
    if depth >= max_depth or environment.is_terminal(state=state):
        return 0.0
    
    action = action_sampler.sample()
    next_state = environment.state_transition_model(state=state, action=action).sample()[0]
    reward = environment.reward(state=state, action=action)
    
    return reward + discount_factor * random_rollout_action_sampler(
        state=next_state, 
        depth=depth + 1, 
        action_sampler=action_sampler, 
        environment=environment, 
        discount_factor=discount_factor,
        max_depth=max_depth
    )
