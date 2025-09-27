import random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief, is_terminal_belief
from POMDPPlanners.core.cost import belief_expectation_cost_particle_belief, belief_expectation_cost
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.tree import (
    ActionNode,
    BeliefNode,
)
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    PathSimulationPolicy,
)


class SparsePFT(PathSimulationPolicy):
    """Sparse-PFT (Sparse Progressive Function Transfer) Algorithm for POMDP Planning.

    Sparse-PFT combines the efficiency of sparse sampling with progressive function transfer
    and Monte Carlo Tree Search for POMDP planning. It addresses the curse of dimensionality
    by limiting the number of children per belief-action node while using sophisticated
    exploration strategies to guide tree construction.

    Algorithm Overview:
    The algorithm operates by:
    1. **Sparse Branching**: Limits each action node to a fixed number of belief children
    2. **Progressive Selection**: Uses modified UCB to balance exploration and exploitation
    3. **Adaptive Sampling**: Samples existing children or generates new ones based on capacity
    4. **Random Rollouts**: Estimates values from leaf nodes using random simulations

    Key Features:
    - **Sparse Tree Structure**: Controls memory usage by limiting belief children per action
    - **Enhanced UCB**: Uses modified UCB formula with beta parameter for better exploration
    - **Efficient Sampling**: Balances between exploring existing branches and generating new ones
    - **Discrete Actions**: Optimized for discrete action spaces with discrete or mixed observations
    - **Terminal State Handling**: Properly detects when all particles reach terminal states

    Mathematical Foundation:
    The algorithm uses a modified UCB selection criterion:
        UCB(s,a) = Q(s,a) + c_ucb * beta_ucb * N(s) * (1/√N(s,a))

    Where:
    - Q(s,a): Action-value estimate
    - c_ucb: Base exploration constant
    - beta_ucb: Additional exploration parameter
    - N(s): Visit count of belief node
    - N(s,a): Visit count of action node

    Attributes:
        environment: The discrete-action POMDP environment for planning
        discount_factor: Discount factor for future rewards (0 < γ ≤ 1)
        gamma: Alternative discount parameter for value computation
        depth: Maximum search depth for tree expansion
        c_ucb: Base exploration constant for UCB formula
        beta_ucb: Additional exploration parameter for enhanced UCB
        belief_child_num: Maximum number of belief children per action node
        n_simulations: Number of MCTS simulations to perform

    Example:
        Using Sparse-PFT on Tiger POMDP with controlled tree growth::

            from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.core.belief import get_initial_belief

            # Create Tiger POMDP environment
            tiger = TigerPOMDP(discount_factor=0.95)

            # Create Sparse-PFT planner with controlled branching
            sparse_pft = SparsePFT(
                environment=tiger,
                discount_factor=0.95,
                gamma=0.95,                  # Discount for recursive calls
                depth=12,                    # Maximum planning depth
                c_ucb=1.0,                   # Base exploration constant
                beta_ucb=2.0,                # Enhanced exploration parameter
                belief_child_num=5,          # Max 5 belief children per action
                n_simulations=1000,          # Number of MCTS simulations
                name="SparsePFT_Tiger"
            )

            # Plan action from initial belief
            initial_belief = get_initial_belief(tiger, n_particles=1000)
            action, run_data = sparse_pft.action(initial_belief)

            print(f"Selected action: {action[0]}")
            print(f"Tree metrics collected: {len(run_data.info_variables)}")

            # Access tree statistics for analysis
            tree_metrics = run_data.info_variables
            for metric in tree_metrics:
                print(f"{metric.name}: {metric.value}")

    Example:
        Comparing different sparse branching strategies::

            # Conservative branching (small tree, faster planning)
            conservative_pft = SparsePFT(
                environment=tiger,
                discount_factor=0.95,
                gamma=0.95,
                depth=15,
                c_ucb=1.0,
                beta_ucb=1.0,                # Lower exploration
                belief_child_num=3,          # Fewer children (sparser tree)
                n_simulations=500,
                name="Conservative_SparsePFT"
            )

            # Aggressive branching (larger tree, more thorough search)
            aggressive_pft = SparsePFT(
                environment=tiger,
                discount_factor=0.95,
                gamma=0.95,
                depth=10,
                c_ucb=2.0,                   # Higher base exploration
                beta_ucb=3.0,                # Higher enhanced exploration
                belief_child_num=8,          # More children per action
                n_simulations=2000,          # More simulations
                name="Aggressive_SparsePFT"
            )

            # Compare performance
            conservative_action, conservative_data = conservative_pft.action(initial_belief)
            aggressive_action, aggressive_data = aggressive_pft.action(initial_belief)

            print("Conservative approach:")
            print(f"  Action: {conservative_action[0]}")
            print(f"  Planning time: {conservative_data.info_variables}")

            print("Aggressive approach:")
            print(f"  Action: {aggressive_action[0]}")
            print(f"  Planning time: {aggressive_data.info_variables}")

    Example:
        Using Sparse-PFT with different environments requiring parameter tuning::

            from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP

            # For environments with complex dynamics, tune parameters differently
            mountain_car = MountainCarPOMDP(discount_factor=0.99)

            mountain_pft = SparsePFT(
                environment=mountain_car,
                discount_factor=0.99,
                gamma=0.99,
                depth=20,                    # Deeper search for longer episodes
                c_ucb=1.5,                   # Moderate exploration
                beta_ucb=1.8,                # Balanced exploration enhancement
                belief_child_num=6,          # Moderate branching
                n_simulations=1500,          # More simulations for complex dynamics
                name="SparsePFT_MountainCar"
            )

            # The algorithm automatically handles the different observation space
            mountain_belief = get_initial_belief(mountain_car, n_particles=500)
            mountain_action, mountain_data = mountain_pft.action(mountain_belief)

            print(f"Mountain Car action: {mountain_action[0]}")  # One of [-1, 0, 1]

    Algorithm Details:

    **Tree Construction Process:**
    1. Start with root belief node
    2. For each simulation:
       - Traverse tree using enhanced UCB until leaf
       - Expand leaf with all available actions
       - For each action, maintain at most belief_child_num children
       - Sample existing children or generate new ones based on capacity
       - Perform random rollout from leaf for value estimation
       - Backpropagate values up the tree

    **Memory Management:**
    The sparse structure ensures memory usage stays bounded even for long planning
    horizons by limiting the branching factor at each action node.

    **Exploration Strategy:**
    The enhanced UCB formula with beta_ucb parameter provides better exploration
    control compared to standard UCB, especially important in sparse trees where
    exploration opportunities are limited.

    **Comparison with Other Algorithms:**
    - **vs POMCP**: More memory efficient due to sparse structure, may sacrifice some optimality
    - **vs PFT-DPW**: Works with discrete actions, uses different progressive strategy
    - **vs Sparse Sampling**: Adds MCTS tree structure for better sequential planning

    References:
    - Browne, C., et al. "A Survey of Monte Carlo Tree Search Methods." IEEE TCIAIG 2012.
    - Kearns, M., et al. "Near-Optimal Reinforcement Learning in Polynomial Time." ML 2002.
    """

    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        gamma: float,
        depth: int,
        c_ucb: float,
        beta_ucb: float,
        belief_child_num: int,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        name: str = "SparsePFT",
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        if not isinstance(environment, DiscreteActionsEnvironment):
            raise TypeError("environment must be a DiscreteActionsEnvironment instance")
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor must be a float")
        if not isinstance(gamma, float):
            raise TypeError("gamma must be a float")
        if not isinstance(depth, int):
            raise TypeError("depth must be an int")
        if not isinstance(c_ucb, float):
            raise TypeError("c_ucb must be a float")
        if not isinstance(beta_ucb, float):
            raise TypeError("beta_ucb must be a float")
        if not isinstance(belief_child_num, int):
            raise TypeError("belief_child_num must be an int")
        if not (1 >= discount_factor >= 0):
            raise ValueError("discount_factor must be between 0 and 1")

        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=time_out_in_seconds,
            action_sampler=None,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.gamma = gamma
        self.depth = depth
        self.c_ucb = c_ucb
        self.beta_ucb = beta_ucb
        self.belief_child_num = belief_child_num

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None
            return 0

        if is_terminal_belief(belief=belief_node.belief, env=self.environment):
            belief_node.visit_count += 1
            return 0

        if belief_node.is_leaf:
            for action in self.environment.get_actions():  # type: ignore
                action_node = ActionNode(action=action, parent=belief_node, children=tuple())

            state = belief_node.belief.sample()
            belief_node.visit_count += 1

            return self.random_rollout(state=state, depth=depth)

        action_node = self.get_explored_action_node(belief_node=belief_node)

        if len(action_node.children) == self.belief_child_num:
            next_belief_node, immediate_reward = self._sample_next_existing_belief(
                action_node=action_node
            )
        else:
            next_belief_node, immediate_reward = self._generate_belief(action_node=action_node)

        return_sample = immediate_reward + self.gamma * self._simulate_path(
            belief_node=next_belief_node, depth=depth + 1
        )

        self.update_nodes(
            belief_node=belief_node,
            action_node=action_node,
            return_sample=return_sample,
        )

        return return_sample

    def get_explored_action_node(self, belief_node: BeliefNode) -> ActionNode:
        children_visit_counts = np.array([child.visit_count for child in belief_node.children])
        unvisited_action_indices = np.where(children_visit_counts == 0)[0]
        if len(unvisited_action_indices) > 0:
            return belief_node.children[np.random.choice(unvisited_action_indices)]

        q_vals = np.array([child.q_value for child in belief_node.children])
        children_visit_counts = np.array([child.visit_count for child in belief_node.children])

        sprase_pft_exploration_addtion = (
            self.c_ucb
            * self.beta_ucb
            * belief_node.visit_count
            * 1
            / np.sqrt(children_visit_counts)
        )
        selected_action_index = np.argmax(q_vals + sprase_pft_exploration_addtion)

        return belief_node.children[selected_action_index]

    def _sample_next_existing_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        child_visit_counts = np.array([child.visit_count for child in action_node.children])
        if sum(child_visit_counts) == 0:
            # If no children have been visited, randomly select one and return with its immediate cost
            sampled_belief_node = np.random.choice(action_node.children)
            expected_reward = -sampled_belief_node.immediate_cost
            return sampled_belief_node, expected_reward

        weights = child_visit_counts / sum(child_visit_counts)
        sampled_belief_node = np.random.choice(action_node.children, p=weights)
        expected_reward = -sampled_belief_node.immediate_cost
        return sampled_belief_node, expected_reward

    def _generate_belief(self, action_node: ActionNode) -> Tuple[BeliefNode, float]:
        belief = action_node.parent.belief  # type: ignore
        state = belief.sample()
        next_state = self.environment.state_transition_model(
            state=state, action=action_node.action
        ).sample()[0]
        next_observation = self.environment.observation_model(
            next_state=next_state, action=action_node.action
        ).sample()[0]

        next_belief = belief.update(
            action=action_node.action,
            observation=next_observation,
            pomdp=self.environment,
        )

        next_belief_node = BeliefNode(
            belief=next_belief, observation=next_observation, parent=action_node
        )
        next_belief_node.immediate_cost = belief_expectation_cost(
            belief=belief, action=action_node.action, env=self.environment
        )
        immediate_reward = -next_belief_node.immediate_cost

        return next_belief_node, immediate_reward

    def random_rollout(self, state: Any, depth: int) -> float:
        if depth > self.depth or self.environment.is_terminal(state=state):
            return 0

        action = random.choice(self.environment.get_actions())  # type: ignore
        next_state, next_observation, reward = self.environment.sample_next_step(
            state=state, action=action
        )

        return reward + self.discount_factor * self.random_rollout(
            state=next_state, depth=depth + 1
        )

    def update_nodes(self, belief_node: BeliefNode, action_node: ActionNode, return_sample: float):
        belief_node.visit_count += 1
        action_node.visit_count += 1

        if action_node.immediate_cost is None:
            action_node.immediate_cost = belief_expectation_cost(
                belief=belief_node.belief,
                action=action_node.action,
                env=self.environment,
            )

        action_node.q_value += (return_sample - action_node.q_value) / action_node.visit_count
        belief_node.v_value = max(child.q_value for child in belief_node.children)

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED)
