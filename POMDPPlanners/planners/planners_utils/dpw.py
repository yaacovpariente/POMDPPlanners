from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import Tree


class ActionSampler(ABC):
    """Abstract base class for action sampling strategies in PFT-DPW.

    Action samplers provide domain-specific strategies for generating new actions
    during progressive widening. This allows PFT-DPW to work with continuous or
    large discrete action spaces by intelligently sampling promising actions.

    The ActionSampler interface enables flexible action space exploration by allowing
    custom sampling strategies that can incorporate domain knowledge, belief state
    information, or specialized sampling distributions.

    The class is serializable and can be safely pickled/unpickled for distributed
    computing, caching, or saving/loading configurations.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
        >>>
        >>> class ContinuousControlSampler(ActionSampler):
        ...     def __init__(self, action_bounds=(-1.0, 1.0), action_dim=2):
        ...         self.action_bounds = action_bounds
        ...         self.action_dim = action_dim
        ...
        ...     def sample(self, belief_node: Optional[BeliefNode] = None):
        ...         # Sample uniformly from action space
        ...         low, high = self.action_bounds
        ...         return np.random.uniform(low, high, size=self.action_dim)
        >>>
        >>> # Usage with PFT-DPW
        >>> sampler = ContinuousControlSampler(action_bounds=(-2.0, 2.0), action_dim=4)
        >>> action = sampler.sample()  # Returns 4D action vector  # doctest: +SKIP
        >>>
        >>> # Serialization works automatically
        >>> import pickle
        >>> serialized = pickle.dumps(sampler)  # doctest: +SKIP
        >>> restored_sampler = pickle.loads(serialized)  # doctest: +SKIP

        Discrete action sampler with custom distribution::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

            class WeightedDiscreteActionSampler(ActionSampler):
                def __init__(self, actions, probabilities=None):
                    self.actions = actions
                    # Use uniform probabilities if none provided
                    if probabilities is None:
                        self.probabilities = np.ones(len(actions)) / len(actions)
                    else:
                        self.probabilities = np.array(probabilities)
                        self.probabilities /= np.sum(self.probabilities)  # Normalize

                def sample(self, belief_node: Optional[BeliefNode] = None):
                    return np.random.choice(self.actions, p=self.probabilities)

            # Prefer certain actions over others
            actions = ["up", "down", "left", "right", "stay"]
            probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform
            sampler = WeightedDiscreteActionSampler(actions, probs)

        Belief-informed action sampler::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

            class AdaptiveActionSampler(ActionSampler):
                def __init__(self, base_actions, exploration_noise=0.1):
                    self.base_actions = base_actions
                    self.exploration_noise = exploration_noise

                def sample(self, belief_node: Optional[BeliefNode] = None):
                    if belief_node is not None and belief_node.visit_count > 10:
                        # Use belief state to inform sampling
                        best_action = self._get_best_action_from_belief(belief_node)
                        # Add exploration noise
                        noise = np.random.normal(0, self.exploration_noise, len(best_action))
                        return best_action + noise
                    else:
                        # Random exploration for new nodes
                        return np.random.choice(self.base_actions)

                def _get_best_action_from_belief(self, belief_node):
                    # Simplified: return action from best child
                    if belief_node.children:
                        best_child = max(belief_node.children, key=lambda x: x.q_value)
                        return best_child.action
                    return np.random.choice(self.base_actions)

            sampler = AdaptiveActionSampler([0, 1, 2, 3], exploration_noise=0.05)

        Multi-modal action sampler for hybrid control::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

            class MultiModalActionSampler(ActionSampler):
                def __init__(self, discrete_actions, continuous_bounds, mode_prob=0.5):
                    self.discrete_actions = discrete_actions
                    self.continuous_bounds = continuous_bounds
                    self.mode_prob = mode_prob  # Probability of discrete vs continuous

                def sample(self, belief_node=None):
                    if np.random.random() < self.mode_prob:
                        # Sample discrete action
                        return {"type": "discrete", "action": np.random.choice(self.discrete_actions)}
                    else:
                        # Sample continuous action
                        low, high = self.continuous_bounds
                        continuous_action = np.random.uniform(low, high, size=2)
                        return {"type": "continuous", "action": continuous_action}

            # For environments with both discrete and continuous actions
            discrete_acts = ["stop", "emergency_brake", "lane_change"]
            continuous_bounds = (-5.0, 5.0)  # Steering/acceleration range
            sampler = MultiModalActionSampler(discrete_acts, continuous_bounds)

        Goal-directed action sampler::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

            class GoalDirectedActionSampler(ActionSampler):
                def __init__(self, goal_position, action_magnitude=1.0, goal_bias=0.7):
                    self.goal_position = np.array(goal_position)
                    self.action_magnitude = action_magnitude
                    self.goal_bias = goal_bias

                def sample(self, belief_node=None):
                    if np.random.random() < self.goal_bias and belief_node is not None:
                        # Sample action towards goal based on current belief
                        current_position = self._estimate_position(belief_node)
                        direction = self.goal_position - current_position
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            return direction * self.action_magnitude

                    # Random exploration
                    angle = np.random.uniform(0, 2 * np.pi)
                    return self.action_magnitude * np.array([np.cos(angle), np.sin(angle)])

                def _estimate_position(self, belief_node):
                    # Simplified: use mean of particles in belief
                    if hasattr(belief_node.belief, 'particles'):
                        positions = [p[:2] for p in belief_node.belief.particles]  # First 2D as position
                        return np.mean(positions, axis=0)
                    return np.array([0.0, 0.0])

            # Navigation towards specific goal
            goal = [10.0, 5.0]
            sampler = GoalDirectedActionSampler(goal, action_magnitude=2.0, goal_bias=0.8)
    """

    @abstractmethod
    def sample(self, belief_node: Optional[BeliefNode] = None) -> Any:
        """Sample a new action for progressive widening.

        Args:
            belief_node: Optional belief node context for informed sampling

        Returns:
            A sampled action compatible with the environment's action space
        """

    def __getstate__(self):
        """Return state for serialization.

        This method is automatically called by pickle to get the object's state.
        Subclasses should override this if they have non-serializable attributes.

        Returns:
            Dictionary containing the object's state
        """
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from serialization.

        This method is automatically called by pickle to restore the object's state.
        Subclasses should override this if they need custom restoration logic.

        Args:
            state: Dictionary containing the object's state
        """
        vars(self).update(state)

    def __reduce__(self):
        """Custom reduction for pickle to handle abstract base classes.

        This ensures that concrete subclasses can be properly serialized
        even when referenced through the abstract base class.

        Returns:
            Tuple for pickle reconstruction
        """
        # Get the actual class of the instance (not the abstract base)
        cls = self.__class__
        # Get the current state
        state = self.__getstate__()
        # Return reconstruction tuple: (class, (), state)
        return (cls, (), state)

    def __eq__(self, other):
        """Equality comparison for serialization testing.

        Args:
            other: Another ActionSampler instance

        Returns:
            True if samplers are equivalent, False otherwise
        """
        if not isinstance(other, self.__class__):
            return False
        return self.__getstate__() == other.__getstate__()

    def __hash__(self):
        """Hash based on state for consistent behavior.

        Returns:
            Hash value based on the sampler's state
        """
        return hash(str(self.__getstate__()))


def action_progressive_widening(
    belief_node: BeliefNode,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    min_visit_count_per_action: int = 1,
) -> ActionNode:
    """Select or add action using progressive widening strategy.

    Progressive widening gradually expands the action space based on visit counts.
    New actions are added when ⌊n^α_a⌋ > ⌊(n-1)^α_a⌋, where n is the visit count.
    Otherwise, existing actions are selected using UCB1.

    The progressive widening mechanism balances exploration and exploitation by:
    1. Initially adding new actions frequently (exploration phase)
    2. Gradually reducing the rate of new actions as visit count increases
    3. Eventually relying primarily on UCB1 selection (exploitation phase)

    Args:
        belief_node: Current belief node to select action from
        alpha_a: Progressive widening exponent (0 < alpha_a ≤ 1). Lower values
                create fewer actions, higher values create more actions.
        action_sampler: Action sampler for generating new actions
        exploration_constant: UCB1 exploration constant for existing actions

    Returns:
        Selected or newly created action node

    Examples:
        Basic usage with continuous action sampler::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import (
                ActionSampler, action_progressive_widening
            )
            from POMDPPlanners.core.tree import BeliefNode
            from POMDPPlanners.core.belief import WeightedParticleBelief

            # Create action sampler
            class SimpleActionSampler(ActionSampler):
                def sample(self, belief_node=None):
                    return np.random.uniform(-1, 1, size=2)

            # Create belief node
            particles = [[0.0, 0.0], [1.0, 1.0]]
            log_weights = np.log(np.array([0.5, 0.5]))
            belief = WeightedParticleBelief(particles, log_weights)
            belief_node = BeliefNode(belief=belief)

            # Progressive widening
            action_sampler = SimpleActionSampler()
            action_node = action_progressive_widening(
                belief_node=belief_node,
                alpha_a=0.5,  # Moderate exploration
                action_sampler=action_sampler,
                exploration_constant=1.0
            )

        Comparing different alpha_a values::

            # Conservative exploration (fewer new actions)
            conservative_action = action_progressive_widening(
                belief_node=belief_node,
                alpha_a=0.25,  # Low alpha = fewer actions
                action_sampler=action_sampler,
                exploration_constant=1.0
            )

            # Aggressive exploration (more new actions)
            aggressive_action = action_progressive_widening(
                belief_node=belief_node,
                alpha_a=0.75,  # High alpha = more actions
                action_sampler=action_sampler,
                exploration_constant=1.0
            )

        Progressive widening in a loop (simulating MCTS)::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
            from POMDPPlanners.core.tree import BeliefNode, ActionNode
            from POMDPPlanners.core.belief import WeightedParticleBelief

            # Setup
            class DiscreteActionSampler(ActionSampler):
                def __init__(self, actions):
                    self.actions = actions

                def sample(self, belief_node=None):
                    return np.random.choice(self.actions)

            particles = [[0], [1], [2]]
            log_weights = np.log(np.array([1/3, 1/3, 1/3]))
            belief = WeightedParticleBelief(particles, log_weights)
            root_node = BeliefNode(belief=belief)

            sampler = DiscreteActionSampler(['up', 'down', 'left', 'right'])

            # Simulate multiple selections
            for i in range(10):
                root_node.visit_count = i  # Simulate increasing visits
                action_node = action_progressive_widening(
                    belief_node=root_node,
                    alpha_a=0.5,
                    action_sampler=sampler,
                    exploration_constant=1.41  # sqrt(2)
                )
                print(f"Visit {i}: {len(root_node.children)} actions, selected {action_node.action}")

        Tuning progressive widening parameters::

            # Effect of alpha_a on action creation
            visit_counts = range(1, 21)
            alpha_values = [0.25, 0.5, 0.75, 1.0]

            for alpha in alpha_values:
                action_counts = []
                for n in visit_counts:
                    # Calculate when new actions would be created
                    should_create = floor(n ** alpha) > floor((n-1) ** alpha) if n > 0 else True
                    action_counts.append(1 if should_create else 0)

                total_new_actions = sum(action_counts)
                print(f"Alpha {alpha}: {total_new_actions} new actions in 20 visits")
    """
    if belief_node.depth == 0:
        for action_node in belief_node.children:
            if action_node.visit_count < min_visit_count_per_action:
                return action_node

    if (
        belief_node.is_leaf
        or belief_node.visit_count == 0
        or len(belief_node.children) <= k_a * belief_node.visit_count**alpha_a
    ):
        action = action_sampler.sample()
        action_node = belief_node.get_child(action=action)
        if action_node is None:
            action_node = ActionNode(action=action, parent=belief_node)

        return action_node

    return ucb1_exploration(belief_node=belief_node, exploration_constant=exploration_constant)


def ucb1_exploration(belief_node: BeliefNode, exploration_constant: float) -> ActionNode:
    """Select action from existing children using UCB1 criterion.

    Uses Upper Confidence Bounds (UCB1) to balance exploration and exploitation:
    UCB1(a) = Q(a) + c * sqrt(log(N) / N(a))
    where Q(a) is the average reward, N is parent visits, N(a) is action visits,
    and c is the exploration constant.

    The UCB1 algorithm provides theoretical guarantees for multi-armed bandit problems
    and is widely used in Monte Carlo Tree Search algorithms. It automatically balances
    exploitation of good actions (high Q-values) with exploration of uncertain actions
    (low visit counts).

    Args:
        belief_node: Belief node with existing action children
        exploration_constant: Controls exploration vs exploitation trade-off.
                            Higher values favor exploration, lower values favor exploitation.
                            Common values: √2 ≈ 1.41 (theoretical optimum), 0.5-2.0 (practical range)

    Returns:
        Action node with highest UCB1 value

    Examples:
        Basic UCB1 action selection::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration
            from POMDPPlanners.core.tree import BeliefNode, ActionNode
            from POMDPPlanners.core.belief import WeightedParticleBelief

            # Create belief node with action children
            particles = [[0.0], [1.0]]
            log_weights = np.log(np.array([0.5, 0.5]))
            belief = WeightedParticleBelief(particles, log_weights)
            belief_node = BeliefNode(belief=belief)
            belief_node.visit_count = 100

            # Add action nodes with different Q-values and visit counts
            actions_data = [
                {"action": "up", "q_value": 0.8, "visits": 30},
                {"action": "down", "q_value": 0.6, "visits": 20},
                {"action": "left", "q_value": 0.9, "visits": 40},
                {"action": "right", "q_value": 0.4, "visits": 10}
            ]

            for data in actions_data:
                action_node = ActionNode(action=data["action"], parent=belief_node)
                action_node.q_value = data["q_value"]
                action_node.visit_count = data["visits"]

            # Select action using UCB1
            selected_action = ucb1_exploration(
                belief_node=belief_node,
                exploration_constant=1.41  # sqrt(2)
            )
            print(f"Selected action: {selected_action.action}")

        Comparing exploration constants::

            # Low exploration (favor exploitation)
            conservative_action = ucb1_exploration(
                belief_node=belief_node,
                exploration_constant=0.1
            )

            # High exploration (favor exploration)
            exploratory_action = ucb1_exploration(
                belief_node=belief_node,
                exploration_constant=3.0
            )

            # Balanced approach (theoretical optimum)
            balanced_action = ucb1_exploration(
                belief_node=belief_node,
                exploration_constant=1.41  # sqrt(2)
            )

        Manual UCB1 calculation and verification::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration

            # Calculate UCB1 values manually
            exploration_constant = 1.0
            ucb1_values = []

            for child in belief_node.children:
                exploration_term = exploration_constant * np.sqrt(
                    np.log(belief_node.visit_count) / child.visit_count
                )
                ucb1 = child.q_value + exploration_term
                ucb1_values.append(ucb1)
                print(f"Action {child.action}: Q={child.q_value:.2f}, "
                      f"exploration={exploration_term:.3f}, UCB1={ucb1:.3f}")

            # Verify our function selects the highest UCB1
            expected_best_idx = np.argmax(ucb1_values)
            selected_action = ucb1_exploration(belief_node, exploration_constant)
            actual_best_idx = belief_node.children.index(selected_action)

            assert expected_best_idx == actual_best_idx, "UCB1 selection mismatch"

        UCB1 in dynamic scenarios::

            import numpy as np
            from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration

            # Simulate how UCB1 selection changes over time
            belief_node.visit_count = 1

            for round_num in range(1, 11):
                belief_node.visit_count = round_num * 10

                # Select action
                selected = ucb1_exploration(belief_node, exploration_constant=1.41)

                # Update the selected action (simulate learning)
                selected.visit_count += 1
                selected.q_value += (np.random.normal(0.5, 0.1) - selected.q_value) / selected.visit_count

                print(f"Round {round_num}: Selected {selected.action}, "
                      f"Q={selected.q_value:.3f}, visits={selected.visit_count}")

        Exploration vs exploitation analysis::

            # Create scenario with clear best action vs uncertain actions
            exploration_constants = [0.1, 0.5, 1.0, 1.41, 2.0, 5.0]
            selection_counts = {c: {"up": 0, "down": 0, "left": 0, "right": 0} for c in exploration_constants}

            for exploration_c in exploration_constants:
                # Run multiple selections
                for _ in range(100):
                    selected = ucb1_exploration(belief_node, exploration_c)
                    selection_counts[exploration_c][selected.action] += 1

                print(f"Exploration constant {exploration_c}:")
                for action, count in selection_counts[exploration_c].items():
                    print(f"  {action}: {count}% selections")

        UCB1 with confidence intervals::

            import math
            from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration

            # Calculate confidence intervals for each action
            exploration_constant = 1.41
            confidence_level = 0.95

            for child in belief_node.children:
                # UCB1 upper confidence bound
                confidence_radius = exploration_constant * math.sqrt(
                    math.log(belief_node.visit_count) / child.visit_count
                )

                lower_bound = child.q_value - confidence_radius
                upper_bound = child.q_value + confidence_radius

                print(f"Action {child.action}: "
                      f"Q={child.q_value:.3f} ± {confidence_radius:.3f} "
                      f"[{lower_bound:.3f}, {upper_bound:.3f}]")

            # The selected action has the highest upper bound
            selected = ucb1_exploration(belief_node, exploration_constant)
            print(f"Selected: {selected.action} (highest upper confidence bound)")
    """
    q_vals = [child.q_value for child in belief_node.children]
    children_visit_counts = [
        max(child.visit_count, 1) for child in belief_node.children
    ]  # Avoid division by zero

    ucb = q_vals + exploration_constant * np.sqrt(
        np.log(belief_node.visit_count) / children_visit_counts
    )

    return belief_node.children[np.argmax(ucb)]


def action_progressive_widening_arena(
    tree: Tree,
    belief_id: int,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    min_visit_count_per_action: int = 1,
) -> int:
    """Arena variant of :func:`action_progressive_widening`.

    Returns the action-node ID (an int) selected by progressive widening +
    UCB1 from belief node ``belief_id`` in ``tree``. Mirrors the
    semantics of the legacy helper but operates on the column-store tree.
    """
    # Root-only: ensure every existing action has been visited at least
    # min_visit_count_per_action times before considering widening.
    if tree.parent_id[belief_id] is None:
        for cid in tree.children_ids[belief_id]:
            if tree.visit_count[cid] < min_visit_count_per_action:
                return cid

    children = tree.children_ids[belief_id]
    belief_visits = tree.visit_count[belief_id]
    is_leaf = len(children) == 0

    if is_leaf or belief_visits == 0 or len(children) <= k_a * belief_visits**alpha_a:
        action = action_sampler.sample()
        existing_id = tree.get_action_child_indexed(belief_id, action)
        if existing_id is not None:
            return existing_id
        existing_id = tree.get_action_child(belief_id, action)
        if existing_id is not None:
            return existing_id
        return tree.add_action_node(action=action, parent_id=belief_id)

    return ucb1_exploration_arena(
        tree=tree, belief_id=belief_id, exploration_constant=exploration_constant
    )


def ucb1_exploration_arena(tree: Tree, belief_id: int, exploration_constant: float) -> int:
    """Arena variant of :func:`ucb1_exploration`. Returns the action-node ID."""
    children = tree.children_ids[belief_id]
    belief_visits = tree.visit_count[belief_id]
    log_n = float(np.log(belief_visits))
    best_id = children[0]
    best_ucb = -float("inf")
    for cid in children:
        visits = max(tree.visit_count[cid], 1)
        ucb = tree.q_value[cid] + exploration_constant * (log_n / visits) ** 0.5
        if ucb > best_ucb:
            best_ucb = ucb
            best_id = cid
    return best_id
