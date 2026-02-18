"""BetaZero planner: neural MCTS for POMDPs.

This module implements the BetaZero algorithm (Moss et al., 2024 — arXiv:2306.00249),
which adapts AlphaZero to POMDPs by planning in belief space. It combines online MCTS
with PUCT and neural network priors for both action selection and leaf value estimation.
Offline policy-iteration training is orchestrated via
:class:`~POMDPPlanners.training.PolicyTrainer`.

Classes:
    BetaZero: Main planner extending ``DoubleProgressiveWideningMCTSPolicy``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import PolicyRunData, PolicySpaceInfo, TrainablePolicy
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    BeliefRepresentation,
    ParticleMeanStdRepresentation,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
    BetaZeroActionSampler,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    AbstractBetaZeroNetwork,
    BetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.puct import (
    puct_action_progressive_widening,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.training_buffer import (
    TrainingBuffer,
    TrainingExample,
)
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import (
    DoubleProgressiveWideningMCTSPolicy,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


@dataclass
class _PendingExample:
    belief_features: np.ndarray
    policy_target: np.ndarray


@dataclass
class _BatchedEpisodeState:
    belief: Belief
    state: Any
    rewards: List[float]
    features_list: List[np.ndarray]
    policies_list: List[np.ndarray]
    active: bool


class BetaZero(DoubleProgressiveWideningMCTSPolicy, TrainablePolicy):
    """BetaZero: Neural MCTS for POMDPs.

    Extends ``DoubleProgressiveWideningMCTSPolicy`` with three key innovations
    from the BetaZero paper:

    1. **PUCT selection**: Replaces UCB1 using learned policy priors.
    2. **Neural value estimation**: Replaces random rollouts at leaf nodes.
    3. **Policy iteration via ``fit()``**: Collects episodes, computes
       Q-weighted policy targets, and trains the network.

    The planner has two modes:
    - **Online planning** via ``action(belief)``: builds an MCTS tree with
      PUCT and network value estimates.
    - **Offline training** via ``fit()``: alternates data collection and
      network training.

    Attributes:
        network: Dual-head neural network for policy and value prediction.
        belief_representation: Belief → feature-vector mapping φ(b).
        z_q: Exponent for Q-value term in policy target.
        z_n: Exponent for visit-count term in policy target.
        temperature: Temperature τ for sharpening/smoothing policy target.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
        >>> from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
        >>>
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> sampler = DiscreteActionSampler(env.get_actions())
        >>> planner = BetaZero(
        ...     environment=env,
        ...     discount_factor=0.95,
        ...     depth=3,
        ...     name="BetaZero_Tiger",
        ...     action_sampler=sampler,
        ...     n_simulations=20,
        ...     state_dim=1,
        ... )
        >>> belief = get_initial_belief(env, n_particles=10)
        >>> actions, run_data = planner.action(belief)
        >>> actions[0] in env.get_actions()
        True
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
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_visit_count_per_action: int = 1,
        # BetaZero-specific
        network: Optional[AbstractBetaZeroNetwork] = None,
        belief_representation: Optional[BeliefRepresentation] = None,
        state_dim: Optional[int] = None,
        z_q: float = 1.0,
        z_n: float = 1.0,
        temperature: float = 1.0,
        n_buffer: int = 1,
        training_batch_size: int = 256,
        training_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        track_gradients: bool = False,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize BetaZero planner.

        Args:
            environment: The POMDP environment to plan for.
            discount_factor: Discount factor γ.
            depth: Maximum MCTS search depth.
            name: Policy identifier.
            action_sampler: Action sampling strategy for progressive widening.
            k_a: Action widening coefficient.
            alpha_a: Action widening exponent.
            k_o: Observation widening coefficient.
            alpha_o: Observation widening exponent.
            exploration_constant: PUCT exploration constant c.
            time_out_in_seconds: Time limit for planning.
            n_simulations: Number of MCTS simulations.
            min_visit_count_per_action: Min visits per action at root.
            network: Pre-trained network (created automatically if ``None``).
            belief_representation: Belief feature extractor
                (``ParticleMeanStdRepresentation`` created if ``None``).
            state_dim: State dimensionality (required if ``belief_representation``
                is ``None``).
            z_q: Q-value exponent in policy target.
            z_n: Visit-count exponent in policy target.
            temperature: Temperature τ for policy target.
            n_buffer: Number of policy-iteration slots to retain in the
                replay buffer.  With the default ``n_buffer=1`` only the
                current iteration's data is used for training (on-policy).
                Set ``n_buffer > 1`` for a rolling window of recent iterations.
            training_batch_size: Mini-batch size during training.
            training_epochs: Epochs per ``fit()`` iteration.
            learning_rate: Adam learning rate.
            weight_decay: L2 regularisation weight λ.
            hidden_sizes: Widths of hidden layers in the network trunk.
            track_gradients: When ``True``, gradient and weight norms are
                computed during training and included in the metrics dict.
            log_path: Optional log directory.
            debug: Enable debug logging.
            use_queue_logger: Use queue-based logging.

        Raises:
            ValueError: If neither ``belief_representation`` nor ``state_dim``
                is provided.
        """
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            name=name,
            action_sampler=action_sampler,
            k_a=k_a,
            alpha_a=alpha_a,
            k_o=k_o,
            alpha_o=alpha_o,
            exploration_constant=exploration_constant,
            min_visit_count_per_action=min_visit_count_per_action,
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.min_visit_count_per_action = min_visit_count_per_action
        self.z_q = z_q
        self.z_n = z_n
        self.temperature = temperature
        self.hidden_sizes = hidden_sizes
        self.track_gradients = track_gradients

        # Training hyper-parameters
        self.n_buffer = n_buffer
        self.training_batch_size = training_batch_size
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Belief representation
        self.belief_representation = self._init_belief_representation(
            belief_representation, state_dim
        )

        # Neural network
        self.network = network or self._create_default_network(hidden_sizes)

        # Wire network into action sampler if it's a BetaZeroActionSampler
        if isinstance(self.action_sampler, BetaZeroActionSampler):
            self.action_sampler.set_network_and_representation(
                self.network, self.belief_representation
            )

        # Training state
        self._buffer = TrainingBuffer(n_buffer=n_buffer)
        self._collecting_data = False
        self._pending_examples: List[_PendingExample] = []
        self._last_tree: Optional[BeliefNode] = None

    # ── Initialisation helpers ────────────────────────────────────────

    def _init_belief_representation(
        self,
        belief_representation: Optional[BeliefRepresentation],
        state_dim: Optional[int],
    ) -> BeliefRepresentation:
        if belief_representation is not None:
            return belief_representation
        if state_dim is None:
            raise ValueError("Either belief_representation or state_dim must be provided")
        return ParticleMeanStdRepresentation(state_dim=state_dim)

    def _create_default_network(self, hidden_sizes: Tuple[int, ...]) -> BetaZeroNetwork:
        belief_dim = self.belief_representation.feature_dim
        env_space = self.environment.space_info

        if env_space.action_space == SpaceType.DISCRETE:
            return BetaZeroNetwork(
                belief_dim=belief_dim,
                action_space_type="discrete",
                n_actions=len(self.environment.get_actions()),  # type: ignore[attr-defined]
                hidden_sizes=hidden_sizes,
            )
        return BetaZeroNetwork(
            belief_dim=belief_dim,
            action_space_type="continuous",
            action_dim=self._infer_action_dim(),
            hidden_sizes=hidden_sizes,
        )

    def _infer_action_dim(self) -> int:
        sample = self.action_sampler.sample()
        return int(np.asarray(sample).shape[0])

    # ── MCTS overrides ────────────────────────────────────────────────

    def _learn_tree(self, belief: Belief) -> BeliefNode:
        tree = super()._learn_tree(belief)
        self._last_tree = tree
        return tree

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        """Select an action via MCTS with PUCT and network value estimates.

        If data collection is active (during ``fit()``), also stores a
        pending training example from the tree root.
        """
        result = super().action(belief)

        if self._collecting_data and self._last_tree and not self._last_tree.is_leaf:
            features = self.belief_representation(belief)
            policy_target = self._compute_q_weighted_policy_target(self._last_tree)
            self._pending_examples.append(
                _PendingExample(belief_features=features, policy_target=policy_target)
            )

        return result

    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        if depth > self.depth:
            belief_node.parent = None
            return 0.0

        if self.environment.is_terminal(belief_node.belief.sample()):
            belief_node.visit_count += 1
            return 0.0

        action_priors = self._get_action_priors(belief_node)

        action_node = puct_action_progressive_widening(
            belief_node=belief_node,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a,
            action_priors=action_priors,
            min_visit_count_per_action=self.min_visit_count_per_action,
        )

        return_sample = self._simulate_return(
            belief_node=belief_node, action_node=action_node, depth=depth
        )

        self._update_node_statistics(
            belief_node=belief_node, action_node=action_node, total=return_sample
        )

        return return_sample

    def _simulate_return(
        self, belief_node: BeliefNode, action_node: ActionNode, depth: int
    ) -> float:
        if len(action_node.children) <= self.k_o * action_node.visit_count**self.alpha_o:
            next_belief_node, immediate_reward = self._sample_new_belief_node(
                belief_node=belief_node, action_node=action_node
            )
            leaf_value = self._network_leaf_value(next_belief_node)
            total = immediate_reward + self.discount_factor * leaf_value
        else:
            next_belief_node, immediate_reward = self._sample_existing_belief_node(
                belief_node=belief_node, action_node=action_node
            )
            total = immediate_reward + self.discount_factor * self._simulate_path(
                belief_node=next_belief_node, depth=depth + 1
            )
        return total

    def _sample_new_belief_node(
        self, belief_node: BeliefNode, action_node: ActionNode
    ) -> Tuple[BeliefNode, float]:
        immediate_reward = belief_expectation_reward(
            belief=belief_node.belief, action=action_node.action, env=self.environment
        )
        belief_node.immediate_cost = -immediate_reward

        _, next_observation, _ = self.environment.sample_next_step(
            state=belief_node.belief.sample(), action=action_node.action
        )
        next_belief = belief_node.belief.update(
            observation=next_observation,
            action=action_node.action,
            pomdp=self.environment,
        )
        next_belief_node = BeliefNode(belief=next_belief, parent=action_node)
        return next_belief_node, immediate_reward

    def _sample_existing_belief_node(
        self, belief_node: BeliefNode, action_node: ActionNode
    ) -> Tuple[BeliefNode, float]:
        immediate_reward = -belief_node.immediate_cost  # type: ignore[operator]
        next_belief_node = action_node.sample_child_node()
        return next_belief_node, immediate_reward

    def _update_node_statistics(
        self, belief_node: BeliefNode, action_node: ActionNode, total: float
    ) -> None:
        belief_node.visit_count += 1
        action_node.visit_count += 1
        action_node.q_value += (total - action_node.q_value) / action_node.visit_count
        belief_node.v_value = np.max([child.q_value for child in belief_node.children])

    # ── Network helpers ───────────────────────────────────────────────

    def _network_leaf_value(self, belief_node: BeliefNode) -> float:
        features = self.belief_representation(belief_node.belief)
        _, value = self.network.predict(features)
        return value

    def _get_action_priors(self, belief_node: BeliefNode) -> Optional[np.ndarray]:
        if not belief_node.children:
            return None
        if self.network.action_space_type == "discrete":
            return self._discrete_action_priors(belief_node)
        return None

    def _discrete_action_priors(self, belief_node: BeliefNode) -> np.ndarray:
        features = self.belief_representation(belief_node.belief)
        policy, _ = self.network.predict(features)
        actions = self.environment.get_actions()  # type: ignore[attr-defined]
        child_actions = [child.action for child in belief_node.children]
        priors = np.array(
            [
                policy[actions.index(a)] if a in actions else 1.0 / len(child_actions)
                for a in child_actions
            ]
        )
        prior_sum = priors.sum()
        if prior_sum > 0:
            priors = priors / prior_sum
        else:
            priors = np.ones(len(child_actions)) / len(child_actions)
        return priors

    # ── Q-weighted policy target ──────────────────────────────────────

    def _compute_q_weighted_policy_target(self, tree: BeliefNode) -> np.ndarray:
        """Compute π_qw(b,a) ∝ [softmax(Q)^z_q · (N(b,a)/ΣN)^z_n]^(1/τ).

        Returns a probability distribution over the tree root's child actions.
        For discrete action spaces, maps to full action-space vector.
        """
        children = tree.children
        q_values = np.array([child.q_value for child in children])
        visit_counts = np.array([child.visit_count for child in children], dtype=np.float64)

        q_term = self._softmax_q_term(q_values)
        n_term = self._visit_count_term(visit_counts)

        logits = (self.z_q * np.log(q_term + 1e-10) + self.z_n * np.log(n_term + 1e-10)) / max(
            self.temperature, 1e-10
        )
        logits -= logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        if self.network.action_space_type == "discrete":
            return self._map_to_full_action_vector(children, probs)
        return self._compute_continuous_policy_target(children, probs)

    def _compute_continuous_policy_target(self, children: tuple, probs: np.ndarray) -> np.ndarray:
        child_actions = np.array([child.action for child in children])
        weighted_mean = probs @ child_actions
        diff = child_actions - weighted_mean
        weighted_var = probs @ (diff**2)
        weighted_log_std = 0.5 * np.log(weighted_var + 1e-8)
        return np.concatenate([weighted_mean, weighted_log_std])

    def _softmax_q_term(self, q_values: np.ndarray) -> np.ndarray:
        shifted = q_values - q_values.max()
        exp_q = np.exp(shifted)
        return exp_q / exp_q.sum()

    def _visit_count_term(self, visit_counts: np.ndarray) -> np.ndarray:
        total = visit_counts.sum()
        if total == 0:
            return np.ones_like(visit_counts) / len(visit_counts)
        return visit_counts / total

    def _map_to_full_action_vector(self, children: tuple, probs: np.ndarray) -> np.ndarray:
        actions = self.environment.get_actions()  # type: ignore[attr-defined]
        full = np.zeros(len(actions), dtype=np.float64)
        for child, p in zip(children, probs):
            if child.action in actions:
                full[actions.index(child.action)] = p
        total = full.sum()
        if total > 0:
            full /= total
        return full

    # ── TrainablePolicy hooks ────────────────────────────────────────

    def begin_collecting(self) -> None:
        self._buffer.begin_iteration()
        self._collecting_data = True

    def end_collecting(self) -> None:
        self._collecting_data = False

    def prepare_episode(self) -> None:
        self._pending_examples.clear()

    def finalize_episode(self, history) -> None:
        self._finalize_episode_data(history)

    def train_step(self) -> Dict[str, List[float]]:
        return self._train_network_on_buffer()

    def buffer_size(self) -> int:
        return len(self._buffer)

    def collect_episodes_batched(
        self,
        initial_belief_fn: Callable[[], Belief],
        n_episodes: int,
        episode_length: int,
    ) -> None:
        self._collect_episodes_batched(initial_belief_fn, n_episodes, episode_length)

    def get_network(self) -> AbstractBetaZeroNetwork:
        return self.network

    def get_metric_keys(self) -> List[str]:
        keys = ["total_loss", "value_loss", "policy_loss"]
        if self.track_gradients:
            keys += [
                "grad_norm/global",
                "grad_norm/trunk",
                "grad_norm/policy_head",
                "grad_norm/value_head",
                "weight_norm/global",
            ]
        return keys

    # ── Episode data helpers ─────────────────────────────────────────

    def _finalize_episode_data(self, history) -> None:
        rewards = [step.reward for step in history.history if step.reward is not None]
        discounted_returns = self._compute_discounted_returns(rewards)

        for i, pending in enumerate(self._pending_examples):
            if i < len(discounted_returns):
                self._buffer.add(
                    TrainingExample(
                        belief_features=pending.belief_features,
                        policy_target=pending.policy_target,
                        value_target=discounted_returns[i],
                    )
                )

    # ── Batched network-only episode collection ──────────────────────

    def _collect_episodes_batched(
        self,
        initial_belief_fn: Callable[[], Belief],
        n_episodes: int,
        episode_length: int,
    ) -> None:
        self._buffer.begin_iteration()
        episodes = self._init_batched_episodes(initial_belief_fn, n_episodes)

        for _ in range(episode_length):
            active_indices = [i for i, ep in enumerate(episodes) if ep.active]
            if not active_indices:
                break

            features_batch = self._build_feature_batch(episodes, active_indices)
            policies, _ = self.network.predict_batch(features_batch)

            for batch_idx, ep_idx in enumerate(active_indices):
                ep = episodes[ep_idx]
                policy = policies[batch_idx]
                ep.features_list.append(features_batch[batch_idx].copy())
                ep.policies_list.append(policy.copy())

                action = self._sample_action_from_policy(policy)
                self._step_single_episode(ep, action)

        for ep in episodes:
            self._store_batched_episode_data(ep)

    def _init_batched_episodes(
        self, initial_belief_fn: Callable[[], Belief], n_episodes: int
    ) -> List[_BatchedEpisodeState]:
        episodes: List[_BatchedEpisodeState] = []
        for _ in range(n_episodes):
            belief = initial_belief_fn()
            state = belief.sample()
            episodes.append(
                _BatchedEpisodeState(
                    belief=belief,
                    state=state,
                    rewards=[],
                    features_list=[],
                    policies_list=[],
                    active=True,
                )
            )
        return episodes

    def _build_feature_batch(
        self,
        episodes: List[_BatchedEpisodeState],
        active_indices: List[int],
    ) -> np.ndarray:
        features = [self.belief_representation(episodes[i].belief) for i in active_indices]
        return np.stack(features, axis=0)

    def _sample_action_from_policy(self, policy: np.ndarray) -> Any:
        if self.network.action_space_type == "discrete":
            return self._sample_discrete_action(policy)
        return self._sample_continuous_action(policy)

    def _sample_discrete_action(self, probs: np.ndarray) -> Any:
        actions = self.environment.get_actions()  # type: ignore[attr-defined]
        probs = np.maximum(probs, 0.0)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(len(actions)) / len(actions)
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def _sample_continuous_action(self, policy_output: np.ndarray) -> np.ndarray:
        half = len(policy_output) // 2
        mean = policy_output[:half]
        log_std = policy_output[half:]
        std = np.exp(np.clip(log_std, -5.0, 2.0)) + 0.1
        return np.random.normal(mean, std).astype(np.float32)

    def _step_single_episode(self, episode: _BatchedEpisodeState, action: Any) -> None:
        next_state, observation, reward = self.environment.sample_next_step(
            state=episode.state, action=action
        )
        episode.state = next_state
        episode.rewards.append(reward)
        try:
            episode.belief = episode.belief.update(
                observation=observation, action=action, pomdp=self.environment
            )
        except ValueError:
            episode.active = False
            return
        if self.environment.is_terminal(next_state):
            episode.active = False

    def _store_batched_episode_data(self, episode: _BatchedEpisodeState) -> None:
        if not episode.rewards:
            return
        discounted_returns = self._compute_discounted_returns(episode.rewards)
        n_steps = min(len(episode.features_list), len(discounted_returns))
        for i in range(n_steps):
            self._buffer.add(
                TrainingExample(
                    belief_features=episode.features_list[i],
                    policy_target=episode.policies_list[i],
                    value_target=discounted_returns[i],
                )
            )

    def _compute_discounted_returns(self, rewards: List[float]) -> List[float]:
        n = len(rewards)
        returns = [0.0] * n
        if n == 0:
            return returns
        returns[-1] = rewards[-1]
        for t in range(n - 2, -1, -1):
            returns[t] = rewards[t] + self.discount_factor * returns[t + 1]
        return returns

    def _train_network_on_buffer(self) -> Dict[str, List[float]]:
        return self.network.fit(
            buffer=self._buffer,
            n_epochs=self.training_epochs,
            batch_size=self.training_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            track_gradients=self.track_gradients,
        )

    # ── Serialisation ─────────────────────────────────────────────────

    def save(self, filepath=None) -> Path:
        """Save policy config and network weights to a directory.

        Args:
            filepath: Directory path. If ``None``, uses default.

        Returns:
            Directory where the policy was saved.
        """
        if filepath is None:
            filepath = Path("saved_policies") / self.environment.name / "BetaZero" / self.name
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        config_path = filepath / "policy_config.json"
        config_data = {
            "policy_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "discount_factor": self.discount_factor,
            "depth": self.depth,
            "name": self.name,
            "k_a": self.k_a,
            "alpha_a": self.alpha_a,
            "k_o": self.k_o,
            "alpha_o": self.alpha_o,
            "exploration_constant": self.exploration_constant,
            "z_q": self.z_q,
            "z_n": self.z_n,
            "temperature": self.temperature,
            "hidden_sizes": list(self.hidden_sizes),
            "belief_dim": self.belief_representation.feature_dim,
            "action_space_type": self.network.action_space_type,
            "n_actions": self.network.n_actions,
            "action_dim": self.network.action_dim,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        self.network.save_weights(filepath / "network_weights.pt")
        return filepath

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)
