# SPDX-License-Identifier: MIT

"""ConstrainedZero planner: neural MCTS for Chance-Constrained POMDPs.

This module implements the ConstrainedZero algorithm, which extends BetaZero to solve
CC-POMDPs. It adds a 3-head network with a failure probability head, safety-constrained
PUCT (SPUCT), adaptive failure threshold calibration via conformal inference, and
constrained policy targets for training.

References:
    Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., & Kochenderfer, M. J. (2024).
    ConstrainedZero: Chance-Constrained POMDP Planning Using Learned Probabilistic Failure
    Surrogates and Adaptive Safety Constraints. Proceedings of the Thirty-Third International
    Joint Conference on Artificial Intelligence (IJCAI), 6752-6760. arXiv:2405.00644.
    https://arxiv.org/abs/2405.00644

Implementation note:
    Operates on the column-store arena
    :class:`POMDPPlanners.core.tree.arena.Tree` via its BetaZero parent.
    ``_failure_dict`` and ``_delta_dict`` are keyed by integer node IDs
    (instead of ``id(node)`` as in the legacy anytree version).

Classes:
    ConstrainedZero: Main planner extending ``BetaZero``.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.beta_zero.belief_representation import (
    BeliefRepresentation,
)
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_network import (
    AbstractBetaZeroNetwork,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_puct import (
    spuct_action_progressive_widening_arena,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training import (
    train_constrained_network,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_training_buffer import (
    ConstrainedTrainingBuffer,
    ConstrainedTrainingExample,
)
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero_network import (
    ConstrainedZeroNetwork,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


class ConstrainedZero(BetaZero):
    """ConstrainedZero: Neural MCTS for Chance-Constrained POMDPs.

    Extends ``BetaZero`` with:

    1. **3-head network**: Adds a failure probability head alongside policy and value.
    2. **SPUCT selection**: Safety-constrained PUCT that masks unsafe actions.
    3. **Adaptive Delta (conformal inference)**: Calibrates the failure threshold
       during tree search using online conformal inference.
    4. **Failure propagation**: Tracks failure probability per action node using
       ``p = p_immediate + delta_compounding * (1 - p_immediate) * p_next``.
    5. **Constrained policy targets**: Applies safety mask during target computation.

    Attributes:
        failure_fn: User-provided function ``state -> bool`` defining failure.
        delta_0: Nominal failure probability threshold.
        eta: Learning rate for adaptive Delta calibration.
        delta_compounding: Discount factor for failure propagation.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
        >>> from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import ConstrainedZero
        >>>
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> sampler = DiscreteActionSampler(env.get_actions())
        >>> planner = ConstrainedZero(
        ...     environment=env,
        ...     discount_factor=0.95,
        ...     depth=3,
        ...     name="CZ_Tiger",
        ...     action_sampler=sampler,
        ...     n_simulations=20,
        ...     state_dim=1,
        ...     failure_fn=lambda s: False,
        ... )
        >>> belief = get_initial_belief(env, n_particles=10)
        >>> actions, run_data = planner.action(belief)
        >>> actions[0] in env.get_actions()
        True
    """

    network: ConstrainedZeroNetwork
    _buffer: ConstrainedTrainingBuffer

    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        failure_fn: Callable[[Any], bool],
        delta_0: float = 0.01,
        eta: float = 0.00001,
        delta_compounding: float = 1.0,
        # Inherited BetaZero params
        k_a: float = 1.0,
        alpha_a: float = 0.5,
        k_o: float = 1.0,
        alpha_o: float = 0.5,
        exploration_constant: float = 1.0,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_visit_count_per_action: int = 1,
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
        use_dropout: bool = True,
        p_dropout: float = 0.2,
        track_gradients: bool = False,
        normalize_inputs: bool = True,
        normalize_values: bool = True,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize ConstrainedZero planner.

        Args:
            environment: The POMDP environment to plan for.
            discount_factor: Discount factor gamma.
            depth: Maximum MCTS search depth.
            name: Policy identifier.
            action_sampler: Action sampling strategy for progressive widening.
            failure_fn: Function ``state -> bool`` that returns True if the
                state is a failure state.
            delta_0: Nominal failure probability threshold (default 0.01).
            eta: Learning rate for adaptive Delta calibration (default 0.00001).
            delta_compounding: Discount factor for failure propagation (default 1.0).
            k_a: Action widening coefficient.
            alpha_a: Action widening exponent.
            k_o: Observation widening coefficient.
            alpha_o: Observation widening exponent.
            exploration_constant: PUCT exploration constant c.
            time_out_in_seconds: Time limit for planning.
            n_simulations: Number of MCTS simulations.
            min_visit_count_per_action: Min visits per action at root.
            network: Pre-trained 3-head network (created automatically if ``None``).
            belief_representation: Belief feature extractor.
            state_dim: State dimensionality (required if ``belief_representation``
                is ``None``).
            z_q: Q-value exponent in policy target.
            z_n: Visit-count exponent in policy target.
            temperature: Temperature tau for policy target.
            n_buffer: Number of policy-iteration slots to retain in the
                replay buffer (default 1 = on-policy, current iteration only).
            training_batch_size: Mini-batch size during training.
            training_epochs: Epochs per ``fit()`` iteration.
            learning_rate: Adam learning rate.
            weight_decay: L2 regularisation weight.
            hidden_sizes: Widths of hidden layers in the network trunk.
            use_dropout: If True, apply dropout in the shared network trunk (default True).
            p_dropout: Dropout probability passed to ConstrainedZeroNetwork (default 0.2).
            track_gradients: When ``True``, gradient and weight norms are
                computed during training and included in the metrics dict.
                Includes an additional ``"grad_norm/failure_head"`` key
                compared to ``BetaZero``.
            normalize_inputs: When ``True``, belief features are z-scored
                before being fed to the network during training and inference.
            normalize_values: When ``True``, value targets are z-scored
                during training and predicted values are de-normalised at
                inference time.
            log_path: Optional log directory.
            debug: Enable debug logging.
            use_queue_logger: Use queue-based logging.
        """
        self.failure_fn = failure_fn
        self.delta_0 = delta_0
        self.eta = eta
        self.delta_compounding = delta_compounding
        self.use_dropout = use_dropout
        self.p_dropout = p_dropout

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
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            min_visit_count_per_action=min_visit_count_per_action,
            network=network,
            belief_representation=belief_representation,
            state_dim=state_dim,
            z_q=z_q,
            z_n=z_n,
            temperature=temperature,
            n_buffer=n_buffer,
            training_batch_size=training_batch_size,
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_sizes=hidden_sizes,
            track_gradients=track_gradients,
            normalize_inputs=normalize_inputs,
            normalize_values=normalize_values,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Replace the buffer with a constrained version
        self._buffer = ConstrainedTrainingBuffer(n_buffer=n_buffer)

        # External dicts for failure and adaptive delta storage
        self._failure_dict: Dict[int, float] = {}
        self._delta_dict: Dict[int, float] = {}

    # ── Network creation override ─────────────────────────────────────

    def _create_default_network(self, hidden_sizes: Tuple[int, ...]) -> ConstrainedZeroNetwork:
        belief_dim = self.belief_representation.feature_dim
        env_space = self.environment.space_info

        if env_space.action_space == SpaceType.DISCRETE:
            return ConstrainedZeroNetwork(
                belief_dim=belief_dim,
                action_space_type="discrete",
                n_actions=len(self.environment.get_actions()),  # type: ignore[attr-defined]
                hidden_sizes=hidden_sizes,
                use_dropout=self.use_dropout,
                p_dropout=self.p_dropout,
            )
        return ConstrainedZeroNetwork(
            belief_dim=belief_dim,
            action_space_type="continuous",
            action_dim=self._infer_action_dim(),
            hidden_sizes=hidden_sizes,
            use_dropout=self.use_dropout,
            p_dropout=self.p_dropout,
        )

    # ── MCTS overrides ────────────────────────────────────────────────

    def _learn_tree(self, belief: Belief) -> Tuple[Tree, int]:
        self._failure_dict.clear()
        self._delta_dict.clear()
        return super()._learn_tree(belief)

    def _simulate_path(self, tree: Tree, belief_id: int, depth: int) -> float:
        if depth > self.depth:
            return 0.0

        if self.environment.is_terminal(tree.get_belief(belief_id).sample()):
            tree.increment_visit_count(belief_id)
            return 0.0

        action_priors = self._get_action_priors(tree=tree, belief_id=belief_id)
        delta_prime = self._get_delta_prime(belief_id)

        action_id = spuct_action_progressive_widening_arena(
            tree=tree,
            belief_id=belief_id,
            alpha_a=self.alpha_a,
            action_sampler=self.action_sampler,
            exploration_constant=self.exploration_constant,
            k_a=self.k_a,
            failure_dict=self._failure_dict,
            delta_prime=delta_prime,
            action_priors=action_priors,
            min_visit_count_per_action=self.min_visit_count_per_action,
        )

        return_sample, failure_sample = self._simulate_return_constrained(
            tree=tree, belief_id=belief_id, action_id=action_id, depth=depth
        )

        self._update_node_statistics_constrained(
            tree=tree,
            belief_id=belief_id,
            action_id=action_id,
            total=return_sample,
            failure=failure_sample,
        )

        return return_sample

    def _simulate_return_constrained(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        depth: int,
    ) -> Tuple[float, float]:
        p_immediate = self._estimate_belief_failure_prob(tree.get_belief(belief_id))

        action_children_count = len(tree.get_children_ids(action_id))
        action_visits = tree.get_visit_count(action_id)
        if action_children_count <= self.k_o * action_visits**self.alpha_o:
            next_belief_id, immediate_reward = self._sample_new_belief_node(
                tree=tree, belief_id=belief_id, action_id=action_id
            )
            leaf_value, leaf_failure = self._network_leaf_value_and_failure(
                tree.get_belief(next_belief_id)
            )
            total = immediate_reward + self.discount_factor * leaf_value
            failure = self._compound_failure(p_immediate, leaf_failure)
        else:
            next_belief_id, immediate_reward = self._sample_existing_belief_node(
                tree=tree, belief_id=belief_id, action_id=action_id
            )
            next_return = self._simulate_path(tree=tree, belief_id=next_belief_id, depth=depth + 1)
            total = immediate_reward + self.discount_factor * next_return
            p_next = self._get_subtree_failure(tree, next_belief_id)
            failure = self._compound_failure(p_immediate, p_next)

        return total, failure

    def _update_node_statistics_constrained(
        self,
        tree: Tree,
        belief_id: int,
        action_id: int,
        total: float,
        failure: float,
    ) -> None:
        tree.increment_visit_count(belief_id)
        tree.update_action_q_with_return(action_id, total)
        children = tree.get_children_ids(belief_id)
        if children:
            tree.v_value[belief_id] = float(max(tree.get_q_value(cid) for cid in children))

        self._update_action_failure(tree=tree, action_id=action_id, failure=failure)
        self._update_adaptive_delta(tree=tree, belief_id=belief_id, action_id=action_id)

    # ── Network helpers ───────────────────────────────────────────────

    def _network_leaf_value_and_failure(self, belief: Belief) -> Tuple[float, float]:
        features = self._get_normalized_features(self.belief_representation(belief))
        _, value, failure_prob = self.network.predict(features)
        return self._get_denormalized_value(value), failure_prob

    def _network_leaf_value(self, belief: Belief) -> float:
        value, _ = self._network_leaf_value_and_failure(belief)
        return value

    def _discrete_action_priors(self, tree: Tree, belief_id: int) -> np.ndarray:
        features = self._get_normalized_features(
            self.belief_representation(tree.get_belief(belief_id))
        )
        policy, _, _ = self.network.predict(features)
        actions = self.environment.get_actions()  # type: ignore[attr-defined]
        children = tree.get_children_ids(belief_id)
        child_actions = [tree.get_action(cid) for cid in children]
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

    # ── Failure tracking helpers ──────────────────────────────────────

    def _estimate_belief_failure_prob(self, belief: Belief) -> float:
        particles = [belief.sample() for _ in range(20)]
        failures = sum(1 for s in particles if self.failure_fn(s))
        return failures / len(particles)

    def _compound_failure(self, p_immediate: float, p_next: float) -> float:
        return p_immediate + self.delta_compounding * (1.0 - p_immediate) * p_next

    def _get_delta_prime(self, belief_id: int) -> float:
        stored = self._delta_dict.get(belief_id, self.delta_0)
        return max(self.delta_0, stored)

    def _update_action_failure(self, tree: Tree, action_id: int, failure: float) -> None:
        old = self._failure_dict.get(action_id, 0.0)
        n = tree.get_visit_count(action_id)
        self._failure_dict[action_id] = old + (failure - old) / max(n, 1)

    def _update_adaptive_delta(self, tree: Tree, belief_id: int, action_id: int) -> None:
        f = self._failure_dict.get(action_id, 0.0)
        current_delta = self._delta_dict.get(belief_id, self.delta_0)
        err = 1.0 if f > current_delta else 0.0
        new_delta = current_delta + self.eta * (err - self.delta_0)
        lb, ub = self._compute_delta_bounds(tree, belief_id)
        self._delta_dict[belief_id] = float(np.clip(new_delta, lb, ub))

    def _compute_delta_bounds(self, tree: Tree, belief_id: int) -> Tuple[float, float]:
        child_failures = [
            self._failure_dict[cid]
            for cid in tree.get_children_ids(belief_id)
            if cid in self._failure_dict
        ]
        if not child_failures:
            return self.delta_0 * 0.1, min(1.0, self.delta_0 * 10.0)
        return min(child_failures), max(child_failures)

    def _get_subtree_failure(self, tree: Tree, belief_id: int) -> float:
        children = tree.get_children_ids(belief_id)
        if not children:
            return 0.0
        total_visits = sum(tree.get_visit_count(cid) for cid in children)
        if total_visits == 0:
            return 0.0
        weighted_sum = sum(
            self._failure_dict.get(cid, 0.0) * tree.get_visit_count(cid) for cid in children
        )
        return weighted_sum / total_visits

    # ── Constrained Q-weighted policy target ──────────────────────────

    def _compute_q_weighted_policy_target(self, tree: Tree, belief_id: int) -> np.ndarray:
        children = tree.get_children_ids(belief_id)
        q_values = np.array([tree.get_q_value(cid) for cid in children])
        visit_counts = np.array([tree.get_visit_count(cid) for cid in children], dtype=np.float64)
        failure_probs = np.array([self._failure_dict.get(cid, 0.0) for cid in children])

        q_term = self._softmax_q_term(q_values)
        n_term = self._visit_count_term(visit_counts)

        logits = (self.z_q * np.log(q_term + 1e-10) + self.z_n * np.log(n_term + 1e-10)) / max(
            self.temperature, 1e-10
        )
        logits -= logits.max()
        probs = np.exp(logits)

        # Apply safety mask
        delta_prime = self._get_delta_prime(belief_id)
        safety_mask = (failure_probs <= delta_prime).astype(np.float64)
        if safety_mask.sum() == 0:
            safety_mask = np.ones_like(safety_mask)
        probs = probs * safety_mask
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(len(children)) / len(children)

        if self.network.action_space_type == "discrete":
            return self._map_to_full_action_vector(tree, children, probs)
        return self._compute_continuous_policy_target(tree, children, probs)

    # ── TrainablePolicy overrides ────────────────────────────────────

    def get_metric_keys(self) -> List[str]:
        keys = ["total_loss", "value_loss", "policy_loss", "failure_loss"]
        if self.track_gradients:
            keys += [
                "grad_norm/global",
                "grad_norm/trunk",
                "grad_norm/policy_head",
                "grad_norm/value_head",
                "grad_norm/failure_head",
                "weight_norm/global",
            ]
        return keys

    # ── Episode data / training overrides ─────────────────────────────

    def _finalize_episode_data(self, history) -> None:
        rewards = [step.reward for step in history.history if step.reward is not None]
        discounted_returns = self._compute_discounted_returns(rewards)
        # Match the documented semantic of
        # ``ConstrainedTrainingExample.failure_target`` — a single
        # episode-level "any failure occurred?" boolean, applied to every
        # training example. Avoids the ``discounted_returns``
        # (None-filtered) vs. per-step ``failure_targets`` (full-history)
        # length mismatch the previous implementation could hit.
        failure_target = float(self._compute_episode_failure(history))

        for i, pending in enumerate(self._pending_examples):
            if i < len(discounted_returns):
                self._buffer.add(
                    ConstrainedTrainingExample(
                        belief_features=pending.belief_features,
                        policy_target=pending.policy_target,
                        value_target=discounted_returns[i],
                        failure_target=failure_target,
                    )
                )

    def _compute_episode_failure(self, history) -> bool:
        for step in history.history:
            if step.state is not None and self.failure_fn(step.state):
                return True
            if step.next_state is not None and self.failure_fn(step.next_state):
                return True
        return False

    def _compute_per_timestep_failures(self, history) -> List[float]:
        per_step_failure = self._extract_per_step_failure_flags(history)
        return self._backward_accumulate_failures(per_step_failure)

    def _extract_per_step_failure_flags(self, history) -> List[bool]:
        flags: List[bool] = []
        for step in history.history:
            failed = False
            if step.state is not None and self.failure_fn(step.state):
                failed = True
            if step.next_state is not None and self.failure_fn(step.next_state):
                failed = True
            flags.append(failed)
        return flags

    def _backward_accumulate_failures(self, per_step_failure: List[bool]) -> List[float]:
        n = len(per_step_failure)
        if n == 0:
            return []
        targets = [0.0] * n
        targets[-1] = 1.0 if per_step_failure[-1] else 0.0
        for t in range(n - 2, -1, -1):
            if per_step_failure[t] or targets[t + 1] > 0.0:
                targets[t] = 1.0
            else:
                targets[t] = 0.0
        return targets

    def _train_network_on_buffer(self) -> Dict[str, List[float]]:
        if self.normalize_inputs or self.normalize_values:
            self._update_normalization_stats()
        return train_constrained_network(
            network=self.network,
            buffer=self._buffer,
            n_epochs=self.training_epochs,
            batch_size=self.training_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            track_gradients=self.track_gradients,
            input_mean=self._input_mean if self.normalize_inputs else None,
            input_std=self._input_std if self.normalize_inputs else None,
            value_mean=self._value_mean if self.normalize_values else None,
            value_std=self._value_std if self.normalize_values else None,
        )
