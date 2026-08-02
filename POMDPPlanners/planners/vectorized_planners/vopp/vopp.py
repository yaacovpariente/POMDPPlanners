# SPDX-License-Identifier: MIT

"""Vectorized Online POMDP Planner (VOPP / PORPP).

This module implements :class:`VOPPPlanner`, a fully GPU-vectorized online
POMDP solver following Hoerger, Sudrajat and Kurniawati, *Vectorized Online
POMDP Planning* (arXiv:2510.27191). VOPP is the vectorized realization of
Partially Observable Reference Policy Programming (PORPP): rather than
interleaving numerical optimization with expectation estimation, it solves the
value function analytically (a log-sum-exp over action *preferences*) and
leaves only expectations to Monte Carlo.

The planner drives three collaborating pieces, none of which contains any
planning-policy logic itself:

* a :class:`~POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree.VectorizedBeliefTree`
  holding the tree topology, per-action visit/reward statistics, and two
  registered belief fields -- ``preferences`` (the ``|A|``-vector ``Psi`` of
  eq. 4) and ``value`` (the scalar belief value ``V``);
* a :class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
  supplying the batched transition / observation / reward / terminal / key
  kernels; and
* a root belief represented as a tensor of state particles.

Each planning iteration runs two fully vectorized passes over the tree:

#. **Forward search** (Algorithm 2): from the root belief, sample ``n_p``
   states, and repeatedly -- for every live episode in parallel -- sample an
   action from ``softmax(eta * Psi)``, push it through the generative model,
   accumulate the immediate reward on the action node, drop terminated
   episodes, and append the successor belief nodes. This descends one tree
   level per step until ``max_depth`` is exceeded, where a value heuristic
   scores the leaf states.
#. **Preference backup** (Algorithm 3): from the deepest level up to the root,
   compute per-action ``Q = mean_reward + gamma * weighted_future_value`` and
   apply the PORPP preference update (eq. 5)
   ``Psi <- Psi - L_eta[Psi] + Q`` on every belief, recomputing each belief's
   value ``V = L_eta[Psi]`` as it goes.

After the configured budget of iterations the planner returns the root action
with the highest preference.

VOPP assumes a fixed, finite *representative* action set indexed
``0 .. num_actions - 1``; the model's ``action_keys`` must return exactly that
integer index so it can address a column of the dense ``preferences`` field.

References:
    Hoerger, M., Sudrajat, M., & Kurniawati, H. (2026). Vectorized Online POMDP
    Planning. arXiv:2510.27191. https://arxiv.org/abs/2510.27191

Example:
    Planning one step in Continuous Light-Dark::

        >>> import torch
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP,
        ... )
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
        ...     ContinuousLightDarkVectorizedModel,
        ... )
        >>> from POMDPPlanners.planners.vectorized_planners import VOPPPlanner
        >>> _ = torch.manual_seed(0)
        >>> env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
        >>> model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))
        >>> planner = VOPPPlanner(
        ...     model,
        ...     num_actions=model.num_actions,
        ...     num_particles=256,
        ...     max_depth=5,
        ...     num_planning_iterations=8,
        ...     discount_factor=0.95,
        ... )
        >>> root_particles = torch.tensor([[2.0, 2.0]]).repeat(256, 1)
        >>> action = planner.plan(root_particles)
        >>> 0 <= action < model.num_actions
        True
"""

from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.core.policy import PolicyInfoVariable
from POMDPPlanners.core.tree.vectorized_belief_tree import VectorizedBeliefTree
from POMDPPlanners.utils.tree_statistics import compute_vectorized_tree_metrics

ValueHeuristic = Callable[[Tensor], Tensor]

_PREFERENCES_FIELD = "preferences"
_VALUE_FIELD = "value"


def _zero_value_heuristic(states: Tensor) -> Tensor:
    """Default leaf heuristic: zero future value for every leaf state."""
    return torch.zeros(states.shape[0], dtype=states.dtype, device=states.device)


class VOPPPlanner:
    """Fully vectorized online POMDP planner (VOPP / PORPP).

    The planner owns a :class:`VectorizedBeliefTree` and repeatedly expands and
    backs it up entirely with batched tensor operations. A single call to
    :meth:`plan` samples particles from the supplied root belief, runs the
    configured number of forward-search / preference-backup iterations, and
    returns the greedy root action.

    Attributes:
        device: Device every tensor lives on (taken from the model).
        num_actions: Size of the fixed representative action set.

    Example:
        See the module-level docstring for a runnable example.
    """

    def __init__(
        self,
        model: VectorizedGenerativeModel,
        num_actions: int,
        *,
        temperature: float = 2.0,
        num_particles: int = 1000,
        max_depth: int = 10,
        discount_factor: float = 0.95,
        num_planning_iterations: int = 100,
        value_heuristic: Optional[ValueHeuristic] = None,
        belief_capacity: int = 1024,
        action_capacity: int = 1024,
        value_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialise the planner and its belief tree.

        Args:
            model: Batched generative model driving the forward search.
            num_actions: Number of representative actions; the ``preferences``
                field has this many columns and actions index them directly.
            temperature: PORPP temperature ``eta`` (must be positive).
            num_particles: Episodes sampled in parallel per iteration (``n_p``).
            max_depth: Maximum search depth; leaves sit one level below it.
            discount_factor: POMDP discount ``gamma`` in ``(0, 1]``.
            num_planning_iterations: Forward-search / backup passes per
                :meth:`plan` call (the planning budget).
            value_heuristic: Optional leaf value estimator mapping
                ``[n, ds]`` states to ``[n]`` values; defaults to zeros.
            belief_capacity: Initial preallocated belief-node capacity.
            action_capacity: Initial preallocated action-node capacity.
            value_dtype: Floating dtype for preferences, values, and rewards.

        Raises:
            ValueError: If ``num_actions`` or ``temperature`` is non-positive,
                or ``max_depth`` / ``num_particles`` / ``num_planning_iterations``
                is not positive.
        """
        self._validate_hyperparameters(
            num_actions, temperature, num_particles, max_depth, num_planning_iterations
        )
        self._model = model
        self.num_actions = num_actions
        self.device = model.device
        self._temperature = float(temperature)
        self._num_particles = num_particles
        self._max_depth = max_depth
        self._discount_factor = float(discount_factor)
        self._num_iterations = num_planning_iterations
        self._value_heuristic = value_heuristic or _zero_value_heuristic
        self._value_dtype = value_dtype
        self._tree = VectorizedBeliefTree(
            device=self.device,
            belief_capacity=belief_capacity,
            action_capacity=action_capacity,
            value_dtype=value_dtype,
        )
        self._tree.register_belief_field(
            _PREFERENCES_FIELD, (num_actions,), dtype=value_dtype, default=0.0
        )
        self._tree.register_belief_field(_VALUE_FIELD, dtype=value_dtype, default=0.0)

    @staticmethod
    def _validate_hyperparameters(
        num_actions: int,
        temperature: float,
        num_particles: int,
        max_depth: int,
        num_planning_iterations: int,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if num_planning_iterations <= 0:
            raise ValueError("num_planning_iterations must be positive")

    @property
    def tree(self) -> VectorizedBeliefTree:
        """The belief tree built by the most recent :meth:`plan` call."""
        return self._tree

    def tree_metrics(self) -> List[PolicyInfoVariable]:
        """Root-tree analysis metrics for the most recent :meth:`plan` call.

        Returns the same :class:`~POMDPPlanners.utils.tree_statistics.TreeMetrics`
        set the MCTS planners report (root action visit min / max / entropy,
        number of root actions, root visit count, max depth, leaf flag), making
        VOPP directly comparable to them.

        Returns:
            A list of :class:`~POMDPPlanners.core.policy.PolicyInfoVariable`
            describing the current tree.
        """
        return compute_vectorized_tree_metrics(self._tree)

    def plan(self, root_particles: Tensor) -> int:
        """Plan from a particle-set root belief and return the greedy action.

        The tree is cleared, then ``num_planning_iterations`` forward-search /
        preference-backup passes refine the root action preferences. The action
        with the highest root preference is returned.

        Args:
            root_particles: ``[num_root_particles, ds]`` states representing the
                current belief; iterations resample ``num_particles`` of them
                with replacement.

        Returns:
            The index of the greedy root action in ``[0, num_actions)``.

        Raises:
            ValueError: If ``root_particles`` is not a 2-D tensor on the
                planner's device.
        """
        self._validate_root_particles(root_particles)
        self._tree.clear()
        root = self._tree.root_index
        for _ in range(self._num_iterations):
            states = self._sample_root_states(root_particles)
            beliefs = torch.full(
                (states.shape[0],), root, dtype=self._tree.index_dtype, device=self.device
            )
            leaf_beliefs, leaf_values = self._forward_search(states, beliefs)
            self._backup(leaf_beliefs, leaf_values)
        root_preferences = self._tree.belief_field(_PREFERENCES_FIELD)[root]
        return int(torch.argmax(root_preferences).item())

    def _validate_root_particles(self, root_particles: Tensor) -> None:
        if root_particles.dim() != 2:
            raise ValueError("root_particles must be a 2-D [num_particles, ds] tensor")
        if root_particles.device != self.device:
            raise ValueError(f"root_particles must be on device {self.device}")

    def _sample_root_states(self, root_particles: Tensor) -> Tensor:
        indices = torch.randint(
            0, root_particles.shape[0], (self._num_particles,), device=self.device
        )
        return root_particles[indices]

    # ------------------------------------------------------------------ #
    # Forward search (Algorithm 2)
    # ------------------------------------------------------------------ #

    def _forward_search(self, states: Tensor, beliefs: Tensor) -> Tuple[Tensor, Tensor]:
        """Expand the tree one level per step, returning ``(leaf_beliefs, H)``."""
        depth = 0
        while depth <= self._max_depth:
            actions = self._sample_actions(beliefs)
            action_nodes, _ = self._tree.get_or_create_actions(
                beliefs, self._model.action_keys(actions)
            )
            next_states = self._model.sample_next_states(states, actions)
            rewards = self._model.rewards(states, actions, next_states)
            self._tree.update_action_statistics(action_nodes, rewards.to(self._value_dtype))
            survivors = ~self._model.terminal_mask(next_states)
            if not bool(survivors.any()):
                return self._empty_leaves()
            states, beliefs = self._expand_survivors(action_nodes, next_states, actions, survivors)
            depth += 1
        return beliefs, self._value_heuristic(states).to(self._value_dtype)

    def _sample_actions(self, beliefs: Tensor) -> Tensor:
        preferences = self._tree.belief_field(_PREFERENCES_FIELD)[beliefs]
        probabilities = torch.softmax(self._temperature * preferences, dim=1)
        return torch.multinomial(probabilities, 1).squeeze(1)

    def _expand_survivors(
        self,
        action_nodes: Tensor,
        next_states: Tensor,
        actions: Tensor,
        survivors: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        surviving_states = next_states[survivors]
        surviving_actions = actions[survivors]
        observations = self._model.sample_observations(surviving_states, surviving_actions)
        observation_keys = self._model.observation_keys(observations)
        child_beliefs, _ = self._tree.get_or_create_beliefs(
            action_nodes[survivors], observation_keys
        )
        return surviving_states, child_beliefs

    def _empty_leaves(self) -> Tuple[Tensor, Tensor]:
        empty_index = torch.empty(0, dtype=self._tree.index_dtype, device=self.device)
        empty_value = torch.empty(0, dtype=self._value_dtype, device=self.device)
        return empty_index, empty_value

    # ------------------------------------------------------------------ #
    # Preference backup (Algorithm 3)
    # ------------------------------------------------------------------ #

    def _backup(self, leaf_beliefs: Tensor, leaf_values: Tensor) -> None:
        """Propagate leaf values to the root via the PORPP preference update."""
        num_beliefs = self._tree.num_belief_nodes
        belief_visits = self._init_leaf_values(leaf_beliefs, leaf_values, num_beliefs)
        stats = self._action_statistics()
        for depth in range(self._max_depth, -1, -1):
            self._backup_depth(depth, belief_visits, stats)

    def _init_leaf_values(
        self, leaf_beliefs: Tensor, leaf_values: Tensor, num_beliefs: int
    ) -> Tensor:
        belief_visits = torch.zeros(num_beliefs, dtype=self._value_dtype, device=self.device)
        if leaf_beliefs.numel() == 0:
            return belief_visits
        value = self._tree.belief_field(_VALUE_FIELD)
        belief_visits.index_add_(0, leaf_beliefs, torch.ones_like(leaf_values))
        heuristic_sum = torch.zeros(num_beliefs, dtype=self._value_dtype, device=self.device)
        heuristic_sum.index_add_(0, leaf_beliefs, leaf_values)
        leaves = belief_visits > 0
        value[leaves] = heuristic_sum[leaves] / belief_visits[leaves]
        return belief_visits

    def _action_statistics(self) -> dict:
        num_actions = self._tree.num_action_nodes
        visits = self._tree.action_visit_count[:num_actions].to(self._value_dtype)
        return {
            "parent_belief": self._tree.action_parent_belief[:num_actions],
            "key": self._tree.action_key[:num_actions],
            "depth": self._tree.action_depth[:num_actions],
            "visits": visits,
            "mean_reward": self._tree.action_reward_sum[:num_actions] / visits.clamp_min(1.0),
            "count": num_actions,
        }

    def _backup_depth(self, depth: int, belief_visits: Tensor, stats: dict) -> None:
        beliefs_at_depth = self._tree.belief_nodes_at_depth(depth)
        if beliefs_at_depth.numel() == 0:
            return
        actions_at_depth = torch.nonzero(stats["depth"] == depth, as_tuple=False).flatten()
        q_values = self._action_q_values(depth, belief_visits, stats)
        self._accumulate_belief_visits(belief_visits, actions_at_depth, stats)
        self._apply_preference_update(beliefs_at_depth, actions_at_depth, q_values, stats)

    def _action_q_values(self, depth: int, belief_visits: Tensor, stats: dict) -> Tensor:
        child_beliefs = self._tree.belief_nodes_at_depth(depth + 1)
        future = torch.zeros(stats["count"], dtype=self._value_dtype, device=self.device)
        if child_beliefs.numel() > 0:
            value = self._tree.belief_field(_VALUE_FIELD)
            weighted = belief_visits[child_beliefs] * value[child_beliefs]
            parents = self._tree.belief_parent_action[child_beliefs]
            future.index_add_(0, parents, weighted)
        weighted_future = future / stats["visits"].clamp_min(1.0)
        return stats["mean_reward"] + self._discount_factor * weighted_future

    def _accumulate_belief_visits(
        self, belief_visits: Tensor, actions_at_depth: Tensor, stats: dict
    ) -> None:
        totals = torch.zeros_like(belief_visits)
        totals.index_add_(
            0, stats["parent_belief"][actions_at_depth], stats["visits"][actions_at_depth]
        )
        belief_visits[:] = torch.where(totals > 0, totals, belief_visits)

    def _apply_preference_update(
        self,
        beliefs_at_depth: Tensor,
        actions_at_depth: Tensor,
        q_values: Tensor,
        stats: dict,
    ) -> None:
        preferences = self._tree.belief_field(_PREFERENCES_FIELD)
        value = self._tree.belief_field(_VALUE_FIELD)
        current_value = self._log_sum_exp(preferences[beliefs_at_depth])
        preferences[beliefs_at_depth] = preferences[beliefs_at_depth] - current_value.unsqueeze(1)
        preferences[
            stats["parent_belief"][actions_at_depth], stats["key"][actions_at_depth]
        ] += q_values[actions_at_depth]
        value[beliefs_at_depth] = self._log_sum_exp(preferences[beliefs_at_depth])

    def _log_sum_exp(self, preferences: Tensor) -> Tensor:
        return torch.logsumexp(self._temperature * preferences, dim=1) / self._temperature
